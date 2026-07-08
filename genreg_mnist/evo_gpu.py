"""GPU backend for the image-pipe genome batteries (MNIST-Pipe / CIFAR-Pipe).

Pure arithmetic acceleration — the evolution (ga_step, selection, mutation,
champion gating) stays in numpy on the CPU, byte-identical to the CPU path.
Only the FITNESS EVALUATIONS (dense GEMMs + elementwise activations + block
pooling + Fisher statistics) run on the GPU, because that is >95% of the
wall-clock. No gradients are ever computed: torch is used under no_grad as a
matrix calculator. TF32 is disabled so GPU fitness matches CPU fitness to
float32 rounding.

Everything degrades gracefully: HAS_GPU is False without torch/CUDA and the
pipelines fall back to their numpy paths.
"""
import numpy as np

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:                                # pragma: no cover
    torch = None
    HAS_GPU = False

DEV = "cuda" if HAS_GPU else "cpu"


def to_dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def _acts_t(x, a):
    """Torch mirror of mnist_pipe._acts (same 8-function catalog)."""
    if a == 0:
        return torch.relu(x)
    if a == 1:
        return torch.abs(x)
    if a == 2:
        return torch.sin(x)
    if a == 3:
        return torch.cos(x)
    if a == 4:
        return torch.exp(-x * x)
    if a == 5:
        return torch.where(x > 0, x, 0.1 * x)
    if a == 6:
        return x * x
    return torch.tanh(x)


def _pool_t(resp, R, pools):
    """(B,R,R) -> (B, sum r*c) crop-block mean pools (matches _pool_resp)."""
    B = resp.shape[0]
    parts = []
    for r, c in pools:
        hr, hc = R // r, R // c
        parts.append(resp[:, :r * hr, :c * hc].reshape(B, r, hr, c, hc)
                     .mean(dim=(2, 4)).reshape(B, r * c))
    return torch.cat(parts, dim=1)


# --------------------------------------------------------------------------
# Joint-head fitness (mean log-softmax of true class - L2), numpy in/out
# --------------------------------------------------------------------------
class JointFitGPU:
    def __init__(self, F, y):
        self.Fg = to_dev(F.astype(np.float32))
        self.yg = to_dev(y.astype(np.int64)).view(1, -1, 1)
        self.N, self.nf = F.shape

    @torch.no_grad()
    def __call__(self, W, b, l2=0.0):
        P = len(W)
        Wg = to_dev(W)                             # (P,nf,10)
        bg = to_dev(b)                             # (P,10)
        z = (self.Fg @ Wg.permute(1, 0, 2).reshape(self.nf, P * 10)) \
            .reshape(self.N, P, 10).permute(1, 0, 2) + bg[:, None, :]
        logp = torch.log_softmax(z, dim=-1)
        ch = logp.gather(2, self.yg.expand(P, self.N, 1))[..., 0]
        fit = ch.mean(dim=1)
        if l2 > 0:
            fit = fit - l2 * (Wg * Wg).sum(dim=(1, 2))
        return fit.cpu().numpy()

    @torch.no_grad()
    def acc1(self, W, b):
        """Top-1 accuracy of a single head (nf,10),(10,) on this pool."""
        Wg = to_dev(W); bg = to_dev(b)
        pred = (self.Fg @ Wg + bg).argmax(dim=1)
        return float((pred == self.yg.view(-1)).float().mean().item())


# --------------------------------------------------------------------------
# Binary (detector / pairwise) fitness: pools uploaded once, minibatch
# indices sampled on CPU per generation
# --------------------------------------------------------------------------
class BinaryFitGPU:
    def __init__(self, Fp, Fn):
        self.Fp = to_dev(Fp.astype(np.float32))
        self.Fn = to_dev(Fn.astype(np.float32))

    @torch.no_grad()
    def __call__(self, w, b, ip, inn, l2=0.0):
        """w (P,nf), b (P,), ip/inn int index arrays into the pools ->
        (fit (P,), acc (P,)) numpy — same math as LinearPop.fitness."""
        wg = to_dev(w); bg = to_dev(b)
        Fp = self.Fp[to_dev(ip.astype(np.int64))]
        Fn = self.Fn[to_dev(inn.astype(np.int64))]
        zp = (wg @ Fp.T + bg[:, None]).clamp(-30, 30)
        zn = (wg @ Fn.T + bg[:, None]).clamp(-30, 30)
        lp = -torch.log1p(torch.exp(-zp)).mean(dim=1)
        ln = -torch.log1p(torch.exp(zn)).mean(dim=1)
        acc = ((zp > 0).float().mean(dim=1) + (zn < 0).float().mean(dim=1)) / 2
        fit = lp + ln
        if l2 > 0:
            fit = fit - l2 * (wg * wg).sum(dim=1)
        return fit.cpu().numpy(), acc.cpu().numpy()


# --------------------------------------------------------------------------
# Detector-bank fitness: conv responses + activations + pools + Fisher,
# whole population per call
# --------------------------------------------------------------------------
class DetbankFitGPU:
    def __init__(self, patches, y, R, pools, nc=10):
        """patches (N, R*R, KD) im2col'd images, y (N,) labels."""
        self.N, self.npos, self.kd = patches.shape
        self.R, self.pools, self.nc = R, pools, nc
        self.Pf = to_dev(patches.reshape(-1, self.kd).astype(np.float32))
        self.masks = [to_dev((y == c).astype(np.float32)) for c in range(nc)]
        self.counts = [float(m.sum().item()) for m in self.masks]

    @torch.no_grad()
    def __call__(self, K, b, act):
        """K (P,KD), b (P,), act (P,) ints -> Fisher fitness (P,) numpy."""
        P = len(K)
        Kg = to_dev(K); bg = to_dev(b)
        resp = (self.Pf @ Kg.T + bg).reshape(self.N, self.R, self.R, P)
        D = sum(r * c for r, c in self.pools)
        pooled = torch.empty((P, self.N, D), device=DEV)
        for a in range(8):
            ids = np.where(act == a)[0]
            if len(ids) == 0:
                continue
            idt = to_dev(ids.astype(np.int64))
            blk = _acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * self.N, self.R, self.R)
            pooled[idt] = _pool_t(blk, self.R, self.pools).reshape(len(ids), self.N, D)
        mu = pooled.mean(dim=1)                    # (P,D)
        bt = torch.zeros((P, D), device=DEV)
        wi = torch.zeros((P, D), device=DEV)
        for c in range(self.nc):
            m = self.masks[c]
            n_c = self.counts[c]
            mc = (pooled * m[None, :, None]).sum(dim=1) / n_c
            bt += n_c * (mc - mu) ** 2
            d = (pooled - mc[:, None, :]) * m[None, :, None]
            wi += (d * d).sum(dim=1)
        return (bt / (wi + 1e-8)).mean(dim=1).cpu().numpy()


@torch.no_grad()
def bank_features_gpu(patches_fn, X, bank, R, pools, chunk=4096):
    """Full-corpus bank features on GPU. `patches_fn(Xc) -> (n, R*R, KD)`."""
    nb = len(bank["K"])
    D = sum(r * c for r, c in pools)
    Kg = to_dev(bank["K"]); bg = to_dev(bank["b"])
    out = np.empty((len(X), nb * D), np.float32)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        Pf = to_dev(patches_fn(Xc).reshape(-1, bank["K"].shape[1]).astype(np.float32))
        resp = (Pf @ Kg.T + bg).reshape(len(Xc), R, R, nb)
        for a in range(8):
            ids = np.where(bank["act"] == a)[0]
            if len(ids) == 0:
                continue
            idt = to_dev(ids.astype(np.int64))
            blk = _acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * len(Xc), R, R)
            pooled = _pool_t(blk, R, pools).reshape(len(ids), len(Xc), D) \
                .permute(1, 0, 2).cpu().numpy()
            for k, j in enumerate(ids):
                out[lo:lo + len(Xc), j * D:(j + 1) * D] = pooled[:, k, :]
    return out
