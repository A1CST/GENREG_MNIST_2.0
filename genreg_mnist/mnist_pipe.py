"""MNIST-Pipe — the WordPipe recipe applied to images (2026-07-07).

The point of this module is a proof OUTSIDE language that the GA-abstraction
thesis holds: don't evolve the representation — BUILD it from corpus
statistics; evolution learns tiny relationships inside the pre-built space.
MNIST is the clean test because the output space is only 10 symbols.

Exactly the EvoLang/WordPipe decomposition, transposed to images:

  LAYER 1 — STATISTICS (built, never evolved). A fixed feature bank computed
            from the raw pixels + the training set's own statistics: zone ink
            densities, row/column ink profiles, gradient-orientation
            histograms at two cell scales (HOG-lite), and unsupervised PCA of
            the raw pixels (the "SVD features" of the image world). No labels
            touch the feature construction. The features are the environment.

  LAYER 2 — SEMANTIC genomes (evolved). Decompose "which digit?" into tiny
            specialists, each with one clean survival condition:
              * 10 DETECTORS  — "is this a 3, yes or no?" One linear head per
                digit over the fixed stats (NF+1 params each), soft BCE
                fitness (mean log-prob), balanced positives vs negatives.
              * 45 PAIRWISE disambiguators — "4 or 9?" One linear head per
                digit pair, trained ONLY on images of those two digits. The
                confusable-pair structure of MNIST is the environment being
                decomposed, same as language was decomposed into order /
                selection / agreement.

  LAYER 3 — OUTPUT mixer (evolved). A 10x10 matrix + bias over the detector
            logits, soft fitness = mean log-softmax prob of the true digit.
            At inference the pairwise specialists referee the mixer's top-2
            when they are close. Gate for every layer: beat the layer below
            it on the held-out split.

Everything gradient-free: tournament + elitism + energy starvation + self-
adaptive mutation (the shared GENREG machinery), numpy inference only.
Baselines per GENREG_RULES VII: majority class (11.35%) and nearest
class-centroid in stats space (no evolution) — every evolved layer must beat
the non-evolved floor. Test 10k is never touched during training; champions
are selected on a 5k validation slice carved off the train set.
"""
import gzip
import os
import pickle
import struct
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "corpora", "mnist")
CACHE = os.path.join(ROOT, "demo", "mnist_genomes.pkl")

N_VAL = 5000                     # carved off the end of train for champion gating


# --------------------------------------------------------------------------
# Data (idx format, cached gz files in corpora/mnist/)
# --------------------------------------------------------------------------
_MNIST = None


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist():
    """(Xtr (55k,28,28) f32 0-1, ytr, Xva (5k,...), yva, Xte (10k,...), yte)."""
    global _MNIST
    if _MNIST is None:
        X = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
        y = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")).astype(np.int64)
        Xt = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
        yt = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
        n = len(X) - N_VAL
        _MNIST = (X[:n], y[:n], X[n:], y[n:], Xt, yt)
    return _MNIST


# --------------------------------------------------------------------------
# LAYER 1 — the statistics layer (BUILT, not evolved)
# --------------------------------------------------------------------------
def deskew(X):
    """Moment-based deskew (round-2 statistics layer). Per image: estimate the
    shear angle from the ink's second moments (mu11/mu02) and resample with the
    inverse shear so every digit stands upright. Pure image statistics — no
    labels, nothing evolved; this enriches the ENVIRONMENT, not the organism.
    Vectorised bilinear remap over the whole batch."""
    N = len(X)
    ys, xs = np.mgrid[0:28, 0:28].astype(np.float32)
    m = X.sum(axis=(1, 2)) + 1e-8
    cy = (X * ys).sum(axis=(1, 2)) / m
    cx = (X * xs).sum(axis=(1, 2)) / m
    dy = ys[None] - cy[:, None, None]
    dx = xs[None] - cx[:, None, None]
    mu11 = (X * dx * dy).sum(axis=(1, 2)) / m
    mu02 = (X * dy * dy).sum(axis=(1, 2)) / m
    alpha = np.clip(mu11 / (mu02 + 1e-8), -1.0, 1.0)
    out = np.empty_like(X)
    idx = np.arange(N)
    for lo in range(0, N, 4096):                   # chunked to bound temp memory
        sl = idx[lo:lo + 4096]
        src_x = xs[None] + alpha[sl, None, None] * (ys[None] - cy[sl, None, None])
        x0 = np.floor(src_x).astype(np.int64)
        w = (src_x - x0).astype(np.float32)
        x0c = np.clip(x0, 0, 27); x1c = np.clip(x0 + 1, 0, 27)
        rows = np.broadcast_to(ys.astype(np.int64)[None], x0c.shape)
        n3 = sl[:, None, None]
        valid = ((src_x >= -1) & (src_x <= 28)).astype(np.float32)
        out[sl] = (X[n3, rows, x0c] * (1 - w) + X[n3, rows, x1c] * w) * valid
    return out



def _zone_ink(X, g):
    """Mean ink in a g x g grid of zones -> (N, g*g). 28 must divide by g cleanly
    enough; we use reshape-mean over 28//g blocks (g in {4, 7})."""
    N = len(X)
    s = 28 // g
    return X[:, :g * s, :g * s].reshape(N, g, s, g, s).mean(axis=(2, 4)).reshape(N, g * g)


def _profiles(X):
    """Row + column ink profiles, downsampled to 14 each -> (N, 28)."""
    rows = X.mean(axis=2).reshape(len(X), 14, 2).mean(axis=2)
    cols = X.mean(axis=1).reshape(len(X), 14, 2).mean(axis=2)
    return np.concatenate([rows, cols], axis=1)


def _grad_hist(X, cells, nbin=8):
    """Gradient-orientation histograms (HOG-lite): magnitude-weighted vote into
    `nbin` orientation bins per cell of a cells x cells grid -> (N, cells^2*nbin)."""
    gy = np.zeros_like(X); gx = np.zeros_like(X)
    gy[:, 1:-1, :] = X[:, 2:, :] - X[:, :-2, :]
    gx[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)                       # [-pi, pi)
    b = ((ang + np.pi) / (2 * np.pi) * nbin).astype(np.int64) % nbin
    N = len(X)
    s = 28 // cells
    out = np.zeros((N, cells, cells, nbin), np.float32)
    cy = (np.arange(28) // s).clip(0, cells - 1)
    cx = (np.arange(28) // s).clip(0, cells - 1)
    n_idx = np.repeat(np.arange(N), 28 * 28)
    cy_idx = np.tile(np.repeat(cy, 28), N)
    cx_idx = np.tile(np.tile(cx, 28), N)
    np.add.at(out, (n_idx, cy_idx, cx_idx, b.reshape(-1)), mag.reshape(-1))
    out = out.reshape(N, -1)
    return out / (out.sum(axis=1, keepdims=True) + 1e-6)


_STATCACHE = {}


def stat_features(X, pca=None):
    """The fixed statistics bank for a batch of images -> (N, NF) float32.
    `pca` = (mean, comps) built from the TRAIN set only (see build_pca)."""
    parts = [_zone_ink(X, 4), _zone_ink(X, 7), _profiles(X),
             _grad_hist(X, 4), _grad_hist(X, 7)]
    if pca is not None:
        mu, comps = pca
        parts.append((X.reshape(len(X), -1) - mu) @ comps.T)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_pca(Xtr, D=64):
    """Unsupervised PCA of the raw train pixels (the data-built component of the
    environment — same role as the SVD word features). Returns (mean, comps)."""
    F = Xtr.reshape(len(Xtr), -1)
    mu = F.mean(axis=0)
    # covariance eigendecomposition on 784 dims (cheap, exact)
    C = np.cov((F - mu).T.astype(np.float64))
    w, v = np.linalg.eigh(C)
    comps = v[:, ::-1][:, :D].T.astype(np.float32)
    return mu.astype(np.float32), comps


RFF_D = 1024                     # random-Fourier lift width (stats layer v3)
RFF_SEED = 424242                # fixed — the projection is derivable, not stored


def rff_lift(F, s, D=RFF_D, seed=RFF_SEED):
    """Random Fourier features (v3 environment): z = sqrt(2/D) cos(F W / s + b)
    with W ~ N(0,1), b ~ U[0,2pi). A FIXED random projection — approximates the
    RBF kernel's feature space, so the linear genomes get a nonlinear
    environment while staying linear organisms. Data-independent (seeded), so
    it is derivable at load, never stored, never evolved. `s` is the RBF
    bandwidth from the median heuristic on TRAIN features (a data statistic)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((F.shape[1], D)).astype(np.float32)
    b = (rng.random(D) * 2 * np.pi).astype(np.float32)
    return (np.sqrt(2.0 / D) * np.cos(F @ W / s + b)).astype(np.float32)


def median_bandwidth(F, n=2000, seed=0):
    """Median pairwise distance over a train sample (the classic unsupervised
    RBF bandwidth heuristic)."""
    rng = np.random.default_rng(seed)
    S = F[rng.choice(len(F), n, replace=False)]
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(np.median(d2[np.triu_indices(n, 1)])))


def _shift(X, dy, dx):
    """Integer-pixel shift with zero fill (train-pool augmentation)."""
    out = np.zeros_like(X)
    ys0, ys1 = max(0, dy), min(28, 28 + dy)
    xs0, xs1 = max(0, dx), min(28, 28 + dx)
    out[:, ys0:ys1, xs0:xs1] = X[:, ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def build_features(version=2, augment=0, v4_bank=None, v4_cache=None):
    """Build + standardise the full stats layer for train/val/test. Standardise
    with TRAIN statistics only. Cached in memory per (version, augment).
    version 1 = raw images; version 2 = deskewed (the shipped environment);
    version 3 = deskewed + a random-Fourier lift (kept for the record — CUT:
    the GA cannot exploit the lifted directions, minibatch noise dominates).
    `augment` = extra shifted copies of each TRAIN image appended to the train
    pool (environment enrichment on the data side — attacks the val->test
    generalisation gap). Val and test are NEVER augmented."""
    key = ("F", version, augment, v4_bank, v4_cache)
    if key in _STATCACHE:
        return _STATCACHE[key]
    Xtr, ytr, Xva, yva, Xte, yte = load_mnist()
    if version >= 2:
        Xtr, Xva, Xte = deskew(Xtr), deskew(Xva), deskew(Xte)
    pca = build_pca(Xtr)
    Ftr = stat_features(Xtr, pca)
    mu = Ftr.mean(axis=0); sd = Ftr.std(axis=0) + 1e-6
    Ftr = (Ftr - mu) / sd
    Fva = (stat_features(Xva, pca) - mu) / sd
    Fte = (stat_features(Xte, pca) - mu) / sd
    if version == 4:
        # evolved-detector environment: bank pools + built stats, PCA'd to a
        # readout-sized space (all unsupervised given the frozen bank)
        cache_f = v4_cache or os.path.join(DATA_DIR, "feats_v4.npz")
        if os.path.exists(cache_f):
            z = np.load(cache_f)
            Ftr2, Fva2, Fte2 = z["Ftr"], z["Fva"], z["Fte"]
        else:
            with open(v4_bank or DETBANK, "rb") as fh:
                bank = pickle.load(fh)
            Btr = bank_features(Xtr, bank)
            bmu = Btr.mean(0); bsd = Btr.std(0) + 1e-6
            Ftr2 = np.concatenate([Ftr, (Btr - bmu) / bsd], axis=1)
            Fva2 = np.concatenate([Fva, (bank_features(Xva, bank) - bmu) / bsd], axis=1)
            Fte2 = np.concatenate([Fte, (bank_features(Xte, bank) - bmu) / bsd], axis=1)
            Dp = 1024
            C = (Ftr2.T @ Ftr2).astype(np.float64) / len(Ftr2)
            m = Ftr2.mean(0).astype(np.float64)
            C -= np.outer(m, m)
            w, v = np.linalg.eigh(C)
            comps = v[:, ::-1][:, :Dp].astype(np.float32)
            Ftr2 = Ftr2 @ comps; Fva2 = Fva2 @ comps; Fte2 = Fte2 @ comps
            pmu = Ftr2.mean(0); psd = Ftr2.std(0) + 1e-6
            Ftr2 = (Ftr2 - pmu) / psd; Fva2 = (Fva2 - pmu) / psd; Fte2 = (Fte2 - pmu) / psd
            np.savez_compressed(cache_f, Ftr=Ftr2, Fva=Fva2, Fte=Fte2)
        _STATCACHE[key] = {"Ftr": Ftr2, "ytr": ytr, "Fva": Fva2, "yva": yva,
                           "Fte": Fte2, "yte": yte, "pca": pca, "mu": mu, "sd": sd,
                           "nf": Ftr2.shape[1], "version": 4}
        return _STATCACHE[key]
    if version == 3:
        s = median_bandwidth(Ftr)
        Ztr = rff_lift(Ftr, s)
        mu2 = Ztr.mean(axis=0); sd2 = Ztr.std(axis=0) + 1e-6   # equalise gene influence
        Ftr = np.concatenate([Ftr, (Ztr - mu2) / sd2], axis=1)
        Fva = np.concatenate([Fva, (rff_lift(Fva, s) - mu2) / sd2], axis=1)
        Fte = np.concatenate([Fte, (rff_lift(Fte, s) - mu2) / sd2], axis=1)
    if augment > 0:
        assert version < 3, "augment is built for the v2 environment (no lift)"
        rng = np.random.default_rng(99)
        shifts = [(dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)
                  if (dy, dx) != (0, 0)]
        blocks, yblocks = [Ftr], [ytr]
        for k in range(augment):
            pick = rng.integers(0, len(shifts), size=len(Xtr))   # per-image shift
            Fa = np.empty_like(blocks[0][:len(Xtr)])
            for si, (dy, dx) in enumerate(shifts):
                m = pick == si
                if m.any():
                    Fa[m] = (stat_features(_shift(Xtr[m], dy, dx), pca) - mu) / sd
            blocks.append(Fa); yblocks.append(ytr)
        Ftr = np.concatenate(blocks); ytr = np.concatenate(yblocks)
    _STATCACHE[key] = {"Ftr": Ftr, "ytr": ytr, "Fva": Fva, "yva": yva,
                       "Fte": Fte, "yte": yte, "pca": pca, "mu": mu, "sd": sd,
                       "nf": Ftr.shape[1], "version": version}
    return _STATCACHE[key]


# --------------------------------------------------------------------------
# LAYER 1c (v4 environment) — EVOLVED DETECTOR BANK, the MANTIS recipe kept
# genome-pure: each detector is a 5x5 conv kernel genome + bias + an EVOLVED
# ACTIVATION gene from the 8-function catalog (the GENREG signature
# primitive). Survival condition per detector: Fisher class-separability of
# its pooled responses — a local, clean fitness; NO end-to-end accuracy, no
# gradients, and the classifier never enters the detector's landscape.
# Harvested across seeds, greedily decorrelated into a fixed bank; the bank's
# pooled responses then BECOME the environment (same status as HOG/PCA).
# --------------------------------------------------------------------------
DETBANK = os.path.join(ROOT, "demo", "mnist_detbank.pkl")
POOLS = ((3, 3), (4, 2), (2, 4))                 # mean pools -> 25 dims/detector


def _acts(x, a):
    """Activation catalog applied per detector: 0 relu, 1 abs, 2 sin, 3 cos,
    4 gaussian, 5 leaky, 6 square, 7 tanh."""
    if a == 0:
        return np.maximum(x, 0)
    if a == 1:
        return np.abs(x)
    if a == 2:
        return np.sin(x)
    if a == 3:
        return np.cos(x)
    if a == 4:
        return np.exp(-x * x)
    if a == 5:
        return np.where(x > 0, x, 0.1 * x)
    if a == 6:
        return x * x
    return np.tanh(x)


def _im2col5(X):
    """(N,28,28) -> (N, 24*24, 25) sliding 5x5 patches."""
    N = len(X)
    s0, s1, s2 = X.strides
    v = np.lib.stride_tricks.as_strided(X, (N, 24, 24, 5, 5), (s0, s1, s2, s1, s2))
    return np.ascontiguousarray(v.reshape(N, 576, 25))


def _pool_resp(resp):
    """(N,24,24) activated response -> (N,25) multi-shape mean pools."""
    N = len(resp)
    out = []
    for r, c in POOLS:
        hr, hc = 24 // r, 24 // c
        out.append(resp[:, :r * hr, :c * hc].reshape(N, r, hr, c, hc)
                   .mean(axis=(2, 4)).reshape(N, r * c))
    return np.concatenate(out, axis=1)


def _fisher(feats, y, nc=10):
    """(N,D) features, labels -> mean Fisher score across dims:
    between-class variance / within-class variance."""
    mu = feats.mean(0)
    bt = np.zeros(feats.shape[1]); wi = np.zeros(feats.shape[1])
    for c in range(nc):
        fc = feats[y == c]
        mc = fc.mean(0)
        bt += len(fc) * (mc - mu) ** 2
        wi += ((fc - mc) ** 2).sum(0)
    return float((bt / (wi + 1e-8)).mean())


def _pool_all(resp, act):
    """(N,24,24,P) raw responses + per-detector activation ids -> (P,N,25)
    pooled features, activations applied group-wise (8 calls, not P)."""
    N, _, _, P = resp.shape
    pooled = np.empty((P, N, 25), np.float32)
    for a in range(8):
        ids = np.where(act == a)[0]
        if len(ids) == 0:
            continue
        blk = _acts(resp[..., ids], a).transpose(3, 0, 1, 2).reshape(len(ids) * N, 24, 24)
        pooled[ids] = _pool_resp(blk).reshape(len(ids), N, 25)
    return pooled


def _fisher_all(pooled, y, nc=10):
    """(P,N,25) -> (P,) mean Fisher per detector, vectorised across the pop."""
    P, N, D = pooled.shape
    mu = pooled.mean(1)
    bt = np.zeros((P, D)); wi = np.zeros((P, D))
    for c in range(nc):
        fc = pooled[:, y == c, :]
        mc = fc.mean(1)
        bt += fc.shape[1] * (mc - mu) ** 2
        wi += ((fc - mc[:, None, :]) ** 2).sum(1)
    return (bt / (wi + 1e-8)).mean(1).astype(np.float32)


def evolve_detbank(n_bank=96, rounds=48, pop=48, gens=60, sub=2500, seed=7,
                   corr_cap=0.95, log=print, out=None, act_catalog=True):
    """Evolve the detector bank. Per round: one population of conv-kernel
    genomes climbs Fisher separability on a fixed balanced subsample (a
    deterministic landscape — the small-pool lesson doesn't apply because
    Fisher over 3k images is a statistic, not a fit). Champions harvested
    every 20 gens, then greedy selection by Fisher with a correlation cap
    keeps the bank DIVERSE, not 192 copies of the best. Saved to DETBANK."""
    Xtr, ytr = load_mnist()[0], load_mnist()[1]
    Xtr = deskew(Xtr)
    rng0 = np.random.default_rng(seed)
    harvest = []                                   # (fisher, K, b, act, pooled_sub)
    for rd in range(rounds):
        rs = np.random.default_rng(seed + 101 * rd)
        idx = np.concatenate([rs.choice(np.where(ytr == c)[0], sub // 10, replace=False)
                              for c in range(10)])
        Xs, ys = Xtr[idx], ytr[idx]
        P = _im2col5(Xs)                           # (n,576,25)
        Pf = P.reshape(-1, 25)
        dbf = None
        try:
            from genreg_mnist import evo_gpu
            if evo_gpu.HAS_GPU:
                dbf = evo_gpu.DetbankFitGPU(P, ys, 24, POOLS)
        except ImportError:
            pass
        K = (rs.standard_normal((pop, 25)) * 0.4).astype(np.float32)
        b = np.zeros(pop, np.float32)
        act = (rs.integers(0, 8, pop).astype(np.float32) if act_catalog
               else np.zeros(pop, np.float32))     # relu-only ablation
        sigma = np.full(pop, 0.08, np.float32)
        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8 if act_catalog \
                else np.zeros(pop, np.int64)
            if dbf is not None:
                fit = dbf(K, b, ai)
            else:
                resp = (Pf @ K.T + b).reshape(len(Xs), 24, 24, pop)
                fit = _fisher_all(_pool_all(resp, ai), ys)
            if gen % 20 == 0 or gen == gens:
                o = np.argsort(fit)[::-1][:8]
                for j in o:
                    harvest.append((float(fit[j]), K[j].copy(), float(b[j]),
                                    int(ai[j]), None))
            pd = {"K": K, "b": b, "act": act, "sigma": sigma}
            ga_step(pd, fit, rng0)
            K, b, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8               # activation gene: discrete catalog
        log(f"  [detbank] round {rd + 1}/{rounds}: best fisher "
            f"{max(h[0] for h in harvest):.3f}, harvested {len(harvest)}")
    # greedy diverse selection: by fisher, correlation cap vs already-picked,
    # ALL candidates re-pooled on one common reference set (cross-round
    # correlations are meaningless on different images)
    rr = np.random.default_rng(seed + 9999)
    ridx = np.concatenate([rr.choice(np.where(ytr == c)[0], 200, replace=False)
                           for c in range(10)])
    Pref = _im2col5(Xtr[ridx]).reshape(-1, 25)
    harvest.sort(key=lambda h: -h[0])
    picked, feats = [], []
    for f, K, b, a, _ in harvest:
        resp = (Pref @ K + b).reshape(len(ridx), 24, 24)
        v = _pool_resp(_acts(resp, a)).reshape(-1)
        v = (v - v.mean()) / (v.std() + 1e-8)
        if any(abs(float(v @ u) / len(v)) > corr_cap for u in feats):
            continue
        picked.append((K, b, a)); feats.append(v)
        if len(picked) >= n_bank:
            break
    bank = {"K": np.stack([p[0] for p in picked]),
            "b": np.array([p[1] for p in picked], np.float32),
            "act": np.array([p[2] for p in picked], np.int64)}
    out = out or DETBANK
    with open(out, "wb") as f:
        pickle.dump(bank, f)
    log(f"detector bank: {len(picked)} genomes (from {len(harvest)} harvested) "
        f"-> {out}")
    return bank


def bank_features(X, bank, chunk=2048):
    """(N,28,28) DESKEWED images -> (N, n_bank*25) pooled evolved features."""
    nb = len(bank["K"])
    out = np.empty((len(X), nb * 25), np.float32)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        P = _im2col5(Xc).reshape(-1, 25)
        resp = (P @ bank["K"].T + bank["b"]).reshape(len(Xc), 24, 24, nb)
        for j in range(nb):
            out[lo:lo + len(Xc), j * 25:(j + 1) * 25] = \
                _pool_resp(_acts(resp[..., j], int(bank["act"][j])))
    return out


# --------------------------------------------------------------------------
# Shared GA mechanics — same machinery as wordpipe.ga_step (tournament +
# elitism + energy starvation + self-adaptive sigma), copied so the image
# program carries no text-corpus dependency.
# --------------------------------------------------------------------------
def ga_step(params, fit, rng, elite_frac=0.1, tourn_k=4, starve_frac=0.08,
            sigma_lo=5e-3, sigma_hi=0.4, mag_scale=False):
    """`mag_scale`: perturb each gene proportionally to ITS OWN magnitude
    (plus a 5%-of-mean floor so near-zero genes can still move). A single
    global sigma only explores efficiently near one weight scale; scaled to
    the gene, large weights (strong features) and small weights (fine
    distinctions) each get a meaningful step. sigma then means RELATIVE
    step size."""
    P = len(fit)
    order = np.argsort(fit)[::-1]
    n_elite = max(1, int(round(P * elite_frac)))
    n_starve = int(round(P * starve_frac))
    elite = order[:n_elite]
    alive = order[:P - n_starve] if n_starve > 0 else order
    n_child = P - n_elite
    parents = np.empty(n_child, np.int64)
    for i in range(n_child):
        picks = alive[rng.integers(0, len(alive), size=tourn_k)]
        parents[i] = picks[np.argmax(fit[picks])]
    sigma = params["sigma"]
    csig = sigma[parents] * np.exp(0.2 * rng.standard_normal(n_child).astype(np.float32))
    csig = np.clip(csig, sigma_lo, sigma_hi)
    new = {}
    for name, arr in params.items():
        if name == "sigma":
            new["sigma"] = np.concatenate([sigma[elite].copy(), csig])
            continue
        keep = arr[elite].copy()
        child = arr[parents].copy()
        shape = (n_child,) + (1,) * (arr.ndim - 1)
        step = rng.standard_normal(child.shape).astype(np.float32) * csig.reshape(shape)
        if mag_scale:
            mag = np.abs(child)
            floor = 0.05 * mag.reshape(n_child, -1).mean(1).reshape(shape) + 1e-8
            step = step * (mag + floor)
        child += step
        new[name] = np.concatenate([keep, child])
    params.clear()
    params.update(new)
    return int(order[0])


# --------------------------------------------------------------------------
# LAYER 2a — DETECTOR genomes (one per digit). Genome = linear head w (NF) +
# bias. score = F @ w + b. Soft BCE fitness on balanced pos/neg minibatches.
# --------------------------------------------------------------------------
class LinearPop:
    def __init__(self, pop, nf, seed):
        rng = np.random.default_rng(seed)
        self.w = (rng.standard_normal((pop, nf)) * (1.0 / np.sqrt(nf))).astype(np.float32)
        self.b = np.zeros(pop, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def logits(self, F):                            # (N,nf) -> (P,N)
        return self.w @ F.T + self.b[:, None]

    def fitness(self, Fp, Fn, l2=0.0):
        """Mean log-prob: positives toward 1, negatives toward 0 (soft, dense).
        `l2` = landscape pressure against weight mass — with a lifted (high-dim)
        environment, unpenalised genes chase minibatch noise (val degrades while
        train climbs); the cost makes generalising directions the only ones
        worth their magnitude."""
        zp = np.clip(self.logits(Fp), -30, 30)
        zn = np.clip(self.logits(Fn), -30, 30)
        lp = -np.log1p(np.exp(-zp)).mean(1)          # log sigmoid(zp)
        ln = -np.log1p(np.exp(zn)).mean(1)           # log (1 - sigmoid(zn))
        acc = ((zp > 0).mean(1) + (zn < 0).mean(1)) / 2
        fit = lp + ln
        if l2 > 0:
            fit = fit - l2 * (self.w * self.w).sum(1)
        return fit, acc

    def champion(self, idx):
        return np.concatenate([self.w[idx], [self.b[idx]]]).astype(np.float32)


def _train_binary(Fp_tr, Fn_tr, Fp_va, Fn_va, name, gens=1200, pop=200,
                  minibatch=256, seed=7, log=print, log_every=200, warm=None,
                  l2=0.0, gpu=True):
    """Shared trainer for detector + pairwise genomes: linear head, balanced
    minibatches, champion picked on held-out fitness. `warm` = a prior champion
    (nf_old+1 vector, nf_old <= nf) to bootstrap the population from — the
    GENREG two-phase rule: never re-learn a proven solution from random init.
    New dims (the lifted block) start at 0; sigma starts low so mutation
    refines instead of destroying."""
    nf = Fp_tr.shape[1]
    rng = np.random.default_rng(seed)
    popn = LinearPop(pop, nf, seed)
    if warm is not None:
        w0 = np.zeros(nf, np.float32)
        w0[:len(warm) - 1] = warm[:-1]
        popn.w[:] = w0[None] + rng.standard_normal(popn.w.shape).astype(np.float32) * 0.005
        popn.w[0] = w0                       # one exact copy of the proven genome
        popn.b[:] = warm[-1]
        popn.sigma[:] = 0.02
    bf = None
    if gpu:
        from genreg_mnist import evo_gpu
        if evo_gpu.HAS_GPU:
            bf = evo_gpu.BinaryFitGPU(Fp_tr, Fn_tr)
    best_fit, best_acc, champ = -1e9, 0.0, None
    for gen in range(1, gens + 1):
        ip = rng.integers(0, len(Fp_tr), size=minibatch)
        inn = rng.integers(0, len(Fn_tr), size=minibatch)
        fit, _ = bf(popn.w, popn.b, ip, inn, l2=l2) if bf is not None \
            else popn.fitness(Fp_tr[ip], Fn_tr[inn], l2=l2)
        pd = {"w": popn.w, "b": popn.b, "sigma": popn.sigma}
        ga_step(pd, fit, rng)
        popn.w, popn.b, popn.sigma = pd["w"], pd["b"], pd["sigma"]
        if gen % log_every == 0 or gen == 1:
            vfit, vacc = popn.fitness(Fp_va, Fn_va)
            if float(vfit[0]) > best_fit:
                best_fit = float(vfit[0]); best_acc = float(vacc[0])
                champ = popn.champion(0)
            log(f"  [{name}] gen {gen}: val_logprob={vfit[0]:.4f} val_acc={vacc[0]:.4f}")
    return champ, round(best_acc, 4)


def train_detectors(gens=1200, pop=200, seed=7, log=print, D=None):
    """Evolve the 10 one-vs-rest detector genomes. GATE per digit: held-out
    balanced acc must beat 0.5 by a wide margin (it does — this is the easy layer)."""
    D = D if D is not None else build_features()
    Ftr, ytr, Fva, yva = D["Ftr"], D["ytr"], D["Fva"], D["yva"]
    dets, accs = {}, {}
    for d in range(10):
        Fp, Fn = Ftr[ytr == d], Ftr[ytr != d]
        Vp, Vn = Fva[yva == d], Fva[yva != d]
        champ, acc = _train_binary(Fp, Fn, Vp, Vn, f"det{d}", gens=gens, pop=pop,
                                   seed=seed + d, log=log)
        dets[d] = champ; accs[d] = acc
        log(f"[detector {d}] done: balanced val_acc={acc}")
    return {"det": dets, "det_val_acc": accs}


# --------------------------------------------------------------------------
# LAYER 2b — PAIRWISE disambiguator genomes ("4 or 9?"), one per digit pair,
# trained only on those two digits. positive = the smaller digit of the pair.
# --------------------------------------------------------------------------
def train_pairwise(gens=800, pop=150, seed=7, log=print, D=None):
    D = D if D is not None else build_features()
    Ftr, ytr, Fva, yva = D["Ftr"], D["ytr"], D["Fva"], D["yva"]
    pairs, accs = {}, {}
    for a in range(10):
        for b in range(a + 1, 10):
            Fp, Fn = Ftr[ytr == a], Ftr[ytr == b]
            Vp, Vn = Fva[yva == a], Fva[yva == b]
            champ, acc = _train_binary(Fp, Fn, Vp, Vn, f"pw{a}{b}", gens=gens,
                                       pop=pop, seed=seed + 10 * a + b, log=log,
                                       log_every=400)
            pairs[(a, b)] = champ; accs[f"{a}v{b}"] = acc
        log(f"[pairwise {a}v*] done")
    return {"pairs": pairs, "pair_val_acc": accs}


# --------------------------------------------------------------------------
# LAYER 3 — OUTPUT mixer genome over the 10 detector logits.
# --------------------------------------------------------------------------
def det_logits(dets, F):
    """(N, NF) -> (N, 10) raw detector logits from the frozen champions."""
    W = np.stack([dets[d][:-1] for d in range(10)], axis=1)   # (NF, 10)
    b = np.array([dets[d][-1] for d in range(10)], np.float32)
    return F @ W + b


class MixerPop:
    """Genome = W (10,10) + b (10). logits = s @ W + b; soft fitness = mean
    log-softmax prob of the true digit."""

    def __init__(self, pop, seed):
        rng = np.random.default_rng(seed)
        eye = np.eye(10, dtype=np.float32)
        self.W = (eye[None] + rng.standard_normal((pop, 10, 10)).astype(np.float32) * 0.05)
        self.b = np.zeros((pop, 10), np.float32)
        self.sigma = np.full(pop, 0.05, np.float32)

    def logits(self, S):                            # (N,10) -> (P,N,10)
        return np.einsum("nd,pde->pne", S, self.W) + self.b[:, None, :]

    def fitness(self, S, y):
        z = self.logits(S)
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ch = np.take_along_axis(logp, y[None, :, None].repeat(len(self.W), 0), axis=2)[..., 0]
        acc = (z.argmax(-1) == y[None]).mean(1)
        return ch.mean(1), acc

    def champion(self, idx):
        return (self.W[idx].copy(), self.b[idx].copy())


def train_mixer(dets, gens=1500, pop=200, minibatch=1024, seed=7, log=print, D=None):
    """GATE: held-out top-1 must beat raw argmax over the detector logits."""
    D = D if D is not None else build_features()
    Str = det_logits(dets, D["Ftr"]); Sva = det_logits(dets, D["Fva"])
    ytr, yva = D["ytr"], D["yva"]
    base_acc = float((Sva.argmax(1) == yva).mean())
    rng = np.random.default_rng(seed)
    popn = MixerPop(pop, seed)
    best_fit, best_acc, champ = -1e9, 0.0, None
    for gen in range(1, gens + 1):
        s = rng.integers(0, len(Str), size=minibatch)
        fit, _ = popn.fitness(Str[s], ytr[s])
        pd = {"W": popn.W, "b": popn.b, "sigma": popn.sigma}
        ga_step(pd, fit, rng)
        popn.W, popn.b, popn.sigma = pd["W"], pd["b"], pd["sigma"]
        if gen % 100 == 0 or gen == 1:
            vfit, vacc = popn.fitness(Sva, yva)
            if float(vfit[0]) > best_fit:
                best_fit = float(vfit[0]); best_acc = float(vacc[0])
                champ = popn.champion(0)
            log(f"  [mixer] gen {gen}: val_logprob={vfit[0]:.4f} "
                f"val_acc={vacc[0]:.4f} (argmax base {base_acc:.4f})")
    return {"mixer": champ, "mixer_val_acc": round(best_acc, 4),
            "argmax_val_acc": round(base_acc, 4)}


# --------------------------------------------------------------------------
# LAYER 2/3 JOINT REFINE — the GENREG two-phase rule applied to the whole
# linear stack. The detector columns + mixer fold algebraically into ONE
# 677x10 linear genome: F@Wd@Wm + (bd@Wm + bm). Phase 1 bred the parts
# separately (their own survival conditions); phase 2 evolves the folded
# genome JOINTLY on the composed objective (mean log-softmax of the true
# digit) — because argmax-over-10 is a joint decision the one-vs-rest
# training never optimised. Warm-started, low sigma, big minibatches; the
# champion is tracked on val so the worst case is zero regression.
# --------------------------------------------------------------------------
def fold_stack(champs):
    """(det champions + mixer) -> (W0 (nf,10), b0 (10,)) single linear head.
    If a jointly-refined head already exists it IS the folded stack, further
    evolved — continue from it (bootstrap rule), don't re-fold the parts."""
    if "joint" in champs:
        return champs["joint"][0].copy(), champs["joint"][1].copy()
    dets = champs["det"]
    Wd = np.stack([dets[d][:-1] for d in range(10)], axis=1)   # (nf,10)
    bd = np.array([dets[d][-1] for d in range(10)], np.float32)
    if "mixer" in champs:
        Wm, bm = champs["mixer"]
        return (Wd @ Wm).astype(np.float32), (bd @ Wm + bm).astype(np.float32)
    return Wd.astype(np.float32), bd


class JointPop:
    def __init__(self, pop, nf, W0, b0, seed):
        rng = np.random.default_rng(seed)
        self.W = W0[None].repeat(pop, 0) + \
            rng.standard_normal((pop, nf, 10)).astype(np.float32) * 0.002
        self.W[0] = W0                                # one exact proven genome
        self.b = b0[None].repeat(pop, 0)
        self.sigma = np.full(pop, 0.005, np.float32)

    def fitness(self, F, y, l2=0.0):
        """Mean log-softmax of the true digit, minus an L2 weight cost. The cost
        is the generalisation half of the landscape: unpenalised, mutation's
        random walk grows weight norms and the population climbs train NLL
        while val decays (measured — same L2 the closed-form ceiling needed)."""
        P = len(self.W)
        nf = self.W.shape[1]
        # one big GEMM instead of einsum: (N,nf) @ (nf, P*10) — MKL-threaded,
        # ~an order of magnitude faster at this pool size
        z = (F @ self.W.transpose(1, 0, 2).reshape(nf, P * 10)) \
            .reshape(len(F), P, 10).transpose(1, 0, 2) + self.b[:, None, :]
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ch = np.take_along_axis(logp, y[None, :, None].repeat(P, 0), axis=2)[..., 0]
        fit = ch.mean(1)
        if l2 > 0:
            fit = fit - l2 * (self.W * self.W).reshape(P, -1).sum(1)
        return fit


class JointQPop(JointPop):
    """JointPop + an evolved PRECISION gene per output neuron (bits, 3..16;
    16 ~ fp16, 3 ~ int3). Fitness always evaluates the QUANTIZED weights
    (symmetric linear quant on each neuron's dynamic range, straight-through
    on the latent float genes), and precision carries a fitness cost — so a
    neuron keeps high bits only where fine distinctions pay for themselves
    and the model self-compresses during training."""

    def __init__(self, pop, nf, W0, b0, seed, bits0=12.0):
        super().__init__(pop, nf, W0, b0, seed)
        self.kbits = np.full((pop, 10), bits0, np.float32)

    def quantized(self):
        k = np.clip(np.round(self.kbits), 3, 16)            # (P,10)
        q = (2.0 ** (k - 1) - 1)[:, None, :]                # levels per neuron
        s = np.abs(self.W).max(axis=1, keepdims=True) + 1e-8
        Wq = np.round(self.W / s * q) / q * s
        return Wq.astype(np.float32), k

    def fitness(self, F, y, l2=0.0, bit_cost=0.0):
        Wq, k = self.quantized()
        P = len(Wq); nf = Wq.shape[1]
        z = (F @ Wq.transpose(1, 0, 2).reshape(nf, P * 10)) \
            .reshape(len(F), P, 10).transpose(1, 0, 2) + self.b[:, None, :]
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ch = np.take_along_axis(logp, y[None, :, None].repeat(P, 0), axis=2)[..., 0]
        fit = ch.mean(1)
        if l2 > 0:
            fit = fit - l2 * (Wq * Wq).reshape(P, -1).sum(1)
        if bit_cost > 0:
            fit = fit - bit_cost * (k.mean(1) / 32.0)
        return fit


def train_joint(champs, gens=6000, pop=60, minibatch=16384, seed=7, log=print,
                D=None, rotate=0, l2=1e-4, mag_scale=True, warm_init=None,
                gpu=True):
    """Jointly refine the folded linear stack. GATE: held-out top-1 must beat
    the unfolded det+mixer stack (it starts there, so no regression possible).
    The fitness pool is ONE fixed `minibatch`-sized subset for the whole run
    (rotate=0): per-gen resampling drowned the refinement signal in noise, and
    25-gen rotation produced SERIAL PER-BATCH OVERFITTING (population climbs
    each 4096 batch, rotation invalidates the climb; |W|^2 grew straight
    through the L2 penalty — both measured, see CHANGELOG 2026-07-08). A fixed
    16k pool is deterministic AND too large for a 6,770-param linear genome to
    overfit (~2.4 samples/param — the closed-form fit generalises at this
    ratio). Low sigma floor for the same reason: refinement steps must stay
    smaller than the remaining signal. L2 keeps the random walk's norm growth
    priced in."""
    D = D if D is not None else build_features()
    Ftr, ytr, Fva, yva = D["Ftr"], D["ytr"], D["Fva"], D["yva"]
    W0, b0 = warm_init if warm_init is not None else fold_stack(champs)

    def vacc(W, b):
        return float(((Fva @ W + b).argmax(1) == yva).mean())

    base = vacc(W0, b0)
    rng = np.random.default_rng(seed)
    popn = JointPop(pop, Ftr.shape[1], W0, b0, seed)
    best_acc, champ = base, (W0.copy(), b0.copy())
    if minibatch <= 0 or minibatch >= len(Ftr):      # full-train landscape
        s = np.arange(len(Ftr))
        log(f"  [joint] fitness pool = FULL train ({len(Ftr)})")
    else:
        s = rng.integers(0, len(Ftr), size=minibatch)
    jf, use_gpu = None, False
    if gpu:
        from genreg_mnist import evo_gpu
        if evo_gpu.HAS_GPU:
            use_gpu = True
            jf = evo_gpu.JointFitGPU(Ftr[s], ytr[s])
            log("  [joint] fitness on GPU")
    for gen in range(1, gens + 1):
        if rotate > 0 and gen % rotate == 0:
            s = rng.integers(0, len(Ftr), size=minibatch)
            if use_gpu:
                from genreg_mnist import evo_gpu
                jf = evo_gpu.JointFitGPU(Ftr[s], ytr[s])
        fit = jf(popn.W, popn.b, l2=l2) if jf is not None \
            else popn.fitness(Ftr[s], ytr[s], l2=l2)
        pd = {"W": popn.W, "b": popn.b, "sigma": popn.sigma}
        ga_step(pd, fit, rng, sigma_lo=5e-4, mag_scale=mag_scale)
        popn.W, popn.b, popn.sigma = pd["W"], pd["b"], pd["sigma"]
        if gen % 100 == 0 or gen == 1:
            a = vacc(popn.W[0], popn.b[0])
            if a > best_acc:
                best_acc = a; champ = (popn.W[0].copy(), popn.b[0].copy())
            if gen % 500 == 0 or gen == 1:
                wn = float((popn.W[0] ** 2).sum())
                log(f"  [joint] gen {gen}: val_acc={a:.4f} best={best_acc:.4f} "
                    f"fit0={fit[0]:.4f} |W|^2={wn:.1f} (base {base:.4f})")
    return {"joint": champ, "joint_val_acc": round(best_acc, 4),
            "joint_base_val_acc": round(base, 4)}


def train_joint_q(champs, gens=4000, pop=60, seed=7, log=print, D=None,
                  l2=1e-4, bit_cost=0.01, bits0=12.0):
    """Round-6 trainer: full-train landscape + magnitude-scaled mutation +
    evolved per-neuron precision (JointQPop). The champion saved is the
    QUANTIZED model (what you'd deploy), with its per-neuron bit widths."""
    D = D if D is not None else build_features()
    Ftr, ytr, Fva, yva = D["Ftr"], D["ytr"], D["Fva"], D["yva"]
    W0, b0 = fold_stack(champs)

    def vacc(W, b):
        return float(((Fva @ W + b).argmax(1) == yva).mean())

    rng = np.random.default_rng(seed)
    popn = JointQPop(pop, Ftr.shape[1], W0, b0, seed, bits0=bits0)
    Wq0, k0 = popn.quantized()
    base = vacc(Wq0[0], b0)
    best_acc, champ, champ_bits = base, (Wq0[0].copy(), b0.copy()), k0[0].copy()
    log(f"  [jointq] fitness pool = FULL train ({len(Ftr)}), bits0={bits0} "
        f"(quantised warm-start val {base:.4f})")
    for gen in range(1, gens + 1):
        fit = popn.fitness(Ftr, ytr, l2=l2, bit_cost=bit_cost)
        pd = {"W": popn.W, "b": popn.b, "kbits": popn.kbits, "sigma": popn.sigma}
        ga_step(pd, fit, rng, sigma_lo=5e-4, mag_scale=True)
        popn.W, popn.b, popn.kbits, popn.sigma = pd["W"], pd["b"], pd["kbits"], pd["sigma"]
        if gen % 100 == 0 or gen == 1:
            Wq, k = popn.quantized()
            a = vacc(Wq[0], popn.b[0])
            if a > best_acc:
                best_acc = a
                champ, champ_bits = (Wq[0].copy(), popn.b[0].copy()), k[0].copy()
            if gen % 500 == 0 or gen == 1:
                log(f"  [jointq] gen {gen}: val_acc={a:.4f} best={best_acc:.4f} "
                    f"fit0={fit[0]:.4f} bits={np.round(k[0]).astype(int).tolist()} "
                    f"(base {base:.4f})")
    kb = float((677 * champ_bits).sum() / 8 / 1024)
    return {"joint": champ, "joint_val_acc": round(best_acc, 4),
            "joint_base_val_acc": round(base, 4),
            "joint_bits": champ_bits.tolist(),
            "joint_kb": round(kb, 2)}


# --------------------------------------------------------------------------
# Inference / evaluation over any subset of layers
# --------------------------------------------------------------------------
def centroid_baseline(version=2):
    """No-evolution floor: nearest class-centroid in the stats space."""
    D = build_features(version)
    cents = np.stack([D["Ftr"][D["ytr"] == d].mean(0) for d in range(10)])
    d2 = ((D["Fte"][:, None, :] - cents[None]) ** 2).sum(-1)
    return float((d2.argmin(1) == D["yte"]).mean())


def predict(champs, F, use_mixer=True, use_pairs=True, pair_margin=3.0,
            use_joint=True):
    """Full pipeline prediction on features F -> (pred (N,), logits (N,10)).
    The joint-refined head (if trained + enabled) replaces det+mixer — it IS
    det+mixer, folded and evolved further. Pairwise referees fire only when
    the top-2 logits are within `pair_margin` (the confusable zone they were
    bred for)."""
    if "det" not in champs:
        use_joint = True                          # v4 champs: joint IS the stack
    if use_joint and "joint" in champs:
        Wj, bj = champs["joint"]
        L = F @ Wj + bj
    else:
        S = det_logits(champs["det"], F)
        if use_mixer and "mixer" in champs:
            W, b = champs["mixer"]
            L = S @ W + b
        else:
            L = S
    pred = L.argmax(1)
    if use_pairs and champs.get("pairs"):
        srt = np.argsort(L, axis=1)
        top1, top2 = srt[:, -1], srt[:, -2]
        margin = np.take_along_axis(L, top1[:, None], 1)[:, 0] \
            - np.take_along_axis(L, top2[:, None], 1)[:, 0]
        close = margin < pair_margin
        for i in np.where(close)[0]:
            a, b2 = int(min(top1[i], top2[i])), int(max(top1[i], top2[i]))
            wb = champs["pairs"].get((a, b2))
            if wb is None:
                continue
            z = float(F[i] @ wb[:-1] + wb[-1])       # >0 -> a, <0 -> b2
            pred[i] = a if z > 0 else b2
    return pred, L


def evaluate(champs, split="test", use_mixer=True, use_pairs=True, pair_margin=3.0,
             use_joint=True):
    """Accuracy + confusion matrix on val or test for a layer subset."""
    D = build_features(champs.get("feat_version", 2) if isinstance(champs, dict) else 2)
    F, y = (D["Fte"], D["yte"]) if split == "test" else (D["Fva"], D["yva"])
    pred, _ = predict(champs, F, use_mixer, use_pairs, pair_margin, use_joint)
    acc = float((pred == y).mean())
    conf = np.zeros((10, 10), np.int64)
    np.add.at(conf, (y, pred), 1)
    return {"acc": round(acc, 4), "confusion": conf.tolist(), "n": len(y)}


def tune_pair_margin(champs, log=print):
    """Pick the pairwise-referee margin on the VALIDATION split (never test)."""
    best_m, best_acc = 0.0, evaluate(champs, "val", True, False)["acc"]
    for m in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        acc = evaluate(champs, "val", True, True, m)["acc"]
        log(f"  margin {m}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m, best_acc


# --------------------------------------------------------------------------
# Full battery
# --------------------------------------------------------------------------
def run_all(det_gens=1200, pair_gens=800, mixer_gens=1500, seed=7, log=print,
            augment=0):
    t0 = time.time()
    log("=== MNIST-Pipe battery ===")
    log("building statistics layer (features are the environment)…")
    D = build_features(2, augment)
    log(f"stats layer v{D['version']}: {D['nf']} fixed dims "
        f"(deskew + zones + profiles + grad-hist + PCA), augment={augment} "
        f"-> train pool {len(D['ytr'])}")
    log(f"centroid baseline (no evolution): {centroid_baseline():.4f}")

    log("--- LAYER 2a: detector genomes (10x one-vs-rest) ---")
    champs = train_detectors(gens=det_gens, seed=seed, log=log, D=D)
    log("--- LAYER 3: output mixer ---")
    champs.update(train_mixer(champs["det"], gens=mixer_gens, seed=seed, log=log, D=D))
    log("--- LAYER 2b: pairwise disambiguators (45x one-vs-one) ---")
    champs.update(train_pairwise(gens=pair_gens, seed=seed, log=log, D=D))

    log("--- gating on validation ---")
    m, vacc = tune_pair_margin(champs, log=log)
    champs["pair_margin"] = m
    log(f"chosen pair_margin={m} (val_acc={vacc:.4f})")

    res = {
        "centroid_test": centroid_baseline(),
        "det_argmax_test": evaluate(champs, "test", False, False)["acc"],
        "mixer_test": evaluate(champs, "test", True, False)["acc"],
        "full_test": evaluate(champs, "test", True, True, m)["acc"],
    }
    champs["results"] = res
    champs["feat_version"] = D["version"]
    champs["augment"] = augment
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(champs, f)
    log(f"saved champions -> {CACHE}")
    log(f"TEST: centroid {res['centroid_test']:.4f} | detectors(argmax) "
        f"{res['det_argmax_test']:.4f} | +mixer {res['mixer_test']:.4f} | "
        f"+pairwise {res['full_test']:.4f}   ({time.time() - t0:.0f}s)")
    return res


def run_v4(det_gens=3000, joint_gens=6000, pair_gens=1500, seed=7, log=print):
    """The full battery on the EVOLVED-DETECTOR environment (feat v4):
    detectors bred in the new space -> folded as the joint warm start ->
    joint refine on the full-train landscape (mag-scaled mutation) ->
    pairwise referees -> margin gate on val -> one-shot test."""
    t0 = time.time()
    log("=== MNIST-Pipe v4 battery (evolved-detector environment) ===")
    D = build_features(4)
    log(f"environment v4: {D['nf']} dims (66-genome evolved bank + built stats, PCA)")
    log(f"centroid floor: {centroid_baseline(4):.4f}")
    if det_gens > 0:
        log("--- detectors (10x one-vs-rest, v4 space) ---")
        champs = train_detectors(gens=det_gens, seed=seed, log=log, D=D)
    else:
        champs = {}
    # centroid head as the joint warm start — a pure train statistic
    # (W = class means, b = -|mu|^2/2), and a far better basin entry (0.9872)
    # than the folded one-vs-rest logits (0.9494, argmax-uncalibrated).
    mu_c = np.stack([D["Ftr"][D["ytr"] == c].mean(0) for c in range(10)], axis=1)
    warm = (mu_c.astype(np.float32), (-0.5 * (mu_c ** 2).sum(0)).astype(np.float32))
    log("--- joint refine (centroid warm start, full-train landscape) ---")
    champs.update(train_joint(champs, gens=joint_gens, seed=seed, log=log,
                              D=D, minibatch=0, warm_init=warm))
    log("--- pairwise referees (45x one-vs-one, v4 space) ---")
    champs.update(train_pairwise(gens=pair_gens, seed=seed, log=log, D=D))
    champs["feat_version"] = 4
    log("--- gating on validation ---")
    m, vacc = tune_pair_margin(champs, log=log)
    champs["pair_margin"] = m
    log(f"chosen pair_margin={m} (val_acc={vacc:.4f})")
    res = {
        "centroid_test": centroid_baseline(4),
        "joint_test": evaluate(champs, "test", True, False)["acc"],
        "full_test": evaluate(champs, "test", True, True, m)["acc"],
    }
    champs["results"] = res
    with open(CACHE, "wb") as f:
        pickle.dump(champs, f)
    log(f"saved champions -> {CACHE}")
    log(f"TEST: centroid {res['centroid_test']:.4f} | joint {res['joint_test']:.4f} "
        f"| +pairwise {res['full_test']:.4f}   ({time.time() - t0:.0f}s)")
    return res


def run_joint_refine(joint_gens=6000, seed=7, log=print, pool=16384, quant=False):
    """Round-3 entry: bootstrap from the saved champions (never re-learn),
    jointly refine the folded det+mixer head, re-gate the pairwise margin on
    val, one-shot test eval, save."""
    t0 = time.time()
    with open(CACHE, "rb") as f:
        champs = pickle.load(f)
    log("=== MNIST-Pipe joint refine (two-phase, bootstrapped) ===")
    D = build_features()
    if quant:
        champs.update(train_joint_q(champs, gens=joint_gens, seed=seed, log=log, D=D))
    else:
        champs.update(train_joint(champs, gens=joint_gens, seed=seed, log=log, D=D,
                                  minibatch=pool))
    log("--- re-gating pairwise margin on validation (joint head) ---")
    m, vacc = tune_pair_margin(champs, log=log)
    champs["pair_margin"] = m
    log(f"chosen pair_margin={m} (val_acc={vacc:.4f})")
    res = champs.get("results", {})
    res.update({
        "joint_test": evaluate(champs, "test", True, False, use_joint=True)["acc"],
        "full_test": evaluate(champs, "test", True, True, m, use_joint=True)["acc"],
    })
    champs["results"] = res
    with open(CACHE, "wb") as f:
        pickle.dump(champs, f)
    log(f"saved champions -> {CACHE}")
    log(f"TEST: joint {res['joint_test']:.4f} | +pairwise {res['full_test']:.4f}"
        f"   ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-gens", type=int, default=1200)
    ap.add_argument("--pair-gens", type=int, default=800)
    ap.add_argument("--mixer-gens", type=int, default=1500)
    ap.add_argument("--joint-gens", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--augment", type=int, default=0,
                    help="extra shifted train copies (environment enrichment)")
    ap.add_argument("--joint-only", action="store_true",
                    help="round-3: joint refine from saved champions")
    ap.add_argument("--joint-pool", type=int, default=16384,
                    help="joint fitness pool size; 0 = full train set")
    ap.add_argument("--quant", action="store_true",
                    help="evolve per-neuron precision (self-compressing joint head)")
    ap.add_argument("--v4", action="store_true",
                    help="full battery on the evolved-detector environment")
    args = ap.parse_args()
    if args.v4:
        run_v4(args.det_gens, args.joint_gens, args.pair_gens, args.seed)
    elif args.joint_only:
        run_joint_refine(args.joint_gens, args.seed, pool=args.joint_pool,
                         quant=args.quant)
    else:
        run_all(args.det_gens, args.pair_gens, args.mixer_gens, args.seed,
                augment=args.augment)
