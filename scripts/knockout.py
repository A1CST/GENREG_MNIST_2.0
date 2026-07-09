"""Detector knockout analysis on the shipped model.

Rebuilds the v4 environment with the projection intact, then knocks out
detectors by mean-imputing their 25 pooled feature columns (standardized
columns set to zero) before the PCA projection, and re-evaluates the frozen
classifier head. No parameter is retrained; this is post-hoc analysis of the
final model.

Reports: every single-detector knockout, the top-10-by-Fisher group
knockout, a random-10 group knockout, and cumulative knockout by Fisher
rank. Writes results/knockout.json.

    python scripts/knockout.py
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_mnist import mnist_pipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_projected_env(bank):
    """Recompute the pre-PCA pipeline so knockouts can intervene before the
    projection. Deterministic; matches build_features(version=4)."""
    Xtr, ytr, Xva, yva, Xte, yte = mp.load_mnist()
    Xtr, Xva, Xte = mp.deskew(Xtr), mp.deskew(Xva), mp.deskew(Xte)
    pca = mp.build_pca(Xtr)
    Str = mp.stat_features(Xtr, pca)
    mu = Str.mean(0); sd = Str.std(0) + 1e-6
    Str = (Str - mu) / sd
    Sva = (mp.stat_features(Xva, pca) - mu) / sd
    Ste = (mp.stat_features(Xte, pca) - mu) / sd
    Btr = mp.bank_features(Xtr, bank)
    bmu = Btr.mean(0); bsd = Btr.std(0) + 1e-6
    Ctr = np.concatenate([Str, (Btr - bmu) / bsd], axis=1)
    Cva = np.concatenate([Sva, (mp.bank_features(Xva, bank) - bmu) / bsd], axis=1)
    Cte = np.concatenate([Ste, (mp.bank_features(Xte, bank) - bmu) / bsd], axis=1)
    C = (Ctr.T @ Ctr).astype(np.float64) / len(Ctr)
    m = Ctr.mean(0).astype(np.float64)
    C -= np.outer(m, m)
    w, v = np.linalg.eigh(C)
    comps = v[:, ::-1][:, :1024].astype(np.float32)
    P = Ctr @ comps
    pmu = P.mean(0); psd = P.std(0) + 1e-6
    return {"Cva": Cva, "Cte": Cte, "yva": yva, "yte": yte,
            "comps": comps, "pmu": pmu, "psd": psd, "n_stats": Str.shape[1]}


def acc_with_knockout(env, head, drop, split="test"):
    C = (env["Cte"] if split == "test" else env["Cva"]).copy()
    y = env["yte"] if split == "test" else env["yva"]
    for j in drop:
        lo = env["n_stats"] + j * 25
        C[:, lo:lo + 25] = 0.0                    # standardized 0 = train mean
    F = ((C @ env["comps"]) - env["pmu"]) / env["psd"]
    W, b = head
    return float(((F @ W + b).argmax(1) == y).mean())


if __name__ == "__main__":
    with open(os.path.join(ROOT, "checkpoints", "mnist_detbank.pkl"), "rb") as f:
        bank = pickle.load(f)
    with open(os.path.join(ROOT, "checkpoints",
                           "mnist_genomes_v4_9903.pkl"), "rb") as f:
        head = pickle.load(f)["joint"]
    nb = len(bank["K"])
    print(f"rebuilding projected environment ({nb} detectors)...", flush=True)
    env = build_projected_env(bank)

    base = acc_with_knockout(env, head, [])
    print(f"baseline (no knockout): test {base:.4f}", flush=True)

    singles = []
    for j in range(nb):
        a = acc_with_knockout(env, head, [j])
        singles.append(round(a, 4))
    singles = np.array(singles)
    worst = int(np.argmin(singles))
    print(f"single knockouts: mean {singles.mean():.4f}, "
          f"min {singles.min():.4f} (detector {worst}), "
          f"max {singles.max():.4f}", flush=True)

    # bank is stored in Fisher-rank order (greedy selection sorted by Fisher)
    top10 = acc_with_knockout(env, head, list(range(10)))
    rng = np.random.default_rng(0)
    rand10 = [acc_with_knockout(env, head,
                                rng.choice(nb, 10, replace=False).tolist())
              for _ in range(5)]
    print(f"top-10-by-Fisher knockout: {top10:.4f}", flush=True)
    print(f"random-10 knockout (5 draws): {np.mean(rand10):.4f} "
          f"± {np.std(rand10):.4f}", flush=True)

    cumulative = []
    for k in (0, 5, 10, 20, 33, 50, 66):
        cumulative.append({"k": k,
                           "acc": round(acc_with_knockout(env, head,
                                                          list(range(k))), 4)})
    print("cumulative by Fisher rank:", cumulative, flush=True)

    out = {"baseline": round(base, 4), "singles": singles.tolist(),
           "single_mean": round(float(singles.mean()), 4),
           "single_min": round(float(singles.min()), 4),
           "single_min_detector": worst,
           "top10_knockout": round(top10, 4),
           "random10_knockout_mean": round(float(np.mean(rand10)), 4),
           "random10_knockout_std": round(float(np.std(rand10)), 4),
           "cumulative_by_fisher_rank": cumulative}
    with open(os.path.join(ROOT, "results", "knockout.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("-> results/knockout.json", flush=True)
