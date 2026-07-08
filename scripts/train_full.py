"""Train the full pipeline from scratch (single seed).

    python scripts/train_full.py [--seed 7]

Stages: detector-bank evolution -> v4 environment build -> joint head from
the centroid warm start on the full-train landscape -> pairwise referees ->
margin gate on validation -> one test evaluation. Writes the checkpoint to
checkpoints/mnist_genomes_seed<seed>.pkl. With a CUDA GPU this takes roughly
10 minutes; CPU-only expect a few hours.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_mnist import mnist_pipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--joint-gens", type=int, default=6000)
    ap.add_argument("--pair-gens", type=int, default=1500)
    a = ap.parse_args()

    ck = os.path.join(ROOT, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    bank_p = os.path.join(ck, f"mnist_detbank_seed{a.seed}.pkl")

    mp.evolve_detbank(seed=a.seed, out=bank_p)
    D = mp.build_features(4, v4_bank=bank_p,
                          v4_cache=os.path.join(ck, f"feats_seed{a.seed}.npz"))
    mu_c = np.stack([D["Ftr"][D["ytr"] == c].mean(0) for c in range(10)], axis=1)
    warm = (mu_c.astype(np.float32), (-0.5 * (mu_c ** 2).sum(0)).astype(np.float32))

    champs = {}
    champs.update(mp.train_joint(champs, gens=a.joint_gens, seed=a.seed, D=D,
                                 minibatch=0, warm_init=warm))
    champs.update(mp.train_pairwise(gens=a.pair_gens, seed=a.seed, D=D))
    champs["feat_version"] = 4

    def val(mrg):
        pred, _ = mp.predict(champs, D["Fva"], True, mrg > 0, mrg, use_joint=True)
        return float((pred == D["yva"]).mean())

    best_m, best = 0.0, val(0.0)
    for m in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12):
        v = val(m)
        if v > best:
            best, best_m = v, m
    champs["pair_margin"] = best_m

    pred, _ = mp.predict(champs, D["Fte"], True, best_m > 0, best_m, use_joint=True)
    acc = float((pred == D["yte"]).mean())
    champs["results"] = {"test": acc, "val": best, "margin": best_m}
    out = os.path.join(ck, f"mnist_genomes_seed{a.seed}.pkl")
    with open(out, "wb") as f:
        pickle.dump(champs, f)
    print(f"seed {a.seed}: val {best:.4f} (margin {best_m})  TEST {acc:.4f} -> {out}")
