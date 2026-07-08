"""MNIST-Pipe v4 multi-seed protocol (5 seeds, full pipeline per seed).

Per seed: evolve the detector bank -> build the v4 environment -> joint head
from the centroid warm start on the full-train landscape -> pairwise referees
-> margin gate on the validation split -> ONE evaluation of the untouched
test set. Everything seeded; per-seed artifacts and a results JSON are written
to demo/seeds/. Also records wall-clock per stage and fitness-evaluation
counts for the compute-budget report.
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_mnist import mnist_pipe as mp

SEEDS = [7, 11, 23, 42, 101]
OUT = os.path.join(mp.ROOT, "demo", "seeds")
os.makedirs(OUT, exist_ok=True)

DET_ROUNDS, DET_POP, DET_GENS, DET_SUB = 48, 48, 60, 2500
JOINT_GENS, JOINT_POP = 6000, 60
PAIR_GENS, PAIR_POP, PAIR_MB = 1500, 150, 256


def eval_stack(champs, D, split, margin):
    F, y = (D["Fte"], D["yte"]) if split == "test" else (D["Fva"], D["yva"])
    pred, _ = mp.predict(champs, F, True, margin > 0, margin, use_joint=True)
    conf = np.zeros((10, 10), np.int64)
    np.add.at(conf, (y, pred), 1)
    return float((pred == y).mean()), conf


def run_seed(s):
    t0 = time.time()
    res = {"seed": s}
    bank_p = os.path.join(OUT, f"bank_{s}.pkl")
    cache_p = os.path.join(OUT, f"feats_{s}.npz")

    t = time.time()
    bank = mp.evolve_detbank(seed=s, out=bank_p, rounds=DET_ROUNDS, pop=DET_POP,
                             gens=DET_GENS, sub=DET_SUB)
    res["bank_n"] = int(len(bank["K"]))
    res["t_bank"] = round(time.time() - t, 1)

    t = time.time()
    D = mp.build_features(4, v4_bank=bank_p, v4_cache=cache_p)
    res["nf"] = int(D["nf"])
    res["t_features"] = round(time.time() - t, 1)

    # centroid floor + warm start (train statistics only)
    cents = np.stack([D["Ftr"][D["ytr"] == c].mean(0) for c in range(10)])
    d2 = ((D["Fte"][:, None, :] - cents[None]) ** 2).sum(-1)
    res["centroid_test"] = round(float((d2.argmin(1) == D["yte"]).mean()), 4)
    mu_c = cents.T.astype(np.float32)
    warm = (mu_c, (-0.5 * (mu_c ** 2).sum(0)).astype(np.float32))

    t = time.time()
    champs = {}
    champs.update(mp.train_joint(champs, gens=JOINT_GENS, pop=JOINT_POP,
                                 seed=s, D=D, minibatch=0, warm_init=warm))
    res["joint_val"] = champs["joint_val_acc"]
    res["t_joint"] = round(time.time() - t, 1)

    t = time.time()
    champs.update(mp.train_pairwise(gens=PAIR_GENS, pop=PAIR_POP, seed=s, D=D))
    res["t_pairs"] = round(time.time() - t, 1)

    # margin gate on VALIDATION only
    best_m, best_acc = 0.0, eval_stack(champs, D, "val", 0.0)[0]
    for m in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        a = eval_stack(champs, D, "val", m)[0]
        if a > best_acc:
            best_acc, best_m = a, m
    res["pair_margin"] = best_m
    res["val_full"] = round(best_acc, 4)

    # single test evaluation
    acc, conf = eval_stack(champs, D, "test", best_m)
    res["test_full"] = round(acc, 4)
    res["confusion"] = conf.tolist()
    res["t_total"] = round(time.time() - t0, 1)

    # fitness-evaluation accounting
    res["fitness_evals"] = {
        "bank_genome_evals": DET_ROUNDS * DET_GENS * DET_POP,
        "bank_images_per_eval": DET_SUB,
        "joint_genome_evals": JOINT_GENS * JOINT_POP,
        "joint_images_per_eval": int(len(D["ytr"])),
        "pair_genome_evals": 45 * PAIR_GENS * PAIR_POP,
        "pair_images_per_eval": 2 * PAIR_MB,
    }
    with open(os.path.join(OUT, f"champs_{s}.pkl"), "wb") as f:
        pickle.dump({**champs, "feat_version": 4, "pair_margin": best_m,
                     "results": res}, f)
    return res


if __name__ == "__main__":
    all_res = []
    for s in SEEDS:
        print(f"===== SEED {s} =====", flush=True)
        r = run_seed(s)
        all_res.append(r)
        print(f"SEED {s}: test={r['test_full']} val={r['val_full']} "
              f"joint_val={r['joint_val']} bank={r['bank_n']} "
              f"margin={r['pair_margin']} ({r['t_total']}s)", flush=True)
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump(all_res, f, indent=1)
    accs = [r["test_full"] for r in all_res]
    print(f"MEAN±STD test: {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
          f"(n={len(accs)}, min {min(accs)}, max {max(accs)})", flush=True)
