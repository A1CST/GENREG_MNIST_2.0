"""MNIST-Pipe v4 ablation battery. One cell per major design decision, all
other settings held at the shipped configuration (seed 7 throughout; the
multi-seed protocol covers variance separately). Each cell reports held-out
VALIDATION accuracy (test stays untouched by ablations — it was spent on the
shipped configuration only) plus the fitness trajectory endpoint.

Cells:
  environment : v2 built-stats only vs v4 evolved detector bank
  activations : evolved 8-function catalog vs relu-only bank
  warm start  : centroid head vs detector-fold vs cold (small random)
  mutation    : magnitude-scaled vs global-sigma
  landscape   : full-train vs fixed-16k pool vs per-generation minibatch
  l2          : 1e-4 vs 0
  standard GA : cold start + global sigma + per-gen minibatch (the classic
                setup) on the same environment — the baseline comparison
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_mnist import mnist_pipe as mp

OUT = os.path.join(mp.ROOT, "demo", "seeds")
os.makedirs(OUT, exist_ok=True)
SEED = 7
GENS, POP = 6000, 60


def centroid_warm(D):
    mu_c = np.stack([D["Ftr"][D["ytr"] == c].mean(0) for c in range(10)], axis=1)
    return (mu_c.astype(np.float32), (-0.5 * (mu_c ** 2).sum(0)).astype(np.float32))


def cold_warm(D, seed=SEED):
    rng = np.random.default_rng(seed)
    nf = D["Ftr"].shape[1]
    return ((rng.standard_normal((nf, 10)) * (1.0 / np.sqrt(nf))).astype(np.float32),
            np.zeros(10, np.float32))


def joint_cell(name, D, warm, mag_scale=True, minibatch=0, rotate=0, l2=1e-4,
               gens=GENS):
    t = time.time()
    r = mp.train_joint({}, gens=gens, pop=POP, seed=SEED, D=D, minibatch=minibatch,
                       rotate=rotate, l2=l2, mag_scale=mag_scale, warm_init=warm,
                       log=lambda *a: None)
    out = {"cell": name, "val_acc": r["joint_val_acc"],
           "warm_val": r["joint_base_val_acc"], "t": round(time.time() - t, 1)}
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    results = []
    bank_p = os.path.join(OUT, "bank_7.pkl")
    cache_p = os.path.join(OUT, "feats_7.npz")
    if not os.path.exists(bank_p):                 # standalone fallback
        mp.evolve_detbank(seed=SEED, out=bank_p)
    D4 = mp.build_features(4, v4_bank=bank_p, v4_cache=cache_p)
    D2 = mp.build_features(2)
    w4, w2 = centroid_warm(D4), centroid_warm(D2)

    # environment
    results.append(joint_cell("env_v4_evolved_bank", D4, w4))
    results.append(joint_cell("env_v2_built_stats_only", D2, w2))

    # bank activations: relu-only bank -> its own environment
    relu_bank = os.path.join(OUT, "bank_7_relu.pkl")
    relu_cache = os.path.join(OUT, "feats_7_relu.npz")
    if not os.path.exists(relu_bank):
        mp.evolve_detbank(seed=SEED, out=relu_bank, act_catalog=False)
    Dr = mp.build_features(4, v4_bank=relu_bank, v4_cache=relu_cache)
    results.append(joint_cell("bank_relu_only", Dr, centroid_warm(Dr)))

    # warm start
    results.append(joint_cell("warm_cold_random", D4, cold_warm(D4)))
    # detector-fold warm start: quick one-vs-rest detectors in v4 space
    dets = mp.train_detectors(gens=1500, seed=SEED, log=lambda *a: None, D=D4)
    Wd, bd = mp.fold_stack({"det": dets["det"]})
    results.append(joint_cell("warm_detector_fold", D4, (Wd, bd)))

    # mutation
    results.append(joint_cell("mutation_global_sigma", D4, w4, mag_scale=False))

    # landscape
    results.append(joint_cell("landscape_fixed_16k", D4, w4, minibatch=16384))
    results.append(joint_cell("landscape_pergen_minibatch", D4, w4,
                              minibatch=4096, rotate=1))

    # l2
    results.append(joint_cell("l2_off", D4, w4, l2=0.0))

    # standard GA baseline: cold + global sigma + per-gen minibatch
    results.append(joint_cell("standard_ga_baseline", D4, cold_warm(D4),
                              mag_scale=False, minibatch=4096, rotate=1))

    with open(os.path.join(OUT, "ablations.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("ablations saved", flush=True)
