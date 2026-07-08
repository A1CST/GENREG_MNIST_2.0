"""Generate the result charts from results/seeds.json + results/ablations.json
and the record checkpoint. Professional, monochrome-plus-one-accent styling.

    python scripts/make_charts.py
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CH = os.path.join(ROOT, "charts")
os.makedirs(CH, exist_ok=True)

ACCENT = "#2563eb"
GREY = "#6b7280"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def confusion_chart(seeds):
    conf = np.array(seeds[0]["confusion"])          # seed 7 = record protocol
    acc = seeds[0]["test_full"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    off = conf.copy().astype(float)
    np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap="Reds", vmin=0, vmax=max(1, np.nanmax(off)))
    for i in range(10):
        for j in range(10):
            v = conf[i, j]
            if i == j:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="#065f46", fontweight="bold")
            elif v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix, held-out test set (seed 7, acc {acc:.4f})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="off-diagonal errors")
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "confusion_matrix.png"))
    plt.close(fig)


def ladder_chart(seeds):
    accs = [r["test_full"] for r in seeds]
    cent = [r["centroid_test"] for r in seeds]
    stages = [
        ("majority class", 0.1135),
        ("centroid floor\n(no classifier evolution)", float(np.mean(cent))),
        ("v2 pipeline (built stats),\nbest of rounds 1-6", 0.9821),
        ("v4 full pipeline\n(evolved environment)", float(np.mean(accs))),
        ("closed-form ceiling\n(diagnostic only)", 0.9920),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = np.arange(len(stages))
    vals = [s[1] for s in stages]
    colors = [GREY, GREY, GREY, ACCENT, "#d1d5db"]
    ax.bar(xs, vals, color=colors, width=0.62)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.004, f"{v * 100:.2f}%", ha="center", fontsize=10)
    ax.set_xticks(xs, [s[0] for s in stages], fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy ladder (test set, mean over 5 seeds where applicable)")
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "accuracy_ladder.png"))
    plt.close(fig)


def seeds_chart(seeds):
    accs = np.array([r["test_full"] for r in seeds])
    labels = [str(r["seed"]) for r in seeds]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(labels, accs, color=ACCENT, width=0.55)
    m, s = accs.mean(), accs.std()
    ax.axhline(m, color="#111827", lw=1, ls="--")
    ax.fill_between([-0.5, len(accs) - 0.5], m - s, m + s, color=ACCENT, alpha=0.12)
    ax.set_ylim(min(accs) - 0.004, max(accs) + 0.004)
    ax.set_xlabel("seed"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Test accuracy across seeds: {m:.4f} ± {s:.4f} (n={len(accs)})")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.0005, f"{a:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "seed_variance.png"))
    plt.close(fig)


def ablation_chart(abl):
    order = ["env_v4_evolved_bank", "env_v2_built_stats_only", "bank_relu_only",
             "warm_cold_random", "warm_detector_fold", "mutation_global_sigma",
             "landscape_fixed_16k", "landscape_pergen_minibatch", "l2_off",
             "standard_ga_baseline"]
    names = {"env_v4_evolved_bank": "shipped configuration",
             "env_v2_built_stats_only": "built stats only (no evolved bank)",
             "bank_relu_only": "bank without evolved activations (relu)",
             "warm_cold_random": "cold random start (no centroid warm)",
             "warm_detector_fold": "detector-fold warm start",
             "mutation_global_sigma": "global-sigma mutation (no magnitude scaling)",
             "landscape_fixed_16k": "fixed 16k fitness pool",
             "landscape_pergen_minibatch": "per-generation minibatch",
             "l2_off": "no L2 weight cost",
             "standard_ga_baseline": "standard GA baseline"}
    d = {r["cell"]: r["val_acc"] for r in abl}
    rows = [(names[k], d[k]) for k in order if k in d]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ys = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    colors = [ACCENT if i == 0 else GREY for i in range(len(rows))]
    ax.barh(ys, vals, color=colors, height=0.6)
    for y, v in zip(ys, vals):
        ax.text(v + 0.001, y, f"{v * 100:.2f}%", va="center", fontsize=9)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(min(vals) - 0.01, max(vals) + 0.012)
    ax.set_xlabel("validation accuracy (test untouched by ablations)")
    ax.set_title("Ablation study (one design decision changed per row)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "ablations.png"))
    plt.close(fig)


def bank_chart():
    with open(os.path.join(ROOT, "checkpoints", "mnist_detbank.pkl"), "rb") as f:
        bank = pickle.load(f)
    names = ["relu", "abs", "sin", "cos", "gaussian", "leaky", "square", "tanh"]
    cnt = np.bincount(bank["act"], minlength=8)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].bar(names, cnt, color=ACCENT, width=0.6)
    axes[0].set_title("Evolved activation genes (66-detector bank)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("detectors")
    K = bank["K"][:24].reshape(-1, 5, 5)
    grid = np.full((4 * 6 + 3, 6 * 6 + 5), np.nan)
    for i, k in enumerate(K):
        r, c = divmod(i, 6)
        grid[r * 7:r * 7 + 5, c * 7:c * 7 + 5] = k
    axes[1].imshow(grid, cmap="RdBu_r")
    axes[1].set_title("First 24 evolved 5x5 kernels (by Fisher rank)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "detector_bank.png"))
    plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RES, "seeds.json")) as f:
        seeds = json.load(f)
    confusion_chart(seeds)
    ladder_chart(seeds)
    seeds_chart(seeds)
    ap = os.path.join(RES, "ablations.json")
    if os.path.exists(ap):
        with open(ap) as f:
            ablation_chart(json.load(f))
    bank_chart()
    print("charts ->", CH)
