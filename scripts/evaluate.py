"""Evaluate a trained checkpoint on the untouched MNIST test set.

    python scripts/evaluate.py [checkpoints/mnist_genomes_v4_9903.pkl]

Prints test accuracy, the per-class breakdown, and the confusion matrix.
The feature environment (detector bank responses + built statistics + PCA)
is rebuilt deterministically from the training data and the bank checkpoint.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_mnist import mnist_pipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(ckpt):
    with open(ckpt, "rb") as f:
        champs = pickle.load(f)
    fv = champs.get("feat_version", 4)
    bank = os.path.join(ROOT, "checkpoints", "mnist_detbank.pkl")
    D = mp.build_features(fv, v4_bank=bank) if fv == 4 else mp.build_features(fv)
    m = champs.get("pair_margin", 0.0)
    pred, _ = mp.predict(champs, D["Fte"], True, m > 0, m, use_joint=True)
    y = D["yte"]
    acc = float((pred == y).mean())
    print(f"test accuracy: {acc:.4f}  ({int((pred != y).sum())} errors / {len(y)})")
    conf = np.zeros((10, 10), np.int64)
    np.add.at(conf, (y, pred), 1)
    print("per-class accuracy:",
          " ".join(f"{d}:{conf[d, d] / conf[d].sum():.3f}" for d in range(10)))
    print("confusion matrix (rows = true, cols = predicted):")
    for row in conf:
        print("  " + " ".join(f"{v:4d}" for v in row))


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "checkpoints", "mnist_genomes_v4_9903.pkl")
    main(ckpt)
