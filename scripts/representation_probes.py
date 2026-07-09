"""Representation-quality probes: freeze the evolved environment, train
standard gradient-based readouts on it, and compare.

If logistic regression, a linear SVM, and a small MLP all perform well on
the frozen features, the evolved detector bank is a generally useful
representation rather than one narrowly co-adapted to the evolved head.
These probes are diagnostics of the representation; their weights are not
part of the model.

Requires scikit-learn.

    python scripts/representation_probes.py
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_mnist import mnist_pipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier

    bank = os.path.join(ROOT, "checkpoints", "mnist_detbank.pkl")
    D = mp.build_features(4, v4_bank=bank)
    Xtr, ytr, Xte, yte = D["Ftr"], D["ytr"], D["Fte"], D["yte"]

    probes = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "linear_svm": LinearSVC(C=0.1),
        "mlp_64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                random_state=0),
    }
    with open(os.path.join(ROOT, "checkpoints",
                           "mnist_genomes_v4_9903.pkl"), "rb") as f:
        head = pickle.load(f)["joint"]
    evolved = float(((Xte @ head[0] + head[1]).argmax(1) == yte).mean())
    out = {"evolved_head": round(evolved, 4)}
    print(f"evolved head (reference): {evolved:.4f}", flush=True)
    for name, clf in probes.items():
        t = time.time()
        clf.fit(Xtr, ytr)
        acc = float(clf.score(Xte, yte))
        out[name] = round(acc, 4)
        print(f"{name}: test {acc:.4f}  ({time.time() - t:.0f}s)", flush=True)
    with open(os.path.join(ROOT, "results", "representation_probes.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("-> results/representation_probes.json", flush=True)
