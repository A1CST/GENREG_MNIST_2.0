"""Download the MNIST idx files into corpora/mnist/ (the layout the pipeline
expects). Tries two mirrors."""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "corpora", "mnist")
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/"]

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for f in FILES:
        dst = os.path.join(DATA, f)
        if os.path.exists(dst):
            print("have", f)
            continue
        for m in MIRRORS:
            try:
                urllib.request.urlretrieve(m + f, dst)
                print("downloaded", f, "from", m)
                break
            except Exception as exc:
                print("failed", m + f, exc)
    print("done ->", DATA)
