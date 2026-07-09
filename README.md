# GENREG MNIST 2.0

A gradient-free MNIST classifier. Every trainable inference parameter is
produced by evolution: the convolutional feature detectors are evolved, the
classifier head is evolved, and the auxiliary referee classifiers are
evolved. The surrounding environment (deskew, pooling, standardization, an
unsupervised PCA projection) consists of fixed statistics computed from the
training images without labels. There is no backpropagation, no gradient,
and no closed-form fit in the model. (A closed-form logistic regression
appears once in the experiments as a diagnostic ceiling estimate; its
weights are never used.)

## Results

| | |
|---|---|
| Test accuracy (5 seeds, full pipeline re-run per seed) | **99.10% ± 0.05%** (min 99.00, max 99.14; 95% CI [99.03, 99.17]) |
| Best single run | 99.14% |
| Evolved parameters at inference | ~15,300 (66 detector genomes + one 1024x10 linear head) |
| Evolved-weight checkpoint | 47 KB |
| Fixed environment constants (PCA projection, derivable from training data) | ~9.5 MB |
| Training wall-clock per seed (RTX 4080, fitness on GPU) | 10-21 min (median 11 min) |

Context (learned parameters, checkpoint, accuracy):

| Model | Learned params | Accuracy | Checkpoint |
|---|---|---|---|
| MANTIS (predecessor: evolved features, least-squares readout) | 50K | 98.55% | 40 KB |
| LeNet-5 (backprop CNN) | 60K | 99.05% | ~250 KB |
| GENREG MNIST 2.0 (evolution end to end) | ~15.3K | 99.10% ± 0.05% | 47 KB (+9.5 MB fixed PCA) |

![accuracy ladder](charts/accuracy_ladder.png)

![confusion matrix](charts/confusion_matrix.png)

![seed variance](charts/seed_variance.png)

## How it works

Three stages, each with its own survival condition; full detail in
[docs/METHOD.md](docs/METHOD.md).

1. **Environment.** Built image statistics (deskew, zone densities,
   profiles, gradient histograms, pixel PCA; 677 dims, unsupervised) are
   concatenated with the pooled responses of an **evolved detector bank**:
   66 genomes, each a 5x5 convolution kernel plus a bias plus an evolved
   activation gene from an 8-function catalog, selected on Fisher
   class-separability of its own pooled response map and decorrelated under
   a correlation cap. The concatenation is PCA-reduced to 1024 dims and
   frozen. The classifier never sees pixels.

2. **Classifier genome.** One linear head (10,250 genes) warm-started from
   the class-centroid statistic and evolved on a deterministic fitness
   landscape: mean log-softmax probability of the true digit over the full
   55,000-image training set, minus an L2 weight cost. Tournament selection
   with elitism and starvation; self-adaptive mutation scaled per-gene by
   the gene's own magnitude.

3. **Referees and gating.** 45 one-vs-one genomes referee close top-2
   decisions, enabled only if a validation-selected margin says they help.
   On the final configuration the gate selects margin 0 for 4 of 5 seeds:
   the head sits near the environment ceiling and the referees are
   correctly disabled.

Every statistic in the environment — deskew moments, standardization, the
PCA projection — is fit on the training images only and contains no label
information; evolution operates entirely after the projection. The one
labeled quantity in the environment is the Fisher separability score that
selects detector genomes, and it uses training labels only.

**Design position.** This is deliberately not end-to-end. The system
evolves the adaptive components (detectors, classifier, referees) inside
fixed environments built from data statistics. Fixing the environment is
what makes each genome's fitness landscape small, deterministic, and
climbable by selection; the ablation table quantifies what each adaptive
component earns inside it.

![detector bank](charts/detector_bank.png)

## Quick start

```
pip install numpy                    # torch optional (GPU); matplotlib optional (charts)
python scripts/download_data.py     # MNIST idx files -> corpora/mnist/
python scripts/evaluate.py          # rebuilds the environment, evaluates the shipped checkpoint
```

## Training from scratch

```
python scripts/train_full.py --seed 7        # full pipeline, one seed
python scripts/run_seeds.py                  # the 5-seed protocol (writes results JSON)
python scripts/run_ablations.py              # the ablation battery
python scripts/make_charts.py                # regenerate charts from results
```

With a CUDA GPU a full seed takes roughly 11 minutes; CPU-only, expect
several hours (the fitness evaluations fall back to numpy automatically).

## Experimental hygiene

- **Multiple random seeds.** The headline number is the mean and standard
  deviation over 5 seeds (7, 11, 23, 42, 101), where each seed re-runs the
  entire pipeline: bank evolution, environment build, classifier evolution,
  referee evolution, and gating. Per-seed results, confusion matrices, and
  timings are in `results/seeds.json`.
- **Test set never used during evolution.** All fitness is computed on the
  55,000-image training split. The test set is evaluated exactly once per
  seed, after every selection decision is frozen.
- **Validation set used for model selection.** Champions at every stage and
  the referee margin are selected on a fixed 5,000-image validation split
  carved off the training set. The validation-to-test gap is reported per
  seed (0.04-0.18 points across the five seeds).
- **Compute budget.** Per seed on one RTX 4080 (fitness evaluations on GPU
  under torch no_grad, TF32 disabled; everything else numpy on CPU):
  bank evolution 79-219 s, environment build 93-126 s, classifier evolution
  100-153 s, referees 341-815 s; total 619-1288 s. CPU fallback is 30-60x
  slower on the large stages. Total GPU time for every experiment reported
  in this repository (the 5-seed protocol plus the 10-cell ablation battery,
  including the extra bank evolved for the activation ablation): about
  1.6 GPU-hours on one RTX 4080. The original single record run used the
  CPU path only (78 minutes).
- **Fitness evaluations.** Per seed: 138,240 bank genome-evaluations (2,500
  images each), 360,000 classifier genome-evaluations (55,000 images each,
  about 2.0e10 image-genome evaluations), 10,125,000 referee
  genome-evaluations (512 images each).
- **Full reproducibility.** All seeds are explicit, every stage is
  deterministic given its seed, and the code in this repository is the code
  that produced the numbers. `scripts/run_seeds.py` regenerates
  `results/seeds.json` end to end. GPU and CPU fitness paths agree to under
  5e-7; identical seeds reproduce identical bank compositions.
- **Ablations.** Every major design decision is removed or replaced one at
  a time in `scripts/run_ablations.py`, reported on validation (the test
  set is reserved for the shipped configuration): the evolved bank vs built
  statistics alone, evolved activation genes vs relu-only, centroid warm
  start vs detector-fold vs cold, magnitude-scaled vs global-sigma
  mutation, full-training-set fitness vs fixed-pool vs per-generation
  minibatch, and the L2 cost. Results in `results/ablations.json` and the
  chart below.
- **Standard GA baseline.** The classic configuration (cold start, global
  sigma, per-generation minibatch fitness) is run on the identical
  environment and budget as a comparison cell in the same table.
- **Failure cases.** Documented with the evidence that cut them in
  [docs/FAILURES.md](docs/FAILURES.md): a random-Fourier feature lift,
  shift augmentation, small fitness pools (memorized), the detector-fold
  warm start, and referee behavior at the environment ceiling.
- **Limitations.** Stated in [docs/LIMITATIONS.md](docs/LIMITATIONS.md),
  including the environment ceiling, the size of the fixed PCA constants,
  validation-set consumption by repeated gating, and the open problem of
  transferring the recipe to natural images.

![ablations](charts/ablations.png)

Ablation results (validation accuracy, identical environment and generation
budget unless the cell changes them; seed 7):

| Configuration | Validation | Delta vs shipped |
|---|---|---|
| Shipped configuration | 99.22% | — |
| Built statistics only (no evolved detector bank) | 97.80% | -1.42 |
| Bank without evolved activation genes (relu only) | 98.90% | -0.32 |
| Cold random start (no centroid warm start) | 83.20% | -16.02 |
| Detector-fold warm start | 97.92% | -1.30 |
| Global-sigma mutation (no magnitude scaling) | 99.04% | -0.18 |
| Fixed 16k fitness pool (instead of full training set) | 99.18% | -0.04 |
| Per-generation minibatch fitness | 99.08% | -0.14 |
| No L2 weight cost | 99.14% | -0.08 |
| No PCA (evolution on the raw 2,327 standardized dims) | 94.60% | -4.62 |
| Standard GA baseline (cold start, global sigma, minibatch) | 92.60% | -6.62 |

Two readings of this table. First, the warm start dominates at this budget:
evolution from a random genome cannot cover the distance in 6,000
generations, which is why every stage bootstraps from either a data
statistic or a previously proven genome. Second, with the warm start in
place, magnitude-scaled mutation accounts for most of the remaining
evolutionary progress: global-sigma mutation improves on its own warm start
by only +0.04 where magnitude scaling gains +0.22.

## Detector knockout analysis

Post-hoc analysis of the shipped model (`scripts/knockout.py`): detectors
are removed by mean-imputing their pooled feature columns before the PCA
projection, with the classifier head frozen; nothing is retrained. Test
accuracy is reported because no selection decision depends on these numbers.

| Knockout | Test accuracy |
|---|---|
| None (baseline) | 99.03% |
| Worst single detector (of 66, full sweep) | 98.78% |
| Mean over all single knockouts | 98.98% |
| Top 10 by Fisher score | 98.01% |
| Random 10 (5 draws) | 96.54% ± 1.33% |
| All 66 (built statistics only, same projection) | 68.61% |

Three observations. The representation is redundant rather than fragile: no
single detector costs more than 0.25 points, and some single knockouts are
within noise of the baseline. Importance is distributed: random groups of
ten can hurt substantially more than the ten highest-Fisher detectors, so
Fisher rank at selection time does not determine importance to the final
head. And the bank carries the classifier: removing all of it through the
same projection collapses accuracy to 68.61%. Full per-detector results in
`results/knockout.json`.

## Representation-quality probes

The evolved detector bank frozen, standard gradient-based readouts trained
on the same 1024-dimensional environment (`scripts/representation_probes.py`;
diagnostics of the representation, not part of the model):

| Readout | Test accuracy |
|---|---|
| Evolved linear head (the model) | 99.03% |
| Logistic regression (scikit-learn defaults, C=1.0) | 98.64% |
| Linear SVM (C=0.1) | 98.69% |
| MLP, one hidden layer of 64 | 98.85% |

All standard readouts perform well on the frozen environment, so the
evolved detector bank is a generally useful representation rather than one
narrowly co-adapted to the evolved head. Two honest caveats: the
gradient-based probes use off-the-shelf hyperparameters (a multinomial
logistic fit with tuned regularization — the ceiling diagnostic reported
above — reaches 99.20% on the same features), and the linear SVM emitted a
convergence warning at its iteration limit. The evolved head outperforming
the default-configured probes on its own representation is noted without
being leaned on.

## Reading these results critically

The ablation table invites several objections. They are addressed here
directly, with numbers, because each one is either correct and conceded or
answerable from the data in this repository.

**"The hand-built statistics already reach 97.8%; evolution only adds 1.4
points."** The percentage-point framing understates what happens near the
top of this benchmark. In error terms, the built-statistics environment
leaves a 2.20% validation error and the shipped configuration leaves 0.78%:
the evolved detector bank removes 65% of the remaining errors. The knockout
analysis makes the same point from the other direction: removing the evolved
bank from the shipped model (same projection, frozen head) collapses test
accuracy from 99.03% to 68.61%. The classifier as shipped depends on the
evolved representation, not on the hand-built statistics alone.

**"The warm start does the work; cold-start evolution only reaches 83.2%."**
The cold-start number is real and reported at equal budget (6,000
generations). Two things follow from the table, not from rhetoric. First,
bootstrapping is the stated design principle, not a concealed trick: every
stage starts from either a data statistic or a previously proven genome, and
the centroid head is one line of arithmetic on the training set. Second, the
claim that evolution "only fine-tunes" is environment-dependent in a way the
table quantifies: on the built-statistics environment the same algorithm
climbs from a 91.96% warm start to 97.80% (+5.84); on the evolved
environment it climbs from 99.00% to 99.22% (+0.22). The head has less to do
on the evolved environment precisely because the evolved detector bank
already did the work — the contribution moved into the bank, it did not
disappear. What the shipped system never does is fit weights to labels by
gradient or by closed form; how far pure cold-start evolution could go with
a larger budget is left open, and the 83.2% cell reports the budget that was
actually spent.

**"PCA is closed-form linear algebra, so the system is not purely
evolutionary."** PCA is closed-form, and this README does not hide it: the
projection is a fixed, unsupervised rotation computed from training images
only, containing no label information. The claim made here is narrower and
precise: no parameter that maps features to labels is produced by a gradient
or a closed-form fit. Whether the projection does hidden discriminative work
is testable, and was tested. A projection that only discards dimensions
cannot add class information to a linear classifier, and the closed-form
diagnostic confirms it: 99.04% validation on the raw 2,327 standardized
dimensions versus 99.26% on the 1,024 PCA dimensions — the same ceiling
either way. What PCA does change is the evolutionary search: at the same
6,000-generation budget the evolved head reaches 94.60% on the raw
dimensions versus 99.22% on the whitened ones. The honest characterization
is therefore: the projection is a search-conditioning aid for evolution, not
a source of class information, and evolution at this budget depends on it.

**"A CNN reaches the same accuracy from raw pixels with less compute."**
Correct, and stated in the limitations: LeNet-5 solves this benchmark from
raw pixels with backpropagation, and gradient methods are far more
sample-efficient per wall-clock second. This repository does not claim
compute efficiency and does not claim end-to-end representation learning.
It demonstrates that a system whose every trainable inference parameter is
produced by selection — features and classifier both — reaches 99.10% ±
0.05% under a documented protocol. Whether that is worth 1.6 GPU-hours is
a judgment the reader can make with the budget in front of them.

## Repository layout

```
genreg_mnist/mnist_pipe.py   the full pipeline (data, environment, genomes, batteries)
genreg_mnist/evo_gpu.py      GPU fitness evaluation (optional; CPU fallback automatic)
scripts/                     download, train, evaluate, seeds, ablations,
                             knockout, representation probes, charts
checkpoints/                 shipped bank + classifier checkpoints
results/                     seeds.json, ablations.json, knockout.json,
                             representation_probes.json
docs/                        METHOD.md, FAILURES.md, LIMITATIONS.md
```

## License

MIT.
