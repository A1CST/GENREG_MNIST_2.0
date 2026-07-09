# Method

Every trainable inference parameter in this system is produced by evolution.
The environment around those parameters (deskew, pooling, standardization,
PCA) consists of fixed statistics computed from the training images; the PCA
projection is fit on training images only and contains no label information,
and evolution operates entirely after the projection. There is no
backpropagation, no gradient, and no closed-form fit anywhere in the model.
(A closed-form logistic regression appears once, as a diagnostic ceiling
probe; its weights are never used. The representation-quality probes in
`scripts/representation_probes.py` use gradient-based readouts as
diagnostics of the frozen environment; their weights are likewise never part
of the model.)

## Data protocol

- Train: first 55,000 MNIST training images.
- Validation: last 5,000 MNIST training images. Used for champion selection
  and for the referee margin gate. Never trained on.
- Test: the 10,000-image MNIST test set. Used exactly once per configuration,
  after all selection decisions are frozen. Never used during evolution or
  model selection.

## Stage 1: the environment

The classifier genomes never see pixels. They see a frozen feature
environment assembled in three parts:

1. **Built statistics** (677 dims, no labels, no learning): moment-based
   deskew, zone ink densities at two grids (4x4, 7x7), row/column ink
   profiles, 8-bin gradient-orientation histograms at two cell grids, and
   64 PCA components of the raw pixels. Standardized by training statistics.

2. **Evolved detector bank** (66 genomes, 77 genes each): 5x5 convolution
   kernels plus a bias plus one activation gene selecting from an
   8-function catalog (relu, abs, sin, cos, gaussian, leaky relu, square,
   tanh). Fitness is Fisher class-separability (between-class variance over
   within-class variance) of the detector's pooled response map on a fixed
   2,500-image balanced subsample — a local survival condition; downstream
   accuracy never enters the detector's landscape. 48 independently seeded
   populations (48 genomes, 60 generations each) are harvested every 20
   generations; candidates are then greedily selected by Fisher score under
   a 0.95 correlation cap, computed on one common reference set, yielding a
   diverse bank rather than 48 copies of one detector. Each detector's
   response map is mean-pooled at three grid shapes (3x3, 4x2, 2x4),
   contributing 25 dims.

3. **PCA to 1024 dims** of the concatenated (built + bank) features,
   standardized. Unsupervised, computed from training data only.

## Stage 2: the classifier genome

One linear head (1024 x 10 + 10 = 10,250 genes), evolved as follows:

- **Warm start**: the class-centroid head (W = class means, b = -|mu|^2/2),
  a pure training statistic. It opens at 99.0% validation.
- **Fitness**: mean log-softmax probability of the true digit over the FULL
  55,000-image training set, minus 1e-4 times the squared weight norm.
  The full-set landscape is deterministic; improvements on it cannot be
  minibatch memorization (see docs/FAILURES.md for what smaller pools did).
- **Selection**: tournament (k=4) with 10% elitism and 8% starvation of the
  lowest-fitness slice per generation; population 60; 6,000 generations.
- **Mutation**: self-adaptive per-genome sigma (log-normal drift, floor
  5e-4), scaled per-gene by the gene's own magnitude plus a 5%-of-mean
  floor, so large and small weights are explored at their own scales.
- **Champion**: the highest validation accuracy seen, evaluated every 100
  generations. Since generation 1 holds the warm start, the procedure
  cannot regress below its starting point.

## Stage 3: pairwise referees and the margin gate

45 one-vs-one linear genomes (one per digit pair, trained only on those two
digits, same trainer as the detectors) referee the joint head's decision
when the top-2 logit margin is small. The margin is selected on validation
from {0, 0.5, ..., 12}. On the final configuration the gate selects margin
0.0 — the joint head is close enough to the environment ceiling that the
referees add nothing, and the gate correctly disables them. They
contributed up to +1.2 test points on earlier, weaker configurations.

## The GA in one paragraph

`ga_step` is shared by every stage: sort by fitness, keep the top 10%
unmutated (elites), exclude the bottom 8% from parenthood (starvation),
fill the rest by k=4 tournament, mutate children with a self-adaptive sigma
inherited log-normally from the parent, optionally scaled per-gene by
magnitude. No crossover. All arrays are numpy; when a CUDA device is
present, fitness evaluations (and only fitness evaluations) run on it via
torch under no_grad with TF32 disabled, verified to match the CPU path to
under 5e-7.
