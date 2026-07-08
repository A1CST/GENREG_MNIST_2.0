# Failure cases

Every approach that was tried and cut during development, with the evidence
that cut it. All numbers are balanced validation accuracy on the 5k holdout
unless stated otherwise; the test set was never consulted for any of these
decisions.

## 1. Random Fourier feature lift of the statistics layer

A fixed random projection of the 677-dimensional built-statistics environment
into 1024 additional cosine features (RFF approximation of an RBF kernel),
intended to give the linear genomes a nonlinear environment.

- Cold start: a single detector genome reached 0.9598 in 5,000 generations
  versus 0.9785 on the unlifted features at the same budget. The larger
  search space diluted the same mutation budget.
- Warm start from the unlifted champion (latent genes for the lifted block
  initialized at zero): opened at 0.9786 and then decayed; validation
  log-probability fell from -0.17 to -0.49 while training fitness climbed.
  The lifted directions were exploited to fit minibatch noise.
- L2 penalties at 3e-4 and 1e-3 slowed the decay without producing a genome
  that beat the warm start.

Verdict: cut. The GA could not extract generalizing signal from the lifted
directions at any tested budget. The code remains in the repository
(`rff_lift`, feature version 3) for reproduction.

## 2. Shifted-copy training augmentation

Two extra copies of each training image at random integer shifts in
[-2, 2]^2, appended to the training pool (validation and test untouched).

- Detector genome on the augmented pool: 0.9581 at 5,000 generations versus
  0.9785 unaugmented at the same budget. The augmented task is harder and
  the fixed generation budget bought less progress on the clean objective.

Verdict: cut at this budget. Available via `--augment`.

## 3. Small fitness pools are memorized

The joint head was first refined on a fixed 16,384-image pool (chosen to be
"too large to overfit" at 6,770 parameters). It was not:

- Final champion negative log-likelihood on its own pool: about -0.009
  (near-perfect fit). The same genome scored -0.086 on the full training
  set. The population had memorized the pool; validation accuracy stalled
  while pool fitness kept improving.
- Rotating 4,096-image pools every 25 generations produced serial per-batch
  overfitting instead: the population climbed each batch, the rotation
  invalidated the climb, and champion weight norms grew from 478 through an
  L2 penalty of 1e-4 to 828.
- Per-generation resampled minibatches drowned the refinement signal in
  evaluation noise entirely (see the ablation table).

Fix that shipped: the full 55,000-image training set as one deterministic
fitness landscape. Fitness improvements on that landscape cannot be
memorization, and the validation curve tracked fitness monotonically from
then on.

## 4. Detector-fold warm start for the joint head

Folding the 10 one-vs-rest detector heads (plus the 10x10 mixer) into a
single linear head is algebraically exact, but the folded head opened at
0.9494 on the evolved-environment features while the class-centroid head — a
pure training statistic — opened at 0.9902. One-vs-rest heads are trained
independently and their logit scales are not argmax-calibrated. The shipped
configuration warm-starts from the centroid head.

## 5. Pairwise referees at the environment ceiling

45 one-vs-one linear genomes referee the joint head's close top-2 decisions.
On weaker joint heads they contributed up to +1.2 points of test accuracy
(margin selected on validation). On the final v4 configuration, where the
joint head sits within 0.03 of the environment's closed-form ceiling, the
validation margin gate selected margin 0.0 — the referees were disabled
because they had nothing left to add. The gate mechanism is retained: it is
the correct behavior in both regimes.

## 6. Per-neuron evolved precision (neutral, not negative)

Evolving a bit-width gene per output neuron with quantization inside the
fitness evaluation compressed an earlier joint head from 32-bit to a mean of
10.5 bits per weight at unchanged accuracy (98.19 versus 98.21, a 2-image
difference on the 10k test set), with the hardest digit classes retaining
the most precision. It neither helped nor hurt accuracy; it is a deployment
optimization and is not part of the headline configuration.
