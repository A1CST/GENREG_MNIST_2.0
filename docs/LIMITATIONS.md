# Limitations

1. **MNIST is a solved benchmark.** The contribution is not the absolute
   number; it is that the number was reached with selection as the only
   learning operator, in a small parameter budget, with each stage gated on
   held-out data. Standard convolutional baselines exceed 99.3% and ensembles
   exceed 99.7%.

2. **The environment ceiling binds.** A closed-form multinomial logistic
   probe on the same frozen features reaches 99.20% test. The evolved
   classifier reaches 99.03%: evolution captured most, not all, of what the
   environment supports, and nothing in this pipeline can exceed the
   environment. Further progress requires a richer environment, not a better
   classifier genome.

3. **The PCA projection is a large fixed constant.** The evolved weights
   total roughly 47 KB, but the 2327-to-1024 PCA projection used to
   assemble the environment is ~9.5 MB of float32 if shipped as weights. It
   is unsupervised and exactly derivable from the training images plus the
   detector bank, so it can be recomputed at setup instead of shipped, but a
   fair size comparison against a CNN checkpoint should count it.

4. **The margin gate and champion selection consume the validation set.**
   Champions are selected on validation at every stage and the referee
   margin is tuned on it. Across many rounds this adds selection pressure on
   a 5k split; the observed validation-to-test gap in the final
   configuration is 0.21 points, consistent with the closed-form probe's own
   0.27-point gap, but the risk grows with every additional gated decision.

5. **Fisher separability is a proxy.** Detector genomes are selected for a
   between/within variance ratio on 2,500-image subsamples, not for
   downstream accuracy. It worked here; there is no guarantee it selects the
   right detectors on harder data. On CIFAR-10 the same procedure produced a
   bank with much lower diversity (15 usable detectors versus 66) and a far
   lower ceiling — the transfer of this exact recipe to natural images is an
   open problem, not a demonstrated result.

6. **Single dataset, single architecture family.** All classifier genomes
   are linear heads over a frozen environment. No claim is made about
   evolving deeper structures, and none about datasets beyond MNIST.

7. **Compute comparisons are not apples-to-apples.** The pipeline evaluates
   fitness on the full training set for hundreds of thousands of
   genome-evaluations; gradient methods reach comparable accuracy in a
   handful of epochs. The interesting property is the absence of gradients,
   not evaluation efficiency.

8. **Timing figures are hardware-specific.** GPU wall-clock numbers were
   measured on one RTX 4080 with the fitness evaluations under torch
   no_grad and TF32 disabled; the CPU fallback is 30-60x slower for the
   large stages.
