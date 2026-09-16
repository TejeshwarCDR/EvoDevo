# NCA timescale (lambda, state_bound) calibration

Fitness-blind (evodevo.analysis.structural.settling_profile only -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes/cell, init_condition=`center_seed` (the P3 decision), T_max=40, hidden_channels=3, seed=0.

**Target (fixed, not modified)**: rule_based's own settling distribution, measured once at these settings via `structural.settling_profile` -- the same function used in results/depth/DEPTH_STUDY.md, reused here rather than re-implemented.

rule_based: median settle_step=2.0000, mean settle_step=6.5655, never_settled_fraction=0.1155.

## Joint sweep: lambda (decay) x state_bound (B)

state_bound is crossed with decay because it sets the argmax margin the readout must cross to flip, so it affects the MEASURED settle step even though it plays no role in the contraction condition itself (results/nca/CONVERGENCE.md, Proposition 2: the certificate depends on `state_bound * K(theta)`, not on B or K individually).

| lambda | B | Wasserstein-1 | KS | never_settled_fraction | median settle_step | mean settle_step |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 2 | 30.3260 | 0.8635 | 0.4005 | 39.0000 | 36.8495 **<- selected** |
| 0.05 | 5 | 31.1055 | 0.8815 | 0.3860 | 39.0000 | 37.6690 |
| 0.05 | 10 | 31.3850 | 0.8790 | 0.4090 | 39.0000 | 37.9505 |
| 0.1 | 2 | 31.3930 | 0.8820 | 0.4645 | 39.0000 | 37.9575 |
| 0.1 | 5 | 31.9060 | 0.8790 | 0.5140 | 40.0000 | 38.4715 |
| 0.1 | 10 | 32.2235 | 0.8800 | 0.5440 | 40.0000 | 38.7890 |
| 0.15 | 2 | 31.5910 | 0.8805 | 0.4970 | 39.0000 | 38.1565 |
| 0.15 | 5 | 32.3020 | 0.8805 | 0.5900 | 40.0000 | 38.8675 |
| 0.15 | 10 | 32.7015 | 0.8820 | 0.6730 | 40.0000 | 39.2670 |
| 0.2 | 2 | 31.6335 | 0.8780 | 0.5565 | 40.0000 | 38.1990 |
| 0.2 | 5 | 32.5875 | 0.8815 | 0.6790 | 40.0000 | 39.1530 |
| 0.2 | 10 | 32.9080 | 0.8825 | 0.7630 | 40.0000 | 39.4735 |
| 0.3 | 2 | 31.5400 | 0.8770 | 0.6085 | 40.0000 | 38.1055 |
| 0.3 | 5 | 32.7405 | 0.8815 | 0.7785 | 40.0000 | 39.3060 |
| 0.3 | 10 | 33.0910 | 0.8825 | 0.8635 | 40.0000 | 39.6565 |
| 0.5 | 2 | 31.1500 | 0.8725 | 0.7105 | 40.0000 | 37.7155 |
| 0.5 | 5 | 32.8245 | 0.8795 | 0.8885 | 40.0000 | 39.3900 |
| 0.5 | 10 | 33.2210 | 0.8835 | 0.9475 | 40.0000 | 39.7865 |
| 0.7 | 2 | 31.1940 | 0.8730 | 0.7775 | 40.0000 | 37.7595 |
| 0.7 | 5 | 32.9520 | 0.8805 | 0.9380 | 40.0000 | 39.5175 |
| 0.7 | 10 | 33.3495 | 0.8835 | 0.9835 | 40.0000 | 39.9150 |
| 1.0 | 2 | 32.6625 | 0.8770 | 0.9435 | 40.0000 | 39.2280 |
| 1.0 | 5 | 33.3585 | 0.8830 | 0.9950 | 40.0000 | 39.9240 |
| 1.0 | 10 | 33.4155 | 0.8840 | 0.9980 | 40.0000 | 39.9810 |

Selection criterion: minimise Wasserstein-1 (a full-distribution distance, not just matched medians -- matching only the median is exactly the limitation P5's mutation calibration hit when it matched only the mean phenotypic displacement and missed a 7-cell IQR gap in the tails; KS is reported alongside as a second, independent distribution-shape check, not used for selection).

**Selected: lambda\* = 0.05, B\* = 2** (Wasserstein-1 = 30.3260, KS = 0.8635).

## Held-out verification

A fresh, independent NCA sample (N=2000, seed=1 -- distinct from the sweep's seed=0) at (lambda\*=0.05, B\*=2), compared against the SAME fixed rule_based target distribution measured above (the target is not resampled: it is fixed once and never modified, per the task's constraint).

| quantity | sweep (seed=0) | verification (seed=1) |
| --- | --- | --- |
| Wasserstein-1 | 30.3260 | 30.4860 |
| KS | 0.8635 | 0.8635 |
| never_settled_fraction | 0.4005 | 0.4125 |
| median settle_step | 39.0000 | 39.0000 |
| mean settle_step | 36.8495 | 37.0095 |

Verification Wasserstein-1 (30.4860) differs from the sweep's selection value (30.3260) by 0.1600 -- within a small-sample-noise tolerance of 0.5 steps, so the selected (lambda*, B*) generalises to unseen genomes at this setting and is not an artifact of the sweep's particular sample.

## Tail behaviour at lambda*, B* (P5-style: matching a median or mean is not enough)

| settle within <= N steps | rule_based (target) | nca @ lambda*,B* (sweep sample) | nca @ lambda*,B* (verification sample) |
| --- | --- | --- | --- |
| 2 | 0.6285 | 0.0210 | 0.0210 |
| 5 | 0.8615 | 0.0210 | 0.0210 |
| 10 | 0.8830 | 0.0210 | 0.0210 |
| 20 | 0.8845 | 0.0350 | 0.0285 |

## How much of the 37-vs-2 gap this closes

Risk-register item 4's baseline gap (results/depth/DEPTH_STUDY.md, unbounded_residual, center_seed): NCA median settle_step=37 vs rule_based=2, a gap of 35 steps.
**Negative result on the median**: at (lambda\*, B\*), NCA median settle_step=39.0000 vs rule_based=2.0000 -- a gap of 37.0000 steps, WORSE than (not an improvement on) the original 35-step baseline gap. Every cell in the swept lambda x B grid (lambda in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0], B in [2, 5, 10]) has a median settle_step of 39-40, at or near T_max=40: within this grid, decay/state_bound tuning does not bring NCA's median settling speed anywhere close to rule_based's. lambda* and B* were still selected on Wasserstein-1 (the least-bad cell in the grid, not a successful match) since a selection criterion is required regardless of whether the calibration goal was reached.

**This does not claim the gap is solved -- it is not.** The Wasserstein-1 distance at (lambda\*, B\*) is 30.3260 steps (out of a maximum possible 40), driven by BOTH a residual median/mean gap and the tail differences in the table below. Within the tested grid, the best cell sits at its lowest tested B (2) and the never_settled_fraction is lowest at the lowest tested lambda (0.05) for every B -- suggestive (not conclusive: W1 itself is not strictly monotone in lambda at fixed B, see the table) that smaller lambda and/or B values, outside this task's specified grid, might close more of the gap. That is future work, not evidence that this grid's selection has actually solved the mismatch.

## Notes

This calibration never computes or consults a fitness, reward, or task-performance quantity: only settling_profile's decode()-derived, fitness-blind measurement is used. The rule_based arm was measured once and never modified to ease matching.

