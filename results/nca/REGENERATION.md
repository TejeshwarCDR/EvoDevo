# NCA regeneration / self-repair (Corollary 3)

Fitness-blind (only decode()/develop()/NCAEncoder._step()/develop_rules() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes per (k, damage_fraction) cell, T=40, init_condition=`center_seed` (P3), seed=0. NCA at update_rule=bounded_leaky, lambda\*=0.05, B\*=2 (docs/decisions.md, 'NCA timescale calibration'), H=3 (N5-resolved). Damage: NCA zeroes a random d-fraction of cells' full state vector; rule_based sets a random d-fraction of cells to EMPTY; direct damages the genome directly (see the explicit note on this asymmetry below).

## NCA at PRIMARY settings (standard random_genome(), init_gain=1.0)

Certified fraction under the PRIMARY genome distribution: 0/18000 (0.00%) -- consistent with results/nca/CONVERGENCE.md's finding that `init_gain=1.0` (calibrated for near-unit pre-activation variance, not for a small Lipschitz constant) makes essentially no randomly-drawn genome satisfy the certificate. **The theory therefore makes no recovery guarantee for the genome population evolution would actually sample at these settings** -- the numbers below describe empirical (uncertified) behaviour only, not a test of Corollary 3 (that test needs a population that actually contains certified genomes -- see the DIAGNOSTIC section below).

| k | T-k | d | n | mean Hamming | median Hamming | fraction fully recovered (Hamming=0) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 30 | 0.1 | 2000 | 2.9750 | 3.0000 | 0.1215 |
| 10 | 30 | 0.2 | 2000 | 5.6995 | 6.0000 | 0.0335 |
| 10 | 30 | 0.4 | 2000 | 8.8445 | 9.0000 | 0.0150 |
| 20 | 20 | 0.1 | 2000 | 2.2420 | 2.0000 | 0.1485 |
| 20 | 20 | 0.2 | 2000 | 4.6545 | 5.0000 | 0.0395 |
| 20 | 20 | 0.4 | 2000 | 7.7480 | 8.0000 | 0.0175 |
| 30 | 10 | 0.1 | 2000 | 1.6790 | 2.0000 | 0.1930 |
| 30 | 10 | 0.2 | 2000 | 3.7655 | 4.0000 | 0.0455 |
| 30 | 10 | 0.4 | 2000 | 6.8800 | 7.0000 | 0.0120 |

## NCA at DIAGNOSTIC settings (mixed genome scale, NOT the primary distribution)

Genomes drawn at scales (0.001, 0.01, 0.1, 1.0, 5.0) (uniformly at random per genome) instead of the standard z~N(0,1) `random_genome()` draw -- purely so this population actually CONTAINS certified genomes to test Corollary 3 against; this is a diagnostic measurement, not what evolution would sample (see results/nca/CONVERGENCE.md, which uses the same device for the same reason).

| k | T-k | d | certified? | n | mean Hamming | median Hamming | fraction fully recovered (Hamming=0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 30 | 0.1 | certified | 1181 | 1.6139 | 2.0000 | 0.1643 |
| 10 | 30 | 0.1 | uncertified | 819 | 4.4554 | 4.0000 | 0.0733 |
| 10 | 30 | 0.2 | certified | 1199 | 4.0350 | 5.0000 | 0.1401 |
| 10 | 30 | 0.2 | uncertified | 801 | 7.4869 | 7.0000 | 0.0212 |
| 10 | 30 | 0.4 | certified | 1223 | 8.0343 | 10.0000 | 0.1055 |
| 10 | 30 | 0.4 | uncertified | 777 | 10.6371 | 11.0000 | 0.0077 |
| 20 | 20 | 0.1 | certified | 1181 | 1.6156 | 2.0000 | 0.1677 |
| 20 | 20 | 0.1 | uncertified | 819 | 3.1453 | 3.0000 | 0.1062 |
| 20 | 20 | 0.2 | certified | 1199 | 4.0367 | 5.0000 | 0.1418 |
| 20 | 20 | 0.2 | uncertified | 801 | 5.9950 | 6.0000 | 0.0262 |
| 20 | 20 | 0.4 | certified | 1223 | 8.0442 | 10.0000 | 0.1071 |
| 20 | 20 | 0.4 | uncertified | 777 | 9.1351 | 9.0000 | 0.0077 |
| 30 | 10 | 0.1 | certified | 1181 | 1.6215 | 2.0000 | 0.1660 |
| 30 | 10 | 0.1 | uncertified | 819 | 2.0855 | 2.0000 | 0.1453 |
| 30 | 10 | 0.2 | certified | 1199 | 4.0392 | 5.0000 | 0.1460 |
| 30 | 10 | 0.2 | uncertified | 801 | 4.5943 | 5.0000 | 0.0212 |
| 30 | 10 | 0.4 | certified | 1223 | 8.0425 | 10.0000 | 0.1104 |
| 30 | 10 | 0.4 | uncertified | 777 | 7.9009 | 8.0000 | 0.0090 |

### Does certified NCA's mean Hamming distance trend toward 0 as T-k grows? (diagnostic population)

| d | mean Hamming @ T-k=10 | @ T-k=20 | @ T-k=30 | monotonically non-increasing in T-k? |
| --- | --- | --- | --- | --- |
| 0.1 | 1.6215 | 1.6156 | 1.6139 | YES |
| 0.2 | 4.0392 | 4.0367 | 4.0350 | YES |
| 0.4 | 8.0425 | 8.0442 | 8.0343 | NO |

## The rigorous test: continuous state-distance geometric bound (diagnostic population, certified genomes only)

For each certified genome, predicted_final_distance = L(theta)^(T-k) * post_damage_distance (the state distance measured immediately after the perturbation at step k). The theory requires observed final_state_distance <= predicted_final_distance (up to floating-point slack) -- this is the actual pass/fail criterion for Corollary 3, independent of whether the discrete readout has fully re-converged yet.

| k | d | n certified | bound violations | max ratio (median) | max ratio (p90) | max ratio (worst) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.1 | 1181 | 0 | 0.998 | 1.000 | 1.000 |
| 10 | 0.2 | 1199 | 0 | 0.998 | 1.000 | 1.000 |
| 10 | 0.4 | 1223 | 0 | 0.998 | 1.000 | 1.000 |
| 20 | 0.1 | 1181 | 0 | 0.999 | 1.000 | 1.000 |
| 20 | 0.2 | 1199 | 0 | 0.999 | 1.000 | 1.000 |
| 20 | 0.4 | 1223 | 0 | 0.999 | 1.000 | 1.000 |
| 30 | 0.1 | 1181 | 0 | 0.999 | 1.000 | 1.000 |
| 30 | 0.2 | 1199 | 0 | 0.999 | 1.000 | 1.000 |
| 30 | 0.4 | 1223 | 0 | 0.999 | 1.000 | 1.000 |

**Theory verdict: HOLDS.** 0/10809 certified-genome trials violated the predicted geometric bound (observed state distance after recovery never exceeded L(theta)^(T-k) times the post-damage distance). Corollary 3's core claim -- that certified genomes recover from bounded mid-trajectory damage at (at least) the guaranteed contraction rate -- is confirmed, not merely illustrated.

## rule_based: recovery Hamming distance vs (T-k) (no certificate concept)

| k | T-k | d | n | mean Hamming | median Hamming | fraction fully recovered |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 30 | 0.1 | 2000 | 0.4270 | 0.0000 | 0.7480 |
| 10 | 30 | 0.2 | 2000 | 0.9055 | 0.0000 | 0.6185 |
| 10 | 30 | 0.4 | 2000 | 1.5095 | 0.0000 | 0.5195 |
| 20 | 20 | 0.1 | 2000 | 0.4260 | 0.0000 | 0.7495 |
| 20 | 20 | 0.2 | 2000 | 0.9035 | 0.0000 | 0.6210 |
| 20 | 20 | 0.4 | 2000 | 1.5240 | 0.0000 | 0.5250 |
| 30 | 10 | 0.1 | 2000 | 0.4210 | 0.0000 | 0.7515 |
| 30 | 10 | 0.2 | 2000 | 0.8740 | 0.0000 | 0.6245 |
| 30 | 10 | 0.4 | 2000 | 1.4405 | 0.0000 | 0.5285 |

## direct: no development, hence no recovery process (explicit asymmetry)

DirectEncoder's genome IS the phenotype -- there is no intermediate developmental state and therefore no step k, no elapsed recovery time T-k, and no possibility of self-repair by construction. The only meaningful analogue measured here is damaging the genome ITSELF (setting a random d-fraction of its 25 cells to EMPTY) before a single one-shot decode(); this number is the same regardless of when in a hypothetical 'trajectory' the damage would have occurred, because there is no trajectory. It is reported once per d, not as a function of k or T-k.

| d | n | mean Hamming | median Hamming | fraction unaffected (Hamming=0) |
| --- | --- | --- | --- | --- |
| 0.1 | 2000 | 0.8245 | 1.0000 | 0.3470 |
| 0.2 | 2000 | 2.0000 | 2.0000 | 0.0770 |
| 0.4 | 2000 | 3.9955 | 4.0000 | 0.0050 |

**This is the structural finding for direct, not a missing measurement**: since damage to a d-fraction of the genome changes exactly that same d-fraction of the phenotype (decode() is a pure reshape), direct's mean Hamming distance is, by construction, `n_damaged * (1 - empty_prob)` -- the number of damaged cells, discounted by the fraction that were already EMPTY before the damage (a cell set to EMPTY that was already EMPTY produces no Hamming change). With `empty_prob=0.6` (the default), that predicts means of 2*0.4=0.80, 5*0.4=2.00, 10*0.4=4.00 for d=0.1/0.2/0.4 -- matching the measured 0.8245/2.0000/3.9955 above closely (small residual gap from `n_damaged`'s own rounding). The exact figure is a detail of the empty-cell base rate; the STRUCTURAL point is that whatever cells the damage actually changes STAY changed forever regardless of any 'recovery time' -- there is no mechanism by which it COULD be lower, because there is no process running after the damage that could undo it. This is precisely the property axis (regeneration) on which a developmental encoding can structurally outperform a direct one, and direct's numbers above are the zero-recovery baseline that NCA's certified genomes are compared against.

## Notes

No further action required: the geometric bound held with zero violations across every (k, d) cell tested. The monotonic-Hamming-trend check above is a secondary, softer illustration of the same result at the discrete-readout level, and is expected to be noisier (and not necessarily monotonic at every d, since Hamming is a discrete function of a continuous quantity, near ties can transiently move it non-monotonically) than the rigorous state-distance test -- that is expected and not itself a theory failure; see this module's own docstring and results/nca/CONVERGENCE.md for why argmax-readout convergence can lag the underlying state contraction.

This module never computes or consults a fitness, reward, or task-performance quantity: every number above is derived from decode()/develop()/_step() alone.

