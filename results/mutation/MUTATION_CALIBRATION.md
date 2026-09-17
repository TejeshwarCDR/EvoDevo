# Mutation calibration: equalising phenotypic displacement

Fitness-blind (evodevo/analysis/mutation_calibration.py; only decode()/mutate()/validate_body are ever called -- NO PPO, NO task, NO reward, NO fitness quantity anywhere in this module). N=2000 valid random parents per grid point, 10-point grid per arm, init_condition=`center_seed` (P3), dev_steps=20 (P4 T_primary), seed=0 (sweep) / 10000 (independent verification sample).

## Displacement vs. parameter sweep

![displacement vs parameter](figs/displacement_vs_param.png)

### direct (`mutation_rate`)

| param | mean | median | q25 | q75 | iqr | std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0100 | 0.1495 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3836 |
| 0.0533 | 0.8855 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9335 |
| 0.0967 | 1.4935 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | 1.1734 |
| 0.1400 | 2.2915 | 2.0000 | 1.0000 | 3.0000 | 2.0000 | 1.4455 |
| 0.1833 | 2.9590 | 3.0000 | 2.0000 | 4.0000 | 2.0000 | 1.6478 |
| 0.2267 | 3.6735 | 4.0000 | 2.0000 | 5.0000 | 3.0000 | 1.8083 |
| 0.2700 | 4.3905 | 4.0000 | 3.0000 | 6.0000 | 3.0000 | 1.8926 |
| 0.3133 | 5.1200 | 5.0000 | 4.0000 | 6.0000 | 2.0000 | 2.0036 |
| 0.3567 | 5.8145 | 6.0000 | 4.0000 | 7.0000 | 3.0000 | 2.1338 |
| 0.4000 | 6.5430 | 6.0000 | 5.0000 | 8.0000 | 3.0000 | 2.2568 |

### nca (`mutation_sigma`)

| param | mean | median | q25 | q75 | iqr | std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0100 | 0.9065 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0415 |
| 0.0644 | 5.1480 | 5.0000 | 3.0000 | 7.0000 | 4.0000 | 2.9710 |
| 0.1189 | 8.3215 | 8.0000 | 6.0000 | 11.0000 | 5.0000 | 3.8709 |
| 0.1733 | 10.7270 | 11.0000 | 8.0000 | 14.0000 | 6.0000 | 4.2583 |
| 0.2278 | 12.6060 | 13.0000 | 10.0000 | 16.0000 | 6.0000 | 4.4756 |
| 0.2822 | 14.0410 | 14.0000 | 11.0000 | 17.0000 | 6.0000 | 4.4823 |
| 0.3367 | 15.1865 | 16.0000 | 12.0000 | 18.0000 | 6.0000 | 4.4194 |
| 0.3911 | 16.0345 | 16.0000 | 13.0000 | 19.0000 | 6.0000 | 4.3061 |
| 0.4456 | 16.7165 | 17.0000 | 14.0000 | 20.0000 | 6.0000 | 4.1942 |
| 0.5000 | 17.2735 | 18.0000 | 15.0000 | 20.0000 | 5.0000 | 4.0741 |

### rule_based (`mutation_rate`)

| param | mean | median | q25 | q75 | iqr | std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0200 | 0.6460 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.2884 |
| 0.0733 | 2.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.8212 |
| 0.1267 | 4.0120 | 0.0000 | 0.0000 | 5.0000 | 5.0000 | 7.2780 |
| 0.1800 | 5.4445 | 0.0000 | 0.0000 | 9.0000 | 9.0000 | 8.0789 |
| 0.2333 | 6.7555 | 1.0000 | 0.0000 | 14.0000 | 14.0000 | 8.6380 |
| 0.2867 | 7.8735 | 4.0000 | 0.0000 | 17.0000 | 17.0000 | 8.9766 |
| 0.3400 | 9.0015 | 6.0000 | 0.0000 | 18.2500 | 18.2500 | 9.1393 |
| 0.3933 | 10.0225 | 7.0000 | 0.0000 | 19.0000 | 19.0000 | 9.2753 |
| 0.4467 | 10.8565 | 9.0000 | 1.0000 | 20.0000 | 19.0000 | 9.2569 |
| 0.5000 | 11.5245 | 11.0000 | 2.0000 | 20.0000 | 18.0000 | 9.2032 |

## Target displacement D*

| arm | achievable mean-displacement range within its grid |
| --- | --- |
| direct | [0.1495, 6.5430] |
| nca | [0.9065, 17.2735] |
| rule_based | [0.6460, 11.5245] |

Jointly achievable range across all three arms: [0.9065, 6.5430]. Recommended target band: 2.0-3.0 cells changed out of 25 -- too small and mutation is mostly neutral (the EA barely explores beyond its parent), too large and the EA degenerates toward random search (each offspring becomes nearly unrelated to its parent, destroying the locality mutation-based search depends on).

**Outcome (a): NCA's displacement curve now reaches into the originally recommended 2.0-3.0 band.** nca's grid floor (`mutation_sigma`=0.01) now produces a mean displacement of 0.9065 cells -- above the previous re-run's floor of 0.2235 cells at the same nominal `mutation_sigma`=0.01 (the comparison point cited by OLD_* in this module is the MOST RECENT prior calibration, not necessarily the original pre-N1 unbounded architecture -- see the 'Comparison to the original' section below for which run that actually is). D* is RE-SELECTED within the recommended band and ALL THREE ARMS are re-calibrated to it: **Chosen D* = 2.5000** (was 2.5000).

## Solved parameter values and verification

Each arm's parameter was solved by interpolating its sweep curve for D*=2.5000, then verified on a FRESH, independent N=2000 sample (different seed) -- stated tolerance: |mean - D*| <= 0.5 cells.

| arm | parameter | solved value | verification mean | verification median | verification IQR | within tolerance? |
| --- | --- | --- | --- | --- | --- | --- |
| direct | mutation_rate | 0.1535 | 2.5030 | 2.0000 | 2.0000 | YES |
| nca | mutation_sigma | 0.0305 | 2.6885 | 2.0000 | 3.0000 | YES |
| rule_based | mutation_rate | 0.0779 | 2.6330 | 0.0000 | 0.0000 | YES |

## Comparison to the original (P5, 2026-09-15) calibration

Reference point only -- the original run is never recomputed here, just cited. Original: D*=2.5000, nca's old unbounded architecture, `init_scale`-era genome. This re-run: D*=2.5000, nca at bounded_leaky/(lambda\*=0.2, B\*=5.0)/H=0/N2 layer-wise-scaled genome.

| arm | old param | old verify mean | old IQR | new param | new verify mean | new IQR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 0.1535 | 2.5030 | 2.0000 | 0.1535 | 2.5030 | 2.0000 |
| nca | 0.1177 | 2.4235 | 3.0000 | 0.0305 | 2.6885 | 3.0000 |
| rule_based | 0.0781 | 2.6705 | 1.0000 | 0.0779 | 2.6330 | 0.0000 |

**Cross-arm IQR spread**: old=2.0000 cells (IQRs direct=2.0000, nca=3.0000, rule_based=1.0000), new=3.0000 cells (IQRs direct=2.0000, nca=3.0000, rule_based=0.0000). 
The spread did NOT narrow (widened by 1.0000 cells).

**NCA's own IQR specifically**: old=3.0000 cells (at D*=2.5000), new=3.0000 cells (at D*=2.5000). 
NCA's IQR did not narrow (changed by -0.0000 cells).

## Calibrated distributions side by side

![calibrated distributions](figs/calibrated_distributions.png)

Means at calibration (all targeting D*=2.5000): direct=2.5030, nca=2.6885, rule_based=2.6330.

IQRs at calibration: direct=2.0000, nca=3.0000, rule_based=0.0000 (spread across arms: 3.0000 cells). 
**The tails do NOT match as well as the means do**: despite all three arms hitting essentially the same mean displacement, their IQRs differ by 3.0000 cells -- i.e. this calibration equalises the first moment of phenotypic displacement but not its spread, and the tails (which drive search) can still differ materially between arms even after calibration. Q25/Q75 by arm: direct=[1.0000, 3.0000], nca=[1.0000, 4.0000], rule_based=[0.0000, 0.0000].

## Limitations (threats to validity)

Matching the first moment (mean) of phenotypic displacement does NOT equalise the mutation operators' underlying structure, and this limitation applies regardless of how well the calibration above numerically succeeds. Rule-based mutation is symbolic and discontinuous: a single mutated rule can change the interpreter's control flow for the rest of development (e.g. changing which condition an entire branch matches), producing a step-function-like effect on the phenotype. NCA mutation perturbs a fraction of the network's weights by a small additive Gaussian nudge applied simultaneously across the whole genome, changing every subsequent developmental step's computation by a small, continuous amount rather than altering one discrete decision. Direct mutation independently re-samples a subset of the 25 cells' categorical values directly in phenotype space, with no development in between at all. These are three qualitatively different operators that happen, at the calibrated settings, to move the decoded body by a similar expected Hamming distance -- they are not thereby equivalent operators, and any conclusion that depends on the operators' higher-order behaviour (e.g. how often a mutation is neutral vs. catastrophic, or how mutation effects compose over multiple generations) is not covered by this calibration and remains an open threat to validity.
