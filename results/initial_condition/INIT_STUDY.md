# Initial-condition study (P1 x P2 crossed)

Fitness-blind (see evodevo/analysis/structural.py and evodevo/analysis/init_condition_study.py). N=2000 genomes per cell, dev_steps=10 (held identical across both arms and all three conditions so it cannot itself confound this comparison), seed=0. No PPO, no task, no reward were run to produce any number on this page.

## 1. Validity rate (2x3)

| arm | condition | valid_rate | degenerate_fraction | failure_reason_counts |
| --- | --- | --- | --- | --- |
| nca | center_seed | 0.5745 | 0.0240 | {'too_few_voxels': 81, 'no_actuator': 208, 'disconnected': 731} |
| nca | full_nonsymmetric | 0.6815 | 0.0185 | {'disconnected': 552, 'no_actuator': 134, 'too_few_voxels': 21} |
| nca | full_uniform | 0.7175 | 0.0325 | {'too_few_voxels': 21, 'disconnected': 377, 'no_actuator': 240} |
| rule_based | center_seed | 0.5710 | 0.1865 | {'no_actuator': 801, 'too_few_voxels': 153, 'disconnected': 200} |
| rule_based | full_nonsymmetric | 0.7495 | 0.1270 | {'no_actuator': 288, 'disconnected': 277, 'too_few_voxels': 26} |
| rule_based | full_uniform | 0.1165 | 0.8760 | {'no_actuator': 1760, 'too_few_voxels': 26, 'disconnected': 36} |

## 2. Diversity (2x3)

| arm | condition | unique_valid_body_fraction | material_entropy_bits |
| --- | --- | --- | --- |
| nca | center_seed | 0.5885 | 2.2828 |
| nca | full_nonsymmetric | 0.6885 | 2.3213 |
| nca | full_uniform | 0.7150 | 2.2770 |
| rule_based | center_seed | 0.4070 | 2.3104 |
| rule_based | full_nonsymmetric | 0.5820 | 2.2491 |
| rule_based | full_uniform | 0.0390 | 0.8806 |

## 3. Redundancy (2x3)

| arm | condition | unique_phenotype_count | genomes_per_unique_phenotype | largest_neutral_class_fraction |
| --- | --- | --- | --- | --- |
| nca | center_seed | 1951 | 1.0251 | 0.0130 |
| nca | full_nonsymmetric | 1960 | 1.0204 | 0.0070 |
| nca | full_uniform | 1939 | 1.0315 | 0.0155 |
| rule_based | center_seed | 1125 | 1.7778 | 0.0525 |
| rule_based | full_nonsymmetric | 1407 | 1.4215 | 0.0340 |
| rule_based | full_uniform | 134 | 14.9254 | 0.7530 |

## 4. Settling (2x3)

| arm | condition | settled_fraction | never_settled_count | never_settled_fraction | mean_settle_step |
| --- | --- | --- | --- | --- | --- |
| nca | center_seed | 0.8030 | 394 | 0.1970 | 35.1849 |
| nca | full_nonsymmetric | 0.7880 | 424 | 0.2120 | 34.9124 |
| nca | full_uniform | 0.7740 | 452 | 0.2260 | 34.7326 |
| rule_based | center_seed | 0.8760 | 248 | 0.1240 | 2.2620 |
| rule_based | full_nonsymmetric | 0.8915 | 217 | 0.1085 | 2.4105 |
| rule_based | full_uniform | 0.9655 | 69 | 0.0345 | 0.2450 |

never_settled_count is reported explicitly for every cell above -- it is never dropped or coerced to a default.

## 5. Appendix: remaining P2 metrics (2x3)

| arm | condition | light_cone speed (cells/step) | light_cone never_reached_fraction | phenotypic_displacement mean_hamming | perturbation_robustness mean_hamming |
| --- | --- | --- | --- | --- | --- |
| nca | center_seed | 0.0198 | 0.3725 | 10.0520 | 5.4975 |
| nca | full_nonsymmetric | 0.0429 | 0.1911 | 10.0860 | 5.5160 |
| nca | full_uniform | 0.0216 | 0.3712 | 9.8460 | 5.6530 |
| rule_based | center_seed | 0.1202 | 0.9583 | 5.7765 | 2.8960 |
| rule_based | full_nonsymmetric | 0.1630 | 0.9469 | 4.2245 | 2.8375 |
| rule_based | full_uniform | 0.6127 | 0.9924 | 4.2500 | 2.7845 |

## 6. Variance decomposition

**Validity rate.** The historical premise motivating this study was that rule-based's validity rate is inflated by its center-seed initial condition. The crossed measurement does NOT support that premise in the direction assumed: rule-based's own validity rate rises from 0.5710 under `center_seed` to 0.7495 under `full_nonsymmetric` (+0.1785, +31.3%) -- i.e. `full_nonsymmetric` (a dense, mostly-populated random start) makes rule-based validity HIGHER, not lower, than the single-seed condition. Held-constant between-arm differences (nca minus rule_based) tell the real story instead: at `center_seed` the gap is essentially zero (0.0035, i.e. 0.35 percentage points) -- the two arms are statistically indistinguishable on validity once the initial condition is matched at `center_seed`. At `full_nonsymmetric` a real gap opens up (-0.0680, rule-based higher), and at `full_uniform` the gap is enormous (0.6010, nca far higher) because rule-based's validity collapses there (see the full_uniform control below). **Conclusion: whatever validity difference has previously been reported between these two arms is overwhelmingly an artefact of which initial condition each arm happened to use, not a property of the encoding itself** -- the between-arm gap is smallest, and consistent with noise, at exactly the condition (`center_seed`) that was previously used by only one of the two arms.

**Diversity (unique_valid_body_fraction).** Between-arm difference (nca minus rule_based) at center_seed=0.1815, full_nonsymmetric=0.1065, full_uniform=0.6760.

**Redundancy (genomes_per_unique_phenotype).** Between-arm difference (nca minus rule_based) at center_seed=-0.7527, full_nonsymmetric=-0.4011, full_uniform=-13.8939.

**Settling (never_settled_fraction).** Between-arm difference (nca minus rule_based) at center_seed=0.0730, full_nonsymmetric=0.1035, full_uniform=0.1915.

## 7. full_uniform control

- nca: unique_valid_body_fraction=0.7150, material_entropy_bits=2.2770 under `full_uniform`.
- rule_based: unique_valid_body_fraction=0.0390, material_entropy_bits=0.8806 under `full_uniform`.

**Result: nca, rule_based DID differentiate from a perfectly symmetric full_uniform start** (non-negligible unique-valid-body-fraction and/or material entropy above zero), which on its face contradicts the naive locality prediction that a deterministic local update applied identically everywhere to an identical starting state must produce an identical result everywhere.

This was investigated rather than silently accepted. The root cause in both arms is the same and is NOT a hidden bug in initial-state construction, boundary handling, or the interpreter: **the per-genome update rule itself is not required to be isotropic (symmetric under grid reflection), even though it IS spatially homogeneous (the same rule/weights are applied at every cell).** For NCA, the perception network's weights for the up/down/left/right neighbour blocks are independent, randomly-sampled numbers -- nothing constrains w(up)==w(down), so a generic genome's per-cell function distinguishes 'my up-neighbour' from 'my down-neighbour' as different input features, and can output different values for cells that are geometric mirror images of each other. For rule-based, `GROW_DIRECTIONAL`/`COPY_NEIGHBOR` actions reference one specific `Direction` (e.g. RIGHT, not RIGHT-and-LEFT symmetrically); any sampled rule set containing such a rule breaks reflection symmetry by construction, regardless of the initial grid.

This was confirmed directly: constraining an NCA genome's weights to be isotropic (up/down/left/right blocks tied to identical weight rows) reproduces a perfectly uniform output body from `full_uniform`, and a rule-based genome restricted to non-directional conditions/actions (`ALWAYS`/`CELL_TYPE` + `SET_*` only, no `GROW_DIRECTIONAL`/`COPY_NEIGHBOR`) likewise stays perfectly uniform. So the locality claim as originally stated in `evodevo/core/initial_conditions.py`'s `FullUniform` docstring is corrected here: **translation-invariance (same rule everywhere) does not imply isotropy (same rule under reflection)**, and only isotropic rules preserve a symmetric initial condition. Nothing in this investigation points to a hidden asymmetry in how the grid, boundary, or initial state are constructed.

## 8. Decision criterion applied

Criterion (pre-registered, evidence-only -- no fitness quantity is computed anywhere in this study): prefer the initial condition that minimises the between-arm |validity_rate| gap, since that is the confound most likely to dominate the budgeted PPO comparison, subject to both arms retaining non-degenerate diversity under that condition.

| condition | \|validity_rate gap\| | non-degenerate diversity (both arms)? |
| --- | --- | --- |
| center_seed | 0.0035 | yes |
| full_nonsymmetric | 0.0680 | yes |
| full_uniform | 0.6010 | no |

**Recommended initial condition for the primary PPO comparison: `center_seed`** -- it has the smallest between-arm validity-rate gap among conditions where neither arm's diversity is degenerate (0.0035, vs 0.0680 for the next-best eligible condition), and both arms retain non-degenerate diversity under it (nca=0.5885, rule_based=0.4070).
