# NCA readout instrumentation and H (hidden-channel) ablation

Fitness-blind (evodevo.analysis.structural only -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes/cell, init_condition=`center_seed` (the P3 decision), T_max=40, update_rule=bounded_leaky at (lambda\*=0.2, B\*=5.0) (docs/decisions.md, 'NCA timescale calibration'), seed=0. The argmax readout rule itself is unchanged everywhere in this study -- these metrics instrument it, they do not redesign it.

## 1. What fraction of material decisions are near-ties?

| H | mean margin | median margin | fraction of CELLS near-tie (<0.01) | fraction of BODIES with >=1 near-tie cell |
| --- | --- | --- | --- | --- |
| 0 | 1.5005 | 1.1443 | 0.0057 | 0.1300 |
| 3 | 1.5158 | 1.1574 | 0.0052 | 0.1190 |
| 6 | 1.5259 | 1.1678 | 0.0055 | 0.1270 |

At the primary setting (H=0): 0.57% of individual cell material decisions are near-ties (margin < 0.01, effectively a coin flip), and 13.0% of bodies contain at least one such cell -- i.e. a non-trivial minority of decoded phenotypes have their material assignment resting on a near-tie in at least one cell, even though the aggregate margin distribution is not dominated by ties (mean/median margin well above the threshold).

## 2. What fraction of bodies are degenerate/uniform? Has the init change helped?

| H | uniform_fraction | by material (0=empty..4) |
| --- | --- | --- |
| 0 | 0.0095 | 0:0.0050, 1:0.0015, 2:0.0025, 3:0.0005, 4:0.0000 |
| 3 | 0.0070 | 0:0.0040, 1:0.0000, 2:0.0010, 3:0.0010, 4:0.0010 |
| 6 | 0.0045 | 0:0.0020, 1:0.0015, 2:0.0005, 3:0.0005, 4:0.0000 |

Old init (pre-N2, single-scale `init_scale=0.5`, unbounded_residual): degenerate_fraction = 0.0195 (docs/decisions.md, 2026-09-02 'Pre-launch audit reconfirmation of NCA PPO-free calibration' row (init_scale=0.5, unbounded_residual, full_nonsymmetric, dev_steps=10, N=2000, seed=0)).
New init (H=0, layer-wise-scaled, bounded_leaky at lambda\*/B\*): uniform_fraction = 0.0095.
**The init change reduced the degenerate/uniform-body fraction** by 0.0100 (51.3% relative).

## 3. Does H=0 differ measurably from H=3 (and H=6)?

| H | valid_rate | degenerate_fraction | unique_valid_body_fraction | material_entropy_bits | never_settled_fraction | mean settle_step | mean margin | fraction cells near-tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6770 | 0.0045 | 0.6635 | 2.3166 | 0.6255 | 36.6676 | 1.5005 | 0.0057 |
| 3 | 0.6865 | 0.0080 | 0.6760 | 2.3196 | 0.6715 | 37.5008 | 1.5158 | 0.0052 |
| 6 | 0.6850 | 0.0060 | 0.6785 | 2.3198 | 0.6945 | 37.3453 | 1.5259 | 0.0055 |

### H=0 vs H=3 (primary), relative gap per metric

| metric | H=0 | H=3 | relative gap |
| --- | --- | --- | --- |
| valid_rate | 0.6770 | 0.6865 | 1.38% |
| degenerate_fraction | 0.0045 | 0.0080 | 43.75% |
| unique_valid_body_fraction | 0.6635 | 0.6760 | 1.85% |
| material_entropy_bits | 2.3166 | 2.3196 | 0.13% |
| never_settled_fraction | 0.6255 | 0.6715 | 6.85% |
| mean_margin | 1.5005 | 1.5158 | 1.01% |
| fraction_cells_near_tie | 0.0057 | 0.0052 | 8.39% |

### Hidden channel utilisation directly (H=3, H=6)

| H | channel | spatial variance (across cells, t=T) | temporal variance (across steps) | correlation with final material (t=T/2) |
| --- | --- | --- | --- | --- |
| 3 | 0 | 5.8593 | 2.2453 | -0.0121 |
| 3 | 1 | 5.5858 | 2.2185 | -0.0073 |
| 3 | 2 | 5.8160 | 2.2435 | 0.0046 |
| 6 | 0 | 5.6226 | 2.1966 | -0.0055 |
| 6 | 1 | 5.8127 | 2.2328 | 0.0046 |
| 6 | 2 | 5.7744 | 2.2246 | -0.0007 |
| 6 | 3 | 5.6723 | 2.2266 | 0.0162 |
| 6 | 4 | 5.7571 | 2.2373 | -0.0044 |
| 6 | 5 | 5.9236 | 2.2630 | -0.0052 |

Correlation caveat: material ids are categorical, not ordinal, so a near-zero Pearson correlation is suggestive but not conclusive evidence of no relationship (see evodevo/analysis/structural.py::hidden_utilisation docstring).

### Verdict

**The hidden channels do not measurably affect the encoding's structural properties.** Every structural metric with at least one side above the 0.03 absolute floor (excluding tiny-baseline metrics like degenerate_fraction, where a 43.75%-looking 'relative gap' is really just a 0.35-percentage-point absolute difference and not a meaningful signal -- shown in the table above regardless, nothing hidden) differs by less than 10% (largest such gap: 6.85%), and hidden_utilisation's own per-channel correlation with the final material readout is small for every channel at both H=3 and H=6 (table above). This is reported as the finding, not softened: three (or six) hidden channels appear to be carried through development without materially changing validity, diversity, settling behaviour, or decision confidence, at least at these settings -- a legitimate, publishable result, and grounds to consider dropping them rather than carrying unexamined channels forward.

## Notes

This module never computes or consults a fitness, reward, or task-performance quantity; every number above is derived from decode()/develop() alone via evodevo.analysis.structural. The argmax readout rule is unchanged -- this study measures it, it does not redesign it.

