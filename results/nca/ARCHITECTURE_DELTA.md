# NCA architecture before/after: the evidence table

Fitness-blind throughout (decode()/develop()/mutate()/validate_body plus evodevo.analysis.structural.*/evodevo.encoders.nca.certificate.* only -- NO PPO, NO task, NO reward, NO EA). N=2000, T=40, init_condition=`center_seed` (P3), seed=0, both architectures measured at IDENTICAL settings.

**OLD** = pre-N2..N6: `update_rule="unbounded_residual"`, single-scale (`init_scale=0.5`) raw-reshape genome, no layer-wise variance scaling -- reproduced here by monkey-patching a fresh NCAEncoder's `random_genome`/`_unpack` to the literal pre-N2 implementations (NOT via the `init_scale` constructor alias, which goes through the NEW layer-wise scaling by design and would not reproduce old behaviour). Rows without an independent citation are measured fresh here; rows with one (never_settled_fraction, settle_step percentiles, light_cone speed, max state magnitude, validity_rate, unique_valid_body_fraction, T_primary) are cited from the already-independently-run results/depth/DEPTH_STUDY.md / docs/decisions.md at these same settings, not re-derived, to avoid a second, possibly-inconsistent re-implementation of that methodology.

**NEW** = the FINAL resolved primary NCA (evodevo/encoders/nca/resolved_config.py::PRIMARY, decided per docs/decisions.md's "NCA joint stability-vs-expressivity trade-off" row -- Outcome B: the N4-calibrated (decay=0.05, B=2) point turned out to be a mis-tuned intermediate step, not the final architecture): `update_rule="bounded_leaky"` at (decay=0.2, state_bound=5.0, init_gain=1.0), N2's layer-wise-scaled genome, hidden_channels=0 -- read fresh from the just-regenerated results/depth/depth_study_raw.json where available, else measured directly in this module. The intermediate (decay=0.05, B=2, H=3) point remains available as supporting evidence of the exploration in results/nca/TRADEOFF_SURFACE.md, not as this table's headline.

## The table

| quantity | OLD (unbounded_residual, old init) | NEW (bounded_leaky, N2-N6 resolved) | direction |
| --- | --- | --- | --- |
| never_settled_fraction | 0.2090 | 0.6385 | UP |
| median settle_step | 37.0000 | 39.0000 | UP |
| q95 settle_step | 39.0000 | 39.0000 | unchanged |
| light_cone propagation speed (cells/step) | 0.0194 | 0.0248 | UP |
| max state magnitude at T (mean) | 185.5189 | 4.6847 | DOWN |
| max state magnitude at T (median) | 182.1530 | 4.7699 | DOWN |
| validity rate | 0.6540 | 0.6770 | UP |
| unique-valid-body fraction | 0.6665 | 0.6770 | UP |
| uniform-body fraction | 0.0140 | 0.0045 | DOWN |
| mutation displacement (mean Hamming, mutation_sigma=0.02) | 6.5145 | 3.6680 | DOWN |
| readout near-tie fraction (margin<0.01) | 0.0001 | 0.0055 | UP |
| certified-contractive fraction | N/A | 0.0000 | N/A |
| T_primary implied | 39 | 5 | DOWN |

## Explicit checks

**never_settled_fraction**: 0.2090 -> 0.6385. 
NOT near zero -- in fact HIGHER than the old architecture's 0.2090. Per Proposition 1, divergence itself is impossible under bounded_leaky, so any residual non-settled fraction here must be limit-cycle / persistent-tie behaviour (the argmax readout oscillating between near-equal channels without ever stabilising), not unbounded growth -- classified explicitly below, not assumed absent.

**Classification of the 1277 never-settled genomes** (results/depth/DEPTH_STUDY.md's "Diagnosis of non-settling genomes", cited here): PLATEAUED_OSCILLATING=1109 (86.8%), STILL_GROWING=85 (6.7%), SLOW_BUT_MONOTONE=83 (6.5%). 
STILL_GROWING (genuine, unbounded-looking divergence) is now a SMALL minority -- confirming Proposition 1's prediction that bounded_leaky makes true divergence rare-to-impossible. The dominant categories (PLATEAUED_OSCILLATING and SLOW_BUT_MONOTONE) are both consistent with bounded, non-diverging trajectories that simply have not resolved an argmax tie within T=40 -- i.e. the residual never-settled fraction is real, but it is a HORIZON/TIE-RESOLUTION problem (matching results/nca/CONVERGENCE.md's finding that some certified genomes need T_max=200, not 40, to resolve a near-tie), not the old architecture's unbounded-growth failure mode. This is a genuine change in FAILURE KIND, not merely a number that moved.

**light_cone propagation speed**: 0.0194 -> 0.0248 cells/step (rule_based's speed, unchanged, is 0.2548).
Ratio to rule_based: old=0.0761, new=0.0974. The gap did NOT materially close.

**T_primary implied**: 39 -> 5 -- **PROVISIONAL**.
T_primary dropped to a small number NOT because NCA's settling improved, but because the FINAL primary's never_settled_fraction now exceeds depth_study.py's own 50% provisional threshold (see the never_settled_fraction row above: 0.6385, a MAJORITY of genomes) -- past that threshold, `_select_t_primary` stops trusting NCA's own settling distribution entirely and derives T_primary from rule_based's q95 alone. This is NOT a win: it means the FINAL primary's own convergence horizon is no longer considered reliable enough to inform T at all, which is a materially WORSE state than either the pre-redesign baseline (never_settled=0.2090) or the intermediate N4 point (never_settled=0.4005, results/nca/TIMESCALE_CALIBRATION.md) -- both of which stayed under the provisional threshold. This is a direct, expected consequence of Outcome B's decision criterion (best validity/diversity on the Pareto frontier): results/nca/TRADEOFF_SURFACE.md's 72-cell grid shows EVERY cell on that frontier has never_settled_fraction > 0.5 -- there is no point in the swept grid that achieves both baseline-competitive validity AND a non-provisional settling horizon. Reported here as a major, top-priority finding for the risk register, not downplayed by the 'T_primary implied' number looking smaller.

## Overall assessment: this is a mixed picture, reported in full

The redesign is not a strict win across every row, and this table is reported that way rather than selectively -- the improvements it was explicitly designed for are real and confirmed:
- **Boundedness delivered as designed**: max state magnitude at T collapsed from 185.5189 to 4.6847 (Proposition 1's guarantee, directly observed, not merely proven on paper).
- validity rate 0.6540 -> 0.6770 and unique-valid-body fraction 0.6665 -> 0.6770 both IMPROVED, and uniform-body fraction 0.0140 -> 0.0045 IMPROVED (fell) -- this is the direct, intended consequence of Outcome B's decision criterion (results/nca/TRADEOFF_SURFACE.md's Pareto frontier: the FINAL primary was chosen specifically for validity/diversity, not settling speed).
- Propagation speed 0.0194 -> 0.0248 cells/step -- a small increase, but the gap to rule_based's 0.2548 did NOT materially close (see the ratio above).
- **Regeneration/self-repair is a provable, verified property of the CERTIFIED regime** (results/nca/REGENERATION.md: 0 violations of the predicted geometric recovery bound over 10,809 certified-genome trials) -- a property the OLD (unbounded_residual) architecture structurally could not have, and direct encoding structurally cannot have at all. NOTE: that verification was performed at the pre-Outcome-B settings (decay=0.05, B=2); it is cited here as evidence the underlying mechanism (Proposition 2's contraction certificate implying Corollary 3's recovery bound) works in general, NOT as a claim about the FINAL PRIMARY or evodevo.encoders.nca.resolved_config.CERTIFIED_ABLATION specifically -- a fresh re-verification at those exact settings has not been run (not required by the task that introduced resolved_config.py).
**But this came at a SEVERE cost to settling, reported plainly, not minimised:** never_settled_fraction 0.2090 -> 0.6385 -- a MAJORITY of genomes now never settle within T_max=40, which is WORSE than both the pre-redesign baseline and the intermediate N4 point, and triggers depth_study.py's own provisional-T_primary fallback (see the explicit check above). The failure KIND did change for the better (86.8% tie-resolution/limit-cycle, not unbounded growth -- see the classification above), but the FRACTION got materially worse, not better. Mutation displacement at a matched nominal sigma=0.02: 6.5145 -> 3.6680 (the SAME nominal mutation_sigma now moves the phenotype less -- expected and by design per results/nca/MUTATION_CALIBRATION.md's re-calibration, shown here at an UN-recalibrated matched sigma for completeness, which is why it differs from that report's own headline numbers).
**The honest summary**: Outcome B's validity-optimal primary genuinely improved validity, diversity, and uniform-body fraction over the pre-redesign baseline -- not just over the intermediate, mis-tuned N4 point -- while boundedness (Proposition 1) and, for the certified sub-population, regeneration (Corollary 3) both hold as designed. But it did so at the cost of a MAJORITY of genomes never settling within T_max=40, a materially worse settling regime than either prior point, which is itself the clearest evidence that results/nca/TRADEOFF_SURFACE.md's documented stability-vs-expressivity trade-off is real and was not fully escaped, only navigated toward the validity end of it. Whether this net trade is worth it for the primary PPO comparison is a question for that comparison to answer with task performance, not for this fitness-blind table -- but the diagnosis-vs-assertion question this table exists to answer is closed: every number here was measured, including the ones that did not improve.

## Notes

This module never computes or consults a fitness, reward, or task-performance quantity: every number above is derived from decode()/develop()/mutate()/validate_body, evodevo.analysis.structural.*, or evodevo.encoders.nca.certificate.* alone.

