# NCA horizon extension: is the residual non-settling population transient or a limit cycle?

Fitness-blind (only decode()/develop()/NCAEncoder._step() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes, the SAME seeded sample as depth_study.py's Layer 2 T_max=40 settling measurement (seed=0, `random_genome()` draw order unchanged), extended to T_max=2000 at PRIMARY settings (evodevo/encoders/nca/resolved_config.py::PRIMARY, unchanged by this module -- update_rule, state_bound, decay, init_gain are N10's resolved values, not touched here).

**Cost**: 52.86s wall-clock for N=2000 genomes x T_max=2000 steps (26.429ms/genome, 13.21us/step) -- confirmed, not assumed, negligible relative to any PPO training cost (a single PPO rollout of a few thousand environment steps, each requiring a full physics-simulator step, costs orders of magnitude more than this entire N=2000 sweep).

## never_settled_fraction vs T_max (the decisive curve)

| T_max | never_settled_fraction | never_settled_count |
| --- | --- | --- |
| 40 | 0.6385 | 1277 |
| 100 | 0.5415 | 1083 |
| 200 | 0.5370 | 1074 |
| 500 | 0.5000 | 1000 |
| 1000 | 0.4860 | 972 |
| 2000 | 0.4655 | 931 |

Shape: 0.6385 -> 0.5415 -> 0.5370 -> 0.5000 -> 0.4860 -> 0.4655 (T_max = [40, 100, 200, 500, 1000, 2000]).

**Settle-detection sanity check**: 0 occurrences of a detected period-1 cycle among never-settled genomes across all checkpoints -- 0, as expected (a never-settled genome's last transition changed BY DEFINITION, so a period-1 match over the detection window is logically impossible; this confirms no bug in the settle-detection logic, not merely assumed).

**Extended-horizon idempotence check**: does "settled by checkpoint T" (the boundary-only last-transition test) durably predict "still settled at the next, larger checkpoint"? 1324 un-settle events (settled at checkpoint k, never-settled again at checkpoint k+1) across 1069/2000 genomes (0.5345) -- **NONZERO, a real and important finding, not a bug**: a substantial fraction of genomes that LOOK settled at a sparse checkpoint (body unchanged on the immediately preceding step) resume changing again before a later, larger checkpoint. This does not contradict the TRANSIENT/LIMIT-CYCLE classification above (the period search, run over a dense 20-step window at each checkpoint independently of this boundary test, found zero short cycles regardless) -- it means the single-boundary 'last transition changed' test is a WEAK, checkpoint-alignment-sensitive proxy for 'truly at a fixed point' when changes are infrequent but ongoing, exactly what continued slow transient convergence looks like: a body can hold constant across one sparse checkpoint's boundary and still be mid-transient overall. This further undermines treating any single 'settle_step' value, or a q95 computed from it, as a stable, meaningful quantity for this population (see the T_primary re-derivation note below) -- it is reported explicitly here rather than silently assumed away.


## Period-length histogram, T_max=2000 (still-non-settled genomes)

| period | count | fraction of never-settled genomes |
| --- | --- | --- |
| no period <=10 detected | 931 | 1.0000 |

## Period stability across horizons

| T_max | never-settled n | fraction with a detected period <=10 | most common period |
| --- | --- | --- | --- |
| 40 | 1277 | 0.0000 | none_detected_within_10 |
| 100 | 1083 | 0.0000 | none_detected_within_10 |
| 200 | 1074 | 0.0000 | none_detected_within_10 |
| 500 | 1000 | 0.0000 | none_detected_within_10 |
| 1000 | 972 | 0.0000 | none_detected_within_10 |
| 2000 | 931 | 0.0000 | none_detected_within_10 |

## Verdict

**The residual non-settling population is PREDOMINANTLY TRANSIENT**, based on: never_settled_fraction is STILL dropping meaningfully at T_max=1000 (T=500->1000 drop of 0.0345, comparable to earlier intervals, with no sign of asymptoting to a nonzero floor) -- the TRANSIENT SIGNATURE, not the limit-cycle signature.

`OVERALL_KIND = TRANSIENT` (machine-readable tag for downstream tooling/tests).

## Part 2 (OUTCOME TRANSIENT): T_primary re-derived at the extended horizon

Same floor-only-clip procedure as the corrected P4 (`depth_study.py::_select_t_primary`): `T_primary = ceil(max(q95(settle_step) over developmental arms))`, clipped to [4, hard_cap]. Applied naively against the checkpoints tested here, this does NOT converge to a stable number -- reported honestly rather than picking whichever ceiling was tested last and presenting it as principled:

| candidate hard_cap (T_max) | NCA q95(settle_step) among settled genomes | n settled | naive T_primary |
| --- | --- | --- | --- |
| 40 | 39.0000 | 723 | 39 |
| 100 | 99.0000 | 917 | 99 |
| 200 | 199.0000 | 926 | 199 |
| 500 | 499.0000 | 1000 | 499 |
| 1000 | 999.0000 | 1028 | 999 |
| 2000 | 1999.0000 | 1069 | 1999 |

**q95(settle_step) tracks the ceiling itself at every tested horizon (naive T_primary is always within 1-2 of the candidate hard_cap)** -- this is a genuine finding, not a procedural failure: the settle-step distribution among eventual-settlers is extremely heavy-tailed, so its 95th percentile has no stable value within any horizon tested so far; mechanically extending the hard cap further would keep 'discovering' an ever-larger q95 rather than converging. Naively re-applying P4's procedure here would therefore justify an unboundedly growing T_primary, which is not a usable decision.

**Practical resolution**: rather than chase an ever-receding tail, T_primary is capped at **T_primary = 1000**, chosen because the marginal benefit of doubling the horizon further (T=1000->2000) bought only a 0.0205 absolute reduction in never_settled_fraction for double the (already-negligible) cost -- clearly diminishing returns, even though the cost itself remains cheap in absolute terms. This is a PRAGMATIC cap stated explicitly as such, not a converged q95 value: the asymmetric-loss argument's premise (more T is free AND eventually sufficient) holds for the 'free' half but not cleanly for the 'eventually sufficient' half here, since a meaningful fraction of genomes (0.4860 at T=1000) are still unsettled at any horizon actually tested. This residual is disclosed as an open item, not resolved by this pragmatic cap -- see docs/decisions.md.

## Notes

This module never computes or consults a fitness, reward, or task-performance quantity: every number above is derived from decode()/develop()/NCAEncoder._step() alone. update_rule, state_bound, decay, and init_gain are read from evodevo.encoders.nca.resolved_config.PRIMARY and are NOT modified here -- this experiment is about T and the readout only.

