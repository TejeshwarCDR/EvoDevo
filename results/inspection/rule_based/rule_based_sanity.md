> **SUPERSEDED as of 2026-09-19 (architecture context, not the numbers
> themselves).** This validity_rate=0.5975 was measured under the
> pre-redesign rule_based defaults (`conflict_resolution="row_major_last_wins"`,
> `position_predicate="enabled"`) — before R2/R3
> (`docs/decisions.md`'s "Make the rule_based interpreter provably
> order-independent by default" and "Restore translation-equivariance
> (Proposition R3)" rows). The current resolved PRIMARY
> (`evodevo/encoders/rule_based/resolved_config.py::PRIMARY`:
> `conflict_resolution="first_rule_index"`, `position_predicate="disabled"`)
> measures validity_rate=0.5470 fresh (`results/rule_based/ARCHITECTURE_DELTA.md`).

# rule_based Encoder Sanity Report

N = 2000 random genomes, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | Value |
|---|---|
| Validity rate | 0.5975 |
| Unique body hashes | 1096 (0.5480) |
| All-empty fraction | 0.0540 |
| Uniform non-empty fraction | 0.1205 |
| Degenerate fraction (all-empty + uniform) | 0.1745 |

### Invalid-genome failure reasons

| Reason category | Count |
|---|---|
| disconnected | 149 |
| no_actuator | 766 |
| too_few_voxels | 118 |

## Example developmental trajectories

![trajectories](rule_based_sanity_trajectories.png)

## Decision gate

- PASS: validity rate and degenerate-body fraction are both within the decision-gate thresholds.
