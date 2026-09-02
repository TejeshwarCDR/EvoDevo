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
