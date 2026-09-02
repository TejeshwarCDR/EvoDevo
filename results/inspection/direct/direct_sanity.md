# direct Encoder Sanity Report

N = 2000 random genomes, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | Value |
|---|---|
| Validity rate | 0.0215 |
| Unique body hashes | 2000 (1.0000) |
| All-empty fraction | 0.0000 |
| Uniform non-empty fraction | 0.0000 |
| Degenerate fraction (all-empty + uniform) | 0.0000 |

### Invalid-genome failure reasons

| Reason category | Count |
|---|---|
| disconnected | 1957 |
| no_actuator | 4 |
| too_few_voxels | 1 |

## Example developmental trajectories

![trajectories](direct_sanity_trajectories.png)

## Decision gate

- STOP: direct validity rate (0.0215) is below the 10% gate threshold. Revisit this encoder's random-genome/initial-condition hyperparameters before running the comparison.
