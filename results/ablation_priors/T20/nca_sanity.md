# NCA Developmental Engine Sanity Report (E0)

N = 500 random genomes per encoder, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | NCA | Direct (reference) |
|---|---|---|
| Validity rate | 0.6840 | 0.0140 |
| Unique body hashes | 498 (0.9960) | 500 (1.0000) |
| All-empty fraction | 0.0040 | 0.0000 |
| Uniform non-empty fraction | 0.0080 | 0.0000 |
| Degenerate fraction (all-empty + uniform) | 0.0120 | 0.0000 |

### Invalid-genome failure reasons

| Reason category | NCA count | Direct count |
|---|---|---|
| disconnected | 142 | 493 |
| no_actuator | 24 | 1 |
| too_few_voxels | 6 | 0 |

## Developmental stability (NCA only)

Never-settled fraction (still changing on the final transition, t=19 -> t=20): 0.4220

Settle-step distribution (step index after which the argmax body never changes again):

| Step | Count |
|---|---|
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |
| 5 | 0 |
| 6 | 0 |
| 7 | 1 |
| 8 | 1 |
| 9 | 4 |
| 10 | 3 |
| 11 | 3 |
| 12 | 4 |
| 13 | 11 |
| 14 | 7 |
| 15 | 13 |
| 16 | 25 |
| 17 | 39 |
| 18 | 66 |
| 19 | 110 |
| 20 | 211 |

## Residual state magnitude at t=T (NCA only)

max |state| across cells/channels at t=T, across genomes: mean=88.608, median=87.247, min=44.715, max=165.150

## Example developmental trajectories

![trajectories](nca_sanity_trajectories.png)

## Decision gate

- PASS: validity rate and degenerate-body fraction are both within the decision-gate thresholds. See docs/decisions.md for the recorded numbers.
