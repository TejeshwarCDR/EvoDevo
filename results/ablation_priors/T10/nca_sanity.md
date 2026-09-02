# NCA Developmental Engine Sanity Report (E0)

N = 500 random genomes per encoder, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | NCA | Direct (reference) |
|---|---|---|
| Validity rate | 0.6700 | 0.0140 |
| Unique body hashes | 498 (0.9960) | 500 (1.0000) |
| All-empty fraction | 0.0020 | 0.0000 |
| Uniform non-empty fraction | 0.0120 | 0.0000 |
| Degenerate fraction (all-empty + uniform) | 0.0140 | 0.0000 |

### Invalid-genome failure reasons

| Reason category | NCA count | Direct count |
|---|---|---|
| disconnected | 137 | 493 |
| no_actuator | 40 | 1 |
| too_few_voxels | 9 | 0 |

## Developmental stability (NCA only)

Never-settled fraction (still changing on the final transition, t=9 -> t=10): 0.6620

Settle-step distribution (step index after which the argmax body never changes again):

| Step | Count |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 1 |
| 3 | 2 |
| 4 | 2 |
| 5 | 2 |
| 6 | 4 |
| 7 | 15 |
| 8 | 36 |
| 9 | 106 |
| 10 | 331 |

## Residual state magnitude at t=T (NCA only)

max |state| across cells/channels at t=T, across genomes: mean=40.761, median=40.128, min=21.689, max=77.255

## Example developmental trajectories

![trajectories](nca_sanity_trajectories.png)

## Decision gate

- PASS: validity rate and degenerate-body fraction are both within the decision-gate thresholds. See docs/decisions.md for the recorded numbers.
