# NCA Developmental Engine Sanity Report (E0)

N = 500 random genomes per encoder, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | NCA | Direct (reference) |
|---|---|---|
| Validity rate | 0.6940 | 0.0140 |
| Unique body hashes | 492 (0.9840) | 500 (1.0000) |
| All-empty fraction | 0.0020 | 0.0000 |
| Uniform non-empty fraction | 0.0240 | 0.0000 |
| Degenerate fraction (all-empty + uniform) | 0.0260 | 0.0000 |

### Invalid-genome failure reasons

| Reason category | NCA count | Direct count |
|---|---|---|
| disconnected | 128 | 492 |
| no_actuator | 29 | 4 |
| too_few_voxels | 7 | 0 |

## Developmental stability (NCA only)

Never-settled fraction (still changing on the final transition, t=9 -> t=10): 0.7040

Settle-step distribution (step index after which the argmax body never changes again):

| Step | Count |
|---|---|
| 0 | 0 |
| 1 | 3 |
| 2 | 4 |
| 3 | 4 |
| 4 | 2 |
| 5 | 2 |
| 6 | 5 |
| 7 | 14 |
| 8 | 31 |
| 9 | 83 |
| 10 | 352 |

## Residual state magnitude at t=T (NCA only)

max |state| across cells/channels at t=T, across genomes: mean=37.651, median=36.846, min=18.142, max=72.419

## Example developmental trajectories

![trajectories](nca_sanity_trajectories.png)

## Decision gate

- PASS: validity rate and degenerate-body fraction are both within the decision-gate thresholds. See docs/decisions.md for the recorded numbers.
