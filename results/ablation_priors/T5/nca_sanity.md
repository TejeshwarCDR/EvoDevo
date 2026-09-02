# NCA Developmental Engine Sanity Report (E0)

N = 500 random genomes per encoder, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | NCA | Direct (reference) |
|---|---|---|
| Validity rate | 0.6400 | 0.0140 |
| Unique body hashes | 488 (0.9760) | 500 (1.0000) |
| All-empty fraction | 0.0080 | 0.0000 |
| Uniform non-empty fraction | 0.0260 | 0.0000 |
| Degenerate fraction (all-empty + uniform) | 0.0340 | 0.0000 |

### Invalid-genome failure reasons

| Reason category | NCA count | Direct count |
|---|---|---|
| disconnected | 148 | 493 |
| no_actuator | 50 | 1 |
| too_few_voxels | 7 | 0 |

## Developmental stability (NCA only)

Never-settled fraction (still changing on the final transition, t=4 -> t=5): 0.9180

Settle-step distribution (step index after which the argmax body never changes again):

| Step | Count |
|---|---|
| 0 | 0 |
| 1 | 3 |
| 2 | 4 |
| 3 | 8 |
| 4 | 26 |
| 5 | 459 |

## Residual state magnitude at t=T (NCA only)

max |state| across cells/channels at t=T, across genomes: mean=18.187, median=17.767, min=9.884, max=33.333

## Example developmental trajectories

![trajectories](nca_sanity_trajectories.png)

## Decision gate

- PASS: validity rate and degenerate-body fraction are both within the decision-gate thresholds. See docs/decisions.md for the recorded numbers.
