# NCA Developmental Engine Sanity Report (E0)

N = 500 random genomes per encoder, seed = 0. No PPO was run for this experiment.

## Validity and diversity

| Metric | NCA | Direct (reference) |
|---|---|---|
| Validity rate | 0.6300 | 0.0140 |
| Unique body hashes | 455 (0.9100) | 500 (1.0000) |
| All-empty fraction | 0.0240 | 0.0000 |
| Uniform non-empty fraction | 0.0720 | 0.0000 |
| Degenerate fraction (all-empty + uniform) | 0.0960 | 0.0000 |

### Invalid-genome failure reasons

| Reason category | NCA count | Direct count |
|---|---|---|
| disconnected | 125 | 493 |
| no_actuator | 89 | 1 |
| too_few_voxels | 29 | 0 |

## Developmental stability (NCA only)

Never-settled fraction (still changing on the final transition, t=0 -> t=1): 1.0000

Settle-step distribution (step index after which the argmax body never changes again):

| Step | Count |
|---|---|
| 0 | 0 |
| 1 | 500 |

## Residual state magnitude at t=T (NCA only)

max |state| across cells/channels at t=T, across genomes: mean=2.650, median=2.593, min=1.418, max=5.193

## Example developmental trajectories

![trajectories](nca_sanity_trajectories.png)

## Decision gate

- PASS: validity rate and degenerate-body fraction are both within the decision-gate thresholds. See docs/decisions.md for the recorded numbers.
