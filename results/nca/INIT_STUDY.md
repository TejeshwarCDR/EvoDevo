# NCA genome init study (layer-wise variance scaling)

Fitness-blind (evodevo/experiments/inspect_encoder.py's `_validity_stats`, and a direct pre-activation-variance measurement below). N=2000 genomes, init_condition=`full_nonsymmetric`, hidden_channels=3, hidden_width=16 (genome_length=792: layer 1 fan-in 5C=40, layer 2 fan-in h=16), update_rule=bounded_leaky. No PPO, no task, no reward were run to produce any number on this page.

**OLD init**: genome ~ N(0, init_scale=0.5) is the flat weight vector directly (NCAEncoder's behaviour before this change -- a single scale applied identically to layer 1's fan-in-40 weights and layer 2's fan-in-16 weights).
**NEW init**: genome z ~ N(0, 1) (independent of scale); `_unpack` applies layer-wise scaling W1 = z * sqrt(init_gain / fan_in1), W2 = z * sqrt(init_gain / fan_in2), with init_gain=1.0 -- the layer-wise-correct He/Xavier-style fan-in scaling.

## Pre-activation variance (standardised unit-variance layer input, standard fan-in analysis)

| init | layer | fan_in | pre-activation variance | target |
| --- | --- | --- | --- | --- |
| old (single-scale, init_scale=0.5) | 1 | 40 | 10.1454 | 1.0 |
| old (single-scale, init_scale=0.5) | 2 | 16 | 4.2616 | 1.0 |
| new (layer-wise, init_gain=1.0) | 1 | 40 | 0.9993 | 1.0 |
| new (layer-wise, init_gain=1.0) | 2 | 16 | 1.0144 | 1.0 |

**Old init**: every weight is drawn at the same fixed scale (std=init_scale=0.5, i.e. per-weight variance 0.25) regardless of which layer it is in. Pre-activation variance is fan_in * per-weight-variance, so it scales directly with fan_in: layer 1 (fan_in=40) lands at 10.1454 (~= 40 * 0.25), layer 2 (fan_in=16) lands at 4.2616 (~= 16 * 0.25). Neither is near the 1.0 target, and the two layers end up at different effective gains (2.38x apart) purely because they have different fan-in (40 vs 16) at the same weight scale.
**New init**: layer 1 variance 0.9993, layer 2 variance 1.0144 -- both close to the 1.0 target and close to each other, because the per-layer sqrt(init_gain/fan_in) scaling cancels out the fan_in difference between the two layers.

## Body degeneracy and diversity, at bounded_leaky

| init | valid_rate | degenerate_fraction (all-empty + uniform) | unique_body_fraction |
| --- | --- | --- | --- |
| old (single-scale) | 0.6955 | 0.0150 | 0.9875 |
| new (layer-wise) | 0.7005 | 0.0270 | 0.9740 |

Old init: 1975/2000 unique bodies, 0.0050 all-empty, 0.0100 uniform-nonempty.
New init: 1948/2000 unique bodies, 0.0060 all-empty, 0.0210 uniform-nonempty.

## Notes

This is a measurement of the init/mutation-scale change in isolation (N4/N5's bounded_leaky update rule and the boundary/spectral-normalise machinery are unchanged and held fixed here, at their defaults, for both rows -- only how the genome maps to layer 1 and layer 2 weights differs between rows). Whether the layer-wise-correct pre-activation variance translates into a mutation-sensitivity improvement is N6's question (re-tuning mutation_sigma), not measured here.

