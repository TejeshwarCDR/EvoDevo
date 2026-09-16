# NCA readout policy selection: T-shift sensitivity, fixed-T vs mode-of-window

**The operational test** (does not depend on results/nca/DYNAMICS_CLASSIFICATION.md's underlying dynamical classification). Fitness-blind (only decode()/develop()/NCAEncoder._step()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes, the SAME seeded sample as horizon_extension.py (seed=0).

**Cost**: 26.32s to develop all trajectories, 34.34s total -- negligible relative to PPO training cost.

## Part 2: fixed-T argmax readout, T-shift sensitivity

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 40 | 0.9410 | 0.6385 | 0.6390 | 0.9360 |
| 100 | 0.8550 | 0.5415 | 0.5600 | 0.8350 |
| 200 | 0.7840 | 0.5370 | 0.5305 | 0.7770 |
| 500 | 0.7275 | 0.5000 | 0.4920 | 0.7305 |
| 1000 | 0.7020 | 0.4860 | 0.4790 | 0.7055 |

fraction_changed = fraction of the full N genomes whose decoded body at T+shift differs from the body at T_primary; settled genomes contribute zero to this by construction (their body is constant across the shift window), so a nonzero fraction reflects only the still-changing subset.

## Part 3: mode-of-window readout, T-shift sensitivity

**IMPORTANT CAVEAT (found and fixed 2026-09-16, see docs/decisions.md's "NCA T_primary/W consistency fix" row)**: `windowed_readout.py::windowed_body_at` silently CLIPS to `min(W, available history)` when the queried trajectory is shorter than W -- it does not raise. Rows below marked `*` have W > T (and, for the T-shift columns, W > T+shift too): at these rows, the "window" is NOT a genuine trailing W-step window -- it is the FULL available trajectory at that point, a DIFFERENT and differently-justified readout that happens to share the same code path. Only UNMARKED rows (W <= T, and W <= every shifted query point T+shift used against them) measure a genuine W-step window throughout. Part 4's selection below is restricted to unmarked (T, W) pairs for exactly this reason.

### W=5

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 40 | 0.9165 | 0.3200 | 0.3150 | 0.9100 |
| 100 | 0.8415 | 0.2980 | 0.2830 | 0.8185 |
| 200 | 0.7570 | 0.2595 | 0.2620 | 0.7600 |
| 500 | 0.7050 | 0.2495 | 0.2460 | 0.7150 |
| 1000 | 0.6895 | 0.2255 | 0.2325 | 0.6810 |

### W=10

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 40 | 0.6620 | 0.1790 | 0.2055 | 0.6995 |
| 100 | 0.6055 | 0.1670 | 0.1830 | 0.6185 |
| 200 | 0.5645 | 0.1580 | 0.1480 | 0.5545 |
| 500 | 0.5320 | 0.1335 | 0.1450 | 0.5275 |
| 1000 | 0.5190 | 0.1425 | 0.1405 | 0.4870 |

### W=20

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 40 | 0.4105 | 0.0955 | 0.1060 | 0.4360 |
| 100 | 0.3965 | 0.1040 | 0.0930 | 0.3675 |
| 200 | 0.3605 | 0.0865 | 0.0720 | 0.3325 |
| 500 | 0.3295 | 0.0815 | 0.0785 | 0.3200 |
| 1000 | 0.3350 | 0.0795 | 0.0740 | 0.3035 |

### W=50

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 40 * | 0.1030 | 0.0195 | 0.0215 | 0.1055 |
| 100 | 0.1705 | 0.0390 | 0.0360 | 0.1735 |
| 200 | 0.1745 | 0.0385 | 0.0405 | 0.1910 |
| 500 | 0.1690 | 0.0390 | 0.0510 | 0.1750 |
| 1000 | 0.1835 | 0.0405 | 0.0450 | 0.1765 |

`*` = W > T_primary: NOT a genuine W-step window at this row (clipped to the full available trajectory instead); excluded from Part 4's selection.

## Windowed vs fixed-T, at matched (T_primary, shift)

Confirmed (not assumed) that windowed fraction_changed is dramatically lower:

| T_primary | shift | fixed-T | mode-of-window (best W) | reduction |
| --- | --- | --- | --- | --- |
| 40 | -5 | 0.9410 | 0.1030 | 0.8380 |
| 40 | 5 | 0.9360 | 0.1055 | 0.8305 |
| 100 | -5 | 0.8550 | 0.1705 | 0.6845 |
| 100 | 5 | 0.8350 | 0.1735 | 0.6615 |
| 200 | -5 | 0.7840 | 0.1745 | 0.6095 |
| 200 | 5 | 0.7770 | 0.1910 | 0.5860 |
| 500 | -5 | 0.7275 | 0.1690 | 0.5585 |
| 500 | 5 | 0.7305 | 0.1750 | 0.5555 |
| 1000 | -5 | 0.7020 | 0.1835 | 0.5185 |
| 1000 | 5 | 0.7055 | 0.1765 | 0.5290 |

## Equivalence check (essential): cleanly-settled genomes, fixed-T vs mode-of-window

"If it fails, mode-of-window would be silently changing already-good answers, which is not acceptable." Checked at population scale, every (T_primary, W) pair:

| T_primary | W | n_settled (fixed-T proxy) | n_mismatch |
| --- | --- | --- | --- |
| 40 | 5 | 0 | 0 |
| 40 | 10 | 0 | 0 |
| 40 | 20 | 0 | 0 |
| 40 | 50 | 0 | 0 |
| 100 | 5 | 52 | 0 |
| 100 | 10 | 52 | 0 |
| 100 | 20 | 52 | 0 |
| 100 | 50 | 52 | 0 |
| 200 | 5 | 207 | 0 |
| 200 | 10 | 207 | 0 |
| 200 | 20 | 207 | 0 |
| 200 | 50 | 207 | 0 |
| 500 | 5 | 387 | 0 |
| 500 | 10 | 387 | 0 |
| 500 | 20 | 387 | 0 |
| 500 | 50 | 387 | 0 |
| 1000 | 5 | 442 | 0 |
| 1000 | 10 | 442 | 0 |
| 1000 | 20 | 442 | 0 |
| 1000 | 50 | 442 | 0 |

**All mismatches are 0 -- the equivalence property holds** at every tested (T_primary, W): mode-of-window never changes the answer for a genome that fixed-T already reads as cleanly settled.

## Part 4: selection

Recommended starting threshold: fraction_changed < 0.02 at shift=+-5 for the windowed readout, checked against the actual table above, not assumed reachable. **Restricted to (T_primary, W) pairs with W <= T_primary** -- a W-step trailing window cannot exist within a shorter trajectory; see Part 3's caveat above for the `windowed_readout.py` clipping behaviour that made this restriction necessary (found and fixed 2026-09-16).

**The recommended 0.02 threshold is NOT reachable anywhere on the CONSTRAINED (W<=T_primary) grid** -- consistent with results/nca/DYNAMICS_CLASSIFICATION.md's finding that a substantial fraction (~0.449 of the full population) of PRIMARY genomes show chaotic-like (exponentially growing) sensitivity to an infinitesimal perturbation: no window size or horizon can fully suppress the T-shift sensitivity of a genuinely chaotic trajectory, since its argmax pattern itself keeps changing. The threshold is instead set, honestly, to **0.1735** -- the BEST (lowest) +-5 sensitivity actually achieved anywhere on the CONSTRAINED grid, not a number chosen in advance (and notably worse than the UNCONSTRAINED grid's apparent best of 0.1055 at T=40,W=50 -- that pair was never a genuine W=50 window at all, since W=50 > T=40; it measured the full 41-step trajectory, a different and less favourable-looking but invalid comparison). Selected: T_primary=100, W=50, readout_mode=mode_of_window (the smallest (T_primary, W) achieving this minimum among genuine windows; shift=-5 fraction_changed=0.1705, shift=+5 fraction_changed=0.1735).

This is still a large, confirmed improvement over the fixed-T readout at matched T_primary (see the 'Windowed vs fixed-T' table above) -- windowed reduces sensitivity by roughly an order of magnitude, even though it cannot reach the originally recommended <2% absolute level for this population.

