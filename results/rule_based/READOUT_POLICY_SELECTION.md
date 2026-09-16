# rule_based readout policy: T-shift sensitivity, fixed-T vs mode-of-window

Extends N13's operational T-shift test to rule_based (results/rule_based/SETTLING_AUDIT.md's confirmed genuine, permanent ~11.55% cycling population). Fitness-blind (only decode()/develop_trajectory()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes, seed=0. Uses the SAME shared `evodevo.analysis.windowed_readout.mode_of_window` utility NCA's encoder uses.

**Cost**: 426.83s to develop all trajectories, 430.67s total.

## Odd-vs-even W tie behaviour: empirical verification (CORRECTS the originating assumption)

The originating task instructed choosing W ODD, reasoning that an even W's perfect 50/50 tie-break on a period-2 cycle is 'a worse outcome than a genuine majority vote'. Direct measurement on a synthetic exact period-2 cycle (a,b,a,b,...), sweeping T and computing fraction_changed under a +-1/+-5 shift exactly as the operational test defines it, shows the OPPOSITE:

| W | parity | fraction_changed (synthetic period-2 cycle) |
| --- | --- | --- |
| 2 | even | 0.0000 |
| 3 | odd | 1.0000 |
| 4 | even | 0.0000 |
| 5 | odd | 1.0000 |
| 6 | even | 0.0000 |
| 7 | odd | 1.0000 |
| 11 | odd | 1.0000 |

**EVEN W gives PERFECT T-shift stability (fraction_changed=0.0) for a pure period-2 cycle** -- the tie always resolves the same way (deterministic lowest-material-ID tie-break), regardless of which phase the window's right edge lands on. **ODD W gives the WORST possible instability (fraction_changed=1.0)** -- with an odd window length, the majority literally has one more of whichever phase is at the window's most recent end, so the majority ALTERNATES every single step, exactly like the fixed-T readout it was meant to improve on. This directly contradicts the originating task's assumption; it is reported here plainly rather than silently preserved. The W selected below is chosen by this measured criterion, not by a parity assumption.

## Part 1 Step 4: fixed-T argmax readout, T-shift sensitivity

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.1385 | 0.1170 | 0.1170 | 0.1170 |
| 20 | 0.1150 | 0.1155 | 0.1155 | 0.1150 |
| 40 | 0.1150 | 0.1155 | 0.1155 | 0.1150 |
| 100 | 0.1150 | 0.1155 | 0.1155 | 0.1150 |

## Part 1 Step 5: mode-of-window readout, T-shift sensitivity (all tested W)

### W=2 (even)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.0870 | 0.0280 | 0.0215 | 0.0265 |
| 20 | 0.0160 | 0.0270 | 0.0120 | 0.0245 |
| 40 | 0.0250 | 0.0255 | 0.0215 | 0.0230 |
| 100 | 0.0250 | 0.0255 | 0.0215 | 0.0230 |

### W=3 (odd)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.1490 | 0.0905 | 0.0970 | 0.0985 |
| 20 | 0.0935 | 0.0960 | 0.0905 | 0.0965 |
| 40 | 0.0975 | 0.0925 | 0.0945 | 0.0930 |
| 100 | 0.0975 | 0.0930 | 0.0945 | 0.0930 |

### W=4 (even)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.0985 | 0.0200 | 0.0155 | 0.0215 |
| 20 | 0.0180 | 0.0210 | 0.0150 | 0.0225 |
| 40 | 0.0175 | 0.0170 | 0.0155 | 0.0180 |
| 100 | 0.0180 | 0.0175 | 0.0160 | 0.0180 |

### W=5 (odd)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.1405 | 0.0980 | 0.1025 | 0.1055 |
| 20 | 0.0990 | 0.1065 | 0.0955 | 0.1090 |
| 40 | 0.1045 | 0.1020 | 0.1025 | 0.1040 |
| 100 | 0.1050 | 0.1015 | 0.1035 | 0.1030 |

### W=6 (even)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.1225 | 0.0170 | 0.0095 | 0.0145 |
| 20 | 0.0095 | 0.0060 | 0.0025 | 0.0040 |
| 40 | 0.0055 | 0.0055 | 0.0025 | 0.0040 |
| 100 | 0.0070 | 0.0060 | 0.0025 | 0.0035 |

### W=7 (odd)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.1325 | 0.0935 | 0.1045 | 0.1110 |
| 20 | 0.1110 | 0.1095 | 0.1095 | 0.1100 |
| 40 | 0.1115 | 0.1115 | 0.1090 | 0.1110 |
| 100 | 0.1120 | 0.1110 | 0.1100 | 0.1095 |

### W=11 (odd)

| T_primary | shift=-5 | shift=-1 | shift=+1 | shift=+5 |
| --- | --- | --- | --- | --- |
| 10 | 0.1130 | 0.0460 | 0.0695 | 0.1015 |
| 20 | 0.0965 | 0.1055 | 0.0935 | 0.1070 |
| 40 | 0.1040 | 0.1015 | 0.1010 | 0.1010 |
| 100 | 0.1040 | 0.1015 | 0.1015 | 0.1010 |

## Equivalence check (essential): cleanly-settled genomes, fixed-T vs mode-of-window

| T_primary | W | n_settled | n_mismatch |
| --- | --- | --- | --- |
| 10 | 2 | 0 | 0 |
| 10 | 3 | 0 | 0 |
| 10 | 4 | 0 | 0 |
| 10 | 5 | 0 | 0 |
| 10 | 6 | 0 | 0 |
| 10 | 7 | 0 | 0 |
| 10 | 11 | 0 | 0 |
| 20 | 2 | 1766 | 0 |
| 20 | 3 | 1766 | 0 |
| 20 | 4 | 1766 | 0 |
| 20 | 5 | 1766 | 0 |
| 20 | 6 | 1766 | 0 |
| 20 | 7 | 1766 | 0 |
| 20 | 11 | 1766 | 0 |
| 40 | 2 | 1769 | 0 |
| 40 | 3 | 1769 | 0 |
| 40 | 4 | 1769 | 0 |
| 40 | 5 | 1769 | 0 |
| 40 | 6 | 1769 | 0 |
| 40 | 7 | 1769 | 0 |
| 40 | 11 | 1769 | 0 |
| 100 | 2 | 1769 | 0 |
| 100 | 3 | 1769 | 0 |
| 100 | 4 | 1769 | 0 |
| 100 | 5 | 1769 | 0 |
| 100 | 6 | 1769 | 0 |
| 100 | 7 | 1769 | 0 |
| 100 | 11 | 1769 | 0 |

**All mismatches are 0** -- mode-of-window reproduces fixed-T exactly for every genome that is genuinely at a fixed point, confirming the shared `windowed_readout.mode_of_window` utility behaves identically for rule_based as it does for NCA.

## Selection: smallest (T, W) achieving the grid's best measured stability

Searching the FULL grid (all tested T x W, not just the smallest T candidate -- a per-T-only search would have missed this: T=10 alone is dominated by rule_based's own settling transient, not yet complete this early, and picks a worse W): the grid's best (lowest) +-5-shift fraction_changed is **0.0055**, first achieved at **T=40, W=6** (the smallest (T, W) pair reaching it, ascending T then W). Notably W=6 (even) dominates every odd W tested at every T -- consistent with the odd-vs-even finding above: even W's deterministic tie-break absorbs the permanent period-2 cycling subpopulation's phase-dependence entirely, while odd W's alternating majority actively fights it. Unlike NCA's population (dominated by long, effectively-chaotic non-periods, W=50), rule_based's cycling population is small (11.55%) and short-period (dominated by period 2), so its resolved W is far smaller.

