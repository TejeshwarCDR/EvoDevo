# rule_based exact cycle characterisation (Proposition R1)

Fitness-blind (only decode()/develop_rules()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). Replaces results/rule_based/SETTLING_AUDIT.md's SAMPLED (fixed-horizon, T_max=100) settling characterisation with the EXACT one Proposition R1 makes available: since the grid-state space is finite and `develop_rules` is deterministic for a fixed genome/interpreter settings, every genome's trajectory is eventually periodic with an EXACT transient length mu and period p -- computed here by exact cycle detection (visited-state hash set, first-repeat rule), not estimated.

## Exact (mu, p) for N=2000 genomes, CURRENT dynamics (R2 first_rule_index + R3 position-disabled)

**Search cap = 500 steps** (well above SETTLING_AUDIT.md's observed mu+p<=100 within its own T_max=100 window, so the cap is justified by data, not picked arbitrarily). **0/2000 genomes hit the cap** (none) -- zero genomes exceeded the cap, confirming it was generous enough for this population.

Among the 2000 resolved genomes: mu median=2.0000, mu max=21, mu mean=2.2905; p median=1.0000, p max=60; **max observed (mu+p) = 75** -- this is the actual empirical justification for the 500-step cap being more than sufficient.

**Fixed points (p=1, i.e. "settled" in SETTLING_AUDIT.md's sense): 1784/2000 (0.8920).** **Genuine cycles (p>1): 216/2000 (0.1080).**

### Period histogram (p>1 only)

| period | count | fraction of cycling genomes |
| --- | --- | --- |
| 2 | 161 | 0.7454 |
| 3 | 19 | 0.0880 |
| 4 | 15 | 0.0694 |
| 5 | 2 | 0.0093 |
| 6 | 11 | 0.0509 |
| 8 | 5 | 0.0231 |
| 10 | 1 | 0.0046 |
| 12 | 1 | 0.0046 |
| 60 | 1 | 0.0046 |

## Reconciliation with results/rule_based/SETTLING_AUDIT.md

SETTLING_AUDIT.md (sampled, T_max=100, under the ORIGINAL pre-R2/R3 dynamics) reported: never_settled_fraction=0.1155 at T_max=100, 99.57% of the never-settled population with a detected period <=20 (73.16% period-2), i.e. ~0.1150 of the full population confirmed cyclic.

This module's EXACT figure for the SAME pre-R2/R3 dynamics (row_major_last_wins, position enabled, same genome draw order/seed): **0.1155 of genomes are genuinely cyclic (p>1)**, vs SETTLING_AUDIT.md's sampled 0.1150 -- closely matching (within expected small differences from SETTLING_AUDIT.md's own T_max=100 window occasionally under- or over-counting near the boundary), confirming the sampled estimate was accurate.

## Effect of R2 (conflict policy) + R3 (position disabled) on the cycling population

| dynamics | fraction fixed-point (p=1) | fraction cycling (p>1) | dominant period |
| --- | --- | --- | --- |
| pre-R2/R3 baseline (row_major_last_wins, position enabled) | 0.8845 | 0.1155 | 2 |
| current (first_rule_index, position disabled) | 0.8920 | 0.1080 | 2 |

**The cycling population SHRANK by 0.0075** under R2+R3's combined effect, measured directly rather than assumed to be unaffected.

## Connection to the resolved windowed readout (W=6)

N15/N16 adopted `readout_window=6` (even, validity-aware tie-break) when the dominant observed period was 2. Under the CURRENT dynamics: dominant observed period = 2, and W=6 is STILL a multiple of it.

**25/216 cycling genomes (0.1157) have a period that does NOT divide W=6** -- these are the genomes the window does not cleanly absorb (their mode-of-window readout can still show residual phase-dependence, per N15's own even-W-divisibility argument).

This fraction is small -- **W=6 remains an appropriate choice** under the current dynamics; no revision warranted by this evidence.

## Paper-facing contrast: symbolic vs continuous developmental encodings

For rule_based, (mu, p) are EXACT and DECIDABLE for any genome, computed in a single pass with a hash set, no sampling, no ambiguity, no unresolved cases (beyond the disclosed cap, empirically never hit here). For NCA, the analogous question -- does the residual non-settling population represent genuine limit cycles or slow transient convergence? -- required FOUR separate diagnostic rounds (results/nca/HORIZON_EXTENSION.md's extended-horizon sweep to T=2000, plus results/nca/DYNAMICS_CLASSIFICATION.md's widened period search AND corrected perturbation-sensitivity probe) and remained EMPIRICALLY UNRESOLVED in the strong sense: the population was classified as "predominantly transient" / "chaotic-like" from indirect statistical evidence (never_settled_fraction trends, log-distance growth slopes), never PROVEN, because NCA's continuous state space is not finite and Proposition R1's pigeonhole argument does not apply to it at all. This is a genuine, structural difference between the two encodings' analyzability, not a difference in how much effort was spent on each -- rule_based's finite, discrete, deterministic state space makes exact characterisation possible in principle AND in practice; NCA's does not.

