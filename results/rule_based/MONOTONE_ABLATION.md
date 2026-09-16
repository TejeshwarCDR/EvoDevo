# rule_based monotone ablation: termination-guaranteed development (Proposition R4)

Fitness-blind (only decode()/develop_rules()/random_genome()/validate_body() are used -- NO PPO, NO task, NO reward, NO EA). The rule-based counterpart of NCA's `spectral_normalise` certified ablation.

**Rank chosen**: EMPTY (0) < SOFT (1) < RIGID (2) < H_ACT (3) < V_ACT (4) -- justified as increasing "commitment/structural complexity": EMPTY is the least-committed state and the only state GROW_DIRECTIONAL may ever target (so it must be the rank-0 floor); SOFT is deformable/least structured; RIGID is a fixed, undeformable commitment stronger than SOFT; H_ACT/V_ACT (actuators) are the most specialised voxel roles, ranked highest. Any other fixed total order with EMPTY lowest would equally satisfy the proposition below -- this is the specific order the implementation uses.

## Proposition R4, proved and verified

**Proof sketch**: under `action_monotonicity="monotone"`, a SELF-affecting action (SET_*, COPY_NEIGHBOR) is applied only if it does not decrease the acting cell's RANK; GROW_DIRECTIONAL needs no such gate at all, since it only ever targets an EMPTY (RANK 0) cell with a non-EMPTY (RANK >= 1) write -- unconditionally rank-increasing by construction. So EVERY write this interpretation permits either increases some cell's rank or does not happen. `Phi(s) = sum of RANK over all 25 cells` is therefore non-decreasing every step, and bounded above by 25*4=100 (every cell at the maximum rank). A non-decreasing, upper-bounded integer sequence must stop increasing within at most 100 steps; once Phi stops increasing, no write happened at all this step (every possible write would have strictly increased some cell's rank), so the GRID itself is a fixed point -- no genome can exhibit a period>1 cycle under monotone, and termination is guaranteed within at most 25*4=100 steps.

### Phi non-decreasing check, N=2000, T up to 100 steps per genome

**0/5710 Phi-decrease violations** across every step of every genome's development -- ZERO, exactly as the proposition requires.

**Max Phi observed: 100** (bound: 100).

### Termination step distribution

mean=1.8550, median=2.0000, max=7 (bound: 100).

**Max observed termination step (7) <= the derived bound of 100** -- confirmed within bound.

### Cycle check (period > 1 forbidden under monotone)

Reusing evodevo.analysis.rule_based_cycles.py's exact cycle detection (Proposition R1's apparatus) under monotone: **0/2000 genomes with period > 1** -- ZERO, exactly as R4 guarantees. (0 genomes hit the cycle-detector's own search cap, unrelated to this check.)

**Coincidence note**: the derived bound of 100 steps coincides numerically with the empirically selected shared T_primary=100 (results/T_PRIMARY_UNIFIED.md). This is remarked on here as a numerical coincidence, NOT presented as evidence of anything -- T_primary=100 was selected purely from NCA's own T-shift-stability sweep (results/nca/READOUT_POLICY_SELECTION.md), entirely independent of this ablation's 25*4=100 rank-sum bound, which follows from the grid size (25 cells) and material count (5, hence max rank 4) alone.

## Structural cost: monotone vs unrestricted

N=2000, T_primary=100.

| metric | unrestricted | monotone | delta |
| --- | --- | --- | --- |
| valid_rate | 0.5470 | 0.6530 | 0.1060 |
| degenerate_fraction | 0.2380 | 0.1165 | -0.1215 |
| unique_valid_body_fraction | 0.3315 | 0.3865 | 0.0550 |
| material_entropy_bits | 2.3043 | 2.3155 | 0.0112 |
| uniform_body_fraction | 0.2380 | 0.1165 | -0.1215 |
| never_settled_fraction | 0.1080 | 0.0000 | -0.1080 |
| mean_settle_step | 2.1754 | 1.8550 | -0.3204 |
| mean_hamming_displacement | 5.8280 | 5.1735 | -0.6545 |
| fraction_zero_displacement | 0.5230 | 0.5480 | 0.0250 |

**Exact settling under monotone, by construction**: never_settled_fraction = 0.0000 -- 0 (100% settled within T_primary), exactly as guaranteed.

**Cost, reported plainly**: monotone changes valid_rate by 0.1060 relative to unrestricted -- monotone does NOT cost validity in this sample -- the termination guarantee comes essentially for free on this metric, though other metrics above should also be read for the full picture (e.g. diversity, displacement).

## Status: ABLATION, not primary -- but the data does NOT match the a priori expectation

`action_monotonicity="monotone"` REMAINS an ABLATION in this report (RuleBasedEncoder's own class default stays "unrestricted") -- but the measured cost table above does NOT match the naive expectation that a termination guarantee trades away expressivity. On FOUR of the five core structural metrics, monotone is BETTER than unrestricted, not worse: valid_rate +0.1060, degenerate_fraction -0.1215, unique_valid_body_fraction +0.0550, uniform_body_fraction -0.1215 -- PLUS the termination/no-cycling guarantee itself, entirely free of charge on these metrics. The one place a real cost shows up is mutation sensitivity: mean_hamming_displacement drops by -0.6545 and fraction_zero_displacement rises by 0.0250 -- monotone genomes are LESS sensitive to mutation (more neutral mutations), a genuine, disclosed downside for evolvability/search even though it costs nothing on the structural metrics measured directly.

**Re-decision, as explicitly instructed when the cost measurement shows otherwise**: this result warrants SERIOUS CONSIDERATION of monotone as more than a pure ablation -- it is not asserted here that it should replace "unrestricted" as the primary (that is a larger, cascading decision -- T_primary, the windowed readout, and N6's mutation calibration were all resolved under "unrestricted" dynamics, and switching the primary would require re-running that whole chain, explicitly out of this task's scope). What IS decided here: monotone is flagged as a genuinely STRONG CANDIDATE, not dismissed as "the ablation, as expected" -- any future task that revisits rule_based's primary configuration should treat this cost table as evidence FOR monotone on structural grounds, with the one real, disclosed trade-off being reduced mutation sensitivity.

