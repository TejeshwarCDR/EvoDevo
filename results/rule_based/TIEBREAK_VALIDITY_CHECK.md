# rule_based tie-break validity check: does the even-W tie-break discard valid phases?

Fitness-blind (only decode()/develop_trajectory()/random_genome()/validate_body() are used -- NO PPO, NO task, NO reward, NO EA). N=200 confirmed period-2 cycling genomes (SETTLING_AUDIT.md's own detection method, T_max=100, seed=0), evaluated at the resolved (T_primary=100, W=6).

## Per-genome measurements

**1.0000 of sampled genomes' windowed readout equals the deterministic tie-break exactly** -- confirming these are genuinely 50/50-tied cases (every differing cell between A and B resolves to the tie-break value), not cases where the window happens to contain enough non-tied history to produce a real majority. For a pure period-2 cycle sampled deep into its trajectory with W=6 (even), this is expected to be at or near 1.0.

## Validity comparison: windowed (tie-break) vs fixed-at-A vs fixed-at-B

| readout | validity rate |
| --- | --- |
| mode_of_window (tie-break) | 0.3100 |
| fixed at phase A | 0.3250 |
| fixed at phase B | 0.6100 |
| EITHER phase A or B valid | 0.6250 |

## The concrete harm case

Fraction of the 200-genome sample where mode_of_window's tie-broken body is INVALID while AT LEAST ONE of the genome's two real alternating phases (A or B) WAS valid: **0.3150** (63/200).

Threshold: 0.0500 (5%), justified as the point past which the tie-break mechanism is discarding a legitimately-available valid body for a non-trivial share of the very population this readout policy exists to serve -- below this, the mechanism is a minor, acceptable simplification; at or above it, genome-specific information matters enough to warrant a more careful mechanism.

**Verdict: the original tie-break is NOT acceptable as a default.** 0.3150 > 0.0500 -- a non-trivial share of cycling genomes have a legitimately valid phase that the tie-break discards. Proceeding to test the genome-aware alternative.

## Genome-aware alternative (tested regardless, for completeness)

Candidate: for a genome in an exact 2-way tie, prefer whichever of its own two cycle phases (A or B) is independently valid (checked via `validate_body`); if both or neither are valid, fall back to the original tie-break.

| readout | validity rate |
| --- | --- |
| original tie-break (mode_of_window) | 0.3100 |
| genome-aware tie-break | 0.6250 |

**T-shift stability of the genome-aware alternative, verified (not assumed)**: 50/50 genomes' own (A, B) pair -- and hence the genome-aware readout, which depends only on it -- is UNCHANGED when re-derived at every T+shift query point (+-1, +-5) used by the operational T-shift test. Confirmed stable by construction, as expected for a mechanism that depends only on the genome's fixed cycle content, never on t.

## Recommendation

**Adopt the genome-aware tie-break.** It improves validity by 0.3150 (0.3100 -> 0.6250) while remaining T-shift stable by construction, directly addressing the concrete harm case measured above.

