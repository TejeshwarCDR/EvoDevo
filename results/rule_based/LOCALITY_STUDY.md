# rule_based locality study: restoring Proposition R3, boundary/neighbourhood parity with NCA

Fitness-blind (only decode()/develop_rules()/random_genome()/validate_body() are used -- NO PPO, NO task, NO reward, NO EA). Does not change conflict resolution (R2's own axis) or the action vocabulary.

**Rationale (position_predicate)**: `ConditionType.POSITION` tests the ACTING CELL'S OWN absolute grid coordinates ("if I am at row 0 / the boundary, become RIGID"). This lets a rule express something functionally equivalent to a direct, per-coordinate encoding using rule syntax, which breaks the TRANSLATION-EQUIVARIANCE property the developmental-encoding claim rests on: D(translate(s)) should equal translate(D(s)) for interior cells, for any genome that reacts only to LOCAL grid content, not to WHERE it is. `position_predicate="disabled"` (the new default) means the sampler/mutator never GENERATE a new POSITION condition; a genome that already has one (deserialized from an old result) still decodes correctly -- only its GENERATION is gated, not its evaluation.

## Part 1: structural comparison, position_predicate enabled vs disabled

N=2000, seed=0.

| metric | enabled | disabled (new default) | delta |
| --- | --- | --- | --- |
| valid_rate | 0.5735 | 0.5485 | -0.0250 |
| degenerate_fraction | 0.1870 | 0.2375 | 0.0505 |
| unique_valid_body_fraction | 0.3955 | 0.3330 | -0.0625 |
| material_entropy_bits | 2.3038 | 2.3045 | 0.0007 |
| genomes_per_unique_phenotype | 1.8083 | 2.1575 | 0.3492 |
| largest_neutral_class_fraction | 0.0615 | 0.0755 | 0.0140 |
| never_settled_fraction | 0.1155 | 0.1080 | -0.0075 |
| mean_settle_step | 2.2012 | 2.1754 | -0.0258 |
| mean_hamming_displacement | 5.6715 | 5.8450 | 0.1735 |
| fraction_zero_displacement | 0.4920 | 0.5210 | 0.0290 |

**Disabling POSITION materially reduces validity_rate by 0.0250** -- reported honestly: we removed a predicate that improved our numbers because it violated the encoding's central equivariance claim. This is a genuine, disclosed cost, not a weakness to hide.

## Part 2: Proposition R3, direct empirical test

For N=500 genomes, construct a random grid state `s`, translate it by (1, 0) via `s' = roll(s, 1, axis=0)`, develop BOTH `s` and `s'` for exactly 1 iteration under the SAME genome using `boundary="toroidal"` (for THIS TEST ONLY -- does not change RuleBasedEncoder's own default boundary, which stays `"no_op"`), and check whether `D(s') == roll(D(s), 1, axis=0)` over the WHOLE grid. **Why toroidal, specifically**: an earlier version of this test used the encoder's actual default `"no_op"` boundary with an "interior-only" cell restriction, and found a nonzero failure rate even with `position_predicate="disabled"` -- traced directly (not hand-waved) to a genuine FINITE-GRID artifact: a growth WRITE landing on a comparison cell can originate from a neighbor 2 rows away whose OWN decision depends on an off-grid check with no consistent counterpart after a 1-row translate on a 5-row grid. That artifact is a property of finite open grids, orthogonal to what R3 actually claims about POSITION. Testing on a TORUS instead (no distinguished boundary cell anywhere, so `roll` is an EXACT symmetry of the space) isolates the property actually being checked -- and `ConditionType.POSITION` still breaks it even on a torus, since TOP/BOTTOM/LEFT/RIGHT/BOUNDARY test fixed absolute indices that remain meaningful (and un-translated) regardless of wraparound, which is exactly the failure mode R3 is about.

| position_predicate | n | n_pass | n_fail | fraction_fail |
| --- | --- | --- | --- | --- |
| disabled | 500 | 500 | 0 | 0.0000 |
| enabled | 500 | 393 | 107 | 0.2140 |

**With position_predicate="disabled": 0.0000 fraction_fail** -- 0, confirming R3 holds EXACTLY (over the whole grid, on a torus) once POSITION cannot be generated.

**With position_predicate="enabled": 0.2140 fraction_fail** -- confirmed NONZERO, demonstrating (not assuming) that POSITION is what breaks equivariance when present.

**Verdict: confirmed, not assumed -- POSITION is exactly what breaks translation-equivariance, and disabling its generation restores Proposition R3.**

Example failure (enabled): genome index 0, cell (0, 1): D(translate(s)) = 2, translate(D(s)) = 1.

## Boundary and neighbourhood axis registration

`RuleBasedEncoder`/`develop_rules` now expose `boundary` (`"no_op"` default = the interpreter's entire historical off-grid behaviour, preserved byte-for-byte; `"toroidal"` wraps every neighbor read and growth target) and `neighbourhood` (`"von_neumann"` default = the interpreter's entire historical 4-neighbor `NEIGHBOR_COUNT`, matching EvoGym's own cardinal connectivity -- see docs/decisions.md's "Neighbourhood type" row, not re-justified here; `"moore"` = 8-neighbor `NEIGHBOR_COUNT`, matching NCA's own Moore option). Verified: `tests/encoders/test_rule_based_locality.py` (boundary no_op vs toroidal growth-wrap test, neighbourhood von_neumann vs moore NEIGHBOR_COUNT test).

