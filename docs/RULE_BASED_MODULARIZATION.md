# Rule-Based Encoder: Reference-Checkout Freeze/Removal + Structural Parity

Date: 2026-09-02

Summary of two changes, executed together: (1) froze the rule-based port's
equivalence proof into a permanent regression fixture, then removed
`external/rule-based-reference/` entirely; (2) reorganized
`encoders/rule_based/` to structural parity with `encoders/nca/`. Neither
change touched `external/evogym/` (a separate, already-scoped migration),
and neither touched decode/mutate/crossover behavior.

## Pre-flight

- Checked for a running primary benchmark (`tmux ls`, `ps aux | grep
  benchmark`) before starting: none found. Proceeded.

## Part 1 — Froze the port-equivalence proof

- The existing test, `tests/encoders/test_rule_based.py::test_ported_interpreter_matches_original_across_20_seeded_genomes`,
  compared the ported interpreter live against
  `external/rule-based-reference` on 20 seeded genomes (plus a companion
  5-seed mutation test).
- Generated and committed `tests/encoders/fixtures/rule_based_port_equivalence.json`:
  the 20 sampled genomes (serialized via `RuleBasedEncoder.serialize_genome`)
  and their resulting bodies, plus the 5 parent/mutated-child genome pairs
  and their bodies, all produced by running **both** the original and ported
  interpreters side by side (in the same process, immediately before
  deletion) and asserting identical output (modulo the documented SOFT/HARD
  fix). The fixture's header records the source commit
  (`703e711902cbb347d2b3385332be2913fee9657f`, 2026-09-02T07:43:59+05:30),
  the freeze date, and an explicit note that it is permanent and
  unregenerable once the reference checkout is gone.
- Rewrote the test file to replay the ported interpreter against this frozen
  fixture (`test_ported_interpreter_matches_frozen_port_equivalence_fixture`,
  `test_ported_mutation_matches_frozen_port_equivalence_fixture`) instead of
  importing the live checkout.
- Ran the rewritten test **while the original checkout was still present**:
  9/9 passed in `tests/encoders/test_rule_based.py`. This was the last
  chance to catch a mismatch with the original still on disk to compare
  against — it passed, so Part 2 proceeded.

## Part 2 — Removed `external/rule-based-reference/`

- Deleted the directory entirely, including its nested `.git`.
- Removed its `.gitignore` entry (and the explanatory note that existed only
  to justify that entry); `external/evogym/` is the sole remaining nested-repo
  ignore rule.
- Updated `README.md` and `docs/READY_TO_RUN.md` (this project has no
  `docs/dependency-architecture.md` or `REPRODUCE.md` — those were updated in
  their place) to state: EvoGym at `external/evogym/` is the project's only
  external dependency; the rule-based reference implementation was ported in
  full during initial development, its equivalence to the original is
  permanently proven by the frozen fixture, and the original checkout is no
  longer needed or present.
- Recorded the removal decision in `docs/decisions.md`, including the
  fixture's provenance (commit hash, date) and the explicit, accepted
  tradeoff: future re-verification against the original source is no longer
  possible, only against this frozen snapshot. (The original checkout was
  itself `.gitignore`d and never tracked by this project's own git history,
  so the fixture and the ported modules' header comments are now the only
  surviving record of the original's identity.)

## Part 3 — Structural parity with `encoders/nca/`

`encoders/nca/` is one file (`encoder.py`) holding both the adapter and the
network logic. `encoders/rule_based/` legitimately needs more code (a
ported rule language: types, genome operators, interpreter) that doesn't
belong inlined into the adapter — so, matching the principle rather than
the file count, the port's three modules were moved into a clearly-named
internal submodule instead of being loose top-level files with legacy
names:

```
encoders/rule_based/
  __init__.py          exports RuleBasedEncoder (unchanged)
  encoder.py            Encoder-contract adapter (unchanged behavior)
  _rules/                internal implementation, not part of the public API
    __init__.py
    types.py            was rule_types.py
    genome.py            was rule_encoding.py
    interpreter.py       was interpreter.py (unchanged name, now nested)
```

- All internal imports (`encoder.py`, the three `_rules/` modules, and
  `tests/encoders/test_rule_based.py`) were updated to the new paths via
  `git mv` + import-path edits. No other file in the tree imported the old
  paths (`tests/visualization/test_development.py` and
  `tests/evolution/test_ea_crossover.py` only ever imported
  `evodevo.encoders.rule_based.encoder`, which is unaffected).
- `RuleBasedEncoder.describe()` already exposed every hyperparameter
  (`developmental_iterations`, `mutation_rate`, `min_rules`, `max_rules`,
  `condition_types`, `action_types`) at the same completeness level as
  `NCAEncoder.describe()` — no change needed there.
- Updated docstring/header comments in `encoder.py` and the three `_rules/`
  modules to keep the credit-and-provenance header (original author, source
  commit) while pointing at the frozen fixture instead of the now-removed
  checkout path, and to avoid the literal string `external/rule-based-reference`
  anywhere under `evodevo/` (see Part 4 grep below).
- No decode/mutate/crossover logic was changed — this was a pure file-move
  plus import-path and comment update. Re-ran the frozen-fixture
  equivalence tests after the move: still 9/9 passing, confirming behavior
  is unaffected.

## Part 4 — Full re-verification

| Check | Result |
|---|---|
| Full test suite (`pytest tests/ -q`) | **278 passed, 0 failed** — matches the pre-change baseline exactly (two live-reference tests were replaced by two frozen-fixture tests of equal count, net zero change) |
| Encoder contract suite (`tests/encoders/test_contract.py`) | 28 passed — direct, nca, rule_based identical |
| `bash scripts/rule_based/test.sh` | 145 passed, 1 deselected |
| `grep -rn "rule-based-reference" evodevo/` | **0 matches** |
| Locked primary config hash | **Unchanged** — `git status` confirms zero modifications to any file under `evodevo/experiments/configs/`, `evodevo/experiments/benchmark.py`, or `evodevo/core/config.py`; the hash's inputs (the three resolved `ExperimentConfig` objects) are untouched by this change |
| `python -c "import evodevo"` | Succeeds |
| CLI dry-run (`python -m evodevo.experiments.benchmark --config evodevo/experiments/configs/benchmark.yaml --dry-run`) | Succeeds, produces the expected 15-run manifest |

## Status

**READY TO RUN** under the existing lock (`docs/READY_TO_RUN.md`,
config hash `7e93f59a82ba780e2ea4229a3d60681d7e98c08cfc33e3479a63af56d88c24ce`).
This change touched no config value, no decode/mutate/crossover behavior,
and no file the lock's hash is computed from — it is a test-fixture,
dependency-removal, and file-layout change only. `docs/READY_TO_RUN.md` has
been annotated with a post-lock note pointing here.
