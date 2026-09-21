# Project Loose Ends Resolved

Date: 2026-09-19
Base commit before this consolidated workstream: `092fcd5` (main)
Commits produced by this workstream (oldest first): `ff5905b`, `bbe54c6`, `23ade98`, `a89cf1a`, `e869ecf`, `163a7f9`, `0a630ca`

This document consolidates five prior tasks completed in sequence in this
session, each of which found and fixed real issues rather than confirming a
clean bill of health by assumption. It reports what was checked, what was
found, what was fixed, and how each fix was verified.

## Part 1 — Lab Machine Setup Follow-Up (3 fixes)

Three specific follow-up items from the initial lab-machine-setup verification:

1. **`setuptools<81` pin** (commit `ff5905b`). EvoGym's `evogym/envs/base.py`
   still imports the deprecated `pkg_resources` module, which setuptools
   dropped as of 84.0.0. Added the pin to `requirements.txt` with an
   explanatory comment; re-ran `pip install -r requirements.txt` (only
   `setuptools` itself changed, already at 78.1.1 from an earlier manual
   fix — no other package moved); confirmed EvoGym still imports and
   creates/steps/closes an env. **Confirmed applied**: present in
   `requirements.txt`, verified against the live venv.
2. **Benchmark tool's self-inflicted dirty-tree gate** (commit `bbe54c6`).
   `run_benchmark()` and `run_experiment()` each independently called
   `check_git_state()`; `run_benchmark()`'s own bookkeeping writes
   (`factors.json`, the manifest) could dirty the tree between its own
   check and a later, genuinely-new cell's independent re-check inside the
   same invocation — observed live (`make benchmark SURROGATE=1` failed
   this way after resuming 15 already-complete cells). Fixed by computing
   `(commit_hash, dirty_files)` exactly once at the top of
   `run_benchmark()` and threading it through every `run_experiment()` call
   via a new `precomputed_git_state` parameter. **Confirmed applied and
   tested**: `tests/experiments/test_benchmark.py::
   test_resuming_completed_cells_does_not_spuriously_dirty_gate_a_new_cell`
   reproduces the exact failure (confirmed to fail on the pre-fix code) and
   passes on the fix; also re-verified live end-to-end
   (`make benchmark SURROGATE=1` completed the previously-blocked
   `rule_based_monotone` cells with zero `DirtyTreeError`).
3. **tmux**: not installed, and installation remains **blocked** — this
   session has no passwordless sudo, so `apt-get install` cannot complete
   non-interactively. Unresolved; requires the user to run
   `sudo apt-get install -y tmux` or grant passwordless sudo. Unrelated to
   code/documentation consistency, carried here for completeness only.

## Part 2 — Staleness Audit (`results/STALENESS_AUDIT.md`)

Full detail: **`results/STALENESS_AUDIT.md`** (commit `a89cf1a`).

Every one of the 69 `.md` files under `docs/` and `results/` was checked
against a reference table built from `evodevo/encoders/nca/resolved_config.py`,
`evodevo/encoders/rule_based/resolved_config.py`, `docs/decisions.md`, and
`evodevo/experiments/configs/benchmark.yaml` — not sampled, not assumed
current from memory.

**16 files** made a "current X is Y" claim that no longer matched the
current resolved architecture and were given a superseded banner (old vs.
new values, plus the `decisions.md` row that made the correction) — none
were deleted. The remaining 53 were confirmed current or correctly
self-scoped as historical (explicit before/after studies, archives,
ablation-grid data, process docs).

**`results/BENCHMARK_REPORT.md` and `docs/FINAL_SMOKE_TEST.md`**, named
explicitly for extra scrutiny, were both resolved, not silently skipped:
both were checked via `git log --follow` against `resolved_config.py`'s
introduction (`a5abe84`/`0af2992`, 2026-09-17) and confirmed to predate the
*entire* N1-N17/R1-R8 architecture-redesign workstream (`7675f0d`/`0a9a61b`,
both 2026-09-15/2026-09-02). Both carry the most prominent banners in the
audit, explicitly stating that the real, final benchmark/smoke-test result
under the current resolved architecture **does not exist yet** — neither
file is left in a state a reader could mistake for a completed current
result.

## Part 3 — `docs/decisions.md` Internal Consistency Sweep

Full detail: `results/STALENESS_AUDIT.md`'s "Part 6" section (commit `e869ecf`).

- **Row count**: 47 data rows (counted programmatically, using the exact
  same row-inclusion logic the enforcing checker itself uses).
- **OPEN rows**: **0**, confirmed by re-running the actual enforcing
  function (`evodevo.analysis.readiness_check.scan_decisions_open_rows()`,
  which matches only the Chosen column, not the word "OPEN" anywhere in a
  row) directly against the live file — not informal grep, which returns a
  misleading 5 (all prose discussing past OPEN status). Its two existing
  regression tests also pass.
- **Contradictions found and resolved**: **5**. An older row stating a
  "final" value on the same topic (hidden channel count, mutation tuning —
  two hops, NCA timescale calibration, rule_based monotone status) as a
  later row's different final value, with no cross-reference either
  direction. Each older row now carries an explicit
  `(SUPERSEDED -- see the "..." row below)` tag in its topic cell, matching
  the document's own pre-existing convention — no Chosen/Rationale content
  was edited or removed (append-only convention preserved). One additional
  case (an order-independence percentage mismatch) was found to already be
  a deliberate, disclosed choice by a later row and correctly left as-is.
- **Artifact citation check**: all 33 unique `results/*.md` paths cited
  across every row checked for existence and cross-referenced against Part
  2's staleness verdicts. 32 exist and are used correctly (no row treats a
  superseded report's stale number as live evidence). The one that doesn't
  exist (`results/RESULTS.md`, cited only in the document's own header) is
  a deliberate forward reference to the not-yet-produced final results
  write-up, not a broken citation.

## Part 4 — Silent-Clip and Duplicated-Literal Sweep

Full commit: `163a7f9`.

**Silent-clip bug class** (a fixed ceiling/bound silently substituting a
caller's value instead of raising or warning — recurred 4 times previously
in this project's history). Searched `evodevo/` exhaustively via pattern
grep (`min(max(`, bare `min()`/`max()` clamps, `clip`/`clamp`/`truncat`
keywords) plus manual triage of every hit.

| Site | Before | After | Test |
|---|---|---|---|
| `DirectEncoder.__init__` | `mutation_rate * factors.mutation_scale` silently clipped to 1.0 | Warns with exact values when the clip triggers | `test_mutation_rate_overflow_warns_loudly_instead_of_silently_clipping` + a no-warning counterpart, `tests/encoders/test_direct.py` |
| `RuleBasedEncoder.__init__` | same bug | same fix | same pair, `tests/encoders/test_rule_based.py` |
| `windowed_readout.py::windowed_body_at` | silently clipped both an out-of-range `t` and an under-length window (`w > t+1`, returning the full available prefix instead of a genuine w-step window) — the exact mechanism behind the already-fixed "NCA T_primary/W consistency" bug | Warns on both conditions; the window-case message is stable (no interpolated values) so Python's own warning dedup collapses a legitimate diagnostic sweep's many intentional hits into one notice per process | 3 new tests, `tests/analysis/test_windowed_readout.py` |
| `rule_based_tiebreak_check.py::verify_genome_aware_readout_is_t_shift_stable` | a T-shift query genuinely exceeds its computed trajectory under current settings (`T_MAX_AUDIT=100 < RESOLVED_T+max(shift)=105`); verified harmless for the exact-period-2 population tested, but undisclosed | Warns | New file, `tests/analysis/test_rule_based_tiebreak_check.py` (no test file existed for this module before) |

Verified (not assumed) several other candidate sites are already compliant
or safe-by-construction: `depth_study.py`'s two `T_MAX_HARD_CAP` clips
already `warnings.warn`; `interpreter.py`'s `HARD_MAX_ITERATIONS` bound and
`rule_based_readout_policy.py::_required_trajectory_length` already raise;
`readiness_check.py`'s and `mutation_calibration.py`'s clips are fully
disclosed in their generated report text; `horizon_extension.py`'s
"naive_t_primary" deliberately illustrates the bug for the report; several
local `_bodies_at`/t-clamp closures in one-off analysis scripts are bounded
by an upstream raising guard or self-consistent arithmetic and cannot
trigger under current code; `NCAEncoder`'s `np.clip` for "replicate"
boundary padding is a documented spatial boundary mode, not a silent
violation of a caller's request.

**Duplicated resolved-value literal risk** (the same class as the earlier
six-copies-of-lambda*/B* finding for NCA, now checked exhaustively for
rule_based). Found **7+ call sites across 4 analysis modules**
(`rule_based_monotone_report.py`, `rule_based_cycles.py`,
`rule_based_capacity_study.py`, `rule_based_context_report.py`) hardcoding
`position_predicate="disabled"`/`conflict_resolution="first_rule_index"` as
bare literals instead of importing `PRIMARY` from
`evodevo.encoders.rule_based.resolved_config` — plus one instance each in
`nca_readiness_check.py` (`update_rule="bounded_leaky"`, inconsistent with
the same file already importing `PRIMARY.decay`/`PRIMARY.state_bound`) and
`windowed_validity_check.py` (`NCA_W`/`RB_W` local constants, inconsistent
with the same file already importing `NCA_PRIMARY` for other fields). All
fixed to import from the relevant `resolved_config.py`. Added an AST-based
(not regex) regression test,
`test_no_module_hardcodes_a_resolved_dynamics_literal_instead_of_importing_primary`
in `tests/encoders/test_rule_based_resolved_config_single_source.py`, which
scans both call-site keyword arguments and function-signature defaults —
the latter check caught one additional site (`exact_cycle`'s own
parameter defaults) that the initial manual grep missed, and was added to
close that gap.

Every fix in this part was verified to fail against the pre-fix code (via
`git stash`) and pass against the fix — not merely written and assumed
correct.

## Part 5 — Documentation Accuracy (`README.md`, `docs/READY_TO_RUN.md`)

Full commit: `0a630ca`.

Rewrote both documents to describe the current project accurately: the
one-pipeline/four-arm architecture (`direct`, `nca`, `rule_based`,
`rule_based_monotone`), the current `fast_dev`/`primary_matrix` profile
mechanism and its guards, and the current status — **architecture design
work is complete and validated; the real experimental matrix has not been
run**. Every CLI flag referenced was verified against live `--help` output
(`--surrogate`, `--allow-dirty`, `--profile`, `--i-know-this-is-fast-dev`),
and the `make benchmark` Makefile target was checked against its actual
`Makefile` recipe (it does **not** pass `--i-know-this-is-fast-dev`, so
plain `make benchmark` would hit the guard immediately — corrected in the
README rather than left implying it "just works").

While cross-checking `docs/PRE_PARALLELIZATION_READINESS.md`'s dynamics
table against the real production config path for this write-up (not
trusting the table from memory), found that **the calibrated per-arm
mutation strength never reached the real `benchmark.yaml`-driven path
either** — the same wiring-gap shape already fixed once for
`hidden_channels`, just never checked for this field, because
`resolved_config.py` deliberately excludes mutation values (a solved value
is an artifact, not a design choice, per its own docstring) and the
existing single-source-of-truth tests had no field to check this against.
Every real run was silently using each encoder's raw, uncalibrated class
default (`direct=0.1`, `nca=0.1`, `rule_based=0.2`) instead of the
calibrated `0.1535`/`0.0305`/`0.0779` the project's entire mutation-tuning
workstream produced. **Fixed**: `benchmark.py::_load_calibrated_mutation_kwargs()`
reads the single committed calibration artifact
(`results/mutation/mutation_calibration_raw.json`) lazily, at resolve time
rather than import time, and raises with an actionable message if missing;
keyed by each arm's base encoder so `rule_based_monotone` also correctly
gets `rule_based`'s own calibrated rate. Verified via the real config path
(`test_real_benchmark_yaml_wires_calibrated_mutation_strength`, all four
arms, confirmed to fail against the pre-fix code — `0.2` instead of the
expected `0.0779` for `rule_based`) and documented in a new
`docs/decisions.md` row and `PRE_PARALLELIZATION_READINESS.md`'s corrected
table. This finding and fix directly affected what "architecture finalised
and validated" could honestly claim in the README — it was fixed before,
not after, writing that claim.

## Test Suite

Full suite (`python -m pytest -q`, includes `tests/evaluation/test_ppo_runner.py::test_real_ppo_smoke`, a genuine real-PPO run — confirmed executed, not skipped):

```
744 passed, 33 warnings in 629.84s (0:10:29)
```

**0 failed.** This includes every new test added across Parts 1-4: the
dirty-tree-gate regression test, both mutation-rate-overflow-warning test
pairs, the `windowed_body_at` clip-warning tests, the new
`test_rule_based_tiebreak_check.py` file, the AST-based duplicated-literal
regression test, and the four-arm calibrated-mutation-wiring test. All 33
warnings were reviewed; each maps to an intentional/documented behavior
(deprecation shims kept for `results/` reproducibility, the new clip
warnings firing correctly in legitimate existing test scenarios, a pandas
API-evolution notice, unrelated third-party deprecations) — none indicate
a masked failure.

## Git Status

Tree is clean. Every part's changes are isolated into its own commit, in
order:

```
23ade98 Add lab machine setup verification record
a89cf1a Audit docs/ and results/ for staleness against the current resolved config
e869ecf Confirm zero OPEN rows and resolve 5 internal contradictions in decisions.md
163a7f9 Close the silent-clip bug class project-wide; dedupe rule_based literals
0a630ca Rewrite README/READY_TO_RUN for current state; wire calibrated mutation
```

(on top of `bbe54c6`/`ff5905b`, Part 1's two code-level fixes, committed in
an earlier session turn). Two files with edits spanning two logical parts
(`docs/decisions.md`: Part 3's 5 supersession tags + Part 5's new mutation-
wiring row; `results/STALENESS_AUDIT.md`: Part 2's audit + Part 3's "Part 6"
section) were deliberately split via temporary truncation/restoration so
each commit contains exactly its own part's content, not a bundled diff.

A test run's own regenerated-report side effect
(`results/rule_based/ORDER_INDEPENDENCE_AUDIT.md`'s wall-clock timing line,
which a test rewrites on every full-suite run as a documented, expected
side effect) was reverted after each full-suite run to keep the tree clean
— not committed, since it carries no information (a timing number that
varies run to run on this machine).

## Final Verdict: **CODEBASE AND DOCUMENTATION CONSISTENT**

Every check in Parts 1-5 passed or was fixed and re-verified: the lab
environment fixes are applied and tested (tmux excepted, an environment
blocker unrelated to code/doc consistency); every `docs/`/`results/` file
making a current-state claim is either accurate or clearly marked
superseded with specifics; `docs/decisions.md` has zero OPEN rows and no
remaining internal contradiction; the silent-clip bug class and the
duplicated-resolved-literal risk are closed project-wide, each with a
regression test that was confirmed to catch the pre-fix bug; `README.md`
and `docs/READY_TO_RUN.md` accurately describe the current architecture and
status, including a real wiring bug (calibrated mutation strength never
reaching production) found and fixed in the course of verifying that
accuracy, not papered over; and the full test suite — 744 tests, including
a real-PPO run — passes with zero failures.

**This verdict is about internal consistency and accuracy only.** It does
**not** mean the project is ready to launch the real experimental matrix.
Parallelization design and the pre-flight wall-clock compute projection are
deliberately untouched by every task in this workstream, and
`primary_matrix`'s `population_size`/`offspring_size`/`eval_budget` remain
an explicit, disclosed open scoping decision (`docs/decisions.md`'s "Do NOT
guess..." row) — not resolved here, not implied resolved by this verdict.
Launch readiness is a separate, later gate; this document certifies only
that what exists today — the architecture design, its documentation, and
its own internal decision record — is consistent, accurate, and not
self-contradictory.
