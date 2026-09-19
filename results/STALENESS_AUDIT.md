# Staleness Audit: docs/ and results/ Markdown Files vs. Current Resolved Configuration

Date: 2026-09-19

## Method

1. **Reference table** established by reading, in full: `evodevo/encoders/nca/resolved_config.py`, `evodevo/encoders/rule_based/resolved_config.py`, `docs/decisions.md` (every row, 55 rows), and `evodevo/experiments/configs/benchmark.yaml`. See the table below.
2. **Every** `.md` file under `docs/` and `results/` was enumerated with `find docs results -iname "*.md"` (69 files, listed in full in the verdict table below) — not sampled.
3. Each file was checked for a claim of the form "current X is Y" / "the primary/final configuration is Y" against the reference table, using (a) targeted `grep` across all 69 files for known old-value signatures (`unbounded_residual`, `row_major_last_wins`, `hidden_channels.*3`/`H=3`, old mutation values `0.0201`/`0.1177`/`0.1707`/`0.3057`, old `D*=4.9915`, old `T_primary=39`/`T_primary=20`/`dev_steps: 39`, old timescale-calibration values `decay=0.05`/`B*=2`), (b) `git log --follow` on every candidate and every specially-named file to establish exactly when it was last substantively generated relative to `resolved_config.py`'s introduction (commits `a5abe84`/`0af2992`, 2026-09-17), and (c) direct reading of the file's own framing — an explicit "OLD vs. NEW" / "before vs. after" comparison table is not stale even if it cites an old value, since it is the value's own documented history, not a false current-state claim.
4. Files found STALE were given a superseded banner at the top (not deleted), citing the specific old vs. new values and the `docs/decisions.md` row that made the correction, per the task's required banner format.

## Reference table (current, as of this audit)

**NCA** (`evodevo/encoders/nca/resolved_config.py::PRIMARY`): `update_rule="bounded_leaky"`, `init_gain=1.0`, `decay=0.2`, `state_bound=5.0`, `hidden_channels=0`, `spectral_normalise=False`, `readout_mode="mode_of_window"`, `readout_window=50`, `mutation_sigma=0.0305` (third re-run, `results/mutation/MUTATION_CALIBRATION.md`).

**rule_based** (`evodevo/encoders/rule_based/resolved_config.py::PRIMARY`): `conflict_resolution="first_rule_index"`, `position_predicate="disabled"`, `boundary="no_op"`, `neighbourhood="von_neumann"`, `action_monotonicity="unrestricted"` for PRIMARY — **but `MONOTONE_ABLATION` (`action_monotonicity="monotone"`) is now a co-primary PPO-matrix candidate alongside PRIMARY** (`PPO_MATRIX_CANDIDATES`, deferred decision, not "kept as ablation only"), `readout_mode="mode_of_window"`, `readout_window=6`, `mutation_rate=0.0779` (current regenerated artifact).

**Shared `T_primary` = 100** (`evodevo/experiments/configs/benchmark.yaml`'s `factors.dev_steps`), via the T-shift-stability criterion + the NCA T_primary/W consistency fix — NOT the earlier `q95(settle_step)`-based 39, 20, or the NCA-only provisional 5.

**Mutation**: `D*=2.5000`; `direct mutation_rate=0.1535`; `nca mutation_sigma=0.0305`; `rule_based mutation_rate=0.0779` — NOT the original P5 values (`D*=4.9915`, `0.3057`/`0.0201`/`0.1707`) or the intermediate second re-run (`nca mutation_sigma=0.1177`).

**Profile mechanism**: `PROFILES = {"fast_dev": {ppo_timesteps:5000, population_size:15, offspring_size:15, eval_budget:300}, "primary_matrix": {ppo_timesteps:1_000_000}}`; `benchmark.yaml` is currently on `fast_dev`; `primary_matrix`'s `population_size`/`offspring_size`/`eval_budget` are a deliberate, disclosed OPEN decision (no value adopted).

## Verdict table (all 69 files)

Legend: **CURRENT** = checked, accurate, no action. **SUPERSEDED** = banner applied (see list below for specifics). **N/A** = not a current-state claim file (self-labeled historical/archive/ablation-data/process/planning doc).

| File | Verdict |
|---|---|
| docs/adding-an-encoder.md | N/A — encoder-contract process doc, no hardcoded resolved values |
| docs/analysis-protocol.md | CURRENT — regenerated with `ABLATION_WITHIN_ARM` distinction (commit `35298f8`) |
| docs/ARCHITECTURE_SNAPSHOT.md | **SUPERSEDED** — banner applied |
| docs/CODE_STRUCTURE_REFERENCE.md | **SUPERSEDED** — banner applied |
| docs/decisions.md | N/A — the source of truth itself; append-only decision log by design |
| docs/evodevo-finalization-prompts.md | N/A — process/prompt-history doc |
| docs/evodevo-implementation-roadmap.md | N/A — original planning doc; its `[DEC]` items are explicitly recommendations pending resolution in `docs/decisions.md`, not current-state claims |
| docs/evogym_technical_reference.md | N/A — describes EvoGym's own (external, unchanged) API, not NCA/rule_based resolved config |
| docs/FINAL_SMOKE_TEST.md | **SUPERSEDED** — banner applied (special handling, see below) |
| docs/INCIDENT_VALIDITY_FAILURE.md | N/A — incident report, scoped to its own dated incident |
| docs/LAB_MACHINE_SETUP.md | N/A — environment setup record, no NCA/rule_based config claims |
| docs/module-ownership.md | N/A — directory-ownership map, no resolved values |
| docs/paper-manifest.md | CURRENT — uses explicit OLD/NEW framing for `unbounded_residual`/`row_major_last_wins` |
| docs/PRE_PARALLELIZATION_READINESS.md | CURRENT — most recent commit in the repo (`092fcd5`), explicitly regenerated for T_primary/hidden_channels/profiles/ablation labels |
| docs/READY_TO_RUN.md | **SUPERSEDED** — banner applied |
| docs/RESULTS_EXPLAINED.md | **SUPERSEDED** — banner applied |
| docs/RULE_BASED_MODULARIZATION.md | N/A — process/verification record for a specific completed refactor |
| docs/rule-language-spec.md | CURRENT — corrected order-independence citations (`docs/decisions.md`'s "Citation-mismatch resolution" row) |
| results/ablation_priors/E3_nca/nca_sanity.md | N/A — raw data for one labeled ablation-grid cell, no prose "current" claim |
| results/ablation_priors/H0/nca_sanity.md | N/A — same |
| results/ablation_priors/H3/nca_sanity.md | N/A — same |
| results/ablation_priors/T10/nca_sanity.md | N/A — same |
| results/ablation_priors/T1/nca_sanity.md | N/A — same |
| results/ablation_priors/T20/nca_sanity.md | N/A — same |
| results/ablation_priors/T5/nca_sanity.md | N/A — same |
| results/ablations.md | **SUPERSEDED** — banner applied (partial: "not yet run" status still true) |
| results/archive/discarded-progress-reporting-cutover/README.md | N/A — self-labeled archive, explains its own supersession |
| results/archive/pre-final-launch-v2/README.md | N/A — self-labeled archive |
| results/archive/pre-final-run-direct/README.md | N/A — self-labeled archive |
| results/BENCHMARK_REPORT.md | **SUPERSEDED** — banner applied (special handling, see below) |
| results/COMPARISON_READINESS.md | **SUPERSEDED** — banner applied (numbers only; its own BLOCKER finding stands) |
| results/depth/DEPTH_STUDY.md | **SUPERSEDED** — banner applied (T_primary number only) |
| results/FINAL_STRUCTURAL_COMPARISON.md | CURRENT — regenerated `011e71f`, explicit OLD/NEW framing throughout |
| results/initial_condition/INIT_STUDY.md | CURRENT — P3 `center_seed` decision never superseded |
| results/inspection/direct/direct_sanity.md | CURRENT — direct encoder never modified by any workstream |
| results/inspection/nca/nca_sanity.md | **SUPERSEDED** — banner applied |
| results/inspection/rule_based/rule_based_sanity.md | **SUPERSEDED** — banner applied |
| results/mutation/MUTATION_CALIBRATION.md | CURRENT — verified current content matches decisions.md row 54's regenerated numbers exactly (D*=2.5000, nca=0.0305, rule_based=0.0779) |
| results/nca/ARCHITECTURE_DELTA.md | CURRENT — explicit before/after table, "NEW = the FINAL resolved primary" |
| results/nca/CONVERGENCE.md | CURRENT — explicit "before/after" columns, after-column labeled "the FINAL resolved primary" |
| results/nca/DYNAMICS_CLASSIFICATION.md | CURRENT — non-gating diagnostic, correctly scoped to its own measurement |
| results/nca/HORIZON_EXTENSION.md | CURRENT — correctly scoped diagnostic feeding into the (separately current) T_primary derivation |
| results/nca/INIT_STUDY.md | CURRENT — part of the a5abe84 redesign commit, scoped diagnostic |
| results/nca/MUTATION_CALIBRATION_READOUT_CASCADE.md | CURRENT — explicit cascade/before-after framing |
| results/NCA_READINESS.md | **SUPERSEDED** — banner applied (partial: hidden_channels + T_primary claims only) |
| results/nca/READOUT_POLICY_SELECTION.md | CURRENT — is itself the derivation of the current readout policy |
| results/nca/READOUT_STUDY.md | CURRENT — explicit H-ablation comparison, correctly scoped |
| results/nca/REGENERATION.md | CURRENT — explicitly scoped to the CERTIFIED_ABLATION regime, not PRIMARY |
| results/nca_sanity.md | **SUPERSEDED** — banner applied |
| results/nca/TIMESCALE_CALIBRATION.md | CURRENT — explicit "negative result" report about a retracted intermediate point, correctly labeled as such |
| results/nca/TRADEOFF_SURFACE.md | CURRENT — is itself the derivation of the current Outcome-B PRIMARY |
| results/rule_based/ARCHITECTURE_DELTA.md | CURRENT — explicit OLD/NEW table |
| results/rule_based/CAPACITY_STUDY.md | CURRENT — self-discloses the stale-792-vs-current-501 distinction |
| results/rule_based/CONFLICT_POLICY.md | CURRENT — explicit baseline-vs-alternatives comparison (confirmed accurate historical snapshot per decisions.md row 47) |
| results/rule_based/CONTEXT_ANALYSIS.md | CURRENT — scoped to `position_predicate="disabled"`, "the encoder's current default" |
| results/rule_based/CYCLE_CHARACTERISATION.md | CURRENT — explicitly labeled "CURRENT dynamics (R2 first_rule_index + R3 position-disabled)" |
| results/rule_based/LOCALITY_STUDY.md | CURRENT — explicitly says "`position_predicate=\"disabled\"` (the new default)" |
| results/rule_based/MONOTONE_ABLATION.md | **SUPERSEDED** — banner applied (status line only; data unaffected) |
| results/rule_based/MUTATION_DISPLACEMENT_CHECK.md | CURRENT — explicit before/after check under new defaults |
| results/rule_based/ORDER_INDEPENDENCE_AUDIT.md | CURRENT — corrected figures (0.1295/0.3040), own subject is documenting the pre-R2 baseline explicitly |
| results/RULE_BASED_READINESS.md | **SUPERSEDED** — banner applied (partial: monotone-status claim only) |
| results/rule_based/READOUT_POLICY_SELECTION.md | CURRENT — is itself part of the current readout-policy derivation |
| results/rule_based/SETTLING_AUDIT.md | CURRENT — correctly historically-scoped diagnosis; its `T_primary=39` mention is explicitly discussed as "does this change T_primary's validity" at the time, not asserted as the current value |
| results/rule_based/TIEBREAK_VALIDITY_CHECK.md | CURRENT — re-run and reconfirmed under current defaults (decisions.md row 47) |
| results/runs/strategy_comparison/DECOMPOSITION.md | CURRENT — accurately scoped to its own stated 3-arm dataset, does not claim 4-arm coverage |
| results/STATS_PROTOCOL_VALIDATION.md | CURRENT — Section 2 uses `strategy_comparison_v2`, shows `ablation_within_arm` label correctly |
| results/structural/STRUCTURAL.md | CURRENT — regenerated `011e71f`, no stale hardcoded NCA/rule_based literals found |
| results/T_PRIMARY_UNIFIED.md | CURRENT — is itself the canonical source for the current shared `T_primary=100` |
| results/windowed_validity_check/VALIDITY_CHECK.md | CURRENT — explicitly cites "T_primary=100 (the corrected shared T_primary)" |

## SUPERSEDED files: specific old-vs-new values

1. **`results/BENCHMARK_REPORT.md`** (special handling — see below)
2. **`docs/FINAL_SMOKE_TEST.md`** (special handling — see below)
3. **`docs/RESULTS_EXPLAINED.md`** — same stale run as #1 (lay explainer); NCA non-settling/state-magnitude figures are pre-redesign.
4. **`docs/READY_TO_RUN.md`** — lock hash `56b3ed67...` computed against the pre-redesign resolved config; NCA "FROZEN" claim did not hold.
5. **`results/COMPARISON_READINESS.md`** — `T_primary=39` (Section 3) → current shared `T_primary=100`. Its own Section-7 BLOCKER finding about `results/BENCHMARK_REPORT.md` remains valid.
6. **`results/NCA_READINESS.md`** — (a) treats `hidden_channels=3` as still-undecided ("Decide whether to drop...") → already decided, current PRIMARY is `hidden_channels=0`. (b) `T_primary=5` (Outcome-B-provisional) → current shared `T_primary=100`.
7. **`results/RULE_BASED_READINESS.md`** — Section 6/12/"kept as ablation" → current: `monotone` is a co-primary PPO-matrix candidate (DEFER decision), not ablation-only.
8. **`results/rule_based/MONOTONE_ABLATION.md`** — own "Status: ABLATION, not primary" line → superseded by the same DEFER/co-primary decision as #7.
9. **`results/depth/DEPTH_STUDY.md`** — headline `T_primary = 39` (q95-based, method later found ill-posed) → current shared `T_primary=100` (T-shift-stability criterion).
10. **`results/nca_sanity.md`** — pre-redesign `dev_steps=10`/`hidden_channels=3`/unbounded-residual numbers, cited as live evidence by #1.
11. **`results/inspection/nca/nca_sanity.md`** — identical content/issue to #10.
12. **`results/inspection/rule_based/rule_based_sanity.md`** — `validity_rate=0.5975` under pre-R2/R3 defaults → current PRIMARY measures 0.5470.
13. **`results/ablations.md`** — "`hidden_channels=3` is the NCA default" → current PRIMARY is `hidden_channels=0`. ("not yet run" status remains accurate.)
14. **`REPO_INVENTORY_REPORT.md`** — entire pre-restructure repo layout (its own commit message says so), predates the current directory structure entirely.
15. **`docs/ARCHITECTURE_SNAPSHOT.md`** — Section B.2 (NCA: plain residual update, `hidden_channels=3`) and B.3 (rule_based: no conflict_resolution/position_predicate/readout axes) both pre-redesign.
16. **`docs/CODE_STRUCTURE_REFERENCE.md`** — same as #15, describes `_step` as unbounded residual with no `update_rule` concept.

## Special handling: `results/BENCHMARK_REPORT.md` and `docs/FINAL_SMOKE_TEST.md`

Both were checked against `git log --follow` on the file itself vs. on `evodevo/encoders/nca/resolved_config.py` / `evodevo/encoders/rule_based/resolved_config.py`:

| File | Last substantive commit | resolved_config.py introduced |
|---|---|---|
| `results/BENCHMARK_REPORT.md` | `7675f0d`, 2026-09-15 | `a5abe84`/`0af2992`, 2026-09-17 |
| `docs/FINAL_SMOKE_TEST.md` | `0a9a61b`, 2026-09-15 (content dated 2026-09-02) | `a5abe84`/`0af2992`, 2026-09-17 |

**Both predate the entire redesign.** Both banners are the most prominent in this audit and explicitly state: the real, final benchmark/smoke-test result under the current resolved architecture **does not yet exist**, and will be produced by the actual primary-matrix run once `evodevo/experiments/configs/benchmark.yaml`'s `profile` is moved to `primary_matrix` and its `population_size`/`offspring_size`/`eval_budget` open decision (`docs/decisions.md`'s "Do NOT guess..." row) is resolved. Neither file was deleted — both are kept for historical/methodological value (the smoke test in particular proved the pipeline's plumbing, which remains true; only the encoder configuration it exercised is stale).

## Coverage confirmation

All 69 `.md` files under `docs/` and `results/` appear in the verdict table above. `results/BENCHMARK_REPORT.md` and `docs/FINAL_SMOKE_TEST.md` are both explicitly resolved as SUPERSEDED with prominent banners, not silently skipped.

## Part 6 — `docs/decisions.md` Internal Consistency Audit

### 1. Zero-OPEN-rows confirmation (via the actual enforcing convention, not informal grep)

`docs/decisions.md` itself states: *"No row may remain 'OPEN' in the final submission."* The project's own enforcing mechanism is `evodevo.analysis.readiness_check.scan_decisions_open_rows()` — it does **not** match the word "OPEN" anywhere in a row (an earlier version did, and mis-flagged a resolved row whose own rationale prose merely *discussed* its prior OPEN status); it matches only rows whose **Chosen** column (the 4th of 5 table columns) is the bare marker `OPEN`.

Re-ran this exact function directly against the live file:

```
from evodevo.analysis.readiness_check import scan_decisions_open_rows
scan_decisions_open_rows()  ->  []   (0 rows)
```

Also ran its two existing regression tests (`tests/analysis/test_readiness_check.py` and `tests/analysis/test_rule_based_readiness_check.py`, both `test_scan_decisions_open_rows_ignores_the_word_open_in_resolved_rationale`) — both pass, confirming the detection *logic itself* is sound, not just that it happened to return zero this time.

**For contrast, informal `grep -c "OPEN" docs/decisions.md` returns 5** — every one of those 5 hits is prose discussing a row's *former* OPEN status or the file's own header rule, not a live OPEN row. This is exactly the false-positive class the project's own convention was built to avoid, and confirms why "re-run the actual test" (not grep) was the right instruction: informal grep would have produced a misleading non-zero count.

**Result: 0 OPEN rows, confirmed by the real enforcing convention. Total data rows in the decisions table: 47** (counted programmatically with the exact same row-inclusion logic `scan_decisions_open_rows()` itself uses — header and separator lines excluded; one of the table's rows, dated 2026-09-16 on the readout-policy resolution, has its Rationale cell continue onto a further unwrapped paragraph in the raw file that is not itself a new `|`-prefixed table row, so a naive line-range count over-estimates).

### 2. Contradiction sweep (full read, start to finish, all 47 rows)

Read every row in full. `docs/decisions.md` is largely self-policing already — 6 rows (dated topics: the two original T_primary rows, the TRANSIENT-vs-LIMIT-CYCLE row, the readout-policy row, the unified-T_primary row, and the mutation re-run row) already carry an explicit `(SUPERSEDED -- see ... row below/above)` tag in their own topic cell, and one entire correction (the order-independence percentage mismatch, 12.95%/30.40% vs. a stale 15.90%/34.45%) was deliberately left unmarked on the *original* R2/R3 rows by the row that found it (its own text: *"docs/decisions.md's own R2/R3 rows are NOT edited (append-only convention) -- this row is the correction of record instead"*) — a disclosed, reasoned choice, not an oversight, so it was left as-is.

**5 genuine contradictions were found** — an older row stating a "final"/adopted value on the same topic as a later row's different final value, with no cross-reference either direction — and each was fixed by adding a short `(SUPERSEDED -- see the "..." row below)` tag to the **older** row's topic cell, in the exact style the document already uses elsewhere. No Chosen/Rationale content was edited or removed (append-only convention preserved).

| Topic | Older row (now tagged) | Newer row (already correct) | What changed |
|---|---|---|---|
| Hidden channel count (H) | "Hidden channel count (H)" — `H=3` is PRIMARY | "Hidden channel count (H) -- final re-resolution" — DROP H to 0 | `H`: 3 → 0 |
| Mutation tuning (values) | "Mutation tuning procedure" (P5 original) — `D*=4.9915`, `nca σ=0.0201` | "...RE-RUN under the bounded/normalised NCA architecture" — `D*=2.5000`, `nca σ=0.1177` | D* and all three arms' parameters re-solved |
| Mutation tuning (values, 2nd hop) | "...RE-RUN under the bounded/normalised..." (itself now also tagged) | "...THIRD re-run, under the Outcome-B PRIMARY" — `nca σ=0.0305` | nca `mutation_sigma` re-solved again for the final architecture |
| NCA timescale calibration | "NCA timescale calibration..." — selected `lambda*=0.05, B*=2` | "NCA joint stability-vs-expressivity trade-off" — RETRACTS N4's point, adopts `decay=0.2, state_bound=5.0` | decay/state_bound: 0.05/2 → 0.2/5.0 |
| rule_based monotone status | "Implement and verify the termination-guaranteed 'monotone' ablation..." — "monotone remains the ablation" | "Monotone vs unrestricted: an explicit decision" — DEFER, run both as co-primary | status: ablation-only → co-primary PPO-matrix candidate |

Re-ran `scan_decisions_open_rows()` after these edits: still 0 (the tags were added to the topic cell, never the Chosen cell, so the OPEN-detection convention is unaffected).

**Note on a pre-existing, unrelated parsing quirk found while verifying:** rows discussing the mutation tolerance (`\|mean - D\*\| <= 0.5`) use escaped literal pipe characters for absolute-value notation in prose. A naive pipe-count/split (as both informal `grep -c '|'` and the readiness-checker's own naive `line.split("|")` do) over-counts these as extra table cells. This predates this audit (confirmed present in the original row text before any edit here) and does not affect the OPEN-row check's correctness for the current file (verified: still 0, and the check's own logic only looks for an exact `"OPEN"` string match, which no misaligned cell produces) — flagged here for visibility, not fixed, since it's outside this task's scope.

### 3. Every `results/*.md` citation: existence + Part-2-staleness cross-check

Extracted all 33 unique `results/*.md` paths cited across every row (`grep -oE`), and separately checked the one citation in the document's own header (not a row).

- **32 of 33 row-cited files exist** and were cross-checked against Part 2/5's staleness verdict table above.
- **1 does not exist**: `results/RESULTS.md`, cited only in the document's own header ("...before the results in `results/RESULTS.md` are considered final"). This is not a broken citation to something that was supposed to already exist — it is an explicit forward reference to the eventual final paper-facing results write-up, which by design does not exist yet (the actual primary-matrix PPO run has not happened; see `results/BENCHMARK_REPORT.md`'s own superseded banner and `docs/PRE_PARALLELIZATION_READINESS.md`). No fix needed; noted for completeness per this task's instruction to confirm every citation resolves to a real file.
- **No row cites a now-superseded report's stale number as if it were still live evidence.** The one place this risk was real — row 49 ("Monotone vs unrestricted") citing `results/rule_based/MONOTONE_ABLATION.md` — correctly uses that report's underlying structural *measurements* (which are unaffected by the status contradiction) to justify a *different* conclusion than that report's own now-superseded "kept as ablation" status line; it does not repeat the stale status claim. This is exactly the pattern Part 2's own banner on `results/rule_based/MONOTONE_ABLATION.md` already documents.
- Row 39 (the "T=40-staleness audit" row) explicitly cites `results/NCA_READINESS.md` and `results/COMPARISON_READINESS.md` — both flagged SUPERSEDED in Part 2/5 above — but only for a narrow, explicitly-scoped purpose (whether their `T=40` mentions are stale relative to the shared `T_primary`), and correctly concludes those specific mentions are historical citations, not live claims. This narrow use is accurate and is not the same claim Part 2/5 flagged (Part 2 flagged NCA_READINESS.md's *hidden_channels* and *T_primary=5* claims, and COMPARISON_READINESS.md's *headline* `T_primary=39`, none of which row 39 relies on). No contradiction.
- `docs/READY_TO_RUN.md` (flagged SUPERSEDED in Part 2/5) is cited exactly once, in row 24, purely to describe the historical act of recording a lock hash there in 2026-09-02 — not as a claim that the hash is still valid today. That row's own rationale already states the hash is invalidated by any future config change, which is exactly what later happened; no contradiction.
- No row cites `results/BENCHMARK_REPORT.md`, `docs/FINAL_SMOKE_TEST.md`, `docs/RESULTS_EXPLAINED.md`, `docs/ARCHITECTURE_SNAPSHOT.md`, `docs/CODE_STRUCTURE_REFERENCE.md`, or `REPO_INVENTORY_REPORT.md` at all (all six flagged SUPERSEDED in Part 2/5) — so there is no risk of `decisions.md` treating any of them as live evidence.

### Verification

After all edits: `scan_decisions_open_rows()` still returns `[]`; the two existing readiness-check regression tests for this function still pass; `docs/decisions.md`'s table structure (5 columns per row) is intact for every edited row.

**Acceptance check**: 0 OPEN rows confirmed via the actual enforcing test (not informal grep). All 5 found contradictions resolved with an explicit supersession note on the older row, matching the document's own established tagging convention — none silently merged or deleted.
