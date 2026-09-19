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
