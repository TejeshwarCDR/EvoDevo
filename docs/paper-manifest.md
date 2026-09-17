# Figure and table manifest

Every figure/table the final report will contain, fixed in advance of any
real PPO result (see `docs/PRE_PARALLELIZATION_READINESS.md`). No actual
data appears below -- only numbering, titles, caption skeletons, exact
axes/columns, and which arms/tasks each item covers. This document is
task-agnostic by construction: nothing below assumes `Walker-v0`
specifically (see the Appendix's explicit grep confirmation), since a second
task is the next workstream after this one.

Arm set referenced throughout: `direct`, `nca`, `rule_based`, `rule_based_monotone`
(the co-primary pair from `docs/decisions.md`'s "Monotone vs unrestricted"
row, registered as a first-class arm by this same change -- see
`docs/decisions.md`'s "Wire the resolved readout policy..." row's 2026-09-16
update). Strategy set: `ea`, `random_search` (P7).

---

## Figure 1 -- Dual-panel budget curves

**Title**: Best-so-far fitness vs. evaluation budget, per task.

**Caption skeleton**: "Best-so-far fitness across `{N_SEEDS}` seeds for each
of `{ARM_LIST}` on `{TASK}`, under the `ea` strategy. (a) indexed against
evaluations spent (the budget-governing counter); (b) indexed against
evaluations valid (real, decode-and-validate-passing search attempts) --
see `docs/analysis-protocol.md` Sec 4/6 for why both are shown. Line: median
across seeds; band: interquartile range (25th-75th percentile)."

**Axes**: (a) x = `evals_spent`, y = best-so-far fitness. (b) x =
`evals_valid_cumulative`, y = best-so-far fitness. One figure per task, one
line+band per arm.

**Arms/tasks**: all arms registered at report time, `ea` strategy only (the
`random_search` decomposition is Figure 4, not this figure). One instance
per task in the matrix (currently 1: Walker-v0).

**Source**: `evodevo/analysis/budget_curves.py::plot_two_panel_comparison`
(median+IQR, per `docs/analysis-protocol.md` Sec 4).

---

## Table 1 -- Validity / diversity / structural-metric comparison

**Title**: Cross-arm validity, diversity, and structural-metric summary.

**Caption skeleton**: "Per-arm summary at the resolved primary configuration
(`docs/PRE_PARALLELIZATION_READINESS.md`'s resolved-config table), N=`{N}`
genomes/arm, seed=`{SEED}`, `init_condition=center_seed`, `dev_steps={T_PRIMARY}`."

**Columns**: `arm`, `valid_rate`, `degenerate_fraction`, `unique_valid_body_fraction`,
`uniform_body_fraction`, `never_settled_fraction`, `mean_hamming_displacement`
(one-mutation), `fraction_zero_displacement`.

**Arms/tasks**: all four arms (this is the table `rule_based_monotone`
becomes real in, once Part 1 registers it -- previously only reportable for
`direct`/`nca`/`rule_based`). Task-agnostic structural metrics (not
PPO/task-performance), so one instance suffices regardless of task count,
but re-run per task if any structural factor (e.g. `grid_shape`) ever varies
by task.

**Source**: `evodevo/analysis/structural.py` / `evodevo/analysis/structural_report.py`
(existing), extended to include `rule_based_monotone` as a fourth row.

---

## Table 2 -- NCA architecture before/after

**Title**: NCA architecture revision: before vs. after.

**Already exists -- cite, do not recreate**: `results/nca/ARCHITECTURE_DELTA.md`,
produced by `evodevo/analysis/architecture_delta.py`. Covers
`never_settled_fraction`, light-cone propagation speed, `T_primary`, max
state magnitude, validity/diversity, and regeneration, comparing the
pre-N2..N6 architecture (`update_rule="unbounded_residual"`) against the
final resolved PRIMARY (`update_rule="bounded_leaky"`).

**Report action**: reproduce its existing table verbatim (import/format the
committed `.md`'s table), with a one-line pointer to the source report.

---

## Table 3 -- rule_based architecture before/after

**Title**: rule_based architecture revision: before vs. after.

**Already exists -- cite, do not recreate**: produced by
`evodevo/analysis/rule_based_architecture_delta.py`, comparing the OLD
(`row_major_last_wins`/position-enabled) vs. NEW (`first_rule_index`/
position-disabled) resolved defaults, per `docs/decisions.md`'s "Re-run the
R2/R3 cascade for rule_based only" row.

**Report action**: same as Table 2 -- reproduce verbatim, cite the source.

---

## Figure 2 -- Exact-cycle vs. empirically-intractable dynamics characterisation

**Title**: Dynamics classification: exact (rule_based) vs. empirical (NCA).

**Caption skeleton**: "rule_based's discrete, finite-state dynamics admit an
EXACT cycle/reachability characterisation (`{N_STATES}`-state enumeration,
`docs/decisions.md`'s 'Exact rule reachability/coverage/redundancy' row);
NCA's continuous-state dynamics do not, and are instead characterised
empirically via horizon extension and settling/never-settled classification
over `{N}` sampled trajectories."

**Axes/columns**: side-by-side panel or table -- (left) rule_based's exact
cycle-length distribution / period histogram (from `CYCLE_CHARACTERISATION.md`);
(right) NCA's empirical never-settled-fraction vs. horizon (`T`) curve, from
`evodevo/analysis/horizon_extension.py` and `evodevo/analysis/dynamics_classification.py`'s
STILL_GROWING / PLATEAUED_OSCILLATING / SLOW_BUT_MONOTONE breakdown.

**Arms/tasks**: `rule_based` (exact) and `nca` (empirical) only -- this
contrast does not apply to `direct` (no development) or extend meaningfully
to `rule_based_monotone` beyond noting its hard termination guarantee
(Proposition R4, already exact by construction: 0 cycles).

**Why this figure**: identified in prior workstreams as a genuine, reportable
cross-arm finding (one developmental arm is exactly analysable, the other
is not, and this is itself a property of the encodings being compared, not
a limitation of the analysis) -- it needs its own figure rather than being
folded into Table 1's numeric summary, which cannot convey "exact" vs.
"empirical" as a qualitative distinction.

**Source**: cite `results/rule_based/CYCLE_CHARACTERISATION.md`,
`results/nca/HORIZON_EXTENSION.md`, `results/nca/DYNAMICS_CLASSIFICATION.md`
(if not yet produced under this name, generate via the modules named above --
no new logging is required, see the Appendix).

---

## Figure 3 -- Regeneration / robustness curves

**Title**: Post-perturbation regeneration and structural robustness.

**Caption skeleton**: "Fraction of genomes recovering a valid/certified body
within `{K}` steps of a single-voxel perturbation, per arm, N=`{N}` trials."

**Axes**: x = steps-since-perturbation, y = fraction recovered (or
structural-distance-to-original), one line per arm.

**Arms/tasks**: both developmental arms (`nca`, `rule_based`, and
`rule_based_monotone` once real) -- `direct` has no development/regeneration
concept and is excluded with a one-line note, not silently omitted.

**Source**: `evodevo/analysis/regeneration.py` (existing; per
`results/nca/REGENERATION.md`'s 0-bound-violations-over-10,809-trials
result, cited rather than recomputed for NCA); extend to `rule_based`/
`rule_based_monotone` if not already covered there.

---

## Table 4 -- Statistical significance table

**Title**: Pairwise statistical comparisons (Holm-Bonferroni corrected).

**Caption skeleton**: "Every planned pairwise comparison in the current
experimental matrix (`docs/analysis-protocol.md` Sec 3), Mann-Whitney U with
Vargha-Delaney A12 effect size, Holm-Bonferroni corrected across the full
`m={M}`-comparison family."

**Columns**: `comparison` (`cell_a` vs `cell_b`), `kind` (`cross_arm` /
`ablation_within_arm` / `arm_vs_own_random_search` -- see
`docs/analysis-protocol.md` Sec 3's "cross_arm vs. ablation_within_arm"
subsection: `cross_arm` is a representation-choice question between two
DIFFERENT base encoders, `ablation_within_arm` is an ablation question
between two arms sharing ONE base encoder -- e.g. `rule_based` vs.
`rule_based_monotone` -- derived generically from shared base-encoder
identity, never hardcoded by name), `n_a`, `n_b`, `mean_a`, `mean_b`,
`vargha_delaney_a12`, `mann_whitney_u`, `p_raw`, `p_holm`,
`significant_at_alpha`. The `kind` column must be rendered visibly (not
just available in the underlying data) so a reader never mistakes an
ablation result for a cross-arm one or vice versa.

**Arms/tasks**: the full planned family for whatever matrix is in effect at
report time (currently `m=10`: 6 cross-arm-family EA pairs -- 5 `cross_arm`
+ 1 `ablation_within_arm` (`rule_based` vs `rule_based_monotone`) -- + 4
`arm_vs_own_random_search` pairs, 1 task -- see `docs/analysis-protocol.md`
Sec 3 for the exact derivation). Doubles automatically per added task.

**Source**: `evodevo/analysis/stats_protocol.py::run_protocol`, exactly as
rehearsed in `results/STATS_PROTOCOL_VALIDATION.md` Sec 2.

---

## Figure 4 -- Light-cone propagation-speed comparison

**Title**: Information/perturbation propagation speed across arms.

**Caption skeleton**: "Fitted propagation speed (cells/step) from a
single-voxel perturbation, per developmental arm, `T_primary={T}`."

**Axes/columns**: bar or point comparison, one value per developmental arm
(`nca`, `rule_based`, `rule_based_monotone`), y = fitted cells/step.

**Arms/tasks**: developmental arms only (`direct` has no propagation
concept -- excluded with a note). Task-agnostic (a structural/dynamics
measurement, not task performance).

**Source**: `evodevo/analysis/structural.py`'s `light_cone` instrument
(existing; already produced `results/structural/figs/light_cone.png` and
`results/structural/nca/light_cone.jsonl`/rule_based counterpart) --
extend to include `rule_based_monotone`.

---

## Table 5 -- Description-length / capacity comparison

**Title**: Genome description-length (capacity) comparison.

**Caption skeleton**: "Bits of description per genome, by arm and
architecture point, per `docs/decisions.md`'s 'Justify MIN_RULES/MAX_RULES
capacity bounds' row."

**Columns**: `arm`, `architecture_point` (where applicable, e.g. rule_based's
`MIN_RULES`/`MAX_RULES` bound), `bits_of_description`.

**Already exists -- cite, do not recreate**: `evodevo/analysis/rule_based_capacity_study.py`'s
result (direct=58.0 bits, rule_based=21.4-111.0 bits, NCA=16032/25344 bits
depending on architecture point).

**Report action**: reproduce verbatim; extend with a `rule_based_monotone`
row only if its capacity genuinely differs from `rule_based`'s (it does not,
by construction -- monotone changes `action_monotonicity`, not the rule
genome's own description length -- so this table may simply note "identical
to rule_based" for that row rather than re-deriving it).

---

## Appendix: logging schema gaps found and fixed

**Charter**: every figure/table above was cross-checked against the current
run logger (`evodevo/evolution/eval_bookkeeping.py::EvaluationTracker`, the
single shared per-evaluation/per-generation logging implementation used by
both `ea.py` and `random_search.py`) to confirm it actually logs every field
each figure needs. Gaps are fixed here, not deferred to be discovered
mid-run.

**Gaps found and fixed** (this change):

1. **Per-generation morphology diversity**: not logged before this change --
   `record_generation_summary()`'s output had no aggregate diversity metric,
   only per-individual `body_metrics()` (unaggregated). **Fixed**: added
   `generation_unique_valid_bodies` (count of distinct `body_hash` values
   among that generation's own valid individuals) to the per-generation
   summary record, needed by any future per-generation diversity curve
   (Table 1/Figure 1 use run-level, not per-generation, diversity today, but
   a within-run diversity-over-generations figure was previously
   unbuildable from the logs at all).

2. **Per-generation-only (non-cumulative) validity breakdown**: the existing
   `evals_invalid_reason_counts` field is CUMULATIVE (as-of-this-generation,
   including every prior generation) -- there was no way to see a single
   generation's own invalid-reason mix in isolation. **Fixed**: added
   `generation_invalid_reason_counts` (this generation's own batch only) and
   `generation_size` (this generation's own evaluation count, the
   denominator needed to turn the counts into a per-generation rate).

**Gap NOT fixed, and why that is correct rather than an oversight**:

3. **Exact-cycle-vs-chaos comparison (Figure 2)**: this needs NO new
   per-run logging. It is computed entirely from `depth_study.py`/
   `dynamics_classification.py`/`horizon_extension.py`'s own OFFLINE,
   standalone sampling of freshly-generated genome populations (N=2000-style
   runs, independent of any EA/random_search run's log) -- never from
   per-evaluation or per-generation records inside `run_ea`/`run_random_search`.
   Adding a logging field for this would duplicate data these modules
   already produce and would not actually feed Figure 2, which reads their
   `.md`/`.json` output directly. Confirmed by inspecting all three modules:
   none of them consume `run_log.jsonl`/`generation_log.jsonl`.

**Task-agnosticism check (explicit, per this task's MUST NOT list)**:

```
$ git diff evodevo/evolution/eval_bookkeeping.py | grep -i "walker\|upstepper"
(no output -- confirmed no hardcoded task name in the logging-schema change)
```

`evodevo/evolution/eval_bookkeeping.py`, `ea.py`, `random_search.py`, and
`logging.py` collectively reference `task` only via `config.task` (an opaque
string passed through to the evaluator) -- confirmed by inspection during
this same change, and unchanged by the two new fields above, both of which
read only `body_hash`/`valid`/`reason_category` off each evaluation's own
record.
