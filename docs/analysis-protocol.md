# Statistical analysis protocol (LOCKED before any real PPO result exists)

This document fixes, in advance, how every arm-vs-arm comparison in this
project's final report will be computed. It is written and committed before
any real PPO result has been produced (see `docs/decisions.md` and
`docs/PRE_PARALLELIZATION_READINESS.md` for provenance), specifically so the
protocol cannot be chosen or adjusted after seeing results. The single
generic implementation of everything below is
`evodevo/analysis/stats_protocol.py`; its two independent validations are in
`results/STATS_PROTOCOL_VALIDATION.md`.

## 1. Primary hypothesis test: Mann-Whitney U

**Choice**: Mann-Whitney U, two-sided, for comparing two arms'/cells'
final-fitness distributions across seeds.

**Justification**: (a) it is non-parametric -- it makes no assumption that
final fitness is normally distributed across seeds, which this project has
no basis to assume for a population-based EA's output (fitness landscapes
here are not expected to be Gaussian, and with n~5 seeds per arm there is no
realistic way to test normality anyway); (b) it is the standard choice for
comparing two independent small samples on an ordinal/interval scale without
distributional assumptions; (c) `evodevo/analysis/statistics.py::mann_whitney_u`
already implements it (U statistic + a normal-approximation two-sided
p-value) and is reused here rather than reimplemented -- see "scipy" below
for why the normal approximation, not an exact test, is used.

**scipy note**: scipy is confirmed NOT a dependency of this project
(`requirements.txt` has no `scipy` entry, and `python -c "import scipy"`
fails in this project's environment). `evodevo/analysis/statistics.py`'s own
module docstring already documents this constraint and its consequence: the
Mann-Whitney p-value is a normal approximation, "accurate only for larger
samples -- treat the p-value as a rough indicator at n~5". This protocol
inherits that constraint rather than introducing a new scipy dependency
purely to get an exact small-sample p-value.

## 2. Effect size: Vargha-Delaney A12

**Choice**: reported alongside every p-value, never a p-value alone.
`A12 = P(a > b) + 0.5*P(a == b)`, computed as `(cliffs_delta(a, b) + 1) / 2`
(`evodevo/analysis/stats_protocol.py::vargha_delaney_a12`), a linear
reparametrisation of the existing `cliffs_delta` effect size already in
`evodevo/analysis/statistics.py`.

**Justification**: a p-value says whether an effect is *detectable* at this
sample size; it says nothing about its *size*. With `n~5` seeds/arm (see
&sect;5), even a real, practically meaningful difference in final fitness may
not reach `p<0.05`, and conversely a p-value near the threshold conveys
almost nothing about magnitude. A12 has a direct, reportable probabilistic
interpretation ("probability a random seed of arm A outperforms a random
seed of arm B") that is meaningful and interpretable even when the p-value
is inconclusive -- exactly the situation this project's low seed count
makes likely. A12=0.5 is no effect; A12 close to 0 or 1 is a large effect,
independent of whether Holm-Bonferroni (&sect;3) marks it significant.

## 3. Multiple-comparison correction: Holm-Bonferroni

**Choice**: Holm-Bonferroni step-down correction
(`evodevo/analysis/stats_protocol.py::holm_bonferroni`), applied once across
the WHOLE planned comparison family for a given experimental matrix, not
per-task or per-comparison-kind separately.

**The planned comparison family, computed explicitly (not estimated)**: only
comparisons with a single, clean scientific interpretation are counted --
not every unordered pair of `(arm, strategy)` cells (`evodevo/analysis/
stats_protocol.py::enumerate_planned_comparisons` enumerates exactly this,
generically over arbitrary arms/strategies/tasks):

1. every cross-arm pair under the primary strategy (`ea`) -- the main
   scientific question, how the arms compare under real search;
2. for every arm, its `ea` performance vs. its own `random_search`
   performance -- the prior-vs-search decomposition (P7's purpose:
   separating what an encoding's prior achieves for free from what the EA's
   search adds).

Explicitly excluded: cross-arm-and-cross-strategy comparisons (e.g. `direct`
under `ea` vs. `rule_based` under `random_search`), which mix two different
axes of variation into one number with no single clean interpretation.

**`cross_arm` vs. `ablation_within_arm`, and why the distinction matters**:
within family 1 above, `enumerate_planned_comparisons` further splits
cross-arm-family pairs into two `kind` labels, derived generically from
whether the two arms share a base encoder registry entry (never by hardcoding
an encoder's name): `cross_arm` when they do not (e.g. `direct` vs. `nca` --
a REPRESENTATION-CHOICE question, "which encoding is better"), and
`ablation_within_arm` when they do (e.g. `rule_based` vs.
`rule_based_monotone` -- an ABLATION question, "does this one changed knob
on the SAME encoding affect task performance"). These are different
scientific questions with different implications for interpretation: a
significant `cross_arm` result supports choosing one encoding over another;
a significant `ablation_within_arm` result speaks to whether a specific
design choice (here, monotone's termination guarantee) costs or preserves
task performance within one encoding, and says nothing about how that
encoding compares to the others. Labelling both identically in the
significance table (Table 4, `docs/paper-manifest.md`) would read as if
they answered the same kind of question; they do not. The mechanism
generalises automatically to any future encoder that gains its own ablation
variant -- no special-casing is required.

**For the CURRENT experimental matrix** -- 4 arms (`direct`, `nca`,
`rule_based`, `rule_based_monotone`, per `docs/decisions.md`'s co-primary
decision and its Part-1 registration fix), 2 strategies (`ea`,
`random_search`), 1 task (`Walker-v0`):

- cross-arm pairs under `ea`: `C(4, 2) = 6`
- arm-vs-own-random_search: `4`
- **total, `m = 10`**

This is a genuine function of the matrix shape, not a fixed constant: adding
a second task doubles it to `m = 20` with zero code change (each task
contributes its own 6 + 4 = 10), adding a fifth arm would raise the
cross-arm term to `C(5,2) = 10` for `14` per task, etc.
`enumerate_planned_comparisons` computes this from `(arms, strategies,
tasks)` directly -- the final report must call it with the actual matrix in
effect at that time and cite its returned count, not a hand-typed number.

**Why Holm-Bonferroni specifically (vs. plain Bonferroni or no correction)**:
running `m=10` (or more, as tasks are added) independent hypothesis tests
without correction inflates the family-wise false-positive rate well above
the nominal `alpha` (at `m=10`, independent tests, the uncorrected
probability of at least one false positive is `1-(1-0.05)^10 ~ 40%`). Plain
Bonferroni (`alpha/m` per test) controls this but is more conservative than
necessary. Holm-Bonferroni controls the same family-wise error rate exactly
as strictly as Bonferroni while being uniformly more powerful (it only
applies the harshest correction to the single smallest p-value, progressively
less to each next-smallest) -- there is no reason to accept Bonferroni's
extra conservatism when Holm dominates it. `results/STATS_PROTOCOL_VALIDATION.md`
&sect;1 validates this behaviour directly under a simulated null.

## 4. Convergence-curve reporting convention: median + IQR

**Choice**: median line + IQR (25th/75th percentile) band across seeds,
computed by `evodevo/analysis/stats_protocol.py::median_iqr_curve` and used
by `evodevo/analysis/budget_curves.py`'s two-panel figure (evals_spent /
evals_valid, per `docs/decisions.md`'s dual-counter framing) -- **not**
mean&plusmn;SD.

**Correction to this document's own premise**: this convention was assumed
to already be `budget_curves.py`'s behaviour when this task was scoped; it
was not. As of the commit prior to this one, `budget_curves.py::_plot_panel`
computed a MEAN line with a MIN/MAX envelope across seeds. This document
locks median+IQR as the convention going forward, and
`budget_curves.py::_plot_panel` has been updated in this same change to
delegate to `median_iqr_curve` so the one existing curve-plotting code path
matches this document from the moment it is committed, rather than
diverging from it until a future, separate fix.

**Justification for median+IQR over mean&plusmn;SD or mean/min-max**: at
`n~5` seeds, a single outlier seed (e.g. one that never finds a valid genome
and stays sentinel-dominated, or one lucky early spike) has an outsized
effect on both a mean and a min/max envelope -- either can make a curve look
systematically better or worse than what most seeds actually show. The
median is robust to that single seed by construction, and the IQR band
communicates the middle 50% of seeds' spread without being stretched to an
outlier's extreme.

## 5. Seed-count floor: 5, a pragmatic minimum, not power-derived

**Floor in use**: 5 seeds per arm/strategy/task cell, carried over from the
P4/mutation-calibration and N-series decisions already in `docs/decisions.md`
(and reflected in `evodevo/experiments/configs/benchmark.yaml`'s
`seeds: [0, 1, 2, 3, 4]`).

**Is this sufficient for the planned tests at a stated power/alpha? Stated
plainly: no formal power analysis was performed, and one is not feasible
given this project's compute budget** -- a real per-seed PPO run
(`ppo_timesteps=5000`, `eval_budget=300`) costs on the order of tens of
minutes per seed (`docs/decisions.md`, "Direct-encoding real-PPO baseline
budget" row); running enough extra seeds to power a formal a-priori
calculation (typically dozens per arm for a two-sample Mann-Whitney U at
conventional 80% power against a medium effect size) is not achievable in
this project's environment. 5 is disclosed here as a pragmatic minimum
inherited from earlier compute-budget decisions, not derived for this
specific hypothesis test.

**What this means concretely, so results are not over-read**: at
`n_a=n_b=5`, the smallest possible two-sided Mann-Whitney U p-value is
`2 / C(10,5) = 2/252 ~ 0.0079` -- achievable ONLY when every value in one
group exceeds every value in the other (complete separation, `A12` at its
extreme of 0 or 1). Any result short of near-total separation between two
arms' 5-seed samples will not reach conventional significance, let alone
survive Holm-Bonferroni correction across `m=10` comparisons. The test is
therefore well-powered only for LARGE effects; a non-significant result at
this seed count must be reported as "no detectable difference at this
sample size", never as evidence of equivalence (this mirrors
`evodevo/analysis/statistics.py`'s own module-docstring warning, reused
verbatim in spirit here). A12 (&sect;2) is reported precisely so a real but
statistically-undetectable-at-n=5 effect is still visible in the report.

## 6. Sentinel-fitness handling: included at the sentinel value

**Choice**: an invalid genome's sentinel fitness (`sentinel_fitness=-10.0`,
per `docs/decisions.md`'s "Invalid-genome policy" row) is INCLUDED, at its
sentinel value, in every statistical comparison in this protocol -- never
excluded/filtered, and never imputed to some other value.

**Justification, and consistency with the P6 dual-counter framing**: the
Invalid-genome policy is explicit that a sentinel-scored genome consumes
exactly one evaluation slot and is never resampled or free
(`docs/decisions.md` row 12). An arm/seed whose final best-so-far fitness is
still the sentinel value (i.e. it never found a single valid genome within
its budget) is a real, reportable experimental outcome for that arm under
that budget -- excluding it would silently discard exactly the failure mode
the sentinel policy exists to make visible, and would bias the comparison in
favour of an arm/strategy that frequently fails to find anything valid.
`evals_valid_cumulative` (the P6 "real search attempts" counter, already the
x-axis of `budget_curves.py`'s panel (b)) is reported ALONGSIDE the fitness
comparison as a companion diagnostic -- it explains *why* a seed's final
fitness is sentinel-dominated, but it is never used to filter that seed out
of the statistical test itself. `stats_protocol.py` enforces this by
convention rather than by code: it consumes whatever per-seed final-fitness
array it is given (the standard array coming from
`evodevo.analysis.aggregate.per_seed_finals`, which already includes every
seed, sentinel-dominated or not) and performs no filtering of its own.
