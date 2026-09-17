"""The single, generic implementation of this project's LOCKED statistical
protocol (see docs/analysis-protocol.md for the full justification of every
choice made here -- this module only implements it).

Covers, generically over an arbitrary set of arms/strategies/tasks (never
hardcoded to three arms or to Walker-v0):
  - the primary pairwise test (Mann-Whitney U, via evodevo.analysis.statistics)
  - the effect size reported alongside every p-value (Vargha-Delaney A12)
  - the Holm-Bonferroni multiple-comparison correction, applied over the
    ACTUAL planned comparison family for a given (arms, strategies, tasks)
    matrix -- see enumerate_planned_comparisons()
  - the convergence-curve summary convention (median + IQR across seeds)

Sentinel-fitness handling (docs/analysis-protocol.md, and consistent with
docs/decisions.md's "Invalid-genome policy" row): this module does not filter
or special-case sentinel values itself -- it consumes whatever per-seed final
-fitness arrays it is given (e.g. from evodevo.analysis.aggregate.per_seed_finals,
which already includes an all-invalid seed's sentinel-dominated value as a
real data point). Callers must NOT pre-filter sentinel-dominated seeds out
before passing data in here.

This module never imports scipy (not a project dependency -- see
evodevo.analysis.statistics's own module docstring); every implementation
here is numpy-only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evodevo.analysis.statistics import cliffs_delta, mann_whitney_u

DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------


def vargha_delaney_a12(a: Sequence[float], b: Sequence[float]) -> float:
    """Vargha-Delaney A12: the probability a randomly drawn value from `a`
    exceeds a randomly drawn value from `b`, plus half the probability of a
    tie -- P(a>b) + 0.5*P(a==b). In [0, 1]; 0.5 means fully overlapping
    distributions, 1.0/0.0 means `a`/`b` is stochastically always larger.
    A linear reparametrisation of Cliff's delta (A12 = (delta + 1) / 2),
    reported because it has a direct probabilistic reading ("probability of
    superiority") that delta's [-1, 1] scale does not."""
    return (cliffs_delta(a, b) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


@dataclass
class HolmResult:
    p_raw: List[float]
    p_holm: List[float]
    reject: List[bool]
    alpha: float
    m: int


def holm_bonferroni(p_values: Sequence[float], alpha: float = DEFAULT_ALPHA) -> HolmResult:
    """Holm-Bonferroni step-down correction over a family of `m` p-values.

    Returns, per ORIGINAL input position, the Holm-adjusted p-value and
    whether it is rejected at `alpha`. Implemented via the standard
    "adjusted p-value" formulation (matches R's stats::p.adjust(method=
    "holm")): sort ascending, adjusted_i = max(adjusted_1..i, p_i * (m - rank)),
    clipped to 1.0 -- comparing the adjusted p-value directly against alpha
    reproduces the sequential step-down rejection decision exactly, while
    also giving a reportable per-comparison adjusted p-value (a plain
    Bonferroni p*m bound would be more conservative and is NOT what this
    project uses)."""
    p_values = list(p_values)
    m = len(p_values)
    if m == 0:
        return HolmResult(p_raw=[], p_holm=[], reject=[], alpha=alpha, m=0)

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        candidate = p_values[idx] * factor
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)

    reject = [adjusted[i] <= alpha for i in range(m)]
    return HolmResult(p_raw=p_values, p_holm=adjusted, reject=reject, alpha=alpha, m=m)


# ---------------------------------------------------------------------------
# Convergence-curve summary
# ---------------------------------------------------------------------------


def median_iqr_curve(values: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The locked convergence-curve convention: median + IQR (25th/75th
    percentile) reduced along `axis` (the seed axis by default, the last
    one). Returns (median, q25, q75), each with `axis` removed. Uses
    nanmedian/nanpercentile so a seed with missing values at some x-position
    (e.g. before forward-filling a pivoted, ragged per-seed curve) does not
    poison the whole reduction."""
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=axis)
    q25 = np.nanpercentile(values, 25, axis=axis)
    q75 = np.nanpercentile(values, 75, axis=axis)
    return median, q25, q75


# ---------------------------------------------------------------------------
# Planned-comparison enumeration (generic over arms/strategies/tasks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Cell:
    """One (arm, strategy, task) run-matrix cell -- the atomic unit
    stats_protocol compares. `arm` is an arm_id (see
    evodevo.experiments.benchmark.ArmSpec.arm_id), not necessarily a bare
    registry encoder name."""

    arm: str
    strategy: str
    task: str


#: Comparison kinds. "cross_arm": two arms with DIFFERENT base encoders --
#: a representation-choice question. "ablation_within_arm": two arms that
#: SHARE a base encoder (e.g. rule_based / rule_based_monotone) -- an
#: ablation question (does this one changed knob affect task performance?),
#: not a representation-choice one; conflating the two would read
#: confusingly in a significance table. "arm_vs_own_random_search": the
#: prior-vs-search decomposition, unrelated to either of the above.
CROSS_ARM = "cross_arm"
ABLATION_WITHIN_ARM = "ablation_within_arm"
ARM_VS_OWN_RANDOM_SEARCH = "arm_vs_own_random_search"


@dataclass
class ComparisonSpec:
    cell_a: Cell
    cell_b: Cell
    kind: str


def enumerate_planned_comparisons(
    arms: Sequence[str],
    strategies: Sequence[str],
    tasks: Sequence[str],
    primary_strategy: str = "ea",
    arm_base_encoder: Optional[Dict[str, str]] = None,
) -> List[ComparisonSpec]:
    """The hypothesis-scoped comparison family LOCKED by docs/analysis-protocol.md:
    only comparisons with a single, clean scientific interpretation, never
    every unordered pair of (arm, strategy) cells (which would include
    interpretation-free mixtures like "arm A under EA vs. arm B under
    random_search").

    Per task, for every task in `tasks`:
      1. every cross-arm pair under `primary_strategy` (the main scientific
         question: how do the arms compare under real search) -- C(len(arms), 2).
         Labelled CROSS_ARM if the two arms have different base encoders, or
         ABLATION_WITHIN_ARM if they share one (e.g. rule_based vs.
         rule_based_monotone) -- an ablation question, not a
         representation-choice one, and reported as a visibly distinct
         `kind` so a significance table never conflates the two.
      2. for every arm, that arm's `primary_strategy` vs. each OTHER strategy
         in `strategies` (the prior-vs-search decomposition: how much of an
         arm's performance is its encoding's prior vs. the EA's search
         contribution) -- len(arms) * (len(strategies) - 1).

    `arm_base_encoder`: {arm_id: base_encoder_registry_name}, e.g. from
    {a.arm_id: a.encoder for a in bconfig.resolved_arms()}
    (evodevo.experiments.benchmark.ArmSpec) -- this is how the
    cross_arm/ablation_within_arm distinction is DERIVED, generically, from
    whether two arms share a base encoder, rather than special-casing any
    encoder by name; a future encoder gaining its own ablation variant is
    handled automatically. Defaults to `{arm: arm for arm in arms}` (every
    arm is its own base encoder, i.e. every pair is CROSS_ARM) for backward
    compatibility with callers that don't have arm/encoder metadata.

    Purely a function of (arms, strategies, tasks): adding a second task, a
    fifth arm, or a third strategy changes the returned family size with NO
    code change here -- this is what "not hardcoded to three arms or Walker"
    means in practice. For the CURRENT matrix (4 arms, {ea, random_search},
    1 task): 6 cross-arm-or-ablation-within-arm (5 cross_arm + 1
    ablation_within_arm, rule_based vs rule_based_monotone) + 4
    arm-vs-own-random_search = 10 comparisons."""
    if arm_base_encoder is None:
        arm_base_encoder = {arm: arm for arm in arms}
    other_strategies = [s for s in strategies if s != primary_strategy]
    comparisons: List[ComparisonSpec] = []
    for task in tasks:
        for arm_a, arm_b in itertools.combinations(arms, 2):
            same_base_encoder = arm_base_encoder[arm_a] == arm_base_encoder[arm_b]
            comparisons.append(
                ComparisonSpec(
                    cell_a=Cell(arm_a, primary_strategy, task),
                    cell_b=Cell(arm_b, primary_strategy, task),
                    kind=ABLATION_WITHIN_ARM if same_base_encoder else CROSS_ARM,
                )
            )
        for arm in arms:
            for strategy in other_strategies:
                comparisons.append(
                    ComparisonSpec(
                        cell_a=Cell(arm, primary_strategy, task),
                        cell_b=Cell(arm, strategy, task),
                        kind=ARM_VS_OWN_RANDOM_SEARCH,
                    )
                )
    return comparisons


# ---------------------------------------------------------------------------
# Top-level protocol runner
# ---------------------------------------------------------------------------


@dataclass
class PairwiseResult:
    cell_a: Cell
    cell_b: Cell
    kind: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    a12: float
    mann_whitney_u: float
    p_raw: float
    p_holm: float
    significant_holm: bool


@dataclass
class ProtocolResult:
    comparisons: List[PairwiseResult]
    holm: HolmResult
    alpha: float


def run_protocol(
    fitness_by_cell: Dict[Tuple[str, str, str], Sequence[float]],
    arms: Sequence[str],
    strategies: Sequence[str],
    tasks: Sequence[str],
    primary_strategy: str = "ea",
    alpha: float = DEFAULT_ALPHA,
    arm_base_encoder: Optional[Dict[str, str]] = None,
) -> ProtocolResult:
    """`fitness_by_cell`: {(arm, strategy, task): [per-seed final fitness]},
    e.g. built from evodevo.analysis.aggregate.per_seed_finals()'s output
    re-keyed by (arm_id, strategy, task) -- the standard shared "final
    fitness per seed" definition every consumer of this module should use,
    so this test, evodevo.analysis.report, and budget_curves.py never
    silently disagree on what "final fitness" means.

    `arm_base_encoder` is passed straight through to
    enumerate_planned_comparisons() -- see its docstring for the
    cross_arm/ablation_within_arm distinction it derives.

    Enumerates the planned comparison family via
    enumerate_planned_comparisons(), runs Mann-Whitney U + A12 on each, then
    applies ONE Holm-Bonferroni correction across the whole family (not
    per-task or per-kind separately -- the family is the unit of correction)."""
    plan = enumerate_planned_comparisons(
        arms, strategies, tasks, primary_strategy=primary_strategy, arm_base_encoder=arm_base_encoder
    )

    raw_results = []
    for spec in plan:
        key_a = (spec.cell_a.arm, spec.cell_a.strategy, spec.cell_a.task)
        key_b = (spec.cell_b.arm, spec.cell_b.strategy, spec.cell_b.task)
        values_a = np.asarray(fitness_by_cell[key_a], dtype=float)
        values_b = np.asarray(fitness_by_cell[key_b], dtype=float)

        a12 = vargha_delaney_a12(values_a, values_b)
        u, p_raw = mann_whitney_u(values_a, values_b)

        raw_results.append(
            dict(
                cell_a=spec.cell_a,
                cell_b=spec.cell_b,
                kind=spec.kind,
                n_a=len(values_a),
                n_b=len(values_b),
                mean_a=float(np.mean(values_a)),
                mean_b=float(np.mean(values_b)),
                a12=a12,
                mann_whitney_u=u,
                p_raw=p_raw,
            )
        )

    holm = holm_bonferroni([r["p_raw"] for r in raw_results], alpha=alpha)

    comparisons = [
        PairwiseResult(
            cell_a=r["cell_a"],
            cell_b=r["cell_b"],
            kind=r["kind"],
            n_a=r["n_a"],
            n_b=r["n_b"],
            mean_a=r["mean_a"],
            mean_b=r["mean_b"],
            a12=r["a12"],
            mann_whitney_u=r["mann_whitney_u"],
            p_raw=r["p_raw"],
            p_holm=holm.p_holm[i],
            significant_holm=holm.reject[i],
        )
        for i, r in enumerate(raw_results)
    ]

    return ProtocolResult(comparisons=comparisons, holm=holm, alpha=alpha)
