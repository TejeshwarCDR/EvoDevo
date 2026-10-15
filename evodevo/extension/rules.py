"""Pre-registered decision rules for the campaign extension.

Registered in docs/campaign_prep/ext/PREREGISTRATION_EXT.md on 2026-10-15,
before any preflight measurement on the extension tasks existed. Every
function here is a pure function of preflight measurements: none of them
reads an arm-versus-arm comparison of evolutionary outcomes, so none of them
can bias the research question. The thresholds are module constants and are
not to be changed after preflight data exist (any change is a dated
"Deviation from pre-registration" row, never an edit here).

Rules
-----
D6-ext  PPO timestep budget per task, from learning curves (B4x).
D16-ext PPO seeds K per task, from repeated trainings (B5x).
GUARD-ext completion-time guard on the projected campaign duration.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field

from evodevo.preflight.stages.b4_learning_curves import SELECT_KNEE_RULE_TEXT, select_knee

# --- D6-ext constants -------------------------------------------------------
CURVE_MAX_TIMESTEPS = 2_000_000
CURVE_EVAL_INTERVAL = 100_000
QUALIFYING_FRACTION_OF_BEST = 0.10    # same as the Walker-v0 fallback rule
MIN_QUALIFYING_BODIES = 2
NON_LEARNING_BUDGET = 1_000_000       # = the Walker-v0 budget
BUDGET_FLOOR = 500_000                # D6's stated lower expectation
BUDGET_CAP = 1_500_000                # compute cap, declared before data
FALLBACK_CANDIDATES = (1_000_000, 1_200_000, 1_500_000, 2_000_000)
FALLBACK_FRACTION_OF_FINAL = 0.85     # same as the Walker-v0 fallback rule

# --- D16-ext constants ------------------------------------------------------
# Geometric mean of the two original within/between ratios (Walker-v0 0.165
# -> K=1, UpStepper-v0 0.666 -> K=2), so this rule reproduces both original
# D16 decisions.
NOISE_RATIO_THRESHOLD = 0.33
MAX_OF_2_REQUIRED_IMPROVEMENT = 0.5   # max-of-2 must at least halve the ratio
MIN_NOISE_BODIES = 3
MIN_SPREAD = 1e-9

# --- GUARD-ext constants ----------------------------------------------------
GUARD_THRESHOLD_H = 360.0             # 15 days: launch ~2026-10-22 -> ~2026-11-06
ESTIMATE_MIN_RELATIVE_ERROR = 0.10    # band half-width floor (see estimate rule)


def checkpoint_index(timesteps: int, interval: int = CURVE_EVAL_INTERVAL) -> int:
    """Curve point i was recorded at the first PPO step >= (i+1)*interval."""
    if timesteps % interval:
        raise ValueError(f"{timesteps} is not a multiple of the checkpoint interval {interval}")
    return timesteps // interval - 1


def fitness_at(curve: list, timesteps: int, interval: int = CURVE_EVAL_INTERVAL) -> float | None:
    index = checkpoint_index(timesteps, interval)
    return float(curve[index][1]) if 0 <= index < len(curve) else None


def final_fitness(curve: list) -> float | None:
    return float(curve[-1][1]) if curve else None


def round_to_interval(timesteps: float, interval: int = CURVE_EVAL_INTERVAL) -> int:
    """Curve timesteps sit a few PPO steps past each checkpoint (rollouts are
    collected in 2048-step batches); report the checkpoint itself."""
    return int(round(timesteps / interval) * interval)


def median_series(curves: list) -> list:
    """Per-checkpoint median across bodies, aligned by checkpoint index
    (identical to the original B4 stage's _median_series)."""
    if not curves:
        return []
    series = []
    for i in range(max(len(c) for c in curves)):
        points = [c[i] for c in curves if i < len(c)]
        series.append((statistics.median(p[0] for p in points), statistics.median(p[1] for p in points)))
    return series


@dataclass
class BudgetDecision:
    task: str
    ppo_timesteps: int
    branch: str                     # registered_knee | fallback | non_learning
    disclosed_floor: bool           # True when the value is a floor, not a converged knee
    clipped: str | None             # None | "floor" | "cap"
    qualifying_bodies: list
    details: dict = field(default_factory=dict)


def qualifying_bodies(bodies: dict) -> list:
    """Bodies whose 2M fitness is >= 10% of the best body's (best must be > 0)."""
    finals = {name: final_fitness(curve) for name, curve in bodies.items() if curve}
    finals = {name: value for name, value in finals.items() if value is not None}
    if not finals:
        return []
    best = max(finals.values())
    if best <= 0:
        return []
    return sorted(name for name, value in finals.items() if value >= QUALIFYING_FRACTION_OF_BEST * best)


def _complete(curve: list) -> bool:
    return len(curve) >= checkpoint_index(CURVE_MAX_TIMESTEPS) + 1


def decide_budget(task: str, bodies: dict) -> BudgetDecision:
    """D6-ext. ``bodies`` maps a body label to its curve [(timesteps, fitness), ...]
    recorded every 100k steps to 2M. Bodies whose training failed or stopped
    early are excluded before anything else is computed."""
    complete = {name: curve for name, curve in bodies.items() if curve and _complete(curve)}
    excluded = sorted(set(bodies) - set(complete))
    qualifying = qualifying_bodies(complete)
    details = {"excluded_incomplete_bodies": excluded,
               "final_fitness": {name: final_fitness(c) for name, c in complete.items()},
               "rule_text": SELECT_KNEE_RULE_TEXT}
    if len(qualifying) < MIN_QUALIFYING_BODIES:
        details["reason"] = (f"{len(qualifying)} qualifying bodies (< {MIN_QUALIFYING_BODIES}); "
                             "no learning signal to calibrate a budget")
        return BudgetDecision(task, NON_LEARNING_BUDGET, "non_learning", True, None, qualifying, details)

    series = median_series([complete[name] for name in qualifying])
    knee = select_knee(series)
    details["median_series"] = series
    details["knee"] = {key: knee[key] for key in ("knee_timesteps", "knee_found", "knee_window_index")}
    if knee["knee_found"]:
        raw = round_to_interval(knee["knee_timesteps"])
        value = min(max(raw, BUDGET_FLOOR), BUDGET_CAP)
        clipped = None if value == raw else ("floor" if value > raw else "cap")
        details["raw_knee_checkpoint"] = raw
        return BudgetDecision(task, value, "registered_knee", clipped == "cap", clipped, qualifying, details)

    # Fallback (the Walker-v0 rule, with the ordering condition restricted to
    # qualifying bodies: near-zero non-learners' order is noise).
    table = []
    chosen = None
    final_order = sorted(qualifying, key=lambda name: final_fitness(complete[name]), reverse=True)
    for candidate in FALLBACK_CANDIDATES:
        fractions = {name: fitness_at(complete[name], candidate) / final_fitness(complete[name])
                     for name in qualifying}
        order = sorted(qualifying, key=lambda name: fitness_at(complete[name], candidate), reverse=True)
        fraction_ok = all(value >= FALLBACK_FRACTION_OF_FINAL for value in fractions.values())
        order_ok = order == final_order
        table.append({"T": candidate, "fractions_of_2M": fractions, "order": order,
                      "fraction_ok": fraction_ok, "order_ok": order_ok})
        if chosen is None and fraction_ok and order_ok:
            chosen = candidate
    details["fallback_table"] = table
    details["final_order"] = final_order
    floor = chosen is None or chosen == CURVE_MAX_TIMESTEPS
    raw = chosen if chosen is not None else CURVE_MAX_TIMESTEPS
    value = min(raw, BUDGET_CAP)
    clipped = "cap" if value < raw else None
    details["fallback_choice"] = raw
    return BudgetDecision(task, value, "fallback", floor or clipped == "cap", clipped, qualifying, details)


@dataclass
class SeedsDecision:
    task: str
    ppo_seeds: int
    branch: str                     # noise_rule | insufficient_spread | insufficient_bodies
    details: dict = field(default_factory=dict)


def _pooled_sd(groups: list) -> float | None:
    numerator = denominator = 0.0
    for values in groups:
        if len(values) > 1:
            numerator += (len(values) - 1) * statistics.variance(values)
            denominator += len(values) - 1
    return (numerator / denominator) ** 0.5 if denominator else None


def decide_seeds(task: str, repeats: dict) -> SeedsDecision:
    """D16-ext. ``repeats`` maps a body label to the fitness values of its
    repeated, independently seeded trainings at the task's chosen budget
    (failed trainings already removed)."""
    usable = {name: list(values) for name, values in repeats.items() if len(values) >= 2}
    details = {"bodies_used": sorted(usable), "bodies_excluded": sorted(set(repeats) - set(usable))}
    if len(usable) < MIN_NOISE_BODIES:
        details["reason"] = f"{len(usable)} bodies with >= 2 completed repeats (< {MIN_NOISE_BODIES})"
        return SeedsDecision(task, 1, "insufficient_bodies", details)
    means = [statistics.mean(values) for values in usable.values()]
    between = statistics.stdev(means)
    details["between_body_sd"] = between
    if between < MIN_SPREAD:
        details["reason"] = "bodies are indistinguishable at this budget; repeat noise cannot change a ranking"
        return SeedsDecision(task, 1, "insufficient_spread", details)
    single = _pooled_sd(list(usable.values()))
    max2 = _pooled_sd([[max(a, b) for a, b in itertools.combinations(values, 2)]
                       for values in usable.values()])
    ratio_single = single / between
    ratio_max2 = max2 / between if max2 is not None else None
    use_two = (ratio_single > NOISE_RATIO_THRESHOLD and ratio_max2 is not None
               and ratio_max2 <= MAX_OF_2_REQUIRED_IMPROVEMENT * ratio_single)
    details.update(pooled_within_sd_single=single, pooled_within_sd_max_of_2=max2,
                   ratio_single=ratio_single, ratio_max_of_2=ratio_max2,
                   threshold=NOISE_RATIO_THRESHOLD, required_improvement=MAX_OF_2_REQUIRED_IMPROVEMENT)
    return SeedsDecision(task, 2 if use_two else 1, "noise_rule", details)


GUARD_TIMESTEP_CEILING = 1_200_000   # step (b) ceiling = the UpStepper-v0 budget


def project_phase_band(reference_makespan_h: float, busy_share: float, ratio: float, k: int,
                       relative_error: float) -> dict:
    """Projected phase duration (hours).

    The original campaign's realised Walker-v0 phase makespan is the
    reference: the pool's realised throughput, utilisation and scheduling
    tail are all inside it. It is scaled by the share of that phase's
    worker-busy time spent on the arms being run (the three core arms),
    by the measured per-training cost ratio (new task at its budget over
    Walker-v0 at 1M, same bodies, same batch) and by K. The band is
    +/- max(10%, |backtest error|).
    """
    if not 0 < busy_share <= 1 or ratio <= 0 or k < 1 or reference_makespan_h <= 0:
        raise ValueError("invalid projection inputs")
    mean = reference_makespan_h * busy_share * ratio * k
    half = max(ESTIMATE_MIN_RELATIVE_ERROR, abs(relative_error))
    return {"p5": mean * (1 - half), "mean": mean, "p95": mean * (1 + half)}


def apply_guard(task_order: list, decided: dict, project, threshold_h: float = GUARD_THRESHOLD_H):
    """GUARD-ext, applied mechanically to the p95 projected total.

    ``decided[task]`` = {"ppo_timesteps": T, "ppo_seeds": K} from D6-ext and
    D16-ext; ``project(task, T, K)`` returns the phase band in hours.
      (a) while over: K=2 -> K=1 on the latest task (D7 order) that has K=2;
      (b) while over: T -> 1.2M on the latest task with T > 1.2M (disclosed
          as a floor);
      (c) still over: nothing else is cut (no arm, seed, task, generation);
          the campaign proceeds and the later completion is reported.
    Returns (final {task: {ppo_timesteps, ppo_seeds}}, steps, total band, outcome).
    """
    current = {task: dict(decided[task]) for task in task_order}
    steps = []

    def total():
        band = {q: 0.0 for q in ("p5", "mean", "p95")}
        per_task = {}
        for task in task_order:
            phase = project(task, current[task]["ppo_timesteps"], current[task]["ppo_seeds"])
            per_task[task] = phase
            for q in band:
                band[q] += phase[q]
        return band, per_task

    def record(name):
        band, per_task = total()
        steps.append({"step": name, "design": {t: dict(v) for t, v in current.items()},
                      "phase_h": per_task, "total_h": band, "over": band["p95"] > threshold_h})
        return band

    band = record("none")
    for task in reversed(task_order):
        if band["p95"] <= threshold_h:
            break
        if current[task]["ppo_seeds"] > 1:
            current[task]["ppo_seeds"] = 1
            band = record(f"a_K1_{task}")
    for task in reversed(task_order):
        if band["p95"] <= threshold_h:
            break
        if current[task]["ppo_timesteps"] > GUARD_TIMESTEP_CEILING:
            current[task]["ppo_timesteps"] = GUARD_TIMESTEP_CEILING
            band = record(f"b_T1.2M_{task}")
    outcome = ("within_guard" if band["p95"] <= threshold_h else
               "over_guard_proceed_with_later_completion")
    return current, steps, band, outcome
