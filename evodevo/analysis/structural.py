"""Fitness-blind structural measurement substrate for the genotype-phenotype
map of each encoding arm. Metrics 8-10 (readout_margin, uniform_body_fraction,
hidden_utilisation) instrument NCA's ARGMAX READOUT itself -- how confident
each cell's material decision is, how often development collapses to a
single-material body, and whether the H hidden channels the readout discards
actually carry information -- see results/nca/READOUT_STUDY.md. The argmax
readout rule itself is NOT changed anywhere in this module.

NO PPO, NO task, NO reward, NO EA anywhere in this module -- every
subsequent decision in this workstream depends on these seven metrics being
measurable independently of anything downstream of decode(). This is
enforced mechanically by tests/analysis/test_structural_no_ppo_imports.py,
which inspects this module's own import statements (not just "it happens
not to be slow" -- an actual import-graph assertion).

Every metric function has the signature
    metric(encoder, factors, n_samples, rng, ...) -> a result dataclass
and ALSO writes one JSONL record per sampled genome to
results/structural/<arm_name>/<metric_name>.jsonl -- raw enough to re-derive
every aggregate number, and to re-plot, without re-sampling. `factors` is
accepted by every metric (even ones whose behaviour does not depend on it)
so the signature is uniform and each JSONL run can be attributed to the
ExperimentFactors it was measured under.

INSTRUMENTATION NOTE: several metrics need per-STEP internal state (the
NCA's continuous cell state; the rule-based interpreter's intermediate
grids) that the Encoder contract does not expose publicly (decode() and
develop_trajectory() only ever return finished/argmax bodies). Rather than
widen the Encoder contract for measurement-only needs, this module reaches
into NCAEncoder/RuleBasedEncoder's own internals (`_unpack`, `_step`,
`_initial_state`, `_initial_condition`) and reuses
rule_based._rules.interpreter.develop_rules(..., initial_grid=...) to
resume a discrete grid one iteration at a time. This is deliberate,
documented coupling for analysis tooling, not a change to either encoder's
public behaviour.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.core.materials import N_MATERIALS
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.encoders.rule_based._rules.interpreter import develop_rules
from evodevo.morphology.metrics import body_hash
from evodevo.morphology.validate import validate_body

T_MAX_DEFAULT = 40
RESULTS_ROOT = "results/structural"


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _jsonl_path(arm_name: str, metric_name: str, output_root: str = RESULTS_ROOT) -> str:
    directory = os.path.join(output_root, arm_name)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{metric_name}.jsonl")


def _write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _categorize_reasons(reasons: List[str]) -> List[str]:
    """Substrings of validate_body() failure reasons -> a stable category
    label, matching the categorization already used by
    evodevo/experiments/inspect_encoder.py (kept independent here since that
    module is fine to import from but this file intentionally has as few
    dependencies as possible)."""
    categories = []
    for reason in reasons:
        if "min_voxels" in reason:
            categories.append("too_few_voxels")
        elif "connected component" in reason:
            categories.append("disconnected")
        elif "no actuator" in reason or "actuator" in reason:
            categories.append("no_actuator")
        elif "0..4" in reason:
            categories.append("invalid_material_values")
        elif "shape" in reason:
            categories.append("wrong_shape")
        elif "dtype" in reason:
            categories.append("non_integer_dtype")
        else:
            categories.append("other")
    return categories


def _is_degenerate(body: np.ndarray) -> bool:
    """A body is degenerate if every cell holds the same material (all-empty
    or a uniform non-empty slab) -- structurally trivial regardless of
    validity."""
    return bool(np.unique(body).size == 1)


def _nca_readout(state: np.ndarray) -> np.ndarray:
    return np.argmax(state[:, :, :N_MATERIALS], axis=-1).astype(int)


def _nca_trajectory_from_state(encoder: NCAEncoder, genome, state0: np.ndarray, t_max: int):
    """Full (t=0..t_max) readout trajectory AND raw state trajectory,
    starting from an arbitrary state0 (not necessarily encoder's own fixed
    initial state) -- lets light_cone/perturbation_robustness resume from a
    perturbed/damaged mid-trajectory state using the genome's own weights."""
    w1, b1, w2, b2 = encoder._unpack(genome)
    state = state0.copy()
    bodies = [_nca_readout(state)]
    states = [state.copy()]
    magnitudes = []
    for _ in range(t_max):
        state = encoder._step(state, w1, b1, w2, b2)
        bodies.append(_nca_readout(state))
        states.append(state.copy())
        magnitudes.append(float(np.max(np.abs(state))))
    return bodies, magnitudes, states


def _rule_based_trajectory_from_grid(encoder: RuleBasedEncoder, genome, grid0: np.ndarray, t_max: int):
    """Full (t=0..t_max) grid trajectory, one interpreter iteration at a
    time (develop_rules(..., iterations=1, initial_grid=...) is exactly
    equivalent to a single synchronous update step -- see interpreter.py),
    so this is O(t_max) rather than develop_trajectory()'s O(t_max^2)
    from-scratch-per-t approach, and lets us resume from an arbitrary
    (possibly perturbed/damaged) intermediate grid.

    BUG FIX (2026-09-16, found while verifying the monotone ablation):
    this previously called develop_rules() with NO interpreter kwargs at
    all, silently ignoring the encoder's own configured
    conflict_resolution/copy_neighbor_reads/boundary/neighbourhood/
    action_monotonicity and using the interpreter's raw defaults instead
    -- meaning settling_profile/light_cone/perturbation_robustness (every
    caller of _full_trajectory for rule_based) were measuring the WRONG
    dynamics whenever an encoder was constructed with any non-default
    interpreter setting. Fixed by threading every one of the encoder's
    own settings through explicitly, exactly as encoder.decode()/
    develop_trajectory() already do."""
    grid = np.array(grid0, dtype=int, copy=True)
    bodies = [grid.copy()]
    magnitudes = []
    for _ in range(t_max):
        next_grid = develop_rules(
            genome, shape=encoder.grid_shape, iterations=1, initial_grid=grid,
            conflict_resolution=encoder.conflict_resolution,
            copy_neighbor_reads=encoder.copy_neighbor_reads,
            boundary=encoder.boundary,
            neighbourhood=encoder.neighbourhood,
            action_monotonicity=encoder.action_monotonicity,
        )
        magnitudes.append(int(np.count_nonzero(next_grid != grid)))
        grid = next_grid
        bodies.append(grid.copy())
    return bodies, magnitudes


def _full_trajectory(encoder, genome, t_max: int) -> Tuple[List[np.ndarray], List[float]]:
    """Dispatches to the right per-arm trajectory function. Returns
    (bodies[0..t_max], magnitude_trace[1..t_max]); magnitude_trace[i] is the
    NCA's max-abs continuous state after step i+1, or the rule-based
    interpreter's changed-cell count at step i+1 -- comparable in ROLE (both
    answer "how much did development move at this step") but not in units,
    per the task's own instruction. A non-developmental encoder (direct)
    has no intermediate state: its trajectory is the single instantaneous
    decode()."""
    if isinstance(encoder, NCAEncoder):
        bodies, magnitudes, _ = _nca_trajectory_from_state(encoder, genome, encoder._initial_state, t_max)
        return bodies, magnitudes
    if isinstance(encoder, RuleBasedEncoder):
        grid0 = encoder._initial_condition.as_discrete_grid(encoder.grid_shape)
        return _rule_based_trajectory_from_grid(encoder, genome, grid0, t_max)
    return [encoder.decode(genome)], []


def _settle_info(bodies: List[np.ndarray]) -> Tuple[Optional[int], bool]:
    """(settle_step, settled). settle_step is the last step at which the
    decoded body changed (None if it was still changing at the final
    recorded transition, i.e. never observed to settle within the
    trajectory's length). A trajectory of length 1 (non-developmental) is
    trivially settled at step 0."""
    T = len(bodies) - 1
    if T == 0:
        return 0, True
    changed_at = [t + 1 for t in range(T) if not np.array_equal(bodies[t], bodies[t + 1])]
    if not changed_at:
        return 0, True
    last_change = changed_at[-1]
    if last_change == T:
        return None, False
    return last_change, True


# ---------------------------------------------------------------------------
# 1. validity_rate
# ---------------------------------------------------------------------------


@dataclass
class ValidityRateResult:
    n: int
    valid_rate: float
    failure_reason_counts: Dict[str, int]
    degenerate_count: int
    degenerate_fraction: float


def validity_rate(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    output_root: str = RESULTS_ROOT,
) -> ValidityRateResult:
    reason_counts: Dict[str, int] = {}
    valid_count = 0
    degenerate_count = 0
    records = []

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        body = encoder.decode(genome)
        result = validate_body(body, grid_shape=encoder.grid_shape)
        degenerate = _is_degenerate(body)

        if result.valid:
            valid_count += 1
        if degenerate:
            degenerate_count += 1
        for category in _categorize_reasons(result.reasons):
            reason_counts[category] = reason_counts.get(category, 0) + 1

        records.append({
            "index": i,
            "valid": result.valid,
            "failure_categories": _categorize_reasons(result.reasons),
            "degenerate": degenerate,
        })

    _write_jsonl(_jsonl_path(encoder.name, "validity_rate", output_root), records)

    return ValidityRateResult(
        n=n_samples,
        valid_rate=valid_count / n_samples,
        failure_reason_counts=reason_counts,
        degenerate_count=degenerate_count,
        degenerate_fraction=degenerate_count / n_samples,
    )


# ---------------------------------------------------------------------------
# 2. settling_profile
# ---------------------------------------------------------------------------


@dataclass
class SettlingProfileResult:
    n: int
    t_max: int
    settled_count: int
    settled_fraction: float
    never_settled_count: int
    never_settled_fraction: float
    mean_settle_step: Optional[float]


def settling_profile(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    t_max: int = T_MAX_DEFAULT, output_root: str = RESULTS_ROOT,
) -> SettlingProfileResult:
    records = []
    settle_steps: List[int] = []
    never_settled = 0

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        bodies, magnitudes = _full_trajectory(encoder, genome, t_max)
        settle_step, settled = _settle_info(bodies)

        if settled:
            settle_steps.append(settle_step)
        else:
            never_settled += 1

        records.append({
            "index": i,
            "settle_step": settle_step,  # None (-> JSON null) if never_settled
            "settled": settled,
            "state_magnitude_trace": magnitudes,
        })

    _write_jsonl(_jsonl_path(encoder.name, "settling_profile", output_root), records)

    settled_count = n_samples - never_settled
    return SettlingProfileResult(
        n=n_samples,
        t_max=t_max,
        settled_count=settled_count,
        settled_fraction=settled_count / n_samples,
        never_settled_count=never_settled,
        never_settled_fraction=never_settled / n_samples,
        mean_settle_step=(float(np.mean(settle_steps)) if settle_steps else None),
    )


# ---------------------------------------------------------------------------
# 3. light_cone
# ---------------------------------------------------------------------------


@dataclass
class LightConeResult:
    n: int
    t_max: int
    perturb_cell: Tuple[int, int]
    mean_first_diff_step_by_distance: Dict[int, Optional[float]]
    never_reached_fraction: float
    propagation_speed_cells_per_step: Optional[float]


def _perturbed_initial_state_nca(encoder: NCAEncoder, perturb_cell: Tuple[int, int]) -> np.ndarray:
    state = encoder._initial_state.copy()
    r, c = perturb_cell
    # Perturbation magnitude equal to init_gain (the variance-scaling gain
    # the genome's own weights are drawn at), added to every channel of the
    # perturbed cell -- an arbitrary but fixed and documented sign.
    state[r, c, :] += encoder.init_gain
    return state


def _perturbed_initial_grid_rule_based(encoder: RuleBasedEncoder, perturb_cell: Tuple[int, int]) -> np.ndarray:
    from evodevo.core.materials import RIGID, SOFT

    grid = encoder._initial_condition.as_discrete_grid(encoder.grid_shape).copy()
    r, c = perturb_cell
    # Flip the perturbed cell's discrete state to a different material --
    # SOFT unless it's already SOFT, in which case RIGID -- guaranteeing an
    # actual change at t=0 regardless of the named initial condition.
    grid[r, c] = SOFT if grid[r, c] != SOFT else RIGID
    return grid


def _perturbed_genome_direct(encoder, genome, perturb_cell: Tuple[int, int]) -> np.ndarray:
    rows, cols = encoder.grid_shape
    r, c = perturb_cell
    idx = r * cols + c
    perturbed = np.array(genome, copy=True)
    perturbed[idx] = (int(perturbed[idx]) + 1) % N_MATERIALS
    return perturbed


def _first_diff_steps(reference_bodies: List[np.ndarray], perturbed_bodies: List[np.ndarray]) -> np.ndarray:
    """Per-cell first step at which perturbed_bodies differs from
    reference_bodies; None (object dtype) if it never differs within the
    recorded trajectory."""
    rows, cols = reference_bodies[0].shape
    first_diff = np.full((rows, cols), None, dtype=object)
    for t in range(len(reference_bodies)):
        diff_mask = reference_bodies[t] != perturbed_bodies[t]
        newly = diff_mask & (first_diff == None)  # noqa: E711
        for r, c in zip(*np.where(newly)):
            first_diff[r, c] = t
    return first_diff


def light_cone(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    t_max: int = T_MAX_DEFAULT, output_root: str = RESULTS_ROOT,
) -> LightConeResult:
    rows, cols = encoder.grid_shape
    perturb_cell = (rows // 2, cols // 2)
    distances = {
        (r, c): abs(r - perturb_cell[0]) + abs(c - perturb_cell[1])
        for r in range(rows) for c in range(cols)
    }

    per_distance_steps: Dict[int, List[int]] = {}
    total_other_cells = 0
    never_reached_count = 0
    records = []

    for i in range(n_samples):
        genome = encoder.random_genome(rng)

        if isinstance(encoder, NCAEncoder):
            ref_bodies, _, _ = _nca_trajectory_from_state(encoder, genome, encoder._initial_state, t_max)
            perturbed_state0 = _perturbed_initial_state_nca(encoder, perturb_cell)
            pert_bodies, _, _ = _nca_trajectory_from_state(encoder, genome, perturbed_state0, t_max)
        elif isinstance(encoder, RuleBasedEncoder):
            ref_grid0 = encoder._initial_condition.as_discrete_grid(encoder.grid_shape)
            ref_bodies, _ = _rule_based_trajectory_from_grid(encoder, genome, ref_grid0, t_max)
            pert_grid0 = _perturbed_initial_grid_rule_based(encoder, perturb_cell)
            pert_bodies, _ = _rule_based_trajectory_from_grid(encoder, genome, pert_grid0, t_max)
        else:
            ref_bodies = [encoder.decode(genome)]
            pert_genome = _perturbed_genome_direct(encoder, genome, perturb_cell)
            pert_bodies = [encoder.decode(pert_genome)]

        first_diff = _first_diff_steps(ref_bodies, pert_bodies)

        cell_record = {}
        for (r, c), dist in distances.items():
            if (r, c) == perturb_cell:
                continue
            step = first_diff[r, c]
            total_other_cells += 1
            if step is None:
                never_reached_count += 1
            else:
                per_distance_steps.setdefault(dist, []).append(int(step))
            cell_record[f"{r},{c}"] = None if step is None else int(step)

        records.append({"index": i, "first_diff_step_by_cell": cell_record})

    _write_jsonl(_jsonl_path(encoder.name, "light_cone", output_root), records)

    mean_by_distance = {
        d: (float(np.mean(steps)) if steps else None)
        for d, steps in sorted(per_distance_steps.items())
    }

    all_distances, all_steps = [], []
    for d, steps in per_distance_steps.items():
        for s in steps:
            all_distances.append(d)
            all_steps.append(s)

    speed = None
    if len(all_steps) >= 2 and np.std(all_steps) > 0:
        # distance ~= speed * step -- ordinary least-squares slope of
        # distance regressed on step, over every (cell, genome) pair that
        # was ever reached. This is the "fitted effective propagation
        # speed" the task asks for; it is not claimed to be more than a
        # linear approximation of a process that need not be linear.
        slope, _intercept = np.polyfit(all_steps, all_distances, 1)
        speed = float(slope)

    return LightConeResult(
        n=n_samples,
        t_max=t_max,
        perturb_cell=perturb_cell,
        mean_first_diff_step_by_distance=mean_by_distance,
        never_reached_fraction=never_reached_count / total_other_cells,
        propagation_speed_cells_per_step=speed,
    )


# ---------------------------------------------------------------------------
# 4. phenotypic_displacement
# ---------------------------------------------------------------------------


@dataclass
class PhenotypicDisplacementResult:
    n: int
    mean_hamming_distance: float
    median_hamming_distance: float
    fraction_zero_displacement: float


def phenotypic_displacement(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    output_root: str = RESULTS_ROOT, max_attempts_multiplier: int = 200,
) -> PhenotypicDisplacementResult:
    """N valid random PARENTS (rejection sampling at the analysis level --
    unlike the Encoder contract's own random_genome(), which must never
    resample until valid, this measurement tool is explicitly asking "what
    does mutation do starting from a valid body", so it filters for one)."""
    records = []
    hammings: List[int] = []
    collected = 0
    attempts = 0
    max_attempts = n_samples * max_attempts_multiplier

    while collected < n_samples and attempts < max_attempts:
        attempts += 1
        parent = encoder.random_genome(rng)
        parent_body = encoder.decode(parent)
        if not validate_body(parent_body, grid_shape=encoder.grid_shape).valid:
            continue

        child = encoder.mutate(parent, rng)
        child_body = encoder.decode(child)
        hamming = int(np.count_nonzero(parent_body != child_body))

        hammings.append(hamming)
        records.append({"index": collected, "hamming_distance": hamming})
        collected += 1

    if collected < n_samples:
        raise RuntimeError(
            f"{encoder.name}: only found {collected}/{n_samples} valid random parents "
            f"within {max_attempts} attempts"
        )

    _write_jsonl(_jsonl_path(encoder.name, "phenotypic_displacement", output_root), records)

    hammings_arr = np.array(hammings)
    return PhenotypicDisplacementResult(
        n=n_samples,
        mean_hamming_distance=float(hammings_arr.mean()),
        median_hamming_distance=float(np.median(hammings_arr)),
        fraction_zero_displacement=float(np.mean(hammings_arr == 0)),
    )


# ---------------------------------------------------------------------------
# 5. redundancy
# ---------------------------------------------------------------------------


@dataclass
class RedundancyResult:
    n: int
    unique_phenotype_count: int
    genomes_per_unique_phenotype: float
    largest_neutral_class_size: int
    largest_neutral_class_fraction: float


def redundancy(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    output_root: str = RESULTS_ROOT,
) -> RedundancyResult:
    counts: Dict[str, int] = {}
    records = []

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        body = encoder.decode(genome)
        h = body_hash(body)
        counts[h] = counts.get(h, 0) + 1
        records.append({"index": i, "body_hash": h})

    _write_jsonl(_jsonl_path(encoder.name, "redundancy", output_root), records)

    unique_count = len(counts)
    largest = max(counts.values())
    return RedundancyResult(
        n=n_samples,
        unique_phenotype_count=unique_count,
        genomes_per_unique_phenotype=n_samples / unique_count,
        largest_neutral_class_size=largest,
        largest_neutral_class_fraction=largest / n_samples,
    )


# ---------------------------------------------------------------------------
# 6. diversity
# ---------------------------------------------------------------------------


@dataclass
class DiversityResult:
    n: int
    unique_valid_body_count: int
    unique_valid_body_fraction: float
    material_entropy_bits: float


def diversity(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    output_root: str = RESULTS_ROOT,
) -> DiversityResult:
    valid_hashes = set()
    material_counts = np.zeros(N_MATERIALS, dtype=np.int64)
    records = []

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        body = encoder.decode(genome)
        valid = validate_body(body, grid_shape=encoder.grid_shape).valid
        h = body_hash(body)
        if valid:
            valid_hashes.add(h)

        values, counts = np.unique(body, return_counts=True)
        for value, count in zip(values, counts):
            material_counts[int(value)] += int(count)

        records.append({"index": i, "valid": valid, "body_hash": h})

    _write_jsonl(_jsonl_path(encoder.name, "diversity", output_root), records)

    probs = material_counts / material_counts.sum()
    nonzero = probs[probs > 0]
    entropy_bits = float(-(nonzero * np.log2(nonzero)).sum())

    return DiversityResult(
        n=n_samples,
        unique_valid_body_count=len(valid_hashes),
        unique_valid_body_fraction=len(valid_hashes) / n_samples,
        material_entropy_bits=entropy_bits,
    )


# ---------------------------------------------------------------------------
# 7. perturbation_robustness
# ---------------------------------------------------------------------------


@dataclass
class PerturbationRobustnessResult:
    n: int
    damage_fraction: float
    mean_hamming_distance: float
    median_hamming_distance: float
    fraction_unaffected: float


def perturbation_robustness(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    damage_fraction: float = 0.2, output_root: str = RESULTS_ROOT,
) -> PerturbationRobustnessResult:
    """Damages a developmental run mid-trajectory (step k = floor(T/2)) by
    zeroing a random `damage_fraction` of cells (NCA: zero their whole
    continuous state vector; rule-based: flip them to a random material),
    then lets development continue to completion with the SAME genome, and
    reports the Hamming distance to the undamaged final body.

    ASYMMETRY FOR THE DIRECT ARM: DirectEncoder's genome IS its phenotype --
    there is no intermediate developmental state, and therefore no
    mid-trajectory step to damage. The only meaningful analogue is damaging
    the GENOME itself before its one-shot decode(). This is a deliberate,
    documented difference from the developmental arms (which are damaged
    partway through development and can, in principle, partially "regrow"
    away from the damage before the final readout is taken); direct has no
    such recovery mechanism by construction, since there is nothing after
    the damage but a single reshape."""
    records = []
    hammings: List[int] = []
    rows, cols = encoder.grid_shape
    n_cells = rows * cols
    n_damaged = max(1, int(round(damage_fraction * n_cells)))

    for i in range(n_samples):
        genome = encoder.random_genome(rng)

        if isinstance(encoder, NCAEncoder):
            T = encoder.dev_steps
            bodies, _, states = _nca_trajectory_from_state(encoder, genome, encoder._initial_state, T)
            undamaged_final = bodies[-1]
            k = T // 2
            w1, b1, w2, b2 = encoder._unpack(genome)
            state = states[k].copy()
            flat_idx = rng.choice(n_cells, size=n_damaged, replace=False)
            for idx in flat_idx:
                r, c = divmod(int(idx), cols)
                state[r, c, :] = 0.0
            for _ in range(T - k):
                state = encoder._step(state, w1, b1, w2, b2)
            damaged_final = _nca_readout(state)

        elif isinstance(encoder, RuleBasedEncoder):
            T = encoder.developmental_iterations
            grid0 = encoder._initial_condition.as_discrete_grid(encoder.grid_shape)
            bodies, _ = _rule_based_trajectory_from_grid(encoder, genome, grid0, T)
            undamaged_final = bodies[-1]
            k = T // 2
            grid = bodies[k].copy()
            flat_idx = rng.choice(n_cells, size=n_damaged, replace=False)
            for idx in flat_idx:
                r, c = divmod(int(idx), cols)
                grid[r, c] = int(rng.integers(0, N_MATERIALS))
            for _ in range(T - k):
                grid = develop_rules(
                    genome, shape=encoder.grid_shape, iterations=1, initial_grid=grid,
                    conflict_resolution=encoder.conflict_resolution,
                    copy_neighbor_reads=encoder.copy_neighbor_reads,
                    boundary=encoder.boundary,
                    neighbourhood=encoder.neighbourhood,
                    action_monotonicity=encoder.action_monotonicity,
                )
            damaged_final = grid

        else:
            undamaged_final = encoder.decode(genome)
            damaged_genome = np.array(genome, copy=True)
            flat_idx = rng.choice(n_cells, size=n_damaged, replace=False)
            for idx in flat_idx:
                damaged_genome[idx] = int(rng.integers(0, N_MATERIALS))
            damaged_final = encoder.decode(damaged_genome)

        hamming = int(np.count_nonzero(undamaged_final != damaged_final))
        hammings.append(hamming)
        records.append({"index": i, "hamming_distance": hamming})

    _write_jsonl(_jsonl_path(encoder.name, "perturbation_robustness", output_root), records)

    hammings_arr = np.array(hammings)
    return PerturbationRobustnessResult(
        n=n_samples,
        damage_fraction=damage_fraction,
        mean_hamming_distance=float(hammings_arr.mean()),
        median_hamming_distance=float(np.median(hammings_arr)),
        fraction_unaffected=float(np.mean(hammings_arr == 0)),
    )


# ---------------------------------------------------------------------------
# 8. readout_margin
# ---------------------------------------------------------------------------


@dataclass
class ReadoutMarginResult:
    n: int
    n_cells: int
    near_tie_threshold: float
    mean_margin: float
    median_margin: float
    fraction_cells_near_tie: float
    fraction_bodies_with_near_tie_cell: float


def readout_margin(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    near_tie_threshold: float = 0.01, output_root: str = RESULTS_ROOT,
) -> ReadoutMarginResult:
    """Per-cell (max logit - second-max logit) at t=T, over the 5 material
    channels NCA's argmax readout actually decides between. Requires an
    NCAEncoder: rule_based and direct both produce a discrete body directly,
    with no pre-argmax continuous logits to measure a margin over -- this
    metric is specifically about the confidence of NCA's argmax decision,
    not a generic property every arm's decode() exposes."""
    if not isinstance(encoder, NCAEncoder):
        raise TypeError(
            f"readout_margin requires an NCAEncoder (got {type(encoder).__name__}): only NCA exposes "
            "continuous pre-argmax material logits (state[:, :, :N_MATERIALS]) -- rule_based/direct "
            "decode() straight to a discrete body with no logits to take a margin over."
        )

    records = []
    all_margins: List[np.ndarray] = []
    bodies_with_tie = 0

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        _, _, states = _nca_trajectory_from_state(encoder, genome, encoder._initial_state, encoder.dev_steps)
        final_logits = states[-1][:, :, :N_MATERIALS]  # (rows, cols, N_MATERIALS)
        sorted_logits = np.sort(final_logits, axis=-1)
        margin = (sorted_logits[..., -1] - sorted_logits[..., -2]).flatten()  # (rows*cols,)

        all_margins.append(margin)
        has_tie = bool(np.any(margin < near_tie_threshold))
        if has_tie:
            bodies_with_tie += 1

        records.append({
            "index": i,
            "margins": margin.tolist(),
            "has_near_tie_cell": has_tie,
        })

    _write_jsonl(_jsonl_path(encoder.name, "readout_margin", output_root), records)

    margins_arr = np.concatenate(all_margins)
    return ReadoutMarginResult(
        n=n_samples,
        n_cells=int(margins_arr.size),
        near_tie_threshold=near_tie_threshold,
        mean_margin=float(margins_arr.mean()),
        median_margin=float(np.median(margins_arr)),
        fraction_cells_near_tie=float(np.mean(margins_arr < near_tie_threshold)),
        fraction_bodies_with_near_tie_cell=bodies_with_tie / n_samples,
    )


# ---------------------------------------------------------------------------
# 9. uniform_body_fraction
# ---------------------------------------------------------------------------


@dataclass
class UniformBodyFractionResult:
    n: int
    uniform_fraction: float
    uniform_count_by_material: Dict[int, int]
    uniform_fraction_by_material: Dict[int, float]


def uniform_body_fraction(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    output_root: str = RESULTS_ROOT,
) -> UniformBodyFractionResult:
    """Fraction of decoded bodies holding a single material across all 25
    cells (the same "uniform" test as validity_rate's degenerate_fraction /
    _is_degenerate, arm-agnostic), broken down by WHICH material -- directly
    quantifies the single-material collapse the P7 surrogate investigation
    surfaced for NCA."""
    counts_by_material: Dict[int, int] = {m: 0 for m in range(N_MATERIALS)}
    uniform_count = 0
    records = []

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        body = encoder.decode(genome)
        unique_vals = np.unique(body)
        is_uniform = bool(unique_vals.size == 1)
        material = int(unique_vals[0]) if is_uniform else None

        if is_uniform:
            uniform_count += 1
            counts_by_material[material] = counts_by_material.get(material, 0) + 1

        records.append({"index": i, "uniform": is_uniform, "material": material})

    _write_jsonl(_jsonl_path(encoder.name, "uniform_body_fraction", output_root), records)

    return UniformBodyFractionResult(
        n=n_samples,
        uniform_fraction=uniform_count / n_samples,
        uniform_count_by_material=counts_by_material,
        uniform_fraction_by_material={m: c / n_samples for m, c in counts_by_material.items()},
    )


# ---------------------------------------------------------------------------
# 10. hidden_utilisation
# ---------------------------------------------------------------------------


@dataclass
class HiddenUtilisationResult:
    n: int
    hidden_channels: int
    per_channel_spatial_variance: List[float]  # variance across the 25 cells at t=T, averaged over genomes
    per_channel_temporal_variance: List[float]  # variance across t=0..T per cell, averaged over cells/genomes
    per_channel_correlation_with_readout: List[float]  # Pearson r: channel@T/2 vs final material id, pooled


def hidden_utilisation(
    encoder, factors: ExperimentFactors, n_samples: int, rng: np.random.Generator,
    output_root: str = RESULTS_ROOT,
) -> HiddenUtilisationResult:
    """Whether NCA's H hidden channels -- carried through development but
    discarded by the argmax readout (only channels 0..N_MATERIALS-1 feed it)
    -- actually do anything: their spatial variance (across cells, at t=T),
    their temporal variance (across steps, per cell), and their correlation
    with the material the readout ultimately picks (measured at t=T/2, half
    a step of causal slack before the final readout, so a channel that
    merely mirrors the already-decided material at t=T is not what's being
    tested). Low variance on both axes means a channel is inert; low
    |correlation| means a channel varies but doesn't covary with the
    decision the readout actually makes.

    CAVEAT on the correlation number: material ids 0..N_MATERIALS-1 are a
    categorical label, not an ordinal scale, so a linear Pearson correlation
    against them is a crude, order-dependent indicator -- a large value is
    good evidence of a real relationship, but a near-zero value does not
    rule out a real, non-monotonic (e.g. one-material-vs-rest) relationship
    between a channel and the decision. Requires an NCAEncoder; H=0 returns
    an explicitly empty result (not an error) so it can be called uniformly
    across an H ablation sweep."""
    if not isinstance(encoder, NCAEncoder):
        raise TypeError(
            f"hidden_utilisation requires an NCAEncoder (got {type(encoder).__name__}): hidden channels "
            "are an NCA-specific concept (state channels beyond the N_MATERIALS the readout uses)."
        )

    H = encoder.hidden_channels
    if H == 0:
        _write_jsonl(_jsonl_path(encoder.name, "hidden_utilisation", output_root), [])
        return HiddenUtilisationResult(
            n=n_samples,
            hidden_channels=0,
            per_channel_spatial_variance=[],
            per_channel_temporal_variance=[],
            per_channel_correlation_with_readout=[],
        )

    records = []
    spatial_vars_per_genome: List[List[float]] = []  # [genome][channel]
    temporal_vars_per_genome: List[List[float]] = []
    mid_values_pooled = [[] for _ in range(H)]  # [channel] -> flat list over (genome, cell)
    final_material_pooled = []  # flat list over (genome, cell), shared across channels

    for i in range(n_samples):
        genome = encoder.random_genome(rng)
        T = encoder.dev_steps
        _, _, states = _nca_trajectory_from_state(encoder, genome, encoder._initial_state, T)
        mid_step = T // 2
        final_state = states[-1]
        mid_state = states[mid_step]
        final_material = _nca_readout(final_state).flatten()  # (rows*cols,)

        channel_stack = np.stack(states, axis=0)  # (T+1, rows, cols, channels)

        genome_spatial = []
        genome_temporal = []
        genome_mid_values = []
        for h in range(H):
            ch_idx = N_MATERIALS + h
            spatial_var = float(np.var(final_state[:, :, ch_idx]))
            temporal_var = float(np.mean(np.var(channel_stack[:, :, :, ch_idx], axis=0)))
            mid_values = mid_state[:, :, ch_idx].flatten()

            genome_spatial.append(spatial_var)
            genome_temporal.append(temporal_var)
            genome_mid_values.append(mid_values.tolist())
            mid_values_pooled[h].extend(mid_values.tolist())

        spatial_vars_per_genome.append(genome_spatial)
        temporal_vars_per_genome.append(genome_temporal)
        final_material_pooled.extend(final_material.tolist())

        records.append({
            "index": i,
            "spatial_variance_per_channel": genome_spatial,
            "temporal_variance_per_channel": genome_temporal,
            "mid_step_values_per_channel": genome_mid_values,
            "final_material": final_material.tolist(),
        })

    _write_jsonl(_jsonl_path(encoder.name, "hidden_utilisation", output_root), records)

    final_material_arr = np.array(final_material_pooled, dtype=float)
    correlations = []
    for h in range(H):
        channel_arr = np.array(mid_values_pooled[h], dtype=float)
        if np.std(channel_arr) == 0.0 or np.std(final_material_arr) == 0.0:
            # a constant channel (or constant readout) has undefined linear
            # correlation; 0 (no LINEAR relationship detectable) is the
            # honest report, not NaN.
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(channel_arr, final_material_arr)[0, 1]))

    spatial_arr = np.array(spatial_vars_per_genome)  # (n_samples, H)
    temporal_arr = np.array(temporal_vars_per_genome)  # (n_samples, H)

    return HiddenUtilisationResult(
        n=n_samples,
        hidden_channels=H,
        per_channel_spatial_variance=[float(v) for v in spatial_arr.mean(axis=0)],
        per_channel_temporal_variance=[float(v) for v in temporal_arr.mean(axis=0)],
        per_channel_correlation_with_readout=correlations,
    )
