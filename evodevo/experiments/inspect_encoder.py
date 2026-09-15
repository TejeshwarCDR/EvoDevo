"""PPO-free characterization for ANY registered encoder: samples N random
genomes, reports validity rate with failure-reason breakdown, unique-body
count, degeneracy rate, and -- for encoders that expose a develop()
trajectory hook (currently NCA) -- developmental settling behaviour.

This generalizes what was originally NCA-specific (evodevo/experiments/
nca_sanity.py, kept as a thin deprecated shim over this module so existing
results remain reproducible without modification).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evodevo.core.factors import extract_factors
from evodevo.core.materials import N_MATERIALS
from evodevo.core.seeding import make_rng
from evodevo.encoders import get_encoder
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.morphology.metrics import body_hash
from evodevo.morphology.validate import validate_body

VALIDITY_GATE = 0.10
DEGENERACY_GATE = 0.50

# Substrings of validate_body() failure reasons -> a stable category label.
# Order matters: checked top-to-bottom, first match wins.
_REASON_CATEGORIES = [
    ("min_voxels", "too_few_voxels"),
    ("connected component", "disconnected"),
    ("no actuator", "no_actuator"),
    ("0..4", "invalid_material_values"),
    ("shape", "wrong_shape"),
    ("dtype", "non_integer_dtype"),
]


def _build_encoder(encoder_name: str, seed: int, kwargs: dict):
    """Every registered encoder now requires an explicit `factors`
    (ExperimentFactors) kwarg. This tool's callers historically passed flat
    per-arm hyperparameter overrides (e.g. {'dev_steps': 1}) straight
    through as encoder_kwargs -- extract_factors() pulls those factor-shaped
    keys out and builds the ExperimentFactors from them (defaulting to
    dev_steps=10/full_nonsymmetric, this module's historical NCA-sanity-check
    behaviour, for anything not explicitly overridden), so existing
    ablation-report call sites keep working unchanged."""
    factors, remaining_kwargs = extract_factors(kwargs, seed=seed)
    return get_encoder(encoder_name, factors=factors, **remaining_kwargs)


def _categorize_reason(reason: str) -> str:
    for substr, category in _REASON_CATEGORIES:
        if substr in reason:
            return category
    return "other"


def _validity_stats(bodies) -> dict:
    n = len(bodies)
    valid_count = 0
    reason_counts: dict = {}
    hashes = set()
    all_empty_count = 0
    uniform_nonempty_count = 0

    for body in bodies:
        result = validate_body(body)
        if result.valid:
            valid_count += 1
        for reason in result.reasons:
            category = _categorize_reason(reason)
            reason_counts[category] = reason_counts.get(category, 0) + 1

        hashes.add(body_hash(body))

        unique_vals = np.unique(body)
        if unique_vals.size == 1:
            if int(unique_vals[0]) == 0:
                all_empty_count += 1
            else:
                uniform_nonempty_count += 1

    return {
        "n": n,
        "valid_rate": valid_count / n,
        "invalid_reason_counts": reason_counts,
        "unique_body_count": len(hashes),
        "unique_body_fraction": len(hashes) / n,
        "all_empty_fraction": all_empty_count / n,
        "uniform_nonempty_fraction": uniform_nonempty_count / n,
        "degenerate_fraction": (all_empty_count + uniform_nonempty_count) / n,
    }


def _settling_step(trajectory):
    """Return (settled_at, never_settled) for a t=0..T argmax-body trajectory.

    settled_at is the step index after which the body never changes again.
    never_settled is True if the body was still changing on the very last
    transition (t=T-1 -> t=T), so stability within the trajectory's length
    was not observed.
    """
    T = len(trajectory) - 1
    last_change = -1
    for t in range(T):
        if not np.array_equal(trajectory[t], trajectory[t + 1]):
            last_change = t
    settled_at = last_change + 1
    never_settled = last_change == T - 1
    return settled_at, never_settled


def _histogram(values, max_value: int):
    counts = [0] * (max_value + 1)
    for v in values:
        counts[v] += 1
    return counts


def _characterize_encoder(encoder, rng, n: int, n_example_trajectories: int = 0):
    """Sample n random genomes from `encoder` and return (stats, example_trajectories).

    If the encoder exposes a develop(genome, return_trajectory=True) method
    (currently only NCAEncoder), developmental settling behaviour is also
    computed and folded into the returned stats dict; otherwise only the
    generic validity/diversity statistics are computed.
    """
    bodies = []
    settle_steps = []
    never_settled_count = 0
    max_abs_state = []
    example_trajectories = []

    supports_trajectory = hasattr(encoder, "develop")

    for i in range(n):
        genome = encoder.random_genome(rng)

        if supports_trajectory:
            trajectory = encoder.develop(genome, return_trajectory=True)
            body = trajectory[-1]
            settled_at, never_settled = _settling_step(trajectory)
            settle_steps.append(settled_at)
            if never_settled:
                never_settled_count += 1
            if i < n_example_trajectories:
                example_trajectories.append(trajectory)
        else:
            body = encoder.decode(genome)

        bodies.append(body)

        # NCA-specific residual-magnitude diagnostic: this reaches into the
        # encoder's own private stepping numerics, so it is deliberately
        # gated on the concrete class (only meaningful for NCA's weight-
        # driven residual update) rather than attempted generically for
        # every encoder.
        if isinstance(encoder, NCAEncoder):
            w1, b1, w2, b2 = encoder._unpack(genome)
            state = encoder._initial_state.copy()
            for _ in range(encoder.dev_steps):
                state = encoder._step(state, w1, b1, w2, b2)
            max_abs_state.append(float(np.max(np.abs(state))))

    stats = _validity_stats(bodies)

    if supports_trajectory:
        dev_steps = getattr(encoder, "dev_steps", max(settle_steps, default=0))
        stats["settle_step_distribution"] = _histogram(settle_steps, dev_steps)
        stats["never_settled_fraction"] = never_settled_count / n if n else 0.0

    if max_abs_state:
        stats["state_magnitude_at_T"] = {
            "mean": float(np.mean(max_abs_state)),
            "median": float(np.median(max_abs_state)),
            "max": float(np.max(max_abs_state)),
            "min": float(np.min(max_abs_state)),
        }

    return stats, example_trajectories


def _plot_trajectories(trajectories, path: str, encoder_name: str) -> None:
    if not trajectories:
        return
    n_rows = len(trajectories)
    n_cols = len(trajectories[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2), squeeze=False)
    cmap = plt.get_cmap("tab10", N_MATERIALS)

    for row, trajectory in enumerate(trajectories):
        for col, body in enumerate(trajectory):
            ax = axes[row][col]
            ax.imshow(body, cmap=cmap, vmin=0, vmax=N_MATERIALS - 1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={col}", fontsize=8)

    fig.suptitle(f"Example {encoder_name} developmental trajectories (random genomes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_report(encoder_name: str, stats: dict, path: str, figure_relpath: str, compare_to: str = None) -> None:
    primary = stats[encoder_name]

    lines = [
        f"# {encoder_name} Encoder Sanity Report",
        "",
        f"N = {stats['n']} random genomes"
        + (" per encoder" if compare_to else "")
        + f", seed = {stats['seed']}. No PPO was run for this experiment.",
        "",
        "## Validity and diversity",
        "",
    ]

    if compare_to:
        reference = stats[f"{compare_to}_reference"]
        lines += [
            f"| Metric | {encoder_name} | {compare_to} (reference) |",
            "|---|---|---|",
            f"| Validity rate | {primary['valid_rate']:.4f} | {reference['valid_rate']:.4f} |",
            f"| Unique body hashes | {primary['unique_body_count']} ({primary['unique_body_fraction']:.4f}) "
            f"| {reference['unique_body_count']} ({reference['unique_body_fraction']:.4f}) |",
            f"| All-empty fraction | {primary['all_empty_fraction']:.4f} | {reference['all_empty_fraction']:.4f} |",
            f"| Uniform non-empty fraction | {primary['uniform_nonempty_fraction']:.4f} "
            f"| {reference['uniform_nonempty_fraction']:.4f} |",
            f"| Degenerate fraction (all-empty + uniform) | {primary['degenerate_fraction']:.4f} "
            f"| {reference['degenerate_fraction']:.4f} |",
            "",
            "### Invalid-genome failure reasons",
            "",
            f"| Reason category | {encoder_name} count | {compare_to} count |",
            "|---|---|---|",
        ]
        all_reasons = sorted(set(primary["invalid_reason_counts"]) | set(reference["invalid_reason_counts"]))
        for reason in all_reasons:
            lines.append(
                f"| {reason} | {primary['invalid_reason_counts'].get(reason, 0)} "
                f"| {reference['invalid_reason_counts'].get(reason, 0)} |"
            )
    else:
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Validity rate | {primary['valid_rate']:.4f} |",
            f"| Unique body hashes | {primary['unique_body_count']} ({primary['unique_body_fraction']:.4f}) |",
            f"| All-empty fraction | {primary['all_empty_fraction']:.4f} |",
            f"| Uniform non-empty fraction | {primary['uniform_nonempty_fraction']:.4f} |",
            f"| Degenerate fraction (all-empty + uniform) | {primary['degenerate_fraction']:.4f} |",
            "",
            "### Invalid-genome failure reasons",
            "",
            "| Reason category | Count |",
            "|---|---|",
        ]
        for reason in sorted(primary["invalid_reason_counts"]):
            lines.append(f"| {reason} | {primary['invalid_reason_counts'][reason]} |")

    if "settle_step_distribution" in primary:
        dev_steps = len(primary["settle_step_distribution"]) - 1
        lines += [
            "",
            f"## Developmental stability ({encoder_name} only)",
            "",
            f"Never-settled fraction (still changing on the final transition, "
            f"t={dev_steps - 1} -> t={dev_steps}): {primary['never_settled_fraction']:.4f}",
            "",
            "Settle-step distribution (step index after which the argmax body never changes again):",
            "",
            "| Step | Count |",
            "|---|---|",
        ]
        for step, count in enumerate(primary["settle_step_distribution"]):
            lines.append(f"| {step} | {count} |")

    if "state_magnitude_at_T" in primary:
        mag = primary["state_magnitude_at_T"]
        lines += [
            "",
            f"## Residual state magnitude at t=T ({encoder_name} only)",
            "",
            f"max |state| across cells/channels at t=T, across genomes: "
            f"mean={mag['mean']:.3f}, median={mag['median']:.3f}, min={mag['min']:.3f}, max={mag['max']:.3f}",
        ]

    lines += [
        "",
        "## Example developmental trajectories",
        "",
        f"![trajectories]({figure_relpath})",
        "",
        "## Decision gate",
        "",
    ]

    gate_notes = []
    if primary["valid_rate"] < VALIDITY_GATE:
        gate_notes.append(
            f"STOP: {encoder_name} validity rate ({primary['valid_rate']:.4f}) is below the "
            f"{VALIDITY_GATE:.0%} gate threshold. Revisit this encoder's random-genome/initial-"
            "condition hyperparameters before running the comparison."
        )
    if primary["degenerate_fraction"] > DEGENERACY_GATE:
        gate_notes.append(
            f"STOP: {encoder_name} degenerate-body fraction ({primary['degenerate_fraction']:.4f}) is above "
            f"the {DEGENERACY_GATE:.0%} gate threshold. Revisit this encoder's random-genome/initial-"
            "condition hyperparameters before running the comparison."
        )
    if not gate_notes:
        gate_notes.append(
            "PASS: validity rate and degenerate-body fraction are both within the "
            "decision-gate thresholds."
        )
    for note in gate_notes:
        lines.append(f"- {note}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def inspect_encoder(
    encoder_name: str,
    seed: int = 0,
    n: int = 2000,
    output_dir: str = "results",
    n_example_trajectories: int = 5,
    encoder_kwargs: dict = None,
    compare_to: str = None,
    compare_to_kwargs: dict = None,
    report_prefix: str = None,
) -> dict:
    """Sample n random genomes from the registered encoder `encoder_name`
    (via evodevo.encoders.get_encoder) and write a PPO-free characterization
    report. If `compare_to` names another registered encoder, its stats are
    computed too and included as a side-by-side reference column, exactly
    as the original NCA-vs-direct sanity check did.

    Writes {output_dir}/{report_prefix or encoder_name}_sanity.{json,md} and
    a trajectories PNG, and returns the stats dict (also JSON-dumped as-is).

    Every registered encoder now requires an explicit `factors`
    (ExperimentFactors) constructor kwarg; `encoder_kwargs`/`compare_to_kwargs`
    may include factor fields directly (e.g. {"dev_steps": 1}), which are
    pulled out and used to build it (see _build_encoder) -- unset factor
    fields default to dev_steps=10, init_condition="full_nonsymmetric".
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = make_rng(seed)

    encoder = _build_encoder(encoder_name, seed, encoder_kwargs)
    primary_stats, example_trajectories = _characterize_encoder(
        encoder, rng, n, n_example_trajectories=n_example_trajectories
    )

    stats = {
        "seed": seed,
        "n": n,
        encoder_name: primary_stats,
        f"{encoder_name}_config": encoder.describe(),
    }

    if compare_to:
        reference_encoder = _build_encoder(compare_to, seed, compare_to_kwargs)
        reference_stats, _ = _characterize_encoder(reference_encoder, rng, n, n_example_trajectories=0)
        stats[f"{compare_to}_reference"] = reference_stats
        stats[f"{compare_to}_config"] = reference_encoder.describe()

    prefix = report_prefix or encoder_name
    json_path = os.path.join(output_dir, f"{prefix}_sanity.json")
    md_path = os.path.join(output_dir, f"{prefix}_sanity.md")
    figure_name = f"{prefix}_sanity_trajectories.png"

    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    _plot_trajectories(example_trajectories, os.path.join(output_dir, figure_name), encoder_name)
    _write_report(encoder_name, stats, md_path, figure_relpath=figure_name, compare_to=compare_to)

    return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, help="a registered encoder name, e.g. direct/nca/rule_based")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to results/inspection/{encoder}",
    )
    parser.add_argument("--compare-to", default=None, help="optional second registered encoder name for reference")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or os.path.join("results", "inspection", args.encoder)
    stats = inspect_encoder(
        args.encoder, seed=args.seed, n=args.n, output_dir=output_dir, compare_to=args.compare_to
    )
    block = stats[args.encoder]
    msg = (
        f"{args.encoder} validity_rate={block['valid_rate']:.4f} "
        f"degenerate_fraction={block['degenerate_fraction']:.4f}"
    )
    if "never_settled_fraction" in block:
        msg += f" never_settled_fraction={block['never_settled_fraction']:.4f}"
    print(msg)


if __name__ == "__main__":
    main()
