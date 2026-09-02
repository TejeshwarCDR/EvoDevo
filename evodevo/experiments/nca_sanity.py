"""E0: developmental-engine sanity check. No PPO; characterises the NCA
encoder's prior over morphologies (and the direct encoder's, as a reference)
from random genomes alone, before any evolutionary or PPO compute is spent.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.nca import N_MATERIALS, NCAEncoder
from evodevo.morphology.metrics import body_hash
from evodevo.morphology.validate import validate_body
from evodevo.core.seeding import make_rng

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
    transition (t=T-1 -> t=T), so stability within dev_steps was not observed.
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


def _plot_trajectories(trajectories, path: str) -> None:
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

    fig.suptitle("Example NCA developmental trajectories (random genomes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_report(stats: dict, path: str, figure_relpath: str) -> None:
    nca = stats["nca"]
    direct = stats["direct_reference"]
    dev_steps = stats["nca_config"]["dev_steps"]

    lines = [
        "# NCA Developmental Engine Sanity Report (E0)",
        "",
        f"N = {stats['n']} random genomes per encoder, seed = {stats['seed']}. "
        "No PPO was run for this experiment.",
        "",
        "## Validity and diversity",
        "",
        "| Metric | NCA | Direct (reference) |",
        "|---|---|---|",
        f"| Validity rate | {nca['valid_rate']:.4f} | {direct['valid_rate']:.4f} |",
        f"| Unique body hashes | {nca['unique_body_count']} ({nca['unique_body_fraction']:.4f}) "
        f"| {direct['unique_body_count']} ({direct['unique_body_fraction']:.4f}) |",
        f"| All-empty fraction | {nca['all_empty_fraction']:.4f} | {direct['all_empty_fraction']:.4f} |",
        f"| Uniform non-empty fraction | {nca['uniform_nonempty_fraction']:.4f} "
        f"| {direct['uniform_nonempty_fraction']:.4f} |",
        f"| Degenerate fraction (all-empty + uniform) | {nca['degenerate_fraction']:.4f} "
        f"| {direct['degenerate_fraction']:.4f} |",
        "",
        "### Invalid-genome failure reasons",
        "",
        "| Reason category | NCA count | Direct count |",
        "|---|---|---|",
    ]

    all_reasons = sorted(set(nca["invalid_reason_counts"]) | set(direct["invalid_reason_counts"]))
    for reason in all_reasons:
        lines.append(
            f"| {reason} | {nca['invalid_reason_counts'].get(reason, 0)} "
            f"| {direct['invalid_reason_counts'].get(reason, 0)} |"
        )

    lines += [
        "",
        "## Developmental stability (NCA only)",
        "",
        f"Never-settled fraction (still changing on the final transition, t={dev_steps - 1} -> t={dev_steps}): "
        f"{nca['never_settled_fraction']:.4f}",
        "",
        "Settle-step distribution (step index after which the argmax body never changes again):",
        "",
        "| Step | Count |",
        "|---|---|",
    ]
    for step, count in enumerate(nca["settle_step_distribution"]):
        lines.append(f"| {step} | {count} |")

    mag = nca["state_magnitude_at_T"]
    lines += [
        "",
        "## Residual state magnitude at t=T (NCA only)",
        "",
        f"max |state| across cells/channels at t=T, across genomes: "
        f"mean={mag['mean']:.3f}, median={mag['median']:.3f}, min={mag['min']:.3f}, max={mag['max']:.3f}",
        "",
        "## Example developmental trajectories",
        "",
        f"![trajectories]({figure_relpath})",
        "",
        "## Decision gate",
        "",
    ]

    gate_notes = []
    if nca["valid_rate"] < VALIDITY_GATE:
        gate_notes.append(
            f"STOP: NCA validity rate ({nca['valid_rate']:.4f}) is below the "
            f"{VALIDITY_GATE:.0%} gate threshold. Revisit init_scale / initial state / "
            "hidden_channels before running the comparison."
        )
    if nca["degenerate_fraction"] > DEGENERACY_GATE:
        gate_notes.append(
            f"STOP: NCA degenerate-body fraction ({nca['degenerate_fraction']:.4f}) is above the "
            f"{DEGENERACY_GATE:.0%} gate threshold. Revisit init_scale / initial state / "
            "hidden_channels before running the comparison."
        )
    if not gate_notes:
        gate_notes.append(
            "PASS: validity rate and degenerate-body fraction are both within the "
            "decision-gate thresholds. See docs/decisions.md for the recorded numbers."
        )
    for note in gate_notes:
        lines.append(f"- {note}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_sanity(
    seed: int = 0,
    n: int = 2000,
    output_dir: str = "results",
    n_example_trajectories: int = 5,
    nca_kwargs: dict = None,
) -> dict:
    """nca_kwargs overrides/extends the NCA encoder's own hyperparameters
    (e.g. {'dev_steps': 1}), reused as-is by ablation_report.py to
    characterise each ablation cell's prior over morphologies without
    spending any PPO compute."""
    os.makedirs(output_dir, exist_ok=True)

    rng = make_rng(seed)
    nca = NCAEncoder(**(nca_kwargs or {}))
    direct = DirectEncoder()

    nca_bodies = []
    nca_settle_steps = []
    nca_never_settled = 0
    nca_max_abs_state = []
    example_trajectories = []

    for i in range(n):
        genome = nca.random_genome(rng)
        trajectory = nca.develop(genome, return_trajectory=True)
        nca_bodies.append(trajectory[-1])

        settled_at, never_settled = _settling_step(trajectory)
        nca_settle_steps.append(settled_at)
        if never_settled:
            nca_never_settled += 1

        # Reuse the encoder's own private stepping numerics (rather than
        # re-deriving them) to get the raw residual state at t=T.
        w1, b1, w2, b2 = nca._unpack(genome)
        state = nca._initial_state.copy()
        for _ in range(nca.dev_steps):
            state = nca._step(state, w1, b1, w2, b2)
        nca_max_abs_state.append(float(np.max(np.abs(state))))

        if i < n_example_trajectories:
            example_trajectories.append(trajectory)

    direct_bodies = [direct.decode(direct.random_genome(rng)) for _ in range(n)]

    nca_stats = _validity_stats(nca_bodies)
    nca_stats["settle_step_distribution"] = _histogram(nca_settle_steps, nca.dev_steps)
    nca_stats["never_settled_fraction"] = nca_never_settled / n
    nca_stats["state_magnitude_at_T"] = {
        "mean": float(np.mean(nca_max_abs_state)),
        "median": float(np.median(nca_max_abs_state)),
        "max": float(np.max(nca_max_abs_state)),
        "min": float(np.min(nca_max_abs_state)),
    }

    direct_stats = _validity_stats(direct_bodies)

    stats = {
        "seed": seed,
        "n": n,
        "nca": nca_stats,
        "direct_reference": direct_stats,
        "nca_config": nca.describe(),
        "direct_config": direct.describe(),
    }

    with open(os.path.join(output_dir, "nca_sanity.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    figure_name = "nca_sanity_trajectories.png"
    _plot_trajectories(example_trajectories, os.path.join(output_dir, figure_name))
    _write_report(stats, os.path.join(output_dir, "nca_sanity.md"), figure_relpath=figure_name)

    return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    stats = run_sanity(seed=args.seed, n=args.n, output_dir=args.output_dir)
    print(
        f"NCA validity_rate={stats['nca']['valid_rate']:.4f} "
        f"degenerate_fraction={stats['nca']['degenerate_fraction']:.4f} "
        f"never_settled_fraction={stats['nca']['never_settled_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
