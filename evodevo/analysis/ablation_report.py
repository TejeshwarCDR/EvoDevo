"""Ablation report: A1 (dev_steps), A2 (hidden_channels), E3 (second task).

Combines the actual EA+PPO fitness results (from scripts/run_ablations.sh
output directories) with cheap, PPO-free E0-style prior statistics per
hyperparameter cell (reusing evodevo.experiments.nca_sanity.run_sanity),
since a change in the encoding's prior over morphologies can explain a
fitness difference at least as well as fitness alone does.

Gracefully reports "not yet run" for any arm whose PPO results don't exist
yet -- the prior statistics do not require PPO and are always computed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evodevo.analysis.aggregate import discover_runs, load_runs
from evodevo.analysis.plots import plot_fitness_curves
from evodevo.experiments.nca_sanity import run_sanity

A1_DEV_STEPS = [1, 5, 10, 20]
A2_HIDDEN_CHANNELS = [0, 3]
# hidden_channels=3 (the NCA default) is identical to A1's dev_steps=10 cell;
# reuse that tag's runs instead of duplicating the compute.
A2_TAG_BY_HIDDEN_CHANNELS = {0: "H0", 3: "T10"}
SECOND_TASK = "UpStepper-v0"
PRIOR_N = 500  # cheap, PPO-free; independent of eval_budget


def _load_tagged_runs(root_dir: str, encoder: str, task: str, tag=None) -> pd.DataFrame:
    base = os.path.join(root_dir, encoder, task)
    if tag:
        base = os.path.join(base, tag)
    return load_runs(discover_runs(base))


def _fitness_summary(df: pd.DataFrame, arm_label) -> dict:
    if df.empty:
        return {
            "arm": arm_label,
            "status": "not yet run",
            "n_seeds": 0,
            "final_best_fitness_mean": float("nan"),
            "final_best_fitness_std": float("nan"),
            "evolved_validity_rate": float("nan"),
            "unique_bodies_explored": 0,
        }

    finals = []
    for _seed, g in df.groupby("seed"):
        g = g.sort_values("eval_index")
        finals.append(g["fitness"].cummax().iloc[-1])

    return {
        "arm": arm_label,
        "status": "complete",
        "n_seeds": int(df["seed"].nunique()),
        "final_best_fitness_mean": float(np.mean(finals)),
        "final_best_fitness_std": float(np.std(finals)),
        "evolved_validity_rate": float(df["valid"].mean()),
        "unique_bodies_explored": int(df["body_hash"].nunique()),
    }


def _prior_summary(arm_label, nca_kwargs: dict, seed: int, n: int, output_dir: str) -> dict:
    stats = run_sanity(seed=seed, n=n, output_dir=output_dir, n_example_trajectories=0, nca_kwargs=nca_kwargs)
    nca = stats["nca"]
    return {
        "arm": arm_label,
        "prior_validity_rate": nca["valid_rate"],
        "prior_unique_body_fraction": nca["unique_body_fraction"],
        "prior_degenerate_fraction": nca["degenerate_fraction"],
        "prior_never_settled_fraction": nca["never_settled_fraction"],
    }


def _merge(fitness_row: dict, prior_row: dict) -> dict:
    row = dict(fitness_row)
    row.update({k: v for k, v in prior_row.items() if k != "arm"})
    return row


def _rows_to_markdown_table(rows, columns) -> list:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = "nan" if np.isnan(value) else f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _relabelled(df: pd.DataFrame, arm_label: str) -> pd.DataFrame:
    """Copy df with its `encoder` column overwritten by arm_label, so
    plot_fitness_curves (which groups by `encoder`) can be reused to plot
    per-ablation-arm curves instead of per-encoder curves."""
    if df.empty:
        return df
    out = df.copy()
    out["encoder"] = arm_label
    return out


def _save_fitness_curve_figure(dfs_by_arm: dict, path: str) -> bool:
    """dfs_by_arm: {arm_label: df}. Returns False (and writes nothing) if every
    arm is empty (no PPO results yet)."""
    import matplotlib.pyplot as plt

    frames = [_relabelled(df, arm) for arm, df in dfs_by_arm.items() if not df.empty]
    if not frames:
        return False

    combined = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots()
    plot_fitness_curves(combined, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def run_ablation_report(root_dir: str = "results/runs", output_dir: str = "results", prior_seed: int = 0) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # --- A1: dev_steps ---
    a1_rows = []
    a1_dfs = {}
    for T in A1_DEV_STEPS:
        df = _load_tagged_runs(root_dir, "nca", "Walker-v0", tag=f"T{T}")
        arm_label = f"dev_steps={T}"
        a1_dfs[arm_label] = df
        fitness_row = _fitness_summary(df, arm_label=arm_label)
        prior_row = _prior_summary(
            arm_label, {"dev_steps": T}, prior_seed, PRIOR_N, os.path.join(output_dir, "ablation_priors", f"T{T}")
        )
        a1_rows.append(_merge(fitness_row, prior_row))
    a1_figure = "ablation_a1_dev_steps_fitness.png"
    a1_has_figure = _save_fitness_curve_figure(a1_dfs, os.path.join(output_dir, a1_figure))

    # --- A2: hidden_channels ---
    a2_rows = []
    a2_dfs = {}
    for H in A2_HIDDEN_CHANNELS:
        tag = A2_TAG_BY_HIDDEN_CHANNELS[H]
        df = _load_tagged_runs(root_dir, "nca", "Walker-v0", tag=tag)
        arm_label = f"hidden_channels={H}"
        a2_dfs[arm_label] = df
        fitness_row = _fitness_summary(df, arm_label=arm_label)
        prior_row = _prior_summary(
            arm_label,
            {"hidden_channels": H},
            prior_seed,
            PRIOR_N,
            os.path.join(output_dir, "ablation_priors", f"H{H}"),
        )
        a2_rows.append(_merge(fitness_row, prior_row))
    a2_figure = "ablation_a2_hidden_channels_fitness.png"
    a2_has_figure = _save_fitness_curve_figure(a2_dfs, os.path.join(output_dir, a2_figure))

    # --- E3: second task, both encoders ---
    e3_rows = []
    e3_dfs = {}
    for encoder in ("direct", "nca"):
        df = _load_tagged_runs(root_dir, encoder, SECOND_TASK)
        arm_label = f"{encoder} ({SECOND_TASK})"
        e3_dfs[arm_label] = df
        fitness_row = _fitness_summary(df, arm_label=arm_label)
        if encoder == "nca":
            # Task never changes the encoder's prior; reuse the default-hyperparameter
            # prior stats rather than re-sampling identically.
            prior_row = _prior_summary(
                arm_label, {}, prior_seed, PRIOR_N, os.path.join(output_dir, "ablation_priors", "E3_nca")
            )
            e3_rows.append(_merge(fitness_row, prior_row))
        else:
            e3_rows.append(fitness_row)
    e3_figure = "ablation_e3_second_task_fitness.png"
    e3_has_figure = _save_fitness_curve_figure(e3_dfs, os.path.join(output_dir, e3_figure))

    columns_nca = [
        "arm",
        "status",
        "n_seeds",
        "final_best_fitness_mean",
        "final_best_fitness_std",
        "evolved_validity_rate",
        "unique_bodies_explored",
        "prior_validity_rate",
        "prior_degenerate_fraction",
        "prior_never_settled_fraction",
    ]
    columns_direct = [
        "arm",
        "status",
        "n_seeds",
        "final_best_fitness_mean",
        "final_best_fitness_std",
        "evolved_validity_rate",
        "unique_bodies_explored",
    ]

    lines = [
        "# Ablation Report",
        "",
        "3 seeds per arm, identical PPO/EA budget to the main comparison "
        "(`population_size`, `offspring_size`, `eval_budget`, `ppo_timesteps` unchanged "
        "across arms within each axis). \"Prior\" statistics are computed on random "
        "genomes only (no PPO, no evolution; N="
        f"{PRIOR_N}) and characterise the encoding's raw morphology-generation behaviour "
        "under each hyperparameter setting, independent of fitness.",
        "",
        "Any arm marked `not yet run` has no completed EA+PPO runs on disk yet "
        "(run `scripts/run_ablations.sh` to produce them); its prior statistics are "
        "still reported since they require no PPO compute.",
        "",
        "## A1: developmental steps (`dev_steps`)",
        "",
        "T=1 is the key control: it isolates whether *iterated* development matters, "
        "or whether the network's expressiveness alone (a single perception+update pass) "
        "explains any observed effect.",
        "",
    ]
    lines += _rows_to_markdown_table(a1_rows, columns_nca)
    if a1_has_figure:
        lines += ["", f"![A1 fitness curves]({a1_figure})"]
    else:
        lines += ["", "*(figure omitted: no A1 arm has completed PPO results yet)*"]
    lines += [
        "",
        "## A2: hidden channels (`hidden_channels`)",
        "",
        "`hidden_channels=3` is the NCA default and is identical to A1's `dev_steps=10` "
        "cell; its numbers are reused here rather than re-run.",
        "",
    ]
    lines += _rows_to_markdown_table(a2_rows, columns_nca)
    if a2_has_figure:
        lines += ["", f"![A2 fitness curves]({a2_figure})"]
    else:
        lines += ["", "*(figure omitted: no A2 arm has completed PPO results yet)*"]
    lines += [
        "",
        f"## E3: second task ({SECOND_TASK})",
        "",
        "Tests whether any effect seen on Walker-v0 is task-specific.",
        "",
    ]
    lines += _rows_to_markdown_table(e3_rows, columns_direct + [c for c in columns_nca if c not in columns_direct])
    if e3_has_figure:
        lines += ["", f"![E3 fitness curves]({e3_figure})"]
    else:
        lines += ["", "*(figure omitted: no E3 arm has completed PPO results yet)*"]
    lines += [
        "",
        "## What the data shows, what remains a hypothesis, what a larger study needs",
        "",
        "- **What the data shows:** whichever `final_best_fitness_mean`/`std` and prior-statistic "
        "differences are populated above, from actually-executed arms, at 3 seeds each. "
        "With only 3 seeds per cell, treat any ordering between adjacent settings (e.g. "
        "`dev_steps=5` vs `dev_steps=10`) as indicative, not conclusive -- across-seed std "
        "columns should be read alongside the means, not ignored.",
        "- **What remains a hypothesis:** any causal story connecting a prior-statistic "
        "difference (e.g. validity rate or never-settled fraction) to a fitness difference. "
        "The prior statistics and the fitness results are both real, but this report does not "
        "run a statistical test tying them together; a correlation across ablation cells is "
        "suggestive, not proof, at n=4 (A1) or n=2 (A2) points.",
        "- **What a larger study would need:** more seeds per cell (5+, matching the main "
        "comparison), a formal significance test (e.g. a seed-level bootstrap or "
        "Mann-Whitney U across arms) rather than eyeballing mean/std, and -- if `dev_steps=1` "
        "turns out to perform comparably to `dev_steps=10` -- a targeted follow-up isolating "
        "network expressiveness from iteration count more directly (e.g. matching parameter "
        "count across T while ablating recurrence).",
        "",
    ]

    with open(os.path.join(output_dir, "ablations.md"), "w") as f:
        f.write("\n".join(lines))

    return {"a1": a1_rows, "a2": a2_rows, "e3": e3_rows}


def main() -> None:
    run_ablation_report()


if __name__ == "__main__":
    main()
