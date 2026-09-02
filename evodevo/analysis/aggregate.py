"""Load one or many run directories into a tidy per-evaluation DataFrame."""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import pandas as pd


def _load_single_run(run_dir: str) -> pd.DataFrame:
    log_path = os.path.join(run_dir, "run_log.jsonl")
    meta_path = os.path.join(run_dir, "run_meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    config = meta["config"]

    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    df = pd.DataFrame.from_records(records)
    df["encoder"] = config["encoder"]
    df["task"] = config["task"]
    df["seed"] = config["seed"]
    df["run_dir"] = run_dir
    return df


def load_runs(run_dirs: Iterable[str]) -> pd.DataFrame:
    """Load a collection of run directories (each containing run_log.jsonl and
    run_meta.json) into a single tidy DataFrame, one row per evaluation."""
    frames = [_load_single_run(d) for d in run_dirs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def discover_runs(root_dir: str) -> list:
    """Find every completed run (a directory containing run_meta.json) under root_dir."""
    run_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if "run_meta.json" in filenames:
            run_dirs.append(dirpath)
    return sorted(run_dirs)


def load_all_runs(root_dir: str) -> pd.DataFrame:
    return load_runs(discover_runs(root_dir))


def comparison_table(df: pd.DataFrame, reference_encoder: str = "direct") -> pd.DataFrame:
    """Per-encoder summary for the head-to-head comparison.

    "Evaluations to 50%/90% of the direct arm's final best" assumes the
    reference arm's final best fitness is positive (walking, not
    sentinel-dominated); if it is not, the 50%/90% thresholds are reported
    as-is but their "closer to the goal" interpretation no longer holds and
    this should be called out rather than read at face value.
    """
    if df.empty:
        return pd.DataFrame()

    reference_finals = []
    for _seed, g in df[df["encoder"] == reference_encoder].groupby("seed"):
        g = g.sort_values("eval_index")
        reference_finals.append(g["fitness"].cummax().iloc[-1])
    reference_best = float(np.mean(reference_finals)) if reference_finals else float("nan")

    rows = []
    for encoder, enc_df in df.groupby("encoder"):
        final_bests = []
        evals_to_50 = []
        evals_to_90 = []

        for _seed, g in enc_df.groupby("seed"):
            g = g.sort_values("eval_index")
            best_so_far = g["fitness"].cummax()
            final_bests.append(best_so_far.iloc[-1])

            for threshold, bucket in ((0.5 * reference_best, evals_to_50), (0.9 * reference_best, evals_to_90)):
                hit = g.loc[(best_so_far >= threshold).values, "eval_index"]
                bucket.append(float(hit.iloc[0] + 1) if len(hit) else np.nan)

        rows.append(
            {
                "encoder": encoder,
                "n_seeds": int(enc_df["seed"].nunique()),
                "final_best_fitness_mean": float(np.mean(final_bests)),
                "final_best_fitness_std": float(np.std(final_bests)),
                "evaluations_to_50pct_of_reference_mean": float(np.nanmean(evals_to_50)),
                "evaluations_to_90pct_of_reference_mean": float(np.nanmean(evals_to_90)),
                "validity_rate": float(enc_df["valid"].mean()),
                "cache_hit_rate": float(enc_df["cache_hit"].mean()) if "cache_hit" in enc_df.columns else float("nan"),
                "unique_bodies_explored": int(enc_df["body_hash"].nunique()),
                "mean_wall_clock_s_per_eval": float(enc_df["wall_clock_s"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("encoder").reset_index(drop=True)
