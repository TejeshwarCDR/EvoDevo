"""DEPRECATED: this module's NCA-specific characterization logic has been
generalized into evodevo.experiments.inspect_encoder, which works for any
registered encoder. This module is now a thin shim over it, kept ONLY so
that existing callers (evodevo/analysis/ablation_report.py) and existing
results (results/nca_sanity.json, results/ablation_priors/*/nca_sanity.json)
remain reproducible without any code changes.

New code should call:
    evodevo.experiments.inspect_encoder.inspect_encoder(
        "nca", ..., compare_to="direct", report_prefix="nca",
    )
directly instead of run_sanity().
"""
from __future__ import annotations

import argparse
import warnings

from evodevo.experiments.inspect_encoder import inspect_encoder


def run_sanity(
    seed: int = 0,
    n: int = 2000,
    output_dir: str = "results",
    n_example_trajectories: int = 5,
    nca_kwargs: dict = None,
) -> dict:
    """DEPRECATED: use evodevo.experiments.inspect_encoder.inspect_encoder
    instead. nca_kwargs overrides/extends the NCA encoder's own
    hyperparameters (e.g. {'dev_steps': 1}), reused as-is by
    ablation_report.py to characterise each ablation cell's prior over
    morphologies without spending any PPO compute."""
    warnings.warn(
        "evodevo.experiments.nca_sanity.run_sanity is deprecated; call "
        "evodevo.experiments.inspect_encoder.inspect_encoder('nca', ..., "
        "compare_to='direct', report_prefix='nca') directly instead. This shim exists "
        "only so existing results/ artifacts remain reproducible.",
        DeprecationWarning,
        stacklevel=2,
    )
    return inspect_encoder(
        "nca",
        seed=seed,
        n=n,
        output_dir=output_dir,
        n_example_trajectories=n_example_trajectories,
        encoder_kwargs=nca_kwargs,
        compare_to="direct",
        report_prefix="nca",
    )


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
