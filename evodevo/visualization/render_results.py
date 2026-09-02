"""CLI: for each arm's best evolved body, produce a static morphology PNG,
a rollout gif+mp4, and (where available) a development-trajectory figure,
writing to results/media/{encoder}/{task}/. Also writes a single
side-by-side comparison figure of every arm's best morphology.

Rendering must NEVER crash a benchmark run: every rendering call in this
module is wrapped in try/except that warns and records the failure rather
than raising, so a failed gif never loses a completed PPO run's results.

Usage:
    python -m evodevo.visualization.render_results \
        --runs-root results/runs --task Walker-v0 --output-root results/media
"""
from __future__ import annotations

import argparse
import os
import warnings
from typing import List, Optional

import numpy as np

from evodevo.analysis.aggregate import discover_runs, load_runs
from evodevo.encoders import get_encoder, list_encoders
from evodevo.visualization.development import plot_development_trajectory
from evodevo.visualization.morphology import plot_body, plot_body_grid
from evodevo.visualization.render import render_body_rollout

DEFAULT_RUNS_ROOT = "results/runs"
DEFAULT_OUTPUT_ROOT = "results/media"


def _find_best_run(runs_root: str, encoder: str, task: str) -> Optional[dict]:
    """{"seed", "run_dir", "fitness"} for the highest-fitness completed
    evaluation among every seed of (encoder, task), or None if no
    completed run exists for this arm/task."""
    base = os.path.join(runs_root, encoder, task)
    run_dirs = discover_runs(base)
    if not run_dirs:
        return None

    df = load_runs(run_dirs)
    if df.empty:
        return None

    best_row = df.loc[df["fitness"].idxmax()]
    return {"seed": int(best_row["seed"]), "run_dir": str(best_row["run_dir"]), "fitness": float(best_row["fitness"])}


def _load_best_body(run_dir: str) -> np.ndarray:
    npz_path = os.path.join(run_dir, "best_body.npz")
    data = np.load(npz_path)
    return data["body"]


def _warn_and_record(outcome: dict, message: str) -> None:
    outcome.setdefault("errors", []).append(message)
    warnings.warn(f"[render_results] {message}", stacklevel=2)


def render_one_arm(
    runs_root: str,
    encoder: str,
    task: str,
    output_root: str,
    n_rollout_steps: int = 200,
) -> dict:
    """Renders everything for one arm's best evolved body. Never raises --
    every failure is caught, warned about, and recorded in the returned
    dict's "errors" list."""
    # Resolve to absolute paths BEFORE any rendering happens: creating an
    # EvoGym env (inside render_body_rollout, below) can change the
    # process's current working directory as a side effect on some
    # platforms (observed on macOS) -- a relative runs_root/output_root
    # would then silently resolve against the wrong directory for this
    # call, or for a later encoder's call in the same process.
    runs_root = os.path.abspath(runs_root)
    output_root = os.path.abspath(output_root)

    output_dir = os.path.join(output_root, encoder, task)
    os.makedirs(output_dir, exist_ok=True)
    outcome: dict = {"encoder": encoder, "task": task, "output_dir": output_dir, "errors": []}

    best = _find_best_run(runs_root, encoder, task)
    if best is None:
        _warn_and_record(outcome, f"{encoder}/{task}: no completed run found under {runs_root}")
        return outcome
    outcome.update(best)

    try:
        body = _load_best_body(best["run_dir"])
        outcome["body"] = body
    except Exception as e:  # noqa: BLE001
        _warn_and_record(outcome, f"{encoder}/{task}: failed to load best_body.npz ({e})")
        return outcome

    # 1. Static morphology PNG.
    try:
        import matplotlib.pyplot as plt

        ax = plot_body(body, title=f"{encoder} (seed {best['seed']}, fitness={best['fitness']:.2f})")
        morphology_path = os.path.join(output_dir, "morphology.png")
        ax.figure.savefig(morphology_path, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
        outcome["morphology_path"] = morphology_path
    except Exception as e:  # noqa: BLE001
        _warn_and_record(outcome, f"{encoder}/{task}: morphology PNG failed ({e})")

    # 2. Rollout gif+mp4. No trained controller is currently persisted by
    #    run_experiment.py (only the best body + connections are saved),
    #    so this always falls back to random actions -- render_body_rollout
    #    makes that explicit via RolloutResult.label, and this label is
    #    never hidden from the caller.
    try:
        rollout = render_body_rollout(
            body,
            controller_path=None,
            task=task,
            n_steps=n_rollout_steps,
            output_path=os.path.join(output_dir, "rollout"),
            seed=best["seed"],
        )
        outcome["rollout_mp4_path"] = rollout.mp4_path
        outcome["rollout_gif_path"] = rollout.gif_path
        outcome["rollout_label"] = rollout.label
    except Exception as e:  # noqa: BLE001
        _warn_and_record(outcome, f"{encoder}/{task}: rollout gif/mp4 failed ({e})")

    # 3. Development trajectory, if this encoder exposes one. Genomes
    #    themselves are not currently persisted anywhere (only decoded
    #    bodies are), so this necessarily renders a FRESH random genome's
    #    development, not literally the best individual's own genome -- it
    #    demonstrates the encoder's developmental process in general.
    try:
        encoder_instance = get_encoder(encoder, grid_shape=body.shape)
        rng = np.random.default_rng(best["seed"])
        genome = encoder_instance.random_genome(rng)
        trajectory_row = plot_development_trajectory(
            encoder_instance, genome, title=f"{encoder}: example developmental trajectory"
        )
        if trajectory_row is None:
            outcome["development_trajectory_path"] = None  # e.g. direct: no developmental process
        else:
            import matplotlib.pyplot as plt

            trajectory_path = os.path.join(output_dir, "development_trajectory.png")
            trajectory_row[0].figure.savefig(trajectory_path, dpi=120, bbox_inches="tight")
            plt.close(trajectory_row[0].figure)
            outcome["development_trajectory_path"] = trajectory_path
    except Exception as e:  # noqa: BLE001
        _warn_and_record(outcome, f"{encoder}/{task}: development trajectory figure failed ({e})")

    return outcome


def render_comparison_figure(outcomes: List[dict], output_path: str) -> Optional[str]:
    """A single side-by-side figure of every arm's best morphology, for the
    benchmark report. Returns the output path, or None if fewer than one
    arm has a body to show (never raises)."""
    labelled = [(o["encoder"], o["body"]) for o in outcomes if o.get("body") is not None]
    if not labelled:
        warnings.warn("[render_results] no arm has a body to render a comparison figure from", stacklevel=2)
        return None

    try:
        import matplotlib.pyplot as plt

        labels = [f"{enc}" for enc, _body in labelled]
        bodies = [body for _enc, body in labelled]
        axes = plot_body_grid(bodies, labels=labels)
        fig = axes[0].figure
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[render_results] comparison figure failed ({e})", stacklevel=2)
        return None


def render_all(
    runs_root: str = DEFAULT_RUNS_ROOT,
    task: str = "Walker-v0",
    output_root: str = DEFAULT_OUTPUT_ROOT,
    encoders: Optional[List[str]] = None,
    n_rollout_steps: int = 200,
) -> dict:
    # Resolve BOTH paths to absolute ONCE, before rendering ANY arm: env
    # creation changes the process cwd as a side effect (see render_one_arm's
    # comment), and that change persists for the rest of the process -- if
    # each arm re-resolved a relative path independently, only the FIRST
    # arm would resolve correctly; every arm after it would resolve against
    # the already-changed cwd instead of the caller's original one.
    runs_root = os.path.abspath(runs_root)
    output_root = os.path.abspath(output_root)

    encoders = encoders or list_encoders()
    outcomes = [render_one_arm(runs_root, encoder, task, output_root, n_rollout_steps) for encoder in encoders]

    comparison_path = render_comparison_figure(
        outcomes, os.path.join(output_root, f"comparison_{task}.png")
    )

    return {"outcomes": outcomes, "comparison_path": comparison_path}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--task", default="Walker-v0")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--n-rollout-steps", type=int, default=200)
    args = parser.parse_args(argv)

    result = render_all(
        runs_root=args.runs_root,
        task=args.task,
        output_root=args.output_root,
        encoders=args.encoders,
        n_rollout_steps=args.n_rollout_steps,
    )

    for outcome in result["outcomes"]:
        status = "OK" if not outcome["errors"] else f"{len(outcome['errors'])} issue(s)"
        print(f"[{status}] {outcome['encoder']}/{outcome['task']} -> {outcome['output_dir']}")
        for error in outcome["errors"]:
            print(f"    - {error}")

    if result["comparison_path"]:
        print(f"comparison figure: {result['comparison_path']}")
    else:
        print("comparison figure: not written (see warnings above)")


if __name__ == "__main__":
    main()
