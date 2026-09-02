import importlib.util
import os
import tempfile
import warnings

import numpy as np
import pytest

from evodevo.visualization.render_results import render_all, render_comparison_figure, render_one_arm

HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None

GOOD_WALKER = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)

DISCONNECTED_BODY = np.array(
    [
        [1, 0, 0, 0, 3],
        [4, 1, 2, 0, 4],
        [3, 0, 3, 0, 2],
        [0, 3, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ],
    dtype=int,
)


def _write_fake_completed_run(runs_root, encoder, task, seed, body, fitness=5.0):
    import json

    run_dir = os.path.join(runs_root, encoder, task, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, "best_body.npz"), body=body)
    with open(os.path.join(run_dir, "run_log.jsonl"), "w") as f:
        f.write(json.dumps({"eval_index": 0, "fitness": fitness, "valid": True, "body_hash": "x"}) + "\n")
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({"config": {"encoder": encoder, "task": task, "seed": seed}}, f)
    return run_dir


def test_render_one_arm_never_raises_on_a_forced_invalid_body_and_still_writes_morphology():
    """A forced rendering failure (an invalid "best" body reaching the
    rollout stage) must not raise -- the pipeline logs it and moves on,
    and anything that CAN still be produced (the morphology PNG, which
    doesn't touch the simulator) must still be written."""
    with tempfile.TemporaryDirectory() as runs_root, tempfile.TemporaryDirectory() as output_root:
        _write_fake_completed_run(runs_root, "direct", "Walker-v0", 0, DISCONNECTED_BODY)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outcome = render_one_arm(runs_root, "direct", "Walker-v0", output_root)

        assert outcome["errors"]  # the rollout failure was recorded
        assert any("rollout" in e for e in outcome["errors"])
        # the morphology PNG (no simulator involved) must still exist.
        assert os.path.exists(os.path.join(output_root, "direct", "Walker-v0", "morphology.png"))
        # no rollout files should exist, since rendering it was refused.
        assert not os.path.exists(os.path.join(output_root, "direct", "Walker-v0", "rollout.gif"))


def test_render_one_arm_reports_missing_run_without_raising():
    with tempfile.TemporaryDirectory() as runs_root, tempfile.TemporaryDirectory() as output_root:
        outcome = render_one_arm(runs_root, "nca", "Walker-v0", output_root)
        assert outcome["errors"]
        assert "no completed run found" in outcome["errors"][0]


def test_render_comparison_figure_handles_partial_bodies():
    with tempfile.TemporaryDirectory() as tmp:
        outcomes = [
            {"encoder": "direct", "body": GOOD_WALKER},
            {"encoder": "nca", "body": None},  # e.g. this arm had no completed run
        ]
        path = render_comparison_figure(outcomes, os.path.join(tmp, "comparison.png"))
        assert path is not None
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_render_comparison_figure_returns_none_when_no_bodies_available():
    with tempfile.TemporaryDirectory() as tmp:
        outcomes = [{"encoder": "direct", "body": None}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = render_comparison_figure(outcomes, os.path.join(tmp, "comparison.png"))
        assert path is None


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_render_all_resolves_relative_paths_before_any_env_creation_changes_cwd():
    """Regression test for a real bug found during development: creating an
    EvoGym env (even with render_mode="img", never "human") changes the
    process's current working directory as a side effect on some platforms
    (observed on macOS, likely a GLFW/OpenGL context side effect). Since
    render_all() renders multiple arms in one process, only the FIRST arm
    resolved correctly before this was fixed -- every arm after it silently
    wrote its output under the wrong (already-cwd-changed) directory. This
    test passes a genuinely RELATIVE runs_root/output_root (relative to the
    process's cwd at call time, changed here via os.chdir to a controlled
    temp directory) and multiple encoders, and checks every arm's output
    landed under the correct, intended directory."""
    import json

    with tempfile.TemporaryDirectory() as workdir:
        runs_root = "runs"
        output_root = "media"
        os.makedirs(os.path.join(workdir, runs_root), exist_ok=True)

        for encoder in ("direct", "nca"):
            run_dir = os.path.join(workdir, runs_root, encoder, "Walker-v0", "seed_0")
            os.makedirs(run_dir, exist_ok=True)
            np.savez(os.path.join(run_dir, "best_body.npz"), body=GOOD_WALKER)
            with open(os.path.join(run_dir, "run_log.jsonl"), "w") as f:
                f.write(json.dumps({"eval_index": 0, "fitness": 1.0, "valid": True, "body_hash": "x"}) + "\n")
            with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
                json.dump({"config": {"encoder": encoder, "task": "Walker-v0", "seed": 0}}, f)

        original_cwd = os.getcwd()
        try:
            os.chdir(workdir)
            result = render_all(
                runs_root=runs_root,
                task="Walker-v0",
                output_root=output_root,
                encoders=["direct", "nca"],
                n_rollout_steps=3,
            )
        finally:
            os.chdir(original_cwd)

        for encoder in ("direct", "nca"):
            outcome = next(o for o in result["outcomes"] if o["encoder"] == encoder)
            assert not outcome["errors"], outcome["errors"]
            expected_dir = os.path.join(workdir, output_root, encoder, "Walker-v0")
            assert os.path.exists(os.path.join(expected_dir, "morphology.png"))
            assert os.path.exists(os.path.join(expected_dir, "rollout.mp4"))
            assert os.path.getsize(os.path.join(expected_dir, "rollout.mp4")) > 0
            assert os.path.exists(os.path.join(expected_dir, "rollout.gif"))
            assert os.path.getsize(os.path.join(expected_dir, "rollout.gif")) > 0
