import importlib.util
import os
import subprocess

import numpy as np
import pytest

from evodevo.evaluation.cache import FitnessCache
from evodevo.evaluation.ppo_runner import (
    DEFAULT_SENTINEL_FITNESS,
    evaluate_body,
    evaluate_body_surrogate,
    make_evaluator,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

INVALID_BODY = np.zeros((5, 5), dtype=int)  # all-empty: fails validation

HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None


def test_surrogate_deterministic():
    r1 = evaluate_body_surrogate(GOOD_WALKER, seed=0)
    r2 = evaluate_body_surrogate(GOOD_WALKER, seed=0)
    assert r1.fitness == r2.fitness
    assert r1.valid and r2.valid


def test_surrogate_invalid_body_returns_sentinel():
    result = evaluate_body_surrogate(INVALID_BODY)
    assert not result.valid
    assert result.fitness == DEFAULT_SENTINEL_FITNESS
    assert result.reason is not None


def test_surrogate_custom_sentinel():
    result = evaluate_body_surrogate(INVALID_BODY, sentinel_fitness=-99.0)
    assert result.fitness == -99.0


def test_make_evaluator_surrogate():
    fn = make_evaluator("surrogate")
    assert fn is evaluate_body_surrogate


def test_make_evaluator_real():
    fn = make_evaluator("real")
    assert fn is evaluate_body


def test_make_evaluator_unknown_raises():
    with pytest.raises(ValueError):
        make_evaluator("bogus")


def test_cache_hit_and_miss():
    cache = FitnessCache()
    result = evaluate_body_surrogate(GOOD_WALKER, seed=0)

    fetched = cache.get(result.body_hash)
    assert fetched is None
    assert cache.misses == 1
    assert cache.hits == 0

    cache.put(result.body_hash, result)
    fetched = cache.get(result.body_hash)
    assert fetched is result
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.hit_rate == 0.5


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_real_ppo_smoke():
    result = evaluate_body(
        GOOD_WALKER,
        task="Walker-v0",
        ppo_timesteps=2000,
        eval_episodes=1,
        seed=0,
    )
    assert result.valid
    assert np.isfinite(result.fitness)


# --- Regression: evaluate_body()'s extraction of _rollout_fitness() and the
# addition of evaluate_body_with_curve() must not have changed evaluate_body
# ()'s own behaviour. Compares the committed HEAD version of this module
# against whatever is currently on disk (which may equal HEAD, or may carry
# this exact refactor as an uncommitted change) on a fixed set of valid and
# invalid bodies, at the same real-PPO settings.

DISCONNECTED_BODY = np.array(
    [
        [0, 1, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 3],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)

NO_ACTUATOR_BODY = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)

GOOD_WALKER_2 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 3, 1, 3, 0],
        [0, 1, 0, 1, 0],
        [0, 3, 1, 3, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)

_PARITY_BODIES = [
    ("GOOD_WALKER", GOOD_WALKER),
    ("GOOD_WALKER_2", GOOD_WALKER_2),
    ("INVALID_BODY_too_few_voxels", INVALID_BODY),
    ("DISCONNECTED_BODY", DISCONNECTED_BODY),
    ("NO_ACTUATOR_BODY", NO_ACTUATOR_BODY),
]


def _load_ppo_runner_at_head():
    """Loads the git-committed HEAD version of evodevo/evaluation/
    ppo_runner.py as its own, separately-named module -- via `git show`
    into a temp file, not `git stash` (stashing would touch the whole
    working tree, not just this one file, for the duration of the test).
    Its own imports are all absolute (`from evodevo.morphology...`, see the
    module's own docstring/imports), so loading it from an arbitrary temp
    path resolves correctly against the normal evodevo package already on
    sys.path -- no path surgery needed beyond the load itself."""
    result = subprocess.run(
        ["git", "show", "HEAD:evodevo/evaluation/ppo_runner.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    head_source = result.stdout

    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="ppo_runner_head_")
    tmp_path = os.path.join(tmp_dir, "ppo_runner_head.py")
    with open(tmp_path, "w") as f:
        f.write(head_source)

    spec = importlib.util.spec_from_file_location("ppo_runner_head_parity_check", tmp_path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses's own annotation-resolution logic looks the module up via
    # sys.modules[cls.__module__] -- it must be registered there BEFORE
    # exec_module() runs the @dataclass-decorated class bodies, or it fails
    # with "'NoneType' object has no attribute '__dict__'" (caught by
    # running this test).
    import sys as _sys

    _sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del _sys.modules[spec.name]
        raise
    return module


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_evaluate_body_is_bit_identical_between_head_and_working_tree():
    """See docs/decisions.md / PREFLIGHT_READINESS.md Section 9's
    "ppo_runner.py change" subsection: this proves the _rollout_fitness()
    extraction (and the addition of evaluate_body_with_curve(), which
    evaluate_body() itself never calls) left evaluate_body() itself
    bit-identical -- same fitness, valid, reason, body_hash -- for a fixed
    set of 5 bodies (2 valid, 3 invalid via 3 different validation
    failures), at ppo_timesteps=2000, seed=0."""
    head_module = _load_ppo_runner_at_head()

    for name, body in _PARITY_BODIES:
        head_result = head_module.evaluate_body(
            body, task="Walker-v0", ppo_timesteps=2000, eval_episodes=1, seed=0,
        )
        working_tree_result = evaluate_body(
            body, task="Walker-v0", ppo_timesteps=2000, eval_episodes=1, seed=0,
        )

        assert head_result.valid == working_tree_result.valid, name
        assert head_result.reason == working_tree_result.reason, name
        assert head_result.body_hash == working_tree_result.body_hash, name
        assert head_result.fitness == working_tree_result.fitness, (
            f"{name}: fitness differs -- HEAD={head_result.fitness!r} "
            f"working_tree={working_tree_result.fitness!r}"
        )
