"""Regression test for evodevo.experiments.lock_hash's documented hashing
procedure. Pins a hand-constructed, fixed set of resolved configs to a known
sha256 digest, so the procedure itself (resolve -> to_dict() -> sorted JSON
-> sha256) can never silently change behaviour across commits without this
test failing -- see lock_hash.py's module docstring for why this matters."""
from __future__ import annotations

from evodevo.core.config import ExperimentConfig
from evodevo.experiments.lock_hash import compute_lock_hash

# Re-pinned 2026-09-27: ExperimentConfig gained ppo_seeds/ppo_seed_aggregation
# (D16), so to_dict() -- step 2 of the procedure -- now serializes two more
# keys. The procedure itself is unchanged: hashing these same fixed configs
# with those two keys dropped still reproduces the previous pin
# 78e865e0061b5e8ebca8020c62f5f8bffff125980a384d3696e34f70e7d3c226 exactly.
_PINNED_HASH = "dec52f1b5606d62cba7a785915c05eee32ba3886fe047362c129902322fed231"


def _fixed_resolved_configs():
    common = dict(
        seed=0,
        population_size=3,
        offspring_size=3,
        eval_budget=6,
        ppo_timesteps=100,
        ppo_eval_episodes=1,
        grid_shape=(5, 5),
        sentinel_fitness=-10.0,
        cache_enabled=False,
        use_crossover=False,
        min_voxels=3,
        min_actuators=1,
    )
    direct = ExperimentConfig(encoder="direct", task="Walker-v0", output_dir="unused/direct", **common)
    nca = ExperimentConfig(encoder="nca", task="Walker-v0", output_dir="unused/nca", **common)
    return {("direct", "Walker-v0"): direct, ("nca", "Walker-v0"): nca}


def test_compute_lock_hash_matches_pinned_value():
    resolved = _fixed_resolved_configs()
    assert compute_lock_hash(resolved) == _PINNED_HASH


def test_compute_lock_hash_is_independent_of_dict_iteration_order():
    resolved = _fixed_resolved_configs()
    reordered = dict(reversed(list(resolved.items())))
    assert compute_lock_hash(reordered) == compute_lock_hash(resolved) == _PINNED_HASH


def test_compute_lock_hash_changes_if_any_field_changes():
    resolved = _fixed_resolved_configs()
    baseline = compute_lock_hash(resolved)

    import dataclasses

    mutated = dict(resolved)
    mutated[("direct", "Walker-v0")] = dataclasses.replace(
        mutated[("direct", "Walker-v0")], eval_budget=7
    )
    assert compute_lock_hash(mutated) != baseline
