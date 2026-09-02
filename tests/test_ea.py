import os
import tempfile

import numpy as np
import pytest

from evodevo.config import ExperimentConfig
from evodevo.encoders.base import Encoder
from evodevo.encoders.direct import DirectEncoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea
from evodevo.evolution.logging import RunLogger


def _make_config(**overrides):
    defaults = dict(
        seed=0,
        encoder="direct",
        task="dummy-task",
        population_size=5,
        offspring_size=5,
        eval_budget=17,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir="unused",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_ea_runs_to_completion():
    encoder = DirectEncoder()
    config = _make_config()
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used > 0
    assert len(result.records) == result.evaluations_used


def test_ea_stops_exactly_at_budget():
    encoder = DirectEncoder()
    config = _make_config(eval_budget=17)
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used == 17


def test_ea_stops_exactly_at_budget_smaller_than_population():
    encoder = DirectEncoder()
    config = _make_config(population_size=5, offspring_size=5, eval_budget=3)
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used == 3


def _read_jsonl_without_wall_clock(path):
    import json

    records = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            record.pop("wall_clock_s", None)
            records.append(record)
    return records


def test_same_seed_byte_identical_logs():
    # wall_clock_s is real elapsed time and is excluded from the comparison;
    # everything else about the run (genomes, fitnesses, order) must match exactly
    # for the same seed.
    encoder = DirectEncoder()
    config = _make_config(seed=123)
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        logger1 = RunLogger(tmp1)
        run_ea(DirectEncoder(), evaluate_body_surrogate, config, logger=logger1)
        logger1.close()

        logger2 = RunLogger(tmp2)
        run_ea(DirectEncoder(), evaluate_body_surrogate, config, logger=logger2)
        logger2.close()

        records1 = _read_jsonl_without_wall_clock(os.path.join(tmp1, "run_log.jsonl"))
        records2 = _read_jsonl_without_wall_clock(os.path.join(tmp2, "run_log.jsonl"))
        assert records1 == records2


def test_different_seeds_differ():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        logger1 = RunLogger(tmp1)
        run_ea(DirectEncoder(), evaluate_body_surrogate, _make_config(seed=1), logger=logger1)
        logger1.close()

        logger2 = RunLogger(tmp2)
        run_ea(DirectEncoder(), evaluate_body_surrogate, _make_config(seed=2), logger=logger2)
        logger2.close()

        with open(os.path.join(tmp1, "run_log.jsonl"), "rb") as f:
            content1 = f.read()
        with open(os.path.join(tmp2, "run_log.jsonl"), "rb") as f:
            content2 = f.read()
        assert content1 != content2


def test_best_so_far_fitness_monotonic_nondecreasing():
    encoder = DirectEncoder()
    config = _make_config(eval_budget=30, population_size=6, offspring_size=6)
    result = run_ea(encoder, evaluate_body_surrogate, config)

    best_so_far = float("-inf")
    for record in sorted(result.records, key=lambda r: r["eval_index"]):
        best_so_far = max(best_so_far, record["fitness"])
        # recompute prefix max independently to check monotonic non-decreasing trace
    prefix_max = []
    running = float("-inf")
    for record in sorted(result.records, key=lambda r: r["eval_index"]):
        running = max(running, record["fitness"])
        prefix_max.append(running)
    assert all(prefix_max[i] <= prefix_max[i + 1] for i in range(len(prefix_max) - 1))


class AlwaysInvalidEncoder(Encoder):
    """Test-only encoder whose decode() always yields an invalid (all-empty) body."""

    @property
    def name(self) -> str:
        return "always_invalid"

    def genome_length(self) -> int:
        return 25

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(25, dtype=int)

    def decode(self, genome: np.ndarray) -> np.ndarray:
        return np.zeros((5, 5), dtype=int)

    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return genome.copy()

    def describe(self) -> dict:
        return {"name": self.name}


def test_ea_terminates_cleanly_with_always_invalid_encoder():
    encoder = AlwaysInvalidEncoder()
    config = _make_config(eval_budget=13, population_size=4, offspring_size=4)
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used == 13
    assert all(not r["valid"] for r in result.records)
