"""Bit-identity regression tests for the proposed-generation EA API."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.nca import NCAEncoder
from evodevo.encoders.rule_based import RuleBasedEncoder
from evodevo.evaluation.ppo_runner import evaluate_body, evaluate_body_surrogate
from evodevo.evolution.ea import EvolutionRun, run_ea
from evodevo.evolution.logging import RunLogger

# The immediately preceding commit is deliberately pinned: this test compares
# the refactor against the old implementation, not against a second copy that
# can silently be edited to agree with it.
PRE_PROPOSE_TELL_EA_COMMIT = "bfb913917b2603acef310f44fdfb491fea33564a"


def _config(arm, seed=0, budget=300, **overrides):
    values = dict(seed=seed, encoder=arm, task="Walker-v0", population_size=15,
                  offspring_size=15, eval_budget=budget, ppo_timesteps=1,
                  ppo_eval_episodes=1, output_dir="unused")
    values.update(overrides)
    return ExperimentConfig(**values)


def _encoder(arm):
    factors = ExperimentFactors(dev_steps=1, init_condition="center_seed", seed=0)
    if arm == "direct":
        return DirectEncoder(factors=factors)
    if arm == "nca":
        return NCAEncoder(factors=factors)
    if arm == "rule_based":
        return RuleBasedEncoder(factors=factors)
    if arm == "rule_based_monotone":
        return RuleBasedEncoder(factors=factors, action_monotonicity="monotone")
    raise AssertionError(arm)


ARMS = ("direct", "nca", "rule_based", "rule_based_monotone")


def _old_run_ea():
    source = subprocess.check_output(
        ["git", "show", f"{PRE_PROPOSE_TELL_EA_COMMIT}:evodevo/evolution/ea.py"], text=True,
    )
    spec = importlib.util.spec_from_loader("pinned_pre_propose_tell_ea", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "pinned_pre_propose_tell_ea.py", "exec"), module.__dict__)
    return module.run_ea


def _without_clock(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        row.pop("wall_clock_s", None)
        rows.append(row)
    return rows


def _outcome(encoder, result):
    records = [{k: v for k, v in row.items() if k != "wall_clock_s"} for row in result.records]
    return (
        [(encoder.serialize_genome(i.genome), i.genome_id, i.parent_id, i.fitness, i.valid)
         for i in result.population],
        result.evaluations_used, result.best_fitness, records, result.generation_records,
    )


def _evaluate(candidates, config):
    results = []
    for candidate in candidates:
        result = evaluate_body_surrogate(candidate.body, task=config.task, seed=config.seed)
        result.eval_index = candidate.eval_index
        results.append(result)
    return results


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("seed", range(5))
def test_pinned_pre_refactor_surrogate_bit_identity(tmp_path, arm, seed):
    old = _old_run_ea()
    config = _config(arm, seed)
    old_logger = RunLogger(str(tmp_path / "old"))
    new_logger = RunLogger(str(tmp_path / "new"))
    try:
        old_result = old(_encoder(arm), evaluate_body_surrogate, config, logger=old_logger)
        new_result = run_ea(_encoder(arm), evaluate_body_surrogate, config, logger=new_logger)
    finally:
        old_logger.close()
        new_logger.close()
    assert _without_clock(tmp_path / "old" / "run_log.jsonl") == _without_clock(tmp_path / "new" / "run_log.jsonl")
    assert _outcome(_encoder(arm), old_result) == _outcome(_encoder(arm), new_result)


@pytest.mark.parametrize("arm", ARMS)
def test_tell_result_order_is_independent(arm):
    config = _config(arm, budget=30)
    ordered, shuffled = EvolutionRun(_encoder(arm), config), EvolutionRun(_encoder(arm), config)
    while not ordered.done:
        left, right = ordered.propose(), shuffled.propose()
        left_results = _evaluate(left, config)
        right_results = _evaluate(right, config)
        ordered.tell(left_results)
        shuffled.tell(list(reversed(right_results)))
    assert _outcome(_encoder(arm), ordered.result()) == _outcome(_encoder(arm), shuffled.result())


@pytest.mark.parametrize("arm", ARMS)
def test_generation_boundary_checkpoint_round_trip(arm):
    config = _config(arm, budget=300)
    uninterrupted = EvolutionRun(_encoder(arm), config)
    checkpoints = []
    while not uninterrupted.done:
        candidates = uninterrupted.propose()
        uninterrupted.tell(_evaluate(candidates, config))
        checkpoints.append(uninterrupted.get_state())
        # The public checkpoint contract is portable JSON, not merely a
        # Python dict that happens to work until a process restart.
        json.dumps(checkpoints[-1])
    expected = _outcome(_encoder(arm), uninterrupted.result())
    for state in checkpoints:
        restored = EvolutionRun.from_state(_encoder(arm), config, state)
        while not restored.done:
            candidates = restored.propose()
            restored.tell(_evaluate(candidates, config))
        assert _outcome(_encoder(arm), restored.result()) == expected


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_pinned_pre_refactor_real_mode_bit_identity(arm):
    old = _old_run_ea()
    config = _config(arm, budget=30, ppo_timesteps=2000)
    old_result = old(_encoder(arm), evaluate_body, config)
    new_result = run_ea(_encoder(arm), evaluate_body, config)
    old_rows = [(r["fitness"], r["valid"], r["reason"], r["body_hash"]) for r in old_result.records]
    new_rows = [(r["fitness"], r["valid"], r["reason"], r["body_hash"]) for r in new_result.records]
    assert old_rows == new_rows
