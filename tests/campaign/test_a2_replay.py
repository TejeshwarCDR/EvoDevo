"""The replay harness behind the 'bit-identical on the final code' check of Amendment 2."""
import json
from pathlib import Path

import pytest

from evodevo.campaign import replay_check
from evodevo.evaluation.ppo_runner import PPOTaskResult

ORIGINAL = Path("results/campaign/20260930T044002Z")


def test_the_default_items_are_the_two_gate_r_replays():
    assert [replay_check.parse_item(i) for i in replay_check.DEFAULT_ITEMS] == [
        ("Walker-v0", "direct", 1, 2), ("UpStepper-v0", "rule_based", 0, 299)]


class FakePool:
    """Returns the recorded per-seed fitness (plus ``drift``) instead of training."""
    instances = []

    def __init__(self, n_workers, plan, mode, **kw):
        self.n_workers, self.plan, self.queue = n_workers, plan, []
        FakePool.instances.append(self)

    drift = 0.0

    def submit_ppo_task(self, task, timeout_s=None):
        self.queue.append(task)

    def next_ppo_result(self, **kw):
        task = self.queue.pop(0)
        run = ORIGINAL / "runs" / task.run_key.split("/")[0] / task.task / task.run_key.split("/")[2]
        row = [json.loads(line) for line in (run / "run_log.jsonl").read_text().splitlines()][task.eval_index]
        fitness = row["per_seed_fitness"][task.ppo_k] + FakePool.drift
        return task, PPOTaskResult(fitness, True, None, 1.0, None, len(self.queue), [self.plan.slots[0][0]],
                                   None, None, None, None)

    def shutdown(self, wait=True):
        pass


@pytest.fixture
def fake_pool(monkeypatch):
    import evodevo.parallel.worker_pool as worker_pool
    FakePool.instances, FakePool.drift = [], 0.0
    monkeypatch.setattr(worker_pool, "EvaluationPool", FakePool)
    return FakePool


def test_identical_values_pass_and_every_training_gets_its_own_core_and_directory(fake_pool, tmp_path, capsys):
    status = replay_check.main(["--output-dir", str(tmp_path), "--cpus", "0,2,4"])
    assert status == 0
    report = json.loads((tmp_path / "compare.json").read_text())
    assert report["all_identical"] and len(report["items"]) == 2
    walker, upstepper = report["items"]
    assert walker["recorded_per_seed"] == walker["replayed_per_seed"] and len(walker["replayed_per_seed"]) == 1
    assert upstepper["recorded_per_seed"] == upstepper["replayed_per_seed"] and len(upstepper["replayed_per_seed"]) == 2
    pool = fake_pool.instances[0]
    assert pool.n_workers == 3 and [slot[0] for slot in pool.plan.slots] == [0, 2, 4]  # K=1 + K=2 trainings


def test_one_ulp_of_drift_is_a_mismatch(fake_pool, tmp_path):
    fake_pool.drift = 1e-12
    assert replay_check.main(["--output-dir", str(tmp_path)]) == 1
    assert json.loads((tmp_path / "compare.json").read_text())["all_identical"] is False


def test_replay_needs_one_cpu_per_training(fake_pool, tmp_path, capsys):
    assert replay_check.main(["--output-dir", str(tmp_path), "--cpus", "0,2"]) == 2
    assert "need 3 CPUs" in capsys.readouterr().err


def test_output_directories_are_unique_per_training(monkeypatch, fake_pool, tmp_path):
    seen = []
    original = FakePool.submit_ppo_task
    monkeypatch.setattr(FakePool, "submit_ppo_task", lambda self, task, timeout_s=None: (
        seen.append((task.artifact_dir, task.eval_index, task.ppo_k)), original(self, task, timeout_s)))
    replay_check.main(["--output-dir", str(tmp_path)])
    stems = {(directory, f"e{index:05d}_k{k}") for directory, index, k in seen}
    assert len(stems) == len(seen) == 3
