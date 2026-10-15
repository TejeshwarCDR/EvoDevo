"""Mechanical derivation of the extension decisions from synthetic inputs."""
import json
import math

import pytest

from evodevo.extension import derive as derive_module
from evodevo.extension import spec

LABELS = [spec.body_label(arm, kind) for arm in spec.ARMS for kind in spec.BODY_KINDS]


def _curve(level, knee):
    return [[math.ceil((i + 1) * 1e5 / 2048) * 2048, level * min(1.0, (i + 1) * 1e5 / knee)] for i in range(20)]


def make_preflight(root, label="PRODUCTION", noisy_task=None, cost=None, commits=None, drop_curve_row=False,
                   fail_curve=False):
    run = root / "preflight"
    run.mkdir()
    (run / "preflight_manifest.json").write_text(json.dumps(
        {"label": label, "params": {},
         "git_commits": commits or [{"git_commit": "abc123", "stage": "all", "dirty": False}]}))
    (run / "bodies.json").write_text(json.dumps({"bodies": {
        name: {"arm": name.split(":")[0], "kind": name.split(":")[1], "body": [[3] * 5] * 5,
               "body_hash": f"{index:064d}"} for index, name in enumerate(LABELS)}}))
    knees = {"ObstacleTraverser-v1": 1_200_000, "Climber-v0": 700_000, "CaveCrawler-v0": 900_000}
    decisions = {}
    from evodevo.extension.rules import decide_budget
    import dataclasses
    for task, knee in knees.items():
        curves = {name: _curve(1 + i, knee) for i, name in enumerate(LABELS)}
        decisions[task] = dataclasses.asdict(decide_budget(task, curves))
    (run / "budgets.json").write_text(json.dumps({"decisions": decisions}))
    curve_rows = []
    for task in spec.TASK_ORDER:
        for name in LABELS:
            bad = fail_curve and task == "Climber-v0" and name == LABELS[0]
            curve_rows.append({"key": f"curve|{task}|{name}", "stage": "curves", "task": task, "body": name,
                               "completed": not bad, "curve": [] if bad else _curve(1, 1e6),
                               "reason": "simulation_instability: boom" if bad else None, "error": None})
    if drop_curve_row:
        curve_rows.pop()
    (run / "curves.jsonl").write_text("".join(json.dumps(r) + "\n" for r in curve_rows))
    cost = cost or {"ObstacleTraverser-v1": 1.4, "Climber-v0": 1.3, "CaveCrawler-v0": 1.45}
    rows = []
    for i, name in enumerate(LABELS):
        base = 1000.0 + 50 * i
        for task, timesteps in spec.REFERENCE_TASKS.items():
            factor = 1.0 if task == "Walker-v0" else 1.15
            rows.append({"key": f"reference|{task}|{name}", "reason": None, "error": None,
                         "stage": "reference", "task": task, "body": name, "rep": 0, "timesteps": timesteps,
                         "completed": True, "fitness": 1.0, "wall_clock_s": base * factor * (2.3172 if i % 2 else 1),
                         "core_kind": "E" if i % 2 else "P"})
        for task in spec.TASK_ORDER:
            t = decisions[task]["ppo_timesteps"]
            for rep in range(4):
                fitness = (1 + i) + (0.01 * rep)
                if task == noisy_task and rep == 1 and i % 2 == 0:
                    fitness = 0.0
                rows.append({"key": f"noise|{task}|{name}|r{rep}", "reason": None, "error": None,
                             "stage": "noise", "task": task, "body": name, "rep": rep, "timesteps": t,
                             "completed": True, "fitness": fitness,
                             "wall_clock_s": base * cost[task] * t / 1e6 * (2.3172 if (i + rep) % 2 else 1),
                             "core_kind": "E" if (i + rep) % 2 else "P"})
    (run / "noise.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def make_reference(root, walker_h=132.0, upstepper_h=302.0):
    campaign = root / "campaign"
    (campaign / "runs" / "direct" / "Walker-v0" / "seed_0").mkdir(parents=True)
    (campaign / "runs" / "direct" / "Walker-v0" / "seed_0" / "blob").write_bytes(b"x" * 1000)
    (campaign / "manifest.json").write_text(json.dumps({"campaign_id": "ORIG", "status": "complete",
                                                       "git_commit": "deadbeef"}))
    metrics = {task: {"status": "complete", "makespan_s": h * 3600, "worker_count": 18,
                      "worker_utilization": 0.97, "worker_busy_s": h * 3600 * 18 * 0.97}
               for task, h in (("Walker-v0", walker_h), ("UpStepper-v0", upstepper_h))}
    (campaign / "phase_metrics.json").write_text(json.dumps(metrics))
    rows = []
    for arm, share in (("direct", 0.25), ("nca", 0.25), ("rule_based", 0.25), ("rule_based_monotone", 0.25)):
        for task in ("Walker-v0", "UpStepper-v0"):
            for i in range(10):
                rows.append({"run_key": f"{arm}/{task}/seed_0", "result": {"wall_clock_s": 1000.0 * share * 4}})
    (campaign / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return campaign


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    monkeypatch.setattr(spec, "DOCS_DIR", docs)
    monkeypatch.setattr(spec, "DECISIONS_JSON", docs / "EXT_DECISIONS.json")
    monkeypatch.setattr(spec, "DECISIONS_MD", docs / "EXT_DECISIONS.md")
    monkeypatch.setattr(spec, "ESTIMATE_JSON", docs / "final_estimate_ext.json")
    monkeypatch.setattr(spec, "CAMPAIGN_YAML", tmp_path / "campaign_ext.yaml")
    monkeypatch.setattr(spec, "REPO_ROOT", tmp_path)
    return tmp_path


def test_derivation_applies_every_rule(tmp_path, isolated_outputs):
    result = derive_module.derive(make_preflight(tmp_path), make_reference(tmp_path))
    record, estimate = result["record"], result["estimate"]
    assert record["status"] == "FINAL" and record["task_order"] == list(spec.TASK_ORDER)
    assert {t: record["tasks"][t]["ppo_timesteps"] for t in spec.TASK_ORDER} == {
        "ObstacleTraverser-v1": 1_200_000, "Climber-v0": 700_000, "CaveCrawler-v0": 900_000}
    assert all(record["tasks"][t]["ppo_seeds"] == 1 for t in spec.TASK_ORDER)
    ratios = estimate["inputs"]["cost_ratios"]
    # P-equivalent normalisation removes the E-core factor exactly.
    assert ratios["ObstacleTraverser-v1"]["ratio"] == pytest.approx(1.4 * 1.2)
    assert ratios["UpStepper-v0"]["ratio"] == pytest.approx(1.15)
    # Backtest: 132 h x 1.15 x K=2 = 303.6 h vs 302 h realised.
    assert estimate["backtest"]["predicted_upstepper_h"] == pytest.approx(132 * 1.15 * 2)
    assert estimate["backtest"]["relative_error"] == pytest.approx((303.6 - 302) / 302)
    band = estimate["result"]["phase_h"]["Climber-v0"]
    assert band["mean"] == pytest.approx(132 * 0.75 * 1.3 * 0.7)
    assert band["p95"] == pytest.approx(band["mean"] * 1.10)
    t_ppo = estimate["inputs"]["t_ppo_mean_sd_at_b3_footing"]
    assert set(t_ppo) == {f"{a}|{t}" for a in spec.ARMS for t in spec.TASK_ORDER}
    assert estimate["disk"]["three_x_total_bytes"] == pytest.approx(3 * estimate["disk"]["total_bytes"])


def test_collapse_noise_selects_k2_and_the_guard_may_remove_it(tmp_path, isolated_outputs):
    result = derive_module.derive(make_preflight(tmp_path, noisy_task="Climber-v0"), make_reference(tmp_path))
    climber = result["record"]["tasks"]["Climber-v0"]
    assert climber["d16_ext"]["ppo_seeds"] == 2
    steps = [s["step"] for s in result["record"]["guard"]["steps"]]
    if result["record"]["guard"]["steps"][0]["over"]:
        assert "a_K1_Climber-v0" in steps and climber["ppo_seeds"] == 1 and climber["guard_changed"]
    else:
        assert climber["ppo_seeds"] == 2


def test_test_preflight_is_refused_unless_allowed(tmp_path, isolated_outputs):
    run, reference = make_preflight(tmp_path, label="TEST"), make_reference(tmp_path)
    with pytest.raises(ValueError, match="TEST preflight run"):
        derive_module.derive(run, reference)
    assert derive_module.derive(run, reference, allow_test=True)["record"]["label"] == "TEST"


def test_incomplete_reference_campaign_is_refused(tmp_path, isolated_outputs):
    reference = make_reference(tmp_path)
    manifest = json.loads((reference / "manifest.json").read_text())
    manifest["status"] = "running"
    (reference / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not 'complete'"):
        derive_module.derive(make_preflight(tmp_path), reference)


def test_written_artifacts_are_consistent_and_tamper_evident(tmp_path, isolated_outputs, monkeypatch):
    run, reference = make_preflight(tmp_path), make_reference(tmp_path)
    derive_module.write(derive_module.derive(run, reference))
    text = spec.CAMPAIGN_YAML.read_text()
    assert "tasks: [ObstacleTraverser-v1, Climber-v0, CaveCrawler-v0]" in text
    assert "encoders: [direct, nca, rule_based]" in text and "decisions_path:" in text
    assert "    Climber-v0: 700000" in text
    record = json.loads(spec.DECISIONS_JSON.read_text())
    assert derive_module.render_yaml(record) == text
    assert "Backtest" in spec.DECISIONS_MD.read_text()
    # Hand-editing the generated YAML is detected by `check`.
    spec.CAMPAIGN_YAML.write_text(text.replace("Climber-v0: 700000", "Climber-v0: 800000"))
    problems = derive_module.check(None, None)
    assert "campaign_ext.yaml is not the rendering of EXT_DECISIONS.json" in problems


def _commits(*entries):
    return [{"git_commit": sha, "stage": stage, "dirty": dirty} for sha, stage, dirty in entries]


@pytest.mark.parametrize("commits, reason", [
    (_commits(("a", "bodies", False), ("a", "curves", True)), "dirty working tree"),
    (_commits(("a", "bodies", False), ("b", "curves", False)), "span 2 git commits"),
])
def test_dirty_or_multi_commit_runs_are_refused(tmp_path, isolated_outputs, commits, reason):
    run, reference = make_preflight(tmp_path, commits=commits), make_reference(tmp_path)
    with pytest.raises(ValueError, match=reason):
        derive_module.derive(run, reference)
    assert derive_module.derive(run, reference, allow_test=True)["record"]["preflight_git_commits"] == commits


def test_missing_trainings_are_refused_and_failures_are_recorded(tmp_path, isolated_outputs):
    reference = make_reference(tmp_path)
    run = make_preflight(tmp_path, drop_curve_row=True)
    with pytest.raises(ValueError, match="curves.jsonl has 17 recorded trainings, expected 18"):
        derive_module.derive(run, reference)


def test_failed_training_counts_as_recorded_and_is_reported(tmp_path, isolated_outputs):
    reference = make_reference(tmp_path)
    result = derive_module.derive(make_preflight(tmp_path, fail_curve=True), reference)
    record = result["record"]
    assert record["preflight_trainings"] == {"curves": 18, "noise": 84}
    assert record["preflight_failed_trainings"] == [
        f"curve|Climber-v0|{LABELS[0]}: simulation_instability: boom"]
    report = derive_module.render_report(record, result["estimate"])
    assert "## Preflight completeness" in report and "simulation_instability: boom" in report


def test_check_is_byte_for_byte(tmp_path, isolated_outputs):
    run, reference = make_preflight(tmp_path), make_reference(tmp_path)
    derive_module.write(derive_module.derive(run, reference))
    # (The plan-build check needs the real decision record under the real repo root.)
    assert [p for p in derive_module.check(run, reference) if not p.startswith("plan does not build")] == []
    # Whitespace-only hand edits of the JSON, the report and the YAML are all caught.
    original = spec.ESTIMATE_JSON.read_text()
    spec.ESTIMATE_JSON.write_text(original.replace('"method"', ' "method"', 1))
    assert any("canonical dump" in p for p in derive_module.check(None, None))
    spec.ESTIMATE_JSON.write_text(original)
    report = spec.DECISIONS_MD.read_text()
    spec.DECISIONS_MD.write_text(report + "\n")
    assert any("EXT_DECISIONS.md" in p for p in derive_module.check(None, None))
    spec.DECISIONS_MD.write_text(report)
    # A re-derivation from different inputs differs.
    (tmp_path / "x").mkdir()
    other = make_preflight(tmp_path / "x", noisy_task="Climber-v0")
    assert any("differs from a fresh derivation" in p for p in derive_module.check(other, reference))
