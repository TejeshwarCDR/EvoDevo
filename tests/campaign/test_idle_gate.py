"""Step-5 machine-idle gate: remote-operation baseline vs competing work."""
from types import SimpleNamespace

from evodevo.campaign.idle_gate import ActivitySample, assess_samples, classify_process_cpu
from evodevo.campaign import launch_checks


def _sample(system=3.5, operator=95.0, competing=1.0):
    return ActivitySample(
        system_percent=system,
        operator_cpu_percent=operator,
        competing_cpu_percent=competing,
        top_operator=(("gnome-shell", operator),),
        top_competing=(("python", competing),),
    )


def test_required_remote_desktop_baseline_is_accepted():
    refusals, record = assess_samples([_sample(), _sample(), _sample()])
    assert refusals == []
    assert record["passed"] is True
    assert record["operator_process_names"] == [
        "anydesk", "claude", "code", "codex", "gnome-shell", "xorg"]


def test_sustained_single_core_competitor_is_rejected():
    refusals, record = assess_samples([
        _sample(system=7.0, competing=100.0),
        _sample(system=7.1, competing=99.0),
        _sample(system=7.0, competing=101.0),
    ])
    assert refusals and "competing process CPU" in refusals[0]
    assert record["passed"] is False


def test_unattributed_multicore_load_is_rejected():
    refusals, _record = assess_samples([
        _sample(system=30.0), _sample(system=29.0), _sample(system=31.0)])
    assert any("total system CPU" in refusal for refusal in refusals)


def test_runaway_allowlisted_stack_is_not_a_bypass():
    refusals, _record = assess_samples([
        _sample(system=10.0, operator=210.0),
        _sample(system=10.0, operator=205.0),
        _sample(system=10.0, operator=215.0),
    ])
    assert any("operator baseline CPU" in refusal for refusal in refusals)


def test_one_transient_spike_does_not_fail_sustained_gate():
    refusals, record = assess_samples([
        _sample(), _sample(system=30.0, competing=500.0), _sample()])
    assert refusals == []
    assert record["passed"] is True


def test_process_classification_is_exact_and_case_insensitive():
    operator, competing, top_operator, top_competing = classify_process_cpu([
        ("Xorg", 30.0), ("code", 5.0), ("python", 90.0),
        ("code-helper-impostor", 70.0),
    ])
    assert operator == 35.0
    assert competing == 160.0
    assert top_operator[0] == ("xorg", 30.0)
    assert top_competing[0] == ("python", 90.0)


def test_validation_launch_check_enforces_idle_gate(monkeypatch):
    monkeypatch.setattr(launch_checks, "notifier_status", lambda: {
        "config_path": "/tmp/n", "configured": False, "channels": []})
    monkeypatch.setattr(launch_checks, "mirror_check", lambda _args: ([], {"ok": True}))
    monkeypatch.setattr(launch_checks, "check_machine_idle", lambda: (
        ["machine has competing work"], {"required": True, "passed": False}))
    args = SimpleNamespace(mode="validation", no_notify=False, no_mirror=True,
                           output_root="results", mirror_dir=None)
    refusals, record = launch_checks.launch_refusals(args)
    assert refusals == ["machine has competing work"]
    assert record["machine_idle"] == {"required": True, "passed": False}


def test_nonvalidation_launch_does_not_run_step5_gate(monkeypatch):
    monkeypatch.setattr(launch_checks, "notifier_status", lambda: {
        "config_path": "/tmp/n", "configured": True, "channels": ["shell"]})
    monkeypatch.setattr(launch_checks, "mirror_check", lambda _args: ([], {"ok": True}))
    monkeypatch.setattr(launch_checks, "check_machine_idle", lambda: (_ for _ in ()).throw(
        AssertionError("Step-5 gate must only run for validation mode")))
    args = SimpleNamespace(mode="surrogate", no_notify=False, no_mirror=True,
                           output_root="results", mirror_dir=None)
    refusals, record = launch_checks.launch_refusals(args)
    assert refusals == []
    assert record["machine_idle"] == {"required": False}
