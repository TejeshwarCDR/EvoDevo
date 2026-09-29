from __future__ import annotations

import argparse

from evodevo.campaign import launch_checks, production


def _args(**changes):
    values = {
        "mode": "real", "no_notify": True, "no_mirror": True,
        "production_power": True, "cpu_profile": production.PROFILE,
        "workers": len(production.WORKER_CPUS), "local_config": None,
        "output_root": "results/campaign", "mirror_dir": None,
        "allow_same_disk_mirror": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_production_power_check_records_verified_state(monkeypatch):
    observed = {"turbo_enabled": True, "no_turbo": 0,
                "p_core_max_khz": {str(cpu): production.P_MAX_KHZ for cpu in range(16)},
                "pl1_uw": production.PL1_UW, "pl2_uw": production.PL2_UW}
    monkeypatch.setattr(launch_checks, "power_refusals", lambda: ([], observed))
    refusals, record = launch_checks.launch_refusals(_args())
    assert refusals == []
    assert record["production_power"] == {"required": True, **observed}


def test_production_power_check_fails_closed_for_wrong_topology_or_control(monkeypatch):
    monkeypatch.setattr(launch_checks, "power_refusals", lambda: (["PL1 is wrong"], {"pl1_uw": 1}))
    refusals, record = launch_checks.launch_refusals(_args(workers=10, cpu_profile="e-cores"))
    assert "production launch requires CPU profile 'thermal-validated'" in refusals
    assert "production launch requires exactly 18 workers" in refusals
    assert "PL1 is wrong" in refusals
    assert record["production_power"]["pl1_uw"] == 1


def test_power_refusals_requires_every_selected_control(monkeypatch):
    state = {"turbo_enabled": True, "no_turbo": 0,
             "p_core_max_khz": {str(cpu): production.P_MAX_KHZ for cpu in range(16)},
             "pl1_uw": production.PL1_UW, "pl2_uw": production.PL2_UW}
    monkeypatch.setattr(production, "power_state", lambda: state)
    assert production.power_refusals() == ([], state)
    state["p_core_max_khz"]["3"] = 5_500_000
    reasons, _ = production.power_refusals()
    assert len(reasons) == 1 and "policies 0-15" in reasons[0]
