"""The validated production machine contract.

This module is deliberately stdlib-only: it is used by launcher refusal
checks before a scheduler or a worker can import the PPO stack.  The values
are not tuning defaults; they are the specific configuration validated in
``results/thermal_worker_study/FINAL_SELECTION_20260930.md``.
"""
from __future__ import annotations

from pathlib import Path

from evodevo.parallel.topology import AffinityPlan, Topology, allocation_order


PROFILE = "thermal-validated"
WORKER_CPUS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
P_WORKER_CPUS = WORKER_CPUS[:8]
E_WORKER_CPUS = WORKER_CPUS[8:]
SCHEDULER_CPUS = (26, 27)
P_MAX_KHZ = 4_200_000
PL1_UW = 125_000_000
PL2_UW = 160_000_000


def contract() -> dict:
    """Serializable configuration identity persisted in production manifests."""
    return {
        "profile": PROFILE,
        "workers": len(WORKER_CPUS),
        "p_worker_cpus": list(P_WORKER_CPUS),
        "e_worker_cpus": list(E_WORKER_CPUS),
        "scheduler_cpus": list(SCHEDULER_CPUS),
        "p_max_khz": P_MAX_KHZ,
        "pl1_uw": PL1_UW,
        "pl2_uw": PL2_UW,
        "turbo_enabled": True,
    }


def _require_topology(topology: Topology) -> None:
    cores = {core.logical_id: core for core in topology.cores}
    missing = sorted(set(WORKER_CPUS + SCHEDULER_CPUS) - set(cores))
    if missing:
        raise ValueError(f"{PROFILE} profile requires logical CPUs {missing}, which are unavailable")
    bad_p = [cpu for cpu in P_WORKER_CPUS
             if cores[cpu].kind != "P" or not cores[cpu].is_physical_primary]
    bad_e = [cpu for cpu in E_WORKER_CPUS + SCHEDULER_CPUS if cores[cpu].kind != "E"]
    if bad_p or bad_e:
        raise ValueError(
            f"{PROFILE} profile requires P primaries {list(P_WORKER_CPUS)} and E CPUs "
            f"{list(E_WORKER_CPUS + SCHEDULER_CPUS)}; topology classification disagrees "
            f"(bad P={bad_p}, bad E={bad_e})"
        )


def placement(topology: Topology, workers: int) -> tuple[AffinityPlan, list[int]]:
    """Return the only worker/control placement validated for production."""
    if workers != len(WORKER_CPUS):
        raise ValueError(f"{PROFILE} profile requires exactly {len(WORKER_CPUS)} workers, got {workers}")
    _require_topology(topology)
    return AffinityPlan(
        n_workers=workers,
        slots=tuple((cpu,) for cpu in WORKER_CPUS),
        oversubscribed=False,
        allocation_order=tuple(allocation_order(topology)),
    ), list(SCHEDULER_CPUS)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def power_state() -> dict:
    """Read the kernel state that must be true before production launch."""
    pstate = Path("/sys/devices/system/cpu/intel_pstate")
    rapl = Path("/sys/class/powercap/intel-rapl:0")
    frequencies = {
        str(cpu): _read_int(Path(f"/sys/devices/system/cpu/cpufreq/policy{cpu}/scaling_max_freq"))
        for cpu in range(16)
    }
    return {
        "turbo_enabled": _read_int(pstate / "no_turbo") == 0,
        "no_turbo": _read_int(pstate / "no_turbo"),
        "p_core_max_khz": frequencies,
        "pl1_uw": _read_int(rapl / "constraint_0_power_limit_uw"),
        "pl2_uw": _read_int(rapl / "constraint_1_power_limit_uw"),
    }


def power_refusals() -> tuple[list[str], dict]:
    """Return fail-closed reasons and an auditable read-only state record."""
    state = power_state()
    refusals = []
    if state["no_turbo"] != 0:
        refusals.append(f"Turbo must be enabled (intel_pstate/no_turbo=0), observed {state['no_turbo']!r}")
    wrong = {cpu: value for cpu, value in state["p_core_max_khz"].items() if value != P_MAX_KHZ}
    if wrong:
        refusals.append(f"P-core policies 0-15 must be capped at {P_MAX_KHZ} kHz, observed {wrong}")
    if state["pl1_uw"] != PL1_UW:
        refusals.append(f"RAPL PL1 must be {PL1_UW} uW, observed {state['pl1_uw']!r}")
    if state["pl2_uw"] != PL2_UW:
        refusals.append(f"RAPL PL2 must be {PL2_UW} uW, observed {state['pl2_uw']!r}")
    return refusals, state
