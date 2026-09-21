"""D5 B1: the machine profile.

Stage `machine_profile` -- cost "free" (pure introspection plus one
sequential-disk-write measurement; no PPO, no simulator, no worker pool).
Everything here is either read directly off the OS/hardware or measured in
this one process; nothing is assumed. In particular, D13's risk register
(swap use, ulimits, sdb1, the NVIDIA driver, sleep targets, unattended-
upgrades, coretemp) is recorded here as VALUES this run actually observed,
not as a plan someone wrote down earlier -- a D13 row that only cites a
prior setup document is not this stage's job; re-measuring it is.

Deliberately never imports numpy/torch/evogym/psutil at module scope (see
evodevo/parallel/threading_control.py and worker_pool.py for why this
discipline matters elsewhere in this project, and evodevo/preflight/
__main__.py's own docstring for why --dry-run/--help must stay fast and
free of heavy or optional dependencies) -- every such import here is lazy,
inside the function that needs it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

from evodevo.parallel.threading_control import THREAD_ENV_VARS
from evodevo.preflight.run_dir import atomic_write_json

DEFAULT_DISK_PROBE_MB = 512


class MachineProfileStage:
    name = "machine_profile"
    description = (
        "Machine profile (D5 B1): topology, RAM/disk incl. measured write rate, "
        "unconfigured-default thread behaviour, environment provenance, accelerator "
        "presence, CPU model/freq/governor, and D13 risk state."
    )
    estimated_cost = "free"
    depends_on = ()

    def __init__(self, disk_probe_mb: int = DEFAULT_DISK_PROBE_MB, self_check_stamp=None):
        self.disk_probe_mb = disk_probe_mb
        self.self_check_stamp = self_check_stamp  # set only by --self-check

    def run(self, ctx) -> dict:
        profile = {
            "topology": _topology_summary(ctx.topology),
            "memory_disk": _memory_disk_summary(ctx.run_dir.path, self.disk_probe_mb),
            "unconfigured_threading": _probe_unconfigured_threading(),
            "environment": _environment_provenance(),
            "accelerator": _accelerator_summary(),
            "cpu": _cpu_summary(ctx.topology),
            "risk_state": _risk_state_summary(),
            "self_check_stamp": self.self_check_stamp,
        }

        # Merge into machine.json (which already holds the static snapshot
        # write_machine_snapshot() wrote at run-dir creation) rather than
        # clobber it -- both are "what this machine looked like", just at
        # different levels of detail/cost.
        machine_path = ctx.run_dir.machine_path
        try:
            with open(machine_path) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}
        existing["b1_machine_profile"] = profile
        atomic_write_json(machine_path, existing)

        ctx.events.record(self.name, "measurement", {"disk_write_rate_mb_s": profile["memory_disk"]["disk_write_rate_mb_s"]})

        return profile


# --- topology ----------------------------------------------------------


def _topology_summary(topology) -> dict:
    p_primaries = sorted(c.logical_id for c in topology.cores if c.kind == "P" and c.is_physical_primary)
    e_cores = sorted(c.logical_id for c in topology.cores if c.kind == "E")
    return {
        "logical_count": topology.logical_count,
        "physical_count": topology.physical_count,
        "p_core_physical_count": len(p_primaries),
        "p_core_physical_ids": p_primaries,
        "p_core_logical_ids": sorted(topology.p_core_ids),
        "e_core_count": len(e_cores),
        "e_core_ids": e_cores,
        "hybrid": topology.hybrid,
        "detection_method": topology.detection_method,
    }


# --- memory / disk -------------------------------------------------------


def _sdb1_status() -> dict:
    mount_point = None
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "/dev/sdb1":
                    mount_point = parts[1]
                    break
    except OSError:
        pass
    if mount_point is None:
        return {"mounted": False, "mount_point": None, "free_gb": None}
    try:
        usage = shutil.disk_usage(mount_point)
        return {"mounted": True, "mount_point": mount_point, "free_gb": round(usage.free / 1e9, 2)}
    except OSError:
        return {"mounted": True, "mount_point": mount_point, "free_gb": None}


def _measure_disk_write_rate(run_dir_path: str, size_mb: int) -> Optional[float]:
    """Writes a size_mb temp file into run_dir_path, fsyncs, times only the
    write+fsync, then deletes it. The temp file is ALWAYS removed in a
    `finally`, even if the write itself raises partway through (disk full,
    permission change mid-run, an injected fault in a test) -- this probe
    must never leave debris in a run directory that's meant to survive for
    half a day."""
    path = os.path.join(run_dir_path, f".b1_disk_probe_{os.getpid()}.tmp")
    chunk = os.urandom(1024 * 1024)  # 1 MiB of incompressible data per write
    try:
        start = time.time()
        with open(path, "wb") as f:
            for _ in range(size_mb):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.time() - start
        if elapsed <= 0:
            return None
        return round(size_mb / elapsed, 2)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _memory_disk_summary(run_dir_path: str, disk_probe_mb: int) -> dict:
    result: dict = {}
    try:
        import psutil

        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        result["ram_total_mb"] = round(vmem.total / 1e6, 1)
        result["ram_available_mb"] = round(vmem.available / 1e6, 1)
        result["swap_total_mb"] = round(swap.total / 1e6, 1)
        result["swap_used_mb"] = round(swap.used / 1e6, 1)
    except Exception as e:  # noqa: BLE001 -- a missing psutil metric degrades to null
        result["ram_total_mb"] = None
        result["ram_available_mb"] = None
        result["swap_total_mb"] = None
        result["swap_used_mb"] = None
        result["memory_probe_error"] = f"{type(e).__name__}: {e}"

    try:
        result["disk_free_root_gb"] = round(shutil.disk_usage("/").free / 1e9, 2)
    except OSError:
        result["disk_free_root_gb"] = None

    try:
        result["disk_free_run_dir_gb"] = round(shutil.disk_usage(run_dir_path).free / 1e9, 2)
    except OSError:
        result["disk_free_run_dir_gb"] = None

    sdb1 = _sdb1_status()
    result["sdb1_mounted"] = sdb1["mounted"]
    result["sdb1_mount_point"] = sdb1["mount_point"]
    result["sdb1_free_gb"] = sdb1["free_gb"]

    result["disk_probe_mb"] = disk_probe_mb
    result["disk_write_rate_mb_s"] = _measure_disk_write_rate(run_dir_path, disk_probe_mb)

    return result


# --- unconfigured-default thread probe --------------------------------


def _probe_unconfigured_threading() -> dict:
    """Launches a bare `python -c` subprocess with THREAD_ENV_VARS
    stripped from its environment and configure_threading() never called,
    and asks it what torch.get_num_threads() actually defaults to. This is
    the oversubscription risk evodevo.parallel.threading_control's
    configure_threading(1) exists to fix, documented here as a MEASURED
    number rather than an assertion that the risk existed."""
    probe_code = (
        "import json\n"
        "import torch\n"
        "import os\n"
        f"env_vars = {list(THREAD_ENV_VARS)!r}\n"
        "print(json.dumps({'torch_num_threads': torch.get_num_threads(), "
        "'env': {v: os.environ.get(v) for v in env_vars}}))\n"
    )
    env = os.environ.copy()
    for var in THREAD_ENV_VARS:
        env.pop(var, None)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if completed.returncode != 0:
            return {
                "torch_num_threads": None,
                "env": None,
                "error": f"probe subprocess exited {completed.returncode}: {completed.stderr[-2000:]}",
            }
        last_line = completed.stdout.strip().splitlines()[-1]
        data = json.loads(last_line)
        return data
    except Exception as e:  # noqa: BLE001 -- this probe must never fail the whole stage
        return {"torch_num_threads": None, "env": None, "error": f"{type(e).__name__}: {e}"}


# --- environment provenance ---------------------------------------------


def _library_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def _evogym_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", "external/evogym", "rev-parse", "--short=8", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _evogym_version() -> Optional[str]:
    try:
        import importlib.metadata

        return importlib.metadata.version("evogym")
    except Exception:
        return None


def _safe_lock_hash() -> dict:
    try:
        from evodevo.experiments.lock_hash import compute_lock_hash_from_benchmark_config

        return {"value": compute_lock_hash_from_benchmark_config(), "error": None}
    except Exception as e:  # noqa: BLE001 -- record the failure, don't fail B1 over it
        return {"value": None, "error": f"{type(e).__name__}: {e}"}


def _environment_provenance() -> dict:
    from evodevo.core.provenance import git_commit_hash, git_dirty_files

    dirty_files = git_dirty_files()
    lock_hash = _safe_lock_hash()

    return {
        "python_version": sys.version,
        "numpy_version": _library_version("numpy"),
        "torch_version": _library_version("torch"),
        "gymnasium_version": _library_version("gymnasium"),
        "stable_baselines3_version": _library_version("stable_baselines3"),
        "evogym_version": _evogym_version(),
        "evogym_commit": _evogym_commit(),
        "project_git_commit": git_commit_hash(),
        "project_dirty": bool(dirty_files),
        "project_dirty_files": dirty_files,
        "lock_hash": lock_hash["value"],
        "lock_hash_error": lock_hash["error"],
    }


# --- accelerator ----------------------------------------------------------


def _accelerator_summary() -> dict:
    """Documentation only -- evodevo/evaluation/ppo_runner.py never passes
    device= explicitly to stable_baselines3's PPO(), so SB3's own
    device="auto" default governs; that resolves to cuda if available,
    else cpu, which is reproduced here (not imported from SB3, since SB3
    only resolves it lazily inside PPO's own constructor) purely to report
    what it WOULD pick. Nothing here changes what device is actually used."""
    try:
        import torch
    except ImportError:
        return {
            "torch_importable": False, "cuda_available": None,
            "device_name": None, "ppo_runner_device": None,
        }

    cuda_available = torch.cuda.is_available()
    device_name = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = None

    return {
        "torch_importable": True,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "ppo_runner_device": "cuda" if cuda_available else "cpu",
    }


# --- CPU model / frequency / governor -----------------------------------


def _cpu_model_name() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _read_int_file(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpu_freq_policy(topology) -> dict:
    """A measurement whose frequency policy is unrecorded cannot be
    reproduced: two runs at the same "governor=powersave" can still behave
    completely differently depending on the scaling driver, whether
    turbo/HWP boost is enabled, and per-core energy/performance hints
    (especially on a P/E hybrid part, where P- and E-cores can carry
    different EPP values). All of this is read fresh, per B1 run -- never
    assumed identical to a prior run -- and B7's D13 risk table carries a
    note that this policy must be IDENTICAL between the pre-flight and the
    real campaign, not just "close enough"."""
    p_core = next((c for c in topology.cores if c.kind == "P"), None)
    e_core = next((c for c in topology.cores if c.kind == "E"), None)

    def _epp_for(core) -> Optional[str]:
        if core is None:
            return None
        return _read_text_file(f"/sys/devices/system/cpu/cpu{core.logical_id}/cpufreq/energy_performance_preference")

    def _cur_freq_khz_for(logical_id: int) -> Optional[int]:
        return _read_int_file(f"/sys/devices/system/cpu/cpu{logical_id}/cpufreq/scaling_cur_freq")

    no_turbo_raw = _read_text_file("/sys/devices/system/cpu/intel_pstate/no_turbo")
    idle_freqs_mhz = {}
    for core in (p_core, e_core):
        if core is None:
            continue
        freq_khz = _cur_freq_khz_for(core.logical_id)
        idle_freqs_mhz[f"{core.kind}_core_cpu{core.logical_id}"] = freq_khz / 1000 if freq_khz is not None else None

    return {
        "scaling_driver": _read_text_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
        "energy_performance_preference_p_core": _epp_for(p_core),
        "energy_performance_preference_e_core": _epp_for(e_core),
        "epp_p_core_cpu_id": p_core.logical_id if p_core is not None else None,
        "epp_e_core_cpu_id": e_core.logical_id if e_core is not None else None,
        "intel_pstate_no_turbo": (None if no_turbo_raw is None else no_turbo_raw == "1"),
        "intel_pstate_status": _read_text_file("/sys/devices/system/cpu/intel_pstate/status"),
        "idle_freq_mhz_by_core": idle_freqs_mhz,
    }


def _cpu_summary(topology) -> dict:
    base_freq_khz = _read_int_file("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")
    max_freq_khz = max((c.max_freq_khz for c in topology.cores if c.max_freq_khz), default=None)
    governor = None
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            governor = f.read().strip()
    except OSError:
        pass

    return {
        "model": _cpu_model_name(),
        "base_freq_mhz": base_freq_khz / 1000 if base_freq_khz else None,
        "max_freq_mhz": max_freq_khz / 1000 if max_freq_khz else None,
        "governor": governor,
        "freq_policy": _cpu_freq_policy(topology),
    }


# --- D13 risk state -----------------------------------------------------


def _nvidia_driver_loaded() -> bool:
    return os.path.isdir("/sys/module/nvidia") or os.path.isdir("/proc/driver/nvidia")


def _sleep_target_states() -> dict:
    targets = ("sleep.target", "suspend.target", "hibernate.target", "hybrid-sleep.target")
    result = {}
    for target in targets:
        try:
            completed = subprocess.run(
                ["systemctl", "is-enabled", target], capture_output=True, text=True, timeout=10,
            )
            result[target] = (completed.stdout.strip() or completed.stderr.strip()) or None
        except Exception as e:  # noqa: BLE001 -- record as unknown, never fail the stage
            result[target] = f"unknown: {e}"
    return result


def _unattended_upgrades_enabled() -> Optional[bool]:
    try:
        completed = subprocess.run(
            ["systemctl", "is-enabled", "unattended-upgrades.service"],
            capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip() == "enabled"
    except Exception:
        return None


def _coretemp_readable() -> bool:
    try:
        import psutil

        temps = psutil.sensors_temperatures()
        return bool(temps.get("coretemp"))
    except Exception:
        return False


def _process_ulimits() -> dict:
    """As seen by THIS (the pre-flight) process right now -- not whatever
    a launcher script or a prior setup session configured elsewhere. See
    evodevo/parallel/worker_pool.py's own launcher-script ulimit -n
    additions for the config-time counterpart to this runtime measurement."""
    try:
        import resource

        soft_n, hard_n = resource.getrlimit(resource.RLIMIT_NOFILE)
        soft_c, hard_c = resource.getrlimit(resource.RLIMIT_CORE)
        return {"nofile_soft": soft_n, "nofile_hard": hard_n, "core_soft": soft_c, "core_hard": hard_c}
    except Exception as e:  # noqa: BLE001
        return {"nofile_soft": None, "nofile_hard": None, "core_soft": None, "core_hard": None, "error": str(e)}


def _risk_state_summary() -> dict:
    try:
        import psutil

        swap = psutil.swap_memory()
        swap_total_mb = round(swap.total / 1e6, 1)
        swap_used_mb = round(swap.used / 1e6, 1)
        swap_in_use = swap.used > 0
    except Exception:
        swap_total_mb = swap_used_mb = swap_in_use = None

    ulimits = _process_ulimits()

    return {
        "swap_total_mb": swap_total_mb,
        "swap_used_mb": swap_used_mb,
        # Any swap use during the campaign is an anomaly, not a cushion --
        # recorded plainly so a reader doesn't have to infer it from the
        # raw MB figure.
        "swap_in_use": swap_in_use,
        "ulimit_n_soft": ulimits["nofile_soft"],
        "ulimit_n_hard": ulimits["nofile_hard"],
        "ulimit_c_soft": ulimits["core_soft"],
        "ulimit_c_hard": ulimits["core_hard"],
        "sdb1_mounted": _sdb1_status()["mounted"],
        "nvidia_driver_loaded": _nvidia_driver_loaded(),
        "sleep_targets": _sleep_target_states(),
        "unattended_upgrades_enabled": _unattended_upgrades_enabled(),
        "coretemp_readable": _coretemp_readable(),
    }
