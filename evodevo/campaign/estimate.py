"""Which capacity estimate a campaign uses, and how its priors are read.

The original campaign (Walker-v0, UpStepper-v0) was planned from the
gitignored Monte Carlo output at ``results/campaign_prep/final_estimate/
final_estimate.json`` ("legacy" format: ``walker_phase_h``/``upstepper_phase_h``
keys inside a guard-sequence result). A campaign launched from a benchmark
YAML with ``estimate_path`` records that path in its manifest, and every
reader (dashboard, monitor, dry run) uses it.

The v2 format (``"format": "evodevo-estimate-v2"``) is task-generic:

    {"format": "evodevo-estimate-v2",
     "inputs": {"t_ppo_mean_sd_at_b3_footing": {"<arm>|<task>": [mean_s, sd_s], ...}},
     "result": {"task_order": [...],
                "phase_h": {"<task>": {"p5": h, "mean": h, "p95": h}, ...},
                "total_h": {"p5": h, "mean": h, "p95": h}},
     "guard": {"threshold_h": h, ...},
     "disk": {"total_bytes": n, "three_x_total_bytes": n}}

Nothing here changes how the legacy file is interpreted.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ESTIMATE_PATH = REPO_ROOT / "results/campaign_prep/final_estimate/final_estimate.json"
V2_FORMAT = "evodevo-estimate-v2"
LEGACY_GUARD_H = 432.0
E_CORE_FACTOR = 2.3172


def estimate_path_for_manifest(manifest: dict | None) -> Path:
    raw = (manifest or {}).get("estimate_path")
    if not raw:
        return LEGACY_ESTIMATE_PATH
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_v2(estimate: dict) -> bool:
    return isinstance(estimate, dict) and estimate.get("format") == V2_FORMAT


def _legacy_result(estimate: dict) -> dict:
    """The legacy file's final-campaign row, located by shape (as before)."""
    def locate(obj):
        if isinstance(obj, dict):
            value = obj.get("result")
            if isinstance(value, dict) and "walker_phase_h" in value and "total_h" in value:
                return value
            for child in obj.values():
                found = locate(child)
                if found:
                    return found
        if isinstance(obj, list):
            for child in obj:
                found = locate(child)
                if found:
                    return found
        return None
    return locate(estimate) or {}


def phase_band(estimate: dict, task: str) -> dict:
    """{p5, mean, p95} hours for one phase, or {} when the estimate has none."""
    if is_v2(estimate):
        return dict(estimate.get("result", {}).get("phase_h", {}).get(task, {}))
    result = _legacy_result(estimate)
    key = "walker_phase_h" if task == "Walker-v0" else "upstepper_phase_h"
    return dict(result.get(key, {}))


def total_band(estimate: dict) -> dict:
    if is_v2(estimate):
        return dict(estimate.get("result", {}).get("total_h", {}))
    return dict(_legacy_result(estimate).get("total_h", {}))


def guard_threshold_h(estimate: dict) -> float:
    if is_v2(estimate) and estimate.get("guard", {}).get("applies") is False:
        return float("inf")  # Amendment 2 (k): no cost guard
    if is_v2(estimate):
        return float(estimate.get("guard", {}).get("threshold_h", LEGACY_GUARD_H))
    return LEGACY_GUARD_H


def slowest_ecore_p95_s(estimate: dict, task: str) -> float:
    """B3-footing P-core mean + 1.645 SD over arms, scaled to an E-core."""
    pairs = estimate["inputs"]["t_ppo_mean_sd_at_b3_footing"]
    values = [mean + 1.645 * sd for key, (mean, sd) in pairs.items() if key.endswith("|" + task)]
    if not values:
        raise ValueError(f"no measured t_PPO prior for {task}")
    return max(values) * E_CORE_FACTOR


def unit_p95_ecore_s(estimate: dict, key: str) -> float:
    """Slowest-E-core p95 PPO-task duration for a status key. An Amendment 2
    estimate states it per phase/stage in ``a2.p95_ecore_s`` (budget-scaled);
    any other estimate is read per task as before."""
    stated = (estimate.get("a2") or {}).get("p95_ecore_s", {}) if isinstance(estimate, dict) else {}
    if key in stated:
        return float(stated[key])
    return slowest_ecore_p95_s(estimate, key)
