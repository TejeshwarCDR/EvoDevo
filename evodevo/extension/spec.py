"""Fixed design of the campaign extension (stdlib only).

Everything here was registered on 2026-10-15 in
docs/campaign_prep/ext/PREREGISTRATION_EXT.md, before any extension
preflight measurement existed.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# D7 order: the hard task first (the one difficulty tier the original
# campaign lacks), then the two medium tasks.
TASK_ORDER = ("ObstacleTraverser-v1", "Climber-v0", "CaveCrawler-v0")
TIER = {"ObstacleTraverser-v1": "hard", "Climber-v0": "medium", "CaveCrawler-v0": "medium"}
ARMS = ("direct", "nca", "rule_based")
SEEDS = (0, 1, 2, 3, 4)
SHARED = {"population_size": 15, "offspring_size": 15, "eval_budget": 300, "ppo_seed_aggregation": "max"}

# Reference tasks re-measured in the same batch to calibrate costs against
# the original campaign's realised phase durations.
REFERENCE_TASKS = {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000}
REFERENCE_K = {"Walker-v0": 1, "UpStepper-v0": 2}

# Representative bodies: per core arm, the original campaign's Walker-v0
# final best body ("evolved") and the first structurally valid random genome
# drawn from the arm's campaign encoder ("random").
BODY_KINDS = ("evolved", "random")
EXPECTED_BODIES = len(ARMS) * len(BODY_KINDS)
RANDOM_BODY_SEED = 20261015
MAX_RANDOM_ATTEMPTS = 10_000

# B4x: one training per (task, body) to 2M with a checkpoint every 100k.
CURVE_SEED = 20261015
# B5x: repeated trainings per (task, body) at the task's D6-ext budget.
NOISE_REPS = 4
NOISE_SEED_BASE = 20261016

# Pool safety timeouts (seconds) -- generous E-core bounds, not estimates.
CURVE_TIMEOUT_S = 8 * 3600
NOISE_TIMEOUT_S = 5 * 3600

DOCS_DIR = REPO_ROOT / "docs/campaign_prep/ext"
DECISIONS_JSON = DOCS_DIR / "EXT_DECISIONS.json"
DECISIONS_MD = DOCS_DIR / "EXT_DECISIONS.md"
ESTIMATE_JSON = DOCS_DIR / "final_estimate_ext.json"
CAMPAIGN_YAML = REPO_ROOT / "evodevo/experiments/configs/campaign_ext.yaml"
PREFLIGHT_ROOT = REPO_ROOT / "results/extension_preflight"


def body_label(arm: str, kind: str) -> str:
    return f"{arm}:{kind}"


_BASE_METHOD_CONFIG = (Path(__file__).resolve().parents[2]
                       / "evodevo/experiments/configs/direct/obstacletraverser.yaml")


def campaign_eval_episodes() -> int:
    """``ppo_eval_episodes`` the campaign's runs use (shared base config)."""
    from evodevo.core.config import ExperimentConfig
    return int(ExperimentConfig.from_method_yaml(str(_BASE_METHOD_CONFIG)).ppo_eval_episodes)
