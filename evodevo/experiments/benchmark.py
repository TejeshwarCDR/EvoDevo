"""Unified N-way benchmark orchestrator: runs every (encoder, task, seed)
combination in a benchmark config through the standard EA pipeline
(evodevo.experiments.run_experiment.run_experiment), after verifying every
resolved per-encoder config is pairwise fairness-comparable.

MUST NOT be used to patch or re-run a single arm after seeing results -- see
docs/decisions.md and CONTRIBUTING.md's scientific-integrity note. If a bug
is found partway through a benchmark, fix it and re-run every arm from
scratch, not just the one that was affected.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml

from evodevo.core.config import SHARED_BASE_KEYS, ConfigOverrideError, ExperimentConfig
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import make_evaluator
from evodevo.experiments.fairness import FairnessViolation, assert_comparable
from evodevo.experiments.run_experiment import DONE_MARKER, _run_output_dir, run_experiment

DEFAULT_CONFIGS_ROOT = "evodevo/experiments/configs"
DEFAULT_OUTPUT_DIR = "results/runs"
MANIFEST_NAME = "benchmark_manifest.json"


def _task_to_config_stem(task: str) -> str:
    """'Walker-v0' -> 'walker', 'UpStepper-v0' -> 'upstepper' -- matches the
    naming convention of evodevo/experiments/configs/<method>/*.yaml."""
    stem = re.sub(r"-v\d+$", "", task)
    return stem.lower()


@dataclass
class BenchmarkConfig:
    encoders: List[str]
    tasks: List[str]
    seeds: List[int]
    configs_root: str = DEFAULT_CONFIGS_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    surrogate: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def config_path(self, encoder: str, task: str) -> str:
        return os.path.join(self.configs_root, encoder, f"{_task_to_config_stem(task)}.yaml")


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _library_versions() -> dict:
    versions = {"python": sys.version}
    for lib in ("numpy", "yaml", "pytest", "pandas", "matplotlib"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def resolve_configs(bconfig: BenchmarkConfig) -> dict:
    """Returns {(encoder, task): ExperimentConfig}, using bconfig.seeds[0] as
    a representative seed (seed is exempt from the fairness protocol, so any
    seed produces identical values on every other field)."""
    representative_seed = bconfig.seeds[0]
    resolved = {}
    for encoder in bconfig.encoders:
        for task in bconfig.tasks:
            path = bconfig.config_path(encoder, task)
            config = ExperimentConfig.from_method_yaml(path)
            resolved[(encoder, task)] = dataclasses.replace(config, seed=representative_seed)
    return resolved


def check_fairness(bconfig: BenchmarkConfig) -> dict:
    """Resolve every (encoder, task) config and assert pairwise
    comparability within each task, BEFORE anything is run. Raises
    FairnessViolation on any mismatch. Returns the resolved configs dict."""
    resolved = resolve_configs(bconfig)
    for task in bconfig.tasks:
        for enc_a, enc_b in itertools.combinations(bconfig.encoders, 2):
            assert_comparable(resolved[(enc_a, task)], resolved[(enc_b, task)])
    return resolved


def _shared_fields(config: ExperimentConfig) -> dict:
    d = config.to_dict()
    return {k: d[k] for k in sorted(SHARED_BASE_KEYS)}


def _run_config_for(bconfig: BenchmarkConfig, resolved_configs: dict, encoder: str, task: str, seed: int) -> ExperimentConfig:
    """The per-run config actually used to execute (encoder, task, seed):
    the resolved (encoder, task) config with output_dir forced to
    bconfig.output_dir. This makes bconfig.output_dir the single
    authoritative root for both the manifest and every run's output
    directory -- a per-method configs/<method>/*.yaml's own `output_dir`
    field is NOT used for benchmark-orchestrated runs (it still is when
    that config is run directly via run_experiment.py outside a
    benchmark)."""
    return dataclasses.replace(resolved_configs[(encoder, task)], seed=seed, output_dir=bconfig.output_dir)


def _resolved_config_path(bconfig: BenchmarkConfig, encoder: str, task: str, seed: int) -> str:
    """Where a fully-resolved, self-contained per-run config is written so
    run_experiment.py (which always reloads its config from a file path)
    executes against bconfig.output_dir rather than whatever `output_dir`
    the source configs/<method>/*.yaml happens to contain."""
    directory = os.path.join(bconfig.output_dir, "_resolved_configs", encoder, _task_to_config_stem(task))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"seed_{seed}.yaml")


def build_manifest(bconfig: BenchmarkConfig, resolved_configs: dict) -> dict:
    """The resolved shared config (proven identical across every arm by
    check_fairness), each encoder's describe(), the git commit, library
    versions, and the full planned (encoder, task, seed) run matrix --
    written BEFORE any run starts."""
    any_config = next(iter(resolved_configs.values()))
    shared_config = _shared_fields(any_config)

    encoder_describe = {}
    for encoder in bconfig.encoders:
        any_encoder_config = next(cfg for (enc, _task), cfg in resolved_configs.items() if enc == encoder)
        encoder_describe[encoder] = get_encoder(encoder, grid_shape=any_encoder_config.grid_shape).describe()

    run_matrix = [
        {
            "encoder": encoder,
            "task": task,
            "seed": seed,
            "output_dir": _run_output_dir(_run_config_for(bconfig, resolved_configs, encoder, task, seed), seed),
        }
        for encoder in bconfig.encoders
        for task in bconfig.tasks
        for seed in bconfig.seeds
    ]

    return {
        "shared_config": shared_config,
        "encoders": encoder_describe,
        "git_commit": _git_commit_hash(),
        "library_versions": _library_versions(),
        "surrogate": bconfig.surrogate,
        "run_matrix": run_matrix,
    }


def _is_run_complete(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, DONE_MARKER))


def estimate_wall_clock(bconfig: BenchmarkConfig, resolved_configs: dict) -> dict:
    """Measures ONE real evaluation's wall-clock cost per encoder (using
    that encoder's own resolved hyperparameters and evaluator mode) and
    extrapolates the whole benchmark's estimated total wall-clock from it."""
    evaluator = make_evaluator("surrogate" if bconfig.surrogate else "real")
    per_encoder_cost = {}

    for encoder in bconfig.encoders:
        any_task = bconfig.tasks[0]
        config = resolved_configs[(encoder, any_task)]
        encoder_instance = get_encoder(encoder, grid_shape=config.grid_shape)
        rng = np.random.default_rng(config.seed)
        genome = encoder_instance.random_genome(rng)
        body = encoder_instance.decode(genome)

        start = time.time()
        evaluator(
            body,
            task=config.task,
            ppo_timesteps=config.ppo_timesteps,
            eval_episodes=config.ppo_eval_episodes,
            seed=config.seed,
        )
        per_encoder_cost[encoder] = time.time() - start

    total_estimate = 0.0
    n_runs_per_encoder = len(bconfig.tasks) * len(bconfig.seeds)
    for encoder in bconfig.encoders:
        any_task = bconfig.tasks[0]
        eval_budget = resolved_configs[(encoder, any_task)].eval_budget
        total_estimate += per_encoder_cost[encoder] * eval_budget * n_runs_per_encoder

    return {
        "per_encoder_measured_cost_s": per_encoder_cost,
        "estimated_total_wall_clock_s": total_estimate,
    }


def _write_manifest(bconfig: BenchmarkConfig, manifest: dict) -> str:
    os.makedirs(bconfig.output_dir, exist_ok=True)
    path = os.path.join(bconfig.output_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    return path


def run_benchmark(bconfig: BenchmarkConfig, dry_run: bool = False) -> dict:
    # 1. Fairness check FIRST, before anything is run or written.
    resolved_configs = check_fairness(bconfig)

    # 2. Manifest written before any run starts.
    manifest = build_manifest(bconfig, resolved_configs)
    manifest_path = _write_manifest(bconfig, manifest)

    if dry_run:
        timing = estimate_wall_clock(bconfig, resolved_configs)
        manifest["dry_run_timing_estimate"] = timing
        _write_manifest(bconfig, manifest)
        print(json.dumps(manifest["run_matrix"], indent=2))
        print(
            f"Estimated total wall-clock: {timing['estimated_total_wall_clock_s']:.1f}s "
            f"(measured per-encoder costs: {timing['per_encoder_measured_cost_s']})"
        )
        return manifest

    # 3. Execute, resuming (skipping) any already-completed run.
    completed, skipped = [], []
    for entry in manifest["run_matrix"]:
        encoder, task, seed = entry["encoder"], entry["task"], entry["seed"]
        output_dir = entry["output_dir"]

        if _is_run_complete(output_dir):
            print(f"[skip] {encoder}/{task}/seed_{seed} already complete at {output_dir} -- resuming")
            skipped.append(entry)
            continue

        run_config = _run_config_for(bconfig, resolved_configs, encoder, task, seed)
        resolved_path = _resolved_config_path(bconfig, encoder, task, seed)
        run_config.to_yaml(resolved_path)

        print(f"[run]  {encoder}/{task}/seed_{seed}")
        run_experiment(resolved_path, seed, surrogate=bconfig.surrogate, overwrite=False)
        completed.append(entry)

    manifest["completed_this_invocation"] = completed
    manifest["skipped_already_done"] = skipped
    _write_manifest(bconfig, manifest)
    print(f"manifest written to {manifest_path}")
    print(f"completed {len(completed)} run(s), skipped {len(skipped)} already-done run(s)")

    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a benchmark YAML config")
    parser.add_argument("--dry-run", action="store_true", help="print the run matrix and a wall-clock estimate; run nothing")
    parser.add_argument("--surrogate", action="store_true", help="override the config's surrogate flag to True")
    args = parser.parse_args(argv)

    bconfig = BenchmarkConfig.from_yaml(args.config)
    if args.surrogate:
        bconfig = dataclasses.replace(bconfig, surrogate=True)

    try:
        run_benchmark(bconfig, dry_run=args.dry_run)
    except (FairnessViolation, ConfigOverrideError) as e:
        # Both are "abort before anything runs" outcomes: ConfigOverrideError
        # fires earlier (at config-load time, if a per-method config tries to
        # override a shared-base key); FairnessViolation fires afterward (if
        # two already-loaded configs still disagree on a required-equal
        # field). Either way, nothing has been written to output_dir yet.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
