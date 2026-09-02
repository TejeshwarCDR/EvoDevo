"""Computes the SHA-256 "lock hash" that pins the primary benchmark run's
resolved configuration (see docs/READY_TO_RUN.md and its successors).

This module exists to replace an earlier, unreproducible ad hoc hash
(`7e93f59a...`, recorded in the original docs/READY_TO_RUN.md): that value
was computed manually during a prior session with no script or documented
procedure behind it, and could not be reproduced later even against the
exact commit it was computed at, using the same underlying config data (see
docs/decisions.md, "Retire unreproducible lock hash", 2026-09-02). This is
the single, documented, re-runnable procedure going forward -- anyone at any
future point can re-run this exact code and get the same digest for the same
resolved configs, with zero ambiguity.

PROCEDURE (changing any step invalidates every previously-recorded lock
hash and requires updating tests/experiments/test_lock_hash.py's pinned
expected value):

1. Resolve the primary configs via
   `evodevo.experiments.benchmark.resolve_configs()` -- the exact code path
   `run_benchmark()` itself uses to build each (encoder, task) ->
   ExperimentConfig mapping, not a reimplementation of config loading.
2. Serialize each resolved `ExperimentConfig` via its own `.to_dict()`
   method (`evodevo/core/config.py`) -- the object's own canonical
   serialization, not a hand-rolled field list that could drift from the
   dataclass definition.
3. Combine the per-(encoder, task) dicts into a single JSON-serializable
   list of `{"encoder": ..., "task": ..., "config": <to_dict() output>}`
   entries, sorted by `(encoder, task)` -- so the result does not depend on
   dict/YAML iteration order.
4. `json.dumps(..., sort_keys=True, separators=(",", ":"))` -- `sort_keys`
   makes key order within each entry irrelevant; the compact `(",", ":")`
   separators make the exact byte output independent of any given Python
   version's default separator whitespace.
5. SHA-256 of the resulting UTF-8-encoded string; the hex digest is the
   lock hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Dict, Tuple

from evodevo.core.config import ExperimentConfig
from evodevo.experiments.benchmark import BenchmarkConfig, resolve_configs

DEFAULT_BENCHMARK_CONFIG = "evodevo/experiments/configs/benchmark.yaml"


def _serialize(resolved: Dict[Tuple[str, str], ExperimentConfig]) -> str:
    """Step 3+4 of the module docstring's procedure."""
    entries = [
        {"encoder": encoder, "task": task, "config": config.to_dict()}
        for (encoder, task), config in resolved.items()
    ]
    entries.sort(key=lambda e: (e["encoder"], e["task"]))
    return json.dumps(entries, sort_keys=True, separators=(",", ":"))


def compute_lock_hash(resolved: Dict[Tuple[str, str], ExperimentConfig]) -> str:
    """The full documented procedure (steps 2-5) applied to an already
    resolved `{(encoder, task): ExperimentConfig}` mapping, as produced by
    `evodevo.experiments.benchmark.resolve_configs()`. Takes the resolved
    mapping directly (rather than a YAML path) so callers -- including
    tests -- can hash a fixed, hand-constructed set of configs without
    touching any file on disk."""
    blob = _serialize(resolved).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_lock_hash_from_benchmark_config(path: str = DEFAULT_BENCHMARK_CONFIG) -> str:
    """Convenience wrapper covering step 1 too: loads a `BenchmarkConfig`
    from YAML (the same entry point `evodevo.experiments.benchmark.main()`
    uses) and resolves it via `resolve_configs()` -- the real orchestrator
    code path, not a reimplementation -- before hashing."""
    bconfig = BenchmarkConfig.from_yaml(path)
    resolved = resolve_configs(bconfig)
    return compute_lock_hash(resolved)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_BENCHMARK_CONFIG,
        help="Path to a benchmark.yaml (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    print(compute_lock_hash_from_benchmark_config(args.config))


if __name__ == "__main__":
    main()
