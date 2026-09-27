"""Single reproducible entry point for a single (encoder, task, seed) run.

Usage:
    python -m evodevo.experiments.run_experiment --config path.yaml --seed N [--surrogate] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
import yaml

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.core.provenance import DirtyTreeError, check_git_state, write_factors_json
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import make_evaluator
from evodevo.evolution.ea import run_ea
from evodevo.evolution.logging import RunLogger
from evodevo.evolution.random_search import run_random_search

DONE_MARKER = "run_meta.json"

# The strategy axis (P7): {ea, random_search} crossed with every encoder.
# random_search is the control that separates the encoding's representational
# PRIOR (what it achieves) from the EA's SEARCH contribution (what selection
# and mutation add on top) -- see evodevo/analysis/budget_curves.py's
# decomposition and evodevo/evolution/random_search.py's module docstring.
STRATEGIES = {"ea": run_ea, "random_search": run_random_search}
DEFAULT_STRATEGY = "ea"


def _compute_connections(body: np.ndarray):
    """Best-effort connectivity computation. Returns None if evogym's compiled
    simulator is not available (e.g. surrogate-mode / pipeline testing)."""
    try:
        from evogym import get_full_connectivity

        return get_full_connectivity(body)
    except ImportError:
        return None


def _run_output_dir(config: ExperimentConfig, seed: int, tag: str = None) -> str:
    parts = [config.output_dir, config.encoder, config.task]
    if tag:
        parts.append(tag)
    parts.append(f"seed_{seed}")
    return os.path.join(*parts)


def _combine_tag_and_strategy(tag: Optional[str], strategy: str) -> Optional[str]:
    """Folds `strategy` into `tag` so the default strategy ("ea") never
    changes an existing output path, while a non-default strategy always
    disambiguates it -- shared by run_experiment() below and by
    evodevo.experiments.benchmark._output_dir_for(), so a benchmark's
    precomputed manifest path always matches what a direct run_experiment()
    invocation with the same (tag, strategy) independently computes."""
    if strategy == DEFAULT_STRATEGY:
        return tag
    return strategy if tag is None else f"{tag}_{strategy}"


def run_experiment(
    config_path: str,
    seed: int,
    surrogate: bool,
    overwrite: bool,
    encoder_kwargs: dict = None,
    tag: str = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    allow_dirty: bool = False,
    strategy: str = DEFAULT_STRATEGY,
    arm_id: Optional[str] = None,
    precomputed_git_state: Optional[Tuple[str, List[str]]] = None,
) -> str:
    """encoder_kwargs overrides/extends the encoder's own hyperparameters
    (e.g. {'dev_steps': 1} for the NCA encoder) without touching the encoder
    class, the EA, or the evaluator -- used for hyperparameter ablations.
    `tag` disambiguates the output directory when multiple ablation arms
    share the same (encoder, task, seed). `progress_callback`, if given, is
    invoked with each evaluation's record dict for live stdout reporting
    (evodevo.evolution.progress) -- purely additive, never affects persisted
    output. `strategy` selects which search strategy consumes the budget
    (see STRATEGIES above); both strategies share the exact same
    evaluation pipeline (evodevo.evolution.eval_bookkeeping), so switching
    strategy never changes budget semantics, invalid-genome policy, or the
    P6 dual counters -- only which genome gets evaluated next. `arm_id` is a
    purely additive identity label recorded verbatim into run_meta.json
    (defaulting to config.encoder when not given) -- it does NOT affect
    output-dir computation (that is still driven by `tag`/`strategy` alone,
    exactly as before); it exists so downstream analysis can group runs by
    arm identity (e.g. "rule_based" vs "rule_based_monotone", both of which
    share config.encoder="rule_based") without reconstructing it from `tag`.

    `allow_dirty` gates the provenance check below (see
    evodevo.core.provenance): a dirty working tree raises DirtyTreeError
    unless explicitly overridden -- for local iteration only, never for a
    run whose output is cited in a report or a docs/decisions.md row. This
    is a standalone provenance wrapper: it does not touch ExperimentConfig's
    shape, SHARED_BASE_KEYS, or the lock hash in evodevo.experiments.lock_hash.

    `precomputed_git_state`, if given, is an already-computed
    (commit_hash, dirty_files) pair and skips this function's own
    check_git_state call entirely -- used by evodevo.experiments.benchmark's
    run_benchmark(), which checks cleanliness exactly once at the very start
    of a benchmark invocation (before its own manifest/factors bookkeeping
    writes anything) and threads that single result through every
    run_experiment() call in its run matrix. Without this, each call here
    would re-derive git state independently and see the benchmark's own
    prior writes (e.g. results/runs/factors.json, the manifest) as tree
    dirtiness, spuriously blocking a later run in the same invocation. Left
    as None (the default) for run_experiment()'s own standalone CLI use,
    where a fresh check_git_state call here is exactly right."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}'; available: {sorted(STRATEGIES)}")
    run_strategy = STRATEGIES[strategy]

    if precomputed_git_state is not None:
        commit_hash, dirty_files = precomputed_git_state
    else:
        commit_hash, dirty_files = check_git_state(allow_dirty)

    loaded = ExperimentConfig.from_method_yaml(config_path)
    # --seed overrides the config's seed; everything else comes from the YAML.
    config_dict = loaded.to_dict()
    config_dict["seed"] = seed
    config_dict["grid_shape"] = tuple(config_dict["grid_shape"])
    config = ExperimentConfig(**config_dict)
    if config.ppo_seeds > 1:
        # D16 (docs/decisions.md, 2026-09-27): multi-seed PPO evaluation per
        # genome is decided but not yet implemented in the evaluator. Refuse
        # here -- the single entry point every run passes through -- rather
        # than silently evaluating each genome once.
        raise NotImplementedError(
            f"ppo_seeds={config.ppo_seeds} (> 1) requested for {config.encoder}/{config.task}, but "
            "multi-seed PPO evaluation per genome is not implemented yet; refusing to run as K=1."
        )

    # Fold strategy into the output-dir tag so the default (strategy="ea")
    # keeps its exact pre-existing output path -- only a non-default
    # strategy (random_search) disambiguates the path, keeping this a
    # backward-compatible addition rather than a breaking rename.
    effective_tag = _combine_tag_and_strategy(tag, strategy)
    output_dir = _run_output_dir(config, seed, tag=effective_tag)
    done_path = os.path.join(output_dir, DONE_MARKER)
    if os.path.exists(done_path) and not overwrite:
        raise FileExistsError(
            f"a completed run already exists at {output_dir} "
            "(pass --overwrite to replace it)"
        )
    os.makedirs(output_dir, exist_ok=True)

    factors, remaining_kwargs = extract_factors(encoder_kwargs, seed=config.seed)
    write_factors_json(output_dir, factors, commit_hash, dirty_files)
    encoder = get_encoder(config.encoder, factors=factors, grid_shape=config.grid_shape, **remaining_kwargs)
    evaluator = make_evaluator("surrogate" if surrogate else "real")

    heartbeat_path = os.path.join(config.output_dir, ".heartbeat.json")
    logger = RunLogger(output_dir, heartbeat_path=heartbeat_path, encoder=config.encoder, seed=seed,
                       on_eval=progress_callback)
    try:
        result = run_strategy(encoder, evaluator, config, logger=logger)
    finally:
        logger.close()

    meta_config = config.to_dict()
    meta_config["encoder_kwargs"] = encoder_kwargs or {}
    meta_config["tag"] = tag
    meta_config["strategy"] = strategy
    meta_config["arm_id"] = arm_id or config.encoder
    logger.write_meta(meta_config, encoder.describe(), cache_hit_rate=result.cache_hit_rate,
                       resolved_policy=result.resolved_policy)

    if result.population:
        best = max(result.population, key=lambda ind: ind.fitness)
        best_body = encoder.decode(best.genome)
        connections = _compute_connections(best_body)
        npz_path = os.path.join(output_dir, "best_body.npz")
        if connections is not None:
            np.savez(npz_path, body=best_body, connections=connections)
        else:
            np.savez(npz_path, body=best_body)

    return output_dir


def _parse_encoder_kwargs(pairs) -> dict:
    kwargs = {}
    for pair in pairs or []:
        key, _, raw_value = pair.partition("=")
        kwargs[key] = yaml.safe_load(raw_value)
    return kwargs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig YAML file")
    parser.add_argument("--seed", required=True, type=int, help="seed for this run (overrides config seed)")
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="use the fast physics-free surrogate evaluator instead of real PPO",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an already-completed run at the target output directory",
    )
    parser.add_argument(
        "--encoder-kwarg",
        action="append",
        dest="encoder_kwargs",
        metavar="KEY=VALUE",
        help="override/extend an encoder hyperparameter (e.g. --encoder-kwarg dev_steps=1); repeatable",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="disambiguates the output directory when multiple arms share (encoder, task, seed), e.g. for ablations",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow running with an uncommitted working tree (local iteration only -- "
        "never for a run whose output is cited in a report or a docs/decisions.md row)",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=sorted(STRATEGIES),
        help="search strategy consuming the eval budget (default: %(default)s); "
        "'random_search' is the no-selection/no-mutation control (see evodevo/evolution/random_search.py)",
    )
    parser.add_argument(
        "--arm-id",
        default=None,
        help="identity label recorded into run_meta.json for downstream analysis grouping "
        "(default: the config's encoder); does not affect the output directory",
    )
    args = parser.parse_args(argv)

    try:
        output_dir = run_experiment(
            args.config,
            args.seed,
            args.surrogate,
            args.overwrite,
            encoder_kwargs=_parse_encoder_kwargs(args.encoder_kwargs),
            tag=args.tag,
            allow_dirty=args.allow_dirty,
            strategy=args.strategy,
            arm_id=args.arm_id,
        )
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except DirtyTreeError:
        sys.exit(1)

    print(f"run complete: {output_dir}")


if __name__ == "__main__":
    main()
