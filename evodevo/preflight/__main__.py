"""python -m evodevo.preflight -- the pre-flight measurement harness CLI,
the single entry point for the D5 campaign pre-flight.

IMPORT ORDER IS LOAD-BEARING -- READ BEFORE TOUCHING THIS FILE'S IMPORTS.
configure_threading(1, strict=True), called just below, MUST run before
numpy or torch (or anything that transitively imports them: evogym,
stable_baselines3, even a stray "from evodevo.encoders import get_encoder")
is imported anywhere in this process. OMP_NUM_THREADS/MKL_NUM_THREADS/etc.
are read ONCE, by OpenMP/MKL's own C initialization code, at that library's
FIRST import in the process -- see evodevo/parallel/threading_control.py's
own docstring for the full mechanism. Every worker this process later
spawns configures its OWN threading independently in its own initializer
(see evodevo/parallel/worker_pool.py) -- but THIS process, the orchestrator
itself, only gets one chance, right here, before anything else runs.

This is why the call sits at this module's own top level, immediately
after the one import it needs (threading_control itself, which is
numpy/torch-free), and BEFORE every other import in this file -- all of
which are lazy-imported inside functions below wherever they are not
already known to be numpy/torch-free at module scope, so that adding a
numpy/torch-importing dependency to one of THEM later can never silently
move the goalposts here. A FUTURE REFACTOR THAT ADDS AN INNOCENT-LOOKING
TOP-LEVEL IMPORT ABOVE THE configure_threading() CALL BELOW -- one that
transitively pulls in numpy or torch -- WOULD SILENTLY BREAK THE ENTIRE
MEASUREMENT: every worker in every subsequent EvaluationPool would still
configure itself correctly (that guard is independent), but THIS process's
own threading would default back to every-core BLAS/OpenMP contention, and
nothing would raise -- the run would just be mysteriously, silently
slower. If you are about to add a top-level import here, check what it
imports, transitively, first.

THIS MODULE IS A CLI ENTRY POINT ONLY and must never be imported by other
Python code in this project, including its own tests -- always invoke it
via `python -m evodevo.preflight ...` in a fresh subprocess. Importing it
in-process risks tripping ThreadingConfiguredTooLateError the instant
numpy/torch is already loaded for any other reason in that process, which
is true of nearly every other test in this suite. See
tests/preflight/test_cli.py's import-order regression test for the one
deliberate, isolated exception (it does this import in its own subprocess,
specifically to prove this property).
"""
from __future__ import annotations

import sys

from evodevo.parallel.threading_control import ThreadingConfiguredTooLateError, configure_threading

try:
    _THREADING_REPORT = configure_threading(1, strict=True)
except ThreadingConfiguredTooLateError as _exc:
    print(f"error: {_exc}", file=sys.stderr)
    print(
        "evodevo.preflight.__main__'s own import order is load-bearing -- see this file's module "
        "docstring. This almost always means numpy or torch was already imported (directly or "
        "transitively) before this module itself was imported -- e.g. because something imported "
        "it in-process instead of via `python -m evodevo.preflight`.",
        file=sys.stderr,
    )
    sys.exit(1)

# Everything below is numpy/torch-free at module scope, per each imported
# module's own established discipline (see evodevo/parallel/worker_pool.py
# and evodevo/preflight/*.py's own docstrings) -- safe to import now, after
# configure_threading() has already run above. Stage classes themselves
# (evodevo.preflight.stages.b1_machine etc.) are still imported lazily,
# inside build_registry(), on principle -- so a numpy/torch import added to
# one of THEM can never retroactively matter here.
import argparse
import os
import signal
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from evodevo.preflight.context import PreflightContext
from evodevo.preflight.dirty_check import dirty_paths_outside_results, project_is_dirty
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import (
    RunDir,
    RunDirNotFoundError,
    atomic_write_json,
    create_or_resume_run_dir,
    read_json,
    write_machine_snapshot,
)
from evodevo.preflight.stages import (
    SELF_CHECK_STAMP,
    DependencyNotDoneError,
    StageRegistry,
    UnknownStageError,
    run_stages,
)
from evodevo.preflight.telemetry import TelemetrySampler

# The exact --self-check overrides, verbatim from the prompt that defined
# this bounded validation run: every value here exists purely to prove
# every stage's code path executes, every artifact is written, every
# figure renders -- NONE of these numbers may ever be cited as a real
# measurement (see SELF_CHECK_STAMP, stamped into every artifact this
# produces). ppo_timesteps/arms/tasks/proj_* are not individually named in
# that list but are set here to equally trivial values for the same
# reason: the top-level MUST NOT for this whole prompt forbids "any
# full-length PPO training... at production settings", full stop, which
# would otherwise still happen inside B3/B6 even with every OTHER stage's
# settings reduced.
#
# cost_develop_n/cost_sentinel_n=5 (default 100 each) matter far more than
# they look: B3's t_develop/t_sentinel measurements pay a decode()-only
# cost that is NOT PPO and is NOT governed by --ppo-timesteps at all, and
# rule_based's decode() was measured (see PREFLIGHT_READINESS.md Section 9,
# "why the full arm grid did not fit in 3 minutes") at ~232ms/call --
# roughly 200,000x direct's ~0.001ms/call. At the previous, unreduced
# default of 100 (develop, always paid in full) plus up to
# cost_max_draws=200 (sentinel search), rule_based alone could cost
# 100*0.232s + up to 200*0.232s =~ 70s of PURE DECODE TIME per self-check
# run, dwarfing the PPO cost this whole override dict otherwise controls.
SELF_CHECK_OVERRIDES = {
    "concurrency": 2,
    "sweep_levels": "1,2",
    "sweep_genomes": 2,
    "sweep_timesteps": 500,
    "sweep_cooldown_s": 0.0,
    "sweep_warm_up": False,
    "cost_reps": 1,
    "ppo_timesteps": 500,
    "cost_max_draws": 200,
    "cost_develop_n": 5,
    "cost_sentinel_n": 5,
    "noise_bodies": 1,
    "noise_reps": 2,
    "curve_bodies": 1,
    "curve_max_timesteps": 2000,
    "curve_eval_interval": 500,
    "arms": "direct",
    "tasks": "Walker-v0",
    "proj_arms": "direct",
    "proj_tasks": "Walker-v0",
    "proj_populations": "2",
    "proj_generations": "2",
    "proj_seeds": "1",
    "telemetry_interval": 1.0,
}

# --self-check-full: identical trivial numeric settings to SELF_CHECK_
# OVERRIDES above, but the full production arm x task grid (direct, nca,
# rule_based x Walker-v0, UpStepper-v0) instead of a single (direct,
# Walker-v0) combination -- so the pre-flight's first contact with every
# arm/task combination happens here, in minutes, not hours into the real
# run. Resolves readiness Open Question 2: the answer was that self-check
# must honour the full grid; this is that. Given a wider 10-minute budget
# (not 3) specifically because rule_based's real per-decode cost, even
# reduced via cost_develop_n/cost_sentinel_n above, still multiplies by 3
# arms x 2 tasks instead of 1x1.
SELF_CHECK_FULL_OVERRIDES = dict(
    SELF_CHECK_OVERRIDES,
    arms="direct,nca,rule_based",
    tasks="Walker-v0,UpStepper-v0",
    proj_arms="direct,nca,rule_based",
    proj_tasks="Walker-v0,UpStepper-v0",
)


class PreflightInterrupted(BaseException):
    """Raised by the SIGINT/SIGTERM handler installed in main().

    Deliberately a BaseException, not an Exception -- the same relationship
    KeyboardInterrupt itself has to Exception -- so that stages/run_stages'
    own broad `except Exception` handling (which marks a stage "failed" on
    a genuine bug) never catches this and mislabels an operator-requested
    stop as a failure. Left uncaught by everything except main()'s own
    handler, it propagates cleanly through every `with pool:` /
    `with ctx.make_pool(...) as pool:` block a stage is inside at the
    moment the signal arrives, running their normal cleanup (pool
    shutdown) as it unwinds -- and past run_stages' `except Exception`
    around stage.run(ctx), leaving the manifest exactly as "running" (the
    status written just before stage.run() was called), which is exactly
    the state that makes --run-id resume re-run that stage rather than
    skip it.
    """

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001 -- frame is part of the signal.signal callback signature
        raise PreflightInterrupted(signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def build_registry(args: argparse.Namespace, self_check_stamp: Optional[str] = None) -> StageRegistry:
    """The production stage registry -- all seven D5 stages.

    self_check_stamp is None on every normal invocation; --self-check
    passes SELF_CHECK_STAMP through here so every stage that produces a
    figure or JSON artifact stamps it, verbatim, in both places (see
    evodevo/preflight/stages/__init__.py's SELF_CHECK_STAMP docstring).
    """
    from evodevo.preflight.stages.b1_machine import MachineProfileStage
    from evodevo.preflight.stages.b2_sweep import ConcurrencySweepStage
    from evodevo.preflight.stages.b3_cost import EvaluationCostStage
    from evodevo.preflight.stages.b4_learning_curves import LearningCurvesStage
    from evodevo.preflight.stages.b5_noise import EvaluationNoiseStage
    from evodevo.preflight.stages.b6_projection import ProjectionStage
    from evodevo.preflight.stages.b7_reliability import ReliabilityStage

    registry = StageRegistry()
    registry.register(MachineProfileStage(disk_probe_mb=args.disk_probe_mb, self_check_stamp=self_check_stamp))
    registry.register(
        ConcurrencySweepStage(
            levels=[int(x) for x in _split_csv(args.sweep_levels)],
            n_genomes=args.sweep_genomes,
            timesteps=args.sweep_timesteps,
            order=args.sweep_order,
            cooldown_s=args.sweep_cooldown_s,
            tolerance=args.sweep_tolerance,
            warm_up=args.sweep_warm_up,
            # evaluator_mode intentionally NOT exposed as a CLI flag here -- see module docstring.
            self_check_stamp=self_check_stamp,
        )
    )
    registry.register(
        EvaluationCostStage(
            concurrency=args.concurrency,
            arms=_split_csv(args.arms),
            tasks=_split_csv(args.tasks),
            cost_reps=args.cost_reps,
            ppo_timesteps=args.ppo_timesteps,
            cost_max_draws=args.cost_max_draws,
            t_develop_n=args.cost_develop_n,
            t_sentinel_n=args.cost_sentinel_n,
            # evaluator_mode intentionally NOT exposed as a CLI flag here -- see module docstring.
            self_check_stamp=self_check_stamp,
        )
    )
    registry.register(
        LearningCurvesStage(
            curve_bodies=args.curve_bodies,
            curve_max_timesteps=args.curve_max_timesteps,
            curve_eval_interval=args.curve_eval_interval,
            concurrency=args.concurrency,
            # evaluator_mode intentionally NOT exposed as a CLI flag here -- see module docstring.
            self_check_stamp=self_check_stamp,
        )
    )
    registry.register(
        EvaluationNoiseStage(
            noise_bodies=args.noise_bodies,
            noise_reps=args.noise_reps,
            # evaluator_mode/cache_enabled intentionally NOT exposed as CLI flags -- see module docstring.
            self_check_stamp=self_check_stamp,
        )
    )
    registry.register(
        ProjectionStage(
            populations=[int(x) for x in _split_csv(args.proj_populations)],
            generations=[int(x) for x in _split_csv(args.proj_generations)],
            seeds=[int(x) for x in _split_csv(args.proj_seeds)],
            arms=_split_csv(args.proj_arms),
            tasks=_split_csv(args.proj_tasks),
            overhead=args.proj_overhead,
            concurrency=args.concurrency,
            self_check_stamp=self_check_stamp,
        )
    )
    registry.register(
        ReliabilityStage(
            thermal_threshold_c=args.thermal_threshold_c, ram_margin=args.ram_margin,
            self_check_stamp=self_check_stamp,
        )
    )
    return registry


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _print_dry_run(run_dir_path: str, registry: StageRegistry, self_check: bool = False) -> None:
    print(f"run directory: {run_dir_path}")
    if self_check:
        print(f"mode: --self-check ({SELF_CHECK_STAMP}) -- trivial settings, NOT a measurement")
    order = registry.ordered()
    if not order:
        print("no stages registered.")
        return
    header = f"{'stage':<30} {'cost':<10} {'depends_on':<30} description"
    print(header)
    print("-" * len(header))
    for name in order:
        stage = registry.get(name)
        deps = ",".join(stage.depends_on) or "-"
        print(f"{name:<30} {stage.estimated_cost:<10} {deps:<30} {stage.description}")


def _utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _apply_self_check_overrides(args: argparse.Namespace, overrides: dict = SELF_CHECK_OVERRIDES) -> None:
    for key, value in overrides.items():
        setattr(args, key, value)


def _create_self_check_run_dir(base_dir: str, run_id: str) -> RunDir:
    """Creates a NEW run directory under a caller-chosen, fixed run_id
    (`selfcheck_<timestamp>` or `selfcheck_full_<timestamp>`), mirroring
    create_or_resume_run_dir's own "new run" path -- which cannot be reused
    directly here because its run_id= branch is resume-only (requires the
    directory to already exist; see run_dir.py's own docstring for why
    resume must never silently create)."""
    run_dir = RunDir(os.path.join(base_dir, run_id))
    if os.path.exists(run_dir.path):
        raise FileExistsError(f"self-check run directory {run_dir.path!r} already exists")
    os.makedirs(base_dir, exist_ok=True)
    run_dir.ensure_skeleton()
    write_machine_snapshot(run_dir)
    atomic_write_json(run_dir.manifest_path, {"run_id": run_id, "stages": {}})
    return run_dir


def _record_project_dirty(run_dir: RunDir, project_dirty: bool) -> None:
    """Records whether the working tree had uncommitted changes (outside
    results/) at the moment this run started, in manifest.json, for EVERY
    run -- self-check, self-check-full, and a real (possibly --allow-dirty)
    launch alike. A real run against a dirty tree is only possible via
    --allow-dirty, but its manifest must say so plainly rather than let a
    reader assume the git commit hash recorded per stage (see
    stages/__init__.py's run_stages()) is the whole story."""
    manifest = read_json(run_dir.manifest_path)
    manifest["project_dirty"] = project_dirty
    atomic_write_json(run_dir.manifest_path, manifest)




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evodevo.preflight",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", default="results/preflight",
        help="root directory all pre-flight runs live under (default: %(default)s)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="resume an existing run directory by name (its folder name under --base-dir); "
        "default creates a new run",
    )
    parser.add_argument(
        "--only", default=None, metavar="STAGE[,STAGE...]",
        help="run only the named stage(s) (comma-separated), skipping everything else",
    )
    parser.add_argument(
        "--skip", default=None, metavar="STAGE[,STAGE...]",
        help="exclude the named stage(s) (comma-separated) from this invocation",
    )
    parser.add_argument(
        "--force", default=None, metavar="STAGE[,STAGE...]",
        help="re-run the named stage(s) (comma-separated) even if already marked done",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the ordered stage list with estimated cost and exit; runs nothing, "
        "writes only the run directory skeleton",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="permit launching a real (non-self-check) run against an uncommitted working tree "
        "(anything `git status --porcelain` reports outside results/). Without this, a dirty "
        "tree is refused before any run directory is created -- a real pre-flight's own recorded "
        "git commit hash (per stage) must actually reflect what ran. --self-check/"
        "--self-check-full are always exempt but always RECORD project_dirty in their manifest.",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="A BOUNDED, NON-PRODUCTION EXECUTION: forces every stage to deliberately trivial "
        "settings on a single (direct, Walker-v0) arm/task combination, runs under a fresh "
        "results/preflight/selfcheck_<ts> run directory, stamps every artifact SELF-CHECK -- NOT "
        "A MEASUREMENT, and must finish in well under 3 minutes. Proves the code path for one "
        "combination executes; produces no number anyone should cite. Overrides --base-dir/"
        "--run-id and every stage-sizing flag listed in this help text. Before a real launch, "
        "prefer --self-check-full, which additionally proves every arm x task combination.",
    )
    parser.add_argument(
        "--self-check-full", action="store_true",
        help="THE RECOMMENDED PRE-LAUNCH CHECK: --self-check's identical trivial settings, but "
        "across the FULL production arm x task grid (direct, nca, rule_based x Walker-v0, "
        "UpStepper-v0), not a single combination -- so the first contact with all 6 combinations "
        "happens here, in minutes, not hours into the real run. Runs under a fresh "
        "results/preflight/selfcheck_full_<ts> run directory; must finish in well under 10 "
        "minutes (wider than --self-check's 3, since rule_based's real decode() cost, even "
        "reduced, still multiplies by 3 arms x 2 tasks). Mutually exclusive with --self-check.",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=None,
        help="if set, sends this process SIGTERM after this many seconds of wall clock from "
        "startup -- the run stops the same way an operator-sent SIGTERM would (resumable, not "
        "corrupted), rather than running unbounded (default: no timeout)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="disable the live terminal dashboard; fall back to plain line-by-line logging "
        "(also the automatic behaviour whenever stdout is not a TTY)",
    )
    parser.add_argument(
        "--dashboard-interval", type=float, default=2.0,
        help="seconds between dashboard redraws (default: %(default)s)",
    )
    parser.add_argument(
        "--telemetry-interval", type=float, default=5.0,
        help="seconds between background telemetry samples (default: %(default)s)",
    )
    parser.add_argument(
        "--disk-probe-mb", type=int, default=512,
        help="[B1] size (MB) of the temp file written to measure sequential disk write rate "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-levels", default="8,12,16,20,24,28", metavar="N[,N...]",
        help="[B2] comma-separated worker counts to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-genomes", type=int, default=120,
        help="[B2] number of genomes in the fixed workload every level evaluates -- raised from the "
        "original 24 after the D13 fix: 24 was too few relative to worker count once this "
        "machine's E-cores enter the pool (level 12+), so a single straggler genome dominated a "
        "level's wall-clock instead of measuring steady-state throughput; see b2_sweep.py's "
        "DEFAULT_N_GENOMES docstring (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-timesteps", type=int, default=20000,
        help="[B2] REDUCED ppo_timesteps per genome for this stage only -- measures the SHAPE of "
        "the concurrency curve, never the campaign's real PPO budget (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-order", choices=["ascending", "shuffled"], default="shuffled",
        help="[B2] level visit order -- shuffled (default) guards against a thermal-drift confound "
        "masquerading as a concurrency effect; see the stage's own docstring",
    )
    parser.add_argument(
        "--sweep-cooldown-s", type=float, default=60.0,
        help="[B2] pause between levels, letting the machine's thermal state settle (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-tolerance", type=float, default=0.05,
        help="[B2] relative throughput tolerance (fraction) defining the 'knee' concurrency level "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-warm-up", dest="sweep_warm_up", action="store_true", default=True,
        help="[B2] D13 fix: run one discarded warm-up level (at the smallest --sweep-levels entry) "
        "before the timed/shuffled sweep, to absorb the one-time cold-cache/cold-CPU-frequency "
        "cost that shuffling alone does not control for (default: on)",
    )
    parser.add_argument(
        "--no-sweep-warm-up", dest="sweep_warm_up", action="store_false",
        help="[B2] disable the discarded warm-up level (see --sweep-warm-up)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="[B3] worker count to measure evaluation cost at (default: B2's own knee_level)",
    )
    parser.add_argument(
        "--arms", default="direct,nca,rule_based", metavar="ARM[,ARM...]",
        help="[B3] comma-separated arms to measure cost for (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks", default="Walker-v0,UpStepper-v0", metavar="TASK[,TASK...]",
        help="[B3] comma-separated tasks to measure cost for, each independently -- never "
        "extrapolated from another (default: %(default)s)",
    )
    parser.add_argument(
        "--cost-reps", type=int, default=3,
        help="[B3] valid genomes per arm x task to measure t_PPO over (default: %(default)s)",
    )
    parser.add_argument(
        "--ppo-timesteps", type=int, default=1_000_000,
        help="[B3] FULL campaign ppo_timesteps for t_PPO measurement -- never reduced here; that is "
        "--sweep-timesteps's job (default: %(default)s)",
    )
    parser.add_argument(
        "--cost-max-draws", type=int, default=5000,
        help="[B3] cap on genome draws while looking for --cost-reps valid genomes per arm, before "
        "the result is marked partial (default: %(default)s)",
    )
    parser.add_argument(
        "--cost-develop-n", type=int, default=100,
        help="[B3] number of decode()-only samples for t_develop, paid once per arm regardless of "
        "validity -- for arms with a slow decode() (rule_based measured at ~232ms/call vs direct's "
        "~0.001ms), this is the dominant cost of the whole stage; see --self-check-full's own "
        "investigation notes in PREFLIGHT_READINESS.md Section 9 (default: %(default)s)",
    )
    parser.add_argument(
        "--cost-sentinel-n", type=int, default=100,
        help="[B3] target number of INVALID-genome decode samples for t_sentinel per arm, bounded by "
        "--cost-max-draws -- same slow-decode cost concern as --cost-develop-n (default: %(default)s)",
    )
    parser.add_argument(
        "--curve-bodies", type=int, default=3,
        help="[B4] bodies per task (spanning low/middle/high fitness observed in B3) to train full "
        "learning curves for (default: %(default)s)",
    )
    parser.add_argument(
        "--curve-max-timesteps", type=int, default=2_000_000,
        help="[B4] PPO timesteps to train each curve body to (default: %(default)s)",
    )
    parser.add_argument(
        "--curve-eval-interval", type=int, default=100_000,
        help="[B4] timesteps between periodic curve checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--noise-bodies", type=int, default=3,
        help="[B5] bodies per task (spanning low/middle/high fitness observed in B3) to re-evaluate "
        "repeatedly (default: %(default)s)",
    )
    parser.add_argument(
        "--noise-reps", type=int, default=5,
        help="[B5] repeated evaluations per body, each with a different PPO seed (default: %(default)s)",
    )
    parser.add_argument(
        "--proj-populations", default="15", metavar="N[,N...]",
        help="[B6] candidate population_size grid (default: %(default)s)",
    )
    parser.add_argument(
        "--proj-generations", default="20", metavar="N[,N...]",
        help="[B6] candidate generations grid (default: %(default)s)",
    )
    parser.add_argument(
        "--proj-seeds", default="5", metavar="N[,N...]",
        help="[B6] candidate seed-count grid (default: %(default)s)",
    )
    parser.add_argument(
        "--proj-arms", default="direct,nca,rule_based", metavar="ARM[,ARM...]",
        help="[B6] arms to include in the projection grid (default: %(default)s)",
    )
    parser.add_argument(
        "--proj-tasks", default="Walker-v0,UpStepper-v0", metavar="TASK[,TASK...]",
        help="[B6] tasks to include in the projection grid (default: %(default)s)",
    )
    parser.add_argument(
        "--proj-overhead", type=float, default=None,
        help="[B6] explicit overhead factor override (default: measured from B2's realised "
        "throughput versus its per-genome latency at the chosen concurrency)",
    )
    parser.add_argument(
        "--thermal-threshold-c", type=float, default=85.0,
        help="[B7] package temperature (C) above which time-spent is reported (default: %(default)s)",
    )
    parser.add_argument(
        "--ram-margin", type=float, default=0.8,
        help="[B7] fraction of total RAM usable when computing the implied max safe concurrency "
        "(default: %(default)s)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_check and args.self_check_full:
        print("error: --self-check and --self-check-full are mutually exclusive", file=sys.stderr)
        return 1

    self_check_stamp = None
    is_any_self_check = args.self_check or args.self_check_full
    if args.self_check:
        _apply_self_check_overrides(args, SELF_CHECK_OVERRIDES)
        self_check_stamp = SELF_CHECK_STAMP
        # --base-dir is deliberately NOT forced here -- it already defaults
        # to "results/preflight" (build_parser()'s own default), which is
        # exactly where the self-check is meant to land in normal
        # operator use. Respecting an explicit --base-dir override (rather
        # than clobbering it) is what makes this whole flag testable in
        # isolation -- see tests/preflight/test_cli.py.
        args.run_id = f"selfcheck_{_utc_compact_timestamp()}"
    elif args.self_check_full:
        _apply_self_check_overrides(args, SELF_CHECK_FULL_OVERRIDES)
        self_check_stamp = SELF_CHECK_STAMP
        args.run_id = f"selfcheck_full_{_utc_compact_timestamp()}"

    registry = build_registry(args, self_check_stamp=self_check_stamp)

    project_dirty = project_is_dirty(REPO_ROOT)
    if not is_any_self_check and not args.dry_run and project_dirty and not args.allow_dirty:
        print("error: working tree has uncommitted changes outside results/ -- refusing to launch.", file=sys.stderr)
        print("offending paths:", file=sys.stderr)
        for path in dirty_paths_outside_results(REPO_ROOT):
            print(f"  {path}", file=sys.stderr)
        print("", file=sys.stderr)
        print("commit them first, e.g.:", file=sys.stderr)
        print('  git add -A -- . ":!results"', file=sys.stderr)
        print('  git commit -m "..."', file=sys.stderr)
        print("", file=sys.stderr)
        print("or, if you accept running against a dirty tree, re-run with --allow-dirty.", file=sys.stderr)
        return 1

    try:
        if is_any_self_check:
            run_dir = _create_self_check_run_dir(args.base_dir, args.run_id)
        else:
            run_dir = create_or_resume_run_dir(base_dir=args.base_dir, run_id=args.run_id)
    except (RunDirNotFoundError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        _print_dry_run(run_dir.path, registry, self_check=is_any_self_check)
        return 0

    _record_project_dirty(run_dir, project_dirty)

    resolved_config = dict(vars(args))
    resolved_config["run_id"] = run_dir.run_id
    resolved_config["resolved_at"] = datetime.now(timezone.utc).isoformat()
    resolved_config["project_dirty"] = project_dirty
    atomic_write_json(os.path.join(run_dir.path, "resolved_config.json"), resolved_config)

    from evodevo.parallel.topology import detect_topology

    events = EventLog(run_dir.events_path)
    telemetry = TelemetrySampler(run_dir.telemetry_path, interval_s=args.telemetry_interval, disk_path=run_dir.path)
    ctx = PreflightContext(
        run_dir=run_dir,
        events=events,
        telemetry=telemetry,
        topology=detect_topology(),
        config=resolved_config,
    )

    dashboard = None
    use_dashboard = not args.no_dashboard and sys.stdout.isatty()
    if use_dashboard:
        from evodevo.preflight.dashboard import Dashboard

        dashboard = Dashboard(ctx, registry, interval_s=args.dashboard_interval)
        dashboard.start()
    else:
        from evodevo.preflight.dashboard import PlainLogger

        dashboard = PlainLogger(ctx, registry, interval_s=args.dashboard_interval)
        dashboard.start()

    _install_signal_handlers()

    timeout_timer = None
    if args.timeout_s is not None:
        timeout_timer = threading.Timer(args.timeout_s, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timeout_timer.daemon = True
        timeout_timer.start()

    telemetry.start()
    exit_code = 0
    interrupted = False
    try:
        run_stages(
            ctx, registry,
            only=_split_csv(args.only), skip=_split_csv(args.skip), force=_split_csv(args.force),
        )
    except PreflightInterrupted:
        interrupted = True
        exit_code = 130
    except (DependencyNotDoneError, UnknownStageError) as e:
        print(f"error: {e}", file=sys.stderr)
        exit_code = 1
    except Exception as e:  # noqa: BLE001 -- a stage failure must exit cleanly
        # (non-zero, message printed), not dump a raw traceback over a live
        # tmux dashboard; the full error is already in manifest.json/events.jsonl.
        print(f"pre-flight stopped: {type(e).__name__}: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        if timeout_timer is not None:
            timeout_timer.cancel()
        if dashboard is not None:
            dashboard.stop()
        telemetry.stop()
        events.close()

    resume_cmd = f"python -m evodevo.preflight --run-id {run_dir.run_id} --base-dir {args.base_dir}"
    if interrupted:
        print(f"\ninterrupted -- telemetry stopped, events flushed, in-flight stage left resumable.", file=sys.stderr)
        print(f"resume with:\n  {resume_cmd}\n", file=sys.stderr)

    if is_any_self_check:
        kind = "self-check-full" if args.self_check_full else "self-check"
        note = f"{kind} complete" if exit_code == 0 else f"{kind} FAILED"
        print(f"\n{note} -- {SELF_CHECK_STAMP} -- run directory: {run_dir.path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
