"""`python -m evodevo.campaign --dry-run` (and `run_campaign.sh --dry-run`).

Writes NOTHING (no run directory, manifest, mirror, tmux session or
bytecode: __main__ sets sys.dont_write_bytecode first) and pins nothing
(worker/scheduler placement is computed, never applied). Prints the plan,
the pre-registered estimate as a calendar from --launch-at, the disk
projection against free space on the main disk and the mirror, and every
launch-refusal check's current status.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from .estimate import LEGACY_ESTIMATE_PATH as FINAL_ESTIMATE
DISK_ESTIMATE = REPO_ROOT / "results/campaign_prep/disk/disk_estimate.json"
GUARD_H = 432.0


def _existing(path):
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _free_gb(path):
    return shutil.disk_usage(_existing(path)).free / 1e9


def _stamp(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%d %H:%M %a")


def _command_output(*command):
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=10,
                              check=False).stdout.strip() or "not available"
    except (OSError, subprocess.SubprocessError):
        return "not available"


def operational_environment() -> dict:
    """Read-only facts relevant to detached multi-day campaign survival."""
    session = os.environ.get("XDG_SESSION_ID", "self")
    kill = _command_output("loginctl", "show-session", session, "-p", "KillUserProcesses", "--value")
    sshd = _command_output("systemctl", "is-active", "sshd")
    if sshd == "not available" or sshd in {"inactive", "unknown"}:
        sshd = _command_output("systemctl", "is-active", "ssh")
    tmux = _command_output("tmux", "list-sessions")
    return {"kill_user_processes": kill, "sshd": sshd, "tmux_sessions": tmux}


def design_applies(plan) -> tuple[bool, str]:
    """The committed estimate models ONE design: the accepted campaign
    profile (4 arms x seeds 0-4, pop 15 / eval_budget 300, D6 timesteps,
    D16 K). Anything else gets no calendar rather than a wrong one."""
    from evodevo.experiments.benchmark import CAMPAIGN, PROFILES
    profile = PROFILES[CAMPAIGN]
    estimate = json.loads(FINAL_ESTIMATE.read_text())
    design = estimate["final_configuration"]["design"]
    for phase in plan.phases:
        arms = sorted({run.arm for run in phase.runs})
        seeds = sorted({run.seed for run in phase.runs})
        cfg = phase.runs[0].config
        expected = (sorted(design["arms"][phase.task]), [0, 1, 2, 3, 4],
                    profile["ppo_timesteps"][phase.task], design["K"][phase.task],
                    profile["population_size"], profile["eval_budget"])
        actual = (arms, seeds, phase.ppo_timesteps, phase.ppo_seeds, cfg["population_size"], cfg["eval_budget"])
        if actual != expected:
            return False, f"{phase.task}: plan {actual} != estimated design {expected}"
    return True, "plan matches the estimated design exactly"


def estimate_applies(plan, workers: int) -> tuple[bool, str]:
    applies, reason = design_applies(plan)
    if not applies:
        return applies, reason
    estimated_workers = json.loads(FINAL_ESTIMATE.read_text())["inputs"]["workers"]["n"]
    if workers != estimated_workers:
        return False, (f"worker count {workers} != committed estimate's "
                       f"{estimated_workers}; no stale calendar is shown")
    return True, reason


def production_projection_applies(plan, args) -> bool:
    from .production import PROFILE, WORKER_CPUS
    return (args.workers == len(WORKER_CPUS) and args.cpu_profile == PROFILE
            and design_applies(plan)[0])


def render(args, plan, campaign_id) -> str:
    from evodevo.campaign.launch_checks import launch_refusals
    from evodevo.campaign.scheduler import cpu_placement
    from evodevo.parallel.topology import detect_topology
    from evodevo.preflight.dirty_check import dirty_paths_outside_results

    out = ["=== EvoDevo campaign DRY RUN (nothing is written, nothing is started) ==="]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = dirty_paths_outside_results(str(REPO_ROOT))
    out += [f"config: {args.config}", f"campaign id (would be): {campaign_id}",
            f"config hash: {plan.config_hash}", f"git commit: {commit}",
            f"dirty tree: {'YES -- ' + '; '.join(dirty) if dirty else 'no (clean)'}", ""]

    out.append(f"PLAN (D7 phase order) -- guard outcome: {plan.guard_outcome}")
    total_runs = 0
    for index, phase in enumerate(plan.phases, 1):
        arms = sorted({run.arm for run in phase.runs})
        seeds = sorted({run.seed for run in phase.runs})
        cfg = phase.runs[0].config
        total_runs += len(phase.runs)
        out.append(f"  phase {index}: {phase.task}: {len(phase.runs)} runs = {len(arms)} arms x "
                   f"{len(seeds)} seeds {seeds}; ppo_timesteps={phase.ppo_timesteps:,}; "
                   f"K (ppo_seeds)={phase.ppo_seeds} aggregation={phase.aggregation}; "
                   f"population={cfg['population_size']} offspring={cfg['offspring_size']} "
                   f"eval_budget={cfg['eval_budget']} genomes; PPO-task timeout={phase.timeout_s:.0f} s")
        out.append(f"           arms: {', '.join(arms)}")
    out.append(f"  total runs: {total_runs}")
    out.append("")

    from evodevo.experiments.benchmark import BenchmarkConfig
    from .estimate import is_v2, load
    from .plan import resolve_estimate_path
    estimate_file = resolve_estimate_path(BenchmarkConfig.from_yaml(args.config))
    v2_estimate = load(estimate_file)
    if is_v2(v2_estimate):
        return _render_v2(out, args, plan, v2_estimate, estimate_file)
    estimate = json.loads(FINAL_ESTIMATE.read_text())
    applies, why = estimate_applies(plan, args.workers)
    production_applies = production_projection_applies(plan, args)
    launch = dt.datetime.fromisoformat(args.launch_at) if args.launch_at else dt.datetime.now().replace(
        second=0, microsecond=0)
    result = estimate["guard_sequence"][-1]["result"]
    walker, upstepper, total = result["walker_phase_h"], result["upstepper_phase_h"], result["total_h"]
    if production_applies:
        from .production_runtime import projection
        measured = projection()
        out.append(f"THERMAL-VALIDATED PROJECTION ({measured['source']}; measured "
                   f"{measured['measured_genomes_per_minute']:.2f} genomes/min)")
        out.append(f"  projected total: about {measured['total_h']:.1f} h "
                   f"({measured['total_h'] / 24:.1f} days); this is a rate-scaled projection, "
                   "not a new confidence interval")
        out.append(f"  Walker-v0 phase: about {measured['walker_h']:.1f} h; UpStepper-v0 phase: "
                   f"about {measured['upstepper_h']:.1f} h")
        out.append(f"  Walker milestone at about {100 * measured['walker_fraction']:.1f}% of total runtime")
        out.append(f"  CALENDAR from launch {_stamp(launch)} (local time; approximate)")
        out.append(f"    Walker milestone / phase end: {_stamp(launch + dt.timedelta(hours=measured['walker_h']))}")
        out.append(f"    campaign end:                 {_stamp(launch + dt.timedelta(hours=measured['total_h']))}")
    elif not applies:
        out.append(f"ESTIMATE ({FINAL_ESTIMATE.relative_to(REPO_ROOT)}, {estimate['n_trials']} Monte Carlo "
                   f"trials; pre-registered budget guard: p95 <= {GUARD_H:.0f} h -> guard step applied: "
                   f"{estimate['guard_outcome']})")
        out.append(f"  NOT APPLICABLE to this plan ({why}); no calendar is shown.")
    else:
        out.append(f"ESTIMATE ({FINAL_ESTIMATE.relative_to(REPO_ROOT)}, {estimate['n_trials']} Monte Carlo "
                   f"trials; pre-registered budget guard: p95 <= {GUARD_H:.0f} h -> guard step applied: "
                   f"{estimate['guard_outcome']})")
        out.append(f"  total: mean {total['mean']:.1f} h, p5 {total['p5']:.1f} h, p95 {total['p95']:.1f} h "
                   f"({'within' if total['p95'] <= GUARD_H else 'OVER'} the {GUARD_H:.0f} h guard)")
        out.append(f"  Walker-v0 phase: mean {walker['mean']:.1f} h (p5 {walker['p5']:.1f}, p95 {walker['p95']:.1f}); "
                   f"UpStepper-v0 phase: mean {upstepper['mean']:.1f} h (p5 {upstepper['p5']:.1f}, "
                   f"p95 {upstepper['p95']:.1f})")
        pct = result["walker_milestone_pct_of_total"]["of_means"]
        out.append(f"  Walker milestone at {pct:.1f}% of total runtime (package generated automatically "
                   "when the Walker-v0 phase ends)")
        out.append(f"  CALENDAR from launch {_stamp(launch)} (local time)")
        out.append(f"    {'':<22} {'p5':<22}{'mean':<22}p95")
        for label, stats in (("Walker milestone", walker), ("Walker-v0 phase end", walker),
                             ("campaign end", total)):
            cells = [_stamp(launch + dt.timedelta(hours=stats[q])) for q in ("p5", "mean", "p95")]
            out.append(f"    {label:<22} {cells[0]:<22}{cells[1]:<22}{cells[2]}")
    out.append("")

    disk = json.loads(DISK_ESTIMATE.read_text())
    return _render_tail(out, args, disk, applies or production_applies, DISK_ESTIMATE.relative_to(REPO_ROOT))


def _design_matches_v2(plan, estimate) -> tuple[bool, str]:
    """A v2 estimate states the exact design it modelled; refuse to show a
    calendar for any other plan."""
    design = estimate.get("design", {})
    for phase in plan.phases:
        modelled = design.get("tasks", {}).get(phase.task)
        if modelled is None:
            return False, f"{phase.task} is not in the estimate"
        arms = sorted({run.arm for run in phase.runs})
        seeds = sorted({run.seed for run in phase.runs})
        cfg = phase.runs[0].config
        actual = dict(arms=arms, seeds=seeds, ppo_timesteps=phase.ppo_timesteps, ppo_seeds=phase.ppo_seeds,
                      population_size=cfg["population_size"], eval_budget=cfg["eval_budget"])
        expected = {key: modelled.get(key) for key in actual}
        if actual != expected:
            return False, f"{phase.task}: plan {actual} != estimated design {expected}"
    if [phase.task for phase in plan.phases] != list(estimate.get("result", {}).get("task_order", [])):
        return False, "phase order differs from the estimate"
    return True, "plan matches the estimated design exactly"


def _render_v2(out, args, plan, estimate, estimate_file):
    applies, why = _design_matches_v2(plan, estimate)
    launch = dt.datetime.fromisoformat(args.launch_at) if args.launch_at else dt.datetime.now().replace(
        second=0, microsecond=0)
    result, guard = estimate.get("result", {}), estimate.get("guard", {})
    try:
        label = estimate_file.relative_to(REPO_ROOT)
    except ValueError:
        label = estimate_file
    out.append(f"ESTIMATE ({label}; {estimate.get('method', 'v2 estimate')}; pre-registered guard: "
               f"p95 <= {guard.get('threshold_h', float('nan')):.0f} h -> outcome: {guard.get('outcome', 'n/a')})")
    if not applies:
        out.append(f"  NOT APPLICABLE to this plan ({why}); no calendar is shown.")
    elif estimate.get("workers") is not None and args.workers != estimate.get("workers"):
        out.append(f"  NOT APPLICABLE: worker count {args.workers} != estimate's {estimate.get('workers')}; "
                   "no calendar is shown.")
    else:
        total = result["total_h"]
        out.append(f"  total: mean {total['mean']:.1f} h, p5 {total['p5']:.1f} h, p95 {total['p95']:.1f} h "
                   f"({'within' if total['p95'] <= guard.get('threshold_h', float('inf')) else 'OVER'} the guard)")
        elapsed = {"p5": 0.0, "mean": 0.0, "p95": 0.0}
        out.append(f"  CALENDAR from launch {_stamp(launch)} (local time)")
        out.append(f"    {'':<30} {'p5':<22}{'mean':<22}p95")
        for task in result["task_order"]:
            band = result["phase_h"][task]
            for q in elapsed:
                elapsed[q] += band[q]
            out.append(f"    {task + ' (' + format(band['mean'], '.1f') + ' h)':<30} "
                       + "".join(f"{_stamp(launch + dt.timedelta(hours=elapsed[q])):<22}" for q in ("p5", "mean"))
                       + _stamp(launch + dt.timedelta(hours=elapsed['p95'])))
    out.append("")
    disk = estimate.get("disk") or {}
    return _render_tail(out, args, disk, applies, label)


def _render_tail(out, args, disk, disk_applies, disk_label):
    from evodevo.campaign.launch_checks import launch_refusals
    from evodevo.campaign.scheduler import cpu_placement
    from evodevo.parallel.topology import detect_topology
    from evodevo.preflight.dirty_check import dirty_paths_outside_results
    dirty = dirty_paths_outside_results(str(REPO_ROOT))
    refusals, record = launch_refusals(args)
    if not disk:
        disk = {"total_bytes": float("nan"), "three_x_total_bytes": float("nan")}
    need = disk["three_x_total_bytes"] / 1e9
    main_free = _free_gb(args.output_root)
    out.append(f"DISK ({disk_label}): projected {disk['total_bytes'] / 1e9:.1f} GB; "
               f"D10 3x margin requires {need:.1f} GB free"
               + ("" if disk_applies else " (projection is for the estimated design, not this plan)"))
    out.append(f"  main ({_existing(args.output_root)}): {main_free:.1f} GB free -> "
               f"{'OK' if main_free >= need else 'INSUFFICIENT'}")
    if args.no_mirror:
        out.append("  mirror: DISABLED (--no-mirror)")
    elif args.mirror_dir and Path(args.mirror_dir).is_dir():
        mirror_free = _free_gb(args.mirror_dir)
        out.append(f"  mirror ({args.mirror_dir}): {mirror_free:.1f} GB free -> "
                   f"{'OK' if mirror_free >= need else 'INSUFFICIENT'}")
    else:
        out.append(f"  mirror ({args.mirror_dir}): directory does not exist")
    mirror = record["mirror"]
    if not args.no_mirror:
        out.append(f"MIRROR PHYSICAL DISK: results {mirror.get('results_source')} on disk(s) "
                   f"{mirror.get('results_disks')}; mirror {mirror.get('mirror_source')} (mounted at "
                   f"{mirror.get('mirror_mount')}) on disk(s) {mirror.get('mirror_disks')}; "
                   f"same physical disk: {mirror.get('same_physical_disk')}")
    notifier = record["notifier"]
    out.append(f"NOTIFIER: {notifier['config_path']}: "
               + (f"configured ({', '.join(notifier['channels'])}); prove delivery with "
                  "`python -m evodevo.campaign.alerts --test`" if notifier["configured"]
                  else "NOT CONFIGURED" + (" (--no-notify given)" if args.no_notify else "")))
    out.append("")

    topology = detect_topology()
    affinity, scheduler_cpus = cpu_placement(topology, args.workers, args.cpu_profile)
    kinds = {core.logical_id: ("P-primary" if core.kind == "P" and core.is_physical_primary else
                               "P-sibling" if core.kind == "P" else core.kind + "-core")
             for core in topology.cores}
    out.append(f"CPU PLACEMENT (computed, not applied): profile={args.cpu_profile}; "
               f"{args.workers} workers, one CPU each"
               + (" -- OVERSUBSCRIBED" if affinity.oversubscribed else ""))
    for slot, cpus in enumerate(affinity.slots):
        out.append(f"  worker {slot:>2}: cpu {','.join(map(str, cpus))} ({kinds.get(cpus[0], '?')})")
    out.append(f"  scheduler: cpus {scheduler_cpus} ({', '.join(sorted({kinds.get(c, '?') for c in scheduler_cpus}))})")
    out.append("")
    environment = operational_environment()
    out.append("OPERATIONAL ENVIRONMENT (info only; no service or session is changed)")
    out.append(f"  systemd KillUserProcesses: {environment['kill_user_processes']}")
    out.append(f"  sshd active: {environment['sshd']}")
    out.append(f"  existing tmux sessions: {environment['tmux_sessions']}")
    out.append("")
    out.append("LAUNCH CHECKS: " + ("PASS -- a real launch would be accepted" if not refusals and not dirty
                                     else "WOULD BE REFUSED:"))
    for reason in (["dirty working tree"] if dirty else []) + refusals:
        out.append(f"  - {reason}")
    return "\n".join(out) + "\n"
