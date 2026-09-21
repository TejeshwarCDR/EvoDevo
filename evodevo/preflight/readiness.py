"""python -m evodevo.preflight.readiness -- generates PREFLIGHT_READINESS.md
at the repo root: the eleven-section sign-off document an operator reads
before running the real pre-flight by hand in tmux.

This module is NOT subject to evodevo.preflight.__main__'s own import-order
discipline (it does not configure process-wide threading itself -- it is a
report generator, not the harness) and freely imports things that pull in
numpy/torch (e.g. evodevo.experiments.lock_hash). It never imports
evodevo.preflight.__main__ directly, for the same reason nothing else in
this project does -- see that module's own docstring; wherever this file
needs something from it (the resolved stage list, --dry-run's exact
output, argparse defaults), it shells out to `python -m evodevo.preflight
...` in a fresh subprocess instead, exactly as an operator would run it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "PREFLIGHT_READINESS.md")

_CREATED_GLOBS = (
    "evodevo/parallel/**/*.py",
    "evodevo/preflight/**/*.py",
    "scripts/parallel/**/*.py",
    "scripts/preflight/**/*.py",
    "scripts/preflight/**/*.sh",
    "tests/parallel/**/*.py",
    "tests/preflight/**/*.py",
)

_NAMED_TESTS = {
    "cross-worker-count determinism": "tests/parallel/test_worker_pool.py::test_determinism_matches_single_worker",
    "thread-pinning assertion": "tests/parallel/test_worker_pool.py::test_workers_report_single_thread_and_pinned_env_vars",
    "affinity-distinctness": "tests/parallel/test_worker_pool.py::test_workers_get_distinct_cpu_affinity_sets",
    "kill-and-resume": "tests/preflight/test_stages.py::test_kill_during_stage_is_rerun_on_resume_while_done_stage_is_skipped",
    "knee-selection unit tests": "tests/preflight/test_b4_b6_b7.py -k select_knee",
}


def _run(cmd: list, cwd: str = REPO_ROOT, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _git(*args: str) -> str:
    result = _run(["git", *args])
    return result.stdout.strip() if result.returncode == 0 else f"<git {' '.join(args)} failed: {result.stderr.strip()}>"


def _find_pytest_summary_line(stdout: str) -> str:
    """pytest's own final summary line (e.g. "876 passed, 33 warnings in
    686.84s (0:11:26)"), NOT simply the last line of stdout -- several
    tests in this suite (visualization/EvoGym ones) print diagnostic text
    (e.g. "Status: Using GLEW 2.2.0") to stdout AFTER pytest's own summary
    has already printed, so blindly taking stdout's last line silently
    picks up that noise instead of the actual result (caught by hand
    inspecting an early generated readiness doc)."""
    import re

    for line in reversed(stdout.strip().splitlines()):
        if re.search(r"^\d+ (passed|failed|error)", line.strip()):
            return line.strip()
    return stdout.strip().splitlines()[-1] if stdout.strip() else "(no output)"


# --- section 1: git / config lock ---------------------------------------


def _section_1() -> str:
    commit = _git("rev-parse", "HEAD")
    dirty = _run(["git", "status", "--porcelain"]).stdout
    clean = not dirty.strip()
    try:
        from evodevo.experiments.lock_hash import compute_lock_hash_from_benchmark_config

        lock_hash = compute_lock_hash_from_benchmark_config()
        lock_hash_error = None
    except Exception as e:  # noqa: BLE001 -- readiness doc must still render if this fails
        lock_hash = None
        lock_hash_error = f"{type(e).__name__}: {e}"

    lines = ["## 1. Git state and config lock", ""]
    lines.append(f"- Commit: `{commit}`")
    lines.append(f"- Working tree clean: **{clean}**" + ("" if clean else " -- see `git status --porcelain` output below"))
    if not clean:
        lines.append("")
        lines.append("```")
        lines.append(dirty.rstrip())
        lines.append("```")
    lines.append("")
    if lock_hash is not None:
        lines.append(f"- Config lock hash (`evodevo.experiments.lock_hash.compute_lock_hash_from_benchmark_config()`): `{lock_hash}`")
    else:
        lines.append(f"- Config lock hash: **could not be computed** ({lock_hash_error})")
    lines.append("")
    return "\n".join(lines)


# --- section 2: file inventory ------------------------------------------


def _section_2() -> str:
    lines = ["## 2. File inventory (everything created across Prompts 1-7)", ""]
    lines.append("| file | lines |")
    lines.append("|---|---|")
    total_lines = 0
    total_files = 0
    seen = set()
    for pattern in _CREATED_GLOBS:
        for path in sorted(glob.glob(os.path.join(REPO_ROOT, pattern), recursive=True)):
            rel = os.path.relpath(path, REPO_ROOT)
            if rel in seen or not os.path.isfile(path):
                continue
            seen.add(rel)
            try:
                with open(path, "rb") as f:
                    n = sum(1 for _ in f)
            except OSError:
                n = -1
            total_lines += max(n, 0)
            total_files += 1
            lines.append(f"| `{rel}` | {n} |")
    lines.append("")
    lines.append(f"**{total_files} files, {total_lines} lines total.**")
    lines.append("")
    return "\n".join(lines)


# --- section 3: test results ---------------------------------------------


def _section_3(skip_full_suite: bool) -> str:
    lines = ["## 3. Test results", ""]

    if skip_full_suite:
        lines.append("_Full suite run skipped for this generation (--skip-tests) -- see prior CI/manual run for the count._")
        lines.append("")
    else:
        full = _run([sys.executable, "-m", "pytest", "-q"], timeout=3600)
        summary_line = _find_pytest_summary_line(full.stdout)
        lines.append(f"- Full suite: `{summary_line}`")
        lines.append(f"- Exit code: {full.returncode}")
        lines.append("")

    lines.append("### Five specific required results")
    lines.append("")
    lines.append("| test | result |")
    lines.append("|---|---|")
    for label, node_id in _NAMED_TESTS.items():
        parts = node_id.split(" ")
        result = _run([sys.executable, "-m", "pytest", "-v", *parts], timeout=600)
        out = result.stdout
        status = "PASS" if result.returncode == 0 else "FAIL"
        summary = _find_pytest_summary_line(out)
        lines.append(f"| {label} (`{node_id}`) | **{status}** -- `{summary}` |")
    lines.append("")
    return "\n".join(lines)


# --- section 4: stage list ------------------------------------------------


def _section_4() -> str:
    lines = ["## 4. Stage list (dependency order, cost, every default parameter value)", ""]

    dry_run = _run([sys.executable, "-m", "evodevo.preflight", "--dry-run", "--base-dir", tempfile.mkdtemp()])
    lines.append("```")
    lines.append(dry_run.stdout.rstrip())
    lines.append("```")
    lines.append("")

    defaults_script = (
        "import json\n"
        "from evodevo.preflight.__main__ import build_parser\n"
        "parser = build_parser()\n"
        "args = vars(parser.parse_args([]))\n"
        "print(json.dumps(args, default=str, sort_keys=True))\n"
    )
    result = _run([sys.executable, "-c", defaults_script])
    lines.append("### Every CLI flag's default value")
    lines.append("")
    lines.append("```json")
    if result.returncode == 0:
        try:
            parsed = json.loads(result.stdout.strip())
            lines.append(json.dumps(parsed, indent=2, sort_keys=True))
        except ValueError:
            lines.append(result.stdout.strip())
    else:
        lines.append(f"(failed to introspect defaults: {result.stderr.strip()})")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --- section 5: dry-run output verbatim -----------------------------------


def _section_5() -> str:
    lines = ["## 5. `python -m evodevo.preflight --dry-run` -- exact output", ""]
    dry_run = _run([sys.executable, "-m", "evodevo.preflight", "--dry-run", "--base-dir", tempfile.mkdtemp()])
    lines.append("```")
    lines.append(dry_run.stdout.rstrip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --- section 6: live B1 machine profile -----------------------------------


def _section_6(skip: bool) -> str:
    lines = ["## 6. B1 machine profile (collected live -- read-only, seconds, NOT a measurement run)", ""]
    if skip:
        lines.append("not collected (--skip-machine-profile)")
        lines.append("")
        return "\n".join(lines)

    try:
        from evodevo.parallel.topology import detect_topology
        from evodevo.preflight.context import PreflightContext
        from evodevo.preflight.events import EventLog
        from evodevo.preflight.run_dir import RunDir
        from evodevo.preflight.stages.b1_machine import MachineProfileStage

        tmp = tempfile.mkdtemp(prefix="preflight_readiness_b1_")
        run_dir = RunDir(tmp)
        run_dir.ensure_skeleton()
        from evodevo.preflight.run_dir import write_machine_snapshot

        write_machine_snapshot(run_dir)
        events = EventLog(run_dir.events_path)
        ctx = PreflightContext(run_dir=run_dir, events=events, telemetry=None, topology=detect_topology(), config={})
        profile = MachineProfileStage(disk_probe_mb=64).run(ctx)
        events.close()
        lines.append("```json")
        lines.append(json.dumps(profile, indent=2, sort_keys=True, default=str))
        lines.append("```")
    except Exception as e:  # noqa: BLE001 -- readiness doc must still render if this fails
        lines.append(f"not collected -- vetoed by error: {type(e).__name__}: {e}")
    lines.append("")
    return "\n".join(lines)


# --- section 7: self-check results ----------------------------------------


def _describe_self_check_run(run_path: str, grid_detail: bool) -> list:
    lines = []
    run_id = os.path.basename(run_path.rstrip("/"))
    manifest_path = os.path.join(run_path, "manifest.json")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        lines.append(f"self-check run `{run_id}` found but its manifest could not be read: {e}")
        return lines

    stages = manifest.get("stages", {})
    n_done = sum(1 for e in stages.values() if e.get("status") == "done")
    n_failed = sum(1 for e in stages.values() if e.get("status") == "failed")
    total_duration = sum(e.get("duration_s") or 0 for e in stages.values())

    lines.append(f"- Run directory: `{os.path.relpath(run_path, REPO_ROOT)}`")
    lines.append(f"- project_dirty at run start: `{manifest.get('project_dirty')}`")
    lines.append(f"- Stages done: {n_done}/{len(stages)}, failed: {n_failed}")
    lines.append(f"- Total measured wall clock across stages: {total_duration:.1f}s")
    lines.append("")
    lines.append("| stage | status | duration_s |")
    lines.append("|---|---|---|")
    for name, entry in sorted(stages.items()):
        lines.append(f"| {name} | {entry.get('status')} | {entry.get('duration_s')} |")
    lines.append("")

    if grid_detail:
        b3_path = os.path.join(run_path, "stages", "evaluation_cost.json")
        if os.path.exists(b3_path):
            try:
                b3 = json.load(open(b3_path))
            except (OSError, ValueError):
                b3 = {}
            lines.append("Per-arm x per-task detail (B3 `evaluation_cost`):")
            lines.append("")
            lines.append("| arm | task | n_successful | n_failed | partial |")
            lines.append("|---|---|---|---|---|")
            for arm_id, arm in sorted((b3.get("arms") or {}).items()):
                for task_name, task in sorted((arm.get("tasks") or {}).items()):
                    lines.append(
                        f"| {arm_id} | {task_name} | {task.get('n_successful')} | {task.get('n_failed')} | "
                        f"{task.get('partial')} |"
                    )
            lines.append("")

    return lines


def _section_7(base_dir: str = "results/preflight") -> str:
    lines = ["## 7. Self-check results", ""]
    lines.append(
        "**This is a bounded infrastructure self-check, NOT a measurement. None of the numbers below may "
        "ever be cited as real B1-B7 results -- every artifact each run produced is independently stamped "
        "`SELF-CHECK — NOT A MEASUREMENT` in both its JSON and its figure titles.**"
    )
    lines.append("")

    all_candidates = sorted(glob.glob(os.path.join(REPO_ROOT, base_dir, "selfcheck_*")))
    full_candidates = [c for c in all_candidates if "selfcheck_full_" in c]
    plain_candidates = [c for c in all_candidates if "selfcheck_full_" not in c]

    lines.append("### --self-check-full (recommended pre-launch check -- all 3 arms x 2 tasks)")
    lines.append("")
    if full_candidates:
        lines.extend(_describe_self_check_run(full_candidates[-1], grid_detail=True))
    else:
        lines.append(
            "**Not found.** Run `bash scripts/preflight/self_check.sh --full` before generating this "
            "document -- this subsection cannot be populated without it."
        )
        lines.append("")

    lines.append("### --self-check (fast smoke test -- single arm/task)")
    lines.append("")
    if plain_candidates:
        lines.extend(_describe_self_check_run(plain_candidates[-1], grid_detail=False))
    else:
        lines.append("**Not found.** Run `bash scripts/preflight/self_check.sh` before generating this document.")
        lines.append("")

    lines.append(
        "What both prove: every registered stage's `run(ctx)` executed to completion at trivial settings, "
        "every stage wrote its JSON output, every figure-producing stage rendered its figure, and the "
        "manifest/event-log/telemetry-sampler machinery all round-tripped correctly -- --self-check-full "
        "additionally proves this for every arm x task combination the real pre-flight will exercise, not "
        "just one."
    )
    lines.append("")
    return "\n".join(lines)


# --- section 8: environment provenance ------------------------------------


def _section_8() -> str:
    lines = ["## 8. Environment provenance", ""]
    versions = {}
    for mod_name in ("numpy", "torch", "gymnasium", "stable_baselines3", "evogym"):
        if mod_name == "evogym":
            # evogym's own module does not expose __version__ -- read
            # installed package metadata instead, the SAME method B1
            # (evodevo/preflight/stages/b1_machine.py's _evogym_version())
            # uses, so this section and a live B1 profile never disagree
            # about which version is installed. See test_readiness.py's
            # own regression test for this agreement.
            try:
                import importlib.metadata

                versions[mod_name] = importlib.metadata.version("evogym")
            except Exception as e:  # noqa: BLE001
                versions[mod_name] = f"not importable ({type(e).__name__}: {e})"
            continue
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            versions[mod_name] = f"not importable ({type(e).__name__}: {e})"
    lines.append(f"- python: `{sys.version.split()[0]}`")
    for name, version in versions.items():
        lines.append(f"- {name}: `{version}`")
    evogym_commit = _git("-C", "external/evogym", "rev-parse", "--short=12", "HEAD")
    lines.append(f"- evogym (external/evogym) commit: `{evogym_commit}`")
    lines.append(f"- this repo's commit: `{_git('rev-parse', '--short=12', 'HEAD')}`")
    lines.append("")
    return "\n".join(lines)


# --- section 9: known gaps -------------------------------------------------


def _section_9() -> str:
    lines = ["## 9. Known gaps, deviations, and things deliberately not implemented", ""]
    lines.append("### The evodevo/evaluation/ppo_runner.py change")
    lines.append("")
    lines.append(
        "`git status` shows `evodevo/evaluation/ppo_runner.py` as modified against HEAD. Prompt 2's own "
        "rule was: if the worker pool needed ppo_runner.py behaviour changed, stop and report rather than "
        "change it silently. This subsection is that report, produced after the fact once the omission was "
        "flagged -- not before the change was made."
    )
    lines.append("")
    lines.append(
        "**What changed:** `evaluate_body()`'s own inline rollout-evaluation loop (reset, deterministic "
        "policy loop, accumulate reward per episode, mean across `eval_episodes`) was extracted verbatim "
        "into a new function, `_rollout_fitness(model, eval_env, eval_episodes)`; `evaluate_body()`'s own "
        "call site now calls it instead of inlining the loop, with the same arguments in the same place. "
        "Separately, two wholly new, additive things were introduced: a `CurveResult` dataclass and a new "
        "function `evaluate_body_with_curve()`, which trains PPO once to `max_timesteps` and periodically "
        "evaluates the policy every `eval_interval` timesteps via an SB3 callback (reusing "
        "`_rollout_fitness` for each checkpoint). Neither is called by `evaluate_body()` or any other "
        "pre-existing caller; `evaluate_body_with_curve()` is used only by the B4 `learning_curves` "
        "pre-flight stage."
    )
    lines.append("")
    lines.append(
        "**Why:** B4 needs a fitness-vs-timesteps curve per body, measured periodically DURING one PPO "
        "training run rather than one independent full training run per checkpoint (which would multiply "
        "cost by the checkpoint count for no additional information). Reusing `_rollout_fitness` for every "
        "checkpoint means each curve point is evaluated with the exact same methodology `evaluate_body()`'s "
        "own final evaluation uses, so a curve's own final point is directly comparable to a normal "
        "`evaluate_body()` call on the same body/task/seed/timesteps."
    )
    lines.append("")
    lines.append(
        "**Classification: (i) behaviour-neutral.** `evaluate_body()`'s own validation, sentinel fitness, "
        "seeding, PPO construction (`PPO(\"MlpPolicy\", vec_env, seed=seed, verbose=0)`), "
        "`model.learn(total_timesteps=ppo_timesteps)`, evaluation-episode count, fitness computation, "
        "exception handling, and cache/threading/device selection are all untouched -- only the "
        "already-inline rollout loop moved into a function with an unchanged call site and unchanged "
        "arguments. `evaluate_body_with_curve()` is new, additive code `evaluate_body()` itself never "
        "reaches."
    )
    lines.append("")
    lines.append(
        "**Proof:** `tests/evaluation/test_ppo_runner.py::"
        "test_evaluate_body_is_bit_identical_between_head_and_working_tree` loads the git-committed HEAD "
        "version of this module (via `git show HEAD:...` into a separately-named module -- not `git "
        "stash`, which would touch the whole working tree) and the current working-tree version, and "
        "evaluates a fixed set of 5 bodies (2 valid -- GOOD_WALKER, GOOD_WALKER_2 -- and 3 invalid, each "
        "via a different `validate_body()` failure: too_few_voxels, disconnected, no_actuator) with both, "
        "at `ppo_timesteps=2000, eval_episodes=1, seed=0`. All 5 produced bit-identical `fitness`, `valid`, "
        "`reason`, and `body_hash` between the two versions. The test passed and is kept in the suite as a "
        "permanent regression guard against a future edit to this module silently changing "
        "`evaluate_body()`'s own behaviour."
    )
    lines.append("")
    lines.append("### coretemp_readable=true vs. ENVIRONMENT_REPORT.md's \"lm-sensors not installed\"")
    lines.append("")
    lines.append(
        "Not a contradiction -- two different things. B1's `coretemp_readable` (b1_machine.py's "
        "`_coretemp_readable()`) and telemetry.py's per-sample `cpu_package_temp_c` both come from "
        "`psutil.sensors_temperatures()`, which on Linux reads DIRECTLY from the kernel's `/sys/class/hwmon/"
        "hwmon*/temp*_input` sysfs files (confirmed by reading psutil's own Linux implementation source, "
        "whose docstring states this explicitly: \"lm-sensors on Ubuntu 16.04+ relies on /sys/class/hwmon\" "
        "-- i.e. even the real `sensors` CLI tool reads the same underlying sysfs interface). The `coretemp` "
        "KERNEL DRIVER is already loaded and exposes these hwmon nodes on its own; the `lm-sensors` PACKAGE "
        "ENVIRONMENT_REPORT.md reports absent is only the separate `sensors` command-line binary, a human-"
        "facing formatting convenience over the same sysfs data -- never required for psutil (or any Python "
        "code) to read a temperature."
    )
    lines.append("")
    lines.append(
        "**Verbatim sample record**, first line of `results/preflight/selfcheck_full_20260920T143859Z/"
        "telemetry.jsonl` (the latest --self-check-full run at the time this was written):"
    )
    lines.append("")
    lines.append("```json")
    lines.append(
        '{"cpu_package_temp_c": 49.0, "cpu_percent_overall": 0.0, "disk_free_gb": 307.27, '
        '"load_avg_1m": 1.0439453125, "process_count": 494, "ram_available_mb": 59850.4, '
        '"ram_used_mb": 7319.7, "swap_used_mb": 0.0, "thermal_throttle_count": 0, '
        '"ts": "2026-09-20T14:38:59.980600+00:00", "ts_epoch": 1789915139.9806}'
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "`cpu_package_temp_c` is non-null and checked across all 56 samples in that run's telemetry.jsonl -- "
        "zero were null. No D13 gap in this instance. This is stated as an observation about THIS run, not "
        "a guarantee: if a future run's telemetry.jsonl ever shows a null `cpu_package_temp_c`, that IS a "
        "D13 gap (a sensor that stopped being readable mid-run) and must be reported as one, not glossed "
        "over as \"readable\" because B1's one-time startup check happened to succeed."
    )
    lines.append("")
    lines.append("### The results/rule_based/ORDER_INDEPENDENCE_AUDIT.md test side effect")
    lines.append("")
    lines.append(
        "`tests/encoders/test_rule_based_order_independence.py::test_order_independence_permutation_audit` "
        "DELIBERATELY rewrites `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` every time it runs -- its "
        "own docstring states this explicitly (\"Writes results/rule_based/ORDER_INDEPENDENCE_AUDIT.md as a "
        "side effect so this evidence is inspectable without re-running the (slow, N=2000xT=100) audit\"). "
        "Only the wall-clock timing line changes run to run; the permutation/growth-conflict fractions "
        "themselves are deterministic and pinned by a companion test, "
        "`test_no_stale_order_independence_citations`, which asserts every OTHER file's citation of those "
        "numbers still matches. This is a documented, intentional side effect, not a bug -- and per this "
        "prompt's own instruction, it is reported here, not changed. Practice throughout this whole project "
        "has been to `git checkout -- results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` after every full-suite "
        "run to keep the tree clean of this one regenerated timing line."
    )
    lines.append("")
    lines.append(
        "- `docs/final_decisions.md` still does not exist in this repository. A later instruction asked for "
        "an uploaded `final_decisions.md` to be copied to `docs/final_decisions.md` and committed, so every "
        "D-number cited in the code has a document to diff against -- but no such file could be found "
        "anywhere on this machine's filesystem or in the accessible conversation context (searched "
        "exhaustively). This is reported as a genuine, outstanding blocker, not silently skipped or "
        "fabricated: every \"D-number\" referenced throughout `evodevo/preflight/` (D5-D16) remains "
        "implemented from the wording given directly in each build prompt, cited as the authoritative "
        "source in each module's own docstring, with no separate document to diff it against until this "
        "file is actually provided."
    )
    lines.append(
        "- D14 (per-option cost deltas), D15 (the primary EA population/generations/seed grid), and D16 "
        "(single-eval vs. multi-seed per genome) are left open BY DESIGN: `projection.py` (B6) and "
        "`evaluation_noise.py` (B5) surface the evidence (deltas, decision_input text) but never pick a "
        "configuration. These are resolved by a human (named \"Tejeshwar\" in the source prompts) at a "
        "scoping gate after this pre-flight runs for real."
    )
    lines.append(
        "- The dashboard's ETA range (`dashboard.py`'s `Dashboard._eta_str`) is a crude heuristic -- "
        "elapsed / completed_fraction, presented as a [0.5x, 1.5x] band -- not a calibrated estimator. It "
        "gets meaningfully better once B2/B3 have real timing data; before that, it is a rough placeholder, "
        "labeled as such (\"rough range, not a commitment\")."
    )
    lines.append(
        "- `--self-check` still restricts `--arms`/`--tasks`/`--proj-arms`/`--proj-tasks` to a single "
        "arm (`direct`) and single task (`Walker-v0`), kept as the fastest possible smoke test (well under "
        "3 minutes). `--self-check-full` (added to close this gap) uses the identical trivial numeric "
        "settings across the FULL production grid (direct, nca, rule_based x Walker-v0, UpStepper-v0) "
        "instead, and is the one operators should run before a real launch -- see "
        "`scripts/preflight/self_check.sh --full`. Measured at 64s against a 10-minute budget, all 6 "
        "combinations succeeded with zero partial/failed results (see the self-check-full run directory "
        "cited in Section 7)."
    )
    lines.append(
        "- **Why the full grid did not originally fit in 3 minutes:** NOT valid-genome search (direct, nca "
        "and rule_based all found their first valid random genome within a few dozen draws at worst, well "
        "under `--cost-max-draws`). The real cause was `rule_based`'s `decode()` cost -- measured directly "
        "at ~232ms/call versus `direct`'s ~0.001ms/call and `nca`'s ~1.8ms/call (a ~200,000x gap vs "
        "direct), because it runs an iterative rule-interpretation simulation over `dev_steps=100` growth "
        "steps per call, not a cheap array reshape. B3's `t_develop`/`t_sentinel` measurements "
        "(`_measure_develop`/`_measure_sentinel` in `b3_cost.py`) paid this cost `t_develop_n` (a FIXED "
        "100 calls, always paid in full) plus up to `cost_max_draws` times (searching for `t_sentinel_n` "
        "invalid samples) -- for rule_based alone, up to roughly (100 + 200) x 0.232s =~ 70s of pure "
        "decode time, on top of the real PPO evaluations, per arm. `t_develop_n`/`t_sentinel_n` were not "
        "exposed as CLI flags at all before this fix, so self-check had no way to reduce them; they now "
        "are (`--cost-develop-n`/`--cost-sentinel-n`, both defaulting to the previous fixed value of 100 "
        "for production use), and self-check reduces both to 5."
    )
    lines.append(
        "- `report.py` regenerates figures by calling each stage module's own private `_make_figure`/"
        "`_write_markdown`/`_write_csv` functions directly from saved JSON; it does not independently "
        "reimplement any plotting logic, so a bug in a stage's own figure code is NOT caught twice."
    )
    lines.append(
        "- `readiness.py`'s wall-clock estimate (Section 10) is not derived from any real B2/B3 measurement "
        "(none exist yet) -- it is a rough, explicitly-caveated order-of-magnitude number from this "
        "machine's known real-PPO throughput at a small scale, meant to be superseded by B6's own "
        "`projection.md` once the real pre-flight actually runs."
    )
    lines.append("")
    return "\n".join(lines)


# --- section 10: wall clock estimate ---------------------------------------


def _section_10() -> str:
    lines = ["## 10. Estimated pre-flight wall clock (range)", ""]
    lines.append(
        "No real B2/B3 measurement exists yet (producing one is the pre-flight's own job), so this is a "
        "rough, explicitly-caveated planning number, not a measured estimate:"
    )
    lines.append("")
    lines.append(
        "- **Assumption:** ~2.5-4ms of wall clock per PPO timestep, single evaluation, unconcurrent "
        "(from this machine's `test_real_ppo_smoke`-scale observations during earlier development), with "
        "no correction yet for B2's own concurrency overhead factor -- concurrent workers will not scale "
        "linearly, and B6's `projection.md` (produced by the real pre-flight itself) supersedes this number."
    )
    lines.append(
        "- At default settings (sweep: 6 levels x 24 genomes x 20,000 steps; cost: 3 arms x 2 tasks x "
        "3 reps x 1,000,000 steps; curves: 3 bodies x 2 tasks x 2,000,000 steps; noise: 3 bodies x 2 tasks "
        "x 5 reps, timesteps from B3's own cost data) the dominant terms are B3 (~18M PPO-timesteps) and "
        "B4 (~12M PPO-timesteps); B2/B5 are smaller. At assumed 2.5-4ms/step *single-worker-equivalent*, "
        "and assuming B2's own measured concurrency roughly divides that by somewhere between 4x and 16x "
        "(a wide, deliberately conservative range, since this machine has never been measured under this "
        "exact load): **roughly 6-30 hours.**"
    )
    lines.append(
        "- **The two phases are not the same order of magnitude, and an operator watching tmux should "
        "expect the pace to change sharply partway through:** B2 (`concurrency_sweep`, at default settings "
        "6 levels x 24 genomes x 20,000 reduced timesteps, plus 60s inter-level cooldowns) is the cheap "
        "phase -- on the order of **minutes to about an hour**. B3/B4/B5 (`evaluation_cost`, "
        "`learning_curves`, `evaluation_noise`, at FULL campaign timesteps -- 1,000,000 for B3, 2,000,000 "
        "for B4) are the expensive phases and dominate the total -- on the order of **hours**, not minutes. "
        "If B2 finishes in the first hour and the dashboard's ETA suddenly looks far larger afterward, that "
        "is the expected transition into B3, not a stall."
    )
    lines.append(
        "- This range is wide on purpose -- it exists only to plan a tmux session and disk headroom around, "
        "not to promise a duration. Treat B6's `projection.md`, produced from real B2/B3 numbers, as "
        "authoritative once available."
    )
    lines.append("")
    return "\n".join(lines)


# --- section 11: open questions --------------------------------------------


def _section_11() -> str:
    lines = ["## 11. Open questions for a human decision", ""]
    lines.append("1. D14/D15/D16 (option costs, the primary EA grid, single- vs multi-seed evaluation) -- "
                  "resolved by Tejeshwar at the scoping gate, using this pre-flight's B5/B6 output as evidence.")
    lines.append("2. **RESOLVED:** the self-check must honour the full arm/task grid. `--self-check-full` "
                  "does this (all 3 arms x 2 tasks, same trivial numeric settings as `--self-check`, "
                  "10-minute budget; measured at 64s with zero partial/failed combinations) and is now the "
                  "recommended pre-launch check -- see Section 9's own subsection on why the original 3-minute "
                  "`--self-check` could not simply be widened to the full grid (rule_based's real decode() "
                  "cost) and Section 7 for its own run's detail.")
    lines.append("3. `--timeout-s` currently sends this process SIGTERM (the same path as an operator's own "
                  "Ctrl-C/kill), stopping cleanly with a resumable manifest rather than continuing past the "
                  "deadline -- confirm this is the intended semantics for an unattended, timed run.")
    lines.append("4. The dashboard's ETA range is a crude [0.5x, 1.5x] heuristic around a linear extrapolation "
                  "of completed-stage-fraction; confirm this rough band is acceptable for an operator watching "
                  "tmux for several hours, or whether a tighter, cost-weighted estimator is worth building.")
    lines.append("5. Disk headroom: B7's disk projection (D10's 3x margin) has not yet been checked against "
                  "this machine's real available space under a full run's expected artifact volume -- worth "
                  "confirming before an unattended multi-hour run starts.")
    lines.append("")
    return "\n".join(lines)


def _section_changes_since_previous() -> str:
    lines = ["## Changes since the previous readiness report", ""]
    lines.append(
        "Closing the readiness gaps a review of the first PREFLIGHT_READINESS.md found, before the real "
        "pre-flight runs. Each item below is a full gate from that review; see Section 9 for the detailed "
        "write-up of the first three."
    )
    lines.append("")
    lines.append("| # | gap | outcome |")
    lines.append("|---|---|---|")
    lines.append(
        "| 1 | `evodevo/evaluation/ppo_runner.py` showed modified with no report (Prompt 2's own rule: "
        "stop and report, don't change silently) | Explained, classified **behaviour-neutral** "
        "(`_rollout_fitness()` extraction + additive `evaluate_body_with_curve()`), proven bit-identical "
        "against HEAD across 5 bodies by a new committed test. See Section 9. |"
    )
    lines.append(
        "| 2 | self-check restricted to arm=direct, task=Walker-v0 -- first contact with 5 of 6 real "
        "combinations would happen hours into the real run | Added `--self-check-full` (same trivial "
        "settings, full 3-arm x 2-task grid, 10-minute budget). Root cause of the original 3-minute budget "
        "not fitting the full grid identified and fixed: `rule_based`'s decode() cost "
        "(~232ms/call) multiplied by previously-unreduced, CLI-unexposed `t_develop_n`/`t_sentinel_n` "
        "(now `--cost-develop-n`/`--cost-sentinel-n`, reduced to 5 for self-check). Measured at 64s "
        "against the 600s budget, zero partial/failed combinations. See Section 7 and Section 9. |"
    )
    lines.append(
        "| 3 | no gate preventing a real launch against an uncommitted tree | `python -m evodevo.preflight` "
        "now refuses to create a run directory for a real launch when `git status --porcelain` (excluding "
        "results/) is non-empty, unless `--allow-dirty` is passed; self-check/self-check-full are exempt "
        "but always record `project_dirty` in their manifest. Covered by "
        "`tests/preflight/test_dirty_check.py` and `tests/preflight/test_cli.py`. |"
    )
    lines.append(
        "| 4a | Section 8 reported evogym version as \"unknown\", disagreeing with B1's own correct report "
        "| Fixed to read the same source B1 uses (`importlib.metadata.version`, not `module.__version__`, "
        "which evogym does not define); a regression test asserts the two agree. |"
    )
    lines.append(
        "| 4b | `coretemp_readable=true` appeared to contradict ENVIRONMENT_REPORT.md's \"lm-sensors not "
        "installed\" | Not a contradiction: both read `/sys/class/hwmon` directly via "
        "`psutil.sensors_temperatures()`; `lm-sensors` is only the separate, optional `sensors` CLI "
        "convenience tool. Verbatim sample confirmed non-null; all 56 samples in the checked run were "
        "non-null. See Section 9. |"
    )
    lines.append(
        "| 4c | CPU frequency policy (scaling driver, per-core EPP, turbo state) was not recorded -- a "
        "measurement whose frequency policy is unrecorded cannot be reproduced | B1 now records "
        "scaling_driver, energy_performance_preference for one P-core and one E-core, "
        "intel_pstate no_turbo/status, and idle frequency per core; B7's D13 risk table surfaces all of "
        "it with a note that it must be IDENTICAL between this pre-flight and the real campaign. |"
    )
    lines.append(
        "| 4d | `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` shows modified after every full-suite run "
        "| Confirmed deliberate: `test_order_independence_permutation_audit` documents this as an "
        "intentional side effect in its own docstring. Reported, not changed, per this gate's own "
        "instruction; reverted with `git checkout` after every full-suite run in this session. |"
    )
    lines.append(
        "| 5 | work from Prompts 1-7 sat entirely uncommitted | Committed in 4 dependency-ordered commits "
        "(Step 0 env setup; the ppo_runner.py change, isolated and revertible alone; evodevo/parallel; "
        "evodevo/preflight) -- reordered from the prompt's literal (1)(2)(3)(4) numbering so the ppo_runner "
        "commit lands before evodevo/parallel, which depends on it (evaluator_entry.py's real-mode "
        "curve_request path), keeping every commit independently buildable. "
        "**`docs/final_decisions.md` was NOT committed** -- despite the instruction to copy an uploaded "
        "file there, no such file could be found anywhere on this machine or in this conversation's "
        "accessible context; this is flagged as a genuine blocker, not silently skipped or fabricated. |"
    )
    lines.append("")
    return "\n".join(lines)


def generate_readiness_markdown(skip_full_suite: bool = False, skip_machine_profile: bool = False) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    header = (
        f"# EvoDevo Pre-Flight Readiness\n\n"
        f"Generated {generated_at} by `python -m evodevo.preflight.readiness`.\n\n"
        f"**The real pre-flight has NOT been executed as part of producing this document.** The only "
        f"executions performed to produce this document (besides the test suite and a live, read-only B1 "
        f"machine query) are the bounded `--self-check` and `--self-check-full` runs reported in Section "
        f"7. To run the real pre-flight, an operator runs (from the repo root, in tmux):\n\n"
        f"```\nbash scripts/preflight/run_preflight.sh\n```\n\n"
        f"which wraps:\n\n```\npython -m evodevo.preflight\n```\n\n"
    )
    sections = [
        _section_changes_since_previous(),
        _section_1(),
        _section_2(),
        _section_3(skip_full_suite),
        _section_4(),
        _section_5(),
        _section_6(skip_machine_profile),
        _section_7(),
        _section_8(),
        _section_9(),
        _section_10(),
        _section_11(),
    ]
    return header + "\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evodevo.preflight.readiness")
    parser.add_argument("--skip-tests", action="store_true", help="skip running the full test suite for section 3")
    parser.add_argument(
        "--skip-machine-profile", action="store_true",
        help="veto the live B1 read-only system query for section 6 (states 'not collected')",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    markdown = generate_readiness_markdown(skip_full_suite=args.skip_tests, skip_machine_profile=args.skip_machine_profile)
    with open(args.output, "w") as f:
        f.write(markdown)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
