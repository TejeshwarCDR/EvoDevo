# Claude Code prompts for the extension campaign (lab host)

Run these one at a time in Claude Code on the lab machine, in the repository root. Send each prompt's report
back before running the next one. The sequence and commands are also in [RUNBOOK_EXT.md](RUNBOOK_EXT.md).

**The extension is merged into `main`. Do NOT run `git pull` (or Prompt 1) on the lab machine until the
original campaign (Walker-v0 and UpStepper-v0) is complete.** The running campaign's milestone subprocess and
any rebuilt worker import whatever code is on disk at that moment. Only Prompt 0 may be run while it runs.

| Prompt | When | Changes anything? |
|---|---|---|
| 0 | now, while the original campaign runs | no (writes one report under `results/ext_prep/`) |
| 1 | after the original campaign is `complete` | reconciles `main` with the rewritten history (no pull or fast-forward); runs the tests and a stamped rehearsal campaign |
| 2 | after prompt 1 passes | starts the extension preflight (~9–11 h, unattended) |
| 3 | after the preflight finishes | derives decisions, commits them to `main` and pushes (no force) |
| 4 | after prompt 3 | writes the audit report; launch only after an explicit GO |

---

## Prompt 0 — read-only record of the running campaign

```
Produce a read-only record of the original EvoDevo campaign while it runs. This is READ-ONLY.

HARD CONSTRAINTS
- Do NOT pull, merge, check out, edit, stage, commit or stash anything, and run no `git fetch`.
- Do NOT touch the running campaign: no writes, moves or deletes under results/campaign/, no signals, no
  edits to campaign.local.yaml or anything under results/campaign_prep/. Reading is fine.
- Write nothing under results/campaign/ or results/campaign_prep/ and no signals to any campaign process.
- The only code you may run is read-only status tooling and the probe in section E, which must run with
  `nice -n 19 taskset -c 27` and single-threaded env vars (OMP/MKL/OPENBLAS_NUM_THREADS=1).
- Redact the ntfy topic and any token as <REDACTED>.
- Write exactly one file: results/ext_prep/LAB_RECORD_<UTC date>.md (create the folder). Print its path.

SECTIONS
A. Git (read-only, no fetch): branch, HEAD (hash, date, message), `git status --short`,
   `git log origin/main..HEAD --oneline` and `git log HEAD..origin/main --oneline` against the refs already
   present locally, and `git ls-remote --heads origin` (network read, writes nothing).
B. The running campaign (find it with `ls -t results/campaign/*/manifest.json | head`):
   - campaign_id, manifest status, phase, phase_status (with ISO timestamps), started_at, git_commit,
     launch_checks (summarised), mirror_dir (redacted to mount name), no_notify.
   - The full output of `scripts/campaign/status_campaign.sh --campaign <ID>`.
   - Per-arm progress on UpStepper-v0 (evaluations done / 300 per seed) and the dashboard ETA range.
   - From results/campaign/<ID>/phase_metrics.json: every field for Walker-v0 (and UpStepper-v0 if
     present).
   - From journal.jsonl: PPO-task counts and summed result.wall_clock_s per (arm, task); count of rows
     whose result.reason starts with "infra_failure:"; any row with completed=false (count by reason).
   - From alerts.jsonl: counts by condition and status, and the last 5 alerts verbatim.
   - Whether the campaign was ever stopped/resumed: search transcript.log for "Resume:" and for scheduler
     start lines; report each start with its timestamp.
   - Walker-v0 milestone package: list milestones/Walker-v0/ (top level and best_bodies/), the
     package_manifest.json status/completion_fraction, and stats/final_fitness_summary.csv verbatim.
C. Files the extension depends on (exist? size? mtime?): results/campaign_prep/final_estimate/final_estimate.json,
   results/campaign_prep/disk/disk_estimate.json, results/mutation/mutation_calibration_raw.json,
   milestones/Walker-v0/best_bodies/{direct,nca,rule_based}.npz.
D. Machine: `lscpu | head -20`, `free -h`, `df -h / <backup mount>`, `uptime`, current package temperature
   and the last hour's max if the dashboard shows it, `.venv/bin/python -V`, `.venv/bin/pip show evogym
   stable-baselines3 torch gymnasium numpy | grep -E 'Name|Version'`, `tmux ls`.
E. Environment probe (single-threaded, nice 19, taskset -c 27): with the Walker-v0 best body
   milestones/Walker-v0/best_bodies/direct.npz, build `gym.make(env_id, body=body,
   connections=get_full_connectivity(body))` for Walker-v0, UpStepper-v0, ObstacleTraverser-v1,
   Climber-v0 and CaveCrawler-v0 (import evogym.envs first). For each: observation and action shapes,
   spec.max_episode_steps, and 300 random-action steps (reset on termination) with the reward range and
   mean ms/step. Report any construction error verbatim.
F. Anything inconsistent or surprising, one line each.
```

---

## Prompt 1 — close out the original campaign, reconcile `main` with the rewritten history, test and rehearse

```
The original campaign must be finished before anything here runs. Work through the steps in order and STOP
at the first failure, reporting it. Write results/ext_prep/STEP1_REPORT.md as you go and print its path.
Never use force, stash or a branch switch. The one permitted `git reset --hard` is the history reconciliation in
STEP 2, and only after its proof (2c) passes.

STEP 1 — PRECONDITIONS (stop if any fails; nothing is fetched or changed before all of them pass)
a. ORIG = the campaign in results/campaign/ whose manifest has both Walker-v0 and UpStepper-v0. Its manifest
   status must be "complete", phase_status["Walker-v0"].status and phase_status["UpStepper-v0"].status both
   "complete", milestone_status[<task>].ok true for both tasks, and CAMPAIGN_COMPLETE.txt must name it.
   Quote these fields.
b. No campaign worker, scheduler, monitor or preflight process is running (`pgrep -af 'evodevo'` is empty,
   ignoring this shell), and `tmux ls` shows no live evodevo-campaign session. If an evodevo-campaign session
   exists and its dashboard shows CAMPAIGN COMPLETE, close it with `tmux kill-session -t evodevo-campaign`;
   otherwise STOP.
c. `git branch --show-current` is `main` and `git status --short` is empty.
d. Verify the original campaign's mirror: run the repository's mirror verification for ORIG against its
   recorded mirror_dir (evodevo.campaign.mirror verify_mirror / verify_checksum_manifest) and report the
   result. If it is not verified, run one final sync with the same function and verify again.
e. `git fetch origin --prune` (this is the first fetch; it only reads). Record `git rev-parse origin/main`: this
   is the current origin/main SHA used everywhere below.

STEP 2 — HISTORY RECONCILIATION (replaces any `git pull` / `git merge --ff-only`)
The history of `main` on GitHub was rewritten after this checkout was made (same file contents and dates,
new commit hashes, one author identity). This checkout is on the OLD history, so `git merge --ff-only
origin/main` and `git pull` WILL FAIL; do not try them. Run this only now, with UpStepper-v0 complete (step 1a)
and the working tree clean (step 1c). It deletes nothing the campaign needs.
a. Record in the report: `git rev-parse HEAD`, `git status -sb`, `git stash list`, `git branch -vv`.
b. Create a safety ref of the old state: `git branch backup/pre-rewrite-lab HEAD`.
c. Prove this checkout holds no unique tracked work. All three must hold, otherwise STOP and report (the lab
   has content that is not in the rewritten history) and do not reset:
   - `git status --porcelain --untracked-files=no` prints nothing;
   - `T=$(git rev-parse HEAD^{tree}); git log origin/main --format='%H %T %aI %s' | grep "$T"` prints a
     rewritten twin whose author date (`git log -1 --format=%aI HEAD`) and subject
     (`git log -1 --format=%s HEAD`) equal the lab HEAD's;
   - if the twin is not exactly one line, STOP and report.
d. Only if (c) passes: `git reset --hard origin/main`. This is safe because the tracked content is identical
   (c); ignored files such as results/ and the .venv are untouched by reset; and `backup/pre-rewrite-lab`
   preserves the old history. Without a passing (c), do not reset.
e. Record the commit you end on (`git rev-parse HEAD`): it must equal the current origin/main SHA from 1e (no
   "or later"; if origin/main moved since 1e, fetch again and use the new SHA). Confirm these exist:
   evodevo/extension/{__main__,preflight,derive,rules,spec}.py,
   evodevo/experiments/configs/campaign_ext.yaml, campaign_ext_mini_validation.yaml, campaign_ext_validation.yaml,
   docs/campaign_prep/ext/{PREREGISTRATION_EXT,RUNBOOK_EXT,LAB_PROMPTS}.md, tests/extension/.
f. Note for the record: the original campaign's manifest.json records an OLD git commit (pre-rewrite). The
   old-to-new SHA map is held outside the repository (and as docs/provenance/commit_hash_map.tsv). The
   extension's `derive` only records the preflight's `git_commit(s)` and does not require any commit to be an
   ancestor of main (evodevo/extension/derive.py; evodevo/campaign/plan.py has no commit logic), so the old
   recorded SHA is expected and is not an error.

STEP 3 — PROOF THAT THE ORIGINAL CAMPAIGN IS UNCHANGED BY THE NEW CODE
With the REAL results/campaign_prep/final_estimate/final_estimate.json, compute
CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")).config_hash
in a fresh `.venv/bin/python -c` and compare it with results/campaign/<ORIG>/manifest.json config_hash.
They MUST be identical; if not, STOP and report both.

STEP 4 — TESTS (use `-p no:cacheprovider`; the lab has final_estimate.json, psutil, Linux CPUs, tmux and setsid)
a. Required, must be 100% green:
   `.venv/bin/python -m pytest -q -p no:cacheprovider tests/extension tests/campaign/test_extension_invariance.py
   tests/campaign/test_extension_generic.py`  (expected: 76 passed).
b. Full suite: `.venv/bin/python -m pytest -q -m "not slow" -p no:cacheprovider`. Expected on the lab: 0 failed.
   The 40 tests that fail on a Mac for environment reasons only (missing final_estimate.json, no psutil,
   no Linux CPU layout/sched_setaffinity, no setsid/tmux; identical failures at d51138e, the rewritten twin of the old 7774f76) MUST pass here:
   tests/campaign/{test_dry_run,test_entry_point,test_equivalence_validator,test_mirror_disk,test_notifier,
   test_operations,test_production_launcher_lifecycle,test_validation_profile}.py,
   tests/parallel/test_worker_pool.py, tests/preflight/{test_b1_b2,test_cli,test_report,test_telemetry}.py.
   Any failure is a STOP unless the same test ID fails identically at d51138e in a temporary `git worktree`
   (run it with that worktree as cwd and PYTHONPATH, print `evodevo.__file__`, then remove the worktree).
   Report the summary line and every failure with its first error line.

STEP 5 — REAL-LAUNCHER REHEARSAL OF THE EXTENSION GRID
This writes only a NEW campaign directory results/campaign/ext-mini-rehearsal and the mirror folder below.
It must never write into results/campaign/<ORIG> or results/extension_preflight/ (the production preflight).
a. `mkdir -p <BACKUP_MOUNT>/validation/ext-mini`, then a dry run:
   `xvfb-run -a sudo scripts/campaign/run_production_campaign.sh --dry-run --config
   evodevo/experiments/configs/campaign_ext_mini_validation.yaml --mirror-dir <BACKUP_MOUNT>/validation/ext-mini
   --local-config campaign.local.yaml`. Quote the PLAN and LAUNCH CHECKS sections (checks must PASS).
b. Launch: `xvfb-run -a sudo scripts/campaign/run_production_campaign.sh --campaign-id ext-mini-rehearsal --config
   evodevo/experiments/configs/campaign_ext_mini_validation.yaml --mirror-dir
   <BACKUP_MOUNT>/validation/ext-mini --local-config campaign.local.yaml --stamp 'VALIDATION - NOT A RESULT'`.
   Wait for it to finish (poll `scripts/campaign/status_campaign.sh --campaign ext-mini-rehearsal` every
   5 minutes; expect well under 2 h). If it stalls for 30 minutes, stop it with
   scripts/campaign/stop_campaign.sh and report.
c. Verify and report, with numbers:
   - phases ran in the order ObstacleTraverser-v1, Climber-v0, CaveCrawler-v0, each "complete" before the next
     started; arms exactly direct, nca, rule_based (no rule_based_monotone); seeds 0 and 1;
   - every run has 30 evaluations; journal rows == valid evaluations x K per phase (K=2 on Climber-v0 only);
     every journal row has torch_num_threads 1 and cpu_ids inside the production worker CPUs;
   - each phase's milestone package is complete: list its stats/ files; Climber-v0 must contain the
     mean_of_2_* files and the other two must not; every package has stats/floor_effect_check.csv; the
     rule_based_vs_monotone.csv files contain only a header;
   - manifest has estimate_path = evodevo/experiments/configs/ext_validation_estimate.json, config_path, and
     git_commit_history with one entry;
   - the dashboard frame shows ">360 h" (the extension guard) and an ETA;
   - the mirror verified; the phase/complete notifications were delivered (quote alerts.jsonl kinds);
   - every artifact carries the VALIDATION stamp.
d. Close the rehearsal's tmux session once it shows CAMPAIGN COMPLETE. Leave `git status --short` empty.
```

---

## Prompt 2 — start the extension preflight

```
Start the extension preflight. It runs unattended for about 9–11 hours (84 B5x-stage rows + 18 curves, 4 repeats).
Do not modify code or configuration and never pass --allow-dirty or any --test-* flag.

1. Preconditions (refuse and stop otherwise): on branch main, `git status --short` empty (a dirty tree is
   refused), HEAD == origin/main, no evodevo process running, no evodevo tmux session, no existing
   results/extension_preflight/ext-<date> from a different commit. ORIG = the completed original campaign id
   (as in Prompt 1).
2. Start it in a detached tmux session named ext-preflight, logging to results/ext_prep/preflight.log:
   sudo scripts/campaign/with_cpu_power_cap.sh --turbo --p-max-mhz 4200 --p-cpus 0-15 --pl1-w 125 --pl2-w 160 \
     --run-as-user ubuntu -- .venv/bin/python -m evodevo.extension preflight all \
     --reference-campaign results/campaign/<ORIG> --run-id ext-<UTC yyyymmdd>
   (run the sudo inside the tmux window so the power controls live as long as the job).
3. After 10 minutes report: results/extension_preflight/ext-<date>/preflight_manifest.json fields `label`
   (MUST be PRODUCTION; if TEST, stop and report), `production_params` (true), `git_commits` (one distinct
   commit, every `dirty` false), the bodies.json summary (6 bodies: label, kind, body_hash, walker_fitness or
   attempt), `.venv/bin/python -m evodevo.extension preflight status --run-id ext-<date>`, that 18 trainings are
   running on CPUs 0,2,4,6,8,10,12,14,16-25 (ps -eo pid,psr,comm or the pool's pinning), and the package
   temperature (it must stay below 95 °C; if it exceeds 95 °C for 5 minutes, stop the job with Ctrl-C in the
   tmux window and report).
4. Then stop. The run is resumable: if it is interrupted, re-run the identical command from the SAME commit
   (a different commit makes the run multi-commit, which `derive` refuses). Expected final counts:
   curves.jsonl 18 rows; noise.jsonl 84 rows = 72 B5x trainings (6 bodies x 3 tasks x 4 repeats) + 12 cost
   references. Budget rule: D6-ext knee clipped to [500,000, 1,500,000]; K = 2 only if r1 > 0.33 and
   r_max2 <= 0.5 r1; GUARD-ext threshold 360 h.
```

---

## Prompt 3 — derive the decisions

```
The extension preflight has finished. Apply the pre-registered rules and prepare the launch. Do not edit any
generated file by hand and do not change any rule constant.

1. `.venv/bin/python -m evodevo.extension preflight status --run-id ext-<date>`: every training recorded
   (curves.jsonl 18; noise.jsonl 84 = 72 noise trainings + 12 cost references); list any failed training with
   its reason. Failed trainings are excluded by the rules, never re-run by hand. Read preflight_manifest.json:
   `label` must be PRODUCTION, all `git_commits` entries one commit with dirty false. If not, STOP: derive
   will refuse it and you must not use --allow-test.
2. `.venv/bin/python -m evodevo.extension derive --run-id ext-<date> --reference-campaign results/campaign/<ORIG>`
   and then the identical `check` command (never pass --allow-test); check must print "extension check passed".
3. Write docs/campaign_prep/ORIGINAL_CAMPAIGN_RECORD.md: ORIG id, launch commit, start/end per phase, phase
   makespans, PPO-task counts and infra failures, mirror verification, any stops/resumes. These are facts
   only, taken from its manifest, phase_metrics.json, journal.jsonl and transcript.log.
4. Dry run of the real extension campaign:
   sudo scripts/campaign/run_production_campaign.sh --dry-run --config evodevo/experiments/configs/campaign_ext.yaml \
     --mirror-dir <BACKUP_MOUNT> --local-config campaign.local.yaml
   Its LAUNCH CHECKS may only list "dirty working tree" at this point.
5. Commit exactly: docs/campaign_prep/ext/{EXT_DECISIONS.json,EXT_DECISIONS.md,final_estimate_ext.json},
   evodevo/experiments/configs/campaign_ext.yaml and docs/campaign_prep/ORIGINAL_CAMPAIGN_RECORD.md, with message
   "Extension decisions derived from preflight ext-<date>". Push to origin main with a plain `git push origin main` (no force; if rejected, STOP and report).
6. Re-run the dry run: LAUNCH CHECKS must now PASS. Report: the full EXT_DECISIONS.md, the dry-run PLAN,
   ESTIMATE/CALENDAR and DISK sections, and the commit hash.
```

---

## Prompt 4 — audit report

```
Write an audit of the extension campaign's readiness to results/ext_prep/AUDIT_EXT.md and print it. Read-only:
change nothing, start nothing. Every claim must cite a file:line, a command output or a JSON field.

1. Provenance: HEAD hash, branch (main), `git status --short` (must be empty), origin/main == HEAD, and the
   commits since d51138e (the rewritten twin of the old 7774f76). Confirm the pre-registration commit (PREREGISTRATION_EXT.md) is older than every
   timestamp in results/extension_preflight/<run-id>/ (compare the commit date with the first row's
   started_at in curves.jsonl and the bodies.json created_at).
2. Rules applied as registered: for each task, quote EXT_DECISIONS.json d6_ext (branch, qualifying bodies,
   knee or fallback table, any clip/cap) and d16_ext (ratios, K). Recompute each with
   evodevo/extension/rules.py directly from curves.jsonl/noise.jsonl in a fresh python process and show
   they match. Confirm rules.py constants equal PREREGISTRATION_EXT.md sections 4–6.
3. Estimate: quote final_estimate_ext.json result, backtest (predicted vs actual UpStepper-v0 h, relative
   error), cost_ratios, observed vs assumed E/P ratio, guard steps and outcome, disk vs free space.
4. Plan: run `python -m evodevo.extension check` (must pass) and print, from CampaignPlan.from_benchmark on
   campaign_ext.yaml: config_hash, phases (task, ppo_timesteps, K, aggregation, timeout_s, runs, arms, seeds),
   population/offspring/eval_budget, factors and campaign_encoder_kwargs per arm. Confirm the encoder
   kwargs per arm equal the original campaign's for direct/nca/rule_based (compare with the original
   manifest's plan).
5. Original campaign unaffected: recompute benchmark.yaml's plan hash with the new code and compare with the
   original manifest config_hash (must match).
6. Launch readiness: the dry-run's LAUNCH CHECKS (must PASS), notifier configured, mirror mounted on a
   separate disk with free space >= the 3x disk estimate, no evodevo process/tmux session, package
   temperature now, rehearsal campaign ext-mini-rehearsal complete.
7. List every deviation, failed preflight training, warning or anomaly found, or "none".
8. End with one line: READY TO LAUNCH or NOT READY (reasons).
```
