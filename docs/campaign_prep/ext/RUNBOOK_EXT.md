# Extension campaign runbook (lab host)

Authoritative sequence for the ObstacleTraverser-v1 → Climber-v0 → CaveCrawler-v0 campaign.
Rules and analysis: [PREREGISTRATION_EXT.md](PREREGISTRATION_EXT.md). Branch: `main` (the extension is already merged).

**Never pull, fetch-and-merge, check out or edit code on the lab host while the original campaign is running.**
Its milestone subprocess and any rebuilt worker import whatever code is on disk at that moment. The history
reconciliation (Step 1, below) happens only after the original campaign is `complete`.

| Step | When | Duration | What |
|---|---|---|---|
| 0 | now (original campaign running) | 10 min | Read-only record of the original campaign; nothing changes |
| 1 | original campaign `complete` | ~1.5 h | verify the original is complete, reconcile `main` with the rewritten history, tests, real-launcher mini rehearsal |
| 2 | after step 1 passes | ~9–11 h, unattended | extension preflight (B4x 18 curves, then B5x 72 noise trainings + 12 cost references = 84 rows; 4 repeats) under the production power controls |
| 3 | after step 2 | 15 min | derive decisions, check, dry run, commit to `main`, push (no force) |
| 4 | after step 3 | 15 min | audit report, then launch only on an explicit GO |

## Step 1 — close-out and rehearsal

```bash
cd ~/projects/EvoDevo
ORIG=$(ls -t results/campaign/*/manifest.json | head -1 | xargs dirname | xargs basename)
python3 -c "import json;m=json.load(open('results/campaign/$ORIG/manifest.json'));print(m['status'],m['phase_status'])"
tmux ls                                   # the original 'evodevo-campaign' session must be closed first
git branch --show-current                 # must be main
git status --short                        # must be empty (results/ is ignored)
pgrep -af evodevo                         # must show no campaign worker / scheduler
git fetch origin --prune
git rev-parse origin/main                 # the current origin/main SHA: the commit to end on
# HISTORY RECONCILIATION. The history of main was rewritten (same content and dates, new hashes), so
# `git pull` / `git merge --ff-only origin/main` WILL FAIL on this checkout. Do this instead:
git rev-parse HEAD; git status -sb; git stash list; git branch -vv    # record all four
git branch backup/pre-rewrite-lab HEAD                                # safety ref of the old history
git status --porcelain --untracked-files=no                           # must print nothing
T=$(git rev-parse HEAD^{tree}); git log origin/main --format='%H %T %aI %s' | grep "$T"
git log -1 --format='%aI %s' HEAD         # the one twin line above must have this author date and subject
# Only if the tracked tree is clean AND exactly one twin matches (same tree, date, subject):
git reset --hard origin/main              # safe: tracked content identical; results/ and .venv are ignored
                                          # and untouched; the backup ref keeps the old history
git rev-parse HEAD                        # must equal the origin/main SHA recorded above
.venv/bin/python -m pytest -q -p no:cacheprovider tests/extension tests/campaign/test_extension_invariance.py \
    tests/campaign/test_extension_generic.py                          # required: 76 passed
.venv/bin/python -m pytest -q -m "not slow" -p no:cacheprovider      # full suite: 0 failed on the lab
```

The full suite needs the lab's `results/campaign_prep/final_estimate/final_estimate.json`, `psutil`, the Linux
CPU layout, `tmux` and `setsid`; on a laptop 40 tests fail for exactly those reasons (identical at d51138e, the rewritten twin of the old 7774f76).

Real-launcher rehearsal of the extension grid (real PPO at 2,048 timesteps, 30 genomes, 2 seeds, all three
tasks; K=2 on Climber-v0 exercises the multi-seed path). Same wrapper, tmux, notifier and mirror as production.
It writes only `results/campaign/ext-mini-rehearsal` and the mirror folder, never the original campaign's
results or `results/extension_preflight/`:

```bash
mkdir -p <BACKUP_MOUNT>/validation/ext-mini
xvfb-run -a sudo scripts/campaign/run_production_campaign.sh --campaign-id ext-mini-rehearsal \
  --config evodevo/experiments/configs/campaign_ext_mini_validation.yaml \
  --mirror-dir <BACKUP_MOUNT>/validation/ext-mini --local-config campaign.local.yaml \
  --stamp 'VALIDATION - NOT A RESULT'
```

## Step 2 — extension preflight

```bash
tmux new -s ext-preflight
sudo scripts/campaign/with_cpu_power_cap.sh --turbo --p-max-mhz 4200 --p-cpus 0-15 --pl1-w 125 --pl2-w 160 \
  --run-as-user ubuntu -- .venv/bin/python -m evodevo.extension preflight all \
  --reference-campaign results/campaign/$ORIG --run-id ext-$(date -u +%Y%m%d)
# progress, from another shell:
.venv/bin/python -m evodevo.extension preflight status --run-id ext-YYYYMMDD
```

The run is resumable. Re-running the same command from the same commit skips every training already recorded
in `results/extension_preflight/<run-id>/{curves,noise}.jsonl`. It refuses a dirty tree; never pass
`--allow-dirty` or `--test-*` (those mark the run TEST and `derive` refuses a TEST or multi-commit run).
Expected: curves.jsonl 18 rows, noise.jsonl 84 rows (72 B5x trainings + 12 cost references).

## Step 3 — derivation

```bash
.venv/bin/python -m evodevo.extension derive --run-id ext-YYYYMMDD --reference-campaign results/campaign/$ORIG
.venv/bin/python -m evodevo.extension check  --run-id ext-YYYYMMDD --reference-campaign results/campaign/$ORIG
git add docs/campaign_prep/ext evodevo/experiments/configs/campaign_ext.yaml
git commit -m "Extension decisions derived from preflight ext-YYYYMMDD"
git push origin main        # plain push, never --force
sudo scripts/campaign/run_production_campaign.sh --dry-run \
  --config evodevo/experiments/configs/campaign_ext.yaml --mirror-dir <BACKUP_MOUNT> --local-config campaign.local.yaml
```

## Step 4 — launch (only after the audit's GO)

```bash
sudo scripts/campaign/run_production_campaign.sh --campaign-id ext-$(date -u +%Y%m%dT%H%M%SZ) \
  --config evodevo/experiments/configs/campaign_ext.yaml \
  --mirror-dir <BACKUP_MOUNT> --local-config campaign.local.yaml
```

Resume after a stop or crash (the scheduler also prints its exact resume line):

```bash
sudo scripts/campaign/run_production_campaign.sh --resume <EXT_ID> \
  --config evodevo/experiments/configs/campaign_ext.yaml \
  --mirror-dir <BACKUP_MOUNT> --local-config campaign.local.yaml
```

Status at any time: `scripts/campaign/status_campaign.sh --campaign <EXT_ID>`.

## Amendment 2 — the campaign that replaces the 3-task design

Amendment 2 of [PREREGISTRATION_EXT.md](PREREGISTRATION_EXT.md) retires Steps 2–4 above for the 3-task design
(ObstacleTraverser-v1, Climber-v0, CaveCrawler-v0): `EXT_DECISIONS.json` is the preflight record only
(`SUPERSEDED_BY_AMENDMENT_2.json` holds its sha256) and `campaign_ext.yaml` can no longer be planned or launched.
Step 1 (close-out, history reconciliation, tests) and the extension preflight `ext-20261023` stand.

**One campaign, strict order** (`campaign_a2.yaml`, 60 evolutionary runs, 15 per phase, then the re-training stages):

| # | unit | what | PPO steps | trainings |
|---:|---|---|---:|---:|
| 1 | `ObstacleTraverser-v1:s0-4` | P1 (INTERIM package) | 1,500,000 | 15 runs |
| 2 | `reeval:ObstacleTraverser-v1:s0-4` | 15 champions x 10 | 1,500,000 | 150 |
| 3 | `ObstacleTraverser-v1:s5-9` | P2 (final package, seeds 0–9) | 1,500,000 | 15 runs |
| 4 | `reeval:ObstacleTraverser-v1:s5-9` | 15 champions x 10 | 1,500,000 | 150 |
| 5 | `UpStepper-v0:s5-9` | P3 (K = 2, original per-run configs) | 1,200,000 | 15 runs |
| 6 | `reeval:UpStepper-v0:s0-9` | 30 champions (original 0–4 read-only, new 5–9) x 10 | 1,200,000 | 300 |
| 7 | `Walker-v0:s5-9` | P4 (original per-run configs) | 1,000,000 | 15 runs |
| 8 | `reeval:Walker-v0:s0-9` | 30 champions x 10 | 1,000,000 | 300 |
| 9–11 | `robustness:ObstacleTraverser-v1`, `:UpStepper-v0`, `:Walker-v0` | 30 champions x 3 at 2x the budget | 3,000,000 / 2,400,000 / 2,000,000 | 90 each |

Re-evaluation PPO seeds are `ppo_seed(20261101, k)`, k = 0..9; robustness `ppo_seed(20261201, k)`, k = 0..2. Every
re-training is an ordinary pool task on the same pinned workers and writes its policy and log below
`retrain/<stage>/<task>/<arm>/seed_<s>/e<champion eval index>/`, a path unique per (task, arm, seed, champion
evaluation index, stage, k). One row per training goes to `reeval.jsonl` / `robustness.jsonl` in the campaign directory
(campaign id, run key, source campaign and commit, seed, champion eval index, body hash, budget, PPO seed, fitness, wall
time, core kind). A stopped or crashed stage resumes without repeating a finished training.

**Decision record.** `python -m evodevo.extension amend2 derive --t-obstacle 1500000` writes `A2_DECISIONS.json`,
`A2_DECISIONS.md`, `final_estimate_a2.json` and `campaign_a2.yaml`; `python -m evodevo.extension amend2 check
[--reference-campaign results/campaign/20260930T044002Z]` verifies them byte for byte and refuses hand edits. `derive`
refuses a dirty tree, a gates report whose first line is not `GATE T: <T> | GATE R: PASS | GATE S: saving possible yes`
for the same T, and a preflight record that no longer matches `SUPERSEDED_BY_AMENDMENT_2.json`.

**Launch checks added for Amendment 2** (in `--check-only`, `--dry-run` and the real launch): every Walker-v0/UpStepper-v0
run config equals the original campaign's except for the seed (and today's code builds the same configs), and every
recorded library version (python, torch, stable-baselines3, gymnasium, numpy, cloudpickle, evogym and its commit) equals
the installed one. A TEST-labelled config may skip the config check with `--skip-config-equality-check`; a PRODUCTION
config refuses the flag.

**Rehearsal and launch.** `campaign_a2_rehearsal.yaml` (generated by `amend2 rehearsal`, TEST label, real PPO at 2,048 steps,
30 genomes per run, Obstacle blocks {0} and {1}, UpStepper-v0 {5}, Walker-v0 {5}, 2 trainings per champion, robustness at
2x) uses the same production wrapper:

```bash
mkdir -p /mnt/evodevo-backup/validation/a2-rehearsal
sudo scripts/campaign/run_production_campaign.sh --campaign-id a2-rehearsal \
  --config evodevo/experiments/configs/campaign_a2_rehearsal.yaml --skip-config-equality-check \
  --mirror-dir /mnt/evodevo-backup/validation/a2-rehearsal --local-config campaign.local.yaml \
  --stamp 'VALIDATION - NOT A RESULT'
mkdir -p /mnt/evodevo-backup/a2-production
sudo scripts/campaign/run_production_campaign.sh --campaign-id a2-$(date -u +%Y%m%dT%H%M%SZ) \
  --config evodevo/experiments/configs/campaign_a2.yaml \
  --mirror-dir /mnt/evodevo-backup/a2-production --local-config campaign.local.yaml
```

Resume: the same command with `--resume <A2_ID>` instead of `--campaign-id` (and the same `--config`/`--mirror-dir`); the
plan hash must match the manifest's or the resume is refused. Each phase and stage end sends a notification and writes a
package under `results/campaign/<A2_ID>/milestones/<unit>/` (colons in a unit id become `__`): `ObstacleTraverser-v1__s0-4`
(INTERIM), `ObstacleTraverser-v1__s5-9` (seeds 0–9), `UpStepper-v0__s5-9` and `Walker-v0__s5-9` (original seeds 0–4 merged,
provenance column, monotone contrast from the original seeds 0–4 only), `reeval__<...>` (secondary endpoint),
`robustness__<task>` and the final `robustness__all` (task budget versus 2x).

**Replay check.** `python -m evodevo.campaign replay --output-dir <DIR>` re-runs recorded original evaluations through the
real worker path, each training pinned to its own core, and exits 0 only if every fitness is bit-identical to the recorded
one (default: Walker-v0 direct seed_1 eval 2 and UpStepper-v0 rule_based seed_0 eval 299, about 1 h).
