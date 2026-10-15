# Extension campaign runbook (lab host)

Authoritative sequence for the ObstacleTraverser-v1 → Climber-v0 → CaveCrawler-v0 campaign.
Rules and analysis: [PREREGISTRATION_EXT.md](PREREGISTRATION_EXT.md). Branch: `campaign-ext`.

**Never pull, check out or edit code on the lab host while the original campaign is running.** Its milestone
subprocess and any rebuilt worker import whatever code is on disk at that moment.

| Step | When | Duration | What |
|---|---|---|---|
| 0 | now (original campaign running) | 10 min | Read-only record of the original campaign; nothing changes |
| 1 | original campaign `complete` | ~1.5 h | close out the original, check out `campaign-ext`, tests, real-launcher mini rehearsal |
| 2 | after step 1 passes | ~8–10 h, unattended | extension preflight (B4x, then B5x) under the production power controls |
| 3 | after step 2 | 15 min | derive decisions, check, dry run, commit, push |
| 4 | after step 3 | 15 min | audit report, then launch only on an explicit GO |

## Step 1 — close-out and rehearsal

```bash
cd ~/projects/EvoDevo
ORIG=$(ls -t results/campaign/*/manifest.json | head -1 | xargs dirname | xargs basename)
python3 -c "import json;m=json.load(open('results/campaign/$ORIG/manifest.json'));print(m['status'],m['phase_status'])"
tmux ls                                   # the original 'evodevo-campaign' session must be closed first
git status --short                        # must be empty (results/ is ignored)
git fetch origin && git checkout campaign-ext && git pull --ff-only
.venv/bin/python -m pytest -q -m "not slow" -p no:cacheprovider    # full suite
.venv/bin/python -m pytest -q -p no:cacheprovider tests/extension tests/campaign/test_extension_invariance.py \
    tests/campaign/test_extension_generic.py
```

Real-launcher rehearsal of the extension grid (real PPO at 2,048 timesteps, 30 genomes, 2 seeds, all three
tasks; K=2 on Climber-v0 exercises the multi-seed path). Same wrapper, tmux, notifier and mirror as production:

```bash
sudo scripts/campaign/run_production_campaign.sh --campaign-id ext-mini-rehearsal \
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

The run is resumable. Re-running the same command skips every training already recorded in
`results/extension_preflight/<run-id>/{curves,noise}.jsonl`.

## Step 3 — derivation

```bash
.venv/bin/python -m evodevo.extension derive --run-id ext-YYYYMMDD --reference-campaign results/campaign/$ORIG
.venv/bin/python -m evodevo.extension check  --run-id ext-YYYYMMDD --reference-campaign results/campaign/$ORIG
git add docs/campaign_prep/ext evodevo/experiments/configs/campaign_ext.yaml
git commit -m "Extension decisions derived from preflight ext-YYYYMMDD"
git push origin campaign-ext
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
