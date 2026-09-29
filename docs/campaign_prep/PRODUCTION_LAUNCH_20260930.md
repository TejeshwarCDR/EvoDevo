# Production launch contract — 2026-09-30

The scientific campaign has **not** been launched. This document records the
only accepted production configuration and the required guarded entry point.

## Selected configuration

| Item | Required value |
|---|---|
| Workers | 18 |
| P-worker CPUs | 0, 2, 4, 6, 8, 10, 12, 14 |
| E-worker CPUs | 16–25 |
| Scheduler/control CPUs | 26–27 only |
| Turbo | enabled |
| P-policy cap | 4.2 GHz for policies 0–15 |
| RAPL | PL1 125 W; PL2 160 W |
| Measured rate | 42.96 genomes/min |
| Full-campaign projection | about 434 h / 18.1 days |
| Walker milestone/phase end | about 132 h / 5.5 days |

The selected real-PPO validation ran 960 genomes at 20,000 PPO steps each for
22.3 minutes. It observed 84 C maximum and 77 C p95 package temperature,
97.1% worker utilization, zero failures, verified affinity and no thermal
throttle-counter increase. The comparison matrix is
`results/thermal_worker_study/FINAL_SELECTION_20260930.md`.

## Fail-closed operation

Use `scripts/campaign/run_production_campaign.sh`, never a bare real-mode
`run_campaign.sh`, for production. It requires root only to set and verify the
reversible host controls, then drops the campaign process to the sudo-invoking
user. It holds those controls until the scheduler stops and restores every
changed sysfs value afterwards. The regular launcher verifies the exact
topology plus turbo, every P-policy cap, PL1 and PL2 before a real launch;
there is no silent fallback to the historic 20-worker or 10-E-core layouts.

Run it from a durable user tmux session. Its `--dry-run` variant performs no
campaign work and is the required final preflight:

```bash
sudo scripts/campaign/run_production_campaign.sh --dry-run \
  --mirror-dir /mnt/evodevo-backup --local-config campaign.local.yaml
```

After a passing final dry-run and a clean worktree, the same command without
`--dry-run` starts the real campaign. Do not use that command until explicitly
ready to begin the scientific run.

For a controlled stop, use `scripts/campaign/stop_campaign.sh`. A stopped
production campaign must be resumed through the root-held production wrapper:

```bash
sudo scripts/campaign/run_production_campaign.sh --resume CAMPAIGN_ID \
  --mirror-dir /mnt/evodevo-backup --local-config campaign.local.yaml
```
