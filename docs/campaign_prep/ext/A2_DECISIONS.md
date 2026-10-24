# Amendment 2 decisions (PRODUCTION)

Derived 2026-10-24T20:00:25+00:00 at commit `5895934d8168b2d7f75902082846bd12d4dd1cea` from preflight run `results/extension_preflight/ext-20261023`, T_obstacle = 1,500,000 (Gate T), calibrated on campaign `20260930T044002Z`. Gate line: `GATE T: 1500000 | GATE R: PASS | GATE S: saving possible yes`. Rules: [PREREGISTRATION_EXT.md](PREREGISTRATION_EXT.md), Amendment 2.

| Task | ppo_timesteps | K | source of ppo_timesteps |
|---|---:|---:|---|
| ObstacleTraverser-v1 | 1,500,000 | 1 | Gate T (A2_GATES_REPORT.md): the registered D6-ext fallback test failed at 1.0M and 1.2M, so the registered cap 1.5M applies |
| UpStepper-v0 | 1,200,000 | 2 | original campaign per-run configs, unchanged (seeds 5-9 pool with seeds 0-4) |
| Walker-v0 | 1,000,000 | 1 | original campaign per-run configs, unchanged (seeds 5-9 pool with seeds 0-4) |

## Execution order (one campaign, strict order)

| # | unit | kind | seeds / sources | PPO steps | trainings | h p5 | h mean | h p95 | mean finish (IST) |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `ObstacleTraverser-v1:s0-4` | evolution (runs) | seeds 0-4 x 3 arms | 1,500,000 | 15 | 293.1 | 325.7 | 358.3 | Mon 2026-11-09 01:40 IST |
| 2 | `reeval:ObstacleTraverser-v1:s0-4` | reeval | ObstacleTraverser-v1:s0-4 | 1,500,000 | 150 | 15.2 | 16.9 | 18.6 | Mon 2026-11-09 18:35 IST |
| 3 | `ObstacleTraverser-v1:s5-9` | evolution (runs) | seeds 5-9 x 3 arms | 1,500,000 | 15 | 293.1 | 325.7 | 358.3 | Mon 2026-11-23 08:16 IST |
| 4 | `reeval:ObstacleTraverser-v1:s5-9` | reeval | ObstacleTraverser-v1:s5-9 | 1,500,000 | 150 | 15.2 | 16.9 | 18.6 | Tue 2026-11-24 01:10 IST |
| 5 | `UpStepper-v0:s5-9` | evolution (runs) | seeds 5-9 x 3 arms | 1,200,000 | 15 | 261.2 | 290.2 | 319.2 | Sun 2026-12-06 03:22 IST |
| 6 | `reeval:UpStepper-v0:s0-9` | reeval | original seeds 0-4 + UpStepper-v0:s5-9 | 1,200,000 | 300 | 13.2 | 14.6 | 16.1 | Sun 2026-12-06 18:00 IST |
| 7 | `Walker-v0:s5-9` | evolution (runs) | seeds 5-9 x 3 arms | 1,000,000 | 15 | 107.7 | 119.7 | 131.6 | Fri 2026-12-11 17:40 IST |
| 8 | `reeval:Walker-v0:s0-9` | reeval | original seeds 0-4 + Walker-v0:s5-9 | 1,000,000 | 300 | 10.9 | 12.1 | 13.3 | Sat 2026-12-12 05:44 IST |
| 9 | `robustness:ObstacleTraverser-v1` | robustness | ObstacleTraverser-v1:s0-4 + ObstacleTraverser-v1:s5-9 | 3,000,000 | 90 | 18.9 | 21.0 | 23.2 | Sun 2026-12-13 02:47 IST |
| 10 | `robustness:UpStepper-v0` | robustness | original seeds 0-4 + UpStepper-v0:s5-9 | 2,400,000 | 90 | 8.4 | 9.4 | 10.3 | Sun 2026-12-13 12:10 IST |
| 11 | `robustness:Walker-v0` | robustness | original seeds 0-4 + Walker-v0:s5-9 | 2,000,000 | 90 | 7.0 | 7.7 | 8.5 | Sun 2026-12-13 19:54 IST |

**Projected total (informational; no cost guard applies, Amendment 2 (k)):** 1159.9 h mean (1043.9-1275.9 h, p5-p95) = 48.3 days from the planned launch Mon 2026-10-26 12:00 IST: p5 Tue 2026-12-08 23:55 IST, mean Sun 2026-12-13 19:54 IST, p95 Fri 2026-12-18 15:53 IST.

## Re-training stages

Re-evaluation: 10 trainings per champion, PPO seeds `ppo_seed(20261101, k)`, k = 0..9, at each task's own budget. Robustness: 3 trainings per champion, `ppo_seed(20261201, k)`, k = 0..2, at 2x the budget: ObstacleTraverser-v1 3,000,000, UpStepper-v0 2,400,000, Walker-v0 2,000,000 (in that order; 270 trainings).

## Gates

- Gate T: T_obstacle = 1,500,000. registered D6-ext fallback test: every qualifying body >= 85% of its 2M fitness and rank order kept. Fails at 1.0M and 1.2M; 1.5M is the registered cap and fails as well; only 2M passes (trivially).
- Gate R: PASS; config differs only in the seed; library versions identical (evogym f63c649c); 12/12 (including both K=2 trainings) replayed original evaluations bit-identical.
- Gate S: saving possible yes: the worker already saves each policy with its optimizer state after the fitness rollout; no observation normaliser exists; learning rate constant 3e-4

## Inputs

- Gates report `docs/campaign_prep/ext/A2_GATES_REPORT.md` sha256 `5ac40f6f02061f57a135bcb22dd741d92692aff70e97e12c1e98551646ae3dba`
- Preflight record `docs/campaign_prep/ext/EXT_DECISIONS.json` sha256 `930213b12778c4e161e604078d64420107ebc8d07a3180d36273c7b933f44bc9` (superseded; evidence only)
- Original campaign `results/campaign/20260930T044002Z` commit `7774f76c1473815ece05c488ab34c91cf84b8020` config hash `e50111cbefae4089462d81b33d79f38c1b97e302d3c8f63eed4f3fd052a55cd0`
