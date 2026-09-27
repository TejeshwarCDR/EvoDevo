> Version-controlled copy, made 2026-09-28, of `results/campaign_prep/CAMPAIGN_PREP_P2.md` (a gitignored output directory). Relative paths below are as in the original.

# Campaign prep P2: decisions locked, configuration encoded, final estimates

Date: 2026-09-27. Input gate: `results/campaign_prep/CAMPAIGN_PREP_P1.md` shows **D6 CLEAR** (commit `1e79a56`). P1 Steps 3–5 are complete, including the t_PPO batch and the measured scaling.

The campaign was **not** run. `ea.py`, `ppo_runner.py` and every encoder are unmodified. The only PPO beyond P1's approved batch was the Step 5 policy-size probe: 2 trainings of 2,048 steps.

## Commits

| Commit | Contents |
|---|---|
| `f2689b1` | Pre-registered analysis plan (earlier pass; verified unchanged here) |
| `1e79a56` | D6 application plus deviation from pre-registration (earlier pass) |
| `a27c61a` | **decisions.md**: MONO, D14, D15, D16, concurrency, budget guard with its outcome, infra-failure policy |
| `c7172d1` | **Configuration**: campaign profile, schema validation, K>1 guard, and tests |
| `feed8ec` | **Lock**: re-locked config hash recorded in `docs/READY_TO_RUN.md` |
| `77bc842` | **Simulator**: `simulate_campaign.py` (variable-width generations) plus `final_estimate.py` |

## Step 1 — Decisions (`docs/decisions.md`, all ACCEPTED 2026-09-27, decided by Tejeshwar)

| Item | Decision | Main source |
|---|---|---|
| a. MONO | `rule_based_monotone` co-primary on both tasks. Fallback under the guard: Walker-v0 only. | P1 Steps 3b and 4e (monotone ÷ rule_based t_PPO 1.005 / 1.006) |
| b. D14 | No options adopted. Second grid size about 1,000 h; `direct_repaired` and `nca_compact` about 505 h each; a larger budget would be arbitrary under D6, and +7% already exceeds 432 h. | Final estimate: p95 406.0 h, 26 h headroom |
| c. D15 | Population 15, offspring 15, `eval_budget` 300 **genomes**, seeds 0–4, 4 arms, Walker-v0 then UpStepper-v0 | `campaign` profile |
| d. D6 | Walker-v0 1,000,000; UpStepper-v0 1,200,000. Already recorded in `1e79a56`, citing `learning_curves.json` and P1 Steps 2d/2e, so not duplicated. | P1 Step 2 |
| e. D16 | Walker-v0 K=1. UpStepper-v0 K=2 with fitness = MAX over completed seeds; budget counts genomes; every per-seed value logged; mean-of-K as sensitivity; seed-failure rule (`simulation_instability` if none completed; partial completion flagged). The same-day PROPOSED row is marked superseded. | P1 Step 5: UpStepper-v0 fraction single 0.666 / mean-of-2 0.384 / max-of-2 0.061 |
| f. Concurrency | 20 workers | B2 in corrected run `20260922T122730Z_4be8402b`: knee = optimum = 20 |
| g. Budget guard | If p95 > 432 h: (a) monotone Walker-only, then (b) drop monotone, then (c) UpStepper-v0 K=1 (per your addendum 2). **Outcome: none applied.** | Step 4 |
| h. Infra failures | Up to 2 retries on a fresh worker, then `infra_failure`, treated as sentinel, counted separately, alert above 0.5% | — |

## Step 2 — Pre-registered analysis plan

Per your addendum (1), this was verified rather than rewritten. The "Pre-registered analysis plan" section committed in `f2689b1` is **present and byte-identical** in `docs/decisions.md` at HEAD. It was committed before any campaign code existed; the campaign profile arrived later, in `c7172d1`.

## Step 3 — Configuration

**a. The `campaign` profile.** It lives in `evodevo/experiments/benchmark.py::PROFILES`, and `evodevo/experiments/configs/benchmark.yaml` now says `profile: campaign` with `tasks: [Walker-v0, UpStepper-v0]`. It holds:
- population 15, offspring 15, `eval_budget` 300
- `ppo_timesteps` {Walker-v0: 1,000,000, UpStepper-v0: 1,200,000}
- `ppo_seeds` {Walker-v0: 1, UpStepper-v0: 2}
- `ppo_seed_aggregation` "max"

Per-task mappings resolve per task in `resolve_configs`, identically for every arm on a task, so `check_fairness` still passes.

`ExperimentConfig` gains `ppo_seeds` (default 1) and `ppo_seed_aggregation` (default "max"; allowed {max, mean}). Both are validated at construction and added to `SHARED_BASE_KEYS` and `shared/base.yaml`.

Schema validation (`CampaignProfileError`) rejects:
- unknown aggregation names
- K < 1, or a non-integer K
- missing per-task values
- missing required fields

`resolved_config.py` was **not** touched, since it lives in the encoder package. The profile lives in the experiment layer instead.

**b. `OpenProfileDecisionError`.** It is not raised for `campaign`: every field is decided, and **no field remains open**. It is kept, unchanged, for the legacy `primary_matrix` profile. That profile still omits population, offspring and budget, and is superseded by `campaign`.

**c. The K > 1 guard.** Two layers, both raising `NotImplementedError`:
1. `run_benchmark` calls `check_ppo_seeds_supported`, which raises `PpoSeedsNotImplementedError` (a `NotImplementedError` subclass) before anything is written. It is skipped on `--dry-run`, which evaluates nothing.
2. `run_experiment` refuses any config with `ppo_seeds > 1`. This is the single entry point every run passes through.

Tests: `tests/experiments/test_campaign_profile.py` (new). Full suite: **955 passed, 0 failed**. `test_random_search_parity` is scoped to Walker-v0 (K=1).

**d. The lock hash.**
- **Old:** `c1685531d6e6543215b012b204d820c15e1a9284bbd6b0097cdc01aa1224f7ed` (fast_dev; recomputed from a clean checkout of the prior HEAD, matching the recorded value).
- **New:** `e5c2aa7547d4e08e6e0dad9e57f5a6e928355b3c671606d867f7709b404af638` (campaign, at `c7172d1`).

The procedure regression test's pin changed from `78e865e0…` to `dec52f1b…` only because `to_dict()` now serializes the two new keys. Dropping them reproduces the old pin exactly.

## Step 4 — Final estimate

Script: `scripts/capacity_planning/final_estimate.py`, 500 trials. Output: `results/campaign_prep/final_estimate/final_estimate.json`.

The t_PPO inputs were defined *before* the batch results were read:
- **Walker-v0:** B3's measured 1M values (budget = 1M, so no scaling).
- **UpStepper-v0:** B3 × **1.182**, the measured 1M → 1.2M scaling from a same-genome anchor, corrected for batch conditions. This is a measured factor, per your addendum (3); pure linear would be 1.200.
- **Monotone:** its own P1-measured means ÷ condition factor c = 1.0596.

Validity is resolved in the scheduler. UpStepper-v0's K=2 seeds are separate pool tasks. 20 workers with the P/E model (8 P-cores plus 12 E-cores at 2.3172×), D7 strict phases, lowest-generation-first.

| | Mean | p5 | p95 |
|---|---|---|---|
| Walker-v0 phase | 123.0 h | 121.9 h | 124.1 h |
| UpStepper-v0 phase | 280.2 h | 277.8 h | 282.9 h |
| **Total** | **403.2 h (16.8 d)** | **400.6 h** | **406.0 h** |

**Walker milestone: 30.5% of total runtime** (at about 123 h).

**Budget guard, applied mechanically:** p95 406.0 ≤ 432 h, so **no step applied**. The configuration after the guard is the full design: 4 arms × 2 tasks, UpStepper-v0 K=2.

Sensitivities, as p95:
- pure linear ×1.2: 408.8 h
- no condition correction: **425.2 h**
- resampled per-seed validity: 414.6 h

**Caveat:** the p5–p95 band covers only per-evaluation randomness around fixed means, not uncertainty in the means themselves (n = 3 per arm and task). The sensitivities are the better guide, and all stay under 432 h, but headroom is only about 6%.

## Step 5 — Disk estimate

Scripts: `scripts/campaign_prep/disk_estimate.py` and `policy_size_probe.py` (moved from `results/campaign_prep/disk/` and committed 2026-09-28). Output: `results/campaign_prep/disk/disk_estimate.json`.

Counts are upper bounds: every genome is treated as valid, giving 6,000 genomes per task and 18,000 PPO trainings.

| Item | Basis | GB |
|---|---|---|
| Per-evaluation JSONL | 12,000 records × 2 KB (measured about 520 B; ×4 margin for D10's extra fields) | 0.024 |
| Per-PPO-seed journal | 18,000 records × 2 KB | 0.036 |
| SB3 policies, one per PPO training | Probe (2,048-step training, largest 5×5 body): **269 KB** Walker-v0, **288 KB** UpStepper-v0 (19.6k / 21.2k parameters); ×2 margin | 10.151 |
| Checkpoints | Every generation kept; 30 genomes × the largest genome (nca, 10.4 KB) + 5 KB state; ×2 | 0.506 |
| Telemetry and transcript | Measured 2.26 MB/h in pre-flight × 406 h × 2 | 1.837 |
| Milestone packages | **Assumed** 50 MB per package, hourly × 406 h (no measured precedent; generous) | 20.298 |
| **Total** | | **32.9 GB** |

- 3× the total is **98.6 GB**.
- Free space on `/` is **302.1 GB** (282 GiB in `df -h` terms).
- **The D10 3× rule is satisfied.**

The estimate is conservative in two ways:
- SB3 policy size depends on body dimensions, not training timesteps, so the probe's size carries over to the knee budgets unchanged. The largest body bounds all smaller ones.
- D10 itself says to save policies only for final best bodies, so the 10.2 GB policy line is an upper bound.

## Acceptance

1. Every decision written as ACCEPTED, with rationale and sources: **yes** (`a27c61a`; D6 in `1e79a56`).
2. The pre-registered analysis plan was committed before any campaign code: **yes** (`f2689b1`, verified unchanged).
3. The campaign profile validates, the K>1 guard exists, and the hash is re-locked: **yes** (`c7172d1`, `feed8ec`; 955 tests pass).
4. The final estimate is reported with the guard outcome and the Walker milestone percentage: **yes** (403.2 h mean, 406.0 h p95, guard not triggered, milestone 30.5%).
5. Disk: **the 3× rule is satisfied** (98.6 GB needed vs. 302.1 GB free).

## Notes

- `CAPACITY_PLANNING_REPORT.md` has its new "Final campaign estimate" section at the top, and its 3-arm scope caveat is marked resolved. The file lives under the gitignored `results/preflight/`, so this change is on disk but **not committed**. The same applies to both P1/P2 reports under `results/campaign_prep/`, now in `.gitignore` as P1 Step 4 permitted. The `.gitignore` edit itself is also uncommitted.
- The other `scripts/capacity_planning/` scripts from earlier rounds remain untracked, as they were before this prompt.
- The pre-existing uncommitted changes in `evodevo/preflight/*` and `tests/preflight/*` were not touched.
