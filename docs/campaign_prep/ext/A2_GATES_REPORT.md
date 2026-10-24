GATE T: 1500000 | GATE R: PASS | GATE S: saving possible yes

# Amendment 2 decision gates — report

Written for: the researcher deciding Amendment 2 (they will act on the three gate outcomes and the cost calendar). Read-only run: nothing committed, pushed, launched or edited; `results/campaign/20260930T044002Z` untouched. Generated 2026-10-24. HEAD `5e5a7a1e159387797997e8cea9ee8fea27985328`; original campaign commit 7774f76 / twin d51138e2694b85e0cde6296af499ac1d95aa9fe5.

## 1. Summary

| Gate | Result |
|---|---|
| T | **T_obstacle = 1,500,000.** The 1.0M row and the 1.2M row each fail both conditions, so the rule falls through to 1.5M. The 1.5M row itself also fails both conditions (see §2); the rule gives 1.5M by construction, not because it passes. |
| R | **PASS** — R2 identical, R3 identical, all 12 replays bit-identical with current code. R6 not needed, not run, `~/evodevo_original_tree` not created. |
| S | Saving policy + optimizer state is already done by the worker after training and is side-effect free; no observation normaliser exists. |
| Cost | At T=1.5M the whole of Amendment 2 projects to ~1,150 h mean (≈48 days; launch 2026-10-26 12:00 IST → ~2026-12-13). Far above the 360 h GUARD-ext threshold. |

## 2. Gate T — ObstacleTraverser-v1 PPO budget

Method: `curves.jsonl` rows for ObstacleTraverser-v1 (6 bodies, all completed, 20 checkpoints). `rules.qualifying_bodies` gave all 6 as qualifying. The fallback table is computed inline inside `rules.decide_budget`, so I called `decide_budget` itself with `rules.select_knee` replaced in memory by a stub reporting "no knee found" (no file changed), which forces its own fallback branch. (For the record, the real decision was `registered_knee` at 600k; this gate deliberately ignores the knee.) The fallback branch returned `fallback_choice` 2,000,000 (capped to 1,500,000 by `BUDGET_CAP`), consistent with the rule below.

Fraction of each body's 2M fitness reached at T:

| T | direct:evolved | direct:random | nca:evolved | nca:random | rule_based:evolved | rule_based:random | fraction_ok (all ≥0.85) | order_ok |
|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 1,000,000 | 0.1913 | 0.7961 | 1.0481 | 0.9486 | 0.3839 | 0.9554 | **False** | **False** |
| 1,200,000 | 0.5781 | 0.7329 | 1.2615 | 1.0001 | 0.5423 | 0.9549 | **False** | **False** |
| 1,500,000 | 0.5781 | 0.2234 | 1.0226 | 0.9479 | 0.3847 | 0.9534 | False | False |
| 2,000,000 | 1 | 1 | 1 | 1 | 1 | 1 | True | True |

Order at 2M: rule_based:evolved > rule_based:random > direct:random > nca:random > nca:evolved > direct:evolved. Orders at 1.0M / 1.2M / 1.5M all differ from that.

**Rule: 1.0M passes? no. 1.2M passes? no. → 1,500,000.** T_obstacle = **1,500,000**.

Note: at 1.5M the table shows fraction_ok False (direct:evolved 0.58, direct:random 0.22, rule_based:evolved 0.38) and order_ok False; i.e. the 1.5M budget does NOT satisfy the Walker-style 85%/order criteria either. Only 2M does. The 1.5M value is the rule's fallback (and equals `BUDGET_CAP`).

Per body (fitness; single preflight training per body, seed 20261015):

| body | 600k | 1.0M | 1.2M | 2M | first checkpoint within 5% of own 2M value |
|---|---:|---:|---:|---:|---:|
| direct:evolved | 0.0438 | 0.2081 | 0.6288 | 1.0877 | 2,000,000 |
| direct:random | 0.8046 | 0.9158 | 0.8431 | 1.1503 | 1,300,000 |
| nca:evolved | 1.1587 | 1.1624 | 1.3991 | 1.1091 | 500,000 |
| nca:random | 1.1237 | 1.0676 | 1.1255 | 1.1254 | 600,000 |
| rule_based:evolved | 1.1422 | 1.1897 | 1.6805 | 3.0990 | 2,000,000 |
| rule_based:random | 1.1948 | 1.1999 | 1.1993 | 1.2559 | 500,000 |

Rank correlation between bodies' fitness at T_obstacle=1.5M and at 2M (n=6; implemented by hand with numpy, no scipy in the venv — I did not install anything): **Spearman 0.543, Kendall τ-b 0.333** (identical on the 6 qualifying bodies, which is all of them). With n=6 and single trainings these are very imprecise.

**rule_based:evolved after 1.0M: yes, still rising, but erratically.** 1.19 (1.0M) → 1.15 (1.1M) → 1.68 (1.2M) → 1.17 (1.3M) → 1.74 (1.4M) → 1.19 (1.5M) → … 1.74 (1.8M) → 1.17 (1.9M) → 3.10 (2.0M). The curve is a ~1.15–1.2 plateau with occasional jumps to ~1.7 and a final jump to 3.10; at 1.5M it sits on the low plateau. One training per body cannot separate "late learning" from seed noise here.

## 3. Gate R — can current code reproduce the original Walker-v0 / UpStepper-v0 protocol?

**Outcome: PASS.**

### R1 — code trace and hunk table (d51138e → HEAD)

Worker path traced: `evodevo/campaign/scheduler.py` (propose → `PPOTask`) → `evodevo/parallel/worker_pool.py` (`_worker_initializer`, `_run_ppo_task`) → `evodevo/parallel/threading_control.py`, `affinity.py` → `evodevo/evaluation/ppo_runner.py` (`evaluate_ppo_task`, `_rollout_fitness`, `ppo_seed`, `aggregate_seed_results`) → `evodevo/core/{config,factors,seeding}.py`, `evodevo/encoders/*` (decode), `evodevo/morphology/*`, `evodevo/evolution/{ea,eval_bookkeeping,logging}.py`, EvoGym/SB3. `git diff d51138e HEAD --stat` for `evodevo/core evolution parallel encoders morphology preflight analysis visualization external scripts requirements.txt experiments/run_experiment.py` is **empty** (no hunks). Hunks in the evaluation path:

| file | lines (new) | class | reason |
|---|---|---|---|
| evaluation/ppo_runner.py | 66-69 `PPOTask.curve_eval_interval` field (default None) | NO-OP | new defaulted field; campaign tasks never set it |
| evaluation/ppo_runner.py | 95-96 `PPOTaskResult.curve` (default None) | NO-OP | new defaulted field |
| evaluation/ppo_runner.py | 162-163 `if getattr(task,"curve_eval_interval",None): return _evaluate_curve_task(task)` | NO-OP | branch only when curve_eval_interval is set (preflight); campaign leaves None, falls through to the unchanged original body |
| evaluation/ppo_runner.py | 202-216 `_evaluate_curve_task` | NO-OP | new function, only reachable from the branch above |
| campaign/scheduler.py | 65-83 `encoder_and_config_for` (new) and 194 `_make_run` now calls it | NO-OP | pure extraction of the previous inline code (same statements, same order); R2 confirms identical configs/hash |
| campaign/scheduler.py | 741-755 `cli_main` manifest fields (`git_commit_history`, `estimate_path`, `config_path`) | NO-OP | bookkeeping in the manifest only |
| campaign/plan.py | 15-80 decision-record/estimate helpers; 108-170 `from_benchmark` | NO-OP for this protocol | `decisions_path`/`estimate_path` None → legacy path; priors identical; plan hash for benchmark.yaml equals the manifest's (R2). Timeout formula unchanged numerically (timeouts equal manifest, R2) |
| experiments/benchmark.py | 358-373 new optional fields (default None) | NO-OP | defaults reproduce original behaviour |
| experiments/benchmark.py | 534-542 `resolve_configs` task/yaml consistency check | NO-OP | only raises on mismatch; Walker/UpStepper yamls match |
| campaign/milestone.py | 33-63, 353-355, 443-446, 457-525, 566-572 | NO-OP for evaluations | post-hoc reporting (K-aware sensitivity file names, floor-effect check for non-original tasks); not on the genome-evaluation path. Output names for UpStepper K=2 unchanged |
| campaign/alerts.py, status.py, monitor.py, dashboard.py, dry_run.py | 7+ small hunks each | NO-OP for evaluations | monitoring/dry-run/estimate plumbing; legacy values preserved for Walker/UpStepper |
| campaign/estimate.py | new file | NO-OP | estimate reader |
| experiments/configs/{direct,nca,rule_based}/{cavecrawler,climber,obstacletraverser}.yaml | new files | NO-OP | new tasks only; walker/upstepper yamls and `configs/shared` unchanged |
| experiments/configs/campaign_ext*.yaml, ext_validation_estimate.json | new/derive-modified | NO-OP | extension campaign configs |
| evodevo/extension/* | new package | NO-OP | not imported by the campaign worker |

**BEHAVIOURAL hunks: none found.** One thing to flag for Amendment 2 (not an evaluation change): `BenchmarkConfig` has a single global `seeds` list, and the scheduler keys phase status/metrics by task name (`scheduler.py` ~441-510). Two phases for ObstacleTraverser-v1 (seeds 0-4, 5-9) with per-phase seeds cannot be expressed as-is; Amendment 2's plan code needs new work (not part of this gate; nothing was tested).

### R2 — config field diff (current code vs original manifest)

Built with current code from the original `benchmark.yaml` (copy in scratch with seeds 0-9 and an absolute `configs_root`). 
- `CampaignPlan.from_benchmark(benchmark.yaml)` with seeds 0-4: `config_hash` **e50111cb…a55cd0 = manifest `config_hash`** (identical). (`plan.to_dict() == manifest['plan']` printed False only because the JSON-stored manifest holds lists where the dataclass dict holds tuples; the hash is computed over the JSON form and matches.)
- For each of direct/nca/rule_based × Walker-v0/UpStepper-v0, the seed-5 config (and seeds 6-9) differs from the manifest seed-0 config in exactly one field: `seed` 0→5 (…→9). Seeds 0-4 configs are identical to the manifest's.
- Confirmed in the seed-5 configs: Walker `ppo_timesteps` 1,000,000, `ppo_seeds` 1; UpStepper 1,200,000, `ppo_seeds` 2; `ppo_seed_aggregation` max; `eval_budget` 300; `population_size` 15; `offspring_size` 15; `ppo_eval_episodes` 1; `use_crossover` False; `grid_shape` [5,5]; `min_voxels` 3; `min_actuators` 1; `sentinel_fitness` -10; `cache_enabled` False; encoder kwargs identical per arm (direct mutation_rate 0.15353558052434457; nca mutation_sigma 0.03045437279788335, hidden_channels 0; rule_based mutation_rate 0.07785310734463277, readout mode_of_window, window 6; all dev_steps 100, init_condition center_seed). Phase timeouts: Walker 13013.800927747687 s, UpStepper 15119.187441965214 s — both equal to manifest.
- PPO hyperparameters are not config fields: they are SB3 defaults in `ppo_runner.evaluate_ppo_task` (`PPO("MlpPolicy", vec_env, seed=task.seed, verbose=0)`), unchanged between the commits.

### R3 — versions

| item | recorded | installed | same? |
|---|---|---|---|
| python | 3.10.20 (GCC 13.3.0) — thermal-study `machine.json`, SB3 `system_info.txt` in campaign policy zips | 3.10.20 | yes |
| OS/kernel | Linux-7.0.0-31-generic-x86_64-glibc2.39 (policy zip system_info) | same | yes |
| torch | 2.14.0+cpu | 2.14.0+cpu | yes |
| stable-baselines3 | 2.9.0 | 2.9.0 | yes |
| gymnasium | 1.3.0 | 1.3.0 | yes |
| numpy | 1.26.4 | 1.26.4 | yes |
| cloudpickle | 3.1.2 | 3.1.2 | yes |
| evogym | 2.0.0, commit f63c649c688adda1accad5f86731aad96606388a | 2.0.0, `external/evogym` HEAD f63c649c688adda1accad5f86731aad96606388a; simulator `.so` built 2026-09-19 22:38, before the campaign | yes |
| full pip freeze (41 packages, ENVIRONMENT_REPORT.md) | — | `pip list --format=freeze` normalised diff: **identical**; venv last modified 2026-09-19 | yes |

No `run_meta.json` exists in the campaign run directories (only in old `results/runs`); the recorded versions are therefore from the SB3 `system_info.txt` inside the campaign's policy zips, `thermal_worker_study/…/machine.json`, `ENVIRONMENT_REPORT.md` and `docs/LAB_MACHINE_SETUP.md`. Minor: LAB_MACHINE_SETUP mentions an earlier setuptools pin 78.1.1; the final freeze records 80.10.2 and that is what is installed. CPU model i7-14700K is the same machine.

### R4 — seed scheme

- Evolution RNG: `EvolutionRun.rng = np.random.default_rng(config.seed)` (ea.py:43), i.e. the run seed alone. Genome decode uses no RNG.
- PPO seed: `ppo_seed(base_seed=config.seed, k)` (ppo_runner.py:119): k=0 → the run seed itself; k≥1 → `SeedSequence([base_seed, k]).generate_state(1)[0]`. It is passed to `make_vec_env(seed=…)` for training and evaluation envs and to `PPO(seed=…)`. **The evaluation index does not enter the PPO seed**: every evaluation in a run trains with the same PPO seed(s) (k=0: run seed; UpStepper k=1: derived value). Worker `seed_everything(slot)` is overwritten by PPO's own seeding (replays confirm).
- Run seeds 0-9: PPO seeds k=0: 0…9; k=1: 3964924996, 1189033389, 307626447, 457190280, 2570931004 (seeds 0-4) and 3796490668, 1175807978, 369571992, 3990686019, 3933992529 (seeds 5-9). All 20 values distinct.
- Preflight: curve seed and random-body base 20261015; noise `ppo_seed(20261016, rep)` = 20261016, 1261988398, 445569080, 2374680399. Planned re-evaluation bases (if derived with `ppo_seed(base,k)`, k=0..9): 20261101 → 20261101, 2209434231, 901980336, 3324932687, 1141897181, 3742006676, 1736745405, 2204562315, 2954520885, 2766414702; 20261201 → 20261201, 123862953, 699303471, 3272518155, 343254231, 2149602028, 328476621, 3594614917, 44242627, 3033923287.
- Checked: **no collision** between {run seeds 0-9 and their k=1 values} and {preflight seeds, re-evaluation k=0..9 values for both bases}. Seeds 5-9 do not collide with seeds 0-4. The re-evaluation bases are not defined in code yet, so this holds for the `ppo_seed` scheme I assumed.

### R5 — replay of 12 recorded evaluations with current code

Selection rule (deterministic): for each task × arm the first run (seed 0-4) with a valid, non-cache-hit, fully-recorded generation-0 evaluation (first by eval index) and last-generation (19) evaluation (last by eval index). Resulting runs: Walker direct seed 1, nca seed 0, rule_based seed 0; UpStepper direct seed 1, nca seed 0, rule_based seed 0. Each recorded genome was deserialised and decoded with the encoder built from the manifest config; decoded body and body hash equal the recorded ones (asserted). Harness (`worker_pool._worker_initializer` with `configure_threading(1, strict)`, `pin_current_process`, `seed_everything(slot)`, then `worker_pool._run_ppo_task` with the phase timeout) ran each PPO training (18 trainings: 6 Walker + 6×2 UpStepper k=0/k=1) as a separate process pinned to its own core among 0,2,…,14,16-25 (all 18 running concurrently), as dispatch units are in the campaign. Thread env confirmed `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=1`, `*_DYNAMIC=FALSE`, `torch.get_num_threads()=1`. Wall time 0.45–1.0 h per training (Walker ~3.1-3.4 ks on E-cores; UpStepper ~1.6-1.7 ks on P-cores and 3.6-3.7 ks on E-cores 16-19).

| task | arm | run | eval idx | gen | recorded (aggregate) | replayed (aggregate) | per-seed recorded | per-seed replayed | equal? |
|---|---|---|---|---|---|---|---|---|---|
| Walker-v0 | direct | seed_1 | 2 | 0 | 7.002670207295523 | 7.002670207295523 | [7.002670207295523] | [7.002670207295523] | YES (bit-identical) |
| Walker-v0 | direct | seed_1 | 299 | 19 | 9.018117525381967 | 9.018117525381967 | [9.018117525381967] | [9.018117525381967] | YES (bit-identical) |
| Walker-v0 | nca | seed_0 | 0 | 0 | 7.309246875840472 | 7.309246875840472 | [7.309246875840472] | [7.309246875840472] | YES (bit-identical) |
| Walker-v0 | nca | seed_0 | 299 | 19 | 6.925461149323382 | 6.925461149323382 | [6.925461149323382] | [6.925461149323382] | YES (bit-identical) |
| Walker-v0 | rule_based | seed_0 | 1 | 0 | 7.738111215439858 | 7.738111215439858 | [7.738111215439858] | [7.738111215439858] | YES (bit-identical) |
| Walker-v0 | rule_based | seed_0 | 299 | 19 | 10.56990822145599 | 10.56990822145599 | [10.56990822145599] | [10.56990822145599] | YES (bit-identical) |
| UpStepper-v0 | direct | seed_1 | 2 | 0 | 1.530395527455397 | 1.530395527455397 | [-2.6625799686153186, 1.530395527455397] | [-2.6625799686153186, 1.530395527455397] | YES (bit-identical) |
| UpStepper-v0 | direct | seed_1 | 299 | 19 | 1.4749613378003232 | 1.4749613378003232 | [1.4749613378003232, 1.4714388526172542] | [1.4749613378003232, 1.4714388526172542] | YES (bit-identical) |
| UpStepper-v0 | nca | seed_0 | 0 | 0 | 1.4937089784361888 | 1.4937089784361888 | [1.3694645317416416, 1.4937089784361888] | [1.3694645317416416, 1.4937089784361888] | YES (bit-identical) |
| UpStepper-v0 | nca | seed_0 | 299 | 19 | 1.4719884762912159 | 1.4719884762912159 | [1.422651746413976, 1.4719884762912159] | [1.422651746413976, 1.4719884762912159] | YES (bit-identical) |
| UpStepper-v0 | rule_based | seed_0 | 1 | 0 | 1.4471236254885298 | 1.4471236254885298 | [1.394162094524404, 1.4471236254885298] | [1.394162094524404, 1.4471236254885298] | YES (bit-identical) |
| UpStepper-v0 | rule_based | seed_0 | 299 | 19 | 2.7980774242004145 | 2.7980774242004145 | [1.4641060002159065, 2.7980774242004145] | [1.4641060002159065, 2.7980774242004145] | YES (bit-identical) |

**All 12 evaluations: aggregate fitness AND each per-seed fitness equal at full float precision (`==` on Python floats, repr shown).**

Defect in my harness (does not affect fitness): all replays wrote their saved policy zips into one shared directory with the campaign's file-name stems (`e<eval_idx>_k<k>.zip`), and several replays share a stem (e.g. all six `e00299_k0`), so they overwrote one another. A byte/tensor comparison of the replayed zips with the recorded zips therefore matched only for 8 of 18 files (those that were not overwritten) and showed different network shapes for the rest; that comparison is not meaningful and I did not repeat the 1-hour run. The fitness comparison above is unaffected (fitness is computed before the save). Separately, I confirmed on one recorded policy (Walker direct seed 1 eval 2) that reloading the saved zip and rolling out in a fresh eval env with the same seed reproduces the recorded fitness 7.002670207295523 exactly.

### R6
Not required (no replay differed). Not run. `~/evodevo_original_tree` was not created.

### Outcome
R2 identical, R3 identical, 12/12 replays bit-identical with current code → **PASS**. (Caveat: 12 of the original 12,000 evaluations (4 arms × 5 seeds × 300 × 2 tasks), one environment, P and E cores both exercised; recorded wall clocks in the campaign were taken with the P-core 4.2 GHz cap, replay at default frequency — no effect on results observed.)

## 4. Gate S — saving PPO state (report only; nothing changed)

- **Saved per training today** (`ppo_runner.evaluate_ppo_task`, whenever `artifact_dir` is set, which the campaign scheduler does): after `model.learn()` and after `_rollout_fitness`, `model.save(<run>/policies/e{eval_index:05d}_k{k}.zip)` and `<run>/training_logs/e{eval_index:05d}_k{k}.json` (`{"timesteps","fitness","seed"}`, ~64-73 B). The zip is SB3's standard zip: `data` (~17.5 kB JSON of hyperparameters and class refs), `policy.pth` (~68 kB), `policy.optimizer.pth` (~136 kB Adam state), `pytorch_variables.pth` (1.3 kB), `_stable_baselines3_version`, `system_info.txt` — total ≈223 kB (Walker direct, 14-dim action space) to ≈276 kB (UpStepper nca). So **the policy is saved**, and the optimizer state is already in it. Saved only for valid, trained evaluations (e.g. 91 files for Walker direct seed 0).
- **Learning-rate schedule:** constant 3e-4 (SB3 default `FloatSchedule`; `clip_range` constant 0.2). No decay over any horizon. Other PPO settings are SB3 defaults: n_steps 2048, batch 64, 10 epochs, γ 0.99, λ 0.95, ent_coef 0, vf_coef 0.5, max_grad_norm 0.5; Adam lr 3e-4, eps 1e-5.
- **Observation normalisation:** none. The env stack is `make_vec_env` → `DummyVecEnv`; no `VecNormalize` anywhere in `ppo_runner.py`. There is therefore no normaliser state to save.
- **What a post-training save needs without touching RNG streams or fitness:** nothing beyond what already happens. `model.save` draws from no random stream (serialises state dicts and JSON); it executes after the fitness rollout, so it cannot alter the recorded fitness; a saved policy reloaded with `PPO.load` and evaluated in an identically seeded env reproduces the fitness (verified). Not currently saved: torch/numpy RNG states and the rollout buffer; reading RNG state (`torch.get_rng_state()`) also draws nothing, but resuming training bit-exactly would additionally need the env RNG and last observation — out of scope here. A normaliser, were one added, would be saved with `VecNormalize.save`, which is also draw-free. Practical caveat: the saved zips contain the model after the last (overshooting) update (e.g. `num_timesteps` 1,001,472 for a 1M budget).
- **Champion (milestone code):** per run, `evodevo/campaign/milestone.py` ~line 226-240: `group = group.sort_values("eval_index")`, `best = group.loc[group["fitness"].idxmax()]` — the evaluation with the maximum recorded fitness (aggregated over PPO seeds with `max`, over all rows of the run_log including invalid sentinel −10 rows, which cannot win). **Tie rule: pandas `idxmax` returns the first occurrence, i.e. the lowest eval_index.** Per-arm "best_bodies" (~line 402) use `group.loc[group["fitness"].idxmax()]` on the concatenated, unsorted frame of all seeds, so a cross-seed exact tie is broken by frame order (first run directory read). The primary endpoint (`analysis/aggregate.per_seed_finals`) is the value of the cumulative max at the last evaluation, so it is tie-independent. In the EA itself survivor selection is `sorted(..., key=fitness, reverse=True)` (stable: earlier population entry wins ties).

## 5. Cost projection

Method = `derive`'s: `project_phase_band(walker_h=173.1823 h, core_arm_busy_share=0.690984, ratio, K, backtest relative error 0.0799559)`; band half-width max(10%, |0.080|) = 10%. Obstacle cost ratio 1.088637 at 600k (preflight, vs Walker 1M), scaled linearly in T (as `derive.project` does for a guard-lowered budget); UpStepper ratio 1.212532 at 1.2M, K=2; Walker ratio 1, K=1. Hours:

| item | p5 | mean | p95 |
|---|---:|---:|---:|
| Phase 1 / Phase 2 each, Obstacle @ 1.0M (reference) | 195.4 | 217.1 | 238.8 |
| Phase 1 / Phase 2 each, Obstacle @ 1.2M | 234.5 | 260.5 | 286.6 |
| Phase 1 / Phase 2 each, Obstacle @ **1.5M** | 293.1 | 325.7 | 358.3 |
| Phase 3 UpStepper seeds 5-9, 3 arms (1.2M, K=2) | 261.2 | 290.2 | 319.2 |
| Phase 4 Walker seeds 5-9, 3 arms (1M, K=1) | 107.7 | 119.7 | 131.6 |
| Re-evaluation 30 champions × 10 trainings, Obstacle @1.5M | 29.6 | 32.9 | 36.1 |
| Re-evaluation, UpStepper @1.2M | 13.2 | 14.6 | 16.1 |
| Re-evaluation, Walker @1M | 10.9 | 12.1 | 13.3 |
| 2M robustness 30×3 Obstacle | 12.6 | 14.0 | 15.4 |
| 2M robustness 30×3 UpStepper | 7.0 | 7.8 | 8.6 |
| 2M robustness 30×3 Walker | 7.0 | 7.7 | 8.5 |
| 2M robustness, all 90 champions × 3 | 26.6 | 29.6 | 32.5 |

Re-evaluation and robustness are not phases of the `derive` method, so I used a pool-throughput variant of the same reference data: hours = n_trainings × (mean pool-busy seconds per Walker 1M training of the three core arms in the original campaign, 2526.6 s) × (task cost ratio × T/T_measured) ÷ (18 workers × 0.99802 utilisation) + half a mean training for the tail; same ±10% band. Assumptions to check: 300 trainings per task (30 champions × 10), one training of K per champion at the task's budget (no K multiplier), champions cost like average campaign bodies, no generation barriers. The campaign's own all-arm UpStepper/Walker per-training ratio was 1.137 versus the preflight ratio 1.213 used here, and the backtest over-predicted the original UpStepper phase by 8.0% — the projection for UpStepper may be on the high side.

Calendar, sequential on the one machine, launch **2026-10-26 12:00 IST**, T_obstacle = 1.5M (p5 / mean / p95 finish):

| step | finish p5 | finish mean | finish p95 |
|---|---|---|---|
| Phase 1 Obstacle seeds 0-4 | Sat 7 Nov 17:06 | Mon 9 Nov 01:40 | Tue 10 Nov 10:15 |
| Phase 2 Obstacle seeds 5-9 | Thu 19 Nov 22:13 | Sun 22 Nov 15:21 | Wed 25 Nov 08:30 |
| Phase 3 UpStepper seeds 5-9 | Mon 30 Nov 19:24 | Fri 4 Dec 17:33 | Tue 8 Dec 15:43 |
| Phase 4 Walker seeds 5-9 | Sat 5 Dec 07:06 | Wed 9 Dec 17:13 | Mon 14 Dec 03:21 |
| Re-eval Obstacle | Sun 6 Dec 12:40 | Fri 11 Dec 02:04 | Tue 15 Dec 15:29 |
| Re-eval UpStepper | Mon 7 Dec 01:50 | Fri 11 Dec 16:43 | Wed 16 Dec 07:35 |
| Re-eval Walker | Mon 7 Dec 12:42 | Sat 12 Dec 04:47 | Wed 16 Dec 20:52 |
| 2M robustness (90×3) | Tue 8 Dec 15:20 | Sun 13 Dec 10:22 | Fri 18 Dec 05:24 |

Cumulative mean 1,150 h (≈48 days). Same schedule with T=1.2M: phases end Fri 6 Nov / Tue 17 Nov / Sun 29 Nov / Fri 4 Dec (mean); everything finished mean Tue 8 Dec 00:06 (p5 3 Dec 18:05, p95 12 Dec 06:06); cumulative mean 1,020 h. Weekday names are for 2026.

## 6. Anything abnormal

1. **The rule's answer (1.5M) is not a passing budget**: its row fails fraction_ok and order_ok; only 2M passes. It is the cap/fallback value, not evidence of convergence at 1.5M. The preflight rests on one training per body (n=6) and rule_based:evolved has a 3.10 jump at 2M that is a single outlier-like observation.
2. **Cost is very large**: Amendment 2 projects to ~1,150 h at 1.5M versus the 360 h GUARD-ext threshold (and 486.9 h for the three-task extension it replaces). The GUARD-ext rule is not applied here; I only report.
3. **Plan code cannot yet express Amendment 2**: single global `seeds` list; phase state keyed by task name (two ObstacleTraverser phases would collide in `phase_status`/`phase_metrics`). Not tested by me.
4. **PPO seed does not depend on the evaluation index** (all evaluations of a run train with the same PPO seed, k=0 = run seed); this is how the original campaign ran and why replays are exactly reproducible.
5. **My R5 policy-zip comparison was invalid** because of file-name collisions in a shared output directory (§R5). Fitness results are unaffected. The extra comparison could be redone with per-item output directories (~1 h) if wanted.
6. The version evidence for the original campaign comes from policy-zip `system_info.txt`, a pre-campaign machine profile and ENVIRONMENT_REPORT, not from a per-run record; no `run_meta.json` exists in the campaign directory.
7. `scipy` is not in the venv; rank correlations were computed by hand (no packages installed or changed).
8. No sudo used; replay processes ran at default CPU frequency.

## 7. Final state checks
- `pgrep -af evodevo`: the only match is `/sbin/mount.ntfs /dev/sdb1 /mnt/evodevo-backup -o rw` (the pre-existing backup-disk mount, matched by the path name); no campaign, replay or other evodevo process is running. `tmux ls`: no server running.
- `git status -sb`: unchanged — `M evodevo/experiments/configs/campaign_ext.yaml` plus untracked `docs/campaign_prep/ext/EXT_DECISIONS.json`, `EXT_DECISIONS.md`, `final_estimate_ext.json`. No commits made.
- `~/evodevo_original_tree` does not exist (R6 was not needed).
- Replay outputs are in `~/evodevo_private_notes/replay_r5_current/` (results, logs, `compare.json`); scratch scripts are in the session scratchpad. Nothing under `results/` was written.
