# EvoGym_Project — Technical Inventory Report

Read-only inspection. No files were modified, moved, renamed, or deleted; no implementation code was written or refactored. All claims cite exact paths/classes/functions; anything unverifiable by static reading is flagged explicitly rather than assumed.

## Table of Contents
- [0. Critical structural finding (git)](#0-critical-structural-finding-git)
- [A. Repo Inventory](#a-repo-inventory)
- [B. Encoding Implementations](#b-encoding-implementations)
- [C. Shared Pipeline Components](#c-shared-pipeline-components)
- [D. Configuration & Experiment Entry Points](#d-configuration--experiment-entry-points)
- [E. Visualization / Rendering](#e-visualization--rendering)
- [F. Results / Outputs Already Produced](#f-results--outputs-already-produced)
- [G. Tests](#g-tests)
- [H. Naming & Structure Inconsistencies](#h-naming--structure-inconsistencies)
- [I. Dependency & Environment State](#i-dependency--environment-state)
- [J. Gaps Against a Fair Three-Way Comparison](#j-gaps-against-a-fair-three-way-comparison)
- [Summary for reorganization planning](#summary-for-reorganization-planning)

---

## 0. Critical structural finding (git)

- `/Users/tejeshwarcdr/Academics/EvoGym_Project` is **not its own git repository**. `git rev-parse --show-toplevel` from inside it resolves to `/Users/tejeshwarcdr` (the whole home directory is the worktree), on branch `feature-branch-backend`, with **zero commits** and `git ls-files` returning nothing from inside the project. None of `evodevo/`, `docs/`, `results/`, `tests/`, `scripts/`, or the vendored `evogym/` copy is under version control right now.
- `evogym-rule-based-encoding/.git` **is** a real, independent repository with its own history (`703e711 Initial project implementation`, plus upstream EvoGym history).
- It is **not a git submodule**: no `.gitmodules` entry anywhere in the outer tree references it, and its own `.gitmodules` only registers its *own* nested submodules (pybind11, glfw, glew, two RL example repos) — unrelated to the outer project.
- **Conclusion**: `evogym-rule-based-encoding/` is a plain, unintegrated nested git checkout dropped inside the main folder — a separate project with its own history, wired into the main repo via no mechanism at all (no submodule pointer, no subtree merge, no symlink).

---

## A. Repo Inventory

Pruned tree (excludes vendored C++ build artifacts, `.venv`, `.git` internals, `examples/`/`tutorials/` inside the vendored EvoGym library itself):

```
EvoGym_Project/
├── README.md                          — stale: claims "skeleton only, nothing implemented" — contradicted by the working evodevo/ pipeline below
├── pytest.ini                         — registers `slow` marker (real-PPO tests skipped unless evogym simulator is built)
├── requirements.txt                   — numpy<2.0.0, gymnasium, stable-baselines3, pyyaml, pytest, pandas, matplotlib
├── docs/
│   ├── decisions.md                   — decision log; several rows still marked OPEN
│   ├── evodevo-implementation-roadmap.md
│   └── evogym_technical_reference.md
├── evodevo/                           — the real pipeline
│   ├── config.py                      — frozen `ExperimentConfig` dataclass, YAML round-trip
│   ├── seeding.py                     — `make_rng`, `seed_everything`
│   ├── encoders/
│   │   ├── base.py                    — abstract `Encoder` contract
│   │   ├── direct.py                  — `DirectEncoder` (Baseline)
│   │   ├── nca.py                     — `NCAEncoder`
│   │   └── __init__.py                — registry {"direct":DirectEncoder,"nca":NCAEncoder}, `get_encoder()`
│   ├── morphology/
│   │   ├── validate.py                — `validate_body()` (shared validation)
│   │   └── metrics.py                 — `body_metrics()`, `body_hash()`
│   ├── evaluation/
│   │   ├── ppo_runner.py              — `evaluate_body` (real PPO), `evaluate_body_surrogate`, `make_evaluator`
│   │   └── cache.py                   — `FitnessCache` (body-hash keyed)
│   ├── evolution/
│   │   ├── ea.py                      — `run_ea` ((mu+lambda) EA), encoding-agnostic
│   │   └── logging.py                 — `RunLogger` (JSONL + run_meta.json)
│   ├── experiments/
│   │   ├── run_experiment.py          — CLI entry point, single (encoder, task, seed) run
│   │   ├── fairness.py                — `assert_comparable()` fairness gate
│   │   ├── nca_sanity.py              — E0 sanity report (NCA vs direct priors, no PPO)
│   │   └── configs/{direct_upstepper,direct_walker,nca_upstepper,nca_walker}.yaml
│   └── analysis/
│       ├── aggregate.py               — `load_runs`, `discover_runs`, `comparison_table`
│       ├── plots.py                   — `plot_fitness_curves`, `plot_morphology_comparison`
│       └── ablation_report.py         — A1/A2/E3 ablation report generator
├── scripts/
│   ├── run_baseline.sh                — direct encoder, Walker-v0, seeds 0-4
│   ├── run_comparison.sh              — direct vs nca, calls `assert_comparable()` first
│   └── run_ablations.sh               — A1 (dev_steps∈{1,5,10,20}), A2 (hidden_channels∈{0,3}), E3 (2nd task)
├── tests/                             — 11 files, 1069 lines (see §G)
├── results/                           — partial outputs (see §F)
├── evogym/                            — vendored copy of the EvoGym library itself (pip-installed from source per requirements.txt comment)
└── evogym-rule-based-encoding/        — separate, unintegrated project (own .git, own vendored evogym/ copy)
    ├── README.md, LICENSE, MANIFEST.in, setup.py, pyproject.toml, environment.yml, requirements*.txt, build.log
    ├── PROJECT_REPORT.md, PROJECT_FINAL_TECHNICAL_REPORT.md, FINAL_THREE_WAY_COMPARISON_REPORT.md — prior narrative results
    ├── evogym/                        — vendored EvoGym core (envs/, sim.py, world.py, viewer.py, compiled simulator_cpp .so)
    ├── examples/
    │   ├── ga/run.py                  — V1 GA entry point (also used as historical baseline)
    │   ├── ga/run_developmental.py    — V2 (rule-based developmental) GA entry point — the "Rule-Based" arm
    │   ├── ga/test_run_developmental.py
    │   ├── ppo/run.py, args.py, eval.py, callback.py, group_ppo.py — this codebase's own PPO wrapper
    │   ├── utils/rule_types.py, rule_encoding.py, rule_interpreter.py — "V1" static rule language
    │   ├── utils/developmental_rule_types.py, developmental_rule_encoding.py, developmental_interpreter.py — "V2" developmental rule language
    │   ├── utils/test_rule_encoding.py, test_developmental_rules.py
    │   ├── utils/algo_utils.py, mp_group.py
    │   ├── cppn_neat/, bo/            — other baselines (CPPN-NEAT, Bayesian Opt), not part of the 3-way comparison
    │   └── make_gifs.py, visualize.py — rendering
    ├── saved_data/                    — test_ga/ (gens 0-3: controller .zip, structure .npz, output.txt) present; several other dirs referenced by the final report are NOT present on disk (see §F)
    └── tests/                         — EvoGym library's own env tests (screen_free/, requires_screen/), not rule-encoding-specific
```

---

## B. Encoding Implementations

### Shared interface — `evodevo/encoders/base.py`
```python
class Encoder(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def genome_length(self) -> int: ...
    @abstractmethod
    def random_genome(self, rng) -> np.ndarray: ...
    @abstractmethod
    def decode(self, genome) -> np.ndarray: ...   # must return (5,5) int array, no validity handling
    @abstractmethod
    def mutate(self, genome, rng) -> np.ndarray: ...  # must not loop until valid
    @abstractmethod
    def describe(self) -> dict: ...
```
Invalid-genome handling is deliberately excluded from the encoder contract by design (docstring) — it's a pipeline concern handled downstream in `ppo_runner.py`.

### Baseline — `evodevo/encoders/direct.py`, class `DirectEncoder`
- **Genome**: flat `np.ndarray[int]`, length 25 (`5×5`), values in `{0..4}` (`N_MATERIALS=5`: EMPTY/RIGID/SOFT/H_ACT/V_ACT). Example: `[0,2,2,3,0, 0,2,2,2,0, ...]` reshaped row-major to 5×5.
- **Decode**: pure reshape to `(5,5)` — genotype ≡ phenotype, no development step.
- **Mutate**: per-cell Bernoulli(`mutation_rate=0.1`) resample from the material distribution.
- **Crossover**: none (mutation-only).
- Correctly subclasses `Encoder` (all 5 abstract members implemented).
- **Hardcoded**: `N_MATERIALS=5`, default `grid_shape=(5,5)`, `mutation_rate=0.1`, `empty_prob=0.6`.
- **Runs end-to-end?** Imports only `numpy` and `evodevo.encoders.base` — no unresolved imports at the encoder level (physics/evogym dependency is isolated to `ppo_runner.py`). Not executed in this inspection — assessed by static reading only.
- Per `docs/decisions.md`: random direct genomes are valid only ~2.25% of the time (measured, N=2000, seed 0), which drove `eval_budget` up rather than encoder retuning.

### NCA — `evodevo/encoders/nca.py`, class `NCAEncoder`
- **Genome**: flat `float64` vector, weights only, for a 2-layer perception MLP. With default hyperparams (`channels = N_MATERIALS(5) + hidden_channels(3) = 8`, `input_dim = 5×channels = 40` [self + 4 cardinal neighbours], `hidden_width=16`): length = `40×16 + 16 + 16×8 + 8 = 792`.
- Fixed, non-evolved initial cell state (`_INIT_STATE_SEED = 20260902`, `normal(scale=0.1, size=(5,5,channels))`) — identical for every individual and generation, deliberately asymmetric by design (a symmetric init would prevent differentiation under a deterministic synchronous update).
- **Decode/develop**: `dev_steps=10` synchronous residual updates (`state += tanh(perceive(state) @ w1 + b1) @ w2 + b2`), zero-padded von-Neumann (4-neighbour) perception only — Moore neighbourhood explicitly unsupported (raises `ValueError`). Final decode is `argmax` over the first `N_MATERIALS` channels → `(5,5)` int array.
- **Mutate**: Gaussian perturbation (`mutation_sigma=0.1`) applied to a `mutation_fraction=1.0` (all, by default) subset of genome indices.
- **Crossover**: none.
- Correctly implements the `Encoder` interface.
- **Hardcoded**: `N_MATERIALS=5`, 4-neighbour offsets, `_INIT_STATE_SEED`, `_INIT_STATE_SCALE=0.1`, `hidden_channels=3`, `hidden_width=16`, `dev_steps=10`, `init_scale=0.5`.
- **Runs end-to-end?** Same as direct.py — imports only numpy/base, no unresolved imports found by static reading; not executed.
- Per `docs/decisions.md`: validity rate 69.15% (N=2000, seed 0) — passes stated gates — but 68.35% of genomes are still changing state on the final developmental step (never settle within `dev_steps=10`), and residual magnitude at t=T is ~40.7 vs an init scale of 0.1 — the update is explicitly not bounded/self-stabilizing; recorded as a known limitation, not currently a blocker.

### Rule-Based — `evogym-rule-based-encoding/examples/utils/`
Two rule languages exist: "V1" (static, `rule_types.py`/`rule_encoding.py`/`rule_interpreter.py`) and "V2" (developmental, `developmental_rule_types.py`/`developmental_rule_encoding.py`/`developmental_interpreter.py`). The framing in `FINAL_THREE_WAY_COMPARISON_REPORT.md` indicates **V2 is the "Rule-Based" arm** of the 3-way comparison; V1 is a prior/simpler iteration kept for reference.
- **Genome**: `DevelopmentalRuleGenome` = an ordered, order-meaningful (first-match) list of 2–12 (`MIN_RULES=2`, `MAX_RULES=12`) `DevelopmentalRule` objects. Each rule is `condition_type(+params) -> action_type(+params)`. `ConditionType` = ALWAYS/CELL_TYPE/NEIGHBOR_COUNT/EMPTY_NEIGHBOR/POSITION; `ActionType` = 5 `SET_*` + `GROW_DIRECTIONAL` + `COPY_NEIGHBOR`. Not a fixed-shape numeric vector like the other two — no comparable "shape/dtype" example beyond the rule-object list itself.
- **Decode**: `developmental_interpreter.py::develop_rules()` — deterministic, bounded-iteration (`DEFAULT_MAX_ITERATIONS=5`, `HARD_MAX_ITERATIONS=10`), synchronous, two-pass (self-actions, then `GROW_DIRECTIONAL`) update from a single SOFT seed cell at grid center.
- **Mutate**: `mutate_developmental_rules` — 7 structural operators (condition type/params, action type/params, add/remove/reorder rule), each applied independently at `mutation_rate=0.2`.
- **Crossover**: `crossover_developmental_rules` — one-point cut on rule lists with length repair to stay within `[MIN_RULES, MAX_RULES]`. Unlike evodevo's Baseline/NCA, this arm has both mutation **and** crossover.
- **Interface conformance**: does **not** implement `evodevo.encoders.base.Encoder` — no shared base class at all; standalone.
- **Hardcoded**: same 5-symbol numeric material vocabulary as evodevo (0=EMPTY..4=V_ACT) but independently redefined (`VoxelState.EMPTY/SOFT/HARD/ACTUATOR_HORIZONTAL/ACTUATOR_VERTICAL`), not imported/shared; `MIN_ACTUATORS=2`; iteration caps as above.
- **Validity check**: `actuator_count >= MIN_ACTUATORS(2)` and `is_connected(body)`/`has_actuator(body)` (EvoGym's own helpers) in `generate_robot_from_developmental_rules()` — returns `None` on failure, never repairs/resamples. This is a **separate, independently implemented** validation path, not a call into `evodevo/morphology/validate.py`.
- **Entry point**: `examples/ga/run_developmental.py`.
- **Runs end-to-end?** Imports `evogym` (`is_connected`, `has_actuator`, `get_full_connectivity`) and local `utils.*` modules; self-contained within `evogym-rule-based-encoding/examples/`. Not wired into `evodevo/encoders/` in any way. Not executed in this inspection.

---

## C. Shared Pipeline Components

- **Morphology validation**: `evodevo/morphology/validate.py::validate_body()` checks, in order: shape == grid_shape, integer dtype, values ⊆ {0..4}, `occupied_count >= min_voxels(3)`, single 4-connected component, ≥1 actuator. Both `DirectEncoder` and `NCAEncoder` rely on this **same** function, called downstream in `ppo_runner.py::evaluate_body`/`evaluate_body_surrogate` — not inside the encoders themselves. The Rule-Based arm uses a **separate**, duplicated check (`is_connected`+`has_actuator`+`actuator_count>=2`) with a different minimum-voxel floor (2 actuators vs 3 total occupied cells) — **not shared code**.
- **Connectivity**: both sides use EvoGym's own `get_full_connectivity(body)` (genuinely shared, since it's library code) — called independently in `ppo_runner.py`, `run_experiment.py`, and `developmental_rule_encoding.py:214`.
- **PPO evaluation**:
  - `evodevo/evaluation/ppo_runner.py::evaluate_body` — `PPO("MlpPolicy", vec_env, seed=seed, verbose=0)`, all SB3 hyperparameters left at library defaults, `model.learn(total_timesteps=ppo_timesteps)` (config value `5000` in all 4 yamls), `ppo_eval_episodes=1` deterministic rollout for fitness. Also `evaluate_body_surrogate` — explicitly labeled non-scientific, physics-free stand-in (fitness = actuator_count + horizontal_symmetry_score), used for pipeline testing only.
  - `evogym-rule-based-encoding/examples/ppo/run.py::run_ppo` + `args.py` — explicit, tuned hyperparameters: `learning_rate=2.5e-4, n_steps=128, batch_size=4, n_epochs=4, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, max_grad_norm=0.5, ent_coef=0.01, clip_range=0.1`, default `total_timesteps=1e6`, own `EvalCallback` (periodic eval every `eval_interval=1e5` steps, best-reward tracking) rather than a single terminal rollout.
  - **Inconsistency**: fundamentally different PPO stacks (SB3 defaults + single post-hoc eval vs explicit tuning + periodic best-so-far eval); timesteps differ by orders of magnitude (evodevo configs: 5000; rule-based's *executed* 3-way comparison: 300; rule-based's own "normal" default: 1e6).
- **Evolutionary loop**:
  - `evodevo/evolution/ea.py::run_ea` — (mu+lambda) truncation, encoder-agnostic (docstring forbids importing physics/RL libs directly). `population_size=15`, `offspring_size=15` (all 4 yamls), hard `eval_budget=300`. Selection: sort survivors by fitness desc, truncate to mu; parents chosen uniformly at random. Termination: `while evaluations_used < budget`. **Every** `evaluate_genome()` call increments `evaluations_used`, valid or not (invalid → sentinel fitness `-10.0` default, no resampling).
  - `evogym-rule-based-encoding/examples/ga/run.py`/`run_developmental.py` — variable survivor count via `get_percent_survival_evals()`; the *executed* 3-way comparison used `pop_size=5, max_evaluations=20`. Reproduction **retries** invalid/duplicate offspring (init cap `MAX_INIT_ATTEMPTS_PER_INDIVIDUAL=500`; reproduction loop runs `while num_children < ... and num_evaluations < max_evaluations`). **Only valid+unique children increment `num_evaluations`** — invalid genomes do not consume the evaluation budget here, unlike evodevo.
  - This is a substantive, documented difference in invalid-genome policy between the two codebases.
- **Caching**: `evodevo/evaluation/cache.py::FitnessCache` — dict keyed by SHA-256 `body_hash`, tracks hits/misses/hit_rate. `cache_enabled: false` by default in all yaml configs, and **no call site** wires `FitnessCache` into `run_ea`/`run_experiment` — it's built and unit-tested in isolation (`tests/test_ppo_runner.py::test_cache_hit_and_miss`) but appears unintegrated into the real run path.
- **Logging**: `evodevo/evolution/logging.py::RunLogger` — writes `run_log.jsonl` (one line per evaluation) and `run_meta.json` (config, encoder `describe()`, `git rev-parse HEAD` — which, given §0, resolves to the home-directory's commit, not a meaningful project commit — and library versions). Output convention: `{output_dir}/{encoder}/{task}/[{tag}/]seed_{seed}/`.

---

## D. Configuration & Experiment Entry Points

- **Baseline/NCA entry point**: `evodevo/experiments/run_experiment.py` (CLI, single encoder/task/seed run), plus `scripts/run_baseline.sh`, `scripts/run_comparison.sh`, `scripts/run_ablations.sh`.
- **Configs** — all 4 yamls under `evodevo/experiments/configs/` are numerically identical except `encoder`/`task`:
```yaml
seed: 0
population_size: 15
offspring_size: 15
eval_budget: 300
ppo_timesteps: 5000
ppo_eval_episodes: 1
grid_shape: [5, 5]
output_dir: results/runs
```
  `evodevo/experiments/fairness.py::assert_comparable()` mechanically enforces equality of task, eval_budget, ppo_timesteps, ppo_eval_episodes, grid_shape, population_size, offspring_size, sentinel_fitness, cache_enabled — genuinely enforced, not just aspirational, and `run_comparison.sh` calls it before running direct vs nca.
- **Rule-Based entry point**: `examples/ga/run_developmental.py` (argparse CLI: `exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores, developmental_iterations`). The *executed* comparison documented in `FINAL_THREE_WAY_COMPARISON_REPORT.md` used: `pop_size=5, max_evaluations=20, structure_shape=(5,5), env_name=Walker-v0, num_cores=3`, PPO `total_timesteps=300, n_steps=64, batch_size=16, n_epochs=2, eval_interval=150, n_evals=1`, `developmental_iterations=3`, seeds 0-4.

**Parameters that differ but should be identical for fair comparison:**

| Parameter | evodevo (Direct/NCA) | Rule-Based (executed run) |
|---|---|---|
| population_size | 15 | 5 |
| eval_budget / max_evaluations | 300 | 20 |
| ppo_timesteps | 5000 | 300 |
| seeds | 0 (single per yaml; scripts loop 0-4) | 0-4 |
| task/env | Walker-v0 / UpStepper-v0 | Walker-v0 only |
| PPO hyperparams | SB3 library defaults | explicitly tuned (n_steps=64, batch_size=16, n_epochs=2, clip_range not default) |
| invalid-genome policy | consumes eval budget, sentinel fitness -10.0 | retried until valid/unique or attempt cap; does not consume max_evaluations |
| reproduction operators | mutation-only (both Direct and NCA) | mutation + crossover |

Population is 3× smaller, eval budget 15× smaller, PPO timesteps ~17× smaller in the Rule-Based side's already-executed comparison than in evodevo's configs — these are two separate experiments run under two different budgets by two different codebases, not one apples-to-apples comparison.

---

## E. Visualization / Rendering

- `evodevo/experiments/nca_sanity.py::_plot_trajectories` — matplotlib, explicit `matplotlib.use("Agg")` set headless before `pyplot` import, grid of per-timestep body snapshots (`tab10` colormap over materials) → `results/nca_sanity_trajectories.png` (confirmed present on disk).
- `evodevo/analysis/plots.py::plot_fitness_curves` / `plot_morphology_comparison` — fitness curves and box plots; **no explicit `Agg` backend set in this file itself** — relies on `nca_sanity.py`'s earlier import-time side effect or the environment default. This is a latent fragility if this module is ever imported standalone in a GUI/interactive environment.
- `evodevo/analysis/ablation_report.py::_save_fitness_curve_figure` → `ablation_a1_dev_steps_fitness.png` / `_a2_hidden_channels_` / `_e3_second_task_` (skipped if no runs exist yet).
- No `imageio`/gif/video export code anywhere under `evodevo/`; `imageio` is not in `requirements.txt` — evodevo has no animated-render capability, static PNGs only.
- Rule-Based side: `examples/make_gifs.py`, `examples/visualize.py`, plus EvoGym's own `evogym/viewer.py`/`render_mode` support (`human`/`img` per its README). These belong to the vendored EvoGym library/examples, not to the rule-based encoding's own analysis code. `imageio==2.14.1`, `pygifsicle==1.0.5`, `opencv-python`, `pyopengl` are declared in its `environment.yml`.
- Whether the Rule-Based side's gif/video pipeline currently works could not be determined from static reading alone — no output artifacts (gif/video files) were found on disk to confirm it has actually run successfully.

---

## F. Results / Outputs Already Produced

- `results/nca_sanity.json` + `.md` + `results/nca_sanity_trajectories.png` — E0 sanity run, real artifacts (N=2000, seed 0).
- `results/ablation_priors/{E3_nca,H0,H3,T1,T5,T10,T20}/nca_sanity.{json,md}` — PPO-free prior-characterization runs for ablation cells (N=500 each), real artifacts.
- `results/ablations.md` — ablation report; per `docs/decisions.md` the fitness half is "not yet run," only priors are populated.
- `results/runs/direct/Walker-v0/seed_0/` — **complete** run (`best_body.npz`, `run_log.jsonl`, `run_meta.json`).
- `results/runs/direct/Walker-v0/seed_1/` — **incomplete**: only `run_log.jsonl` present, no `run_meta.json`/`best_body.npz`.
- **No `nca` runs exist anywhere under `results/runs/`** — NCA has been prior-sanity-checked but not run end-to-end through the real EA+PPO pipeline in this snapshot.
- Rule-Based side: `evogym-rule-based-encoding/saved_data/test_ga/` (generations 0-3: `controller/*.zip`, `structure/*.npz`, `output.txt`, `metadata.txt`) is present — a small smoke-test GA run.
- `FINAL_THREE_WAY_COMPARISON_REPORT.md` references `saved_data/baseline_seed{0-4}/`, `saved_data/multiseed_seed{0-4}/`, `saved_data/developmental_multiseed_seed{0-4}/`, and `saved_data/final_three_way_comparison/` (with 6 JSON summaries + 15 PNGs) — **none of these directories were found on disk** in this checkout, only `test_ga/` exists. **This means the report's underlying numbers cannot currently be independently re-verified from files present in this repository** — flagged explicitly as unverifiable, not assumed missing due to deletion vs. never-committed vs. different environment.

---

## G. Tests

Main repo `tests/` (1069 lines, 11 files):
- `test_ablation_report.py`, `test_config.py`, `test_direct_encoder.py`
- `test_ea.py` — covers `run_ea`: completion, exact-stop-at-budget, byte-identical logs under same seed, monotonic best-so-far, always-invalid-encoder termination
- `test_fairness.py`, `test_metrics.py`, `test_nca_encoder.py`, `test_nca_sanity.py`
- `test_ppo_runner.py` — covers the surrogate path fully + cache hit/miss + `make_evaluator`; the **real**-PPO path (`evaluate_body`) is covered only by one `@pytest.mark.slow test_real_ppo_smoke`, skipped unless the compiled evogym simulator is built — real PPO correctness is not exercised in ordinary runs
- `test_run_experiment.py`, `test_seeding.py`
- `test_validate.py` — 10 functions covering all `validate_body` branches, plus 2 cross-checks against EvoGym's own `is_connected`/`has_actuator`

**Coverage gaps**: `validate.py` and `ea.py` have solid direct unit tests. `ppo_runner.py`'s real PPO path is effectively untested in a normal run (gated behind `slow` + compiled simulator).

Rule-Based side: `examples/utils/test_rule_encoding.py`, `test_developmental_rules.py`, `examples/ga/test_run_developmental.py` — its own report claims "141 passed" for these combined; not independently re-run in this inspection.

**No test anywhere calls into both `evodevo` and `evogym-rule-based-encoding` together** — there is no cross-repo integration test, consistent with them being unintegrated codebases.

---

## H. Naming & Structure Inconsistencies

- Material naming: evodevo's `MATERIALS = {"EMPTY":0,"RIGID":1,"SOFT":2,"H_ACT":3,"V_ACT":4}` (`validate.py`) vs Rule-Based's `VoxelState.{EMPTY,SOFT,HARD,ACTUATOR_HORIZONTAL,ACTUATOR_VERTICAL}` — same numeric vocabulary, different label names, independently defined (not shared/imported).
- `_largest_component_size` is **duplicated verbatim** (same ~20-line flood-fill) in both `evodevo/morphology/validate.py` and `evodevo/morphology/metrics.py` — an in-repo, single-codebase duplication trivially shareable but currently not.
- Genome class naming: `DirectEncoder`/`NCAEncoder` (evodevo, both subclass a common `Encoder` ABC) vs `RuleGenome`/`DevelopmentalRuleGenome` (Rule-Based, neither implements the `Encoder` interface — no relationship between the two type hierarchies at all).
- No circular imports detected in files read; evodevo's dependency direction is strictly `experiments → evolution/evaluation → encoders/morphology`, partly enforced by design docstrings (e.g. `ea.py` forbidding import of the physics simulator).
- Likely-orphaned/unintegrated code: `evodevo/evaluation/cache.py::FitnessCache` — implemented and unit-tested in isolation, `cache_enabled` config flag exists, but no call site wires it into `run_ea`/`run_experiment` — has zero effect on any real run currently.
- `README.md` (main repo root) is stale, claiming "no encoder/EA/PPO logic implemented yet" despite a fully working `evodevo/` pipeline, populated `results/`, and 11 test files.

---

## I. Dependency & Environment State

- **Main repo** `requirements.txt`: `numpy<2.0.0, gymnasium, stable-baselines3, pyyaml, pytest, pandas, matplotlib`, plus a comment that `evogym` must be installed separately from source (the vendored `evogym/` directory). No Python version pin found anywhere in the main repo; no separate `pyproject.toml`/`environment.yml` at the top level.
- **Vendored `evogym/`** (the dependency copy, at repo root): has its own `environment.yml`, `requirements.txt`/`-dev`/`-working`, `pyproject.toml`, `setup.py` — standard EvoGym library packaging, supports Python 3.7-3.10 per its README.
- **`evogym-rule-based-encoding/environment.yml`**: pins `python=3.7.11`, `gym==0.22.0` (pre-gymnasium fork, **not** `gymnasium`), `numpy==1.21.5`, `torch==1.10.2`, `matplotlib==3.5.1`, git-pinned forks of `neat-python`/`GPyOpt`, plus `pygifsicle`, `opencv-python`, `pyopengl(-accelerate)`, `h5py`, `ttkbootstrap`.
- **`evogym-rule-based-encoding/requirements.txt`** (separate, shorter file): lists `neat-python`, `GPyOpt`, `stable-baselines3`, `imageio`, `pygifsicle` — **inconsistent with its own `environment.yml`** (uses `stable-baselines3`, absent from `environment.yml`; doesn't pin `gym==0.22.0`/`numpy==1.21.5`).
- **Cross-project conflict**: main repo uses `gymnasium` + loose `numpy<2.0.0`; Rule-Based uses old `gym==0.22.0` + pinned `numpy==1.21.5` — different, likely incompatible environments (different Gym APIs entirely). Strong evidence the two codebases were never meant to run in the same Python environment.
- **Python version**: main repo unconstrained; Rule-Based pins exactly `3.7.11`; the actual installed local `.venv` found in the repo is Python 3.10 — the active venv doesn't even match the Rule-Based side's own pin, another sign these were never run together.

---

## J. Gaps Against a Fair Three-Way Comparison

1. **No shared codebase/interface**: Rule-Based does not implement `evodevo.encoders.base.Encoder`; it cannot be dropped into `evodevo/encoders/__init__.py`'s registry or run through `evodevo/evolution/ea.py`/`evodevo/evaluation/ppo_runner.py` as-is. An adapter class (e.g. `RuleBasedEncoder(Encoder)` wrapping `developmental_rule_encoding.py`) would be needed before a true one-pipeline 3-way run is possible.
2. **Environment/dependency incompatibility**: `gym==0.22.0`/`numpy==1.21.5`/`python=3.7.11` (Rule-Based) vs `gymnasium`/`numpy<2.0.0` (evodevo) cannot currently be installed into one interpreter without resolving the gym→gymnasium breaking API change that `ppo_runner.py` already assumes.
3. **Wildly different already-executed budgets** (see table in §D): population 15 vs 5, eval_budget 300 vs 20, ppo_timesteps 5000 vs 300, SB3 defaults vs tuned hyperparameters. Neither existing dataset can be merged into a single fair comparison without rerunning at least one side under matched settings.
4. **Different invalid-genome policies**: evodevo consumes eval budget on invalid genomes with sentinel fitness -10.0; Rule-Based's GA retries until a valid+unique child is found and only counts accepted children against `max_evaluations` — a structurally different search-cost accounting that alone can explain apparent "search efficiency" differences independent of representation.
5. **NCA has no completed real-PPO runs** in this snapshot (`results/runs/` has zero `nca/` entries) — the Baseline-vs-NCA comparison evodevo is built for has not been executed end-to-end even within evodevo's own codebase; only priors exist for NCA.
6. **Rule-Based's own report's supporting data files are missing from this checkout** (`saved_data/baseline_seed*`, `multiseed_seed*`, `developmental_multiseed_seed*`, `final_three_way_comparison/`) — the report text is present but its numbers are not independently re-verifiable from files on disk here.
7. **Reproduction-operator confound acknowledged but unresolved**: Rule-Based's own report flags that its baseline is mutation-only while both rule GA variants use crossover+mutation; the same asymmetry class exists between evodevo's Direct/NCA (mutation-only, both) and any future Rule-Based-in-evodevo arm if it keeps crossover — needs an explicit decision (disable crossover for parity, or accept/document the confound as Rule-Based's own report chose to do).
8. **No cross-repo analysis script**: `evodevo/analysis/aggregate.py`/`plots.py`/`ablation_report.py` only understand evodevo's own `run_log.jsonl`/`run_meta.json` layout; nothing ingests Rule-Based's `saved_data/.../metrics.json`/`output.txt` format into the same schema. A genuine 3-way analysis/plotting step needs either a new loader for the Rule-Based log format, or the Rule-Based arm run *through* evodevo's pipeline (requires gap #1's adapter).
9. **`docs/decisions.md` still has rows marked OPEN** (neighbourhood type, hidden-channel count, dev-steps count, initial-state policy, project-level invalid-genome policy, mutation-tuning procedure) — per the doc's own header, no row may remain OPEN in the final submission, so the project's own gating criterion for calling results final is not yet met, independent of the Rule-Based integration question.

---

## Summary for reorganization planning

Prioritized structural problems to resolve before a clean, unified, three-way comparative pipeline is possible:

1. **Bring the whole project under one real git repo.** Currently the "main repo" has zero commits and isn't tracked at all (the tracked repo root is accidentally the home directory), while `evogym-rule-based-encoding/` is its own separate git history. Any reorg must first establish version control at the intended project root before merging anything.
2. **Decide the integration mechanism for the Rule-Based code** (submodule vs subtree merge vs port-in-place vs adapter-only) — right now it's a bare nested checkout with no relationship to the outer project.
3. **Write a `RuleBasedEncoder` adapter conforming to `evodevo.encoders.base.Encoder`** so all three arms can run through one `ea.py`/`ppo_runner.py`, or explicitly decide the three arms will remain separate pipelines with a shared post-hoc analysis format instead.
4. **Resolve the environment conflict**: `gym`+Python 3.7 (Rule-Based) vs `gymnasium`+unconstrained Python (evodevo) cannot coexist in one interpreter as-is; this blocks any shared execution environment regardless of the adapter question.
5. **Re-run at minimum the smaller-budget side (Rule-Based) at matched population/eval-budget/PPO-timestep settings** — the existing "final" comparison numbers were produced under a 3×/15×/17× smaller budget than evodevo's own Direct/NCA configs and cannot be merged as-is.
6. **Standardize invalid-genome accounting** across all three arms (does an invalid/duplicate genome consume search budget or not) before comparing sample efficiency.
7. **Complete NCA's real-PPO run** — it currently has priors only, no `results/runs/nca/*` data, so even the two-way evodevo comparison this project is built for is incomplete.
8. **Locate or regenerate the Rule-Based report's missing `saved_data/` artifacts**, or mark its published numbers as provisional/unverifiable until the underlying files are found.
9. **Build a common result-loader/schema** so `evodevo/analysis/*` can ingest Rule-Based's log format (or vice versa) for a single cross-approach plot/table.
10. **Wire up or remove `FitnessCache`** (currently dead-in-practice) and **fix the stale root `README.md`** and **close the OPEN rows in `docs/decisions.md`** as lower-priority cleanup once the above structural issues are settled.
