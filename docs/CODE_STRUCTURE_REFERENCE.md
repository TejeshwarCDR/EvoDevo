# EvoGym_Project — Code Structure Reference

This document is produced entirely from direct inspection of the source tree at
`/Users/tejeshwarcdr/Academics/FYP/EvoGym_Project/.claude/worktrees/agent-a0a83c152ec67c94c`
(a git worktree of the repo, branch `main`) on 2026-09-09. Every claim below cites
an exact file path and, where practical, a line number, function, or class name.
Where something claimed by other docs in this repo (e.g. `docs/evodevo-finalization-prompts.md`)
does not actually exist on disk, that discrepancy is flagged explicitly rather than assumed.

---

## PART A — REPOSITORY STRUCTURE

### A.1 Annotated directory tree

```
.
├── Makefile                          # make targets: test, test-core, test-<method>, inspect-<method>, benchmark, docs, clean
├── README.md, CONTRIBUTING.md        # top-level project docs (not modified by this reference)
├── REPO_INVENTORY_REPORT.md          # a pre-existing repo inventory doc (not authored by this task)
├── pytest.ini                        # pytest configuration
├── requirements.txt                  # pinned/loose runtime deps; evogym explicitly NOT listed (installed separately, see Part D.5)
├── docs/                             # project documentation (this file lives here); pre-existing docs untouched
├── evodevo/                          # the project's own Python package — see Part B for full breakdown
│   ├── core/                         # shared constants/config/RNG spine (config.py, materials.py, seeding.py)
│   ├── encoders/                     # the Encoder ABC, registry, and the three encoding implementations
│   │   ├── direct/                   # DirectEncoder: flat categorical genome, one gene per voxel
│   │   ├── nca/                      # NCAEncoder: Neural Cellular Automata developmental encoder
│   │   └── rule_based/               # RuleBasedEncoder + ported V2 rule language (rule_types/rule_encoding/interpreter)
│   ├── morphology/                   # native (evogym-free) body validation, connectivity, and descriptive metrics
│   ├── evaluation/                   # the one fitness evaluator allowed to import evogym/stable_baselines3 (ppo_runner.py), plus FitnessCache
│   ├── evolution/                    # the encoding-agnostic (mu+lambda) EA loop, run logging, and stdout progress reporting
│   ├── experiments/                  # CLI entry points: single-run, N-way benchmark, fairness check, encoder inspection
│   │   └── configs/                  # YAML configs, split into shared/base.yaml + per-method overrides + benchmark.yaml
│   ├── analysis/                     # offline post-hoc analysis: aggregate run logs into DataFrames, statistics, plots, markdown reports
│   └── visualization/                # environment sanity checks + rendering (rollout GIF/MP4) + static body/trajectory plotting
├── scripts/                          # thin shell wrappers around `python -m evodevo...` invocations, one dir per method + benchmark/
├── tests/                            # pytest suite, directory-mirrored to evodevo/ (see Part F)
│   └── integration/                  # EXISTS but is EMPTY (no files) — flagged as a finding, see A.1 note below
└── results/                          # run outputs, ablation priors, archived results, rendered media (data, not code)
```

**Note on `tests/integration/`**: `find` confirms this directory exists but contains zero files (not even `__init__.py`). There is no integration-test suite distinct from the directory-mirrored unit/contract tests described in Part F.

**Note on `external/`**: neither `external/evogym/` nor `external/rule-based-reference/` exist anywhere under this repo checkout (confirmed by `find . -maxdepth 2 -iname "external*"` returning nothing). Multiple docs (`docs/evodevo-finalization-prompts.md:93-94,198,206,223,227`, `README.md:26`, `docs/decisions.md:18,21`, `docs/READY_TO_RUN.md:30`) reference `external/evogym/` and `external/rule-based-reference/` as if they are present, vendored checkouts. On this machine/worktree they are **not present** — see Part D.5 for where evogym is actually installed from instead. `evodevo/visualization/check.py:10,60,67,78,106` also instructs users to `pip install -e external/evogym` and points at `external/evogym/README.md`, which likewise does not exist in this checkout.

### A.2 Entry points

No `pyproject.toml` or `setup.py` exists in the repo (confirmed: `find`/`cat` both empty), so there are **no `console_scripts` entry points** — every invocation is `python -m evodevo.<pkg>.<module>` or a wrapper shell script.

| Command | Defined in | When to use |
|---|---|---|
| `python -m evodevo.experiments.run_experiment --config PATH --seed N [--surrogate] [--overwrite] [--encoder-kwarg K=V ...] [--tag T]` | `evodevo/experiments/run_experiment.py:114-159` (`main`) | Run a single (encoder, task, seed) EA experiment end-to-end. |
| `python -m evodevo.experiments.benchmark --config PATH [--dry-run] [--surrogate]` | `evodevo/experiments/benchmark.py:282-306` (`main`) | Run/resume the full N-way (encoder × task × seed) benchmark matrix defined by a `BenchmarkConfig` YAML; `--dry-run` prints the run matrix and a wall-clock estimate without running anything. |
| `python -m evodevo.experiments.inspect_encoder --encoder NAME [--seed N] [--n N] [--output-dir DIR] [--compare-to NAME]` | `evodevo/experiments/inspect_encoder.py:374-402` (`main`) | PPO-free characterization of a single encoder's prior over morphologies (validity rate, degeneracy, unique bodies). |
| `python -m evodevo.experiments.nca_sanity --seed N --n N --output-dir DIR` | `evodevo/experiments/nca_sanity.py:54-71` (`main`) | **Deprecated** thin shim (see module docstring, lines 1-13) over `inspect_encoder`, kept only so old results stay reproducible. |
| `python -m evodevo.analysis.report --root DIR --output PATH --figures-dir DIR` | `evodevo/analysis/report.py:279-291` (`main`) | Generate `results/BENCHMARK_REPORT.md` (and figures) from a tree of completed run directories. |
| `python -m evodevo.analysis.ablation_report` | `evodevo/analysis/ablation_report.py:291-296` (`main`) | Build the ablation-priors report from `results/runs` + `results/ablation_priors/*`. |
| `python -m evodevo.visualization.check` | `evodevo/visualization/check.py:197-223` (`main`) | Environment sanity check: is evogym importable, is its compiled simulator present, can an env actually step and render a frame, does gif/mp4 writing work. |
| `python -m evodevo.visualization.render_results ...` | `evodevo/visualization/render_results.py:206-236` (`main`) | Render rollout GIF/MP4 for the best body of one or more completed runs. |

Shell wrappers (all under `scripts/`, each a thin `python -m ...` invocation per the Makefile comments):
- `scripts/direct/{run,run_baseline,test,inspect}.sh`, `scripts/nca/{run,test,inspect}.sh`, `scripts/rule_based/{run,test,inspect}.sh` — per-method run/test/inspect wrappers.
- `scripts/benchmark/{run_comparison,run_ablations,render_results}.sh` — orchestrator-level wrappers.

Makefile targets (`Makefile:1-79`): `test` (full suite), `test-core` (shared-core only, `-m "not slow"`), `test-direct`/`test-nca`/`test-rule-based` (delegate to `scripts/<method>/test.sh`), `inspect-direct`/`inspect-nca`/`inspect-rule-based` (delegate to `scripts/<method>/inspect.sh`), `benchmark` (runs `evodevo.experiments.benchmark` then `evodevo.analysis.report`; `SURROGATE=1` passes `--surrogate`), `docs` (checks a fixed doc-file list exists), `clean` (removes `__pycache__`/`.pytest_cache`/egg-info, explicitly never touching `external/`, `.venv/`, or `results/` per `Makefile:74-79`, even though `external/` does not currently exist — see A.1).

### A.3 Config resolution

Configs live at `evodevo/experiments/configs/shared/base.yaml` (the fields every arm must share) and `evodevo/experiments/configs/{direct,nca,rule_based}/{walker,upstepper}.yaml` (per-method, per-task overrides), plus `evodevo/experiments/configs/benchmark.yaml` (the top-level N-way run-matrix config).

Verified against actual loader code, `evodevo/core/config.py`:

- `ExperimentConfig` (`evodevo/core/config.py:39-136`) is a **frozen dataclass**. `SHARED_BASE_KEYS` (`config.py:18-32`) is a frozenset of exactly the keys `configs/shared/base.yaml` is allowed/required to define: `population_size, offspring_size, eval_budget, ppo_timesteps, ppo_eval_episodes, grid_shape, sentinel_fitness, cache_enabled, min_voxels, min_actuators, use_crossover`.
- `ExperimentConfig.from_yaml(path)` (`config.py:63-72`): loads one fully self-contained YAML file with `yaml.safe_load`, converts `grid_shape` to a tuple if present, and constructs the dataclass directly — no merging.
- `ExperimentConfig.from_base_and_override(base_path, override_path)` (`config.py:74-103`): loads both YAML files independently, computes `conflicting = sorted(k for k in override_data if k in base_data)`, and **raises `ConfigOverrideError`** (`config.py:35-36`) if the override file redefines *any* key already in the base file — this is enforced structurally, not just checked after the fact. If no conflict, merges via plain dict spread `{**base_data, **override_data}` (override wins on non-conflicting keys, though by construction there should be none), converts `grid_shape` to tuple, and constructs `ExperimentConfig(**merged)`.
- `ExperimentConfig.from_method_yaml(path, base_path=None)` (`config.py:105-127`): the actual resolution entry point used by real runs. If `base_path` is not given, it derives `configs_dir/shared/base.yaml` from `path`'s grandparent directory (`os.path.dirname(os.path.dirname(...))`) — i.e. for `evodevo/experiments/configs/direct/walker.yaml`, the derived base path is `evodevo/experiments/configs/shared/base.yaml`. If that derived (or an explicit) base path does not exist, and no explicit base was passed, it falls back to `from_yaml(path)` (self-contained config, e.g. ad hoc test fixtures); an *explicitly passed* nonexistent `base_path` is **not** silently ignored — it goes on to call `from_base_and_override` which will raise `FileNotFoundError` opening it.
- `evodevo/experiments/run_experiment.py:61` calls `ExperimentConfig.from_method_yaml(config_path)` to load, then overrides `seed` from the CLI `--seed` argument (`run_experiment.py:62-66`) — every other field comes verbatim from the resolved YAML.
- `evodevo/experiments/benchmark.py:88-99` (`resolve_configs`) calls `ExperimentConfig.from_method_yaml` once per `(encoder, task)` pair using `BenchmarkConfig.config_path(encoder, task)` (`benchmark.py:62-63`, which builds `{configs_root}/{encoder}/{task_stem}.yaml`), then overwrites `seed` to `bconfig.seeds[0]` (a "representative" seed) via `dataclasses.replace`. `check_fairness` (`benchmark.py:102-110`) then calls `assert_comparable` (`evodevo/experiments/fairness.py:31-49`) pairwise across encoders for the *same* task, **before any run starts** — `assert_comparable` requires `task` plus every field in `SHARED_BASE_KEYS` (imported directly from `core.config`, `fairness.py:12,24`) to be identical between the two configs, raising `FairnessViolation` (`fairness.py:27-28`) otherwise.
- For actual execution, `benchmark.py:118-127` (`_run_config_for`) takes the resolved `(encoder, task)` config and does `dataclasses.replace(..., seed=seed, output_dir=bconfig.output_dir)` — i.e. the benchmark orchestrator overrides both `seed` and `output_dir` per run, and a per-method config's own `output_dir` field is ignored when run under the benchmark (only used when that config is run directly via `run_experiment.py`, per the comment at `benchmark.py:124-126`). The fully-resolved per-run config is then written to `results/runs/_resolved_configs/<encoder>/<task_stem>/seed_<N>.yaml` (`benchmark.py:130-137`, `_resolved_config_path`) and `run_experiment.py` is invoked against *that* file path (never against the original `configs/<method>/*.yaml`), because `run_experiment.py` always reloads its config from a path (`run_experiment.py:44-61`).

---

## PART B — CORE MODULE-BY-MODULE BREAKDOWN

### B.1 `evodevo/core/`

**`config.py`** — see Part A.3 for the full merge-logic breakdown. Public surface: `SHARED_BASE_KEYS: frozenset[str]` (line 18), `ConfigOverrideError(ValueError)` (line 35), `ExperimentConfig` dataclass (line 39) with fields `seed: int, encoder: str, task: str, population_size: int, offspring_size: int, eval_budget: int, ppo_timesteps: int, ppo_eval_episodes: int, output_dir: str, grid_shape: tuple = (5,5), sentinel_fitness: float = -10.0, cache_enabled: bool = False, use_crossover: bool = False, min_voxels: int = 3, min_actuators: int = 1`, plus classmethods `from_yaml`, `from_base_and_override`, `from_method_yaml`, and instance methods `to_dict() -> dict[str, Any]` (line 129) and `to_yaml(path) -> None` (line 134). Imports: `dataclasses`, `os`, `yaml` — **no evogym/gym/gymnasium/stable_baselines3/torch imports** ("none"). Imported by: `evodevo/experiments/fairness.py`, `evodevo/experiments/benchmark.py`, `evodevo/experiments/run_experiment.py`, and 9 test files (see grep results — every experiments/config test).

**`materials.py`** — single authoritative material vocabulary (module docstring, lines 1-13, explicitly states it replaces three previously-independent redefinitions). Exact constants: `EMPTY = 0` (line 16), `RIGID = 1` (17), `SOFT = 2` (18), `H_ACT = 3` (19), `V_ACT = 4` (20), `N_MATERIALS = 5` (22), `MATERIAL_NAMES = ("EMPTY","RIGID","SOFT","H_ACT","V_ACT")` (24), `ACTUATORS = frozenset({H_ACT, V_ACT})` (26), `MATERIALS = {"EMPTY":0,"RIGID":1,"SOFT":2,"H_ACT":3,"V_ACT":4}` (29-35). Docstring explicitly notes `FIXED=5` (EvoGym's terrain-voxel id) is deliberately excluded. No imports beyond `from __future__ import annotations`. Imported by 9 files across `encoders/direct`, `encoders/nca`, `encoders/rule_based/{interpreter,rule_types}`, `morphology/{metrics,validate}`, `experiments/inspect_encoder.py`, `visualization/{morphology,development}.py`, plus tests.

**`seeding.py`** — the project's declared single source of randomness (docstring line 1: "All randomness in this project must flow from these two functions"). `make_rng(seed: int) -> np.random.Generator` (line 9-10): thin wrapper on `np.random.default_rng(seed)`. `seed_everything(seed: int) -> None` (line 13-23): calls `random.seed(seed)`, `np.random.seed(seed)`, then tries `import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if available`, swallowing `ImportError` if torch is absent. This is the **only file in `core/` that imports torch**, and only conditionally/lazily inside the function body (line 17), never at module level. Imported by `evodevo/experiments/inspect_encoder.py` (`make_rng`) and `tests/core/test_seeding.py`. See Part E.3 — `seed_everything` itself is **not called anywhere** in the non-test source tree (grep confirms no call site outside `tests/core/test_seeding.py`), which is a finding: the "single spine" function for full-stack seeding (numpy + random + torch) appears to be dead code in production paths; the actual EA/encoder RNG flow instead uses `make_rng`/`np.random.default_rng` directly (see Part E.3).

### B.2 `evodevo/encoders/base.py` — the `Encoder` ABC

Full method list, verified by reading the entire file (109 lines):
- `name -> str` (abstract property, line 28-31)
- `genome_length() -> int` (abstract method, line 33-35)
- `genome_dtype -> Any` (abstract property, line 37-46): docstring requires a numpy dtype for fixed-shape numeric genomes, or Python `object` for variable-length non-numeric genomes (rule_based).
- `random_genome(rng: np.random.Generator) -> np.ndarray` (abstract, line 48-50)
- `decode(genome: np.ndarray) -> np.ndarray` (abstract, line 52-55): docstring mandates return of a `(5,5)` int material-ID array with **no validity handling**.
- `mutate(genome: np.ndarray, rng: np.random.Generator) -> np.ndarray` (abstract, line 57-60): "Must not loop until a valid body results."
- `supports_crossover() -> bool` (concrete, line 62-67): defaults `False`.
- `crossover(parent_a, parent_b, rng) -> ...` (concrete, line 69-77): default raises `NotImplementedError`.
- `serialize_genome(genome) -> Any` (abstract, line 79-85)
- `deserialize_genome(obj: Any) -> ...` (abstract, line 87-92): docstring requires exact round-trip.
- `describe() -> dict` (abstract, line 94-97): JSON-serialisable hyperparameters for run logs.
- `develop_trajectory(genome) -> Any` (concrete, line 99-108): defaults to `None`; for developmental encoders, returns the full t=0..T body sequence.

Imports: `abc.ABC/abstractmethod`, `typing.Any`, `numpy` — **none of evogym/gym/gymnasium/stable_baselines3/torch**. Docstring (lines 1-18) explicitly states the contract is designed so a new encoder never needs to touch `registry.py`, `evolution/ea.py`, `evaluation/ppo_runner.py`, or `morphology/`. Imported by `registry.py`, `encoders/__init__.py`, all three concrete encoders, `evolution/ea.py` (type hint only), `visualization/development.py`, and 2 test files.

### B.3 `evodevo/encoders/registry.py`

Explicit (non-dynamic) plugin registry, 37 lines. `_REGISTRY: Dict[str, Type[Encoder]]` module-level dict (line 14). `register_encoder(name: str, cls: Type[Encoder]) -> None` (line 17-25): raises `TypeError` if `cls` isn't an `Encoder` subclass, raises `ValueError` if `name` is already registered to a *different* class (idempotent re-registration to the same class is allowed). `get_encoder(name: str, **kwargs) -> Encoder` (line 28-33): looks up and instantiates, raising `ValueError` listing available names on miss. `list_encoders() -> list` (line 36-37): `sorted(_REGISTRY)`. Docstring (lines 1-7) states registration must be explicit via imports in `encoders/__init__.py`, never filesystem scanning, so a broken encoder fails loudly at import time. Imports only `evodevo.encoders.base.Encoder` — none of evogym/gym/torch/sb3. Registration actually happens in `evodevo/encoders/__init__.py:10-12`: `register_encoder("direct", DirectEncoder)`, `register_encoder("nca", NCAEncoder)`, `register_encoder("rule_based", RuleBasedEncoder)`.

### B.4 `evodevo/encoders/direct/` (`encoder.py`, 56 lines)

`DirectEncoder(Encoder)` — genome is a flat `(25,)` `int64` array, one categorical material choice per voxel (grid 5×5 = 25). Constructor (line 11-16): `grid_shape=(5,5), mutation_rate=0.1, empty_prob=0.6`; precomputes `_material_probs` as `[empty_prob] + [remaining]*4` where `remaining = (1-empty_prob)/(N_MATERIALS-1)`. `name -> "direct"`. `genome_dtype -> np.int64`. `genome_length() -> 25`. `random_genome(rng)` (line 35-36): `rng.choice(N_MATERIALS, size=25, p=_material_probs)`. `decode(genome)` (line 38-39): pure `np.reshape` to `(5,5)`, no arithmetic. `mutate(genome, rng)` (line 41-47): per-gene Bernoulli mask at `mutation_rate`, resamples masked genes from the same categorical distribution. `serialize_genome`/`deserialize_genome`: list round-trip via `np.asarray(...).tolist()`/`np.asarray(obj, dtype=int)`. `describe()`: dict of `name, grid_shape, mutation_rate, empty_prob`. Imports `evodevo.core.materials.N_MATERIALS`, `evodevo.encoders.base.Encoder` — none of evogym/gym/torch/sb3.

### B.5 `evodevo/encoders/rule_based/` (4 files)

**`rule_types.py`** (250 lines) — ported (per header comment lines 1-5) from `external/rule-based-reference/.../developmental_rule_types.py` commit `703e711`. Defines `VoxelState`, `Direction`, `PositionType`, `ConditionType`, `ActionType` Enums (lines 40-89); `GROWABLE_STATES` tuple excluding EMPTY (94-99); `MIN_NEIGHBOR_THRESHOLD=1`, `MAX_NEIGHBOR_THRESHOLD=4` (107-108); the `DevelopmentalRule` dataclass (194-249) with `condition_type, condition_params, action_type, action_params`, `__post_init__` validation via `_validate_condition_params`/`_validate_action_params` (131-192), `.copy()`, `.to_dict()`, `.__repr__`. Pure Python stdlib (`copy.deepcopy`, `dataclasses`, `enum`, `typing`) — no evodevo/evogym/gym/torch/sb3 imports at all.

**`rule_encoding.py`** (348 lines) — ported from `.../developmental_rule_encoding.py`. `MIN_RULES=2`, `MAX_RULES=12`, `MIN_ACTUATORS=2` (lines 41-46). `DevelopmentalRuleGenome` class (49-84): ordered `List[DevelopmentalRule]`, enforces rule-count bounds in `__init__`, has `.copy()`, `__len__`, `__iter__`, `__eq__`. `sample_developmental_rules() -> DevelopmentalRuleGenome` (153-193): builds 2 guaranteed growth rules (seed-cell grow + empty-neighbor-propagate) then fills to a random length in `[3,7]` with growth-biased random rules, shuffles. `generate_robot_from_developmental_rules(rules, shape, iterations)` (196-220): runs `develop_rules` once, checks actuator count ≥ `MIN_ACTUATORS` and `is_connected(body) and has_actuator(body)`; returns `(body, get_full_connectivity(body))` or `None` — **imports `is_connected`/`has_actuator` from `evodevo.morphology.validate`** (line 25) and `get_full_connectivity` from `evodevo.morphology.connectivity` (line 24), i.e. this is where the "native, evogym-free" morphology reimplementations get consumed by the rule-based arm (see Part D.4). Mutation operators (`_mutate_condition_type/_params`, `_mutate_action_type/_params`, `_add_rule`, `_remove_rule`, `_reorder_rules`, lines 227-286) each applied independently with probability `mutation_rate` inside `mutate_developmental_rules` (289-304). Crossover: `crossover_developmental_rules_at(parent1, parent2, cut1, cut2)` (311-338, deterministic one-point with length-repair) and `crossover_developmental_rules` (341-347, random cut points). Uses Python's global `random` module throughout (not `np.random.Generator`) — this is deliberate per the encoder adapter's docstring (see B.5/`encoder.py` below). No evogym/gym/torch/sb3 imports.

**`interpreter.py`** (320 lines) — ported from `.../developmental_interpreter.py`, with one **documented deliberate deviation**: `HARD = RIGID`, `HORIZONTAL = H_ACT`, `VERTICAL = V_ACT` (lines 144-146) are sourced from `evodevo.core.materials`, correcting a pre-existing bug in the original where `SOFT=1/HARD=2` did not match EvoGym's real `RIGID=1/SOFT=2` (full explanation in the module docstring, lines 124-146, and `docs/decisions.md:18`). `develop_rules(genome, shape=(5,5), iterations=DEFAULT_MAX_ITERATIONS, seed_position=None) -> np.ndarray` (249-319) is the core interpreter — see Part C for its full call-by-call walkthrough. `DEFAULT_MAX_ITERATIONS=5`, `HARD_MAX_ITERATIONS=10` (171-172). Helper functions: `_in_bounds`, `_condition_matches` (evaluates `ConditionType.ALWAYS/CELL_TYPE/NEIGHBOR_COUNT/EMPTY_NEIGHBOR/POSITION` against the *current* grid), `_apply_action` (applies `SET_*`/`GROW_DIRECTIONAL`/`COPY_NEIGHBOR`). Imports `evodevo.core.materials` and `evodevo.encoders.rule_based.rule_types` — no evogym/gym/torch/sb3.

**`encoder.py`** (169 lines) — `RuleBasedEncoder(Encoder)`, the **non-ported** adapter (module docstring lines 1-7 explicitly states this). Genome type: `DevelopmentalRuleGenome` (variable-length list of `DevelopmentalRule`, **not** a fixed numpy array); `genome_dtype -> object` (line 101-103, explicitly documented as "not meaningful ... for allocation purposes"); `genome_length() -> MAX_RULES` (12), documented as a nominal upper bound never actually used for allocation. `_seed_global_random(rng)` (62-75): derives an int seed from the caller's `np.random.Generator` via `int(rng.integers(0, 2**31-1))` and calls Python's global `random.seed(seed)` — because the ported V2 operators use the global `random` module internally; the docstring explicitly flags this as safe only because the EA loop is single-threaded/sequential (see Part E.3 for the RNG-flow implication). `random_genome`, `mutate`, `crossover` all call `_seed_global_random(rng)` first, then delegate to the corresponding `rule_encoding` function. `decode(genome)` (116-124): calls `develop_rules` directly (not `generate_robot_from_developmental_rules`, since `decode()` per the ABC contract must never itself signal validity). `develop_trajectory(genome)` (130-140): calls `develop_rules` once per `t` in `range(iterations+1)` (not incremental — restarts from scratch each time, per the code comment). `supports_crossover() -> True` (142-143) — the only encoder that returns `True` here. Imports `evodevo.encoders.base.Encoder`, `evodevo.encoders.rule_based.{interpreter,rule_encoding,rule_types}` — no evogym/gym/torch/sb3.

### B.6 `evodevo/encoders/nca/encoder.py` (184 lines)

`NCAEncoder(Encoder)` — genome is a flat `float64` weight vector for a 2-layer MLP perception network. Constructor (37-73): `grid_shape=(5,5), hidden_channels=3, hidden_width=16, dev_steps=10, neighbourhood="von_neumann" (only supported value, else ValueError), init_scale=0.5, mutation_sigma=0.1, mutation_fraction=1.0`. `channels = N_MATERIALS + hidden_channels` (=8 by default); `input_dim = 5*channels` (own cell + 4 cardinal neighbours). Genome length = `w1_size + b1_size + w2_size + b2_size` where `w1: (input_dim, hidden_width)`, `b1: (hidden_width,)`, `w2: (hidden_width, channels)`, `b2: (channels,)`. `_INIT_STATE_SEED = 20260902`, `_INIT_STATE_SCALE = 0.1` (module-level, lines 26-27) — the fixed, non-evolved initial cell state, generated once via `np.random.default_rng(_INIT_STATE_SEED).normal(...)` (line 30-33) and identical across every individual/generation (documented rationale: symmetric zero-init would prevent any left-right/up-down differentiation under deterministic synchronous updates). `random_genome(rng) -> rng.normal(scale=init_scale, size=genome_length)`. `_unpack(genome)` (95-106): slices the flat vector into `w1,b1,w2,b2`. `_perceive(state)` (108-123): computes 4 cardinally-shifted copies of `state` (zero-padded at boundaries) and concatenates with `state` itself along the channel axis → `(rows,cols,5*channels)`. `_step(state,w1,b1,w2,b2)` (125-131): `hidden=tanh(flat@w1+b1)`, `delta=hidden@w2+b2`, returns `state+delta` (residual, synchronous). `develop(genome, return_trajectory=True)` (133-149): unpacks weights, copies the fixed initial state, applies `_step` `dev_steps` times, and at every step (including t=0) computes `readout(s) = argmax(s[:,:,:N_MATERIALS], axis=-1)` to get the discrete material-ID body for that step. `decode(genome) = develop(genome, True)[-1]` (151-153). `develop_trajectory` = `develop(genome, True)` (155-159). `mutate(genome, rng)` (161-168): picks `round(mutation_fraction*n)` genome indices without replacement and adds `N(0, mutation_sigma)` noise to each. `describe()`: full hyperparameter dict. Imports `evodevo.core.materials.N_MATERIALS`, `evodevo.encoders.base.Encoder` — none of evogym/gym/torch/sb3 (NCA's own forward/backward math is hand-rolled numpy, not a torch network).

### B.7 `evodevo/morphology/` (3 files)

**`validate.py`** (79 lines) — `is_connected(body) -> bool` (14-26): "Native equivalent of `evogym.is_connected`" (explicit docstring claim). Computes `occupied = body != EMPTY`; if zero occupied voxels returns `False` (matches evogym's "no start voxel" behaviour per the docstring); otherwise returns `largest_component_size(occupied) == occupied.sum()`. `has_actuator(body) -> bool` (29-33): "Native equivalent of `evogym.has_actuator`" — `np.isin(body, list(ACTUATORS)).any()`. `ValidationResult` dataclass (36-39): `valid: bool, reasons: list`. `validate_body(body, grid_shape=(5,5), min_voxels=3, min_actuators=1) -> ValidationResult` (42-79): checks shape, integer dtype, value range (⊆ `{0..4}`), min occupied voxels, single-connected-component, and actuator count, accumulating human-readable reason strings (used verbatim by `inspect_encoder.py`'s reason categorizer). Imports `evodevo.core.materials.{ACTUATORS,EMPTY,MATERIALS}` and `evodevo.morphology.connectivity.largest_component_size` — no evogym import (this is a real, native reimplementation, not a thin evogym wrapper — see Part D.4).

**`connectivity.py`** (60 lines) — `get_full_connectivity(robot: np.ndarray) -> np.ndarray` (7-36): "Native equivalent of `evogym.get_full_connectivity`". Iterates the grid in row-major flat order (`for i in range(robot.size): x=i%width; y=i//width`), for each non-empty voxel checks the right neighbour then the down neighbour, appending `[flat_idx, neighbour_flat_idx]` pairs; returns `np.array(out).T` (shape `(2,k)`) or `np.empty((0,2)).T` if no edges. `largest_component_size(occupied: np.ndarray) -> int` (39-59): explicit iterative (stack-based, not recursive) 4-connected flood fill over a boolean grid, used by both `validate.py` and `metrics.py`. No evogym import.

**`metrics.py`** (51 lines) — `body_metrics(body) -> dict` (12-44): computes per-material voxel counts, `actuator_fraction`, `horizontal_symmetry_score`/`vertical_symmetry_score` (fraction of cells equal to their mirror under `[:, ::-1]`/`[::-1, :]`), and `largest_component_size` — this exact dict is spread (`**body_metrics(body)`) into every EA run-log record (see Part E.1). `body_hash(body) -> str` (47-50): `sha256(shape_str + b"|" + dtype_str + b"|" + body.tobytes())`, the canonical body-identity key used by `FitnessCache` and dedup logic throughout. Imports `evodevo.core.materials`, `evodevo.morphology.connectivity.largest_component_size` — no evogym.

### B.8 `evodevo/evaluation/` (2 files)

**`ppo_runner.py`** (168 lines) — module docstring (lines 1-7): "the only module in the project allowed to import stable_baselines3 or evogym environments." `DEFAULT_SENTINEL_FITNESS = -10.0` (20). `EvalResult` dataclass (23-29): `fitness, valid, reason, wall_clock_s, body_hash`. `evaluate_body(body, task, ppo_timesteps, eval_episodes, seed, output_dir=None, sentinel_fitness=-10.0) -> EvalResult` (32-116): see Part D.3 for the full call chain — validates first via `validate_body`, and only on success does it `import evogym.envs` (line 54) and `from evogym import get_full_connectivity` (55) and `from stable_baselines3 import PPO` / `from stable_baselines3.common.env_util import make_vec_env` (56-57), all as **local, deferred imports inside the function body** (not module-level) so importing `ppo_runner` itself never requires evogym/sb3 to be installed unless real evaluation is actually invoked. Any exception during the try block (PPO training/eval) is caught broadly (`except Exception as exc`, line 102) and converted into a sentinel-fitness `EvalResult` with `reason=f"simulation_instability: {type(exc).__name__}: {exc}"` — by explicit design (comment 61-70) so one unstable body cannot abort a multi-hour benchmark or give any arm a budget advantage. `evaluate_body_surrogate(...)` (119-159): PPO-free stand-in, fitness `= h_act_count + v_act_count + horizontal_symmetry_score` from `body_metrics` — explicitly documented as "NOT a scientific result," used only for pipeline testing. `make_evaluator(mode: str) -> Callable` (162-167): returns `evaluate_body` for `"real"`, `evaluate_body_surrogate` for `"surrogate"`, else raises `ValueError`. Imports `evodevo.morphology.{metrics,validate}` at module level; evogym/sb3 only locally inside `evaluate_body` — this is the confirmed architectural boundary (see Part D).

**`cache.py`** (32 lines) — `FitnessCache`: `_store: dict`, `hits`/`misses` counters, `get(body_hash)` (13-18, increments hit/miss), `put(body_hash, result)` (20-21), `__len__`, `hit_rate` property (26-31, `hits/(hits+misses)` or `None` if empty). No evogym/torch/gym imports. Consumed only by `evolution/ea.py`.

### B.9 `evodevo/evolution/` (3 files)

**`ea.py`** (177 lines) — module docstring (lines 1-7): "may only call methods on the Encoder interface and the evaluator callable it is given. It must never import the physics simulator package or the RL training library, nor any specific encoder class" — verified true by reading the full file: imports are `evodevo.encoders.base.Encoder` (type hint), `evodevo.evaluation.cache.FitnessCache`, `evodevo.evolution.logging.RunLogger`, `evodevo.morphology.metrics.{body_hash,body_metrics}`, `numpy`, stdlib — **no evogym/gym/torch/sb3**. `Individual` dataclass (22-28): `genome_id, parent_id, genome, fitness, valid`. `RunResult` dataclass (31-37): `population, evaluations_used, best_fitness, records, cache_hit_rate`. `run_ea(encoder, evaluator, config, logger=None, cache=None) -> RunResult` (40-176): see Part D.6 for the full top-to-bottom chain. Key logic: RNG is `np.random.default_rng(config.seed)` (47); a `(mu+lambda)` scheme where `mu=population_size`, `lam=offspring_size`; `cache_enabled` gates whether a `FitnessCache` is used at all (a cache hit **still consumes one evaluation from budget**, per explicit fairness-preserving comment 52-56); `use_crossover` is validated against `encoder.supports_crossover()` once at startup (73-77), raising `ValueError` on mismatch rather than failing silently mid-run; each generation selects the top-`mu` survivors by fitness (151), then produces `lam` offspring via `encoder.crossover` or `encoder.mutate` depending on `use_crossover`, using `rng.integers` to pick parent(s) from survivors uniformly at random (158-164) — note: **not** fitness-weighted, not tournament selection, pure uniform-random-parent-from-survivors "mu+lambda" truncation selection.

**`logging.py`** (99 lines) — `RunLogger` class: writes newline-delimited JSON to `<output_dir>/run_log.jsonl` (line 23, opened once at construction, appended-to via `log()`), optionally rewrites a shared `.heartbeat.json` file after every evaluation via atomic `os.replace` (44-57), and optionally invokes an `on_eval` callback (purely for stdout progress reporting, never affects persisted content — line 33-34, 41-42). `write_meta(config_dict, encoder_describe, cache_hit_rate=None)` (62-76) writes `run_meta.json` containing `config, encoder, cache_hit_rate, git_commit (via `git rev-parse HEAD`, 79-87), library_versions` (numpy/yaml/pytest only, 90-98). No evogym/torch/sb3 imports.

**`progress.py`** (92 lines) — `ProgressTracker` (22-44): tracks `eval_budgets: Dict[(encoder,task,seed), int]` and cumulative progress across the whole benchmark matrix; `for_run(encoder,task,seed) -> _RunProgress`. `_RunProgress.__call__(record)` (60-91): prints one stdout line per completed evaluation with a rolling-mean ETA (window default 20, `deque(maxlen=window)`), tagging each line `cache hit` / `validity failure` / `real-PPO`. Explicitly documented (module docstring 1-6) as purely additive — never touches persisted JSONL/heartbeat/meta. No evogym/torch/sb3 imports.

### B.10 `evodevo/experiments/`

All four files named in the task exist, plus two extra (`inspect_encoder.py`, `nca_sanity.py`); there is **no `lock_hash.py`** anywhere in the repo (`find`/`grep` confirm) — the fairness/reproducibility "locking" role that name might suggest is instead played jointly by `fairness.py` (pre-run assertion) and `benchmark.py`'s manifest + resolved-config writing (post-resolution, pre-run persistence of the exact config every arm ran under, `benchmark.py:140-172, 217-222`).

- **`run_experiment.py`** (159 lines) — see Part A.3/D.3/D.6. `_compute_connections(body)` (25-33): best-effort `from evogym import get_full_connectivity`, returns `None` on `ImportError` (used only for the saved `best_body.npz` artifact, not for evaluation itself). `run_experiment(config_path, seed, surrogate, overwrite, encoder_kwargs=None, tag=None, progress_callback=None) -> str` (44-103): the single (encoder,task,seed) run driver — loads config, builds output dir (`_run_output_dir`, 36-41), refuses to overwrite a completed run unless `--overwrite`, builds the encoder via `get_encoder`, builds the evaluator via `make_evaluator`, runs `run_ea`, writes meta, and saves the best individual's decoded body + connectivity to `best_body.npz`.
- **`benchmark.py`** (306 lines) — see Part A.3. `BenchmarkConfig` dataclass (47-63) with `encoders, tasks, seeds, configs_root, output_dir, surrogate`. `run_benchmark(bconfig, dry_run=False)` (225-279): fairness check → manifest write → (if not dry-run) execute every `(encoder,task,seed)` in the matrix, **skipping** any already-`DONE_MARKER`-complete run dir (resumable). `estimate_wall_clock` (179-214): measures one real evaluation's cost per encoder and extrapolates total.
- **`fairness.py`** (49 lines) — see Part A.3/B.1.
- **`inspect_encoder.py`** (403 lines) — generalized PPO-free encoder characterization; see summary embedded above (Part A.2 table) and full read above. Imports `evodevo.core.{materials,seeding}`, `evodevo.encoders.{get_encoder, nca.encoder.NCAEncoder}`, `evodevo.morphology.{metrics.body_hash, validate.validate_body}`, `matplotlib` (Agg backend) — no evogym/torch/sb3.
- **`nca_sanity.py`** (71 lines) — deprecated shim, see B.10 note above; issues a `DeprecationWarning` (34-40) and delegates to `inspect_encoder.inspect_encoder(...)`.

### B.11 `evodevo/analysis/`

- **`aggregate.py`** (187 lines) — `load_runs`/`discover_runs`/`load_all_runs` build a tidy per-evaluation `pandas.DataFrame` from `run_log.jsonl` + `run_meta.json` pairs (`_load_single_run`, 30-51, reads every JSONL line as one record, tags each row with `encoder/task/seed/run_dir` from `run_meta.json`'s `config`). `per_seed_finals(df) -> dict` (76-90): `{encoder: [cummax(fitness) at last eval_index, per seed]}` — the canonical "final best fitness per seed" definition shared by both summary tables. `three_way_table(df) -> DataFrame` (93-135): N-way summary (final fitness mean/std, evaluations-to-50%/90%-of-**best-arm's**-final, validity rate, unique bodies, cache hit rate, mean wall-clock). `comparison_table(df, reference_encoder="direct") -> DataFrame` (138-186): same idea but thresholds are relative to one named reference arm instead of the best arm. Module docstring (1-9) explicitly notes rule_based's non-numeric genome never reaches this loader since JSONL records only ever contain body-level scalars/dicts. Imports `pandas`, `numpy`, stdlib — no evogym/torch/sb3.
- **`statistics.py`** (176 lines) — dependency-free (no scipy) pairwise statistics for small-n (≈5 seeds) comparisons, with an explicit, prominent docstring warning (lines 1-17) that no result here should be read as more than "indicative," and that overlapping-CI results mean "no detectable difference," not "equivalent." `cliffs_delta(a,b)` (49-58), `mann_whitney_u(a,b)` (61-82, normal-approximation two-sided p-value), `bootstrap_ci_mean_difference(a,b,n_boot=10000,ci=0.95,seed=0)` (85-105, independent resampling per arm), `compare_two_arms(...) -> PairedComparison` (108-151), `three_way_comparison(values_by_encoder, ...) -> list[PairedComparison]` (154-176, all pairwise combinations). `PairedComparison` dataclass (29-46). No evogym/torch/sb3.
- **`plots.py`** (93 lines, per current diff `git status` shows this file is locally modified) — `plot_fitness_curves(df, ax=None)` (22-48): per-encoder mean±std best-so-far curve across seeds vs. `eval_index`. `plot_morphology_comparison(df, metrics=MORPHOLOGY_METRICS, axes=None)` (55-73): bar comparison of best-body metrics across encoders. `plot_validity_and_efficiency(three_way_df, axes=None)` (74-93): validity-rate and evaluations-to-threshold bars. Uses `matplotlib`; imports `evodevo.analysis.aggregate` helpers. No evogym/torch/sb3.
- **`report.py`** (291 lines, also locally modified per `git status`) — `generate_report(...)` (129-278, main orchestration): loads the benchmark manifest (`_load_manifest`, 40-47, itself importing `evodevo.experiments.benchmark` only for path/constant reuse — this is the one place `analysis/` imports from `experiments/`), aggregates all runs, computes the three-way table and pairwise comparisons, renders figures via `_write_figures` (93-128, calls into `plots.py`), and writes the final `results/BENCHMARK_REPORT.md`-style markdown (`_df_to_markdown`, 48-64; `_headline`, 65-92). `main()` (279-291) is the CLI (`--root`, `--output`, `--figures-dir`).
- **`ablation_report.py`** (296 lines) — `run_ablation_report(root_dir="results/runs", output_dir="results", prior_seed=0)` (131-291): builds the ablation-priors report combining tagged-run fitness summaries (`_load_tagged_runs`, `_fitness_summary`) with PPO-free "prior" characterizations (`_prior_summary`, 71-82, calls `evodevo.experiments.inspect_encoder.inspect_encoder` under the hood based on grep of importers — consistent with `nca_sanity.py`'s note that `ablation_report.py` is an existing caller of the NCA sanity/inspect machinery). No evogym/torch/sb3 imports.

### B.12 `evodevo/visualization/`

All four named files exist, plus one extra (`development.py`).

- **`check.py`** (223 lines) — environment sanity CLI. `check_simulator_importable()` (54-83): checks `importlib.util.find_spec("evogym")` then `"evogym.simulator_cpp"`, then actually does `import evogym` / `import evogym.envs` (71-72) inside a try/except, producing actionable remediation text (`pip install -e external/evogym`, referencing a nonexistent-in-this-checkout path — see A.1). `check_env_creation_and_frame_capture()` (84-128): `import gymnasium as gym` (88), `import evogym.envs`, `from evogym import get_full_connectivity` (89) — actually constructs an env and steps/renders a frame. `check_imageio_gif_and_mp4()` (129-174) and `check_gifsicle_optional()` (175-187): media-writing checks, no evogym. `run_all()`/`main()` (188-223). This module imports evogym/gymnasium **only inside function bodies**, never at module top level.
- **`development.py`** (60 lines) — `plot_development_trajectory(encoder, genome, ax_row=None, title=None)` (26-60): calls `encoder.develop_trajectory(genome)` and renders each step with `evodevo.visualization.morphology`'s body-plotting helpers. Imports `evodevo.core.materials`, `evodevo.encoders.base.Encoder` — no evogym.
- **`morphology.py`** (83 lines) — `plot_body(body, ax=None, title=None)` (36-54) and `plot_body_grid(bodies, labels=None, axes=None)` (55-83): static matplotlib rendering of a material-ID grid via a fixed 5-color colormap; `_legend_handles()` (32-35). No evogym.
- **`render.py`** (161 lines) — module docstring (line 8) explicitly states it must never import evogym/stable_baselines3 at module level (mirrors `ppo_runner.py`'s pattern). `_make_env(body, task, seed=0)` (34-59): `import evogym.envs` (50), `from evogym import get_full_connectivity` (52), `import gymnasium as gym` (51) — all local. `_load_controller(controller_path)` (60-65): `from stable_baselines3 import PPO` (61, local). `render_body_rollout(...)` (66-140): builds env, optionally loads a trained PPO controller, steps the env collecting frames, writes GIF/MP4 via `imageio`. `_maybe_optimize_gif` (141-161): optional `gifsicle` shell-out, degrades gracefully if absent.
- **`render_results.py`** (236 lines) — `_find_best_run`/`_load_best_body` (33-56): locate the best individual across a run tree and load `best_body.npz`. `render_one_arm(...)` (61-155) and `render_comparison_figure(...)` (156-179): orchestrates `render.render_body_rollout` per arm and stitches a side-by-side comparison figure (this is the source of `results/media/comparison_Walker-v0.png` per `git status`). `render_all(...)`/`main()` (180-236). Imports `evodevo.analysis.aggregate`, `evodevo.morphology.validate` — evogym only transitively via `render.py`'s deferred imports, not directly here (confirmed: no `evogym`-literal import line in this file per the earlier grep, only its docstring/skip-marker references it).

---

## PART C — THE THREE ENCODER CLASSES, IN FULL

### C.1 `DirectEncoder`

```
DirectEncoder(Encoder)
├── name -> "direct"
├── genome_dtype -> np.int64
├── genome_length() -> 25                          [grid_shape[0]*grid_shape[1]]
├── random_genome(rng) -> (25,) int array           [categorical draw, empty_prob-weighted]
├── decode(genome) -> (5,5) int array               [pure reshape]
├── mutate(genome, rng) -> (25,) int array          [per-gene Bernoulli resample]
├── serialize_genome / deserialize_genome           [list <-> np.ndarray round trip]
└── describe() -> dict
```
Genome: `np.ndarray`, shape `(25,)`, dtype `int64`, values in `{0,1,2,3,4}`.

**`decode(genome)` walkthrough** (`evodevo/encoders/direct/encoder.py:38-39`):
1. `np.asarray(genome, dtype=int)` — coerce to a plain int array (no-op if already int64).
2. `.reshape(self.grid_shape)` — reshape the flat 25-vector into `(5,5)` in row-major (C) order. No validation, no arithmetic beyond the reshape — this is the entire function.

### C.2 `NCAEncoder`

```
NCAEncoder(Encoder)
├── name -> "nca"
├── genome_dtype -> np.float64
├── genome_length() -> w1_size+b1_size+w2_size+b2_size
├── random_genome(rng) -> flat float64 weight vector   [N(0, init_scale)]
├── _unpack(genome) -> (w1, b1, w2, b2)
├── _perceive(state) -> (rows,cols,5*channels)          [self + 4 cardinal zero-padded shifts]
├── _step(state,w1,b1,w2,b2) -> state + MLP(perceive(state))   [residual]
├── develop(genome, return_trajectory) -> [body_0..body_T]
├── decode(genome) -> develop(...)[-1]
├── develop_trajectory(genome) -> develop(genome, True)
├── mutate(genome, rng) -> flat float64 vector          [Gaussian perturbation of a random index subset]
└── describe() -> dict
```
Genome: `np.ndarray`, shape `(w1_size+b1_size+w2_size+b2_size,)` (with defaults `hidden_channels=3, hidden_width=16` → `channels=8, input_dim=40`: `w1=40*16=640, b1=16, w2=16*8=128, b2=8` → genome length 792), dtype `float64`.

**`decode(genome)` walkthrough** (`evodevo/encoders/nca/encoder.py:151-153`, calling into `develop` at 133-149):
1. `_unpack(genome)` (95-106): slice the flat vector sequentially into `w1 (40,16)`, `b1 (16,)`, `w2 (16,8)`, `b2 (8,)`.
2. `state = self._initial_state.copy()` — start from the **fixed** `(5,5,8)` array generated once at construction time from `np.random.default_rng(20260902)` (never re-seeded per genome/individual).
3. `readout(state) = argmax(state[:,:,:5], axis=-1).astype(int)` computed once at `t=0` and appended to `trajectory`.
4. Loop `dev_steps` (default 10) times, each iteration calling `_step`:
   a. `_perceive(state)` (108-123): build `parts = [state]`; for each of the 4 `NEIGHBOUR_OFFSETS` `(-1,0),(1,0),(0,-1),(0,1)`, build a zero-initialized array of `state`'s shape and copy the appropriately-offset, in-bounds slab of `state` into it (out-of-bounds positions stay zero — zero padding); concatenate `[state]+shifted×4` along the last axis → `(5,5,40)`.
   b. `flat = perception.reshape(-1, 40)` — flatten the 25 cells into a batch dimension.
   c. `hidden = tanh(flat @ w1 + b1)` — `(25,16)`.
   d. `delta = (hidden @ w2 + b2).reshape(5,5,8)`.
   e. `state = state + delta` — residual update, applied synchronously (the whole grid's `delta` is computed from the *old* `state` before any cell is updated).
   f. `readout(state)` appended to `trajectory` for this step.
5. After the loop, `decode` returns `trajectory[-1]`, i.e. `argmax` of the first 5 channels of the final state, giving a `(5,5)` int material-ID body. No connectivity/validity check inside `decode`.

### C.3 `RuleBasedEncoder`

```
RuleBasedEncoder(Encoder)
├── name -> "rule_based"
├── genome_dtype -> object                          [not a numpy array]
├── genome_length() -> MAX_RULES (12, nominal only)
├── random_genome(rng) -> DevelopmentalRuleGenome    [seeds global `random` from rng, then sample_developmental_rules()]
├── decode(genome) -> (5,5) int array                [develop_rules(genome, shape, iterations)]
├── mutate(genome, rng) -> DevelopmentalRuleGenome   [seeds global random, mutate_developmental_rules()]
├── crossover(a, b, rng) -> DevelopmentalRuleGenome  [seeds global random, crossover_developmental_rules()]
├── supports_crossover() -> True                     [only encoder to override this]
├── develop_trajectory(genome) -> [body_0..body_iterations]  [re-runs develop_rules once per t, not incremental]
├── serialize_genome / deserialize_genome            [list-of-dicts <-> DevelopmentalRuleGenome, enum-aware]
└── describe() -> dict
```
Genome: `DevelopmentalRuleGenome`, an ordered Python list of 2–12 `DevelopmentalRule` dataclass instances (not a numeric array).

**`decode(genome)` walkthrough** (`evodevo/encoders/rule_based/encoder.py:116-124`, calling `develop_rules` in `interpreter.py:249-319`):
1. `develop_rules(genome, shape=(5,5), iterations=developmental_iterations)` (default `developmental_iterations = DEFAULT_MAX_ITERATIONS = 5`).
2. Inside `develop_rules`: validate `0 <= iterations <= HARD_MAX_ITERATIONS(10)`, else `ValueError`.
3. Initialize `grid = np.zeros((5,5), dtype=int)`; `seed_position = (rows//2, cols//2) = (2,2)`; set `grid[2][2] = SOFT` (=2).
4. For each of `iterations` (default 5) synchronous update steps:
   a. **Resolution pass** (read-only): for every cell `(i,j)` in row-major order, scan `genome.rules` in list order and take the **first** rule whose `_condition_matches(rule,i,j,grid,shape)` is `True` (checking `ALWAYS`/`CELL_TYPE`(compare to `STATE_TO_VOXEL[state]`)/`NEIGHBOR_COUNT`(count of 4-neighbours that are non-EMPTY in the *current* grid, compared to a threshold)/`EMPTY_NEIGHBOR`(is a specific-direction neighbour EMPTY)/`POSITION`(boundary/interior/top/bottom/left/right)); append `(i,j,rule)` to `decisions` if a match is found, otherwise the cell is skipped this iteration.
   b. `next_grid = grid.copy()`.
   c. **Self-action write pass**: for every `(i,j,rule)` in `decisions` whose `action_type != GROW_DIRECTIONAL`, call `_apply_action`: `SET_EMPTY/SET_SOFT/SET_HARD/SET_ACTUATOR_HORIZONTAL/SET_ACTUATOR_VERTICAL` unconditionally overwrite `next_grid[i][j]`; `COPY_NEIGHBOR(direction)` sets `next_grid[i][j] = grid[si][sj]` (the neighbour's *current*, pre-iteration value) if in bounds, else no-op.
   d. **Growth write pass** (applied after all self-actions, deliberately, so growth cannot be immediately erased by an unrelated cell's same-iteration self-action): for every `(i,j,rule)` in `decisions` with `action_type == GROW_DIRECTIONAL`, compute target `(ti,tj)`; if in-bounds **and** `grid[ti][tj] == EMPTY` (checked against the pre-iteration grid), set `next_grid[ti][tj] = STATE_TO_VOXEL[voxel_type]`; same-target collisions between two growing cells resolve by row-major processing order (last writer wins).
   e. `grid = next_grid`.
5. After all iterations, return `grid` — a `(5,5)` int array using `evodevo.core.materials` codes directly (EMPTY=0, RIGID=1, SOFT=2, H_ACT=3, V_ACT=4). No validity check inside `decode`/`develop_rules` itself.

---

## PART D — THE COMPLETE EVODEVO-TO-EVOGYM CONNECTION

### D.1 Every file importing `evogym` (verified via `grep -rn "evogym" --include="*.py" .`, and separately `grep` for `import gym`/`import gymnasium`)

| File:line | Import statement |
|---|---|
| `evodevo/evaluation/ppo_runner.py:54` | `import evogym.envs  # noqa: F401  registers evogym gym environments` |
| `evodevo/evaluation/ppo_runner.py:55` | `from evogym import get_full_connectivity` |
| `evodevo/experiments/run_experiment.py:29` | `from evogym import get_full_connectivity` (inside `_compute_connections`, wrapped in try/except `ImportError`) |
| `evodevo/visualization/check.py:71-72` | `import evogym  # noqa: F401`; `import evogym.envs  # noqa: F401` |
| `evodevo/visualization/check.py:87-89` | `import gymnasium as gym`; `import evogym.envs  # noqa: F401`; `from evogym import get_full_connectivity` |
| `evodevo/visualization/render.py:50-52` | `import evogym.envs  # noqa: F401  registers evogym gym environments`; `import gymnasium as gym` (implied by function context; see line 51); `from evogym import get_full_connectivity` |
| `evodevo/visualization/render.py:61` | `from stable_baselines3 import PPO` |
| `evodevo/evaluation/ppo_runner.py:56-57` | `from stable_baselines3 import PPO`; `from stable_baselines3.common.env_util import make_vec_env` |
| `tests/morphology/test_validate.py:7` | `from evogym import has_actuator, is_connected` (inside a try/except gating `HAS_EVOGYM`) |
| `tests/morphology/test_evogym_equivalence.py:14` | `evogym = pytest.importorskip("evogym")` |
| `tests/evaluation/test_ppo_runner.py:27`, `tests/visualization/test_render.py:32`, `tests/visualization/test_render_results.py:11` | `importlib.util.find_spec("evogym.simulator_cpp")` (presence check, not an import) |

No file imports bare `import gym` (the old, pre-gymnasium OpenAI Gym package) anywhere in `evodevo/` or `tests/` — only `gymnasium` (via `evogym.envs`' own registration and `check.py`/`render.py`'s explicit `import gymnasium as gym`).

### D.2 What each import is used for

- `evogym.envs` (imported for side effect only, `# noqa: F401`) — registers EvoGym's Gymnasium environments (e.g. `Walker-v0`) into the global Gymnasium registry so `gym.make("Walker-v0", ...)` / `make_vec_env("Walker-v0", ...)` can resolve the id. This is why it's imported immediately before `make_vec_env` is called in `ppo_runner.py:54-57` and `render.py:50-52`.
- `evogym.get_full_connectivity` — used in `ppo_runner.py:59` to compute the `connections` array passed as an env constructor kwarg (EvoGym environments need voxel adjacency, not just material IDs, to build the physical simulation). Also used in `run_experiment.py:29-33` (best-effort, for the saved `best_body.npz` artifact) and `render.py`/`check.py` for the same env-construction purpose.
- `evogym.{is_connected, has_actuator}` — used **only in test code** (`tests/morphology/test_validate.py`, `tests/morphology/test_evogym_equivalence.py`) as the ground truth that `evodevo/morphology/validate.py`'s native reimplementations are asserted equal to.
- `stable_baselines3.PPO` / `make_vec_env` — the actual RL training/rollout machinery; used in `ppo_runner.py:evaluate_body` to train and evaluate a PPO controller for one robot body, and in `render.py:_load_controller` to load a trained controller for rollout rendering.

### D.3 Exact call chain: validated body → returned fitness score

Starting from `evaluate_body(body, task, ppo_timesteps, eval_episodes, seed, ...)` in `evodevo/evaluation/ppo_runner.py:32-116`:

1. `h = compute_body_hash(body)` (`ppo_runner.py:42`, → `evodevo.morphology.metrics.body_hash`, `metrics.py:47-50`) — sha256 of shape+dtype+bytes.
2. `validation = validate_body(body)` (`ppo_runner.py:44`, → `evodevo.morphology.validate.validate_body`, `validate.py:42-79`) — shape/dtype/value-range/connectivity/actuator checks, using `is_connected`-equivalent logic and `largest_component_size` (`morphology/connectivity.py:39-59`) internally.
3. If invalid: **short-circuit**, return `EvalResult(fitness=sentinel_fitness, valid=False, reason=...)` — evogym is never touched.
4. If valid: `import evogym.envs` (`ppo_runner.py:54`) — registers EvoGym's Gymnasium env ids as a side effect.
5. `from evogym import get_full_connectivity` (55); `connections = get_full_connectivity(body)` (59) — EvoGym's **own** connectivity function (not the native reimplementation) computes the `(2,k)` adjacency array the simulator needs.
6. `from stable_baselines3 import PPO`; `from stable_baselines3.common.env_util import make_vec_env` (56-57).
7. `vec_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs={"body": body, "connections": connections})` (75) — this is the actual `gym.make`-equivalent call: `make_vec_env` internally calls `gymnasium.make(task, **env_kwargs)` once per requested sub-env (here `n_envs=1`) and wraps it in a `DummyVecEnv`/`SubprocVecEnv`, with `task` being an EvoGym-registered id like `"Walker-v0"`.
8. `model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)` (77) — constructs a fresh PPO agent with SB3's default `MlpPolicy` and all other SB3 default hyperparameters (none are overridden here) against this one body's env.
9. `model.learn(total_timesteps=ppo_timesteps)` (78) — trains PPO for exactly `config.ppo_timesteps` environment steps (from the config, default 5000 per `configs/shared/base.yaml`).
10. `eval_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs=env_kwargs)` (80) — a **fresh** second vec-env instance for evaluation (not the training env).
11. Rollout loop over `eval_episodes` (default 1, `configs/shared/base.yaml: ppo_eval_episodes: 1`): `obs = eval_env.reset()`; while not done, `action, _ = model.predict(obs, deterministic=True)`, `obs, reward, dones, _ = eval_env.step(action)`, accumulate `total_reward`; append `total_reward` to `episode_returns` (81-91).
12. `fitness = float(np.mean(episode_returns))` (93) — the mean per-episode return is the returned fitness score.
13. `return EvalResult(fitness=fitness, valid=True, reason=None, wall_clock_s=..., body_hash=h)` (95-101).
14. `finally:` both vec-envs are `.close()`d (113-116) regardless of success/failure.
15. Any exception anywhere in steps 4-13 is caught (`except Exception as exc`, 102) and converted to a sentinel-fitness, `valid=False` result with `reason="simulation_instability: ..."` (105-111) — this is the *only* place a real-PPO evaluation can fail without crashing the process.

Upstream of `evaluate_body`, the full chain from the EA loop is (see also D.6): `evodevo.evolution.ea.run_ea` → `evaluate_genome` closure (`ea.py:84-139`) → `body = encoder.decode(genome)` (90) → (cache check, optional) → `result = evaluator(body, task=config.task, ppo_timesteps=config.ppo_timesteps, eval_episodes=config.ppo_eval_episodes, seed=config.seed, sentinel_fitness=...)` (102-109) where `evaluator` is whatever `make_evaluator(mode)` returned (`evaluate_body` in real mode) → the 15-step chain above.

### D.4 Confirmed native (non-evogym-importing) reimplementations

Confirmed by reading `evodevo/morphology/validate.py` and `evodevo/morphology/connectivity.py` in full, and grepping both files for `evogym` (zero hits in either):

- `is_connected(body) -> bool` — `evodevo/morphology/validate.py:14-26`. Docstring: `"""Native equivalent of ``evogym.is_connected``."""`
- `has_actuator(body) -> bool` — `evodevo/morphology/validate.py:29-33`. Docstring: `"""Native equivalent of ``evogym.has_actuator``..."""`
- `get_full_connectivity(robot: np.ndarray) -> np.ndarray` — `evodevo/morphology/connectivity.py:7-36`. Docstring: `"""Native equivalent of ``evogym.get_full_connectivity``."""`

All three are genuinely native reimplementations (numpy/stdlib only), not thin wrappers that secretly import evogym. `docs/decisions.md:21` documents the exact provenance: these were added specifically so `evodevo/encoders/rule_based/rule_encoding.py` (which previously imported the three directly from `evogym`, a documented "encoder-isolation contract violation") could instead import them from `evodevo.morphology.*`. Equivalence to the real evogym functions is asserted by `tests/morphology/test_evogym_equivalence.py:137-149` (`test_is_connected_matches_evogym`, `test_has_actuator_matches_evogym`, `test_get_full_connectivity_matches_evogym`), which imports real `evogym` directly (permitted in test code) and compares outputs across a parametrized set of hand-built and seeded-random bodies.

Note: `ppo_runner.py:55` still calls **evogym's own** `get_full_connectivity` (not the native one) when actually building an env for real PPO training — the native reimplementation in `morphology/connectivity.py` is used by the rule-based encoder's internal validity gate (`rule_encoding.py:24,220`) and by `validate_body`'s connectivity check, not by the real-evaluation path, which legitimately needs to call into evogym anyway once a body is already known-valid.

### D.5 Where the external EvoGym dependency actually lives and how it's installed

- `requirements.txt:1` (a comment, not a pip line): `# evogym must be installed separately from source (see evogym/ in this repo).` — this comment itself is stale/inaccurate for the current checkout, since there is no `evogym/` directory anywhere in this repo (confirmed by `find`).
- No `external/evogym/` directory exists in this checkout (see A.1).
- `pip show evogym` (run against `/Users/tejeshwarcdr/Academics/FYP/EvoGym_Project/.venv`) reports: `Name: evogym`, `Version: 2.0.0`, `Home-page: https://github.com/EvolutionGym/evogym`, `Author-email: Jagdeep Singh Bhatia <jagdeep@mit.edu>`.
- The actual install provenance, read from `.venv/lib/python3.10/site-packages/evogym-*/direct_url.json` (pip's editable-install record): `{"dir_info": {}, "url": "file:///Users/tejeshwarcdr/Academics/EvoGym_Project/evogym"}`. This is a **plain (non-editable — `dir_info` has no `"editable": true` key) install from a local path outside this repo entirely**, at `/Users/tejeshwarcdr/Academics/EvoGym_Project/evogym` — note this path has **no `FYP` component and no `.claude/worktrees/...` component**, i.e. it points at a *different* checkout of the project on this machine than the one this document was written from (`/Users/tejeshwarcdr/Academics/FYP/EvoGym_Project/...`). This is a finding worth flagging: the currently-active Python environment's evogym installation is not traceable to any file inside this repo/worktree — it was built from a separate, sibling checkout that this worktree does not have access to, so `external/evogym/` (which several docs assume exists) may exist there but not here.

### D.6 Top-to-bottom connection diagram

```
python -m evodevo.experiments.run_experiment --config C --seed N
  └─ run_experiment.run_experiment()                        [run_experiment.py:44]
       ├─ ExperimentConfig.from_method_yaml(C)               [core/config.py:105]  (base+override merge, Part A.3)
       ├─ encoder = get_encoder(config.encoder, ...)         [encoders/registry.py:28]
       ├─ evaluator = make_evaluator("real")                 [evaluation/ppo_runner.py:162] -> evaluate_body
       ├─ logger = RunLogger(output_dir, ...)                [evolution/logging.py:16]
       └─ run_ea(encoder, evaluator, config, logger)         [evolution/ea.py:40]
            ├─ rng = np.random.default_rng(config.seed)      [ea.py:47]
            ├─ for each of `population_size` initial individuals:
            │     genome = encoder.random_genome(rng)        [Encoder.random_genome — Part C]
            │     evaluate_genome(genome, parent_id=None, generation=0)   [ea.py:84]
            │        ├─ body = encoder.decode(genome)        [Encoder.decode — Part C walkthroughs]
            │        ├─ (optional) cache.get(body_hash)      [evaluation/cache.py:13]
            │        ├─ result = evaluator(body, task=config.task, ppo_timesteps=..., eval_episodes=..., seed=config.seed, sentinel_fitness=...)
            │        │     = evaluate_body(...)               [evaluation/ppo_runner.py:32]
            │        │        ├─ validate_body(body)          [morphology/validate.py:42]  (uses is_connected/has_actuator native logic + connectivity/largest_component_size)
            │        │        ├─ if invalid -> sentinel EvalResult, evogym never touched
            │        │        ├─ if valid:
            │        │        │    ├─ import evogym.envs                        [registers Gymnasium env ids]
            │        │        │    ├─ connections = evogym.get_full_connectivity(body)
            │        │        │    ├─ vec_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs={body, connections})  [-> gymnasium.make(task, ...) internally]
            │        │        │    ├─ model = PPO("MlpPolicy", vec_env, seed=seed)
            │        │        │    ├─ model.learn(total_timesteps=ppo_timesteps)
            │        │        │    ├─ eval_env = make_vec_env(task, ...)  (fresh instance)
            │        │        │    ├─ rollout eval_episodes episodes, deterministic model.predict
            │        │        │    └─ fitness = mean(episode_returns)   -> EvalResult(fitness, valid=True, ...)
            │        │        └─ except Exception -> sentinel EvalResult, reason="simulation_instability: ..."
            │        ├─ (optional) cache.put(body_hash, result)
            │        ├─ record = {eval_index, generation, genome_id, parent_id, fitness, valid, reason, body_hash, wall_clock_s, cache_hit, **body_metrics(body)}
            │        └─ logger.log(record)                    [evolution/logging.py:36]  -> appends run_log.jsonl line + heartbeat + progress callback
            ├─ while evaluations_used < eval_budget:
            │     survivors = top-`population_size` by fitness from current population
            │     for `offspring_size` children:
            │         parent(s) = uniform-random from survivors  [rng.integers]
            │         child_genome = encoder.crossover(...) or encoder.mutate(...)   [config.use_crossover gate]
            │         evaluate_genome(child_genome, parent_id, generation)   (same inner chain as above)
            │     population = survivors + offspring
            └─ return RunResult(population, evaluations_used, best_fitness, records, cache_hit_rate)
       └─ logger.write_meta(...)  -> run_meta.json
       └─ best individual's body re-decoded, connections computed (best-effort), saved to best_body.npz
```

---

## PART E — DATA FLOW AND STATE

### E.1 Exact contents of one evaluation record before it is written to disk

Constructed at `evodevo/evolution/ea.py:116-128` inside the `evaluate_genome` closure, immediately before `logger.log(record)` (`ea.py:131`) writes it as one JSON line to `run_log.jsonl` (`evolution/logging.py:37`):

```python
record = {
    "eval_index": evaluations_used - 1,     # int, populated by ea.py's evaluations_used counter (line 113, 116)
    "generation": generation,               # int, the run_ea generation loop variable (ea.py:142/150)
    "genome_id": genome_id,                 # int, monotonically increasing, assigned in evaluate_genome (ea.py:87-88)
    "parent_id": parent_id,                 # Optional[int], None for gen-0 individuals, else the sampled survivor's genome_id (ea.py:161/165)
    "fitness": result.fitness,              # float, from EvalResult.fitness (evaluation/ppo_runner.py:23-29), either real PPO mean return or sentinel_fitness
    "valid": result.valid,                  # bool, from EvalResult.valid
    "reason": result.reason,                # Optional[str], from EvalResult.reason (validation failure text or "simulation_instability: ..." or None)
    "body_hash": result.body_hash,          # str (sha256 hex), from EvalResult.body_hash <- morphology/metrics.py:body_hash
    "wall_clock_s": result.wall_clock_s,    # float, from EvalResult.wall_clock_s (time.time() delta measured inside evaluate_body/evaluate_body_surrogate)
    "cache_hit": cache_hit,                 # bool, set locally in evaluate_genome depending on whether FitnessCache.get() found this body_hash (ea.py:92-99)
    # ** body_metrics(body) spread in verbatim (morphology/metrics.py:12-44): **
    "occupied_count": ..., "empty_count": ..., "rigid_count": ..., "soft_count": ...,
    "h_act_count": ..., "v_act_count": ..., "actuator_fraction": ...,
    "horizontal_symmetry_score": ..., "vertical_symmetry_score": ...,
    "largest_component_size": ...,
}
```
`RunLogger.log()` (`evolution/logging.py:36-42`) writes `json.dumps(record, sort_keys=True) + "\n"`, flushes immediately, updates the heartbeat file, and (if wired) invokes the stdout progress callback — the record's own persisted content is never touched by the progress path.

### E.2 State persisting across evaluations vs. freshly created each time

**Persists across the whole run:**
- `rng = np.random.default_rng(config.seed)` (`ea.py:47`) — one `Generator` instance threaded through every `random_genome`/`mutate`/`crossover` call for the entire run.
- `cache: Optional[FitnessCache]` (`ea.py:57-61`) — if `config.cache_enabled` (default `False`), one `FitnessCache` instance accumulates `body_hash -> EvalResult` across the whole run, plus running `hits`/`misses` counters (`evaluation/cache.py:7-31`).
- `population: List[Individual]` and `records: list` — accumulate across generations within `run_ea` (`ea.py:141-176`).
- `evaluations_used`, `next_genome_id`, `best_fitness` — closure-scoped counters (`nonlocal`, `ea.py:79-85`) that persist across every call to `evaluate_genome` within one `run_ea` invocation.
- The `RunLogger`'s open file handle to `run_log.jsonl` (`evolution/logging.py:23`) stays open for the whole run, appending one line per evaluation.
- `NCAEncoder._initial_state` (`encoders/nca/encoder.py:66`) — computed once at encoder construction (not per-genome, not per-generation) and reused as the fixed developmental starting point for **every** genome evaluated by that encoder instance for the entire run.

**Freshly created every single evaluation (inside `evaluate_body`, `evaluation/ppo_runner.py:32-116`):**
- `vec_env` and `eval_env` — two brand-new `make_vec_env(...)` instances per call (one for training, one for evaluation), both explicitly `.close()`d in the `finally` block (113-116). No env is ever reused across bodies.
- `model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)` (line 77) — a brand-new PPO agent (random weight init, fresh optimizer state) is trained from scratch for every single evaluated body; nothing about PPO's learned policy or replay/rollout buffers persists from one evaluated body to the next, even across genomes with very similar bodies.
- `connections = get_full_connectivity(body)` — recomputed per call, not cached (a `body_hash`-based `FitnessCache`, if enabled, is the only reuse mechanism, and it caches the whole `EvalResult`, not intermediate connectivity).

### E.3 RNG creation/seeding sites and derivation from one seed

Full result of `grep -rn "random.seed(\|np.random.seed(\|np.random.default_rng(\|RandomState(\|default_rng(\|Random(" evodevo`:

| Site | Derivation |
|---|---|
| `evodevo/core/seeding.py:10` (`make_rng`) | Direct `np.random.default_rng(seed)` — a pure function of its `seed` arg. |
| `evodevo/core/seeding.py:14-15` (`seed_everything`) | `random.seed(seed)`, `np.random.seed(seed)` (+ conditional torch seeding). |
| `evodevo/evolution/ea.py:47` | `np.random.default_rng(config.seed)` — this **is** the single top-level seed for an entire EA run; every genome sampled/mutated/crossed-over in that run derives from repeated draws off this one `Generator`. |
| `evodevo/encoders/nca/encoder.py:31` (`_make_initial_state`) | `np.random.default_rng(_INIT_STATE_SEED)` where `_INIT_STATE_SEED = 20260902` is a **hardcoded module-level constant**, entirely independent of `config.seed`. This is deliberate (module docstring, `nca/encoder.py:1-7,17-27`): the NCA developmental starting condition must be identical across every seed/individual/generation so the only source of variation is genome weights. **Finding**: this means the NCA encoder's initial condition does *not* vary with `config.seed` at all — two different benchmark seeds for the NCA arm differ only in which genomes get sampled/mutated, never in the developmental starting point. |
| `evodevo/encoders/rule_based/encoder.py:74-75` (`_seed_global_random`) | `seed = int(rng.integers(0, 2**31-1))` then `_global_random.seed(seed)` — derives a seed from the caller's `Generator` (itself derived from `config.seed` via `ea.py:47`) and pushes it into Python's **global, process-wide** `random` module state before every `random_genome`/`mutate`/`crossover` call. This is explicitly flagged by the code's own docstring (lines 62-73) as **only safe because the EA loop is strictly single-threaded and sequential** — it mutates global interpreter state, so it would not be safe under any concurrent evaluation scheme. Still fully derived from `config.seed` in the end (via the `np.random.Generator` chain), so determinism is preserved for the current sequential usage, but this is a real, documented fragility if the EA loop is ever parallelized. |
| `evodevo/experiments/benchmark.py:190` (`estimate_wall_clock`) | `np.random.default_rng(config.seed)` — independent one-off `Generator` used only to sample a single genome for timing estimation, not tied to any actual run's RNG stream. |
| `evodevo/analysis/statistics.py:95` (`bootstrap_ci_mean_difference`) | `np.random.default_rng(seed)` with a **separate `seed: int = 0` default parameter**, unrelated to any `ExperimentConfig.seed` — this is post-hoc statistical resampling of already-collected results, not simulation randomness, so it is correctly independent, but is worth noting as a *third* distinct "seed" concept in the codebase (config seed, NCA init-state seed, bootstrap seed). |
| `evodevo/visualization/render_results.py:136` | `np.random.default_rng(best["seed"])` — reuses the recorded run's own seed for any incidental rendering-time randomness (not used for scientific results). |

**Overall finding on seeding**: for a given `run_ea` invocation, *almost* everything genuinely deterministic (genome sampling, mutation, crossover, rule-based's global-`random` bridge) derives from the single `config.seed` passed into `np.random.default_rng` at `ea.py:47`. The one deliberate, documented exception is `NCAEncoder`'s fixed initial developmental state (`_INIT_STATE_SEED = 20260902`), which is **not** derived from `config.seed` and is identical across all seeds/runs by design. `seed_everything` (`core/seeding.py:13-23`), despite being described as the project's seeding "spine" alongside `make_rng`, is **not called anywhere in the production code paths** (`run_experiment.py`, `benchmark.py`, `ea.py`, `ppo_runner.py` all avoid it) — grep confirms its only call site is `tests/core/test_seeding.py`. In particular, PPO training itself (`ppo_runner.py:77`, `PPO(..., seed=seed, ...)`) relies on SB3's own `seed=` plumbing rather than a prior `seed_everything()` call, and stable_baselines3/torch's own internal RNG state is not otherwise globally reset by anything in `evodevo/` before training — this is a plausible (though not proven here) source of nondeterminism in the *torch*-side PPO training itself across otherwise-identical reruns, since `seed_everything`'s `torch.manual_seed`/`torch.cuda.manual_seed_all` calls are simply never invoked in the real run path.

---

## PART F — TESTS AS DOCUMENTATION

Tests that exercise the **real** EvoGym integration (not purely encoder-level/mocked), identified by grepping `tests/` for `evogym`/`HAS_SIMULATOR`/`HAS_EVOGYM`:

- **`tests/morphology/test_evogym_equivalence.py`** (`evogym = pytest.importorskip("evogym")`, line 14) — asserts `evodevo`'s native `is_connected`/`has_actuator`/`get_full_connectivity` produce byte-identical results to the real, installed `evogym` package's own functions (`test_is_connected_matches_evogym`, `test_has_actuator_matches_evogym`, `test_get_full_connectivity_matches_evogym`, lines 137-149) across a parametrized set of hand-built and seeded-random bodies. This is the direct verification evidence for the claim in Part D.4 that the morphology reimplementations are behaviorally equivalent to evogym's originals, not merely similarly-named.
- **`tests/morphology/test_validate.py`** — imports `from evogym import has_actuator, is_connected` guarded by try/except setting `HAS_EVOGYM` (lines 7-11); `test_connectivity_matches_evogym` and `test_actuator_check_matches_evogym` (lines 135-143, both `@pytest.mark.skipif(not HAS_EVOGYM, ...)`) cross-check the native `evodevo.morphology.validate` functions against real evogym on concrete bodies.
- **`tests/evaluation/test_ppo_runner.py`** — `HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None` (line 27); tests gated by `@pytest.mark.skipif(not HAS_SIMULATOR, ...)` (line 82) exercise `evaluate_body` against the real compiled evogym simulator + real (short) PPO training, verifying the full real-evaluation path (env construction, `make_vec_env`, `PPO.learn`, rollout, fitness computation) actually runs end-to-end and returns a well-formed `EvalResult`, not just that the function is importable.
- **`tests/visualization/test_render.py`** and **`tests/visualization/test_render_results.py`** — both gated the same way on `evogym.simulator_cpp` presence; exercise `render_body_rollout`/`render_one_arm` against a real evogym env + real (or a genuinely trained-for-the-test) PPO controller, verifying frames are actually captured and GIF/MP4 files are actually written from real simulated rollouts.

All of the above are automatically **skipped** (not failed) when `evogym.simulator_cpp` (the compiled C++ extension) isn't built/importable in the active environment — the pattern the codebase uses throughout (`check.py`'s whole purpose is diagnosing exactly this) to let the rest of the suite run on machines without a working evogym build. Every other encoder/morphology/evolution/analysis test in `tests/` operates purely on the native, evogym-free code paths (numpy arrays, the `Encoder` ABC, `evaluate_body_surrogate`, `FitnessCache`, `run_ea`, `aggregate`/`statistics`) and never touches evogym at all, confirmed by the `grep -rn "evogym"` sweep in Part D.1 returning no hits for any other test file.

---

## END: Quick Reference

| Module | One-line purpose | Most important function/class |
|---|---|---|
| `evodevo/core/config.py` | Loads/merges/enforces the shared-vs-per-method experiment configuration | `ExperimentConfig.from_method_yaml` |
| `evodevo/core/materials.py` | Single authoritative voxel-material vocabulary (EMPTY/RIGID/SOFT/H_ACT/V_ACT) | `MATERIALS` / `N_MATERIALS` |
| `evodevo/core/seeding.py` | Declared (but under-used in production paths) central RNG helpers | `make_rng` |
| `evodevo/encoders/base.py` | The frozen `Encoder` plugin contract every encoding must satisfy | `Encoder` (ABC) |
| `evodevo/encoders/registry.py` | Explicit, non-dynamic encoder name → class registry | `get_encoder` |
| `evodevo/encoders/direct/encoder.py` | Flat 25-gene categorical baseline encoding | `DirectEncoder` |
| `evodevo/encoders/nca/encoder.py` | Weights-only Neural Cellular Automata developmental encoding | `NCAEncoder` |
| `evodevo/encoders/rule_based/*.py` | Ported symbolic developmental rule-language encoding | `RuleBasedEncoder` / `develop_rules` |
| `evodevo/morphology/validate.py` | Native body validity + `is_connected`/`has_actuator` equivalents | `validate_body` |
| `evodevo/morphology/connectivity.py` | Native `get_full_connectivity` equivalent + flood-fill sizing | `get_full_connectivity` |
| `evodevo/morphology/metrics.py` | Descriptive body metrics + the canonical body-identity hash | `body_hash` |
| `evodevo/evaluation/ppo_runner.py` | The one module allowed to call evogym/stable_baselines3; real fitness evaluation | `evaluate_body` |
| `evodevo/evaluation/cache.py` | Body-hash-keyed fitness memoization | `FitnessCache` |
| `evodevo/evolution/ea.py` | Encoding-agnostic (mu+lambda) evolutionary loop | `run_ea` |
| `evodevo/evolution/logging.py` | Per-evaluation JSONL logging + run metadata | `RunLogger` |
| `evodevo/evolution/progress.py` | Additive stdout ETA/progress reporting | `ProgressTracker` |
| `evodevo/experiments/run_experiment.py` | Single (encoder,task,seed) run entry point | `run_experiment` |
| `evodevo/experiments/benchmark.py` | N-way benchmark orchestrator with fairness gating and resumability | `run_benchmark` |
| `evodevo/experiments/fairness.py` | Runtime assertion that two configs are comparable | `assert_comparable` |
| `evodevo/experiments/inspect_encoder.py` | PPO-free per-encoder morphology-prior characterization | `inspect_encoder` |
| `evodevo/analysis/aggregate.py` | Loads run logs into tidy DataFrames + summary tables | `load_all_runs` / `three_way_table` |
| `evodevo/analysis/statistics.py` | Small-n, dependency-free pairwise statistical comparison | `compare_two_arms` |
| `evodevo/analysis/plots.py` | Matplotlib figures for fitness/morphology/validity comparisons | `plot_fitness_curves` |
| `evodevo/analysis/report.py` | Assembles the final markdown benchmark report | `generate_report` |
| `evodevo/analysis/ablation_report.py` | Combines tagged-run results with PPO-free priors for ablations | `run_ablation_report` |
| `evodevo/visualization/check.py` | Diagnoses whether the evogym simulator/environment actually works | `run_all` |
| `evodevo/visualization/development.py` | Plots a developmental encoder's trajectory | `plot_development_trajectory` |
| `evodevo/visualization/morphology.py` | Static body-grid plotting | `plot_body_grid` |
| `evodevo/visualization/render.py` | Real evogym+PPO rollout rendering to GIF/MP4 | `render_body_rollout` |
| `evodevo/visualization/render_results.py` | Finds/renders the best body per run + comparison figures | `render_all` |
