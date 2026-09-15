# Architecture Snapshot

Read-only snapshot of the EvoDevo repository (`/Users/tejeshwarcdr/Academics/EvoGym_Project`) as of 2026-09-03, git commit `9e5cef87f5777b7c66397fea008a53a44229b907` (per `results/runs/benchmark_manifest.json:148`). Every claim below cites an exact `file:line` that was actually read, or the exact command that was actually run. Where something could not be executed, that is stated explicitly rather than guessed.

---

## PART A — High-level architecture

### A.1 Shared modules and the no-forbidden-import guarantee

The one-pipeline/three-decoder principle: everything downstream of `Encoder.decode()` is written once and shared by all three arms. The truly shared modules (imported identically regardless of which encoder is active) are:

- `evodevo/morphology/validate.py` — `validate_body`, `is_connected`, `has_actuator` (native evogym-equivalents)
- `evodevo/morphology/connectivity.py` — `get_full_connectivity`, `largest_component_size`
- `evodevo/morphology/metrics.py` — body metrics/hash (referenced by `evodevo/evolution/ea.py:18-19`)
- `evodevo/evaluation/ppo_runner.py` — `evaluate_body`/`evaluate_body_surrogate` (the only module allowed to import `evogym`/`stable_baselines3`, per its own docstring at `evodevo/evaluation/ppo_runner.py:1-7`)
- `evodevo/evolution/ea.py` — `run_ea`, the encoding-agnostic (mu+lambda) loop
- `evodevo/evolution/logging.py`, `evodevo/evolution/progress.py`, `evodevo/evaluation/cache.py`
- `evodevo/encoders/base.py` — the frozen `Encoder` ABC
- `evodevo/experiments/fairness.py` — `assert_comparable`

**Re-verified by actual grep** (not assumed) that no encoder module imports `evogym`, `gym`, `stable_baselines3`, or `torch`:

```
$ grep -rn "^import\|^from" evodevo/encoders/direct/ evodevo/encoders/nca/ evodevo/encoders/rule_based/ evodevo/encoders/base.py
```

Full result (15 import lines total across `direct/encoder.py`, `nca/encoder.py`, `rule_based/encoder.py`, `rule_based/_rules/{types,interpreter,genome}.py`, `base.py`): every import is one of `numpy`, `evodevo.core.materials`, `evodevo.encoders.base`, `evodevo.encoders.rule_based._rules.*`, `evodevo.morphology.connectivity`/`validate`, `abc`, `typing`, `dataclasses`, `enum`, `copy`, `random` (Python stdlib), or `__future__`. Zero matches for `evogym`, `gym`, `stable_baselines3`, or `torch`. This confirms `README.md:35-37`'s claim and `docs/decisions.md`'s row 21 (which records that this exact grep, `grep -rnE "^\s*(import|from)\s+(evogym|stable_baselines3|gym|torch)\b" evodevo/encoders/`, was previously run and returned zero matches after a prior contract violation was fixed).

The one documented exception to full evogym-freedom in encoder-adjacent code is `evodevo/encoders/rule_based/_rules/genome.py:27-28`, which imports `get_full_connectivity`/`has_actuator`/`is_connected` from `evodevo.morphology.*` (the shared native re-implementations, not `evogym` itself) — consistent with the contract.

### A.2 Genome → logged fitness: call chain

Traced through actual code (`evodevo/evolution/ea.py`, `evodevo/evaluation/ppo_runner.py`, `evodevo/morphology/validate.py`, `evodevo/morphology/connectivity.py`, `evodevo/evolution/logging.py`):

```
run_ea()                                        evodevo/evolution/ea.py:40
 ├─ encoder.random_genome(rng) / encoder.mutate()/.crossover()
 │    (per-encoder: DirectEncoder.random_genome    evodevo/encoders/direct/encoder.py:35
 │                  NCAEncoder.random_genome        evodevo/encoders/nca/encoder.py:92
 │                  RuleBasedEncoder.random_genome  evodevo/encoders/rule_based/encoder.py:118)
 └─ evaluate_genome(genome, parent_id, generation)  evodevo/evolution/ea.py:84
      ├─ body = encoder.decode(genome)              evodevo/evolution/ea.py:90
      │    (DirectEncoder.decode: reshape            evodevo/encoders/direct/encoder.py:38
      │     NCAEncoder.decode: develop()->argmax      evodevo/encoders/nca/encoder.py:151-153
      │     RuleBasedEncoder.decode: develop_rules()  evodevo/encoders/rule_based/encoder.py:122-130)
      ├─ [optional] cache.get(body_hash)             evodevo/evolution/ea.py:95-99
      ├─ result = evaluator(body, task=..., ppo_timesteps=..., ...)
      │    -> evaluate_body(body, task, ppo_timesteps, eval_episodes, seed, ...)
      │         evodevo/evaluation/ppo_runner.py:32
      │         ├─ validate_body(body)               evodevo/morphology/validate.py:42
      │         │    ├─ is_connected-equivalent via largest_component_size()
      │         │    │     evodevo/morphology/connectivity.py:39
      │         │    └─ has_actuator-equivalent check  evodevo/morphology/validate.py:68
      │         │  -> if invalid: return EvalResult(fitness=sentinel_fitness, valid=False, ...)
      │         │     evodevo/evaluation/ppo_runner.py:44-52  [no PPO invoked]
      │         ├─ [if valid] get_full_connectivity(body) (evogym's own)
      │         │     evodevo/evaluation/ppo_runner.py:55,59
      │         ├─ make_vec_env(task, ..., env_kwargs={"body":body,"connections":connections})
      │         │     evodevo/evaluation/ppo_runner.py:75  -> gym.make() internally (evogym env registration
      │         │     via `import evogym.envs` at evodevo/evaluation/ppo_runner.py:54)
      │         ├─ PPO("MlpPolicy", vec_env, seed=seed).learn(total_timesteps=ppo_timesteps)
      │         │     evodevo/evaluation/ppo_runner.py:77-78
      │         └─ rollout loop -> fitness = mean(episode_returns)
      │               evodevo/evaluation/ppo_runner.py:80-101
      ├─ record = {...} built (fitness, valid, reason, body_hash, wall_clock_s, cache_hit, body_metrics)
      │     evodevo/evolution/ea.py:116-128
      └─ logger.log(record)                          evodevo/evolution/ea.py:130-131
             -> RunLogger.log() writes JSONL + heartbeat + optional progress callback
                evodevo/evolution/logging.py (see fairness/timing sections below for its exact behavior)
```

### A.3 Fairness controls mechanically enforced in code

1. **`assert_comparable()`** — `evodevo/experiments/fairness.py:31-49`. Compares every field in `_REQUIRED_EQUAL_FIELDS` (`task` plus `SHARED_BASE_KEYS`, `evodevo/experiments/fairness.py:24`) between two `ExperimentConfig`s and raises `FairnessViolation` (`evodevo/experiments/fairness.py:27-28`) on any mismatch, listing every violating field.
2. **Eval-budget accounting in `ea.py`** — `evaluations_used += 1` happens unconditionally inside `evaluate_genome`, after either a cache hit or a real/surrogate evaluator call (`evodevo/evolution/ea.py:113`), and both the initial-population loop (`evodevo/evolution/ea.py:143-147`) and the offspring loop (`evodevo/evolution/ea.py:149-166`) check `evaluations_used >= budget` before evaluating another genome — so budget is spent per-evaluation, not per-generation.
3. **Cache-hit-consumes-budget** — explicit code comment and enforcement at `evodevo/evolution/ea.py:52-61`: `evaluations_used += 1` (line 113) executes identically whether `cache_hit` was True or False; a hit is also logged with `cache_hit=True` in the record (`evodevo/evolution/ea.py:126`). `cache_enabled` defaults to `False` for the primary comparison (`evodevo/evolution/ea.py:57`, confirmed `cache_enabled: false` in every resolved config, e.g. `results/runs/_resolved_configs/direct/walker/seed_0.yaml:14`).
4. **Sentinel fitness for invalid genomes** — `DEFAULT_SENTINEL_FITNESS = -10.0` (`evodevo/evaluation/ppo_runner.py:20`); `evaluate_body` returns this fitness with `valid=False` and a `reason` string, both for structural invalidity (`evodevo/evaluation/ppo_runner.py:44-52`) and for any exception during PPO training/rollout (`evodevo/evaluation/ppo_runner.py:102-111`, deliberately scoped broad-except so one unstable body cannot abort the whole multi-hour benchmark, per the comment at lines 61-70).
5. **use_crossover symmetry check** — `evodevo/evolution/ea.py:72-77`: if `config.use_crossover` is True but `encoder.supports_crossover()` is False, `run_ea` raises `ValueError` immediately at startup rather than silently falling back to mutation. Default is `False` for every arm in the primary run (`ea.py:66-72` comment; confirmed `use_crossover: false` in all three resolved seed_0 configs).
6. **Encoder ABC contract** — `evodevo/encoders/base.py:27-108` defines the exact frozen method set every encoder must implement (see A.4/Part C.4 below); a shared contract test suite (`tests/encoders/test_contract.py`, referenced in `README.md:38`) exercises every encoder against it.
7. **`min_voxels`/`min_actuators` validity floor** — applied identically to every encoding via the single shared `validate_body()` call in `ppo_runner.py:44` (not duplicated per-encoder), with values from the shared config (`min_voxels: 3`, `min_actuators: 1` in every resolved config, e.g. `results/runs/_resolved_configs/direct/walker/seed_0.yaml:16-17`).

---

## PART B — The three encoders

### B.1 direct

**File structure** (`find evodevo/encoders/direct -type f`):
```
evodevo/encoders/direct/__init__.py
evodevo/encoders/direct/encoder.py
```
`__init__.py:1` re-exports `DirectEncoder` from `encoder.py`.

**Genome representation**: fixed-length `np.ndarray`, `dtype=np.int64` (`evodevo/encoders/direct/encoder.py:23-24`), length `grid_shape[0]*grid_shape[1]` = 25 for the default (5,5) grid (`evodevo/encoders/direct/encoder.py:26-27`). One categorical material ID (0-4) per voxel, flat row-major.

**Actual `random_genome(seed=0)` run** (`.venv/bin/python`, `np.random.default_rng(0)`):
```
genome type: <class 'numpy.ndarray'>
shape: (25,) dtype: int64
first 10: [1 0 0 0 3 4 1 2 0 4]
```

**decode() logic** (`evodevo/encoders/direct/encoder.py:38-39`): a single-line pure reshape, `np.asarray(genome, dtype=int).reshape(self.grid_shape)` — no validity handling, no iteration.

**mutate()** (`evodevo/encoders/direct/encoder.py:41-47`): copies the genome, builds a boolean mask via `rng.random(genome.shape) < self.mutation_rate` (default 0.1), and for every masked position redraws a new material from `_material_probs` (a fixed distribution biased toward `empty_prob`=0.6 for material 0, uniform over the remaining 4 materials otherwise, set in `__init__` at lines 15-16). **No crossover**: `DirectEncoder` does not override `supports_crossover()`/`crossover()`, so it inherits the ABC defaults (`supports_crossover() -> False`, `crossover()` raises `NotImplementedError`) from `evodevo/encoders/base.py:62-77`.

**Actual `describe()` output**: `{'name': 'direct', 'grid_shape': [5, 5], 'mutation_rate': 0.1, 'empty_prob': 0.6}`

**Locked primary-config hyperparameters** (`results/runs/_resolved_configs/direct/walker/seed_0.yaml`): `population_size=15, offspring_size=15, eval_budget=300, ppo_timesteps=5000, ppo_eval_episodes=1, grid_shape=[5,5], sentinel_fitness=-10.0, cache_enabled=false, use_crossover=false, min_voxels=3, min_actuators=1` (lines 4-17).

**Documented limitations/quirks**: `docs/decisions.md` row dated 2026-09-02 ("Direct-encoding random-genome validity rate under default hyperparameters", `docs/decisions.md:15`): random direct genomes are valid only "~2.25-2.4% of the time" at default `empty_prob=0.6`/`mutation_rate=0.1`, deliberately left un-retuned; a second row (`docs/decisions.md:14`) records that `eval_budget` was scaled to 300 specifically to give mutation a "~99%+ analytic chance of finding at least one valid body per seed" given this low prior. `results/BENCHMARK_REPORT.md:41` reports its realized run-time validity rate as `0.2507` — much higher than the ~2.25% random-genome prior, because (per `docs/RESULTS_EXPLAINED.md:62`) evolution concentrates search once a valid body is found.

### B.2 nca

**File structure**:
```
evodevo/encoders/nca/__init__.py
evodevo/encoders/nca/encoder.py
```
`__init__.py:1` re-exports `N_MATERIALS, NCAEncoder`.

**Genome representation**: fixed-length `np.ndarray`, `dtype=np.float64` (`evodevo/encoders/nca/encoder.py:80-81`), length 792 for the default config — a flat weights-only vector for a 2-layer perception network (`w1`: 40×16, `b1`: 16, `w2`: 16×8, `b2`: 8; sizes computed at `evodevo/encoders/nca/encoder.py:68-73`). `channels = N_MATERIALS(5) + hidden_channels(3) = 8`; `input_dim = 5*channels = 40` (own cell + 4 cardinal neighbours) (lines 63-64).

**Actual `random_genome(seed=0)` run**:
```
genome type: <class 'numpy.ndarray'>
shape: (792,) dtype: float64
first 10: [ 0.06286511 -0.06605243  0.32021133  0.05245006 -0.26783469  0.18079753
  0.65200002  0.47354048 -0.35186762 -0.63271074]
```
(drawn from `rng.normal(scale=self.init_scale)`, `init_scale=0.5`, `evodevo/encoders/nca/encoder.py:92-93`.)

**decode() logic** (`evodevo/encoders/nca/encoder.py:151-153` calling `develop()` at lines 133-149): unpacks the genome into `(w1,b1,w2,b2)` (`_unpack`, lines 95-106); starts from a **fixed, non-evolved** initial cell state `_INIT_STATE_SEED=20260902` generated once via `np.random.default_rng` (lines 26-33, `_make_initial_state`); runs `dev_steps=10` synchronous residual update steps (`_step`, lines 125-131: perceive 4-neighbour+self state via zero-padded shifts (`_perceive`, lines 108-123), `tanh` hidden layer, linear output added residually to state); readout at each step is `argmax` over the first `N_MATERIALS` channels (`readout`, lines 137-138). `decode()` returns only the final trajectory element.

**mutate()** (`evodevo/encoders/nca/encoder.py:161-168`): copies genome, selects `n_mutate = round(mutation_fraction * n)` indices without replacement (`mutation_fraction=1.0` by default, i.e. every weight), adds `rng.normal(scale=mutation_sigma=0.1)` Gaussian noise to each. **No crossover** (inherits ABC defaults, same as direct — no override present in `evodevo/encoders/nca/encoder.py`).

**Actual `describe()` output**: `{'name': 'nca', 'grid_shape': [5, 5], 'hidden_channels': 3, 'hidden_width': 16, 'dev_steps': 10, 'neighbourhood': 'von_neumann', 'init_scale': 0.5, 'mutation_sigma': 0.1, 'mutation_fraction': 1.0, 'channels': 8, 'genome_length': 792}`

**Locked primary-config hyperparameters** (`results/runs/_resolved_configs/nca/walker/seed_0.yaml`): shared-pipeline fields identical to direct (lines 4-17); NCA-specific hyperparameters come from `describe()` above (encoder hyperparameters are not part of `ExperimentConfig`, per `evodevo/experiments/fairness.py:6-9`).

**Documented limitations/quirks**: `docs/decisions.md:16` ("E0 sanity gate... NCA validity rate and degeneracy"): validity rate 0.6915, degenerate fraction 0.0195 at N=2000 seed=0 — both pass gates. Critically, the same row discloses: "68.35% of genomes were still changing on the very last developmental step (t=9→t=10) and never observably settled within `dev_steps=10`; mean max-abs residual-state magnitude at t=T was ~40.7 (vs. an initial-state scale of 0.1), indicating the residual update is not bounded/self-stabilizing at these defaults." This is reconfirmed byte-for-byte in `docs/decisions.md:20` and carried into `results/BENCHMARK_REPORT.md:69` and `docs/RESULTS_EXPLAINED.md:71` as an accepted, disclosed limitation.

### B.3 rule_based

**File structure**:
```
evodevo/encoders/rule_based/__init__.py
evodevo/encoders/rule_based/encoder.py
evodevo/encoders/rule_based/_rules/__init__.py
evodevo/encoders/rule_based/_rules/genome.py       (350 lines)
evodevo/encoders/rule_based/_rules/interpreter.py  (323 lines)
evodevo/encoders/rule_based/_rules/types.py        (250 lines)
```
`encoder.py:1-13` documents that `encoder.py` itself is new adapter code, not a port; `_rules/*.py` are the ported modules (module-header provenance comments at `_rules/interpreter.py:1-17` and `_rules/genome.py:1-9`).

**Genome representation**: `DevelopmentalRuleGenome`, a variable-length (`MIN_RULES=2..MAX_RULES=12`, `_rules/genome.py:44-45`) ordered list of `DevelopmentalRule` objects — **not** a fixed-shape numeric array; `genome_dtype` returns Python `object` (`evodevo/encoders/rule_based/encoder.py:107-109`). Each `DevelopmentalRule` = `(condition_type, condition_params) -> (action_type, action_params)`.

**Actual `random_genome(seed=0)` run** (via `_seed_global_random(rng)` deriving a seed from `np.random.default_rng(0)` and reseeding Python's global `random` module, `evodevo/encoders/rule_based/encoder.py:68-81,118-120`):
```
genome type: <class 'evodevo.encoders.rule_based._rules.genome.DevelopmentalRuleGenome'>
repr: DevelopmentalRuleGenome(
    DevelopmentalRule(cell_type{'state': <VoxelState.HARD: 'hard'>} -> grow_directional{'direction': <Direction.RIGHT: 'right'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>})
    DevelopmentalRule(empty_neighbor{'direction': <Direction.DOWN: 'down'>} -> grow_directional{'direction': <Direction.DOWN: 'down'>, 'voxel_type': <VoxelState.SOFT: 'soft'>})
    DevelopmentalRule(empty_neighbor{'direction': <Direction.LEFT: 'left'>} -> copy_neighbor{'direction': <Direction.LEFT: 'left'>})
    DevelopmentalRule(cell_type{'state': <VoxelState.SOFT: 'soft'>} -> grow_directional{'direction': <Direction.RIGHT: 'right'>, 'voxel_type': <VoxelState.SOFT: 'soft'>})
    DevelopmentalRule(cell_type{'state': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>} -> copy_neighbor{'direction': <Direction.LEFT: 'left'>})
)
```
(5 rules, within `[_SAMPLE_MIN_RULES=3, _SAMPLE_MAX_RULES=7]`, `_rules/genome.py:151-152,165`.)

**decode() logic** (`evodevo/encoders/rule_based/encoder.py:122-130`): calls `develop_rules(genome, shape=self.grid_shape, iterations=self.developmental_iterations)` directly (the raw interpreter, not `generate_robot_from_developmental_rules`, deliberately — per the docstring, `decode()` must never itself signal validity). The interpreter (`_rules/interpreter.py`) starts from a single-SOFT-seed-at-centre initial grid and runs a **two-pass synchronous update per iteration**: pass 1 resolves every cell's first-matching rule purely against the current (pre-iteration) grid (row-major, first-match-wins per `_rules/interpreter.py:60-64`); pass 2 writes self-directed actions first, then `GROW_DIRECTIONAL` actions (deliberately ordered so growth into a cell cannot be immediately erased by that cell's own same-iteration self-action) — full rationale documented at `_rules/interpreter.py:44-60`. No `random.*` calls inside the interpreter itself (deterministic given genome+shape+iterations, `_rules/interpreter.py:32-36`).

**mutate()** (`evodevo/encoders/rule_based/encoder.py:132-134`): reseeds global `random`, then delegates to `mutate_developmental_rules(genome, mutation_rate=0.2)` (ported operator in `_rules/genome.py`, structural mutation operators including `_mutate_condition_type`/`_mutate_condition_params`/`_mutate_action_type`/`_mutate_action_params`/`_add_rule` at `_rules/genome.py:229-260+`).

**crossover()** — **supported** (`supports_crossover() -> True`, `evodevo/encoders/rule_based/encoder.py:148-149`), the only one of the three arms to do so; delegates to ported `crossover_developmental_rules(parent_a, parent_b)` (`evodevo/encoders/rule_based/encoder.py:151-155`).

**Actual `describe()` output**: `{'name': 'rule_based', 'grid_shape': [5, 5], 'developmental_iterations': 5, 'mutation_rate': 0.2, 'min_rules': 2, 'max_rules': 12, 'condition_types': ['always', 'cell_type', 'neighbor_count', 'empty_neighbor', 'position'], 'action_types': ['set_empty', 'set_soft', 'set_hard', 'set_actuator_horizontal', 'set_actuator_vertical', 'grow_directional', 'copy_neighbor']}`

**Locked primary-config hyperparameters** (`results/runs/_resolved_configs/rule_based/walker/seed_0.yaml`): shared-pipeline fields identical to the other two arms (lines 4-17). Note `use_crossover: false` (line 15) even though this is the one arm capable of crossover — the primary comparison is mutation-only for every arm (`docs/decisions.md` row 19, `evodevo/evolution/ea.py:63-71`); a `use_crossover=True` run of this encoder would be a separately-labelled ablation arm (`rule_based_xover`), never merged into the primary run.

**Documented limitations/quirks**: `docs/decisions.md` row 18 discloses a corrected pre-existing bug in the ported original: the original reference's SOFT/HARD numeric codes were swapped relative to EvoGym's real material vocabulary; this project's port corrects them (`RIGID=1`/`SOFT=2`) rather than reproducing the bug faithfully — "the one deliberate, documented deviation from an otherwise verbatim port." Row 21 (`docs/decisions.md:21`) records a since-fixed contract violation: `rule_encoding.py` originally imported `is_connected`/`has_actuator`/`get_full_connectivity` directly from `evogym`, moved to the shared `evodevo/morphology/*` native equivalents. Row 22 records the permanent deletion of `external/rule-based-reference/` (the original standalone checkout) after freezing its port-equivalence proof into `tests/encoders/fixtures/rule_based_port_equivalence.json` — meaning the port's fidelity **can never be re-verified against the original source again**, only against the frozen fixture; this is called out as an explicit, accepted tradeoff.

---

## PART C — Repo structure and EvoGym integration

### C.1 Annotated directory tree (excluding `.venv`, `__pycache__`, `.git` internals; `external/evogym/` collapsed)

```
.
├── CONTRIBUTING.md, Makefile, README.md, REPO_INVENTORY_REPORT.md, pytest.ini, requirements.txt
├── docs/                              decisions log, incident/results docs, module ownership, roadmap
├── evodevo/
│   ├── analysis/                      report.py, ablation_report.py, aggregate.py, plots.py, statistics.py
│   ├── core/                          config.py (ExperimentConfig), materials.py, seeding.py
│   ├── encoders/
│   │   ├── base.py                    Encoder ABC (frozen contract)
│   │   ├── registry.py
│   │   ├── direct/encoder.py
│   │   ├── nca/encoder.py
│   │   └── rule_based/encoder.py, _rules/{types,genome,interpreter}.py
│   ├── evaluation/                    cache.py, ppo_runner.py (only evogym/sb3-importing module)
│   ├── evolution/                     ea.py, logging.py, progress.py
│   ├── experiments/                   run_experiment.py, benchmark.py, fairness.py, inspect_encoder.py,
│   │                                  lock_hash.py, nca_sanity.py, configs/{shared,direct,nca,rule_based}/
│   └── visualization/                 check.py, development.py, morphology.py, render.py, render_results.py
├── external/
│   └── evogym/                        the sole external dependency (nested git checkout)
├── results/
│   ├── BENCHMARK_REPORT.md            generated three-way comparison report
│   ├── ablation_priors/, ablations.md deferred-ablation PPO-free prior data
│   ├── archive/                       superseded run data, kept with per-directory README (never deleted)
│   ├── figures/                       fitness_curves.png, morphology_comparison.png, validity_efficiency.png
│   ├── inspection/{direct,nca,rule_based}/  PPO-free per-encoder characterization
│   ├── media/{direct,nca,rule_based}/ per-arm rollout gif/mp4, morphology.png, development_trajectory.png
│   └── runs/                          current/latest primary run: benchmark_manifest.json, primary_run.log,
│                                       .heartbeat.json, _resolved_configs/, direct|nca|rule_based/Walker-v0/seed_*/
├── scripts/                            benchmark/, direct/, nca/, rule_based/ (test.sh, inspect.sh, run scripts)
└── tests/                              (not enumerated in depth here; 279+ tests per README/decisions log)
```
(Full raw listing captured via `find . -path ./.venv -prune -o -path ./external/evogym -prune -o ... -print`.)

### C.2 Where/how evogym is invoked

Grep for evogym-related calls across `morphology/` and `evaluation/` (`grep -n "evogym\|is_connected\|has_actuator\|get_full_connectivity\|gym.make\|PPO(" evodevo/morphology/*.py evodevo/evaluation/*.py`):

- `evodevo/morphology/connectivity.py:7-8` — `get_full_connectivity()` is a **native re-implementation** ("Native equivalent of `evogym.get_full_connectivity`"), not a call into evogym.
- `evodevo/morphology/validate.py:14-15,29-30` — `is_connected()`/`has_actuator()` are likewise native re-implementations, docstring-labelled as evogym-equivalents.
- `evodevo/evaluation/ppo_runner.py:54` — `import evogym.envs` (registers evogym's gym environments) — the actual evogym touchpoint.
- `evodevo/evaluation/ppo_runner.py:55,59` — `from evogym import get_full_connectivity` then `connections = get_full_connectivity(body)` — this call site uses evogym's **real** function (distinct from the native copy in `morphology/connectivity.py`, which exists so encoder-adjacent code never needs to import evogym at all).
- `evodevo/evaluation/ppo_runner.py:75,80` — `make_vec_env(task, n_envs=1, seed=seed, env_kwargs={"body":body,"connections":connections})` (stable_baselines3 helper that internally calls `gym.make`/env construction against the evogym-registered `task` id, e.g. `Walker-v0`).
- `evodevo/evaluation/ppo_runner.py:77-78` — `PPO("MlpPolicy", vec_env, seed=seed, verbose=0)` then `.learn(total_timesteps=ppo_timesteps)` — the PPO training loop, from `stable_baselines3`.
- `evodevo/evaluation/ppo_runner.py:87-91` — deterministic rollout loop (`model.predict(..., deterministic=True)`, `eval_env.step(action)`) used for the fitness-evaluation episodes.

So: `morphology/` never touches evogym directly (it re-implements the same semantics natively so the encoder-isolation contract holds); `evaluation/ppo_runner.py` is the **sole** module that imports and calls evogym/gym/stable_baselines3, exactly as its module docstring states (`ppo_runner.py:1-7`).

### C.3 External dependency confirmation

```
$ ls external/
evogym
```
`external/evogym/` is present (nested checkout with its own `.git`, `setup.py`, `evogym/` package, etc.). A search for any rule-based-reference directory:
```
$ find . -maxdepth 2 -iname "*rule-based-reference*"
(no output)
```
confirms `external/rule-based-reference/` no longer exists — consistent with `docs/decisions.md` row 22's record of its deliberate, permanent deletion after the port-equivalence proof was frozen into `tests/encoders/fixtures/rule_based_port_equivalence.json`. `external/evogym/` is therefore the project's sole external dependency, as claimed in `README.md:19` and `evodevo/encoders/rule_based/encoder.py:9-12`.

### C.4 Encoder ABC — exact current method signatures (`evodevo/encoders/base.py`, quoted verbatim)

```python
27  class Encoder(ABC):
28      @property
29      @abstractmethod
30      def name(self) -> str:
31          ...
33      @abstractmethod
34      def genome_length(self) -> int:
35          ...
37      @property
38      @abstractmethod
39      def genome_dtype(self) -> Any:
48      @abstractmethod
49      def random_genome(self, rng: np.random.Generator) -> np.ndarray:
50          ...
52      @abstractmethod
53      def decode(self, genome: np.ndarray) -> np.ndarray:
54          """Return a (5, 5) int array of material IDs. No validity handling here."""
55          ...
57      @abstractmethod
58      def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
59          """Return a mutated copy of genome. Must not loop until a valid body results."""
60          ...
62      def supports_crossover(self) -> bool:
67          return False
69      def crossover(self, parent_a, parent_b, rng: np.random.Generator):
74          raise NotImplementedError(...)
79      @abstractmethod
80      def serialize_genome(self, genome) -> Any:
85          ...
87      @abstractmethod
88      def deserialize_genome(self, obj: Any):
92          ...
94      @abstractmethod
95      def describe(self) -> dict:
96          """JSON-serialisable hyperparameters, written into every run log."""
97          ...
99      def develop_trajectory(self, genome) -> Any:
108         return None
```
(Line numbers exactly as in `evodevo/encoders/base.py`.)

---

## PART D — Execution flow and results

### D.1 Entry points, as they currently exist

From `Makefile:1-90` and `README.md:57-65`:

- **Single-arm experiment**: `python3 -m evodevo.experiments.run_experiment --config <path> --seed <int> [--surrogate] [--overwrite] [--encoder-kwarg KEY=VALUE ...] [--tag TAG]` (CLI defined at `evodevo/experiments/run_experiment.py:114-157`).
- **Full three-way benchmark**: `make benchmark` (optionally `SURROGATE=1`), which runs:
  ```
  python3 -m evodevo.experiments.benchmark --config evodevo/experiments/configs/benchmark.yaml [--surrogate]
  python3 -m evodevo.analysis.report --root results/runs --output results/BENCHMARK_REPORT.md --figures-dir results/figures
  ```
  (`Makefile:64-68`; CLI for `benchmark.py` at `evodevo/experiments/benchmark.py:282-304`, `--dry-run`/`--surrogate` supported; CLI for `report.py` at `evodevo/analysis/report.py:279-289`.)
- **Analysis/report generation** alone: `python -m evodevo.analysis.report --root <runs_root> --output <path> --figures-dir <dir>`.
- **Rendering pipeline**: `scripts/benchmark/render_results.sh [TASK] [RUNS_ROOT] [OUTPUT_ROOT]`, wrapping `python3 -m evodevo.visualization.render_results --runs-root ... --task ... --output-root ...`.
- **Per-encoder tests**: `make test-direct` / `test-nca` / `test-rule-based` → `bash scripts/<method>/test.sh`; `make test` runs the full suite; `make test-core` runs shared-core-only.
- **Per-encoder PPO-free characterization**: `make inspect-direct` / `inspect-nca` / `inspect-rule-based` → `bash scripts/<method>/inspect.sh` (writes to `results/inspection/<method>/`).

### D.2 Final benchmark_manifest.json (full, verbatim)

```json
{
  "completed_this_invocation": [ /* 15 entries: direct/nca/rule_based x Walker-v0 x seed 0-4 */ ],
  "encoders": {
    "direct": {"empty_prob": 0.6, "grid_shape": [5, 5], "mutation_rate": 0.1, "name": "direct"},
    "nca": {"channels": 8, "dev_steps": 10, "genome_length": 792, "grid_shape": [5, 5], "hidden_channels": 3, "hidden_width": 16, "init_scale": 0.5, "mutation_fraction": 1.0, "mutation_sigma": 0.1, "name": "nca", "neighbourhood": "von_neumann"},
    "rule_based": {"action_types": ["set_empty", "set_soft", "set_hard", "set_actuator_horizontal", "set_actuator_vertical", "grow_directional", "copy_neighbor"], "condition_types": ["always", "cell_type", "neighbor_count", "empty_neighbor", "position"], "developmental_iterations": 5, "grid_shape": [5, 5], "max_rules": 12, "min_rules": 2, "mutation_rate": 0.2, "name": "rule_based"}
  },
  "git_commit": "9e5cef87f5777b7c66397fea008a53a44229b907",
  "library_versions": {"matplotlib": "3.10.9", "numpy": "1.26.4", "pandas": "2.3.3", "pytest": "9.1.1", "python": "3.10.20 (main, Mar  3 2026, 00:49:35) [Clang 21.0.0 (clang-2100.0.123.102)]", "yaml": "6.0.3"},
  "run_matrix": [ /* identical 15 entries to completed_this_invocation */ ],
  "shared_config": {"cache_enabled": false, "eval_budget": 300, "grid_shape": [5, 5], "min_actuators": 1, "min_voxels": 3, "offspring_size": 15, "population_size": 15, "ppo_eval_episodes": 1, "ppo_timesteps": 5000, "sentinel_fitness": -10.0, "use_crossover": false},
  "skipped_already_done": [],
  "surrogate": false
}
```
(Each of the 15 `completed_this_invocation`/`run_matrix` entries is `{encoder, output_dir: "results/runs/<encoder>/Walker-v0/seed_<n>", seed, task: "Walker-v0"}` for `encoder ∈ {direct, nca, rule_based}`, `seed ∈ {0,1,2,3,4}` — full path `results/runs/benchmark_manifest.json`.)

### D.3 Three-way comparison table (verbatim from `results/BENCHMARK_REPORT.md:39-43`)

| encoder | n_seeds | final_best_fitness_mean | final_best_fitness_std | evaluations_to_50pct_of_best_arm_mean | evaluations_to_90pct_of_best_arm_mean | validity_rate | unique_bodies_explored | cache_hit_rate | mean_wall_clock_s_per_eval |
|---|---|---|---|---|---|---|---|---|---|
| direct | 5 | -1.7371 | 4.1335 | 146.2500 | 131.6667 | 0.2507 | 1203 | 0.0000 | 1.9918 |
| nca | 5 | 0.2772 | 0.0937 | 58.4000 | 163.3333 | 0.8073 | 1498 | 0.0000 | 6.6742 |
| rule_based | 5 | 0.1783 | 0.0749 | 130.6667 | 223.0000 | 0.7760 | 604 | 0.0000 | 6.2188 |

Statistical comparison table (`results/BENCHMARK_REPORT.md:57-61`): all three pairwise 95% CIs on final-fitness mean difference include zero (direct-nca: [-6.1547, 0.1696]; direct-rule_based: [-6.0629, 0.2658]; nca-rule_based: [-0.0011, 0.2073]) — headline is "no detectable difference between arms" (`results/BENCHMARK_REPORT.md:3`).

### D.4 Summary of the three investigations

- **Display-tag reconciliation** (`docs/INCIDENT_VALIDITY_FAILURE.md`, "Addendum: live-display reconciliation" section, lines 144-239): investigated whether a user watching the live progress stream during the `direct`/Walker-v0/seed_0 run seeing what looked like 208 consecutive "validity failure" tags indicated a display bug. Traced the exact tag-selection logic (`evodevo/evolution/progress.py`'s `_RunProgress.__call__`, cited at lines 153-162) and confirmed `RunLogger.log()` passes the identical record object to both the persisted JSONL write and the progress callback (no copy/mutation in between, `evodevo/evolution/logging.py`, cited lines 174-181). Directly verified two real successes (eval_index 135, 153) were correctly tagged `real-PPO` in the persisted log. Concluded no bug exists — the illusion is explained by extreme timing skew: bursts of ~10+ near-instantaneous (sub-millisecond) invalid-body rejections bracket rare ~8-9s real-PPO lines, making the real successes trivially easy for a human to miss on a scrolling terminal (lines 202-239).

- **Config-hash retirement** (`docs/decisions.md`, row dated 2026-09-02, "Retire unreproducible lock hash", `docs/decisions.md:23`): found that the original lock hash `7e93f59a...` could not be reproduced by any reasonable implementation of its own named recipe ("sha256 of the three resolved ExperimentConfig objects, sorted-key JSON") and had evidently been computed ad hoc, by hand, once. Before retiring it, ruled out config drift conclusively via `git diff` across the config files/`ExperimentConfig` dataclass (empty), manual diff of every other changed file in the range, and an isolated git-worktree re-computation at the exact lock commit — both HEAD and the lock commit produced the identical digest `6d6487c9...` under a reimplemented recipe, proving the discrepancy was a serialization-methodology difference, not drift. Retired `7e93f59a...` (kept as a historical record, not deleted) and replaced it with a documented, test-pinned procedure (`evodevo/experiments/lock_hash.py`, `tests/experiments/test_lock_hash.py`), producing lock hash v2 `56b3ed67ca5e74e6303e3917921be7f347c6957b44880092afe711c2377b53c8`, reconfirmed in `docs/FINAL_SMOKE_TEST.md:57-65` immediately before the primary run's launch.

- **Direct-arm timing/validity investigation** (`docs/INCIDENT_VALIDITY_FAILURE.md`, main body, Parts A-C and "Root cause"/"Statistical assessment" sections, lines 1-142; corroborated by `docs/RESULTS_EXPLAINED.md` section 5, "Run integrity: timing investigation (clean)", lines 54-64): investigated an apparent all-validity-failure stretch (208/208 tagged failures observed live) on the `direct`/Walker-v0/seed_0 run. Read the persisted 237-record run log directly: 229 failures, all carrying the exact literal string `"not a single connected component under cardinal adjacency"` from `validate_body()` (`evodevo/morphology/validate.py:66`) — none carrying the `simulation_instability:` exception-path prefix that would indicate a masked exception. Ruled out the masked-exception hypothesis via direct reproduction (a hand-verified valid body run through `evaluate_body` both with and without its own try/except, both succeeding identically, ~8.8s). Re-verified the config lock hash matched exactly (`7e93f59a...` computed vs. locked, "MATCH", lines 96-100). Statistically, an observed 3.4% valid rate against a 2.25% prior over 237 evaluations is ~1.2 standard deviations from expectation — unremarkable; long failure streaks are expected at this validity rate (binomial/geometric-gap reasoning, lines 118-134). Conclusion: "Root cause: None found. No fix is proposed because none is needed" (line 136-138). The companion, separately-scoped timing question — the full 15-run primary benchmark finishing in ~7.5h against an ~11h pre-run estimate (a 32% speedup) — was investigated and found benign: all 15 runs completed their full 300 evaluations, config hashes matched post-hoc, cache was confirmed off with zero hits across all runs, zero exceptions were silently swallowed across 4,500 logged evaluations, and the actual explanation was that realized in-run validity rates (evolution concentrating search once valid regions are found) came in well above the pre-run random-genome priors that the original 11h estimate had assumed (`docs/RESULTS_EXPLAINED.md:56-64`).

### D.5 Open items in `docs/decisions.md`

`docs/decisions.md:3` states the project's own closure rule: *"Every [DEC] item from `evodevo-implementation-roadmap.md` must be resolved here before the results in `results/RESULTS.md` are considered final. No row may remain 'OPEN' in the final submission."*

**This rule is currently violated.** Reading the table verbatim, six rows still carry `OPEN` in the "Chosen" column, all with blank Date/Rationale (`docs/decisions.md:7-12`):

| Decision | Options considered |
|---|---|
| Neighbourhood type | von Neumann (4-neighbour) vs Moore (8-neighbour) |
| Hidden channel count (H) | e.g. 0, 3, ... |
| Developmental steps (T) | e.g. 1, 5, 10, 20 |
| Initial state at t=0 | fixed non-symmetric vs symmetric vs evolved |
| Invalid-genome policy | sentinel fitness + consume budget vs resample-until-valid |
| Mutation tuning procedure | shared rate/sigma vs per-arm pilot tuning |

Note: every one of these six nominally-OPEN rows is, in practice, addressed elsewhere in the same file by a later, dated, non-OPEN row with a concrete rationale — e.g. "Invalid-genome policy" is functionally resolved by the sentinel-fitness mechanism documented throughout (`ppo_runner.py:20`, `ea.py` comments) and "Mutation tuning procedure" is addressed by the "Per-arm mutation-strength tuning asymmetry" threat-to-validity disclosed in both `results/BENCHMARK_REPORT.md:68` and `docs/RESULTS_EXPLAINED.md:72`. But as literal rows in the decisions table, they are still marked `OPEN` and were not observed to have been mechanically closed (no `[x]`/date/rationale filled in) as of this snapshot — this is a direct contradiction of the file's own stated closure rule and should be flagged as such in any teaching discussion of this repo's process discipline.

---

## Verification notes / what could not be run

- All B.2/B.5 outputs above (`random_genome(seed=0)`, `describe()`) were executed for real via `.venv/bin/python` against the current code in this repo, not recalled from memory or docs — full transcript reproduced in Part B.
- The `grep` for forbidden imports (A.1) was run for real against the current `evodevo/encoders/` tree.
- Running the full PPO pipeline (`evaluate_body`) end-to-end was **not** re-executed as part of producing this snapshot (it would take several seconds to minutes per call and mutate no files, but was judged unnecessary/out of scope for a static snapshot); its behavior is instead documented from direct reading of `evodevo/evaluation/ppo_runner.py` plus the already-recorded reproductions in `docs/INCIDENT_VALIDITY_FAILURE.md` and `docs/FINAL_SMOKE_TEST.md`, which this document cites rather than re-derives.
- `tests/` directory contents were not individually enumerated file-by-file (only referenced via README/docs claims of "279 tests" / "278/281 passed" at various points) — the test suite was not re-run as part of this snapshot.
