# EvoDevo Capstone — Implementation Roadmap & Claude Code Prompt Sequence

**Sources of truth:** `project-history.md` (project constraints, scientific position) and `evogym_technical_reference.md` (EvoGym API facts, commit `f63c649`).

**Experimental question this codebase must answer:**

> Under an identical evaluation pipeline, fixed 5×5 grid, and equal evaluation budget, does an NCA developmental encoding differ from a direct encoding in final task fitness, search efficiency, and morphological properties — and in which direction?

Nothing in this plan assumes NCA wins. The architecture is deliberately symmetric so that the comparison, not the encoding, is the deliverable.

---

## Part 0 — Notation for decisions

| Tag | Meaning |
|---|---|
| **[EG]** | Established by the EvoGym reconnaissance report; treat as fact. |
| **[PH]** | Established by `project-history.md`; treat as a project constraint. |
| **[DEC]** | A research/implementation decision **not** settled by either document. Must be chosen explicitly, recorded in `docs/decisions.md`, and justified. |

---

## Part 1 — Architectural principle: one pipeline, two decoders

The single most important structural decision is that **both encodings share every stage except the decoder.**

```
                    ┌─────────────────────┐
   Direct genome →  │                     │
   (25 integers)    │   Encoder interface │ → body: np.ndarray (5,5), int
                    │   .decode(genome)   │
   NCA genome    →  │                     │
   (weight vector)  └─────────────────────┘
                                 ↓
                        morphology validator      [shared]
                                 ↓
                        get_full_connectivity()   [shared, EG]
                                 ↓
                        gym.make('Walker-v0', body=…, connections=…)  [shared, EG]
                                 ↓
                        PPO training (fixed budget)  [shared, EG]
                                 ↓
                        evaluation reward → fitness  [shared]
                                 ↓
                        EA selection on the genome   [shared]
```

Everything below the encoder box is written **once**. If any of it is duplicated per-encoding, the comparison is compromised, because a difference in results could come from the pipeline rather than the representation.

**Target repository layout:**

```
evodevo/
  encoders/
    base.py           # Encoder ABC: decode(genome) -> (5,5) int array; genome_space(); random_genome(rng); mutate(genome, rng)
    direct.py         # DirectEncoder — 25 categorical values
    nca.py            # NCAEncoder — weight vector, developmental rollout
  morphology/
    validate.py       # shape/material/connectivity/actuator/min-voxel checks
    repair.py         # the ONE shared invalid-genome policy
    metrics.py        # descriptive morphology metrics (voxel counts, symmetry, compactness)
  evaluation/
    ppo_runner.py     # body -> PPO train -> mean eval reward. The only place SB3 is touched.
    cache.py          # body-hash -> fitness memoization
  evolution/
    ea.py             # encoding-agnostic (μ+λ) EA loop
    logging.py        # per-generation JSONL records
  experiments/
    configs/          # YAML: seed, budget, encoder, task, EA params
    run_experiment.py # single entry point, takes a config
  analysis/
    aggregate.py      # load run logs -> tidy dataframe
    plots.py          # fitness-vs-evaluations curves, morphology metric comparisons
  tests/
docs/
  decisions.md        # every [DEC] recorded with rationale and date
```

---

## Part 2 — Ordered build sequence (what, and why in this order)

| Stage | Build | Why it comes here |
|---|---|---|
| 1 | Repo skeleton, config, seeding, decisions log | Reproducibility must exist before any result exists. |
| 2 | Morphology validator + metrics | Every later stage depends on "is this body legal?" |
| 3 | Encoder ABC + DirectEncoder | Defines the contract the NCA must later satisfy. Direct is the simpler of the two. |
| 4 | PPO evaluation wrapper (with a cheap surrogate mode) | The expensive component; needs to be isolated and testable before the EA drives it thousands of times. |
| 5 | Encoding-agnostic EA loop | Validated using the direct encoder and the surrogate fitness. |
| 6 | **Baseline run: direct encoding, real PPO** | This is the scientific control. Must produce sane numbers before NCA exists. |
| 7 | NCAEncoder (development engine) | Introduced only once the surrounding pipeline is trusted. |
| 8 | NCA sanity/ablation experiments | Establish that development is actually doing something. |
| 9 | Head-to-head comparison runs | The actual result. |
| 10 | Analysis, plots, write-up | — |

The critical ordering claim: **do not write the NCA until stage 6 has produced a believable direct-encoding fitness curve.** If the pipeline is broken, an NCA result is uninterpretable, and debugging a novel encoding on top of an unvalidated pipeline is where undergraduate projects lose weeks.

---

## Part 3 — The NCA specification to implement (fixed before coding)

Consistent with `project-history.md` §2.4 (fixed grid, no growth) and the NCA explainer already produced.

**Cell state.** Each of the 25 voxels holds a vector of length `C = 5 + H`:
- 5 **material channels** — logits over `{EMPTY, RIGID, SOFT, H_ACT, V_ACT}` = IDs `{0,1,2,3,4}` **[EG]**. `FIXED` (5) is excluded **[EG, report §I]**.
- `H` **hidden channels** — free-form, no fixed meaning. **[DEC]** Recommend `H = 3` for the minimum viable version; `H` is an ablation axis.

**Neighbourhood. [DEC]** Recommend 4-neighbour (von Neumann) with zero-padding outside the grid, because EvoGym connectivity itself is cardinal **[EG: `get_full_connectivity` connects right/bottom neighbours only]**. 8-neighbour is an ablation.

**Network.** Input = own state ⊕ 4 neighbour states = `5C` numbers. One hidden layer, then a residual update:

```
s_i(t+1) = s_i(t) + f_θ( concat(s_i(t), s_up, s_down, s_left, s_right) )
```

**[DEC]** Recommend hidden width 16 and `tanh` on the hidden layer. With `H=3`, `C=8`, input `=40`: params = (40·16+16) + (16·8+8) = **792**.

**Initial state (t=0). [DEC]** Not settled by `project-history.md` §13.4. Recommend **Option A from the NCA explainer**: a *fixed*, identical, non-symmetric initial state for every individual, so the genome is weights-only and development is the only thing evolution shapes. A fully symmetric init risks the symmetry-breaking failure mode; record the chosen init in `decisions.md`.

**Development.** `T` synchronous steps. **[DEC]** Recommend `T = 10`, with `T ∈ {1, 5, 10, 20}` as an ablation — note that `T = 1` is effectively a non-developmental control and is scientifically valuable for exactly that reason.

**Readout.** `body[r,c] = argmax(material_channels[r,c])` at `t = T`, cast to `int` **[EG: bodies are integer material IDs]**.

**Genome.** The flat 792-length float vector `θ`. Nothing else. **[PH: one genome per robot; evolution acts on the genome, not the morphology]**

---

## Part 4 — Fairness protocol (must be enforced in code, not in prose)

These are the things that, if unequal, invalidate the comparison. Each should be asserted at run start and written into the run log.

| Axis | Rule |
|---|---|
| Budget | Equal number of **PPO evaluations**, not equal generations. Both runs terminate at the same evaluation count. This is the single most important control. |
| PPO config | Identical `total_timesteps`, policy, hyperparameters, eval protocol for every body in both arms **[EG: `run_ppo` via SB3 `MlpPolicy`]**. |
| Task | Identical env ID and `max_episode_steps` for both arms. |
| Seeds | Identical seed list across arms (e.g. seeds 0–4). Seed controls EA init, mutation stream, and PPO. |
| Invalid genomes | **One shared policy** applied identically to both encoders. **[DEC]** Recommend: an invalid body is assigned a fixed sentinel fitness and *still consumes an evaluation slot from the budget*, mirroring `bo/run.py`'s cost-`10` convention **[EG]**. Resample-until-valid (as `cppn_neat/population.py` does **[EG]**) silently gives a free advantage to whichever encoding produces more invalids — do not use it unless applied to both and counted against the budget. |
| Caching | Body-hash caching either on for both or off for both. Note it favours whichever encoding revisits bodies more often; report cache hit rates per arm. |
| Mutation strength | Not equalisable across a categorical and a continuous space. **Do not pretend it is.** Instead, tune each arm's mutation rate/σ on a *separate, smaller* pilot budget and report the tuning procedure — otherwise you are comparing your tuning effort, not the encodings. |

---

## Part 5 — Minimum credible experiment set

| ID | Experiment | Purpose |
|---|---|---|
| E0 | Development sanity: random NCA genomes → morphology validity rate, diversity, stability of `s(t)` | Confirms the developmental engine works at all. No PPO. Cheap. |
| E1 | Direct baseline on `Walker-v0`, 5 seeds | The control. |
| E2 | NCA on `Walker-v0`, 5 seeds, same budget | The comparison. |
| E3 | Second task (**[DEC]** recommend `UpStepper-v0` or `GapJumper-v0`) | Tests whether any observed difference is task-specific. |
| A1 | Ablation: `T ∈ {1, 5, 10, 20}` | `T=1` isolates whether *iteration* matters or only the network's expressiveness. |
| A2 | Ablation: `H ∈ {0, 3}` | Isolates whether hidden channels contribute. |
| A3 | Random-genome control: sample N random genomes per encoding, no evolution | Separates "the encoding's prior over morphologies" from "what evolution found". Often the most revealing and the cheapest. |

**Metrics to log per evaluation** (not just per generation): best-so-far fitness, evaluations consumed, wall-clock, body hash, validity flag, and morphology descriptors (occupied voxel count, actuator count, H/V actuator ratio, left–right symmetry score, number of distinct materials). Fitness-vs-evaluations curves with across-seed variance bands are the primary result figure; a bare "best fitness" table is not a credible result.

---

## Part 6 — Explicitly deferred (do not build)

Justified by `project-history.md` §16 and undergraduate scope:

- Regeneration / damage-recovery experiments (Horibe et al.'s headline property — interesting, but a separate paper).
- True voxel growth; the grid stays fixed **[PH]**.
- Material change during locomotion; development completes before PPO **[PH]**.
- Asynchronous or stochastic cell updates (Mordvintsev et al.'s 0.5 update probability) — synchronous only **[PH §13.7]**.
- Sample-pool / persistence training tricks (they assume a gradient signal; you have fitness only).
- Gradient-based NCA training. **[DEC]** If ever added, it is a *third* arm, not a modification of the evolutionary arm.
- Crossover for the NCA arm — mutation-only ES is sufficient and matches EvoGym's own GA example, which implements no crossover despite the header **[EG]**.
- Symbolic-rule encoder — your teammate's track. Build the `Encoder` ABC so it can drop in later as a third arm, but do not implement it.
- Custom connectivity topologies; always use `get_full_connectivity` **[EG]**.
- Multi-objective / QD (MAP-Elites) search.
- Distributed/cluster orchestration beyond simple process-level parallelism.

---

# Part 7 — Claude Code Prompt Sequence

Each prompt is self-contained, single-objective, and states its blast radius. Run them one at a time and confirm acceptance criteria before proceeding.

---

### Prompt 1 — Repository skeleton and reproducibility spine

```
Create the skeleton of a research repository for an Evolutionary-Developmental Robotics capstone. Do not implement any algorithm logic yet — only structure, configuration, and reproducibility infrastructure.

CREATE:
- Package `evodevo/` with subpackages: encoders/, morphology/, evaluation/, evolution/, experiments/, analysis/ (each with __init__.py).
- `evodevo/config.py`: a frozen dataclass `ExperimentConfig` with fields: seed:int, encoder:str, task:str, grid_shape:tuple=(5,5), population_size:int, offspring_size:int, eval_budget:int, ppo_timesteps:int, ppo_eval_episodes:int, output_dir:str. Include from_yaml() and to_dict().
- `evodevo/seeding.py`: `make_rng(seed)` returning a numpy Generator, plus `seed_everything(seed)` that also seeds python `random` and torch if present. All randomness in this project must flow from these.
- `evodevo/experiments/configs/` with two example YAML configs (direct_walker.yaml, nca_walker.yaml) that differ ONLY in the `encoder` field.
- `docs/decisions.md` with a table (Date | Decision | Options considered | Chosen | Rationale) and these pre-filled rows: neighbourhood type, hidden channel count, developmental steps T, initial state, invalid-genome policy, mutation tuning procedure. Mark all as "OPEN".
- `pyproject.toml` or `requirements.txt` pinning: numpy<2.0.0, gymnasium, stable-baselines3, pyyaml, pytest. Note that evogym must be installed separately from source.
- `tests/` with `test_config.py` and `test_seeding.py`.
- `README.md` stating the research question and the one-pipeline-two-decoders principle.

MUST NOT: install or modify anything inside the evogym source tree; write any encoder, EA, or PPO code.

TESTS: `pytest tests/ -q` passes.

ACCEPTANCE: config round-trips through YAML; `make_rng(42)` produces identical sequences across two calls; repo imports cleanly with `python -c "import evodevo"`.
```

---

### Prompt 2 — Morphology validation and metrics

```
Implement shared morphology validation and descriptive metrics. These are used identically by every encoding, so correctness here is critical.

CREATE:
- `evodevo/morphology/validate.py`:
  - `MATERIALS = {EMPTY:0, RIGID:1, SOFT:2, H_ACT:3, V_ACT:4}` (deliberately excluding FIXED=5, which EvoGym reserves for terrain).
  - `ValidationResult` dataclass: `valid: bool`, `reasons: list[str]`.
  - `validate_body(body, grid_shape=(5,5), min_voxels=3) -> ValidationResult` checking, in order: exact shape; integer dtype; all values in 0..4; at least `min_voxels` non-empty voxels; single connected component under CARDINAL adjacency; at least one actuator (3 or 4).
  - Implement connectivity and actuator checks independently in this file, then cross-check them against `evogym.is_connected` and `evogym.has_actuator` in tests (import guarded so tests skip if evogym is unbuilt).
- `evodevo/morphology/metrics.py`: `body_metrics(body) -> dict` returning occupied_count, empty_count, rigid_count, soft_count, h_act_count, v_act_count, actuator_fraction, horizontal_symmetry_score (fraction of cells matching their left-right mirror), vertical_symmetry_score, largest_component_size, and `body_hash(body) -> str` (stable hash of the byte representation).

MUST NOT: touch encoders, EA, PPO, or the evogym source tree.

TESTS: `tests/test_validate.py` covering: all-empty body rejected; disconnected two-blob body rejected; body with no actuator rejected; body containing value 5 rejected; wrong shape rejected; a known-good hand-written 5x5 walker accepted; single-voxel body rejected by min_voxels even though evogym's own helpers would accept it. `tests/test_metrics.py` covering symmetry scores on a hand-built symmetric and asymmetric body, and hash stability.

ACCEPTANCE: all tests pass; the `min_voxels` rationale is recorded in docs/decisions.md as a project-level constraint stricter than EvoGym's own.
```

---

### Prompt 3 — Encoder interface and direct encoding

```
Define the encoder contract that ALL representations in this project must satisfy, then implement the direct (non-developmental) baseline encoding.

CREATE:
- `evodevo/encoders/base.py`: abstract base class `Encoder` with:
  - `name: str` property
  - `genome_length() -> int`
  - `random_genome(rng) -> np.ndarray`
  - `decode(genome) -> np.ndarray`  # returns a (5,5) int array of material IDs
  - `mutate(genome, rng) -> np.ndarray`
  - `describe() -> dict`  # serialisable hyperparameters, written into every run log
  This interface is FROZEN after this prompt. Any later encoding (NCA, symbolic rules) must fit it without changing the signature.
- `evodevo/encoders/direct.py`: `DirectEncoder(grid_shape=(5,5), mutation_rate=0.1, empty_prob=0.6)`.
  - Genome: flat integer array of length 25, values 0..4.
  - `decode` reshapes to (5,5) — no other processing.
  - `mutate`: per-cell, with probability mutation_rate, resample from the 5 materials using a categorical distribution with EMPTY weighted at empty_prob (matching EvoGym's own GA example convention) and the remaining mass split evenly.
  - `random_genome`: same categorical sampling for all 25 cells.
  - IMPORTANT: mutate must NOT loop until validity. Validity handling is a pipeline concern, not an encoder concern, and must be identical across encodings.
- `evodevo/encoders/__init__.py`: `get_encoder(name, **kwargs)` registry.

MUST NOT: implement NCA; add any resample-until-valid logic; touch evaluation or EA code.

TESTS: `tests/test_direct_encoder.py`: decode returns correct shape and dtype; decode is deterministic for a fixed genome; mutate with rate 0.0 is identity; mutate with rate 1.0 and a fixed rng is reproducible; random_genome with the same seed twice is identical; genome_length()==25; describe() is JSON-serialisable.

ACCEPTANCE: tests pass; `get_encoder("direct")` works; the base class docstring explicitly states the invalid-genome policy belongs to the pipeline.
```

---

### Prompt 4 — PPO evaluation wrapper with a surrogate mode

```
Implement the single shared fitness evaluator. This is the only module allowed to import stable_baselines3 or evogym environments.

CREATE:
- `evodevo/evaluation/ppo_runner.py`:
  - `evaluate_body(body, task, ppo_timesteps, eval_episodes, seed, output_dir=None) -> EvalResult`
    - `EvalResult` dataclass: fitness:float, valid:bool, reason:str|None, wall_clock_s:float, body_hash:str.
    - Validate the body first using evodevo.morphology.validate. If invalid, return immediately with the sentinel fitness (configurable, default -10.0) WITHOUT training.
    - Compute `connections = evogym.get_full_connectivity(body)`.
    - Build a vectorised env via stable_baselines3 make_vec_env with the task ID and `env_kwargs={'body': body, 'connections': connections}`, matching EvoGym's own examples/ppo/run.py convention.
    - Train `PPO("MlpPolicy", ...)` for exactly ppo_timesteps.
    - Evaluate over eval_episodes fresh episodes; fitness = mean episode return.
    - Import `evogym.envs` at call time so environments register.
  - `evaluate_body_surrogate(body, ...) -> EvalResult`: a FAST, deterministic, physics-free stand-in used only for testing the EA loop. It must return a value derived from body_metrics (e.g. rewards actuator count and horizontal symmetry) and must be clearly documented as NOT a scientific result.
  - A `make_evaluator(mode)` factory returning either the real or surrogate function.
- `evodevo/evaluation/cache.py`: `FitnessCache` keyed on body_hash, with hit/miss counters exposed for logging.

MUST NOT: implement the EA; modify anything in the evogym source tree; hard-code actuator ordering (if actuator indices are ever needed, obtain them via `env.unwrapped.get_actuator_indices('robot')`).

TESTS: `tests/test_ppo_runner.py`:
- Surrogate mode: deterministic for a fixed body; invalid body returns sentinel fitness without attempting training.
- Real mode: a single smoke test with ppo_timesteps=2000 on Walker-v0 and a hand-written valid body, marked with `@pytest.mark.slow` and skipped if evogym.simulator_cpp is not importable.
- Cache: repeated identical bodies produce one miss and one hit.

ACCEPTANCE: surrogate tests pass in under a second; the slow smoke test completes and returns a finite fitness when evogym is built; no module outside evaluation/ imports stable_baselines3.
```

---

### Prompt 5 — Encoding-agnostic evolutionary loop

```
Implement the evolutionary algorithm. It must be entirely encoding-agnostic: it may only call methods on the Encoder interface and the evaluator function.

CREATE:
- `evodevo/evolution/ea.py`: `run_ea(encoder, evaluator, config) -> RunResult`
  - (mu + lambda) scheme: initialise `population_size` random genomes, evaluate all, then repeatedly select the top mu as survivors, produce lambda offspring by mutation of randomly chosen survivors, evaluate offspring only.
  - TERMINATION IS ON EVALUATION COUNT, NOT GENERATIONS. Stop as soon as evaluations_used >= config.eval_budget. Invalid genomes consume a budget slot. This is a deliberate fairness control and must be enforced here, not left to the caller.
  - Survivors are never re-evaluated (their fitness is carried forward), matching EvoGym's own GA example.
  - No crossover.
- `evodevo/evolution/logging.py`: `RunLogger` writing newline-delimited JSON, one record PER EVALUATION with: eval_index, generation, genome_id, parent_id, fitness, valid, reason, body_hash, wall_clock_s, and all body_metrics. Also write a `run_meta.json` containing the full config dict, encoder.describe(), git commit hash, and library versions.

MUST NOT: import evogym, stable_baselines3, or any specific encoder class. The EA must be testable with the surrogate evaluator and the direct encoder alone.

TESTS: `tests/test_ea.py`:
- Runs to completion with the direct encoder + surrogate evaluator under a small budget.
- Evaluation count at termination equals eval_budget exactly (not more).
- Two runs with the same seed produce byte-identical JSONL logs.
- Two runs with different seeds differ.
- Best-so-far fitness is monotonically non-decreasing.
- An encoder whose decode always returns an invalid body still terminates cleanly at the budget.

ACCEPTANCE: all tests pass; `grep -r "evogym\|stable_baselines3" evodevo/evolution/` returns nothing.
```

---

### Prompt 6 — Experiment entry point and the direct baseline run

```
Wire everything into a single reproducible entry point and produce the direct-encoding baseline.

CREATE:
- `evodevo/experiments/run_experiment.py`: CLI taking `--config path.yaml --seed N [--surrogate]`. It must construct the encoder from the registry, build the evaluator, run the EA, and write all outputs to `output_dir/{encoder}/{task}/seed_{N}/`. It must refuse to start if the output directory already contains a completed run unless `--overwrite` is passed (EvoGym's own examples use interactive prompts here; do not — this must be non-interactive and script-safe).
- `scripts/run_baseline.sh`: runs the direct encoder on Walker-v0 for seeds 0-4.
- `evodevo/analysis/aggregate.py`: load one or many run directories into a tidy pandas DataFrame (one row per evaluation, columns including encoder, task, seed).
- `evodevo/analysis/plots.py`: `plot_fitness_curves(df)` producing best-so-far fitness vs evaluations, one line per encoder, shaded across-seed min/max band.

MUST NOT: implement the NCA encoder. Do not tune PPO hyperparameters yet.

TESTS: `tests/test_run_experiment.py`: end-to-end run in surrogate mode with a tiny budget produces run_meta.json and a JSONL log with exactly eval_budget records; aggregate.py loads it; rerunning without --overwrite raises.

ACCEPTANCE (this is the scientific gate — do not proceed to the NCA until all hold):
1. Five real-PPO seeds complete without crashing.
2. Best-so-far fitness curves increase above the random-initial-population mean on at least 4 of 5 seeds.
3. The best evolved body is a plausible walker (occupied voxels >= 3, has actuators) and is saved as a .npz with body and connections.
4. Wall-clock per evaluation is measured and recorded, so the full comparison budget can be estimated before committing to it.
5. Write the observed per-evaluation cost and projected total experiment cost into docs/decisions.md.
```

---

### Prompt 7 — NCA encoder: development engine

```
Implement the Neural Cellular Automata developmental encoder. It must satisfy the existing frozen Encoder interface without modifying base.py.

CREATE:
- `evodevo/encoders/nca.py`: `NCAEncoder(grid_shape=(5,5), hidden_channels=3, hidden_width=16, dev_steps=10, neighbourhood="von_neumann", init_scale=0.5, mutation_sigma=0.1, mutation_fraction=1.0)`.

SPECIFICATION (implement exactly this; deviations must be raised, not silently made):
- Cell state length C = 5 + hidden_channels. Channels 0..4 are material logits for {EMPTY,RIGID,SOFT,H_ACT,V_ACT}; the rest are hidden.
- Perception: concatenate own state with the 4 cardinal neighbour states, zero-padded outside the grid. Input dimension = 5*C.
- Network: Linear(5C -> hidden_width) -> tanh -> Linear(hidden_width -> C). Implement in pure NumPy (no torch) — it is small, and this keeps the developmental rollout dependency-free and fast.
- Update: RESIDUAL and SYNCHRONOUS. s(t+1) = s(t) + f(perception(s(t))), all 25 cells computed from s(t) before any is written. Assert synchronicity in a test.
- Initial state t=0: a FIXED, deterministic, non-symmetric array identical for every individual (do not evolve it — genome is weights-only). Expose it as a module-level constant with a comment explaining that symmetric initial conditions can prevent differentiation.
- Readout at t=dev_steps: body[r,c] = argmax over channels 0..4, as int.
- Genome: the flat float vector of all weights and biases. genome_length() must be computed, not hard-coded.
- random_genome: Gaussian scaled by init_scale.
- mutate: add N(0, mutation_sigma) to a randomly chosen `mutation_fraction` of entries. No resample-until-valid.
- Also expose `develop(genome, return_trajectory=True)` returning the list of intermediate (5,5) argmax bodies — needed for the sanity experiment and for figures.

MUST NOT: modify encoders/base.py, the EA, the evaluator, or the validator. Do not add growth, alive/dead masking, asynchronous updates, or gradient training.

TESTS: `tests/test_nca_encoder.py`:
- decode returns a (5,5) int array with values in 0..4.
- Determinism: same genome -> identical body, twice.
- Synchronicity: a hand-constructed genome/state where a sequential (asynchronous) update would give a different result than a synchronous one; assert the synchronous answer.
- Boundary handling: corner and edge cells receive zero-padded neighbours, verified against a hand-computed perception vector.
- genome_length matches the analytic parameter count for several (hidden_channels, hidden_width) settings.
- Residual property: with all weights zero, the body at t=T equals the body read from the fixed initial state.
- develop() returns dev_steps+1 trajectory entries.

ACCEPTANCE: all tests pass; `get_encoder("nca")` works; the EA from Prompt 5 runs unmodified with the NCA encoder in surrogate mode.
```

---

### Prompt 8 — NCA sanity experiment (E0, no PPO)

```
Before spending any PPO compute on the NCA arm, characterise the developmental engine itself. This experiment is cheap and must pass before the comparison runs.

CREATE:
- `evodevo/experiments/nca_sanity.py`: sample N=2000 random NCA genomes (seeded) and report:
  1. Validity rate under validate_body, with a breakdown by failure reason.
  2. Morphological diversity: number of unique body hashes.
  3. Developmental stability: for each genome, the number of steps after which the argmax body stops changing; report the distribution and the fraction that never settle within dev_steps.
  4. Degeneracy check: fraction of genomes producing a uniform body (all one material) or an all-empty body.
  5. State magnitude: max absolute cell-state value at t=T, to detect divergence from the residual update.
  6. The same statistics for the DIRECT encoder's random genomes, as a side-by-side reference (this is ablation A3, the random-genome control).
- Output: a markdown report to `results/nca_sanity.md` plus the raw stats as JSON, and a figure showing developmental trajectories for 5 example genomes (t=0..T grids).

MUST NOT: call PPO; modify the NCA encoder to improve these numbers before reporting the as-designed values first.

TESTS: reruns with the same seed reproduce identical statistics.

ACCEPTANCE / DECISION GATE:
- If validity rate is below ~10%, or over ~50% of genomes produce degenerate uniform bodies, STOP and revisit init_scale / initial state / hidden_channels BEFORE running the comparison. Record the original numbers and any change made in docs/decisions.md — the pre-fix numbers are themselves a reportable finding about the encoding's prior.
- If a large fraction never settle within dev_steps, record it; this is the developmental-instability limitation and belongs in the final report either way.
```

---

### Prompt 9 — Fairness harness and the head-to-head comparison

```
Add explicit enforcement of the fairness protocol, then execute the main comparison.

CREATE:
- `evodevo/experiments/fairness.py`: `assert_comparable(config_a, config_b)` raising if the two configs differ in any of: task, eval_budget, ppo_timesteps, ppo_eval_episodes, grid_shape, population_size, offspring_size, invalid-genome sentinel value, or cache mode. It must permit differences ONLY in `encoder` and encoder-specific hyperparameters. Call this at the start of any comparison run.
- `scripts/run_comparison.sh`: direct and NCA, Walker-v0, seeds 0-4, identical budgets, invoking assert_comparable first.
- Extend `evodevo/analysis/aggregate.py` with `comparison_table(df)` reporting per encoder: final best fitness (mean ± std across seeds), evaluations to reach 50%/90% of the direct arm's final best, validity rate, cache hit rate, unique bodies explored, and mean wall-clock per evaluation.
- Extend plots.py with a morphology-metric comparison figure (actuator fraction, occupied count, symmetry score for best bodies per arm).

MUST NOT: change any encoder, the EA, or the evaluator during or after seeing comparison results. If a bug is found, fix it and RERUN BOTH ARMS from scratch — never patch one arm's results.

TESTS: `tests/test_fairness.py`: assert_comparable passes for two configs differing only in encoder; raises for configs differing in ppo_timesteps, eval_budget, or sentinel value.

ACCEPTANCE:
1. Both arms complete 5 seeds at identical evaluation counts.
2. comparison_table() runs and produces the table.
3. The report explicitly states the result direction — including if the direct encoding wins or if the arms are statistically indistinguishable. A null or negative result is a valid outcome and must be reported as such, per the project's stated falsifiability commitment.
4. Report across-seed variance, not just means; with 5 seeds, avoid strong claims and say so.
```

---

### Prompt 10 — Ablations and second task

```
Run the minimal ablation set needed to interpret the comparison result.

CREATE:
- `scripts/run_ablations.sh` covering:
  - A1: NCA with dev_steps T in {1, 5, 10, 20}, 3 seeds each. T=1 is the key control: it isolates whether ITERATED development matters, or whether the network's expressiveness alone explains any effect.
  - A2: NCA with hidden_channels in {0, 3}, 3 seeds each.
  - E3: both encoders on a second task (UpStepper-v0 or GapJumper-v0), 3 seeds each, same budget rules.
- `evodevo/analysis/ablation_report.py` producing `results/ablations.md` with one table per ablation axis and the corresponding figures.

MUST NOT: introduce new encodings, change the EA, or alter the PPO budget between ablation arms.

ACCEPTANCE:
- Each ablation reports fitness AND the E0-style descriptive statistics (validity rate, unique bodies, settling behaviour), since an ablation that changes the encoding's prior over morphologies explains fitness differences better than fitness alone does.
- The report explicitly separates: what the data shows, what remains a hypothesis, and what a larger study would need to resolve.
```

---

### Prompt 11 — Final reproducibility and results package

```
Freeze the project into a reproducible, submittable state.

CREATE / UPDATE:
- `results/RESULTS.md`: research question, method summary, fairness protocol as actually enforced, all figures, all tables, and an explicit "threats to validity" section covering: small seed count, PPO budget below EvoGym's own recommended >1e6 for serious runs, single-task-family evaluation, per-arm mutation tuning asymmetry, and the fact that a 5x5 grid may be too small for developmental encodings to show their advantages.
- `REPRODUCE.md`: exact commands, environment versions, evogym commit, and expected wall-clock, sufficient for someone to rerun from scratch.
- Complete `docs/decisions.md`: every row resolved, no "OPEN" entries remaining.
- Ensure every figure in RESULTS.md is regenerable by a single script from the raw JSONL logs.

MUST NOT: adjust any result, re-run a single arm to improve a number, or soften a negative finding.

ACCEPTANCE: a clean checkout plus REPRODUCE.md reproduces every figure from the archived logs; docs/decisions.md has zero OPEN rows; RESULTS.md states the finding in one sentence at the top, whichever direction it points.
```

---

## Part 8 — Recommended execution order

```
1  → Repo skeleton and reproducibility spine
2  → Morphology validation and metrics
3  → Encoder interface + direct encoding
4  → PPO evaluation wrapper (+ surrogate)
5  → Encoding-agnostic EA loop
6  → Experiment entry point + DIRECT BASELINE RUN     ◄── HARD GATE
7  → NCA encoder (development engine)
8  → NCA sanity experiment E0 (no PPO)                ◄── HARD GATE
9  → Fairness harness + head-to-head comparison
10 → Ablations + second task
11 → Final reproducibility and results package
```

**The two hard gates matter more than anything else in this plan.**

Gate at step 6: if the direct baseline does not evolve, the pipeline is broken, and any NCA number produced afterwards is meaningless. Do not proceed on hope.

Gate at step 8: if random NCA genomes almost never produce valid morphologies, the comparison will measure your invalid-genome policy rather than the encoding. Diagnose it with cheap, PPO-free experiments before spending the compute.

**Realistic parallelisation for a team:** prompts 1–3 and prompt 4 can proceed concurrently (the surrogate evaluator decouples them); the teammate's symbolic-rule encoder can be built against the frozen `Encoder` ABC any time after prompt 3, and slots in as a third arm at prompt 9 with no changes to the pipeline.
