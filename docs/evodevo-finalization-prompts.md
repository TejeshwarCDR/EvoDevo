# EvoDevo — Finalization Prompt Sequence for Claude Code

**Basis:** `REPO_INVENTORY_REPORT.md`, `project-history.md`, `evogym_technical_reference.md`, `evodevo-implementation-roadmap.md`.

Run these **one at a time**, confirming acceptance criteria before proceeding. Prompts 1–4 are structural and must be sequential. Prompts 5–8 finalize.

---

## Design decisions this sequence commits to

These resolve the blockers the inventory identified. They are stated here so you can veto any of them before you start running prompts.

| # | Blocker (from inventory) | Resolution baked into these prompts |
|---|---|---|
| 1 | Project root is not a git repo (tracked root is `$HOME`) | Prompt 1 initializes a real repo at the project root, before anything moves. Non-negotiable and first. |
| 2 | Rule-Based needs `gym==0.22` + Python 3.7; evodevo needs `gymnasium` + 3.10 | **Do not merge environments.** The rule-based *encoding logic* (`developmental_rule_types/encoding/interpreter`) is pure Python + NumPy + three EvoGym helper functions — it never touches `gym`. Only the teammate's *GA/PPO stack* needs the old environment, and we are not using that stack. Port the three encoding modules in; leave the rest as an archived reference checkout. |
| 3 | Rule-Based doesn't implement `Encoder` | Prompt 4 writes a `RuleBasedEncoder` adapter. Genome is the rule list; `decode()` runs the existing interpreter unchanged. |
| 4 | Crossover asymmetry (Rule-Based has it, Direct/NCA don't) | Primary 3-way comparison is **mutation-only for all three arms**. Rule-Based-with-crossover runs as a separate, explicitly labelled ablation arm. The ABC gains an optional `crossover()` defaulting to `None`. |
| 5 | Invalid-genome accounting differs (consume budget vs. retry-free) | Single shared policy in `ea.py`: invalid genomes consume budget and receive sentinel fitness. Applies to all three arms identically. The retry-based accounting is not reproduced. |
| 6 | Budgets differ 3×/15×/17× | All prior comparative numbers are treated as **unusable**. Every arm is re-run under one config. Prompt 8 executes this. |
| 7 | NCA has no real-PPO runs | Prompt 8 runs it. Until then no comparative claim exists. |
| 8 | Rule-Based `saved_data/` artifacts missing | Not recoverable. Prior reports are archived as provisional and superseded, not cited as results. |

---

# PROMPT 1 — Version control and workspace restructure

```
Establish version control and restructure this project into a clean modular workspace. This prompt performs MOVES and RENAMES only — do not change any logic, fix any bug, or alter any algorithm behaviour. Behaviour must be byte-identical before and after, except for import paths.

CRITICAL FIRST STEP — version control:
The project root is currently NOT a git repository; `git rev-parse --show-toplevel` resolves to the user's home directory with zero commits, meaning nothing here is tracked. Before touching a single file:
1. Verify this is still the case. If it is, run `git init` at the project root.
2. Write a `.gitignore` covering: .venv/, __pycache__/, *.pyc, *.so, build/, dist/, *.egg-info/, .pytest_cache/, results/**/controller/*.zip, and any large simulator build artifacts.
3. Make an initial commit of the CURRENT state, before any reorganization, so the pre-reorg state is recoverable.
4. Commit after the restructure as a second, separate commit.
Do NOT run `git init` in the home directory and do NOT modify the home directory's existing git state in any way.

TARGET STRUCTURE:
evodevo/
  core/                      # shared, encoding-agnostic — changing this affects all three arms
    config.py                # moved from evodevo/config.py
    seeding.py               # moved from evodevo/seeding.py
    materials.py             # NEW FILE, empty stub with a TODO comment only — populated in Prompt 2
  morphology/                # unchanged location: validate.py, metrics.py
  encoders/
    base.py
    registry.py              # registry logic moved out of __init__.py into its own module
    direct/                  # each encoder becomes a PACKAGE, not a single file
      __init__.py            # exports the encoder class
      encoder.py             # moved from encoders/direct.py
    nca/
      __init__.py
      encoder.py             # moved from encoders/nca.py
    rule_based/
      __init__.py            # stub only — populated in Prompt 4
  evaluation/                # unchanged: ppo_runner.py, cache.py
  evolution/                 # unchanged: ea.py, logging.py
  experiments/
    run_experiment.py
    fairness.py
    configs/
      shared/                # NEW: shared base config fragments
      direct/                # per-encoder config dirs
      nca/
      rule_based/
  analysis/                  # unchanged: aggregate.py, plots.py, ablation_report.py
  visualization/             # NEW empty package with __init__.py — populated in Prompt 7
tests/
  core/                      # test_config.py, test_seeding.py
  morphology/                # test_validate.py, test_metrics.py
  encoders/
    test_contract.py         # NEW empty stub — populated in Prompt 3
    test_direct.py           # moved from test_direct_encoder.py
    test_nca.py              # moved from test_nca_encoder.py
  evaluation/                # test_ppo_runner.py
  evolution/                 # test_ea.py
  experiments/               # test_fairness.py, test_run_experiment.py
  analysis/                  # test_ablation_report.py
  integration/               # NEW empty dir with .gitkeep
scripts/
  direct/                    # per-method scripts
  nca/
  rule_based/
  benchmark/                 # cross-method scripts; run_comparison.sh moves here
docs/
external/
  evogym/                          # the vendored EvoGym library, moved from repo root
  rule-based-reference/            # the teammate's checkout, moved from repo root and RENAMED
results/                     # unchanged location

MOVES REQUIRED:
- Root `evogym/` -> `external/evogym/`. Preserve the compiled simulator and its own .git if present.
- Root `evogym-rule-based-encoding/` -> `external/rule-based-reference/`. It has its OWN .git with real history — preserve it intact, do not delete it, do not merge its history, and add it to the outer .gitignore so the outer repo does not try to track its internals. Add `external/rule-based-reference/ARCHIVED.md` stating: this is a read-only reference checkout of the teammate's original codebase, retained for provenance; the active rule-based implementation lives in evodevo/encoders/rule_based/; its PROJECT_REPORT.md / FINAL_THREE_WAY_COMPARISON_REPORT.md describe runs under different budgets and their supporting saved_data/ artifacts are absent from this checkout, so their numbers are provisional and superseded.

RULES:
- Use `git mv` for every move so history is preserved from the initial commit forward.
- Update every import statement affected by the moves. After this prompt, `grep -rn "from evodevo.config\|from evodevo.seeding\|encoders.direct import\|encoders.nca import" evodevo/ tests/ scripts/` must return only correct new-style paths.
- `pytest.ini` must be updated for the new tests/ layout; keep the `slow` marker registration.
- Do not touch anything inside external/ other than moving it and adding ARCHIVED.md.
- Do not delete any file. If something looks orphaned, leave it and note it.

TESTS: `pytest -q` must pass with the same number of passing tests as before the restructure. Record the before/after counts in the commit message.

ACCEPTANCE:
1. `git log --oneline` at the project root shows exactly two commits and `git rev-parse --show-toplevel` resolves to the project root, not $HOME.
2. Full test suite passes with an unchanged pass count.
3. `python -c "import evodevo"` succeeds.
4. `python -m evodevo.experiments.run_experiment --help` succeeds.
5. No file was deleted; `git status` is clean.
```

---

# PROMPT 2 — Harden the shared core

```
Fix the shared-core defects the repository inventory identified. Scope is strictly evodevo/core/, evodevo/morphology/, evodevo/evaluation/, and evodevo/analysis/plots.py. Do not touch any encoder, the EA loop's selection logic, or anything in external/.

1. SINGLE SOURCE OF TRUTH FOR MATERIALS
   Populate `evodevo/core/materials.py` as the one authoritative definition of the voxel vocabulary: EMPTY=0, RIGID=1, SOFT=2, H_ACT=3, V_ACT=4, an explicit N_MATERIALS=5, a MATERIAL_NAMES tuple for plotting/logging, an ACTUATORS frozenset {3,4}, and a documented note that FIXED=5 is EvoGym terrain material and is never valid in an evolved robot body.
   Then make `morphology/validate.py`, `morphology/metrics.py`, and both encoders import from it instead of redefining constants locally. Currently the vocabulary is defined independently in at least three places (evodevo's MATERIALS dict, direct.py's N_MATERIALS, nca.py's N_MATERIALS) — after this change there must be exactly one definition.

2. DE-DUPLICATE CONNECTIVITY
   `_largest_component_size` is duplicated verbatim between morphology/validate.py and morphology/metrics.py. Move one canonical implementation into `evodevo/morphology/connectivity.py` and have both import it. Behaviour must be identical; verify with existing tests.

3. WIRE UP OR REMOVE FitnessCache
   `evaluation/cache.py::FitnessCache` is implemented, unit-tested, and has a `cache_enabled` config flag, but no call site — it currently has zero effect on any run. Wire it into the evaluation path in `evolution/ea.py` so that when `cache_enabled` is true, a body whose hash was already evaluated reuses the stored fitness.
   CRITICAL FAIRNESS RULE: a cache hit must still consume one evaluation from the budget, and must be logged with a `cache_hit: true` field. If a cache hit were free, whichever encoding revisits bodies more often would gain a silent budget advantage — that would confound exactly the search-efficiency comparison this project exists to make. Log per-run cache hit rate into run_meta.json so any residual effect is visible in the results.
   Keep `cache_enabled: false` as the default for the primary comparison.

4. HEADLESS PLOTTING
   `analysis/plots.py` produces figures but never sets the matplotlib backend; it currently works only by accident, via an import-time side effect in nca_sanity.py. Set `matplotlib.use("Agg")` before importing pyplot in EVERY module under analysis/ and any future module that renders. Add a test asserting plots.py can be imported and can save a figure in a subprocess with no DISPLAY set.

5. VALIDATION POLICY RECONCILIATION
   The rule-based arm (to be ported in Prompt 4) uses a different validity floor: it requires >= 2 actuators, while evodevo's validate_body requires >= 3 occupied voxels and >= 1 actuator. Extend `validate_body` with explicit, config-driven parameters `min_voxels` (default 3) and `min_actuators` (default 1), so a single function can express both policies and the chosen policy is recorded in the run config rather than hard-coded per arm. Do NOT change the default behaviour — existing tests must pass unchanged.

MUST NOT: change EA selection logic, encoder behaviour, PPO hyperparameters, or the sentinel-fitness value.

TESTS: all existing tests pass unchanged. Add: test_materials.py (single-source assertions, e.g. no module outside materials.py defines a material integer literal — enforce with a grep-style test); test_connectivity.py; a cache-integration test in tests/evolution/ asserting a cache hit still increments evaluations_used and is flagged in the log; the headless-plotting subprocess test.

ACCEPTANCE:
1. `grep -rn "H_ACT.*=.*3\|N_MATERIALS *= *5" evodevo/ --include=*.py` returns hits only in core/materials.py.
2. Full suite green.
3. A run with cache_enabled=true and cache_enabled=false consumes the identical number of evaluations for the same seed.
```

---

# PROMPT 3 — Encoder plugin architecture and conformance contract

```
Turn the encoder layer into a true plugin system with a shared conformance test suite. This is the mechanism that lets team members work on different encodings without colliding: the contract is central and stable; each encoding is an isolated package that must pass the same tests.

1. EXTEND THE ENCODER ABC (evodevo/encoders/base.py)
   Keep all five existing abstract members with unchanged signatures — DirectEncoder and NCAEncoder must continue to satisfy the interface without modification. Add:
   - `supports_crossover() -> bool`, defaulting to False.
   - `crossover(parent_a, parent_b, rng) -> genome`, a non-abstract method raising NotImplementedError by default.
   - `genome_dtype` property, so the EA and logger can handle both fixed-shape numeric genomes (direct, nca) and variable-length object genomes (rule-based) without special-casing.
   - `serialize_genome(genome) -> JSON-serialisable object` and `deserialize_genome(obj) -> genome`. Required because the rule-based genome is a variable-length list of rule objects that cannot be logged as a numpy array. Implement these concretely for DirectEncoder and NCAEncoder.
   Document in the class docstring that invalid-genome handling remains a pipeline concern and encoders must never resample-until-valid.

2. REGISTRY (evodevo/encoders/registry.py)
   Registry maps name -> encoder class, with `register_encoder(name, cls)`, `get_encoder(name, **kwargs)`, and `list_encoders()`. Registration must happen via explicit import in `evodevo/encoders/__init__.py` — no dynamic filesystem scanning, which would make failures silent and hard to debug.

3. CONFORMANCE SUITE (tests/encoders/test_contract.py)
   Write ONE parametrized test module that runs against EVERY encoder in the registry automatically, so a new encoding is validated the moment it registers. It must assert, for each encoder:
   - decode() returns a (5,5) integer array with all values in 0..4, never 5.
   - decode() is deterministic: the same genome yields an identical body twice.
   - random_genome() with the same rng seed is reproducible, and with different seeds differs.
   - mutate() with a fixed seed is reproducible.
   - mutate() never returns a genome that fails deserialize(serialize(g)) round-trip equality.
   - mutate() does not loop until valid: statistically, over 500 random genomes, at least one decodes to an invalid body (if an encoder produces 100% valid bodies this test should warn loudly, since it usually means hidden repair logic).
   - describe() is JSON-serialisable and includes every hyperparameter that affects decode().
   - serialize/deserialize round-trips exactly.
   - The encoder module does not import evogym, stable_baselines3, gym, or torch — assert by inspecting the module's imports. Encoders must stay physics-free and RL-free.

4. Move the existing per-encoder tests to only cover encoder-SPECIFIC behaviour (e.g. NCA synchronicity, residual property, boundary zero-padding; direct's reshape identity), deleting assertions now covered generically by the contract suite.

5. DEVELOPER DOCUMENTATION
   Write `docs/adding-an-encoder.md`: the exact steps to add a new encoding (create package under encoders/, implement the ABC, register it, add config dir, run the contract suite), with the explicit rule that a contributor must not modify base.py, registry.py, ea.py, ppo_runner.py, or morphology/ to make their encoder work — if they need to, that is a signal the contract is wrong and should be discussed, not worked around.

MUST NOT: change DirectEncoder or NCAEncoder decode/mutate behaviour. Their outputs for a given seed must be bit-identical before and after this prompt — add a regression test pinning a few known (seed -> body) pairs to prove it.

TESTS: contract suite passes for direct and nca; regression test pins existing behaviour; full suite green.

ACCEPTANCE: adding a deliberately broken dummy encoder to the registry causes the contract suite to fail with a clear message naming the violated clause.
```

---

# PROMPT 4 — Port the Rule-Based encoding as a third arm

```
Integrate the teammate's symbolic developmental-rules encoding as a first-class third arm, conforming to the Encoder contract.

KEY CONSTRAINT — read carefully before starting:
The reference checkout at external/rule-based-reference/ requires gym==0.22.0 and Python 3.7.11, which is incompatible with this project's gymnasium + Python 3.10 environment. DO NOT attempt to merge the environments and DO NOT install its requirements.
However, its ENCODING logic is dependency-light: `examples/utils/developmental_rule_types.py`, `developmental_rule_encoding.py`, and `developmental_interpreter.py` depend only on Python stdlib, numpy, and three EvoGym helper functions (is_connected, has_actuator, get_full_connectivity) — none of which touch the gym/gymnasium API. Only the teammate's GA and PPO stack needs the old environment, and we are not using that stack; we use this project's ea.py and ppo_runner.py.
Therefore: PORT the three encoding modules, leave everything else in external/ untouched.

1. PORT (evodevo/encoders/rule_based/)
   - `rule_types.py`, `rule_encoding.py`, `interpreter.py` — ported from the V2 developmental_* files (NOT the V1 static rule files, which are a superseded earlier iteration).
   - Change ONLY what is necessary: import paths, and replacing the locally redefined `VoxelState` vocabulary with an import from `evodevo.core.materials`. The report notes VoxelState uses the same integers (0..4) under different names (SOFT/HARD/ACTUATOR_HORIZONTAL/ACTUATOR_VERTICAL) — map them explicitly and add a test asserting the numeric mapping is unchanged from the original, so the port provably preserves semantics.
   - Do NOT redesign the rule language, alter the operators, or change MIN_RULES/MAX_RULES/iteration caps. This is a port, not a rewrite. Preserve the two-pass (self-actions then GROW_DIRECTIONAL) synchronous update and the single-SOFT-seed-at-centre initial condition exactly.
   - Add a header comment to each ported file crediting the original author and citing the source path and commit hash from external/rule-based-reference/.

2. ADAPTER (evodevo/encoders/rule_based/encoder.py, class RuleBasedEncoder)
   - Genome: the variable-length ordered list of DevelopmentalRule objects.
   - `decode(genome)`: run the ported interpreter's develop_rules for `developmental_iterations` steps and return the (5,5) int body. If the interpreter's generate function returns None on failure, decode must instead return whatever grid it produced — decode MUST return a body and MUST NOT signal validity, because validity is the pipeline's job and returning None here would break contract parity with the other two arms.
   - `mutate`: wrap `mutate_developmental_rules` (7 structural operators, rate 0.2).
   - `crossover` + `supports_crossover() -> True`: wrap `crossover_developmental_rules`.
   - `serialize_genome`/`deserialize_genome`: rules to/from JSON dicts. This is required for logging and must round-trip exactly.
   - `describe()`: must expose developmental_iterations, mutation_rate, MIN_RULES, MAX_RULES, and the operator set.
   - Register as "rule_based".

3. CROSSOVER PARITY DECISION
   The primary three-way comparison runs MUTATION-ONLY for all three arms, because Direct and NCA have no crossover and an asymmetric reproduction operator would confound the representation comparison. `ea.py` must therefore take an explicit `use_crossover` config flag defaulting to False, and must assert at startup that if use_crossover is True the encoder reports supports_crossover(). Rule-Based-with-crossover becomes a separate labelled ablation arm ("rule_based_xover"), not the primary arm.
   Record this in docs/decisions.md with the rationale.

4. CONFIGS: create evodevo/experiments/configs/rule_based/ mirroring the direct/ and nca/ configs exactly — identical population_size, offspring_size, eval_budget, ppo_timesteps, ppo_eval_episodes, grid_shape, sentinel_fitness, cache_enabled. Only `encoder` and encoder-specific hyperparameters may differ.

MUST NOT: modify external/rule-based-reference/ in any way; import gym; install its environment; change base.py, ea.py selection logic (beyond the use_crossover flag), ppo_runner.py, or the other two encoders.

TESTS:
- The generic contract suite from Prompt 3 must pass for rule_based with no special-casing.
- tests/encoders/test_rule_based.py: a semantic-equivalence test that runs the PORTED interpreter and the ORIGINAL interpreter (imported directly from external/rule-based-reference by file path, guarded to skip if unavailable) on the same 20 seeded genomes and asserts identical bodies. This is the proof that the port preserves behaviour — it is the single most important test in this prompt.
- Serialization round-trip on genomes containing every rule/action type.
- A run of ea.py with the rule_based encoder in surrogate mode completes at the exact eval budget.

ACCEPTANCE:
1. `list_encoders()` returns direct, nca, rule_based.
2. Contract suite green for all three.
3. Port-equivalence test passes (or is explicitly skipped with a loud reason, in which case say so in the summary — a skipped equivalence test means the port is unverified).
4. `grep -rn "^import gym$\|from gym import" evodevo/` returns nothing.
```

---

# PROMPT 5 — Isolated per-method pipelines

```
Give each of the three methods a self-contained run/test/inspect pipeline, so a team member can iterate on their own arm without running or breaking the others.

1. PER-METHOD ENTRY POINTS
   `scripts/{direct,nca,rule_based}/run.sh`   — single-method EA run over a seed range, taking config path and seeds as arguments.
   `scripts/{direct,nca,rule_based}/test.sh`  — runs ONLY that method's tests: the contract suite parametrized to that encoder, plus its method-specific tests, plus the shared core tests it depends on. Must complete in under ~60 seconds using surrogate evaluation.
   `scripts/{direct,nca,rule_based}/inspect.sh` — PPO-free characterization for that encoder: sample N random genomes, report validity rate with failure-reason breakdown, unique-body count, degeneracy rate, and (for developmental encoders) settling behaviour, writing to results/inspection/{encoder}/.
   Generalize the existing `evodevo/experiments/nca_sanity.py` into `evodevo/experiments/inspect_encoder.py`, which works for ANY registered encoder rather than being NCA-specific. Keep nca_sanity.py as a thin deprecated shim that calls it, so existing results remain reproducible.

2. MAKEFILE at project root with targets: `make test` (everything), `make test-direct`, `make test-nca`, `make test-rule-based`, `make test-core`, `make inspect-<method>`, `make benchmark`, `make docs`, `make clean`. Each target must print what it is about to do.

3. OWNERSHIP BOUNDARIES
   Create `docs/module-ownership.md` mapping directories to owners: encoders/nca/* -> Tejeshwar; encoders/rule_based/* -> teammate; encoders/direct/*, core/, morphology/, evaluation/, evolution/, analysis/, visualization/ -> shared/joint-review. State the rule plainly: changes to shared modules require agreement from both owners because they alter every arm's results, whereas changes inside an encoder package affect only that arm. Add a CONTRIBUTING.md summarizing the workflow: branch per method, run `make test-<your-method>` before pushing, run `make test` before merging.

4. CONFIG INHERITANCE
   Implement a base/override config mechanism: `configs/shared/base.yaml` holds every parameter that MUST be identical across arms (population_size, offspring_size, eval_budget, ppo_timesteps, ppo_eval_episodes, grid_shape, sentinel_fitness, cache_enabled, min_voxels, min_actuators, use_crossover); each per-method config inherits from it and may only add encoder-specific keys. Loading a per-method config that overrides a shared-base key must RAISE with a message naming the offending key. This makes the fairness protocol structurally impossible to violate by editing a config, rather than merely checked after the fact.
   Refactor fairness.py's assert_comparable to build on this rather than duplicate it, keeping it as a second line of defence at run time.

MUST NOT: change encoder or EA behaviour. This prompt is tooling and configuration only.

TESTS: tests/experiments/test_config_inheritance.py — a per-method config overriding a shared key raises; a valid per-method config loads and equals the expected merged dict; all three methods' configs load and produce configs that pass assert_comparable pairwise.

ACCEPTANCE:
1. `make test-nca` passes and does not execute rule_based or direct-specific tests.
2. `make test` passes.
3. All three `inspect.sh` scripts produce a report without invoking PPO.
4. Attempting to set a different ppo_timesteps in configs/nca/ fails loudly at load time.
```

---

# PROMPT 6 — Unified three-way benchmark pipeline

```
Build the single pipeline that runs and compares all three arms under identical conditions, and the analysis layer that reports the result.

1. BENCHMARK ORCHESTRATOR (evodevo/experiments/benchmark.py)
   CLI taking a benchmark config listing: encoders (direct, nca, rule_based), tasks, seed list, and the shared base config. For each (encoder, task, seed) it runs the standard EA pipeline and writes to results/runs/{encoder}/{task}/seed_{n}/.
   It must, BEFORE running anything: call assert_comparable pairwise across all three resolved configs and abort on any mismatch; write a `benchmark_manifest.json` recording the resolved shared config, each encoder's describe(), git commit, library versions, and the planned run matrix.
   It must support resume (skip completed runs, detected by presence of a valid run_meta.json) and `--dry-run` printing the matrix and estimated wall-clock from measured per-evaluation cost.

2. ANALYSIS (extend evodevo/analysis/)
   - `aggregate.py`: generalize load_runs/discover_runs to N encoders rather than assuming two; add a `three_way_table()` producing per encoder: final best fitness (mean ± std across seeds), evaluations to reach 50%/90% of the best arm's final fitness, validity rate, unique bodies explored, cache hit rate, mean wall-clock per evaluation.
   - Add a `statistics.py` with a paired comparison across seeds. With ~5 seeds, use a non-parametric test (Mann-Whitney U or a bootstrap CI over seed means) and REPORT EFFECT SIZE AND CONFIDENCE INTERVALS, not just a p-value. The module must include a docstring stating plainly that 5 seeds cannot support strong claims and that overlapping CIs must be reported as "no detectable difference", not as equivalence.
   - `plots.py`: three-arm fitness-vs-evaluations curves with per-seed variance bands; morphology-metric comparison across arms; a validity-rate/search-efficiency panel.
   - A results loader must handle the rule_based arm's variable-length serialized genomes without crashing.

3. REPORT GENERATOR (evodevo/analysis/report.py)
   Produce `results/BENCHMARK_REPORT.md` containing: the research question, the enforced fairness protocol as ACTUALLY resolved from the manifest (not as aspirationally documented), the three-way table, all figures, the statistical comparison with CIs, and a mandatory "Threats to validity" section covering at minimum: seed count, PPO budget far below EvoGym's own >1e6 recommendation for serious runs, single task family, per-arm mutation-strength tuning asymmetry, the NCA's known non-settling behaviour (68% of genomes still changing at t=T) and unbounded residual magnitude, and the possibility that a 5x5 grid is too small for developmental encodings to express any advantage.
   The generator must state the result direction neutrally in a single opening sentence, and must be capable of writing "no detectable difference between arms" or "the direct baseline outperformed both developmental encodings" as its headline. Do not build in any assumption that a developmental encoding wins.

MUST NOT: modify encoders, ea.py, or ppo_runner.py. If a bug is discovered here, fix it and re-run ALL arms from scratch — never patch or re-run a single arm after seeing results.

TESTS: end-to-end benchmark in surrogate mode across all three encoders, 2 seeds, tiny budget, asserting: identical evaluation counts per arm; manifest written; three_way_table populated for all three; report generated; resume skips completed runs; a deliberately mismatched config aborts before any run starts.

ACCEPTANCE: `make benchmark SURROGATE=1` produces a complete BENCHMARK_REPORT.md with all three arms and all figures, in under two minutes.
```

---

# PROMPT 7 — EvoGym visualization and rendering

```
Build the visualization layer. The inventory found evodevo has NO animated-render capability at all (static PNGs only, imageio not even a dependency), while the reference checkout's gif tooling has no confirmed working output on disk. Treat rendering as unproven and verify every step with an actual output file.

1. DEPENDENCIES: add imageio and imageio-ffmpeg to requirements.txt. Do not add pygifsicle (an external binary dependency that will fail on machines without it); if gif optimization is wanted, make it optional and degrade gracefully.

2. RENDERING MODULE (evodevo/visualization/)
   - `render.py`: `render_body_rollout(body, controller_path, task, n_steps, output_path, fps)` — instantiate the EvoGym env with render_mode="img" (NOT "human"), step the trained controller, collect frames, write mp4 and gif. Must work headless. Must accept a controller path OR fall back to random actions with a clear label on the output when no controller is supplied.
   - `morphology.py`: `plot_body(body)` and `plot_body_grid(bodies, labels)` — static voxel-grid figures using the material colours and names from core/materials.py, with a legend. Used for best-robot figures in the report.
   - `development.py`: `plot_development_trajectory(encoder, genome)` — renders t=0..T for any DEVELOPMENTAL encoder by calling a trajectory hook. Add an optional `develop_trajectory(genome) -> list[body]` method to the Encoder ABC returning None by default; implement it for NCA (which already has develop(return_trajectory=True)) and rule_based; direct returns None and the plotting function must handle that gracefully rather than crash. This figure — the same rule producing different structure over developmental time — is the single most communicative visual this project can produce for a review panel, and it exists for two of the three arms.
   - Every module sets matplotlib Agg before pyplot import.

3. ROBUSTNESS: rendering must NEVER crash a benchmark run. Wrap all rendering calls in the pipeline in try/except that logs a warning and continues. A failed gif must not lose a completed PPO run.

4. CLI: `scripts/benchmark/render_results.sh` — for each arm's best evolved body, produce: a static morphology PNG, a rollout gif and mp4, and (where available) a development trajectory figure. Output to results/media/{encoder}/{task}/. Also generate a single side-by-side comparison figure of the three arms' best morphologies for the report.

5. DIAGNOSTIC: `python -m evodevo.visualization.check` — a self-test verifying the EvoGym simulator is importable, an env can be created with render_mode="img", a frame can be captured, and imageio can write both gif and mp4. It must print a precise, actionable diagnosis for each failure (missing compiled simulator, missing ffmpeg, missing OpenGL/mesa libs, headless display issues) referencing the fixes documented in EvoGym's own README, rather than a bare traceback.

MUST NOT: modify encoders' decode/mutate behaviour; add rendering imports to ea.py, ppo_runner.py, or any encoder module; use render_mode="human" anywhere in an automated path.

TESTS: tests/visualization/ — plot_body produces a non-empty PNG; plot_body_grid handles 1 and N bodies; development trajectory returns None gracefully for direct; a @pytest.mark.slow rollout test producing an actual gif, skipped if the simulator is unbuilt.

ACCEPTANCE:
1. `python -m evodevo.visualization.check` reports all-green, or names exactly what is missing and how to fix it.
2. At least one real gif and one real mp4 exist on disk under results/media/ and are non-zero-length and openable.
3. A forced rendering failure does not abort a benchmark run.
```

---

# PROMPT 8 — Execute, document, and finalize

```
Execute the real experiments, close every open project obligation, and produce the final documentation set. This prompt has a long execution phase — report progress as you go and do not skip the gates.

1. PRE-FLIGHT GATES (do not proceed past a failing gate; report and stop)
   a. `make test` fully green.
   b. `python -m evodevo.visualization.check` green.
   c. `make benchmark SURROGATE=1` produces a complete three-arm report.
   d. Measure real per-evaluation wall-clock with a single short real-PPO run, and print the projected total cost of the full matrix BEFORE launching it. If the projection exceeds a reasonable budget, stop and report the number rather than launching.

2. EXECUTE
   - Primary: all three encoders x Walker-v0 x seeds 0-4, mutation-only, identical shared config. Note that no prior comparative numbers in this repository are reusable: NCA has zero completed real-PPO runs, direct has one complete and one partial seed, and the rule-based arm's prior numbers came from a 15x smaller evaluation budget with different invalid-genome accounting and its supporting artifacts are absent from disk. Everything is re-run from scratch under one config.
   - Secondary task: all three encoders x a second task (UpStepper-v0 or GapJumper-v0) x seeds 0-2.
   - Ablations: NCA dev_steps in {1,5,10,20} and hidden_channels in {0,3}, 3 seeds each. The T=1 cell is the critical control — it isolates whether ITERATED development matters or only the network's expressiveness.
   - Labelled ablation arm: rule_based with crossover enabled, 3 seeds, reported separately from the primary mutation-only comparison and never merged into it.

3. CLOSE PROJECT OBLIGATIONS
   - docs/decisions.md: resolve EVERY row still marked OPEN (neighbourhood type, hidden-channel count, dev-steps, initial-state policy, invalid-genome policy, mutation-tuning procedure), each with the option chosen, the evidence, and the date. The document's own header states no row may remain OPEN in the final submission — enforce that with a test that greps for "OPEN" and fails.
   - Add newly-made decisions from Prompts 1-7: rule-based integration mechanism, crossover parity, cache-hit budget accounting, validation-policy parameterization, port-equivalence verification status.
   - Document the mutation-strength tuning procedure honestly: mutation rate/sigma cannot be equalized across a categorical genome, a continuous weight vector, and a variable-length rule list. State how each arm's setting was arrived at, and list this in threats-to-validity.

4. DOCUMENTATION SET
   - `README.md` (root) — REPLACE the current stale one, which still claims nothing is implemented despite a working pipeline and populated results. Must contain: one-paragraph research question, the one-pipeline-three-decoders architecture diagram, quickstart (install, make test, make benchmark), repo map, links to the component guides, and the headline result in one sentence.
   - `docs/architecture.md` — end-to-end flow from genome through decode, validation, connectivity, EvoGym instantiation, PPO, fitness, and selection; the fairness protocol and where each clause is mechanically enforced in code; the extension points; a module dependency diagram showing that encoders never import physics/RL.
   - `docs/methods/baseline.md`, `docs/methods/rule-based.md`, `docs/methods/nca.md` — one standalone guide per arm: representation, genome format with a concrete worked example, decode procedure, operators, hyperparameters and where they live, how to run and test it in isolation, its known limitations, and its literature grounding. NCA cites Mordvintsev et al. 2020 and Horibe et al. 2021; rule-based credits the teammate's original implementation and cites the CA/artificial-embryogeny lineage from project-history.md.
   - `docs/results.md` — the benchmark report with figures, statistics, and threats to validity.
   - `REPRODUCE.md` — exact commands, versions, EvoGym commit, expected wall-clock, sufficient to reproduce every figure from archived logs.

5. SCIENTIFIC INTEGRITY REQUIREMENTS (non-negotiable)
   - Report the result in whichever direction the data points. A null result or a direct-baseline win is a valid, publishable-for-a-capstone outcome and must be stated plainly, per this project's own stated commitment to falsifiability.
   - Never re-run a single arm after seeing results in order to improve its numbers. If any bug is found at any point, fix it and re-run ALL arms.
   - Report across-seed variance and confidence intervals everywhere, never bare means.
   - State explicitly that the novelty claim is the controlled three-way comparison under identical constraints, not the invention of any one encoding — NCA-for-soft-robots (Horibe et al. 2021) and symbolic developmental rules both predate this work.

ACCEPTANCE:
1. results/BENCHMARK_REPORT.md exists with all three arms, 5 seeds, real PPO, figures, and statistics.
2. `grep -c "OPEN" docs/decisions.md` returns 0 and the enforcing test passes.
3. Root README.md is accurate and states the actual headline result.
4. All five documentation files exist and are internally consistent with the results.
5. `make test` green; git history is clean with descriptive commits.
6. A final summary lists: what was proven, what remains hypothesis, and what a larger study would need.
```

---

## Recommended execution order

```
1 → Git init + workspace restructure          ◄── HARD GATE: nothing is under version control right now
2 → Harden shared core
3 → Encoder plugin architecture + contract suite
4 → Port Rule-Based as third arm              ◄── HARD GATE: port-equivalence test must pass
5 → Isolated per-method pipelines
6 → Unified three-way benchmark pipeline
7 → Visualization and rendering               ◄── HARD GATE: visualization.check must be green
8 → Execute, document, finalize
```

**Prompt 1 is genuinely urgent and should be run before anything else.** Nothing in the project is currently tracked by git — the accidental repository root is the home directory with zero commits. Any restructuring done before `git init` is irreversible if it goes wrong.

**Between prompts 4 and 5** is the natural point to hand `docs/adding-an-encoder.md` and `docs/module-ownership.md` to your teammate, so they can start working inside `encoders/rule_based/` against the frozen contract while you work in `encoders/nca/`.
