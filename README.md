# EvoDevo: Direct vs. Developmental (NCA) vs. Rule-Based Encodings in EvoGym

**Research question:** Under an identical evaluation pipeline, fixed 5x5 grid, and equal evaluation budget, how does an NCA developmental encoding compare to a direct encoding and a rule-based generative encoding in final task fitness, search efficiency, and morphological properties — and in which direction?

This project makes no assumption that any one encoding wins. The architecture is deliberately symmetric so that the comparison, not the encoding, is the deliverable.

## One pipeline, three decoders

All three encodings share every stage of the pipeline except the decoder itself:

```
genome -> Encoder.decode(genome) -> body (5x5 int array)
       -> morphology validator                [shared]
       -> get_full_connectivity()             [shared, EvoGym]
       -> gym.make('Walker-v0', ...)          [shared, EvoGym]
       -> PPO training (fixed budget)          [shared]
       -> evaluation reward -> fitness         [shared]
       -> EA selection on the genome           [shared]
```

Everything below the encoder box is written once, so that any observed difference in results comes from the representation, not from asymmetries in the pipeline. `evodevo/experiments/benchmark.py` enforces this before any run starts: it resolves every (encoder, task) config and asserts pairwise fairness (`evodevo/experiments/fairness.py`) across all three arms — identical shared config, identical `use_crossover`, identical budget accounting — and aborts if any pair disagrees.

Encoders:
- **direct** (`evodevo/encoders/direct/`): genome is the body grid itself.
- **nca** (`evodevo/encoders/nca/`): a neural cellular automaton develops the body over a fixed number of steps from a seed genome.
- **rule_based** (`evodevo/encoders/rule_based/`): a ported rule-based/grammar generative encoding. Its original reference implementation was ported in full during initial development; behavioral equivalence to that original was proven once, across 20 seeded genomes, and is now permanently frozen as a regression fixture (`tests/encoders/fixtures/rule_based_port_equivalence.json`) — the original checkout is no longer needed or present (see `docs/decisions.md`). The project's only remaining external dependency is EvoGym at `external/evogym/`.

Every encoder implements the same ABC (`evodevo/encoders/base.py`) — `decode`, `random_genome`, `mutate`, `crossover`/`supports_crossover`, `describe` — and is exercised by a shared contract test suite (`tests/encoders/test_contract.py`) so no arm can silently diverge from the interface the EA relies on. Encoder modules are barred from importing `evogym`, `stable_baselines3`, `gym`, or `torch` directly (enforced by grep in the test/audit process) — anything they need from those packages is exposed instead through `evodevo/morphology/` shared utilities, so an encoder can be exercised end-to-end without the physics simulator.

## Repository layout

```
evodevo/
  core/           ExperimentConfig, shared-base-key config inheritance/override rules
  encoders/       direct / nca / rule_based, plus the shared Encoder ABC and registry
  morphology/     body validation, connectivity, metrics (shared across all encoders)
  evaluation/     ppo_runner.py (real PPO evaluator) + a surrogate evaluator for fast iteration
  evolution/      ea.py (encoding-agnostic mu+lambda EA), logging.py, progress.py, cache.py
  experiments/    run_experiment.py (single run), benchmark.py (N-way orchestrator), fairness.py
  analysis/       report.py / ablation_report.py -- aggregate stats + Markdown report generation
  visualization/  rendering, morphology/development figures, a self-test diagnostic
scripts/          per-method run/test/inspect scripts + benchmark scripts
docs/             decisions.md, module-ownership.md, adding-an-encoder.md, READY_TO_RUN.md, ...
tests/            279 tests across core, encoders, evolution, evaluation, morphology, visualization
results/          runs/ (live output), archive/ (superseded runs, kept not deleted), figures/
```

## Running it

```bash
make test                 # full suite (279 tests)
make test-direct           # direct-encoder contract + method-specific + shared-core tests
make test-nca               # same, for nca
make test-rule-based        # same, for rule_based
make inspect-direct         # PPO-free morphology-prior characterization (no EA)
make benchmark               # real-PPO three-way comparison -> results/BENCHMARK_REPORT.md
make benchmark SURROGATE=1   # same, with the fast physics-free evaluator instead of PPO
```

`evodevo/experiments/benchmark.py` writes a manifest before any run starts, resumes (skips) any already-`DONE_MARKER`-complete `(encoder, task, seed)` run, and prints one live progress line per completed evaluation — current run, index within it, overall matrix progress, rolling mean wall-clock, and an HH:MM:SS ETA for both the current run and the full matrix (`evodevo/evolution/progress.py`). This is purely additive: it never changes the persisted `run_log.jsonl` / `run_meta.json` / heartbeat content.

## Data hygiene

`results/archive/` holds superseded run data, never deleted, each with a README explaining why it was superseded (a bug fix, a deliberate clean-restart cutover, etc.) — see `results/archive/*/README.md`. `results/runs/` holds only the current/latest run. Per `docs/decisions.md` and `CONTRIBUTING.md`, a bug found partway through a benchmark is fixed and **all arms are re-run from scratch**, never patched arm-by-arm after seeing results.

## Status

The pipeline, all three encoders, the EA, fairness checking, the benchmark orchestrator, live progress reporting, aggregate statistics/report generation, and visualization are implemented and covered by the full test suite (279 tests, plus each encoder's own contract/method-specific suites — see `make test-*` above). The primary three-way comparison (direct/nca/rule_based x Walker-v0 x seeds 0-4, `population_size=15, offspring_size=15, eval_budget=300, ppo_timesteps=5000, grid_shape=(5,5)`) is config-locked (hash in `docs/READY_TO_RUN.md`) and ready to execute; `results/runs/` is currently empty, awaiting a fresh end-to-end run under the current code path. See `docs/decisions.md` for every implementation decision, `docs/module-ownership.md` for who owns which module, and `docs/adding-an-encoder.md` to add a fourth arm.
