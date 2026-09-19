# EvoDevo: Direct vs. Developmental (NCA) vs. Rule-Based Encodings in EvoGym

**Research question:** Under an identical evaluation pipeline, fixed 5x5 grid, and equal evaluation budget, how does a Neural Cellular Automaton (NCA) developmental encoding compare to a direct encoding and a rule-based generative encoding in final task fitness, search efficiency, and morphological properties — and in which direction?

This project makes no assumption that any one encoding wins. The architecture is deliberately symmetric so that the comparison, not the encoding, is the deliverable.

## One pipeline, four arms

All four arms share every stage of the pipeline except the decoder itself:

```
genome -> Encoder.decode(genome) -> body (5x5 int array)
       -> morphology validator                [shared]
       -> get_full_connectivity()             [shared, EvoGym]
       -> gym.make('Walker-v0', ...)          [shared, EvoGym]
       -> PPO training (fixed budget)          [shared]
       -> evaluation reward -> fitness         [shared]
       -> EA selection on the genome           [shared]
```

Everything below the encoder box is written once, so that any observed difference in results comes from the representation, not from asymmetries in the pipeline. `evodevo/experiments/benchmark.py` enforces this before any run starts: it resolves every (arm, task) config and asserts pairwise fairness (`evodevo/experiments/fairness.py`) across every arm — identical shared config, identical `use_crossover`, identical budget accounting — and aborts if any pair disagrees. Per-arm dynamics knobs (an encoder's own update rule, readout policy, mutation strength) are deliberately outside this fairness check — see `evodevo/encoders/*/resolved_config.py` below.

Arms:
- **direct** (`evodevo/encoders/direct/`): genome is the body grid itself.
- **nca** (`evodevo/encoders/nca/`): a neural cellular automaton develops the body over a fixed number of steps from a seed genome, using a bounded (`bounded_leaky`) update rule and a `mode_of_window` readout — see `evodevo/encoders/nca/resolved_config.py::PRIMARY`.
- **rule_based** (`evodevo/encoders/rule_based/`): a rule-based/grammar generative encoding, with an order-independent conflict-resolution policy and a `mode_of_window` readout — see `evodevo/encoders/rule_based/resolved_config.py::PRIMARY`. Its original reference implementation was ported in full during initial development; behavioral equivalence to that original was proven once, across 20 seeded genomes, and is now permanently frozen as a regression fixture (`tests/encoders/fixtures/rule_based_port_equivalence.json`) — the original checkout is no longer needed or present (see `docs/decisions.md`).
- **rule_based_monotone**: a co-primary variant of `rule_based` (same encoder class, `evodevo/encoders/rule_based/resolved_config.py::MONOTONE_ABLATION`) with a termination-guaranteed ("monotone") development rule instead of `rule_based`'s unrestricted one. Structurally it dominates `rule_based` on 4/5 core metrics with one disclosed cost (reduced mutation sensitivity) — whether that trade-off matters for task performance is exactly what the primary matrix is for, so both run as first-class arms rather than picking one in advance (`docs/decisions.md`'s "Monotone vs unrestricted: an explicit decision" row).

The project's only external dependency is EvoGym at `external/evogym/`.

Every arm's encoder implements the same ABC (`evodevo/encoders/base.py`) — `decode`, `random_genome`, `mutate`, `crossover`/`supports_crossover`, `describe` — and is exercised by a shared contract test suite (`tests/encoders/test_contract.py`) so no arm can silently diverge from the interface the EA relies on. Encoder modules are barred from importing `evogym`, `stable_baselines3`, `gym`, or `torch` directly (enforced by grep in the test/audit process) — anything they need from those packages is exposed instead through `evodevo/morphology/` shared utilities, so an encoder can be exercised end-to-end without the physics simulator.

## Repository layout

```
evodevo/
  core/           ExperimentConfig, shared-base-key config inheritance/override rules
  encoders/       direct / nca / rule_based, the shared Encoder ABC/registry, and each
                  arm's resolved_config.py (single source of truth for its dynamics)
  morphology/     body validation, connectivity, metrics (shared across all encoders)
  evaluation/     ppo_runner.py (real PPO evaluator) + a surrogate evaluator for fast iteration
  evolution/      ea.py (encoding-agnostic mu+lambda EA), random_search.py (no-selection
                  control), logging.py, progress.py, cache.py
  experiments/    run_experiment.py (single run), benchmark.py (N-way orchestrator), fairness.py
  analysis/       the N/R-series diagnostic and calibration modules, statistics/stats_protocol,
                  report.py / ablation_report.py -- aggregate stats + Markdown report generation
  visualization/  rendering, morphology/development figures, a self-test diagnostic
scripts/          per-method run/test/inspect scripts + benchmark scripts
docs/             decisions.md (the full decision trail), PRE_PARALLELIZATION_READINESS.md
                  (the current resolved-configuration/readiness gate), module-ownership.md,
                  adding-an-encoder.md, READY_TO_RUN.md (how to actually run this today), ...
tests/            the full test suite (run `make test` for the current count)
results/          runs/ (benchmark output), archive/ (superseded runs, kept not deleted),
                  and one report/analysis artifact per N/R-series diagnostic
```

## Running it

```bash
make test                   # full test suite
make test-direct             # direct-encoder contract + method-specific + shared-core tests
make test-nca                # same, for nca
make test-rule-based         # same, for rule_based
make inspect-direct          # PPO-free morphology-prior characterization (no EA)
make benchmark SURROGATE=1   # four-arm comparison, fast physics-free evaluator
```

`evodevo/experiments/configs/benchmark.yaml` currently targets the `fast_dev` profile (a pipeline-validation-scale budget: `ppo_timesteps=5000`, `population_size=15`, `offspring_size=15`, `eval_budget=300` — not the scientific comparison budget). `make benchmark` (real PPO, no `SURROGATE=1`) will refuse to run against this profile without an explicit `--i-know-this-is-fast-dev` acknowledgement (`FastDevRealRunGuardError`) — the Makefile target does not pass that flag, so invoke `python -m evodevo.experiments.benchmark` directly if you actually want a real-PPO smoke test at `fast_dev` scale; see `docs/READY_TO_RUN.md` for the exact command and what it is (and is not) useful for. `evodevo/experiments/benchmark.py` writes a manifest before any run starts, resumes (skips) any already-complete `(arm, task, seed)` run, and prints one live progress line per completed evaluation. See `docs/PRE_PARALLELIZATION_READINESS.md` for the full resolved-configuration table (every dynamics field, per arm, cross-checked against the real production config path) and `docs/READY_TO_RUN.md` for a step-by-step guide to actually running this today.

## Data hygiene

`results/archive/` holds superseded run data, never deleted, each with a README explaining why it was superseded (a bug fix, a deliberate clean-restart cutover, etc.) — see `results/archive/*/README.md`. Per `docs/decisions.md` and `CONTRIBUTING.md`, a bug found partway through a benchmark is fixed and **all arms are re-run from scratch**, never patched arm-by-arm after seeing results.

## Current status

**Architecture design work is complete and validated; the real experimental matrix has not yet been run.** Concretely:

- **Done**: the pipeline, all four arms (including the NCA and rule_based architecture redesigns — see `docs/decisions.md`'s N1-N17/R1-R8 rows for the full derivation of `bounded_leaky`, `mode_of_window`, order-independence, the monotone co-primary variant, and every structural/mutation/readout-policy calibration behind them), the EA plus a `random_search` no-selection control, the fairness protocol, the `fast_dev`/`primary_matrix` profile mechanism, the statistics protocol (with Holm-Bonferroni correction and an explicit ablation-within-arm vs. cross-arm distinction), the benchmark manifest/progress/figure infrastructure, and visualization are implemented and covered by the full test suite (`make test`).
- **Not done**: no real-PPO primary-matrix result exists yet. `results/runs/` currently holds only a pre-redesign run (predates the entire N1-N17/R1-R8 workstream — see `results/BENCHMARK_REPORT.md`'s own superseded-banner for detail) kept for historical/methodological value, not as a current result. The `primary_matrix` profile's `population_size`/`offspring_size`/`eval_budget` remain an explicit, disclosed open scoping decision (deliberately not guessed — see `docs/decisions.md`'s "Do NOT guess..." row).
- **Next steps**: the pre-flight wall-clock compute projection and parallel-execution design, ahead of actually launching the primary matrix.

See `docs/decisions.md` for every implementation decision (the full, append-only decision trail) and `docs/PRE_PARALLELIZATION_READINESS.md` for the consolidated readiness gate (resolved-configuration table, test-suite status, and the explicit "NOT covered by this document" scope). `docs/module-ownership.md` documents who owns which module, and `docs/adding-an-encoder.md` explains how to add a further arm.
