# Contributing

This project compares three morphology-generation encodings (Direct, NCA,
Rule-Based) under one shared, fairness-enforced pipeline. See
`docs/module-ownership.md` for who owns what and why, and
`docs/adding-an-encoder.md` if you are adding a new encoding rather than
working on an existing one.

## Workflow

1. **Branch per method.** Work on one encoder's arm (`nca`, `rule_based`,
   `direct`, or a shared module) on its own branch. Don't mix an encoder
   change with a shared-module change in the same branch unless they are
   genuinely inseparable.

2. **Run your method's tests before pushing.**

   ```
   make test-nca            # or: make test-direct / make test-rule-based
   ```

   This runs only the contract suite (parametrized to your encoder), your
   method-specific tests, and the shared-core tests your encoder depends
   on -- surrogate evaluation only, no real PPO, so it finishes in well
   under a minute. If it doesn't pass, don't push.

3. **Run the full suite before merging.**

   ```
   make test
   ```

   This is the only way to be sure a shared-module change (or an
   accidental import/registration mistake) hasn't broken one of the other
   two arms. A green `make test-<your-method>` on your own branch does
   NOT guarantee `make test` is green -- shared modules affect everyone.

4. **Touching a shared module?** (`evodevo/core/`, `evodevo/morphology/`,
   `evodevo/evaluation/`, `evodevo/evolution/`, `evodevo/analysis/`,
   `evodevo/visualization/`, `evodevo/encoders/base.py`,
   `evodevo/encoders/registry.py`, or `evodevo/encoders/direct/*` -- see
   `docs/module-ownership.md`) get agreement from the other owner before
   merging. These modules affect every arm's results, so an undiscussed
   change here can silently invalidate someone else's already-reported
   numbers.

5. **Adding config changes?** Per-method configs
   (`evodevo/experiments/configs/<method>/*.yaml`) may only add
   encoder-specific keys -- they inherit every fairness-critical field
   (population size, eval budget, PPO timesteps, etc.) from
   `evodevo/experiments/configs/shared/base.yaml` and will fail loudly at
   load time if they try to override one. If you need a shared field to
   differ for a real reason, that's a fairness-protocol decision to make
   explicitly (e.g. a labelled ablation arm), not a config edit -- raise
   it with the other owner first.

## Useful commands

| Command | What it does |
|---|---|
| `make test` | Full test suite, every arm + shared core |
| `make test-core` | Shared-core tests only |
| `make test-direct` / `make test-nca` / `make test-rule-based` | One method's tests only |
| `make inspect-direct` / `make inspect-nca` / `make inspect-rule-based` | PPO-free characterization of one encoder's prior over morphologies |
| `make benchmark` | Cross-method comparison run |
| `make docs` | Sanity-check the documentation set exists |
| `make clean` | Remove caches/build artifacts |

## Scientific integrity

This project's fairness protocol (`evodevo/experiments/fairness.py`,
`evodevo/core/config.py::SHARED_BASE_KEYS`) exists to make sure any
difference in results between arms can be attributed to the encoding, not
to an accidental asymmetry in population size, PPO budget, or validation
policy. Don't work around `assert_comparable()` or the base/override config
mechanism to "just get a run working" -- if a fairness check is blocking
you, that means the comparison you're about to run isn't a fair one yet,
and the fix is a discussion, not a bypass.
