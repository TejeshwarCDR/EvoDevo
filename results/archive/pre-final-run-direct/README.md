# Pre-final-run archive: direct/Walker-v0, seeds 0-3

This directory holds a real-PPO `direct`/Walker-v0 run (seeds 0-3 only, not
seed 4) that predates the frozen three-way comparison and is **not** part of
the final direct/nca/rule_based results.

Why it is archived rather than merged or deleted:

- The run's shared config (`population_size=15`, `offspring_size=15`,
  `eval_budget=300`, `ppo_timesteps=5000`) matches the current primary-run
  config, so it is not simply a stale/incompatible-budget artifact.
- However, it predates the NCA PPO-free calibration decision being locked in
  (see `docs/decisions.md`'s NCA sanity-gate and reconfirmation rows) and
  predates the full three-way fairness lock (`assert_comparable` verified
  pairwise across all three resolved configs immediately before the primary
  launch). Mixing it into the final comparison would let one arm's data
  originate from a run that was not gated by the same pre-launch checks the
  other two arms' data was.
- It is also incomplete: only seeds 0-3 are present, not the full 0-4 seed
  set the primary comparison requires.

This data is preserved for reference (e.g. sanity-checking wall-clock
estimates, or eyeballing early direct-encoder behavior) but must not be
merged with `results/runs/` or `results/BENCHMARK_REPORT.md` for the final
three-way comparison. Nothing here was deleted.
