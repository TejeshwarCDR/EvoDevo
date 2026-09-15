# EA-minus-random decomposition (strategy_comparison benchmark)

Real data from `results/runs/strategy_comparison/` -- 5400 evaluation records, encoders=['direct', 'nca', 'rule_based'], strategies=['ea', 'random_search'], 3 seeds/cell, surrogate evaluator, eval_budget=300/cell.

**Purpose and scope**: this benchmark uses `evodevo.evaluation.ppo_runner.evaluate_body_surrogate` -- a fast, deterministic, physics-free proxy (a cheap function of actuator count and horizontal symmetry, explicitly NOT a scientific result; no PPO training is run) -- so it exists to produce a real, citable decomposition artifact and exercise the six-cell {ea, random_search} x {direct, nca, rule_based} pipeline end-to-end, not to draw locomotion-performance conclusions. nca and rule_based both saturate to an identical 26.0000 ceiling with zero across-seed variance under BOTH strategies: the surrogate's bounded scoring function is reached quickly regardless of strategy for these two arms at this budget, which is exactly why the guard clause below correctly refuses to call their results interpretable rather than reporting a spurious "no advantage" finding. Re-running this same matrix under `--surrogate` false (real PPO) is required before any interpretable nca/rule_based decomposition number exists; direct's result here is real, non-degenerate signal from the surrogate.

**Decomposition framing**: `random_search` (no selection, no mutation, no population -- fresh i.i.d. samples from the encoder's own genome distribution) measures what the encoding's representational PRIOR achieves for free. `ea - random_search` measures what SEARCH (selection + mutation) adds on top. These are different claims: a large `random_search` result with a small `ea - random_search` gap means the encoding's prior is doing the work, not the algorithm.

| encoder | counter | random_search (mean +/- std) | ea (mean +/- std) | EA-minus-random advantage | seed-to-seed noise floor | interpretable? |
| --- | --- | --- | --- | --- | --- | --- |
| direct | evals_spent | 10.0800 +/- 0.5133 | 14.1333 +/- 1.2760 | 4.0533 | 1.2760 | yes |
| direct | evals_valid | 10.0800 +/- 0.5133 | 14.1333 +/- 1.2760 | 4.0533 | 1.2760 | yes |
| nca | evals_spent | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |
| nca | evals_valid | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |
| rule_based | evals_spent | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |
| rule_based | evals_valid | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |

**nca's search-efficiency result under `evals_spent` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.
**nca's search-efficiency result under `evals_valid` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.
**rule_based's search-efficiency result under `evals_spent` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.
**rule_based's search-efficiency result under `evals_valid` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.

