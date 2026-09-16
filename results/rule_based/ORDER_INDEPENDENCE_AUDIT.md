# rule_based interpreter order-independence audit

Fitness-blind (only develop_rules()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes (seed=0..1999, same random_genome() draw convention as elsewhere in this project), T=100. Does NOT change interpreter behaviour: `_cell_order` is a test-only hook, None (default) reproduces the exact prior row-major behaviour (verified: test_cell_order_hook_default_reproduces_row_major_exactly).

**Cost**: 27.47s wall-clock for N=2000 genomes x 2 orderings x T=100 steps.

## Part 1a: the permutation test

Row-major cell order vs ONE FIXED random permutation (seed=12345) of the same 25 cells, applied identically at every iteration, for the full development. **Fraction of genomes whose decoded body at T=100 DIFFERS under permutation: 0.1295** (259/2000).

Example (genome index 4), row-major vs permuted:
```
row-major:
[[1 1 2 1 0]
 [1 2 1 1 2]
 [1 0 2 1 0]
 [2 2 2 2 2]
 [0 0 0 0 0]]
permuted:
[[1 1 1 1 0]
 [1 2 1 1 2]
 [1 0 2 1 0]
 [1 2 2 1 2]
 [0 0 0 0 0]]
```

## Part 1b: growth-conflict frequency (case (a))

Instrumented directly (whitebox replica of develop_rules' own Pass-1/Pass-2 loop, verified byte-identical to develop_rules' real output for every genome -- test_whitebox_replica_matches_develop_rules_row_major and the per-genome assertion in this test). **608/2000 genomes (0.3040) have AT LEAST ONE growth conflict somewhere across their 100-iteration development.** Total conflict events across the full sample: 5336. Max conflicts in a single genome's development: 348.

Growth conflicts are NOT rare -- case (a) is a LIVE bug (order-dependence that actually manifests for a meaningful fraction of real genomes), not merely a latent/theoretical one.

## Part 1c: copy_neighbor read semantics (case (b))

**Confirmed by direct code reading, cross-checked with a targeted test (test_copy_neighbor_reads_pre_iteration_state_not_live_state)**: `_apply_action`'s COPY_NEIGHBOR branch reads `current_grid[si][sj]`, and `current_grid` is ALWAYS the untouched pre-iteration `grid` (never `next_grid`) at every call site in `develop_rules`'s Pass 2 loop. copy_neighbor reads PRE-ITERATION state, not live/partially-updated state -- case (b) is NOT a source of order-dependence; the interpreter's self-action pass (Pass 2a) is fully order-independent by construction (every self-action writes to its own acting cell, and at most one decision exists per cell, so no two self-actions can ever collide on the same write target regardless of enumeration order).

## Verdict

**The current interpreter is NOT order-independent.** 0.1295 of sampled genomes' decoded bodies change under a permuted cell enumeration order. The cause is EXCLUSIVELY case (a), growth-conflict resolution (Pass 2b) -- case (b) is confirmed not to contribute (Part 1c). Pass 1 (matching) and Pass 2a (self-actions) are order-independent by construction, exactly as the two-pass design's own justification claims; that justification simply never covered Pass 2b, which the docstring already documents as resolving conflicts by enumeration order (`interpreter.py`'s own GROW_DIRECTIONAL action-semantics comment), just without ever having been empirically checked end-to-end until now.

