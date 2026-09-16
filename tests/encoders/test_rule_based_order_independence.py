"""PART 1 of the rule-language formal-spec task: empirically checks whether
`develop_rules`'s two-pass design is actually order-independent, which its
own docstring claims but had never been verified.

The two-pass design's justification (Pass 1 is read-only) is sound for Pass
1 -- cell scan order cannot bias MATCHING, since every cell's decision is
resolved purely from the untouched pre-iteration grid regardless of visit
order. It does NOT automatically cover Pass 2's WRITES:
  (a) GROWTH CONFLICT: two different acting cells' GROW_DIRECTIONAL both
      target the same empty neighbor in the same iteration -- whichever
      acting cell is enumerated LAST wins (last write to next_grid[ti][tj]
      wins), making the result a function of enumeration order.
  (b) COPY_NEIGHBOR READ SEMANTICS: does it read the untouched
      pre-iteration grid, or a partially-updated grid? (answered by code
      inspection below, and cross-checked with a targeted test)

This test does NOT change interpreter behaviour -- `develop_rules`'s new
`_cell_order` parameter is a test-only hook (private, leading underscore,
no production caller ever passes it; None -- the default -- reproduces the
exact prior row-major behaviour byte-for-byte, see
tests/encoders/test_rule_based.py's existing frozen-fixture equivalence
test, still green).

UPDATE (2026-09-16, the conflict-resolution-policy follow-up): every
`develop_rules()` call in this file now explicitly passes
`conflict_resolution="row_major_last_wins"` -- this file specifically
audits THAT historical, order-DEPENDENT policy (still fully selectable,
just no longer the class default; see `evodevo.encoders.rule_based.
_rules.interpreter.DEFAULT_CONFLICT_RESOLUTION` and
`results/rule_based/CONFLICT_POLICY.md`). The companion check that the
NEWLY-ADOPTED DEFAULT policy is order-independent (0/2000, as required)
lives in `evodevo/analysis/conflict_policy_selection.py`'s own
order-independence re-audit, not here.
"""
from __future__ import annotations

import random as _global_random

import numpy as np

from evodevo.core.materials import EMPTY
from evodevo.encoders.rule_based._rules.genome import sample_developmental_rules
from evodevo.encoders.rule_based._rules.interpreter import (
    _apply_self_action,
    _condition_matches,
    _DIRECTION_OFFSET,
    _growth_target,
    _in_bounds,
    _resolve_growth_conflicts,
    develop_rules,
)
from evodevo.encoders.rule_based._rules.types import ActionType

ROW_MAJOR_LAST_WINS = "row_major_last_wins"

SHAPE = (5, 5)
T = 100
N = 2000
SEED = 0


def _row_major_order(shape=SHAPE):
    rows, cols = shape
    return [(i, j) for i in range(rows) for j in range(cols)]


def _permuted_order(rng: np.random.Generator, shape=SHAPE):
    order = _row_major_order(shape)
    perm = rng.permutation(len(order))
    return [order[k] for k in perm]


def _draw_genome(seed: int):
    rng = np.random.default_rng(seed)
    _global_random.seed(int(rng.integers(0, 2**31 - 1)))
    return sample_developmental_rules()


def _count_growth_conflicts(genome, shape, iterations, cell_order=None):
    """Whitebox replica of develop_rules's own Pass-1/Pass-2 loop under
    conflict_resolution="row_major_last_wins" specifically, using the SAME
    internal helpers (_condition_matches/_apply_self_action/_growth_target/
    _resolve_growth_conflicts), purely to COUNT how often >=2 acting cells
    target the same empty growth neighbor in the same iteration --
    develop_rules itself returns only the final grid, with no
    conflict-count instrumentation, so this diagnostic-only replica is
    necessary to measure this without changing production code. Returns
    (conflict_count_per_iteration: list[int], final_grid: np.ndarray) --
    final_grid is cross-checked against a direct develop_rules() call to
    confirm this replica is faithful."""
    rows, cols = shape
    order = cell_order if cell_order is not None else _row_major_order(shape)

    grid = np.zeros(shape, dtype=int)
    si, sj = rows // 2, cols // 2
    from evodevo.core.materials import SOFT
    grid[si][sj] = SOFT

    conflicts_per_iter = []
    for _ in range(iterations):
        decisions = []
        for i, j in order:
            for rule_index, rule in enumerate(genome.rules):
                if _condition_matches(rule, i, j, grid, shape):
                    decisions.append((i, j, rule, rule_index))
                    break

        next_grid = grid.copy()
        for i, j, rule, _rule_index in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                _apply_self_action(rule, i, j, grid, next_grid, shape, "pre_iteration")

        # count growth conflicts: >=2 acting cells whose GROW_DIRECTIONAL
        # targets the SAME in-bounds, currently-EMPTY cell.
        growth_candidates = []
        for i, j, rule, rule_index in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                continue
            target = _growth_target(rule, i, j, grid, shape)
            if target is not None:
                ti, tj = target
                growth_candidates.append((ti, tj, i, j, rule, rule_index))

        by_target = {}
        for ti, tj, i, j, rule, rule_index in growth_candidates:
            by_target.setdefault((ti, tj), []).append((i, j))
        n_conflicts = sum(1 for targets in by_target.values() if len(targets) >= 2)
        conflicts_per_iter.append(n_conflicts)

        resolved_writes = _resolve_growth_conflicts(growth_candidates, ROW_MAJOR_LAST_WINS)
        for (ti, tj), voxel_code in resolved_writes.items():
            next_grid[ti][tj] = voxel_code

        grid = next_grid

    return conflicts_per_iter, grid


def test_whitebox_replica_matches_develop_rules_row_major():
    """Sanity check: the whitebox replica above must reproduce develop_rules'
    own output exactly under row-major order + row_major_last_wins, before
    it is trusted for conflict counting."""
    genome = _draw_genome(seed=1)
    _, replica_grid = _count_growth_conflicts(genome, SHAPE, T)
    real_grid = develop_rules(genome, shape=SHAPE, iterations=T, conflict_resolution=ROW_MAJOR_LAST_WINS)
    np.testing.assert_array_equal(replica_grid, real_grid)


def test_cell_order_hook_default_reproduces_row_major_exactly():
    """_cell_order=None (the default, used by every real caller) must be
    byte-identical to the pre-existing row-major behaviour, under
    row_major_last_wins."""
    genome = _draw_genome(seed=2)
    default_grid = develop_rules(genome, shape=SHAPE, iterations=T, conflict_resolution=ROW_MAJOR_LAST_WINS)
    explicit_row_major = develop_rules(
        genome, shape=SHAPE, iterations=T, conflict_resolution=ROW_MAJOR_LAST_WINS, _cell_order=_row_major_order()
    )
    np.testing.assert_array_equal(default_grid, explicit_row_major)


def test_copy_neighbor_reads_pre_iteration_state_not_live_state():
    """Targeted check of copy_neighbor's read semantics (Part 1's case b):
    build a 2-rule genome where cell A's rule is COPY_NEIGHBOR(right) and
    cell B (A's right neighbor) simultaneously SET_SOFTs itself from EMPTY.
    If copy_neighbor reads LIVE (partially-updated) state, A would end up
    SOFT (B's new value). If it reads PRE-ITERATION state, A ends up EMPTY
    (B's old value). The interpreter's own next_grid.copy() + separate
    current_grid parameter (passed as the untouched `grid`, never
    `next_grid`) should make this PRE-ITERATION -- verified directly."""
    from evodevo.encoders.rule_based._rules.genome import DevelopmentalRuleGenome
    from evodevo.encoders.rule_based._rules.types import ConditionType, Direction, DevelopmentalRule

    # A rule that fires for the seed cell only (POSITION doesn't
    # discriminate individual cells finely enough here) -- use CELL_TYPE
    # instead: seed the grid with SOFT at (2,2) (A) and EMPTY at (2,3) (B),
    # give A a COPY_NEIGHBOR(RIGHT) rule and B an ALWAYS->SET_HARD rule (B
    # is EMPTY, doesn't match CELL_TYPE(SOFT), so needs a condition that
    # fires on EMPTY specifically -- CELL_TYPE(EMPTY) works).
    from evodevo.encoders.rule_based._rules.types import VoxelState, ActionType as AT

    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
                           AT.COPY_NEIGHBOR, {"direction": Direction.RIGHT}),
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.EMPTY},
                           AT.SET_HARD, {}),
    ])
    initial = np.zeros(SHAPE, dtype=int)
    from evodevo.core.materials import SOFT
    initial[2, 2] = SOFT  # A
    # (2,3) is EMPTY (B) by construction of np.zeros

    result = develop_rules(genome, shape=SHAPE, iterations=1, initial_grid=initial, conflict_resolution=ROW_MAJOR_LAST_WINS)
    # If copy_neighbor read PRE-iteration state: A (2,2) becomes EMPTY (B's
    # old value, 0), not HARD (B's NEW value).
    assert result[2, 2] == EMPTY, (
        f"copy_neighbor appears to read LIVE/partially-updated state (got {result[2,2]}, "
        f"expected EMPTY={EMPTY} for pre-iteration semantics)"
    )


def test_order_independence_permutation_audit():
    """THE PERMUTATION TEST (Part 1's main deliverable). For N genomes,
    compares develop_rules' row-major result against a single fixed
    permuted cell order, at T=100. Reports (does not merely assert) the
    fraction of genomes whose decoded body differs, and separately counts
    growth conflicts per genome/iteration. Writes results/rule_based/
    ORDER_INDEPENDENCE_AUDIT.md as a side effect so this evidence is
    inspectable without re-running the (slow, N=2000xT=100) audit."""
    import os
    import time

    perm_rng = np.random.default_rng(12345)
    permuted_order = _permuted_order(perm_rng)
    assert sorted(permuted_order) == sorted(_row_major_order())  # genuine permutation, not a bug

    t0 = time.perf_counter()
    n_differ = 0
    total_conflicts = 0
    genomes_with_any_conflict = 0
    max_conflicts_single_genome = 0
    example_differing_genome = None

    for idx in range(N):
        genome = _draw_genome(seed=idx)

        row_major_result = develop_rules(genome, shape=SHAPE, iterations=T, conflict_resolution=ROW_MAJOR_LAST_WINS)
        permuted_result = develop_rules(
            genome, shape=SHAPE, iterations=T, conflict_resolution=ROW_MAJOR_LAST_WINS, _cell_order=permuted_order
        )

        differs = not np.array_equal(row_major_result, permuted_result)
        if differs:
            n_differ += 1
            if example_differing_genome is None:
                example_differing_genome = (idx, row_major_result.copy(), permuted_result.copy())

        conflicts_per_iter, replica_grid = _count_growth_conflicts(genome, SHAPE, T)
        assert np.array_equal(replica_grid, row_major_result)  # replica fidelity, every genome
        genome_conflicts = sum(conflicts_per_iter)
        total_conflicts += genome_conflicts
        if genome_conflicts > 0:
            genomes_with_any_conflict += 1
        max_conflicts_single_genome = max(max_conflicts_single_genome, genome_conflicts)

    elapsed = time.perf_counter() - t0
    fraction_differ = n_differ / N
    fraction_with_conflict = genomes_with_any_conflict / N

    lines = []
    lines.append("# rule_based interpreter order-independence audit")
    lines.append("")
    lines.append(
        "Fitness-blind (only develop_rules()/random_genome() are used -- NO PPO, NO task, NO reward, NO "
        f"EA). N={N} genomes (seed=0..{N-1}, same random_genome() draw convention as elsewhere in this "
        f"project), T={T}. Does NOT change interpreter behaviour: `_cell_order` is a test-only hook, "
        "None (default) reproduces the exact prior row-major behaviour (verified: "
        "test_cell_order_hook_default_reproduces_row_major_exactly)."
    )
    lines.append("")
    lines.append(f"**Cost**: {elapsed:.2f}s wall-clock for N={N} genomes x 2 orderings x T={T} steps.")
    lines.append("")

    lines.append("## Part 1a: the permutation test\n")
    lines.append(
        f"Row-major cell order vs ONE FIXED random permutation (seed=12345) of the same 25 cells, applied "
        "identically at every iteration, for the full development. "
        f"**Fraction of genomes whose decoded body at T={T} DIFFERS under permutation: "
        f"{fraction_differ:.4f}** ({n_differ}/{N})."
    )
    lines.append("")
    if example_differing_genome is not None:
        idx, rm, pm = example_differing_genome
        lines.append(f"Example (genome index {idx}), row-major vs permuted:")
        lines.append("```")
        lines.append(f"row-major:\n{rm}")
        lines.append(f"permuted:\n{pm}")
        lines.append("```")
    lines.append("")

    lines.append("## Part 1b: growth-conflict frequency (case (a))\n")
    lines.append(
        f"Instrumented directly (whitebox replica of develop_rules' own Pass-1/Pass-2 loop, verified "
        f"byte-identical to develop_rules' real output for every genome -- "
        "test_whitebox_replica_matches_develop_rules_row_major and the per-genome assertion in this test). "
        f"**{genomes_with_any_conflict}/{N} genomes ({fraction_with_conflict:.4f}) have AT LEAST ONE growth "
        f"conflict somewhere across their {T}-iteration development.** Total conflict events across the full "
        f"sample: {total_conflicts}. Max conflicts in a single genome's development: {max_conflicts_single_genome}."
    )
    lines.append("")
    if fraction_with_conflict > 0.01:
        lines.append(
            "Growth conflicts are NOT rare -- case (a) is a LIVE bug (order-dependence that actually "
            "manifests for a meaningful fraction of real genomes), not merely a latent/theoretical one."
        )
    else:
        lines.append(
            "Growth conflicts are rare in this sampled population -- case (a) is a LATENT bug (the "
            "order-dependence exists in the algorithm and CAN manifest, per the permutation test above, but "
            "essentially never arises for typical sampled genomes at this grid size/rule distribution)."
        )
    lines.append("")

    lines.append("## Part 1c: copy_neighbor read semantics (case (b))\n")
    lines.append(
        "**Confirmed by direct code reading, cross-checked with a targeted test "
        "(test_copy_neighbor_reads_pre_iteration_state_not_live_state)**: `_apply_action`'s COPY_NEIGHBOR "
        "branch reads `current_grid[si][sj]`, and `current_grid` is ALWAYS the untouched pre-iteration "
        "`grid` (never `next_grid`) at every call site in `develop_rules`'s Pass 2 loop. copy_neighbor reads "
        "PRE-ITERATION state, not live/partially-updated state -- case (b) is NOT a source of "
        "order-dependence; the interpreter's self-action pass (Pass 2a) is fully order-independent by "
        "construction (every self-action writes to its own acting cell, and at most one decision exists per "
        "cell, so no two self-actions can ever collide on the same write target regardless of enumeration "
        "order)."
    )
    lines.append("")

    lines.append("## Verdict\n")
    if fraction_differ > 0:
        lines.append(
            f"**The current interpreter is NOT order-independent.** {fraction_differ:.4f} of sampled genomes' "
            "decoded bodies change under a permuted cell enumeration order. The cause is EXCLUSIVELY case "
            "(a), growth-conflict resolution (Pass 2b) -- case (b) is confirmed not to contribute (Part 1c). "
            "Pass 1 (matching) and Pass 2a (self-actions) are order-independent by construction, exactly as "
            "the two-pass design's own justification claims; that justification simply never covered Pass "
            "2b, which the docstring already documents as resolving conflicts by enumeration order "
            "(`interpreter.py`'s own GROW_DIRECTIONAL action-semantics comment), just without ever having "
            "been empirically checked end-to-end until now."
        )
    else:
        lines.append(
            "**No decoded-body difference was observed under this permutation for this sample** -- despite "
            f"{genomes_with_any_conflict}/{N} genomes having at least one growth conflict internally, none "
            "of those conflicts happened to change the FINAL T=100 decoded body for this particular "
            "permutation/sample. This does NOT mean the interpreter is order-independent in general (the "
            "conflict-resolution mechanism is provably order-dependent by construction -- see the "
            "GROW_DIRECTIONAL semantics in `interpreter.py`'s own docstring); it means this specific "
            "permutation/sample did not surface a case where the conflicting write mattered to the T=100 "
            "outcome (e.g. the eventually-overwritten cell converged to the same state via other rules "
            "regardless of which write won)."
        )
    lines.append("")

    report = "\n".join(lines) + "\n"
    out_path = "results/rule_based/ORDER_INDEPENDENCE_AUDIT.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    # The test itself always passes -- it reports a finding, it does not
    # assert order-independence (which would be asserting the answer in
    # advance instead of measuring it).
    assert 0.0 <= fraction_differ <= 1.0


# Recurrence safeguard (docs/decisions.md's "Citation-mismatch resolution:
# order-independence percentages" row): the committed
# ORDER_INDEPENDENCE_AUDIT.md's permutation-test/growth-conflict fractions
# (12.95%/30.40%) were previously mis-cited elsewhere as 15.90%/34.45% --
# a stale-prose drift that went undetected for a while because nothing
# checked these hand-written citations against the actual generated
# artifact. This test greps the percentage strings across every location
# that cites them and fails if any of them still carries the stale figures,
# so a future edit can't silently reintroduce this specific drift.
_STALE_CITATIONS = ("15.90%", "34.45%")
_CITING_FILES = (
    "docs/rule-language-spec.md",
    "evodevo/encoders/rule_based/_rules/interpreter.py",
    "evodevo/analysis/rule_based_architecture_delta.py",
)


def test_no_stale_order_independence_citations():
    for path in _CITING_FILES:
        with open(path, "r") as f:
            text = f.read()
        for stale in _STALE_CITATIONS:
            assert stale not in text, (
                f"{path} still contains the stale order-independence citation {stale!r} -- "
                "the committed results/rule_based/ORDER_INDEPENDENCE_AUDIT.md states 12.95%/30.40%; "
                "see docs/decisions.md's 'Citation-mismatch resolution: order-independence percentages' row."
            )
