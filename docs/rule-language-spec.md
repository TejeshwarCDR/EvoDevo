# The V2 developmental rule language: formal specification

Source of truth for every claim below: `evodevo/encoders/rule_based/_rules/{types,genome,interpreter}.py`, read
directly (not inferred, not copied from an earlier informal description) as of 2026-09-16. Where an earlier
informal description (a template, an architecture-review note, or this document's own originating task) made
an assumption that the code does not actually support, that assumption is corrected below and the correction
is called out explicitly rather than silently reconciled.

This document is diagnosis/specification only — it does not change interpreter behaviour. See
`results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` for the companion empirical audit that motivated Part A of
this spec (whether the two-pass design is actually order-independent, as B.3's operator below documents
precisely).

## A. Grammar (BNF, complete — every condition and action, every parameter, no ellipses)

```
genome        ::= rule rule rest-of-genome        (2 <= |genome.rules| <= 12, see A.1)

rule          ::= condition "->" action

condition     ::= "always"
                 | "cell_type" "(" state ")"
                 | "neighbor_count" "(" int ")"
                 | "empty_neighbor" "(" direction ")"
                 | "position" "(" position ")"

action        ::= "set_empty"
                 | "set_soft"
                 | "set_hard"
                 | "set_actuator_horizontal"
                 | "set_actuator_vertical"
                 | "grow_directional" "(" direction "," growable-state ")"
                 | "copy_neighbor" "(" direction ")"

state         ::= "empty" | "soft" | "hard" | "actuator_horizontal" | "actuator_vertical"

growable-state ::= "soft" | "hard" | "actuator_horizontal" | "actuator_vertical"
                    (every `state` EXCEPT "empty" -- growing an empty voxel is a no-op by
                    definition and is excluded from the sampleable/mutable vocabulary;
                    see types.py's GROWABLE_STATES)

direction     ::= "up" | "down" | "left" | "right"

position      ::= "boundary" | "interior" | "top" | "bottom" | "left" | "right"
                    (NOTE: `position`'s "left"/"right" are DISTINCT enum members from
                    `direction`'s "left"/"right" -- PositionType and Direction are
                    separate Python Enums that happen to share two string values; they
                    are never interchangeable in a rule)

int           ::= 1 | 2 | 3 | 4     (NEIGHBOR_COUNT's threshold; see A.2 -- there is NO
                                      selectable comparator and NO per-state filter)
```

### A.1 Genome-level constraints (from `genome.py`)

- `MIN_RULES = 2`, `MAX_RULES = 12`: a genome is a list of 2 to 12 `rule`s, inclusive.
- Rule order is semantically meaningful (see B.2, first-match semantics) — a genome is an ORDERED list, not a
  set.
- `MIN_ACTUATORS = 2`: not a grammar constraint on rules themselves, but a project-level VALIDITY constraint
  applied to the *decoded body* after development finishes (checked in `generate_robot_from_developmental_
  rules`, via `evodevo.morphology.validate`), not by the interpreter or the genome constructor.

### A.2 Corrections to previously-assumed grammar shapes (read from the code, not invented)

**`neighbor_count` has NO comparator parameter and NO per-state filter.** An earlier template for this
grammar assumed `neighbor_count(state, cmp, int)` with a selectable comparator (`==`/`>=`/`<=`) and a
specific neighbor state to count. The actual implementation (`interpreter.py::_condition_matches`,
`ConditionType.NEIGHBOR_COUNT` branch) is:

```python
threshold = rule.condition_params["threshold"]
count = 0
for di, dj in _DIRECTION_OFFSET.values():
    ni, nj = i + di, j + dj
    if _in_bounds(ni, nj, shape) and grid[ni][nj] != EMPTY:
        count += 1
return count >= threshold
```

- It counts **all four cardinal neighbors that are in-bounds AND non-EMPTY** — there is no way to count only
  neighbors of a *specific* state (e.g. "count of SOFT neighbors" is not expressible; only "count of
  occupied neighbors" is).
- The comparison is **hard-coded to `>=`** — `==` and `<=` do not exist in the language at all; the only
  parameter is `threshold ∈ {1, 2, 3, 4}` (`MIN_NEIGHBOR_THRESHOLD=1`, `MAX_NEIGHBOR_THRESHOLD=4`, from
  `types.py`).
- The correct grammar production is therefore `"neighbor_count" "(" int ")"`, not the 3-argument form a
  template assumed. This is documented here explicitly per this task's own "resolve the currently
  underspecified ones by reading the implementation" instruction.

**`position` is disabled by default as of 2026-09-16 (SUPERSEDES this section's own earlier finding).** When
this spec was first written, `position` was confirmed NOT disabled anywhere (`_random_condition` sampled
every `ConditionType` including `POSITION` with equal weight, no gate). That finding was correct AT THE TIME
and is preserved above as the historical record of what was found by reading the code. R3 (section F) has
since been established (not merely left open): `POSITION` tests the acting cell's own ABSOLUTE grid
coordinates, which breaks translation-equivariance (confirmed empirically, `results/rule_based/
LOCALITY_STUDY.md`: 0/500 genomes violate `D(translate(s)) == translate(D(s))` with `position_predicate=
"disabled"`, vs. 21.40% with `"enabled"`). `genome.py::sample_developmental_rules`/`mutate_developmental_rules`
now take a `position_predicate` parameter (`"disabled"` default, `"enabled"` selectable) gating GENERATION of
new `POSITION` conditions only — a genome that already has one (deserialized from an old result, or sampled
under `"enabled"`) still decodes and evaluates it correctly; only the sampler/mutator's ability to introduce
a NEW one is gated. The grammar in section A is unchanged (`position` remains part of the language); only
its DEFAULT SAMPLING WEIGHT changed, from "equal to every other condition type" to "zero, unless explicitly
enabled". Measured cost of this fix: `valid_rate` drops by 0.0250 (0.5735 -> 0.5485, N=2000) — disabling
POSITION removed a predicate that measurably improved validity, reported honestly rather than hidden, exactly
because the predicate's contribution to validity came from violating the encoding's own locality claim.

## B. Operational semantics

### B.1 State and rule structure

A `DevelopmentalRule` is `condition (+ condition_params) -> action (+ action_params)`, exactly the grammar of
section A. A genome (`DevelopmentalRuleGenome`) is an ordered list of 2–12 such rules.

The developmental grid is a fixed-shape (default 5×5) integer array; each cell holds one of 5 states —
EMPTY(0), RIGID(1), SOFT(2), H_ACT(3), V_ACT(4) (the numeric codes are `evodevo.core.materials`'s; the
rule-language-facing names are `VoxelState.EMPTY/SOFT/HARD/ACTUATOR_HORIZONTAL/ACTUATOR_VERTICAL` — `HARD`
in the rule language maps to the numeric code `RIGID`, a deliberate, documented correction of a pre-existing
bug in the ported reference implementation; see `interpreter.py`'s own header comment).

### B.2 Rule matching (Pass 1)

For a cell `(i, j)`, its matched rule is the FIRST rule (in genome order) whose `condition` evaluates `True`
against the CURRENT grid (the grid as it stood at the start of this iteration — see B.3). If no rule matches,
the cell has no decision this iteration. Matching is READ-ONLY: it never writes to any grid.

### B.3 The two-pass operator, fully specified (the three previously-implicit points, now explicit)

One development iteration, mapping grid `G_t` to `G_{t+1}`, given a fixed genome and a fixed **cell
enumeration order** `π` (a permutation of the grid's cells — the production interpreter always uses
row-major order for `π`; nothing else is possible through any public entry point, see C below for why this
still matters):

1. **MATCH** (Pass 1, read-only, order-irrelevant): for every cell `(i, j)` in the grid, independent of `π`,
   evaluate rules in genome order against `G_t` (B.2) and record the decision `D(i, j)` = the matched rule,
   or `⊥` (no decision) if none matched. **This step's result does not depend on `π`** — each cell's decision
   is computed purely from `G_t`, never from another cell's decision, so no permutation of `π` can change any
   individual `D(i, j)`.
2. **SELF-WRITE** (Pass 2a): let `G' := copy(G_t)`. For every cell `(i, j)` with `D(i, j) ≠ ⊥` whose action is
   `SET_*` or `COPY_NEIGHBOR` (i.e. NOT `GROW_DIRECTIONAL`), apply it: `SET_*` writes the corresponding state
   into `G'[i][j]` unconditionally; `COPY_NEIGHBOR(dir)` writes `G_t[neighbor(i,j,dir)]` into `G'[i][j]` if
   the neighbor is in-bounds, else this is a no-op. **Explicit point (i): self-actions read `G_t`
   (pre-iteration state), never `G'` — `COPY_NEIGHBOR`'s source read and every `SET_*`'s implicit "what was
   here before" are both against the untouched pre-iteration grid.** This step is also order-irrelevant:
   every self-action writes to its OWN acting cell, and Pass 1 guarantees at most one decision per cell, so
   no two self-actions can ever target the same cell — `π` cannot change this step's result either.
3. **GROWTH-WRITE** (Pass 2b): for every cell `(i, j)` with `D(i, j) ≠ ⊥` whose action IS `GROW_DIRECTIONAL
   (dir, voxel_type)`, let `(ti, tj) := neighbor(i, j, dir)`. If `(ti, tj)` is in-bounds **and `G_t[ti][tj] ==
   EMPTY`** (explicit point (iii), corrected below), then `G'[ti][tj] := voxel_type`; otherwise this
   particular growth is a no-op. **Explicit point (ii): growth writes are first COLLECTED (as the subset of
   `D` with a `GROW_DIRECTIONAL` action, from Pass 1) and only then RESOLVED in this step** — growth never
   observes the self-write pass's output, only `G_t`. **This step is `π`-DEPENDENT**: if two different acting
   cells `(i₁,j₁)` and `(i₂,j₂)` both target the SAME `(ti, tj)` in the same iteration (both pass the
   `G_t[ti][tj] == EMPTY` check, since that check is against the shared, unmodified `G_t`), they are applied
   in the SAME relative order as `π` visits `(i₁,j₁)` and `(i₂,j₂)` — the LATER one (in `π`-order) overwrites
   the earlier one's write to `G'[ti][tj]`. This is the interpreter's one genuine, confirmed
   order-dependency — see `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` for its empirical frequency and
   impact.
4. `G_{t+1} := G'`.

**Correction to explicit point (iii) as originally phrased**: an earlier informal description of this
operator described growth as targeting "cells empty after self-actions" — this is **not what the code does**.
The emptiness check in step 3 (`current_grid[ti][tj] == EMPTY`) is against `G_t`, the PRE-ITERATION grid, not
`G'` (the grid as modified by step 2's self-actions). A concrete consequence: if a growth target `(ti, tj)`
was EMPTY at the start of the iteration, but `(ti, tj)` **itself** also has a self-action decision this same
iteration (e.g. its own matched rule is `SET_SOFT`), the self-write happens first (step 2, writing into
`G'[ti][tj]`), and then the growth write (step 3) UNCONDITIONALLY OVERWRITES it (since the pre-iteration
check still passes) — growth wins over a same-cell self-action, not merely "growth only targets cells that
end up empty." This matches the interpreter's own stated design intent (a growth target's own self-action
"can never immediately erase growth that just arrived there") but the mechanism is growth-overwrites-
self-write, not growth-checks-post-self-write-emptiness — a meaningful difference the prior informal
phrasing did not capture.

### B.4 Termination

`develop_rules` applies B.3 exactly `iterations` times (`0 ≤ iterations ≤ HARD_MAX_ITERATIONS = 105`,
a safety-only ceiling, no part of the language) and returns `G_iterations`. There is no other stopping
condition (no fixed-point detection, no early exit) — the interpreter always runs for exactly the requested
number of iterations.

## C. Order-independence: the actual, empirically-measured answer

**The production interpreter is NOT fully order-independent**, despite its own docstring's Pass-1-is-
read-only argument (sound, but incomplete — it never covered Pass 2b). Measured directly (not assumed):
permuting the cell enumeration order `π` changes the T=100 decoded body for **12.95% of a random N=2000
sample** (`results/rule_based/ORDER_INDEPENDENCE_AUDIT.md`; this figure and the growth-conflict figure below
were previously mis-cited here at stale, higher values -- corrected to match the committed audit artifact
exactly, per the fresh re-run documented in docs/decisions.md's "Citation-mismatch resolution:
order-independence percentages" row). The cause is exclusively B.3's growth-conflict
resolution (step 3) — Pass 1 and Pass 2a are order-independent by construction, confirmed by the same audit.
Growth conflicts (≥2 acting cells targeting one empty cell in one iteration) are not rare: 30.40% of sampled
genomes have at least one, across a T=100 development. **The production interpreter fixes `π` to row-major
order always** (there is no configuration, factor, or code path through which any real caller — `decode()`,
`RuleBasedEncoder`, the EA, mutation, or crossover — can select a different `π`); this is why `develop_rules`
remains fully DETERMINISTIC (same genome ⇒ same output, always) even though it is not order-INDEPENDENT
(a different, hypothetical `π` could give a different output for the same genome). Determinism, not
order-independence, is what R1 (section E) actually needs.

## D. Context space

The "context" a rule's condition can observe, at a given cell `(i, j)`, is: the acting cell's own state
(5 possibilities: EMPTY/RIGID/SOFT/H_ACT/V_ACT) together with the state of each of its 4 cardinal neighbors
(UP, DOWN, LEFT, RIGHT).

**The OFF_GRID decision, pinned down from the code, not assumed**: a neighbor that is out-of-bounds is NOT
folded into EMPTY. Two independent code paths confirm this:
- `EMPTY_NEIGHBOR(dir)`'s condition (`_condition_matches`, `ConditionType.EMPTY_NEIGHBOR` branch) is
  `_in_bounds(ni, nj, shape) and grid[ni][nj] == EMPTY` — an out-of-bounds neighbor makes this `False`
  (NOT matched as empty), which would be `True` if OFF_GRID were folded into EMPTY.
- `NEIGHBOR_COUNT`'s per-neighbor test (A.2) is `_in_bounds(ni, nj, shape) and grid[ni][nj] != EMPTY` — an
  out-of-bounds neighbor is never counted as occupied EITHER, which would also differ from EMPTY-folding
  (an EMPTY-folded off-grid neighbor would still correctly not count as occupied, so this second check alone
  is consistent with either interpretation — it is the first check, `EMPTY_NEIGHBOR`, that is decisive).

An out-of-bounds neighbor therefore behaves as neither "empty" nor "occupied" for condition-matching
purposes — a third, DISTINGUISHED category, functionally equivalent to a 6th neighbor-state symbol
`OFF_GRID`. The context space is therefore:

```
|C| = 5 (own cell state) × 6^4 (four neighbors, each ∈ {EMPTY, RIGID, SOFT, H_ACT, V_ACT, OFF_GRID})
    = 5 × 1296
    = 6480
```

not the folded-to-EMPTY alternative (`5 × 5^4 = 3125`). This is stated here explicitly because R4 (section E)
depends on which one is correct.

## E. Worked trace: a 3-rule genome, with a growth conflict, on the NEW fully-specified semantics

Genome (rules in order — first match wins):

```
rule 0:  position(top)          -> grow_directional(down, soft)
rule 1:  cell_type(soft)        -> grow_directional(right, hard)
rule 2:  always                 -> set_empty
```

Initial grid (5×5; `.`=EMPTY, `S`=SOFT; A at (0,2), C at (1,1)):

```
. . A . .
. C . . .
. . . . .
. . . . .
. . . . .
```

**Pass 1 (match, against `G_0`, row-major enumeration — result independent of enumeration order per B.3
step 1)**:
- Every cell in row 0 (`(0,0)..(0,4)`) satisfies `position(top)` (rule 0 tests ONLY the acting cell's row/
  column position — B/A.2's grammar — it does not require the acting cell itself to be SOFT or anything
  else) — **rule 0 matches all 5 of them**, including `(0,2)=A` (rule 0 is checked before rule 1, so A's
  SOFT-ness is never even tested this iteration). Decision for each: `grow_directional(down, soft)`,
  targeting `(1,0), (1,1), (1,2), (1,3), (1,4)` respectively.
- `(1,1)=C` is not in row 0 (rule 0 does not match), but `cell_type(soft)` (rule 1) does — decision:
  `grow_directional(right, hard)`, targeting `(1,2)`.
- Every other cell (`(1,0),(1,2),(1,3),(1,4)`, all of row 2–4) matches neither rule 0 nor rule 1, falls
  through to rule 2 (`always`) — decision: `set_empty` (a no-op for cells already EMPTY).

**Pass 2a (self-writes, from `G_0`)**: every `set_empty` decision writes EMPTY into its own (already-EMPTY)
cell — no visible change. `(0,0),(0,1),(0,3),(0,4)` and `(1,1)=C` have `grow_directional` decisions, not
self-actions, so Pass 2a does not touch them either — they are untouched by this pass, retaining their
`G_0` values (`(0,0),(0,1),(0,3),(0,4)` stay EMPTY; `(1,1)=C` stays SOFT, since `G' := copy(G_0)` at the
start of this pass already carries it forward).

**Pass 2b (growth writes, checked against `G_0`, applied in row-major `π`-order — THIS is where the conflict
is resolved)**:
- `(0,0)→(1,0)`: `G_0[1,0]=EMPTY` ⇒ write `SOFT` to `(1,0)`.
- `(0,1)→(1,1)`: `G_0[1,1]=SOFT` (occupied) ⇒ **no-op** (this is why C survives despite being a growth
  target too — B.3's "growth targets `G_t`, not `G'`" rule does not matter here since `(1,1)` was already
  non-empty in `G_0` to begin with).
- `(0,2)=A → (1,2)`: `G_0[1,2]=EMPTY` ⇒ write `SOFT` to `(1,2)`.
- `(0,3)→(1,3)`: `G_0[1,3]=EMPTY` ⇒ write `SOFT` to `(1,3)`.
- `(0,4)→(1,4)`: `G_0[1,4]=EMPTY` ⇒ write `SOFT` to `(1,4)`.
- `(1,1)=C → (1,2)`: `G_0[1,2]=EMPTY` ⇒ write `HARD` to `(1,2)`. **This is the growth conflict**: `(0,2)`
  and `(1,1)` both target `(1,2)`. In row-major `π`-order, `(0,2)` (row 0) is visited before `(1,1)` (row 1),
  so `(1,1)`'s write is applied SECOND and wins: `(1,2)` ends the iteration as `HARD`, not `SOFT`.

**Resulting grid `G_1`** (verified by direct execution, not hand-derived — `H`=HARD/RIGID, `S`=SOFT):

```
. . S . .
S S H S S
. . . . .
. . . . .
. . . . .
```

This matches `develop_rules(genome, shape=(5,5), iterations=1, initial_grid=G_0)`'s actual output exactly.
**If `π` were permuted so that `(1,1)` were visited before `(0,2)`, `(1,2)` would instead end as `SOFT`** —
this is precisely the order-dependence section C measures at population scale.

## F. Propositions

### R1 — Eventual periodicity (proved here)

**Statement**: for any fixed genome and any fixed, deterministic cell-enumeration order `π` (in particular,
the production interpreter's row-major `π`), the sequence `G_0, G_1, G_2, ...` produced by repeated
application of B.3's operator is eventually periodic: there exist `p ≥ 0` (the pre-period) and `k ≥ 1` (the
period) such that `G_{t+k} = G_t` for all `t ≥ p`.

**Proof (pigeonhole)**: B.3's operator is a deterministic function `f_π : \mathcal{G} \to \mathcal{G}` on the
finite set `\mathcal{G}` of all possible 5×5 grids over the 5-symbol vocabulary — `|\mathcal{G}| = 5^25`
(every one of the 25 cells independently takes one of 5 states; the initial condition and the operator never
change the grid's shape or vocabulary). "Deterministic" here means: for a FIXED genome and FIXED `π`, the
same input grid `G` always produces the same output `f_π(G)` — true by direct inspection of B.3, which
consults only `G_t`, the genome, and `π`, none of which involve any external randomness (per `interpreter.py`
's own DETERMINISM docstring claim, itself now a direct logical consequence of B.3 as specified above, not
merely asserted).

Consider the sequence `G_0, G_1 = f_π(G_0), G_2 = f_π(G_1), \ldots, G_{5^{25}}` — `5^{25} + 1` grids drawn
from a set of size `5^{25}`. By the pigeonhole principle, two of them must coincide: `G_a = G_b` for some
`0 \le a < b \le 5^{25}`. Since `f_π` is a deterministic function, `G_{a+1} = f_π(G_a) = f_π(G_b) = G_{b+1}`,
and by induction `G_{a+m} = G_{b+m}` for every `m \ge 0`. Taking `p := a` and `k := b - a` gives
`G_{t+k} = G_t` for all `t \ge p`, with `k \le 5^{25}`. `∎`

This holds regardless of whether `f_π` is order-independent (R1 needs only DETERMINISM of `f_π` for a FIXED
`π`, which section C already establishes holds for the production interpreter's fixed row-major `π` even
though a DIFFERENT `π'` could give a different, equally-deterministic `f_{π'}`) — this is precisely why R1
could be proved here without first resolving whether the interpreter is order-independent (it is not, per
section C), and why `results/rule_based/SETTLING_AUDIT.md`'s directly-observed period-2/3/6/... cycles among
real sampled genomes are fully consistent with, and an instance of, this proposition (not a separate,
unexplained phenomenon).

### R2 — (status: OPEN, to be established in a future R2 task)

Not established in this document. R1 guarantees SOME eventual period `k ≤ 5^25` exists for every genome;
R2 is the open question of a TIGHTER, practically-useful bound on `k` (or on the pre-period `p`) — the naive
`5^25` bound is astronomically larger than anything ever observed (the largest confirmed real period in this
project's own data is single-digit, `results/rule_based/SETTLING_AUDIT.md`'s period-2/3/6/8/9/10 breakdown).
No claim about a tighter bound is made here; deferred.

### R3 — Translation-equivariance (status: ESTABLISHED, 2026-09-16)

**Statement**: for any genome G that never uses `ConditionType.POSITION`, and any grid state `s` on a
TOROIDAL (periodic) grid, `D_G(translate(s)) = translate(D_G(s))` for every cell, where `translate` is a
shift along either grid axis and `D_G` is one application of B.3's operator under genome G. Genomes that DO
use `POSITION` do not, in general, satisfy this — `POSITION` tests the acting cell's own fixed absolute
coordinates (TOP/BOTTOM/LEFT/RIGHT/BOUNDARY/INTERIOR), which are not preserved under translation, breaking
the property directly at any cell whose matched rule depends on it.

**Empirical confirmation** (`results/rule_based/LOCALITY_STUDY.md`, N=500 genomes each): translating a random
grid state by `(1, 0)` on a torus (`boundary="toroidal"`, so there is no distinguished boundary cell for a
finite-grid artifact to hide in) and comparing `D(translate(s))` to `translate(D(s))` over the WHOLE grid:
**0/500 (0.0000) violate it when `POSITION` cannot be generated** (`position_predicate="disabled"`), vs.
**107/500 (0.2140) when it can** (`position_predicate="enabled"`) — confirming the predicate is the exact
and only cause, not merely a plausible one.

**Why the torus, not the actual `"no_op"` (finite, open) grid the encoder uses by default**: an
open/finite grid has genuine, POSITION-unrelated boundary artifacts of its own (a growth write into a
comparison cell can originate from a neighbor whose own decision depends on an off-grid check with no
consistent counterpart after translation), which would contaminate a naive open-grid test even with
`POSITION` disabled. Testing on a torus isolates the property R3 actually claims (equivariance under
translation for LOCAL rules) from this separate, expected, finite-grid effect — see `LOCALITY_STUDY.md`'s
own methodology note for the direct empirical trace that led to this test design.

**Practical consequence**: `position_predicate="disabled"` (genome.py's new default) restores R3 for the
`RuleBasedEncoder`'s primary sampling/mutation path; `POSITION` remains fully part of the language (section
A) and fully evaluable for any genome that already contains it (deserialized from an old result, or sampled
under `position_predicate="enabled"`) — only its GENERATION by the sampler/mutator is gated.

Measured cost: disabling `POSITION` reduces `valid_rate` by 0.0250 (N=2000, `results/rule_based/
LOCALITY_STUDY.md`) — a real, disclosed expressivity cost of restoring this property, not a free fix.

### R4 — (status: OPEN, to be established in a future R6 task)

Not established in this document. Section D pins down the context-space size (`|C| = 6480`, OFF_GRID
distinguished) that R4 is presumed to depend on; no further claim about R4 itself is made here.
