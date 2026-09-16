"""Single source of truth for rule_based's resolved dynamics configuration --
the rule_based counterpart of evodevo/encoders/nca/resolved_config.py.

Before this module existed, rule_based's resolved settings
(conflict_resolution, position_predicate, boundary, neighbourhood,
action_monotonicity) existed only as RuleBasedEncoder's own constructor
defaults (evodevo/encoders/rule_based/encoder.py) -- correct, but with no
single named object analysis modules could import and compare against, the
same gap that once let NCA's (lambda*, B*) drift into six independent
copies (results/NCA_READINESS.md Section 10) before resolved_config.py was
introduced for that arm. This module closes the same gap for rule_based
pre-emptively, per docs/decisions.md's "Fix the silent-clip bug class
systemically" row.

PRIMARY mirrors RuleBasedEncoder's own constructor defaults for every field
EXCEPT readout_mode/readout_window (see next paragraph) -- see
`evodevo.encoders.rule_based.encoder.RuleBasedEncoder.__init__`:
- conflict_resolution="first_rule_index" (docs/decisions.md's "Make the
  rule_based interpreter provably order-independent by default" row,
  results/rule_based/CONFLICT_POLICY.md)
- position_predicate="disabled" (docs/decisions.md's "Restore
  translation-equivariance (Proposition R3)" row, results/rule_based/
  LOCALITY_STUDY.md)
- boundary="no_op", neighbourhood="von_neumann" (parity axes added in the
  same R3 task; both left at the interpreter's entire historical behaviour)
- action_monotonicity="unrestricted" -- see the "monotone-vs-unrestricted
  decision" paragraph below; this field's meaning changed in that decision.

readout_mode="mode_of_window", readout_window=6: the resolved readout
policy (docs/decisions.md's "unified T_primary (T-shift-stability
criterion, both arms)" row, results/rule_based/READOUT_POLICY_SELECTION.md,
results/rule_based/TIEBREAK_VALIDITY_CHECK.md's genome-aware tie-break,
which `RuleBasedEncoder.decode()`'s `mode_of_window` branch always applies).
UNLIKE every other field above, this does NOT match RuleBasedEncoder's own
class default (which stays "fixed_t"/5, deliberately, per that same
decisions.md row's own explicit instruction to preserve exact historical
behaviour for old call sites) -- RULE_BASED_READINESS.md's Section 0/11/12c
found this meant the resolved readout policy never reached the actual
benchmark path. Per docs/decisions.md's "Wire the resolved readout policy
into the primary benchmark path" row, PRIMARY is now the thing the primary
benchmark config is built from explicitly (see evodevo/experiments/
benchmark.py and its own per-arm encoder_kwargs extension), rather than
being threaded manually by whoever eventually writes the PPO run script.

MONOTONE-VS-UNRESTRICTED DECISION (docs/decisions.md's "Monotone vs
unrestricted: an explicit decision" row): monotone structurally dominates
unrestricted on 4/5 core metrics and eliminates the cycling-population/
readout-window problem entirely (Proposition R4), at one disclosed cost --
a modest rise in neutral (zero-displacement) mutations (fraction_zero_
displacement 0.5230 -> 0.5480), a claim about EA SEARCH DYNAMICS that
structural snapshots alone cannot settle. DECISION: DEFER to the PPO
matrix -- run BOTH as first-class, co-primary configurations rather than
picking one now. PRIMARY.action_monotonicity stays "unrestricted" (i.e.
RuleBasedEncoder's own class default is UNCHANGED, minimising blast
radius), and MONOTONE_ABLATION is elevated from a diagnostic-only ablation
to a co-primary candidate for the PPO matrix (see its own docstring below
and evodevo/experiments/configs/rule_based_monotone/, the second arm this
decision added to the primary comparison).

MIN_RULES/MAX_RULES/HARD_MAX_ITERATIONS are re-exported (not redefined)
from genome.py/interpreter.py, their own single source of truth (module-
level constants those modules already enforce directly) -- re-exporting
here only gives analysis code one place to import every resolved
rule_based quantity from, without creating a second copy of the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from evodevo.encoders.rule_based._rules.genome import MAX_RULES, MIN_RULES
from evodevo.encoders.rule_based._rules.interpreter import HARD_MAX_ITERATIONS


@dataclass(frozen=True)
class RuleBasedResolvedConfig:
    label: str
    conflict_resolution: str
    position_predicate: str
    boundary: str
    neighbourhood: str
    action_monotonicity: str = "unrestricted"
    readout_mode: str = "mode_of_window"
    readout_window: int = 6


PRIMARY = RuleBasedResolvedConfig(
    label="primary",
    conflict_resolution="first_rule_index",
    position_predicate="disabled",
    boundary="no_op",
    neighbourhood="von_neumann",
    action_monotonicity="unrestricted",
    readout_mode="mode_of_window",
    readout_window=6,
)

MONOTONE_ABLATION = RuleBasedResolvedConfig(
    label="monotone_ablation",
    conflict_resolution=PRIMARY.conflict_resolution,
    position_predicate=PRIMARY.position_predicate,
    boundary=PRIMARY.boundary,
    neighbourhood=PRIMARY.neighbourhood,
    action_monotonicity="monotone",
    readout_mode=PRIMARY.readout_mode,
    readout_window=PRIMARY.readout_window,
)

# Per docs/decisions.md's "Monotone vs unrestricted: an explicit decision"
# row: BOTH configurations run as co-primary candidates in the PPO matrix
# (the decision is DEFERRED to task-performance evidence, not made here) --
# this tuple is the single source of truth for "which rule_based dynamics
# configurations does the primary comparison include", consumed by
# evodevo/experiments/benchmark.py when resolving the rule_based_monotone
# arm alongside rule_based. MONOTONE_ABLATION's `label` field keeps its
# historical name for backward compatibility with existing analysis
# modules/reports that already cite it (e.g. results/rule_based/
# MONOTONE_ABLATION.md) -- only its STATUS changed (diagnostic-only ->
# co-primary candidate), not its identity or its own resolved values.
PPO_MATRIX_CANDIDATES = (PRIMARY, MONOTONE_ABLATION)

__all__ = [
    "RuleBasedResolvedConfig", "PRIMARY", "MONOTONE_ABLATION", "PPO_MATRIX_CANDIDATES",
    "MIN_RULES", "MAX_RULES", "HARD_MAX_ITERATIONS",
]
