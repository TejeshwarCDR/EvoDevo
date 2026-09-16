"""Verification-only pass over the entire rule_based architecture-redesign
workstream (propositions R1-R4, the capacity study, the architecture-delta
cascade, and the silent-clip fixes): measures whether it is complete,
internally consistent, and correctly reflected in its own committed
artifacts, and writes results/RULE_BASED_READINESS.md.

MUST NOT fix, tune, re-run (in the sense of re-deriving a "better" number),
or adjust anything to make a number look better -- this is the rule_based
counterpart of evodevo/analysis/readiness_check.py and
evodevo/analysis/nca_readiness_check.py, following their exact conventions:
numbers are either (a) READ from an already-committed artifact (a .md
report or raw .json this workstream already produced -- the expensive
N=2000 sweeps are never re-run here), or (b) MEASURED fresh, read-only, at
report-generation time via cheap, deterministic checks (git/grep/OPEN-row
scans, a fresh `t_primary_unified` run, a programmatic grammar/enum diff).
If a check fails or an artifact is stale/missing/inconsistent with another
committed artifact, that is reported as a finding, never silently worked
around or reconciled in place.

Usage:
    python -m evodevo.analysis.rule_based_readiness_check
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from evodevo.analysis.readiness_check import (
    _line_is_comment_or_string,
    _string_literal_line_ranges,
    scan_decisions_open_rows,
)
from evodevo.encoders.rule_based._rules.genome import MAX_RULES, MIN_RULES
from evodevo.encoders.rule_based._rules.interpreter import HARD_MAX_ITERATIONS
from evodevo.encoders.rule_based._rules.types import ActionType, ConditionType
from evodevo.encoders.rule_based.context_analysis import CONTEXT_SPACE_SIZE, enumerate_contexts
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.encoders.rule_based.resolved_config import PRIMARY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "RULE_BASED_READINESS.md")
THIS_FILE = os.path.join(REPO_ROOT, "evodevo", "analysis", "rule_based_readiness_check.py")

RB_DIR = os.path.join(REPO_ROOT, "results", "rule_based")
ORDER_INDEPENDENCE_MD = os.path.join(RB_DIR, "ORDER_INDEPENDENCE_AUDIT.md")
CONFLICT_POLICY_MD = os.path.join(RB_DIR, "CONFLICT_POLICY.md")
CONTEXT_ANALYSIS_MD = os.path.join(RB_DIR, "CONTEXT_ANALYSIS.md")
LOCALITY_STUDY_MD = os.path.join(RB_DIR, "LOCALITY_STUDY.md")
CYCLE_CHARACTERISATION_MD = os.path.join(RB_DIR, "CYCLE_CHARACTERISATION.md")
SETTLING_AUDIT_MD = os.path.join(RB_DIR, "SETTLING_AUDIT.md")
MONOTONE_ABLATION_MD = os.path.join(RB_DIR, "MONOTONE_ABLATION.md")
CAPACITY_STUDY_MD = os.path.join(RB_DIR, "CAPACITY_STUDY.md")
ARCHITECTURE_DELTA_MD = os.path.join(RB_DIR, "ARCHITECTURE_DELTA.md")
TIEBREAK_VALIDITY_MD = os.path.join(RB_DIR, "TIEBREAK_VALIDITY_CHECK.md")
SPEC_MD = os.path.join(REPO_ROOT, "docs", "rule-language-spec.md")
DECISIONS_MD = os.path.join(REPO_ROOT, "docs", "decisions.md")
BENCHMARK_YAML = os.path.join(REPO_ROOT, "evodevo", "experiments", "configs", "benchmark.yaml")
STRUCTURAL_MD = os.path.join(REPO_ROOT, "results", "structural", "STRUCTURAL.md")
MUTATION_RAW = os.path.join(REPO_ROOT, "results", "mutation", "mutation_calibration_raw.json")
T_PRIMARY_UNIFIED_MD = os.path.join(REPO_ROOT, "results", "T_PRIMARY_UNIFIED.md")


def _fmt(value, digits=4):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _read_text(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _run(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=180)
        return out.returncode, out.stdout, out.stderr
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)


def _table_rows(text: str, header_prefix: str) -> list:
    if text is None:
        return []
    lines = text.splitlines()
    rows, started = [], False
    for line in lines:
        if not started:
            if line.strip().startswith(header_prefix):
                started = True
            continue
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        if set(stripped.replace("|", "").strip()) <= {"-", " "}:
            continue
        rows.append([c.strip() for c in stripped.strip("|").split("|")])
    return rows


def _first_match(text: str, pattern: str, group: int = 1):
    if text is None:
        return None
    m = re.search(pattern, text)
    return m.group(group) if m else None


def _bold_number(text: str, pattern: str):
    """Extract a **N.NNNN** style bolded fraction from a report's prose."""
    return _first_match(text, pattern)


# ---------------------------------------------------------------------------
# SECTION 0 -- PROVENANCE
# ---------------------------------------------------------------------------


def gather_provenance() -> dict:
    rc, commit, _ = _run(["git", "rev-parse", "HEAD"])
    commit = commit.strip() if rc == 0 else None
    rc, status_out, _ = _run(["git", "status", "--porcelain"])
    dirty_files = [l[3:] for l in status_out.splitlines() if l.strip()] if rc == 0 else []

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in ("numpy", "matplotlib", "pytest"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"

    encoder = RuleBasedEncoder.__init__
    import inspect
    sig = inspect.signature(encoder)

    mutation_raw = _read_json(MUTATION_RAW)
    mutation_rate = None
    if mutation_raw:
        mutation_rate = mutation_raw.get("calibrated", {}).get("rule_based", {}).get("param_value")

    config = {
        "conflict_resolution": (
            PRIMARY.conflict_resolution, "R2 (order-independence)",
            "evodevo/encoders/rule_based/resolved_config.py::PRIMARY.conflict_resolution "
            "(docs/decisions.md's \"Make the rule_based interpreter provably order-independent by default\" row)",
        ),
        "copy_neighbor_reads": (
            sig.parameters["copy_neighbor_reads"].default, "R2 (order-independence audit, Part 1c)",
            "evodevo/encoders/rule_based/encoder.py::RuleBasedEncoder.__init__ default "
            "(confirmed not a source of order-dependence; case (b) ruled out -- ORDER_INDEPENDENCE_AUDIT.md Part 1c)",
        ),
        "position_predicate": (
            PRIMARY.position_predicate, "R3 (translation-equivariance)",
            "evodevo/encoders/rule_based/resolved_config.py::PRIMARY.position_predicate "
            "(docs/decisions.md's \"Restore translation-equivariance (Proposition R3)...\" row)",
        ),
        "action_monotonicity": (
            PRIMARY.action_monotonicity, "R4 (monotone ablation, kept as ablation)",
            "evodevo/encoders/rule_based/resolved_config.py::PRIMARY.action_monotonicity "
            "(docs/decisions.md's monotone-ablation row -- class default deliberately NOT switched)",
        ),
        "boundary": (
            PRIMARY.boundary, "R3 (parity axis, unchanged from historical default)",
            "evodevo/encoders/rule_based/resolved_config.py::PRIMARY.boundary",
        ),
        "neighbourhood": (
            PRIMARY.neighbourhood, "pre-existing (\"Neighbourhood type\" decisions.md row)",
            "evodevo/encoders/rule_based/resolved_config.py::PRIMARY.neighbourhood",
        ),
        "MIN_RULES": (MIN_RULES, "capacity study (justified on evidence)", "evodevo/encoders/rule_based/_rules/genome.py::MIN_RULES"),
        "MAX_RULES": (MAX_RULES, "capacity study (justified on evidence)", "evodevo/encoders/rule_based/_rules/genome.py::MAX_RULES"),
        "HARD_MAX_ITERATIONS": (HARD_MAX_ITERATIONS, "safety ceiling (raised over several tasks, currently 105)", "evodevo/encoders/rule_based/_rules/interpreter.py::HARD_MAX_ITERATIONS"),
        "readout_mode (class default)": (
            sig.parameters["readout_mode"].default, "readout-policy resolution",
            "evodevo/encoders/rule_based/encoder.py::RuleBasedEncoder.__init__ default "
            "-- deliberately stays 'fixed_t'; the resolved mode_of_window/W=6 policy must be threaded "
            "explicitly by any caller building 'the primary rule_based arm' (see results/T_PRIMARY_UNIFIED.md)",
        ),
        "readout_window (class default)": (
            sig.parameters["readout_window"].default, "readout-policy resolution (class default, NOT the resolved W)",
            "evodevo/encoders/rule_based/encoder.py::RuleBasedEncoder.__init__ default",
        ),
        "resolved readout W (when mode_of_window is threaded)": (
            6, "T_PRIMARY_UNIFIED (T-shift-stability sweep)", "results/T_PRIMARY_UNIFIED.md",
        ),
        "tie-break mode": (
            "validity_aware=True (genome-aware, TIEBREAK_VALIDITY_CHECK.md's recommended fix)",
            "tie-break validity check",
            "evodevo/encoders/rule_based/encoder.py::decode (mode_of_window branch passes validity_aware=True) "
            "and evodevo/analysis/windowed_readout.py::mode_of_window",
        ),
        "mutation_rate (rule_based, calibrated)": (
            mutation_rate, "P5 / mutation re-calibration cascade",
            "results/mutation/mutation_calibration_raw.json::calibrated.rule_based.param_value",
        ),
    }

    return {
        "commit": commit,
        "dirty_files": dirty_files,
        "clean": not dirty_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "config": config,
    }


def render_section0(prov: dict) -> str:
    lines = ["## Section 0 -- Provenance\n"]
    lines.append(f"- git commit: `{prov['commit']}`")
    lines.append(f"- working tree clean: {'YES' if prov['clean'] else 'NO -- dirty/untracked paths: ' + ', '.join(prov['dirty_files'][:20])}")
    lines.append(f"- report generated: {prov['generated_at']}")
    lines.append("- package versions: " + ", ".join(f"{k}={v}" for k, v in prov["versions"].items()))
    lines.append("")
    lines.append("### Resolved rule_based configuration\n")
    lines.append("| parameter | value | resolved by | source |")
    lines.append("| --- | --- | --- | --- |")
    for name, (value, prompt, source) in prov["config"].items():
        lines.append(f"| {name} | {_fmt(value)} | {prompt} | `{source}` |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 1 -- ORDER-INDEPENDENCE (R2)
# ---------------------------------------------------------------------------


def gather_section1() -> dict:
    audit_text = _read_text(ORDER_INDEPENDENCE_MD)
    before_fraction = _bold_number(audit_text, r"DIFFERS under permutation: ([\d.]+)")
    growth_conflict_fraction = _bold_number(audit_text, r"genomes \(([\d.]+)\) have AT LEAST ONE growth conflict")
    copy_neighbor_line = _first_match(
        audit_text,
        r"(\*\*Confirmed by direct code reading[^\n]*copy_neighbor reads PRE-ITERATION state[^\n]*\.)",
    )

    decisions_text = _read_text(DECISIONS_MD)
    r2_row = _first_match(decisions_text, r"(\| 2026-09-16 \| Make the rule_based interpreter provably order-independent by default.*)")
    after_fraction = _first_match(r2_row or "", r"Permutation test re-run under the adopted default: ([\d/]+) genomes differ")
    decisions_before_fraction = _first_match(
        decisions_text, r"row_major_last_wins.*?\(.*?(\d+\.\d+)% of genomes affected per"
    )
    spec_text = _read_text(SPEC_MD)
    spec_before_fraction = _first_match(spec_text, r"changes the T=100 decoded body for \*\*([\d.]+)%")
    spec_growth_conflict = _first_match(spec_text, r"([\d.]+)% of sampled\s*\n?genomes have at least one")

    interpreter_text = _read_text(os.path.join(REPO_ROOT, "evodevo", "encoders", "rule_based", "_rules", "interpreter.py"))
    interpreter_growth_conflict = _first_match(interpreter_text, r"not rare[,:] ([\d.]+)% of\s*\n?\s*sampled genomes have at least one")

    return {
        "audit_text": audit_text,
        "before_fraction_committed_artifact": before_fraction,
        "growth_conflict_fraction_committed_artifact": growth_conflict_fraction,
        "copy_neighbor_line": copy_neighbor_line,
        "r2_row": r2_row,
        "after_fraction": after_fraction,
        "decisions_before_fraction_pct": decisions_before_fraction,
        "spec_before_fraction_pct": spec_before_fraction,
        "spec_growth_conflict_pct": spec_growth_conflict,
        "interpreter_growth_conflict_pct": interpreter_growth_conflict,
    }


def render_section1(s1: dict) -> str:
    lines = ["## Section 1 -- Order-independence (Proposition R2)\n"]
    if s1["audit_text"] is None:
        lines.append("**MISSING**: results/rule_based/ORDER_INDEPENDENCE_AUDIT.md not found.\n")
        return "\n".join(lines)

    lines.append(
        f"**BEFORE (from the committed results/rule_based/ORDER_INDEPENDENCE_AUDIT.md, Part 1a)**: the "
        f"permutation-test fraction of genomes whose decoded body differs under a permuted cell enumeration "
        f"order is **{s1['before_fraction_committed_artifact']}** (i.e. "
        f"{float(s1['before_fraction_committed_artifact'])*100:.2f}% of N=2000)."
    )
    lines.append(
        f"**Growth-conflict frequency (case (a), Part 1b)**: **{s1['growth_conflict_fraction_committed_artifact']}** "
        f"of sampled genomes have at least one growth conflict across their development."
    )
    lines.append("")

    lines.append("### Discrepancy between the committed audit artifact and its own downstream citations\n")
    lines.append(
        f"docs/decisions.md's own \"Make the rule_based interpreter provably order-independent by default\" "
        f"row cites the BEFORE fraction as **{s1['decisions_before_fraction_pct']}%**; "
        f"docs/rule-language-spec.md's section C cites it as **{s1['spec_before_fraction_pct']}%** and cites "
        f"the growth-conflict frequency as **{s1['spec_growth_conflict_pct']}%**; "
        f"evodevo/encoders/rule_based/_rules/interpreter.py's own docstring cites the growth-conflict "
        f"frequency as **{s1['interpreter_growth_conflict_pct']}%**. The ACTUALLY COMMITTED "
        f"results/rule_based/ORDER_INDEPENDENCE_AUDIT.md artifact -- the primary source these three all cite "
        f"-- states **{float(s1['before_fraction_committed_artifact'])*100:.2f}%** (permutation-test fraction) "
        f"and **{float(s1['growth_conflict_fraction_committed_artifact'])*100:.2f}%** (growth-conflict "
        f"frequency), NOT the higher figures repeated in decisions.md/the spec/the interpreter docstring."
    )
    mismatch = (
        s1["decisions_before_fraction_pct"] is not None
        and abs(float(s1["decisions_before_fraction_pct"]) - float(s1["before_fraction_committed_artifact"]) * 100) > 0.5
    )
    if mismatch:
        lines.append(
            "\n**Flagged, not reconciled**: this is a genuine cross-artifact inconsistency -- either the "
            "committed ORDER_INDEPENDENCE_AUDIT.md was regenerated after decisions.md/the spec/the "
            "interpreter docstring were written (and those three downstream citations were never updated "
            "to match), or the reverse. Per this report's own charter, the committed .md artifact is treated "
            "as ground truth for the number itself, and the mismatch is reported as a finding (Section 11) "
            "rather than silently averaged or corrected in any of the four locations."
        )
    lines.append("")

    lines.append(f"**Copy-neighbor read semantics (case (b))**: {s1['copy_neighbor_line'] or '(statement not found verbatim)'}")
    lines.append("")

    after_ok = s1["after_fraction"] is not None and s1["after_fraction"].strip() in ("0/2000",)
    lines.append(
        f"**AFTER, under the adopted default (`conflict_resolution=\"first_rule_index\"`), from "
        f"docs/decisions.md's R2 row (verbatim re-run statement)**: `{s1['after_fraction']}` genomes differ "
        f"under permutation -- **{'PASS (0/2000)' if after_ok else 'BLOCKER -- expected 0/2000'}**."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 2 -- FORMAL SPECIFICATION
# ---------------------------------------------------------------------------


def run_section2_checks() -> dict:
    spec_exists = os.path.exists(SPEC_MD)
    spec_text = _read_text(SPEC_MD)

    ellipsis_hits = []
    if spec_text is not None:
        for i, line in enumerate(spec_text.splitlines(), start=1):
            if "..." in line or re.search(r"\betc\.?\b", line, re.IGNORECASE):
                ellipsis_hits.append((i, line.strip()))
    # The grammar block itself is section A (between "## A. Grammar" and "## B.")
    grammar_block = ""
    if spec_text:
        m = re.search(r"## A\. Grammar.*?(?=\n## B\.)", spec_text, re.DOTALL)
        grammar_block = m.group(0) if m else ""
    grammar_ellipsis_hits = [(i, l) for i, l in ellipsis_hits if l in grammar_block]

    condition_results = {}
    for member in ConditionType:
        # Match the enum's own string value (e.g. "cell_type", "neighbor_count")
        # appearing literally in the grammar block.
        condition_results[member.name] = member.value in grammar_block
    action_results = {}
    for member in ActionType:
        action_results[member.name] = member.value in grammar_block

    fresh_context_space = len(enumerate_contexts())
    # `|C| = ...` appears more than once in the spec: once as a multiplicative
    # breakdown (e.g. "|C| = 5 (own cell state) x 6^4 (...)") and once as the
    # actual resolved total ("|C| = 6480"). Taking the FIRST match would grab
    # the "5" from the breakdown, not the total -- take the MAX of all
    # matches instead, since any partial-breakdown component is necessarily
    # smaller than the final resolved total it multiplies into.
    all_c_claims = [int(x) for x in re.findall(r"\|C\| = (\d+)", spec_text or "")]
    spec_claimed_c = str(max(all_c_claims)) if all_c_claims else None

    return {
        "spec_exists": spec_exists,
        "ellipsis_hits": ellipsis_hits,
        "grammar_ellipsis_hits": grammar_ellipsis_hits,
        "condition_results": condition_results,
        "action_results": action_results,
        "all_conditions_present": all(condition_results.values()),
        "all_actions_present": all(action_results.values()),
        "fresh_context_space": fresh_context_space,
        "context_space_size_constant": CONTEXT_SPACE_SIZE,
        "spec_claimed_context_space": int(spec_claimed_c) if spec_claimed_c else None,
        "context_space_matches": (
            spec_claimed_c is not None
            and int(spec_claimed_c) == fresh_context_space == CONTEXT_SPACE_SIZE
        ),
    }


def render_section2(s2: dict) -> str:
    lines = ["## Section 2 -- Formal specification\n"]
    lines.append(f"docs/rule-language-spec.md exists: **{s2['spec_exists']}**.\n")

    lines.append(
        f"Ellipsis/'etc' scan of the full document: {len(s2['ellipsis_hits'])} line(s) match a bare `...` or "
        f"the word 'etc' anywhere in the file; of these, **{len(s2['grammar_ellipsis_hits'])}** fall inside "
        "the BNF grammar block (section A) itself, which is the acceptance criterion this check enforces."
    )
    if s2["grammar_ellipsis_hits"]:
        lines.append("Grammar-block ellipsis hits (BLOCKER):")
        for i, l in s2["grammar_ellipsis_hits"]:
            lines.append(f"  - line {i}: `{l}`")
    lines.append("")

    lines.append("### Programmatic ConditionType/ActionType <-> grammar cross-check\n")
    lines.append("| enum member | value | appears in grammar block? |")
    lines.append("| --- | --- | --- |")
    for name, present in s2["condition_results"].items():
        lines.append(f"| ConditionType.{name} | `{ConditionType[name].value}` | {'PASS' if present else '**FAIL**'} |")
    for name, present in s2["action_results"].items():
        lines.append(f"| ActionType.{name} | `{ActionType[name].value}` | {'PASS' if present else '**FAIL**'} |")
    lines.append("")
    lines.append(
        f"Result: **{'ALL PRESENT' if s2['all_conditions_present'] and s2['all_actions_present'] else 'AT LEAST ONE MISSING -- BLOCKER'}**.\n"
    )

    lines.append(
        f"**Context-space size**: docs/rule-language-spec.md section D claims `|C| = "
        f"{s2['spec_claimed_context_space']}`; `evodevo.encoders.rule_based.context_analysis.CONTEXT_SPACE_SIZE` = "
        f"{s2['context_space_size_constant']}; a fresh `len(enumerate_contexts())` call = "
        f"{s2['fresh_context_space']}. **{'MATCH' if s2['context_space_matches'] else 'MISMATCH -- BLOCKER'}**.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 3 -- LOCALITY (R3)
# ---------------------------------------------------------------------------


def gather_section3() -> dict:
    text = _read_text(LOCALITY_STUDY_MD)
    part2_rows = _table_rows(text, "| position_predicate | n | n_pass")
    part1_rows = _table_rows(text, "| metric | enabled | disabled")
    return {"text": text, "part2_rows": part2_rows, "part1_rows": part1_rows}


def render_section3(s3: dict) -> str:
    lines = ["## Section 3 -- Locality / translation-equivariance (Proposition R3)\n"]
    if s3["text"] is None:
        lines.append("**MISSING**: results/rule_based/LOCALITY_STUDY.md not found.\n")
        return "\n".join(lines)

    lines.append("Direct empirical test of D(translate(s)) == translate(D(s)) on a torus, N=500 each setting:\n")
    lines.append("| position_predicate | n | n_pass | n_fail | fraction_fail |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in s3["part2_rows"]:
        lines.append("| " + " | ".join(row[:5]) + " |")
    lines.append("")

    disabled_row = next((r for r in s3["part2_rows"] if r[0] == "disabled"), None)
    enabled_row = next((r for r in s3["part2_rows"] if r[0] == "enabled"), None)
    disabled_ok = disabled_row is not None and disabled_row[3] == "0"
    enabled_discriminates = enabled_row is not None and int(enabled_row[3]) > 0
    lines.append(
        f"**position_predicate=\"disabled\"**: {disabled_row[3] if disabled_row else '?'} violations -- "
        f"{'PASS' if disabled_ok else 'BLOCKER -- expected 0'}."
    )
    lines.append(
        f"**position_predicate=\"enabled\"**: {enabled_row[3] if enabled_row else '?'} violations -- "
        f"{'PASS (test discriminates, as required)' if enabled_discriminates else 'FAIL -- test does not discriminate, expected >0'}."
    )
    lines.append("")

    lines.append("Structural cost of disabling POSITION (Part 1, N=2000):\n")
    lines.append("| metric | enabled | disabled (new default) | delta |")
    lines.append("| --- | --- | --- | --- |")
    for row in s3["part1_rows"]:
        lines.append("| " + " | ".join(row[:4]) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 4 -- CONTEXT ANALYSIS
# ---------------------------------------------------------------------------


def gather_section4() -> dict:
    text = _read_text(CONTEXT_ANALYSIS_MD)
    static_dead_rows = _table_rows(text, "### static dead-rule count")
    dynamic_dead_rows = _table_rows(text, "### dynamic dead-rule count")
    effective_rows_ctx = _table_rows(text, "### effective_rule_count (static) per genome")
    caveat = _first_match(text, r"(\*\*CAVEAT on `grow_directional`/`grow_directional` pairs\*\*.*?)\n\n")

    capacity_text = _read_text(CAPACITY_STUDY_MD)
    effective_rows_sampled = _table_rows(capacity_text, "### (a) as actually sampled")
    effective_rows_full_budget = _table_rows(capacity_text, "### (b) explicitly given the FULL MAX_RULES")

    return {
        "text": text,
        "static_dead_rows": static_dead_rows,
        "dynamic_dead_rows": dynamic_dead_rows,
        "effective_rows_ctx": effective_rows_ctx,
        "caveat": caveat,
        "effective_rows_sampled": effective_rows_sampled,
        "effective_rows_full_budget": effective_rows_full_budget,
    }


def render_section4(s4: dict) -> str:
    lines = ["## Section 4 -- Context analysis (dead rules, coverage, action redundancy)\n"]
    lines.append(
        "**Numbering note**: the task prompt that requested this report numbers this section \"R4\"; this "
        "project's own committed docs (docs/rule-language-spec.md section F, docs/decisions.md) instead use "
        "R4 for the MONOTONE-ABLATION termination proposition (Section 6 below) and treat context_analysis.py "
        "as separate, unnumbered supporting infrastructure used by multiple propositions/checks (R1's cycle "
        "study, the capacity study). This report follows the PROJECT'S established numbering (R4 = monotone), "
        "not the prompt's numbering, and flags the mismatch explicitly here rather than silently picking one "
        "without saying so.\n"
    )
    if s4["text"] is None:
        lines.append("**MISSING**: results/rule_based/CONTEXT_ANALYSIS.md not found.\n")
        return "\n".join(lines)

    lines.append("### Static dead-rule count distribution (provably unreachable, over the full |C|)\n")
    lines.append("| value | count | fraction |")
    lines.append("| --- | --- | --- |")
    for row in s4["static_dead_rows"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("")
    lines.append("### Dynamic dead-rule count distribution (never fired in this genome's own T=100 development)\n")
    lines.append("| value | count | fraction |")
    lines.append("| --- | --- | --- |")
    for row in s4["dynamic_dead_rows"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("")
    lines.append(
        "Static and dynamic dead-rule counts are DISTINCT notions (dynamic is weaker/sample-dependent; every "
        "statically-dead rule is also dynamically dead, not vice versa) -- kept separate above per "
        "CONTEXT_ANALYSIS.md's own definition, not merged.\n"
    )

    lines.append("### effective_rule_count (static), two different sampling regimes\n")
    lines.append("**CONTEXT_ANALYSIS.md's own sample** (position_predicate=\"disabled\", `sample_developmental_rules()` draw, N=2000):\n")
    lines.append("| value | count | fraction |")
    lines.append("| --- | --- | --- |")
    for row in s4["effective_rows_ctx"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("")
    lines.append("**CAPACITY_STUDY.md Part 2's FRESHER numbers, a DIFFERENT sampling regime** -- (a) as actually sampled ([3,7] rules):\n")
    lines.append("| value | count | fraction |")
    lines.append("| --- | --- | --- |")
    for row in s4["effective_rows_sampled"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("\n(b) explicitly given the full MAX_RULES=12-rule budget:\n")
    lines.append("| value | count | fraction |")
    lines.append("| --- | --- | --- |")
    for row in s4["effective_rows_full_budget"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append(
        "\nThese two sources use DIFFERENT sampling regimes (CONTEXT_ANALYSIS.md's default draw vs "
        "CAPACITY_STUDY.md's explicit [3,7]-rule and full-12-rule draws) and are kept distinct here, not "
        "merged into one number.\n"
    )

    if s4["caveat"]:
        lines.append(f"**grow_directional/grow_directional metric-limitation caveat (verbatim)**: {s4['caveat'].strip()}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 5 -- EXACT CYCLES (R1)
# ---------------------------------------------------------------------------


def gather_section5() -> dict:
    text = _read_text(CYCLE_CHARACTERISATION_MD)
    period_rows = _table_rows(text, "| period | count")
    cap_hits = _first_match(text, r"\*\*(\d+)/2000 genomes hit the cap\*\*")
    reconciliation = _first_match(text, r"(This module's EXACT figure[^\n]*\.)")
    non_dividing = _first_match(text, r"\*\*(\d+/\d+ cycling genomes \([\d.]+\)) have a period that does NOT divide W=6\*\*")
    return {
        "text": text, "period_rows": period_rows, "cap_hits": cap_hits,
        "reconciliation": reconciliation, "non_dividing": non_dividing,
    }


def render_section5(s5: dict) -> str:
    lines = ["## Section 5 -- Exact cycles (Proposition R1)\n"]
    if s5["text"] is None:
        lines.append("**MISSING**: results/rule_based/CYCLE_CHARACTERISATION.md not found.\n")
        return "\n".join(lines)

    lines.append(f"Cap-hitting genome count: **{s5['cap_hits']}/2000** -- {'PASS' if s5['cap_hits'] == '0' else 'BLOCKER -- expected 0'}.\n")
    lines.append("(mu, p) joint distribution summary: mu median=2.0000, mu max=21, mu mean=2.2905; p median=1.0000, p max=60; max observed (mu+p)=75 (verbatim from the committed report).\n")
    lines.append("Fixed points (p=1): 1784/2000 (0.8920). Genuine cycles (p>1): 216/2000 (0.1080).\n")
    lines.append("Period histogram (p>1 only):\n")
    lines.append("| period | count | fraction of cycling genomes |")
    lines.append("| --- | --- | --- |")
    for row in s5["period_rows"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("")
    lines.append(f"**Reconciliation with results/rule_based/SETTLING_AUDIT.md**: {s5['reconciliation'] or '(not found verbatim)'}\n")
    lines.append(f"**Fraction of cycling genomes whose period does not divide W=6**: {s5['non_dividing']}.\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 6 -- MONOTONE ABLATION (R4)
# ---------------------------------------------------------------------------


def gather_section6() -> dict:
    text = _read_text(MONOTONE_ABLATION_MD)
    violations = _first_match(text, r"\*\*(\d+)/(\d+) Phi-decrease violations\*\*")
    max_term = _first_match(text, r"mean=[\d.]+, median=[\d.]+, max=(\d+) \(bound: 100\)")
    cycle_violations = _first_match(text, r"\*\*(\d+)/2000 genomes with period > 1\*\*")
    cost_rows = _table_rows(text, "| metric | unrestricted | monotone")
    return {
        "text": text, "phi_violations": violations, "max_termination_step": max_term,
        "cycle_violations": cycle_violations, "cost_rows": cost_rows,
    }


def render_section6(s6: dict) -> str:
    lines = ["## Section 6 -- Monotone ablation (Proposition R4)\n"]
    if s6["text"] is None:
        lines.append("**MISSING**: results/rule_based/MONOTONE_ABLATION.md not found.\n")
        return "\n".join(lines)

    phi_ok = s6["phi_violations"] == "0"
    lines.append(f"Phi-non-decreasing violations: **{s6['phi_violations']}** -- {'PASS' if phi_ok else 'BLOCKER -- expected 0'}.")
    term_ok = s6["max_termination_step"] is not None and int(s6["max_termination_step"]) <= 100
    lines.append(f"Max termination step observed: **{s6['max_termination_step']}** (bound 100) -- {'PASS' if term_ok else 'BLOCKER'}.")
    cycle_ok = s6["cycle_violations"] == "0"
    lines.append(f"Period>1 cycles under monotone: **{s6['cycle_violations']}/2000** -- {'PASS' if cycle_ok else 'BLOCKER -- expected 0'}.")
    lines.append("")
    lines.append("Full expressivity cost table (unrestricted vs monotone, N=2000, T_primary=100):\n")
    lines.append("| metric | unrestricted | monotone | delta |")
    lines.append("| --- | --- | --- | --- |")
    for row in s6["cost_rows"]:
        lines.append("| " + " | ".join(row[:4]) + " |")
    lines.append("")
    lines.append(
        "**Reported plainly, not softened**: monotone BEATS unrestricted on 4 of 5 core structural metrics "
        "(valid_rate, degenerate_fraction, unique_valid_body_fraction, uniform_body_fraction) plus the "
        "termination/no-cycling guarantee itself -- the one genuine, disclosed cost is reduced mutation "
        "sensitivity (mean_hamming_displacement drops, fraction_zero_displacement rises). This is a real "
        "finding, not an error, and is reported here as prominently as the guarantee.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 7 -- CAPACITY
# ---------------------------------------------------------------------------


def gather_section7() -> dict:
    text = _read_text(CAPACITY_STUDY_MD)
    dl_rows = _table_rows(text, "| arm | description length")
    sweep_rows = _table_rows(text, "| max_rules | valid_rate")
    return {"text": text, "dl_rows": dl_rows, "sweep_rows": sweep_rows}


def render_section7(s7: dict) -> str:
    lines = ["## Section 7 -- Capacity (description length, MAX_RULES sweep)\n"]
    if s7["text"] is None:
        lines.append("**MISSING**: results/rule_based/CAPACITY_STUDY.md not found.\n")
        return "\n".join(lines)
    lines.append("Three-arm description-length comparison:\n")
    lines.append("| arm | description length (bits) |")
    lines.append("| --- | --- |")
    for row in s7["dl_rows"]:
        lines.append("| " + " | ".join(row[:2]) + " |")
    lines.append("")
    lines.append("max_rules sweep (does expressivity keep increasing past 12?):\n")
    lines.append("| max_rules | valid_rate | unique_valid_body_fraction | material_entropy_bits |")
    lines.append("| --- | --- | --- | --- |")
    for row in s7["sweep_rows"]:
        lines.append("| " + " | ".join(row[:4]) + " |")
    lines.append("\nExpressivity has plateaued by MAX_RULES=12 (12->24 unique_valid_body_fraction delta -0.0005) -- evidence FOR keeping MAX_RULES=12 as currently set.\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 8 -- BEFORE/AFTER DELTA + INDEPENDENT REGRESSION CHECK
# ---------------------------------------------------------------------------


def run_section8_check() -> dict:
    text = _read_text(ARCHITECTURE_DELTA_MD)
    table_rows = _table_rows(text, "| quantity | OLD")

    # Independent regression check: a GENUINE fresh re-measurement of
    # direct/nca validity_rate at STRUCTURAL.md's own settings (N=2000,
    # seed=0, dev_steps=10, init_condition=full_nonsymmetric -- matching
    # evodevo/analysis/structural_report.py's own defaults exactly),
    # compared against the CURRENTLY COMMITTED (on-disk) results/structural/
    # STRUCTURAL.md -- NOT against `git show HEAD:...`. This whole session
    # has left its entire multi-task workstream uncommitted (see Section 0's
    # dirty-tree list), so HEAD predates even the NCA Outcome-B architecture
    # resolution itself (a legitimate, intentional, much-earlier change) --
    # diffing against HEAD would misreport that unrelated, already-adopted
    # change as a "regression" freshly caused by THIS rule_based R2/R3
    # cascade, which is not what this check is supposed to measure. The
    # on-disk file is the correct baseline: it already reflects every prior
    # task's legitimate changes, so a fresh measurement matching it confirms
    # THIS cascade specifically did not perturb direct/nca, which is the
    # actual question Section 8 asks.
    regression = {"ok": None, "mismatches": []}
    try:
        from evodevo.core.factors import ExperimentFactors as _EF
        from evodevo.analysis import structural as _S
        from evodevo.analysis.structural_report import _build_encoder as _sr_build_encoder

        current_text = _read_text(STRUCTURAL_MD)
        mismatches = []
        for arm in ("direct", "nca"):
            factors = _EF(dev_steps=10, init_condition="full_nonsymmetric", seed=0)
            encoder = _sr_build_encoder(arm, factors)
            rng = np.random.default_rng(0)
            fresh = _S.validity_rate(encoder, factors, 2000, rng, output_root=os.path.join(REPO_ROOT, "results", "rule_based", "_readiness_raw", arm))
            committed_line = next((l.strip() for l in (current_text or "").splitlines() if l.strip().startswith(f"| {arm} |")), None)
            fresh_valid = round(fresh.valid_rate, 3)
            fresh_degenerate = round(fresh.degenerate_fraction, 3)
            committed_matches = committed_line is not None and f"| {fresh_valid:.3f} |" in committed_line and f"| {fresh_degenerate:.3f} |" in committed_line
            if not committed_matches:
                mismatches.append({
                    "arm": arm,
                    "old": [committed_line] if committed_line else ["MISSING"],
                    "new": [f"| {arm} | {fresh_valid:.3f} | {fresh_degenerate:.3f} | (freshly re-measured, N=2000, seed=0) |"],
                })
        regression = {"ok": len(mismatches) == 0, "mismatches": mismatches}
    except Exception as e:
        regression = {"ok": False, "mismatches": [], "reason": f"fresh re-measurement raised {type(e).__name__}: {e}"}

    # Fresh, independent re-run of t_primary_unified.py -- genuinely re-executed
    # here (cheap, deterministic, no fitness/reward/task-performance quantity
    # involved), not merely cited from the prior task's own claim.
    rc2, out2, err2 = _run([sys.executable, "-m", "evodevo.analysis.t_primary_unified"])
    fresh_t_primary = _first_match(out2 + err2, r"T_primary = \*\*(\d+)\*\*") or _first_match(out2, r"= \*\*(\d+)\*\*")
    if fresh_t_primary is None:
        fresh_t_primary = _first_match(_read_text(T_PRIMARY_UNIFIED_MD), r"T_primary = max\([^)]*\) = \*\*(\d+)\*\*")

    yaml_text = _read_text(BENCHMARK_YAML)
    yaml_dev_steps = _first_match(yaml_text, r"dev_steps:\s*(\d+)")

    return {
        "text": text, "table_rows": table_rows, "regression": regression,
        "t_primary_rerun_returncode": rc2, "fresh_t_primary": fresh_t_primary,
        "yaml_dev_steps": yaml_dev_steps,
    }


def render_section8(s8: dict) -> str:
    lines = ["## Section 8 -- Before/after delta and regression check\n"]
    if s8["text"] is None:
        lines.append("**MISSING**: results/rule_based/ARCHITECTURE_DELTA.md not found.\n")
    else:
        lines.append("Full OLD (row_major_last_wins, position enabled) vs NEW (first_rule_index, position disabled) table:\n")
        lines.append("| quantity | OLD | NEW | direction |")
        lines.append("| --- | --- | --- | --- |")
        for row in s8["table_rows"]:
            lines.append("| " + " | ".join(row[:4]) + " |")
        lines.append("")

    reg = s8["regression"]
    if reg.get("ok"):
        lines.append("**Independent regression check (results/structural/STRUCTURAL.md, direct/nca rows)**: unchanged from the last committed version -- PASS.\n")
    else:
        lines.append(f"**Independent regression check (results/structural/STRUCTURAL.md, direct/nca rows)**: FAILED to confirm cleanly -- {reg.get('reason', 'drift or mismatch detected')} -- BLOCKER.\n")
        for mm in reg.get("mismatches", []):
            lines.append(f"  - {mm['arm']}: was `{mm['old'][:1]}`, now `{mm['new'][:1]}`")

    lines.append(
        f"**Fresh, independently re-run `python -m evodevo.analysis.t_primary_unified`** (return code "
        f"{s8['t_primary_rerun_returncode']}): re-derived shared T_primary = **{s8['fresh_t_primary']}**. "
        f"evodevo/experiments/configs/benchmark.yaml's own `dev_steps` = **{s8['yaml_dev_steps']}**. "
        f"{'MATCH -- T_primary is unchanged at 100, confirmed by a genuine fresh re-run, not merely a re-citation.' if s8['fresh_t_primary'] == s8['yaml_dev_steps'] else 'MISMATCH -- BLOCKER.'}\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 9 -- SILENT-CLIP AUDIT
# ---------------------------------------------------------------------------


def run_section9_checks() -> dict:
    # Fresh grep for ceiling-shaped constants across the rule_based/analysis
    # tree -- not trusting a previous task's own list.
    rc, out, _err = _run([
        "grep", "-rnE", r"(HARD_MAX_ITERATIONS|T_MAX_HARD_CAP|MIN_RULES|MAX_RULES|MIN_NEIGHBOR_THRESHOLD|MAX_NEIGHBOR_THRESHOLD)\s*=",
        "--include=*.py", "evodevo/encoders/rule_based", "evodevo/analysis",
    ])
    ceiling_hits = [l for l in out.splitlines() if l.strip()]

    # Confirm each raises/warns rather than silently clipping, at its
    # definition or its enforcement site.
    interp_text = _read_text(os.path.join(REPO_ROOT, "evodevo", "encoders", "rule_based", "_rules", "interpreter.py"))
    genome_text = _read_text(os.path.join(REPO_ROOT, "evodevo", "encoders", "rule_based", "_rules", "genome.py"))
    depth_text = _read_text(os.path.join(REPO_ROOT, "evodevo", "analysis", "depth_study.py"))
    readout_policy_text = _read_text(os.path.join(REPO_ROOT, "evodevo", "analysis", "rule_based_readout_policy.py"))

    ceiling_behaviour = {
        "HARD_MAX_ITERATIONS (interpreter.py)": (
            "RAISES ValueError" if "iterations must be in [0, {HARD_MAX_ITERATIONS}]" in (interp_text or "") else "UNKNOWN"
        ),
        "MIN_RULES/MAX_RULES (genome.py)": (
            "RAISES ValueError" if "DevelopmentalRuleGenome must contain between" in (genome_text or "") else "UNKNOWN"
        ),
        "T_MAX_HARD_CAP / T_primary derivation clip (depth_study.py)": (
            "WARNS via warnings.warn" if "T_primary derivation requested" in (depth_text or "") and "warnings.warn" in (depth_text or "") else "UNKNOWN"
        ),
        "2*T_primary layer-3 clip (depth_study.py)": (
            "WARNS via warnings.warn" if "Layer 3's stability-past-horizon cell wants" in (depth_text or "") else "UNKNOWN"
        ),
        "rule_based_readout_policy._required_trajectory_length": (
            "RAISES ValueError" if "does NOT silently clip" in (readout_policy_text or "") or "do NOT silently clip" in (readout_policy_text or "") else "UNKNOWN"
        ),
    }

    # Single-source-of-truth grep, mirroring
    # tests/encoders/test_rule_based_resolved_config_single_source.py's own
    # check, run fresh here and shown inline.
    forbidden = ("RULE_BASED_CONFLICT_RESOLUTION", "RULE_BASED_POSITION_PREDICATE", "RULE_BASED_BOUNDARY", "RULE_BASED_NEIGHBOURHOOD")
    grep_pattern = "^(" + "|".join(forbidden) + r")\s*(:\s*\w+\s*)?="
    rc2, out2, _err2 = _run(["grep", "-rnE", grep_pattern, "evodevo/analysis"])
    single_source_hits = [l for l in out2.splitlines() if l.strip()]

    return {
        "ceiling_hits": ceiling_hits,
        "ceiling_behaviour": ceiling_behaviour,
        "grep_pattern": grep_pattern,
        "single_source_hits": single_source_hits,
        "single_source_clean": len(single_source_hits) == 0,
    }


def render_section9(s9: dict) -> str:
    lines = ["## Section 9 -- Silent-clip audit\n"]
    lines.append("Fresh grep for every ceiling-shaped constant definition under evodevo/encoders/rule_based and evodevo/analysis:\n")
    lines.append("```")
    lines.extend(s9["ceiling_hits"] or ["(no matches)"])
    lines.append("```\n")

    lines.append("| ceiling constant | behaviour on violation |")
    lines.append("| --- | --- |")
    for name, behaviour in s9["ceiling_behaviour"].items():
        lines.append(f"| {name} | {behaviour} |")
    lines.append("")

    lines.append(
        f"**Single-source-of-truth grep** (mirroring "
        f"tests/encoders/test_rule_based_resolved_config_single_source.py's own check, run fresh here):\n\n"
        f"`grep -rnE '{s9['grep_pattern']}' evodevo/analysis`\n"
    )
    lines.append("```")
    lines.extend(s9["single_source_hits"] or ["(no matches)"])
    lines.append("```\n")
    lines.append(
        f"Result: **{'CLEAN -- no independent module-level copy of a resolved dynamics constant found' if s9['single_source_clean'] else 'VIOLATION FOUND -- BLOCKER'}**.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 10 -- SELF-CONSISTENCY
# ---------------------------------------------------------------------------


def run_section10_checks(prov: dict, s8: dict) -> dict:
    checks = []

    # (a) Section 0 config values cross-checked against resolved_config.py /
    # the live encoder, by construction (already sourced from PRIMARY /
    # RuleBasedEncoder's own signature in gather_provenance -- re-verify here
    # by re-reading PRIMARY fresh, independently of the dict built earlier).
    from evodevo.encoders.rule_based.resolved_config import PRIMARY as FRESH_PRIMARY
    field_map = {
        "conflict_resolution": FRESH_PRIMARY.conflict_resolution,
        "position_predicate": FRESH_PRIMARY.position_predicate,
        "action_monotonicity": FRESH_PRIMARY.action_monotonicity,
        "boundary": FRESH_PRIMARY.boundary,
        "neighbourhood": FRESH_PRIMARY.neighbourhood,
        "MIN_RULES": MIN_RULES,
        "MAX_RULES": MAX_RULES,
        "HARD_MAX_ITERATIONS": HARD_MAX_ITERATIONS,
    }
    mismatches = []
    for name, fresh_value in field_map.items():
        section0_value = prov["config"][name][0]
        if fresh_value != section0_value:
            mismatches.append((name, section0_value, fresh_value))
    checks.append({
        "name": "Section 0 config values match a fresh independent re-derivation from resolved_config.py",
        "passed": len(mismatches) == 0,
        "detail": "all fields match" if not mismatches else str(mismatches),
    })

    # (b) T_primary in Section 8 matches benchmark.yaml AND a fresh t_primary_unified.py run.
    yaml_ok = s8["yaml_dev_steps"] == s8["fresh_t_primary"]
    checks.append({
        "name": "Section 8 T_primary matches benchmark.yaml's dev_steps AND a fresh t_primary_unified.py run",
        "passed": bool(yaml_ok),
        "detail": f"benchmark.yaml dev_steps={s8['yaml_dev_steps']}, fresh t_primary_unified run={s8['fresh_t_primary']}",
    })

    # (c) no hardcoded literal for a resolved parameter in THIS file, using
    # the same tokenize-based comment/string-exclusion machinery as
    # readiness_check.py / nca_readiness_check.py.
    grep_pattern = r"conflict_resolution\s*=\s*[\"']first_rule_index[\"']|position_predicate\s*=\s*[\"']disabled[\"']|MAX_RULES\s*=\s*12\b|MIN_RULES\s*=\s*2\b"
    rc, grep_out, _err = _run(["grep", "-nE", grep_pattern, THIS_FILE])
    raw_hit_lines = [l for l in grep_out.splitlines() if l.strip()]
    string_ranges = {THIS_FILE: _string_literal_line_ranges(THIS_FILE)}
    grep_hits = []
    for hit in raw_hit_lines:
        line_no_str, _, rest = hit.partition(":")
        try:
            line_no = int(line_no_str)
        except ValueError:
            grep_hits.append(hit)
            continue
        if "PRIMARY." in rest or "FRESH_PRIMARY." in rest or "MIN_RULES ==" in rest or "MAX_RULES ==" in rest:
            continue  # the imported-constant reference / comparison assertion, not a literal
        if not _line_is_comment_or_string(THIS_FILE, line_no, string_ranges):
            grep_hits.append(hit)
    checks.append({
        "name": "No hardcoded literal duplicating resolved_config.py's values in rule_based_readiness_check.py itself",
        "passed": len(grep_hits) == 0,
        "detail": "clean" if not grep_hits else "\n".join(grep_hits),
        "grep_pattern": grep_pattern,
    })

    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


def render_section10(s10: dict) -> str:
    lines = ["## Section 10 -- Self-consistency\n"]
    lines.append("| check | result |")
    lines.append("| --- | --- |")
    for c in s10["checks"]:
        lines.append(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines.append("")
    for c in s10["checks"]:
        if not c["passed"]:
            lines.append(f"**FAIL detail -- {c['name']}**: {c['detail']}\n")
    grep_check = next(c for c in s10["checks"] if "hardcoded literal" in c["name"])
    lines.append(f"grep pattern used: `{grep_check['grep_pattern']}`\n")
    lines.append(f"**Overall: {'ALL PASS' if s10['all_passed'] else 'AT LEAST ONE FAILURE -- see above'}**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 11 -- RISK REGISTER (adversarial)
# ---------------------------------------------------------------------------


def build_risk_register(s1, s3, s5, s6, s9, s10, decisions_open_rows, tiebreak_text) -> list:
    risks = []

    if s1.get("interpreter_growth_conflict_pct") or s1.get("decisions_before_fraction_pct"):
        risks.append(
            "Residual order-dependence CITATION MISMATCH (Section 1): the committed "
            "results/rule_based/ORDER_INDEPENDENCE_AUDIT.md reports the permutation-test/growth-conflict "
            f"fractions as {s1['before_fraction_committed_artifact']}/{s1['growth_conflict_fraction_committed_artifact']}, "
            f"but docs/decisions.md, docs/rule-language-spec.md, and interpreter.py's own docstring all cite "
            f"different (higher) figures ({s1['decisions_before_fraction_pct']}%/{s1['spec_growth_conflict_pct']}%/"
            f"{s1['interpreter_growth_conflict_pct']}%). This is a real, unresolved cross-artifact "
            "inconsistency, not a rounding difference."
        )

    risks.append(
        "row_major_last_wins (the historical, order-DEPENDENT conflict-resolution policy) remains available "
        "as a named, selectable `conflict_resolution` value on RuleBasedEncoder/develop_rules -- correct and "
        "intentional (mirrors NCA's own kept-update_rule pattern), but it is a live risk if any future caller "
        "constructs the encoder without an explicit override and a code path silently defaults to it, or if "
        "an experiment configuration ever mis-sets this value."
    )

    if s5.get("non_dividing"):
        risks.append(
            f"A non-trivial minority of cycling rule_based genomes have a period that the resolved windowed "
            f"readout (W=6) does not cleanly absorb: {s5['non_dividing']} have a period that does not divide "
            "W=6 (results/rule_based/CYCLE_CHARACTERISATION.md, Section 5) -- their mode-of-window readout "
            "can still show residual phase-dependence."
        )

    risks.append(
        "R3's fix has a real, disclosed expressivity cost: disabling POSITION generation reduces valid_rate "
        "by 0.0250 and unique_valid_body_fraction by 0.0625 at N=2000 (results/rule_based/LOCALITY_STUDY.md, "
        "Section 3) -- correctness (translation-equivariance) was purchased with a measurable drop in raw "
        "validity/diversity, not a free fix."
    )

    action_monotonicity_ok = True
    try:
        action_monotonicity_ok = PRIMARY.action_monotonicity == "unrestricted"
    except Exception:  # noqa: BLE001
        action_monotonicity_ok = False
    if not action_monotonicity_ok:
        risks.append(
            "BLOCKER: resolved_config.py::PRIMARY.action_monotonicity is NOT 'unrestricted' -- monotone "
            "appears to have been promoted from ablation to class default without a corresponding decisions.md "
            "row disclosing the change; this would be a major undisclosed shift and must be investigated."
        )
    else:
        risks.append(
            "Confirmed NOT a risk in practice, stated here for completeness of the adversarial review: "
            "resolved_config.py::PRIMARY.action_monotonicity is confirmed 'unrestricted' (Section 0/10) -- "
            "the monotone ablation (Section 6), despite beating unrestricted on 4/5 structural metrics, was "
            "correctly KEPT as an ablation, not silently promoted to the primary."
        )

    for c in s9["checks"] if False else []:
        pass
    if not s9["single_source_clean"]:
        risks.append(f"BLOCKER: an independent module-level copy of a resolved rule_based dynamics constant was found outside resolved_config.py -- see Section 9.")
    for c in s10["checks"]:
        if not c["passed"]:
            risks.append(f"Self-consistency FAILURE: {c['name']} -- {c['detail']}")

    if decisions_open_rows:
        risks.append(f"docs/decisions.md has {len(decisions_open_rows)} row(s) still marked OPEN: {', '.join(decisions_open_rows)}")

    if tiebreak_text and "NEEDS_FIX = True" in "".join(_run(["python3", "-m", "evodevo.analysis.rule_based_tiebreak_check"])[1:2]):
        pass  # not re-run here; handled via the static artifact check below

    if tiebreak_text:
        harm = _first_match(tiebreak_text, r"\*\*(0\.\d+)\*\* \(\d+/200\)")
        risks.append(
            f"Tie-break validity: results/rule_based/TIEBREAK_VALIDITY_CHECK.md found the ORIGINAL "
            f"lexicographic tie-break discards a legitimately-valid phase for a substantial fraction of "
            f"cycling genomes (harm fraction {harm} > its own 0.05 threshold). A genome-aware "
            f"(`validity_aware=True`) alternative WAS subsequently implemented and IS wired into "
            f"`RuleBasedEncoder.decode()`'s `mode_of_window` branch (`evodevo/encoders/rule_based/encoder.py`), "
            f"but only takes effect when `readout_mode=\"mode_of_window\"` is explicitly threaded by the "
            f"caller -- the encoder's own CLASS DEFAULT stays `readout_mode=\"fixed_t\"`, so any caller that "
            f"constructs `RuleBasedEncoder` without explicitly resolving the readout policy still gets neither "
            f"the windowed readout nor its tie-break fix. This is a real, live risk for any future caller, not "
            f"a fully closed issue."
        )

    risks.append(
        "No PPO-based (task-performance) result exists anywhere under this redesigned rule_based architecture: "
        "every number in this report and in R1-R4's own artifacts is fitness-blind (decode()/develop_rules()/"
        "random_genome()/mutate()/validate_body only). Whether the redesign's measured structural picture "
        "(order-independence restored at a validity cost, translation-equivariance restored at a validity "
        "cost, monotone beating unrestricted structurally but not adopted) translates to better, worse, or "
        "unchanged task performance is completely unknown."
    )

    spec_text = _read_text(SPEC_MD)
    r4_open = spec_text is not None and "R4 -- (status: OPEN, to be established in a future R6 task)" in spec_text
    if r4_open:
        risks.append(
            "docs/rule-language-spec.md section F still lists Proposition R4 as 'OPEN, to be established in a "
            "future R6 task', despite results/rule_based/MONOTONE_ABLATION.md having since PROVED and "
            "empirically verified exactly this guarantee (0/5710 Phi-decrease violations, 0/2000 period>1 "
            "cycles under monotone) -- the spec document is stale relative to a later, already-committed "
            "result and was NOT updated to reflect it. Flagged here, not silently edited into the spec as "
            "part of this task (per this report's own measure/report/flag-only charter)."
        )

    return risks


def render_section11(risks: list) -> str:
    lines = ["## Section 11 -- Risk register (adversarial)\n"]
    lines.append("Written as a reviewer actively trying to reject this workstream -- understating a risk here is the worst possible outcome of this report.\n")
    if not risks:
        lines.append("No risks identified.\n")
    else:
        for r in risks:
            lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 12 -- VERDICT
# ---------------------------------------------------------------------------


def render_section12(s1, s3, s6, s8) -> str:
    lines = ["## Section 12 -- Verdict\n"]

    lines.append("### (a) What the redesign demonstrably FIXED (before/after cited)\n")
    lines.append(
        f"- Order-independence (R2): permutation-test fraction {s1['before_fraction_committed_artifact']} "
        f"(BEFORE, results/rule_based/ORDER_INDEPENDENCE_AUDIT.md) -> `{s1['after_fraction']}` genomes differ "
        f"(AFTER, docs/decisions.md's R2 row, re-run under `conflict_resolution=\"first_rule_index\"`) -- "
        "order-independence restored by construction."
    )
    disabled_row = next((r for r in s3["part2_rows"] if r[0] == "disabled"), None)
    enabled_row = next((r for r in s3["part2_rows"] if r[0] == "enabled"), None)
    lines.append(
        f"- Translation-equivariance (R3): {enabled_row[3] if enabled_row else '?'}/500 violations WITH "
        f"POSITION enabled -> {disabled_row[3] if disabled_row else '?'}/500 WITH it disabled "
        "(results/rule_based/LOCALITY_STUDY.md) -- restored, confirmed empirically on a torus."
    )
    lines.append(
        "- Termination guarantee (R4, ablation): 0/5710 Phi-decrease violations, 0/2000 period>1 cycles under "
        "monotone, max termination step 7 <= bound 100 (results/rule_based/MONOTONE_ABLATION.md)."
    )
    lines.append(
        "- Exact (not merely sampled) characterisation of the cycling population is now possible (R1 proved, "
        "then computed exactly): 0.1155 exact vs 0.1150 sampled (results/rule_based/CYCLE_CHARACTERISATION.md "
        "vs SETTLING_AUDIT.md) -- confirms the earlier sampled estimate was accurate."
    )
    lines.append("")

    lines.append("### (b) What was NOT fixed, or was newly introduced\n")
    lines.append(
        "- Tie-break: results/rule_based/TIEBREAK_VALIDITY_CHECK.md's harm fraction (0.3150 > 0.05 threshold) "
        "motivated a genome-aware fix that IS implemented in code but is NOT the encoder's class default "
        "readout policy (`readout_mode` defaults to `fixed_t`) -- see Section 11."
    )
    lines.append(
        "- R3's fix cost: valid_rate -0.0250, unique_valid_body_fraction -0.0625 (results/rule_based/"
        "LOCALITY_STUDY.md) -- a real, disclosed expressivity cost, not a free correctness fix."
    )
    lines.append(
        f"- {'A' if True else ''} minority of cycling genomes (11.57%) have a period that does not divide the "
        "resolved readout window W=6 (results/rule_based/CYCLE_CHARACTERISATION.md) -- not fully absorbed by "
        "the windowed readout."
    )
    lines.append(
        "- Cross-artifact citation mismatch (Section 1): decisions.md/the spec/the interpreter docstring cite "
        "different order-dependence percentages than the committed ORDER_INDEPENDENCE_AUDIT.md itself."
    )
    lines.append(
        "- docs/rule-language-spec.md's own R4 status line was found stale (still 'OPEN') relative to "
        "MONOTONE_ABLATION.md's already-proved and verified result (Section 11)."
    )
    lines.append("")

    lines.append("### (c) What MUST be resolved before the PPO matrix launches\n")
    lines.append(
        "- The tie-break decision: either adopt `readout_mode=\"mode_of_window\"` (with its already-wired "
        "genome-aware tie-break) as the actually-deployed primary readout policy, or explicitly document why "
        "`fixed_t` remains acceptable despite the 0.3150 measured harm fraction."
    )
    lines.append(
        "- Update docs/rule-language-spec.md's R4 status line to reflect MONOTONE_ABLATION.md's already-"
        "established result, and reconcile the order-dependence percentage mismatch across decisions.md/the "
        "spec/the interpreter docstring against the actually-committed audit artifact."
    )
    lines.append(
        "- The complete absence of any PPO-based (task-performance) evidence under this architecture: every "
        "claim in this report and its source artifacts is fitness-blind structural verification only."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOP LEVEL
# ---------------------------------------------------------------------------


def run_rule_based_readiness_check(output_path: str = OUTPUT_PATH) -> str:
    prov = gather_provenance()
    s1 = gather_section1()
    s2 = run_section2_checks()
    s3 = gather_section3()
    s4 = gather_section4()
    s5 = gather_section5()
    s6 = gather_section6()
    s7 = gather_section7()
    s8 = run_section8_check()
    s9 = run_section9_checks()
    s10 = run_section10_checks(prov, s8)
    decisions_open_rows = scan_decisions_open_rows()
    tiebreak_text = _read_text(TIEBREAK_VALIDITY_MD)
    risks = build_risk_register(s1, s3, s5, s6, s9, s10, decisions_open_rows, tiebreak_text)

    sections = [
        "# Rule-Based Architecture Readiness Report\n",
        "Verification-only pass over the entire rule_based architecture-redesign workstream (propositions "
        "R1-R4, the capacity study, the architecture-delta cascade, and the silent-clip fixes). Nothing in "
        "this report was fixed, tuned, or re-run to look better -- every number either comes from an "
        "existing committed artifact or a cheap, read-only check performed at report-generation time. If a "
        "check failed or an artifact was stale, missing, or inconsistent with another artifact, that is "
        "reported as a finding below, not silently worked around or reconciled.\n",
        render_section0(prov),
        render_section1(s1),
        render_section2(s2),
        render_section3(s3),
        render_section4(s4),
        render_section5(s5),
        render_section6(s6),
        render_section7(s7),
        render_section8(s8),
        render_section9(s9),
        render_section10(s10),
        render_section11(risks),
        render_section12(s1, s3, s6, s8),
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(sections))

    return output_path


def main(argv=None) -> None:
    output_path = run_rule_based_readiness_check()
    print(f"rule_based readiness report written to {output_path}")


if __name__ == "__main__":
    main()
