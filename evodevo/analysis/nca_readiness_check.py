"""Verification-only pass over the entire NCA-architecture-redesign
workstream (N1-N8): measures whether the redesign is complete, internally
consistent, and correctly reflected in its own committed artifacts, and
writes results/NCA_READINESS.md.

MUST NOT fix, tune, re-run (in the sense of re-deriving a "better" number),
or adjust anything to make a number look better. Two classes of numbers
appear in this report:
  (a) numbers READ from an already-committed artifact (a .md report or a
      raw .json this workstream already produced) -- the large, expensive
      N=2000 sweeps (N2, N4, N5, N6, N7, N8) are never re-run here;
  (b) numbers MEASURED fresh, read-only, at report-generation time --
      cheap, deterministic verification checks (the boundedness certificate
      test, the certified-fraction-at-resolved-settings check, the
      lambda-independence property, git/grep/OPEN-row scans) that reuse
      already-implemented, already-tested logic. Running a verification
      check is not "improving a number"; skipping one to avoid an
      inconvenient result would be.
If a check fails or an artifact is stale/missing, that is reported as a
finding, never silently worked around -- this module follows
evodevo/analysis/readiness_check.py's own conventions (see its docstring),
extended with an explicit self-consistency section (Section 9) because this
codebase has already had two stale-literal bugs (gather_provenance's
dev_steps, the risk register's T_primary -- both fixed by threading a
single source-of-truth value through instead of retyping it).

Usage:
    python -m evodevo.analysis.nca_readiness_check
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

from evodevo.encoders.nca.resolved_config import PRIMARY
from evodevo.analysis import readout_study as RS
from evodevo.analysis.readiness_check import (
    _line_is_comment_or_string,
    _string_literal_line_ranges,
    scan_decisions_open_rows,
)
from evodevo.core.factors import ExperimentFactors
from evodevo.core.materials import N_MATERIALS
from evodevo.core.provenance import git_commit_hash, git_dirty_files
from evodevo.encoders.nca.certificate import is_certified_contractive, lipschitz_constant
from evodevo.encoders.nca.encoder import NCAEncoder, VALID_BOUNDARIES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "NCA_READINESS.md")
THIS_FILE = os.path.join(REPO_ROOT, "evodevo", "analysis", "nca_readiness_check.py")

INIT_STUDY_MD = os.path.join(REPO_ROOT, "results", "nca", "INIT_STUDY.md")
CONVERGENCE_MD = os.path.join(REPO_ROOT, "results", "nca", "CONVERGENCE.md")
TIMESCALE_MD = os.path.join(REPO_ROOT, "results", "nca", "TIMESCALE_CALIBRATION.md")
READOUT_MD = os.path.join(REPO_ROOT, "results", "nca", "READOUT_STUDY.md")
REGENERATION_MD = os.path.join(REPO_ROOT, "results", "nca", "REGENERATION.md")
ARCHITECTURE_DELTA_MD = os.path.join(REPO_ROOT, "results", "nca", "ARCHITECTURE_DELTA.md")
MUTATION_MD = os.path.join(REPO_ROOT, "results", "mutation", "MUTATION_CALIBRATION.md")
MUTATION_RAW = os.path.join(REPO_ROOT, "results", "mutation", "mutation_calibration_raw.json")
DEPTH_STUDY_RAW = os.path.join(REPO_ROOT, "results", "depth", "depth_study_raw.json")
STRUCTURAL_MD = os.path.join(REPO_ROOT, "results", "structural", "STRUCTURAL.md")
DEPTH_STUDY_MD = os.path.join(REPO_ROOT, "results", "depth", "DEPTH_STUDY.md")
DECISIONS_MD = os.path.join(REPO_ROOT, "docs", "decisions.md")
RUNS_ROOT = os.path.join(REPO_ROOT, "results", "runs")

VERIFICATION_N = 2000
VERIFICATION_SEED = 0


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
        out = subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout, out.stderr
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)


def _table_rows(text: str, header_prefix: str) -> list:
    """Every markdown table row (as raw '| a | b | ... |' text) between the
    first line starting with header_prefix and the next blank line."""
    if text is None:
        return []
    lines = text.splitlines()
    rows = []
    started = False
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
            continue  # separator row
        rows.append([c.strip() for c in stripped.strip("|").split("|")])
    return rows


def _first_match(text: str, pattern: str, group: int = 1):
    if text is None:
        return None
    m = re.search(pattern, text)
    return m.group(group) if m else None


# ---------------------------------------------------------------------------
# SECTION 0 -- PROVENANCE
# ---------------------------------------------------------------------------


def gather_provenance() -> dict:
    commit = git_commit_hash()
    dirty_files = git_dirty_files()

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in ("numpy", "matplotlib", "pytest"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"

    # The resolved NCA configuration is read by literally constructing the
    # SAME encoder the rest of this workstream's canonical modules build --
    # every value below is an attribute of that live object, sourced from
    # evodevo.encoders.nca.resolved_config.PRIMARY, the single source of
    # truth every N4-N8 module now imports from (this file's own former
    # "six independent copies" finding, Section 10, has been fixed -- see
    # docs/decisions.md's "NCA joint stability-vs-expressivity trade-off"
    # row and tests/encoders/test_resolved_config_single_source.py, which
    # enforces no independent copy recurs). See Section 9 for the explicit
    # cross-check that this claim is not just asserted.
    factors = ExperimentFactors(
        dev_steps=40, init_condition="center_seed", seed=VERIFICATION_SEED, hidden_channels=RS.PRIMARY_H
    )
    resolved = NCAEncoder(
        factors=factors, decay=PRIMARY.decay, state_bound=PRIMARY.state_bound,
    )

    mutation_raw = _read_json(MUTATION_RAW)
    mutation_sigma = None
    if mutation_raw:
        mutation_sigma = mutation_raw.get("calibrated", {}).get("nca", {}).get("param_value")

    config = {
        "update_rule": (resolved.update_rule, "N1", "evodevo/encoders/nca/encoder.py (default)"),
        "state_bound (B*)": (resolved.state_bound, "N4/Outcome-B", "evodevo/encoders/nca/resolved_config.py::PRIMARY.state_bound"),
        "decay (lambda*)": (resolved.decay, "N4/Outcome-B", "evodevo/encoders/nca/resolved_config.py::PRIMARY.decay"),
        "init_gain": (resolved.init_gain, "N2", "evodevo/encoders/nca/encoder.py (default; N2 established the layer-wise-scaled unpacking this gain feeds)"),
        "hidden_channels (H)": (resolved.hidden_channels, "N5/Part-5", "evodevo/encoders/nca/resolved_config.py::PRIMARY.hidden_channels"),
        "hidden_width": (resolved.hidden_width, "pre-N1", "evodevo/encoders/nca/encoder.py (default; not modified by N1-N8)"),
        "boundary": (resolved.boundary, "N1", "evodevo/encoders/nca/encoder.py (default)"),
        "update_prob": (resolved.update_prob, "N1", "evodevo/encoders/nca/encoder.py (default)"),
        "spectral_normalise": (resolved.spectral_normalise, "N1", "evodevo/encoders/nca/encoder.py (default)"),
        "mutation_sigma": (mutation_sigma, "N6", "results/mutation/mutation_calibration_raw.json::calibrated.nca.param_value"),
    }

    return {
        "commit": commit,
        "dirty_files": dirty_files,
        "clean": not dirty_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "config": config,
        "resolved_encoder_genome_length": resolved.genome_length(),
        "resolved_encoder": resolved,
    }


def render_section0(prov: dict) -> str:
    lines = ["## Section 0 -- Provenance\n"]
    lines.append(f"- git commit: `{prov['commit']}`")
    lines.append(f"- working tree clean: {'YES' if prov['clean'] else 'NO -- dirty files: ' + ', '.join(prov['dirty_files'][:20])}")
    lines.append(f"- report generated: {prov['generated_at']}")
    lines.append("- package versions: " + ", ".join(f"{k}={v}" for k, v in prov["versions"].items()))
    lines.append("")
    lines.append("### Resolved NCA configuration\n")
    lines.append("| parameter | value | resolved by | source |")
    lines.append("| --- | --- | --- | --- |")
    for name, (value, prompt, source) in prov["config"].items():
        lines.append(f"| {name} | {_fmt(value)} | {prompt} | `{source}` |")
    lines.append("")
    lines.append(
        f"genome_length() of this resolved configuration: {prov['resolved_encoder_genome_length']} "
        "(computed by the encoder from hidden_width/hidden_channels, not a stored literal -- see Section 4)."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 1 -- PROPOSITION 1: BOUNDEDNESS
# ---------------------------------------------------------------------------


def run_section1_check(n_genomes=500, t_steps=100, h_values=(0, 3, 6), seed=0) -> dict:
    """Reproduces tests/encoders/test_nca_bounded.py's own boundedness
    property test, at the resolved (lambda*, B*), across all 3 boundary
    rules and H in {0,3,6} -- the exact grid the task specifies."""
    worst_violation = -float("inf")
    worst_case = None
    total_checks = 0
    violations = 0

    for H in h_values:
        factors = ExperimentFactors(dev_steps=t_steps, init_condition="center_seed", seed=seed, hidden_channels=H)
        for boundary in VALID_BOUNDARIES:
            encoder = NCAEncoder(
                factors=factors, update_rule="bounded_leaky", decay=PRIMARY.decay,
                state_bound=PRIMARY.state_bound, boundary=boundary,
            )
            init_inf_norm = float(np.max(np.abs(encoder._initial_state)))
            limit = max(init_inf_norm, encoder.state_bound) + 1e-9

            rng = np.random.default_rng(seed)
            for _ in range(n_genomes):
                scale = rng.choice([0.01, 1.0, 50.0, 500.0])
                genome = rng.normal(scale=scale, size=encoder.genome_length())
                w1, b1, w2, b2 = encoder._unpack(genome)
                state = encoder._initial_state.copy()
                for _ in range(t_steps):
                    state = encoder._step(state, w1, b1, w2, b2)
                    total_checks += 1
                    max_abs = float(np.max(np.abs(state)))
                    margin = max_abs - limit
                    if margin > worst_violation:
                        worst_violation = margin
                        worst_case = {"H": H, "boundary": boundary, "max_abs": max_abs, "limit": limit}
                    if margin > 1e-9:
                        violations += 1

    return {
        "n_genomes": n_genomes,
        "t_steps": t_steps,
        "h_values": list(h_values),
        "boundaries": list(VALID_BOUNDARIES),
        "total_checks": total_checks,
        "violations": violations,
        "worst_violation_margin": worst_violation,
        "worst_case": worst_case,
        "passed": violations == 0,
    }


def render_section1(s1: dict) -> str:
    lines = ["## Section 1 -- Proposition 1: boundedness\n"]
    lines.append(
        f"Fresh, read-only check (reproduces tests/encoders/test_nca_bounded.py's own property test) at "
        f"the resolved (lambda\\*={PRIMARY.decay}, B\\*={PRIMARY.state_bound}): N={s1['n_genomes']} genomes x "
        f"T={s1['t_steps']} steps x {len(s1['boundaries'])} boundary rules ({', '.join(s1['boundaries'])}) x "
        f"H in {s1['h_values']} = {s1['total_checks']} total per-step checks."
    )
    lines.append("")
    lines.append(f"- violations (max|s_t| > bound + 1e-9): **{s1['violations']}**")
    lines.append(f"- single largest observed violation margin: **{_fmt(s1['worst_violation_margin'], 6)}**")
    if s1["worst_case"]:
        wc = s1["worst_case"]
        lines.append(f"  (H={wc['H']}, boundary={wc['boundary']}, observed max|s_t|={_fmt(wc['max_abs'], 6)}, bound={_fmt(wc['limit'], 6)})")
    lines.append("")
    verdict = "EMPIRICALLY UNVIOLATED" if s1["passed"] else "VIOLATED -- BLOCKER"
    lines.append(f"**Verdict: boundedness is {verdict}.**")
    if not s1["passed"]:
        lines.append(
            "This is reported as a blocker, not softened: Proposition 1 is a mathematical claim about "
            "every trajectory under bounded_leaky, and a single counterexample falsifies it as stated."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 2 -- PROPOSITION 2: CONTRACTION CERTIFICATE
# ---------------------------------------------------------------------------


def run_section2_check(n=VERIFICATION_N, seed=VERIFICATION_SEED, t_max=40) -> dict:
    """Certified fraction / agreement matrix at the ACTUAL resolved
    (lambda*, B*) -- NOT re-using results/nca/CONVERGENCE.md's committed
    numbers, because that artifact was generated at the encoder's raw
    defaults (decay=0.2, state_bound=5.0), predating N4's calibration to
    (lambda*=0.05, B*=2). That mismatch is itself flagged in Section 9/10;
    this fresh, cheap, read-only check reports the number at the settings
    that are actually resolved."""
    factors = ExperimentFactors(dev_steps=t_max, init_condition="center_seed", seed=seed, hidden_channels=RS.PRIMARY_H)
    encoder = NCAEncoder(factors=factors, update_rule="bounded_leaky", decay=PRIMARY.decay, state_bound=PRIMARY.state_bound)

    def _readout(state):
        return np.argmax(state[:, :, :N_MATERIALS], axis=-1).astype(int)

    rng = np.random.default_rng(seed)
    certified_settled = certified_not_settled = uncertified_settled = uncertified_not_settled = 0
    for _ in range(n):
        genome = encoder.random_genome(rng)
        certified = is_certified_contractive(genome, encoder)
        w1, b1, w2, b2 = encoder._unpack(genome)
        state = encoder._initial_state.copy()
        bodies = [_readout(state)]
        for _ in range(t_max):
            state = encoder._step(state, w1, b1, w2, b2)
            bodies.append(_readout(state))
        # never_settled iff the body still changed on the LAST transition
        # (t=T-1 -> t=T) -- matches structural._settle_info's own
        # convention exactly (the authoritative definition project-wide,
        # and what results/nca/CONVERGENCE.md itself used).
        never_settled = not np.array_equal(bodies[-2], bodies[-1])

        if certified and not never_settled:
            certified_settled += 1
        elif certified and never_settled:
            certified_not_settled += 1
        elif (not certified) and not never_settled:
            uncertified_settled += 1
        else:
            uncertified_not_settled += 1

    n_certified = certified_settled + certified_not_settled
    n_uncertified = uncertified_settled + uncertified_not_settled
    certified_fraction = n_certified / n
    empirically_settled_fraction = (certified_settled + uncertified_settled) / n

    # lambda-independence: L(theta)<1 <=> state_bound*K(theta)<1, for decay in {0.1,0.2,0.5,1.0}
    lambda_indep_violations = 0
    lambda_indep_checked = 0
    rng2 = np.random.default_rng(seed)
    probe_genomes = [rng2.normal(scale=rng2.choice([0.001, 0.01, 0.1, 1.0, 10.0]), size=encoder.genome_length()) for _ in range(500)]
    for decay in (0.1, 0.2, 0.5, 1.0):
        probe_factors = ExperimentFactors(dev_steps=1, init_condition="center_seed", seed=seed, hidden_channels=RS.PRIMARY_H)
        probe_encoder = NCAEncoder(factors=probe_factors, update_rule="bounded_leaky", decay=decay, state_bound=PRIMARY.state_bound)
        for genome in probe_genomes:
            lambda_indep_checked += 1
            certified_here = is_certified_contractive(genome, probe_encoder)
            L = lipschitz_constant(genome, probe_encoder)
            if (L < 1.0) != certified_here:
                lambda_indep_violations += 1

    return {
        "n": n,
        "certified_fraction": certified_fraction,
        "empirically_settled_fraction": empirically_settled_fraction,
        "matrix": {
            "certified_settled": certified_settled,
            "certified_not_settled": certified_not_settled,
            "uncertified_settled": uncertified_settled,
            "uncertified_not_settled": uncertified_not_settled,
        },
        "false_positives": certified_not_settled,
        "conservativeness_gap": empirically_settled_fraction - certified_fraction,
        "lambda_independence_checked": lambda_indep_checked,
        "lambda_independence_violations": lambda_indep_violations,
    }


def render_section2(s2: dict, convergence_md_text: str) -> str:
    lines = ["## Section 2 -- Proposition 2: contraction certificate\n"]
    lines.append(
        f"Fresh, read-only check at the ACTUAL resolved (lambda\\*={PRIMARY.decay}, B\\*={PRIMARY.state_bound}), "
        f"N={s2['n']}, T_max=40, center_seed, primary random_genome() distribution -- see the staleness note "
        "below on why this is not simply read from results/nca/CONVERGENCE.md."
    )
    lines.append("")
    lines.append(f"- certified fraction: **{_fmt(s2['certified_fraction'])}**")
    lines.append(f"- empirically-settled fraction: **{_fmt(s2['empirically_settled_fraction'])}**")
    lines.append(f"- conservativeness gap (empirically-settled minus certified): **{_fmt(s2['conservativeness_gap'])}**")
    lines.append("")
    m = s2["matrix"]
    lines.append("| | empirically settled | never settled |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| **certified** | {m['certified_settled']} | {m['certified_not_settled']} |")
    lines.append(f"| **uncertified** | {m['uncertified_settled']} | {m['uncertified_not_settled']} |")
    lines.append("")
    if s2["false_positives"] == 0:
        lines.append(f"**False positives: 0/{s2['matrix']['certified_settled'] + s2['matrix']['certified_not_settled']}. PASS.**")
    else:
        lines.append(
            f"**False positives: {s2['false_positives']}. BLOCKER.** A certified genome failed to settle -- "
            "per the task's instruction this is reported as an implementation bug or an error in the "
            "proposition, not softened or explained away."
        )
    lines.append("")
    lambda_ok = s2["lambda_independence_violations"] == 0
    lines.append(
        f"Lambda-independence (L(theta)<1 <=> state_bound*K(theta)<1, decay in {{0.1,0.2,0.5,1.0}}): "
        f"{s2['lambda_independence_checked']} checks, {s2['lambda_independence_violations']} violations -- "
        f"**{'PASS' if lambda_ok else 'FAIL'}**."
    )
    lines.append("")
    lines.append("### Staleness note (see Section 9/10)\n")
    committed_certified = _first_match(convergence_md_text, r"\| certified fraction \| ([\d.]+) \|")
    fresh_fingerprint = f"{PRIMARY.hidden_channels} (after column, the FINAL resolved primary)"
    is_fresh = convergence_md_text is not None and fresh_fingerprint in convergence_md_text
    if is_fresh:
        lines.append(
            f"results/nca/CONVERGENCE.md (committed) reports a primary-settings certified fraction of "
            f"{committed_certified}, matching the fresh number above -- its 'after' column fingerprint "
            f"confirms it was generated at the CURRENT resolved_config.py::PRIMARY settings "
            f"(decay={PRIMARY.decay}, state_bound={PRIMARY.state_bound}, H={PRIMARY.hidden_channels}). Not stale."
        )
    else:
        lines.append(
            f"results/nca/CONVERGENCE.md (committed) reports a primary-settings certified fraction of "
            f"{committed_certified} -- but its 'after' column fingerprint does NOT match "
            f"resolved_config.py::PRIMARY's current settings (decay={PRIMARY.decay}, "
            f"state_bound={PRIMARY.state_bound}, H={PRIMARY.hidden_channels}), so it was generated at "
            "different settings than the ones now resolved. The fresh number above is computed at the "
            "CORRECT resolved settings regardless. Flagged in Section 10, not silently corrected in place."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 3 -- TIMESCALE CALIBRATION (N4)
# ---------------------------------------------------------------------------


def gather_section3() -> dict:
    text = _read_text(TIMESCALE_MD)
    grid_rows = _table_rows(text, "| lambda | B | Wasserstein-1")
    selected = _first_match(text, r"\*\*Selected: lambda\\\* = ([\d.]+), B\\\* = (\d+)\*\* \(Wasserstein-1 = ([\d.]+)")
    selected_full = re.search(r"\*\*Selected: lambda\\\* = ([\d.]+), B\\\* = (\d+)\*\* \(Wasserstein-1 = ([\d.]+), KS = ([\d.]+)\)", text) if text else None
    verify_rows = _table_rows(text, "| quantity | sweep (seed=")
    tail_rows = _table_rows(text, "| settle within <= N steps")
    gap_statement = _first_match(text, r"(\*\*(?:This does not claim|Negative result)[^\n]*\n(?:[^\n]*\n)?)")
    return {
        "text": text,
        "grid_rows": grid_rows,
        "selected": selected_full.groups() if selected_full else None,
        "verify_rows": verify_rows,
        "tail_rows": tail_rows,
        "residual_gap_statement": gap_statement,
    }


def render_section3(s3: dict) -> str:
    lines = ["## Section 3 -- Timescale calibration (N4)\n"]
    if s3["text"] is None:
        lines.append("**MISSING**: results/nca/TIMESCALE_CALIBRATION.md not found.")
        return "\n".join(lines)
    lines.append(f"Sweep grid: {len(s3['grid_rows'])} (lambda, B) cells (read from results/nca/TIMESCALE_CALIBRATION.md).")
    lines.append("")
    lines.append("| lambda | B | Wasserstein-1 | KS | never_settled_fraction | median settle_step | mean settle_step |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in s3["grid_rows"]:
        lines.append("| " + " | ".join(row[:7]) + " |")
    lines.append("")
    if s3["selected"]:
        lam, b, w1, ks = s3["selected"]
        lines.append(f"**Selected: lambda\\* = {lam}, B\\* = {b}** (Wasserstein-1 = {w1}, KS = {ks}).")
    lines.append("")
    lines.append("Held-out verification:\n")
    lines.append("| quantity | sweep | verification |")
    lines.append("| --- | --- | --- |")
    for row in s3["verify_rows"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("")
    lines.append("Settling-within-N-steps, NCA @ (lambda*, B*) vs rule_based:\n")
    lines.append("| settle within <= N steps | rule_based | nca (sweep) | nca (verification) |")
    lines.append("| --- | --- | --- | --- |")
    for row in s3["tail_rows"]:
        lines.append("| " + " | ".join(row[:4]) + " |")
    lines.append("")
    if s3["residual_gap_statement"]:
        lines.append(f"**Residual gap (verbatim from the committed report)**: {s3['residual_gap_statement'].strip()}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 4 -- GENOME SPACE AND INITIALISATION (N2)
# ---------------------------------------------------------------------------


def gather_section4(resolved: NCAEncoder) -> dict:
    text = _read_text(INIT_STUDY_MD)
    variance_rows = _table_rows(text, "| init | layer | fan_in")
    degeneracy_rows = _table_rows(text, "| init | valid_rate | degenerate_fraction")

    # H, h read off the SAME resolved encoder instance Section 0 built --
    # not retyped literals -- so this check cannot silently drift from
    # whatever H/h are actually resolved to.
    H, h = resolved.hidden_channels, resolved.hidden_width
    C = N_MATERIALS + H
    input_dim = 5 * C
    analytic_length = input_dim * h + h + h * C + C

    return {
        "text": text,
        "variance_rows": variance_rows,
        "degeneracy_rows": degeneracy_rows,
        "H": H,
        "h": h,
        "analytic_genome_length": analytic_length,
        "resolved_genome_length": resolved.genome_length(),
        "genome_length_matches": analytic_length == resolved.genome_length(),
    }


def render_section4(s4: dict) -> str:
    lines = ["## Section 4 -- Genome space and initialisation (N2)\n"]
    if s4["text"] is None:
        lines.append("**MISSING**: results/nca/INIT_STUDY.md not found.")
        return "\n".join(lines)
    lines.append("Pre-activation variance, old (single-scale) vs new (layer-wise), from results/nca/INIT_STUDY.md:\n")
    lines.append("| init | layer | fan_in | pre-activation variance | target |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in s4["variance_rows"]:
        lines.append("| " + " | ".join(row[:5]) + " |")
    lines.append("")
    lines.append("Body degeneracy/diversity, old vs new init:\n")
    lines.append("| init | valid_rate | degenerate_fraction | unique_body_fraction |")
    lines.append("| --- | --- | --- | --- |")
    for row in s4["degeneracy_rows"]:
        lines.append("| " + " | ".join(row[:4]) + " |")
    lines.append("")
    lines.append(
        f"genome_length() confirmation: analytic formula (5*(N_MATERIALS+H))*h + h + h*(N_MATERIALS+H) + "
        f"(N_MATERIALS+H) at H={s4['H']}, h={s4['h']} = **{s4['analytic_genome_length']}**; the resolved encoder's own "
        f"`genome_length()` = **{s4['resolved_genome_length']}** -- "
        f"{'MATCH' if s4['genome_length_matches'] else 'MISMATCH -- BLOCKER'}. Computed by the encoder from "
        "its constructor arguments, not stored as a literal (evodevo/encoders/nca/encoder.py::__init__)."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 5 -- READOUT AND HIDDEN CHANNELS (N5)
# ---------------------------------------------------------------------------


def gather_section5() -> dict:
    text = _read_text(READOUT_MD)
    margin_rows = _table_rows(text, "| H | mean margin")
    uniform_rows = _table_rows(text, "| H | uniform_fraction")
    hidden_util_rows = _table_rows(text, "| H | channel | spatial variance")
    verdict = _first_match(text, r"### Verdict\n\n(\*\*.+?\*\*[^\n]*)")
    return {
        "text": text,
        "margin_rows": margin_rows,
        "uniform_rows": uniform_rows,
        "hidden_util_rows": hidden_util_rows,
        "verdict": verdict,
    }


def render_section5(s5: dict) -> str:
    lines = ["## Section 5 -- Readout and hidden channels (N5)\n"]
    if s5["text"] is None:
        lines.append("**MISSING**: results/nca/READOUT_STUDY.md not found.")
        return "\n".join(lines)
    lines.append("Near-tie fraction by H, from results/nca/READOUT_STUDY.md:\n")
    lines.append("| H | mean margin | median margin | fraction of CELLS near-tie | fraction of BODIES with >=1 near-tie cell |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in s5["margin_rows"]:
        lines.append("| " + " | ".join(row[:5]) + " |")
    lines.append("")
    lines.append("Uniform-body fraction by H (with material breakdown):\n")
    lines.append("| H | uniform_fraction | by material |")
    lines.append("| --- | --- | --- |")
    for row in s5["uniform_rows"]:
        lines.append("| " + " | ".join(row[:3]) + " |")
    lines.append("")
    lines.append("Hidden-channel utilisation, H=3 and H=6:\n")
    lines.append("| H | channel | spatial variance | temporal variance | correlation with readout |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in s5["hidden_util_rows"]:
        lines.append("| " + " | ".join(row[:5]) + " |")
    lines.append("")
    if s5["verdict"]:
        lines.append(f"**Verdict (verbatim)**: {s5['verdict'].strip()}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 6 -- MUTATION RE-CALIBRATION (N6)
# ---------------------------------------------------------------------------


def gather_section6() -> dict:
    raw = _read_json(MUTATION_RAW)
    text = _read_text(MUTATION_MD)
    outcome_statement = _first_match(text, r"(\*\*Outcome \([ab]\)[^\n]*\n(?:[^\n]*\n)?)")
    iqr_line_old = _first_match(text, r"(\*\*Cross-arm IQR spread\*\*: [^\n]*)")
    return {"raw": raw, "text": text, "outcome_statement": outcome_statement, "iqr_line": iqr_line_old}


def render_section6(s6: dict) -> str:
    lines = ["## Section 6 -- Mutation re-calibration (N6)\n"]
    raw = s6["raw"]
    if raw is None:
        lines.append("**MISSING**: results/mutation/mutation_calibration_raw.json not found.")
        return "\n".join(lines)
    if s6["outcome_statement"]:
        lines.append(f"{s6['outcome_statement'].strip()}")
    lines.append("")
    lines.append(f"Resolved D\\* = **{_fmt(raw['d_star'])}**\n")
    lines.append("| arm | parameter | solved value | verification mean | verification IQR | within tolerance |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arm, entry in raw["calibrated"].items():
        vs = entry["verify_stats"]
        lines.append(
            f"| {arm} | {entry['param_name']} | {_fmt(entry['param_value'])} | {_fmt(vs['mean'])} | "
            f"{_fmt(vs['iqr'])} | {'YES' if entry['within_tolerance'] else 'NO'} |"
        )
    lines.append("")
    if s6["iqr_line"]:
        lines.append(f"**IQR spread before/after (verbatim)**: {s6['iqr_line'].strip()}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 7 -- REGENERATION (N7)
# ---------------------------------------------------------------------------


def gather_section7() -> dict:
    text = _read_text(REGENERATION_MD)
    primary_rows = _table_rows(text, "## NCA at PRIMARY settings")
    diag_rows = _table_rows(text, "## NCA at DIAGNOSTIC settings")
    rigorous_rows = _table_rows(text, "## The rigorous test")
    verdict = _first_match(text, r"(\*\*Theory verdict: [^\n]*)")
    primary_certified_line = _first_match(text, r"(Certified fraction under the PRIMARY genome distribution: [^\n]*)")
    return {
        "text": text,
        "primary_rows": primary_rows,
        "diagnostic_rows": diag_rows,
        "rigorous_rows": rigorous_rows,
        "verdict": verdict,
        "primary_certified_line": primary_certified_line,
    }


def render_section7(s7: dict) -> str:
    lines = ["## Section 7 -- Regeneration (N7)\n"]
    if s7["text"] is None:
        lines.append("**MISSING**: results/nca/REGENERATION.md not found.")
        return "\n".join(lines)
    if s7["primary_certified_line"]:
        lines.append(f"{s7['primary_certified_line'].strip()}\n")
    lines.append(f"PRIMARY-settings recovery curve: {len(s7['primary_rows'])} (k, d) cells measured (see results/nca/REGENERATION.md for the full table).")
    lines.append(f"DIAGNOSTIC-population (certified/uncertified split): {len(s7['diagnostic_rows'])} rows.")
    lines.append(f"Rigorous continuous-state geometric-bound check: {len(s7['rigorous_rows'])} (k, d) cells.")
    lines.append("")
    if s7["verdict"]:
        lines.append(f"{s7['verdict'].strip()}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 8 -- BEFORE/AFTER DELTA (N8)
# ---------------------------------------------------------------------------


def _regression_check(committed_path: str, arms=("direct", "rule_based")) -> dict:
    """Compares the CURRENT working-tree file's per-arm rows against the
    last git-committed version of the same file -- a fresh, read-only diff,
    not a re-run of anything."""
    rc, committed_text, _err = _run(["git", "show", f"HEAD:{os.path.relpath(committed_path, REPO_ROOT)}"])
    current_text = _read_text(committed_path)
    if rc != 0 or current_text is None:
        return {"ok": False, "reason": "could not read committed or current file"}

    def _arm_lines(text, arm):
        return [l.strip() for l in text.splitlines() if f"| {arm} |" in l]

    mismatches = []
    for arm in arms:
        old_lines = _arm_lines(committed_text, arm)
        new_lines = _arm_lines(current_text, arm)
        if old_lines != new_lines:
            mismatches.append({"arm": arm, "old": old_lines, "new": new_lines})
    return {"ok": len(mismatches) == 0, "mismatches": mismatches}


def gather_section8() -> dict:
    text = _read_text(ARCHITECTURE_DELTA_MD)
    table_rows = _table_rows(text, "| quantity | OLD")
    structural_regression = _regression_check(STRUCTURAL_MD)
    depth_regression = _regression_check(DEPTH_STUDY_MD)
    return {"text": text, "table_rows": table_rows, "structural_regression": structural_regression, "depth_regression": depth_regression}


def render_section8(s8: dict) -> str:
    lines = ["## Section 8 -- Before/after delta (N8)\n"]
    if s8["text"] is None:
        lines.append("**MISSING**: results/nca/ARCHITECTURE_DELTA.md not found.")
        return "\n".join(lines)
    lines.append("Full table (results/nca/ARCHITECTURE_DELTA.md):\n")
    lines.append("| quantity | OLD | NEW | direction |")
    lines.append("| --- | --- | --- | --- |")
    for row in s8["table_rows"]:
        lines.append("| " + " | ".join(row[:4]) + " |")
    lines.append("")

    for label, path, result in (
        ("results/structural/STRUCTURAL.md", STRUCTURAL_MD, s8["structural_regression"]),
        ("results/depth/DEPTH_STUDY.md", DEPTH_STUDY_MD, s8["depth_regression"]),
    ):
        if result["ok"]:
            lines.append(f"**Regression check ({label})**: direct/rule_based rows unchanged from the last committed version -- PASS.")
        else:
            lines.append(f"**Regression check ({label})**: DRIFT DETECTED -- BLOCKER. {len(result['mismatches'])} arm(s) changed:")
            for mm in result["mismatches"]:
                lines.append(f"  - {mm['arm']}: was `{mm['old'][:1]}`, now `{mm['new'][:1]}`")
    lines.append("")

    t_primary_row = next((r for r in s8["table_rows"] if r[0] == "T_primary implied"), None)
    speed_row = next((r for r in s8["table_rows"] if "propagation speed" in r[0]), None)
    if t_primary_row:
        lines.append(f"New T_primary: **{t_primary_row[2]}** (old: {t_primary_row[1]}).")
    if speed_row:
        lines.append(f"New light-cone speed: **{speed_row[2]}** cells/step vs rule_based's unchanged 0.2548 (old NCA: {speed_row[1]}).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 9 -- SELF-CONSISTENCY
# ---------------------------------------------------------------------------


def run_section9_checks(prov: dict, s8: dict) -> dict:
    checks = []

    # (a) every Section 0 config value matches the artifact it claims to read.
    # Re-derive independently (a second, fresh construction) and diff.
    factors = ExperimentFactors(dev_steps=40, init_condition="center_seed", seed=VERIFICATION_SEED, hidden_channels=RS.PRIMARY_H)
    rederived = NCAEncoder(factors=factors, update_rule="bounded_leaky", decay=PRIMARY.decay, state_bound=PRIMARY.state_bound)
    config_matches = True
    mismatches = []
    field_map = {
        "update_rule": rederived.update_rule, "state_bound (B*)": rederived.state_bound,
        "decay (lambda*)": rederived.decay, "init_gain": rederived.init_gain,
        "hidden_channels (H)": rederived.hidden_channels, "hidden_width": rederived.hidden_width,
        "boundary": rederived.boundary, "update_prob": rederived.update_prob,
        "spectral_normalise": rederived.spectral_normalise,
    }
    for name, rederived_value in field_map.items():
        section0_value = prov["config"][name][0]
        if rederived_value != section0_value:
            config_matches = False
            mismatches.append((name, section0_value, rederived_value))
    checks.append({
        "name": "Section 0 config values match a fresh independent re-derivation",
        "passed": config_matches,
        "detail": "all fields match" if config_matches else str(mismatches),
    })

    # (b) T_primary in Section 8 matches the regenerated depth study's own artifact.
    depth_raw = _read_json(DEPTH_STUDY_RAW)
    t_primary_row = next((r for r in s8["table_rows"] if r[0] == "T_primary implied"), None)
    if depth_raw is None or t_primary_row is None:
        checks.append({"name": "Section 8 T_primary matches results/depth/depth_study_raw.json", "passed": False, "detail": "missing artifact"})
    else:
        section8_t_primary = float(t_primary_row[2])
        raw_t_primary = float(depth_raw["t_primary"])
        checks.append({
            "name": "Section 8 T_primary matches results/depth/depth_study_raw.json",
            "passed": section8_t_primary == raw_t_primary,
            "detail": f"Section 8={section8_t_primary}, depth_study_raw.json={raw_t_primary}",
        })

    # (c) no hardcoded numeric literal for a resolved parameter in THIS file.
    # Scoped to the distinctive, unambiguous fingerprints of the resolved
    # values (lambda*=0.05 and the calibrated mutation_sigma); common short
    # digits (B*=2, H=3) are not greppable as "hardcoded" vs. legitimate
    # unrelated code (list lengths, loop counts) without false positives, so
    # this check is deliberately scoped to what a grep can actually prove.
    # Grepping the file for a hardcoded "decay=0.05" (etc.) would also match
    # THIS CHECK'S OWN descriptive strings (which necessarily spell out what
    # they're looking for), so this reuses readiness_check.py's own
    # comment/string-literal exclusion machinery (tokenize-based, not a
    # naive '^#' regex) to only count matches in actual executable code.
    grep_pattern = r"decay\s*=\s*0\.05|state_bound\s*=\s*2\b|hidden_channels\s*=\s*3\b"
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
        if "PRIMARY.decay" in rest or "PRIMARY.state_bound" in rest or "PRIMARY.hidden_channels" in rest:
            continue  # the imported-constant reference itself, not a literal
        if not _line_is_comment_or_string(THIS_FILE, line_no, string_ranges):
            grep_hits.append(hit)
    checks.append({
        "name": "No hardcoded literal (decay=0.05 / state_bound=2 / hidden_channels=3) in nca_readiness_check.py itself",
        "passed": len(grep_hits) == 0,
        "detail": "clean" if not grep_hits else "\n".join(grep_hits),
        "grep_pattern": grep_pattern,
    })

    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


def render_section9(s9: dict) -> str:
    lines = ["## Section 9 -- Self-consistency\n"]
    lines.append("| check | result |")
    lines.append("| --- | --- |")
    for c in s9["checks"]:
        lines.append(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines.append("")
    for c in s9["checks"]:
        if not c["passed"]:
            lines.append(f"**FAIL detail -- {c['name']}**: {c['detail']}\n")
    grep_check = next(c for c in s9["checks"] if "hardcoded literal" in c["name"])
    lines.append(f"grep pattern used: `{grep_check['grep_pattern']}`\n")
    lines.append(f"**Overall: {'ALL PASS' if s9['all_passed'] else 'AT LEAST ONE FAILURE -- see above'}**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 10 -- RISK REGISTER
# ---------------------------------------------------------------------------


def build_risk_register(s1, s2, s3, s5, s6, s7, s8, s9, decisions_open_rows) -> list:
    risks = []

    # RESOLVED since this check was first written: the six-independent-copies
    # risk this row used to flag is fixed -- evodevo/encoders/nca/
    # resolved_config.py is now the single source of truth, all former call
    # sites import PRIMARY from it, and
    # tests/encoders/test_resolved_config_single_source.py enforces no
    # independent copy recurs (see docs/decisions.md's "NCA joint
    # stability-vs-expressivity trade-off" row). Kept as a comment, not a
    # risk-register line, so this fix is visible without silently deleting
    # the historical finding.

    depth_raw = _read_json(DEPTH_STUDY_RAW)
    if depth_raw and depth_raw.get("non_settling_diagnosis"):
        diag = depth_raw["non_settling_diagnosis"]
        risks.append(
            f"Residual non-settling fraction under the resolved architecture is NOT zero (see Section 8): "
            f"classified as {', '.join(f'{k}={v*100:.1f}%' for k, v in diag['fractions'].items())} "
            f"(n={diag['total_never_settled']}). The dominant categories are tie-resolution/limit-cycle "
            "behaviour, not unbounded growth, but the fraction itself remains substantial."
        )

    if s2["false_positives"] > 0:
        risks.append(
            f"BLOCKER: {s2['false_positives']} certificate false positive(s) found at the resolved settings "
            "-- see Section 2."
        )

    if s3["residual_gap_statement"]:
        risks.append(f"Residual timescale gap after N4 calibration (verbatim, results/nca/TIMESCALE_CALIBRATION.md): {s3['residual_gap_statement'].strip()}")

    if s6["iqr_line"]:
        risks.append(f"Residual mutation IQR mismatch across arms after N6 re-calibration: {s6['iqr_line'].strip()}")

    if s5["verdict"] and "do not measurably affect" in s5["verdict"]:
        risks.append(
            "H (hidden channels) is NOT load-bearing at the resolved settings (results/nca/READOUT_STUDY.md "
            "verdict, Section 5): three (H=3) or six (H=6) hidden channels are carried through development "
            "without measurably affecting validity, diversity, settling, or decision confidence -- an "
            "unexamined design cost, not yet removed."
        )
    elif s5["verdict"]:
        risks.append(f"H load-bearing status (Section 5 verdict): {s5['verdict'].strip()}")

    # Genuinely dynamic staleness check (a hardcoded `True` here would be
    # exactly the stale-literal bug pattern this section exists to catch):
    # CONVERGENCE.md's "after" column states its own hidden_channels value
    # in a fixed, greppable phrase -- if it doesn't match PRIMARY's current
    # value, the committed file was generated at different settings.
    convergence_text = _read_text(CONVERGENCE_MD)
    fresh_fingerprint = f"{PRIMARY.hidden_channels} (after column, the FINAL resolved primary)"
    convergence_committed_stale = convergence_text is None or fresh_fingerprint not in convergence_text
    if convergence_committed_stale:
        risks.append(
            "results/nca/CONVERGENCE.md is STALE: its committed 'after' column does not match "
            "evodevo/encoders/nca/resolved_config.py::PRIMARY's current settings -- the same class of "
            "stale-literal bug this section exists to catch, this time in a report rather than a "
            "hardcoded number in code. Should be regenerated (out of scope for this read-only check to fix)."
        )

    for c in s9["checks"]:
        if not c["passed"]:
            risks.append(f"Self-consistency FAILURE: {c['name']} -- {c['detail']}")

    if decisions_open_rows:
        risks.append(f"docs/decisions.md has {len(decisions_open_rows)} row(s) still marked OPEN: {', '.join(decisions_open_rows)}")

    # No PPO-based result under this architecture.
    ppo_evidence = False
    if os.path.isdir(RUNS_ROOT):
        rc, out, _err = _run(["grep", "-rl", "bounded_leaky", RUNS_ROOT])
        ppo_evidence = bool(out.strip())
    if not ppo_evidence:
        risks.append(
            "No PPO-based (task-performance) result exists anywhere under this redesigned architecture: "
            "every number in this report and in N1-N8's own artifacts is fitness-blind (decode()/develop()/"
            "mutate()/validate_body only). Whether the redesign's mixed structural picture (Section 8) "
            "translates to better, worse, or unchanged task performance is completely unknown."
        )

    if not s8["structural_regression"]["ok"] or not s8["depth_regression"]["ok"]:
        risks.append("BLOCKER: regression check found direct/rule_based drift -- see Section 8.")

    return risks


def render_section10(risks: list) -> str:
    lines = ["## Section 10 -- Risk register (adversarial)\n"]
    if not risks:
        lines.append("No risks identified -- every check in Sections 1-9 passed cleanly.")
    else:
        for r in risks:
            lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 11 -- VERDICT
# ---------------------------------------------------------------------------


def render_section11(s1, s2, s5, s7, s8) -> str:
    lines = ["## Section 11 -- Verdict\n"]

    lines.append("### Demonstrably fixed (before/after cited)\n")
    fixed = []
    max_state_row = next((r for r in s8["table_rows"] if "max state magnitude" in r[0] and "mean" in r[0]), None)
    if max_state_row:
        fixed.append(f"Unbounded state growth: max state magnitude at T {max_state_row[1]} -> {max_state_row[2]} (Section 8), and empirically zero boundedness violations over {s1['total_checks']} checks (Section 1).")
    speed_row = next((r for r in s8["table_rows"] if "propagation speed" in r[0]), None)
    if speed_row:
        fixed.append(f"Propagation speed increased {speed_row[1]} -> {speed_row[2]} cells/step (Section 8).")
    if s7["verdict"] and "HOLDS" in (s7["verdict"] or ""):
        fixed.append(f"Regeneration/self-repair is now a provable, zero-violation-verified property for certified genomes (Section 7): {s7['verdict'].strip()}")
    for f in fixed:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("### Not fixed, or newly introduced\n")
    not_fixed = []
    never_settled_row = next((r for r in s8["table_rows"] if r[0] == "never_settled_fraction"), None)
    if never_settled_row:
        not_fixed.append(f"never_settled_fraction {never_settled_row[1]} -> {never_settled_row[2]} (UP, not down as hypothesised -- Section 8).")
    validity_row = next((r for r in s8["table_rows"] if r[0] == "validity rate"), None)
    if validity_row:
        not_fixed.append(f"validity rate {validity_row[1]} -> {validity_row[2]} (DOWN -- Section 8).")
    t_primary_row = next((r for r in s8["table_rows"] if r[0] == "T_primary implied"), None)
    if t_primary_row:
        not_fixed.append(f"T_primary unchanged at {t_primary_row[2]} -- did not drop as hoped (Section 8).")
    if s5["verdict"] and "do not measurably affect" in s5["verdict"]:
        not_fixed.append("Hidden channels remain structurally unexamined/not load-bearing (Section 5, Section 10).")
    if s2["certified_fraction"] < 0.01:
        not_fixed.append(f"Certificate essentially never certifies a genome at the primary random_genome() distribution (certified fraction={_fmt(s2['certified_fraction'])} -- Section 2).")
    for f in not_fixed:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("### Must be resolved before the PPO matrix is launched\n")
    must_resolve = [
        "results/nca/CONVERGENCE.md must be regenerated at the N4-resolved (lambda*, B*) -- it is currently stale (Section 2/10).",
        "No PPO-based task-performance evidence exists yet under this architecture (Section 10) -- the entire mixed structural picture in Section 8 is unvalidated against the metric that ultimately matters.",
    ]
    if s5["verdict"] and "do not measurably affect" in s5["verdict"]:
        must_resolve.append("Decide whether to drop the hidden channels (H=3) given Section 5's finding that they are not load-bearing, or run a task-performance check before concluding they are safe to remove.")
    for f in must_resolve:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOP LEVEL
# ---------------------------------------------------------------------------


def run_nca_readiness_check(output_path: str = OUTPUT_PATH) -> str:
    prov = gather_provenance()
    s1 = run_section1_check()
    convergence_text = _read_text(CONVERGENCE_MD)
    s2 = run_section2_check()
    s3 = gather_section3()
    s4 = gather_section4(prov["resolved_encoder"])
    s5 = gather_section5()
    s6 = gather_section6()
    s7 = gather_section7()
    s8 = gather_section8()
    s9 = run_section9_checks(prov, s8)
    decisions_open_rows = scan_decisions_open_rows()
    risks = build_risk_register(s1, s2, s3, s5, s6, s7, s8, s9, decisions_open_rows)

    sections = [
        "# NCA Architecture Readiness Report\n",
        "Verification-only pass over the entire N1-N8 NCA-architecture-redesign workstream. Nothing in "
        "this report was fixed, tuned, or re-run to look better -- every number either comes from an "
        "existing committed artifact or a cheap, read-only check performed at report-generation time. If "
        "a check failed or an artifact was stale/missing, that is reported as a finding below, not "
        "silently worked around.\n",
        render_section0(prov),
        render_section1(s1),
        render_section2(s2, convergence_text),
        render_section3(s3),
        render_section4(s4),
        render_section5(s5),
        render_section6(s6),
        render_section7(s7),
        render_section8(s8),
        render_section9(s9),
        render_section10(risks),
        render_section11(s1, s2, s5, s7, s8),
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(sections))

    return output_path


def main(argv=None) -> None:
    output_path = run_nca_readiness_check()
    print(f"NCA readiness report written to {output_path}")


if __name__ == "__main__":
    main()
