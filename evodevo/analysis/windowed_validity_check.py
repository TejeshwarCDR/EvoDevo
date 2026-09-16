"""PART 3 of the N15 follow-up: closes the "validity_rate and
unique_valid_body_fraction not re-measured under windowed readout" gap
FINAL_STRUCTURAL_COMPARISON.md flagged for both arms, at the CORRECTED
(T, W) values from Parts 1-2.

Reuses evodevo.analysis.structural.validity_rate/diversity directly (no
reimplementation) against encoders explicitly configured for
readout_mode="mode_of_window" at the resolved (T, W) per arm -- writes its
own jsonl records to a SEPARATE output_root (results/windowed_validity_check)
so the existing committed fixed_t baselines under results/structural/ and
results/rule_based/ are never overwritten.

Fitness-blind: only decode()/random_genome()/validate_body() are used (via
structural.py) -- no PPO, no task, no reward, no EA.

Usage:
    python -m evodevo.analysis.windowed_validity_check
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.nca.resolved_config import PRIMARY as NCA_PRIMARY
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

DEFAULT_N = 2000
DEFAULT_SEED = 0

# Resolved, post-N15-fix values.
SHARED_T_PRIMARY = 100
NCA_W = 50
RB_W = 6

OUTPUT_ROOT = "results/windowed_validity_check"


def _build_nca_windowed(seed: int) -> NCAEncoder:
    factors = ExperimentFactors(
        dev_steps=SHARED_T_PRIMARY, init_condition="center_seed", seed=seed,
        hidden_channels=NCA_PRIMARY.hidden_channels,
    )
    return NCAEncoder(
        factors=factors, update_rule=NCA_PRIMARY.update_rule, decay=NCA_PRIMARY.decay,
        state_bound=NCA_PRIMARY.state_bound, readout_mode="mode_of_window", readout_window=NCA_W,
    )


def _build_nca_fixed(seed: int) -> NCAEncoder:
    factors = ExperimentFactors(
        dev_steps=SHARED_T_PRIMARY, init_condition="center_seed", seed=seed,
        hidden_channels=NCA_PRIMARY.hidden_channels,
    )
    return NCAEncoder(
        factors=factors, update_rule=NCA_PRIMARY.update_rule, decay=NCA_PRIMARY.decay,
        state_bound=NCA_PRIMARY.state_bound, readout_mode="fixed_t",
    )


def _build_rb_windowed(seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=SHARED_T_PRIMARY, init_condition="center_seed", seed=seed)
    return RuleBasedEncoder(factors=factors, readout_mode="mode_of_window", readout_window=RB_W)


def _build_rb_fixed(seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=SHARED_T_PRIMARY, init_condition="center_seed", seed=seed)
    return RuleBasedEncoder(factors=factors, readout_mode="fixed_t")


def measure(name: str, encoder, n: int, seed: int) -> dict:
    rng1 = np.random.default_rng(seed)
    vr = S.validity_rate(encoder, encoder.factors, n, rng1, output_root=OUTPUT_ROOT)
    rng2 = np.random.default_rng(seed)
    dv = S.diversity(encoder, encoder.factors, n, rng2, output_root=OUTPUT_ROOT)
    return {
        "name": name,
        "valid_rate": vr.valid_rate,
        "degenerate_fraction": vr.degenerate_fraction,
        "unique_valid_body_fraction": dv.unique_valid_body_fraction,
        "material_entropy_bits": dv.material_entropy_bits,
    }


def run(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    results = {}
    results["nca_fixed_t"] = measure("nca_fixed_t", _build_nca_fixed(seed), n, seed)
    results["nca_windowed"] = measure("nca_windowed", _build_nca_windowed(seed), n, seed)
    results["rb_fixed_t"] = measure("rb_fixed_t", _build_rb_fixed(seed), n, seed)
    results["rb_windowed"] = measure("rb_windowed", _build_rb_windowed(seed), n, seed)
    return {"n": n, "seed": seed, "t_primary": SHARED_T_PRIMARY, "results": results}


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(result: dict) -> str:
    n = result["n"]
    r = result["results"]
    lines = []
    lines.append("# windowed readout validity check: closing the not-re-measured gap")
    lines.append("")
    lines.append(
        f"N={n}, seed={result['seed']}, T_primary={result['t_primary']} (the corrected shared T_primary, "
        "results/T_PRIMARY_UNIFIED.md). Reuses evodevo.analysis.structural.validity_rate/diversity directly "
        "against fixed_t- and mode_of_window-configured encoders at IDENTICAL (T_primary, other settings) -- "
        "isolating the readout-mode effect."
    )
    lines.append("")
    lines.append("| arm | readout | valid_rate | unique_valid_body_fraction | material_entropy_bits |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key, label, readout in [
        ("nca_fixed_t", "nca", "fixed_t"),
        ("nca_windowed", "nca", f"mode_of_window (W={NCA_W})"),
        ("rb_fixed_t", "rule_based", "fixed_t"),
        ("rb_windowed", "rule_based", f"mode_of_window (W={RB_W})"),
    ]:
        row = r[key]
        lines.append(
            f"| {label} | {readout} | {_fmt(row['valid_rate'])} | "
            f"{_fmt(row['unique_valid_body_fraction'])} | {_fmt(row['material_entropy_bits'])} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_ROOT, "VALIDITY_CHECK.md"))
    args = parser.parse_args()

    result = run(n=args.n, seed=args.seed)
    report = render_report(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
