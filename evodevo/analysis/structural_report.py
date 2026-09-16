"""Generates results/structural/STRUCTURAL.md: one section per structural
metric (evodevo/analysis/structural.py), one table per encoding arm, plus
matplotlib figures under results/structural/figs/.

Like structural.py itself, this module is fitness-blind: it only ever
constructs encoders and calls the seven structural.* metric functions --
NO PPO, NO task, NO reward, NO EA. See
tests/analysis/test_structural_no_ppo_imports.py.

Usage:
    python -m evodevo.analysis.structural_report [--n 2000] [--seed 0]
        [--dev-steps 10] [--init-condition full_nonsymmetric]
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder, list_encoders
from evodevo.encoders.nca.resolved_config import PRIMARY

DEFAULT_N = 2000
DEFAULT_OUTPUT_DIR = "results/structural"
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "STRUCTURAL.md")
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "figs")

# NCA's resolved architecture (evodevo/encoders/nca/resolved_config.py, the single
# source of truth -- see its own docstring and docs/decisions.md's "NCA joint
# stability-vs-expressivity trade-off" row). direct/rule_based get no override
# anywhere in this module -- their numbers must come out byte-identical to the
# previously committed results/structural/STRUCTURAL.md (see
# results/nca/ARCHITECTURE_DELTA.md's regression check).


def _build_encoder(name: str, factors: ExperimentFactors):
    if name == "nca":
        nca_factors = dataclasses.replace(factors, hidden_channels=PRIMARY.hidden_channels)
        return get_encoder(
            name, factors=nca_factors, update_rule=PRIMARY.update_rule, decay=PRIMARY.decay,
            state_bound=PRIMARY.state_bound,
            # Pinned explicitly at fixed_t/10 (NOT PRIMARY.readout_mode) for
            # the same reason as depth_study.py's own pin: this module's
            # default dev_steps=10 is far shallower than PRIMARY's resolved
            # W=50, where mode_of_window collapses toward the initial
            # condition rather than reflecting steady-state behaviour.
            readout_mode="fixed_t", readout_window=10,
        )
    return get_encoder(name, factors=factors)


def _fmt(value, digits=3):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _dict_table(rows: dict, key_header: str, value_header: str) -> str:
    lines = [f"| {key_header} | {value_header} |", "| --- | --- |"]
    for key, value in rows.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def _plot_bar(arm_results: dict, key: str, title: str, ylabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    arms = list(arm_results.keys())
    values = [getattr(arm_results[arm], key) for arm in arms]
    ax.bar(arms, values, color=["#4C72B0", "#DD8452", "#55A868"][: len(arms)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_light_cone(arm_results: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for arm, result in arm_results.items():
        distances = sorted(d for d, v in result.mean_first_diff_step_by_distance.items() if v is not None)
        values = [result.mean_first_diff_step_by_distance[d] for d in distances]
        if distances:
            ax.plot(distances, values, marker="o", label=arm)
    ax.set_xlabel("Manhattan distance from perturbed cell")
    ax.set_ylabel("mean first-difference step")
    ax.set_title("Light cone: propagation speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_histogram(arm_records: dict, field: str, title: str, xlabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for arm, values in arm_records.items():
        finite = [v for v in values if v is not None]
        if finite:
            ax.hist(finite, bins=15, alpha=0.5, label=arm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_structural_report(
    n_samples: int = DEFAULT_N,
    seed: int = 0,
    dev_steps: int = 10,
    init_condition: str = "full_nonsymmetric",
    encoder_names=None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    encoder_names = encoder_names or list_encoders()
    figures_dir = os.path.join(output_dir, "figs")
    os.makedirs(figures_dir, exist_ok=True)

    results = {metric: {} for metric in [
        "validity_rate", "settling_profile", "light_cone", "phenotypic_displacement",
        "redundancy", "diversity", "perturbation_robustness",
    ]}
    raw_hamming = {"phenotypic_displacement": {}, "perturbation_robustness": {}}
    raw_settle_steps = {}

    for name in encoder_names:
        factors = ExperimentFactors(dev_steps=dev_steps, init_condition=init_condition, seed=seed)
        encoder = _build_encoder(name, factors)
        rng = np.random.default_rng(seed)

        results["validity_rate"][name] = S.validity_rate(encoder, factors, n_samples, rng, output_root=output_dir)
        settling = S.settling_profile(encoder, factors, n_samples, rng, output_root=output_dir)
        results["settling_profile"][name] = settling
        results["light_cone"][name] = S.light_cone(encoder, factors, n_samples, rng, output_root=output_dir)
        disp = S.phenotypic_displacement(encoder, factors, n_samples, rng, output_root=output_dir)
        results["phenotypic_displacement"][name] = disp
        results["redundancy"][name] = S.redundancy(encoder, factors, n_samples, rng, output_root=output_dir)
        results["diversity"][name] = S.diversity(encoder, factors, n_samples, rng, output_root=output_dir)
        robustness = S.perturbation_robustness(encoder, factors, n_samples, rng, output_root=output_dir)
        results["perturbation_robustness"][name] = robustness

    _plot_bar(results["validity_rate"], "valid_rate", "Validity rate", "fraction valid",
              os.path.join(figures_dir, "validity_rate.png"))
    _plot_bar(results["settling_profile"], "never_settled_fraction", "Never-settled fraction",
              "fraction", os.path.join(figures_dir, "never_settled_fraction.png"))
    _plot_light_cone(results["light_cone"], os.path.join(figures_dir, "light_cone.png"))
    _plot_bar(results["redundancy"], "genomes_per_unique_phenotype", "Redundancy",
              "genomes per unique phenotype", os.path.join(figures_dir, "redundancy.png"))
    _plot_bar(results["diversity"], "material_entropy_bits", "Material entropy",
              "bits", os.path.join(figures_dir, "diversity.png"))
    _plot_bar(results["perturbation_robustness"], "mean_hamming_distance", "Perturbation robustness",
              "mean Hamming distance", os.path.join(figures_dir, "perturbation_robustness.png"))

    sections = []
    sections.append("# Structural report\n")
    sections.append(
        f"Fitness-blind structural measurements (see evodevo/analysis/structural.py) -- "
        f"N={n_samples}, seed={seed}, dev_steps={dev_steps}, init_condition={init_condition}. "
        "No PPO, no task, no reward were run to produce any number on this page.\n"
    )

    sections.append("## 1. Validity rate\n")
    sections.append("| arm | valid_rate | degenerate_fraction | failure_reason_counts |")
    sections.append("| --- | --- | --- | --- |")
    for name, r in results["validity_rate"].items():
        sections.append(f"| {name} | {_fmt(r.valid_rate)} | {_fmt(r.degenerate_fraction)} | {r.failure_reason_counts} |")
    sections.append("")

    sections.append("## 2. Settling profile\n")
    sections.append("| arm | settled_fraction | never_settled_count | never_settled_fraction | mean_settle_step |")
    sections.append("| --- | --- | --- | --- | --- |")
    for name, r in results["settling_profile"].items():
        sections.append(
            f"| {name} | {_fmt(r.settled_fraction)} | {r.never_settled_count} | "
            f"{_fmt(r.never_settled_fraction)} | {_fmt(r.mean_settle_step)} |"
        )
    sections.append(
        "\nThe never-settled count above is the number of genomes whose decoded body was still "
        f"changing at t_max={next(iter(results['settling_profile'].values())).t_max} -- reported "
        "explicitly, never silently dropped from settle-step statistics.\n"
    )
    sections.append(f"![never-settled fraction](figs/never_settled_fraction.png)\n")

    sections.append("## 3. Light cone (propagation speed)\n")
    sections.append("| arm | perturb_cell | propagation_speed_cells_per_step | never_reached_fraction |")
    sections.append("| --- | --- | --- | --- |")
    for name, r in results["light_cone"].items():
        sections.append(f"| {name} | {r.perturb_cell} | {_fmt(r.propagation_speed_cells_per_step)} | {_fmt(r.never_reached_fraction)} |")
    sections.append(f"\n![light cone](figs/light_cone.png)\n")

    sections.append("## 4. Phenotypic displacement (mutation locality)\n")
    sections.append("| arm | mean_hamming_distance | median_hamming_distance | fraction_zero_displacement |")
    sections.append("| --- | --- | --- | --- |")
    for name, r in results["phenotypic_displacement"].items():
        sections.append(f"| {name} | {_fmt(r.mean_hamming_distance)} | {_fmt(r.median_hamming_distance)} | {_fmt(r.fraction_zero_displacement)} |")
    sections.append("")

    sections.append("## 5. Redundancy\n")
    sections.append("| arm | unique_phenotype_count | genomes_per_unique_phenotype | largest_neutral_class_fraction |")
    sections.append("| --- | --- | --- | --- |")
    for name, r in results["redundancy"].items():
        sections.append(f"| {name} | {r.unique_phenotype_count} | {_fmt(r.genomes_per_unique_phenotype)} | {_fmt(r.largest_neutral_class_fraction)} |")
    sections.append(f"\n![redundancy](figs/redundancy.png)\n")

    sections.append("## 6. Diversity\n")
    sections.append("| arm | unique_valid_body_fraction | material_entropy_bits |")
    sections.append("| --- | --- | --- |")
    for name, r in results["diversity"].items():
        sections.append(f"| {name} | {_fmt(r.unique_valid_body_fraction)} | {_fmt(r.material_entropy_bits)} |")
    sections.append(f"\n![diversity](figs/diversity.png)\n")

    sections.append("## 7. Perturbation robustness\n")
    sections.append(
        "For direct, damage is applied to the genome itself (genome IS phenotype); "
        "for nca/rule_based, damage is injected mid-trajectory at step k=floor(T/2). "
        "See evodevo.analysis.structural.perturbation_robustness's docstring for the full asymmetry.\n"
    )
    sections.append("| arm | damage_fraction | mean_hamming_distance | fraction_unaffected |")
    sections.append("| --- | --- | --- | --- |")
    for name, r in results["perturbation_robustness"].items():
        sections.append(f"| {name} | {r.damage_fraction} | {_fmt(r.mean_hamming_distance)} | {_fmt(r.fraction_unaffected)} |")
    sections.append(f"\n![perturbation robustness](figs/perturbation_robustness.png)\n")

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "STRUCTURAL.md")
    with open(report_path, "w") as f:
        f.write("\n".join(sections))

    return report_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-steps", type=int, default=10)
    parser.add_argument("--init-condition", default="full_nonsymmetric")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    report_path = run_structural_report(
        n_samples=args.n,
        seed=args.seed,
        dev_steps=args.dev_steps,
        init_condition=args.init_condition,
        output_dir=args.output_dir,
    )
    print(f"structural report written to {report_path}")


if __name__ == "__main__":
    main()
