"""Correctness/smoke tests for evodevo/analysis/readout_study.py -- small N,
not the full N=2000 acceptance run (that is the committed
results/nca/READOUT_STUDY.md, produced by `python -m
evodevo.analysis.readout_study`)."""
import numpy as np
import pytest

from evodevo.analysis import readout_study as R


def test_run_ablation_covers_all_h_values_and_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = R.run_ablation(n=15, seed=0)

    assert set(cells.keys()) == set(R.H_VALUES)
    expected_metrics = {
        "validity_rate", "settling_profile", "light_cone", "phenotypic_displacement",
        "redundancy", "diversity", "perturbation_robustness", "readout_margin",
        "uniform_body_fraction", "hidden_utilisation",
    }
    for H in R.H_VALUES:
        assert set(cells[H].keys()) == expected_metrics
        assert cells[H]["hidden_utilisation"].hidden_channels == H


def test_hidden_utilisation_h0_cell_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = R.run_ablation(n=10, seed=1)
    hu0 = cells[0]["hidden_utilisation"]
    assert hu0.hidden_channels == 0
    assert hu0.per_channel_spatial_variance == []
    assert hu0.per_channel_correlation_with_readout == []


def test_render_report_contains_verdict_and_all_h_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = R.run_ablation(n=15, seed=0)
    report = R.render_report(cells, n=15, seed=0)

    assert "### Verdict" in report
    assert "readout_margin" not in report or "margin" in report.lower()
    for H in R.H_VALUES:
        assert f"| {H} |" in report
    assert "lambda\\*=" in report
    assert str(R.PRIMARY.decay) in report
    assert str(R.PRIMARY.state_bound) in report


def test_render_report_verdict_direction_matches_measured_gap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = R.run_ablation(n=15, seed=0)
    report = R.render_report(cells, n=15, seed=0)

    h0 = cells[0]
    h3 = cells[3]

    def rel_gap(a, b):
        if a == 0 and b == 0:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    checks = [
        rel_gap(h0["validity_rate"].valid_rate, h3["validity_rate"].valid_rate),
        rel_gap(h0["validity_rate"].degenerate_fraction, h3["validity_rate"].degenerate_fraction),
        rel_gap(h0["diversity"].unique_valid_body_fraction, h3["diversity"].unique_valid_body_fraction),
        rel_gap(h0["diversity"].material_entropy_bits, h3["diversity"].material_entropy_bits),
        rel_gap(h0["settling_profile"].never_settled_fraction, h3["settling_profile"].never_settled_fraction),
        rel_gap(h0["readout_margin"].mean_margin, h3["readout_margin"].mean_margin),
        rel_gap(h0["readout_margin"].fraction_cells_near_tie, h3["readout_margin"].fraction_cells_near_tie),
    ]
    max_gap = max(checks)

    if max_gap < 0.10:
        assert "do not measurably affect" in report
    else:
        assert "is NOT negligible" in report


def test_old_init_citation_value_is_used_in_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = R.run_ablation(n=10, seed=0)
    report = R.render_report(cells, n=10, seed=0)
    assert f"{R.OLD_INIT_DEGENERATE_FRACTION:.4f}" in report
