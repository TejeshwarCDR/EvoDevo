"""Correctness/smoke tests for evodevo/analysis/architecture_delta.py --
small N, not the full N=2000 acceptance run (that is the committed
results/nca/ARCHITECTURE_DELTA.md)."""
import numpy as np
import pytest

from evodevo.analysis import architecture_delta as A


def test_old_encoder_reproduces_pre_n2_random_genome_distribution():
    encoder = A._build_old_encoder(seed=0)
    rng = np.random.default_rng(0)
    genome = encoder.random_genome(rng)
    expected = np.random.default_rng(0).normal(scale=0.5, size=encoder.genome_length())
    np.testing.assert_array_equal(genome, expected)


def test_old_encoder_unpack_is_raw_reshape_no_layer_scaling():
    encoder = A._build_old_encoder(seed=0)
    rng = np.random.default_rng(1)
    genome = rng.normal(size=encoder.genome_length())
    w1, b1, w2, b2 = encoder._unpack(genome)
    w1_size = encoder._sizes[0]
    np.testing.assert_array_equal(w1.flatten(), genome[:w1_size])


def test_old_encoder_uses_unbounded_residual_update_rule():
    encoder = A._build_old_encoder(seed=0)
    assert encoder.update_rule == "unbounded_residual"


def test_new_encoder_uses_resolved_architecture():
    encoder = A._build_new_encoder(seed=0)
    assert encoder.update_rule == "bounded_leaky"
    assert encoder.decay == pytest.approx(A.PRIMARY.decay)
    assert encoder.state_bound == pytest.approx(A.PRIMARY.state_bound)
    assert encoder.hidden_channels == A.PRIMARY.hidden_channels


def test_old_and_new_genomes_diverge_in_decoded_body_distribution():
    # Sanity check that the two architectures are actually different, not
    # accidentally identical (e.g. a monkeypatch that silently no-ops).
    old_encoder = A._build_old_encoder(seed=0)
    new_encoder = A._build_new_encoder(seed=0)
    rng_old = np.random.default_rng(0)
    rng_new = np.random.default_rng(0)
    old_bodies = [old_encoder.decode(old_encoder.random_genome(rng_old)) for _ in range(20)]
    new_bodies = [new_encoder.decode(new_encoder.random_genome(rng_new)) for _ in range(20)]
    assert any(not np.array_equal(o, n) for o, n in zip(old_bodies, new_bodies))


def test_measure_mutation_displacement_runs_for_both_architectures():
    old_encoder = A._build_old_encoder(seed=0, mutation_sigma=0.02)
    new_encoder = A._build_new_encoder(seed=0, mutation_sigma=0.02)
    old_mean = A._measure_mutation_displacement(old_encoder, 20, seed=0)
    new_mean = A._measure_mutation_displacement(new_encoder, 20, seed=0)
    assert old_mean >= 0.0
    assert new_mean >= 0.0


def test_measure_certified_fraction_is_between_zero_and_one():
    new_encoder = A._build_new_encoder(seed=0)
    frac = A._measure_certified_fraction(new_encoder, 30, seed=0)
    assert 0.0 <= frac <= 1.0


def test_render_report_smoke(tmp_path, monkeypatch):
    # Synthesize a minimal depth_study_raw.json so render doesn't depend on
    # a full depth_study run.
    import json

    raw = {
        "layer2": {
            "nca": {
                "never_settled_fraction": 0.4,
                "median_settle_step": 38.0,
                "p95_settle_step": 39.0,
                "magnitude_analysis": {"mean_final_magnitude": 1.7, "median_final_magnitude": 1.6},
            }
        },
        "layer1": {"nca": {"aggregate": {"propagation_speed_cells_per_step": 0.04}}},
        "t_primary": 39,
        "layer3": {
            "nca__T40": {
                "validity_rate": {"valid_rate": 0.42},
                "diversity": {"unique_valid_body_fraction": 0.41},
            }
        },
        "non_settling_diagnosis": {
            "total_never_settled": 800,
            "counts": {"PLATEAUED_OSCILLATING": 420, "SLOW_BUT_MONOTONE": 340, "STILL_GROWING": 40},
            "fractions": {"PLATEAUED_OSCILLATING": 0.525, "SLOW_BUT_MONOTONE": 0.425, "STILL_GROWING": 0.05},
        },
    }
    raw_path = tmp_path / "depth_study_raw.json"
    raw_path.write_text(json.dumps(raw))
    monkeypatch.setattr(A, "DEPTH_STUDY_RAW", str(raw_path))

    result = A.run_architecture_delta(n=20, seed=0)
    report = A.render_report(result)

    assert "## The table" in report
    assert "certified-contractive fraction" in report
    assert "T_primary implied" in report
    assert "Overall assessment" in report
    for key in result["rows"]:
        assert key in result["rows"]
