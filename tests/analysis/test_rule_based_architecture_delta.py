"""Correctness/smoke tests for evodevo/analysis/rule_based_architecture_delta.py
-- small N, not the full N=2000 acceptance run (that is the committed
results/rule_based/ARCHITECTURE_DELTA.md)."""
import numpy as np

from evodevo.analysis import rule_based_architecture_delta as RBA


def test_old_encoder_uses_pre_r2_r3_defaults():
    encoder = RBA._build_encoder(RBA.OLD_CONFLICT_RESOLUTION, RBA.OLD_POSITION_PREDICATE, seed=0)
    assert encoder.conflict_resolution == "row_major_last_wins"
    assert encoder.position_predicate == "enabled"


def test_new_encoder_uses_resolved_primary_defaults():
    from evodevo.encoders.rule_based.resolved_config import PRIMARY

    encoder = RBA._build_encoder(PRIMARY.conflict_resolution, PRIMARY.position_predicate, seed=0)
    assert encoder.conflict_resolution == PRIMARY.conflict_resolution
    assert encoder.position_predicate == PRIMARY.position_predicate
    assert encoder.boundary == PRIMARY.boundary
    assert encoder.neighbourhood == PRIMARY.neighbourhood


def test_old_and_new_genomes_can_diverge_in_decoded_body_distribution():
    # Sanity check the two configs are actually different, not accidentally
    # identical (e.g. a wiring bug that silently ignores the constructor args).
    from evodevo.encoders.rule_based.resolved_config import PRIMARY

    old_encoder = RBA._build_encoder(RBA.OLD_CONFLICT_RESOLUTION, RBA.OLD_POSITION_PREDICATE, seed=0)
    new_encoder = RBA._build_encoder(PRIMARY.conflict_resolution, PRIMARY.position_predicate, seed=0)
    rng_old = np.random.default_rng(0)
    rng_new = np.random.default_rng(0)
    old_bodies = [old_encoder.decode(old_encoder.random_genome(rng_old)) for _ in range(50)]
    new_bodies = [new_encoder.decode(new_encoder.random_genome(rng_new)) for _ in range(50)]
    assert any(not np.array_equal(o, n) for o, n in zip(old_bodies, new_bodies))


def test_measure_returns_all_expected_keys():
    encoder = RBA._build_encoder(RBA.OLD_CONFLICT_RESOLUTION, RBA.OLD_POSITION_PREDICATE, seed=0)
    metrics = RBA._measure(encoder, n=20, seed=0, label="test_small")
    expected_keys = {
        "valid_rate", "degenerate_fraction", "unique_valid_body_fraction", "material_entropy_bits",
        "genomes_per_unique_phenotype", "largest_neutral_class_fraction", "never_settled_fraction",
        "mean_settle_step", "mean_hamming_displacement", "median_hamming_displacement",
        "fraction_zero_displacement",
    }
    assert set(metrics.keys()) == expected_keys


def test_run_architecture_delta_smoke():
    result = RBA.run_architecture_delta(n=20, seed=0)
    assert result["n"] == 20
    assert result["t_primary"] == RBA.T_PRIMARY
    for key, (old_v, new_v) in result["rows"].items():
        assert old_v is None or isinstance(old_v, (int, float))
        assert new_v is None or isinstance(new_v, (int, float))


def test_render_report_smoke():
    result = RBA.run_architecture_delta(n=20, seed=0)
    report = RBA.render_report(result)
    assert "## The table" in report
    assert "## What changed and why" in report
    assert "row_major_last_wins" in report
    assert "first_rule_index" in report
    for key in result["rows"]:
        assert RBA.ROW_LABELS.get(key, key) in report
