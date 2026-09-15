"""Correctness tests for evodevo/analysis/depth_study.py's selection logic
-- small N, not the full N=2000 acceptance run (verified manually/by the
committed results/depth/DEPTH_STUDY.md)."""
import numpy as np
import pytest

from evodevo.analysis import depth_study as D


def test_select_t_primary_takes_max_of_q95_and_clips():
    layer2 = {
        "nca": {"p95_settle_step": 15.0, "never_settled_fraction": 0.1},
        "rule_based": {"p95_settle_step": 3.0, "never_settled_fraction": 0.05},
    }
    t_primary, provisional = D._select_t_primary(layer2)
    assert t_primary == 15
    assert provisional is False


def test_select_t_primary_clips_to_lower_bound():
    layer2 = {
        "nca": {"p95_settle_step": 1.0, "never_settled_fraction": 0.0},
        "rule_based": {"p95_settle_step": 2.0, "never_settled_fraction": 0.0},
    }
    t_primary, provisional = D._select_t_primary(layer2)
    assert t_primary == 4  # clipped up to the geometric-minimum floor


def test_select_t_primary_has_no_ceiling_below_t_max():
    # A previous version of this function clipped to [4, 20] -- an
    # unjustified ceiling with no compute rationale. The corrected clip is
    # a floor only (up to T_MAX_HARD_CAP), so a q95 of 35 must NOT be
    # capped down to 20.
    layer2 = {
        "nca": {"p95_settle_step": 35.0, "never_settled_fraction": 0.1},
        "rule_based": {"p95_settle_step": 5.0, "never_settled_fraction": 0.05},
    }
    t_primary, provisional = D._select_t_primary(layer2)
    assert t_primary == 35


def test_select_t_primary_clips_to_t_max_hard_cap():
    layer2 = {
        "nca": {"p95_settle_step": 999.0, "never_settled_fraction": 0.1},
        "rule_based": {"p95_settle_step": 5.0, "never_settled_fraction": 0.05},
    }
    t_primary, provisional = D._select_t_primary(layer2)
    assert t_primary == D.T_MAX_HARD_CAP


def test_select_t_primary_provisional_fallback_when_nca_does_not_settle():
    layer2 = {
        "nca": {"p95_settle_step": 39.0, "never_settled_fraction": 0.9},
        "rule_based": {"p95_settle_step": 6.0, "never_settled_fraction": 0.1},
    }
    t_primary, provisional = D._select_t_primary(layer2)
    assert provisional is True
    # Falls back to rule_based's q95 and the geometric bound, not nca's own q95.
    assert t_primary == 6


def test_select_t_primary_provisional_respects_geometric_bound():
    layer2 = {
        "nca": {"p95_settle_step": 39.0, "never_settled_fraction": 0.9},
        "rule_based": {"p95_settle_step": 1.0, "never_settled_fraction": 0.0},
    }
    t_primary, provisional = D._select_t_primary(layer2)
    assert provisional is True
    assert t_primary == D.GEOMETRIC_BOUND_FROM_CENTER  # rule_based's q95 (1) is below the geometric bound (4)


def test_depth_sweep_includes_t_equals_1_and_geometric_minimum():
    # This is what run_layer3 uses to build the sweep -- verify the actual
    # set construction rule (T=1 is the non-droppable non-developmental
    # control and must always be present).
    t_primary = 20
    depths = sorted(set([1, 4, t_primary, min(2 * t_primary, D.T_MAX_HARD_CAP)]))
    assert depths[0] == 1
    assert 4 in depths
    assert t_primary in depths
    assert max(depths) <= D.T_MAX_HARD_CAP


def test_light_cone_raw_pairs_runs_for_both_arms():
    from evodevo.core.factors import ExperimentFactors
    from evodevo.encoders import get_encoder

    for arm in D.ARMS:
        factors = ExperimentFactors(dev_steps=10, init_condition="center_seed", seed=0)
        encoder = get_encoder(arm, factors=factors)
        rng = np.random.default_rng(0)
        pairs, never_reached, total = D._light_cone_raw_pairs(encoder, 10, rng, t_max=8)
        assert total == 24 * 10  # 24 other cells x 10 genomes, on a 5x5 grid
        assert 0 <= never_reached <= total
        for dist, step in pairs:
            assert dist >= 1
            assert 0 <= step <= 8
