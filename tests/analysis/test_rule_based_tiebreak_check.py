"""Correctness/smoke tests for evodevo/analysis/rule_based_tiebreak_check.py
-- small N, not the full N=200 acceptance run (that is the committed
results/rule_based/TIEBREAK_VALIDITY_CHECK.md)."""
import warnings

import pytest

from evodevo.analysis import rule_based_tiebreak_check as T


def test_find_period2_genomes_returns_requested_count():
    genomes, encoder = T.find_period2_genomes(n_target=2, seed=0)
    assert len(genomes) == 2
    for genome, A, B in genomes:
        assert A.shape == (5, 5)
        assert B.shape == (5, 5)


def test_verify_genome_aware_readout_is_t_shift_stable_warns_on_the_known_clip():
    # T_MAX_AUDIT=100 but RESOLVED_T+max(shift)=105 -- the +1/+5 shift
    # queries land past the computed trajectory and get silently clipped
    # unless warned. Verified harmless for exact period-2 genomes (see the
    # function's own docstring), but must not pass silently.
    with pytest.warns(UserWarning, match="exceeds the computed trajectory"):
        result = T.verify_genome_aware_readout_is_t_shift_stable(n_check=2, seed=0)
    assert result["n_check"] == 2
    # Harmless for this population: every checked genome is still reported stable.
    assert result["n_stable"] == result["n_check"]


def test_verify_genome_aware_readout_is_t_shift_stable_no_warning_when_horizon_covers_every_shift(monkeypatch):
    # Widen T_MAX_AUDIT to cover RESOLVED_T + max(shift) so the clip never
    # triggers -- confirms the warning is conditional on the real gap, not
    # unconditional noise.
    monkeypatch.setattr(T, "T_MAX_AUDIT", T.RESOLVED_T + 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        T.verify_genome_aware_readout_is_t_shift_stable(n_check=2, seed=0)
