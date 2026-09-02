"""Generic Encoder-conformance suite.

Runs automatically against EVERY encoder currently in the registry
(evodevo.encoders.list_encoders()), so a new encoding is validated the
moment it registers itself in evodevo/encoders/__init__.py -- no per-encoder
test file needs to duplicate these checks. Encoder-SPECIFIC behaviour (e.g.
NCA's synchronicity, direct's reshape identity) lives in test_direct.py /
test_nca.py instead.

See tests/encoders/_contract_checks.py for the individual check functions,
and tests/encoders/test_contract_self_check.py for a demonstration that a
deliberately broken encoder actually fails these checks.
"""
from __future__ import annotations

import pytest

from evodevo.encoders import get_encoder, list_encoders
from tests.encoders._contract_checks import ALL_CHECKS

ENCODER_NAMES = list_encoders()


@pytest.mark.parametrize("encoder_name", ENCODER_NAMES)
@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda fn: fn.__name__)
def test_encoder_conforms_to_contract(check, encoder_name):
    encoder = get_encoder(encoder_name)
    check(encoder)


def test_at_least_direct_and_nca_are_registered():
    # Guards against a collection-time accident silently reducing this suite
    # to zero parametrizations (which would "pass" while testing nothing).
    assert set(ENCODER_NAMES) >= {"direct", "nca"}
