"""Suite hygiene for the repository-root campaign completion marker."""
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def preserve_root_completion_marker():
    """Campaign integration tests must not leave a synthetic root marker.

    A genuine marker that pre-dates testing is restored byte-for-byte.
    """
    marker = Path(__file__).resolve().parents[1] / "CAMPAIGN_COMPLETE.txt"
    original = marker.read_bytes() if marker.exists() else None
    yield
    if original is None:
        marker.unlink(missing_ok=True)
    else:
        marker.write_bytes(original)
