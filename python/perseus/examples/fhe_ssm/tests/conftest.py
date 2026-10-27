# SPDX-License-Identifier: MIT
"""This example's ladder setting inside each of its tests, and a private copy of ``os.environ`` per
test (the capture driver exports ``FHE_GRAPH_DIR`` / ``FHE_GRAPH_CAPTURE_TOKEN``; upstream's own
conftest does the same for python/tests)."""

import os

import pytest

from perseus.examples.fhe_ssm import LADDER_RADIX_BITS


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow_shape: a check at the model's own shape (ring 2^16, d = 1,024); deselect with -m 'not slow_shape'",
    )


@pytest.fixture(autouse=True)
def _plain_ladder(monkeypatch):
    """``port_env``'s value inside each test, undone after it: set once for the whole run it would
    also switch off the radix ladder in the repository's own tests collected in the same run."""
    saved = dict(os.environ)
    monkeypatch.setenv("LADDER_RADIX_BITS", LADDER_RADIX_BITS)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
