# SPDX-License-Identifier: MIT
"""Put the repository root on sys.path so the tests import ``examples.fhe_ssm_from_primitives``
the way the repository's own tests import ``examples.gpt2_from_primitives``."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from examples.fhe_ssm_from_primitives import LADDER_RADIX_BITS  # noqa: E402


@pytest.fixture(autouse=True)
def _plain_ladder(monkeypatch):
    """This example's ladder setting (``port_env``) inside each of its tests, undone after the test.
    Set once for the whole run, it would also switch off the radix ladder in the repository's own
    tests, which pytest collects in the same run; the undo also reverts what ``port_env`` writes when
    a test builds a fake or a session."""
    monkeypatch.setenv("LADDER_RADIX_BITS", LADDER_RADIX_BITS)
