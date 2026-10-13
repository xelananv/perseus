# SPDX-License-Identifier: MIT
"""Put the repository root on sys.path so the tests import ``examples.fhe_ssm_from_primitives``
the way the repository's own tests import ``examples.gpt2_from_primitives``."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
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


def pytest_deselected(items):
    """Stop the run when a test of this directory is deselected for carrying the ``gpu`` marker. These
    are CPU tests; tests/gpu/conftest.py gives the marker to every collected item whose path contains
    "gpu", and the default run (``-m 'not gpu'``) deselects it, so a module named with "gpu" here, or
    a checkout under such a path, would otherwise drop them from every run without a word. (A
    collection hook would see the list only after the deselection; this one sees what it removed.)"""
    swept = [
        it.nodeid
        for it in items
        if it.get_closest_marker("gpu") and HERE in it.path.resolve().parents
    ]
    if swept:
        raise pytest.UsageError(
            f"{len(swept)} test(s) of {HERE} carry the gpu marker, so the default run would never run "
            f"them ({', '.join(swept[:3])}{', ...' if len(swept) > 3 else ''}): tests/gpu/conftest.py "
            "marks every item whose path contains 'gpu'; keep it out of this directory's module names "
            "and out of the checkout's path"
        )
