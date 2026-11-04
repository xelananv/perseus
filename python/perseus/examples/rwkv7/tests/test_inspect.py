# SPDX-License-Identifier: MIT
"""Prints the checkpoint's shapes and the coefficient sites (run with -s); skips without the files."""
import os

import pytest

from perseus.examples.rwkv7 import inspect_coeffs
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(COEFFS)), reason="no checkpoint / coefficients")
def test_inspect():
    assert inspect_coeffs.main(["--coeffs", COEFFS, "--pth", CKPT]) == 0
