# SPDX-License-Identifier: MIT
"""The circuit on numpy slot vectors (the mirror) against the poly reference, 2 blocks of the real checkpoint: with
exact matrices it is the same polynomial model up to float64 rounding (the layout, the folds, the transposes, the
v_first ride on the imaginary axis); with 24-bit PCMM rounding it moves by the rounding only."""
import numpy as np
import pytest

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.model import Mirror, load_coeffs
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer

N_LAYERS = 2


def rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


@pytest.fixture(scope="module")
def params2():
    from perseus.examples.rwkv7.tests.conftest import CKPT
    return ref.load_params(CKPT, N_LAYERS)


@need
@pytest.mark.parametrize("wbits,tol,fuse", [(None, 1e-6, True),(24, 2e-3, True), (24, 2e-3, False)])
def test_mirror_matches_poly_reference(params2, wbits, tol, fuse):
    ids = tokenizer().encode("\n\nThe Eiffel tower is")
    co = load_coeffs(COEFFS)
    mir = Mirror(params2, co, 1 << 15, wbits=wbits, fuse_l1=fuse)
    rf = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=False))
    errs = []
    for t in ids:
        _, h_ref = rf.step(t)
        h, z = mir.step(params2.emb[t])
        errs.append(rel_rms(h, h_ref))
        assert np.abs(z.real.reshape(1024, -1)[768:]).max() < 1e-9
    print("wbits", wbits, "relErrRms per token", [f"{e:.2e}" for e in errs])
    assert max(errs) < tol
