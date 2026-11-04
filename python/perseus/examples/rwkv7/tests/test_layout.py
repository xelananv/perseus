# SPDX-License-Identifier: MIT
"""The layout kernels on numpy slot vectors against the matrix algebra they stand for (n = 2^15 and 2^14 slots,
i.e. t = 32 and 16 lanes, so M = 2 and 4 state ciphertexts)."""
import numpy as np
import pytest

from perseus.examples.rwkv7.layout import D, HEAD, Layout
from perseus.impl.ops import NumpyOps

H = 12


@pytest.mark.parametrize("n", [1 << 15, 1 << 14])
def test_state_products(n):
    L, ops = Layout(n), NumpyOps()
    rng = np.random.default_rng(0)
    S = rng.normal(size=(H, HEAD, HEAD))
    v, a, r = (rng.normal(size=H * HEAD) for _ in range(3))
    Ss = L.state_slots(S)
    assert np.allclose(L.state_from_slots(Ss, H), S)
    # S (.) w[j], (S a) broadcast over j, v k^T, y = S r
    V = L.transpose(ops, L.natural(v))
    want_V = np.broadcast_to(v.reshape(H, HEAD)[:, :, None], (H, HEAD, HEAD))   # [h, i, j] = v[h, i]
    assert np.allclose(L.state_from_slots(V, H), want_V)
    Sa = [L.block_allreduce(ops, ops.mult(s, L.natural(a))) for s in Ss]
    want_Sa = np.einsum("hij,hj->hi", S, a.reshape(H, HEAD))
    assert np.allclose(L.state_from_slots(Sa, H), np.broadcast_to(want_Sa[:, :, None], S.shape))
    y = L.untranspose(ops, [ops.mult(s, L.natural(r)) for s in Ss])
    want_y = np.einsum("hij,hj->hi", S, r.reshape(H, HEAD)).reshape(-1)
    got = np.real(y).reshape(D, L.t)
    assert np.allclose(got[: H * HEAD], want_y[:, None])
    assert np.allclose(got[H * HEAD:], 0.0)


def test_channel_sum_and_steps():
    L, ops = Layout(1 << 15), NumpyOps()
    x = np.random.default_rng(1).normal(size=768)
    s = L.channel_sum(ops, L.natural(x))
    assert np.allclose(s, x.sum())
    assert 0 not in L.rotation_steps()
