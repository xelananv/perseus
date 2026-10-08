# SPDX-License-Identifier: MIT
"""Perseus's filling layout on the numpy fake at the real shape (2^16 ring: N = 32,768 slots, d = 1,024):
32 conversations per ciphertext (real) and 64 as real/imaginary pairs, through one 1024 x 1024 tile,
a tiled 1024 -> 4096 -> 1024 pair, and the norm's per-conversation statistic."""

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import filling

from perseus.impl import fake
from perseus.impl.ops import FheOps
from perseus.impl.poly import LanePackedOps

N, D, E = 32768, 1024, 4096


def session(complex_payload=False):
    steps = filling.rotation_steps(N, D)
    fhe, inf = fake.make_fake(
        N=N,
        hid=D,
        dim=D,
        H=16,
        H_real=16,
        E=E,
        E_real=E,
        rot_steps=steps | {N - s for s in steps},
        complex_payload=complex_payload,
    )
    return fhe, inf, FheOps(inf, fake.core, fhe.unit)


def rows(n, seed):
    return 0.3 * np.random.default_rng(seed).standard_normal((n, D))


def weight(seed, shape=(D, D)):
    return np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[0])


def test_tile_params_mirror_compute_dg_params():
    assert filling.tile_params(N, D) == (32, 32, 32)  # t, s, G
    assert filling.tile_params(N, E) == (8, 64, 64)  # one 4096-wide piece: 8 lanes only


def test_one_tile_32_conversations_exact_and_counted():
    fhe, _inf, ops = session()
    X, W = rows(32, 1), weight(2)
    y = filling.linear(ops, fake.FakeCt(filling.pack(X, N), 34), filling.EncodedTile.encode(W, N))
    err = np.abs(filling.unpack(y.vec, D, 32) - X @ W).max()
    assert err < 1e-10
    assert sum(c.startswith("rotate") for c in fhe.calls) == 31 + 31  # babies + giants
    assert fhe.calls.count("mult_pt") == D  # one product per diagonal


def test_one_tile_64_conversations_as_pairs_exact():
    fhe, _inf, ops = session(complex_payload=True)
    X, W = rows(64, 3), weight(4)
    y = filling.linear(
        ops, fake.FakeCt(filling.pack_pairs(X, N), 34), filling.EncodedTile.encode(W, N)
    )
    err = np.abs(filling.unpack_pairs(y.vec, D, 64) - X @ W).max()
    assert err < 1e-10
    assert fhe.calls.count("mult_pt") == D  # 64 lanes, same 1,024 products


@pytest.mark.parametrize("pairs", [False, True])
def test_tiled_channel_mix_shapes_exact(pairs):
    _fhe, _inf, ops = session(complex_payload=pairs)
    n = 64 if pairs else 32
    X, Wk, Wv = rows(n, 5), weight(6, (D, E)), weight(7, (E, D))
    enc = filling.pack_pairs if pairs else filling.pack
    dec = filling.unpack_pairs if pairs else filling.unpack
    hidden = filling.linear_tiled(ops, [fake.FakeCt(enc(X, N), 34)], filling.encode_tiled(Wk, N, D))
    assert len(hidden) == 4  # 4,096 channels = 4 ciphertexts
    H = np.concatenate([dec(h.vec, D, n) for h in hidden], axis=1)
    assert np.abs(H - X @ Wk).max() < 1e-10
    chunks = [fake.FakeCt(enc(H[:, c * D : (c + 1) * D], N), 34) for c in range(4)]
    out = filling.linear_tiled(ops, chunks, filling.encode_tiled(Wv, N, D))
    assert np.abs(dec(out[0].vec, D, n) - H @ Wv).max() < 1e-10


def test_norm_statistic_32_conversations():
    fhe, _inf, ops = session()
    X = rows(32, 8)
    v = filling.per_conversation_sum(ops, ops.square(fake.FakeCt(filling.pack(X, N), 34)), N, D)
    want = (X**2).sum(axis=1)
    got = v.vec.reshape(D, 32)  # slot i*t + k -> [i, k]
    assert np.abs(got - want[None, :]).max() < 1e-10  # every channel slot of lane k
    assert sum(c.startswith("rotate") for c in fhe.calls) == 10  # log2(1024)


def test_norm_statistic_64_pairs_through_lane_packed_ops():
    _fhe, _inf, ops = session(complex_payload=True)
    X = rows(64, 9)
    lops = LanePackedOps(ops)
    sq = lops.square(fake.FakeCt(filling.pack_pairs(X, N), 34))  # a^2 + i b^2
    v = filling.per_conversation_sum(ops, sq, N, D)
    want = (X**2).sum(axis=1)
    got = v.vec.reshape(D, 32)
    assert np.abs(got.real - want[None, :32]).max() < 1e-9
    assert np.abs(got.imag - want[None, 32:]).max() < 1e-9
