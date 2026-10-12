# SPDX-License-Identifier: MIT
"""The device route of ``filling.apply_tiles`` on the numpy fake, through a stub of the four device
calls (register_tile, encode_group, mult_ptx, evict_ptx): it must record exactly the CPU route's op
stream and give the same numbers, evict every plaintext it made, refuse an evicted one, and fall back
to the CPU route when the prepared inputs do not share one (level, noise degree)."""

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import filling

from perseus.impl import fake
from perseus.impl.ops import FheOps

N, D, E = 32768, 1024, 4096


class _Token:
    def __init__(self, diagonal, name):
        self.diagonal, self.name = diagonal, name


class DeviceRouteOps(FheOps):
    """FheOps on the fake plus the device encoder's surface, served from the tiles' own diagonals."""

    gpu_encode = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiles = []
        self.live = set()
        self.encoded = 0
        self.evicted = 0

    def register_tile(self, W):
        self.tiles.append(filling.EncodedTile.encode(np.asarray(W), N))
        return len(self.tiles) - 1

    def encode_group(self, tile_ids, baby, g, s, ref, names):
        assert len(tile_ids) == len(baby) == len(names)
        assert all(0 <= b < s for b in baby)
        assert ref is not None
        out = []
        for tid, b, name in zip(tile_ids, baby, names):
            tok = _Token(self.tiles[tid].pt(g, b), name)
            self.live.add(id(tok))
            out.append(tok)
        self.encoded += len(out)
        return out

    def mult_ptx(self, a, tok):
        if id(tok) not in self.live:
            raise ValueError("mult_ptx: the device plaintext was evicted")
        return self.mult_pt(a, tok.diagonal, key=tok.name)

    def evict_ptx(self, tok):
        self.live.remove(id(tok))
        self.evicted += 1


def session(cls=FheOps):
    steps = filling.rotation_steps(N, D)
    fhe, inf = fake.make_fake(
        N=N, hid=D, dim=D, H=16, H_real=16, E=E, E_real=E, rot_steps=steps | {N - s for s in steps}
    )
    return fhe, cls(inf, fake.core, fhe.unit)


def rows(n, seed, width=D):
    return 0.3 * np.random.default_rng(seed).standard_normal((n, width))


def weight(seed, shape):
    return np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[0])


def down_projection(ops, X, W):
    """x (E wide, as four chunks) @ W (E x D): four tiles summed into one output, the multi-tile case."""
    xs = [fake.FakeCt(filling.pack(X[:, i * D : (i + 1) * D], N), 34) for i in range(E // D)]
    dev_ops = ops if getattr(ops, "gpu_encode", False) else None
    tiles = filling.encode_tiled(W, N, D, "wdown", dev_ops=dev_ops)
    (y,) = filling.linear_tiled(ops, xs, tiles)
    return y


def test_device_route_records_the_cpu_route_op_stream_and_numbers():
    X, W = rows(32, 1, E), weight(2, (E, D))
    fhe_cpu, ops_cpu = session()
    y_cpu = down_projection(ops_cpu, X, W)
    fhe_dev, ops_dev = session(DeviceRouteOps)
    y_dev = down_projection(ops_dev, X, W)
    assert fhe_dev.calls == fhe_cpu.calls, "the two routes must record one op stream"
    assert np.array_equal(y_dev.vec, y_cpu.vec)
    assert np.abs(filling.unpack(y_dev.vec, D, 32) - X @ W).max() < 1e-9
    assert ops_dev.encoded == 4 * D  # every (tile, baby) of every giant group, once
    assert ops_dev.evicted == ops_dev.encoded
    assert not ops_dev.live


def test_an_evicted_device_plaintext_is_refused():
    _fhe, ops = session(DeviceRouteOps)
    tile = filling.EncodedTile.encode(weight(3, (D, D)), N, "w", dev_ops=ops)
    x = fake.FakeCt(filling.pack(rows(32, 4), N), 34)
    (tok,) = ops.encode_group([tile.dev], [0], 0, 32, x, ["w.0.0"])
    ops.evict_ptx(tok)
    with pytest.raises(ValueError, match="evicted"):
        ops.mult_ptx(x, tok)


def test_mixed_noise_degrees_take_the_cpu_route():
    X, W = rows(32, 5, 2 * D), weight(6, (2 * D, D))
    fhe, ops = session(DeviceRouteOps)
    tiles = filling.encode_tiled(W, N, D, "w", dev_ops=ops)
    x0 = fake.FakeCt(filling.pack(X[:, :D], N), 34)
    x1 = fake.FakeCt(filling.pack(X[:, D:], N), 34, noise_deg=2)  # same level, a pending rescale
    _t, s, _G = filling.tile_params(N, D)
    rxs = [filling.babies(ops, x, N // D, s) for x in (x0, x1)]
    with pytest.warns(UserWarning, match="CPU route"):
        y = filling.apply_tiles(ops, rxs, [tiles[0][0], tiles[1][0]])
    assert ops.encoded == 0
    assert y is not None
    assert fhe.calls.count("mult_pt") == 2 * D
