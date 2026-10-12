# SPDX-License-Identifier: MIT
"""The device route of ``filling.apply_tiles`` on the numpy fake, through a stub of the device
calls (register_tile, tile_values, gpu_encode_fits, encode_group, mult_ptx, evict_ptx), on a fake
that records every addition with its operands' provenance. The device route must record exactly
the CPU route's op stream (products, rotations and the order of the sums) and give the same
numbers, evict every plaintext it made, encode once per input level, take the CPU route from the
tile fetched back off the device when the encoder cannot serve an input, and keep no host rows on
a device tile. The refusal of an evicted plaintext is the stub's here; the extension's own
(``mult_ptx``) is asserted by ``gpu_encode_check.py`` on a GPU."""

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import filling

from perseus.impl import fake
from perseus.impl.ops import FheOps

N, D, E = 32768, 1024, 4096


def _prov(ct):
    return getattr(ct, "prov", "input")


class RecordingFake(fake.FakeFhe):
    """The fake with every ciphertext stamped by provenance (p<n> a product, a<n> a sum of two
    ciphertexts, c<n> a scalar add or copy, r<n> a rotation; n in creation order) and every
    ciphertext addition recorded with its operands' stamps, so two op streams compare in the
    order of their sums, not only in their products."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stamped = 0

    def _stamp(self, ct, kind):
        self.n_stamped += 1
        ct.prov = f"{kind}{self.n_stamped}"
        return ct

    def _product(self, vec, level, packing):
        return self._stamp(super()._product(vec, level, packing), "p")

    def add(self, a, b):
        out = super().add(a, b)
        if isinstance(b, fake.FakeCt):
            self.calls.append(f"add {_prov(a)} {_prov(b)}")
            return self._stamp(out, "a")
        return self._stamp(out, "c")

    def rotate(self, a, steps):
        return self._stamp(super().rotate(a, steps), "r")


class _Token:
    def __init__(self, diagonal, name):
        self.diagonal, self.name = diagonal, name


class DeviceRouteOps(FheOps):
    """FheOps on the fake plus the device encoder's surface, served from the tiles kept here."""

    gpu_encode = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiles = []  # (W, the same tile encoded for the CPU route)
        self.live = set()
        self.unservable_levels: set[int] = set()  # inputs gpu_encode_fits refuses
        self.encoded = self.groups = self.evicted = self.fetched = 0

    def register_tile(self, W):
        W = np.array(W, dtype=np.float64)
        self.tiles.append((W, filling.EncodedTile.encode(W, N)))
        return len(self.tiles) - 1

    def tile_values(self, tile_id):
        self.fetched += 1
        return self.tiles[tile_id][0]

    def gpu_encode_fits(self, ref):
        return self.level(ref) not in self.unservable_levels

    def encode_group(self, tile_ids, baby, g, s, ref, names):
        assert len(tile_ids) == len(baby) == len(names)
        assert all(0 <= b < s for b in baby)
        assert ref is not None
        out = []
        for tid, b, name in zip(tile_ids, baby, names):
            tok = _Token(self.tiles[tid][1].pt(g, b), name)
            self.live.add(id(tok))
            out.append(tok)
        self.encoded += len(out)
        self.groups += 1
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
    fhe = RecordingFake(steps | {N - s for s in steps}, slots=N)
    inf = fake.FakeInf(fhe, N, D, D, 16, 16, E, E)
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


def mixed_inputs(ops, X):
    """Two prepared inputs that do not share one encode: the second carries a pending rescale,
    which ``babies`` realizes into one level more."""
    x0 = fake.FakeCt(filling.pack(X[:, :D], N), 34)
    x1 = fake.FakeCt(filling.pack(X[:, D:], N), 34, noise_deg=2)
    _t, s, _G = filling.tile_params(N, D)
    return [filling.babies(ops, x, N // D, s) for x in (x0, x1)]


def two_tile_sum(ops, X, W, dev_ops=None):
    tiles = filling.encode_tiled(W, N, D, "w", dev_ops=dev_ops)
    return filling.apply_tiles(ops, mixed_inputs(ops, X), [tiles[0][0], tiles[1][0]])


def test_device_route_records_the_cpu_route_op_stream_and_numbers():
    X, W = rows(32, 1, E), weight(2, (E, D))
    fhe_cpu, ops_cpu = session()
    y_cpu = down_projection(ops_cpu, X, W)
    fhe_dev, ops_dev = session(DeviceRouteOps)
    y_dev = down_projection(ops_dev, X, W)
    assert any(c.startswith("add ") for c in fhe_cpu.calls)  # the sums are part of the record
    assert fhe_dev.calls == fhe_cpu.calls, "the two routes must record one op stream"
    assert np.array_equal(y_dev.vec, y_cpu.vec)
    assert np.abs(filling.unpack(y_dev.vec, D, 32) - X @ W).max() < 1e-9
    _t, _s, G = filling.tile_params(N, D)
    assert ops_dev.encoded == 4 * D  # every (tile, baby) of every giant group, once
    assert ops_dev.groups == G  # the four inputs share one level: one batch per giant group
    assert ops_dev.evicted == ops_dev.encoded
    assert not ops_dev.live
    assert ops_dev.fetched == 0


def test_a_device_tile_keeps_no_host_rows_and_fetches_them_back_exactly():
    _fhe, ops = session(DeviceRouteOps)
    W = weight(3, (D, D))
    dev = filling.EncodedTile.encode(W, N, "w", dev_ops=ops)
    cpu = filling.EncodedTile.encode(W, N, "w")
    assert dev.rows is None
    assert dev.dev == 0
    assert (dev.G, dev.s, dev.d) == (cpu.G, cpu.s, cpu.d)
    with pytest.raises(ValueError, match="host rows"):
        dev.pt(0, 0)
    back = dev.with_rows(ops)
    assert np.array_equal(back.rows, cpu.rows)
    assert back.dev == dev.dev
    assert ops.fetched == 1
    assert cpu.with_rows() is cpu


def test_an_evicted_device_plaintext_is_refused():
    _fhe, ops = session(DeviceRouteOps)
    tile = filling.EncodedTile.encode(weight(3, (D, D)), N, "w", dev_ops=ops)
    x = fake.FakeCt(filling.pack(rows(32, 4), N), 34)
    (tok,) = ops.encode_group([tile.dev], [0], 0, 32, x, ["w.0.0"])
    ops.evict_ptx(tok)
    with pytest.raises(ValueError, match="evicted"):
        ops.mult_ptx(x, tok)


def test_inputs_at_different_encodes_take_one_device_encode_each():
    X, W = rows(32, 5, 2 * D), weight(6, (2 * D, D))
    fhe_cpu, ops_cpu = session()
    y_cpu = two_tile_sum(ops_cpu, X, W)
    fhe_dev, ops_dev = session(DeviceRouteOps)
    y_dev = two_tile_sum(ops_dev, X, W, dev_ops=ops_dev)
    _t, _s, G = filling.tile_params(N, D)
    assert ops_dev.groups == 2 * G  # one batched encode per input per giant group
    assert ops_dev.encoded == 2 * D
    assert ops_dev.evicted == ops_dev.encoded
    assert fhe_dev.calls == fhe_cpu.calls
    assert np.array_equal(y_dev.vec, y_cpu.vec)


def test_an_input_the_encoder_cannot_serve_takes_the_cpu_route_from_the_fetched_tiles():
    X, W = rows(32, 7, 2 * D), weight(8, (2 * D, D))
    fhe_cpu, ops_cpu = session()
    y_cpu = two_tile_sum(ops_cpu, X, W)
    fhe_dev, ops_dev = session(DeviceRouteOps)
    ops_dev.unservable_levels = {36}  # the realized second input
    with pytest.warns(UserWarning, match="CPU route"):
        y_dev = two_tile_sum(ops_dev, X, W, dev_ops=ops_dev)
    assert ops_dev.encoded == 0
    assert ops_dev.fetched == 2  # both tiles, once for the group
    assert fhe_dev.calls == fhe_cpu.calls
    assert np.array_equal(y_dev.vec, y_cpu.vec)
