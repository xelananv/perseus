# SPDX-License-Identifier: MIT
"""Many conversations per ciphertext: Perseus's filling (token-interleaved) layout on the ops protocol.

Channel ``i`` of conversation ``k`` sits at slot ``i*t + k`` with ``t = N / d`` (the dev branch's
``diagonal::encode_linear_input``, ``src/packing/diagonal/diagonal_linear_utils.cu``), so up to ``t``
conversations share one ciphertext, and ``2t`` as real/imaginary pairs run through
``perseus.impl.poly.LanePackedOps``. A rotation by ``j*t`` moves every conversation ``j`` channels
along and wraps exactly where each conversation wraps, so the diagonal method needs no lo/hi halves.

``linear`` mirrors the dev branch's C++ ``diagonal::linear`` for a square ``d x d`` tile
(``compute_dg_params``: ``t_in = t_out = N/d``, ``s * G = d``, ``max_n_tok = N/d``): ``s`` baby
rotations by ``b*t`` from one hoisted decomposition, each realized once, ``G`` giant groups
rotated by ``g*s*t``, each diagonal pre-rotated when it is encoded. Wider matrices are cut into
``d x d`` tiles so the conversation cap stays ``N/d``; in one piece a ``1024 -> 4096`` matrix caps
it at ``N/4096 = 8`` (``max_n_tok = min(t_in, t_out)``).

Levels of the pair route: ``LanePackedOps`` turns every ciphertext product into ``lane_mult`` /
``lane_square``, which end in a scalar ``1/4`` (perseus/impl/poly.py), so each costs two levels
where a real product costs one. Plaintext products (the linears) stay one level for both.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np


def tile_params(N: int, d: int):
    """(t, s, G) for a d x d tile in N slots, as compute_dg_params chooses them."""
    if N % d:
        raise ValueError(f"N={N} is not a multiple of d={d}")
    s = max(1, math.isqrt(d))
    while s > 1 and d % s:
        s -= 1
    return N // d, s, d // s


def pack(rows, N: int):
    """(n, d) rows -> one slot vector, conversation k's channel i at slot i*t + k."""
    rows = np.asarray(rows)
    n, d = rows.shape
    t = N // d
    if n > t:
        raise ValueError(f"{n} conversations do not fit t = {t} lanes; use pack_pairs")
    v = np.zeros(N, dtype=rows.dtype)
    idx = np.arange(d) * t
    for k in range(n):
        v[idx + k] = rows[k]
    return v


def pack_pairs(rows, N: int):
    """Up to 2t conversations: the first t on the real axis, the rest on the imaginary axis."""
    rows = np.asarray(rows, dtype=np.float64)
    t = N // rows.shape[1]
    re = pack(rows[:t], N)
    im = pack(rows[t:], N) if rows.shape[0] > t else np.zeros(N)
    return re + 1j * im


def unpack(vec, d: int, n: int):
    t = vec.shape[0] // d
    idx = np.arange(d) * t
    return np.stack([vec[idx + k] for k in range(n)])


def unpack_pairs(vec, d: int, n: int):
    t = vec.shape[0] // d
    out = [unpack(np.real(vec), d, min(n, t))]
    if n > t:
        out.append(unpack(np.imag(vec), d, n - t))
    return np.concatenate(out)


def lane_const(v, N: int):
    """A per-channel constant for every conversation: value i on slots i*t ... i*t + t - 1."""
    v = np.asarray(v, dtype=np.float64)
    return np.repeat(v, N // v.shape[0])


class _Diagonal:
    """One pre-rotated diagonal, expanded to slots only when read (``np.asarray``): the runtime
    reads a named plaintext once, at registration."""

    def __init__(self, vals, t: int, shift: int):
        self.vals, self.t, self.shift = vals, t, shift

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("a _Diagonal is materialized on read; a no-copy view is not available")
        v = np.roll(np.repeat(self.vals, self.t), self.shift)
        return v if dtype is None else v.astype(dtype)


def _diagonal_rows(W: np.ndarray, s: int, G: int) -> np.ndarray:
    """The ``(G, s, d)`` diagonal rows of a d x d tile: row (g, b) holds W[(j + g*s + b) mod d, j]."""
    d = W.shape[0]
    j = np.arange(d)
    k = np.arange(G)[:, None, None] * s + np.arange(s)[None, :, None]
    return W[(j + k) % d, j]


@dataclasses.dataclass(frozen=True, eq=False)
class EncodedTile:
    """A d x d tile as its ``s * G`` diagonals, mirroring diagonal::encode_weight_matrix for
    d_in = d_out: diagonal (g, b) gives output channel j the weight W[(j + g*s + b) mod d, j],
    repeated over the t lanes of channel j and shifted by g*s*t. ``rows`` holds the d values of
    each diagonal, (G, s, d), on a tile the CPU route encodes; a tile registered on the device
    encoder (``dev``) keeps none, the device holds the only copy, and ``with_rows`` fetches it
    back for a group the device route cannot serve. Compared and hashed by identity."""

    N: int
    G: int
    s: int
    d: int
    rows: np.ndarray | None = None
    name: str | None = None
    # the tile's id on the device encoder (apply_tiles' device route), or None
    dev: int | None = None

    @classmethod
    def encode(cls, W, N: int, name: str | None = None, dev_ops=None):
        W = np.asarray(W)
        d = W.shape[0]
        if W.shape != (d, d):
            raise ValueError("cut wider matrices into d x d tiles (encode_tiled)")
        _, s, G = tile_params(N, d)
        if dev_ops is not None:
            # the d x d tile goes resident on the device once and its diagonals are made there at
            # each use; no host copy beside it (24 blocks of 14 tiles would hold 2.6 GiB twice)
            return cls(N, G, s, d, None, name, dev_ops.register_tile(W))
        return cls(N, G, s, d, _diagonal_rows(W, s, G), name, None)

    def with_rows(self, ops=None):
        """This tile with its diagonal rows on the host: itself when it has them, else the tile
        fetched back from the device encoder (``ops.tile_values``) and diagonalized."""
        if self.rows is not None:
            return self
        if ops is None or self.dev is None:
            raise ValueError(
                f"tile {self.name}: no host rows and no device encoder to fetch them from"
            )
        W = np.asarray(ops.tile_values(self.dev), dtype=np.float64)
        if W.shape != (self.d, self.d):
            raise ValueError(
                f"tile {self.name}: the device returned {W.shape}, not ({self.d}, {self.d})"
            )
        return dataclasses.replace(self, rows=_diagonal_rows(W, self.s, self.G))

    def pt(self, g: int, b: int):
        if self.rows is None:
            raise ValueError(f"tile {self.name}: the CPU route needs host rows (with_rows)")
        t = self.N // self.d
        return _Diagonal(self.rows[g, b], t, g * self.s * t)

    def key(self, g: int, b: int):
        return None if self.name is None else f"{self.name}.{g}.{b}"


def encode_tiled(W, N: int, d: int, name: str | None = None, dev_ops=None):
    """A (d_in, d_out) matrix as [input chunk][output chunk] tiles (``dev_ops``: see EncodedTile.encode)."""
    d_in, d_out = np.shape(W)
    if d_in % d or d_out % d:
        raise ValueError(f"W {np.shape(W)} does not tile into {d} x {d}")
    return [
        [
            EncodedTile.encode(
                W[i * d : (i + 1) * d, o * d : (o + 1) * d],
                N,
                None if name is None else f"{name}.{i}.{o}",
                dev_ops=dev_ops,
            )
            for o in range(d_out // d)
        ]
        for i in range(d_in // d)
    ]


def babies(ops, x, t: int, s: int):
    """The ``s`` baby rotations of ``x`` by ``b*t`` from one hoisted decomposition, each copied
    and realized once before the plaintext products (``diagonal::linear``'s ``x_b``)."""
    x = ops.copy(x)
    out = [x, *ops.rotate_many(x, [b * t for b in range(1, s)])]
    for xb in out:
        ops.realize(xb)
    return out


def _encode_key(ops, x):
    """What decides a plaintext's encode for a product with ``x``: its level and pending rescale
    (``encode_like_params``). None on ops without the accessors."""
    level = getattr(ops, "level", None)
    noise_deg = getattr(ops, "noise_deg", None)
    if level is None or noise_deg is None:
        return None
    return (level(x), noise_deg(x))


def _giant_group(ops, rx, w: EncodedTile, g: int):
    """``sum_b rx[b] * diagonal(g, b)``: giant group ``g`` before its rotation."""
    acc = None
    for b, xb in enumerate(rx):
        p = ops.mult_pt(xb, w.pt(g, b), key=w.key(g, b))
        acc = p if acc is None else ops.add(acc, p)
    return acc


def apply_tiles(ops, rxs, tiles):
    """``sum_i x_i @ W_i`` for prepared inputs ``rxs[i]`` (``babies``) and one tile per input.
    Rotation is linear, so each giant group sums every input's products before its single
    rotation by ``g*s*t``: ``G - 1`` giant rotations however many inputs are summed."""
    if len(rxs) != len(tiles) or not tiles:
        raise ValueError(f"{len(rxs)} prepared inputs for {len(tiles)} tiles")
    G, s, d = tiles[0].G, tiles[0].s, tiles[0].d
    t = tiles[0].N // d
    if any((w.G, w.s, w.d, w.N) != (G, s, d, tiles[0].N) for w in tiles):
        raise ValueError("tiles of one sum must share N, d and the (G, s) split")
    if any(len(rx) != s for rx in rxs):
        raise ValueError(f"every prepared input needs {s} baby rotations")
    device = all(w.dev is not None for w in tiles) and getattr(ops, "gpu_encode", False)
    if device and not all(ops.gpu_encode_fits(rx[0]) for rx in rxs):
        # the device encoder makes degree-1 plaintexts; an input it cannot serve (a pending
        # rescale on a one-prime chain) takes the CPU route, from the tiles fetched back
        warnings.warn(
            "apply_tiles: an input the device encoder cannot serve; CPU route for this group",
            stacklevel=2,
        )
        device = False
    if not device:
        tiles = [w.with_rows(ops if w.dev is not None else None) for w in tiles]
    y = None
    for g in range(G):
        if device:
            acc = _device_giant_groups(ops, rxs, tiles, g, s)
        else:
            acc = None
            for rx, w in zip(rxs, tiles):
                part = _giant_group(ops, rx, w, g)
                acc = part if acc is None else ops.add(acc, part)
        if g:
            acc = ops.rotate(acc, g * s * t)
        y = acc if y is None else ops.add(y, acc)
    return y


def _device_giant_groups(ops, rxs, tiles, g: int, s: int):
    """Giant index ``g`` of every tile on the device route: one batched device encode per distinct
    encode among the inputs (the babies of one input share theirs, so inputs at one level share a
    batch and an input at another gets its own, each at its reference ciphertext's), then the
    products and additions in exactly the CPU route's order (each tile's group summed, then the
    tiles' partials), so the recorded op stream is the same on both routes. The device plaintexts
    are evicted before returning."""
    keys = [_encode_key(ops, rx[0]) for rx in rxs]
    pts: list = [None] * (len(tiles) * s)
    made: list = []
    acc = None
    try:
        for key in dict.fromkeys(keys):  # in first-appearance order
            members = [i for i, k in enumerate(keys) if k == key]
            ids = [tiles[i].dev for i in members for _ in range(s)]
            baby = [b for _ in members for b in range(s)]
            names = [
                tiles[i].key(g, b) or f"tile{tiles[i].dev}.{g}.{b}"
                for i in members
                for b in range(s)
            ]
            group = ops.encode_group(ids, baby, g, s, rxs[members[0]][0], names)
            made.extend(group)
            for n, i in enumerate(members):
                pts[i * s : (i + 1) * s] = group[n * s : (n + 1) * s]
        for i, rx in enumerate(rxs):
            part = None
            for b, xb in enumerate(rx):
                p = ops.mult_ptx(xb, pts[i * s + b])
                part = p if part is None else ops.add(part, p)
            acc = part if acc is None else ops.add(acc, part)
    finally:
        for pt in made:
            ops.evict_ptx(pt)
    return acc


def linear(ops, x, w: EncodedTile):
    """y = x @ W for every conversation in x (the y = x @ W convention)."""
    return apply_tiles(ops, [babies(ops, x, w.N // w.d, w.s)], [w])


def linear_multi(ops, xs, weights):
    """Several wide matrices on the same input chunks (``encode_tiled`` each): the baby
    rotations of every input chunk are made once and shared by every output tile of every
    weight (``diagonal::linear_multi``'s "prepare x once"). Returns one list of output chunks
    per weight."""
    if not weights:
        return []
    for tiles in weights:
        if len(tiles) != len(xs):
            raise ValueError(f"{len(xs)} input chunks for {len(tiles)} tile rows")
    w0 = weights[0][0][0]
    rxs = [babies(ops, x, w0.N // w0.d, w0.s) for x in xs]
    return [
        [apply_tiles(ops, rxs, [tiles[i][o] for i in range(len(xs))]) for o in range(len(tiles[0]))]
        for tiles in weights
    ]


def linear_tiled(ops, xs, tiles):
    """A wide matrix as d x d tiles (``encode_tiled``): xs is the list of input chunks, the
    result the list of output chunks. Up: one input, several outputs; down: several inputs
    summed into one output."""
    return linear_multi(ops, xs, [tiles])[0]


def per_conversation_sum(ops, x, N: int, d: int):
    """Sum over each conversation's d channels, at every one of its slots: a rotate-and-sum
    over steps t, 2t, ..., N/2 (log2 d rotations), tagged t-periodic (one value per lane)."""
    t = N // d
    return ops.tag_reduce(ops.rotate_and_sum(x, t, N), t)


def rotation_steps(N: int, d: int):
    """Every rotation the layout issues: baby and giant steps of a d x d tile, the reduction."""
    t, s, G = tile_params(N, d)
    steps = {b * t for b in range(1, s)} | {g * s * t for g in range(1, G)}
    gap = t
    while gap < N:
        steps.add(gap)
        gap *= 2
    return steps
