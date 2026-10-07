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
        v = np.roll(np.repeat(self.vals, self.t), self.shift)
        return v if dtype is None else v.astype(dtype)


@dataclasses.dataclass(frozen=True)
class EncodedTile:
    """A d x d tile as its ``s * G`` diagonals, mirroring diagonal::encode_weight_matrix for
    d_in = d_out: diagonal (g, b) gives output channel j the weight W[(j + g*s + b) mod d, j],
    repeated over the t lanes of channel j and shifted by g*s*t. ``rows`` holds the d values
    of each diagonal, (G, s, d)."""

    N: int
    rows: np.ndarray
    name: str | None = None

    @classmethod
    def encode(cls, W, N: int, name: str | None = None):
        W = np.asarray(W)
        d = W.shape[0]
        if W.shape != (d, d):
            raise ValueError("cut wider matrices into d x d tiles (encode_tiled)")
        _, s, G = tile_params(N, d)
        j = np.arange(d)
        k = np.arange(G)[:, None, None] * s + np.arange(s)[None, :, None]
        return cls(N, W[(j + k) % d, j], name)

    @property
    def d(self) -> int:
        return self.rows.shape[2]

    def pt(self, g: int, b: int):
        t, s = self.N // self.d, self.rows.shape[1]
        return _Diagonal(self.rows[g, b], t, g * s * t)

    def key(self, g: int, b: int):
        return None if self.name is None else f"{self.name}.{g}.{b}"


def encode_tiled(W, N: int, d: int, name: str | None = None):
    """A (d_in, d_out) matrix as [input chunk][output chunk] tiles."""
    d_in, d_out = np.shape(W)
    if d_in % d or d_out % d:
        raise ValueError(f"W {np.shape(W)} does not tile into {d} x {d}")
    return [
        [
            EncodedTile.encode(
                W[i * d : (i + 1) * d, o * d : (o + 1) * d],
                N,
                None if name is None else f"{name}.{i}.{o}",
            )
            for o in range(d_out // d)
        ]
        for i in range(d_in // d)
    ]


def linear(ops, x, w: EncodedTile):
    """y = x @ W for every conversation in x (the y = x @ W convention)."""
    G, s, d = w.rows.shape
    t = w.N // d
    x = ops.copy(x)
    babies = [x, *ops.rotate_many(x, [b * t for b in range(1, s)])]
    for xb in babies:
        ops.realize(xb)
    y = None
    for g in range(G):
        acc = None
        for b in range(s):
            p = ops.mult_pt(babies[b], w.pt(g, b), key=w.key(g, b))
            acc = p if acc is None else ops.add(acc, p)
        if g:
            acc = ops.rotate(acc, g * s * t)
        y = acc if y is None else ops.add(y, acc)
    return y


def linear_tiled(ops, xs, tiles):
    """A wide matrix as d x d tiles (``encode_tiled``): xs is the list of input chunks, the
    result the list of output chunks. Up: one input, several outputs; down: several inputs
    summed into one output."""
    if len(xs) != len(tiles):
        raise ValueError(f"{len(xs)} input chunks for {len(tiles)} tile rows")
    out = []
    for o in range(len(tiles[0])):
        y = None
        for i, x in enumerate(xs):
            yi = linear(ops, x, tiles[i][o])
            y = yi if y is None else ops.add(y, yi)
        out.append(y)
    return out


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
