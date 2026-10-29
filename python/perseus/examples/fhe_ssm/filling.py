# SPDX-License-Identifier: MIT
"""The dense token layout of perseus.impl.prefill on the ops protocol: channel ``i`` of conversation ``k`` at
slot ``i*t + k``, ``t = N / d``, so ``t`` conversations share a ciphertext (``2t`` as pairs).

The linear is ``perseus.impl.prefill.DiagLinear`` (``b`` baby rotations by ``j*t``, ``g`` giant groups by
``g*b*t``, ``b*g = d`` products per ``d x d`` block). ``DiagLinear.slot_vectors`` materializes every plaintext
(1.07 GB per ``1024 x 4096`` matrix at ring 2^16), so ``encode_diag`` hands it ``LazyDiagonals``, one giant
group on read; ``apply_diag`` is ``prefill.diag_apply`` with the input hint's headroom as a parameter (2
upstream's, 1 the fork's reference circuit). ``diag_slot_index`` / ``diag_gather_reference`` are the device
diagonal encoder's index arithmetic in numpy; ``mult_diag_sum`` is the one call per ``(m, gg)`` group (a session's
hook, ``diaggen``, or ``DiagLinear._sum``). README.md has the derivations.
"""

from __future__ import annotations

import numpy as np

from perseus.impl.prefill import DiagLinear


def tile_params(N: int, d: int) -> tuple[int, int, int]:
    """``(t, b, g)`` for a ``d x d`` block in ``N`` slots, as ``DiagLinear.split`` chooses them."""
    if N % d:
        raise ValueError(f"N={N} is not a multiple of d={d}")
    b, g = DiagLinear.split(d)
    return N // d, b, g


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


class LazyDiagonals:
    """``DiagLinear.slot_vectors(W, N, t)`` one ``(m, g)`` giant group at a time: ``pts[m, g]`` is the
    ``(n_in, b, N)`` array of that group's pre-rotated diagonals, built on read (the only access
    ``DiagLinear._sum`` makes), so a block's weights cost their ``float64`` matrices, not their slot
    expansions (8 MB per group against 1.07 GB per ``1024 x 4096`` matrix at ring 2^16)."""

    def __init__(self, W, N: int, t: int):
        self.W = np.asarray(W, dtype=np.float64)
        self.N, self.t = int(N), int(t)
        self.B = self.N // self.t
        if self.W.shape[0] % self.B or self.W.shape[1] % self.B:
            raise ValueError(f"W {self.W.shape} does not tile into {self.B} x {self.B} blocks")
        self.n_in, self.n_out = self.W.shape[0] // self.B, self.W.shape[1] // self.B
        self.b, self.g = DiagLinear.split(self.B)

    @property
    def shape(self):
        return (self.n_out, self.g, self.n_in, self.b, self.N)

    def __getitem__(self, key):
        m, gg = key
        B, t, i = self.B, self.t, np.arange(self.B)
        out = np.empty((self.n_in, self.b, self.N))
        for n in range(self.n_in):
            blk = self.W[n * B : (n + 1) * B, m * B : (m + 1) * B]
            for j in range(self.b):
                k = gg * self.b + j
                out[n, j] = np.roll(np.repeat(blk[(i + k) % B, i], t), gg * self.b * t)
        return out

    def dense(self):
        """The whole table, as ``DiagLinear.slot_vectors`` returns it (tests; small shapes only)."""
        return np.stack([np.stack([self[m, gg] for gg in range(self.g)]) for m in range(self.n_out)])


def diag_slot_index(v, n: int, m: int, gg: int, jj: int, B: int, t: int):
    """``(row, col)`` in the ``(d_in, d_out)`` matrix of the element slot ``v`` of plaintext ``(n, m, gg, jj)`` holds
    (``DiagLinear.slot_vectors``: ``np.roll(np.repeat(blk[(i + k) % B, i], t), gg b t)``, ``k = gg b + jj``, ``blk``
    the ``(n, m)`` block). The roll moves elements to higher indices, so slot ``v`` reads element ``v - gg b t`` of the
    repeated diagonal: ``i = ((v - gg b t) mod n_slots) // t``, row ``n B + ((i + k) mod B)``, col ``m B + i``; the
    first mod is a mask (``n_slots = B t`` a power of two), the second one conditional subtract (``i + k < 2 B``).
    ``v`` an int or an integer array; the kernel's arithmetic (crates/pc-program/src/compile/diag.rs
    ``TileInfo::slot``)."""
    b, _g = DiagLinear.split(B)
    n_slots = B * t
    if n_slots & (n_slots - 1):
        raise ValueError(f"n_slots = B t = {n_slots} is not a power of two")
    v = np.asarray(v, dtype=np.int64)
    i = ((v - gg * b * t) & (n_slots - 1)) // t
    r = i + gg * b + jj
    r = np.where(r >= B, r - B, r)
    return n * B + r, m * B + i


def diag_gather_reference(W, n_slots: int, t: int, jobs):
    """The slot vectors of plaintexts ``jobs`` = ``[(n, m, gg, jj), ...]`` of the ``(d_in, d_out)`` matrix ``W``:
    ``(len(jobs), n_slots)`` float64, each element read from ``W`` at ``diag_slot_index`` (no rolling, no repeat:
    one gather per slot, as the device does it). Equals ``LazyDiagonals(W, n_slots, t)[m, gg][n, jj]`` bit for bit."""
    W = np.asarray(W, dtype=np.float64)
    B = n_slots // t
    v = np.arange(n_slots)
    out = np.empty((len(jobs), n_slots))
    for q, (n, m, gg, jj) in enumerate(jobs):
        r, c = diag_slot_index(v, int(n), int(m), int(gg), int(jj), B, t)
        out[q] = W[r, c]
    return out


def group_jobs(lin: DiagLinear, m: int, gg: int) -> list[tuple[int, int, int, int]]:
    """The ``(n, m, gg, jj)`` of one ``(m, gg)`` group in the order ``DiagLinear._sum`` reads its terms (n outer, jj
    inner: term ``k = n b + jj``)."""
    return [(n, m, gg, jj) for n in range(lin.n_in) for jj in range(lin.b)]


def diag_name(lin: DiagLinear, n: int, m: int, gg: int, jj: int) -> str:
    """A generated diagonal's plaintext name, ``<linear>#<n>.<m>.<gg>.<jj>`` (the compiler's ``PtForm::DiagGen``)."""
    if not lin.name or "#" in lin.name:
        raise ValueError(f"a generated diagonal needs its linear's name (b<b>.<kind>), got {lin.name!r}")
    return f"{lin.name}#{n}.{m}.{gg}.{jj}"


def mult_diag_sum(ops, xs, lin: DiagLinear, m: int, gg: int):
    """``sum_{n, jj} P[n, m, gg, jj] * xs[n][jj]``, one call per ``(m, gg)`` group: the session's ``mult_diag_sum``
    hook when it has one (``diaggen.use_source``: the fake builds the vectors with ``diag_gather_reference``, the
    capture recorder records generated diagonals), else ``DiagLinear._sum`` (the values: ``pts[m, gg]`` through
    ``ops.mult_pt_sum``), unchanged."""
    hook = getattr(ops.inf, "mult_diag_sum", None)
    if hook is None:
        return lin._sum(ops, xs, m, gg)
    return hook(ops, [x for xn in xs for x in xn], lin, m, gg)


def encode_diag(W, N: int, t: int, name: str | None = None) -> DiagLinear:
    """A ``(d_in, d_out)`` matrix as a ``DiagLinear`` over lazy diagonals (``LazyDiagonals``)."""
    pts = LazyDiagonals(W, N, t)
    return DiagLinear(pts.n_in, pts.n_out, pts.b, pts.g, pts=pts, name=name)


def apply_diag(rt, Xs, lins, hint: int = 1):
    """``prefill.diag_apply`` (prefill.py:1223-1248) with the input hint's headroom as a parameter: every
    input is refreshed unless ``hint`` more product units fit under the ceiling (1 = the reference circuit,
    the fork's: the linear's one product fits and its output may sit at the ceiling; 2 = upstream's rule:
    the products, one unit deeper, stay under the limit). At ``hint=2`` this records ``diag_apply``'s op
    stream exactly (tests/test_filling_fake.py). ``lins`` share ``Xs`` (one set of baby rotations); per
    linear its ``n_out`` output blocks."""
    ops, d = rt.ops, rt.dims
    b = lins[0].b
    if any((lin.b, lin.g, lin.n_in) != (b, lins[0].g, len(Xs)) for lin in lins):
        raise ValueError("the linears of one call must share the split and read every input block")
    for x in Xs:
        ops.bootstrap_hint(x, ops.headroom(hint), True)
    xs = [[x, *ops.rotate_many(x, [j * d.t for j in range(1, b)])] for x in Xs]
    for xn in xs:
        for x in xn:
            ops.realize(x)
    out = []
    for lin in lins:
        blocks = []
        for m in range(lin.n_out):
            acc = None
            for gg in range(lin.g):
                tmp = mult_diag_sum(ops, xs, lin, m, gg)
                if gg:
                    tmp = ops.rotate(tmp, gg * lin.b * d.t)
                if acc is None:
                    acc = tmp
                else:
                    ops.inplace_add(acc, tmp)
            blocks.append(acc)
        out.append(blocks)
    return out


def per_conversation_sum(ops, x, N: int, d: int):
    """Sum over each conversation's d channels, at every one of its slots: a rotate-and-sum
    over steps t, 2t, ..., N/2 (log2 d rotations), tagged t-periodic (one value per lane)."""
    t = N // d
    return ops.tag_reduce(ops.rotate_and_sum(x, t, N), t)


def rotation_steps(N: int, d: int) -> set[int]:
    """Every rotation the layout issues: baby and giant steps of a ``d x d`` block, the reduction."""
    t, b, g = tile_params(N, d)
    steps = {j * t for j in range(1, b)} | {k * b * t for k in range(1, g)}
    gap = t
    while gap < N:
        steps.add(gap)
        gap *= 2
    return steps
