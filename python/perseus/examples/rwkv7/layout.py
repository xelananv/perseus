# SPDX-License-Identifier: MIT
"""The slot layout: one conversation, every channel replicated over the lanes of the filled token layout.

Ring 2^16 gives n = 32,768 complex slots. The filled layout (``fhe_ssm/filling.py``, PCMM's ``reshape(nch, t)``)
puts channel ``c`` of lane ``l`` at slot ``c t + l``, with ``D = 1,024`` channels and ``t = n / D = 32`` lanes. The
model's width 768 is padded to 1,024 (channels 768..1023 hold zeros); d_ffn = 3,072 is three ciphertexts.

**Vectors** ("natural" form): channel ``c`` of a width-768 vector on all ``t`` lanes of channel ``c``. With one
conversation every lane holds the same value, so a PCMM linear (the same matrix on every lane) keeps the form, and
every per-channel constant is a ``lane`` vector.

**The state.** Head ``h``'s 64 x 64 matrix ``S[h, i, j]`` (row ``i`` = value dim, column ``j`` = key dim) lives in
``M = 64 / t = 2`` ciphertexts: ``S_m`` holds ``S[h, i = t m + l, j]`` at channel ``h 64 + j``, lane ``l``. So
``S (.) w`` (``w[h, j]`` on channel ``h 64 + j``) is a plain product with a natural vector, and so are the products
with ``a``, ``b``, ``k`` and ``r`` (all indexed by ``j``). The two things that index by ``i`` move between forms:

* ``transpose(v)``: natural ``v`` -> ``V_m`` with ``v[h 64 + t m + l]`` on channel ``h 64 + j`` lane ``l`` for every
  ``j``: per lane ``l`` a rotation by ``t (t m + l)`` slots brings channel ``h 64 + t m + l`` to the block start,
  a mask keeps (block start, lane ``l``), a rotate-and-sum over channel steps ``-t .. -32 t`` copies it over the
  block. Baby-step giant-step over ``l = nb g + b``: ``nb - 1`` baby rotations (shared by both ``m``), ``M ng`` giant
  rotations, ``64`` mask products (one level), ``6 M`` copy rotations.
* ``untranspose(P_0, P_1)``: ``P_m`` holds ``S[h, t m + l, j] r[h, j]`` at (channel ``h 64 + j``, lane ``l``); a
  rotate-and-sum over channel steps ``t .. 32 t`` leaves ``y[h, t m + l]`` at (block start, lane ``l``); per lane a
  mask and a rotation by ``-(t^2 m + (t - 1) l)`` moves it to (channel ``h 64 + t m + l``, lane 0); a rotate-and-sum
  over lane steps ``-1 .. -t/2`` copies it to every lane. Same baby-step giant-step.
* ``block_allreduce(x, s)``: every channel of a head block gets ``s`` times the block's sum (per lane): a
  rotate-and-sum over ``t .. 32 t``, a block-start mask times ``s`` (one level), a rotate-and-sum over
  ``-t .. -32 t``.
* ``channel_sum(x)``: every slot gets the sum over all 1,024 channels of its lane (steps ``t .. n/2``).

Every function takes the ops protocol (``perseus.impl.ops``), so the same code runs on numpy rows (``NumpyOps``),
the fake, the capture recorder and a device session.

**Channel shards (widths above 1,024; ANYSIZE_DESIGN.md).** A width-C vector is ``S = ceil(C / 1024)`` ciphertexts
("shards"), shard ``s`` holding channels ``1024 s .. 1024 s + 1023`` in the layout above, every shard at the same
``t = 32`` lanes. A head's 64 channels never straddle a shard (1,024 = 16 heads), so every per-head kernel
(``block_allreduce``, ``transpose``, ``untranspose``) runs per shard unchanged, and the state of shard ``s``'s 16
heads is its own ``M`` ciphertexts: ``S M`` state ciphertexts per block. ``channel_sum`` is linear, so a LayerNorm
over C channels adds the shards first and sums once. The ``*_sh`` helpers below return one array per shard; the
unsuffixed ones are the single-shard forms (C <= 1,024), unchanged.
"""
from __future__ import annotations

import numpy as np

D = 1024        # channels per ciphertext (PCMM's nch at t = 32)
HEAD = 64       # head size (a block of channels)


def auto_lanes(C: int, n_slots: int = 1 << 15) -> int:
    """The default lane count for width C (ANYSIZE_DESIGN.md §3): ONE ciphertext per vector, i.e. the fewest lanes
    whose channels hold C (32 up to 1,024, 16 up to 2,048, 8 up to 4,096); wider models shard at 4,096 channels. The
    planner's counts decided it: at width 2,048, decode plans 38 / 42 refreshes in blocks 0 / 1 at t = 16 against
    67 / 73 as two shards at t = 32, and a batched pass costs the same 4 refreshes per conversation and block in both."""
    t = n_slots // D
    while t > 8 and C > n_slots // t:
        t //= 2
    return t


class Layout:
    def __init__(self, n_slots: int, C: int = 768, d_ffn: int = 3072, D: int = D):
        """``D`` channels per ciphertext (default 1,024: t = 32 lanes at 32,768 slots). ``D = 2,048`` / ``4,096`` is the
        fewer-lanes layout (t = 16 / 8): one ciphertext per vector up to that width, PCMM's own shape at that t
        (``impl.pcmm.Shape(t).nch = 32,768 / t``)."""
        if n_slots % D:
            raise ValueError(f"{n_slots} slots is not a multiple of {D} channels")
        self.D = D = int(D)
        self.n = int(n_slots)
        self.t = self.n // D
        if HEAD % self.t:
            raise ValueError(f"t = {self.t} lanes must divide the head size {HEAD}")
        self.M = HEAD // self.t                  # state ciphertexts per block and shard
        self.nb = min(8, self.t)                  # baby steps over the lanes
        self.ng = self.t // self.nb
        self.C, self.d_ffn = int(C), int(d_ffn)
        if C % HEAD:
            raise ValueError(f"C = {C} must be whole heads of {HEAD}")
        self.S = -(-self.C // D)                  # channel shards: ciphertexts per width-C vector
        self.Cp = self.S * D
        self.n_ffn = -(-self.d_ffn // D)          # d_ffn zero-padded to whole ciphertexts
        ch = np.arange(self.n) // self.t
        lane = np.arange(self.n) % self.t
        self._ch, self._lane = ch, lane
        #: shard s's live channels (1 on channels 1024 s + c < C)
        self.lives = [(ch + s * D < C).astype(np.float64) for s in range(self.S)]
        self.live = self.lives[0]
        self.block_start = (ch % HEAD == 0).astype(np.float64)
        self._masks = [((ch % HEAD == 0) & (lane == l)).astype(np.float64) for l in range(self.t)]

    # ── constants ──
    def lane(self, v, width: int | None = None) -> np.ndarray:
        width = self.D if width is None else width
        """A per-channel constant (length <= 1,024, zero-padded) in natural form."""
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size > width:
            raise ValueError(f"{v.size} channels > {width}")
        p = np.zeros(self.D)
        p[: v.size] = v
        return np.repeat(p, self.t)

    def head_const(self, per_head, fill=0.0) -> np.ndarray:
        """A per-head constant on every channel of its block; the padding blocks get ``fill``."""
        per_head = np.asarray(per_head, dtype=np.float64).ravel()
        v = np.full(self.D // HEAD, float(fill))
        v[: per_head.size] = per_head
        return np.repeat(np.repeat(v, HEAD), self.t)

    def pad_heads(self, value: float) -> np.ndarray:
        """``value`` on the padding head blocks (channels >= C), 0 elsewhere: added to a per-head statistic so the
        empty heads feed an inverse square root an in-range argument (their output is multiplied by zero anyway)."""
        return (1.0 - self.live) * float(value)

    def encode_row(self, row) -> np.ndarray:
        """The rows ``pack_tokens`` takes: the row (zero-padded to 1,024) on every lane."""
        r = np.zeros(self.D)
        row = np.asarray(row, dtype=np.float64).ravel()
        r[: row.size] = row
        return np.tile(r, (self.t, 1))

    def decode(self, vec) -> np.ndarray:
        """Lane 0's first C channels of a decrypted slot vector."""
        vec = np.asarray(vec)
        return vec[np.arange(self.C) * self.t]

    def natural(self, v) -> np.ndarray:
        """A vector in natural form as a slot vector (the mirror's input)."""
        return self.lane(v)

    def state_slots(self, S) -> list[np.ndarray]:
        """``S`` (H, 64, 64) as the M slot vectors of the state form (the mirror's and the tests')."""
        H = S.shape[0]
        out = []
        for m in range(self.M):
            v = np.zeros((self.D, self.t))
            for h in range(H):
                # channel h*64 + j, lane l  <-  S[h, t m + l, j]
                v[h * HEAD:(h + 1) * HEAD, :] = S[h, self.t * m:self.t * (m + 1), :].T
            out.append(v.reshape(-1))
        return out

    def state_from_slots(self, Ss, H: int) -> np.ndarray:
        S = np.zeros((H, HEAD, HEAD))
        for m, s in enumerate(Ss):
            v = np.real(np.asarray(s)).reshape(self.D, self.t)
            for h in range(H):
                S[h, self.t * m:self.t * (m + 1), :] = v[h * HEAD:(h + 1) * HEAD, :].T
        return S

    # ── channel shards (C > 1,024: one ciphertext per 1,024 channels) ──
    def split(self, v) -> list[np.ndarray]:
        """A length <= Cp vector, zero-padded to Cp, as S chunks of 1,024."""
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size > self.Cp:
            raise ValueError(f"{v.size} channels > {self.Cp}")
        p = np.zeros(self.Cp)
        p[: v.size] = v
        return [p[s * self.D:(s + 1) * self.D] for s in range(self.S)]

    def lanes(self, v) -> list[np.ndarray]:
        """A per-channel constant of any width <= Cp in natural form, one slot vector per shard."""
        return [self.lane(c) for c in self.split(v)]

    def pad_heads_sh(self, value: float) -> list[np.ndarray]:
        """``pad_heads`` per shard: ``value`` on every channel >= C of each shard."""
        return [(1.0 - lv) * float(value) for lv in self.lives]

    def encode_row_sh(self, row) -> list[np.ndarray]:
        """``pack_tokens`` rows per shard: shard s's 1,024 channels of ``row`` on every lane."""
        return [np.tile(c, (self.t, 1)) for c in self.split(row)]

    def encode_rows_sh(self, rows) -> list[np.ndarray]:
        """``encode_rows`` per shard: row l on lane l (lanes past the chunk repeat its last row)."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or not 1 <= rows.shape[0] <= self.t:
            raise ValueError(f"a chunk of 1..{self.t} rows, got {rows.shape}")
        full = np.zeros((self.t, self.Cp))
        full[: rows.shape[0], : rows.shape[1]] = rows
        full[rows.shape[0]:, : rows.shape[1]] = rows[-1]
        return [full[:, s * self.D:(s + 1) * self.D].copy() for s in range(self.S)]

    def decode_sh(self, vecs) -> np.ndarray:
        """Lane 0's first C channels of the decrypted shards."""
        return np.concatenate([np.asarray(v)[np.arange(self.D) * self.t] for v in vecs])[: self.C]

    def decode_lanes_sh(self, vecs, T: int) -> np.ndarray:
        """(T, C): lane l's first C channels of the decrypted shards."""
        return np.concatenate([np.asarray(v).reshape(self.D, self.t)[:, :T].T for v in vecs], axis=1)[:, : self.C].copy()

    def natural_sh(self, v) -> list[np.ndarray]:
        return self.lanes(v)

    def chunk_sh(self, rows) -> list[np.ndarray]:
        return [r.T.reshape(-1) for r in self.encode_rows_sh(rows)]

    def state_slots_sh(self, S) -> list[np.ndarray]:
        """``S`` (H, 64, 64) as the S M slot vectors of the sharded state form, shard-major (shard s holds heads
        16 s .. 16 s + 15 in ``state_slots``' layout)."""
        hps = self.D // HEAD
        out = []
        for s in range(self.S):
            part = S[s * hps:(s + 1) * hps]
            if part.shape[0] == 0:
                part = np.zeros((1, HEAD, HEAD))
            out += self.state_slots(part)
        return out

    def state_from_slots_sh(self, Ss, H: int) -> np.ndarray:
        hps = self.D // HEAD
        parts = []
        for s in range(self.S):
            h = min(hps, H - s * hps)
            parts.append(self.state_from_slots(Ss[s * self.M:(s + 1) * self.M], h))
        return np.concatenate(parts, axis=0)

    # ── rotations the layout issues ──
    def rotation_steps(self) -> set[int]:
        t, n = self.t, self.n
        s = set()
        g = t
        while g < n:                                   # channel_sum
            s.add(g)
            g *= 2
        g = t
        while g < HEAD * t:                            # block sums / copies
            s |= {g, -g}
            g *= 2
        s |= {t * b for b in range(1, self.nb)}         # transpose babies
        s |= {t * (t * m + self.nb * gg) for m in range(self.M) for gg in range(self.ng)}
        s |= {-(t - 1) * b for b in range(1, self.nb)}  # untranspose babies
        s |= {-(t * t * m + (t - 1) * self.nb * gg) for m in range(self.M) for gg in range(self.ng)}
        g = 1
        while g < t:                                   # lane copies
            s.add(-g)
            g *= 2
        return {x % n for x in s} - {0}

    # ── kernels ──
    def channel_sum(self, ops, x):
        return ops.rotate_and_sum(x, self.t, self.n)

    def block_allreduce(self, ops, x, scale: float = 1.0):
        s = ops.rotate_and_sum(x, self.t, HEAD * self.t)
        s = ops.mult_pt(s, self.block_start * float(scale))
        return ops.rotate_and_sum(s, -self.t, HEAD * self.t)

    def transpose(self, ops, v) -> list:
        t, nb = self.t, self.nb
        babies = [v] + [ops.rotate(v, t * b) for b in range(1, nb)]
        outs = []
        for m in range(self.M):
            acc = None
            for g in range(self.ng):
                G = t * (t * m + nb * g)
                pts = [np.roll(self._masks[nb * g + b], G) for b in range(nb)]
                inner = ops.mult_pt_sum(babies, pts)
                term = inner if G == 0 else ops.rotate(inner, G)
                acc = term if acc is None else ops.add(acc, term)
            outs.append(ops.rotate_and_sum(acc, -t, HEAD * t))
        return outs

    def untranspose(self, ops, Ps):
        return ops.rotate_and_sum(self.untranspose_lane0(ops, Ps), -1, self.t)

    def untranspose_lane0(self, ops, Ps):
        """``untranspose`` before its lane copy: ``y[h, t m + l]`` at (channel ``h 64 + t m + l``, lane 0), zero on
        every other lane (the prefill scan moves it to its token's lane instead of copying it)."""
        t, nb = self.t, self.nb
        acc = None
        for m, P in enumerate(Ps):
            R = ops.rotate_and_sum(P, t, HEAD * t)
            babies = [R] + [ops.rotate(R, -(t - 1) * b) for b in range(1, nb)]
            for g in range(self.ng):
                G = t * t * m + (t - 1) * nb * g
                pts = [np.roll(self._masks[nb * g + b], (t - 1) * b) for b in range(nb)]
                inner = ops.mult_pt_sum(babies, pts)
                term = inner if G == 0 else ops.rotate(inner, -G)
                acc = term if acc is None else ops.add(acc, term)
        return acc

    # ── token lanes (prefill: lane l holds token l of the chunk) ──
    def lane_mask(self, lanes, value: float = 1.0) -> np.ndarray:
        """``value`` on every channel of the given lanes, 0 elsewhere."""
        return np.isin(self._lane, list(lanes)).astype(np.float64) * float(value)

    def encode_rows(self, rows) -> np.ndarray:
        """``pack_tokens`` rows for a chunk: row ``l`` (zero-padded to 1,024) on lane ``l``. Lanes past the chunk
        repeat its last row, so they hold in-distribution values (a lane of zeros sends LayerNorm's Newton out of
        range, and one runaway slot spoils a whole ciphertext's refresh)."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or not 1 <= rows.shape[0] <= self.t:
            raise ValueError(f"a chunk of 1..{self.t} rows, got {rows.shape}")
        out = np.zeros((self.t, self.D))
        out[: rows.shape[0], : rows.shape[1]] = rows
        out[rows.shape[0]:, : rows.shape[1]] = rows[-1]
        return out

    def decode_lanes(self, vec, T: int) -> np.ndarray:
        """(T, C): lane l's first C channels of a decrypted slot vector."""
        v = np.asarray(vec).reshape(self.D, self.t)
        return v[: self.C, :T].T.copy()

    def chunk(self, rows) -> np.ndarray:
        """A chunk as a slot vector (the mirror's input)."""
        return self.encode_rows(rows).T.reshape(-1)
