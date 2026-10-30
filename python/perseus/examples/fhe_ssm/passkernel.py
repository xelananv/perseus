# SPDX-License-Identifier: MIT
"""The pass kernels' lane form: a real-linear map from every lane-slot's value onto the lane-slots of other lanes in
the same channel, on a filled pair ciphertext, as a baby-step giant-step over the lanes, in both directions.

The layout (filling.py, pairs.py): channel ``i`` of lane ``j`` at slot ``m = i t + j``, ``t = N / d`` lanes, the real
axis of lane ``j`` its lane-slot ``2j``, the imaginary axis ``2j + 1``. The term, causal (``sign = -1``: lane ``j``
reads lane ``j - s``, the scan and the token shift) or anti-causal (``sign = +1``: lane ``j`` reads lane ``j + s``, the
carry prep's broadcast)::

    T(x) = sum_{s=0}^{t-1} P_s (.) sh_s(x) + P'_s (.) conj(sh_s(x)),   sh_s(x)[i, j] = x[i, j + sign s]  (0 outside [0, t))

``P_s``, ``P'_s`` are target-indexed (entry ``[i, j]`` belongs to the target lane ``j``) and zero wherever lane
``j + sign s`` falls outside ``[0, t)``. Two complex products per offset realize any real-linear map of the source pair
``(u, v) = (Re, Im)``: ``P y + P' conj(y) = ((p + p') u + (q' - q) v) + i ((q + q') u + (p - p') v)`` for ``P = p + iq``,
``P' = p' + iq'`` (passmasks.py builds the coefficients from the lane-slot maps).

**The pre-rotation rule, both directions.** With ``rot(v, k)[m] = v[(m + k) mod N]`` (``fake.py``: ``np.roll(v, -k)``),
a term whose target ``m`` reads source ``m + delta``, ``delta = G + B`` with the giant ``G = sign NB g`` and the baby
``B = sign b`` (``s = NB g + b``, ``delta = sign s``)::

    rot(M (.) rot(x, B), G)[m] = (M (.) rot(x, B))[m + G] = M[m + G] x[m + G + B] = M[m + G] x[m + delta]

so ``M[m + G] = coef_delta[m]`` for every ``m``: ``M[m'] = coef_delta[(m' - G) mod N]``, numpy ``np.roll(coef_delta,
G)`` (``prerotate``). The slot roll handles the channel wrap on its own. Causal (``G = -NB g``): ``M[i t + j'] =
coef[i t + j' + NB g]``; where ``j' + NB g >= t`` the roll reads channel ``i + 1``'s target lane ``j' + NB g - t``, whose
source ``j' + NB g - t - s = j' - t - b < 0`` has no lane, so that coefficient is zero already: the zero the lane-mix
term writes out (``lanemix.py`` ``src < t``) is redundant there. Anti-causal (``G = +NB g``): ``M[i t + j'] =
coef[i t + j' - NB g]``; where ``j' < NB g`` the roll reads channel ``i - 1``'s target lane ``t + j' - NB g``, whose source
``t + j' + b >= t`` has no lane: zero again. The causal rule's zero is **wrong** anti-causally. Transcribed literally
under ``+NB g`` giants (``M = np.roll(coef, -NB g)``, zero where ``j' + NB g >= t``) every group ``g >= 1`` reads the
targets ``2 NB g`` lanes away from the right ones; mirrored (``np.roll(coef, +NB g)``) with that zero kept, it drops the
targets ``j = j' - NB g >= t - 2 NB g`` of every group ``g >= 1``, live ones included (lanes 0..23 of a home that ends at
lane 31). On a home of at most ``NB`` lanes (no giant step) or one that ends at lane ``2 NB - 1`` or below at ``t = 2 NB
NG = 32`` (no target there reads a zeroed lane) the mirrored variant passes by accident: ``tests/test_passkernel.py``
fails both variants where they can fail and records where the second passes vacuously.

``LaneKernel.apply`` is the lane-mix term's op stream (``lanemix.py`` ``apply``) with ``sign`` and ``NB NG = t``
(``bsgs_split``): the ``NB - 1`` baby rotations of ``x`` (and of ``conj x`` when the conjugate half is on) from one
hoisted decomposition each, realized (no graph node: ``fake._Core.realize_pending_rescale``), per giant group ``NB``
products (``2 NB`` with the conjugate half) summed, one giant rotation by ``sign NB g`` for ``g >= 1``; depth 1. Without
the conjugate half (the carry prep, whose ``P'_s`` is zero in every configuration by construction) there is no
conjugation and no conjugate babies. ``direct`` / ``rows`` are the same term per offset on lanes / on rows (the mirror).

**The pass hook** (the one call every pass-mask product goes through, installed like ``diaggen.use_source``'s):
``passmult(ops, ct, mask_id, vec)`` reads ``ops.inf.mult_pass``; ``use_pass_hook(inf)`` installs it. On the numpy fake
(``fake.FakeInf``) the branch is ``ops.mult_pt(ct, vec)``: the current values, encoded at use, no key (a key would
freeze the first configuration's values in the runtime's caches, ``fake.FakeInf._cached`` / ``FheOps._slot``). The
capture recorder and the planned twin (``fakecap.CapInf``) need their own branch: the product
``inf.fhe.mult(ct, fakecap.Pt(inf._full(vec), *inf._like(ct), Tag(), src=("pass", mask_id)))`` (what
``CapInf.mult_pt`` records, without the value tag and with the mask id as the plaintext's source, the full vector
kept in every mode). Until that branch exists ``use_pass_hook`` refuses a capture recorder, so no capture can record
value-tagged pass masks. A mask id never holds a ``#`` (the driver files ``#`` names as generated diagonals).
"""

from __future__ import annotations

import math

import numpy as np

from perseus.impl import fake, fakecap

#: lane j reads lane j - s (the scan, the token shift) / lane j + s (the carry prep's broadcast)
CAUSAL, ANTICAUSAL = -1, 1
SIGNS = (CAUSAL, ANTICAUSAL)
#: the attribute ``use_pass_hook`` sets on a session's ``inf``, and the fake branch's product counter beside it
HOOK = "mult_pass"
COUNTER = "pass_products"


def bsgs_split(t: int) -> tuple[int, int]:
    """``(NB, NG)`` with ``NB NG = t``: ``2^ceil(k/2) x 2^floor(k/2)`` for ``t = 2^k`` (``t = 32``: 8 x 4, the lane-mix
    term's cut), else the smallest divisor ``NB`` of ``t`` with ``NB^2 >= t``."""
    t = int(t)
    if t < 1:
        raise ValueError(f"t must be a positive lane count, got {t}")
    if t & (t - 1) == 0:
        nb = 1 << (t.bit_length() // 2)  # t = 2^k: bit_length k + 1, so 2^ceil(k / 2)
        return nb, t // nb
    nb = next(n for n in range(max(1, math.isqrt(t)), t + 1) if t % n == 0 and n * n >= t)
    return nb, t // nb


def slot_vector(coef) -> np.ndarray:
    """``(t, d)`` lane-major coefficients (entry ``[j, i]``: lane ``j``, channel ``i``) -> the slot vector, channel
    ``i`` of lane ``j`` at slot ``i t + j``."""
    c = np.asarray(coef)
    if c.ndim != 2:
        raise ValueError(f"a lane array is (t, d), got {c.shape}")
    return np.ascontiguousarray(c.T).reshape(-1)


def lane_array(v, t: int) -> np.ndarray:
    """``slot_vector``'s inverse: the ``(t, d)`` lane array of a slot vector."""
    v = np.asarray(v)
    if v.ndim != 1 or v.shape[0] % int(t):
        raise ValueError(f"{v.shape} slots do not hold {t} lanes")
    return np.ascontiguousarray(v.reshape(-1, int(t)).T)


def prerotate(coef_slot, s: int, nb: int, sign: int) -> np.ndarray:
    """The plaintext baby ``b = s mod NB`` of giant group ``g = s div NB`` multiplies (the module docstring):
    ``np.roll(coef_slot, sign NB g)``."""
    if sign not in SIGNS:
        raise ValueError(f"sign must be one of {SIGNS}, got {sign!r}")
    return np.roll(np.asarray(coef_slot), int(sign) * int(nb) * (int(s) // int(nb)))


def lane_shift(X, s: int, sign: int) -> np.ndarray:
    """``sh_s(X)[j] = X[j + sign s]`` on a ``(t, ...)`` lane array, zero where ``j + sign s`` is outside ``[0, t)``."""
    X = np.asarray(X)
    t, s = X.shape[0], int(s)
    Y = np.zeros_like(X)
    if s >= t:
        return Y
    if s == 0:
        return X.copy()
    if sign == CAUSAL:
        Y[s:] = X[: t - s]
    elif sign == ANTICAUSAL:
        Y[: t - s] = X[s:]
    else:
        raise ValueError(f"sign must be one of {SIGNS}, got {sign!r}")
    return Y


def direct(X, P, Pc, sign: int) -> np.ndarray:
    """``sum_s P[s] sh_s(X) + Pc[s] conj(sh_s(X))`` on ``(t, d)`` complex lanes, offset by offset (``Pc`` None: no
    conjugate half). ``P``, ``Pc``: ``(S, t, d)`` (or ``(S, t, 1)``, broadcast), offsets ``0 .. S - 1``."""
    X = np.asarray(X, dtype=np.complex128)
    T = np.zeros_like(X)
    for s in range(np.shape(P)[0]):
        y = lane_shift(X, s, sign)
        T += P[s] * y
        if Pc is not None:
            T += Pc[s] * np.conj(y)
    return T


def pair_lanes(rows_q) -> np.ndarray:
    """``(2t, d)`` real rows in lane-slot order (``q = 2j + e``) -> the ``(t, d)`` complex lanes ``rows[2j] + i rows[2j + 1]``."""
    R = np.asarray(rows_q, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] % 2:
        raise ValueError(f"lane-slot rows are (2t, d), got {R.shape}")
    return R[0::2] + 1j * R[1::2]


def unpair_lanes(X) -> np.ndarray:
    """``pair_lanes``' inverse: lane ``j``'s real axis at row ``2j``, its imaginary axis at row ``2j + 1``."""
    X = np.asarray(X)
    out = np.empty((2 * X.shape[0], *X.shape[1:]), dtype=np.float64)
    out[0::2] = np.real(X)
    out[1::2] = np.imag(X)
    return out


def rows(rows_q, P, Pc, sign: int) -> np.ndarray:
    """The term on ``(2t, d)`` real rows in lane-slot order (the mirror's form): paired, ``direct``, unpaired."""
    return unpair_lanes(direct(pair_lanes(rows_q), P, Pc, sign))


def bsgs_numpy(x, M, Mc, sign: int, nb: int, ng: int) -> np.ndarray:
    """``LaneKernel.apply``'s arithmetic on a numpy slot vector: ``M[s]`` / ``Mc[s]`` are the pre-rotated plaintexts
    (``Mc`` None: no conjugate half), babies ``rot(., sign b)``, giants ``rot(., sign NB g)``. Each product is
    ``ciphertext * plaintext`` in that order, as the fake computes it (``FakeInf._mult_pt``): numpy's complex multiply
    is not commutative bit for bit (``a * b`` and ``b * a`` differed in the last bit on numpy 2.5.4)."""
    x = np.asarray(x, dtype=np.complex128)
    halves = [x] if Mc is None else [x, np.conj(x)]
    babies = [[np.roll(v, -sign * b) for b in range(nb)] for v in halves]
    total: np.ndarray | None = None
    for g in range(ng):
        acc: np.ndarray | None = None
        for b in range(nb):
            s = nb * g + b
            p = babies[0][b] * M[s]
            if Mc is not None:
                p = p + babies[1][b] * Mc[s]
            acc = p if acc is None else acc + p
        if acc is None:
            raise ValueError(f"a giant group of {nb} babies is empty")
        if g:
            acc = np.roll(acc, -sign * nb * g)
        total = acc if total is None else total + acc
    if total is None:
        raise ValueError(f"no giant group ({ng})")
    return total


class LaneKernel:
    """The lane term over ``t`` lanes in one direction on the ops protocol (the module docstring); the plaintexts come
    from the caller, one product at a time, through ``mult(ops, ct, s, half)`` (``half`` 0: ``P_s``, 1: ``P'_s``),
    which multiplies ``ct`` by the pre-rotated plaintext of offset ``s`` (``mask``)."""

    def __init__(self, t: int, sign: int, conj_half: bool = True):
        if sign not in SIGNS:
            raise ValueError(f"sign must be one of {SIGNS}, got {sign!r}")
        self.t, self.sign, self.conj_half = int(t), int(sign), bool(conj_half)
        self.nb, self.ng = bsgs_split(self.t)

    @property
    def products(self) -> int:
        return self.t * (2 if self.conj_half else 1)

    def rotation_steps(self, N: int) -> set[int]:
        """The baby and giant steps, stored as ``k mod N`` (the band's convention, ``lanemix.rotation_steps``)."""
        return {(self.sign * b) % N for b in range(1, self.nb)} | {
            (self.sign * self.nb * g) % N for g in range(1, self.ng)
        }

    def mask(self, coef_slot, s: int) -> np.ndarray:
        """The pre-rotated plaintext of offset ``s`` from its target-indexed slot vector."""
        return prerotate(coef_slot, s, self.nb, self.sign)

    def apply(self, ops, x, mult):
        """``T(x)`` on the slots: the op stream of the module docstring."""
        sign, nb, ng = self.sign, self.nb, self.ng
        halves = [x, ops.conjugate(x)] if self.conj_half else [x]
        babies = []
        for v in halves:
            rot = [v, *ops.rotate_many(v, [sign * b for b in range(1, nb)])]
            for r in rot:
                ops.realize(r)
            babies.append(rot)
        total = None
        for g in range(ng):
            acc = None
            for b in range(nb):
                s = nb * g + b
                p = mult(ops, babies[0][b], s, 0)
                if self.conj_half:
                    p = ops.add(p, mult(ops, babies[1][b], s, 1))
                acc = p if acc is None else ops.add(acc, p)
            if g:
                acc = ops.rotate(acc, sign * nb * g)
            total = acc if total is None else ops.add(total, acc)
        return total


# ── the pass hook ──────────────────────────────────────────────────────────────────────────────


def pass_mult_fake(ops, ct, mask_id: str, vec):  # noqa: ARG001 (the hook's signature)
    """The numpy fake's branch: the product with the current values, encoded at use, no key; counted on the inf
    (``inf.pass_products``) for the driver's census."""
    setattr(ops.inf, COUNTER, getattr(ops.inf, COUNTER, 0) + 1)
    return ops.mult_pt(ct, vec)


def use_pass_hook(inf) -> None:
    """Install the pass hook on ``inf`` (the module docstring): the fake's branch on ``fake.FakeInf``; a capture
    recorder (``fakecap.CapInf``) needs the recorder's branch, which this module does not build yet. The device never
    runs the hook: its programs are compiled from the twin's recording."""
    if isinstance(inf, fakecap.CapInf):  # a FakeInf subclass: checked first
        raise NotImplementedError(
            "the pass kernels on the capture recorder need the recorder's branch of the pass hook: "
            "inf.fhe.mult(ct, fakecap.Pt(inf._full(vec), *inf._like(ct), Tag(), src=('pass', mask_id)))"
        )
    if isinstance(inf, fake.FakeInf):
        setattr(inf, HOOK, pass_mult_fake)
        setattr(inf, COUNTER, 0)
        return
    raise TypeError(f"the pass hook runs on the fake or the capture recorder, not on {type(inf)!r}")


def passmult(ops, ct, mask_id: str, vec):
    """``ct (.) vec`` for the pass mask ``mask_id``: the one call every pass-mask product goes through."""
    if "#" in mask_id:
        raise ValueError(f"a pass mask id never holds '#' (generated-diagonal names do): {mask_id!r}")
    hook = getattr(ops.inf, HOOK, None)
    if hook is None:
        raise RuntimeError("no pass hook on this session's inf (passkernel.use_pass_hook)")
    return hook(ops, ct, mask_id, vec)
