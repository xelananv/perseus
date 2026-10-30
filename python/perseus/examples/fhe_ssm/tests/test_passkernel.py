# SPDX-License-Identifier: MIT
"""The pass kernels' lane form (passkernel.py), both directions, two ways: the baby-step giant-step on slot vectors
with the slot-roll pre-rotation against the direct per-offset form and against the rows form, on random complex
target-indexed coefficients and on the carry prep's own masks; and the trap: the two wrong pre-rotations of the
anti-causal kernel (T1: the causal rule transcribed for +NB g giants; T2: the mirrored shift with the causal rule's
"zero where j + NB g >= t" kept) must fail where they can fail (a home ending at lane 31 spanning >= 9 lanes) and the
test records where T2 passes vacuously (a home of <= 8 lanes, or one ending at lane <= 15)."""

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, passkernel
from perseus.examples.fhe_ssm.passkernel import ANTICAUSAL, CAUSAL, LaneKernel
from perseus.impl import fake
from perseus.impl.ops import FheOps, NumpyOps

T, D = 32, 4
N = T * D


def rel(got, want):
    """``max |got - want| / max |want|``."""
    return float(np.abs(got - want).max() / np.abs(want).max())


def rel_rms(got, want):
    return float(np.sqrt(np.mean(np.abs(got - want) ** 2)) / np.sqrt(np.mean(np.abs(want) ** 2)))


def random_coefs(rng, sign, t=T, d=D, conj_half=True):
    """Random complex target-indexed coefficients ``(t, t, d)``, zero where lane ``j + sign s`` has no source."""
    j = np.arange(t)
    valid = np.array([((j + sign * s >= 0) & (j + sign * s < t)) for s in range(t)], dtype=np.float64)

    def draw():
        return (rng.standard_normal((t, t, d)) + 1j * rng.standard_normal((t, t, d))) * valid[:, :, None]

    return draw(), (draw() if conj_half else None)


def slot_masks(P, rule, t=T):
    """Per offset the plaintext the BSGS multiplies, by ``rule(coef_slot, s)``."""
    return [rule(passkernel.slot_vector(P[s]), s) for s in range(t)]


def rule_slot_roll(sign, t=T):
    nb, _ng = passkernel.bsgs_split(t)
    return lambda v, s: passkernel.prerotate(v, s, nb, sign)


def rule_t1(t=T):
    """T1: the causal rule (``P~[j] = P_s[j + NB g]``, zero where ``j + NB g >= t``) used under +NB g giants."""
    nb, _ng = passkernel.bsgs_split(t)

    def rule(v, s):
        g = s // nb
        m = np.roll(v, -nb * g)
        lane = np.arange(v.shape[0]) % t
        return m * (lane + nb * g < t)

    return rule


def rule_t2(t=T):
    """T2: the mirrored shift (``P~[j] = P_s[j - NB g]``) with the causal rule's zero (``j + NB g >= t``) kept."""
    nb, _ng = passkernel.bsgs_split(t)

    def rule(v, s):
        g = s // nb
        m = np.roll(v, nb * g)
        lane = np.arange(v.shape[0]) % t
        return m * (lane + nb * g < t)

    return rule


def run_bsgs(x, P, Pc, sign, rule, t=T):
    nb, ng = passkernel.bsgs_split(t)
    M = slot_masks(P, rule, t)
    Mc = None if Pc is None else slot_masks(Pc, rule, t)
    return passkernel.bsgs_numpy(x, M, Mc, sign, nb, ng)


def k2_masks(homes, t=T, d=D):
    """The carry prep's ``R_s`` as ``(t, t, d)``: 1/4 at target lane j of a home ``[e0, e1]`` (lanes, inclusive) where
    ``j + s`` is the home's end lane ``e1``."""
    R = np.zeros((t, t, d))
    for e0, e1 in homes:
        for j in range(e0, e1 + 1):
            R[e1 - j, j, :] = 0.25
    return R


def random_slots(rng, n=N):
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


# ── the trap (written first) ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_anticausal_trap_on_random_coefficients(seed):
    """Random complex target-indexed coefficients, lane j reads lane j + s: the slot roll equals the direct form and
    the rows form; T1 and T2 do not."""
    rng = np.random.default_rng(seed)
    P, Pc = random_coefs(rng, ANTICAUSAL)
    x = random_slots(rng)
    X = passkernel.lane_array(x, T)
    want = passkernel.slot_vector(passkernel.direct(X, P, Pc, ANTICAUSAL))
    rows = passkernel.unpair_lanes(X)
    want_rows = passkernel.rows(rows, P, Pc, ANTICAUSAL)
    np.testing.assert_array_equal(passkernel.pair_lanes(want_rows), passkernel.lane_array(want, T))
    got = run_bsgs(x, P, Pc, ANTICAUSAL, rule_slot_roll(ANTICAUSAL))
    assert rel_rms(got, want) <= 1e-15
    assert rel(got, want) <= 1e-15
    for rule in (rule_t1(), rule_t2()):
        assert rel_rms(run_bsgs(x, P, Pc, ANTICAUSAL, rule), want) > 0.1


@pytest.mark.parametrize("homes", [[(0, 31)], [(16, 31)], [(20, 31)], [(0, 9), (10, 31)]])
def test_anticausal_trap_on_the_carry_preps_masks(homes):
    """The carry prep's own masks (R = 1/4 at the end lane, no conj half) on homes that end at lane 31 and span at
    least 9 lanes: the slot roll broadcasts the end lane to every home lane; T1 and T2 lose lanes."""
    rng = np.random.default_rng(len(homes) + homes[0][0])
    R = k2_masks(homes)
    x = random_slots(rng)
    X = passkernel.lane_array(x, T)
    want = passkernel.slot_vector(passkernel.direct(X, R, None, ANTICAUSAL))
    # the broadcast, by hand: Z[j] = X[e1] / 4 on every lane of the home, 0 elsewhere
    hand = np.zeros_like(X)
    for e0, e1 in homes:
        hand[e0 : e1 + 1] = 0.25 * X[e1]
    np.testing.assert_array_equal(passkernel.lane_array(want, T), hand)
    assert rel(run_bsgs(x, R, None, ANTICAUSAL, rule_slot_roll(ANTICAUSAL)), want) <= 1e-15
    for rule in (rule_t1(), rule_t2()):
        assert rel(run_bsgs(x, R, None, ANTICAUSAL, rule), want) > 0.1


@pytest.mark.parametrize(("homes", "t1_too"), [([(24, 31)], True), ([(26, 31)], True), ([(0, 15)], False),
                                               ([(4, 15)], False)])  # fmt: skip
def test_t2_passes_vacuously_on_short_or_low_homes(homes, t1_too):
    """The vacuity condition: a home of <= 8 lanes (offsets < NB: no giant step) or one ending at lane <= 15 (T2's
    zero falls on lanes j' >= t - NB g that no target of this home reads) hides T2's error; T1 is hidden only when no
    giant step is taken (<= 8 lanes). A trap test on such a home proves nothing."""
    rng = np.random.default_rng(5)
    R = k2_masks(homes)
    x = random_slots(rng)
    want = passkernel.slot_vector(passkernel.direct(passkernel.lane_array(x, T), R, None, ANTICAUSAL))
    assert rel(run_bsgs(x, R, None, ANTICAUSAL, rule_t2()), want) <= 1e-15
    t1 = rel(run_bsgs(x, R, None, ANTICAUSAL, rule_t1()), want)
    assert (t1 <= 1e-15) == t1_too


# ── both directions, the slot form against the direct and rows forms ────────────────────────────


@pytest.mark.parametrize("sign", [CAUSAL, ANTICAUSAL])
@pytest.mark.parametrize("conj_half", [True, False])
def test_bsgs_equals_direct_and_rows(sign, conj_half):
    rng = np.random.default_rng(10 + sign + 2 * conj_half)
    P, Pc = random_coefs(rng, sign, conj_half=conj_half)
    x = random_slots(rng)
    X = passkernel.lane_array(x, T)
    want = passkernel.slot_vector(passkernel.direct(X, P, Pc, sign))
    got = run_bsgs(x, P, Pc, sign, rule_slot_roll(sign))
    assert rel_rms(got, want) <= 1e-15
    via_rows = passkernel.rows(passkernel.unpair_lanes(X), P, Pc, sign)
    assert rel_rms(passkernel.slot_vector(passkernel.pair_lanes(via_rows)), want) <= 1e-15


@pytest.mark.parametrize("sign", [CAUSAL, ANTICAUSAL])
@pytest.mark.parametrize("s", [0, 1, 7, 8, 15, 16, 24, 31])
def test_single_offsets_and_the_wrap_lanes(sign, s):
    """One offset at a time (g = 0 and 3, b = 0 and 7): the term reads lane j + sign s inside the channel only, so the
    lanes next to a channel boundary (j = 0, t - 1) read nothing across it."""
    rng = np.random.default_rng(100 + s)
    P = np.zeros((T, T, D), dtype=np.complex128)
    Pc = np.zeros_like(P)
    j = np.arange(T)
    ok = ((j + sign * s >= 0) & (j + sign * s < T)).astype(np.float64)
    P[s] = (rng.standard_normal((T, D)) + 1j * rng.standard_normal((T, D))) * ok[:, None]
    Pc[s] = (rng.standard_normal((T, D)) + 1j * rng.standard_normal((T, D))) * ok[:, None]
    x = random_slots(rng)
    X = passkernel.lane_array(x, T)
    want = passkernel.direct(X, P, Pc, sign)
    # by hand: lane j of channel i reads lane j + sign s of channel i
    hand = np.zeros_like(X)
    for jj in range(T):
        src = jj + sign * s
        if 0 <= src < T:
            hand[jj] = P[s, jj] * X[src] + Pc[s, jj] * np.conj(X[src])
    assert rel(want, hand) <= 1e-15
    got = passkernel.lane_array(run_bsgs(x, P, Pc, sign, rule_slot_roll(sign)), T)
    assert rel(got, hand) <= 1e-15


@pytest.mark.parametrize(
    ("t", "split"), [(16, (4, 4)), (32, (8, 4)), (64, (8, 8)), (128, (16, 8)), (12, (4, 3))]
)
def test_bsgs_split_generalizes(t, split):
    """``NB NG = t``: the lane-mix cut (8 x 4) at t = 32, and the same rule elsewhere, in both directions."""
    assert passkernel.bsgs_split(t) == split
    rng = np.random.default_rng(t)
    for sign in (CAUSAL, ANTICAUSAL):
        P, Pc = random_coefs(rng, sign, t=t, d=2)
        x = random_slots(rng, t * 2)
        want = passkernel.slot_vector(passkernel.direct(passkernel.lane_array(x, t), P, Pc, sign))
        got = run_bsgs(x, P, Pc, sign, rule_slot_roll(sign, t), t)
        assert rel_rms(got, want) <= 1e-15


# ── the ops-protocol form: the same numbers, the census, the level ──────────────────────────────


def fake_ops(steps):
    fhe, inf = fakes.make_runtime_fake(N=N, hid=D, dim=D, H=1, H_real=1, E=D, E_real=D, complex_payload=True)
    fhe._rot = set(steps) | {s - N for s in steps}
    return fhe, FheOps(inf, fake.core, fhe.unit)


@pytest.mark.parametrize(("sign", "conj_half"), [(CAUSAL, True), (ANTICAUSAL, False), (ANTICAUSAL, True)])
def test_lane_kernel_on_the_fake_equals_numpy_and_counts(sign, conj_half):
    kern = LaneKernel(T, sign, conj_half)
    fhe, ops = fake_ops(kern.rotation_steps(N))
    rng = np.random.default_rng(7)
    P, Pc = random_coefs(rng, sign, conj_half=conj_half)
    M = [kern.mask(passkernel.slot_vector(P[s]), s) for s in range(T)]
    Mc = None if Pc is None else [kern.mask(passkernel.slot_vector(Pc[s]), s) for s in range(T)]
    xv = random_slots(rng)
    x = fake.FakeCt(xv.copy(), 36, 2)

    def mult(o, ct, s, half):
        return o.mult_pt(ct, (Mc if half else M)[s])

    y = kern.apply(ops, x, mult)
    nb, ng = passkernel.bsgs_split(T)
    np.testing.assert_array_equal(y.vec, passkernel.bsgs_numpy(xv, M, Mc, sign, nb, ng))
    want = passkernel.slot_vector(passkernel.direct(passkernel.lane_array(xv, T), P, Pc, sign))
    assert rel_rms(y.vec, want) <= 1e-15
    halves = 2 if conj_half else 1
    assert fhe.calls.count("mult_pt") == halves * T == kern.products
    assert sum(c.startswith("rotate") for c in fhe.calls) == halves * (nb - 1) + (ng - 1)
    assert fhe.calls.count("conjugate") == halves - 1
    assert ops.lvl(y) == 36 + 2 + 2  # depth 1: one product above x (x carried a pending rescale)
    assert fhe.n_bootstraps == 0
    steps = kern.rotation_steps(N)
    assert steps == {(sign * k) % N for k in (1, 2, 3, 4, 5, 6, 7, 8, 16, 24)}
    # the numpy ops (the mirror's) run the same kernel to the same numbers
    ny = kern.apply(NumpyOps(), xv.copy(), lambda o, ct, s, half: o.mult_pt(ct, (Mc if half else M)[s]))
    np.testing.assert_array_equal(ny, y.vec)


def test_the_hook_refuses_what_it_must():
    fhe, ops = fake_ops(set())
    x = fake.FakeCt(np.ones(N, dtype=np.complex128), 34, 2)
    with pytest.raises(RuntimeError, match="pass hook"):
        passkernel.passmult(ops, x, "pass/b0/k1/S/P", np.ones(N))
    passkernel.use_pass_hook(ops.inf)
    y = passkernel.passmult(ops, x, "pass/b0/k1/S/P", np.full(N, 2.0))
    np.testing.assert_array_equal(y.vec, 2 * x.vec)
    assert ops.inf.pass_products == 1
    assert fhe.calls == ["mult_pt"]
    with pytest.raises(ValueError, match="#"):
        passkernel.passmult(ops, x, "pass/b0#k1", np.ones(N))
    with pytest.raises(ValueError, match="sign"):
        LaneKernel(T, 0)
