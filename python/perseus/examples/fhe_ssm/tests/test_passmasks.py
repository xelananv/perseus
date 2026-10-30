# SPDX-License-Identifier: MIT
"""The pass configuration (passmasks.py): every mask against a per-lane-slot construction written here from the
definitions (the scan's segmented recurrence, the shift's lane read, the carry prep's broadcast), applied on slot
vectors through the slot form and on lanes through the direct form; today's tick as the masks' identity case; the
refusal rules across passes; the JSON form; the scenarios of the record (prefill:320, doc:258:64, mixed, churn, tick);
the lane-slot order."""

import json

import numpy as np
import pytest

from perseus.examples.fhe_ssm import filling, passkernel, passmasks
from perseus.examples.fhe_ssm.passkernel import ANTICAUSAL, CAUSAL
from perseus.examples.fhe_ssm.passmasks import GENERATED, PassConfig, Segment, Tracker

T, D = 32, 4
NQ = 2 * T
N = T * D


class Decay:
    def __init__(self, a, mix):
        self.a, self.mix = a, mix


def decay(seed=0, d=D):
    rng = np.random.default_rng(seed)
    return Decay(rng.uniform(0.9088, 0.9918, d), rng.uniform(0.3, 0.9, d))


def conv_all(entry="carry", n=NQ):
    return [Segment.conv(q, entry, f"c{q}") for q in range(n)]


def cfg_of(segments, tokens=None, n=NQ, name=""):
    segs = tuple(segments)
    toks = [0] * n if tokens is None else list(tokens)
    for s in segs:  # copies for padding and parked lane-slots (validate checks them)
        for q in s.padding:
            toks[q] = toks[s.q0]
    for s in segs:
        if s.entry == "parked":
            toks[s.q0] = toks[s.parked_from]
    return PassConfig(segs, tuple(toks), name)


#: configurations for the mask checks: the edge cases the scan, the shift and the carry prep define
CASES = {
    "tick": cfg_of(conv_all()),
    "tick fresh": cfg_of(conv_all("fresh")),
    "seq fresh [0,64)": cfg_of([Segment.seq(0, 64, 0, "fresh", "L")]),
    "seq carry [0,64)": cfg_of([Segment.seq(0, 64, 0, "carry", "L")]),
    "seq carry q0 1 (1 padding)": cfg_of([Segment.seq(0, 64, 1, "carry", "L")]),
    "seq carry q0 62 (62 padding)": cfg_of([Segment.seq(0, 64, 62, "carry", "L")]),
    "two adjacent seqs": cfg_of([Segment.seq(0, 32, 0, "fresh", "A"), Segment.seq(32, 64, 33, "carry", "B")]),
    "home ending at lane 15 + conv": cfg_of(
        [
            Segment.seq(10, 32, 13, "carry", "A"),
            *(Segment.conv(q, "carry", f"c{q}") for q in [*range(10), *range(32, 64)]),
        ]
    ),
    "a lane with one axis carried": cfg_of(
        [
            *(Segment.conv(q, "carry", f"c{q}") for q in range(40)),
            Segment.conv(40, "carry", "x"),
            Segment.conv(41, "fresh", "y"),
            Segment.seq(42, 64, 45, "carry", "B"),
        ]
    ),
    "parked": cfg_of(
        [Segment.conv(0, "carry", "c0"), *(Segment.conv(q, "parked", parked_from=0) for q in range(1, 64))]
    ),
}


def roles_here(cfg):
    """The lane-slot roles from the segments, written here (not passmasks.PassConfig.roles)."""
    out = {}
    for s in cfg.segments:
        if s.kind == "seq":
            for q in range(s.h0, s.q0):
                out[q] = ("pad", s)
            for q in range(s.q0, s.h1):
                out[q] = ("pos", s)
        else:
            out[s.q0] = ("parked" if s.entry == "parked" else "conv", s)
    return out


def scan_matrix(cfg, a):
    """``(d, n, n)``: S[q] = sum_q' M[q, q'] x[q'] (the trainer's recurrence unrolled over a chunk), and ``(d, n)``
    the carried state's weight per lane-slot."""
    n = cfg.lane_slots
    M = np.zeros((a.shape[0], n, n))
    k = np.zeros((a.shape[0], n))
    for q, (role, s) in roles_here(cfg).items():
        if role == "pos":
            for qq in range(s.q0, q + 1):
                M[:, q, qq] = (1 - a) * a ** (q - qq)
            if s.entry == "carry":
                k[:, q] = a ** (q - s.q0 + 1)
        else:
            M[:, q, q] = 1 - a
            if role == "conv" and s.entry == "carry":
                k[:, q] = a
    return M, k


def shift_want(cfg, mix, n_q, cn_q):
    """um[q] = mix n[q] + (1 - mix) n_prev[q], n_prev from the previous position, the carried register or zero."""
    out = np.zeros_like(n_q)
    for q, (role, s) in roles_here(cfg).items():
        if role == "pos" and q > s.q0:
            prev = n_q[q - 1]
        elif (role == "pos" and q == s.q0 and s.entry == "carry") or (role == "conv" and s.entry == "carry"):
            prev = cn_q[q]
        else:
            prev = 0.0
        out[q] = mix * n_q[q] + (1 - mix) * prev
    return out


def carry_want(cfg, s_q, n_q):
    """After the carry prep: every lane-slot of a seq home holds the segment's last position's S and n; a
    conversation (and a parked copy) its own."""
    cs, cn = s_q.copy(), n_q.copy()
    for s in cfg.segments:
        if s.kind == "seq":
            cs[s.h0 : s.h1] = s_q[s.h1 - 1]
            cn[s.h0 : s.h1] = n_q[s.h1 - 1]
    return cs, cn


def slots_of(rows_q):
    return passkernel.slot_vector(passkernel.pair_lanes(rows_q))


def rows_of(v):
    return passkernel.unpair_lanes(passkernel.lane_array(v, T))


def apply_masks(mv, b, x_q, c_q, n_q, cn_q):
    """The block's three pass terms on numpy slot vectors with the masks of ``passmasks.masks`` (the slot form)."""
    nb, ng = passkernel.bsgs_split(T)

    def m(name):
        return mv[passmasks.mask_id(name, b)]

    x, c, nn, cn = slots_of(x_q), slots_of(c_q), slots_of(n_q), slots_of(cn_q)
    M = [m(f"k1/P/{s}") for s in range(T)]
    Mc = [m(f"k1/Pc/{s}") for s in range(T)]
    s_v = passkernel.bsgs_numpy(x, M, Mc, CAUSAL, nb, ng) + c * m("k1/S/P") + np.conj(c) * m("k1/S/Pc")
    r = np.roll(nn, 1)  # rot(n, -1)
    um = (
        nn * m("sh/mixU0") + np.conj(nn) * m("sh/U0c") + r * m("sh/U1") + np.conj(r) * m("sh/U1c")
        + cn * m("sh/V") + np.conj(cn) * m("sh/Vc")
    )  # fmt: skip
    y2 = 1j * (np.conj(s_v) - s_v) + (nn - np.conj(nn))
    z = passkernel.bsgs_numpy(y2, [m(f"k2/R/{s}") for s in range(T)], None, ANTICAUSAL, nb, ng)
    w = z + np.conj(z)
    v = 1j * (np.conj(z) - z)
    c_next = s_v * m("k2/Po") + w + 1j * w
    cn_next = nn * m("k2/Po") + v + 1j * v
    return rows_of(s_v), rows_of(um), rows_of(c_next), rows_of(cn_next)


def rel(got, want):
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


# ── the masks against the definitions ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", sorted(CASES))
def test_masks_realize_the_lane_slot_maps(case):
    cfg = CASES[case]
    passmasks.validate(cfg)
    bp = decay(len(case))
    rng = np.random.default_rng(3)
    x_q, c_q, n_q, cn_q = (rng.standard_normal((NQ, D)) for _ in range(4))
    mv = passmasks.masks(cfg, bp, b=5)
    s_q, um_q, c_next, cn_next = apply_masks(mv, 5, x_q, c_q, n_q, cn_q)
    M, k = scan_matrix(cfg, bp.a)
    want_s = np.einsum("iqr,ri->qi", M, x_q) + k.T * c_q
    assert rel(s_q, want_s) <= 1e-14
    assert rel(um_q, shift_want(cfg, bp.mix, n_q, cn_q)) <= 1e-14
    ws, wn = carry_want(cfg, want_s, n_q)
    assert rel(c_next, ws) <= 1e-14
    assert rel(cn_next, wn) <= 1e-14
    # the lane form (the mirror's): the same maps from the target-indexed coefficients, offset by offset
    co = passmasks.coefficients(cfg, bp.a, bp.mix)
    S = passkernel.direct(passkernel.pair_lanes(x_q), co["k1/P"], co["k1/Pc"], CAUSAL)
    C = passkernel.pair_lanes(c_q)
    S = S + co["k1/S/P"] * C + co["k1/S/Pc"] * np.conj(C)
    assert rel(passkernel.unpair_lanes(S), want_s) <= 1e-14


def test_mask_ids_are_fixed_and_shared_where_block_independent():
    ids = {name: set(passmasks.masks(cfg, decay(1), b=3)) for name, cfg in CASES.items()}
    first = next(iter(ids.values()))
    assert all(v == first for v in ids.values())  # never a function of the configuration
    assert len(first) == 6 + 2 * T + 2 + T + 1 == 105
    assert not any("#" in i for i in first)
    k2 = {i for i in first if i.startswith("pass/k2/")}
    assert k2 == {f"pass/k2/R/{s}" for s in range(T)} | {"pass/k2/Po"}
    assert {i for i in first if i.startswith("pass/b3/")} == first - k2
    # the carry prep's masks depend on the partition only: equal across blocks and decays
    ref_k2 = passmasks.masks(CASES["two adjacent seqs"], decay(9), b=1)
    for b in (0, 7):
        got = passmasks.masks(CASES["two adjacent seqs"], decay(b), b=b)
        for i in k2:
            np.testing.assert_array_equal(got[i], ref_k2[i])


def test_the_tick_is_the_identity_case():
    """Every lane-slot ``conv carry``: P_0 = 1 - a, P_{s >= 1} = P' = 0, P^c = a, P'^c = 0, U = 0, V = 1 - mix,
    V' = 0, R = 0, P_o = 1: today's ``a (.) s + (1 - a) (.) x`` and ``mix (.) n + (1 - mix) (.) n_prev``."""
    bp = decay(4)
    co = passmasks.coefficients(CASES["tick"], bp.a, bp.mix)
    lanes = np.ones((T, 1))
    np.testing.assert_array_equal(co["k1/P"][0], (1 - bp.a) * lanes + 0j)
    assert not co["k1/P"][1:].any()
    assert not co["k1/Pc"].any()
    np.testing.assert_array_equal(co["k1/S/P"], bp.a * lanes)
    assert not co["k1/S/Pc"].any()
    np.testing.assert_array_equal(co["sh/mixU0"], bp.mix * lanes + 0j)
    for name in ("sh/U0c", "sh/U1", "sh/U1c", "sh/Vc"):
        assert not co[name].any()
    np.testing.assert_array_equal(co["sh/V"], (1 - bp.mix) * lanes)
    assert not co["k2/R"].any()
    np.testing.assert_array_equal(co["k2/Po"], np.ones(T))
    mv = passmasks.masks(CASES["tick"], bp, b=0)
    np.testing.assert_array_equal(mv["pass/b0/k1/P/0"], filling.lane_const(1 - bp.a, N) + 0j)
    np.testing.assert_array_equal(mv["pass/b0/k1/S/P"], filling.lane_const(bp.a, N))
    np.testing.assert_array_equal(mv["pass/b0/sh/V"], filling.lane_const(1 - bp.mix, N))
    np.testing.assert_array_equal(mv["pass/k2/Po"], np.ones(N))
    fresh = passmasks.coefficients(CASES["tick fresh"], bp.a, bp.mix)
    assert not fresh["k1/S/P"].any()  # a fresh lane-slot is tick 0: S = (1 - a) x, um = mix n
    assert not fresh["sh/V"].any()


def test_the_conj_halves_and_the_one_axis_lane():
    """A lane with one axis reading the carried register and the other not: V' != 0 (the shift's third conj); a
    seq lane: P' != 0 at offsets >= 0 (the scan's cross-axis terms); a copy map: R' never exists."""
    bp = decay(2)
    co = passmasks.coefficients(CASES["a lane with one axis carried"], bp.a, bp.mix)
    assert np.abs(co["sh/Vc"][20]).max() > 0  # lane 20 = lane-slots 40 (carry) and 41 (fresh)
    # lane 22 = lane-slots 44 (padding) and 45 (B's carried first position)
    assert np.abs(co["sh/Vc"][22]).max() > 0
    seq = passmasks.coefficients(CASES["seq fresh [0,64)"], bp.a, bp.mix)
    # offset 0 inside a sequence: the real axis (position 2j) never reads the imaginary one (the later position 2j + 1,
    # c_eo = 0), the imaginary reads the real (c_oe = (1 - a) a): Im P_0 = Im P'_0 = c_oe / 2, Re P'_0 = 0
    np.testing.assert_array_equal(seq["k1/P"][0].imag, seq["k1/Pc"][0].imag)
    np.testing.assert_allclose(
        seq["k1/P"][0].imag, 0.5 * (1 - bp.a) * bp.a * np.ones((T, 1)), rtol=1e-15, atol=0
    )
    assert not seq["k1/Pc"][0].real.any()
    assert np.abs(seq["k1/Pc"][0]).max() > 0
    assert np.abs(seq["k1/Pc"][31]).max() > 0
    assert not seq["k1/P"][1:, 0].any()  # lane 0 has no lane below it


# ── the rules across passes ────────────────────────────────────────────────────────────────────


def run_tracker(cfgs):
    tr = Tracker(cfgs[0].lane_slots)
    out = []
    for cfg in cfgs:
        out.append(tr.resolve(cfg))
        tr.commit(cfg)
    return out


@pytest.mark.parametrize(
    ("segments", "match"),
    [
        ([Segment.seq(1, 64, 1, "fresh", "L"), Segment.conv(0, "fresh", "c")], "even bounds"),
        ([Segment.seq(0, 63, 0, "fresh", "L"), Segment.conv(63, "fresh", "c")], "even bounds"),
        ([Segment.seq(0, 64, 0, "fresh", "L"), Segment.conv(63, "fresh", "c")], "overlap"),
        ([Segment.seq(0, 62, 0, "fresh", "L")], "no segment"),
        ([Segment.seq(0, 32, 0, "fresh", "L"), Segment.seq(32, 64, 32, "fresh", "L")], "occupies"),
        ([Segment.seq(0, 64, 0, "sometimes", "L")], "entry"),
        ([*conv_all()[:63], Segment.conv(63, "parked", parked_from=63)], "not live"),
        ([Segment.seq(0, 62, 0, "fresh", "L"), Segment.conv(62, "parked", parked_from=60), Segment.conv(63, "parked", parked_from=62)], "not live"),
    ],
)  # fmt: skip
def test_malformed_passes_are_refused(segments, match):
    with pytest.raises(ValueError, match=match):
        passmasks.validate(PassConfig(tuple(segments), tuple([0] * NQ)))


def test_copies_must_hold_their_sources_token():
    good = cfg_of([Segment.seq(0, 64, 3, "fresh", "L")], tokens=list(range(64)))
    passmasks.validate(good)
    bad = PassConfig(good.segments, tuple(range(64)))
    with pytest.raises(ValueError, match="padding"):
        passmasks.validate(bad)
    with pytest.raises(ValueError, match="GENERATED"):
        passmasks.validate(PassConfig(good.segments, tuple([GENERATED] * 64)))


def test_the_registers_rules_across_passes():
    seq = cfg_of([Segment.seq(0, 64, 0, "fresh", "L")])
    seq_c = cfg_of([Segment.seq(0, 64, 0, "carry", "L")])
    # carry on a register nothing wrote
    with pytest.raises(ValueError, match="no pass wrote"):
        run_tracker([cfg_of(conv_all())])
    with pytest.raises(ValueError, match="no previous pass"):
        run_tracker([seq_c])
    # a continuing seq whose home is not inside its previous home (a home only shrinks)
    narrow = cfg_of(
        [Segment.seq(32, 64, 32, "fresh", "L"), *(Segment.conv(q, "fresh", f"c{q}") for q in range(32))]
    )
    wide = cfg_of([Segment.seq(0, 64, 0, "carry", "L")])
    with pytest.raises(ValueError, match="not inside its previous home"):
        run_tracker([narrow, wide])
    # carry on a register another stream wrote
    other = cfg_of([*conv_all("fresh")[:5], Segment.conv(5, "carry", "intruder"), *conv_all("fresh")[6:]])
    with pytest.raises(ValueError, match="stream 'c5' wrote"):
        run_tracker([cfg_of(conv_all("fresh")), other])
    # carry on a register a parked copy wrote
    parked = cfg_of(
        [Segment.conv(0, "fresh", "c0"), *(Segment.conv(q, "parked", parked_from=0) for q in range(1, 64))]
    )
    after = cfg_of([Segment.conv(0, "carry", "c0"), Segment.conv(1, "carry", "c1"),
                    *(Segment.conv(q, "parked", parked_from=0) for q in range(2, 64))])  # fmt: skip
    with pytest.raises(ValueError, match="parked copy"):
        run_tracker([parked, after])
    # carry on a register a still-continuing seq wrote
    half = cfg_of(
        [Segment.seq(32, 64, 32, "carry", "L"), *(Segment.conv(q, "carry", f"r{q}") for q in range(32))]
    )
    with pytest.raises(ValueError, match="still continues"):
        run_tracker([seq, half])
    # a conversation's state cannot be handed to a multi-lane seq segment
    with pytest.raises(ValueError, match="no previous pass"):
        run_tracker([cfg_of(conv_all("fresh")), seq_c])
    # the hand-over: a finished seq's home lanes (padding included) turn conv carry, each reading its final state
    pad = cfg_of([Segment.seq(0, 64, 62, "carry", "L")])
    readers = cfg_of([Segment.conv(q, "carry", f"R{q}") for q in range(64)])
    src = run_tracker([seq, pad, readers, readers])
    assert src[2] == ["L"] * 64
    assert src[3] == [f"R{q}" for q in range(64)]
    # the shrink is allowed, and the same stream may continue as a conversation
    shrink = cfg_of(
        [Segment.seq(32, 64, 32, "carry", "L"), *(Segment.conv(q, "fresh", f"c{q}") for q in range(32))]
    )
    gen = cfg_of(
        [*(Segment.conv(q, "parked", parked_from=63) for q in range(63)), Segment.conv(63, "carry", "L")]
    )
    assert run_tracker([seq, shrink, gen])[2][-1] == "L"


def test_json_round_trip(tmp_path):
    cfgs = passmasks.scenario("mixed", vocab=64, seed=1)
    path = tmp_path / "passes.json"
    path.write_text(json.dumps({"passes": passmasks.dump_list(cfgs)}))
    assert passmasks.load_list(str(path)) == cfgs
    assert passmasks.load_list(json.loads(json.dumps(passmasks.dump_list(cfgs)))) == cfgs
    with pytest.raises(ValueError, match="segment has"):
        passmasks.Segment.from_json({**cfgs[0].segments[0].to_json(), "width": 3})


# ── the scenarios ──────────────────────────────────────────────────────────────────────────────


def test_prefill_320_as_written():
    cfgs = passmasks.scenario("prefill:320", vocab=64, seed=2)
    assert len(cfgs) == 8
    for p in range(5):
        (seg,) = cfgs[p].segments
        assert (seg.kind, seg.home, seg.q0, seg.stream) == ("seq", (0, 64), 0, "L")
        assert seg.entry == ("fresh" if p == 0 else "carry")
    assert cfgs[2].mask_key() == cfgs[3].mask_key() == cfgs[4].mask_key() == cfgs[1].mask_key()
    bp = decay(6)
    for p in (2, 3, 4):  # nothing to restage: the masks are bit-identical
        for k, v in passmasks.masks(cfgs[p], bp, 0).items():
            np.testing.assert_array_equal(v, passmasks.masks(cfgs[1], bp, 0)[k])
    for p in (5, 6, 7):
        live = [s for s in cfgs[p].segments if s.live]
        assert [(s.q0, s.entry, s.stream) for s in live] == [(63, "carry", "L")]
        assert cfgs[p].tokens == (GENERATED,) * 64
        assert all(s.parked_from == 63 for s in cfgs[p].segments if not s.live)
    assert sum(len(c.live_slots()) for c in cfgs[:5]) == 320


@pytest.mark.parametrize(("T_", "q0s"), [(1, [63]), (2, [62]), (3, [61]), (63, [1]), (64, [0]), (65, [0, 63]),
                                         (258, [0, 0, 0, 0, 62])])  # fmt: skip
def test_chunks_for_any_length(T_, q0s):
    cfgs = passmasks.scenario(f"prefill:{T_}", vocab=64)
    assert [c.segments[0].q0 for c in cfgs[: len(q0s)]] == q0s
    assert sum(len(c.live_slots()) for c in cfgs[: len(q0s)]) == T_
    for c in cfgs[: len(q0s)]:
        assert c.segments[0].home == (0, 64)


def test_doc_258_64_as_written():
    cfgs = passmasks.scenario("doc:258:64", vocab=64, seed=3)
    assert len(cfgs) == 8
    assert [c.segments[0].q0 for c in cfgs[:5]] == [0, 0, 0, 0, 62]
    assert [c.segments[0].entry for c in cfgs[:5]] == ["fresh", "carry", "carry", "carry", "carry"]
    src = passmasks.validate_list(cfgs)
    assert src[5] == ["D"] * 64  # every reader starts from the document's final state
    assert src[6] == [f"R{q}" for q in range(64)]
    assert len({cfgs[5].tokens, cfgs[6].tokens, cfgs[7].tokens}) == 3
    assert cfgs[6].mask_key() == cfgs[7].mask_key()


def test_mixed_as_written():
    cfgs = passmasks.scenario("mixed", vocab=64, seed=4)
    assert len(cfgs) == 5
    seqs = {s.stream: s for s in cfgs[0].segments if s.kind == "seq"}
    assert (seqs["A"].home, seqs["A"].q0, seqs["B"].home, seqs["B"].q0) == ((0, 32), 0, (32, 42), 33)
    convs0 = [s for s in cfgs[0].segments if s.kind == "conv"]
    assert len(convs0) == 22
    assert len(cfgs[0].live_slots()) == 32 + 9 + 22  # 64 lane-slots = 32 + 9 + 1 padding + 22
    fresh1 = sorted(s.q0 for s in cfgs[1].segments if s.kind == "conv" and s.entry == "fresh")
    assert fresh1 == [45, 50, 61]
    src = passmasks.validate_list(cfgs)
    assert src[2][:32] == ["A"] * 32  # the 32 readers' hand-over
    b2 = next(s for s in cfgs[2].segments if s.stream == "B")
    assert (b2.home, b2.q0, b2.entry) == ((34, 42), 34, "carry")
    assert cfgs[2].mask_key() == cfgs[3].mask_key() == cfgs[4].mask_key()


def test_churn_and_tick():
    cfgs = passmasks.scenario("churn", vocab=64, seed=5)
    assert len(cfgs) == 6
    parked = [sum(1 for s in c.segments if s.entry == "parked") for c in cfgs]
    fresh = [sum(1 for s in c.segments if s.entry == "fresh") for c in cfgs]
    assert parked == [0, 4, 2, 1, 1, 1]
    assert fresh == [64, 2, 4, 3, 0, 0]
    assert cfgs[4].mask_key() == cfgs[5].mask_key()
    ticks = passmasks.scenario("tick", vocab=64, conversations=10)
    assert len(ticks) == 6
    assert ticks[1].live_slots() == list(range(10))
    assert all(s.parked_from == 0 for s in ticks[1].segments if not s.live)
    for spec in ("tick", "prefill:320", "doc:258:64", "mixed", "churn"):
        cs = passmasks.scenario(spec, vocab=64)
        steady = [p for p in range(1, len(cs)) if cs[p].mask_key() == cs[p - 1].mask_key()]
        assert steady, spec  # a pass whose configuration equals the previous pass's exists
    with pytest.raises(ValueError, match="scenario"):
        passmasks.scenario("prefill", vocab=64)


def test_generated_tokens_follow_the_reply():
    cfgs = passmasks.scenario("prefill:64", vocab=8)
    head = np.eye(8, 4)
    reply = np.zeros((64, 4))
    reply[63, 2] = 1.0
    got = passmasks.resolve_generated(cfgs[1], reply, head)
    assert got.tokens == (2,) * 64
    with pytest.raises(ValueError, match="previous pass"):
        passmasks.resolve_generated(cfgs[1], None, head)


# ── the lane-slot order ────────────────────────────────────────────────────────────────────────


def test_the_lane_slot_order_is_one_permutation():
    pi = passmasks.layout_index(NQ)
    assert list(pi[:4]) == [0, 32, 1, 33]
    assert pi[63] == 63
    assert sorted(pi) == list(range(NQ))
    rows_q = np.random.default_rng(1).standard_normal((NQ, D))
    rows_r = passmasks.to_layout(rows_q)
    np.testing.assert_array_equal(passmasks.from_layout(rows_r), rows_q)
    # today's packing of the permuted rows puts lane-slot q on lane q div 2, axis q mod 2
    v = filling.pack_pairs(rows_r, N)
    np.testing.assert_array_equal(v, slots_of(rows_q))
    np.testing.assert_array_equal(passmasks.from_layout(filling.unpack_pairs(v, D, NQ)), rows_q)
