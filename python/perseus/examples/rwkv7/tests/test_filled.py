# SPDX-License-Identifier: MIT
"""The filled state layout (``--layout filled``, FILLED_DESIGN.md): the state's value groups q and q + M share one
ciphertext on its two axes. Light checks on numpy slot vectors (2^15 / 2^14 slots: t = 32 / 16, so M = 1 / 2 filled
ciphertexts against 2 / 4), then the mirror, the fake and the recorder against the current layout on 1-2 blocks."""
import numpy as np
import pytest

from perseus.examples.rwkv7.layout import D, HEAD, Layout
from perseus.impl.ops import NumpyOps

H = 12


class CountOps(NumpyOps):
    """NumpyOps that counts key switches (rotations by step, conjugations) and products."""

    def __init__(self):
        self.rot, self.conj, self.mult_n, self.pt = {}, 0, 0, 0

    def rotate(self, a, k):
        self.rot[int(k)] = self.rot.get(int(k), 0) + 1
        return super().rotate(a, k)

    def rotate_and_sum(self, a, start, stop):
        gap, sign = abs(int(start)), (1 if start > 0 else -1)
        while gap < stop:
            self.rot[sign * gap] = self.rot.get(sign * gap, 0) + 1
            gap *= 2
        return super().rotate_and_sum(a, start, stop)

    def conjugate(self, a):
        self.conj += 1
        return super().conjugate(a)

    def mult(self, a, b):
        self.mult_n += 1
        return super().mult(a, b)

    def mult_pt(self, a, v, key=None, kind="weight", tagged=True):
        self.pt += 1
        return super().mult_pt(a, v)

    def mult_pt_sum(self, cts, vecs):
        self.pt += len(cts)
        return super().mult_pt_sum(cts, vecs)

    @property
    def n_rot(self):
        return sum(self.rot.values())


@pytest.mark.parametrize("n", [1 << 15, 1 << 14])
def test_filled_state_kernels(n):
    """transpose, the state products, untranspose: the filled layout gives the same matrices as the current one."""
    Lc, Lf = Layout(n), Layout(n, layout="filled")
    assert (Lf.Mv, Lf.M) == (Lc.M, Lc.M // 2)
    rng = np.random.default_rng(0)
    S = rng.normal(size=(H, HEAD, HEAD))
    v, a, r = (rng.normal(size=H * HEAD) for _ in range(3))
    Ss = Lf.state_slots(S)
    assert len(Ss) == Lf.M and np.iscomplexobj(Ss[0])
    assert np.allclose(Lf.state_from_slots(Ss, H), S)
    ops = NumpyOps()
    V = Lf.transpose(ops, Lf.natural(v))
    want_V = np.broadcast_to(v.reshape(H, HEAD)[:, :, None], (H, HEAD, HEAD))
    assert np.allclose(Lf.state_from_slots(V, H), want_V)
    Sa = [Lf.block_allreduce(ops, ops.mult(s, Lf.natural(a))) for s in Ss]
    want_Sa = np.einsum("hij,hj->hi", S, a.reshape(H, HEAD))
    assert np.allclose(Lf.state_from_slots(Sa, H), np.broadcast_to(want_Sa[:, :, None], S.shape))
    yf = Lf.untranspose(ops, [ops.mult(s, Lf.natural(r)) for s in Ss])
    yc = Lc.untranspose(ops, [ops.mult(s, Lc.natural(r)) for s in Lc.state_slots(S)])
    want_y = np.einsum("hij,hj->hi", S, r.reshape(H, HEAD)).reshape(-1)
    got = yf.reshape(D, Lf.t)
    assert np.abs(np.imag(got)).max() < 1e-12            # the readout is back on the real axis
    assert np.allclose(np.real(got[: H * HEAD]), want_y[:, None])
    assert np.allclose(got[H * HEAD:], 0.0)
    assert np.abs(yf - yc).max() < 1e-11


@pytest.mark.parametrize("n", [1 << 15, 1 << 14])
def test_filled_kernel_counts_and_steps(n):
    """The op counts per head block (the cost table's per-conversation column) and the rotation keys: every step the
    filled kernels issue is in ``rotation_steps()``, which is a subset of the current layout's."""
    rng = np.random.default_rng(1)
    S = rng.normal(size=(H, HEAD, HEAD))
    x = rng.normal(size=H * HEAD)
    got = {}
    for lay in ("current", "filled"):
        L, ops = Layout(n, layout=lay), CountOps()
        Ss = L.state_slots(S)
        V = L.transpose(ops, L.natural(x))
        Ps = []
        for s, Vm in zip(Ss, V):
            Sa = L.block_allreduce(ops, ops.mult(s, L.natural(x)), -1.0)
            Sn = ops.add(ops.add(ops.mult(s, L.natural(x)), ops.mult(Sa, L.natural(x))), ops.mult(Vm, L.natural(x)))
            Ps.append(ops.mult(Sn, L.natural(x)))
        L.untranspose_lane0(ops, Ps)
        steps = {k % L.n for k in ops.rot}
        assert steps <= L.rotation_steps(), sorted(steps - L.rotation_steps())
        got[lay] = (ops.n_rot, ops.conj, ops.mult_n, ops.pt, L.rotation_steps())
    (rc, cc, mc, pc, kc), (rf, cf, mf, pf, kf) = got["current"], got["filled"]
    assert kf <= kc
    t = Layout(n).t
    if t == 32:
        # rotations: current transpose 26, state step 2 x 12, untranspose 33 = 83; filled 17 + 12 + 17 = 46, plus one
        # conjugation (FILLED_DESIGN.md §1)
        assert (rc, rf) == (83, 46), (rc, rf)
    assert cf == cc + Layout(n, layout="filled").M
    assert mf * 2 == mc
    assert pf * 2 == pc
    print(f"t={t}: key switches {rc + cc} -> {rf + cf}, products {mc} -> {mf}, plaintext products {pc} -> {pf}, "
          f"keys {len(kc)} -> {len(kf)}")


def test_filled_layout_rejects_odd_groups():
    with pytest.raises(ValueError):
        Layout(1 << 16, D=1024, layout="filled")          # t = 64: one value group
    with pytest.raises(ValueError):
        Layout(1 << 15, layout="half")


# ── the model: filled against current on 2 blocks of the real checkpoint ──

from perseus.examples.rwkv7 import common  # noqa: E402
from perseus.examples.rwkv7.batched import BatchedMirror  # noqa: E402
from perseus.examples.rwkv7.model import Mirror, load_coeffs  # noqa: E402
from perseus.examples.rwkv7.session import make_model  # noqa: E402
from perseus.examples.rwkv7.tests.test_mirror import N_LAYERS, params2, rel_rms  # noqa: E402,F401
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer  # noqa: E402

TEXT = "\n\nThe Eiffel tower is in Paris, and the river"


def _state_err(La, sa, Lb, sb, H):
    """Max abs difference of two blocks' carried states, the state matrices decoded from each layout."""
    e = max(float(np.abs(La.state_from_slots_sh(sa.S, H) - Lb.state_from_slots_sh(sb.S, H)).max()), 0.0)
    for x, y in zip([*sa.xa, *sa.xf], [*sb.xa, *sb.xf]):
        e = max(e, float(np.abs(np.asarray(x) - np.asarray(y)).max()))
    return e


@need
def test_mirror_filled_equals_current(params2):  # noqa: F811
    """Decode, a prefill pass (T = 5 and T = 32) and a batched pass (K = 3 and 32): the filled mirror gives the same
    hidden states and the same carried state (decoded to matrices) as the current one, exact matrices."""
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode(TEXT)
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    H = params2.C // HEAD
    n = 1 << 15
    a, b = (Mirror(params2, co, n, wbits=None, layout=x) for x in ("current", "filled"))
    assert (a.L.M, b.L.M) == (2, 1)
    for m in (a, b):
        m.reset(st)
    errs = []
    for t in ids[1:5]:
        ha, _ = a.step(params2.emb[t])
        hb, zb = b.step(params2.emb[t])
        errs.append(rel_rms(hb, ha))
    se = max(_state_err(a.L, x, b.L, y, H) for x, y in zip(a.state, b.state))
    assert np.iscomplexobj(b.state[0].S[0]) and np.abs(np.imag(b.state[0].S[0])).max() > 1e-3   # both axes carry
    for T in (5, 32):
        rows = [params2.emb[t] for t in (ids * 8)[1:1 + T]]
        pa, _ = a.prefill(rows)
        pb, _ = b.prefill(rows)
        errs.append(rel_rms(pb, pa))
    se = max(se, max(_state_err(a.L, x, b.L, y, H) for x, y in zip(a.state, b.state)))
    for K in (3, 32):
        ba, bb = (BatchedMirror(params2, co, n, K, carry="lane", wbits=None, layout=x) for x in ("current", "filled"))
        ba.reset(st)
        bb.reset(st)
        for i in range(2):
            rows = np.stack([params2.emb[(ids * 8)[1 + c + i]] for c in range(K)])
            ya, _ = ba.step(rows)
            yb, _ = bb.step(rows)
            errs.append(rel_rms(yb, ya))
        for c in (0, K - 1):
            se = max(se, max(_state_err(a.L, x, b.L, y, H) for x, y in zip(ba.conv_state(c), bb.conv_state(c))))
    print(f"filled vs current mirror: hidden rel max {max(errs):.1e}, state abs max {se:.1e}")
    assert max(errs) < 1e-11 and se < 1e-10
    # non-vacuity: a filled state with the axes swapped gives a different token
    bad = Mirror(params2, co, n, wbits=None, layout="filled")
    bad.reset(st)
    for s in bad.state:
        s.S = [np.imag(x) + 1j * np.real(x) for x in s.S]
    hbad, _ = bad.step(params2.emb[ids[1]])
    a.reset(st)
    hgood, _ = a.step(params2.emb[ids[1]])
    assert rel_rms(hbad, hgood) > 1e-3


@need
@pytest.mark.parametrize("capture", [False, True])
def test_fake_filled(params2, capture):  # noqa: F811
    """The fake (runtime level rules) and the capture recorder in the filled layout against the filled mirror and
    against the current-layout fake: same values, one op stream per token, one carry refresh fewer per block."""
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode(TEXT)
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    res = {}
    for lay in ("current", "filled"):
        fhe, _inf, model = make_model(params2, co, capture=capture, n_layers=N_LAYERS, layout=lay)
        if capture:
            fhe.magnitude_suppressed = True
        mir = Mirror(params2, co, model.L.n, n_layers=N_LAYERS, wbits=24, layout=lay)
        model.start(st)
        mir.reset(st)
        hs, errs, counts = [], [], []
        for pos, t in enumerate(ids[1:5]):
            b0 = getattr(fhe, "n_bootstraps", 0)
            h, out = model.step(params2.emb[t], 1 + pos)
            hm, _ = mir.step(params2.emb[t])
            hs.append(h)
            errs.append(rel_rms(h, hm))
            counts.append(getattr(fhe, "n_bootstraps", 0) - b0)
        res[lay] = (hs, errs, counts)
        print(lay, "capture" if capture else "fake", f"vs mirror max {max(errs):.1e}, refreshes per token {counts}")
        assert max(errs) < 1e-9
        assert len(set(counts)) == 1
    d = max(rel_rms(x, y) for x, y in zip(res["filled"][0], res["current"][0]))
    print(f"filled vs current fake: hidden rel max {d:.1e}")
    assert d < 1e-9
    if not capture:
        assert res["current"][2][0] - res["filled"][2][0] >= N_LAYERS     # >= one carry refresh fewer per block


@need
def test_filled_capture_and_plan(tmp_path, monkeypatch):
    """A 1-block batched capture (K = 4) in the filled layout: one state ciphertext per conversation (``s0c<c>``, no
    ``s1c<c>``), complex rows for the public state; with --rwkv7-plan, the b2 plan cuts it with one carry refresh per
    conversation and fewer refreshes than the current layout's plan on the same capture recipe."""
    import json
    import os
    import sys

    from perseus.examples.rwkv7.tests.conftest import ROOT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_prefill import _cap_and_plan
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    if os.environ.get("RWKV7_PLAN") != "1":
        from perseus.examples.rwkv7 import run_capture
        from perseus.examples.rwkv7.tests.conftest import CKPT
        from perseus.examples.rwkv7.tests.test_ref import HARNESS
        cap = tmp_path / "cap_f4"
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                                 "--batch", "4", "--layout", "filled", "--out", str(cap)]) == 0
    else:
        from perseus.examples.rwkv7.tests.test_batched import _planopt_env
        (kv,) = _planopt_env("b2")
        monkeypatch.setenv(*kv.split("=", 1))
        cap, plan = _cap_and_plan(tmp_path, "f4", ["--batch", "4", "--layout", "filled"], HIGH)
        capc, planc = _cap_and_plan(tmp_path, "c4", ["--batch", "4"], HIGH)
        sys.path.insert(0, str(ROOT / "tools" / "rwkv7"))
        import b2_census
        f, c = b2_census.census(str(cap), str(plan))["blocks"][0], b2_census.census(str(capc), str(planc))["blocks"][0]
        print(f"K=4 b2, 1 block: filled {f['total']} (carry {f['deliberate_by_scope'].get('carry')}), current "
              f"{c['total']} (carry {c['deliberate_by_scope'].get('carry')})")
        assert f["deliberate_by_scope"]["carry"] == 4 + 2 and c["deliberate_by_scope"]["carry"] == 2 * 4 + 2
        assert f["placed_per_conv_scopes"] == 0 and f["total"] < c["total"]
    meta = json.load(open(cap / "rwkv7_capture.json"))
    assert meta["layout"] == "filled" and meta["state_names"] == ["s0c0", "s0c1", "s0c2", "s0c3", "xa", "xf"]
    g = json.load(open(cap / "block_0" / "graph.json"))
    names = {x for n in g["nodes"] for x in n["inputs"]}
    assert all(any(f"cache.s0c{c}-lvl" in x for x in names) for c in range(4))
    assert not any("cache.s1c" in x for x in names)


@need
def test_filled_prefill_scopes_escape_b2(tmp_path, monkeypatch):
    """A 1-block filled prefill capture (T = 4) labels its transforms and readout pf_xpose / pf_chain / pf_readout, and
    no step of it contains one of PLANOPT=b2's forbidden substrings (review lens 1, finding 1: they were
    tm.state.readout & co., so b2_plan.sh on a filled prefill capture would have forbidden its readout refreshes)."""
    import json

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_batched import _planopt_env
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_ref import HARNESS
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    cap = tmp_path / "cap_pf4_filled"
    assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                             "--prefill-chunk", "4", "--layout", "filled", "--out", str(cap)]) == 0
    (kv,) = _planopt_env("b2")
    b2 = [x for x in kv.split("=", 1)[1].split(",") if x.startswith("tm.state")]
    steps = {n.get("step") or "" for n in json.load(open(cap / "block_0" / "graph.json"))["nodes"]}
    for lab in ("pf_xpose", "pf_chain", "pf_readout"):
        assert any(s.endswith(f"tm.state.{lab}") for s in steps), lab
    hit = sorted({s for s in steps for f in b2 if f in s})
    assert not hit, hit


@need
@pytest.mark.parametrize("mode,capture", [("prefill", False), ("batched", False), ("batched", True)])
def test_fake_filled_passes(params2, mode, capture):  # noqa: F811
    """A prefill pass (T = 4) and batched passes (K = 3) on the fake / recorder: the filled layout gives the current
    layout's hidden states (2 passes each, then a decode step from the carried state), with fewer refreshes."""
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode(TEXT)
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    res = {}
    for lay in ("current", "filled"):
        kw = {"chunk": 4} if mode == "prefill" else {"batch": 3, "batch_carry": "lane"}
        fhe, _inf, model = make_model(params2, co, capture=capture, n_layers=N_LAYERS, layout=lay, **kw)
        if capture:
            fhe.magnitude_suppressed = True
        model.start(st)
        hs, counts = [], []
        for i in range(2):
            rows = ([params2.emb[t] for t in ids[1 + 4 * i:5 + 4 * i]] if mode == "prefill" else
                    np.stack([params2.emb[ids[1 + c + i]] for c in range(3)]))
            b0 = getattr(fhe, "n_bootstraps", 0)
            h, _o = model.step(rows, 1 + i)
            hs.append(h)
            counts.append(getattr(fhe, "n_bootstraps", 0) - b0)
        if mode == "prefill":
            model.set_chunk(0)
            h, _o = model.step(params2.emb[ids[9]], 3)
            hs.append(h[None])
        res[lay] = (hs, counts)
        print(mode, lay, "capture" if capture else "fake", "refreshes per pass", counts)
    d = max(rel_rms(x, y) for x, y in zip(res["filled"][0], res["current"][0]))
    print(f"{mode}: filled vs current hidden rel max {d:.1e}")
    assert d < 1e-9
    if not capture:
        assert res["filled"][1][-1] < res["current"][1][-1]
