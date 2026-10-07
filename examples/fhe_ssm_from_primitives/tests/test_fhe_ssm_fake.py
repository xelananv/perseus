"""The fhe-ssm port on the CPU: the norm against its closed form, every kernel on the numpy
fake against the same kernel on NumpyOps (layout / protocol), the whole model against the
exact forward (Newton approximation), the rotation band, the hint threshold under the stock and
the runtime level rules, and the carry."""

import json

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import block, geometry, norm, ref, runtime_fake, weights
from examples.fhe_ssm_from_primitives.model import FheSsmModel

from perseus.impl import fake
from perseus.impl.layout import Dims
from perseus.impl.ops import FheOps, NumpyOps
from perseus.impl.rt import Rt

D, DFF, N = 32, 128, 1024


def linear_seed(m, M):
    """ml-eval/fhe_circuit_sim.linear_rsqrt_seed: y0 = a + b v fitted on [m, M]."""
    grid = np.geomspace(m, M, 64)
    tgt = 1.0 / np.sqrt(grid)
    A = np.stack([np.ones_like(grid), grid], axis=1) / tgt[:, None]
    sol, *_ = np.linalg.lstsq(A, np.ones_like(grid), rcond=None)
    return float(sol[0]), float(sol[1])


def synthetic(L=2, d=D, dff=DFF, seed=0, ms=(2e-4, 2e-3)):
    """A random checkpoint in the trainer's state-dict names with seeds for the given
    mean-square window, plus the config the loader checks."""
    rng = np.random.default_rng(seed)
    w = {
        "g_out": 1.0 + 0.1 * rng.standard_normal(d),
        "emb.weight": 0.03 * rng.standard_normal((64, d)),
    }
    seeds = {}
    a, b = linear_seed(*ms)
    for i in range(L):
        p = f"blocks.{i}."
        w[p + "tm.g1"] = 1.0 + 0.1 * rng.standard_normal(d)
        w[p + "tm.mix"] = rng.uniform(0.3, 0.9, d)
        w[p + "tm.a_raw"] = rng.standard_normal(d)
        for k in ("c", "dd"):
            w[p + f"tm.{k}"] = 0.5 * rng.standard_normal(d)
        w[p + "tm.p0"], w[p + "tm.p1"], w[p + "tm.p2"] = (
            0.5 + 0.1 * rng.standard_normal(d),
            0.25 * rng.standard_normal(d),
            0.05 * rng.standard_normal(d),
        )
        w[p + "tm.win.weight"] = rng.standard_normal((d, d)) / np.sqrt(d) * 0.03
        w[p + "tm.wout.weight"] = rng.standard_normal((d, d)) / np.sqrt(d) * 0.03
        w[p + "cm.g2"] = 1.0 + 0.1 * rng.standard_normal(d)
        w[p + "cm.wk.weight"] = rng.standard_normal((dff, d)) / np.sqrt(d) * 0.03
        w[p + "cm.wr.weight"] = rng.standard_normal((dff, d)) / np.sqrt(d) * 0.03
        w[p + "cm.wv.weight"] = rng.standard_normal((d, dff)) / np.sqrt(dff) * 0.03
        for k, s in (
            ("q0", 0.1),
            ("q1", 1.0),
            ("q2", 0.1),
            ("q3", 0.05),
            ("r0", 0.5),
            ("r1", 0.25),
            ("r2", 0.05),
        ):
            w[p + f"cm.{k}"] = s * (1.0 + 0.1 * rng.standard_normal(dff))
        seeds[f"L{i}.tm"] = seeds[f"L{i}.cm"] = {"a": a, "b": b, "iters": 3}
    seeds["ln_out"] = {"a": a, "b": b, "iters": 3}
    cfg = {
        "d_model": d,
        "n_layer": L,
        "d_ffn": dff,
        "a_min": 0.9,
        "a_max": 0.9995,
        "alpha_res": 1.0,
        "nonlin": "poly",
        "shift_mix": True,
        "gates": "expanded",
        "vocab": 64,
    }
    return weights.ModelParams(w, seeds, cfg), w, seeds, cfg


def session(params, strict=False, rules="stock", **kw):
    make = runtime_fake.make_runtime_fake if rules == "runtime" else fake.make_fake
    fhe, inf = make(
        N=N,
        hid=params.d,
        dim=params.d,
        H=1,
        H_real=1,
        E=params.d_ffn,
        E_real=params.d_ffn,
        strict=strict,
        rot_steps=(),
        **kw,
    )
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit)
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps)
    return fhe, inf, model


def test_norm_site_frame_matches_harness_formulas():
    a, b, d = 53.12723596752366, -13744.462664572491, 1024
    site = norm.NormSite.from_seed(a, b, 5, d)
    sigma = -5.5 * b / a
    assert site.sigma == pytest.approx(sigma)
    assert site.iters == 12
    assert site.y0 == pytest.approx(0.1 * (a / np.sqrt(sigma) + b / sigma**1.5))
    assert site.input_scale**2 * d == pytest.approx(sigma)
    assert site.output_scale**2 == pytest.approx(sigma)


@pytest.mark.parametrize("form", norm.NEWTON_FORMS)
def test_newton_converges_over_the_seed_window(form):
    a, b = linear_seed(2e-4, 2e-3)
    site = norm.NormSite.from_seed(a, b, 3, D)
    v = np.geomspace(0.8 * 2e-4, 1.25 * 2e-3, 50) * site.sigma  # the calibration window, framed
    y = norm.newton_rsqrt(NumpyOps(), v, site.y0, site.iters, form)
    np.testing.assert_allclose(y, 1 / np.sqrt(v), rtol=1e-6)


def test_rmsnorm_fake_matches_closed_form_and_stays_on_the_lanes():
    params, *_ = synthetic()
    _fhe, inf, model = session(params)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(D) * 0.03  # ms ~ 9e-4, inside the window
    ct = fake.core.encode_token_input(inf, x)
    ct.vec += (np.arange(N) % model.rt.dims.t != 0) * 5.0  # off-lane junk, as after a linear
    site = params.block(0).norm_tm
    y = norm.rmsnorm(model.rt.ops, model.geo, ct, site, "t")
    got = fake.core.decode_token_output(inf, y)
    want = ref.rmsnorm_exact(x, 1.0)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)
    off = np.ones(N, bool)
    off[np.arange(D) * model.rt.dims.t] = False
    assert np.abs(y.vec[off]).max() == 0.0


def test_norm_statistic_is_tagged_one_periodic():
    """The all-slot sum is a broadcast constant, tagged so a planned refresh can route sparse."""
    params, *_ = synthetic()
    _fhe, inf, model = session(params)
    ct = fake.core.encode_token_input(inf, np.arange(D, dtype=float))
    v = model.geo.sum_all(model.rt.ops.square(ct))
    assert v.period == 1
    np.testing.assert_allclose(v.vec, np.sum(np.arange(D) ** 2.0))


def test_block_fake_matches_mirror():
    params, *_ = synthetic(L=1)
    _fhe, inf, model = session(params)
    mirror = ref.MirrorModel(params, 1)
    rng = np.random.default_rng(2)
    h = rng.standard_normal(D) * 0.03
    model.start()
    w = model.block_weights(0)
    y = block.ssm_block(
        model.rt.ops, model.geo, fake.core.encode_token_input(inf, h), w, model.state[0]
    )
    wm = weights.encode_block(params.block(0), mirror.geo, "b0.")
    ym = block.ssm_block(mirror.ops, mirror.geo, h[None], wm, mirror.state[0])
    np.testing.assert_allclose(fake.core.decode_token_output(inf, y), ym[0], rtol=1e-9, atol=1e-12)
    for got, want in (
        (model.state[0].s, mirror.state[0].s),
        (model.state[0].n_prev, mirror.state[0].n_prev),
    ):
        np.testing.assert_allclose(
            fake.core.decode_token_output(inf, got), want[0], rtol=1e-9, atol=1e-12
        )


@pytest.mark.parametrize("rules", ["stock", "runtime"])
@pytest.mark.parametrize("form", norm.NEWTON_FORMS)
def test_ticks_fake_vs_mirror_vs_exact(form, rules):
    params, *_ = synthetic(L=2, seed=3)
    fhe, _inf, model = session(params, rules=rules)
    model.form = form
    mirror, exact = ref.MirrorModel(params, 1, form), ref.ExactModel(params, 1)
    model.start()
    streams = []
    for pos, tok in enumerate([3, 17, 5]):
        row = params.emb_row(tok)
        c0 = len(fhe.calls)
        got = model.step(row, pos)["out"]
        streams.append(fhe.calls[c0:])
        want_m, want_e = mirror.step(row[None])[0], exact.step(row[None])[0]
        assert ref.rel_err_rms(got, want_m) < 1e-9
        assert ref.rel_err_rms(got, want_e) < 1e-4
        assert model.levels() == [(34, 34)] * 2
    assert streams[1] == streams[2]
    issued = {int(c[len("rotate") :]) % N for c in fhe.calls if c.startswith("rotate")}
    assert issued == model.geo.rotation_steps()


def test_every_tick_enters_at_the_same_levels_runtime_rules():
    """Under the runtime's rules a fresh encryption enters at degree 1 and a bootstrap landing at
    degree 2; with the zero state landed by a bootstrap, tick 0 runs at the levels of every later
    tick, so a plan captured on any tick binds all of them."""
    params, *_ = synthetic(L=2, seed=3)
    fhe, _inf, model = session(params, rules="runtime")
    trail, land = [], fhe._land

    def spy(vec, level, packing):
        trail.append((model.rt.ops.step_label, level))
        return land(vec, level, packing)

    fhe._land = spy
    model.start()
    ticks = []
    for pos, tok in enumerate([3, 17, 5]):
        n0 = len(trail)
        model.step(params.emb_row(tok), pos)
        ticks.append(trail[n0:])
    assert ticks[0] == ticks[1] == ticks[2]


def test_carried_value_has_unit_rms():
    """The normalized value the time-mix carries is x / sqrt(mean(x^2) + eps): rms just under 1
    per lane, the amplitude at which a bootstrap of it is most precise."""
    params, *_ = synthetic(L=1, seed=2)
    _fhe, inf, model = session(params)
    model.start()
    model.step(params.emb_row(4), 0)
    n_prev = fake.core.decode_token_output(inf, model.state[0].n_prev)
    rms = np.sqrt(np.mean(n_prev**2))
    assert 0.95 < rms < 1.0


def test_each_stage_names_its_carried_state():
    """The names the planner reads to seed the carried state at the carry's landing level."""
    params, *_ = synthetic(L=2, seed=1)
    _fhe, inf, model = session(params)
    names = []
    inf.name_ct = lambda _ct, name: names.append(name)
    model.start()
    model.step(params.emb_row(2), 0)
    assert names == [f"block_{b}.cache.{v}-lvl=34" for b in range(2) for v in ("s", "n_prev")]


def test_cold_start_equals_zero_state():
    """Encrypted zeros as the initial state give the trainer's cold start (u_{-1} = S_0 = 0)."""
    params, *_ = synthetic(L=1, seed=4)
    exact = ref.ExactModel(params, 1)
    mirror = ref.MirrorModel(params, 1)
    row = params.emb_row(9)
    assert ref.rel_err_rms(mirror.step(row[None]), exact.step(row[None])) < 1e-4


def test_mirror_keeps_sequences_apart():
    """Several sequences through the mirror at once equal each one alone: no reduction crosses
    rows, over two ticks so the carried state is exercised too."""
    params, *_ = synthetic(L=2, seed=3)
    rows = np.stack([params.emb_row(3), params.emb_row(17)])
    batch, exact = ref.MirrorModel(params, 2), ref.ExactModel(params, 2)
    alone = [ref.MirrorModel(params, 1) for _ in rows]
    for _ in range(2):
        got, want = batch.step(rows), exact.step(rows)
        for i, m in enumerate(alone):
            np.testing.assert_allclose(got[i], m.step(rows[i : i + 1])[0], rtol=1e-12, atol=1e-15)
        assert ref.rel_err_rms(got, want) < 1e-4


def test_far_ceiling_needs_only_the_carry_and_strict_mode_raises():
    """With the ceiling far away nothing refreshes but the carry; at the preset ceiling the
    eager chain needs reactive refreshes, which a strict fake turns into an error."""
    params, *_ = synthetic(L=1, seed=5)
    fhe, _inf, model = session(params, limit=10**6)
    model.start()
    assert fhe.n_bootstraps == 2  # the zero state, landed once
    model.step(params.emb_row(1), 0)
    assert fhe.n_reactive == 0
    assert fhe.n_bootstraps == 4  # + the carry
    fhe, _inf, model = session(params, strict=True)
    model.start()
    with pytest.raises(RuntimeError, match="reactive bootstrap"):
        model.step(params.emb_row(1), 0)


def _linear_operand(params, rules, level, deg):
    fhe, _inf, model = session(params, strict=True, rules=rules)
    v = np.zeros(N)
    v[np.arange(DFF) * model.rt.dims.tp_E] = 0.1
    w = model.geo.weight(params.block(0).wv, "t.wv")
    return fhe, model, fake.FakeCt(v, level, deg), w


@pytest.mark.parametrize(
    ("level", "deg", "fires"),
    [(42, 1, False), (44, 1, False), (42, 2, False), (44, 2, True), (46, 1, True)],
)
def test_hint_refreshes_exactly_when_the_linear_would_not_fit_runtime_rules(level, deg, fires):
    """Runtime rules: a product lands at its operand's effective level (level + unit for a
    pending rescale) and refreshes on reaching the ceiling, so the hint before a linear fires
    exactly when the effective level exceeds limit - unit, and the 128 products of W_v then run
    without a single reactive refresh."""
    params, *_ = synthetic(L=1)
    fhe, model, x, w = _linear_operand(params, "runtime", level, deg)
    x = model.geo.hint(x)
    assert fhe.n_bootstraps == int(fires)
    model.geo.linear(x, w)
    assert fhe.n_reactive == 0


def test_stock_fake_refreshes_one_product_earlier():
    """The stock fake charges the rescale at once: an operand at 44 passes the hint and its
    first product reaches the ceiling. Eager counts on the stock fake overstate the runtime's."""
    params, *_ = synthetic(L=1)
    fhe, model, x, w = _linear_operand(params, "stock", 44, 1)
    x = model.geo.hint(x)
    assert fhe.n_bootstraps == 0
    with pytest.raises(RuntimeError, match="reactive bootstrap at level 46"):
        model.geo.linear(x, w)


def test_weight_folds_reproduce_the_trainer_forward():
    """Undoing the folds (ExactModel) equals the plain statement list with the raw weights."""
    params, w, _seeds, cfg = synthetic(L=2, seed=6)
    exact = ref.ExactModel(params, 2)
    rows = w["emb.weight"][[1, 2]]
    got = exact.step(rows)
    h = rows.copy()
    up = [np.zeros_like(h) for _ in range(2)]
    S = [np.zeros_like(h) for _ in range(2)]
    for i in range(2):
        p = f"blocks.{i}."
        u = ref.rmsnorm_exact(h, w[p + "tm.g1"])
        um = u * w[p + "tm.mix"] + up[i] * (1 - w[p + "tm.mix"])
        up[i] = u
        x = um @ w[p + "tm.win.weight"].T
        a = weights.decay(w[p + "tm.a_raw"], cfg["a_min"], cfg["a_max"])
        S[i] = a * S[i] + (1 - a) * x
        z = w[p + "tm.c"] * S[i] + w[p + "tm.dd"] * x
        h = (
            h
            + (z * (w[p + "tm.p0"] + w[p + "tm.p1"] * z + w[p + "tm.p2"] * z * z))
            @ w[p + "tm.wout.weight"].T
        )
        u2 = ref.rmsnorm_exact(h, w[p + "cm.g2"])
        k, r = u2 @ w[p + "cm.wk.weight"].T, u2 @ w[p + "cm.wr.weight"].T
        act = w[p + "cm.q0"] + w[p + "cm.q1"] * k + w[p + "cm.q2"] * k * k + w[p + "cm.q3"] * k**3
        gate = w[p + "cm.r0"] + w[p + "cm.r1"] * r + w[p + "cm.r2"] * r * r
        h = h + (act * gate) @ w[p + "cm.wv.weight"].T
    want = ref.rmsnorm_exact(h, w["g_out"])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)


def test_load_model_rejects_other_architectures(tmp_path):
    _params, w, seeds, cfg = synthetic(L=1)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps({**cfg, "nonlin": "standard"}))
    with pytest.raises(ValueError, match="nonlin"):
        weights.load_model(
            str(tmp_path / "w.npz"), str(tmp_path / "s.json"), str(tmp_path / "c.json")
        )
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    m = weights.load_model(
        str(tmp_path / "w.npz"), str(tmp_path / "s.json"), str(tmp_path / "c.json")
    )
    assert m.n_layers == 1
    assert m.block(0).win.shape == (D, D)


def test_plain_geometry_vectors_are_identity():
    geo = geometry.PlainGeometry(NumpyOps())
    v = np.arange(5.0)
    assert (geo.lane(v) == v).all()
    assert (geo.expanded(v) == v).all()
    rt = Rt(
        FheOps(
            fake.make_fake(N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF)[1], fake.core, 2
        ),
        Dims(N, D, D, 1, 1, DFF, DFF),
    )
    sg = geometry.SlotGeometry(rt)
    assert sg.lane(v)[np.arange(5) * rt.dims.t].tolist() == v.tolist()
    e = sg.expanded(np.arange(DFF, dtype=float))
    assert np.count_nonzero(e) == DFF - 1
    assert e[rt.dims.tp_E * 1] in np.arange(DFF)
