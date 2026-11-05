# SPDX-License-Identifier: MIT
"""(Memory: the default tests together reach ~5.3 GB in one process; run this file under a 6 GB guard, the heavy ones
one per process.) The any-size port (ANYSIZE_DESIGN.md) on synthetic checkpoints (synth.py: the x070 structure at any width, random
values, coefficients fitted on the exact model's site ranges):

* the layout's shard helpers against the matrix algebra (state round trip, per-shard head reductions);
* the mirror (numpy slots: channel shards, the LoRA split, the sharded LayerNorm) against the plaintext reference in
  poly mode, unclamped, at widths 768, 1,024, 2,048, 2,560 and 4,096 (exact matrices, stored float32: ~1e-7);
* the fewer-lanes layout (one ciphertext per vector: t = 16 at 2,048, 8 at 2,560 / 4,096; ``layout.auto_lanes``, the
  default) in the mirror and, at 2,048, on the fake;
* at one sharded width (2,048 as two shards at t = 32, ``--lanes 32``): the fake (the runtime's level rules, PCMM's fake, the carry and shard landing) and the
  capture recorder against the mirror; a prefill pass against T decode steps; a batched pass against K separate
  decodes (mirror and fake);
* the 1-block capture at 2,048 (and, with --rwkv7-plan, its plan by tools/fhe_ssm/make_plan.sh).
Every synthetic checkpoint is 1-2 blocks at ring 2^15 slots (the mirror) or the fake's 2^16 (numpy vectors)."""
import json
import os
import pathlib

import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref, synth
from perseus.examples.rwkv7.layout import D, HEAD, Layout
from perseus.examples.rwkv7.model import Mirror, load_coeffs
from perseus.impl.ops import NumpyOps

SLOTS = 1 << 15
#: the checks that hold 5-9 GB each: one per process (``--rwkv7-heavy`` / RWKV7_HEAVY=1), never inside the whole suite
heavy = pytest.mark.skipif(not os.environ.get("RWKV7_HEAVY"), reason="5-9 GB: run with --rwkv7-heavy, one test per process")


def rel_rms(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


_CACHE: dict = {}


def synthetic(C, n_layer, tmp=None, vocab=2048):
    """(Params, coefficients path, token ids): a synthetic width-C checkpoint, cached per (C, n_layer)."""
    key = (C, n_layer)
    if key not in _CACHE:
        d = pathlib.Path(tmp or os.environ.get("RWKV7_SYNTH_DIR", "/tmp")) / f"rwkv7_synth_{C}_{n_layer}"
        d.mkdir(parents=True, exist_ok=True)
        ck, cj = d / "ckpt.npz", d / "coeffs.json"
        ids = [int(x) for x in np.random.default_rng(7).integers(1, vocab, 12)]
        if not (ck.exists() and cj.exists()):
            sd = synth.make_state_dict(C, n_layer, vocab=vocab, seed=C)
            np.savez(ck, **sd)
            P = ref.Params.from_state_dict(sd)
            del sd
            json.dump(synth.fit_coeffs(P, ids, margin=0.5), open(cj, "w"))
        _CACHE[key] = (ref.load_params(str(ck)), str(cj), ids, str(ck))
    return _CACHE[key][:3]


# ── layout ──

@pytest.mark.parametrize("C", [768, 2048, 2560, 4096])
def test_layout_shards(C):
    L, ops = Layout(SLOTS, C, 4 * C), NumpyOps()
    H = C // HEAD
    assert L.S == -(-C // D) and len(L.lives) == L.S
    assert sum(float(lv.sum()) for lv in L.lives) == C * L.t
    rng = np.random.default_rng(0)
    S = rng.normal(size=(H, HEAD, HEAD))
    Ss = L.state_slots_sh(S)
    assert len(Ss) == L.S * L.M
    assert np.allclose(L.state_from_slots_sh(Ss, H), S)
    v, r = rng.normal(size=C), rng.normal(size=C)
    assert np.allclose(L.decode_sh(L.natural_sh(v)), v)
    rows = rng.normal(size=(5, C))
    assert np.allclose(L.decode_lanes_sh(L.chunk_sh(rows), 5), rows)
    # y = S r per shard (the heads of shard j are its own), via the shard's transpose / untranspose
    y = []
    for j in range(L.S):
        Pj = [ops.mult(s, L.natural_sh(r)[j]) for s in Ss[j * L.M:(j + 1) * L.M]]
        y.append(L.untranspose(ops, Pj))
    want = np.einsum("hij,hj->hi", S, r.reshape(H, HEAD)).reshape(-1)
    assert np.allclose(L.decode_sh([np.real(x) for x in y]), want)
    # the LayerNorm statistic over C channels: shards added, one channel_sum
    tot = L.channel_sum(ops, sum(L.natural_sh(v)))
    assert np.allclose(np.real(tot), v.sum())


# ── the mirror against the reference ──

@pytest.mark.parametrize("C,n_layer,lanes", [(768, 2, 0), (1024, 2, 0), (2048, 2, 32), (2048, 2, 0),
                                             pytest.param(2560, 1, 32, marks=heavy), pytest.param(2560, 1, 16, marks=heavy),
                                             pytest.param(2560, 1, 0, marks=heavy), pytest.param(4096, 1, 32, marks=heavy),
                                             pytest.param(4096, 1, 0, marks=heavy)])
def test_mirror_synthetic(tmp_path_factory, C, n_layer, lanes):
    """lanes 32: channel shards of 1,024; 0: one ciphertext per vector (t = 16 at 2,048, 8 at 2,560 and 4,096);
    16 at 2,560: two shards of 2,048."""
    P, cj, ids = synthetic(C, n_layer, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    rf = ref.RefModel(P, ref.Sites("poly", cj, clamp=False))
    mir = Mirror(P, co, SLOTS, wbits=None, lanes=lanes)
    errs = []
    for t in ids[:5]:
        _l, hr = rf.step(t)
        h, z = mir.step(P.emb[t])
        assert len(z) == mir.L.S
        errs.append(rel_rms(h, hr))
    out = {k: v[1] for k, v in rf.sites.out_counts.items() if v[1]}
    print(f"C {C} lanes {mir.L.t} shards {mir.L.S} lora {mir.W[0].lora}: mirror vs poly reference relErrRms "
          f"{[f'{e:.1e}' for e in errs]}; sites out of range {out}")
    # a Chebyshev site must hold every test token; a newton_const site's [lo, hi] is its accuracy range (it converges
    # on (conv_lo, conv_hi), ROBUST_NORMS.md), so only a Chebyshev excursion makes the comparison meaningless
    cheb_out = {k: v for k, v in out.items() if not ref.site_kind(k).endswith("rsqrt")}
    assert not cheb_out, "the synthetic fit must hold the test tokens"
    # the folded matrices are stored float32 (exact for a <= 24-bit PCMM rounding): ~1e-7, as test_mirror's 3e-8
    assert max(errs) < 1e-6
    assert rel_rms(hr, np.zeros_like(hr) + hr.mean()) > 1e-2       # not vacuous: the hidden state varies


def test_lora_fold_equals_split(tmp_path_factory):
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    a, b = (Mirror(P, co, SLOTS, wbits=None, lora=x, lanes=32) for x in ("fold", "split"))
    assert a.W[1].lora == "fold" and b.W[1].lora == "split"
    e = max(rel_rms(a.step(P.emb[t])[0], b.step(P.emb[t])[0]) for t in ids[:3])
    print("fold vs split", e)
    assert e < 1e-6


def test_mirror_wbits24_synthetic(tmp_path_factory):
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    rf = ref.RefModel(P, ref.Sites("poly", cj, clamp=False))
    mir = Mirror(P, co, SLOTS, wbits=24, lanes=32)
    errs = [rel_rms(mir.step(P.emb[t])[0], rf.step(t)[1]) for t in ids[:4]]
    print("wbits 24 relErrRms", [f"{e:.1e}" for e in errs])
    assert max(errs) < 2e-3


# ── one sharded width (2,048) on the fake, prefill and batched ──

def _model(P, co, **kw):
    from perseus.examples.rwkv7.session import make_model
    return make_model(P, co, **kw)


@heavy
@pytest.mark.parametrize("capture", [False, True])
def test_fake_synthetic_2048(tmp_path_factory, capture):
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    # the recorder keeps every node's slot vector: 1 block, 3 tokens keeps it under ~5 GB at two shards
    nl, nt = (1, 3) if capture else (2, 4)
    fhe, _inf, model = _model(P, co, capture=capture, n_layers=nl, lanes=32)
    if capture:
        fhe.magnitude_suppressed = True
    assert model.L.S == 2 and model.shard_land
    mir = Mirror(P, co, model.L.n, n_layers=nl, wbits=24, lanes=32)
    model.start()
    errs, counts = [], []
    for pos, t in enumerate(ids[:nt]):
        b0 = getattr(fhe, "n_bootstraps", 0)
        h, out = model.step(P.emb[t], pos)
        counts.append(getattr(fhe, "n_bootstraps", 0) - b0)
        errs.append(rel_rms(h, mir.step(P.emb[t])[0]))
        assert len(out) == 2
    print("fake 2048" + (" (recorder)" if capture else ""), "vs mirror", [f"{e:.1e}" for e in errs],
          "refreshes per token", counts, "state names", model.state_names())
    assert max(errs) < 1e-9
    assert len(set(counts)) == 1
    assert model.state_names() == ["s0", "s1", "s0h1", "s1h1", "xa", "xah1", "xf", "xfh1"]


@pytest.mark.parametrize("T,lanes", [(5, 32), (32, 32), (16, 16)])
def test_prefill_mirror_2048(tmp_path_factory, T, lanes):
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    rng = np.random.default_rng(3)
    chunk = [int(x) for x in rng.integers(1, 2048, T)]
    tail = ids[:2]
    _rf, st = common.prefix_state(P, cj, [ids[0]], 1)
    seq, pf = Mirror(P, co, SLOTS, wbits=None, lanes=lanes), Mirror(P, co, SLOTS, wbits=None, lanes=lanes)
    seq.reset(st)
    pf.reset(st)
    hs = np.stack([seq.step(P.emb[t])[0] for t in chunk])
    hp, z = pf.prefill([P.emb[t] for t in chunk])
    e_h = max(rel_rms(hp[i], hs[i]) for i in range(T))
    e_s = max(float(np.max(np.abs(x - y))) for a, b in zip(pf.state, seq.state) for x, y in zip(a.cts(), b.cts()))
    e_d = max(rel_rms(pf.step(P.emb[t])[0], seq.step(P.emb[t])[0]) for t in tail)
    print(f"2048 prefill T {T} lanes {lanes}: hidden {e_h:.1e} state {e_s:.1e} decode after {e_d:.1e}")
    assert len(z) == pf.L.S
    assert e_h < 1e-10 and e_s < 1e-9 and e_d < 1e-10
    assert rel_rms(hs[0], hs[1]) > 1e-2


@pytest.mark.parametrize("K,carry,lanes", [(3, "lane", 32), (3, "decode", 32), pytest.param(32, "lane", 32, marks=heavy),
                                           (16, "lane", 16)])
def test_batched_mirror_2048(tmp_path_factory, K, carry, lanes):
    from perseus.examples.rwkv7.batched import BatchedMirror
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    _rf, st = common.prefix_state(P, cj, [ids[0]], 1)
    rng = np.random.default_rng(5)
    toks = rng.integers(1, 2048, (3, K))                     # 3 passes, conversation c's token on lane c
    bm = BatchedMirror(P, co, SLOTS, K, carry=carry, wbits=None, lanes=lanes)
    bm.reset(st)
    sep = Mirror(P, co, SLOTS, wbits=None, lanes=lanes)  # one set of weights; K separate states swapped in
    states = []
    for _c in range(K):
        sep.reset(st)
        states.append(sep.state)
    err = 0.0
    for row in toks:
        hb, _z = bm.step(np.stack([P.emb[t] for t in row]))
        for c in range(K):
            sep.state = states[c]
            err = max(err, rel_rms(hb[c], sep.step(P.emb[row[c]])[0]))
    serr = max(float(np.max(np.abs(x - y))) for c in range(K)
               for a, b in zip(bm.conv_state(c), states[c]) for x, y in zip(a.cts(), b.cts()))
    print(f"2048 batched K {K} {carry} lanes {lanes}: hidden {err:.1e} state {serr:.1e}")
    # exact ops in float64 (masks of 1 and 1/2, rotations, adding zeros) can make this bit-exact; not vacuous: the
    # conversations differ
    assert err < 1e-10 and serr < 1e-9
    assert rel_rms(hb[0], hb[1]) > 1e-3


@pytest.mark.parametrize("lanes", [0, 32])
def test_filled_mirror_2048(tmp_path_factory, lanes):
    """The filled state layout above width 1,024 (review lens 1, finding 4): at 2,048 as one t = 16 ciphertext
    (lanes 0) and as two t = 32 shards, the filled mirror gives the current mirror's hidden states for 3 decode
    steps and a K = 3 batched pass, with half the state ciphertexts per block (2 blocks, 2^15 slots). One model is
    alive at a time (two 2,048-wide mirrors together pass 6 GB on this Mac)."""
    import gc

    from perseus.examples.rwkv7.batched import BatchedMirror
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    _rf, st = common.prefix_state(P, cj, [ids[0]], 1)
    rows = np.stack([P.emb[t] for t in ids[4:7]])
    out, n_s = {}, {}
    for x in ("current", "filled"):
        m = Mirror(P, co, SLOTS, wbits=None, lanes=lanes, layout=x)
        m.reset(st)
        n_s[x] = len(m.state[0].S)
        out[x] = [m.step(P.emb[t])[0] for t in ids[1:4]]
        del m
        gc.collect()
        bm = BatchedMirror(P, co, SLOTS, 3, carry="lane", wbits=None, lanes=lanes, layout=x)
        bm.reset(st)
        out[x].append(bm.step(rows)[0])
        del bm
        gc.collect()
    assert n_s["filled"] * 2 == n_s["current"]
    errs = [rel_rms(b_, a_) for a_, b_ in zip(out["current"], out["filled"])]
    print(f"2048 filled vs current, lanes {lanes}: hidden rel max {max(errs):.1e}; state cts per block "
          f"{n_s['current']} -> {n_s['filled']}")
    assert max(errs) < 1e-10
    ya = out["current"][-1]
    assert rel_rms(ya[0], ya[1]) > 1e-3


@heavy
def test_fake_prefill_and_batched_2048(tmp_path_factory):
    """The fake: a 4-token prefill pass + 1 decode step vs 5 decode steps; a K = 3 batched pass (lane carries) vs 3
    separate fake decodes. Two fake models (each holds its PCMM encodings), restarted between runs."""
    import gc
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    _rf, st = common.prefix_state(P, cj, [ids[0]], 1)
    body = ids[1:6]
    _f, _i, m = _model(P, co, n_layers=2, prefill_keys=4, lanes=32)
    m.start(st)
    seq = np.stack([m.step(P.emb[t], 1 + i)[0] for i, t in enumerate(body)])
    m.start(st)
    m.set_chunk(4)
    pf = list(m.step([P.emb[t] for t in body[:4]], 1)[0])
    m.set_chunk(0)
    pf.append(m.step(P.emb[body[4]], 5)[0])
    e_pf = max(rel_rms(a, b) for a, b in zip(pf, seq))
    K = 3
    sep = []
    for c in range(K):
        m.start(st)
        sep.append([m.step(P.emb[ids[1 + c + K * i]], 1 + i)[0] for i in range(2)])
    del m, _f, _i
    gc.collect()
    _f, _i, mb = _model(P, co, n_layers=2, batch=K, batch_carry="lane", lanes=32)
    mb.start(st)
    hb = [mb.step([P.emb[ids[1 + c + K * i]] for c in range(K)], 1 + i)[0] for i in range(2)]
    e_b = max(rel_rms(hb[i][c], sep[c][i]) for c in range(K) for i in range(2))
    print(f"fake 2048: prefill vs sequential {e_pf:.1e}; batched K=3 vs separate {e_b:.1e}")
    assert e_pf < 1e-9 and e_b < 1e-9
    assert rel_rms(hb[1][0], hb[1][1]) > 1e-3


@heavy
def test_capture_2048(tmp_path_factory, monkeypatch):
    """run_capture on a 1-block synthetic width-2048 checkpoint (the shards, the shard landing, every state name an
    input); with RWKV7_PLAN=1 (--rwkv7-plan) also the plan, and the refreshes it places (RWKV7_ANY_DIR keeps both)."""
    import subprocess
    import sys

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.conftest import ROOT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    C = int(os.environ.get("RWKV7_ANY_C", "2048"))
    nb = int(os.environ.get("RWKV7_ANY_BLOCKS", "1"))
    root = pathlib.Path(os.environ.get("RWKV7_ANY_DIR", str(tmp_path_factory.mktemp("any"))))
    _P, cj, ids = synthetic(C, max(nb, 1), tmp_path_factory.getbasetemp())
    ck = _CACHE[(C, max(nb, 1))][3]
    cap = root / f"cap_C{C}_b{nb}"
    if not (cap / "capture_config.json").exists():
        assert run_capture.main(["--pth", ck, "--coeffs", cj, "--ids", ",".join(map(str, ids)), "--blocks", str(nb),
                                 "--lanes", "32", "--out", str(cap)]) == 0
    g = json.load(open(cap / "block_0" / "graph.json"))
    ops = {}
    for n in g["nodes"]:
        ops[n["op_type"]] = ops.get(n["op_type"], 0) + 1
    print(f"C {C} block_0: {len(g['nodes'])} nodes", dict(sorted(ops.items())))
    names = {x for n in g["nodes"] for x in n["inputs"]}
    from perseus.examples.rwkv7.model import state_names
    S = -(-C // D)
    for s in state_names(S, 2):
        assert any(f"cache.{s}-lvl" in x for x in names), s
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    env = {**os.environ, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
           "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
    name = f"plan_C{C}_b{nb}"
    if not (root / "bootstrap_placements" / name / "PLAN_CMD.txt").exists():
        p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), name], env=env,
                           capture_output=True, text=True, timeout=6000)
        print(p.stdout[-3000:], p.stderr[-3000:])
    plan = root / "bootstrap_placements" / name
    from perseus.examples.rwkv7.plan_cost import plan_refreshes
    rep = plan_refreshes(str(cap), str(plan))
    print("refreshes per stage", json.dumps(rep))
    assert (plan / "block_0_placement.json").exists()


# ── the fewer-lanes layout (2,048 channels per ciphertext, t = 16 lanes) at width 2,048 ──

@heavy
def test_fewer_lanes_mirror_and_fake_2048(tmp_path_factory):
    """``lanes=16``: one ciphertext per width-2,048 vector (no shards), M = 4 state ciphertexts per block, PCMM at its
    own t = 16 shape (2,048 channels); the mirror against the poly reference, and the fake against the mirror."""
    P, cj, ids = synthetic(2048, 2, tmp_path_factory.getbasetemp())
    co = load_coeffs(cj)
    mir = Mirror(P, co, SLOTS, wbits=None, lanes=16)
    assert (mir.L.S, mir.L.t, mir.L.M, mir.L.D) == (1, 16, 4, 2048)
    rf = ref.RefModel(P, ref.Sites("poly", cj, clamp=False))
    e_ref = max(rel_rms(mir.step(P.emb[t])[0], rf.step(t)[1]) for t in ids[:4])
    fhe, _inf, model = _model(P, co, n_layers=2, lanes=16)
    assert model.L.t == 16 and model.state_names() == ["s0", "s1", "s2", "s3", "xa", "xf"]
    m24 = Mirror(P, co, SLOTS, wbits=24, lanes=16)
    model.start()
    errs, counts = [], []
    for pos, t in enumerate(ids[:3]):
        b0 = fhe.n_bootstraps
        h, out = model.step(P.emb[t], pos)
        counts.append(fhe.n_bootstraps - b0)
        errs.append(rel_rms(h, m24.step(P.emb[t])[0]))
    print(f"lanes 16 at 2048: mirror vs poly {e_ref:.1e}; fake vs mirror {[f'{e:.1e}' for e in errs]}; "
          f"refreshes per token (eager fake) {counts}")
    assert e_ref < 1e-6 and max(errs) < 1e-9 and len(set(counts)) == 1
