# SPDX-License-Identifier: MIT
"""``RWKV7_LEAN_PREFIX`` in the drivers themselves (review of b339dda2, L1): ``main()`` (decode, decode + prefill)
and ``main_batch`` run end to end on stand-ins for the session, the twins and the scorers, so what is checked is the
drivers' own wiring: which prefix-state sets are encrypted and when, which pass reads which set, and how many sets
are alive (a ``WeakSet`` of every set made) at every pass.

Non-vacuous by construction: a missing decode set at n_pf = 0 fails the "token 1 reads dec0" check; a missing drop
fails the "0 live sets at the next pass" check; an extra (unread) set fails the encryption counts; the flag-off runs
pin the old behaviour (three sets at the prefill warm-up, two held for life, the batched set held for life)."""
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from perseus.examples.rwkv7 import common
from perseus.examples.rwkv7 import run_planned as rp


class KvSet(dict):
    """A prefix-state set (a dict subclass: weak-referenceable, so the test can see when the driver drops it;
    hashed by identity for the WeakSet)."""

    __hash__ = object.__hash__


class Env:
    def __init__(self):
        self.live = weakref.WeakSet()
        self.ev = []
        self.n = {}

    def make(self, tag):
        k = self.n.get(tag, 0)
        self.n[tag] = k + 1
        s = KvSet({0: {0: object()}})
        s.name = f"{tag}{k}"
        self.live.add(s)
        self.ev.append(("enc", s.name))
        return s

    def nlive(self):
        return len(list(self.live))


class FakeSess:
    mem_report = "stand-in"
    has_secret = False

    def planned_stats(self):
        return {}

    def pool_free(self):
        return 1000


def fakes(monkeypatch, env, K=0):
    import perseus
    from perseus import _pc
    from perseus.examples.fhe_ssm import run_planned as fsr
    from perseus.examples.rwkv7 import batched, run_capture

    model_box = {}

    class FakeRun:
        def __init__(self, _rec, plan, _cfg, _labels, _keys, *, model, **_kw):
            self.model, self.plan = model, plan
            self.tag = "pf" if model.chunk else ("bat" if model.batch > 1 else "dec")
            self.sess, self.progs, self.client, self.bundle = FakeSess(), {}, None, None

        def setup(self, _rows):
            return self

        def state_cts(self, _st, share=None):
            s = env.make(self.tag)
            self.prefix_encryptions = 4
            return s

        def token(self, t, _rows, argmax=False, kv_cts=None):
            env.ev.append(("token", self.tag, t, kv_cts.name if kv_cts is not None else None, env.nlive()))
            return [0], {"issue_s": 0.0}

        def reply(self, _reg):
            if self.model.batch > 1:
                return [np.zeros(4) for _ in range(self.model.batch)], 5
            return np.zeros(4), 5

        def close(self):
            pass

    class FakeBoth:
        def __init__(self, run, prun, T):
            self.run, self.pf, self.T = run, prun, T

        def setup(self, *_a):
            return self

        def prefill(self, _rows, kv_cts=None):
            env.ev.append(("prefill", kv_cts.name if kv_cts is not None else None, env.nlive()))
            return [0]

        def reply(self, _regs, T):
            return [np.zeros(4) for _ in range(T)], 5

    class FakeScorer:
        def __init__(self, *_a, **_k):
            self.poly = SimpleNamespace(head=lambda h: np.zeros(5))

        def score(self, pos, _tok, _hidden):
            return {"top1_fhe": 0, "top1_agree": 1, "logits_relErrRms": 0.0, "hidden_relErrRms_vs_mirror": 0.0,
                    "poly_sites_out_of_range": None, "pos": pos}

    class FakeMirror:
        def __init__(self, *_a, **_k):
            pass

        def reset(self, _st):
            pass

        def step(self, rows):
            return [np.zeros(4) for _ in range(len(rows))], None

    def make_model(_P, _co, chunk=0, batch=0, **_k):
        m = SimpleNamespace(n_layers=1, L=SimpleNamespace(n=8, S=1, t=32), W=[SimpleNamespace(lora="fold")],
                            shard_land=None, init_state=None, chunk=chunk, batch=batch or 1)
        model_box.setdefault("models", []).append(m)
        return None, None, m

    P = SimpleNamespace(emb=np.zeros((64, 4)), C=4)
    monkeypatch.setattr(common, "load", lambda a: (P, {}, None))
    monkeypatch.setattr(common, "token_ids", lambda a, text=None: list(range(40)) if text else
                        [1] + [int(x) for x in a.ids.split(",")])
    monkeypatch.setattr(common, "prefix_state", lambda *_a: (None, "st"))
    monkeypatch.setattr(common, "Scorer", FakeScorer)
    monkeypatch.setattr(common, "sha256", lambda p: "x")
    monkeypatch.setattr(common, "environment", dict)
    monkeypatch.setattr(common, "tokenizer", lambda *_a: SimpleNamespace(encode=lambda s: [1] + [2] * len(s)))
    monkeypatch.setattr(rp, "make_model", make_model)
    monkeypatch.setattr(rp, "POOL_TRACE", False)          # the header fields under test: the lean flag's only
    monkeypatch.setattr(rp, "model_kw", lambda a: {})
    monkeypatch.setattr(rp, "Recorder", lambda *a, **k: None)
    monkeypatch.setattr(rp, "probe_rows", lambda *a: [None, None])
    monkeypatch.setattr(rp, "resolve_keys", lambda a: (None, None))
    monkeypatch.setattr(rp, "keys_header", lambda run, b=None: {})
    monkeypatch.setattr(rp, "Rwkv7PlannedRun", FakeRun)
    monkeypatch.setattr(rp, "Rwkv7BatchedRun", FakeRun)
    monkeypatch.setattr(rp, "Rwkv7PrefillRun", FakeBoth)
    monkeypatch.setattr(rp, "Mirror", FakeMirror)
    monkeypatch.setattr(rp, "read_prompts", lambda p: ["abc"] * K)
    monkeypatch.setattr(_pc, "KeySource", SimpleNamespace(generate=lambda *a, **k: None), raising=False)
    monkeypatch.setattr(fsr, "session_config", lambda plan: None)
    monkeypatch.setattr(batched, "BatchedMirror", FakeMirror)
    clock = iter(range(1, 1 << 20))                     # a strictly increasing clock: no zero-length pass
    monkeypatch.setattr(rp, "time", SimpleNamespace(time=lambda: float(next(clock))))
    monkeypatch.setattr(run_capture, "batch_probe", lambda P, ids, K, n: ([[P.emb[0]] * K] * n, None))
    assert perseus


def run_main(monkeypatch, tmp_path, lean, n_prompt, T):
    env = Env()
    fakes(monkeypatch, env)
    monkeypatch.setattr(rp, "LEAN_PREFIX", lean)
    out = tmp_path / "o.jsonl"
    argv = ["--pth", "x", "--coeffs", "c", "--plan", "p", "--out", str(out), "--prefix", "1",
            "--ids", ",".join(str(3 + i) for i in range(n_prompt)), "--tokens", str(n_prompt), "--generate", "2"]
    if T:
        argv += ["--prefill-chunk", str(T), "--pf-plan", "pp"]
    assert rp.main(argv) == 0
    return env.ev, [json.loads(x) for x in out.read_text().splitlines()]


def _encs(ev):
    return [e[1] for e in ev if e[0] == "enc"]


@pytest.mark.parametrize("lean", [False, True])
def test_main_decode_only(monkeypatch, tmp_path, lean):
    """n_pf = 0, no prefill session: the decode set is encrypted and read by the first decode token (t + 1 = 1) only;
    with the flag it is gone by the second token."""
    ev, _recs = run_main(monkeypatch, tmp_path, lean, n_prompt=3, T=0)
    toks = [e for e in ev if e[0] == "token"]
    assert toks[0][2:4] == (0, None)                                   # the warm-up: empty state
    assert _encs(ev) == ["dec0"]
    assert toks[1][2:4] == (1, "dec0")                                 # the first token reads the prefix set
    assert all(e[3] is None for e in toks[2:])
    assert [e[4] for e in toks[2:]] == ([0] * 4 if lean else [1] * 4)  # dropped after token 0 / held for life


@pytest.mark.parametrize("lean", [False, True])
def test_main_prefill_whole_chunks(monkeypatch, tmp_path, lean):
    """n_pf > 0 (9 prompt tokens, T = 4: two chunks, one decoded token): the decode set is never read, so with the
    flag it is never made; the prefill set is read by chunk 0 only and (flag) dropped before chunk 1; the warm-up's
    own set is the only live one at the warm-up (flag) instead of three."""
    ev, _recs = run_main(monkeypatch, tmp_path, lean, n_prompt=9, T=4)
    pf = [e for e in ev if e[0] == "prefill"]
    toks = [e for e in ev if e[0] == "token"]
    warm, chunk0, chunk1 = pf
    if lean:
        assert _encs(ev) == ["pf0", "pf1"]                             # the warm-up's set, then the run's
        assert warm == ("prefill", "pf0", 1)
        assert chunk0 == ("prefill", "pf1", 1)
        assert chunk1 == ("prefill", None, 0)
    else:
        assert _encs(ev) == ["dec0", "pf0", "pf1"]
        assert warm == ("prefill", "pf1", 3)                           # three sets live at the warm-up
        assert chunk0 == ("prefill", "pf0", 2)
        assert chunk1 == ("prefill", None, 2)
    assert all(e[3] is None for e in toks)                             # no decode token reads a prefix set
    assert toks[1][2] == 8 + 1                                         # the first decoded token: t = n_pf (+1)


@pytest.mark.parametrize("lean", [False, True])
def test_main_prefill_session_short_prompt(monkeypatch, tmp_path, lean):
    """A prefill session whose prompt is shorter than T (n_pf = 0): no chunk runs, so the first decode token reads
    the decode set; with the flag the prefill set is never made."""
    ev, _recs = run_main(monkeypatch, tmp_path, lean, n_prompt=3, T=4)
    toks = [e for e in ev if e[0] == "token"]
    pf = [e for e in ev if e[0] == "prefill"]
    assert len(pf) == 1                                                # the warm-up only
    assert toks[1][2:4] == (1, "dec0")
    if lean:
        assert _encs(ev) == ["pf0", "dec0"]                            # the warm-up's set (freed), then the decode set
        assert pf[0][2] == 1 and toks[1][4] == 1 and toks[2][4] == 0
    else:
        assert _encs(ev) == ["dec0", "pf0", "pf1"]
        assert toks[2][4] == 2


@pytest.mark.parametrize("lean", [False, True])
def test_main_batch(monkeypatch, tmp_path, lean):
    """``main_batch``: one set, read by pass 0 only (t = 1), the registers after; with the flag it is dropped after
    pass 0, and the header carries ``prefix_encryptions`` (what the first device run reads to settle the sharing)."""
    K = 4
    env = Env()
    fakes(monkeypatch, env, K=K)
    monkeypatch.setattr(rp, "LEAN_PREFIX", lean)
    out = tmp_path / "b.jsonl"
    argv = ["--pth", "x", "--coeffs", "c", "--plan", "p", "--out", str(out), "--prefix", "1", "--batch", str(K),
            "--prompts", "P", "--harness", "h", "--generate", "1"]
    assert rp.main(argv) == 0
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    toks = [e for e in ev if e[0] == "token"] if (ev := env.ev) else []
    assert _encs(ev) == ["bat0"]
    assert toks[0][2:4] == (0, None)                                   # warm-up
    assert toks[1][2:4] == (1, "bat0")                                 # pass 0 reads the set
    assert len(toks) >= 4 and all(e[3] is None for e in toks[2:])      # later passes: the registers
    assert [e[4] for e in toks[2:]] == ([0] if lean else [1]) * len(toks[2:])
    header = recs[0]
    assert ("prefix_encryptions" in header) == lean
    if lean:
        assert header["prefix_encryptions"] == 4
