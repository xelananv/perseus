# SPDX-License-Identifier: MIT
"""The numpy reference on the real checkpoint: it predicts sensibly in exact mode, and poly mode tracks exact."""
import os
import sys

import numpy as np
import pytest

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT

_DEF_HARNESS = ROOT / "tools" / "rwkv7" / "harness"
if not (_DEF_HARNESS / "rwkv_tokenizer.py").exists():
    _DEF_HARNESS = ROOT.parents[1] / "results" / "vector-d-20261104" / "harness"
HARNESS = os.environ.get("RWKV7_HARNESS", str(_DEF_HARNESS))
need = pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(COEFFS)), reason="no checkpoint")


@pytest.fixture(scope="module")
def params():
    return ref.load_params(CKPT)


def tokenizer():
    sys.path.insert(0, HARNESS)
    from rwkv_tokenizer import RWKVTokenizer
    return RWKVTokenizer()


@need
def test_exact_mode_predicts(params):
    tok = tokenizer()
    ids = tok.encode("The Eiffel tower is in the city of")
    m = ref.RefModel(params, ref.Sites("exact"))
    for t in ids:
        logits, _ = m.step(t)
    nxt = tok.decode([int(np.argmax(logits))])
    print("next:", repr(nxt))
    assert "Paris" in nxt


@need
def test_poly_tracks_exact(params):
    tok = tokenizer()
    ids = tok.encode("In the beginning God created the heaven and the earth. And the earth was")
    ex = ref.RefModel(params, ref.Sites("exact"))
    po = ref.RefModel(params, ref.Sites("poly", COEFFS, clamp=True))
    agree = 0
    for t in ids:
        le, _ = ex.step(t)
        lp, _ = po.step(t)
        agree += int(np.argmax(le) == np.argmax(lp))
    print("top1 agree", agree, "/", len(ids), "outside", {k: v for k, v in po.sites.out_counts.items() if v[1]})
    assert agree >= len(ids) - 1
