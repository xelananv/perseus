# SPDX-License-Identifier: MIT
"""All 12 blocks on the numpy fake (runtime level rules, PCMM fake, 24-bit weights), the first tokens of a prompt:
the decrypted hidden state and the client's logits against the poly reference (clamp on). Marked slow_shape (about
3 GB, a minute); run with -m slow_shape."""
import resource
import time

import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.conftest import CKPT
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer


@need
@pytest.mark.slow_shape
def test_full_model_on_fake():
    P = ref.load_params(CKPT)
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode("\n\nThe Eiffel tower is in the city of")[:6]
    fhe, _inf, model = make_model(P, co)
    rf, st = common.prefix_state(P, COEFFS, ids, 1)
    model.start(st)
    rows = []
    for pos, t in enumerate(ids[1:], start=1):
        b0, t0 = fhe.n_bootstraps, time.time()
        h, out = model.step(P.emb[t], pos)
        lr, hr = rf.step(t)
        lf = rf.head(h)
        rows.append((pos, t, int(np.argmax(lf)), int(np.argmax(lr)), common.rel_rms(lf, lr), common.rel_rms(h, hr),
                     fhe.n_bootstraps - b0, model.rt.ops.level(out), round(time.time() - t0, 1)))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    print("(pos, token, top1 fhe, top1 poly, logits relErrRms, hidden relErrRms, refreshes, out level, s)")
    for r in rows:
        print(r)
    print(f"peak rss {peak:.2f} GB")
    assert all(r[2] == r[3] for r in rows)
    assert max(r[4] for r in rows) < 1e-2
