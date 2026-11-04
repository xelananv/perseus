# SPDX-License-Identifier: MIT
"""The model on the numpy fake under the runtime's level rules (reactive refreshes at the ceiling, PCMM's fake with
its external refreshes, the canonical carry) against the mirror: same values to float64 rounding, every token one
op stream. 2 blocks of the real checkpoint, 4 tokens."""
import numpy as np
import pytest

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.model import Mirror, load_coeffs
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.test_mirror import N_LAYERS, params2, rel_rms  # noqa: F401
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer


@need
@pytest.mark.parametrize("capture", [False, True])
def test_fake_matches_mirror(params2, capture):  # noqa: F811
    ids = tokenizer().encode("\n\nThe Eiffel tower")
    co = load_coeffs(COEFFS)
    fhe, inf, model = make_model(params2, co, capture=capture, n_layers=N_LAYERS)
    if capture:
        fhe.magnitude_suppressed = True
    mir = Mirror(params2, co, model.L.n, n_layers=N_LAYERS, wbits=24)
    rf = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=True))
    model.start()
    n0 = fhe.n_bootstraps if hasattr(fhe, "n_bootstraps") else None
    errs, errs_ref, counts = [], [], []
    for pos, t in enumerate(ids):
        b0 = getattr(fhe, "n_bootstraps", 0)
        r0 = getattr(fhe, "n_reactive", 0)
        h, out = model.step(params2.emb[t], pos)
        hm, _ = mir.step(params2.emb[t])
        _, hr = rf.step(t)
        errs.append(rel_rms(h, hm))
        errs_ref.append(rel_rms(h, hr))
        counts.append((getattr(fhe, "n_bootstraps", 0) - b0, getattr(fhe, "n_reactive", 0) - r0,
                       model.rt.ops.level(out)))
    print("capture" if capture else "fake", "vs mirror", [f"{e:.1e}" for e in errs], "vs poly ref",
          [f"{e:.1e}" for e in errs_ref], "(bootstraps, reactive, out level) per token", counts, "start", n0)
    assert max(errs) < 1e-9
    assert len({c[:2] for c in counts}) == 1          # every token the same refreshes
