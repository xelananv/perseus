# SPDX-License-Identifier: MIT
"""The poly model WITHOUT the clamp (what a ciphertext evaluates; the mirror equals it to 1e-8): per position, the
sites whose output departs from the clamped one, with the input's interval coordinate t (run with -s)."""
import numpy as np

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, params, tokenizer  # noqa: F401


class Both(ref.Sites):
    def __call__(self, name, x):
        x = np.asarray(x, dtype=np.float64)
        y = super().__call__(name, x)
        s = self.cfg.get(name)
        if s is not None:
            self.clamp = True
            yc = super().__call__(name, x)
            self.clamp = False
            lo, hi = float(s["lo"]), float(s["hi"])
            t = np.abs((2 * x - lo - hi) / (hi - lo)).max()
            d = float(np.nanmax(np.abs(y - yc))) if np.all(np.isfinite(y)) else float("inf")
            if d > 1e-3 * max(float(np.abs(yc).max()), 1e-9):
                self.log.append((self.pos, name, round(float(t), 3), f"{d:.3g}"))
        return y


@need
def test_unclamped_departures(params):  # noqa: F811
    ids = tokenizer().encode("\n\nThe Eiffel tower is in the city of")
    s = Both("poly", COEFFS, clamp=False)
    s.log = []
    m = ref.RefModel(params, s)
    for i, t in enumerate(ids[:6]):
        s.pos = i
        m.step(t)
    print(s.log[:30])
