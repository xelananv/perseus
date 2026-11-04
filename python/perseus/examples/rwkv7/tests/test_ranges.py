# SPDX-License-Identifier: MIT
"""Where the poly sites' inputs leave their intervals, and by how much in the interval's own coordinate t (|t| > 1),
per position: the encrypted circuit has no clamp, so an excursion is what it would evaluate there (run with -s)."""
import numpy as np

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, params, tokenizer  # noqa: F401


class TSites(ref.Sites):
    def __call__(self, name, x):
        s = self.cfg.get(name)
        if s is not None:
            lo, hi = float(s["lo"]), float(s["hi"])
            t = np.abs((2 * np.asarray(x) - lo - hi) / (hi - lo))
            if t.max() > 1:
                self.hits.append((self.pos, name, float(t.max()), s.get("degree", s.get("steps"))))
        return super().__call__(name, x)


@need
def test_excursions(params):  # noqa: F811
    tok = tokenizer()
    for text in ("In the beginning God created the heaven and the earth. And the earth was",
                 "\n\nThe Eiffel tower is in the city of Paris, which"):
        ids = tok.encode(text)
        s = TSites("poly", COEFFS, clamp=True)
        s.hits = []
        m = ref.RefModel(params, s)
        for i, t in enumerate(ids):
            s.pos = i
            m.step(t)
        print(repr(text[:20]), ids[:3], [(p, n, round(t, 3), d) for p, n, t, d in s.hits])
