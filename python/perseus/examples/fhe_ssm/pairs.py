# SPDX-License-Identifier: MIT
"""Two conversations per slot: ``x = a + i b`` carries conversation ``k`` on the real axis and ``k + t`` on
the imaginary axis (perseus.impl.prefill's token-pair packing).

A real linear, a real plaintext product, a rotation, a sum and a bootstrap act on both axes at once; a
ciphertext product does not, so a nonlinear section takes the ``split`` route (one conjugation gives
``2a`` and ``2b``, the real kernel runs on each, ``a' + i b'`` repacks them; no level) or the ``lane`` route
(``perseus.impl.poly.LanePackedOps``, a level per product). ``PairLaneOps`` adds the one rule the lane route
lacks: a plaintext added to a pair lands on both axes.
"""

from __future__ import annotations

import numpy as np

from perseus.impl.poly import LanePackedOps


def split(ops, x):
    """``(2 Re x, 2 Im x)`` as two ciphertexts with real payloads: one conjugation, no level."""
    xc = ops.conjugate(x)
    return ops.add(x, xc), ops.mult_i(ops.sub(xc, x))


def repack(ops, a, b):
    """``a + i b`` from two real payloads: the product by ``i`` is a monomial, no level."""
    return ops.add(a, ops.mult_i(b))


class PairLaneOps(LanePackedOps):
    """``LanePackedOps`` whose plaintext adds land on both axes of the pair."""

    def add_pt(self, a, v, key=None, kind="weight", tagged=True):
        both = np.asarray(v) * (1.0 + 1.0j)
        return self._ops.add_pt(a, both, key=None if key is None else key + ".both", kind=kind, tagged=tagged)
