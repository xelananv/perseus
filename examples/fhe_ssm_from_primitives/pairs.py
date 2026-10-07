"""Two conversations per slot: ``x = a + i b`` carries conversation ``k`` on the real axis and
``k + t`` on the imaginary axis (the dev branch's token-pair packing).

A linear with a real matrix, a product with a real plaintext, a rotation, a sum and a
bootstrap act on ``a`` and ``b`` at once. A ciphertext product does not (``x^2 = a^2 - b^2 +
2ab i``), so a nonlinear section takes one of two routes:

``split`` (the dev branch's ``norm_token_pair`` / ``gelu_token_pair``): one conjugation gives
the two real halves ``x + conj x = 2a`` and ``i (conj x - x) = 2b``, the real kernel runs on
each, and ``a' + i b'`` repacks them. Conjugation and the product by ``i`` (a monomial) cost no
level; the factor 2 is folded into a plaintext the kernel multiplies anyway.

``lane``: ``perseus.impl.poly.LanePackedOps`` turns every ciphertext product into ``lane_mult``
/ ``lane_square`` (two products plus conjugations, and a scalar ``1/4`` that costs a level).
``PairLaneOps`` adds the one rule it lacks: a plaintext added to a pair must land on both axes,
``v (1 + i)``.
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
        return self._ops.add_pt(
            a, both, key=None if key is None else key + ".both", kind=kind, tagged=tagged
        )
