# SPDX-License-Identifier: MIT
"""RMSNorm from primitives: ``x / sqrt(mean(x^2) + eps)``, the inverse square root by Newton.

The served circuit (hpc_gpu_port/experimental/gpu_real_model_x.cu, ``rmsnorm``; flags
``--ms-norm 5.5 --newton-robust 0.1 12`` in the recorded runs) runs Newton on the normalized
statistic ``v = sigma (mean(x^2) + eps)`` with ``sigma = -K b / a`` recovered from the exported
linear seed ``y0 = a + b ms``, so that ``v`` and ``y = 1/sqrt(v)`` are both near 1 inside the
loop (their magnitudes are reciprocal; without the frame their ratio reaches 3e4 across the
sites, wider than a bootstrap tolerates). The seed is the constant ``frac (a_n + b_n)``, the
linear seed in that frame at ``v = 1`` scaled by ``frac``; the iteration count is
``max(exported, 12)``.

Here the frame rides two lane masks on the input, both one product from ``x`` and zero off
the lanes: ``sqrt(sigma / d) x``, whose square summed over every slot is ``sigma mean(x^2)``
directly, and ``sqrt(sigma) x``, which the inverse square root turns into
``x / sqrt(mean(x^2) + eps)``: unit rms per lane, the amplitude at which a bootstrap of the
carried value is most precise, as in Perseus's own norm. Newton from the constant seed: the first step is affine in ``v``
(``y1 = 1.5 y0 - 0.5 y0^3 v``, no ciphertext product), the rest either Perseus's
``inv_sqrt_newton`` (three levels per step) or the re-associated ``y (1.5) + ((-0.5 v) y) y^2``
(two levels per step, the harness's ``--newton-depth2`` form). Same polynomial either way.
"""

from __future__ import annotations

import dataclasses
import math

from perseus.impl.poly import inv_sqrt_newton

NEWTON_FORMS = ("perseus", "depth2")


@dataclasses.dataclass(frozen=True)
class NormSite:
    sigma: float
    y0: float
    iters: int
    eps: float
    d: int

    @classmethod
    def from_seed(
        cls,
        a: float,
        b: float,
        iters: int,
        d: int,
        *,
        eps: float = 1e-5,
        k: float = 5.5,
        frac: float = 0.1,
        min_iters: int = 12,
    ) -> NormSite:
        """The site constants from the exported linear seed (a, b, iters)."""
        sigma = -k * b / a
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"seed a={a}, b={b} gives no positive sigma")
        rt = math.sqrt(sigma)
        y0 = frac * (a / rt + b / (sigma * rt))
        return cls(sigma, y0, max(int(iters), min_iters), eps, int(d))

    @property
    def input_scale(self) -> float:
        """The statistic's mask: ``sum (c x)^2 = sigma mean(x^2)``."""
        return math.sqrt(self.sigma / self.d)

    @property
    def output_scale(self) -> float:
        """The output's mask: ``sqrt(sigma) x / sqrt(sigma (mean + eps))`` has unit rms."""
        return math.sqrt(self.sigma)


def newton_rsqrt(ops, v, y0: float, iters: int, form: str = "perseus"):
    """``y <- y (1.5 - 0.5 v y^2)`` ``iters`` times from the constant ``y0``."""
    if form not in NEWTON_FORMS:
        raise ValueError(f"form must be one of {NEWTON_FORMS}, got {form!r}")
    y = ops.add(ops.mult(v, -0.5 * y0**3), 1.5 * y0)
    if iters <= 1:
        return y
    if form == "perseus":
        return inv_sqrt_newton(ops, v, y, iters - 1)
    vh = ops.mult(v, -0.5)
    for _ in range(iters - 1):
        t = ops.mult(ops.mult(vh, y), ops.square(y))
        y = ops.add(ops.mult(y, 1.5), t)
    return y


def rmsnorm(ops, geo, x, site: NormSite, tag: str, form: str = "perseus", gain: float = 1.0):
    """``u / sqrt(mean(u^2) + eps)`` on the lanes for ``u = gain x``, zero elsewhere. The gain
    rides the two masks, so it costs nothing: a split pair hands the norm ``2a`` and ``2b`` and
    ``gain = 0.5`` gives the norm of ``a`` and ``b`` exactly."""
    xs = geo.masked_scale(x, gain * site.input_scale, tag)
    xo = geo.masked_scale(x, gain * site.output_scale, tag + ".out")
    v = geo.sum_all(ops.square(xs))
    v = ops.add(v, site.sigma * site.eps)
    y = newton_rsqrt(ops, v, site.y0, site.iters, form)
    return ops.mult(xo, y)
