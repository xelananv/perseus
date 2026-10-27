# SPDX-License-Identifier: MIT
"""RMSNorm from primitives: ``x / sqrt(mean(x^2) + eps)``, the inverse square root by Newton from a constant
seed, on the normalized statistic ``v = sigma (mean(x^2) + eps)`` (``sigma = -K b / a`` from the exported
linear seed) so that ``v`` and ``y`` stay near 1; the frame rides two lane masks on the input.

``newton_rsqrt``: the first step is affine in ``v``, the rest Perseus's ``inv_sqrt_newton`` (three levels per
step) or the re-associated depth-2 form (same polynomial). ``NormSite.from_seed`` takes the seed fraction
and the minimum step count per site (``site_kwargs``: served 0.1 / 12 everywhere, or 0.3 / 9 in-block with
the output norm at 0.1 / 12). ``rmsnorm_packed`` runs one chain for K halves that reach a site together
(residue masks merge the statistics, ``log2 K`` rotations scatter ``y`` back). README.md: the mechanisms.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from perseus.impl.poly import inv_sqrt_newton

if TYPE_CHECKING:
    from .geometry import FilledGeometry, Geometry, Ops

NEWTON_FORMS = ("perseus", "depth2")
NEWTON_SITES = ("served", "per-site")
#: the served constant seed: frac 0.1, at least 12 steps, at every site
SERVED_FRAC, SERVED_MIN_ITERS = 0.1, 12
#: the per-site setting's in-block seed (the output norm keeps the served one)
PER_SITE_FRAC, PER_SITE_MIN_ITERS = 0.3, 9


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
        frac: float = SERVED_FRAC,
        min_iters: int = SERVED_MIN_ITERS,
    ) -> NormSite:
        """The site constants from the exported linear seed (a, b, iters)."""
        sigma = -k * b / a
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"seed a={a}, b={b} gives no positive sigma")
        rt = math.sqrt(sigma)
        y0 = frac * (a / rt + b / (sigma * rt))
        return cls(sigma, y0, max(int(iters), int(min_iters)), eps, int(d))

    @property
    def input_scale(self) -> float:
        """The statistic's mask: ``sum (c x)^2 = sigma mean(x^2)``."""
        return math.sqrt(self.sigma / self.d)

    @property
    def output_scale(self) -> float:
        """The output's mask: ``sqrt(sigma) x / sqrt(sigma (mean + eps))`` has unit rms."""
        return math.sqrt(self.sigma)


def site_kwargs(newton_sites: str, base: dict | None = None) -> tuple[dict, dict]:
    """``(in-block, output norm)`` keyword sets for ``NormSite.from_seed`` under a sites setting:
    ``served`` uses ``base`` (the driver's --newton-frac / --newton-min-iters / --ms-norm-k) at every
    site; ``per-site`` replaces the in-block frac and minimum with 0.3 / 9 and keeps ``base`` at the
    output norm."""
    if newton_sites not in NEWTON_SITES:
        raise ValueError(f"newton_sites must be one of {NEWTON_SITES}, got {newton_sites!r}")
    base = dict(base or {})
    if newton_sites == "served":
        return base, dict(base)
    return {**base, "frac": PER_SITE_FRAC, "min_iters": PER_SITE_MIN_ITERS}, dict(base)


def newton_rsqrt(ops: Ops, v, y0: float, iters: int, form: str = "perseus"):
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


def rmsnorm(ops: Ops, geo: Geometry, x, site: NormSite, tag: str, form: str = "perseus", gain: float = 1.0):
    """``u / sqrt(mean(u^2) + eps)`` on the lanes for ``u = gain x``, zero elsewhere. The gain
    rides the two masks, so it costs nothing: a split pair hands the norm ``2a`` and ``2b`` and
    ``gain = 0.5`` gives the norm of ``a`` and ``b`` exactly."""
    xs = geo.masked_scale(x, gain * site.input_scale, tag)
    xo = geo.masked_scale(x, gain * site.output_scale, tag + ".out")
    v = geo.sum_all(ops.square(xs))
    v = ops.add(v, site.sigma * site.eps)
    y = newton_rsqrt(ops, v, site.y0, site.iters, form)
    return ops.mult(xo, y)


def residue_masks(d: int, K: int) -> list[np.ndarray]:
    """The per-channel 0/1 masks of the packed statistic: mask ``j`` keeps the channels ``i = j (mod K)``."""
    if K < 1 or d % K:
        raise ValueError(f"{K} halves do not divide d = {d} channels")
    return [(np.arange(d) % K == j).astype(np.float64) for j in range(K)]


def rmsnorm_packed(
    ops: Any,
    geo: FilledGeometry,
    halves,
    site: NormSite,
    tag: str,
    form: str = "perseus",
    gain: float = 1.0,
):
    """``rmsnorm`` of every real half in ``halves`` through one Newton chain (the module docstring);
    returns the normalized halves in order. ``geo`` is the filled geometry the halves live on; ``ops`` its
    session's (``tag_reduce`` and ``rotate_and_sum`` are not part of the mirror's ``NumpyOps``)."""
    K = len(halves)
    if K < 2:
        return [rmsnorm(ops, geo, halves[0], site, tag, form, gain)]
    masks = [geo.lane(m) for m in residue_masks(geo.d, K)]
    xs = [geo.masked_scale(h, gain * site.input_scale, tag) for h in halves]
    xo = [geo.masked_scale(h, gain * site.output_scale, tag + ".out") for h in halves]
    merged = None
    for j, x in enumerate(xs):
        part = ops.mult_pt(geo.sum_all(ops.square(x)), masks[j], key=f"pack.{K}.{j}")
        merged = part if merged is None else ops.add(merged, part)
    ops.tag_reduce(merged, K * geo.t)
    merged = ops.add(merged, site.sigma * site.eps)
    y = newton_rsqrt(ops, merged, site.y0, site.iters, form)
    y = geo.hint(y, 1)
    out = []
    for j, h in enumerate(xo):
        yj = ops.rotate_and_sum(ops.mult_pt(y, masks[j], key=f"pack.{K}.{j}"), geo.t, K * geo.t)
        out.append(ops.mult(h, yj))
    return out
