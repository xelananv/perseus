# SPDX-License-Identifier: MIT
"""The numpy fake with the runtime's level rules, and the capture session, for this model.

``perseus.impl.fake.FakeFhe`` runs the Shadow in its ``fake`` mode (a rescale charged at once); the runtime
leaves it pending, which the Shadow's ``fakecap`` mode models: a product lands at degree 2 at its inputs'
effective level and the reactive refresh fires on that level reaching the ceiling, a plaintext add realizes,
a bootstrap lands at degree 2 at the dense landing of ``quirks.fakecap_landings``, a hint fires on the
effective level. ``RuntimeFakeFhe`` is the fake in that mode (plus the reactive check after a subtraction).
``levels="fork"`` = ceiling 46 / landing 34 (the session whose fake counts the fork's GPU runs reproduced);
``"preset"`` = the n32 GPT-2 preset's 44 / 32. README.md: the evidence and the caveat. The landing's runtime
switches (``LANDING_SWITCHES``: the dense route's StC-first order, the raise drop) come from the environment as
in upstream's GPT-2 capture, so a capture records the refreshes of the runtime it is planned for.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from perseus import _pc  # type: ignore[attr-defined]
from perseus.impl import fake, fakecap
from perseus.impl.layout import Dims

from . import port_env

RULES = ("runtime", "stock")
LEVELS = ("fork", "preset")
PRESET = "n32_gpt2_decode"
#: the fork's session: ceiling 46, landing 34, two primes per level
FORK_LEVELS = (46, 34)
SPARSE_LANDINGS = "1:14,256:22,512:24"


def landings(bts_out: int) -> str:
    """``quirks.fakecap_landings`` with the dense route landing at ``bts_out`` (degree 2)."""
    return f"0:{int(bts_out)},{SPARSE_LANDINGS}"


def level_preset(levels: str) -> tuple[int, int]:
    """``(ceiling, landing)`` of a levels setting (the module docstring)."""
    if levels not in LEVELS:
        raise ValueError(f"levels must be one of {LEVELS}, got {levels!r}")
    if levels == "fork":
        return FORK_LEVELS
    cfg = _pc.Config.preset(PRESET)
    dense = str(cfg["quirks.fakecap_landings"]).split(",")[0]
    return int(cfg["bts.auto_bts_level"]), int(dense.split(":")[1])


#: The runtime switches that move where a session's own refreshes land, as legacy env names (the typed fields
#: ``bts.stc_first``, ``stc_first_routes``, ``stc_first_staged_only``, ``stc_first_cf``, ``raise_drop``): the
#: recorder reads the first three to pick the dense route's order (``fakecap.CapFhe.stc_first``), the compile
#: reads ``raise_drop`` for each route's default drop. ``tools/fhe_ssm/make_plan.sh`` LANDING=high runs
#: ``FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13
#: FIDESLIB_BTS_RAISE_DROP=512:1``.
LANDING_SWITCHES = (
    "FIDESLIB_BTS_STC_FIRST",
    "FIDESLIB_BTS_STC_FIRST_ROUTES",
    "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY",
    "FIDESLIB_BTS_STC_FIRST_CF",
    "FIDESLIB_BTS_RAISE_DROP",
)
#: the fakecap Shadow's StC-first drop when ``quirks.fakecap_stc_drop`` is unset (``crates/pc-py/src/shadow.rs``)
STC_DROP = 6


def landing_switches_set(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The ``LANDING_SWITCHES`` (and ``AUTO_BTS_LEVEL``) set non-empty in ``env`` (default ``os.environ``): what
    moves a session off the reference circuit's landing table, for reports to stamp and warn about."""
    env = os.environ if env is None else env
    return {k: env[k] for k in (*LANDING_SWITCHES, "AUTO_BTS_LEVEL") if env.get(k, "")}


def runtime_config(limit: int, bts_out: int, unit: int, env: Mapping[str, str] | None = None):
    """The fake's Config (``fake.fake_config``) with the dense landing at ``bts_out``, plus every
    ``LANDING_SWITCHES`` variable set (non-empty) in ``env`` (default ``os.environ``), typed by
    ``Config.with_env`` as upstream's capture types its environment (``fakecap.session_config``). With none
    set it is the four-field Config field for field. ``AUTO_BTS_LEVEL``, when set, must equal ``limit``: the
    ceiling is the levels setting's, and the environment saying another is a mismatch, not an override."""
    env = os.environ if env is None else env
    ceiling = env.get("AUTO_BTS_LEVEL", "")
    if ceiling and int(ceiling) != int(limit):
        raise ValueError(f"AUTO_BTS_LEVEL={ceiling} in the environment, but the session's ceiling is {limit}")
    cfg = fake.fake_config(limit, bts_out, unit)
    cfg["quirks.fakecap_landings"] = landings(bts_out)
    switches = {k: v for k, v in landing_switches_set(env).items() if k in LANDING_SWITCHES}
    return cfg.with_env(switches) if switches else cfg


def stc_first_configured(cfg) -> bool:
    """Whether ``cfg`` runs a dense refresh in the StC-first order (``fakecap.CapFhe``'s rule)."""
    return (
        (cfg["bts.stc_first"] or 0) > 0
        and not cfg["bts.stc_first_staged_only"]
        and "dense" in (cfg["bts.stc_first_routes"] or ["dense"])
    )


class RuntimeFakeFhe(fake.FakeFhe):
    """``FakeFhe`` under the runtime's level rules (the Shadow's ``fakecap`` mode). A Config with the
    StC-first order on (``stc_first_configured``) refreshes dense in that order and encodes fresh inputs at
    its landing, as the capture recorder does (``fakecap.CapFhe.inner_bootstrap``, ``bootstrap_output_level``)."""

    MODE = "fakecap"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        cfg = self.config
        self._stc_first_cfg = stc_first_configured(cfg)
        drop = cfg["quirks.fakecap_stc_drop"]
        self._stc_drop = STC_DROP if drop is None else int(drop)

    def stc_first(self) -> bool:
        """The StC-first order for the next dense refresh: the Config's and ``bootstrap.stc_first``'s scope."""
        if not self._stc_first_cfg:
            return False
        from perseus.impl import bootstrap as _pyb

        return bool(_pyb._stc_first["on"])

    def bootstrap_output_level(self):
        return self._bts_out - (self._stc_drop if self.stc_first() else 0)

    def bootstrap(self, a, route=None):
        """A refresh; a dense one (``route`` None) on the StC-first route when the Config runs that order."""
        if route is None and self.stc_first():
            route = _pc.Route(slots=0, stc_first=True)
        super().bootstrap(a, route)

    def maybe_bootstrap(self, a):
        """The reactive refresh after a product or a ciphertext subtraction, counted as such."""
        if a.level >= self._limit and not self.auto_bts_suppressed:
            if self.strict:
                raise _pc.LevelError(f"reactive bootstrap at level {a.level} (strict fake)")
            self.n_reactive += 1
            self.bootstrap(a)

    def sub(self, a, b):
        out = super().sub(a, b)
        if isinstance(b, fake.FakeCt):
            self.maybe_bootstrap(out)
        return out


def make_runtime_fake(
    N=1024,
    hid=32,
    dim=24,
    H=4,
    H_real=3,
    E=128,
    E_real=96,
    limit=46,
    bts_out=34,
    unit=2,
    strict=False,
    rot_steps=(),
    complex_payload=False,
    sparse_slots=(512, 1),
    env=None,
):
    """``fake.make_fake`` with the runtime's rules: (fhe, inf). Composite chains only (the
    plaintext-add rule of a one-prime level depends on the plaintext's level, not modelled). ``env``:
    where the landing's switches are read (``runtime_config``; default ``os.environ``)."""
    if unit < 2:
        raise ValueError("the runtime rules here are those of a composite chain (unit >= 2)")
    port_env()
    fhe = RuntimeFakeFhe(
        rot_steps,
        limit=limit,
        bts_out=bts_out,
        unit=unit,
        strict=strict,
        slots=N,
        complex_payload=complex_payload,
        sparse_slots=sparse_slots,
        config=runtime_config(limit, bts_out, unit, env),
    )
    return fhe, fake.FakeInf(fhe, N, hid, dim, H, H_real, E, E_real)


def make_fake(rules: str, **kw):
    """(fhe, inf) under ``rules``: ``runtime`` (above) or ``stock`` (``perseus.impl.fake`` as shipped)."""
    if rules not in RULES:
        raise ValueError(f"rules must be one of {RULES}, got {rules!r}")
    port_env()
    if rules == "runtime":
        return make_runtime_fake(**kw)
    return fake.make_fake(**kw)


def make_capture_session(
    N: int,
    d: int,
    d_ffn: int,
    *,
    limit=46,
    bts_out=34,
    unit=2,
    rot_steps=(),
    complex_payload=True,
    env=None,
):
    """The capture recorder (``perseus.impl.fakecap``: the runtime's rules plus graph.json per stage)
    at this model's dims and levels: (fhe, inf); the core is ``fakecap.core``. Its Config, which the
    capture stamps as ``capture_config.json``, is ``runtime_config`` with the landing's switches from
    ``env`` (default ``os.environ``)."""
    port_env()
    dims = Dims(N, d, d, 1, 1, d_ffn, d_ffn)
    return fakecap.make_session(
        dims,
        set(rot_steps),
        limit=limit,
        bts_out=bts_out,
        unit=unit,
        complex_payload=complex_payload,
        config=runtime_config(limit, bts_out, unit, env),
    )
