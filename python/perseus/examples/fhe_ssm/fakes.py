# SPDX-License-Identifier: MIT
"""The numpy fake with the runtime's level rules, and the capture session, for this model.

``perseus.impl.fake.FakeFhe`` runs the Shadow in its ``fake`` mode (a rescale charged at once); the runtime
leaves it pending, which the Shadow's ``fakecap`` mode models: a product lands at degree 2 at its inputs'
effective level and the reactive refresh fires on that level reaching the ceiling, a plaintext add realizes,
a bootstrap lands at degree 2 at the dense landing of ``quirks.fakecap_landings``, a hint fires on the
effective level. ``RuntimeFakeFhe`` is the fake in that mode (plus the reactive check after a subtraction).
``levels="fork"`` = ceiling 46 / landing 34 (the session whose fake counts the fork's GPU runs reproduced);
``"preset"`` = the n32 GPT-2 preset's 44 / 32. README.md: the evidence and the caveat.
"""

from __future__ import annotations

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


def runtime_config(limit: int, bts_out: int, unit: int):
    """The fake's Config (``fake.fake_config``) with the dense landing at ``bts_out``."""
    cfg = fake.fake_config(limit, bts_out, unit)
    cfg["quirks.fakecap_landings"] = landings(bts_out)
    return cfg


class RuntimeFakeFhe(fake.FakeFhe):
    """``FakeFhe`` under the runtime's level rules (the Shadow's ``fakecap`` mode)."""

    MODE = "fakecap"

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
):
    """``fake.make_fake`` with the runtime's rules: (fhe, inf). Composite chains only (the
    plaintext-add rule of a one-prime level depends on the plaintext's level, not modelled)."""
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
        config=runtime_config(limit, bts_out, unit),
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
):
    """The capture recorder (``perseus.impl.fakecap``: the runtime's rules plus graph.json per stage)
    at this model's dims and levels: (fhe, inf); the core is ``fakecap.core``."""
    port_env()
    dims = Dims(N, d, d, 1, 1, d_ffn, d_ffn)
    return fakecap.make_session(
        dims,
        set(rot_steps),
        limit=limit,
        bts_out=bts_out,
        unit=unit,
        complex_payload=complex_payload,
        config=runtime_config(limit, bts_out, unit),
    )
