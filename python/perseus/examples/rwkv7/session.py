# SPDX-License-Identifier: MIT
"""Sessions for the CPU paths: the numpy fake under the runtime's level rules and the capture recorder (the planned
driver's CPU twin), both from ``fhe_ssm``'s helpers, at ring 2^16 (32,768 slots: the fake's vectors are numpy arrays,
not CKKS ciphertexts, so the ring size costs megabytes here; PCMM is pinned to N = 2^16, ``impl/pcmm.py``)."""
from __future__ import annotations

from types import SimpleNamespace

from perseus.examples.fhe_ssm import fakes, port_env
from perseus.examples.fhe_ssm import run_fake as ssm_run_fake

from .model import Rwkv7Fhe

LOGN = 16


def level_preset(levels: str = "preset"):
    return fakes.level_preset(levels)


def make_session(*, capture=False, levels="preset", unit=2, strict=False, lanes=32):
    """(fhe, inf, core) at this model's dims (``32,768 / lanes`` channels per ciphertext: 1,024 at the default t = 32
    lanes, so ``pack_tokens`` lays out t lanes)."""
    port_env()
    limit, bts_out = level_preset(levels)
    d = (1 << (LOGN - 1)) // int(lanes)
    dims = SimpleNamespace(d=d, d_ffn=3 * d)
    return ssm_run_fake.make_session(dims, LOGN, rules="runtime", limit=limit, bts_out=bts_out, unit=unit,
                                     strict=strict, complex_payload=True, capture=capture)


def rot_band(fhe, model: Rwkv7Fhe):
    """The session's rotation keys: exactly the steps the model issues."""
    n = model.rt.dims.N
    steps = {int(s) % n for s in model.rotation_steps()}
    fhe.loaded_rot_steps = sorted(steps)
    fhe._rot = steps | {s - n for s in steps}


def make_model(P, coeffs, *, capture=False, levels="preset", n_layers=None, lanes=0, **kw):
    """``lanes``: t (32 / 16 / 8), 0 = ``layout.auto_lanes`` for the checkpoint's width."""
    from .layout import auto_lanes
    lanes = int(lanes) or auto_lanes(P.C, 1 << (LOGN - 1))
    fhe, inf, core = make_session(capture=capture, levels=levels, lanes=lanes)
    model = Rwkv7Fhe(inf, P, coeffs, core=core, n_layers=n_layers, lanes=lanes, **kw)
    rot_band(fhe, model)
    return fhe, inf, model
