# SPDX-License-Identifier: MIT
"""The port side of the device diagonal encoder: where a diagonal linear's plaintexts come from (``--diag-source``).
``values`` (the default): every diagonal a host slot vector (``LazyDiagonals``) multiplied at use, so on a device a
resident ``PtStore`` plaintext (344,064 per tick at the model shape). ``generated``: the device builds each diagonal
inside the fused sum of its ``(m, gg)`` group from the linear's resident float64 matrix (the tile): no PCIe, no slot.
Both record the same op stream (per group the loop ``mult`` + ``add_inplace`` over ``n_in b`` terms, n outer, jj
inner; equal levels, tags and, with values, magnitudes); a generated capture differs only in each diagonal's
``args.pt`` (``kind = "diag"``, ``name = <linear>#<n>.<m>.<gg>.<jj>``), which the compiler fuses into one ``CoeffSum``
per group (crates/pc-program/src/compile/diag.rs): captures and plans are shared, the source is the driver's choice.
``use_source(inf, source)`` installs the hook ``filling.mult_diag_sum`` dispatches to: on the numpy fake the vectors
of ``filling.diag_gather_reference`` (the kernel's index arithmetic) through ``ops.mult_pt_sum``, so the fake's reply
is bit-identical under both sources only if that arithmetic is right; on the capture recorder ``diag_sum_recorder``.
"""

from __future__ import annotations

import numpy as np

from perseus.impl import fake, fakecap

from . import filling

SOURCES = ("values", "generated")
HOOK = "mult_diag_sum"


class GeneratedPt:
    """What the planned twin's recorder keeps for a generated diagonal in place of its slot vector: the name (the
    values are the registered tile's, the same every tick). Keeping the vector would cost 256 KiB per diagonal,
    344,064 per tick at the model shape."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = str(name)

    def __eq__(self, other) -> bool:
        return isinstance(other, GeneratedPt) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("GeneratedPt", self.name))

    def __repr__(self) -> str:
        return f"GeneratedPt({self.name!r})"


def group_vectors(lin, m: int, gg: int) -> np.ndarray:
    """The ``(n_in b, n_slots)`` slot vectors of group ``(m, gg)`` of ``lin`` (a ``DiagLinear`` over
    ``LazyDiagonals``), gathered from its matrix with the kernel's arithmetic."""
    pts = lin.pts
    if not isinstance(pts, filling.LazyDiagonals):
        raise TypeError(
            f"{lin.name}: generated diagonals read the matrix of a LazyDiagonals table, got {type(pts)!r}"
        )
    return filling.diag_gather_reference(pts.W, pts.N, pts.t, filling.group_jobs(lin, m, gg))


def diag_sum_fake(ops, cts, lin, m, gg):
    """The fake's hook: the group's vectors from ``diag_gather_reference``, multiplied and summed by
    ``ops.mult_pt_sum`` exactly as ``DiagLinear._sum`` multiplies ``pts[m, gg]``."""
    return ops.mult_pt_sum(cts, group_vectors(lin, m, gg))


def diag_sum_recorder(inf):
    """The capture recorder's hook: the loop the device's fused sum stands for (a ``mult`` per term, ``add_inplace``
    into the first: ``FheOps.mult_pt_sum``'s fallback, ``impl/ops.py:251-253``), each plaintext a ``Pt`` with
    ``src = ("diag", <linear>#<n>.<m>.<gg>.<jj>)``, at the ciphertext's effective level, degree 1 (``CapInf._like``),
    tagged from its full vector in every mode (``pt_tag`` reads the values: period, live, stride), and holding the
    full vector only when the recorder keeps values (``CapCt.novec`` off: the products' magnitudes are the plan's
    input); the sibling of ``examples/llama/model.py`` ``stream_sum_recorder``."""
    fhe = inf.fhe

    def rec(_ops, cts, lin, m, gg):
        jobs = filling.group_jobs(lin, m, gg)
        vecs = group_vectors(lin, m, gg)
        if len(cts) != len(jobs):
            raise ValueError(f"{lin.name} group ({m}, {gg}): {len(cts)} ciphertexts for {len(jobs)} terms")
        acc = None
        for c, (n, _m, _gg, jj), v in zip(cts, jobs, vecs, strict=True):
            full = inf._full(v)  # what CapInf._pt hands its Pt: the same float64 values
            lv, dg = inf._like(c)
            vec = np.zeros(1) if fakecap.CapCt.novec else full
            pt = fakecap.Pt(
                vec, lv, dg, fakecap.pt_tag(full), src=("diag", filling.diag_name(lin, n, m, gg, jj))
            )
            prod = fhe.mult(c, pt)
            if acc is None:
                acc = prod
            else:
                fhe.inplace_add(acc, prod)
        return acc

    return rec


def use_source(inf, source: str) -> None:
    """Set where ``inf``'s diagonal linears take their plaintexts: ``values`` removes the hook, ``generated`` installs
    the recorder's (capture recorder) or the fake's (numpy fake). The device never runs this: its programs are
    compiled from the twin's recording."""
    if source not in SOURCES:
        raise ValueError(f"diag source must be one of {SOURCES}, got {source!r}")
    if source == "values":
        inf.__dict__.pop(HOOK, None)
        return
    if isinstance(inf, fakecap.CapInf):
        setattr(inf, HOOK, diag_sum_recorder(inf))
    elif isinstance(inf, fake.FakeInf):
        setattr(inf, HOOK, diag_sum_fake)
    else:
        raise TypeError(
            f"--diag-source generated runs on the fake or the capture recorder, not on {type(inf)!r}"
        )


def source_of(inf) -> str:
    return "generated" if HOOK in getattr(inf, "__dict__", {}) else "values"


def diag_tiles(model) -> dict[str, tuple[int, int, int]]:
    """``{linear name: (d_in, d_out, t)}`` of the model's diagonal linears (``Session(diag_tiles=...)``): every
    diagonal linear of every block (five per block on the all-diagonal arms, ``wk`` and ``wr`` on ``pcmm-trunk``),
    the shape of the matrix ``LazyDiagonals`` reads (every fold applied: ``weights.block_weights``). The lane count
    rides with the shape because ``B = n_slots / t`` is the layout's, not the matrix's (a 1024 x 4096 matrix tiles as
    ``B`` = 1024, 512, ...)."""
    from .geometry import KINDS

    out: dict[str, tuple[int, int, int]] = {}
    for b in range(model.n_layers):
        bw = model.block_weights(b)
        for kind in KINDS:
            w = getattr(bw, kind)
            pts = getattr(w, "pts", None)
            if not isinstance(pts, filling.LazyDiagonals):
                continue
            if not w.name or w.name in out:
                raise RuntimeError(f"block {b} {kind}: a diagonal linear without a unique name ({w.name!r})")
            out[w.name] = (int(pts.W.shape[0]), int(pts.W.shape[1]), int(pts.t))
    return out


def tile_matrices(model):
    """``(linear name, the C-contiguous float64 (d_in, d_out) matrix)`` per diagonal linear, in ``diag_tiles`` order,
    one block's matrices built at a time (the upload, ``Session.register_tile``, copies each; 24 blocks of the model
    hold 2.625 GiB of them)."""
    from .geometry import KINDS

    for b in range(model.n_layers):
        bw = model.block_weights(b)
        for kind in KINDS:
            w = getattr(bw, kind)
            pts = getattr(w, "pts", None)
            if isinstance(pts, filling.LazyDiagonals):
                yield w.name, np.ascontiguousarray(pts.W, dtype=np.float64)


def tile_pts(model, name: str) -> filling.LazyDiagonals:
    """The ``LazyDiagonals`` of diagonal linear ``name`` (``b<b>.<kind>``): its folded matrix ``W`` (what
    ``Session.register_tile`` uploads) and geometry."""
    blk, _, kind = name.partition(".")
    if not blk.startswith("b") or not blk[1:].isdigit() or not kind:
        raise ValueError(f"{name!r} is not a diagonal linear's name (b<b>.<kind>)")
    w = getattr(model.block_weights(int(blk[1:])), kind)
    pts = getattr(w, "pts", None)
    if w.name != name or not isinstance(pts, filling.LazyDiagonals):
        raise ValueError(f"{name}: not a diagonal linear of this model ({w.name!r}, {type(pts)!r})")
    return pts


def probe_picks(model, tiles, k: int, seed: int = 0):
    """``k`` random diagonals of the declared ``tiles``: ``[(plaintext name, LazyDiagonals, (n, m, gg, jj))]``."""
    rng = np.random.default_rng(seed)
    names = sorted(tiles)
    out = []
    for _ in range(int(k)):
        name = names[int(rng.integers(len(names)))]
        pts = tile_pts(model, name)
        job = tuple(int(rng.integers(x)) for x in (pts.n_in, pts.n_out, pts.g, pts.b))
        out.append((f"{name}#{job[0]}.{job[1]}.{job[2]}.{job[3]}", pts, job))
    return out
