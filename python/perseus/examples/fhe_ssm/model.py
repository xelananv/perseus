# SPDX-License-Identifier: MIT
"""``FheSsmModel``: the recurrent LM as an ``ImplModel`` over one session (or the fake).

A tick is one token per conversation: fresh encode, the block stages with their carried state, the final
RMSNorm (with ``g_out``) as the reply. Each stage ends with the canonical carry of its two state
ciphertexts, so every tick is one op stream; the state starts as encryptions of zero passed once through a
bootstrap, so tick 0 enters as every later tick does; each stage names its carried state for the planner
(``cache.<name>-lvl=<level>``). The geometry's axes (``linear``, ``diag_hint``, ``pcmm_wbits``) and
``newton_form`` / ``pack_stats`` are the model's; ``newton_sites`` / ``lane_mix`` are the parameters'. Block
numbering: 0..n_layers-1 the blocks, n_layers the tail. README.md: layouts, capture, the key rule.
"""

from __future__ import annotations

import numpy as np

from perseus.impl.driver import ImplModel

from .block import PACK_STATS, BlockState, carry, norm, ssm_block
from .geometry import FilledGeometry, PairGeometry, SlotGeometry
from .norm import NEWTON_FORMS
from .scales import BlockScales
from .weights import BlockWeights, ModelParams, encode_block

LAYOUTS = ("filled", "decode")
PAYLOADS = ("pairs", "real")


class FheSsmModel(ImplModel):
    def __init__(
        self,
        inf,
        params: ModelParams,
        *,
        core=None,
        n_layers=None,
        unit=None,
        profile=False,
        newton_form="perseus",
        canonical_carry=True,
        resident_weights=True,
        layout="filled",
        payload="pairs",
        pair_route="split",
        scales=None,
        pack_stats="off",
        linear="diagonal",
        diag_hint=2,
        pcmm_wbits=16,
    ):
        super().__init__(inf, core=core, unit=unit, profile=profile)
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        if payload not in PAYLOADS:
            raise ValueError(f"payload must be one of {PAYLOADS}, got {payload!r}")
        if newton_form not in NEWTON_FORMS:
            raise ValueError(f"newton_form must be one of {NEWTON_FORMS}, got {newton_form!r}")
        if pack_stats not in PACK_STATS:
            raise ValueError(f"pack_stats must be one of {PACK_STATS}, got {pack_stats!r}")
        paired = layout == "filled" and payload == "pairs"
        if pack_stats != "off" and not (paired and pair_route == "split"):
            raise ValueError(
                "pack_stats packs the two halves of a split pair: needs filled pairs, split route"
            )
        if params.lane_mix != "off" and not paired:
            raise ValueError("the lane mix is cut for the filled layout with pairs")
        if linear != "diagonal" and layout != "filled":
            raise ValueError("PCMM linears need the filled layout")
        self.params = params
        self.n_layers = int(params.n_layers if n_layers is None else n_layers)
        self.form = newton_form
        self.pack_stats = pack_stats
        self.canonical_carry = bool(canonical_carry)
        self.resident = bool(resident_weights)
        self.layout, self.payload, self.pair_route = layout, payload, pair_route
        self.scales: list[BlockScales] | None = None if scales is None else list(scales)
        if self.scales is not None and len(self.scales) < self.n_layers:
            raise ValueError(f"{len(self.scales)} blocks of scales for {self.n_layers} blocks")
        self.geo: SlotGeometry | FilledGeometry | PairGeometry
        if layout == "decode":
            self.geo = SlotGeometry(self.rt)
        else:
            filled = FilledGeometry(
                self.rt,
                params.d,
                params.d_ffn,
                linear=linear,
                diag_hint=diag_hint,
                pcmm_wbits=pcmm_wbits,
                lane_mix=params.lane_mix != "off",
            )
            self.geo = PairGeometry(filled, pair_route) if payload == "pairs" else filled
        self._bw: dict[int, BlockWeights] = {}
        self.state: list[BlockState] = []

    @property
    def filled(self) -> FilledGeometry | None:
        """The filled geometry under the layout (None in the decode layout)."""
        if isinstance(self.geo, PairGeometry):
            return self.geo.base
        return self.geo if isinstance(self.geo, FilledGeometry) else None

    def axes(self) -> dict:
        """The axis settings this model runs under (the driver's report)."""
        f = self.filled
        return {
            "newton_form": self.form,
            "newton_sites": self.params.newton_sites,
            "pack_stats": self.pack_stats,
            "linear": None if f is None else f.linear_mode,
            "diag_hint": None if f is None else f.diag_hint,
            "pcmm_wbits": None if f is None or not f.uses_pcmm else f.pcmm_wbits,
            "lane_mix": self.params.lane_mix,
        }

    def block_weights(self, b):
        """The block's operands; with ``resident_weights`` built once and the per-channel constants
        named for the runtime's plaintext caches, else rebuilt per use and unnamed (a CPU pass over many
        blocks). The linears' plaintexts are streamed either way (``DiagLinear`` encodes at use)."""
        sc = None if self.scales is None else self.scales[b]
        if not self.resident:
            return encode_block(self.params.block(b), self.geo, f"b{b}.", named=False, scales=sc)
        if b not in self._bw:
            self._bw[b] = encode_block(self.params.block(b), self.geo, f"b{b}.", named=True, scales=sc)
        return self._bw[b]

    def preload(self):
        for b in range(self.n_layers):
            self.block_weights(b)

    def load_plans(self, plan_dir, blocks=None, validate=True, variants=()):
        """Every stage's plan by default: the blocks and the tail."""
        if blocks is None:
            blocks = range(self.n_layers + 1)
        return super().load_plans(plan_dir, blocks, validate=validate, variants=variants)

    def rotation_audit(self):
        """Rotation steps the port needs that the session's band lacks (empty = ok)."""
        n = self.rt.dims.N
        loaded = {int(s) % n for s in self.inf.fhe.loaded_rot_steps}
        return sorted({s % n for s in self.geo.rotation_steps()} - loaded)

    @property
    def capacity(self) -> int:
        """Conversations per ciphertext: 64 on pairs, 32 real, 1 in the decode layout."""
        return int(self.geo.capacity)

    def start(self):
        """The initial state: zeros, landed by a bootstrap like every carried value (48
        bootstraps per ciphertext for 24 blocks, once)."""
        ops, zero = self.rt.ops, np.zeros((1, self.params.d))
        self.state = []
        for _ in range(self.n_layers):
            st = BlockState(self.geo.encode_rows(zero), self.geo.encode_rows(zero))
            ops.bootstrap(st.s)
            ops.bootstrap(st.n_prev)
            self.state.append(st)

    def name_state(self, st: BlockState):
        """Name the stage's carried ciphertexts in the captured graph (a no-op on the fake)."""
        for name, ct in (("s", st.s), ("n_prev", st.n_prev)):
            lvl = self.rt.ops.level(ct)
            self.inf.name_ct(ct, f"{self.inf.block_prefix}cache.{name}-lvl={lvl}")

    def token_stages(self, x, pos):  # noqa: ARG002 (ImplModel's hook signature)
        def block_fn(b):
            def compute(h):
                cap = self.enter_stage(b, pos)
                self.name_state(self.state[b])
                with self.rt.step(f"blk{b}"):
                    h = ssm_block(
                        self.rt.ops,
                        self.geo,
                        h,
                        self.block_weights(b),
                        self.state[b],
                        self.form,
                        self.pack_stats,
                    )
                    if self.canonical_carry:
                        carry(self.rt.ops, self.geo, self.state[b])
                self.exit_stage(b, cap)
                return h

            return compute

        def tail_fn(h):
            b = self.n_layers
            cap = self.enter_stage(b, pos)
            with self.rt.step("tail"):
                n = norm(self.geo, h, self.params.norm_out, "out.norm", self.form, self.pack_stats)
                out = self.rt.ops.mult_pt(n, self.geo.lane(self.params.g_out), key="out.g")
            self.exit_stage(b, cap)
            return out

        stages = [(f"blk{b}", f"b{b}.", block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", "out.", tail_fn))
        return stages

    def step(self, rows, pos):
        """One tick: an embedding row per conversation in (``(n, d)``, or one ``(d,)`` row), the
        final-norm rows out, decrypted, in the same shape."""
        rows = np.asarray(rows, dtype=np.float64)
        one = rows.ndim == 1
        rows = rows[None] if one else rows
        self.token_begin(pos)
        x = self.geo.encode_rows(rows)
        out = self.run_stages(x, pos, self.token_stages(x, pos))
        self.token_tail(pos)
        rec = self.token_end(pos)
        got = self.geo.decode_rows(out, rows.shape[0])
        rec["out"] = got[0] if one else got
        rec["out_level"] = self.rt.ops.level(out)
        return rec

    def levels(self):
        """(s, n_prev) levels per block: all at the landing level after a carried tick."""
        return [(self.rt.ops.level(st.s), self.rt.ops.level(st.n_prev)) for st in self.state]
