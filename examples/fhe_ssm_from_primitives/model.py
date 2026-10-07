"""``FheSsmModel``: the recurrent LM as an ``ImplModel`` over one session (or the fake).

A tick is one token per sequence: the client's embedding row enters as a fresh encode, the
24 block stages run with their carried state, the final RMSNorm (with ``g_out``) is the reply
the client decrypts and feeds to the tied output head in plaintext. Each block stage ends with
the canonical carry of its two state ciphertexts, so every tick is the same op stream and one
plan per stage serves them all. The state starts as encryptions of zero passed once through a
bootstrap, so the first tick enters exactly as the carry leaves every later one (a fresh
encryption would enter at degree 1 where a bootstrap lands at degree 2); the harness skips the zero
terms on a cold start instead. Each stage names its carried state for the captured graph as the C++
names a K/V cache read (``cache.<name>-lvl=<level>``), so the planner seeds it at the carry's landing
rather than at the previous stage's exit. Capture on a tick after the first, where the state is
not zero.

Block numbering: 0..n_layers-1 the blocks, n_layers the tail (final norm + gain).
"""

from __future__ import annotations

import numpy as np

from perseus.impl.driver import ImplModel

from .block import BlockState, carry, ssm_block
from .geometry import SlotGeometry
from .norm import rmsnorm
from .weights import BlockWeights, ModelParams, encode_block


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
    ):
        super().__init__(inf, core=core, unit=unit, profile=profile)
        self.params = params
        self.n_layers = int(params.n_layers if n_layers is None else n_layers)
        self.form = newton_form
        self.canonical_carry = bool(canonical_carry)
        self.resident = bool(resident_weights)
        self.geo = SlotGeometry(self.rt)
        self._bw: dict[int, BlockWeights] = {}
        self._gout = None
        self.state: list[BlockState] = []

    def block_weights(self, b):
        """The block's operands; with ``resident_weights`` encoded once and named for the
        runtime's caches, else rebuilt per use and unnamed (a CPU pass over many blocks)."""
        if not self.resident:
            return encode_block(self.params.block(b), self.geo, f"b{b}.", named=False)
        if b not in self._bw:
            self._bw[b] = encode_block(self.params.block(b), self.geo, f"b{b}.")
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
        loaded = {int(s) % self.rt.dims.N for s in self.inf.fhe.loaded_rot_steps}
        return sorted(self.geo.rotation_steps() - loaded)

    def start(self):
        """The initial state: zeros, landed by a bootstrap like every carried value (48
        bootstraps per sequence for 24 blocks, once)."""
        ops, zero = self.rt.ops, np.zeros(self.params.d)
        self.state = []
        for _ in range(self.n_layers):
            st = BlockState(ops.encode_token(zero), ops.encode_token(zero))
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
                        self.rt.ops, self.geo, h, self.block_weights(b), self.state[b], self.form
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
                n = rmsnorm(self.rt.ops, self.geo, h, self.params.norm_out, "out.norm", self.form)
                out = self.rt.ops.mult_pt(n, self.geo.lane(self.params.g_out), key="out.g")
            self.exit_stage(b, cap)
            return out

        stages = [(f"blk{b}", f"b{b}.", block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", "out.", tail_fn))
        return stages

    def step(self, row, pos):
        """One tick for one sequence: embedding row in, final-norm row out (decrypted)."""
        self.token_begin(pos)
        x = self.rt.ops.encode_token(np.asarray(row, dtype=np.float64))
        out = self.run_stages(x, pos, self.token_stages(x, pos))
        self.token_tail(pos)
        rec = self.token_end(pos)
        rec["out"] = self.rt.ops.decode_token(out)[: self.params.d]
        rec["out_level"] = self.rt.ops.level(out)
        return rec

    def levels(self):
        """(s, n_prev) levels per block: all at the landing level after a carried tick."""
        return [(self.rt.ops.level(st.s), self.rt.ops.level(st.n_prev)) for st in self.state]
