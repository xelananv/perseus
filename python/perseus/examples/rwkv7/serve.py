# SPDX-License-Identifier: MIT
"""Continuous batching (SERVE_DESIGN.md): read a new conversation's prompt in single-conversation mode (prefill passes
of T tokens, decode steps for the remainder) while the batch waits, then JOIN it to a free lane of the running batch;
when a conversation has generated its tokens it LEAVES and its lane goes back to the public prefix state.

* **The join, state part:** the single-conversation programs write ``S_0, S_1`` straight into lane c's registers
  (``s0c<c>``, ``s1c<c>``): a register hand-over, no server arithmetic.
* **The join, carry part:** the batched pass keeps the token-shift carries in lane form (one ciphertext per carry,
  conversation c's on lane c), the single programs in natural form (every lane equal). Lane c is overwritten:
  ``xa <- xa * (1 - m_c) + xa_new * m_c``, one plaintext product per operand, one add, one refresh to land the result
  where the batched pass reads it. ``m_c`` is lane c (for c = K - 1 also every lane past K, which mirrors lane K - 1).
  The masks are plaintext INPUTS of one compiled program (re-staged per join), so one program serves every lane.
* **The leave:** lane c's S registers are replaced by fresh encryptions of the public prefix state (no program), and
  the carries by the same join program with the prefix carry as ``xa_new``; several lanes leave in ONE program run
  (the mask is the union of their lanes, the prefix carry is the same for all). A lane re-used before the next pass
  skips its reset (the join overwrites both parts anyway).

This module holds the CPU side: the join model (``Rwkv7Join``: the capture the planner cuts the join plan from), the
numpy / fake servers and ``serve_loop``, the scheduler the device runner (``run_planned --serve``) shares.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

import numpy as np

from perseus.impl.driver import ImplModel

from .layout import Layout, auto_lanes

# ── the join, as arithmetic ──

def lane_mask_of(L, K: int, lanes) -> np.ndarray:
    """1 on the given conversations' lanes (lane K - 1 also takes every lane past K), 0 elsewhere."""
    lanes = sorted({int(c) for c in lanes})
    if not lanes or lanes[0] < 0 or lanes[-1] >= K:
        raise ValueError(f"lanes {lanes} outside 0..{K - 1}")
    sel = set(lanes)
    if K - 1 in sel:
        sel |= set(range(K, L.t))
    return L.lane_mask(sorted(sel))


def join_masks(L, K: int, lanes) -> tuple[np.ndarray, np.ndarray]:
    """(m, 1 - m) for lane(s) ``lanes``."""
    m = lane_mask_of(L, K, [lanes] if np.isscalar(lanes) else lanes)
    return m, 1.0 - m


def join_carry(ops, lane_ct, new_ct, m, keep):
    """``lane * (1 - m) + new * m``: exact in float64 for 0/1 masks (each lane keeps one operand bit for bit)."""
    return ops.add(ops.mult_pt(lane_ct, keep), ops.mult_pt(new_ct, m))


def carry_keys(Sh: int) -> list[str]:
    """The carried token-shift keys of one block, in ``State.cts()`` order: ``xa``, ``xah1``, ..., ``xf``, ..."""
    sfx = [""] + [f"h{j}" for j in range(1, Sh)]
    return [f"xa{x}" for x in sfx] + [f"xf{x}" for x in sfx]


# ── the join program's twin (the capture) ──

class Rwkv7Join(ImplModel):
    """ONE stage: every block's lane-form carries and one conversation's natural-form carries in (all named as carried
    state, ``cache.<key>-lvl=<level>``: ``b<b><carry>`` for the lane form, ``b<b>n<carry>`` for the natural form), the
    joined carries out, each landed by a deliberate refresh in a ``carry`` scope (the batched pass's own landing).
    The capture records lane ``lane``'s masks; on the device they are re-staged per join (run_planned.Rwkv7JoinRun)."""

    def __init__(self, inf, C: int, d_ffn: int, n_layers: int, K: int, *, core=None, unit=None, lanes=0, lane=0):
        super().__init__(inf, core=core, unit=unit)
        lanes = int(lanes) or auto_lanes(C, self.rt.dims.N)
        self.L = Layout(self.rt.dims.N, C, d_ffn, D=self.rt.dims.N // lanes)
        if not 2 <= int(K) <= self.L.t:
            raise ValueError(f"batch K = {K} must be 2..{self.L.t}")
        self.n_layers, self.K = int(n_layers), int(K)
        ck = carry_keys(self.L.S)
        self.keys = [f"b{b}{k}" for b in range(self.n_layers) for k in ck]          # lane form: in and out
        self.new_keys = [f"b{b}n{k}" for b in range(self.n_layers) for k in ck]     # natural form: in
        self.set_lane(lane)

    def set_lane(self, lanes):
        self.lane = lanes
        self.m, self.keep = join_masks(self.L, self.K, lanes)

    def linears(self) -> dict:
        return {}

    def rotation_steps(self) -> set[int]:
        return set()

    def state_names(self) -> list[str]:
        return []

    def start(self, state=None):
        pass

    def encode_input(self, rows):
        """``rows``: the ``pack_tokens`` rows of every input in ``keys + new_keys`` order, each landed by a refresh (as
        every carried value is)."""
        ops = self.rt.ops
        if len(rows) != 2 * len(self.keys):
            raise ValueError(f"{len(rows)} input rows, {2 * len(self.keys)} expected")
        out = []
        for r in rows:
            c = ops.pack_tokens(r, ops.bts_out())
            ops.bootstrap(c)
            out.append(c)
        return out

    def token_stages(self, x, pos):
        ops = self.rt.ops

        def compute(xs):
            cap = self.enter_stage(0, pos)
            for key, ct in zip(self.keys + self.new_keys, xs):
                self.inf.name_ct(ct, f"{self.inf.block_prefix}cache.{key}-lvl={ops.level(ct)}")
            n = len(self.keys)
            out = []
            with self.rt.step("join"):
                for lane, new in zip(xs[:n], xs[n:]):
                    y = join_carry(ops, lane, new, self.m, self.keep)
                    with self.rt.step("carry"):
                        ops.bootstrap(y)
                    out.append(y)
            self.exit_stage(0, cap)
            return out
        return [("join", None, compute)]

    def step(self, rows, pos):
        self.token_begin(pos)
        x = self.encode_input(rows)
        out = self.run_stages(x, pos, self.token_stages(x, pos))
        self.token_tail(pos)
        return out


def make_join(P, K: int, *, capture=False, n_layers=None, lanes=0, lane=0, levels="preset"):
    """(fhe, inf, Rwkv7Join) on the fake session (``capture``: the recorder) at the model's dims."""
    from .session import LOGN, make_session
    lanes = int(lanes) or auto_lanes(P.C, 1 << (LOGN - 1))
    fhe, inf, core = make_session(capture=capture, levels=levels, lanes=lanes)
    n = int(P.n_layer if n_layers is None else n_layers)
    m = Rwkv7Join(inf, P.C, P.d_ffn, n, K, core=core, lanes=lanes, lane=lane)
    fhe.loaded_rot_steps, fhe._rot = [], set()
    return fhe, inf, m


def join_rows(L, n_layers: int, lane_state, new_state) -> list[np.ndarray]:
    """The join twin's input rows (``keys + new_keys`` order) from two ``ref.State``: the lane-form carries (every lane
    the same value, as the public prefix leaves them) and a conversation's natural-form carries."""
    def rows(st):
        out = []
        for b in range(n_layers):
            out += L.encode_row_sh(st.xa[b]) + L.encode_row_sh(st.xf[b])
        return out
    return rows(lane_state) + rows(new_state)


def new_carry_rows(L, n_layers: int, state) -> dict:
    """``{(block, carry key): pack rows}``: a natural-form carry set (the public prefix's, for a leave)."""
    ck = carry_keys(L.S)
    out = {}
    for b in range(n_layers):
        for k, r in zip(ck, L.encode_row_sh(state.xa[b]) + L.encode_row_sh(state.xf[b])):
            out[(b, k)] = r
    return out


# ── the CPU servers: the same join / leave on the numpy mirror and on the fake ──

class MirrorBackend:
    """Numpy slots (no levels, no noise): ``Mirror`` for decode / prefill, ``BatchedMirror`` (lane carries) for the
    batched pass."""

    def __init__(self, P, coeffs, state, n_slots, K, *, n_layers=None, wbits=None, lanes=0, lora="auto",
                 fuse_l1=False, layout="current"):
        from .batched import BatchedMirror
        from .model import Mirror
        self.st = state
        self.single = Mirror(P, coeffs, n_slots, n_layers=n_layers, wbits=wbits, lora=lora, lanes=lanes,
                             fuse_l1=fuse_l1, layout=layout)
        self.bm = BatchedMirror(P, coeffs, n_slots, K, carry="lane", n_layers=n_layers, wbits=wbits, lora=lora,
                                lanes=lanes, fuse_l1=fuse_l1, layout=layout)
        self.L, self.ops = self.single.L, self.single.circ.ops

    def fresh_single(self):
        self.single.reset(self.st)
        return self.single.state

    def fresh_batch(self):
        self.bm.reset(self.st)
        return self.bm.state

    def prefill(self, sstate, rows):
        self.single.state = sstate
        return self.single.prefill(rows)[0]

    def decode(self, sstate, row):
        self.single.state = sstate
        return self.single.step(row)[0]

    def batch(self, bstate, rows):
        self.bm.state = bstate
        return self.bm.step(rows)[0]

    def land(self, ct):
        return ct


class FakeBackend:
    """One ``Rwkv7Fhe`` on the fake (or the capture recorder) switched between its three programs (``set_mode``);
    build it with ``batch=K, batch_carry="lane", prefill_keys=T``."""

    def __init__(self, model, state, T):
        if model.batch < 2 or model.batch_carry != "lane":
            raise ValueError("the fake server needs a model built with batch=K >= 2 and batch_carry='lane'")
        self.m, self.st, self.T, self.K = model, state, int(T), model.batch
        self.L, self.ops = model.L, model.rt.ops
        self.pos = 1

    def _step(self, rows, state, **mode):
        self.m.set_mode(**mode)
        self.m.state = state
        h, _o = self.m.step(rows, self.pos)
        self.pos += 1
        return h

    def fresh_single(self):
        self.m.set_mode()
        self.m.start(self.st)
        return self.m.state

    def fresh_batch(self):
        self.m.set_mode(batch=self.K)
        self.m.start(self.st)
        return self.m.state

    def prefill(self, sstate, rows):
        return self._step(list(rows), sstate, chunk=len(rows))

    def decode(self, sstate, row):
        return self._step(row, sstate)

    def batch(self, bstate, rows):
        return self._step(rows, bstate, batch=self.K)

    def land(self, ct):
        self.ops.bootstrap(ct)
        return ct


class Server:
    """The serving operations on a CPU backend, with the device runner's semantics (``serve_loop`` drives it). Every
    method returns ``(hidden, None)`` (the device's ``TeeServer`` returns the mirror's hidden as the second item).
    ``mask_lane`` and ``leave_resets`` exist for the tests' sabotage (a join onto the wrong lane, a leave that does
    not reset)."""

    def __init__(self, backend, K: int):
        self.be, self.K = backend, int(K)
        self.bstate = backend.fresh_batch()
        self.single = None
        self.mask_lane = lambda c: c
        self.leave_resets = True
        self.counts = collections.Counter()

    def prefill(self, c, rows, first):
        if first:
            self.single = self.be.fresh_single()
        self.counts["prefill"] += 1
        return self.be.prefill(self.single, rows), None

    def decode(self, c, row, first):
        if first:
            self.single = self.be.fresh_single()
        self.counts["decode"] += 1
        return self.be.decode(self.single, row), None

    def _land(self, lanes, single, mask_lanes):
        ops, L = self.be.ops, self.be.L
        m, keep = join_masks(L, self.K, mask_lanes)
        for bs, ss in zip(self.bstate, single):
            for c in lanes:
                bs.S[c] = list(ss.S)                       # the register hand-over (the device writes them in place)
            bs.xa = [self.be.land(join_carry(ops, x, n, m, keep)) for x, n in zip(bs.xa, ss.xa)]
            bs.xf = [self.be.land(join_carry(ops, x, n, m, keep)) for x, n in zip(bs.xf, ss.xf)]

    def join(self, c):
        self.counts["join"] += 1
        self._land([c], self.single, [self.mask_lane(c)])
        self.single = None

    def leave(self, lanes):
        if not self.leave_resets:
            return
        self.counts["leave"] += 1
        for c in lanes:                                    # S: a fresh public prefix state per lane (no program)
            for bs, ss in zip(self.bstate, self.be.fresh_single()):
                bs.S[c] = list(ss.S)
        fresh = self.be.fresh_single()                     # the carries: ONE join run, the union of the lanes
        ops, L = self.be.ops, self.be.L
        m, keep = join_masks(L, self.K, lanes)
        for bs, ss in zip(self.bstate, fresh):
            bs.xa = [self.be.land(join_carry(ops, x, n, m, keep)) for x, n in zip(bs.xa, ss.xa)]
            bs.xf = [self.be.land(join_carry(ops, x, n, m, keep)) for x, n in zip(bs.xf, ss.xf)]

    def batch(self, rows):
        self.counts["batch"] += 1
        return self.be.batch(self.bstate, np.asarray(rows)), None


# ── the scheduler ──

@dataclass
class Conv:
    k: int
    arrival: int
    prompt: list
    toks: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    modes: list = field(default_factory=list)
    lane: int | None = None
    joined: int | None = None
    left: int | None = None
    n_batch: int = 0

    def __post_init__(self):
        self.toks = list(self.prompt)


def serve_loop(server, arrivals, *, K: int, T: int, generate: int, emb, pick, idle_token: int, record=None,
               emit=None, defer_leave=True):
    """Runs a stream of conversations through ``server`` (``Server`` on the CPU, the device runner's on the GPU).

    ``arrivals``: [(arrival pass, prompt ids after the public prefix)]. At the start of batched pass q every
    conversation that has arrived (arrival <= q, in arrival order) takes a free lane while one is free: its prompt is
    read in single-conversation mode while the batch waits (floor(n / T) prefill passes, then decode steps; T = 0:
    decode steps only), its first generated token is the client's pick from the last prompt position, and it joins.
    Each batched pass then advances every lane by one token; a conversation that has fed ``generate`` tokens to the
    batch (records: n prompt positions + ``generate``, as the decode runner's) leaves after that pass. Idle lanes run
    from the public prefix state on ``idle_token``, then on their own picks (in range: one runaway lane would spoil
    the shared ciphertexts' refreshes); they are never scored. With nothing to run, the pass clock jumps to the next
    arrival (no pass is spent).

    ``pick(hidden) -> token id`` (the client's greedy pick); ``record(conv, i, token, hidden, aux, mode, lane)`` per
    scored position; ``emit(dict)`` per event (prefill / decode / join / leave / pass, with seconds). Returns
    (convs, idle): idle[c] is lane c's list of idle segments, each [(pass, token, hidden)] since a reset."""
    if generate < 1:
        raise ValueError("--generate >= 1: a conversation joins the batch to generate")
    emit = emit or (lambda _r: None)
    convs = [Conv(k, int(p), [int(t) for t in ids]) for k, (p, ids) in enumerate(arrivals)]
    if any(not c.prompt for c in convs):
        raise ValueError("an empty prompt (nothing after the public prefix)")
    order = sorted(range(len(convs)), key=lambda k: (convs[k].arrival, k))
    lanes: list = [None] * K
    nxt = [int(idle_token)] * K
    idle = {c: [[]] for c in range(K)}
    pending: set = set()
    queue: collections.deque = collections.deque()
    ia, q = 0, 0

    def rec(cv, tok, h, aux, mode, lane):
        i = len(cv.hidden)
        cv.hidden.append(np.asarray(h))
        cv.modes.append(mode)
        if record is not None:
            record(cv, i, int(tok), h, aux, mode, lane)

    def admit(k, c):
        cv = convs[k]
        n = len(cv.prompt)
        n_pf = (n // T) * T if T else 0
        first = True
        for p in range(n_pf // T if T else 0):
            chunk = cv.prompt[p * T:(p + 1) * T]
            t0 = time.time()
            h, aux = server.prefill(c, [emb[t] for t in chunk], first)
            emit({"kind": "prefill", "conv": k, "lane": c, "at_pass": q, "chunk": p, "T": T,
                  "seconds": round(time.time() - t0, 4)})
            for l, tok in enumerate(chunk):
                rec(cv, tok, h[l], None if aux is None else aux[l], "prefill", c)
            first = False
        for i in range(n_pf, n):
            t0 = time.time()
            h, aux = server.decode(c, emb[cv.prompt[i]], first)
            emit({"kind": "decode", "conv": k, "lane": c, "at_pass": q, "pos": i, "seconds": round(time.time() - t0, 4)})
            rec(cv, cv.prompt[i], h, aux, "decode", c)
            first = False
        g1 = int(pick(cv.hidden[-1]))
        t0 = time.time()
        server.join(c)
        emit({"kind": "join", "conv": k, "lane": c, "at_pass": q, "seconds": round(time.time() - t0, 4)})
        pending.discard(c)
        lanes[c], nxt[c] = k, g1
        cv.toks.append(g1)
        cv.lane, cv.joined = c, q

    while True:
        while ia < len(order) and convs[order[ia]].arrival <= q:
            queue.append(order[ia])
            ia += 1
        while queue and None in lanes:
            c = lanes.index(None)
            if idle[c][-1]:
                idle[c].append([])                         # the lane's idle segment ends at the admission
            admit(queue.popleft(), c)
        if all(x is None for x in lanes):
            if ia >= len(order):
                break
            q = max(q + 1, convs[order[ia]].arrival)       # nothing to run: the clock jumps to the next arrival
            continue
        if pending:
            t0 = time.time()
            done = sorted(pending)
            server.leave(done)
            emit({"kind": "leave", "lanes": done, "at_pass": q, "seconds": round(time.time() - t0, 4)})
            pending.clear()
        rows = np.stack([emb[nxt[c]] for c in range(K)])
        t0 = time.time()
        H, aux = server.batch(rows)
        sec = time.time() - t0
        active = [c for c in range(K) if lanes[c] is not None]
        emit({"kind": "pass", "pass": q, "seconds": round(sec, 4), "active": len(active),
              "lanes": [lanes[c] for c in range(K)]})
        for c in range(K):
            pk = int(pick(H[c]))
            k = lanes[c]
            if k is None:
                idle[c][-1].append((q, nxt[c], np.asarray(H[c])))
                nxt[c] = pk
                continue
            cv = convs[k]
            rec(cv, nxt[c], H[c], None if aux is None else aux[c], "batch", c)
            cv.n_batch += 1
            if cv.n_batch >= generate:
                lanes[c], cv.left, nxt[c] = None, q, int(idle_token)
                idle[c].append([])
                if defer_leave:
                    pending.add(c)
                else:
                    t0 = time.time()
                    server.leave([c])
                    emit({"kind": "leave", "lanes": [c], "at_pass": q, "seconds": round(time.time() - t0, 4)})
            else:
                nxt[c] = pk
                cv.toks.append(pk)
        q += 1
    for c in idle:
        idle[c] = [s for s in idle[c] if s]
    return convs, idle


def idle_token(P, coeffs_path, prefix_ids) -> int:
    """An idle lane's first token: the poly model's greedy pick after the public prefix (public, in distribution)."""
    from . import ref
    m = ref.RefModel(P, ref.Sites("poly", coeffs_path, clamp=True))
    lr = None
    for t in prefix_ids:
        lr, _h = m.step(int(t))
    return int(np.argmax(lr))


def read_arrivals(path, prefix_text="\n\n") -> list[tuple[int, str]]:
    """``--serve FILE``: one conversation per line, ``<arrival pass><TAB or space><prompt>``; ``\\n`` escapes a newline
    (``\\\\`` a backslash), ``#`` starts a comment line. A prompt without the public prefix gets it prepended."""
    out = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        sep = "\t" if "\t" in line else " "
        head, _, s = line.partition(sep)
        try:
            p = int(head)
        except ValueError:
            raise ValueError(f"{path}:{n}: no arrival pass before the prompt ({head!r})") from None
        if p < 0:
            raise ValueError(f"{path}:{n}: arrival pass {p} < 0")
        s = s.replace("\\\\", "\0").replace("\\n", "\n").replace("\0", "\\")
        out.append((p, s if s.startswith(prefix_text) else prefix_text + s))
    return out
