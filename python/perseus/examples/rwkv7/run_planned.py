# SPDX-License-Identifier: MIT
"""The encrypted decode on a device, under a plan (``impl.driver.PlannedRun``): every block a compiled program,
replayed per token; one conversation, one token per step (the prompt too).

The CPU twin (the model on the capture recorder, the same flags and shifts as the capture the plan was cut from)
records two probe tokens on a PUBLIC row (the embedding of token 0; the stream does not depend on the values), the
reference token's stream (token 1, which reads a carried state) is compiled against the plan, the carried state of
every block (``S_0, S_1, x_prev`` of both mixes) becomes registers, every plaintext is staged once (by the digest of
its values), the PCMM planes are uploaded by name. One untimed warm-up token (public row) captures the CUDA graphs;
then the decode from zero state: per token the row is encrypted at block 0's compiled input level, every block runs,
the reply is decrypted, the client applies ``ln_out`` and the head, and the token is scored against the plaintext
polynomial model (``ref.RefModel`` in poly mode, clamp on) and the circuit's numpy mirror.

KEYS (``--keys``): ``inproc`` (the default) generates every key in the session process, which then holds the secret:
a correctness and timing run, recorded as ``keys: insecure-inproc``. ``bundle`` / a directory: the client's key
bundle (``keys.pckb``, made by ``--keygen`` from the server's ``--keyspec`` request), opened with
``Session.from_bundle``: the session holds no secret (``has_secret: false`` in the header); the client's secret
(``client_secret.bin`` beside the bundle, a separate ``_pc.Client`` object the session never sees) encrypts each
input and decrypts each reply as wire frames (fhe_ssm's 11-02 publication path).

Widths above 1,024 (ANYSIZE_DESIGN.md): every stage reads and writes the ``S`` shard ciphertexts of ``z``; the
runtime chains only a block's first output to the next block's entry (``compile_dir``), so the other shards are
compiled at their recorded (level, deg) (``input_metas``, as the carried state) and each block's output shards are
checked against the next block's inputs before the session opens.

    python -m perseus.examples.rwkv7.run_planned --pth P --coeffs C --harness H --plan PLAN --shifts S \\
        --tokens 12 --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

from perseus.examples.fhe_ssm.planned import N32_Q_PRIMES, SsmRecorder, digest

from perseus.impl.driver import PlannedRun, plan_rot_scope, register_levels

from . import common
from .model import Mirror
from .run_capture import model_kw
from .session import make_model


class Recorder(SsmRecorder):
    """The twin's recorder with values always on (PCMM's fake multiplies the slot vectors)."""

    @property
    def values(self):
        return True

    @values.setter
    def values(self, _v):
        pass


class Rwkv7PlannedRun(PlannedRun):
    def __init__(self, recorder, plan_dir, config, labels, keys, *, model, work_dir, pcmm_cf=6, pt_rows=28,
                 stc_cache=8, pool=16, log=print, schedule="program", pcmm_cf_map=None, bundle=None, client=None):
        super().__init__(recorder, plan_dir, config, labels, keys, ref_token=1, probe=2, work_dir=work_dir,
                         pt_rows=pt_rows, dyn_rot=[], stc_cache=stc_cache, pool=pool, log=log)
        self.model, self.pcmm_cf = model, int(pcmm_cf)
        self.schedule = str(schedule)
        self.pcmm_cf_map = None if pcmm_cf_map is None else {str(k): int(v) for k, v in pcmm_cf_map.items()}
        self.ops = None
        self.static = False
        self.bundle, self.client = bundle, client          # a client's key bundle file; its secret (_pc.Client)
        self.zin: dict = {}                                 # block -> the program input index of each z shard

    def shard_inputs(self, r) -> list[int]:
        """The program input index of each of the stage's z shards (the recorder's stage input objects)."""
        xin = r.x_in if isinstance(r.x_in, (list, tuple)) else [r.x_in]
        ins = r.inputs()
        idx = []
        for o in xin:
            hit = [i for i, n in enumerate(ins) if n in r.objs and r.objs[n] is o]
            if len(hit) != 1:
                raise RuntimeError(f"a stage input shard is read {len(hit)} times as a program input")
            idx.append(hit[0])
        if idx[0] != 0:
            raise RuntimeError(f"the first z shard is program input {idx[0]}, not 0 (the runtime chains input 0)")
        return idx

    def _compile_with_schedule(self, fn):
        """``PlannedRun.prepare`` calls ``_pc.Program.compile`` without a schedule and with the carried state's input
        metas only: run ``fn`` with the module's ``Program`` standing in for one whose ``compile`` passes
        ``schedule`` (siblings: sibling ops, refresh pairs under PAIR=1 included, batched into one launch; fhe_ssm's
        ``--schedule``) and adds the recorded (level, deg) of every z shard after the first (S > 1)."""
        from perseus import _pc
        model = getattr(self, "model", None)
        S = getattr(getattr(model, "L", None), "S", 1)
        if self.schedule == "program" and S == 1:
            return fn()
        real, sched, run = _pc.Program, self.schedule, self

        class _Scheduled:
            @staticmethod
            def compile(*a, **k):
                if S > 1:
                    metas = {lab: dict(v) for lab, v in (k.get("input_metas") or {}).items()}
                    for b, r in run.ref.items():
                        lab, ins = run.labels[b], r.inputs()
                        for i in run.shard_inputs(r)[1:]:
                            metas.setdefault(lab, {})[ins[i]] = r.metas[ins[i]]
                    k["input_metas"] = metas
                if sched != "program":
                    k["schedule"] = sched
                return real.compile(*a, **k)
        _pc.Program = _Scheduled
        try:
            return fn()
        finally:
            _pc.Program = real

    def check_shard_chain(self):
        """S > 1: block b's output shard s must land at the (level, deg) block b + 1 compiled for its input shard s
        (``Program.run`` refuses a register otherwise); stops before the session opens, naming the shards."""
        bad = []
        for b in sorted(self.ref):
            if b == 0 or b not in self.zin:
                continue
            po, pi = self.progs[self.labels[b - 1]], self.progs[self.labels[b]]
            for s, i in enumerate(self.zin[b]):
                j = self.reg[b - 1]["out"][s][0]
                w, rd = tuple(po.outputs[j][1:3]), tuple(pi.inputs[i][1:3])
                if w[:2] != rd[:2]:
                    bad.append(f"block {b - 1} writes shard {s} at {w}, block {b} reads {rd}")
        if bad:
            raise RuntimeError("the z shards do not chain: " + "; ".join(bad[:6]) +
                               " (run with the shard landing on: no --no-shard-land)")

    # the store: one slot per distinct (value digest, level)
    def pt_aliases(self):
        out = {}
        for b in sorted(self.ref):
            lab, r = self.labels[b], self.ref[b]
            for nm, _lv, _d, f in self.progs[lab].pt_refs:
                if f in ("stream", "diag"):
                    continue
                out[(lab, nm)] = "v/" + digest(np.asarray(r.pts[nm]))
        return out

    def prepare(self, rows, argmax=False):
        self._compile_with_schedule(lambda: PlannedRun.prepare(self, rows, argmax))
        if any(self.dyn.values()):
            raise RuntimeError(f"dynamic steps in the stream: {self.dyn}")
        for b, r in self.ref.items():
            o = self.records[0][b]
            if set(o.pts) != set(r.pts) or any(not np.array_equal(o.pts[k], r.pts[k]) for k in r.pts):
                raise RuntimeError(f"block {b}: a plaintext differs between the probe tokens (not static)")
        n_state = len(self.model.state_names())
        bad = [b for b in range(self.rec.n_layers) if len(self.kv.get(b, ())) != n_state]
        if bad:
            raise RuntimeError(f"blocks {bad}: the carried state was not recognised as block inputs")
        self.zin = {b: self.shard_inputs(r) for b, r in self.ref.items()}
        bad = {b: len(z) for b, z in self.zin.items() if len(z) != self.model.L.S}
        if bad:
            raise RuntimeError(f"stages {bad}: not {self.model.L.S} z shards in")
        for b, r in self.ref.items():
            if len(r.out_names) != self.model.L.S:
                raise RuntimeError(f"stage {b}: {len(r.out_names)} outputs, {self.model.L.S} z shards expected")
        self.check_shard_chain()
        self.static = True
        self.al = self.pt_aliases()
        return self

    def session_kwargs(self):
        kw = super().session_kwargs()
        D = getattr(self.model.L, "D", 1024)
        kw["size"] = {"hidDim": D, "dim": D, "numHeads": 1, "numHeadsReal": 1, "expDim": 3 * D, "expanded": 3 * D,
                      "seqLen": 1}
        lins = self.model.linears()
        kw["pcmm"] = [(n, int(l.t), int(l.n_in), int(l.n_out), int(l.Wd.shape[0]), float(l.w_scale), int(l.shift))
                      for n, l in lins.items()] or None
        kw["pcmm_cf"] = self.pcmm_cf
        if self.pcmm_cf_map is not None:
            # per-linear CFs (fhe_ssm/pcmm_cf_table.py from the capture); a linear not in the table at pcmm_cf
            kw["pcmm_cf_map"] = {n: c for n, c in self.pcmm_cf_map.items() if n in lins}
            self.log(f"[planned] PCMM CF table: {len(kw['pcmm_cf_map'])} of {len(lins)} linears, "
                     f"CFs {sorted(set(kw['pcmm_cf_map'].values()))}, the rest at {self.pcmm_cf}")
        pairs = {(self.al[(lab, nm)], int(lv)) for lab, p in self.progs.items()
                 for nm, lv, _d, f in p.pt_refs if f not in ("stream", "diag")}
        kw["pt_levels"] = [lv for _a, lv in sorted(pairs)]
        kw["pt_slots"] = len(kw["pt_levels"]) + 8
        if kw["pt_levels"] and self.pt_rows:
            kw["pt_rows"] = max(self.pt_rows, N32_Q_PRIMES - min(kw["pt_levels"]))
        return kw

    def setup(self, rows, argmax=False):
        from perseus import _pc
        from perseus.impl.ops import FheOps
        super().setup(rows, argmax)
        self.ops = FheOps(self.sess.fhe, _pc, self.model.rt.ops.unit)
        for name, lin in self.model.linears().items():
            self.sess.pcmm_upload(name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        n = int(self.store.stage([(nm, lv, v) for (nm, lv), v in self.pt_items().items()]))
        self.log(f"[planned] {len(self.model.linears())} PCMM linears uploaded, {n} plaintext slots staged")
        return self

    def pt_items(self) -> dict:
        """Every plaintext under its shared store name ``{(alias, level): values}``; aliases the programs to it."""
        items = {}
        for b in sorted(self.ref):
            lab, r = self.labels[b], self.ref[b]
            amap = {}
            for nm, lv in self.progs[lab].pt_keys():
                pt = nm.split("/", 1)[1]
                a = self.al[(lab, pt)]
                amap[nm] = a
                v = np.asarray(r.pts[pt])
                if np.iscomplexobj(v):          # every plaintext of this model is real (masks, constants)
                    if np.abs(v.imag).max() > 0:
                        raise RuntimeError(f"{lab} {pt}: a complex plaintext")
                    v = v.real
                items.setdefault((a, int(lv)), np.ascontiguousarray(v, dtype=np.float64))
            self.progs[lab].alias_pts(amap)
            self._alias.update(amap)
        return items

    def encrypt_input(self, rows, p, i, what):
        """``pack_tokens`` rows ``(t, D)`` (row ``tok`` at slots ``ch * t + tok``) encrypted at program input ``i``'s
        compiled level AND degree: ``Program.run`` refuses an input Ct whose (level, deg) differ from the compiled
        ones (crates/pc-py/src/program.rs, run_block), and ``pack_tokens`` always encrypts at degree 1. A fresh
        encryption's scale is ``sf(level)^deg``; a compiled scale that differs would scale the input silently, so it
        is checked here (an unknown compiled scale, 0, is taken as the Ct's)."""
        _nm, lvl, deg, nf = p.inputs[i]
        lvl, deg, nf = int(lvl), int(deg), float(nf)
        r = np.asarray(rows, dtype=np.float64)
        v = np.ascontiguousarray(r.T.ravel())               # slot ch * t + tok, as pack_rows lays it out
        ct = Rwkv7PlannedRun.encrypt(self, v, lvl, deg)
        got = (int(ct.level), int(ct.deg), float(ct.nf))
        if got[:2] != (lvl, deg) or (nf > 0 and abs(np.log2(got[2] / nf)) > 1e-6):
            raise RuntimeError(f"{what}: encrypted at (level, deg, nf) {got}, compiled ({lvl}, {deg}, {nf})")
        return ct, (lvl, deg, nf, got[2])

    # ── the client's end: encryption and decryption ──
    def encrypt(self, v, level, deg):
        """A fresh encryption of the slot vector ``v`` at (level, deg): by the client (``_pc.Client``: its secret,
        a wire frame imported into the session) when a client secret is at hand, else by the session (its secret
        under inproc keys, the bundle's public key otherwise)."""
        if getattr(self, "client", None) is not None:
            from perseus import _pc
            frame = bytes(self.client.encrypt(np.asarray(v, dtype=np.complex128), int(level), int(deg)))
            return _pc.import_ct(self.sess, frame)
        return self.fhe.encrypt(v, level, deg)

    def decrypt(self, ct) -> np.ndarray:
        if self.client is not None:
            from perseus import _pc
            return np.asarray(self.client.decrypt(bytes(_pc.export_ct(self.sess, ct)))[0], dtype=np.complex128)
        if self.bundle is not None:
            raise RuntimeError("a bundle session holds no secret and no client secret was given (client_secret.bin)")
        return np.asarray(self.ops.decrypt_slots_complex(ct))

    def open_session(self, kw):
        from perseus import _pc
        if self.bundle is None:
            return _pc.Session(self.cfg, self.keys, **kw)
        sess = _pc.Session.from_bundle(self.cfg, self.keys, str(self.bundle), **kw)
        if sess.has_secret:
            raise RuntimeError("the bundle session holds a secret")
        return sess

    def key_mode(self) -> str:
        return "insecure-inproc" if self.bundle is None else "bundle"

    def input_rows(self, t, rows):
        """Block 0's ``pack_tokens`` rows per shard for token t (decode: the row on every lane)."""
        return self.model.L.encode_row_sh(rows[t])

    def input_ct(self, t, rows, p):
        return [self.encrypt_input(r, p, i, f"block 0 input shard {s}")[0]
                for s, (r, i) in enumerate(zip(self.input_rows(t, rows), self.zin[0]))]

    def run_stage(self, b, t, xs, kv_cts=None, run=None):
        """Block b on the device: its z shards ``xs`` (Cts or registers) at their program inputs, the carried state
        from the registers (``None`` at t = 0: empty), or from ``kv_cts[b]`` (the public prefix state)."""
        run = run or self
        p = run.progs[run.labels[b]]
        ins = [None] * len(p.inputs)
        for i, x in zip(run.zin[b], xs):
            ins[i] = x
        for i, _j, r in run.reg[b]["kv"]:
            ins[i] = kv_cts[b][i] if kv_cts is not None and b in kv_cts else (None if t == 0 else r)
        return p.run(ins, [(j, r) for _i, j, r in run.reg[b]["kv"]] + list(run.reg[b]["out"]))

    def out_regs(self, b, run=None):
        run = run or self
        return [r for _j, r in run.reg[b]["out"]]

    def state_cts(self, state) -> dict:
        """The public prefix state as each block's carried inputs ``{block: {input position: Ct}}``, each a fresh
        encryption at the compiled input's level and degree (``encrypt_input``). The twin lands its state by a
        bootstrap (``Model.start``); on a planned session an eager bootstrap is refused without ``debug_refresh``
        (``Fhe.bootstrap``) and would land at the landing's meta, not the input's, so the device encrypts in place of
        it: same values, less noise. Untimed setup: the state is public."""
        names = self.model.state_names()
        rows = self.model.state_rows(state)
        out, notes = {}, []
        for b in range(self.rec.n_layers):
            p = self.progs[self.labels[b]]
            cts = {}
            for i, _j, _r in self.reg[b]["kv"]:
                key = re.search(r"cache\.([a-z0-9]+)-lvl", self.ref[b].inputs()[i]).group(1)
                cts[i], m = self.encrypt_input(rows[b][names.index(key)], p, i, f"block {b} state {key}")
                notes.append((b, key, *m))
            out[b] = cts
        self.log(f"[planned] prefix state: (block, name, level, deg, compiled nf, ct nf) {notes[:6]} ... "
                 f"({len(notes)} inputs)")
        return out

    def token(self, t, rows, argmax=False, entry=None, variant=None, kv_cts=None):
        """Every stage on the device; the carried inputs from the registers, or from ``kv_cts`` (``state_cts``)
        on the first token of a sequence that starts from a public state."""
        tt = time.time()
        n_blocks = self.rec.n_layers + 1
        for b in range(n_blocks):
            xs = self.input_ct(t, rows, self.progs[self.labels[0]]) if b == 0 else self.out_regs(b - 1)
            self.run_stage(b, t, xs, kv_cts)
        return self.out_regs(n_blocks - 1), {"issue_s": round(time.time() - tt, 4)}

    def decrypt_regs(self, regs):
        cts = [self.ct(r) for r in regs]
        return [np.real(self.decrypt(c)) for c in cts], int(cts[0].level)

    def reply(self, regs):
        vals, level = self.decrypt_regs(regs)
        return self.model.L.decode_sh(vals), level


class Rwkv7BatchedRun(Rwkv7PlannedRun):
    """K conversations per pass (batched.py) under a plan cut from ``run_capture --batch K``: the input is K rows,
    row c on lane c; the reply is decrypted per lane; every conversation's carried state is its own registers
    (``Rwkv7Fhe.state_names``: ``s0c<c>``, ``s1c<c>`` and the carries)."""

    def input_rows(self, t, rows):
        return self.model.L.encode_rows_sh(np.asarray(rows[t]))

    def reply(self, regs):
        vals, level = self.decrypt_regs(regs)
        return self.model.L.decode_lanes_sh(vals, self.model.batch), level


def read_prompts(path, prefix_text="\n\n") -> list[str]:
    """One prompt per line; ``\\n`` escapes a newline (``\\\\`` a backslash). A prompt that does not start with the
    public prefix gets it prepended."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        s = line.replace("\\\\", "\0").replace("\\n", "\n").replace("\0", "\\")
        out.append(s if s.startswith(prefix_text) else prefix_text + s)
    return out


def main_batch(a) -> int:
    """``--batch K --prompts FILE``: K conversations per pass, conversation c on lane c. Each reads its own prompt
    token by token after the public prefix, then free-runs ``--generate`` steps (the client decrypts all K picks and
    feeds conversation c's back to lane c). Every lane runs every pass to the longest schedule; a conversation's
    records stop at its own prompt length + generate."""
    from perseus import _pc
    from perseus.examples.fhe_ssm.run_planned import session_config

    from .batched import BatchedMirror
    from .run_capture import batch_probe
    K = int(a.batch)
    if not (a.prompts and a.prefix):
        raise SystemExit("--batch needs --prompts and a public prefix (--prefix >= 1)")
    t_all = time.time()
    P, co, shifts = common.load(a)
    tok = common.tokenizer(a.harness, a.vocab)
    if tok is None:
        raise SystemExit("--batch needs --harness (the tokenizer)")
    texts = read_prompts(a.prompts)
    if len(texts) < K:
        raise SystemExit(f"{a.prompts}: {len(texts)} prompts for K = {K}")
    texts = texts[:K]
    all_ids = [tok.encode(s) for s in texts]
    pre = all_ids[0][: a.prefix]
    if any(ids[: a.prefix] != pre for ids in all_ids):
        raise SystemExit(f"the prompts do not share the public prefix ids {pre}")
    prompts = [ids[a.prefix:] for ids in all_ids]
    if any(not p for p in prompts):
        raise SystemExit("an empty prompt (nothing after the prefix)")
    polys, st = [], None
    for ids in all_ids:
        poly, st = common.prefix_state(P, a.coeffs, ids, a.prefix)
        polys.append(poly)
    fhe, _inf, model = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts, batch=K,
                                  batch_carry=a.batch_carry, batch_hoist=a.batch_hoist, **model_kw(a))
    model.init_state = st
    cfg = session_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    bundle, client = resolve_keys(a) if a.keyspec is None else (None, None)
    labels = [f"block_{b}" for b in range(model.n_layers + 1)]
    work = a.work_dir or a.out + ".work"
    out = open(a.out, "w")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)

    cf_map = None
    if a.pcmm_cf_table:
        from perseus.examples.fhe_ssm import pcmm_cf_table
        cf_map = pcmm_cf_table.load(a.pcmm_cf_table)
    run = Rwkv7BatchedRun(Recorder(model, model.n_layers), a.plan, cfg, labels, keys, model=model, work_dir=work,
                          pcmm_cf=a.pcmm_cf, pt_rows=a.pt_rows, stc_cache=a.stc_cache, pool=a.pool,
                          schedule=a.schedule, pcmm_cf_map=cf_map, bundle=bundle, client=client)
    long_ids = common.token_ids(a, common.CALIB_TEXT_LONG)[a.prefix:]
    public, _pt = batch_probe(P, long_ids, K, 2)            # the probes: two public passes
    if a.keyspec is not None:
        run.prepare(public)
        return write_keyspec(a.keyspec, cfg, keys, run.session_kwargs(), model)
    binfo = check_bundle(bundle, cfg, model) if bundle is not None else None
    t0 = time.time()
    run.setup(public)
    setup_s = time.time() - t0
    kv = run.state_cts(st)
    emit({"kind": "header", "driver": "perseus.examples.rwkv7.run_planned --batch", **keys_header(run, binfo),
          "width": P.C, "shards": model.L.S, "lanes": model.L.t, "lora": model.W[0].lora, "shard_land": model.shard_land,
          "blocks": model.n_layers, "batch": K, "batch_carry": a.batch_carry, "batch_hoist": a.batch_hoist,
          "prefix": pre, "prompts": prompts, "generate": a.generate, "plan": a.plan, "pcmm_cf": a.pcmm_cf,
          "wbits": a.wbits, "schedule": a.schedule, "pcmm_cf_table": a.pcmm_cf_table, "pool": a.pool,
          "stc_cache": a.stc_cache, "fuse_l1": a.fuse_l1, "shifts": a.shifts, "setup_s": round(setup_s, 1),
          "programs": {lab: {k: v for k in ("n_ops", "n_bootstraps", "n_launches", "n_finishes")
                             if isinstance(v := getattr(p, k, None), (int, float))}
                       for lab, p in run.progs.items()},
          "mem_report_text": str(run.sess.mem_report),
          "pth_sha256": common.sha256(a.pth), "coeffs_sha256": common.sha256(a.coeffs),
          "environment": common.environment()})
    if not a.no_warmup:
        t0 = time.time()
        reg, _ = run.token(0, public)
        run.reply(reg)
        emit({"kind": "warmup", "seconds": round(time.time() - t0, 3)})
    mirror = BatchedMirror(P, co, model.L.n, K, carry=a.batch_carry, wbits=a.wbits, fuse_l1=a.fuse_l1,
                           hoist=a.batch_hoist, lora=a.lora, lanes=a.lanes)
    mirror.reset(st)
    scorers = [common.Scorer(P, a, poly=polys[c]) for c in range(K)]
    toks = [list(p) for p in prompts]
    ends = [len(p) + a.generate for p in prompts]
    n_pass = max(ends)
    rows_seen, recs, passes = [None], [], []
    last = [None] * K
    for i in range(n_pass):
        for c in range(K):
            if i >= len(toks[c]):
                toks[c].append(int(last[c]))       # free running: conversation c's own pick, back on lane c
        rows = np.stack([P.emb[toks[c][i]] for c in range(K)])
        rows_seen.append(rows)
        t0 = time.time()
        # t = i + 1 keeps PlannedRun's "t == 0: empty state" rule off; pass 0 reads the prefix state Cts
        reg, info = run.token(i + 1, rows_seen, kv_cts=kv if i == 0 else None)
        hidden, level = run.reply(reg)
        sec = time.time() - t0
        hm, _z = mirror.step(rows)
        active = sum(i < e for e in ends)
        prec = {"kind": "pass", "pass": i, "seconds": round(sec, 4), "issue_s": info["issue_s"],
                "reply_level": level, "active": active}
        passes.append(prec)
        emit(prec)
        for c in range(K):
            if i >= ends[c]:
                last[c] = int(np.argmax(scorers[c].poly.head(np.asarray(hidden[c], dtype=np.float64))))
                continue
            rec = {"kind": "token", "mode": "batch", "conv": c, "pass": i,
                   "phase": "prompt" if i < len(prompts[c]) else "generate",
                   **scorers[c].score(a.prefix + i, toks[c][i], hidden[c]),
                   "hidden_relErrRms_vs_mirror": common.rel_rms(hidden[c], hm[c])}
            last[c] = rec["top1_fhe"]
            recs.append(rec)
            emit(rec)
    secs = [p["seconds"] for p in passes]
    useful = sum(p["active"] for p in passes)
    med = float(np.median(secs))
    emit({"kind": "summary", "batch": K, "passes": len(passes), "tokens": len(recs),
          "top1_agree": sum(r["top1_agree"] for r in recs),
          "top1_agree_per_conv": [sum(r["top1_agree"] for r in recs if r["conv"] == c) for c in range(K)],
          "tokens_per_conv": [sum(r["conv"] == c for r in recs) for c in range(K)],
          "prompt_ids": prompts, "generated_ids": [toks[c][len(prompts[c]):ends[c]] for c in range(K)],
          "logits_relErrRms_max": max(r["logits_relErrRms"] for r in recs),
          "logits_relErrRms_median": float(np.median([r["logits_relErrRms"] for r in recs])),
          "hidden_relErrRms_vs_mirror_max": max(r["hidden_relErrRms_vs_mirror"] for r in recs),
          "seconds_per_pass_median": med,
          "seconds_per_pass_after_first": float(np.mean(secs[1:])) if len(secs) > 1 else None,
          "seconds_per_token_per_conversation": med,
          "aggregate_tokens_per_s": round(useful / sum(secs), 4),
          "aggregate_tokens_per_s_full": round(K / med, 4),
          "seconds_per_conversation_token_full": round(med / K, 4),
          "setup_s": round(setup_s, 1), "wall_s": round(time.time() - t_all, 1),
          "planned_stats": str(run.sess.planned_stats())})
    run.close()
    return 0


def _state_key(run, b, i):
    return re.search(r"cache\.([a-z0-9]+)-lvl", run.ref[b].inputs()[i]).group(1)


class Rwkv7PrefillRun:
    """The decode programs AND the T-token prefill programs (prefill.py) in ONE planned session, sharing the carried
    state's registers: a prefill pass writes each block's ``S_0, S_1, xa, xf`` where the decode step reads them, so
    decoding continues from the prefilled state without leaving the device.

    ``dec`` and ``pf`` are two ``Rwkv7PlannedRun`` (each its own twin, graphs and plan: the decode plan and the
    prefill capture's plan); ``prepare`` compiles both, relabels the prefill programs ``pf_block_<b>``, maps the
    prefill state registers onto the decode ones and numbers the prefill stage outputs after the decode registers;
    ``setup`` opens one session for every program (the keys cover both programs' rotations: the session derives them
    from the programs), uploads the PCMM planes once (the same linears, the same shifts: both captures read one
    ``pcmm_shifts.json``) and stages both programs' plaintexts in one store (shared names: equal values share a slot).
    """

    def __init__(self, dec: Rwkv7PlannedRun, pf: Rwkv7PlannedRun, T: int, log=print):
        self.dec, self.pf, self.T, self.log = dec, pf, int(T), log

    def prepare(self, dec_rows, pf_chunks):
        dec, pf = self.dec, self.pf
        dec.prepare(dec_rows)
        pf.prepare(pf_chunks)
        for lab, p in pf.progs.items():
            p.relabel(f"pf_{lab}")
        n_layers = dec.rec.n_layers
        bad = []
        for b in range(n_layers):
            dmap = {_state_key(dec, b, i): (i, j, r) for i, j, r in dec.reg[b]["kv"]}
            pmap = {_state_key(pf, b, i): (i, j) for i, j, _r in pf.reg[b]["kv"]}
            if set(dmap) != set(pmap):
                raise RuntimeError(f"block {b}: decode state {sorted(dmap)} vs prefill state {sorted(pmap)}")
            pd, pp = dec.progs[dec.labels[b]], pf.progs[pf.labels[b]]
            for key, (i, j) in pmap.items():
                di, dj, _r = dmap[key]
                # the register a prefill pass writes is read by the next decode step (and the next pass): its
                # (level, deg) must be the decode input's compiled one, else Program.run refuses it (program.rs)
                w, rd = tuple(pp.outputs[j][1:3]), tuple(pd.inputs[di][1:3])
                w2, rd2 = tuple(pd.outputs[dj][1:3]), tuple(pp.inputs[i][1:3])
                if w[:2] != rd[:2] or w2[:2] != rd2[:2]:
                    bad.append(f"block {b} {key}: prefill writes {w}, decode reads {rd}; decode writes {w2}, "
                               f"prefill reads {rd2}")
            pf.reg[b]["kv"] = [(i, j, dmap[key][2]) for key, (i, j) in pmap.items()]
        if bad:
            raise RuntimeError("the two plans land the carried state differently: " + "; ".join(bad[:6]))
        n = dec.n_regs
        for b in sorted(pf.reg):
            outs = []
            for j, _r in pf.reg[b]["out"]:
                outs.append((j, n))
                n += 1
            pf.reg[b]["out"] = outs
        self.n_regs = n
        pf.n_regs = n                   # its registers are numbered in the shared space (register_levels sizes by it)
        return self

    def session_kwargs(self):
        dec, pf = self.dec, self.pf
        kd, kp = dec.session_kwargs(), pf.session_kwargs()
        if {s[0] for s in kd["pcmm"] or []} != {s[0] for s in kp["pcmm"] or []} or sorted(kd["pcmm"] or []) != sorted(kp["pcmm"] or []):
            raise RuntimeError("the decode and prefill twins encode the PCMM linears differently (shifts?)")
        kw = dict(kd)
        kw["programs"] = kd["programs"] + kp["programs"]
        kw["regs"] = self.n_regs
        writes = []
        for run in (dec, pf):
            for b in sorted(run.ref):
                p = run.progs[run.labels[b]]
                writes += [(p, j, r) for _i, j, r in run.reg[b]["kv"]] + [(p, j, r) for j, r in run.reg[b]["out"]]
        kw["reg_levels"] = register_levels(self.n_regs, writes)
        pairs = {(run.al[(lab, nm)], int(lv)) for run in (dec, pf) for lab, p in run.progs.items()
                 for nm, lv, _d, f in p.pt_refs if f not in ("stream", "diag")}
        kw["pt_levels"] = [lv for _a, lv in sorted(pairs)]
        kw["pt_slots"] = len(kw["pt_levels"]) + 8
        kw["pt_rows"] = max(kd.get("pt_rows") or 0, kp.get("pt_rows") or 0)
        return kw

    def setup(self, dec_rows, pf_chunks):
        from perseus import _pc
        from perseus.impl.ops import FheOps
        t0 = time.time()
        self.prepare(dec_rows, pf_chunks)
        dec, pf = self.dec, self.pf
        kw = self.session_kwargs()
        for run in (dec, pf):
            run.rot_keys = plan_rot_scope(run.plan_dir, [run.progs[run.labels[b]] for b in sorted(run.ref)],
                                          run.dyn_rot)
        sess = dec.open_session(kw)
        for run in (dec, pf):
            run.sess, run.store = sess, sess.pt_store
            for b in sorted(run.ref):
                run.progs[run.labels[b]].bind(sess, run.store)
        sess.complete_setup()
        for run in (dec, pf):
            run.fhe, run._pt_vals = sess.fhe, {}
            run.ops = FheOps(sess.fhe, _pc, run.model.rt.ops.unit)
            run.static = True
        for name, lin in dec.model.linears().items():
            sess.pcmm_upload(name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        items = dec.pt_items()
        for k, v in pf.pt_items().items():
            items.setdefault(k, v)
        n = int(dec.store.stage([(nm, lv, v) for (nm, lv), v in items.items()]))
        self.setup_s = time.time() - t0
        self.log(f"[prefill] one session: {len(kw['programs'])} programs (decode + T = {self.T} prefill), "
                 f"{self.n_regs} registers, {n} plaintext slots, rotation keys decode {dec.rot_keys} prefill "
                 f"{pf.rot_keys}; setup {self.setup_s:.1f} s")
        self.log("[prefill] " + sess.mem_report.replace("\n", " | "))
        self.sess = sess
        return self

    def prefill(self, rows, kv_cts=None):
        """One T-token pass (rows: T <= t embedding rows) on the device; the carried inputs from the shared registers,
        or from ``kv_cts`` (``pf.state_cts``: the public prefix state) on the first pass. Returns the reply register."""
        pf = self.pf
        L = pf.model.L
        n_blocks = pf.rec.n_layers + 1
        for b in range(n_blocks):
            if b == 0:
                p = pf.progs[pf.labels[0]]
                xs = [pf.encrypt_input(r, p, i, f"prefill chunk shard {s}")[0]
                      for s, (r, i) in enumerate(zip(L.encode_rows_sh(rows), pf.zin[0]))]
            else:
                xs = pf.out_regs(b - 1)
            pf.run_stage(b, 1, xs, kv_cts)
        return pf.out_regs(n_blocks - 1)

    def reply(self, regs, T):
        vals, level = self.pf.decrypt_regs(regs)
        return self.pf.model.L.decode_lanes_sh(vals, T), level

    def close(self):
        self.dec.close()


def probe_rows(P, calib_ids, prefix_ids=()):
    """The probes' rows: public and in distribution. The twin records them with values on (PCMM's fake multiplies
    the slot vectors), so an untrained row (``emb[0]``, the World vocabulary's unused token 0) drives the Chebyshev
    and Newton sites out of their intervals and the probe's values overflow to NaN (harmless to the stream, which is
    structure only, but it blinds the recorder's field-tag value check and fills the log with overflow warnings).
    The calibration text's tokens after its ``"\\n\\n"`` (``common.CALIB_TEXT``, the same public text the PCMM shifts
    come from); without a tokenizer, the public prefix's token; else token 0."""
    ids = list(calib_ids)[1:3] if len(calib_ids) >= 3 else [prefix_ids[0]] * 2 if len(prefix_ids) else [0, 0]
    return [P.emb[int(t)] for t in ids]


# ── keys: in-process (insecure) or a client's bundle ──

def resolve_keys(a):
    """(bundle file, client) for ``--keys``: ``inproc`` -> (None, None), the session generates every key and holds
    the secret; ``bundle`` (the directory in ``$RWKV7_KEYS``) or a directory -> its ``keys.pckb`` and a ``_pc.Client``
    on its ``client_secret.bin`` (the client's end of the run: input encryption, reply decryption)."""
    if a.keys == "inproc":
        return None, None
    from perseus import _pc
    d = os.environ.get("RWKV7_KEYS") if a.keys == "bundle" else a.keys
    if not d:
        raise SystemExit("--keys bundle reads the bundle directory from RWKV7_KEYS")
    d = pathlib.Path(d)
    if not (d / "keys.pckb").is_file():
        raise SystemExit(f"{d}: no keys.pckb (the client's bundle for this run's key request: --keyspec, then --keygen)")
    if not (d / "client_secret.bin").is_file():
        raise SystemExit(f"{d}: no client_secret.bin: this runner decrypts and scores every reply (the client's end)")
    client = _pc.Client(_pc.ClientSecret.from_bytes(list((d / "client_secret.bin").read_bytes())), complex=True)
    return d / "keys.pckb", client


def check_bundle(bundle, cfg, model) -> dict:
    """The bundle against this session before it opens: the plan contract, PCMM keys, enough switching keys for the
    layout's rotations (+ conjugation + relinearization), a public key."""
    from perseus import _pc
    info = dict(_pc.bundle_info(str(bundle)))
    want = _pc.contract_hash(cfg)
    if info.get("contract") != want:
        raise SystemExit(f"{bundle}: made for plan contract {info.get('contract')}, this session's Config is {want}")
    ent = dict(info.get("entries") or {})
    need = key_needs(model)
    if not ent.get("pcmm"):
        raise SystemExit(f"{bundle}: no PCMM ring-switch keys (pcmm_t {need['pcmm_t']})")
    if int(ent.get("ksk", 0)) < need["n_rotation_steps"] + 2:
        raise SystemExit(f"{bundle}: {ent.get('ksk', 0)} switching keys, the layout needs >= {need['n_rotation_steps'] + 2}")
    if not ent.get("public_key"):
        raise SystemExit(f"{bundle}: no public key")
    return {"bytes": info.get("bytes"), "entries": ent, "contract": info.get("contract"),
            "params_hash": info.get("params_hash")}


def key_needs(model) -> dict:
    """What the model's layout asks of a key bundle before any program exists (the session's request,
    ``Session.key_manifest``, adds the refresh routes' keys and relinearization)."""
    n = model.L.n
    steps = sorted({int(x) % n for x in model.rotation_steps()} - {0})
    return {"slots": n, "width": model.P.C, "shards": model.L.S, "rotation_steps": steps,
            "n_rotation_steps": len(steps), "conjugation": True, "pcmm_t": [model.L.t]}


def write_keyspec(path, cfg, keys, kw, model) -> int:
    """``--keyspec``: the session's key request (``Session.key_manifest`` on the planned programs; device build)."""
    from perseus import _pc
    doc = {"model": key_needs(model), "session": None, "note": None}
    if hasattr(_pc, "Session") and hasattr(_pc.Session, "key_manifest"):
        doc["session"] = json.loads(_pc.Session.key_manifest(cfg, keys, **kw))
    else:
        doc["note"] = "the session's key request needs the device build of perseus._pc"
    text = json.dumps(doc, indent=1, default=str)
    if path == "-":
        print(text)
    else:
        pathlib.Path(path).write_text(text, encoding="utf-8")
        print(f"[keyspec] {path}: {len(doc['model']['rotation_steps'])} rotation steps; session request "
              f"{'present' if doc['session'] else 'absent (CPU build)'}", flush=True)
    return 0


def keygen_main(argv) -> int:
    """The client role (``--keygen DIR --keyspec-file F``): a fresh secret, the bundle ``DIR/keys.pckb`` for the
    server's key request, the secret beside it (``DIR/client_secret.bin``). Runs without the model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--keygen", required=True)
    ap.add_argument("--keyspec-file", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a, _rest = ap.parse_known_args(argv)
    from perseus import _pc
    if not hasattr(_pc, "Client"):
        raise SystemExit("--keygen needs the device build of perseus._pc (Client)")
    doc = json.load(open(a.keyspec_file, encoding="utf-8"))
    spec = doc.get("session") if isinstance(doc, dict) and "session" in doc else doc
    if not isinstance(spec, dict) or "keys" not in spec:
        raise SystemExit(f"{a.keyspec_file}: no session key request (made on the device build: --keyspec)")
    out = pathlib.Path(a.keygen)
    out.mkdir(parents=True, exist_ok=True)
    secret = _pc.ClientSecret.generate()
    cl = _pc.Client(secret, complex=True)
    t0 = time.time()
    st = dict(cl.keygen(json.dumps(spec), str(out / "keys.pckb"), threads=a.threads))
    (out / "client_secret.bin").write_bytes(bytes(secret.to_bytes()))
    st["wall_s"] = round(time.time() - t0, 1)
    st["info"] = dict(_pc.bundle_info(str(out / "keys.pckb")))
    print(json.dumps({"keygen": st, "bundle": str(out / "keys.pckb"),
                      "client_secret": str(out / "client_secret.bin")}, default=str))
    return 0


def keys_header(run, bundle_info=None) -> dict:
    return {"keys": run.key_mode(), "has_secret": bool(run.sess.has_secret),
            "input_encryption": "client" if run.client is not None else
            ("session-secret" if run.bundle is None else "session-public-key"),
            "bundle_info": bundle_info}


# ── continuous batching on the device (serve.py, SERVE_DESIGN.md) ──

_KEY = re.compile(r"cache\.([a-z0-9]+)-lvl")


class Rwkv7JoinRun(Rwkv7PlannedRun):
    """The join program (``run_capture --join``; ``serve.Rwkv7Join`` is its twin): one stage, every input a carried
    value (the lane-form carries ``b<b><carry>`` and one conversation's natural-form carries ``b<b>n<carry>``), the
    joined carries out. Every input is compiled at its recorded (level, deg); the two masks are plaintexts under their
    own store names (``join/m``, ``join/keep``), re-staged in place per join (``stage_masks``)."""

    MASKS = ("join/m", "join/keep")

    def _compile_with_schedule(self, fn):
        from perseus import _pc
        real, sched, run = _pc.Program, self.schedule, self

        class _Join:
            @staticmethod
            def compile(*a, **k):
                metas = {lab: dict(v) for lab, v in (k.get("input_metas") or {}).items()}
                for b, r in run.ref.items():
                    metas.setdefault(run.labels[b], {}).update({n: r.metas[n] for n in r.inputs() if n in r.metas})
                k["input_metas"] = metas
                if sched != "program":
                    k["schedule"] = sched
                return real.compile(*a, **k)
        _pc.Program = _Join
        try:
            return fn()
        finally:
            _pc.Program = real

    def prepare(self, rows, argmax=False):
        self._compile_with_schedule(lambda: PlannedRun.prepare(self, rows, argmax))
        if any(self.dyn.values()):
            raise RuntimeError(f"join: dynamic steps in the stream: {self.dyn}")
        for b, r in self.ref.items():
            o = self.records[0][b]
            if set(o.pts) != set(r.pts) or any(not np.array_equal(o.pts[k], r.pts[k]) for k in r.pts):
                raise RuntimeError("join: a plaintext differs between the probe tokens (not static)")
        self.map_io()
        self.static = True
        self.al = self.pt_aliases()
        return self

    def map_io(self):
        """``ins`` {key: program input}, ``outs`` {lane key: program output}; refuses a stage that is not the join."""
        if sorted(self.ref) != [0]:
            raise RuntimeError(f"join: stages {sorted(self.ref)}, one expected")
        r, p, m = self.ref[0], self.progs[self.labels[0]], self.model
        self.ins = {}
        for i, nm in enumerate(r.inputs()):
            hit = _KEY.search(nm)
            if hit is None:
                raise RuntimeError(f"join: input {i} ({nm}) is not a named carried value")
            self.ins[hit.group(1)] = i
        want = set(m.keys) | set(m.new_keys)
        if set(self.ins) != want:
            raise RuntimeError(f"join: inputs {sorted(self.ins)}, expected {sorted(want)}")
        if len(r.out_names) != len(m.keys):
            raise RuntimeError(f"join: {len(r.out_names)} outputs, {len(m.keys)} lane carries")
        self.outs = {k: p.output_index(o) for k, o in zip(m.keys, r.out_names)}
        return self

    def pt_aliases(self):
        """The two masks by value (the capture's lane): ``join/m``, ``join/keep``; anything else by digest."""
        out = {}
        m, keep = self.model.m, self.model.keep
        for b in sorted(self.ref):
            lab, r = self.labels[b], self.ref[b]
            for nm, _lv, _d, f in self.progs[lab].pt_refs:
                if f in ("stream", "diag"):
                    continue
                v = np.real(np.asarray(r.pts[nm]))
                out[(lab, nm)] = (self.MASKS[0] if np.array_equal(v, m) else self.MASKS[1] if np.array_equal(v, keep)
                                  else "v/" + digest(np.asarray(r.pts[nm])))
        if set(out.values()) != set(self.MASKS):
            raise RuntimeError(f"join: plaintexts {sorted(set(out.values()))}, expected exactly the two masks")
        return out

    def mask_slots(self) -> list[tuple[str, int]]:
        lab = self.labels[0]
        return sorted({(self.al[(lab, nm)], int(lv)) for nm, lv, _d, f in self.progs[lab].pt_refs
                       if f not in ("stream", "diag")})

    def stage_masks(self, lanes):
        """Lane(s) ``lanes``' masks into the store, in place (a resident key is re-encoded at its own slot, after every
        earlier reader in stream order: the bound program's CUDA graph stays valid)."""
        from .serve import join_masks
        m, keep = join_masks(self.model.L, self.model.K, lanes)
        vals = {self.MASKS[0]: m, self.MASKS[1]: keep}
        return int(self.store.stage([(a, lv, np.ascontiguousarray(vals[a], dtype=np.float64))
                                     for a, lv in self.mask_slots()]))


def _remap_regs(runs, f):
    for run in runs:
        for b in list(run.reg):
            run.reg[b] = {"kv": [(i, j, f[r]) for i, j, r in run.reg[b]["kv"]],
                          "out": [(j, f[r]) for j, r in run.reg[b]["out"]]}


class Rwkv7ServeRun:
    """Decode, prefill (``Rwkv7PrefillRun``), the K-conversation batched pass and the join in ONE planned session
    (SERVE_DESIGN.md §3). The register map:

    * the batched pass's registers: lane c's ``s<m>[h<j>]c<c>``, the lane-form carries, the stage outputs;
    * the single-conversation programs (decode, prefill) read and write ``S`` in the registers of the lane the
      conversation will join (``bind_lane``: the register hand-over), their natural-form carries in their own
      registers (the join reads them), their stage outputs in their own;
    * the join reads the lane-form carry registers and the single carry registers and writes the lane-form ones.

    ``prepare`` (CPU) compiles the four program sets, relabels them (``pf_block_<b>``, ``b_block_<b>``,
    ``join_block_0``), numbers the registers densely and compares the (level, deg) of every hand-over direction
    before any session exists, naming each mismatch: single S out -> batched S in (every lane), batched S / carry out
    -> batched in, single carry out -> join natural-form in, batched carry out -> join lane in, join out -> batched in
    and -> join lane in (two joins before a pass)."""

    def __init__(self, dec, pf, bat, join, T: int, K: int, log=print):
        self.dec, self.pf, self.bat, self.join, self.T, self.K, self.log = dec, pf, bat, join, int(T), int(K), log
        self.single = Rwkv7PrefillRun(dec, pf, T, log=log)

    @property
    def runs(self):
        return (self.dec, self.pf, self.bat, self.join)

    def prepare(self, dec_rows, pf_chunks, b_passes, j_rows):
        dec, pf, bat, join = self.runs
        self.single.prepare(dec_rows, pf_chunks)
        bat.prepare(b_passes)
        join.prepare(j_rows)
        for lab, p in bat.progs.items():
            p.relabel(f"b_{lab}")
        for lab, p in join.progs.items():
            p.relabel(f"join_{lab}")
        n0 = self.single.n_regs
        _remap_regs([bat], {r: r + n0 for b in bat.reg for r in
                            [x[2] for x in bat.reg[b]["kv"]] + [x[1] for x in bat.reg[b]["out"]]})
        self.n_layers = dec.rec.n_layers
        self._keymaps()
        sk = sorted(k for k in self.dkv[0] if k.startswith("s"))
        ck = sorted(k for k in self.dkv[0] if not k.startswith("s"))
        for b in range(self.n_layers):
            want = {f"{s}c{c}" for s in sk for c in range(self.K)} | set(ck)
            if set(self.bkv[b]) != want:
                raise RuntimeError(f"block {b}: batched state {sorted(self.bkv[b])[:8]}..., expected the K = {self.K} "
                                   f"lanes of the decode state {sorted(self.dkv[b])}")
        self.s_keys, self.c_keys = sk, ck
        self.check_landings()
        # the single programs' S onto lane 0 (bind_lane moves them per conversation); then number densely: the decode
        # plan's own S registers are never used
        self.bind_lane(0)
        used = sorted({r for run in (dec, pf, bat) for b in run.reg for r in
                       [x[2] for x in run.reg[b]["kv"]] + [x[1] for x in run.reg[b]["out"]]})
        f = {r: i for i, r in enumerate(used)}
        _remap_regs([dec, pf, bat], f)
        self.n_regs = len(used)
        self.single.n_regs = self.n_regs
        for run in self.runs:
            run.n_regs = self.n_regs
        self._keymaps()
        self.bound = None
        self.bind_lane(0)
        return self

    def _keymaps(self):
        """{block: {state key: (input, output, register)}} for the decode, prefill and batched programs."""
        def km(run):
            return {b: {_state_key(run, b, i): (i, j, r) for i, j, r in run.reg[b]["kv"]}
                    for b in range(self.n_layers)}
        self.dkv, self.pkv, self.bkv = km(self.dec), km(self.pf), km(self.bat)

    def lane_reg(self, b, key, c):
        """Lane c's register of single-conversation state ``key`` (``s0``, ``s1h1``, ...) in block b."""
        return self.bkv[b][f"{key}c{c}"][2]

    def bind_lane(self, c):
        """The register hand-over: the decode and prefill programs read and write S in lane c's registers."""
        if getattr(self, "bound", None) == c:
            return
        for run, kv in ((self.dec, self.dkv), (self.pf, self.pkv)):
            for b in range(self.n_layers):
                s = set(self.s_keys)
                run.reg[b]["kv"] = [(i, j, self.lane_reg(b, k, c) if k in s else r)
                                    for k, (i, j, r) in ((_state_key(run, b, i), (i, j, r))
                                                         for i, j, r in run.reg[b]["kv"])]
        self.bound = c
        self._keymaps()

    def join_map(self):
        """[(block, carry key, lane input, natural input, output)] of the join program."""
        out = []
        for b in range(self.n_layers):
            for k in self.c_keys:
                out.append((b, k, self.join.ins[f"b{b}{k}"], self.join.ins[f"b{b}n{k}"], self.join.outs[f"b{b}{k}"]))
        return out

    def check_landings(self):
        dec, pf, bat, join = self.runs
        pj = join.progs[join.labels[0]]
        bad = []

        def cmp(what, wp, wj, rp, ri):
            w, rd = tuple(wp.outputs[wj][1:3]), tuple(rp.inputs[ri][1:3])
            if w[:2] != rd[:2]:
                bad.append(f"{what}: written at {w[:2]}, read at {rd[:2]}")
        want = {f"b{b}{k}" for b in range(self.n_layers) for k in self.c_keys}
        if set(join.outs) != want or set(join.ins) != want | {f"b{b}n{k}" for b in range(self.n_layers)
                                                               for k in self.c_keys}:
            raise RuntimeError(f"the join program's carries {sorted(join.ins)} do not match the batched pass's "
                               f"{sorted(want)} (a join captured for another block count or width?)")
        for b in range(self.n_layers):
            pd, pp, pb = dec.progs[dec.labels[b]], pf.progs[pf.labels[b]], bat.progs[bat.labels[b]]
            for s in self.s_keys:
                for c in range(self.K):
                    bi, bj, _r = self.bkv[b][f"{s}c{c}"]
                    cmp(f"block {b} {s} decode -> batched lane {c}", pd, self.dkv[b][s][1], pb, bi)
                    cmp(f"block {b} {s} prefill -> batched lane {c}", pp, self.pkv[b][s][1], pb, bi)
                    cmp(f"block {b} {s}c{c} batched -> batched", pb, bj, pb, bi)
            for k in self.c_keys:
                bi, bj, _r = self.bkv[b][k]
                li, ni, jo = join.ins[f"b{b}{k}"], join.ins[f"b{b}n{k}"], join.outs[f"b{b}{k}"]
                cmp(f"block {b} {k} batched -> batched", pb, bj, pb, bi)
                cmp(f"block {b} {k} batched -> join (lane form)", pb, bj, pj, li)
                cmp(f"block {b} {k} join -> batched", pj, jo, pb, bi)
                cmp(f"block {b} {k} join -> join (lane form)", pj, jo, pj, li)
                cmp(f"block {b} {k} decode -> join (natural form)", pd, self.dkv[b][k][1], pj, ni)
                cmp(f"block {b} {k} prefill -> join (natural form)", pp, self.pkv[b][k][1], pj, ni)
        if bad:
            raise RuntimeError(f"the serve programs land the carried state differently ({len(bad)}): "
                               + "; ".join(bad[:8]))

    def session_kwargs(self):
        dec, pf, bat, join = self.runs
        kd, kb, kj = self.single.session_kwargs(), bat.session_kwargs(), join.session_kwargs()
        if sorted(kd["pcmm"] or []) != sorted(kb["pcmm"] or []):
            raise RuntimeError("the decode and batched twins encode the PCMM linears differently (shifts?)")
        if kj.get("pcmm"):
            raise RuntimeError("the join program has PCMM linears")
        kw = dict(kd)
        kw["programs"] = kd["programs"] + kb["programs"] + kj["programs"]
        kw["regs"] = self.n_regs
        kw["pool"] = max(kd["pool"], kb["pool"], kj["pool"])
        kw["stc_cache"] = max(kd["stc_cache"], kb["stc_cache"])
        writes = []
        for run in (dec, pf, bat):
            for b in sorted(run.ref):
                p = run.progs[run.labels[b]]
                writes += [(p, j, r) for _i, j, r in run.reg[b]["kv"]] + [(p, j, r) for j, r in run.reg[b]["out"]]
        for b in range(self.n_layers):                    # the single programs write S into every lane's registers
            for s in self.s_keys:
                for run, kv in ((dec, self.dkv), (pf, self.pkv)):
                    p = run.progs[run.labels[b]]
                    writes += [(p, kv[b][s][1], self.lane_reg(b, s, c)) for c in range(self.K)]
        pj = join.progs[join.labels[0]]
        writes += [(pj, jo, self.bkv[b][k][2]) for b, k, _li, _ni, jo in self.join_map()]
        kw["reg_levels"] = register_levels(self.n_regs, writes)
        pairs = {(run.al[(lab, nm)], int(lv)) for run in self.runs for lab, p in run.progs.items()
                 for nm, lv, _d, f in p.pt_refs if f not in ("stream", "diag")}
        kw["pt_levels"] = [lv for _a, lv in sorted(pairs)]
        kw["pt_slots"] = len(kw["pt_levels"]) + 8
        kw["pt_rows"] = max(kd.get("pt_rows") or 0, kb.get("pt_rows") or 0, kj.get("pt_rows") or 0)
        return kw

    def setup(self, dec_rows, pf_chunks, b_passes, j_rows):
        from perseus import _pc
        from perseus.impl.ops import FheOps
        t0 = time.time()
        self.prepare(dec_rows, pf_chunks, b_passes, j_rows)
        kw = self.session_kwargs()
        for run in self.runs:
            run.rot_keys = plan_rot_scope(run.plan_dir, [run.progs[run.labels[b]] for b in sorted(run.ref)],
                                          run.dyn_rot)
        sess = self.dec.open_session(kw)
        for run in self.runs:
            run.sess, run.store = sess, sess.pt_store
            for b in sorted(run.ref):
                run.progs[run.labels[b]].bind(sess, run.store)
        sess.complete_setup()
        for run in self.runs:
            run.fhe, run._pt_vals = sess.fhe, {}
            run.ops = FheOps(sess.fhe, _pc, run.model.rt.ops.unit)
            run.static = True
        for name, lin in self.dec.model.linears().items():
            sess.pcmm_upload(name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        items = {}
        for run in self.runs:
            for k, v in run.pt_items().items():
                items.setdefault(k, v)
        n = int(self.dec.store.stage([(nm, lv, v) for (nm, lv), v in items.items()]))
        self.sess, self.setup_s = sess, time.time() - t0
        self.log(f"[serve] one session: {len(kw['programs'])} programs (decode + prefill T = {self.T} + batched "
                 f"K = {self.K} + join), {self.n_regs} registers, {n} plaintext slots, rotation keys "
                 f"{[run.rot_keys for run in self.runs]}; setup {self.setup_s:.1f} s")
        self.log("[serve] " + sess.mem_report.replace("\n", " | "))
        return self

    def close(self):
        self.dec.close()


class DeviceServer:
    """``serve.Server``'s operations on the device (``serve_loop`` drives it). ``src`` says where each batched state
    input comes from: a register, or an encryption of the public prefix state (the start, and a lane after a leave).
    ``Program.run`` copies an input Ct into its slab, so ONE encryption per (block, state key, compiled meta) serves
    every program and every lane that reads that value at that meta: the decode and prefill programs' first pass, every
    lane's S, the lane-form carries at the start (all lanes equal) and the join's natural-form input on a leave
    (``prefix``: 4 x blocks ciphertexts at width <= 1,024 instead of one per input). Untimed: the prefix is public."""

    def __init__(self, sr: Rwkv7ServeRun, state, log=print):
        from .serve import new_carry_rows
        self.sr, self.log = sr, log
        dec, pf, bat, join = sr.runs
        self.enc = {}
        self.kv_dec, self.kv_pf, self.kv_bat = (self._prefix(run, state) for run in (dec, pf, bat))
        pj = join.progs[join.labels[0]]
        rows = new_carry_rows(join.model.L, sr.n_layers, state)
        self.new_prefix = {(b, k): self._enc(join, pj, b, k, ni, rows[(b, k)]) for b, k, _li, ni, _jo in sr.join_map()}
        self.levels = {}
        self.log(f"[serve] the public prefix state: {len(self.enc)} encryptions")
        self.reset()

    def _enc(self, run, p, b, base, i, rows):
        key = (b, base, tuple(p.inputs[i][1:4]))
        if key not in self.enc:
            self.enc[key] = run.encrypt_input(rows, p, i, f"prefix b{b} {base}")[0]
        return self.enc[key]

    def _prefix(self, run, state) -> dict:
        """``{block: {input position: Ct}}`` of ``run``'s carried inputs, the public prefix state (``state_cts``'s
        values, shared through ``enc``)."""
        names, rows = run.model.state_names(), run.model.state_rows(state)
        out = {}
        for b in range(self.sr.n_layers):
            p = run.progs[run.labels[b]]
            for i, _j, _r in run.reg[b]["kv"]:
                key = _state_key(run, b, i)
                base = re.sub(r"c\d+$", "", key)                  # s0c3 -> s0: every lane holds the same value
                out.setdefault(b, {})[i] = self._enc(run, p, b, base, i, rows[b][names.index(key)])
        return out

    def reset(self):
        """Every lane at the public prefix state (the batched inputs from their encryptions)."""
        self.src = {(b, k): self.kv_bat[b][i] for b in range(self.sr.n_layers) for k, (i, _j, _r) in self.sr.bkv[b].items()}

    def _reply(self, run, regs, T=None):
        vals, level = run.decrypt_regs(regs)
        self.levels[run] = level
        L = run.model.L
        return L.decode_lanes_sh(vals, T) if T else L.decode_sh(vals)

    def prefill(self, c, rows, first):
        self.sr.bind_lane(c)
        regs = self.sr.single.prefill(rows, kv_cts=self.kv_pf if first else None)
        return self._reply(self.sr.pf, regs, len(rows)), None

    def decode(self, c, row, first):
        sr = self.sr
        sr.bind_lane(c)
        dec = sr.dec
        nb = dec.rec.n_layers + 1
        for b in range(nb):
            xs = dec.input_ct(0, [row], dec.progs[dec.labels[0]]) if b == 0 else dec.out_regs(b - 1)
            dec.run_stage(b, 1, xs, self.kv_dec if first else None)
        return self._reply(dec, dec.out_regs(nb - 1)), None

    def _join(self, lanes, new):
        sr = self.sr
        join = sr.join
        p = join.progs[join.labels[0]]
        join.stage_masks(lanes)
        ins, outs = [None] * len(p.inputs), []
        for b, k, li, ni, jo in sr.join_map():
            r = sr.bkv[b][k][2]
            ins[li] = self.src[(b, k)]
            ins[ni] = new(b, k)
            outs.append((jo, r))
            self.src[(b, k)] = r
        p.run(ins, outs)

    def join(self, c):
        sr = self.sr
        self._join([c], lambda b, k: sr.dkv[b][k][2])          # the single programs' natural-form carry registers
        for b in range(sr.n_layers):
            for s in sr.s_keys:
                self.src[(b, f"{s}c{c}")] = sr.lane_reg(b, s, c)   # written by the single programs: the hand-over

    def leave(self, lanes):
        sr = self.sr
        for c in lanes:
            for b in range(sr.n_layers):
                for s in sr.s_keys:
                    i = sr.bkv[b][f"{s}c{c}"][0]
                    self.src[(b, f"{s}c{c}")] = self.kv_bat[b][i]
        self._join(list(lanes), lambda b, k: self.new_prefix[(b, k)])

    def batch(self, rows):
        sr = self.sr
        bat = sr.bat
        nb = bat.rec.n_layers + 1
        for b in range(nb):
            p = bat.progs[bat.labels[b]]
            xs = bat.input_ct(0, [np.asarray(rows)], p) if b == 0 else bat.out_regs(b - 1)
            ins = [None] * len(p.inputs)
            for i, x in zip(bat.zin[b], xs):
                ins[i] = x
            writes = []
            for k, (i, j, r) in sr.bkv.get(b, {}).items():
                ins[i] = self.src[(b, k)]
                writes.append((j, r))
                self.src[(b, k)] = r
            p.run(ins, writes + list(bat.reg[b]["out"]))
        vals, level = bat.decrypt_regs(bat.out_regs(nb - 1))
        self.levels[bat] = level
        return bat.model.L.decode_lanes_sh(vals, self.sr.K), None


class TeeServer:
    """The device server with the circuit's numpy mirror in lockstep (the same tokens): each call returns the
    device's hidden and the mirror's (``hidden_relErrRms_vs_mirror``)."""

    def __init__(self, dev, mirror):
        self.dev, self.mir = dev, mirror

    def prefill(self, c, rows, first):
        return self.dev.prefill(c, rows, first)[0], self.mir.prefill(c, rows, first)[0]

    def decode(self, c, row, first):
        return self.dev.decode(c, row, first)[0], self.mir.decode(c, row, first)[0]

    def join(self, c):
        self.dev.join(c)
        self.mir.join(c)

    def leave(self, lanes):
        self.dev.leave(lanes)
        self.mir.leave(lanes)

    def batch(self, rows):
        return self.dev.batch(rows)[0], self.mir.batch(rows)[0]


def main_serve(a) -> int:
    """``--serve FILE``: continuous batching (SERVE_DESIGN.md). Conversations arrive at batched passes (FILE rows
    ``<pass> <prompt>``); an arrival is prefilled in single-conversation mode while the batch waits (floor(n / T)
    passes of ``--prefill-chunk`` T, decode steps for the remainder), joined to a free lane of the ``--batch`` K pass,
    generates ``--generate`` tokens there and leaves (its lane back to the public prefix state). Plans: ``--plan``
    (decode), ``--pf-plan``, ``--b-plan`` (batched, lane carries), ``--join-plan`` (``run_capture --join``), one
    shifts file. Every conversation's every position is scored like the decode runner's (top-1 against its own poly
    model, logits relErrRms, hidden vs the circuit's numpy mirror run on the same schedule)."""
    from perseus import _pc
    from perseus.examples.fhe_ssm.run_planned import session_config

    from .run_capture import batch_probe, join_probe
    from .serve import MirrorBackend, Server, idle_token, make_join, read_arrivals, serve_loop
    K, T = int(a.batch), int(a.prefill_chunk)
    if K < 2 or not (a.b_plan and a.join_plan and a.prefix):
        raise SystemExit("--serve needs --batch K >= 2, --b-plan, --join-plan and a public prefix (--prefix >= 1)")
    if T and not a.pf_plan:
        raise SystemExit("--serve with --prefill-chunk T needs --pf-plan")
    if not T:
        raise SystemExit("--serve needs --prefill-chunk T (the prefill programs share the session; T >= 1)")
    if a.batch_carry != "lane":
        raise SystemExit("--serve joins lane-form carries: --batch-carry lane")
    t_all = time.time()
    P, co, shifts = common.load(a)
    tok = common.tokenizer(a.harness, a.vocab)
    if tok is None:
        raise SystemExit("--serve needs --harness (the tokenizer)")
    rows_in = read_arrivals(a.serve)
    all_ids = [tok.encode(s) for _p, s in rows_in]
    pre = all_ids[0][: a.prefix]
    if any(ids[: a.prefix] != pre for ids in all_ids):
        raise SystemExit(f"the prompts do not share the public prefix ids {pre}")
    arrivals = [(p, ids[a.prefix:]) for (p, _s), ids in zip(rows_in, all_ids)]
    if any(not ids for _p, ids in arrivals):
        raise SystemExit("an empty prompt (nothing after the prefix)")
    rf, st = common.prefix_state(P, a.coeffs, all_ids[0], a.prefix)
    head_model = rf
    kw = dict(fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts, **model_kw(a))
    _f1, _i1, dm = make_model(P, co, capture=True, **kw)
    _f2, _i2, pm = make_model(P, co, capture=True, chunk=T, **kw)
    _f3, _i3, bm = make_model(P, co, capture=True, batch=K, batch_carry="lane", batch_hoist=a.batch_hoist,
                              prefill_keys=T, **kw)
    for m in (dm, pm, bm):
        m.init_state = st
    _f4, _i4, jm = make_join(P, K, capture=True, lanes=a.lanes)
    cfg = session_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    bundle, client = resolve_keys(a) if a.keyspec is None else (None, None)
    labels = [f"block_{b}" for b in range(dm.n_layers + 1)]
    work = a.work_dir or a.out + ".work"
    cf_map = None
    if a.pcmm_cf_table:
        from perseus.examples.fhe_ssm import pcmm_cf_table
        cf_map = pcmm_cf_table.load(a.pcmm_cf_table)
    lever = dict(schedule=a.schedule, pcmm_cf_map=cf_map, bundle=bundle, client=client, pcmm_cf=a.pcmm_cf,
                 pt_rows=a.pt_rows, stc_cache=a.stc_cache, pool=a.pool)
    dec = Rwkv7PlannedRun(Recorder(dm, dm.n_layers), a.plan, cfg, labels, keys, model=dm, work_dir=work, **lever)
    pf = Rwkv7PlannedRun(Recorder(pm, pm.n_layers), a.pf_plan, cfg, labels, keys, model=pm, work_dir=work + ".pf",
                         **lever)
    bat = Rwkv7BatchedRun(Recorder(bm, bm.n_layers), a.b_plan, cfg, labels, keys, model=bm, work_dir=work + ".b",
                          **lever)
    join = Rwkv7JoinRun(Recorder(jm, 0), a.join_plan, cfg, ["block_0"], keys, model=jm, work_dir=work + ".j",
                        **lever)
    sr = Rwkv7ServeRun(dec, pf, bat, join, T, K)
    try:
        calib = common.token_ids(a, common.CALIB_TEXT)
    except SystemExit:
        calib = []
    public = probe_rows(P, calib, pre)
    long_ids = common.token_ids(a, common.CALIB_TEXT_LONG)[a.prefix:]
    pf_public = [[P.emb[t] for t in long_ids[i * T:(i + 1) * T]] for i in range(2)]
    b_public, _bt = batch_probe(P, long_ids, K, 2)
    j_rows = join_probe(P, a.coeffs, a, jm.L, jm.n_layers)
    probes = (public, pf_public, b_public, [j_rows, j_rows])
    if a.keyspec is not None:
        sr.prepare(*probes)
        return write_keyspec(a.keyspec, cfg, keys, sr.session_kwargs(), bm)
    binfo = check_bundle(bundle, cfg, bm) if bundle is not None else None
    out = open(a.out, "w")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)
    t0 = time.time()
    sr.setup(*probes)
    setup_s = time.time() - t0
    dev = DeviceServer(sr, st)
    progs = {f"{pre_}{lab}": p for pre_, run in (("", dec), ("pf_", pf), ("b_", bat), ("join_", join))
             for lab, p in run.progs.items()}
    emit({"kind": "header", "driver": "perseus.examples.rwkv7.run_planned --serve", **keys_header(dec, binfo),
          "width": P.C, "shards": dm.L.S, "lanes": dm.L.t, "lora": dm.W[0].lora, "shard_land": dm.shard_land,
          "blocks": dm.n_layers, "batch": K, "prefill_chunk": T, "generate": a.generate, "serve": a.serve,
          "arrivals": [[p, ids] for p, ids in arrivals], "prefix": pre, "plan": a.plan, "pf_plan": a.pf_plan,
          "b_plan": a.b_plan, "join_plan": a.join_plan, "pcmm_cf": a.pcmm_cf, "wbits": a.wbits,
          "schedule": a.schedule, "pcmm_cf_table": a.pcmm_cf_table, "pool": a.pool, "stc_cache": a.stc_cache,
          "fuse_l1": a.fuse_l1, "shifts": a.shifts, "setup_s": round(setup_s, 1), "registers": sr.n_regs,
          "programs": {lab: {k: v for k in ("n_ops", "n_bootstraps", "n_launches", "n_finishes")
                             if isinstance(v := getattr(p, k, None), (int, float))} for lab, p in progs.items()},
          "mem_report_text": str(sr.sess.mem_report),
          "pth_sha256": common.sha256(a.pth), "coeffs_sha256": common.sha256(a.coeffs),
          "environment": common.environment()})
    idle_tok = idle_token(P, a.coeffs, pre)
    if not a.no_warmup:                       # untimed, public: every program once (CUDA graphs), then a clean start
        t0 = time.time()
        wl = [P.emb[t] for t in long_ids]
        dev.prefill(0, pf_public[0], True)
        dev.decode(0, wl[0], False)
        dev.join(0)
        dev.leave([0])
        dev.batch(np.stack(wl[:K]))
        dev.reset()
        emit({"kind": "warmup", "seconds": round(time.time() - t0, 3)})
    mir = Server(MirrorBackend(P, co, st, dm.L.n, K, wbits=a.wbits, lanes=a.lanes, lora=a.lora, fuse_l1=a.fuse_l1), K)
    scorers = {}

    def pick(h):
        return int(np.argmax(head_model.head(np.asarray(h, dtype=np.float64))))

    recs = []

    def record(cv, i, token, hidden, aux, mode, lane):
        if cv.k not in scorers:
            scorers[cv.k] = common.Scorer(P, a, poly=common.prefix_state(P, a.coeffs, all_ids[cv.k], a.prefix)[0])
        r = {"kind": "token", "mode": mode, "conv": cv.k, "lane": lane, "index": i,
             "phase": "prompt" if i < len(cv.prompt) else "generate",
             **scorers[cv.k].score(a.prefix + i, token, hidden),
             "hidden_relErrRms_vs_mirror": common.rel_rms(hidden, aux) if aux is not None else None}
        recs.append(r)
        emit(r)
    events = []

    def ev(e):
        events.append(e)
        emit(e)
    t_loop = time.time()
    convs, _idle = serve_loop(TeeServer(dev, mir), arrivals, K=K, T=T, generate=a.generate, emb=P.emb, pick=pick,
                              idle_token=idle_tok, record=record, emit=ev)
    loop_s = time.time() - t_loop
    by = {k: [e["seconds"] for e in events if e["kind"] == k] for k in ("pass", "prefill", "decode", "join", "leave")}
    gen = [r for r in recs if r["phase"] == "generate"]
    emit({"kind": "summary", "batch": K, "prefill_chunk": T, "conversations": len(convs), "tokens": len(recs),
          "top1_agree": sum(r["top1_agree"] for r in recs),
          "top1_agree_per_conv": [sum(r["top1_agree"] for r in recs if r["conv"] == cv.k) for cv in convs],
          "tokens_per_conv": [len(cv.hidden) for cv in convs],
          "lanes": [[cv.k, cv.lane, cv.arrival, cv.joined, cv.left] for cv in convs],
          "generated_ids": [cv.toks[len(cv.prompt):] for cv in convs],
          "poly_sites_out_of_range_tokens": [(r["conv"], r["mode"], r["pos"], r["poly_sites_out_of_range"])
                                             for r in recs if r["poly_sites_out_of_range"]],
          "logits_relErrRms_max": max(r["logits_relErrRms"] for r in recs),
          "logits_relErrRms_median": float(np.median([r["logits_relErrRms"] for r in recs])),
          "hidden_relErrRms_vs_mirror_max": max(r["hidden_relErrRms_vs_mirror"] for r in recs),
          "passes": len(by["pass"]), "seconds_per_pass_median": float(np.median(by["pass"])) if by["pass"] else None,
          "active_lanes_mean": float(np.mean([e["active"] for e in events if e["kind"] == "pass"])) if by["pass"] else None,
          "prefill_passes": len(by["prefill"]), "prefill_seconds_median": float(np.median(by["prefill"])) if by["prefill"] else None,
          "decode_steps": len(by["decode"]), "decode_seconds_median": float(np.median(by["decode"])) if by["decode"] else None,
          "joins": len(by["join"]), "join_seconds_median": float(np.median(by["join"])) if by["join"] else None,
          "leave_runs": len(by["leave"]), "leave_seconds_median": float(np.median(by["leave"])) if by["leave"] else None,
          "loop_s": round(loop_s, 1), "generated_tokens": len(gen),
          "aggregate_generated_tokens_per_s": round(len(gen) / loop_s, 4) if loop_s else None,
          "aggregate_tokens_per_s": round(len(recs) / loop_s, 4) if loop_s else None,
          "setup_s": round(setup_s, 1), "wall_s": round(time.time() - t_all, 1),
          "planned_stats": str(sr.sess.planned_stats())})
    sr.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True, help="JSON lines, one per token, then the summary line")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--pcmm-cf", type=int, default=6)
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--pool", type=int, default=16, help="ciphertext pool slots (k = 1): the carried state is 4 per block")
    ap.add_argument("--generate", type=int, default=0,
                    help="free-running steps after the prompt: the client's greedy pick (top1_fhe) is fed back")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--schedule", default="program", choices=("program", "siblings"),
                    help="Program.compile's schedule (siblings: what a PAIR=1 plan needs to form its refresh pairs)")
    ap.add_argument("--pcmm-cf-table", default=None,
                    help="a CF per PCMM linear (python -m perseus.examples.fhe_ssm.pcmm_cf_table CAPTURE OUT.json); "
                    "a linear not in it at --pcmm-cf")
    ap.add_argument("--pf-plan", default=None,
                    help="with --prefill-chunk T: the plan cut from run_capture --prefill-chunk T (same shifts file)")
    ap.add_argument("--prompts", default=None,
                    help="with --batch K: one prompt per line (\\n escapes a newline), the first K are the batch")
    ap.add_argument("--keys", default="inproc", metavar="inproc|bundle|DIR",
                    help="inproc: in-process keys (INSECURE: the session holds the secret; recorded insecure-inproc); "
                    "bundle: the client's bundle directory in $RWKV7_KEYS; DIR: that directory (keys.pckb + "
                    "client_secret.bin, from --keygen)")
    ap.add_argument("--serve", default=None, metavar="FILE",
                    help="continuous batching (SERVE_DESIGN.md): FILE rows '<arrival pass> <prompt>'; needs --batch K, "
                    "--prefill-chunk T, --plan (decode), --pf-plan, --b-plan, --join-plan")
    ap.add_argument("--b-plan", default=None, help="with --serve: the batched plan (run_capture --batch K)")
    ap.add_argument("--join-plan", default=None, help="with --serve: the join plan (run_capture --join --batch K)")
    ap.add_argument("--keyspec", default=None, metavar="FILE",
                    help="write the session's key request (Session.key_manifest) for these plans and stop")
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--keygen" in argv:
        return keygen_main(argv)
    a = ap.parse_args(argv)
    if a.serve:
        return main_serve(a)
    if a.batch > 1:
        if a.prefill_chunk:
            raise SystemExit("--batch and --prefill-chunk do not combine")
        return main_batch(a)
    T = int(a.prefill_chunk)
    if T and not (a.pf_plan and a.prefix):
        raise SystemExit("--prefill-chunk needs --pf-plan and a public prefix (--prefix >= 1)")
    from perseus import _pc
    from perseus.examples.fhe_ssm.run_planned import session_config

    t_all = time.time()
    P, co, shifts = common.load(a)
    all_ids = common.token_ids(a)
    poly, st = common.prefix_state(P, a.coeffs, all_ids, a.prefix)
    ids = all_ids[a.prefix:][: a.tokens]
    fhe, _inf, model = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts,
                                  **model_kw(a))
    if a.prefix:
        model.init_state = st                    # the twin's probes start where the device does
    cfg = session_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    bundle, client = resolve_keys(a) if a.keyspec is None else (None, None)
    labels = [f"block_{b}" for b in range(model.n_layers + 1)]
    work = a.work_dir or a.out + ".work"
    out = open(a.out, "w")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)

    cf_map = None
    if a.pcmm_cf_table:
        from perseus.examples.fhe_ssm import pcmm_cf_table
        cf_map = pcmm_cf_table.load(a.pcmm_cf_table)
    lever = dict(schedule=a.schedule, pcmm_cf_map=cf_map, bundle=bundle, client=client)
    run = Rwkv7PlannedRun(Recorder(model, model.n_layers), a.plan, cfg, labels, keys, model=model, work_dir=work,
                          pcmm_cf=a.pcmm_cf, pt_rows=a.pt_rows, stc_cache=a.stc_cache, pool=a.pool, **lever)
    try:                                         # the probes' rows: public, in distribution (probe_rows)
        calib = common.token_ids(a, common.CALIB_TEXT)
    except SystemExit:                           # --ids without --harness: no tokenizer
        calib = []
    public = probe_rows(P, calib, all_ids[: a.prefix])
    both = None
    t0 = time.time()
    if T:
        # the prefill twin: the same weights and shifts, T-token passes; its probes are two public chunks
        _f2, _i2, pmodel = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts, chunk=T,
                                      **model_kw(a))
        pmodel.init_state = st
        prun = Rwkv7PlannedRun(Recorder(pmodel, pmodel.n_layers), a.pf_plan, cfg, labels, keys, model=pmodel,
                               work_dir=work + ".pf", pcmm_cf=a.pcmm_cf, pt_rows=a.pt_rows, stc_cache=a.stc_cache,
                               pool=a.pool, **lever)
        long_ids = common.token_ids(a, common.CALIB_TEXT_LONG)[a.prefix:]
        pf_public = [[P.emb[t] for t in long_ids[i * T:(i + 1) * T]] for i in range(2)]
        both = Rwkv7PrefillRun(run, prun, T)
        if a.keyspec is not None:
            both.prepare(public, pf_public)
            return write_keyspec(a.keyspec, cfg, keys, both.session_kwargs(), model)
        binfo = check_bundle(bundle, cfg, model) if bundle is not None else None
        both.setup(public, pf_public)
    else:
        if a.keyspec is not None:
            run.prepare(public)
            return write_keyspec(a.keyspec, cfg, keys, run.session_kwargs(), model)
        binfo = check_bundle(bundle, cfg, model) if bundle is not None else None
        run.setup(public)
    setup_s = time.time() - t0
    kv_dec = run.state_cts(st) if a.prefix else None
    kv_pf = both.pf.state_cts(st) if both else None
    progs = {lab: p for lab, p in run.progs.items()}
    if both:
        progs.update({f"pf_{lab}": p for lab, p in both.pf.progs.items()})
    emit({"kind": "header", "driver": "perseus.examples.rwkv7.run_planned", **keys_header(run, binfo),
          "width": P.C, "shards": model.L.S, "lanes": model.L.t, "lora": model.W[0].lora, "shard_land": model.shard_land,
          "blocks": model.n_layers, "prefix": all_ids[: a.prefix], "tokens": ids, "plan": a.plan, "pcmm_cf": a.pcmm_cf, "wbits": a.wbits,
          "prefill_chunk": T, "pf_plan": a.pf_plan, "schedule": a.schedule, "pcmm_cf_table": a.pcmm_cf_table,
          "pcmm_cf_table_sha256": common.sha256(a.pcmm_cf_table) if a.pcmm_cf_table else None,
          "fuse_l1": a.fuse_l1, "shifts": a.shifts, "setup_s": round(setup_s, 1),
          "programs": {lab: {k: v for k in ("n_ops", "n_bootstraps", "n_launches", "n_finishes")
                             if isinstance(v := getattr(p, k, None), (int, float))}
                       for lab, p in progs.items()},
          "mem_report_text": str(run.sess.mem_report),
          "pth_sha256": common.sha256(a.pth), "coeffs_sha256": common.sha256(a.coeffs),
          "environment": common.environment()})
    if not a.no_warmup:
        t0 = time.time()
        reg, _ = run.token(0, public)
        run.reply(reg)
        emit({"kind": "warmup", "seconds": round(time.time() - t0, 3)})
        if both:                                 # untimed: a public chunk from the public prefix state
            t0 = time.time()
            both.reply(both.prefill(pf_public[0], kv_cts=both.pf.state_cts(st)), T)
            emit({"kind": "warmup", "prefill_chunk": T, "seconds": round(time.time() - t0, 3)})
    mirror = Mirror(P, co, model.L.n, wbits=a.wbits, fuse_l1=a.fuse_l1, lora=a.lora, lanes=a.lanes)
    mirror.reset(st if a.prefix else None)
    sc = common.Scorer(P, a, mirror=mirror, poly=poly)
    rows = [P.emb[t] for t in ids]
    recs, passes = [], []
    n_prompt = len(ids)
    n_pf = (n_prompt // T) * T if T else 0        # whole chunks prefilled; the remainder decoded token by token
    for c in range(n_pf // T if T else 0):
        chunk = ids[c * T:(c + 1) * T]
        t0 = time.time()
        reg = both.prefill([P.emb[t] for t in chunk], kv_cts=kv_pf if c == 0 else None)
        hid, level = both.reply(reg, T)
        sec = time.time() - t0
        prec = {"kind": "prefill", "pass": c, "T": T, "seconds": round(sec, 4),
                "seconds_per_token": round(sec / T, 4), "reply_level": level}
        passes.append(prec)
        emit(prec)
        for l, tok in enumerate(chunk):           # each lane scored as its own position, token by token
            rec = {"kind": "token", "mode": "prefill", "pass": c, "lane": l,
                   **sc.score(a.prefix + c * T + l, tok, hid[l])}
            recs.append(rec)
            emit(rec)
    for t in range(n_pf, n_prompt + a.generate):
        if t >= n_prompt:                        # free running: the client decrypts, picks greedily, feeds it back
            ids.append(int(recs[-1]["top1_fhe"]))
            rows.append(P.emb[ids[-1]])
        t0 = time.time()
        # the first token reads the prefix state (Cts) unless a prefill pass left it in the registers; t + 1 keeps
        # PlannedRun's "t == 0: empty state" rule off
        reg, info = run.token(t + 1 if a.prefix else t, [None] * (1 if a.prefix else 0) + rows,
                              kv_cts=kv_dec if t == 0 else None)
        hidden, level = run.reply(reg)
        sec = time.time() - t0
        rec = {"kind": "token", "mode": "decode", "seconds": round(sec, 4), "issue_s": info["issue_s"],
               "reply_level": level, **sc.score(a.prefix + t, ids[t], hidden)}
        recs.append(rec)
        emit(rec)
    secs = [r["seconds"] for r in recs if r["mode"] == "decode"]
    pf_s = sum(p["seconds"] for p in passes)
    emit({"kind": "summary", "tokens": len(recs), "prompt_ids": ids[:n_prompt], "generated_ids": ids[n_prompt:],
          "top1_agree": sum(r["top1_agree"] for r in recs),
          "top1_agree_prefill": sum(r["top1_agree"] for r in recs if r["mode"] == "prefill"),
          # tokens whose plaintext poly reference left a site's interval (a ciphertext cannot clamp: from there on
          # the encrypted values diverge), as (mode, position, {site: count})
          "poly_sites_out_of_range_tokens": [(r["mode"], r["pos"], r["poly_sites_out_of_range"]) for r in recs
                                             if r["poly_sites_out_of_range"]],
          "prefilled_tokens": n_pf, "prefill_chunk": T, "prefill_passes": len(passes),
          "prefill_seconds_per_token": round(pf_s / n_pf, 4) if n_pf else None,
          "logits_relErrRms_max": max(r["logits_relErrRms"] for r in recs),
          "logits_relErrRms_median": float(np.median([r["logits_relErrRms"] for r in recs])),
          "hidden_relErrRms_vs_mirror_max": max(r["hidden_relErrRms_vs_mirror"] for r in recs),
          "seconds_per_token_median": float(np.median(secs)) if secs else None, "seconds_per_token_after_first":
          float(np.mean(secs[1:])) if len(secs) > 1 else None, "setup_s": round(setup_s, 1),
          "wall_s": round(time.time() - t_all, 1), "planned_stats": str(run.sess.planned_stats())})
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
