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

KEYS: in-process (``KeySource.generate``): the session holds the secret. A correctness and timing run, not a
deployment (recorded as ``keys: insecure-inproc``).

    python -m perseus.examples.rwkv7.run_planned --pth P --coeffs C --harness H --plan PLAN --shifts S \\
        --tokens 12 --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from perseus.examples.fhe_ssm.planned import N32_Q_PRIMES, SsmRecorder, digest

from perseus.impl.driver import PlannedRun

from . import common
from .layout import D
from .model import STATE, Mirror
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
                 stc_cache=8, pool=16, log=print):
        super().__init__(recorder, plan_dir, config, labels, keys, ref_token=1, probe=2, work_dir=work_dir,
                         pt_rows=pt_rows, dyn_rot=[], stc_cache=stc_cache, pool=pool, log=log)
        self.model, self.pcmm_cf = model, int(pcmm_cf)
        self.ops = None
        self.static = False

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
        super().prepare(rows, argmax)
        if any(self.dyn.values()):
            raise RuntimeError(f"dynamic steps in the stream: {self.dyn}")
        for b, r in self.ref.items():
            o = self.records[0][b]
            if set(o.pts) != set(r.pts) or any(not np.array_equal(o.pts[k], r.pts[k]) for k in r.pts):
                raise RuntimeError(f"block {b}: a plaintext differs between the probe tokens (not static)")
        bad = [b for b in range(self.rec.n_layers) if len(self.kv.get(b, ())) != self.model.L.M + 2]
        if bad:
            raise RuntimeError(f"blocks {bad}: the carried state was not recognised as block inputs")
        self.static = True
        self.al = self.pt_aliases()
        return self

    def session_kwargs(self):
        kw = super().session_kwargs()
        kw["size"] = {"hidDim": D, "dim": D, "numHeads": 1, "numHeadsReal": 1, "expDim": 3 * D, "expanded": 3 * D,
                      "seqLen": 1}
        lins = self.model.linears()
        kw["pcmm"] = [(n, int(l.t), int(l.n_in), int(l.n_out), int(l.Wd.shape[0]), float(l.w_scale), int(l.shift))
                      for n, l in lins.items()] or None
        kw["pcmm_cf"] = self.pcmm_cf
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
        items, n = {}, 0
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
        n = int(self.store.stage([(nm, lv, v) for (nm, lv), v in items.items()]))
        self.log(f"[planned] {len(self.model.linears())} PCMM linears uploaded, {n} plaintext slots staged")
        return self

    def input_ct(self, t, rows, p):
        level = int(p.inputs[0][1])
        return self.fhe.pack_tokens(np.asarray(self.model.L.encode_row(rows[t]), dtype=np.float64), level)

    def state_cts(self, state) -> dict:
        """The public prefix state as each block's carried inputs ``{block: {input position: Ct}}``: packed and
        landed by an eager bootstrap (as the twin's ``start`` lands it), then dropped to the compiled input level if
        it lands richer. Untimed setup: the state is public."""
        names = [*STATE[: self.model.L.M], "xa", "xf"]
        rows = self.model.state_rows(state)
        out, notes = {}, []
        for b in range(self.rec.n_layers):
            p = self.progs[self.labels[b]]
            cts = {}
            for i, _j, _r in self.reg[b]["kv"]:
                key = re.search(r"cache\.([a-z0-9]+)-lvl", self.ref[b].inputs()[i]).group(1)
                lvl, deg = int(p.inputs[i][1]), int(p.inputs[i][2])
                r = np.asarray(rows[b][names.index(key)], dtype=np.float64)
                try:
                    ct = self.fhe.pack_tokens(r, int(self.ops.bts_out()))
                    self.ops.bootstrap(ct)
                    how = "bootstrap"
                except Exception as e:          # an eager refresh refused on this session: pack at the input level
                    ct = self.fhe.pack_tokens(r, lvl)
                    how = f"packed ({type(e).__name__}: {e})"
                if int(ct.level) < lvl:
                    ct = self.ops.level_drop(ct, lvl)
                notes.append((b, key, lvl, deg, int(ct.level), int(ct.noise_deg), how))
                cts[i] = ct
            out[b] = cts
        self.log(f"[planned] prefix state: (block, name, compiled level, deg, ct level, deg, how) {notes[:6]} ...")
        return out

    def token(self, t, rows, argmax=False, entry=None, variant=None, kv_cts=None):
        """Every stage on the device; the carried inputs from the registers, or from ``kv_cts`` (``state_cts``)
        on the first token of a sequence that starts from a public state."""
        tt = time.time()
        n_blocks = self.rec.n_layers + 1
        for b in range(n_blocks):
            x = self.input_ct(t, rows, self.progs[self.labels[0]]) if b == 0 else self.reg[b - 1]["out"][0][1]
            if kv_cts is None or b not in kv_cts:
                self._run_block(b, t, {}, x)
                continue
            p = self.progs[self.labels[b]]
            ins = [None] * len(p.inputs)
            ins[0] = x
            for i, _j, _r in self.reg[b]["kv"]:
                ins[i] = kv_cts[b][i]
            p.run(ins, [(j, r) for _i, j, r in self.reg[b]["kv"]] + list(self.reg[b]["out"]))
        return self.reg[n_blocks - 1]["out"][0][1], {"issue_s": round(time.time() - tt, 4)}

    def reply(self, reg):
        ct = self.ct(reg)
        vals = np.asarray(self.ops.decrypt_slots_complex(ct))
        return self.model.L.decode(np.real(vals)), int(ct.level)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True, help="JSON lines, one per token, then the summary line")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--pcmm-cf", type=int, default=6)
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    from perseus import _pc
    from perseus.examples.fhe_ssm.run_planned import session_config

    t_all = time.time()
    P, co, shifts = common.load(a)
    all_ids = common.token_ids(a)
    poly, st = common.prefix_state(P, a.coeffs, all_ids, a.prefix)
    ids = all_ids[a.prefix:][: a.tokens]
    fhe, _inf, model = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts)
    if a.prefix:
        model.init_state = st                    # the twin's probes start where the device does
    cfg = session_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    labels = [f"block_{b}" for b in range(model.n_layers + 1)]
    work = a.work_dir or a.out + ".work"
    out = open(a.out, "w")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)

    run = Rwkv7PlannedRun(Recorder(model, model.n_layers), a.plan, cfg, labels, keys, model=model, work_dir=work,
                          pcmm_cf=a.pcmm_cf, pt_rows=a.pt_rows, stc_cache=a.stc_cache)
    public = [P.emb[0]] * 2                      # the probes' rows: public (structure only)
    t0 = time.time()
    run.setup(public)
    setup_s = time.time() - t0
    kv0 = run.state_cts(st) if a.prefix else None
    emit({"kind": "header", "driver": "perseus.examples.rwkv7.run_planned", "keys": "insecure-inproc",
          "blocks": model.n_layers, "prefix": all_ids[: a.prefix], "tokens": ids, "plan": a.plan, "pcmm_cf": a.pcmm_cf, "wbits": a.wbits,
          "fuse_l1": a.fuse_l1, "shifts": a.shifts, "setup_s": round(setup_s, 1),
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
    mirror = Mirror(P, co, model.L.n, wbits=a.wbits, fuse_l1=a.fuse_l1)
    mirror.reset(st if a.prefix else None)
    sc = common.Scorer(P, a, mirror=mirror, poly=poly)
    rows = [P.emb[t] for t in ids]
    recs = []
    for t in range(len(ids)):
        t0 = time.time()
        # the first token reads the prefix state (Cts); t + 1 keeps PlannedRun's "t == 0: empty state" rule off
        reg, info = run.token(t + 1 if a.prefix else t, [None] * (1 if a.prefix else 0) + rows,
                              kv_cts=kv0 if t == 0 else None)
        hidden, level = run.reply(reg)
        sec = time.time() - t0
        rec = {"kind": "token", "seconds": round(sec, 4), "issue_s": info["issue_s"], "reply_level": level,
               **sc.score(a.prefix + t, ids[t], hidden)}
        recs.append(rec)
        emit(rec)
    secs = [r["seconds"] for r in recs]
    emit({"kind": "summary", "tokens": len(recs), "top1_agree": sum(r["top1_agree"] for r in recs),
          "logits_relErrRms_max": max(r["logits_relErrRms"] for r in recs),
          "logits_relErrRms_median": float(np.median([r["logits_relErrRms"] for r in recs])),
          "hidden_relErrRms_vs_mirror_max": max(r["hidden_relErrRms_vs_mirror"] for r in recs),
          "seconds_per_token_median": float(np.median(secs)), "seconds_per_token_after_first":
          float(np.mean(secs[1:])) if len(secs) > 1 else None, "setup_s": round(setup_s, 1),
          "wall_s": round(time.time() - t_all, 1), "planned_stats": str(run.sess.planned_stats())})
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
