# SPDX-License-Identifier: MIT
"""The box tools' own logic (light: no model, no capture): run_box.sh's product names never collide across settings
that change the product (review lens 2, finding 1), capture_check.py refuses a capture or plan of another
configuration, the memory model reproduces the final-coefficient session reports, and the batched preset restores
the 64-slot StC cache when K halves (review lens 3, findings 5 and 6)."""
import hashlib
import json
import os
import subprocess
import sys

from perseus.examples.rwkv7.tests.conftest import ROOT

RUN_BOX = ROOT / "tools" / "rwkv7" / "run_box.sh"
CHECK = ROOT / "tools" / "rwkv7" / "capture_check.py"
DEF = {"BLOCKS": "", "B_K": "32", "PLANOPT": "", "LANES": "0", "PF_T": "32", "LORA": "auto", "SHARD_LAND": "1",
       "CFTABLE": "", "LAYOUT": "current", "LEVERS": "", "B_CARRY": "lane", "PCMM_SHIFT": "calib", "PCMM_CF": "6"}
CAPD = ["BLOCKS", "LANES", "LORA", "SHARD_LAND", "LAYOUT", "PCMM_SHIFT"]
KEYX = ["CFTABLE", "PCMM_CF"]
DEP = {"CAP": CAPD, "CFT": CAPD, "PLAN": CAPD + ["LEVERS"], "PFCAP": CAPD + ["PF_T"], "PFPLAN": CAPD + ["PF_T", "LEVERS"],
       "BCAP": CAPD + ["B_K", "B_CARRY"], "BPLAN": CAPD + ["B_K", "B_CARRY", "LEVERS", "PLANOPT"],
       "JCAP": CAPD + ["B_K"], "JPLAN": CAPD + ["B_K", "LEVERS"]}
DEP["KEYDIR_decode"] = DEP["R_RUN"] = DEP["PLAN"] + KEYX
DEP["KEYDIR_pf"] = DEP["R_PF"] = DEP["PFPLAN"] + KEYX
DEP["KEYDIR_b"] = DEP["R_BATCH"] = DEP["BPLAN"] + KEYX
DEP["KEYDIR_s"] = DEP["R_SERVE"] = sorted(set(DEP["BPLAN"] + DEP["PFPLAN"] + KEYX))


def _names(cfg: str) -> dict:
    s = dict(DEF)
    s.update(kv.split("=", 1) for kv in cfg.split())
    env = {k: v for k, v in os.environ.items() if k not in DEF}
    env.update(s)
    env["OUT"] = "/OUT"
    out = subprocess.run(["bash", str(RUN_BOX), "names"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return s, dict(line.split("=", 1) for line in out.stdout.splitlines())


def _norm(x):
    return "" if x == "base" else x


def test_run_box_names_do_not_collide():
    """Over the review's colliding pairs and one change of every naming setting, two configurations share a path
    only when they agree on every setting that product depends on."""
    cfgs = ["", "BLOCKS=2 B_K=32 PLANOPT=", "BLOCKS= B_K=32 PLANOPT=b2", "BLOCKS=4", "B_K=4", "BLOCKS=1", "B_K=1",
            "LANES=32", "LANES=16", "PF_T=16", "B_K=16", "LORA=split", "SHARD_LAND=0", "CFTABLE=1", "LAYOUT=filled",
            "LEVERS=best", "PLANOPT=b2", "PLANOPT=base", "B_CARRY=decode", "PCMM_SHIFT=zero", "PCMM_CF=4",
            "BLOCKS=16 B_K=2", "BLOCKS=1 B_K=6", "B_K=1 PF_T=16", "B_K=16 PF_T=1"]
    by_path = {}
    for c in cfgs:
        s, names = _names(c)
        for k, v in names.items():
            by_path.setdefault(v, []).append((c, s, k))
    bad = []
    for v, users in by_path.items():
        for i, (c1, s1, k1) in enumerate(users):
            for c2, s2, k2 in users[i + 1:]:
                if k1 != k2 or any(_norm(s1[d]) != _norm(s2[d]) for d in DEP[k1]):
                    bad.append((v, k1, c1, k2, c2))
    assert not bad, bad[:5]
    # the review's pairs, verbatim
    assert _names("BLOCKS=2 B_K=32 PLANOPT=")[1]["BPLAN"] != _names("BLOCKS= B_K=32 PLANOPT=b2")[1]["BPLAN"]
    assert _names("BLOCKS=4")[1]["CAP"] != _names("B_K=4")[1]["BCAP"]
    assert _names("PF_T=16")[1]["KEYDIR_pf"] != _names("PF_T=32")[1]["KEYDIR_pf"]
    assert _names("B_K=16")[1]["KEYDIR_b"] != _names("")[1]["KEYDIR_b"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _check(cap, *args, env=None):
    e = {k: v for k, v in os.environ.items() if k != "CAPTURE_MISMATCH_OK"}
    e.update(env or {})
    return subprocess.run([sys.executable, str(CHECK), str(cap), *args], cwd=ROOT, env=e, capture_output=True,
                          text=True, check=False)


def test_capture_check(tmp_path):
    pth, co = tmp_path / "m.npz", tmp_path / "c.json"
    pth.write_bytes(b"weights")
    co.write_text("{}")
    cap = tmp_path / "cap"
    (cap / "block_0").mkdir(parents=True)
    (cap / "block_1").mkdir()
    (cap / "block_0" / "graph.json").write_text('{"a": 1}')
    (cap / "block_1" / "graph.json").write_text('{"b": 2}')
    meta = {"blocks": 2, "batch": 4, "batch_carry": "lane", "prefill_chunk": 0, "pth_sha256": _sha(pth),
            "coeffs_sha256": _sha(co), "width": 768, "shards": 1, "lanes": 32, "layout": "current", "lora": "fold",
            "shard_land": False}
    (cap / "rwkv7_capture.json").write_text(json.dumps(meta))
    base = ["--kind", "b", "--pth", str(pth), "--coeffs", str(co), "--lanes", "0", "--layout", "current", "--blocks",
            "2", "--batch", "4", "--sha-cache", str(tmp_path / "sha.json")]
    r = _check(cap, *base)
    assert r.returncode == 0, r.stdout + r.stderr
    for flag, val, field in (("--batch", "8", "batch"), ("--lanes", "16", "lanes"), ("--blocks", "3", "blocks"),
                             ("--layout", "filled", "layout"), ("--lora", "split", "lora")):
        args = list(base)
        if flag in args:
            args[args.index(flag) + 1] = val
        else:
            args += [flag, val]
        r = _check(cap, *args)
        assert r.returncode == 3 and f"MISMATCH {field}" in r.stdout, (flag, r.stdout)
    co.write_text('{"x": 1}')                                   # other coefficients, same path
    r = _check(cap, *base)
    assert r.returncode == 3 and "MISMATCH coeffs_sha256" in r.stdout
    assert _check(cap, *base, env={"CAPTURE_MISMATCH_OK": "1"}).returncode == 0
    co.write_text("{}")
    # the plan's graph md5 (make_plan.sh's graph= line: cat block_*/graph.json | md5)
    md5 = hashlib.md5(b'{"a": 1}{"b": 2}').hexdigest()
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "PLAN_CMD.txt").write_text(f"graph={cap} (md5 of the stages' graph.json {md5})\n")
    assert _check(cap, *base, "--plan", str(plan)).returncode == 0
    (plan / "PLAN_CMD.txt").write_text(f"graph={cap} (md5 of the stages' graph.json {'0' * 32})\n")
    r = _check(cap, *base, "--plan", str(plan))
    assert r.returncode == 3 and "MISMATCH plan" in r.stdout
    # a decode capture is not a batched one
    r = _check(cap, *[a if a != "b" else "decode" for a in base])
    assert r.returncode == 3 and "MISMATCH batch" in r.stdout


def test_memory_model_reproduces_the_final_reports():
    """ckpt_info.memory_mb against the final-coefficient mem_report_text totals (new_pod/r01b_final/run_best_P1,
    r04b_final/run_best_P1, r01b_final/batch32_best): 70,980.0 / 77,844.9 / 91,019.6 MB, never under, <= +1 %."""
    from perseus.examples.rwkv7.ckpt_info import memory_mb, preset
    from perseus.examples.rwkv7.synth import lora_ranks

    def m(nl, C):
        return {"n_layer": nl, "width": C, "d_ffn": 4 * C, "lora": lora_ranks(C)}
    for mm, kw, total in ((m(12, 768), {"pool": 128, "stc": 160}, 70980.0),
                          (m(24, 1024), {"pool": 224, "stc": 160}, 77844.9),
                          (m(12, 768), {"K": 32, "pool": 1024, "stc": 64}, 91019.6)):
        got = memory_mb(mm, **kw)["total"]
        assert -0.001 <= got / total - 1 <= 0.01, (kw, got, total)
    # the 0.4B batched preset: K = 16 with the 64-slot StC cache restored after halving
    p = preset(m(24, 1024))
    assert p["B_K"] == 16 and p["STC_CACHE_B"] == 64


def test_filled_prefill_scopes_escape_b2():
    """The b2 forbidden substrings cannot match the filled prefill's labels (prefill.py: pf_xpose / pf_chain /
    pf_readout) - review lens 1, finding 1."""
    src = (ROOT / "python" / "perseus" / "examples" / "rwkv7" / "prefill.py").read_text()
    for lab in ("pf_xpose", "pf_chain", "pf_readout"):
        assert f'self._scope("{lab}")' in src
    for bad in ('self._scope("readout")', 'self._scope("xpose")', 'self._scope("chain")'):
        assert bad not in src
