#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The model's own free-running text for the range scan (REVIEW_20261105.md §5.A6): greedy generation with the
research harness (torch), one JSON line per prompt, and the texts concatenated into a scan file. The batched run's
collapse (pass 18, an a_sigmoid excursion) came from a conversation's OWN generated tokens, which no WikiText /
LAMBADA scan covers. The default mode is poly UNCLAMPED (what the ciphertext computes); --exact for the exact model.
(new_pod/gen_greedy.py, generalised: the harness path and a prompt file; KV-free, so O(n^2) per prompt.)

    python tools/rwkv7/gen_greedy.py --harness H --model M.pth --coeffs C.json --prompts tools/rwkv7/batch_prompts.txt \\
        --n 256 --out gen.jsonl --text-out gen.txt
"""
import argparse
import json
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--coeffs", default=None)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--clamp", action="store_true")
    ap.add_argument("--prompts", required=True, help="one prompt per line, \\n escapes (run_planned.read_prompts)")
    ap.add_argument("--max-prompts", type=int, default=32)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-out", required=True, help="prompt + generation of every prompt, \\n\\n-separated")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.harness)
    import torch
    from rwkv7_model import Sites, load_rwkv7, pick_device
    from rwkv_tokenizer import RWKVTokenizer
    prompts = []
    for line in open(a.prompts, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.strip():
            s = line.replace("\\\\", "\0").replace("\\n", "\n").replace("\0", "\\")
            prompts.append(s if s.startswith("\n\n") else "\n\n" + s)
    prompts = prompts[: a.max_prompts]
    tok, sites, dev = RWKVTokenizer(), Sites("exact"), pick_device("auto")
    model = load_rwkv7(a.model, device=dev, sites=sites)
    mode = "exact"
    if not a.exact:
        if not a.coeffs:
            raise SystemExit("--coeffs (or --exact)")
        sites.load_poly(json.load(open(a.coeffs)), clamp=a.clamp)
        sites.mode = "poly"
        mode = "poly" + ("-clamp" if a.clamp else "-unclamped")
    with open(a.out, "w") as out, open(a.text_out, "w") as txt:
        for i, text in enumerate(prompts):
            ids, gen, t0 = tok.encode(text), [], time.time()
            with torch.no_grad():
                for _ in range(a.n):
                    logits = model(torch.tensor([ids + gen], device=dev))[0, -1]
                    if not torch.isfinite(logits).all():
                        break
                    gen.append(int(torch.argmax(logits)))
            rec = {"i": i, "mode": mode, "coeffs": a.coeffs, "prompt": text, "prompt_ids": ids, "generated_ids": gen,
                   "generated_text": tok.decode(gen), "finite": len(gen) == a.n, "seconds": round(time.time() - t0, 1)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            txt.write(text + tok.decode(gen) + "\n\n")
            print(json.dumps({k: rec[k] for k in ("i", "finite", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
