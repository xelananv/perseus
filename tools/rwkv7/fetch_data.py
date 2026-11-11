# SPDX-License-Identifier: MIT
"""Download and SHA-256 verify RWKV-7 World checkpoints and evaluation datasets.

Checkpoints (BlinkDL/rwkv-7-world, Apache-2.0):
  0.1b   RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
  0.4b   RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
  1.5b   RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth
  2.9b   RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth

Datasets (with --with-data):
  Salesforce/wikitext  (wiki.valid.raw, wiki.test.raw)
  EleutherAI/lambada_openai  (data/lambada_test.jsonl)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS: dict[str, tuple[str, str]] = {
    "0.1b": (
        "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
        "60c98129b9529963bff2c164b8ab4bd17c19332ae06dc2dcae32aa3a3739295a",
    ),
    "0.4b": (
        "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
        "c08cf602b33e59a5717b8f0cafaf7f04c50c1c67166e477aa2f47c5ca180da4a",
    ),
    "1.5b": (
        "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
        "23e13ff62ac6f80b28910ffc230324939e80ddb00b98f892a5951c0da071e700",
    ),
    "2.9b": (
        "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
        "5d7901b91ef1646115db03d03d171370bca454061ad426a1256fda9e1c428cf8",
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_hf_file(repo_id: str, filename: str, dest_dir: Path, repo_type: str = "model") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo_id, filename, repo_type=repo_type, local_dir=str(dest_dir))
        return Path(p)
    except ImportError:
        if repo_type == "dataset":
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        else:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        target = dest_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"downloading {url} -> {target} (install huggingface_hub for faster parallel transfer)", flush=True)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target)
        return target


def ensure_model(key: str, model_dir: Path, verify: bool = True) -> Path:
    fname, want_sha = MODELS[key]
    target = model_dir / fname
    ok_file = model_dir / f"{fname}.ok"
    if target.exists():
        if ok_file.exists() and ok_file.read_text(encoding="utf-8").strip() == want_sha:
            print(f"[{key}] already present and verified: {target}")
            return target
        if not verify:
            print(f"[{key}] already present: {target}")
            return target
        got = sha256_file(target)
        if got == want_sha:
            ok_file.write_text(got + "\n", encoding="utf-8")
            print(f"[{key}] verified existing file: {target} ({got[:12]})")
            return target
        raise RuntimeError(f"[{key}] existing file {target} has sha256 {got}, expected {want_sha}")

    t0 = time.time()
    p = download_hf_file("BlinkDL/rwkv-7-world", fname, model_dir)
    if verify:
        got = sha256_file(p)
        if got != want_sha:
            raise RuntimeError(f"[{key}] downloaded {p} has sha256 {got}, expected {want_sha}")
        ok_file.write_text(got + "\n", encoding="utf-8")
    print(f"[{key}] ready: {p} ({p.stat().st_size} B, {time.time() - t0:.1f} s)")
    return p


def ensure_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    need_wt = [split for split, out_name in [("test", "wiki.test.raw"), ("validation", "wiki.valid.raw")]
               if not (data_dir / out_name).exists()]
    if need_wt:
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("pandas + pyarrow required to extract WikiText-103 parquet files: pip install pandas pyarrow") from exc
        for split, out_name in [("test", "wiki.test.raw"), ("validation", "wiki.valid.raw")]:
            out_path = data_dir / out_name
            if out_path.exists():
                print(f"[data] already present: {out_path}")
                continue
            pq = download_hf_file(
                "Salesforce/wikitext",
                f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet",
                data_dir,
                repo_type="dataset",
            )
            text = "".join(pd.read_parquet(pq)["text"])
            out_path.write_text(text, encoding="utf-8")
            print(f"[data] wrote {out_path} ({out_path.stat().st_size} B)")
    else:
        print(f"[data] WikiText-103 already present in {data_dir}")

    lam = data_dir / "data" / "lambada_test.jsonl"
    if lam.exists():
        print(f"[data] already present: {lam}")
    else:
        p = download_hf_file("EleutherAI/lambada_openai", "data/lambada_test.jsonl", data_dir, repo_type="dataset")
        print(f"[data] ready: {p} ({p.stat().st_size} B)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models",
        default="0.1b",
        help="comma-separated sizes from {0.1b,0.4b,1.5b,2.9b} or 'all' (default: 0.1b)",
    )
    ap.add_argument("--model-dir", default=str(ROOT / ".cache" / "rwkv7"), help="checkpoint directory")
    ap.add_argument("--data-dir", default=str(ROOT / ".cache" / "rwkv7" / "data"), help="dataset directory")
    ap.add_argument("--with-data", action="store_true", help="also download WikiText-103 (valid/test) and LAMBADA")
    ap.add_argument("--no-verify", action="store_true", help="skip SHA-256 verification")
    a = ap.parse_args(argv)

    if a.models == "none":
        selected = []
    elif a.models == "all":
        selected = list(MODELS)
    else:
        selected = [m.strip().lower() for m in a.models.split(",") if m.strip()]
        bad = [m for m in selected if m not in MODELS]
        if bad:
            ap.error(f"unknown model size(s) {bad}; expected any of {list(MODELS)}, 'all', or 'none'")

    model_dir = Path(a.model_dir)
    for key in selected:
        ensure_model(key, model_dir, verify=not a.no_verify)

    if a.with_data:
        ensure_data(Path(a.data_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

