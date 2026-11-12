# `tools/rwkv7/` — RWKV-7 Checkpoints, Harness, Coefficients & Runners

Complete step-by-step instructions and measured GPU results are in [`python/perseus/examples/rwkv7/README.md`](../../python/perseus/examples/rwkv7/README.md).

## Quickstart

```bash
# 1. Build Python environment + CUDA extension
bash tools/rwkv7/run_box.sh build

# 2. Download & SHA-256 verify 0.1B checkpoint into .cache/rwkv7/ (use MODELS=0.1b,0.4b WITH_DATA=1 for 0.4B & datasets)
bash tools/rwkv7/run_box.sh fetch

# 3. Verify _pc bindings, planner dependencies, and CPU test suite
bash tools/rwkv7/run_box.sh check

# 4. Run 0.1B encrypted decode on GPU with no secret key in the server session (KEYS=bundle)
export LEVERS=best STC_CACHE=160 KEYS=bundle
bash tools/rwkv7/run_box.sh capture plan
KEYMODE=decode bash tools/rwkv7/run_box.sh keyspec keygen
bash tools/rwkv7/run_box.sh run

# 5. Headline batched decode (K=32, filled layout, b2, low memory)
export LAYOUT=filled PLANOPT=b2 B_K=32 B_GENERATE=32
bash tools/rwkv7/run_box.sh capture_b plan_b bcheck
KEYMODE=b bash tools/rwkv7/run_box.sh keyspec keygen
RWKV7_SHED_TWIN=1 RWKV7_LEAN_PREFIX=1 POOL_B=160 STC_CACHE_B=1 bash tools/rwkv7/run_box.sh batch
```

## Directory Contents

| Path | Description |
|---|---|
| `run_box.sh` | Main box driver (`fetch`, `build`, `check`, `crosscheck`, `capture`, `plan`, `run`, `pf`, `b`, `s`, `keyspec`, `keygen`, `names`). Defaults to local `.cache/rwkv7/` and built-in `coeffs/` / `harness/`. |
| `size_pipeline.sh` | End-to-end pipeline (`calibrate -> fit -> gen -> scan -> gate -> capture -> plan -> keys -> GPU`) for any `RWKV-x070-World-*` checkpoint. |
| `fetch_data.py` | Downloads and SHA-256 verifies `BlinkDL/rwkv-7-world` checkpoints (`0.1b`, `0.4b`, `1.5b`, `2.9b`) and optional WikiText-103 / LAMBADA corpora. |
| `harness/` | Plaintext PyTorch polynomial-nonlinearity harness (`calibrate.py`, `fit.py`, `eval.py`, `scan_ranges.py`, `rsqrt_const.py`, `heal.py`, `rwkv_tokenizer.py`). |
| `raw/` | Reference `BlinkDL/RWKV-LM` Apache-2.0 files and `rwkv_vocab_v20230424.txt` (read by `harness/rwkv_tokenizer.py`). |
| `coeffs/` | Pre-fitted polynomial/Newton coefficients (`0.1b/`, `0.4b/`, `1.5b/`), calibrated `ranges_ctx4096.json`, and gate verdicts (`1.5b/verdict_final.json`). |
| `batch_prompts.txt` | 32 prompts used in `K=32` batched decode runs. |
| `longrun_prompts64.txt` | 64 prompts (`L1`, `L5`, `L9`, `L25` used in `T=32` chunked prefill runs). |
| `serve_arrivals.txt` | 8-conversation arrival schedule used in continuous-batching (`serve`) runs. |
| `capture_check.py`, `gate_verdict.py`, `gen_greedy.py`, `harness_logits.py`, `run_records.py`, `planopt.sh`, `plan_sizes.py`, `size_table.py`, `filled_table.py`, `b2_census.py`, `b2_plan.sh`, `convert_newton_scaled.py`, `merge_relu2.py`, `relu2_abs_targets.py`, `sampler.sh` | Helper scripts for verification, gate evaluation, planning options, and telemetry. |

## License

This directory is under the MIT license ([`LICENSE`](LICENSE)). Any commercial use of these files as part of Perseus (`github.com/gladia-research-group/perseus`) requires prior written permission from Alessandro Zirilli or Davide Marincione, under Perseus's Business Source License 1.1; on their own the files are MIT. Reference files in `raw/` are Apache-2.0 (`BlinkDL/RWKV-LM`).

