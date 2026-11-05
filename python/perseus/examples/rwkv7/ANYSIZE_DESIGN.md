# RWKV-7 under CKKS at any size: design, 2026-11-05

**Goal.** Run any `RWKV-x070-World-*` checkpoint encrypted (0.1B ... 13.3B), with the approximation recipe that
solved quality vs robustness (gate 3) built into a per-size pipeline, and with client-separated keys.

**Status.**
- Built: the layout for widths above 1,024 (two variants, §3), every shape read from the checkpoint, synthetic
  checkpoints for the CPU (`synth.py`), the per-size pipeline (`tools/rwkv7/size_pipeline.sh`), the client key path
  (`run_planned.py --keyspec / --keygen / --keys`).
- Verified on the CPU only (§6). **Nothing above width 1,024 has run on a GPU.** Every second and every gigabyte in
  §4 is a model built from the planner's refresh counts and the pod-3 session reports, not a measurement.

## 1. The constraint

Ring 2^16 has n = 32,768 complex slots. The layout puts channel c of lane l at slot c·t + l, so a ciphertext holds
D channels × t lanes with D·t = 32,768. At t = 32 (what decode, prefill T = 32 and batched K = 32 use today) D =
1,024: a width-768 or -1,024 vector fits one ciphertext. Wider models (2,048 for 1.5B, 2,560 for 2.9B, 4,096 for
7.2B / 13.3B) do not.

**Why the lane count is free to change.** PCMM (the encrypted matrix product) works on exactly this layout: its MLWE
shape at t lanes has `nch = 16 · N1 / (2t) = 32,768 / t` channels per block (`impl/pcmm.py`, `Shape`). So t = 16 gives
PCMM blocks of 2,048 channels and t = 8 of 4,096. Nothing in the circuit assumes t = 32 except the head-state split:
a head's 64 × 64 state puts the value index i on the lanes, so it needs M = 64 / t ciphertexts per block of heads
(2 at t = 32, 4 at 16, 8 at 8).

## 2. What a vector costs, and why the state does not care about the layout

Every ciphertext operation (a product, a rotation, a refresh) costs the same whatever the slots mean: it is a
polynomial operation on the ring, and the ring is fixed. So the encrypted cost of a step is **the number of
ciphertexts it touches**, not the number of channels.
- **Token side** (LayerNorm, PCMM, every polynomial / Newton site, GroupNorm, the kk and bonus reductions): one
  operation per ciphertext of the vector. At width C that is ⌈C / D⌉ ciphertexts.
- **State side**: a conversation's state is C × 64 values (H = C / 64 heads of 64 × 64) = C / 512 ciphertexts of
  32,768 slots, **in every layout** (2 per 1,024 channels at t = 32; 4 for 2,048 channels at t = 16: the same 4).
  The state update, its readout and the per-conversation lane moves therefore cost about the same in every layout.

## 3. The options, judged by encrypted cost (the planner's refreshes)

| option | ciphertexts per vector | state cts per block | lanes (K, T max) | built |
|---|---|---|---|---|
| **channel shards** (`--lanes 32`): D = 1,024, S = ⌈C / 1,024⌉ shards | S | 2 S | 32 | yes |
| **fewer lanes** (`--lanes 16 / 8`, **the default**: `layout.auto_lanes`, one ciphertext per vector) | 1 (up to 4,096) | 64 / t | t = 16 (C ≤ 2,048), 8 (C ≤ 4,096) | yes |
| filled layout (two reals per complex slot, the conjugate split; `fhe_ssm`'s `pairs.py`) | — | — | 2× either | no (§7) |

The brief recommended channel shards ("confirm by cost"). **The planner does not confirm it.** 2-block captures of the
same synthetic width-2,048 checkpoint, the same coefficients, LANDING=high, refreshes per stage (`plan_cost.py`; the
census JSON lines under `.cache/rwkv7_any/`):

| width 2,048 | block 0 | block 1 | per conversation and block (K = 4 → 8) | nodes (block 1) |
|---|---|---|---|---|
| 2 shards at t = 32 | 67 | 73 | 4.0 (block 1: 87 → 103) | 5,573 |
| 1 ciphertext at t = 16 | **38** | **42** | 4.0 (block 1: 55 → 71) | 3,334 |
| (the real 0.1B, final coefficients, t = 32) | 38 | 42 | 4.0 (block 1: 54 → 70) | 3,458 |

**Mechanism.** The token side is ~30 of block 1's 42 refreshes (the norms' Newton chains, the PCMM outputs, the sites'
depth). Two shards double it (each shard is its own ciphertext through every site, every PCMM output and every carry);
one 2,048-channel ciphertext does not. The per-conversation cost is the same 4 refreshes per block in both, as §2
predicts. So at K = 1 fewer lanes is **1.7× fewer refreshes per token** (42 vs 73 in block 1), and in a batched pass
it saves the same token-side refreshes at every K. Shards only win where a run needs K or T above t (16 at 2,048), and
§4 shows memory caps K at or below that anyway (1.5B: K = 16 at t = 16; as shards the model fits only K = 8). Prefill
per prompt token is even: T = 32 tokens over 2 shards' token side, or T = 16 over one.

The price of wider ciphertexts is padding: every PCMM block is D × D, so a LoRA rank (64-480) or a d_ffn that is not a
multiple of D is padded to a whole block. At 2,048 (t = 16) that costs little (1.5B planes 8.3 GB against 7.2 GB as
shards); at t = 8 the 4,096-channel blocks inflate 2.9B's planes to 39.8 GB (18.1 GB as shards). The preset
(`ckpt_info.preset`) therefore takes the default layout only if the memory model fits it on one card, else the first
of t = 16, 32 that fits, else the smallest.

So `--lanes` defaults to one ciphertext per vector (`auto_lanes`); `--lanes 32` keeps channel shards (K, T up to 32)
for the case memory ever allows K > 16. Wider than 4,096 (none of the published sizes) the auto layout shards at 4,096
channels (t = 8).

**What sharding needed, now also used by t = 8 at 2,560** (2,560 at t = 16 is two 2,048-shards): `model.py` runs every
per-channel op per shard (a width-C vector is a list); LayerNorm adds the shards before ONE rotate-and-sum
(`channel_sum` is linear) and ONE Newton chain; a head never straddles a shard (1,024 = 16 heads), so the per-head
reductions, the transposes and the state run per shard; PCMM takes `n_in` / returns `n_out` blocks (fhe_ssm's
block-by-block pattern, here inside `impl.pcmm`'s own multi-block GEMM); prefill and batched run their scans per shard.
Two more decisions above one shard:
- **The a / v LoRA pairs split** (`--lora auto`): at one ciphertext `a1·a2` is folded into one C × C matrix (no
  nonlinearity between them); at S shards the fold costs 2S² blocks of PCMM planes per pair against 3S through the
  rank (e.g. 2.9B at t = 32: 18 vs 9), for one more PCMM output (+1 refresh) per pair.
- **The shard landing** (`shard_land`): the runtime's compile chains only a block's FIRST output into the next block's
  entry (`compile_dir`, `prev_exit = outputs.first()`), so with S > 1 every block refreshes its output shards at block
  end (S deliberate refreshes, like the carried state) and the next block compiles shards 1.. at the recorded landing
  (`input_metas`); `Rwkv7PlannedRun.check_shard_chain` compares every block's output metas with the next block's input
  metas before the session opens. `--no-shard-land` drops the refresh and relies on that check.

## 4. The per-size table (models, not measurements)

Inputs, each named (R5):
- **Refreshes**: the planner on 2-block (1-block for the widest) captures of synthetic checkpoints at each width in
  the default layout, `refreshes/token = block_0 + (layers − 1) · block_1 + tail`, times the synthetic-to-real factor
  (the real 0.1B with the final coefficients against the synthetic 768-wide one, same plans); batched: `block(K) =
  block(8) + slope · (K − 8)` (linear in K, as the 0.1B planner showed: BATCHED_DESIGN.md §2).
- **Seconds**: refreshes × the measured price per planned refresh on pod 3: decode 9.54 s / 478 = 19.96 ms
  (new_pod/RESULTS.md P1–P3; the 478-refresh plan of README.md); batched 46.68 s / 2,063 = 22.63 ms
  (r01b_batch/batch32.jsonl: `seconds_per_pass_median`, the header's compiled `n_bootstraps`). R6: another build than
  this code's (f8138f78 / 87fe1ab2 on speedrun 85971fbd), the old coefficients, the same card. The price per refresh
  includes everything else a token does (rotations, products, PCMM GEMMs) at the 0.1B's mix; the PCMM GEMM grows with
  the parameters (the planes), refreshes with the ciphertext count, so at 7.2B the price is an underestimate.
- **Memory** (`ckpt_info.memory_mb`): items transcribed from the pod-3 `mem_report_text` (r01b/run_P1, r04b/run_P1,
  r01b_batch/batch32; the model reproduces their totals within +1.3 %, +1.1 %, +0.3 %): register 14.0 MB, pool slot
  31.5 MB, StC-cache slot 205.2 MB, keys 8,437 MB, other 20.3 / 21.2 GB (decode / batched); PCMM planes exact (3 int8
  planes per 1,024² block: 0.1B 930.0 MB, 0.4B 2,082.0 MB, both as reported); plaintext store 162 MB per block and
  shard (0.1B; 0.4B measured 128). The card has 97,887 MiB; presets keep the model under 92 % of it. Batched pool
  slots `min(2,048, 32 K S)` is an assumption (0.1B K = 32 exhausted 192 and ran at 1,024).

**Default layout** (one ciphertext per vector, `auto_lanes`; `tools/rwkv7/size_table.py`, its two header lines:
- per-conversation slope factors (real / synthetic at 768): block 0 1.00 (2.0/2.0), block 1 2.00 (4.0/2.0)
- synthetic-to-real factor 1.080 (real 0.1B 2-block decode 38/42/1, synthetic 768 34/39/1); 1-block widths: block_1 = block_0 x 1.132)

| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block (b0, b1) | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass | tok/s aggregate | memory decode / batched GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1B | 12 x 768 | 32 | 1 | 2 | 34, 39 | 501 | 10.0 | 32 | 2090 | 47.3 | 0.677 | 69 / 89 |
| 0.4B | 24 x 1024 | 32 | 1 | 2 | 35, 40 | 1032 | 20.6 | 16 | 2656 | 60.1 | 0.266 | 76 / 70 |
| 1.5B | 24 x 2048 | 16 | 1 | 4 | 38, 42 | 1085 | 21.7 | 16 | 4333 | 98.0 | 0.163 | 83 / 87 |
| 2.9B | 32 x 2560 | 8 | 1 | 8 | 42, 46 (estimate, not planned) | 1586 | 31.7 | 1 | - | - | 0.0316 | 92 / 86 (over the card) |
| 7.2B | 32 x 4096 | 8 | 1 | 8 | 42, 46 (estimate, not planned) | 1586 | 31.7 | 1 | - | - | 0.0316 | 97 / 91 (over the card) |
| 13.3B | 61 x 4096 | 8 | 1 | 8 | 42, 46 (estimate, not planned) | 3027 | 60.4 | 1 | - | - | 0.0166 | 146 / 140 (over the card) |

**Channel shards at t = 32** (`--lanes 32`, the brief's option; same method):

| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block (b0, b1) | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass | tok/s aggregate | memory decode / batched GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1B | 12 x 768 | 32 | 1 | 2 | 34, 39 | 501 | 10.0 | 32 | 2090 | 47.3 | 0.677 | 69 / 89 |
| 0.4B | 24 x 1024 | 32 | 1 | 2 | 35, 40 | 1032 | 20.6 | 16 | 2656 | 60.1 | 0.266 | 76 / 70 |
| 1.5B | 24 x 2048 | 32 | 2 | 4 | 67, 73 | 1887 | 37.7 | 8 | 3511 | 79.5 | 0.101 | 94 / 80 (over the card) |
| 2.9B | 32 x 2560 | 32 | 3 | 6 | 92, 104 (b1 = b0 x ratio) | 3584 | 71.5 | 2 | 3991 | 90.3 | 0.022 | 98 / 86 (over the card) |

Columns: `cts / vector` = ciphertexts per width-C vector; `state cts` = per block and conversation; `K (memory)` = the largest K <= t the memory model fits under 92 % of the card (presets, `ckpt_info.preset`); refreshes per pass and seconds per pass at that K; `tok/s aggregate` = K / s per pass (K = 1: 1 / s per token). **Check against the measurements (old coefficients, R6 caveats above):** the 0.1B row predicts 10.0 s/token (measured 9.51-9.55 s with the 478-refresh plan; the final coefficients plan 497-501) and 47.3 s per K = 32 pass (measured 46.68 s); the 0.4B row predicts 20.6 s (measured 19.08-19.13 s).

The 2,560 / 4,096 rows at t = 8 are **estimates, not plans**: the 1-block t = 8 captures exceeded this Mac's memory guard (11.5 GB footprint in the PCMM fake's 4,096-channel encodings), and so did the sharded 4,096 capture (swap growth). The estimate takes the planned 2,048-wide t = 16 counts (38, 42: one ciphertext per vector) plus 4 deliberate refreshes for the 4 extra state ciphertexts per block (M = 8 at t = 8), and 8 refreshes per conversation and block (twice the t = 16 slope: twice the state); one capture on the box replaces them.

Reading it:
- **Decode time grows with blocks × ciphertexts per vector**, not parameters: 0.4B is 2× 0.1B (24 vs 12 blocks, one
  ciphertext each); 1.5B is again 24 blocks of one (2,048-channel) ciphertext, so it should cost about what 0.4B costs
  plus the larger PCMM GEMMs.
- **The state** is C / 512 ciphertexts per block and conversation in every layout: it sets the per-conversation slope.
- **Memory** stops the larger sizes, not time. 2.9B comes to 92 GB in the model even with the StC cache at 32 slots
  (0.1B used 45 of 160: expect misses, i.e. slower refreshes): over the presets' 92 % margin, under the card's 97.9 GB,
  so it is a try-and-see on the box; **7.2B and 13.3B do not fit one 98 GB card** in any layout (planes 34-44 / 64-85 GB,
  the plaintext store, the registers). Levers, unbuilt: (1) the token-shift fold doubles every first linear's planes
  (x and x_prev both enter PCMM): mixing in the ciphertext instead (6 plaintext products per vector, one level) halves
  them; (2) 16-bit weights (2 planes, −33 %), an accuracy question; (3) streaming the planes from host memory (the
  residency ring the runtime has for weights); (4) two cards (the fhe_ssm two-card route).

## 5. The per-size pipeline (`tools/rwkv7/size_pipeline.sh <pth> <tag> [steps]`)

Every step logs its command, rc and seconds to `$OUT/pipeline.log`; the first failure stops it.

| step | what | product |
|---|---|---|
| info | the checkpoint's shapes without loading it (`ckpt_info.py`), the layout and the memory preset of §4 | `info.json` |
| calib | `calibrate.py`, ctx 4096, 200,000 tokens of wiki.valid, batch 2 | `ranges_ctx4096.json` |
| fit | the final recipe: `fit.py --range minmax --margin 0.5 --target-err 1e-3 --metric rel --max-degree 255 --newton-target 1e-3 --site-target 'ffn\.relu2=1e-4' --rsqrt newton-const --rsqrt-range-factor 10`; ReLU² by `RELU2`: `m02` (gate 3: a margin-0.2 fit merged, `merge_relu2.py`) or `abs` (REVIEW §5.A1: one fit, ReLU² with an absolute target per block = `RELU2_REL` × max relu² over the margin-0.1 interval, `relu2_abs_targets.py`) | `coeffs_final.json` |
| gen | 32 prompts × 256 greedy tokens of the poly model, unclamped (`gen_greedy.py`): the model's own text, where the batched collapse came from | `gen_final.{jsonl,txt}` |
| scan | `scan_ranges.py`, unclamped, poly feed, every token of wiki.test, LAMBADA, the generations | `scanfull_final_*.json` |
| gate | `eval.py` exact, poly clamped, poly unclamped; `gate_verdict.py`: unclamped = clamped, all finite, scans clean (no non-finite, no input past a Newton site's convergence bound), the FINAL_PLAN bars (ppl ≤ +0.5 %, LAMBADA ≥ −0.5 points, top-1 ≥ 98 %); FAIL stops the pipeline | `verdict_final.json` |
| capture, plan | `run_box.sh capture / plan` with this checkpoint and coefficients; `plan_cost.py` | plan, `plan_cost_decode.json` |
| pf, b | the prefill (T = PF_T) and batched (K = B_K from the preset) captures and plans | plans |
| *(DRY=1 stops here)* | | |
| keys | `run_box.sh keyspec` + `keygen` for decode, pf, b (KEYS=bundle, the default on a full run) | `keys_<mode>/` |
| run, run_pf, batch | decode P1–P3 + 32 generated; prefill on P3 and the long prompts (`PF_TEXTS`); batched K, 32 generated (falls back to K / 2) | `run_*.jsonl` |
| records | `run_records.py`: per run the FINAL_PLAN records and the sampler window (`sampler.sh` runs throughout) | `records.json` |

`gate_verdict.py` on gate 3's own 0.1B records (gate/0.1b/eval_final.jsonl, the vector-d exact line, the two full
scans) prints PASS: ppl +0.051 %, LAMBADA −0.019 points, top-1 98.17 %, unclamped = clamped, scans 0 non-finite.

## 6. Verified on the CPU (this Mac: ring ≤ 2^15 slots for the mirror, the fake's 2^16 numpy vectors, 1–2 blocks)

`tests/test_anysize.py` (the 5–9 GB checks behind `--rwkv7-heavy`, one per process), `tests/test_keys.py`, and the
scripts named. Synthetic checkpoints: `synth.py` (x070 structure, random values, coefficients fitted on the exact
model's site ranges with the final fit's per-kind degrees).

| check | result |
|---|---|
| width ≤ 1,024 unchanged: the fake's call log and hidden values, old (`4bba3a93`) vs new, 2 blocks of the 0.1B: decode 3 tokens, a T = 4 prefill pass, K = 3 batched (lane carries, 2 passes; decode carries); the recorder's hidden values | identical (8,910 / 4,559 / 8,016 / 4,108 calls) |
| mirror vs the poly reference, unclamped, exact weights (stored float32), 5 tokens: 768, 1,024, 2,048 (shards; t = 16), 2,560 (3 shards; 2 shards at t = 16; t = 8), 4,096 (4 shards; t = 8) | max relErrRms 1.1e-7, 7.7e-8, 8.5e-8; 8.9e-8, 6.0e-8; 6.0e-8; 5.6e-8, 6.3e-8; 5.8e-8 |
| LoRA fold vs split (2,048, shards); 24-bit PCMM rounding (2,048, shards) | 5.1e-8; 1.3e-6 |
| fake vs mirror, 2,048 as 2 shards, 2 blocks, 4 tokens; the recorder, 1 block, 3 tokens | ≤ 9.9e-15, 272 eager refreshes every token; ≤ 6.9e-15, 144 |
| fake vs mirror, 2,048 at t = 16, 2 blocks, 3 tokens | ≤ 7.1e-15, 164 eager refreshes every token |
| prefill pass vs T decode steps (mirror, 2,048): T = 5, 32 (shards), T = 16 (t = 16) | hidden, carried state, the decode after: bit-identical |
| batched pass vs K separate decodes (mirror, 2,048): K = 3 lane / decode carries, K = 32 (shards), K = 16 (t = 16) | bit-identical; conversations differ (> 1e-3) |
| fake: prefill (T = 4) + 1 decode vs 5 decodes; K = 3 batched vs 3 fake decodes (2,048, shards) | 0.0, 0.0 |
| 1-block capture + plan at 2,048 (shards) through `make_plan.sh` | block 0: 65 refreshes (25 placed, 37 deliberate), 4,837 nodes, 380 rotations |
| the planner census (`plan_sizes.py`, 2-block decode and K = 4, 8 captures) | §3 and §4 |
| keys (`test_keys.py`, stand-ins for the device `_pc`) | flag → key source, the client role's files, a bundle run encrypting / decrypting only through the client, the key request's model part, the shard-chain check: 6 passed |
| `size_pipeline.sh` on the 0.1B, `DRY=1 SMOKE=1` (the harness on the CPU with tiny corpora, 2 blocks for the encrypted model) | every step rc 0: info, calib 5 s, fit 10 + 5 s, gen, scans, three evals, gate, capture, plan (77 refreshes), capture_pf + plan_pf (T = 32, 169 s), capture_b + plan_b (K = 4); stops before the GPU |
| `RELU2=abs` (one fit, absolute ReLU² targets) on the same smoke ranges | every ReLU² target met at margin 0.5 with degree 80–105 against the margin-0.2 fit's 54–64 (≈ 1.6–1.9×, as REVIEW §5.A1 predicted), and a lower absolute error than the m02 fit (block 0: 8.1e-4 vs 1.0e-3). Smoke ranges: not a quality result |
| `gate_verdict.py` on gate 3's records | PASS (§5) |

Not verified here: anything on a device (the t = 16 / 8 PCMM shapes, the shard chaining, the bundle session), the
real 1.5B / 2.9B weights and their quality, the 4,096-wide plans (memory), every second and gigabyte of §4.

## 7. Not built, by expected value

1. The memory levers of §4 (fold-free token shift first: it is a circuit change the planner can price on the CPU).
2. The filled layout (two reals per complex slot): 2× lanes at every width; at one ciphertext per vector it would give
   t = 32 at width 2,048 with the cost of t = 16. The biggest structural lever left (REVIEW §5.B3).
3. The 4,096-wide planner count on this Mac: the 1-block capture's recorder exceeded the memory guard (swap growth);
   §4's 4,096 rows extrapolate from 2,048 and 2,560 (marked). One capture on the box settles it.

## 8. The 1.5B on the box (pod 3; the commands, in order)

```bash
cd /root/perseus-rwkv7 && git fetch && git switch fhe-ssm-rwkv7-any        # once the coordinator has committed it
export PATH=/root/.local/bin:/root/.cargo/bin:$PATH                           # uv / cargo (the 0 s failures of 11-04)
bash tools/rwkv7/run_box.sh build
M=/root/d_models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth
export OUT=/root/out_any/1.5b RUNS=/root/vd/runs/1.5b HARNESS=/root/vd/harness TP=/root/venv_d/bin/python DATA=/root/d_data
PYTHONPATH=python .venv/bin/python -m perseus.examples.rwkv7.ckpt_info $M      # shapes, lanes 16, the preset
# 1. the CPU half and the quality gate (stops on a FAIL; RELU2=abs for REVIEW §5.A1's variant)
DRY=1 bash tools/rwkv7/size_pipeline.sh $M 1.5b
# 2. a 1-block smoke on the GPU with in-process keys (the t = 16 PCMM shape's first device run)
env PTH=$M COEFFS=$RUNS/coeffs_final.json LANES=16 BLOCKS=1 TOKENS=4 bash tools/rwkv7/run_box.sh all
# 3. the full runs with client keys (keyspec + keygen per mode, decode P1-P3, prefill, batched at the preset K = 16)
bash tools/rwkv7/size_pipeline.sh $M 1.5b keys,run,run_pf,batch,records
```

## 9. Likely device failure points, most likely first

1. **PCMM at t = 16 (2,048 channels) has never run on a device.** The Python side is `impl.pcmm`'s own shape rule and the
   fake multiplies correctly, but the key set (`pcmm_keys(make_keys(sk, mod, t))`), the GEMM and the finish run at a t
   the device never saw (the 0.1B / 0.4B and `fhe_ssm` used t = 32). The 1-block smoke (step 2) is the test; fallback
   `LANES=32` (channel shards, 1.7× the refreshes).
2. **Host RAM.** The 0.1B batched run peaked at 150 GB of 186 (new_pod/RESULTS.md). The twin, the prefill twin and the
   mirror each hold the folded matrices and the PCMM planes (1.5B: ≈ 8 GB of planes each, ≈ 18 GB of float64 layer
   weights in `ref.Params`); the batched twin records ≈ 16k-node graphs per block with values. Expect the 1.5B batched
   K = 16 run to exceed host RAM; run it last, at `B_K=8` first if the decode runs show > 100 GB.
3. **Device memory.** The model puts 1.5B decode at 83 GB and batched K = 16 at 87 GB (92 % margin = 90 GB). Its least
   certain items are the plaintext store (scaled from the 0.1B, 162 MB per block) and the batched pool (`32 K S` slots).
   Fallbacks: `STC_CACHE` / `STC_CACHE_B` down (each slot 205 MB), `B_K=8`.
4. **The key bundle.** `--keyspec` asks `Session.key_manifest` for the exact programs; the bundle is checked against the
   plan contract and the layout's rotation count before the session opens (`check_bundle`), and the run refuses a
   session that holds a secret. Untested on a device in this port (the path is fhe_ssm's, which ran on 11-02). A
   mismatch shows at session open, before any timing.
5. **Planner and setup time.** 24 blocks of ≈ 3,300 nodes (decode) are the 0.4B's scale; the batched plan at K = 16 is
   ≈ 24 × 8,000 nodes. Expect tens of minutes of planning on the box.
6. **Quality at 1.5B** is unmeasured: the gate (step 1) decides, and stops the pipeline on a FAIL. The 0.4B gate-3
   result is still pending in REVIEW_20261105.md.
7. **Channel shards (only with `LANES=32`, or widths above 4,096):** the z-shard chain. Each block refreshes its output
   shards (`shard_land`) and the next block compiles them at the recorded landing; `check_shard_chain` stops a mismatch
   before the session opens, naming the block and the shard.
