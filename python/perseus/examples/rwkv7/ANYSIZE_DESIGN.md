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
- **State side**: a conversation's state is C × 64 values (H = C / 64 heads of 64 × 64). In the current layout a
  block holds it in ⌈C / D⌉ · 64 / t ciphertexts (S shards of D channels, M = 64 / t ciphertexts per shard), which is
  C / 512 only when C is a multiple of D: 2 per 1,024 channels at t = 32 and 4 for 2,048 channels at t = 16 (the same
  4), but 2 (not 1.5) for the 0.1B's 768 channels and 8 (not 5) for 2,560 at t = 8 (one padded 4,096-channel shard).
  The filled layout (FILLED_DESIGN.md) holds two value groups per ciphertext, one on each axis: half of that
  (⌈C / D⌉ · 32 / t; 1 at the 0.1B, 2 at 2,048). Between the current-layout options at the same padding, the state
  update, its readout and the per-conversation lane moves therefore cost about the same.

## 3. The options, judged by encrypted cost (the planner's refreshes)

| option | ciphertexts per vector | state cts per block | lanes (K, T max) | built |
|---|---|---|---|---|
| **channel shards** (`--lanes 32`): D = 1,024, S = ⌈C / 1,024⌉ shards | S | 2 S | 32 | yes |
| **fewer lanes** (`--lanes 16 / 8`, **the default**: `layout.auto_lanes`, one ciphertext per vector) | 1 (up to 4,096) | 64 / t | t = 16 (C ≤ 2,048), 8 (C ≤ 4,096) | yes |
| filled layout (`--layout filled`: two value groups of the state per ciphertext, one on each axis; FILLED_DESIGN.md) | as either | half of either | as either | yes (B3; combines with both) |

The brief recommended channel shards ("confirm by cost"). **The planner does not confirm it.** 2-block captures of the
same synthetic width-2,048 checkpoint, the same coefficients, LANDING=high, refreshes per stage (`plan_cost.py`; the
census JSON lines under `.cache/rwkv7_any/`):

| width 2,048 | block 0 | block 1 | per conversation and block (K = 4 → 8) | nodes (block 1) |
|---|---|---|---|---|
| 2 shards at t = 32 | 67 | 73 | 4.0 (block 1: 87 → 103) | 5,573 |
| 1 ciphertext at t = 16 | **38** | **42** | 4.0 (block 1: 55 → 71) | 3,334 |
| (the real 0.1B, final coefficients, t = 32, base plan) | 38 | 42 | 4.0 in block 1 (54 → 70), 2.0 in block 0: the min-cut prune artefact `PLANOPT=b2` removes (2.0 under b2, PLAN_B2.md); the synthetic 768 is 2.0 in both | 3,458 |

**Mechanism.** The token side is ~30 of block 1's 42 refreshes (the norms' Newton chains, the PCMM outputs, the sites'
depth). Two shards double it (each shard is its own ciphertext through every site, every PCMM output and every carry);
one 2,048-channel ciphertext does not. The per-conversation cost is the same 4 refreshes per block in both, as §2
predicts: 4 state ciphertexts per conversation and block at width 2,048 in both layouts, one carry refresh each (the
0.1B's 2.0 at t = 32 is its 2 state ciphertexts). So at K = 1 fewer lanes is **1.7× fewer refreshes per token** (42 vs 73 in block 1), and in a batched pass
it saves the same token-side refreshes at every K. Shards only win where a run needs K or T above t (16 at 2,048), and
§4 shows memory caps K at or below that anyway (1.5B: K = 8 at t = 16, set by host RAM; as shards the models fit
only K = 4). Prefill
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

Inputs, each named (R5). Re-derived 2026-11-05 after review lens 3 (findings 1-5, 14): the first version priced
planner counts at a price per compiled refresh, scaled the batched slope ×2 with the b2 prune artefact, and checked
against old-coefficient runs; its batched column was 35-58 % high and its "47.3 s vs 46.68 s measured" agreement was
two errors cancelling.
- **Refreshes** (all PLANNER counts, `summary.total_bootstraps`, never the device's compiled `n_bootstraps`, which is
  smaller: the final batched plan is 1,424 planned, 1,257 compiled): the planner on 2-block (1-block for the widest)
  captures of synthetic checkpoints at each width, `refreshes/token = f · (block_0 + (layers − 1) · block_1 + tail)`,
  f = the real 0.1B with the final coefficients over the synthetic 768-wide one (same plans) = 1.0797; batched:
  `refreshes/pass = refreshes/token + K · (slope_0 + (layers − 1) · slope_1)`, the slopes per conversation and block
  from the synthetic K = 4 → 8 plans, **unscaled** (not × f, not × the real/synthetic slope ratio): the real 0.1B's 4
  in block 1 was the prune artefact, and the synthetic slopes (2 at t = 32, 4 at t = 16) are b2-like. The pass column
  is therefore a `PLANOPT=b2` plan's. The 2-block extrapolation `b0 + (L − 1) · b1` is confirmed for decode only (the checks
  below, within 1–2.5 %). For the other modes the device's 12 blocks differ from it in both directions (review lens 3,
  finding 3, in compiled sites: the base batched pass 1,760 extrapolated vs 1,257 on the device, its blocks 2–11
  cheaper than block 1; prefill 867 vs 1,059 `pf_block_*` sites), so the pass column carries that uncertainty too.
- **Seconds**: refreshes × the price per PLANNED refresh of pod 3's final-coefficient runs with `LEVERS=best`: decode
  9.703 / 9.7283 / 9.7405 s per token over 497 planned refreshes (`r01b_final/run_best_P1-P3.jsonl`,
  `seconds_per_token_median`; `r01b_final/plan_best.log`) and 19.7367 / 19.7413 / 19.7286 s over 1,011
  (`r04b_final/...`) = **19.51–19.60 ms**; batched 29.8721 s per pass over 1,424 planned refreshes
  (`r01b_final/batch32_best.jsonl`, `seconds_per_pass_median`; `r01b_final/plan_b32_best.log`) = **20.98 ms**.
  `size_table.py --anchors new_pod/` re-derives all of them from the records. That batched anchor is a base plan (33
  paired refreshes per block); b2 removes refreshes about half of which were paired, so its price per planned refresh
  is higher: the s/pass column is therefore a range, 20.98 ms (low) to 22.42 ms (high: the price at which the 0.1B
  b2 row equals the per-op model of review lens 3, 28.45 s), i.e. 26.6–28.5 s at the 0.1B. R6: another build than this code's (f8138f78 / 87fe1ab2 on speedrun 85971fbd), the
  same card. The price includes everything else a token does (rotations, products, PCMM GEMMs) at the 0.1B / 0.4B's
  mix; the PCMM GEMM grows with the parameters, refreshes with the ciphertext count, so from 1.5B up the price is an
  underestimate.
- **Memory** (`ckpt_info.memory_mb`): items transcribed from the pod-3 `mem_report_text` of the final-coefficient runs
  (r01b_final/run_best_P1 70,980.0 MB, r04b_final/run_best_P1 77,844.9, r01b_final/batch32_best 91,019.6; the model
  reproduces them to +0.0 %, +0.8 %, +0.0 %; calibrated on the older reports it under-predicted all three by ≈ 2 %,
  review lens 3, finding 5): register 14.0 MB, pool slot 31.5 MB, StC-cache slot 205.2 MB, keys 8,437.2 MB, other
  21.43 / 22.53 GB (decode / batched); PCMM planes exact (3 int8 planes per 1,024² block: 0.1B 930.0 MB, 0.4B 2,082.0
  MB, both as reported); plaintext store 205.3 MB per block and shard (the 0.1B final decode's; the 0.4B's 148.75 is
  smaller, so this is an upper value). **Pod 3's 0.4B batched K = 16 refusal** (`new_pod/RESULTS.md`: "memory: ct_pool
  = 33822867456: plan needs 103450379264 B, device has 101389893632 B free" at `POOL_B=1024`, `STC_CACHE_B=64`, i.e.
  98,658.0 MiB needed of 96,692.9 MiB free, 2.0 GB short) is the fourth anchor: the 12-block calibration said 95,139
  MiB there (−3.6 %), so the batched "other" grows by 293.2 MiB per block beyond 12 (a one-point fit; the refusal
  names no item) and the model reproduces the refusal. The ceiling for the presets is **91,020 MiB planned**, the
  largest planned total measured to run (`r01b_final/batch32_best`, VRAM peak 94,858 of 97,887 MiB): the runtime
  compares its plan with the free memory at session open (96,693 MiB on pod 3), and what the plan does not count took
  the other ≈ 3.8 GB. Batched: the largest K that fits with a 64-slot StC cache, else 32; the pool starts at `32 K S`
  slots (an assumption: 0.1B K = 32 exhausted 192 and ran at 1,024) and grows in steps of 32 up to 1,024 S while the
  model fits: **0.4B K = 16 gets `POOL_B=768`, `STC_CACHE_B=64`, 90.6 GB** (the refused 1,024 would be 98.7).
- **Host RAM** (`ckpt_info.host_gb`), the second budget. Session setup (the twins' probe recording and
  `Program.compile`) runs on the host before the device is touched; on pod 3 (181 GB for the container) it killed the
  0.4B prefill session (two program sets of 24 blocks, T = 32; host RAM 10.8 → 184.9 GB in 2.7 min, the GPU idle)
  and the 0.4B batched K = 32 session (≈ 185.6 GB), while the 0.1B prefill (12 blocks) fitted and the 0.1B batched
  K = 32 session peaked at 150.5 GB. The model is fitted on the sampler's peaks (`new_pod/monitor/usage_20261104.csv`
  and the windows in `new_pod/RESULTS.md`): decode 6.07 GB + 2.42 GB per block (0.1B 34.7–35.1 GB, 0.4B 64.1–64.2);
  batched 6.07 + (0.868 + 0.349 K) GB per block, floored at decode's 2.42 (0.1B K = 32 150.5, 0.4B K = 16 160.9; the
  K = 32 kill is 295 GB in the model); prefill holds the decode programs plus the pass's, 6.07 + (2.42 + p32 · T / 32)
  GB per block with p32 only **bracketed** to 4.87–12.15 GB by the 0.1B fit and the 0.4B kill, so **0.4B at T = 16
  is 123–210 GB in the model: unsettled** (its top end exceeds 181; the first T = 16 run records it; T = 8 is 93–137
  GB). Per block the cost follows the graph's node count, width-independent at one ciphertext per vector (3,334
  nodes at 2,048 against 3,458 at the 0.1B, §3), so it scales with the ciphertexts per vector and the batched slope
  with the state ciphertexts per conversation; widths 768 and 1,024 are confounded in the two batched anchors. The
  presets keep it under 166.5 GB (92 % of 181), and start 0.4B and above at **T = 16 and K ≤ 16**
  (`size_pipeline.sh`; `PF_T` / `B_K` in the environment override them). This is what moved 1.5B batched from K = 16
  to K = 8: K = 16 is 295 GB of host RAM in the model.

**Default layout** (one ciphertext per vector, `auto_lanes`; `tools/rwkv7/size_table.py --census
.cache/rwkv7_any/synth_{768_b2,1024_b2,2048_b2_t16} --real .cache/rwkv7_any/real_0.1b_b2 --estimate 2560:8:42:46:8
4096:8:42:46:8 --anchors <new_pod>`, regenerated 2026-11-05 with the host-RAM model and the pod-3 device refusal;
its header lines, table and checks verbatim):
- synthetic-to-real factor 1.0797 (real 0.1B 2-block decode 38/42/1, synthetic 768 34/39/1); 1-block widths: block_1 = block_0 x 1.132
- per-conversation slopes: the synthetic plans' (K = 4 -> 8), unscaled and added to the decode count (not multiplied by the factor); the pass column is a b2 plan's
- memory: device MiB / 1,000 under the presets' ceiling 91.02 (the largest planned total measured to run); host RAM of session setup in GB under 166.5 (92 % of pod 3's 181 GB); prefill: the decode + T-token programs, a range (only bracketed by pod 3's 0.1B fit and 0.4B OOM kill at T = 32)
- prices per PLANNED refresh (pod 3, final coefficients, LEVERS=best): decode 19.51-19.60 ms; batched 20.98-22.42 ms (low end: the measured base pass; high end: the 0.1B b2 pass at the per-op model's 28.45 s)

| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block (b0, b1) | slope / conv. / block | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass | tok/s aggregate | device memory decode / batched GB | host RAM decode / batched / prefill (T) GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1B | 12 x 768 | 32 | 1 | 2 | 34, 39 | 2 | 501 | 9.78-9.82 | 32 | 1269 | 26.6-28.5 | 1.125-1.202 | 71 / 91 | 35 / 150 / 94-181 (T = 32) |
| 0.4B | 24 x 1024 | 32 | 1 | 2 | 35, 40 | 2 | 1032 | 20.14-20.23 | 16 | 1800 | 37.8-40.4 | 0.396-0.424 | 78 / 91 | 64 / 161 / 123-210 (T = 16) |
| 1.5B | 24 x 2048 | 16 | 1 | 4 | 38, 42 | 4 | 1085 | 21.18-21.27 | 8 | 1853 | 38.9-41.5 | 0.193-0.206 | 85 / 91 | 64 / 161 / 123-210 (T = 16) |
| 2.9B | 32 x 2560 | 8 | 1 | 8 | 42, 46 (estimate, not planned) | 8 | 1586 | 30.95-31.09 | 1 | - | - | 0.0322-0.0323 | 95 / - (over the ceiling) | 84 / - / 123-181 (T = 8) |
| 7.2B | 32 x 4096 | 8 | 1 | 8 | 42, 46 (estimate, not planned) | 8 | 1586 | 30.95-31.09 | 1 | - | - | 0.0322-0.0323 | 99 / - (over the ceiling) | 84 / - / 123-181 (T = 8) |
| 13.3B | 61 x 4096 | 8 | 1 | 8 | 42, 46 (estimate, not planned) | 8 | 3027 | 59.06-59.32 | 1 | - | - | 0.0169-0.0169 | 150 / - (over the ceiling) | 154 / - / 228-339 (T = 8) |

**Channel shards at t = 32** (`--lanes 32`, the brief's option; same method, `--census
.cache/rwkv7_any/synth_{768_b2,1024_b2,2048_b2,2560_b1}`):

| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block (b0, b1) | slope / conv. / block | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass | tok/s aggregate | device memory decode / batched GB | host RAM decode / batched / prefill (T) GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1B | 12 x 768 | 32 | 1 | 2 | 34, 39 | 2 | 501 | 9.78-9.82 | 32 | 1269 | 26.6-28.5 | 1.125-1.202 | 71 / 91 | 35 / 150 / 94-181 (T = 32) |
| 0.4B | 24 x 1024 | 32 | 1 | 2 | 35, 40 | 2 | 1032 | 20.14-20.23 | 16 | 1800 | 37.8-40.4 | 0.396-0.424 | 78 / 91 | 64 / 161 / 123-210 (T = 16) |
| 1.5B | 24 x 2048 | 32 | 2 | 4 | 67, 73 | 4 | 1887 | 36.83-36.99 | 4 | 2271 | 47.6-50.9 | 0.079-0.084 | 94 / 90 (over the ceiling) | 122 / 122 / 239-414 (T = 16) |
| 2.9B | 32 x 2560 | 32 | 3 | 6 | 92, 104 (b1 = b0 x ratio) | 5.25 | 3584 | 69.93-70.23 | 1 | - | - | 0.0142-0.0143 | 103 / - (over the ceiling) | 239 / - / 472-822 (T = 16) |

Columns: `cts / vector` = ciphertexts per width-C vector; `state cts` = per block and conversation (current layout;
half in the filled one); `slope` = planned refreshes per conversation and block; `K (memory)` = the preset's K: the
largest K <= t (<= 16 from 0.4B up) that fits BOTH the device model (at most 91.02 GB planned) and the host model (at
most 166.5 GB); refreshes and seconds per pass at that K, the seconds a range (low: the measured base pass's price,
20.98 ms; high: 22.42 ms, the price at which the 0.1B row equals the per-op model's b2 pass, 28.45 s); `tok/s
aggregate` = K / s per pass (K = 1: 1 / s per token); device memory in GB (MiB / 1,000) at the preset's pools and
caches; host RAM of session setup in GB, prefill at the preset's T as a range.

**Checked against the measured final-coefficient runs (same build family, R6), stated tolerance ±3 % on decode;
memory against the records it is fitted on:**
- check 0.1B decode: 501 planned refreshes vs the final plan's 497 (+0.8 %); 9.78-9.82 s vs measured 9.70-9.74 s (+0.4 to +1.2 %): within the stated +/-3 %
- check 0.4B decode: 1032 planned refreshes vs the final plan's 1011 (+2.1 %); 20.14-20.23 s vs measured 19.73-19.74 s (+2.0 to +2.5 %): within the stated +/-3 %
- check 0.1B batched (K = 32): 1269 planned refreshes per pass for a b2 plan; no measured b2 pass exists. The real 0.1B's K = 32 b2 2-block plan (planner 100 / 103 for blocks 0 / 1, PLAN_B2.md) extrapolates to 100 + 11 x 103 + 1 = 1234 (+2.8 %); the measured BASE pass is 1424 planned / 29.87 s (the low price itself), and the per-op model puts b2 at 28.28-28.45 s (the high price); this row: 26.6-28.5 s
- check 0.1B memory: decode 70,980.0 MiB planned (r01b_final/run_best_P1), batched K = 32 91,019.6 MiB (batch32_best, VRAM peak 94,858 MiB): the model 71.0 / 91.0; host decode 34.7-35.1 GB, batched K = 32 150.5 GB (sampler peaks); prefill T = 32 fitted (no peak on record): the model 35.1 / 150.5 / 93.6-180.9
- check 0.4B memory: decode 77,844.9 MiB planned (r04b_final/run_best_P1); batched K = 16 at POOL_B 1,024 / STC_CACHE_B 64 REFUSED needing 98,658.0 MiB of 96,692.9 free: the model 98.7 there (fitted), 90.6 at the preset's POOL_B 768; host decode 64.1-64.2 GB, batched K = 16 160.9 GB; K = 32 and prefill T = 32 OOM-killed at ~185 GB of 181: the model 64.2 / 160.9 / 294.9 (K = 32) / 181.1-355.8 (T = 32)

The 2,560 / 4,096 rows at t = 8 are **estimates, not plans**: the 1-block t = 8 captures exceeded this Mac's memory guard (11.5 GB footprint in the PCMM fake's 4,096-channel encodings), and so did the sharded 4,096 capture (swap growth). The estimate takes the planned 2,048-wide t = 16 counts (38, 42: one ciphertext per vector) plus 4 deliberate refreshes for the 4 extra state ciphertexts per block (M = 8 at t = 8), and 8 refreshes per conversation and block (twice the t = 16 slope: twice the state); one capture on the box replaces them.

Reading it:
- **Decode time grows with blocks × ciphertexts per vector**, not parameters: 0.4B is 2× 0.1B (24 vs 12 blocks, one
  ciphertext each); 1.5B is again 24 blocks of one (2,048-channel) ciphertext, so it should cost about what 0.4B costs
  plus the larger PCMM GEMMs.
- **The state** is ⌈C / D⌉ · 64 / t ciphertexts per block and conversation (half in the filled layout): it sets the
  per-conversation slope (one carry refresh per state ciphertext in a b2 plan).
- **Memory** stops the larger sizes, not time, on two budgets. Host RAM caps the batched K (0.4B K = 16, 1.5B
  K = 8) and the prefill T (16 from 0.4B up; 0.4B T = 16 unsettled, above). On the device 2.9B comes to 95 GB in the
  model even with the StC cache at 32 slots (0.1B used 45 of 160: expect misses, i.e. slower refreshes): over the
  presets' 91.0 GB ceiling, 1.7 GB under the 96.7 GB free at session open on pod 3, so it is a try-and-see on the box
  at best; **7.2B and 13.3B do not fit one 98 GB card** in any layout (planes 34-44 / 64-85 GB,
  the plaintext store, the registers). Levers, unbuilt: (1) the token-shift fold doubles every first linear's planes
  (x and x_prev both enter PCMM): mixing in the ciphertext instead (6 plaintext products per vector, one level) halves
  them; (2) 16-bit weights (2 planes, −33 %), an accuracy question; (3) streaming the planes from host memory (the
  residency ring the runtime has for weights); (4) two cards (the fhe_ssm two-card route).

## 5. The per-size pipeline (`tools/rwkv7/size_pipeline.sh <pth> <tag> [steps]`)

Every step logs its command, rc and seconds to `$OUT/pipeline.log`; the first failure stops it. Each GPU step is named
after its record (the record's basename), so `run_records.py` attaches the sampler window to every run. The harness
steps write into `RUNS` (default `/root/vd/runs/<tag>_pipe`, never a gate's published directory) and refuse to
overwrite an existing product (ranges, coefficients, generations, scans, evals) unless `FORCE=1`, which moves the old
file to `<name>.prev-<UTC time>` first. Every capture and plan `run_box.sh` reuses, and every run step, is checked
against the current settings first (`tools/rwkv7/capture_check.py`: the pth and coefficient sha256, lanes, layout,
blocks, K, T, LoRA mode, shard landing, and the plan's graph md5); the names carry every setting that changes them
(`run_box.sh names`).

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
| keys | `run_box.sh keyspec` + `keygen` for decode, pf, b (KEYS=bundle, the default on a full run; b only when B_K ≥ 2) | `run_box.sh names`' `KEYDIR_<mode>` |
| run, run_pf, batch | decode P1–P3 + 32 generated; prefill (T = PF_T, 16 from 0.4B up) on P3 and the long prompts (`PF_TEXTS`); batched K (≤ 16 from 0.4B up), 32 generated. Only on a memory refusal (`RuntimeError: memory:`, a CUDA out of memory, or a SIGKILL, which `run_box.sh` notes as `Killed (rc 137 ...)` in the run's log: pod 3's host OOM kills) the batched run falls back to K / 2, with that K's own capture, plan and (KEYS=bundle) key bundle; any other failure stops. Skipped when B_K < 2 | `run*.jsonl`, `batch*.jsonl`, `serve*.jsonl` |
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
| `size_pipeline.sh` on the 0.1B, `DRY=1 SMOKE=1 BLOCKS=2 B_K=4 GEN_N=8 GEN_P=2` (the harness on the CPU with tiny corpora, 2 blocks for the encrypted model), 2026-11-05 after the fix round | PIPELINE rc=0, every step rc 0: info, calib 8 s, fit 15 + 9 s, gen 4 s, scans 17 / 17 / 2 s, three evals 10 / 27 / 26 s, gate (FAIL on top-1 under SMOKE: smoke corpora, printed and passed over), capture 15 s, plan 8 s, capture_pf 45 s + plan_pf 288 s (T = 32), capture_b 13 s + plan_b 21 s (K = 4); stops before the GPU. A second invocation: the `fit` step stops on the existing `coeffs_final.json` (FORCE=1 to overwrite); `capture,b` reuse every capture and plan after capture_check reports them consistent (9-10 fields + plan md5) |
| `RELU2=abs` (one fit, absolute ReLU² targets) on the same smoke ranges | every ReLU² target met at margin 0.5 with degree 80–105 against the margin-0.2 fit's 54–64 (≈ 1.6–1.9×, as REVIEW §5.A1 predicted), and a lower absolute error than the m02 fit (block 0: 8.1e-4 vs 1.0e-3). Smoke ranges: not a quality result |
| `gate_verdict.py` on gate 3's records | PASS (§5) |

Not verified here: anything on a device (the t = 16 / 8 PCMM shapes, the shard chaining, the bundle session), the
real 1.5B / 2.9B weights and their quality, the 4,096-wide plans (memory), every second and gigabyte of §4.

## 7. Not built, by expected value

1. The memory levers of §4 (fold-free token shift first: it is a circuit change the planner can price on the CPU).
2. ~~The filled layout~~: built since (B3, `--layout filled`, FILLED_DESIGN.md): half the state ciphertexts per
   block at every width (it does not double the lanes: the token side stays one value per slot).
3. The 4,096-wide planner count on this Mac: the 1-block capture's recorder exceeded the memory guard (swap growth);
   §4's 4,096 rows extrapolate from 2,048 and 2,560 (marked). One capture on the box settles it.

## 8. The 1.5B on the box (pod 3; the commands, in order)

```bash
cd /root/perseus-rwkv7 && git fetch && git switch fhe-ssm-rwkv7-any        # once the branch tip is committed
export PATH=/root/.local/bin:/root/.cargo/bin:$PATH                           # uv / cargo (the 0 s failures of 11-04)
bash tools/rwkv7/run_box.sh build
M=/root/d_models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth
export OUT=/root/out_any/1.5b RUNS=/root/vd/runs/1.5b HARNESS=/root/vd/harness TP=/root/venv_d/bin/python DATA=/root/d_data
PYTHONPATH=python .venv/bin/python -m perseus.examples.rwkv7.ckpt_info $M      # shapes, lanes 16, the preset
# 1. the CPU half and the quality gate (stops on a FAIL; RELU2=abs for REVIEW §5.A1's variant)
DRY=1 bash tools/rwkv7/size_pipeline.sh $M 1.5b
# 2. a 1-block smoke on the GPU with in-process keys (the t = 16 PCMM shape's first device run). `check` runs the CPU
#    suite on the 0.1B (CHECK_PTH / CHECK_COEFFS: the suite is written against it) and only the width-agnostic checks
#    on $M; KEYS is required for every device step
env PTH=$M COEFFS=$RUNS/coeffs_final.json LANES=16 BLOCKS=1 TOKENS=4 KEYS=inproc \
    bash tools/rwkv7/run_box.sh check capture plan run
# 3. the full runs with client keys (keyspec + keygen per mode, decode P1-P3, prefill at the preset T = 16, batched at
#    the preset K = 8, POOL_B 576: host RAM caps K, §4; record the host peak of each session's setup from sampler.csv)
bash tools/rwkv7/size_pipeline.sh $M 1.5b keys,run,run_pf,batch,records
```

## 9. Likely device failure points, most likely first

1. **PCMM at t = 16 (2,048 channels) has never run on a device.** The Python side is `impl.pcmm`'s own shape rule and the
   fake multiplies correctly, but the key set (`pcmm_keys(make_keys(sk, mod, t))`), the GEMM and the finish run at a t
   the device never saw (the 0.1B / 0.4B and `fhe_ssm` used t = 32). The 1-block smoke (step 2) is the test; fallback
   `LANES=32` (channel shards, 1.7× the refreshes).
2. **Host RAM** (measured, pod 3): the 0.4B prefill at T = 32 and batched at K = 32 were killed by the host's OOM
   killer at setup (≈ 185 GB of 181, the GPU idle); 0.4B K = 16 peaked at 160.9 GB, 0.1B K = 32 at 150.5. The §4 model
   puts 1.5B decode at 64 GB, batched K = 8 at 161 GB and prefill T = 16 at 123–210 GB (unsettled). The twin, the
   prefill twin and the mirror each hold the folded matrices and the PCMM planes (1.5B: ≈ 8 GB of planes each, ≈ 18 GB
   of float64 layer weights in `ref.Params`), which the per-block fit (made at widths 768 and 1,024) may not carry to
   2,048. Run the batched and prefill sessions last; if decode setup peaks well above the model's 64 GB, start at
   `B_K=4` / `PF_T=8`. A SIGKILL is noted in the run's log and the batched step halves K.
3. **Device memory.** The model (recalibrated on the final runs and on the 0.4B K = 16 refusal) puts 1.5B decode
   at 85 GB and batched K = 8 at 90.8 GB (`POOL_B` 576, `STC_CACHE_B` 64; ceiling 91.0 GB planned, 96.7 GB free at
   session open on pod 3). Its least certain items are the plaintext store (205.3 MB per block, the 0.1B's: the
   0.4B's was 149), the per-block batched "other" (one point) and the batched pool (`32 K S` slots at the floor).
   A refusal names the shortfall ("plan needs X B, device has Y B free"): lower `POOL_B` (31.5 MiB a slot) or
   `STC_CACHE_B` (205.2 MiB a slot) by that much.
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
