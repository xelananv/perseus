# RWKV-7 batched decode under CKKS: design, 2026-11-04

**Goal.** Decode K ≤ 32 independent conversations per encrypted pass, one conversation per lane. The token side is
shared; each conversation keeps its own carried state. The reference point is the measured single-conversation decode:
**9.51–9.55 s per token** on the RTX PRO 6000 Workstation (600 W) (`results/rwkv7-fhe-20261104/new_pod/RESULTS.md`,
P1–P3 summary lines). This file calls it 9.54 s.

**Status.**
- Built: `batched.py`, plus the hooks in `model.py` (`batch=`, `set_batch`, `state_names`), `run_capture.py --batch`,
  `run_planned.py --batch K --prompts FILE` and the `run_box.sh` steps `capture_b | plan_b | bcheck | batch | b`.
- Validated on the CPU twin: `tests/test_batched.py` (§3).
- Planned on the CPU: K = 4, 8, 16, 32 (1 block) and K = 32 (2 blocks).
- **Nothing has run on a GPU.** Every second in this file is an estimate (§4).

## 1. Layout

The filled layout has channel c of lane l at slot 32c + l (`layout.py`). A decode step copies one token over the
t = 32 lanes. A batched pass puts **conversation c's current token on lane c** instead. Lanes past K repeat
conversation K − 1's row and are never read back.

| part | what it is per conversation | how it is computed |
|---|---|---|
| token side: PCMM linears, every Chebyshev / Newton site, LayerNorm, GroupNorm, `kk`, the bonus, `z = h + i·v_first` | lane c | decode's code unchanged: PCMM applies one matrix per lane; the reductions sum over channels within a lane (PREFILL_DESIGN.md §1) |
| WKV state S (12 heads × 64 × 64) | **its own** M = 2 ciphertexts per block, in decode's layout (value index on the lanes) | decode's own `state_step`, once per conversation (below) |
| token-shift carries `xa`, `xf` | two layouts, §1.2 | |

### 1.1 The state update, per conversation (P3's scan machinery, without the chain)

The state update needs six j- or i-indexed vectors in natural form: w, κ̂, κ̂⊙a, k̃, r, v. Here they sit on lane c,
so each conversation does the following:
1. **Bring lane c to lane 0.** The three complex sources `w + i·κ̂`, `(κ̂⊙a) + i·k̃` and `r + i·v` are rotated by +1
   once per conversation (a running rotation: one key, 3 rotations per conversation).
2. **Broadcast.** A lane-0 mask carrying the factor ½ (one level), then the lane copy: `rotate_and_sum(−1, 32)`, a
   doubling, so log₂ 32 = 5 rotations, not 31.
3. **Split each pair with one conjugation.** `x = B + conj B`, `y = i·(conj B − B)`. This is exact because a real
   vector is its own conjugate.
4. **Update and read out.** `transpose(v)`, then decode's `state_step` on **conversation c's own** S_0, S_1, then
   `untranspose_lane0`. The readout lands on lane 0.
5. **Put the readout back on lane c.** Horner: `Q ← rot(Q, +1) + acc_c`, then one rotation by −(K − 1) at the end.

Unlike prefill, conversation c's update reads only its own state. **The K updates are independent**: no
conversation's S chain passes through another's, so the S depth per pass is one decode step's (3 levels), whatever K
is. Prefill's chain grows by T × 3 levels.

### 1.2 The token-shift carries: two layouts, both built

The token shift is RWKV's "previous token's LayerNorm output" input.
- **`carry=decode`** (the brief's layout). Conversation c keeps `xa_c`, `xf_c` in natural form, as decode does.
  - Partner: `x_prev = Σ_c xa_c ⊙ mask(lane c)`. That is K plaintext products, one level.
  - New carry: lane c of `x`, broadcast. That is a rotation, a mask and the 5-rotation copy.
  - Cost per block: 12K rotations and **2K carry refreshes**.
- **`carry=lane`** (a lever, and the default). ONE ciphertext per carry holds conversation c's carry on lane c. That
  ciphertext **is x itself**, so the token shift is decode's own code with no ops: `x_prev = carry`, `carry ← x`.
  - Lanes past K stay equal to lane K − 1, by induction from the public prefix state, which is equal on every lane.
  - Saves 12K rotations, K products and **2(K − 1) refreshes per block** (measured below: 284 → 222 at K = 32).
  - The cost: a lane-form carry is not decode's form, so handing a conversation over to single-conversation decode
    needs one lane broadcast (`BatchedMirror.conv_state` does it on the mirror). Nothing in the runner needs that.

K = 1 is decode itself: `set_batch(1)` keeps the decode circuit, its state and its op stream (tested, §3).

### 1.3 Positions are per lane

RWKV has no position input: a position is only "how many updates this lane's state has seen". Every lane advances one
token per pass:
- conversation c reads its prompt token by token, then free-runs;
- the client decrypts all K top-1 picks and feeds c's back to lane c;
- passes with prompt tokens on some lanes and generated tokens on others need no special case (mixed passes, §5.2).

The runner runs every lane to the longest schedule (max prompt + `--generate`). A conversation's records stop at its
own end; its lane keeps running on its own greedy picks, which keeps its values in range.

## 2. What it costs: the planner's counts

**Source.** `tests/test_batched.py::test_batched_capture_and_plan`:
- `--rwkv7-plan --rwkv7-b-k K [--rwkv7-b-blocks 2] [--rwkv7-b-carry decode] --rwkv7-b-dir .cache/rwkv7_b3`;
- plans by `tools/fhe_ssm/make_plan.sh` with LANDING=high, PAIR=0;
- graph op counts from `block_<b>/graph.json`;
- the per-scope census from `plan_census` (also `tools/rwkv7/plan_census.py`).

The decode rows are P3's (PREFILL_DESIGN.md §3).

| capture | block | nodes | rotate | conj. | mult | adds (add + inplace + sub) | refreshes (plan) |
|---|---|---|---|---|---|---|---|
| decode, 1 token | 0 | 2,232 | 209 | 2 | 788 | – | 31 |
| batched K = 4, lane | 0 | 3,634 | 531 | 14 | 1,221 | 1,669 | 55 |
| batched K = 8, lane | 0 | 5,446 | 939 | 26 | 1,793 | 2,465 | 79 |
| batched K = 16, lane | 0 | 9,070 | 1,755 | 50 | 2,937 | 4,057 | 127 |
| batched K = 32, lane | 0 | 16,307 | 3,382 | 98 | 5,224 | 7,235 | **222** |
| batched K = 32, decode-layout carries | 0 | 17,261 | 3,764 | 98 | 5,352 | 7,617 | 284 |
| batched K = 32, lane (2-block plan) | 1 | 16,824 | 3,385 | 99 | 5,448 | 7,507 | **167** |
| decode, 12 blocks (P1, `test_full_plan`) | all | – | – | – | – | – | 478 per token |

**Refreshes are linear in K:** block 0 = 31 + 6K (55, 79, 127, 222 at K = 4, 8, 16, 32; slope 6.0, 6.0, 5.9). In
block 1 at K = 32 it is 167 = 39 + 4·32.

**Where they sit (the census, K = 32, block 0).**
- Planned placements: 142.
  - 128 of them are in `tm.state.update`, **4 per conversation**. They refresh S_m ⊙ w and the `block_allreduce`
    output (the S·a term), one each per state ciphertext. The update's depth does not fit above the level the carried
    state lands at.
  - The other 14 are on the token side.
- The rest are deliberate refreshes: 66 for the carried state (64 S + 2 lane carries) and the PCMM / tail ones.
- Block 1 has **2** update refreshes per conversation instead of 4. The plan does not say why; different magnitudes
  are the likely cause.
- So each conversation pays its own **6 (block 0) / 4 (block 1) refreshes per block**: 2 carry + 4 / 2 update.

**Rotations, K = 32, block 0.** Per conversation: `bcast` 41 (3 × 5 copy + 26 transpose), `readout` 34, `update` 24 (2 ×
12 `block_allreduce`), `src` 3. That is 102 per conversation, 3,382 per block. The token side's own rotations are ≈ 120.

**Against K decode steps (block 0):**

| | refreshes | rotations + conj. | mult |
|---|---|---|---|
| 32 decode steps | 32 × 31 = 992 | 32 × 211 = 6,752 | 32 × 788 = 25,216 |
| one K = 32 pass | 222 (**4.5× fewer**) | 3,480 (1.9× fewer) | 5,224 (4.8× fewer) |

## 3. Validated on the CPU (the records)

All from `tests/test_batched.py`: 2 blocks of the real checkpoint, 2^15 slots (t = 32, the device's lane count), after
the public prefix `"\n\n"`.

| check | result |
|---|---|
| mirror: K conversations batched vs K separate decode mirrors. Prompts of 2–6 tokens, different per conversation, then 2 free-running greedy steps each (the pick fed back to its own lane). K = 2 (decode carries), 5 (decode carries; lane carries; decode carries + hoisted rotations, B = 3), 32 (lane carries; decode carries) | every conversation's hidden state at every step, and its carried state at the end, **bit-identical** (rel. 0.0, abs. 0.0). The greedy token streams are identical |
| the same, non-vacuity | the conversations differ (rel. > 1e-2), and swapping the lane assignment on one pass breaks the equality (> 1e-3) |
| fake (runtime level rules, PCMM fake, canonical carry), K = 3, prompts of 2 / 4 / 3 tokens + 2 generated, decode carries and lane carries | vs 3 separate fake decode runs: **0.0**; **top-1 15/15** against each conversation's poly model (clamp on); hidden vs poly relErrRms ≤ 3.1e-5 |
| capture recorder, K = 3, lane carries | **top-1 15/15**; hidden vs poly ≤ 3.1e-5 |
| K = 1 | the fake's call log is identical to decode's (8,154 calls for 3 tokens), same rotation keys, hidden states `array_equal` |
| capture + plan | every conversation's `s0c<c>`, `s1c<c>` (and the carries) are block inputs; plans as §2 |
| `read_prompts` | `\n` / `\\` escapes, the prefix prepended when missing |

**Why bit-identical is not a red flag here.** Every op a pass adds is exact in float64: masks of 1 and ½, adding
zeros, rotations, and `x/2 + x/2`. The fake has no CKKS noise. The non-vacuity checks show the equality is not
trivially true. On the device the extra key switches (≈ 102 per conversation per block) add key-switch noise. That is
**unmeasured**.

The runner's device half (`Rwkv7BatchedRun.prepare`: `Program.compile` of a batched plan, the state registers) needs
the CUDA build's `perseus._pc`. Its test is `bcheck` on the box. It was skipped here.

## 4. Expected throughput and latency (an estimate, not a measurement)

**Method** (P3's, PREFILL_DESIGN.md §4):
- **Inputs.** Op counts from §2, and Perseus's published per-op times as **ratios only**: refresh 28.77 ms,
  rotation 214 µs, ct × ct 203 µs, ct × pt 72 µs, add 33 µs (another card and another build, R6).
- **Scale.** From this port's measured decode on the 600 W box, 9.54 s per token. A decode token is 15.37 s (a mult
  priced as ct × pt) to 16.61 s (as ct × ct) in published units, so the scale is 0.621 / 0.574.

**Model.** `pass(K) = A + B·K`.
- Fixed per pass, ≈ decode's token side: refreshes 31 (block 0) + 11 × 39 + 1 = 461. The decode plan has 478, which
  fits. Plus 1.5k rotations, 7.8k mult and 10.5k adds. That is 14.5–15.5 published s, so **A ≈ 8.9–9.0 s**.
- Per conversation:
  - refreshes 6 + 11 × 4 = 50 (**low**: blocks 2–11 like block 1) or 6 × 12 = 72 (**high**: like block 0);
  - plus ≈ 1,260 rotations, 1,720 mult and 2,390 adds;
  - that is 1.91–2.14 published s (low) or 2.54–2.77 (high), so **B ≈ 1.19–1.23 s (low) or 1.58–1.59 s (high)**.

| K | pass (s/token per conversation: the latency) | s per conversation-token (pass / K) | aggregate tokens/s | vs 9.54 s decode |
|---|---|---|---|---|
| 1 (decode) | 9.54 (measured) | 9.54 | 0.105 | 1× |
| 4 | 13.8–15.3 | 3.4–3.8 | 0.26–0.29 | 2.5–2.8× throughput |
| 8 | 18.6–21.7 | 2.3–2.7 | 0.37–0.43 | 3.5–4.1× |
| 16 | 28.2–34.4 | 1.8–2.2 | 0.47–0.57 | 4.4–5.4× |
| **32** | **47–60** | **1.5–1.9** | **0.53–0.68** | **5.1–6.5× throughput, 4.9–6.3× the latency** |

Cross-check at K = 32 from the totals instead of the fit:
- refreshes 222 + 11 × 167 + 1 = 2,060 (low) to 222 + 11 × 232.5 + 1 = 2,781 (high);
- rotations + conj. ≈ 41,800, mult ≈ 65,150, adds ≈ 89,800;
- that is 75.9–84.4 published s (low) and 96.6–105.2 (high), so **47.1–48.5 s and 60.0–60.4 s per pass**.

**What it says.**
- Batching buys throughput, not latency. Each conversation waits ~5–6× longer per token, while the GPU serves 32.
- At K = 32 the per-conversation state work is 38–51 of the 47–60 s. The asymptote is B ≈ 1.2–1.6 s per
  conversation-token, about 13–17 % of a decode step. The next gains therefore come from **refreshes per conversation**
  (§5), not from more sharing of the token side.
- Refreshes are ≈ 75 % of a pass's priced time; the per-conversation rotations ≈ 11 %.

**Memory at K = 32** (the decode run's report, `r01b/run_P1.jsonl` header: `replay.regs 853 MB`, `ct_pool 4032 MB`
at `--pool 128`, `keys 8437 MB`, total 68.5 GB of 97.9 GB):
- **Registers.** A carried ciphertext at the landing level is 2 × 2^16 × 27 limbs × 4 B ≈ 14.2 MB; decode has 61
  registers ≈ 0.85 GB, which fits. With lane carries the batched state is 12 × (64 + 2) = 792 registers plus 13 stage
  outputs, **≈ 11.4 GB (+10.6 GB)**. With decode-layout carries it is 12 × 128 = 1,536, ≈ 21.8 GB.
- **Pool.** A pass holds the token-side values, the three sources (rotated lazily: one copy each at a time) and one
  conversation's temporaries. That is prefill's live set, so `--pool 192` (+2 GB at 31.5 MB a slot).
- **Keys.** +2 (+1 and −31) with the running rotation; hoisting would add 30 more (≈ 5.7 GB at ≈ 190 MB a key).
- **Total ≈ 81 GB** if the replay slabs do not grow with the 7× larger programs. **That is unknown.** If the
  session runs out, lower `STC_CACHE_B` first: `bts.stc_first` is 32.8 GB at 160.

## 5. Goal 2: levers like it, judged by encrypted cost

| lever | built? | judgement |
|---|---|---|
| **lane-form token-shift carries** | **built, default** (`--batch-carry lane`) | −2(K − 1) refreshes and −12K rotations per block, measured by the planner: 284 → 222 at K = 32, block 0 (−22 %). No op at all on the token-shift path |
| **lane → all-lanes copy as a doubling** | already so | `rotate_and_sum(−1, 32)` is log₂ 32 = 5 rotations (and the runtime does a ladder in hoisted radix groups, `FheOps.rotate_and_sum`), not 31. Bringing lane c to lane 0 first is 1 rotation |
| **hoisted lane rotations** (`--batch-hoist B`: `rot(src, c)` from one decomposition, B per call) | built, tested on the mirror (K = 5, B = 3); **off by default** | It saves part of 3 × 31 = 93 rotations per block: at most 93 × 214 µs ≈ 20 ms of ≈ 6.3 s per block in published units, < 0.3 % of a pass. It costs 30 more rotation keys, ≈ 5.7 GB of VRAM, at the point where the registers already add 10.6 GB. Does not pay. The same holds for prefill's +1 chain |
| **PAIR=1 + siblings schedule** (`LEVERS=best`) | exists (P3), applies unchanged | The per-conversation refreshes are pairs of real siblings at one level: S_0 ⊙ w / S_1 ⊙ w, the two `allreduce` outputs, and the S_0 / S_1 carries. The census sample shows them at equal levels (28 / 28, 30 / 30). `Pairing::takes` pairs real StC-first siblings into one complex refresh, so per-conversation refreshes could halve (50 → 25 per pass, low case; 72 → 36, high case). If a paired complex refresh costs one real refresh, that saves 0.43–0.62 s per conversation-token (25 or 36 × 28.77 ms × the 0.57–0.62 scale): **≈ 34–40 s per pass at K = 32** (0.8–0.9 tokens/s, 7.6–9× decode's throughput) if pairing holds. The plan count does not show it (pairing happens in `Program.compile`); only the device's `n_bootstraps` does. Run it as the second batched run |
| **mixed passes** (prompt tokens of some conversations next to decode tokens of others) | **built** (inherent) | Positions are per lane and the circuit does not distinguish a prompt token from a generated one. The runner's passes are mixed whenever prompt lengths differ (the default prompt file: 5–29 tokens) |
| **continuous batching** (a new conversation joins on a freed lane mid-run) | **built** (B4, `serve.py`, `run_planned --serve`; SERVE_DESIGN.md) | The state is a register hand-over (the single-conversation programs write lane c's S registers), the lane-form carries are joined by a small compiled program (`xa ⊙ (1 − m_c) + xa_new ⊙ m_c`, one refresh per carry: 24 per join for the 0.1B, masks re-staged per join), a leave resets the lane to the public prefix state. Bit-identical to each conversation decoded alone on the mirror, identical on the fake (top-1 36/36). Unmeasured on a device. **Deployment caveat:** all K lanes are under one secret key, so a batch serves conversations of one key holder, not K mutually distrusting users |
| **unused slots** (0.1B: channels 768–1023 empty, 4 of 16 head blocks in every state ciphertext) | not built | A conversation's 12 heads fill 24 of 32 head blocks of its two state ciphertexts. Two conversations' states fit 3 ciphertexts instead of 4, which saves 25 % of the per-conversation state refreshes and products: ≈ 15 % of a K = 32 pass. It costs a per-conversation channel rotation of the broadcast vectors (heads 0–3 of every second conversation to blocks 12–15), a second mask set and its own transpose / readout bookkeeping, a new layout through `state_slots` and the planner. **Gone at 0.4B** (1,024 channels = 16 heads, no padding). The biggest unbuilt lever for 0.1B; not testable within this budget |
| **two conversations per transpose / untranspose** (complex packing, PREFILL_DESIGN.md §5.1) | not built | `transpose(v_c + i·v_c')` once plus a conjugation; the same for the readout. ≈ 27 of the 102 rotations per conversation: ≈ 3 % of a pass. Small next to the refreshes |
| **refreshing the shared sources once instead of S per conversation** | not built (a planner question) | The 4 / 2 update refreshes per conversation exist because the update's depth does not fit above the carried state's landing level. A plan that lands the carried state with more headroom (or forces a refresh of the three sources, which is shared over K) could remove them. The mincut placer chose per-conversation refreshes; whether a forced placement beats it is a planner experiment (PLAN_* switches), worth up to −2 / −4 refreshes per conversation per block |

## 6. Device failure points, most likely first

1. **Memory.** +10.6 GB of registers, and replay slabs for programs 7× larger than decode's. Watch the header's
   `mem_report_text`. If it fails, lower `STC_CACHE_B` (80 frees ≈ 16 GB), then try K = 16 (`B_K=16`: half the
   registers).
2. **Setup time.** The twin records 2 probe passes × 12 blocks of ≈ 16k nodes (P3: ≈ 10 s per block-pass on the Mac),
   and `Program.compile` runs over ≈ 200k nodes. Expect minutes, untimed.
3. **Pool exhaustion** in the per-conversation loop. Raise `POOL_B`.
4. **Key-switch noise** of the running +1 rotation (up to 31 chained per source) and the Horner chain (31). Watch
   `hidden_relErrRms_vs_mirror` per conversation: growth with the conversation index (the lane) points here. The cure
   is `--batch-hoist 8` (keys permitting).
5. **Block 1's lighter update** (2 refreshes per conversation instead of 4) holds on public text. A conversation whose
   state grows larger than the capture's could need block 0's margin. If the error jumps in one conversation, re-plan
   with `PLAN_MAG_SAFETY=3`.
6. **Register count.** 805 registers per session against 61 for decode. The runtime's register table has never held
   that many. `bcheck` (Program.compile with a K = 4 plan) catches mapping errors, not capacity.
