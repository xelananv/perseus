# The filled layout for the RWKV-7 port: two value groups of the state per ciphertext (`--layout filled`), 2026-11-05

**Result.**
- **Batched pass, K = 32, PLANOPT=b2:** `Program.compile`'s count goes from **87 / 89 to 55 / 57 refreshes** in
  blocks 0 / 1, exactly one fewer per conversation and block. Rotations go from 3,480 to 2,328 per block, ct × ct
  products from 650 to 490 and plaintext products from 4,268 to 2,188.
- **Estimated pass (12 blocks):** **≈ 17.5–18.9 s instead of ≈ 27.4–29.6 s** on the same model and price, with
  LEVERS=best on both sides. That is ≈ 1.5× the aggregate throughput of `LEVERS=best` + b2 on the current layout
  (§1; an estimate, R6).
- **Decode:** 25 / 28 → 24 / 26 refreshes per block, ≈ 3–5 % per token.
- **Prefill (T = 32): worse.** 52 / 74 → 84 / 104 refreshes, ≈ +19 % per pass with LEVERS=best. The planner's placement causes it,
  not the circuit (§3).
- **Opt-in and default off.** The token side, `z = h + i·v_first` and the three broadcast pairs do not move. The
  current layout's op stream is unchanged.

Every count below comes from the plan directories under `.cache/rwkv7_b3fill/` (gitignored; rebuilt by §7) and from
B2's under `.cache/rwkv7_b2plan/`. The table is `tools/rwkv7/filled_table.py`'s output.

## 0. Which filling, and why not the one the brief sketched

A CKKS slot holds a complex number. Two different things can be put on the imaginary axis.

**(A) The state's value groups (built).**
- Head h's 64 × 64 state `S[h, i, j]` has its value index i on the lanes: group m holds rows i = t·m + l on lane l.
  That makes `M = 64 / t` real ciphertexts (2 at t = 32).
- The filled layout puts group q on the real axis and group q + M on the imaginary axis of ONE ciphertext:
  `S_q + i·S_{q+M}`, so M halves (1 at t = 32).
- **Why it is nearly free:** every operand the state meets in the update (`w`, `κ̂`, `κ̂⊙a`, `k̃`, `r`) is indexed by
  j only. It is a real natural vector and the same for both groups. So `(S_0 + i S_1)·w = S_0 w + i S_1 w` is ONE
  ciphertext product serving both groups, with no split, because one factor is real.
- The block sums, the block-start masks (real plaintexts) and the carry refresh are real-linear, so they serve both
  axes too.

**(B) The tokens / conversations (not built).** This is our own model's "64 pairs": lane l real, lane l + 32
imaginary.
- It puts 2t tokens or conversations through the token side per pass.
- Every ct × ct product of two such values mixes the axes: (a + ib)(c + id) = (ac − bd) + i(ad + bc). So every
  polynomial site, Newton chain and LayerNorm variance must go through a conjugation split.
- It also needs `v_first` moved off the imaginary axis (its own carried ciphertext) and the three broadcast pairs
  reworked.
- **In decode it fills nothing**: there is one token, replicated over the lanes.

Only (A) changes the state's ciphertext count, and the state is where B2's correction lives. The carried S
ciphertexts are complex payloads (route `dense_stcf`), so PAIR=1 cannot pair them. In both layouts they cost one
complex refresh each, so halving their number halves their refreshes, with no overlap with PAIR. (B)'s extra gain on
top of (A) is estimated in §4: at most ≈ 1.2×, at K = 64. That K needs the memory of today's K = 32 session, which
already peaked at 94.9 of 97.9 GB. So (B) is priced, not built.

**The brief's premise about this Mac's ring.** The CPU fake and recorder run at 32,768 slots
(`session.py`, `LOGN = 16`), i.e. t = 32 lanes, so K = 32 and T = 32 are native here. B2's K = 32 captures and mine
are the device's own shape, so nothing below is scaled from K = 16.

## 1. The cost table (2-block captures of the 0.1B, `coeffs_final.json`, LANDING=high; `Program.compile`'s count)

**Columns.**
- **sites:** `Program.compile`'s refresh sites (placed + fired hints + deliberate carries), which is the device's own
  `n_bootstraps` (PLAN_B2.md §2). **pairs:** PAIR=1's merged sibling pairs.
- **12-block refreshes:** block 0 + 11 × block 1 + the tail, counting a pair as one refresh.
- **Op counts:** from the compile check's census (`compile_check.json` `limbs`).
- **Price:** P4's model (BATCHED_DESIGN.md §4). Published per-op times are used as ratios: refresh 28.77 ms,
  rotation or conjugation 0.214 ms, ct × ct 0.203 ms, ct × pt 0.072 ms, add 0.033 ms. They are scaled by 0.574–0.621
  to this port's measured decode. **These are estimates, R6:** another card and another build.
- **+best:** LEVERS=best's plan environment (PAIR=1, `PC_EM_FUSE_RESCALE=1`, siblings schedule).
- **Batched rows** are under PLANOPT=b2.

| plan | sites b0 / b1 | pairs b0 / b1 | rotations + conj. b0 / b1 | ct × ct b0 / b1 | ct × pt b0 / b1 | 12-block refreshes | 12-block rotations + conj. | est. s per pass |
|---|---|---|---|---|---|---|---|---|
| decode, current | 25 / 28 | 0 / 0 | 211 / 215 | 340 / 371 | 142 / 142 | 334 | 2,576 | 6.8–7.3 |
| decode, current +best | 25 / 28 | 1 / 3 | 211 / 215 | 340 / 371 | 142 / 142 | 300 | 2,576 | 6.2–6.7 |
| decode, **filled** | 24 / 26 | 0 / 0 | 175 / 179 | 335 / 366 | 77 / 77 | 311 | 2,144 | 6.3–6.8 |
| decode, **filled** +best | 24 / 26 | 1 / 2 | 175 / 179 | 335 / 366 | 77 / 77 | 288 | 2,144 | 5.9–6.4 |
| prefill T = 32, current | 52 / 74 | 0 / 0 | 3,494 / 3,498 | 650 / 681 | 4,274 / 4,274 | 867 | 41,972 | 24.3–26.3 |
| prefill T = 32, current +best | 52 / 74 | 8 / 17 | 3,494 / 3,498 | 650 / 681 | 4,274 / 4,274 | 672 | 41,972 | 21.1–22.8 |
| prefill T = 32, **filled** | 84 / 104 | 0 / 0 | 2,342 / 2,346 | 490 / 521 | 2,194 / 2,194 | 1,229 | 28,148 | 26.7–28.9 |
| prefill T = 32, **filled** +best | 84 / 104 | 9 / 8 | 2,342 / 2,346 | 490 / 521 | 2,194 / 2,194 | 1,132 | 28,148 | 25.1–27.2 |
| prefill T = 32, filled, forbid xpose + readout (§3) | 75 / 103 | 0 / 0 | 2,342 / 2,346 | 490 / 521 | 2,194 / 2,194 | 1,209 | 28,148 | 26.4–28.5 |
| batched K = 32, b2, current | 87 / 89 | 0 / 0 | 3,480 / 3,484 | 650 / 681 | 4,268 / 4,268 | 1,067 | 41,804 | 27.6–29.8 |
| batched K = 32, b2, current +best | 87 / 89 | 1 / 1 | 3,480 / 3,484 | 650 / 681 | 4,268 / 4,268 | 1,055 | 41,804 | 27.4–29.6 |
| batched K = 32, b2, **filled** | **55 / 57** | 0 / 0 | 2,328 / 2,332 | 490 / 521 | 2,188 / 2,188 | **683** | 27,980 | **17.7–19.1** |
| batched K = 32, b2, **filled** +best | **55 / 57** | 1 / 1 | 2,328 / 2,332 | 490 / 521 | 2,188 / 2,188 | **671** | 27,980 | **17.5–18.9** |

**Per conversation and block (K = 32, filled against current).** The plan's differences are exactly the kernel
counts of §2, times 32:

| | current | filled |
|---|---|---|
| refreshes (the carry of S) | 2 | 1 |
| rotations + conjugations | 104 | 68 |
| ct × ct | 10 | 5 |
| ct × pt | 133 | 68 |

The difference per block is −32 sites, −1,152 rotations + conjugations, −160 ct × ct and −2,080 ct × pt. The 21
rotations + conjugations per conversation that do not change are the lane-to-0 rotations and the three broadcasts
with their splits. The token side (≈ 23 refreshes, ≈ 150 rotations, ≈ 330 ct × ct per block) is the same in both
layouts.

**PAIR=0 against PAIR=1.**
- On the batched pass, PAIR=1 still finds 1 pair per block in either layout. The carries are complex, so PAIR cannot
  touch them, and b2 already removed the pairable per-conversation refreshes (PLAN_B2.md §2).
- So the filled layout's gain on this pass is additive to `LEVERS=best`: 1,055 → 671 refreshes per pass.
- On decode and prefill, PAIR=1 pairs token-side real siblings, about equally in both layouts.

**What the pass prices say, and how far to trust them.**
- The model omits PCMM's GEMM, `mult_const`, `rescale` and host time. On decode it predicts 6.2–6.7 s against the
  measured 9.70–9.74 s per token with LEVERS=best on the final coefficients (`r01b_final/run_best_P*.jsonl`). That
  leaves ≈ 3.0–3.5 s per pass of fixed work the model does not see.
- The batched current b2 row (27.6–29.8 s) matches B2's refresh-only estimate (28.7–30.5 s, PLAN_B2.md §5).
- Adding the decode-calibrated remainder to both sides:
  - current + best + b2: ≈ 30.4–33.1 s per K = 32 pass, ≈ 0.97–1.05 tok/s;
  - filled + best + b2: ≈ 20.5–22.4 s, ≈ 1.43–1.56 tok/s.
- **Expected gain over LEVERS=best + b2 on the batched pass: ≈ 1.45–1.5× aggregate throughput.** The latency per
  conversation drops with it, ≈ 21 s instead of ≈ 31 s per token. **An estimate (R6).** Only a device run measures it.
- **Decode:** ≈ 0.3 s per token in model units (5.9–6.4 against 6.2–6.7), ≈ 3–5 % of the measured 9.7 s.

**Memory (estimate).** At 12 blocks and K = 32 the batched state is 12 × (32 + 2) = 408 carried registers instead of
12 × (64 + 2) = 792. At ≈ 14.2 MB per register at the landing (BATCHED_DESIGN.md §4) that saves ≈ 5.5 GB on the card
that peaked at 94.9 of 97.9 GB. Rotation keys: 36 instead of 42 layout steps (the filled steps are a subset; §2).

## 2. The mechanism, per kernel (`layout.py`; counts per head block at t = 32, checked in `tests/test_filled.py`)

**Shift identity.** Group m's transforms are group 0's composed with a rotation by t²·m slots, which is t·m channels:
- `transpose_m(v) = transpose_0(rot(v, t² m))`;
- `untranspose_m(P) = rot(untranspose_0(P), −t² m)`.

Both transforms are real-linear, made of rotations, real masks and adds.

**`transpose`.**
- Input: `x = rot(v, t² q) + i·rot(v, t² (q + M))` (one rotation and a level-free `mult_i` at t = 32).
- Then group 0's transform once: 7 baby rotations, 3 giants, 32 mask products and a 6-rotation copy.
- **17 rotations and 32 mask products** instead of 26 and 64.

**State step.** `S⊙κ̂`, the block sum, `·bb`, `S⊙w`, `V⊙k̃`, `S′⊙r` run once on the filled ciphertext.
**5 products and 12 rotations** instead of 10 and 24.

**`untranspose`.**
- Group 0's transform once on both axes: block sums 6, babies 7, giants 3, masks at value 1/2.
- One conjugation splits it: `u + conj u` is group q, `i(conj u − u)` is group q + M. The masks' 1/2 makes both exact.
- One rotation by −t² places the imaginary half.
- **17 rotations + 1 conjugation and 32 mask products** instead of 33 and 64.

**Totals.** 83 → 46 rotations + 1 conjugation, 10 → 5 ct × ct, 130 → 65 ct × pt. At t = 16 (2^14 slots,
M = 4 → 2): 145 → 88 key switches.

**Depth is unchanged.**
- A rotation, a `mult_i`, a conjugation and an add cost no level.
- The masks are the same one level they were.
- So S's chain is 1 level per token in both layouts: the recorder's levels along S read 32, 38, 40, 42, … identically
  in both prefill captures.

**Exactness.** The filled kernels' partial sums have disjoint slot supports (each term is masked to its own lanes),
so they reproduce the current kernels bit for bit. Masking by 1/2 and doubling are exact in float64. That is why the
mirror and fake comparisons below are exactly 0.0, and the axis-swap check shows those zeros are not vacuous.

**Payload.** The filled state is a complex payload, so its encryption is complex:
- `Rwkv7Fhe.state_rows` returns complex rows;
- `Rwkv7Fhe.pack_rows` packs `pack(Re) + i·pack(Im)` on the fake and the recorder;
- `run_planned.encrypt_input` keeps complex rows complex for `fhe.encrypt` / `_pc.Client.encrypt`, both of which take
  complex values.

The current layout keeps its float64 path.

## 3. The moved tricks, and the one that regressed

**Nothing on the token side moves.**
- `z = h + i·v_first` still carries `v_first` on the imaginary axis of the stage ciphertext. The state never shares a
  ciphertext with z.
- The three broadcast pairs (`w + i·κ̂`, `(κ̂⊙a) + i·k̃`, `r + i·v`) still share one broadcast each and split with one
  conjugation. Each split vector is real and natural, which is exactly what the filled state wants as a multiplier.

**The added cost is small.**
- One conjugation per head-block readout (+1 per conversation and block in a batched pass, +1 per token in prefill).
- One `mult_i` and one rotation per transpose.
- Both are already counted inside the 46 + 1 above.

**What regressed: the prefill plan.**
- In the current layout the prefill S chain is tagged real (`pack_field` 1). The planner refreshes S itself about
  every 3 tokens (25 placements on S, 2 chains, block 0) on the real route `dense_stcf_real`, whose entries reach
  level 48.
- The filled S is complex (`pack_field` 3). The planner instead scatters refreshes over S's products (`S⊙w`,
  `V⊙k̃`), over the broadcast copies and over the transform sums: 61 `tm.state` placements in block 0.
- **The circuit does not need them.** The fake's own reactive rule (refresh at the ceiling) needs fewer refreshes in
  the filled prefill: 146 → 128 per 4-token pass, 2 blocks (`test_fake_filled_passes`).
- The magnitudes are the same: |S| ≈ 1.10 in both, coefficient magnitude 3–4 × 10⁻⁴ (`output_max_abs`,
  `output_max_coeff` of the S adds).
- Forbidding the transforms (`FORBID_STEPS=…,tm.state.xpose,tm.state.readout`; filled-only scopes in
  `prefill.py`) moves only 9 refreshes into the chain (84 → 75 in block 0).
- **Leading hypothesis, unverified:** the real-route entry (48 against the complex route's 46), plus the planner's
  per-site pricing (`placer/refresh.py` `_build`, `real=` only for real-tagged values), make the S cut cheaper for the
  min-cut in the current layout.
- The same planner question will apply to any complex-payload chain. It is listed here and not fixed: no edit to
  `python/perseus/plan/`.

**Consequence for sessions.** Decode, prefill, batched and join programs hand the same state registers over
(B4's serve session). So one session runs one layout:
- **batched-only or decode + batched sessions:** use `filled`;
- **serve sessions with prefill:** today's choice trades the batched gain (≈ −10 s per pass) against the prefill
  regression (≈ +4.0–4.4 s per 32-token pass with LEVERS=best).

A cheaper route exists and is not built: run prefill in the current layout and convert at the hand-over.
- current → filled is `S_0 + i·S_1`, an add and a `mult_i`, with no level and no key switch;
- filled → current is one conjugation and two adds per ciphertext;
- both fit into the existing join program.

## 4. (B), the token fill: what it would add (an estimate; not built)

**Per block at K = 32 in the filled layout, from §1:**
- **token side:** ≈ 23 refreshes, ≈ 150 rotations, ≈ 330 ct × ct, ≈ 0.76 published s;
- **per conversation:** 1 refresh, 68 rotations + conjugations, 5 ct × ct, 68 ct × pt, ≈ 49 published ms.
- **So at K = 32, 67 % of a block is already per conversation.**

**Filling the tokens as well (K = 64 at t = 32):**
- the token side's site and Newton refreshes double, and PAIR=1 may merge the halves again;
- every site product doubles;
- `v_first` needs a carried ciphertext (+1 refresh per block);
- the per-conversation part is unchanged. The broadcasts become six complex sources per two conversations, which is
  still three per conversation.

**Per conversation per block:**
- (A) alone at K = 32: 2.34 / 32 ≈ 73 published ms;
- (A) + (B) at K = 64: ≈ 62 ms if PAIR merges every split and ≈ 72 ms if it merges none.

**So (B) adds at most ≈ 1.0–1.2×.** It needs a K = 64 state, the registers of today's K = 32 current session (which
peaked at 94.9 of 97.9 GB), and it doubles the time a batch waits for a reading prompt. Not built.

## 5. What is verified on the CPU (`tests/test_filled.py`, plus the existing suite with the layout off)

| check | result |
|---|---|
| kernels at 2^15 / 2^14 slots: transpose, state products, untranspose against the matrix algebra and the current layout | equal to 1e-11; the readout's imaginary part < 1e-12 |
| kernel counts per head block, t = 32 | rotations 83 → 46, conjugations +1, ct × ct 10 → 5, ct × pt 130 → 65; every issued step in `rotation_steps()` (36 keys, a subset of the current 42) |
| mirror, exact matrices, 2 blocks: decode 4 tokens, prefill T = 5 and 32, batched K = 3 and 32 (2 passes each) | hidden states and carried states (decoded to matrices) **0.0** apart from the current layout; an axis-swapped state breaks it by > 1e-3 |
| fake (runtime level rules) and capture recorder, decode 4 tokens, 2 blocks | vs the filled mirror ≤ 1.0e-14; vs the current fake **0.0**; one op stream per token; fake refreshes per token 183 → 143, recorder 115 → 109 |
| fake, prefill T = 4 (2 passes, then a decode step); batched K = 3 on the fake and the recorder (2 passes) | **0.0** against the current layout; refreshes per pass 146 → 128 (prefill), 143 → 125 (batched fake), 125 → 119 (recorder) |
| 1-block K = 4 capture: state names | `s0c0..s0c3, xa, xf` (no `s1c<c>`), layout recorded in `rwkv7_capture.json` |
| 1-block K = 4 capture + b2 plan (`--rwkv7-plan`, final coefficients) | filled 40 (carry 6), current 44 (carry 10), 0 per-conversation placements |
| 2-block captures + plans + compile check (§1) | every compile check 0 check errors, `level richer 0`; filled routes `dense_stcf@26` for the state, StC-first everywhere |

**Unverified (only the device shows them):**
1. Time per pass, and whether the 1,152 removed rotations per block price as published.
2. **Precision of a complex carried state.** The fake has no CKKS noise. A complex refresh refreshes both groups with
   one EvalMod on each axis. Watch `hidden_relErrRms_vs_mirror` per conversation against the current layout's run.
   It should be equal at pass 0; if it grows with the pass index, suspect the complex carry.
3. Complex encryption of the prefix state on the device (`fhe.encrypt` / `Client.encrypt` with complex values). This
   is the first complex input of this port.
4. Blocks 2–11 (the Mac plans two). Expect ≈ 55–60 sites per block at K = 32.

## 6. On the box

```bash
export OUT=... PTH=.../RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth COEFFS=.../coeffs_final.json
B_K=4 LAYOUT=filled bash tools/rwkv7/run_box.sh bcheck            # once: the filled registers through Program.compile
LAYOUT=filled PLANOPT=b2 LEVERS=best B_K=32 B_GENERATE=32 POOL_B=1024 STC_CACHE_B=64 bash tools/rwkv7/run_box.sh b
# the A/B anchor on the same box and build (R6): the same line with LAYOUT unset
```

- `LAYOUT` (`current` | `filled`) adds `--layout` to every step, and the suffix `_filled` to every capture, plan, key
  bundle and record, so a filled run never reuses a current-layout plan.
- It combines with `LEVERS`, `PLANOPT`, `B_K` and `PF_T`.
- For decode alone: `LAYOUT=filled LEVERS=best bash tools/rwkv7/run_box.sh capture plan run`.

## 7. Reproduce (CPU, under `tools/memguard.sh 6`, one process at a time)

```bash
# captures (2 blocks, final coefficients, B2's shifts so the token side is identical)
bash .cache/rwkv7_b3fill/cap.sh cap_dec_filled --layout filled
bash .cache/rwkv7_b3fill/cap.sh cap_pf32_filled --layout filled --prefill-chunk 32 --shifts .cache/rwkv7_b2plan/cap_pf32_b2/pcmm_shifts.json
bash .cache/rwkv7_b3fill/cap.sh cap_K32_filled --layout filled --batch 32 --shifts .cache/rwkv7_b2plan/cap_K32_lane_b2/pcmm_shifts.json
# plans (b2_plan.sh prints the census; PAIR=1 RUNTIME_EXTRA=PC_EM_FUSE_RESCALE=1 for +best)
PY=$PY bash tools/rwkv7/b2_plan.sh .cache/rwkv7_b3fill/cap_K32_filled .cache/rwkv7_b3fill plan_K32_filled_b2 b2
bash .cache/rwkv7_b3fill/plans.sh          # the other rows
python3 tools/rwkv7/filled_table.py "LABEL=PLAN_DIR" ...
```

B2's current-layout plan `plan_K32_b2_fs` had no `compile_check.json`. It was re-checked into
`.cache/rwkv7_b3fill/cc/K32_current_b2/` (177 sites, the same as B2's summary).
