# RWKV-7 prefill under CKKS: design, 2026-11-04

**Goal.** Read T prompt tokens per encrypted pass, at a much lower cost per token than the decode step (13.05 s per
token on the 300 W card, `results/rwkv7-fhe-20261104/box_run4_full/run.jsonl`, summary line). Then hand the state to
the decode loop that already exists.

**Status.**
- Built: `prefill.py` and the `Circuit` hooks in `model.py`.
- Validated on the CPU twin: `tests/test_prefill.py`.
- Planned on the CPU: 1 and 2 blocks, T = 8 and 32.
- Wired into the device runner (`run_planned.py --prefill-chunk T --pf-plan P`) and the box script
  (`run_box.sh capture_pf | plan_pf | pfcheck | run_pf | pf`).
- **Nothing has run on a GPU.** Every second in this file is an estimate (§4).

## 1. The idea: token lanes

The decode step copies one token over the t = 32 lanes of the filled layout (channel c, lane l at slot 32c + l;
`layout.py`). Those 32 copies are pure redundancy. A prefill pass puts **token l of the chunk on lane l** instead.

Then almost the whole block is already token-parallel, at the decode step's cost, with no code change:

| step | why it is per-lane |
|---|---|
| PCMM linears (r, k, v, a, w, g, the LoRAs, Wo, the FFN) | PCMM multiplies every lane by the same matrix (`reshape(1024, t)`) |
| every Chebyshev site, every Newton chain | slot-wise |
| LayerNorm (`channel_sum`), GroupNorm, `kk` and the bonus (`block_allreduce`) | they sum over channels *within* a lane: steps 32, 64, … never cross a lane |
| `z = h + i·v_first` | slot-wise: lane l carries token l's v_first |

Only two things couple tokens: the **token shift** and the **WKV recurrence**.

**Token shift** (`PrefillCircuit.token_shift`):
- `x_prev` of lane l is lane l − 1 of the same ciphertext: one rotation by −1 and a mask (one level).
- Lane 0 instead takes the carried LN output of the previous pass or decode step. That carry is in natural form, every lane equal.
- The carry this pass leaves is lane T − 1 copied to every lane: a rotation by T − 1, a lane-0 mask, and the lane copy (5 rotations).
- So the carry has the decode step's form, and decoding can continue from it directly.

## 2. The WKV recurrence: three options, priced by encrypted cost

`S_t = S_{t−1}(diag(w_t) − κ̂_tᵀ(a_t⊙κ̂_t)) + v_tᵀk̃_t`, read out as `y_t = S_t r_t`, per head (64 × 64).

In CKKS the cost is products of two ciphertexts, rotations (each a key switch) and, above all, **refreshes**. A
refresh is a bootstrap: about 28.77 ms against 0.2 ms for a product, on the published figures. Plaintext FLOPs do
not price anything here.

### (a) Sequential scan inside the pass (chosen)

Keep the decode step's state layout: value index i on the lanes, key index j on the channels, M = 2 ciphertexts per
block. Run the decode step's own `state_step` T times inside the pass. Each token t then needs its six j- or
i-indexed vectors in natural form.

**Broadcast (6 → 3).** The pairs `w + i·κ̂`, `(κ̂⊙a) + i·k̃` and `r + i·v` each share one broadcast:
- a running rotation by +1 (the source rotated once per token);
- a lane-0 mask that carries the factor ½;
- the lane copy (5 rotations).

One conjugation then splits a pair: `x = B + conj B`, `y = i·(conj B − B)`. This is the `z = h + i·v_first` trick
again: a real vector is its own conjugate, so a real and an imaginary part separate exactly.

**Readout.** `untranspose_lane0` leaves token t's `y` on lane 0, and the pass accumulates it Horner-style:
`Q ← rot(Q, +1) + acc_t`. One final rotation by −(T − 1) puts token t's readout on lane t. Lanes beyond a short
chunk get the last token's readout, so GroupNorm sees in-range values there.

**Cost per token per block** (counted from the code):
- rotations: 3 × (1 + 5) + 3 conjugations (the broadcasts) + 26 (`transpose(v)`: 7 baby, 7 giant, 12 copy) +
  2 × 12 (`block_allreduce` in the update) + 33 (`untranspose_lane0`: 12 sum, 14 baby, 7 giant) + 1 (the Horner
  step) ≈ **105**. That is 3,360 per 32-token pass against the census's 3,396 rotations plus 98 conjugations for
  the whole block (§3);
- 8 ct × ct products in the update (4 per state ciphertext), plus 2 for the readout.

**Depth on the S chain: 3 levels per token** (`S⊙κ̂`, the block-start mask, `·bb`; the adds use none). The
broadcasts (1 level) and the transpose mask hang off the token-parallel values, not off S. So the refresh count of
the scan grows linearly with T. The plans put it at about 1 extra refresh per token in block 0 and about 2 in
block 1 (§3).

### (b) Chunked / WY form (as in delta-rule chunk kernels): rejected

The GPU chunk kernels (fla's `chunk_rwkv7`/DPLR) build, per chunk of C tokens, C × C intra-chunk matrices. They
invert `(I + L)` (strictly lower) and apply the inter-chunk state once. Under CKKS this costs three things:
1. **The decay is per channel j and data-dependent.** The intra-chunk factor between tokens s < t is
   `Π_{s<u≤t} w_u[j]`, a C × C × 64 tensor per head. The kernels factor it as `W_t / W_s` with
   `W_t = Π_{u≤t} w_u`. That needs `1/W_s`, and since w ∈ (0.545, 1) (`exp(−e^{−0.5})` is the floor), over 32
   tokens 1/W reaches about 2.7·10⁸. No polynomial interval or CKKS scale holds that. Without the division, every
   (s, t) pair is its own product chain.
2. **The triangular inverse** by Neumann doubling, `Π_k (I + L^{2^k})`: log₂ C = 5 sequential C × C products per
   head, about 10 levels. Each is C rotations plus C products on this layout.
3. **It does not remove the S-dependent per-token work.** The inter-chunk term still applies S_0 to every token's
   `r_t` and `κ̂_t`. On this layout that is one `untranspose` per token: the same rotations the scan's readout costs.

So the WY form keeps the scan's dominant per-token term and adds the C² work, the inverse and an impossible
division. Its advantage on a GPU is matmul density (tensor cores), and CKKS has nothing like it.

### (c) Hybrid: what (a) already is

Everything that does not depend on S runs token-parallel: all of §1, plus `κ̂`, `k̃`, `κ̂⊙a`, the decay and the
gates. The scan does only the S-dependent work. A further hybrid step would be the complex packing of two tokens per
transpose (§5).

## 3. Measured on the CPU (the records)

All numbers here come from `tests/test_prefill.py`. The run commands are in the README.

**Correctness** (numpy mirror and fake, 2 blocks of the real checkpoint, after the public prefix `"\n\n"`):

| check | result |
|---|---|
| mirror: a T-token pass vs T decode steps, then 2 decode steps after the hand-over; T = 32, 7, 1 at 2^15 slots (the device's t = 32) and T = 16 at 2^14 | hidden states, carried state and post-hand-over decode **bit-identical** (rel. 0.0) |
| the same, with the chunk reversed (non-vacuity) | differs (> 1e-3): the lanes are real |
| fake (runtime level rules, PCMM fake, canonical carry) and capture recorder: prefix → 6-token pass → 3 decode steps, vs 9 decode steps | prefill-vs-sequential 0.0; **top-1 9/9** for both against the poly model (clamp on); prefill-vs-poly hidden relErrRms 4.1e-6 … 4.5e-5 |

**Why bit-identical, and why that is not a red flag here.** Every op the pass adds is exact in float64: masks of 1
and ½, adding zeros, rotations, and `x/2 + x/2`. The fake has no CKKS noise. On the device, the extra key switches
(about 105 per token, conjugations included) add key-switch noise. That noise is small next to a refresh's,
but **unmeasured**.

**Plans** (`tools/fhe_ssm/make_plan.sh`, LANDING=high, PAIR=0; `.cache/rwkv7_pf/bootstrap_placements/plan_T*` and
the decode test's plan; graph op counts from the capture's `block_0/graph.json`):

| capture | block | nodes | rotate | conjugate | mult | refreshes (plan) |
|---|---|---|---|---|---|---|
| decode, 1 token | 0 | 2,232 | 209 | 2 | 788 | 31 |
| prefill T = 8 | 0 | 5,612 | 953 | 26 | 1,799 | 38 |
| prefill T = 32 | 0 | 16,868 | 3,396 | 98 | 5,230 | 62 |
| prefill T = 32 (2-block plan) | 1 | – | – | – | – | 100 |
| decode, 12 blocks (P1, `test_full_plan`) | all | – | – | – | – | 478 per token (13 stages) |

Per token, at T = 32:
- block 0: 62/32 ≈ 1.9 refreshes against 31 for a decode step (**16×**);
- block 1: 100/32 ≈ 3.1 against ≈ 40.5 (the decode average of blocks 1–11: (478 − 31 − 1)/11) (**13×**).

The scan's own refreshes are the difference: about 31 per pass in block 0 and about 60 in block 1. The plan does
not say why block 1 needs twice as many (larger state magnitudes are the likely cause).

Planner wall time: 5 s for T = 8 (1 block), 35 s for T = 32 (1 block), 161 s for T = 32 (2 blocks). 12 blocks will
take about 15 min on the box.

## 4. Expected seconds per prompt token (an estimate, not a measurement)

**Inputs.**
- Op counts: block 0's census × 12 for the rotations, products and adds; the plans' refreshes,
  62 + 11 × 100 + 1 = **1,163 per pass**.
- Relative op costs: Perseus's published per-op times (RTX PRO 6000, logN 16, 32-bit: refresh 28.77 ms, rotation
  214 µs, ct × ct 203 µs, ct × pt 72 µs, add 33 µs; quoted in
  `results/perseus-20261006/fork_port/gpu_20261011/prefill_speed_20261024/FACTCHECK_ROUTES_1_4.md:182`).

**R6.** Those times come from another card and another build. They serve **only as ratios**. The scale comes from
this port's own decode, 13.05 s per token on the 300 W card. A `mult` node is priced at both the ct × pt and the
ct × ct cost, which gives a range:

| | refreshes | rotations + conj. | mult | adds | sum (published units) | on the 300 W card |
|---|---|---|---|---|---|---|
| decode token | 478 → 13.75 s | 2,532 → 0.54 s | 9,456 → 0.68–1.92 s | 11,988 → 0.40 s | 15.4–16.6 s | **13.05 s** (measured) → scale 0.79–0.85 |
| prefill pass, T = 32 | 1,163 → 33.5 s | 41,928 → 9.0 s | 62,760 → 4.5–12.7 s | 86,964 → 2.9 s | 49.8–58.0 s | **≈ 42–46 s per pass** |

**≈ 1.3–1.4 s per prompt token, about 9–10× faster than decoding the prompt** (13.05 s). A 64-token prompt is two
passes, ≈ 1.5 min instead of ≈ 14 min.

**Why not 13–16×, as the refresh counts suggest.** In decode, refreshes are 83–89 % of the priced time. In a pass
they are 58–67 %, because the scan's rotations (about 105 per token per block) and products now weigh almost as
much. The next levers attack those (§5).

T = 32 is the best chunk: the token-parallel part costs one decode step's refreshes per pass whatever T is, so it
amortises over T (T = 8 gives 38/8 = 4.75 refreshes per token in block 0, against 1.9 at T = 32). T cannot exceed
the 32 lanes. Longer prompts take several passes; each pass reads the previous pass's state from the registers.

## 5. Not built: the next prefill levers, by expected gain

1. **Two tokens per transpose / untranspose (complex packing).** `transpose(v_t + i·v_{t+1})` once, and split the
   two V's with one conjugation. Likewise `untranspose(P_t + i·P_{t+1})`. Half of 26 + 33 rotations per token,
   less two conjugations, saves about 27 of the 105 (−25 % of the scan's key switches).
2. **Refresh pairs in the scan.** S_0 and S_1 are refreshed together, at one level, both real: exactly the siblings
   PAIR=1 makes into one complex refresh (§6). In block 1 that could halve about 60 refreshes per pass.
3. **Hoisting.** The transpose's 7 baby rotations already share one input. The three broadcast sources rotate by
   +1 each token, chained (one key, a little key-switch noise per step). A hoisted version, rotating by t from the
   source with one decomposition per source, needs 31 more keys.

## 6. The fhe_ssm stack's levers (`cfg-best-em`): what transfers

| lever | transfers? | why | how to enable here |
|---|---|---|---|
| **LANDING=high** | already on | PCMM needs the StC-first dense route | (default) |
| **PAIR=1** | **yes** | The compiler pairs two sibling refreshes of one level and degree on a *real* route (`crates/pc-program/src/program.rs:278`, `Pairing::takes`: `r.real && r.stc_first && …`). The complex `z = h + i·v_first` never takes a real route, so it is never paired; the four carried state ciphertexts at block end, and S_0 / S_1 in the scan, are real siblings. | `LEVERS=best` (plan env `PAIR=1`), run with `--schedule siblings` |
| **siblings schedule** | **yes** | A compile-time batching of sibling ops; pairs only form under it | `run_planned.py --schedule siblings` (new: `_compile_with_schedule` passes it to `Program.compile`) |
| **fused EvalMod** (`PC_EM_FUSE_RESCALE=1`) | **yes** | A runtime kernel switch inside every refresh; it binds no plan and does not depend on the model | `LEVERS=best` (plan env `RUNTIME_EXTRA=PC_EM_FUSE_RESCALE=1`, on the plan's `runtime=` line) |
| **per-site PCMM CF table** | mechanically yes; the gain is doubtful | The table builds from this capture and covers every linear (`tests/test_levers.py`). But this port's band shifts already scale every PCMM output to ≤ 0.25/1.5 before its refresh, so the per-site rule collapses to CF 2–3 everywhere (one linear at 3): in effect a **global** move from the CF 6 the box run used. The rule's offset was calibrated on fhe_ssm at shift 0, so the accuracy of CF 2–3 here is untested. | `CFTABLE=1` (`--pcmm-cf-table`); A/B it against `PCMM_CF=6` on the 1-block smoke before any full run |
| **pack-stats** | **no** | It runs several norm statistics through one Newton chain. Here no two statistics are ever ready at the same point of the dependency graph: LN1 → κ̂ norm → GroupNorm → LN2 is a chain within a block, and blocks are sequential. In a prefill pass the T tokens' statistics already share one chain (one lane each). | not built |

## 7. Device failure points, most likely first

1. **The carried state lands differently in the two plans.** `Program.run` refuses a register whose
   (level, deg) differs from the compiled input's (`crates/pc-py/src/program.rs:692`). A pass writes S_0, S_1, xa
   and xf where the decode step reads them. `Rwkv7PrefillRun.prepare` compares the four metas, both ways, for every
   block **before the session opens** and stops with a named list. Both plans end each block with the same
   deliberate carry refresh under the same landing rule, so they should agree. Run `run_box.sh pfcheck` (1 block,
   T = 4) first. If they differ, re-plan the prefill capture with the decode plan's landing pinned.
2. **Pool and memory.** A pass keeps more ciphertexts live than a decode step: the three broadcast sources, S, Q,
   the token-parallel values and the transpose babies. `run_pf` uses `--pool 192` and `--stc-cache 160`. If the
   pool runs out, raise `POOL_PF`.
3. **Relabelled programs in one session.** The prefill programs are compiled from their own plan directory, then
   `relabel`led `pf_block_<b>` before `bind` (`Program.relabel`, which StreamedPlannedRun uses for the same
   purpose). Two programs per block share one plaintext store under content names. Untested on a device.
4. **Key-switch noise of the chained rotations** (+1 per token on the three broadcast sources, up to 31 in a chain,
   and the Horner accumulator's) and of the conjugations. Watch `hidden_relErrRms_vs_mirror` by lane: growth with the lane index points here.
5. **Refresh precision in block 1's scan.** It is the most refresh-dense stage (100 per pass). If the error jumps
   there, try `PLAN_MAG_SAFETY=3` on the prefill plan.
6. **Compile time and size.** 12 blocks × about 17,000 nodes per block. The twin records two probe passes per
   block (about 10 s per block-pass on this Mac), and setup is untimed.
