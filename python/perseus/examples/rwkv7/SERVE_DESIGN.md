# RWKV-7 continuous batching under CKKS: prefill, then join the batch (design, 2026-11-05)

**Goal.** Serve a stream of conversations on one card: read each new prompt fast in single-conversation mode
(prefill), then generate in the K-lane batched pass, and give a lane back when its conversation ends.

**Status.**
- Built: `serve.py` (the join model, the CPU servers, the scheduler `serve_loop`), the device half in
  `run_planned.py` (`Rwkv7JoinRun`, `Rwkv7ServeRun`, `DeviceServer`, `--serve FILE`), `run_capture.py --join`,
  the `run_box.sh` steps `capture_j | plan_j | scheck | serve | s`, and `tools/rwkv7/serve_arrivals.txt`.
- Validated on the CPU (mirror, fake, capture recorder; §6).
- Planned on the CPU: the join program at 1, 2 and 12 blocks, K = 4 and 32 (§3).
- **Nothing has run on a GPU.** Every second in this file is an estimate (§5).

## 1. What a lane holds, and why the two hand-overs differ

The batched pass (`BATCHED_DESIGN.md` §1) keeps two kinds of carried state per block:

| state | batched layout | single-conversation layout (decode, prefill) | hand-over |
|---|---|---|---|
| WKV state `S_0, S_1` | conversation c's **own** ciphertexts, in decode's layout (registers `s0c<c>`, `s1c<c>`) | the same layout, the same ciphertexts | **register map**: no arithmetic |
| token-shift carries `xa`, `xf` | **lane form**: ONE ciphertext per carry, conversation c's value on lane c (`--batch-carry lane`) | **natural form**: the value on every lane | **lane c overwritten**: one small program |

**Why S needs no arithmetic.** A batched pass updates conversation c's state with decode's own `state_step` on
conversation c's own ciphertexts (`BATCHED_DESIGN.md` §1.1). So the state a prefill pass or a decode step leaves is,
bit for bit, the state the batched pass reads for that conversation. On the device the hand-over is a choice of
register: the single-conversation programs read and write S directly in the registers of the lane the conversation will
join (`Rwkv7ServeRun.bind_lane`). `Program.run` takes the output registers per call, so this is free.

**Why the carries need one.** In lane form the carry ciphertext is shared by all K conversations. Writing the new
conversation's carry must change lane c and leave the other K − 1 lanes untouched:

    xa ← xa ⊙ (1 − m_c) + xa_new ⊙ m_c          (m_c = 1 on lane c's slots, 0 elsewhere)

`xa_new` is in natural form (the same value on every lane), so the mask picks exactly lane c's copy. For c = K − 1 the
mask also covers the lanes past K, which mirror lane K − 1 in the batched pass (`BATCHED_DESIGN.md` §1.2: their carries
equal lane K − 1's by induction; the join keeps the induction true).

**Why the join costs a refresh.** A plaintext product consumes one level: CKKS multiplies the scales, and the rescale
that brings the scale back drops one prime (two primes per level unit at `PLAN_LEVEL_UNIT=2`). The result therefore sits
one level deeper than where the batched pass reads its carries, and `Program.run` refuses an input whose (level, deg)
differs from the compiled one (`crates/pc-py/src/program.rs`, `run_block`). One deliberate refresh per carry lands it
back, under the same rule as the batched pass's own carry refresh (a deliberate refresh in a `carry` scope).

## 2. The serving loop (`serve.serve_loop`, shared by the CPU tests and the device runner)

- **Clock:** the batched pass index q. FILE rows give each conversation's arrival pass.
- **Admission:** at the start of pass q, every conversation that has arrived takes a free lane, in arrival order, while
  one is free. Its prompt (n tokens after the public prefix `"\n\n"`) is read in single-conversation mode **while the
  batch waits**: ⌊n/T⌋ prefill passes, then n mod T decode steps. The first single program reads the public prefix
  state; the rest read what the previous one left. The client's pick from the last prompt position is the first
  generated token. Then the join.
- **Batched pass:** every lane advances one token; conversation c's pick goes back to lane c.
- **Leave:** a conversation that has fed `--generate` tokens to the batch leaves after that pass. Its records are its n
  prompt positions plus `--generate`, as the decode runner's.
- **Idle lanes** run from the public prefix state on a public token (the poly model's pick after the prefix), then on
  their own picks. They are never scored. They must stay in range: on the K = 32 device run one lane's out-of-interval
  value broke all 32 lanes at pass 18, because the refreshes and lane-mixing ops share the ciphertext
  (`results/rwkv7-fhe-20261104/new_pod/RESULTS.md`, "Fidelity collapsed at pass 18").
- **Empty batch:** no pass is spent. The clock jumps to the next arrival.

## 3. The join and leave programs

**The join program** (`run_capture --join --batch K`; its twin is `serve.Rwkv7Join`):
- one stage;
- inputs: every block's lane-form carries (`b<b>xa`, `b<b>xf`, + shards) and one conversation's natural-form carries
  (`b<b>nxa`, `b<b>nxf`), all named as carried state (`cache.<key>-lvl=<level>`), so the planner and the compile seed
  them at the carried landing;
- outputs: the joined lane-form carries;
- the two masks are plaintext **inputs**: `Rwkv7JoinRun` stores them under their own names (`join/m`, `join/keep`)
  and re-stages them in place before each run (`PtStore.stage` re-encodes a resident key at its own slot, after
  every earlier reader in stream order, so the bound program's CUDA graph stays valid). One program serves every lane.
  They are NOT aliased by value digest as the other programs' plaintexts are: the capture's lane-0 mask equals the
  batched pass's own lane-0 mask, and re-staging a shared slot would corrupt it.

**The planner's counts** (`tests/test_serve.py::test_join_capture_and_plan`, `make_plan.sh` LANDING=high, PAIR=0;
products in `.cache/rwkv7_serve/bootstrap_placements/plan_join*`):

| capture | nodes | mult (ct × pt) | add | deliberate refreshes | placed refreshes | rotation keys |
|---|---|---|---|---|---|---|
| K = 4, 1 block | 8 | 4 | 2 | 2 | 0 | 0 |
| K = 4, 2 blocks | 16 | 8 | 4 | 4 | 0 | 0 |
| K = 4, 12 blocks (0.1B) | 96 | 48 | 24 | **24** | 0 | 0 |
| K = 32, 12 blocks | 96 | 48 | 24 | 24 | 0 | 0 |

**Levels** (the planner's convention: levels count primes used, `final_named_levels` in
`plan_join4_b1/block_0_placement.json`, node levels in the capture's `graph.json`): every input sits at 26 (the
carried landing; the batched capture's carries are `cache.xa-lvl=26` too), each product at 28, the add at 28, the
refresh lands at 26. K changes nothing but the mask values.

**The leave** reuses the join program: lane c's S inputs switch back to an encryption of the public prefix state (no
program: `Program.run` copies an input ciphertext into its slab, so the batched pass simply reads the encryption
instead of the register), and the carries take the prefix carry through the join program. Two refinements:
- every lane that leaves after one pass is reset in **one** run (the mask is the union of their lanes; the prefix carry
  is the same for all);
- a lane re-used before the next pass skips its reset (the join overwrites both parts anyway).

## 4. One session, four program sets (`Rwkv7ServeRun`)

`prepare` (CPU, before any session):
1. compiles decode, prefill (`Rwkv7PrefillRun`, as P3 built it), batched and join from their own plans, and relabels
   them `block_<b>`, `pf_block_<b>`, `b_block_<b>`, `join_block_0`;
2. **registers:** the batched registers (every lane's S, the lane-form carries, the stage outputs), the single carries
   and the decode and prefill stage outputs. The decode plan's own S registers are dropped (the single programs use the
   lanes'), and everything is numbered densely. `register_levels` sees the single programs' S writes into **every**
   lane's registers and the join's writes into the carry registers;
3. **landings, every hand-over direction, by name** (`check_landings`; a mismatch stops `prepare` with the list):
   decode / prefill S out → batched S in (each lane); batched S and carry out → batched in; decode / prefill carry out
   → join natural-form in; batched carry out → join lane-form in; join out → batched in; join out → join lane-form in
   (two joins before a pass). The decode ↔ prefill directions are `Rwkv7PrefillRun`'s own check;
4. a join captured for another block count or width is refused by name.

`DeviceServer` (the device side of `serve_loop`):
- `src` records, per batched state input, whether it comes from a register or from an encryption of the prefix state;
- **ONE encryption per (block, state key, compiled meta)** serves every reader: the first single pass, every lane's
  S, the start's lane-form carries (all lanes equal) and the leave's natural-form carry. That is 4 × 12 = 48 ciphertexts
  (≈ 0.7 GB at ≈ 14.2 MB each, `BATCHED_DESIGN.md` §4) instead of (4 + 4 + 2K + 2 + 2) × 12;
- `TeeServer` runs the numpy mirror on the same schedule, so every record carries `hidden_relErrRms_vs_mirror`.

## 5. What a serve pass costs over a plain batched pass (an estimate, not a measurement)

**Units** (the method of `BATCHED_DESIGN.md` §4, R6: published per-op times as ratios only, scaled by this port's
measured decode on the 600 W card, 9.54 s per token, scale 0.574–0.621):
- **one join or leave run:** 24 × 28.77 ms (refresh) + 48 × 0.072 ms (ct × pt) + 24 × 0.033 ms (add) = 694.7 ms
  published, so **≈ 0.40–0.43 s**, plus re-staging two plaintexts;
- **the batched pass itself is unchanged:** the same program. Measured: 46.68 s at K = 32
  (`results/rwkv7-fhe-20261104/new_pod/RESULTS.md`); estimated 13.8–15.3 s at K = 4 (`BATCHED_DESIGN.md` §4).

**Per pass:** at most one leave run, plus one join per admitted conversation: 0.9 % of a K = 32 pass per run, 2.6–3.1 %
of a K = 4 pass. The join is not where the time goes.

**Per arrival, the batch waits for the prompt:**
- each prefill pass: 49.8–58.0 published s for 12 blocks at T = 32 (`PREFILL_DESIGN.md` §4), i.e. **≈ 30.9–33.3 s** on
  the 600 W card (ct × pt pricing with scale 0.621, ct × ct with 0.574). Never measured on a device;
- each remainder token: one decode step, **9.54 s** (measured).

**The default arrivals file** (`tools/rwkv7/serve_arrivals.txt`, 8 conversations of 5–45 prompt tokens, 8 generated
each, B_K = 4, T = 32; the schedule from `serve_loop` on a stand-in server):

| part | count | seconds (estimate) |
|---|---|---|
| prefill passes | 3 | 93–100 |
| remainder decode steps | 88 | 839 |
| joins | 8 | 3.2–3.4 |
| leave runs | 1 | 0.4 |
| batched passes (K = 4, 3.76 active lanes on average) | 17 | 235–260 |
| total | | **≈ 1,170–1,200 s (≈ 20 min) for 64 generated tokens** |

**What it says.** The remainder rule dominates: 88 decode steps are ≈ 70–72 % of the loop. The join and leave are ≈ 0.3 %.
The batched pass's throughput is unchanged; what continuous batching adds is admission latency for the waiting batch.

**Levers (not built), by expected gain:**
1. **A second prefill plan for remainders** (e.g. T = 8): a remainder of r tokens becomes ⌊r/8⌋ short passes plus at
   most 7 decode steps. A prefill pass cannot simply run with fewer tokens: the scan updates the state once per lane,
   so padding lanes would advance it. The cost of a T = 8 pass on the device is unknown (planner: 38 refreshes in
   block 0 against 62 at T = 32, `PREFILL_DESIGN.md` §3).
2. **Several joins in one run:** `xa ← xa ⊙ (1 − Σ m_i) + Σ_i xa_new,i ⊙ m_i` takes n arrivals' carries in one program
   with one refresh per carry. It needs each arrival's natural carries in their own registers until the join. It saves
   ≈ 0.4 s per extra arrival per pass: small.
3. **Lockstep prefill and batch** (prefill the next arrival while the batch runs) buys nothing on one card: the GPU runs
   one program at a time.

**Memory.** The session holds the batched programs plus the decode and prefill programs and their rotation keys
(prefill adds ±1 and ±31, the batched pass +1 and −(K − 1)). Extra registers over a batched session: the single
carries (24) and the decode and prefill stage outputs (13 + 13), ≈ 50 × 14.2 MB ≈ 0.7 GB. The K = 32 batched session
alone peaked at **94.9 GB of the 97.9 GB card** (`RESULTS.md`), so **serving at K = 32 is not expected to fit**. Start
with K = 4. How much the replay slabs of three program sets add is unknown.

## 6. Validated on the CPU (the records)

All from `tests/test_serve.py` and `tests/test_serve_session.py`. 2 blocks of the real checkpoint, 2^15 slots (t = 32
lanes), after the public prefix `"\n\n"`. The schedule: 5 conversations arriving at passes 0, 0, 0, 1, 9, with prompts of
5, 3, 8, 4, 6 tokens, each generating 2. It exercises a queue (conversation 3 arrives while every lane is busy), lane
reuse, a two-lane leave in one run, idle lanes after a leave, and a clock jump.

| check | result |
|---|---|
| mirror, K = 3 / T = 4 (prefill + decode remainder), K = 2 / T = 1 (prefill only, 26 one-token passes), K = 3 / T = 0 (decode only): every conversation at every position vs itself decoded alone from the prefix | hidden states **0.0** (bit-identical), token streams identical; 5 joins, 1 leave run (two lanes), 6 batched passes |
| the same: every idle segment of every lane vs a fresh conversation from the prefix on the idle token | **0.0** (segments of 4, 4 / 2 / 4, 4 passes) |
| non-vacuity: two conversations at the same position | differ (> 1e-2) |
| non-vacuity: the join onto the wrong lane (mask of lane c + 1) | breaks the equality (> 1e-3) |
| non-vacuity: a leave that does not reset | breaks the idle check (> 1e-3) |
| fake (runtime level rules, PCMM fake, every carry and join landed by a refresh), K = 3, T = 4, vs each conversation decoded alone on a fresh fake | **0.0**; **top-1 36/36** against each conversation's own poly model (clamp on); hidden vs poly ≤ 4.5e-5 |
| capture recorder, the same | **0.0**; **top-1 36/36** |
| join capture + plan (`--rwkv7-plan`) | per carry 2 products, 1 add, 1 deliberate refresh; plans as §3 |
| `Rwkv7ServeRun.prepare` on stand-in programs | S of the single programs = lane c's registers for every c; single carries their own; dense numbering (14 registers at K = 2, 1 block); 7 programs; a mismatch named in each of four directions; a join for another block count refused |
| `DeviceServer` on stand-ins recording each input's last writer (provenance) | prefill writes lane 1's S, the decode step reads it there, the join reads the start's prefix carry and the decode step's carry, the batched pass reads lane 1's S from the decode step and lane 0's from the prefix encryption, a leave puts the prefix back; 4 prefix encryptions serve every reader. Removing the join's S switch fails it |
| `Rwkv7JoinRun.prepare` on the recorder (compile stubbed) | every input a named carried value at one recorded meta; outputs in key order; both masks found by value |
| `run_planned --serve --keyspec` end to end (compile and key source stubbed; K = 2, T = 2, 1 block) | the four twins record, prepare merges and checks the landings, 7 programs, 14 registers |
| real plans compiled by `Program.compile` (`test_serve_session_prepare`) | **skipped here** (the CPU build has no `Program`); `run_box.sh scheck` |

**Why bit-identical is not a red flag.** The join's ops are exact in float64: products by 0 and 1, and adding a
zero. The fake has no CKKS noise. The non-vacuity checks show the equality is not trivially true.

## 7. Box commands

```bash
cd /root/perseus-rwkv7
B_K=4 bash tools/rwkv7/run_box.sh scheck                     # once: the serve session's CPU half (1 block, T = 4, K = 4)
BLOCKS=1 B_K=4 S_GENERATE=2 bash tools/rwkv7/run_box.sh s    # smoke: every capture and plan, then serve
B_K=4 bash tools/rwkv7/run_box.sh s                          # 12 blocks, K = 4, T = 32, 8 generated each
LEVERS=best B_K=4 bash tools/rwkv7/run_box.sh s              # on PAIR=1 + fused-EvalMod plans, siblings schedule
KEYMODE=s B_K=4 bash tools/rwkv7/run_box.sh keyspec && KEYMODE=s B_K=4 bash tools/rwkv7/run_box.sh keygen
KEYS=bundle B_K=4 bash tools/rwkv7/run_box.sh serve          # the client's keys
```

`s` runs capture, plan, capture_pf, plan_pf, capture_b, plan_b, capture_j, plan_j, serve; each step reuses what an
earlier one left. Products: `$OUT/serve4[_b1][_best].jsonl` (events with seconds, one token line per conversation and
position, the summary) beside its log.

## 8. Device failure points, most likely first

1. **Memory.** Four program sets in one session; the K = 32 batched session alone peaked at 94.9 GB. Run K = 4 first.
   If it runs out, lower `STC_CACHE_S`, then use 1 block. Watch the header's `mem_report_text`.
2. **A landing mismatch at `prepare`.** Four plans from four captures must agree on where every carried value lands.
   The captures agree (every carried input is `-lvl=26`), and every plan lands deliberate refreshes by one rule. The
   join's inputs are compiled at the recorder's metas for **all** inputs (not only the carried state's, as the
   other runs do), which is new. `prepare` names the direction; run `scheck` first.
3. **Re-staging the masks in place.** The first use of `PtStore.stage` on a resident key in this port. Symptom: the
   lane joined second onward starts wrong (its first batched token's `hidden_relErrRms_vs_mirror` jumps, other lanes
   fine). Fallback: K compiled join programs with baked masks.
4. **Pool exhaustion.** The K = 32 batched run needed `POOL_B=1024 STC_CACHE_B=64`. Serve uses `POOL_S` (256); raise it.
5. **One lane's out-of-interval value spoils all lanes** (measured, K = 32, pass 18). Idle lanes are kept in range, but a
   conversation's own text can still leave an interval. The client-side lane health check is held (`PROJECT_BACKLOG.md`).
6. **The register table:** single programs write into every lane's S registers through `bind_lane`; `reg_levels` covers
   those writes. A miss shows as a register-size error on the first prefill of a lane other than 0.
7. **Setup time:** four twins record their probes (decode, T = 32 prefill, K-lane batched, join) before the session
   opens. Untimed; expect minutes at 12 blocks.

## 9. Not in scope

A client-side health check or lane reset on bad logits (the owner holds it; `PROJECT_BACKLOG.md`).
