# RWKV-7 batched decode: the per-conversation update refreshes were a placer artefact (PLANOPT=b2), 2026-11-05

**Result.** With one planner switch, `FORBID_STEPS` on the three per-conversation step scopes, every conversation pays
**2 refreshes per block (its two state carries) in every block**, instead of 6 / 4 (block 0 / block 1) on the plan
that ran on the device. The circuit is unchanged: same capture, same fake, same recorder. Only the plan moves.
- On the capture whose plan ran on the device (K = 32): block 0 goes **209 → 80** and block 1 **153 → 88** in
  `Program.compile`'s count. That count reproduces the device's `n_bootstraps` exactly (§2).
- Estimated 12-block pass on the final coefficients with LEVERS=best: **≈ 28.3–28.5 s per pass against the measured
  29.87 s** of the base plan (≈ 1.12–1.13 tokens/s against 1.07; ≈ 1.05×; 26.5–26.8 s only if a paired refresh costs
  two singles). This is an estimate, §5, re-anchored after review lens 3 (findings 10, 11): the first estimate here
  (≈ 28.4–30.0 s "instead of the measured 46.68 s") credited b2 with the whole drop from 46.68 s, which came mostly
  from the final coefficients' capture and from pairing.
- On the box: `PLANOPT=b2 bash tools/rwkv7/run_box.sh b` (§6).

Every number below is from the files named beside it. Plans and captures are under `.cache/rwkv7_b2plan/`
(gitignored, rebuilt by the commands in §8).

## 1. The mechanism

**Levels.** A level counts primes used: 0 is fresh and 50 is the top of the n32 chain. `PLAN_LEVEL_UNIT=2`, so one
multiplicative level is 2 units. A product's output sits at max(operand levels) + 2, and an add sits at the max of
its operands. A dense StC-first refresh must start at or below 46 (`PLAN_STC_FIRST_ENTRY=46`,
`PLAN_REFRESH_ENV_CAP_ROUTES=0:46`, `tools/fhe_ssm/make_plan.sh` LANDING=high). It lands at 26: the compile check
reports `dense_stcf@26`, and the capture names every carried input `cache.s0c<c>-lvl=26`.

**One conversation's update** (`batched.py` `wkv_shard`, `model.py` `state_step`):
- `bcast`: lane c of each of the three shared sources (`w + i·κ̂`, `(κ̂⊙a) + i·k̃`, `r + i·v`) is brought to lane 0,
  masked (+2) and copied to every lane;
- `update`: `S⊙κ̂` → block-start mask → `·bb`, plus `S⊙w` and `V·k̃`, giving S′. The readout is `S′⊙r`;
- `readout`: `untranspose_lane0` (+2) and the Horner sum `Q ← rot(Q, 1) + acc_c` over the K conversations.

**What sets the depth of that chain is the broadcast vectors, not S.** S enters at 26, the shallowest operand of
every product it is in. Trace, old capture, block 0, conversation 5, on the base plan
(`final_named_levels` of `plan_old_K32_b2_base`):
- the broadcast w and κ̂ sit at plan level 36;
- so `S⊙κ̂` is at 36, the mask at 38, `·bb` at 40, `S′⊙r` at 42, and the readout at 44.

So brief item 1 (**land the carried state higher**) cannot help. A landing below 26 would not move a single product,
because max(S, broadcast) = broadcast. The landing is set by the bootstrap itself, not the planner:
- the runtime's landing table (`crates/pc-program/src/compile/mod.rs`, `bts.landings`);
- the StC-first recipe (`FIDESLIB_BTS_STC_FIRST=2`, `STC_FIRST_CF=13`).

Raising it would need a shallower EvalMod (precision) or a longer chain (log QP, security). It is not a plan switch,
and it is not binding.

**Why the base plan refreshed inside each conversation anyway.**
- The min-cut placer works in passes on over-budget ops, and wires only each op's binding operand
  (`placer/planner.py`, the P4 note).
- On the device's capture it ended with these placements:
  - block 0: 4 per conversation, on `S_m⊙w` and on the `block_allreduce` output;
  - block 1: 2 per conversation;
  - **and** the four GroupNorm (`tm.gn`) refreshes at places that assume the readout leaves shallow (level 36).
- The prune (`_prune_redundant`) then removes one refresh at a time. It keeps a removal only if the block stays within
  budget and the count drops. Removing **one** conversation's refresh fails that test:
  - the Horner sum Q is an add, so its level is the deepest conversation's;
  - Q then leaves at 44 instead of 36, and the GroupNorm chain, refreshed where it was, goes over budget.
- The cheaper plan needs a **joint** move: drop all K conversations' refreshes together and move the GroupNorm
  refreshes to the readout's exit. That costs the same 4 refreshes. A one-at-a-time prune cannot find it.
- Forbidding the per-conversation scopes makes the cut find it in one go. On the old capture, block 0, the token-side
  placements of the base and b2 plans are **identical**: tm.norm 1, tm.lora 1, tm.kk 6, tm.state.src 1, cm.norm 1.
  Only the four `tm.gn` sites move:
  - base: after the readout leaves at 36;
  - b2: from the GroupNorm's first subtraction (input 44 / 46) on.
- The readout then leaves at 44 and the carry refresh of S′ starts at 40 (base: 32). Both are inside the 46 envelope.

**Brief item 2 (refresh the shared sources)** is what the cut does where it needs to. Every plan refreshes 1–2 of
the three sources once per pass (`tm.state.src` placements). On the final-coefficient capture, block 1 under b2
refreshes one more source than the base plan (2 vs 1) and drops 64 per-conversation refreshes. On the old capture no
extra source refresh was needed at all.

**Brief item 3: block 1's 2 vs block 0's 4**, and why it moves with the coefficients. The number of
per-conversation refreshes follows from which of the three sources is deepest when it reaches the update. That in
turn depends on which source the cut chose to refresh (plan levels of the three `tm.state.src` adds, K = 32):

| capture | block | `w + i·κ̂` | `(κ̂⊙a) + i·k̃` | `r + i·v` | per-conversation placed (base) |
|---|---|---|---|---|---|
| old (P4's, `coeffs_mixed.json`) | 0 | 34 | **refreshed** | 28 | 4 (`S⊙w`, allreduce out, × 2) |
| old | 1 | **refreshed** | **refreshed** | 28 | 2 |
| new (HEAD, `coeffs_final.json`) | 0 | **refreshed** | 32 | 28 | 0 |
| new | 1 | **refreshed** | 40 | 30 | 2 (readout, after `S′⊙r` at 44) |

- These are cut-path effects. They are not magnitudes acting directly: the magnitudes move the sites' depths (the
  final coefficients' constant-seed Newton on the 1/√x sites made `tm.kk` deeper), and the depths move which source
  the cut reaches first.
- `PLAN_MAG_SAFETY=3` leaves every count unchanged (§2).
- Under b2 the question disappears: no block has a per-conversation placement, whatever its sources' levels.

## 2. The counts (K = 32 unless stated; 2-block captures, ring 2^15 = 32 lanes)

Two counts per block:
- **planner** = `summary.total_bootstraps` (placed + deliberate + fired hints). The planner's deliberate count also
  holds the PCMM finishes.
- **compile** = the refresh sites of `Program.compile` (`tools/fhe_ssm/compile_check`, the pc-program crate, CPU
  only), placed + fired hints + deliberate carries.

The compile count is the device's own:
- for the plan that ran, it gives **209 / 153** for blocks 0 / 1;
- the device record's `programs.block_0.n_bootstraps` and `programs.block_1.n_bootstraps` are **209 / 153**
  (`results/rwkv7-fhe-20261104/new_pod/r01b_batch/batch32.jsonl`, header).

"Per conversation per block" is 2 carries plus the per-conversation placements / K.

| capture | plan | block 0: planner / compile | block 1: planner / compile | per-conv placed (b0 / b1) | **per conversation per block** (b0 / b1) | compile, 2 blocks + tail |
|---|---|---|---|---|---|---|
| old, K = 32 (`.cache/rwkv7_b3/cap_K32_lane_b2`, P4's) | base (= the device run's plan) | 222 / **209** | 167 / **153** | 128 / 64 | **6 / 4** | 363 |
| old, K = 32 | **b2** | 93 / **80** | 102 / **88** | 0 / 0 | **2 / 2** | 169 |
| new, K = 32 (`cap_K32_lane_b2`, `coeffs_final.json`) | base | 100 / 87 | 166 / 152 | 0 / 64 | 2 / 4 | 240 |
| new, K = 32 | **b2** | 100 / 87 | 103 / 89 | 0 / 0 | **2 / 2** | 177 |
| new, K = 32 | base, `PLAN_MAG_SAFETY=3` | 100 / 87 | 166 / 152 | 0 / 64 | 2 / 4 | 240 |
| new, K = 32 | b2, `PLAN_MAG_SAFETY=3` | 100 / 87 | 103 / 89 | 0 / 0 | 2 / 2 | 177 |
| new, K = 8 (`cap_K8_lane_b2`) | base | 52 / 39 | 70 / 56 | 0 / 16 | 2 / 4 | 96 |
| new, K = 8 | **b2** | 52 / 39 | 56 / 42 | 0 / 0 | **2 / 2** | 82 |

**Per pass under b2.** Each block is a fixed token-side part plus 2K carries:
- block 0: planner 52 at K = 8 and 100 at K = 32, so 36 + 2.0·K;
- block 1: planner 56 and 103, so 40 + 1.96·K (one fired hint fewer at K = 32).

**With `LEVERS=best`** (PAIR=1, fused EvalMod, siblings schedule; the compile check pairs as `Program.compile` does):

| capture | plan | compile sites (b0 / b1) | refresh pairs (b0 / b1) |
|---|---|---|---|
| new, K = 32 | base + best | 87 / 152 | 1 / 33 |
| new, K = 32 | **b2 + best** | 87 / 89 | 1 / 1 |

Pairing is visible on the CPU, so this corrects `BATCHED_DESIGN.md` §5 (the PAIR row) on two points:
- **The carried states do not pair.** S is a complex payload: route `dense_stcf`, not `dense_stcf_real`. `PC_BTS_PAIR`
  pairs only real-payload StC-first siblings. Block 0 (66 carries + 21 placed) gives 1 pair.
- **The per-conversation refreshes do pair.** Block 1 has 33 pairs on the base plan and 1 under b2, where exactly its
  64 per-conversation refreshes are gone: 32 pairs, all of them. So on the base plan
  `LEVERS=best` would have halved them. b2 removes them instead, so `LEVERS=best` adds ~1 pair per block on top of b2.

**Decode and prefill** (2-block captures, `coeffs_final.json`). b2 does not touch them: they have none of the batched
scopes, and `test_planopt_b2_keeps_the_planners_default_forbidden_steps` checks this. The wider experiment
`PLANOPT=st` forbids all of `tm.state`:

| capture | base: planner b0 / b1, compile total | st: planner b0 / b1, compile total |
|---|---|---|
| decode | 38 / 42, 54 | 38 / 41, 53 (block 1's two `tm.state` placements become one `tm.kk` one) |
| prefill T = 32 | 65 / 88, 127 (`tm.state` 28 / 49 placed: the chain of T updates needs them) | **infeasible** (the planner stops) |

So the run_box variant is batched-only. `st` saves at most one refresh per decode block and is not wired into
run_box.sh.

**Warnings.**
- No plan here has a P3c (drift) warning or `[bts_depth_error]`.
- Every compile check reports 0 check errors and `level richer 0`, and every refresh runs on its StC-first route
  (no fallback to the shipped order).
- The planner's P4 note grows under b2 (block 0: 30 → 8,763 "branches never cuttable"). That note is an info line,
  not a warning: it lists every var that fed an over-budget op in a cut pass but was never a cut candidate. The
  forbidden scopes are never candidates by construction.
- The predicted refresh error per block (planner log, `pred rel_err` median / p90 over the placed sites):
  - old capture, block 0: base 0.00189 / 0.00219, b2 0.00197 / 0.00256;
  - old capture, block 1: base 0.00162 / 0.00212, b2 0.00159 / 0.00214.
  - These are over different site populations: the base includes the per-conversation sites.

## 3. Correctness on the CPU

- **The circuit is the same.** A PLANOPT changes only the planner's environment. Captures, the fake and the recorder
  never read a plan, so the fake still equals K separate decodes (`test_batched.py`, unchanged and passing) for every
  variant.
- **The plan binds.** `compile_check` (`Program.compile`'s own compiler, no CUDA) on every b2 plan: 0 check errors,
  `level richer 0`. The pins match the graph's levels.
- **The new tests** (`test_batched.py`):
  - `test_planopt_b2_keeps_the_planners_default_forbidden_steps`: b2's `FORBID_STEPS` carries `PlanConfig`'s
    default list, and no decode / prefill / join scope matches;
  - `test_batched_plan_b2` (with `--rwkv7-plan`): on a K = 4 one-block capture, 0 per-conversation placements, total
    ≤ the base plan's, the same deliberate refreshes. Result: base 46 (1 per-conversation placement), b2 46.

## 4. What the switch is, and why not a code change

- `FORBID_STEPS` is an existing planner input (`run_bootstrap_all_blocks.sh` → `--forbid-steps`,
  `PlanConfig.forbid_steps`). `make_plan.sh` records it on the plan's `recipe=` line.
- It **replaces** the default list (`.var`, `.mean`, `.qkv`), so `tools/rwkv7/planopt.sh` restates that list, and
  the test above checks it against `PlanConfig`.
- **No edit to `python/perseus/plan/placer/`.** The real fix upstream would be a prune that tries removing a group:
  all K siblings of one scope across conversations, plus a re-cut. That is listed as a possible upstream change, not
  made.
- **One local convenience.** The compile check binary under `.cache/fhe_ssm/compile_check/` (gitignored) is a copy of
  the one built in the `fhe-ssm-speedrun` worktree. Its three path crates (`pc-program`, `pc-config`, `pc-shadow`) do
  not differ between that build's commit `cdd32e2` and this branch: `git diff cdd32e2 HEAD -- crates/...` is empty.
  `make_plan.sh` runs it automatically when present.

## 5. What it should cost (an estimate, not a measurement)

**Re-anchored, 2026-11-05 (review lens 3, findings 10 and 11; supersedes the table below).** The anchor is the
measured final-coefficient pass with LEVERS=best on the BASE plan: 29.8721 s per K = 32 pass
(`new_pod/r01b_final/batch32_best.jsonl`), 1,424 planned refreshes (`r01b_final/plan_b32_best.log`), 1,257 compiled
sites (the record header's `n_bootstraps`). Two things already moved between the old 46.68 s run and that one: the
final coefficients' capture plans nine of the twelve blocks b2-like already (the device shows slope 4 only in blocks
1, 5, 11), and PAIR=1 merges sibling refreshes. b2 removes ≈ 189 sites from blocks 1, 5, 11, but about 96 of them
were paired (base block 1: 33 pairs, b2: 1; the device record 112 pairs), i.e. ≈ 96 refresh launches, not 189. With
the per-op model anchored on the measured pass (scale 0.593; FILLED_DESIGN.md §1): b2 = 29.87 − 3 × (4,872.5 −
3,980.6) published ms × 0.593 = **28.28 s**; the remainder model gives 28.45 s; **b2 ≈ 28.3–28.5 s per pass**
(1.12–1.13 tok/s, ≈ 1.05× the measured pass), 26.5–26.8 s only if a paired refresh costs two singles. b2's own
increment is ≈ 1.4–1.6 s. The old method below, applied to the measured configuration, misses it (46.68 − 806 ×
16.5–17.9 ms = 32.3–33.4 s against 29.87 s) because pairing is unpriced. The ≈ 25.5 s once written for b2 in
new_pod/RESULTS.md is withdrawn there for the same reason.

**The first estimate (old coefficients, old anchor; kept for the record, superseded above).**

**Anchor** (measured):
- 46.68 s per pass, 12 blocks, K = 32, old coefficients (`new_pod/RESULTS.md` § batched; `r01b_batch/batch32.jsonl`);
- `n_bootstraps` per program 209, 153, 153, 216, 153, 215, 154, 150, 217, 148, 147, 147, 1, which sums to **2,063**.

**b2 on the same capture, all 12 blocks.**
- Four blocks carry 4 per-conversation refreshes: 209, 216, 215, 217 = 66 carries + 128 + 15–23 fixed. Eight carry
  2: 147–154 = 66 + 64 + 17–24. That inference holds the device's blocks 2–11 to the Mac's blocks 0 and 1, which are
  exact.
- Remove those placements: 2,063 − (4 × 128 + 8 × 64) = **1,039**. From the Mac's b2 blocks directly:
  80 + 11 × 88 + 1 = **1,049**. So the saving is ≈ 1,014–1,024 refreshes per pass.

**The removed refreshes were single-ciphertext launches.**
- Old capture, block 0: base 209 sites in 153 launches; b2 80 sites in 24 launches. The 129 dropped sites are 129
  dropped launches. The 64 carries run as 8 launches of 8 (`bootstrap_hist` `{"1": 23, "8": 8}`) in both plans.
- So the saving is the same whether the device's cost goes per site or per launch.

**Price** (P4's model, `BATCHED_DESIGN.md` §4):
- refresh 28.77 ms published × this card's scale 0.574–0.621 = **16.5–17.9 ms per refresh**;
- check: 2,063 × 17.2 ms ≈ 35.5 s ≈ 76 % of 46.68 s, which matches P4's "≈ 75 % of a pass is refreshes".

| plan | refreshes per pass (12 blocks, K = 32) | pass | s per conversation-token | aggregate tok/s | vs decode 9.54 s |
|---|---|---|---|---|---|
| base, old coefficients (measured) | 2,063 (device) | **46.68 s** (measured) | 1.46 | 0.6855 (measured) | 6.5× |
| **b2**, old coefficients | ≈ 1,039–1,049 | **≈ 28.4–30.0 s** | 0.89–0.94 | ≈ 1.07–1.13 | ≈ 10–11× |
| **b2**, final coefficients | ≈ 87 + 11 × 89 + 1 = 1,067 | **≈ 28.7–30.5 s** (+18–28 refreshes) | 0.90–0.95 | ≈ 1.05–1.11 | ≈ 10–11× |

- Latency per conversation drops with it: each conversation waits ≈ 29 s per token instead of 46.7 s.
- The per-conversation asymptote B falls from ≈ 1.2–1.6 s to the carries plus the rotations: 24 carry refreshes per
  pass-conversation × 16.5–17.9 ms ≈ 0.40–0.43 s, plus ≈ 1,260 rotations ≈ 0.15–0.17 s.
- **R6:** these are published per-op times on another card and build, used as ratios and scaled to this port's
  measured decode. The final-coefficient row assumes blocks 2–11 look like block 1, as on the old capture.
- **Robustness is unchanged.** The 46.68 s run's fidelity collapse at pass 18 (an `a_sigmoid` site out of its
  interval spreading over the shared ciphertext) is a coefficient matter, and the final coefficients target it.
  A plan cannot fix it.

## 6. On the box

```bash
# the batched run on the b2 plan (own plan dir plan_rwkv7_k32_pob2, records batch_k32_pob2.jsonl, key bundle
# keys_b_k32_pob2; `run_box.sh names` prints them all)
export OUT=... PTH=.../RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth COEFFS=.../coeffs_final.json
KEYS=inproc B_K=32 B_GENERATE=32 POOL_B=1024 STC_CACHE_B=64 PLANOPT=b2 bash tools/rwkv7/run_box.sh b
# with the PAIR + fused-EvalMod levers (b2 leaves ~1 pair per block to take): the brief's form
KEYS=inproc PLANOPT=b2 LEVERS=best bash tools/rwkv7/run_box.sh b
```

`PLANOPT` (`base` | `b2`; `tools/rwkv7/planopt.sh`) changes only these:
- `plan_b` passes the variant's environment to `make_plan.sh`;
- `BPLAN`, the `batch` / `serve` record names and the `b` / `s` key bundle directories gain the suffix `_pob2`, so a
  b2 run never reuses a base plan or overwrites a base record (and, since every setting has its own name token, a
  2-block base smoke no longer shares a name with the full-depth b2 run: `_blk2` vs `_pob2`, review lens 2, finding 1).

Decode, prefill and join plans are not affected. The capture is the same, so `capture_b` is reused.
`tools/rwkv7/b2_plan.sh CAP ROOT NAME [PLANOPT]` re-plans one capture on the CPU and prints its census
(`tools/rwkv7/b2_census.py`).

## 7. What only the device shows

1. **The time per pass**, and so whether the 16.5–17.9 ms price per removed refresh holds. The removed refreshes were
   single launches, so batching does not change the count.
2. **Precision of the deeper carry.** Under b2 the carry refresh of S′ starts at plan level 40 (block 0) instead of
   32, and the readout leaves at 44 instead of 36. The fake has no CKKS noise, so the CPU sees only levels.
   - Watch `hidden_relErrRms_vs_mirror` per conversation against the 46.68 s run's passes 0–17 (logits relErrRms
     median 8.05e-4).
   - If it grows with the pass index, the carry is the suspect.
3. **Blocks 2–11.** The Mac plans two blocks. The 12-block counts come from the device's `programs.block_<b>.n_bootstraps`
   in the record header. Expect ≈ 80–90 per block.
4. **What a pair costs.** The compile check shows that pairs form (§2). Only the device shows whether a paired
   complex refresh costs one real one.
5. Memory: fewer refreshes, and the same registers and pool.

## 8. Reproduce (CPU, ring 2^15, under `tools/memguard.sh 6`)

```bash
# captures (2 blocks; final coefficients)
PYTHONPATH=python $PY -m pytest python/perseus/examples/rwkv7/tests/test_batched.py -k capture_and_plan -s \
  --rwkv7-plan --rwkv7-b-k 32 --rwkv7-b-blocks 2 --rwkv7-b-dir .cache/rwkv7_b2plan --rwkv7-coeffs .cache/rwkv7/coeffs_final.json
# variants (the old capture: .cache/rwkv7_b3/cap_K32_lane_b2, P4's)
PY=$PY bash tools/rwkv7/b2_plan.sh .cache/rwkv7_b2plan/cap_K32_lane_b2 .cache/rwkv7_b2plan plan_K32_b2_fs b2
PY=$PY bash tools/rwkv7/b2_plan.sh .cache/rwkv7_b2plan/cap_K32_lane_b2 .cache/rwkv7_b2plan plan_K32_lane_b2 base
```

Plan dirs here:
- `plan_K32_lane_b2` (base);
- `plan_K32_b2_fs` (b2, made under the variant's first name, `fs`; same environment);
- `plan_K32_b2_{base,fs}_best` and `plan_K32_b2_{base,fs}_mag3`;
- `plan_K8_b2_{base,fs}`;
- `plan_old_K32_b2_{base,fs}`;
- `plan_dec_b2_{base,st}`, `plan_pf32_b2_base`.
