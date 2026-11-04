# RWKV-7 0.1B under CKKS on `perseus.impl`: encrypted decode, one conversation

`python/perseus/examples/rwkv7/` runs RWKV-7 (x070, `RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth`: 12 blocks, width
768, 12 heads of 64, d_ffn 3,072) one token per step, every block on ciphertexts. The client decrypts the last
block's hidden state and applies `ln_out` and the head in the clear. Every non-polynomial operation is the polynomial
of `coeffs_mixed.json` (the research harness's `poly` mode, `results/vector-d-20261104/harness/rwkv7_model.py`).
Status: verified on the CPU (numpy fake with the runtime's level rules, capture recorder, planner). **Nothing has run
on a GPU yet**; `tools/rwkv7/run_box.sh` is the box procedure.

## The circuit

The stage interface is one ciphertext, `z = h + i·v_first`. The residual stream `h` sits on the real axis. Layer 0's
value `v_first`, which every later block's value residual reads, sits on the imaginary axis. A block reads `2h = z + conj z`
(a conjugation costs no level; the factor 2 is folded into the LayerNorm's constants) and `2 v_first = i (conj z − z)`
(the factor 2 is folded into the v-gate polynomial, whose coefficients are halved). It adds its real outputs to `z`, so
`v_first` rides along without needing a register. Block 0 writes it with `z += i·v` (multiplying by `i` costs no level).

Per block (`model.py`, `Circuit.block`; the site names are the harness's):

| step | how |
|---|---|
| `x = LN1(h)` | channel sums (rotate-and-sum over the 1,024 channels, 10 rotations), `2(h−μ)` masked to the 768 live channels, Newton 1/√ on `Σ(2(h−μ))²` (site `ln1.rsqrt`) |
| `r, k, v, a', w', g', v'` | PCMM on the pair `[x, x_prev]`: the token shift `x + (x_prev − x)·μ` is folded into the matrices (`diag(1−μ)W` on `x`, `diag(μ)W` on `x_prev`), and so are `a1·a2` and `v1·v2` (no nonlinearity between them) |
| `w = decay(w0 + w2·tanh(w'))` | sites `att.w_tanh`, `att.decay` (`exp(−e^−0.5 σ(z))` as one polynomial) |
| `a`, `g`, the v residual | sites `att.a_sigmoid`, `att.g_sigmoid` → PCMM `g2`, `att.v_sigmoid` (blocks ≥ 1) |
| `κ̂ = k·k_k/‖k·k_k‖` per head | per-head sum (below), Newton (site `att.kk_rsqrt`); `scale^−1/2` folded into `k_k` |
| state update | `S_m ← S_m⊙w + bcast_j(−(S_m⊙κ̂))⊙(κ̂⊙a) + V_m⊙k̃` with `V = transpose(v)`, for m = 0, 1 |
| `y = S r` | `untranspose(S_0⊙r, S_1⊙r)` |
| GroupNorm + bonus | per-head mean and variance (two per-head reductions), Newton (site `att.gn_rsqrt`), `+ (Σ_head r⊙k̃⊙r_k)⊙v` |
| `h += Wo (y⊙g)` | PCMM |
| channel mix | `LN2` as above; PCMM on `[x2, x2_prev]` → 3 ciphertexts → `relu²` polynomial (site `ffn.relu2`) → PCMM 3→1 |
| carry | the four carried ciphertexts (`S_0`, `S_1`, `x_prev` of both mixes) are refreshed at block end (the canonical carry, as in `fhe_ssm`), so every token runs the same op stream |

**Sites.** A Chebyshev site is `impl.poly.eval_chebyshev_bsgs` (baby-step giant-step) with the JSON's interval and
coefficients and no coefficient trimming. A degree-d series costs about 2√d ciphertext products and a depth of
⌈log₂ d⌉ + 1 levels: 8 levels for the degree-127 `w_tanh` of block 1. An inverse square root is fit.py's Newton
form: a cubic Chebyshev seed on `u = (x + eps)/scale` in `[seed_lo, 1]`, then `steps` iterations of
`y ← y(1.5 − 0.5·u·y²)`, written `1.5y + (y·(−u/2))·y²` (two levels per step). The affine map from the raw
statistic to `u`, and the final `scale^−1/2`, are folded into constants.

**No clamp.** The harness's poly mode clamps each site's input to `[lo, hi]`; a ciphertext cannot be clamped. Inside
the interval the circuit equals poly mode. Outside it, a Chebyshev series of degree ~100 grows like
`cosh(d·acosh|t|)` (about 10¹³ at |t| = 1.03 and d = 127), and Newton diverges once `u·y₀² ≥ 3`. Position 0 of a
text leaves several intervals. With the prefix `"\n\n"`: GroupNorm's variance in block 8 lands 62 % of the interval's
width above its top, Newton diverges there, and the 12-block fake turns to NaN at that position
(`tests/test_unclamped.py`). Hence:

**The public prefix.** Every prompt starts with `"\n\n"`, the World models' document separator. That token is public,
so the state after it is a public constant. It is computed in plaintext on the poly model (clamp on), encrypted, and
the encrypted decode starts at position 1 (`--prefix 1`, the default; `--prefix 0` runs every token encrypted from
zero state). The comparison model continues from the same state. On the texts tested (`tests/test_ranges.py`),
positions ≥ 1 after `"\n\n"` stay inside every Chebyshev interval. Newton sites still go somewhat out of range at
later positions, but stay convergent.

## The layout (`layout.py`)

Ring 2^16 gives 32,768 complex slots. The filled layout (PCMM's `reshape(1024, t)`) puts channel `c` of lane `l` at
slot `c·t + l`, with 1,024 channels and `t = 32` lanes. Width 768 is padded to 1,024. With one conversation every lane
holds the same value ("natural" form). PCMM applies the same matrix on every lane, so it keeps that form, and every
per-channel constant is a plain plaintext.

The 64 × 64 state of head `h` (`S[h, i, j]`, with `i` the value dimension and `j` the key dimension) lives in
`M = 64/t = 2` ciphertexts. `S_m` holds `S[h, i = 32m + l, j]` at channel `64h + j`, lane `l`. Everything indexed by
`j` (`w`, `κ̂`, `a`, `k̃`, `r`) is a natural vector, so its product with the state is a plain slot-wise product. Only
`v` (indexed by `i`) and the readout `y = S r` change form:

* `transpose(v)`: for each lane `l`, a rotation by `32·(32m + l)` slots brings channel `64h + 32m + l` to the block
  start; a mask keeps (block start, lane `l`); a rotate-and-sum over channel steps `−32 … −1024` copies the value
  over the block. Baby-step giant-step over `l = 8g + b` costs 7 baby rotations shared by both `m`, 7 giant
  rotations, 64 mask products (one level), and 12 copy rotations.
* `untranspose`: the block sums `Σ_j` (rotate-and-sum over `+32 … +1024`) leave `y[h, 32m + l]` at (block start,
  lane `l`). A mask and a rotation by `−(1024m + 31l)` move it to (channel `64h + 32m + l`, lane 0). A rotate-and-sum
  over lanes `−1 … −16` copies it to every lane. Same baby-step giant-step.
* the per-head reduction: rotate-and-sum `+32 … +1024`, a block-start mask times the scale (one level),
  rotate-and-sum `−32 … −1024`.

The steps the layout issues, plus PCMM's ±1: 43 rotation keys (the plan's `plan_rot_steps.json`).

## Execution: planned

This runtime's device path runs planned only (`examples/gpt2/run_decode.py`: "the new runtime runs planned only").
There is no eager GPU mode with automatic refreshes, so the route to a correct run is the one `fhe_ssm` already took
to the box: capture → plan → `impl.driver.PlannedRun`. The PCMM linears are the device-proven path of `fhe_ssm`'s
`pcmm24` arm: int8 digit planes resident on the device, one external refresh per output. Diagonal linears would need
a resident plaintext store far beyond the card.

* `run_capture.py` (CPU) first runs the fake on a public calibration text and records each linear's `max |y|`. That
  gives the PCMM band shift (`impl.pcmm.band_shift`'s rule, `pcmm_shifts.json`): the Rust service refreshes
  `y / 2^shift` and multiplies by `2^shift` afterwards (`crates/pc-py/src/pcmm.rs`). Then it runs the capture
  recorder with values on, so the graph carries magnitudes. It captures token 1 after the prefix, under the
  `LANDING=high` switches (StC-first dense refreshes, which PCMM requires).
* `tools/fhe_ssm/make_plan.sh` (`LANDING=high`) produces the plan.
* `run_planned.py` (GPU) is `PlannedRun` with these changes:
  * The twin records two probe tokens on a *public* row, and the stream is checked static (no dynamic step, equal
    plaintexts).
  * Every plaintext is staged once (one slot per distinct value digest and level), and the PCMM planes are uploaded
    by name.
  * The first token reads the prefix state as `Ct` inputs (packed and landed by an eager refresh); later tokens read
    the registers.
  * Each reply is decrypted and scored against the poly model and the circuit's numpy mirror.
  * Keys are in-process (`KeySource.generate`): the session holds the secret. This is a correctness and timing run,
    recorded as `keys: insecure-inproc`.

## What is verified on the CPU (`tests/`, `python -m pytest python/perseus/examples/rwkv7/tests`)

| test | what | result |
|---|---|---|
| `test_ref` | the numpy reference (torch-free checkpoint reader `ckpt.py`) in exact mode predicts; poly vs exact | " Paris" after "The Eiffel tower is in the city of"; top-1 15/15 |
| `test_layout` | transpose, untranspose, per-head reductions vs the matrix algebra, t = 32 and 16 | exact |
| `test_mirror` | the circuit on numpy slots vs the poly reference (no clamp), 2 blocks, 7 tokens | 3e-8 (float32 weights); with 24-bit PCMM rounding 4.5e-5 (one scale per matrix; 6e-4 with the fused 2→7 first linear, `--fuse-l1`) |
| `test_fake` | the fake (runtime level rules, PCMM fake, carry) and the capture recorder vs the mirror, 2 blocks | 1e-14; the same refreshes every token |
| `test_full_fake` (slow_shape) | **all 12 blocks on the fake**, 5 tokens after the prefix, vs the poly model (clamp on) | **top-1 5/5; logits relErrRms ≤ 3.4e-5**; 1,453 eager refreshes per token; 3.7 GB |
| `test_planned_cpu` | the planned driver's CPU half: carried state found as program inputs, stream static | 4 carried inputs per block; no dynamic step |
| `test_capture` / `test_full_plan` (slow_shape) | shifts + capture + plan (1 block / 12 blocks) | 12 blocks: 13 stages, **478 planned refreshes per token**, 43 rotation keys, 15 s |

The 12-block numbers come from the fake. That is float64 arithmetic with the runtime's level accounting: no CKKS
noise, no bootstrap error.

**Not verified:**
* everything on the device: the session, the compile of these graphs (`_pc.Program.compile`; the CPU `compile_check`
  binary was not built here), PCMM's band shift and correction factor on these magnitudes, the eager refresh of the
  prefix state, and every timing;
* whether this runtime's bootstraps carry the complex `z` correctly (the planner routes by the recorder's field tags,
  which never flagged an unsound tag in these captures).

## Box commands

```bash
cd /root/perseus-rwkv7
BLOCKS=1 bash tools/rwkv7/run_box.sh all        # smoke: build, check, capture, plan, run on one block + tail
bash tools/rwkv7/run_box.sh capture && bash tools/rwkv7/run_box.sh plan && bash tools/rwkv7/run_box.sh run   # 12 blocks
TORCH_PY=/path/to/python-with-torch bash tools/rwkv7/run_box.sh crosscheck   # harness (torch) vs the numpy reference
```

Products land in `$OUT` (default `/root/out_rwkv7`): `run[_b1].jsonl` holds one line per token (`top1_agree`,
`logits_relErrRms`, `hidden_relErrRms_vs_poly`, `hidden_relErrRms_vs_mirror`, `seconds`, `poly_sites_out_of_range`)
and a summary line; `run*.log`, `capture*.log`, `plan*.log` and `session.log` sit beside it.

## Upstream file touched

`python/perseus/plan/placer/planner.py`: the per-stage summary's log formatted `predicted_rel_err` with `:.3g`, which
fails when a stage has no placed refresh (the tail here). Now it prints `-` in that case. The log line only; no
planning logic changed.
