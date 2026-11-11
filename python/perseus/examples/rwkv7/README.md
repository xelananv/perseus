# RWKV-7 under CKKS on `perseus.impl`: encrypted decode, prefill, batched decode and continuous batching, any RWKV-x070-World size

`python/perseus/examples/rwkv7/` runs RWKV-7 (x070 — e.g. 0.1B `v2.8`: 12 blocks, width 768, 12 heads of 64, d_ffn 3,072; 0.4B `v2.9`: 24 blocks, width 1,024, 16 heads of 64, d_ffn 4,096) under CKKS, reading dimensions directly from the checkpoint. The client decrypts the final block's hidden state and applies `ln_out` and the output head in plaintext. Every non-polynomial operation uses pre-calibrated Chebyshev polynomials and constant-seed Newton inverse-square-root iterations from `tools/rwkv7/coeffs/` evaluated identically by the research harness (`tools/rwkv7/harness/`).

## Status & Measured GPU Results (2026-11-05, 1× RTX PRO 6000 Blackwell 600 W, 97,887 MiB, CUDA 13.2)

Everything listed below has been built, verified end-to-end on GPU, and scored against the plaintext polynomial model (`top1_agree`, `logits_relErrRms`) and circuit mirror:

| Model | Mode / Configuration | Keys | Speed | Throughput (full lanes / realised) | Fidelity (top-1 vs poly) | Host / Device Peak |
|---|---|---|---|---|---|---|
| **0.1B** | Decode (`P1`–`P3`, `LEVERS=best`) | `bundle` (`has_secret: false`) | **9.74–9.76 s/tok** | 0.103 tok/s | 141 / 141 (`0` out of range) | 41.1 GB / 72,712 MiB |
| **0.1B** | 148-step free run (`P1`) | `bundle` (`has_secret: false`) | **9.77 s/tok** | 0.102 tok/s | 155 / 155 (`0` out of range) | — |
| **0.1B** | Prefill `T=32` (`L1,L5,L9,L25`, `LEVERS=best`) | `bundle` (`has_secret: false`) | **24.2–24.6 s/pass** (**0.76 s/prompt tok**) | **12.8× decode** | 409 / 416 prefill + 198 / 201 decode | 177.6 GB / 82,080 MiB |
| **0.1B** | Batched `K=32` (`filled`, `PLANOPT=b2`, `LEVERS=best`) | `bundle` (`has_secret: false`) | **19.36 s/pass** (0.61 s/tok) | **1.653 tok/s** full / **1.11 tok/s** realised (`16.1×` decode) | 1,300 / 1,306 (`0` out of range) | 100.9 GB / 86,670 MiB (unflagged) |
| **0.1B** | Batched `K=32` (`b2`, `SHED_TWIN=1`, `LEAN_PREFIX=1`) | `bundle` (`has_secret: false`) | **27.99 s/pass** | 1.143 tok/s full / 0.77 tok/s realised | 1,291 / 1,297 (`52/61` passes) | **26.3 GB / 67,448 MiB** |
| **0.1B** | Continuous batching `serve` (`K=4`, `T=32`, 8 convs) | `inproc` (`has_secret: true`) | 14.18 s batch / 29.77 s prefill / 3 ms join | 0.176 tok/s incl. prompt | 245 / 248 | 41.0 GB / 85,900 MiB |
| **0.4B** | Decode (`P1`–`P3`, `LEVERS=best`) | `inproc` (`has_secret: true`) | **19.73–19.74 s/tok** | 0.051 tok/s | 141 / 141 (`0` out of range) | 74.5 GB / 80,810 MiB |
| **0.4B** | Prefill `T=32` (`L1,L5`, `SHED_TWIN=1`, `LEAN_PREFIX=1`) | `bundle` (`has_secret: false`) | **48.8–49.9 s/pass** (**1.53–1.54 s/prompt tok**) | **12.8× decode** | 223 / 224 prefill + 90 / 90 decode | 57.9 GB / 79,920 MiB |
| **0.4B** | Batched `K=32` (`filled`, `b2`, `SHED_TWIN=1`, `LEAN_PREFIX=1`) | `bundle` (`has_secret: false`) | **39.08 s/pass** (1.22 s/tok) | **0.819 tok/s** full / **0.548 tok/s** realised (`16.1×` decode) | 1,302 / 1,306 (`0` out of range) | **39.5 GB / 75,904 MiB** |
| **0.4B** | Continuous batching `serve` (`K=8`, `T=32`, `filled`, `b2`) | `inproc` (`has_secret: true`) | 30.88 s batch / 57.32 s prefill / 5 ms join | 0.086 tok/s incl. prompt | 247 / 248 (`K=32` OOM at CUDA graph init, 97.2 GB) | 75.5 GB / 89,356 MiB |
| **1.5B** | Plaintext fp64 polynomial gate (`coeffs_final.json`) | — | — | — | WikiText PPL **9.618 → 9.800 (+1.89%)**, top-1 **93.07%** | Gate **FAIL** — held before GPU |

> On **1.5B**: unclamped evaluation stays finite (`0` non-finite, `0` convergence escapes), and `coeffs_final.json` equals clamped mode, but 5 sites (`blocks.{6,15,21,22}.att.w_tanh`, `blocks.23.att.a_sigmoid`) miss the 1e-3 Chebyshev target at degree 255, failing the 98% top-1 and ≤ 0.5% perplexity gate bars.

## How to Run (Quickstart)

This branch (`fhe-ssm-rwkv7-any`) bundles everything needed except the Apache-2.0 `.pth` weights:
- **Tokenizer, harness & PyTorch gate scripts:** `tools/rwkv7/harness/` and `tools/rwkv7/raw/`
- **Verified coefficients & range JSONs (0.1B, 0.4B, 1.5B):** `tools/rwkv7/coeffs/`
- **Benchmark prompts & arrivals:** `tools/rwkv7/batch_prompts.txt`, `tools/rwkv7/longrun_prompts64.txt`, `tools/rwkv7/serve_arrivals.txt`
- **Checkpoint & dataset downloader:** `tools/rwkv7/fetch_data.py`

### 1. Clone and Build

Requires Python ≥ 3.10, Rust (`rust-toolchain.toml` pins `1.99.1`), `uv`, and for GPU runs an NVIDIA GPU with CUDA 13.2+ toolkit (`nvcc` on `PATH`).

```bash
git clone -b fhe-ssm-rwkv7-any https://github.com/xelananv/perseus.git
cd perseus
bash tools/rwkv7/run_box.sh build
```

This creates `.venv`, installs `[hf,examples,dev]` dependencies, and builds the CUDA extension `perseus._pc` via `maturin develop --release`.

### 2. Download Weights and Run CPU Verification

```bash
# Download and SHA-256 verify the 0.1B checkpoint into .cache/rwkv7/ (adds ~382 MB)
bash tools/rwkv7/run_box.sh fetch
# Optional: also fetch 0.4B (~900 MB) and WikiText-103 + LAMBADA datasets:
# MODELS=0.1b,0.4b WITH_DATA=1 bash tools/rwkv7/run_box.sh fetch

# Run the CPU test suite and checkpoint shape/coefficient verification (~1 min)
bash tools/rwkv7/run_box.sh check
```

To run the full pytest CPU test suite directly (without a GPU):

```bash
PYTHONPATH=python .venv/bin/python -m pytest -q python/perseus/examples/rwkv7/tests
# Also run the standalone harness unit tests (uses torch if installed in a Python env):
python tools/rwkv7/harness/tests/run_without_pytest.py
```

### 3. Single-Conversation Encrypted Decode on GPU (`0.1B` / `0.4B`)

```bash
# Default PTH is 0.1B in .cache/rwkv7/; COEFFS auto-selects tools/rwkv7/coeffs/0.1b/coeffs_final.json
export LEVERS=best STC_CACHE=160 KEYS=bundle

# Capture (CPU) -> Plan (CPU) -> Keyspec (CPU/Device check) -> Keygen (Client) -> GPU Decode Run
bash tools/rwkv7/run_box.sh capture plan
KEYMODE=decode bash tools/rwkv7/run_box.sh keyspec keygen
TEXT=$'\n\nThe scientists discovered a new species of' TOKENS=64 RUN_EXTRA="--generate 32" RUN_TAG=P1 \
  bash tools/rwkv7/run_box.sh run
```

Outputs land in `$OUT` (`$PWD/.cache/rwkv7_out` by default when `/root/out_rwkv7` does not exist): JSONL per-token scores (`run_best_P1.jsonl`) and logs (`run_best_P1.log`, `session.log`). For an interactive timing-only run without separate client keygen, set `KEYS=inproc` (`has_secret: true`).

To run **0.4B** instead:
```bash
MODELS=0.4b bash tools/rwkv7/run_box.sh fetch
export PTH=.cache/rwkv7/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
# COEFFS automatically resolves to tools/rwkv7/coeffs/0.4b/coeffs_final.json
```

### 4. Encrypted Chunked Prefill (`T = 32`) + Decode Tail

```bash
export LEVERS=best STC_CACHE=160 KEYS=bundle PF_GENERATE=32
bash tools/rwkv7/run_box.sh capture_pf plan_pf pfcheck
KEYMODE=pf bash tools/rwkv7/run_box.sh keyspec keygen
POOL_PF=256 bash tools/rwkv7/run_box.sh run_pf
```

### 5. Headline Batched Decode (`K = 32`, Filled Layout, `b2`, Low Memory)

```bash
export LEVERS=best STC_CACHE=160 KEYS=bundle LAYOUT=filled PLANOPT=b2 B_K=32 B_GENERATE=32
bash tools/rwkv7/run_box.sh capture_b plan_b bcheck
KEYMODE=b bash tools/rwkv7/run_box.sh keyspec keygen
RWKV7_SHED_TWIN=1 RWKV7_LEAN_PREFIX=1 POOL_B=160 STC_CACHE_B=1 bash tools/rwkv7/run_box.sh batch
```

For **0.4B** on a 96 GB card, `RWKV7_SHED_TWIN=1 RWKV7_LEAN_PREFIX=1 POOL_B=160 STC_CACHE_B=1` is required so the 32-conversation session fits inside 70.7 GB VRAM and 39.5 GB host RAM.

### 6. Continuous Batching (`serve`)

```bash
B_K=4 bash tools/rwkv7/run_box.sh scheck
RWKV7_SHED_TWIN=1 B_K=4 KEYS=inproc S_GENERATE=8 POOL_S=256 bash tools/rwkv7/run_box.sh s
```

### 7. Full Any-Size Pipeline (`size_pipeline.sh`)

```bash
# Dry run up to GPU (calibration, fit, scan, quality gate, captures, plans):
DRY=1 bash tools/rwkv7/size_pipeline.sh .cache/rwkv7/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth 0.1b
# Full end-to-end GPU run:
bash tools/rwkv7/size_pipeline.sh .cache/rwkv7/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth 0.1b
```

For a quick reference of every helper script in `tools/rwkv7/`, see [`tools/rwkv7/README.md`](../../../tools/rwkv7/README.md).

## Any size (`ANYSIZE_DESIGN.md`)

Every shape is read from the checkpoint (layers, width, heads, d_ffn, the LoRA ranks of w, a, v, g). Ring 2^16 holds
32 lanes x 1,024 channels, so a width above 1,024 needs one of two layouts, both built (`--lanes`):

- **channel shards** (`--lanes 32`, the default): a width-C vector is `S = ceil(C / 1024)` ciphertexts; every
  per-channel op runs per shard; LayerNorm adds the shards' statistics before one rotate-and-sum and one Newton chain;
  a head never straddles a shard, so the head state is 2 ciphertexts per shard (C / 512 per block); a PCMM linear takes
  and returns shard lists (`n_in`, `n_out` in 1,024-blocks). Above one shard the a / v LoRA pairs run as two PCMMs
  through the rank (`--lora split`, the fold would cost S^2 blocks of planes) and every block refreshes its output
  shards (`shard_land`: the runtime chains only a block's first output into the next block; the others enter as
  registers at a fixed landing, checked before the session opens). K and T stay at 32.
- **fewer lanes** (`--lanes 16` / `8`): 2,048 / 4,096 channels per ciphertext, PCMM at its own t = 16 / 8 shape; one
  ciphertext per vector, 64 / t state ciphertexts per block. K and T <= t.

Which one a size runs is a cost question the planner answers (`ANYSIZE_DESIGN.md` §3); at width <= 1,024 both are the
original layout and the op stream is unchanged (the fake's call log and values are identical to `4bba3a93`'s for
decode, prefill and batched passes on 2 blocks of the 0.1B checkpoint).

**Keys** (`run_planned.py --keys`): `inproc` (the default, INSECURE: the session holds the secret; recorded
`insecure-inproc`) or a client bundle: `--keyspec F` writes the session's key request for the plans that run,
`--keygen DIR --keyspec-file F` is the client role (a fresh secret, `DIR/keys.pckb`, `DIR/client_secret.bin`), and
`--keys DIR` opens the session with `Session.from_bundle` (no secret: `has_secret false` in the header) while a
separate `_pc.Client` encrypts every input and decrypts every reply as wire frames. `run_box.sh keyspec | keygen`
(KEYMODE = decode | pf | b | s) and `KEYS=bundle` wrap it; `run_box.sh` requires `KEYS` for every device step (no
default). **What the split is:** between objects in ONE process, as in fhe_ssm: the session object opens from the
bundle and never holds the secret, but the client object that encrypts and decrypts runs in the same process and reads
`client_secret.bin` (written mode 600) from the bundle directory. It shows the session needs no secret; it is not a
two-machine deployment.

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

**The 1/√x sites' method is a property of the coefficients file**, not a switch: `newton` (a Chebyshev seed, the
original fit), `newton_const` (P6, `ROBUST_NORMS.md`: the constant seed that converges for every input below 3× the
widened top; the gate-3 files) or `newton_scaled` (the same seed, range and convergence region with per-step constants,
about half the depth, but a different error profile: the tolerance spread over the whole widened range instead of
converged on the data, 2× the error below the range, an overflow beyond 1.385× the divergence bound — `ROBUST_NORMS.md`
"newton_scaled"; `tools/rwkv7/convert_newton_scaled.py` converts a `newton_const` file; `tests/test_newton_scaled.py`;
the speed-up map's T1, `results/rwkv7-fhe-20261104/SPEEDUP_MAP_20261105.md`). A converted file is not usable before the
gate passes it against the `newton_const` file. The port, the mirror (`ref.py`) and the harness evaluate all three.

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

**The filled state (`--layout filled`, `LAYOUT=filled` in `run_box.sh`; `FILLED_DESIGN.md`).** The two value groups
of a head's state share ONE ciphertext: `S = S_0 + i·S_1` (at t = 32; group q and q + M in general, M = 64 / (2t)).
Every operand the state meets on the update side (`w`, `κ̂`, `κ̂⊙a`, `k̃`, `r`) is indexed by the key dimension j, so
it is real and the same for both groups: `(S_0 + i S_1)·w = S_0 w + i S_1 w` is one product for both. The block
sums, masks and the carry refresh are real-linear and serve both too. Only the two transforms that index by i change:
group m's transpose is group 0's of `v` shifted by `t²m` slots, so `transpose` feeds group 0's transform the complex
input `v + i·rot(v, t²)`; `untranspose` runs group 0's transform once on both axes (masks at 1/2) and splits it with
one conjugation, rotating the imaginary part by `−t²`. Same depth, half the state ciphertexts, carry refreshes, state
products and plaintext masks; per conversation and block at t = 32, 83 rotations become 46 plus one conjugation. The
token side, `z = h + i·v_first` and the three broadcast pairs are untouched. Default off; the current layout's op
stream is unchanged.

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
  * `RWKV7_SHED_TWIN=1` (default off; host RAM only): the twins otherwise keep both probe tokens' every intermediate
    vector (512 KB each at the fake's fixed ring 2^16) and a copy of every mask for the whole session, about 1.4 GB
    per block at decode and 9.5 GB per block at K = 32 or T = 32; with the flag they keep one slot of each vector
    only the record references and one copy per distinct plaintext. Same streams, plans and staged values
    (`tests/test_shed_twin.py`); the header's `shed_twin` records it (`results/rwkv7-fhe-20261104/new_pod/HOST_RAM_20261105.md`).
  * `RWKV7_LEAN_PREFIX=1` (default off; device memory only): every fresh encryption of the public prefix state is an
    eager `Ct` in a full 31.5 MiB pool slot, and the pool must be sized for all of them at once. With the flag one
    encryption serves every carried input that reads the same value at the same compiled meta (a batched session's K
    conversations start from one state: K M + 2 inputs per block become M + 2), and the decode / prefill driver
    encrypts only the set a pass reads, after the warm-ups, dropping it after that pass. Same values, metas and
    programs; still untimed. Lower `POOL_B` / `POOL_PF` to take the saving (`tests/test_lean_prefix.py`;
    `results/rwkv7-fhe-20261104/new_pod/DEVICE_MEM_20261105.md`). `RWKV7_POOL_TRACE=1` records the lowest free pool
    count (`pool_free_min`) so the pool can be sized from a measurement.
  * **The carried state's refresh levers are flags of `run_capture`, not of a run** (the speed-up map's S1 / S2;
    `tests/test_state_levers.py`): `--carry-drop d` refreshes S's carries (and S's landing in `start()`) under raise
    drop d (1, 3, 5: the runtime's built dense variants), so the recorder lands them 2 d units deeper and S enters every
    block at 26 + 2 d; `--carry-skip-s` runs the two-pass cycle's schedule (S unrefreshed at odd positions: a capture at
    `--capture-tick 1` records pass A, S in at 26 and out at 32 with no S carry; at `--capture-tick 2` pass B, S in at
    32, refreshed 38 → 26). Both are recorded in `rwkv7_capture.json` (`carry_drop`, `carry_skip_s`); the model's
    kwargs default off and read no environment; **`run_planned` refuses a capture that carries one** (its plan has S
    at other levels than the driver's twin carries, and the driver has no two-program cycle). Status (2026-11-06,
    CPU): both passes of the cycle plan and compile with no extra placed refresh and matching hand-over levels (A out
    32 = B in, B out 26 = A in); the dropped carries are refused by the compile check — the compiler resolves a
    deliberate refresh by its captured landing through `route_for_landing` (`crates/pc-program/src/compile/mod.rs:
    176-190`), which tries each slot count's base route at the default raise drop and never the built drop variants
    (real or complex alike; the device builds them, `crates/pc-gpu/src/bts/dense.rs:110`), so S2 needs a compiler
    change (a rebuild) before any device run.

## Prefill: T prompt tokens per pass (`prefill.py`, `PREFILL_DESIGN.md`)

A decode step copies one token over the 32 lanes. A prefill pass puts **token l of a chunk on lane l** instead.

**What runs unchanged.** The linears (PCMM: the same matrix on every lane), the polynomial sites, the Newton chains
and the norms (their reductions stay inside a lane) are the decode step's code at the decode step's cost.

**What changes:**
- **Token shift:** lane l takes lane l − 1, by a rotation and a mask; lane 0 takes the carried LN output.
- **WKV:** a sequential scan inside the pass, on the decode step's own state layout and `state_step`. Each token's
  vectors are broadcast from its lane: three complex pairs, one broadcast each, split by a conjugation. Its readout
  is placed back on its lane (Horner accumulation).

A pass therefore ends with exactly the state a decode step carries, and decoding continues from it on the same
registers. The chunk is the client's T rows. Lanes past a short chunk repeat its last row and are never read.

**Planned cost** (0.1B, CPU plans, LANDING=high):
- block 0: 62 refreshes per 32-token pass against 31 per decode token;
- block 1: 100 against about 40.

**Measured GPU performance (`T = 32`, `coeffs_final.json`, `LEVERS=best`, `KEYS=bundle`):** **24.2–24.6 s per 32-token pass (0.76 s per prompt token, 12.8× faster than decoding the prompt)** on 0.1B, and **48.8–49.9 s per pass (1.53–1.54 s per prompt token, 12.8×)** on 0.4B. The WY / chunked form was rejected on encrypted cost (`PREFILL_DESIGN.md` §2).

**Device path:**
- `run_planned.py --prefill-chunk T --pf-plan PLAN_PF` opens **one session** for the decode programs and the prefill
  programs (`Rwkv7PrefillRun`). Each set is compiled from its own capture and plan, and the prefill set is
  relabelled `pf_block_<b>`.
- The prefill state registers are the decode ones. Before the session opens, a check refuses plans that land the
  carried state at different (level, deg).
- The prompt runs as ⌊n/T⌋ passes. The remainder and any `--generate` steps are decode steps.
- Every prefilled position is scored like a decoded one (top-1 and logits relErrRms against the poly model, hidden
  relErrRms against the mirror). Each pass also gets its own line: seconds per pass and per token.

**Levers from the fhe_ssm stack** (`PREFILL_DESIGN.md` §6):
- transfer, opt-in (`LEVERS=best`): PAIR=1 with `--schedule siblings`, and the fused EvalMod;
- built but doubtful (`CFTABLE=1`): the PCMM CF table. The band shifts already normalise every PCMM output, so it
  collapses to a global CF of 2–3; A/B it on the smoke first;
- does not transfer: pack-stats (no two norm statistics are ever ready together).

## Batched decode: K conversations per pass (`batched.py`, `BATCHED_DESIGN.md`)

Conversation c's current token sits on lane c, so every token-side op serves K ≤ 32 conversations for the price of
one. Each conversation keeps **its own** state ciphertexts per block (decode's layout). Each pass updates them one
conversation at a time with prefill's machinery: lane c → natural form, decode's `state_step`, the readout back to
lane c. The K updates are independent.

The token-shift carries are lane-form by default (`--batch-carry lane`): one ciphertext per carry, conversation c's on
lane c. That ciphertext is the LN output itself, so the token shift costs nothing. `--batch-carry decode` keeps one
natural-form carry per conversation (2(K − 1) more refreshes per block).

Positions are per lane:
- each conversation reads its own prompt (`--prompts FILE`, one per line, `\n` escapes, the public `"\n\n"`
  prefix prepended if missing), then free-runs `--generate` steps;
- the client decrypts all K picks and feeds c's back to lane c.

Planner, K = 32, block 0: **222 refreshes per pass**, against 32 × 31 = 992 for 32 decode steps. `PLANOPT=b2` (`PLAN_B2.md`) removes the per-conversation update refreshes, a placer artefact: every conversation then pays its 2 carries per block; combined with `LAYOUT=filled` and `LEVERS=best` on `coeffs_final.json` (`KEYS=bundle`), measured GPU pass latency at `K = 32` is **19.36 s/pass (1.653 tok/s full-lane / 1.11 tok/s realised, 16.1× decode)** on 0.1B and **39.08 s/pass (0.819 tok/s full-lane / 0.548 tok/s realised, 16.1× decode)** on 0.4B.

## Continuous batching: prefill, then join the batch (`serve.py`, `SERVE_DESIGN.md`)

`run_planned.py --serve FILE` serves a stream of conversations (FILE rows `<arrival pass><TAB><prompt>`) on ONE
session holding four program sets: decode, prefill (`pf_block_<b>`), the K-lane batched pass (`b_block_<b>`) and the
join (`join_block_0`). An arrival is read in single-conversation mode while the batch waits (⌊n/T⌋ prefill passes,
decode steps for the remainder), then joins a free lane:
- **state:** the single-conversation programs write `S_0, S_1` straight into the lane's registers (`s0c<c>`, `s1c<c>`):
  a register hand-over, no server arithmetic;
- **carries:** the join program overwrites lane c of the lane-form carries, `xa ← xa ⊙ (1 − m_c) + xa_new ⊙ m_c`, one
  plaintext product per operand, one add, one refresh per carry (24 per join for the 0.1B); the masks are plaintext
  inputs re-staged per join, so one program serves every lane.

A conversation that has generated `--generate` tokens leaves: its S registers go back to an encryption of the public
prefix state, its carries to the prefix carry through the same join program (all lanes leaving at one pass in one
run; skipped when the lane is reused before the next pass). Idle lanes run from the prefix state on their own picks
(in range). `prepare` refuses, by name, any hand-over whose (level, deg) differs between writer and reader.

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
| `test_prefill` (mirror) | a T-token pass vs T decode steps, and 2 decode steps after the hand-over; T = 32, 7, 1 (t = 32 lanes), T = 16 (t = 16) | hidden states, carried state and post-hand-over decode bit-identical (every added op is exact in float64); a reversed chunk differs |
| `test_prefill` (fake, recorder) | prefix → 6-token pass → 3 decode steps vs 9 decode steps, 2 blocks | identical; **top-1 9/9** both, against the poly model |
| `test_prefill` (capture, `--rwkv7-plan`) | `run_capture --prefill-chunk T` + plan | T = 32: 62 refreshes (block 0), 100 (block 1, 2-block plan); T = 8: 38 |
| `test_prefill_session` | the one-session register merge on stand-in programs; a landing mismatch stops it | merged by name; prefill outputs after the decode registers |
| `test_prefill` (`session_prepare`, box only) | both plans compiled by `Program.compile`, registers merged, landings compared | skipped here (CPU build has no `Program`); `run_box.sh pfcheck` |
| `test_batched` (mirror) | K conversations batched vs K separate decode mirrors; prompts of 2–6 tokens + 2 greedy steps each; K = 2, 5, 32, both carry layouts, hoisted rotations | hidden states, end states and token streams bit-identical; a swapped lane assignment differs |
| `test_batched` (fake, recorder) | K = 3 vs 3 separate fake decodes, against each conversation's poly model | identical; **top-1 15/15** |
| `test_batched` (`k1_is_decode`) | `batch=1` vs decode on the fake | same call log (8,154 calls), keys and values |
| `test_batched` (capture, `--rwkv7-plan --rwkv7-b-k K`) | `run_capture --batch K` + plan | block 0: 55 / 79 / 127 / 222 refreshes at K = 4 / 8 / 16 / 32; block 1 (K = 32): 167; decode-layout carries (K = 32): 284 |
| `test_batched` (`session_prepare`, box only) | a K = 4 batched plan compiled, every conversation's state found as inputs | skipped here; `run_box.sh bcheck` |
| `test_levers` | the CF table from this capture; `--schedule` reaches `Program.compile`; a PAIR=1 + fused-EvalMod plan (`--rwkv7-plan`) | covers all 11 linears (CF 2–3); siblings passed; `PC_BTS_PAIR=1 PC_EM_FUSE_RESCALE=1` on `runtime=` |
| `test_anysize` | synthetic checkpoints (`synth.py`) at widths 768–4,096 in both layouts: the mirror vs the poly reference; at 2,048 the fake and recorder vs the mirror, prefill vs T decodes, batched vs K decodes; the 1-block capture + plan (`--rwkv7-heavy` for the 5–9 GB ones, one per process) | ANYSIZE_DESIGN.md §6 |
| `test_serve` (mirror) | 5 conversations, K = 3 / 2 / 3 lanes, T = 4 / 1 / 0 (prefill + decode remainder, prefill only, decode only), arrivals at passes 0, 0, 0, 1, 9, 2 generated each: a queue, lane reuse, a 2-lane leave, a clock jump | every position of every conversation **bit-identical** to that conversation decoded alone; every idle segment identical to a fresh conversation from the prefix; a join onto the wrong lane and a leave without reset both break it |
| `test_serve` (fake, recorder) | the same schedule (K = 3, T = 4) on the fake and the recorder | identical to the conversation decoded alone on a fresh fake; **top-1 36/36** against each conversation's poly model |
| `test_serve` (`join_capture_and_plan`, `--rwkv7-plan`) | `run_capture --join` + plan | per carry 2 products, 1 add, 1 deliberate refresh; 12 blocks: 24 refreshes, 0 placed, landing 26 = the batched carries' |
| `test_serve_session` | the serve session's CPU half on stand-ins: the hand-over register map, dense numbering, a landing mismatch in each direction, the device server's register flow (provenance), the join twin on the recorder, `--serve --keyspec` end to end with `Program.compile` stubbed | as named; `serve_session_prepare` (real plans) is box only: `run_box.sh scheck` |
| `test_keys` | `--keys / --keyspec / --keygen` on stand-ins for the device `_pc`; the z-shard chain check | flag → key source, client-only encryption / decryption on a bundle run |

The planner tests run with `--rwkv7-plan` (`--rwkv7-pf-t T`, `--rwkv7-pf-blocks N`, `--rwkv7-pf-dir DIR` keep the
products): e.g. `pytest tests/test_prefill.py -k plan -s --rwkv7-plan --rwkv7-pf-t 32 --rwkv7-pf-blocks 2`. The
batched one: `pytest tests/test_batched.py -k capture_and_plan -s --rwkv7-plan --rwkv7-b-k 32 --rwkv7-b-blocks 2
--rwkv7-b-dir DIR` (also `--rwkv7-b-carry decode`, `--rwkv7-b-hoist B`, `--rwkv7-b-planenv PAIR=1`). It prints where
the refreshes sit (`plan_census`).

The 12-block CPU counts above come from the fake (float64 arithmetic with runtime level accounting). Device execution across single-conversation decode, chunked prefill (`T = 32`), batched decode (`K = 32` filled + `b2`), continuous batching (`serve`), and client-keyed bundles (`KEYS=bundle`) is verified in the 2026-11-05 GPU measurements summarized in the Status table above.

## Additional Box Commands

```bash
# Default checkout directory (works after `run_box.sh fetch` + `run_box.sh build`)
KEYS=inproc BLOCKS=1 bash tools/rwkv7/run_box.sh all        # smoke: build, check, capture, plan, run on one block + tail
KEYS=inproc bash tools/rwkv7/run_box.sh capture plan run   # 12 blocks
TORCH_PY=python3 bash tools/rwkv7/run_box.sh crosscheck    # harness (torch) vs the numpy reference

# prefill (T = 32 by default; reuses the decode capture's shifts and the decode plan)
bash tools/rwkv7/run_box.sh pfcheck                          # once: the combined session's CPU half, 1 block, T = 4
KEYS=inproc BLOCKS=1 bash tools/rwkv7/run_box.sh pf                      # smoke: capture, plan, capture_pf, plan_pf, run_pf
KEYS=inproc bash tools/rwkv7/run_box.sh pf                               # 12 blocks: 70 prompt tokens = 2 passes + 6 decoded + 4 generated
KEYS=inproc LEVERS=best bash tools/rwkv7/run_box.sh pf                   # the same on PAIR=1 + fused-EvalMod plans, siblings schedule
KEYS=inproc BLOCKS=1 CFTABLE=1 bash tools/rwkv7/run_box.sh run           # A/B the PCMM CF table against PCMM_CF=6 (decode smoke)

# the filled state layout (FILLED_DESIGN.md; every product gets the suffix _filled): batched under b2 + best
B_K=4 LAYOUT=filled bash tools/rwkv7/run_box.sh bcheck
KEYS=inproc LAYOUT=filled PLANOPT=b2 LEVERS=best B_K=32 B_GENERATE=32 RWKV7_SHED_TWIN=1 RWKV7_LEAN_PREFIX=1 POOL_B=160 STC_CACHE_B=1 bash tools/rwkv7/run_box.sh b

# any size (ANYSIZE_DESIGN.md): the whole per-size pipeline (calibrate, fit, scan, gate, capture, plan, keys, runs)
DRY=1 bash tools/rwkv7/size_pipeline.sh .cache/rwkv7/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth 0.1b   # up to the GPU

# client-separated keys for any run (KEYMODE = decode | pf | b), then KEYS=bundle on run / run_pf / batch
KEYMODE=decode bash tools/rwkv7/run_box.sh keyspec && KEYMODE=decode bash tools/rwkv7/run_box.sh keygen
KEYS=bundle bash tools/rwkv7/run_box.sh run

# batched decode (B_K = 32 conversations, lane carries, tools/rwkv7/batch_prompts.txt)
bash tools/rwkv7/run_box.sh bcheck                           # once: the batched runner's CPU half, K = 4, 1 block
KEYS=inproc BLOCKS=1 B_K=4 B_GENERATE=2 bash tools/rwkv7/run_box.sh b    # smoke: capture, plan, capture_b, plan_b, batch
KEYS=inproc bash tools/rwkv7/run_box.sh b                                # 12 blocks, K = 32
KEYS=inproc LEVERS=best bash tools/rwkv7/run_box.sh b                    # PAIR=1 + fused-EvalMod plans, siblings schedule
KEYS=inproc PLANOPT=b2 bash tools/rwkv7/run_box.sh b                     # no refresh inside any conversation's state work (PLAN_B2.md)
# 0.4B (24 blocks): use RWKV7_SHED_TWIN=1 RWKV7_LEAN_PREFIX=1 with filled + b2 + POOL_B=160 STC_CACHE_B=1 (see Quickstart)

# continuous batching (SERVE_DESIGN.md; tools/rwkv7/serve_arrivals.txt: 8 conversations, 3 longer than one pass)
B_K=4 bash tools/rwkv7/run_box.sh scheck                     # once: the serve session's CPU half (1 block, T = 4, K = 4)
KEYS=inproc BLOCKS=1 B_K=4 S_GENERATE=2 bash tools/rwkv7/run_box.sh s    # smoke: every capture and plan, then serve
RWKV7_SHED_TWIN=1 KEYS=inproc B_K=4 bash tools/rwkv7/run_box.sh s        # 12 blocks, K = 4, T = 32, 8 generated each
```

`serve` writes `$OUT/serve4[_b1][_best].jsonl`: `"prefill"`, `"decode"`, `"join"`, `"leave"` and `"pass"` lines with
seconds, one `"token"` line per conversation and position (`mode` prefill / decode / batch, `lane`, top-1 against that
conversation's poly model, `hidden_relErrRms_vs_mirror` against the mirror run on the same schedule), and a summary
(`lanes`: conversation, lane, arrival, join and leave pass; medians per event kind; aggregate tokens per second).

`batch` writes `$OUT/batch32[_b1][_best].jsonl`:
- `"kind": "pass"` lines (seconds per pass, conversations still active);
- one `"kind": "token"` line per conversation and position (`conv`, `phase` prompt / generate, top-1 against that
  conversation's poly model, `hidden_relErrRms_vs_mirror` against the batched mirror);
- a summary: `seconds_per_pass_median` (each conversation's latency per token), `aggregate_tokens_per_s`,
  `top1_agree_per_conv` and `generated_ids`.

`run_pf` writes `$OUT/run_pf32[_b1][_best].jsonl`:
- `"kind": "prefill"` lines (seconds per pass and per token);
- one `"kind": "token"` line per position (`"mode": "prefill"` or `"decode"`);
- a summary with `prefill_seconds_per_token`, `top1_agree_prefill` and the decode median.

Products land in `$OUT` (default `$PWD/.cache/rwkv7_out`, or `/root/out_rwkv7` if present): `run[_b1].jsonl` holds one line per token (`top1_agree`,
`logits_relErrRms`, `hidden_relErrRms_vs_poly`, `hidden_relErrRms_vs_mirror`, `seconds`, `poly_sites_out_of_range`)
and a summary line; `run*.log`, `capture*.log`, `plan*.log` and `session.log` sit beside it.

## Upstream file touched

`python/perseus/plan/placer/planner.py`: the per-stage summary's log formatted `predicted_rel_err` with `:.3g`, which
fails when a stage has no placed refresh (the tail here). Now it prints `-` in that case. The log line only; no
planning logic changed.

## License

This directory and `tools/rwkv7/` are under the MIT license (`LICENSE` here; each file carries `SPDX-License-Identifier: MIT`).
Any commercial use of these files as part of Perseus (github.com/gladia-research-group/perseus) requires prior written
permission from Alessandro Zirilli or Davide Marincione, under Perseus's Business Source License 1.1; on their own the files are MIT.
