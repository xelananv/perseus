# fhe-ssm on `perseus.impl` (upstream `speedrun`), with its speed axes as flags

`python/perseus/examples/fhe_ssm/` is the fhe-ssm recurrent language model (tag `pbd430a`: 24 blocks,
d = 1,024, d_ffn = 4,096, an LTI diagonal scan, learned polynomial gates, Newton RMSNorm, two carried
values per block) on Perseus's leaf-primitive ops protocol, in the shape of `perseus/examples/gpt2`, with
the prefill-speed levers of the research record as flags on the model and on the driver. It is a port of
the fork's `examples/fhe_ssm_from_primitives` (branch `fhe-ssm-port-dev`) onto this branch's Python layer:
the linears are upstream's `prefill.DiagLinear` and `impl.pcmm`, the fake is upstream's Shadow in the
runtime's mode, the capture is upstream's recorder. The checkpoint, the Newton seeds and the config are
the fhe-ssm repository's (`pbd430a_weights.npz` sha256 `3ec3f1d8…784b`, `pbd430a_seeds.json`,
`pbd430a_config.json`; also on Hugging Face, `xelananv/fhe-ssm-pbd430a`).

By default it serves 64 conversations per ciphertext at ring 2^16: the filled layout with one conversation
on each axis of a complex slot (`--layout filled --payload pairs --pair-route split`). It runs on the numpy
fake (CPU, no GPU, the runtime's level accounting) and on the capture recorder (graphs for the planner);
a planned device run is the next step (below, "What needs a GPU").

## The model

    time-mix     n  = rmsnorm(h)                          norm.rmsnorm (Newton from a constant seed)
                 um = mix (.) n + (1 - mix) (.) n_prev     the token shift (n_prev carried)
                 x  = W_in um                             linear (g1 folded in)
                 S  = a (.) S + (1 - a) (.) x  [+ T(x)]    the scan (S carried); T: the lane-mix term, off
                 z  = c (.) S + dd (.) x
                 g  = z (.) (p0 + p1 z + p2 z^2)
                 h  = h + W_out g
    channel-mix  n2 = rmsnorm(h)
                 k  = W_k n2 ; r = W_r n2                 one prepared input, two linears
                 act  = q0 + q1 k + q2 k^2 + q3 k^3
                 gate = r0 + r1 r + r2 r^2
                 h  = h + W_v (act (.) gate)
    tail         reply = g_out (.) rmsnorm(h)             decrypted by the client, tied head in plaintext

The reference is `spec_decode/plain_recurrent.py` of the fhe-ssm repository. `ref.ExactModel` is that
forward in float64 with the closed-form rsqrt; `ref.MirrorModel` runs the port's own kernels on
`NumpyOps` rows (the same Newton, no slots). So `fake − mirror` is a layout or protocol error and
`mirror − exact` the Newton approximation (or, with PCMM, the weight rounding: the mirror rounds the
matrices PCMM would, `geometry.pcmm_quantize`).

Folds, all exact: the norm gains `g1`, `g2` into the rows of `W_in`, `W_k`, `W_r`; `alpha_res` into
`W_out`; the decay as the plaintexts `a` and `1 − a`; the norm's frame constants on two lane masks
(`sqrt(sigma/d)` for the statistic, `sqrt(sigma)` for the output, so the normalized value has unit rms);
the first Newton step from the constant seed is affine in the statistic; on split pairs the split's factor
2 (`c`, `dd`, `W_k`, `W_r` halved, the norm's masks at gain 0.5); the power-of-two scale folds of
`scales_pbd430a.json` (`gate_tm`, `act_cm`, `gate_cm`: the small values a bootstrap refreshes lifted into
its band, each folded into constants the block multiplies anyway and undone in the next linear).

## Run on the fake

From the checkout, with `PYTHONPATH=python` and this checkout's `.venv`:

```bash
# the tests (CPU, a few seconds; the model-shape ones need ~1 GB and a minute)
python -m pytest -q -m 'not gpu and not slow_shape' python/perseus/examples/fhe_ssm/tests
python -m pytest -q -m 'slow_shape' python/perseus/examples/fhe_ssm/tests
python python/perseus/examples/fhe_ssm/lint.py --check          # ruff (ruff.toml) + mypy (mypy.ini)

# ticks on the fake, every axis a flag (--help names every default)
python -m perseus.examples.fhe_ssm.run_fake --weights pbd430a_weights.npz --seeds pbd430a_seeds.json \
    --config pbd430a_config.json --tokens 2 --blocks 24 [axis flags] --out report.json

# the model-shape arms, one process each under a memory guard (what the table below was made with)
FHE_SSM_WEIGHTS=… FHE_SSM_SEEDS=… FHE_SSM_CONFIG=… MEMGUARD="tools/memguard.sh 3" \
    bash python/perseus/examples/fhe_ssm/tests/model_shape_runs.sh OUT [arm …]
```

Each conversation gets its own random token per tick; the references run all of them as rows. Per tick
the driver prints `relErrRms_vs_mirror`, `relErrRms_vs_exact` (and the worst conversation's),
`bootstraps = reactive + hints + carry + pcmm_external` with `bootstraps_by_step`, `mult_cc`, `mult_pt`,
`mult_scalar`, `rotations`, `conjugations`, `out_level`, `state_level_max`, `model_s`, `peak_rss_gb`.
The report records the axes, the level rules, the norm constants, the perseus commit and the sha256 of the
weights, seeds, config and scales. The times are the fake's own and say nothing about a GPU.

## Flags and defaults

| flag | values (default first) | what it does |
|---|---|---|
| `--newton-form` (`--form`) | `perseus`, `depth2` | the Newton step `y <- y (1.5 - 0.5 v y^2)`: Perseus's `inv_sqrt_newton` (three levels per step) or `1.5 y + ((-0.5 v) y) y^2` (two levels). Same polynomial. |
| `--newton-sites` | `served`, `per-site` | served: seed fraction 0.1, at least 12 steps at every site; per-site: 0.3 / 9 at the 48 in-block sites, `ln_out` kept at 0.1 / 12 (`norm.site_kwargs`) |
| `--pack-stats` | `off`, `halves` | one Newton chain for a split pair's two halves at every norm site (`norm.rmsnorm_packed`): residue masks merge the two statistics, one rotation per half scatters `y` back |
| `--linear` | `diagonal`, `pcmm-trunk`, `pcmm` | every matrix through `prefill.DiagLinear`; `W_in`, `W_out`, `W_v` through `impl.pcmm` with `W_k`, `W_r` diagonal; every matrix through PCMM |
| `--pcmm-wbits` | `16`, `24` | PCMM's integer weight width (2 or 3 int8 digit planes) |
| `--diag-hint` | `1`, `2` | headroom (product units) of the diagonal linear's input hint: 1 = the reference circuit (the fork's: a `headroom(1)` hint, the linear's one product fits under the ceiling); 2 = upstream's `diag_apply` headroom rule (the input refreshed one level earlier) |
| `--lane-mix` | `off`, `random` | the cost-only lane-mixing term per block (`lanemix.py`); no trained kernel, so count only |
| `--diag-source` | `values`, `generated` | where the diagonal linears' plaintexts come from (`diaggen.py`): slot vectors, or gathered from the matrices with the device encoder's index arithmetic (the fake: the same reply bit for bit; the capture: the same graph but each diagonal's `args.pt`). A driver flag (`run_fake`, `run_planned`): captures and plans are shared; "The device diagonal encoder" below |
| `--levels` | `fork`, `preset` | ceiling / landing: 46 / 34 (the fork's session, whose fake counts its C++ GPU runs reproduced), or the n32 GPT-2 preset's 44 / 32 (this runtime's); `--limit`, `--bts-out` override |
| `--rules` | `runtime`, `stock` | the fake's level rules: the runtime's (the Shadow's `fakecap` mode) or `perseus.impl.fake` as shipped (a rescale charged at once: refreshes one product earlier) |
| `--layout`, `--payload`, `--pair-route` | `filled`/`decode`, `pairs`/`real`, `split`/`lane` | 64 conversations as split pairs (default), 64 lane-wise, 32 real, or one sequence in the cachemir decode packing |
| `--scales` | `shipped`, `none`, a file | the power-of-two folds |
| `--conversations` | the capacity | fewer are parked (`geometry.park`: copies of conversation 0, dropped on decode) |
| `--capture DIR`, `--capture-tick` | —, `1` | run on the capture recorder and write `block_<b>/graph.json` per stage at that tick (tick 0's carried state is zero) |
| `--ms-norm-k`, `--newton-frac`, `--newton-min-iters` | 5.5, 0.1, 12 | the norm frame and the served seed (the in-block values under `per-site` are fixed) |
| `--no-carry` | | skip the canonical carry (with a huge `--limit` the levels then measure the depth) |

Every axis off reproduces the reference circuit op for op (`tests/test_axes.py::test_every_axis_off_is_the_reference_circuit`).

## Layouts

**Filled** (`filling.py`, `geometry.FilledGeometry`): channel `i` of conversation `k` at slot `i t + k`,
`t = N / d = 32` at ring 2^16 (upstream's dense token layout, `prefill.py:1-4`). A rotation by `j t` moves
every conversation `j` channels along and wraps where each conversation wraps, so the diagonal method
needs no lo/hi halves. Each matrix is `n_in x n_out` blocks of `d x d` (one `1024 -> 4096` piece would cap
the batch at `N / 4096 = 8`); a d_ffn-wide value is four ciphertexts. The norm's statistic is a
per-conversation rotate-and-sum over steps `t, 2t, …, N/2`, tagged `t`-periodic.

**Pairs** (`pairs.py`, `geometry.PairGeometry`): conversation `k` on the real axis and `k + t` on the
imaginary axis, 64 per ciphertext. Linears, plaintext products and bootstraps act on both axes at once;
a ciphertext product does not, so each nonlinear section takes `split` (one conjugation gives
`x + conj x = 2 Re x` and `i (conj x − x) = 2 Im x`, the real kernels run on each, `a + i b` repacks; no
level, the factor 2 rides the masks and the pre-halved operands) or `lane` (`poly.LanePackedOps`, a level
per product for its `1/4`).

**Decode** (`geometry.SlotGeometry`): one sequence per ciphertext in the cachemir packing (channel `i` at
slot `i t`), through `impl.linear` (the GPT-2 decode linear).

## The linears

**Diagonal** (`filling.encode_diag`, `filling.apply_diag`). `prefill.DiagLinear` is the fork's
`filling.EncodedTile` to the letter: `B = N / t = d`, `DiagLinear.split(1024) = (b 32, g 32)` baby and giant
steps, plaintext `(g, j) = repeat(W[(i + g b + j) mod B, i], t)` rolled by `g b t`, input rotations by `j t`
from one hoisted decomposition, one giant rotation per group, `b g = 1,024` products per block, the babies
realized once (`prefill.py:1128-1260`). `DiagLinear.slot_vectors` materializes every plaintext as a full slot
vector — 1.07 GB for one `1024 x 4096` matrix, 3.75 GB per block — so `encode_diag` hands it
`filling.LazyDiagonals`, a table that expands one giant group (8 MB) on `pts[m, g]`, the only access
`DiagLinear._sum` makes (`prefill.py:1192-1200`; `tests/test_filling_fake.py` holds the table to
`slot_vectors`). `apply_diag` is `prefill.diag_apply` (`prefill.py:1223-1248`) with the input hint's headroom
as a parameter: the default, 1, is the reference circuit (the fork's: a `headroom(1)` hint before each
linear, the linear's one product fitting under the ceiling); 2 is upstream's rule (the input refreshed one
level earlier), and at 2 the copy records `diag_apply`'s op stream exactly (tested). The call to
`DiagLinear._sum` is the port's one use of an upstream private; the identity test guards it. Each `(m, gg)` group
goes through `filling.mult_diag_sum`, which is `DiagLinear._sum` unless the session carries a `mult_diag_sum` hook
(`--diag-source generated`, below). Weights are streamed (unkeyed `mult_pt_sum`: nothing cached on the fake); on a
device the diagonals are generated inside each group's fused sum (the device diagonal encoder, below).

**PCMM** (`impl.pcmm`). `Shape(t)` is pinned to `N = 2^16` (`pcmm.py:13-14, 29-41`): nch = 16 · (4096 / 2t)
= 1,024 channels at t = 32, so one port ciphertext is exactly one PCMM block and a matrix must be whole
1,024-wide blocks (the d = 32 synthetic shape cannot run PCMM; its tests use d = 1,024 random weights). On
the fake `pcmm.run` → `Pcmm.apply` (`pcmm.py:481-506`) rotates each input by −1, hints it at the StC-first
cap (44), computes the exact integer products of the filled-pair layout (`_fake`, `pcmm.py:514-528`:
`reshape(nch, t)` of the slot vector is this layout) and lands each output at 26 degree 1 as one external
refresh (`record_external_bootstrap`), then rotates it back by +1; keys `{1, N − 1}` (`pcmm.rot_steps`,
`pcmm.py:777`). The only approximation is the rounding of `W` to integers: one power-of-two scale per
matrix from its largest entry (`PcmmLinear.encode`, `pcmm.py:275-297`), which the mirror applies too
(`geometry.pcmm_quantize`), so `vs_mirror` isolates the layout and `vs_exact` shows the rounding (16 bits:
6.3e-4 on the reply; 24 bits: 2.1e-6). `pcmm-trunk` sends `W_in`, `W_out`, `W_v` (3 outputs per block)
through PCMM and keeps `W_k`, `W_r` diagonal; `pcmm` sends all five (11 outputs per block). The fake lands
PCMM outputs at (26, deg 1) where the capture recorder's default `pcmm_landing` is (26, 2)
(`fakecap.py:841`): both are upstream's statements of a device landing nobody measured on this model.

**The key rule.** On this branch a device PCMM runs planned only: the recorded external refresh (graph v2
`args.external`, `fakecap.py:358-383`) is executed by the Rust runtime's PCMM service
(`crates/pc-py/src/matmul.rs:303` refuses "this session has no PCMM service"), whose ring-switch keys come
from the client's key bundle (`crates/pc-py/src/session.rs:751-770`: the key manifest's `pcmm_t`;
`:868` `Entry::Pcmm`; `crates/pc-client/src/client.rs:190-200`: the client generates them). The
in-process key generation of `pcmm.py` (`KeyRng`, `make_keys`, `pcmm.py:156-254`: from the secret key on the
evaluating side, the legacy `core.bts` path) must never run for this model: `geometry.pcmm_guard` refuses a
core that carries `bts` when a PCMM linear is on (`tests/test_axes.py::test_pcmm_is_refused_on_a_core_with_in_process_keys`).
The band shift (`PcmmLinear.shift`, `PF_PCMM_CALIB` → `PF_PCMM_CALIB_IN`) is a device calibration the fake
does not exercise (it is exact whatever the shift).

## The norm

The served circuit runs Newton on `v = sigma (mean(x^2) + eps)` with `sigma = −K b / a` (K = 5.5) read off
the exported linear seed `y0 = a + b ms`, so that `v` and `y = 1/sqrt(v)` are both near 1 inside the loop
(without the frame their ratio reaches 3e4 across the sites, wider than a bootstrap tolerates). The seed
is the constant `frac (a_n + b_n)` and the count `max(exported, min_iters)`: 0.1 / 12 everywhere in the
published run, a range purchase (the final norm reached 19.6 x its calibration top in a free-running
record; 0.3 / 8 diverged there, 0.3 / 9 covers every in-block range). Hence `per-site`: 0.3 / 9 at the 48
in-block sites, `ln_out` at 0.1 / 12.

**Packed statistic** (`norm.rmsnorm_packed`). A split pair's two halves reach a site together and each
half's statistic, after the ladder, is `t`-periodic (one value per lane, repeated at every channel). Half
`j` keeps the channels `i ≡ j (mod K)` (one 0/1 mask product), the `K = 2` masked statistics are added: the
merged ciphertext holds, at channel `i` and lane `l`, half `(i mod K)`'s statistic — it depends on
`(i mod K, l)`, so it is `K t = 64`-periodic (`tag_reduce(V, 64)`; 64 | 512, so it stays on the 512-slot
sparse route). One chain runs; half `j` takes `y` back by the same mask and a rotate-and-sum over channel
steps `t, …, (K/2) t` (`log2 K = 1` rotation, a layout key). Price per site: +1 level before the chain (the
merge mask), +1 on the output in the perseus form and +2 in depth2 (the chain's output carries a pending
rescale there, realized by the mask product), 2 + 2 masks, 2 rotations, −33 ct×ct (one chain of 11 steps).
Measured on one site at the model shape (`tests/test_axes.py::test_packed_norm_equals_separate_to_rounding`
at the synthetic shape, identical numbers): packed equals separate to ≤ 1e-14 relative, output level 38 → 40
(perseus), 36 → 40 (depth2), eager refreshes 30 → 17 (perseus), 28 → 16 (depth2).

## The lane-mix term

For `x = x_e + i x_o` and the lane offset `s`, `y_s = rot(x, −s)` (lane `j` reads lane `j − s` on both
axes), `T(x) = Σ_s P_s ⊙ y_s + P'_s ⊙ conj(y_s)`: two complex plaintext products per offset do what one
cannot (apply one map to the real axis and another to the imaginary one). Baby-step giant-step over the 32
lanes, `s = 8 g + b`: the 7 baby rotations of `x` and of `conj x` from one hoisted decomposition each,
group `g` multiplies the babies by `P~[g, b][i t + q] = P_{8g+b}[i, q + 8 g]` (zero where `q + 8 g ≥ t`, so
the giant rotation does not carry a group's sum into the next channel's lanes, and zero on the lanes
`j < s` that have no source), one giant rotation by `−8 g`. Per block: 64 plaintext products, 14 hoisted +
3 giant rotations, 1 conjugation, depth 1 (the term lands at the level of the scan's own `b ⊙ x` product);
keys `−1 … −7, −8, −16, −24` (added to the band when the axis is on). The kernel is random (no trained
one exists): the mirror applies the same term on rows (`LaneMixKernel.rows`), so `vs_mirror` still checks
the slots, and `vs_exact` is meaningless with the axis on (the report says so).

## The fake's rules and the levels

`perseus.impl.fake.FakeFhe` runs the Shadow in its `fake` mode (`fake.py:76`): a rescale charged at once
(d1@34 × d1@34 → L36 d1). The runtime leaves it pending; the Shadow's `fakecap` mode (`fakecap.py:134`)
has those rules, and `fakes.RuntimeFakeFhe` is `FakeFhe` in that mode: a product lands at degree 2 at its
inputs' effective level (d1@34 × d1@34 → L34 d2) and the reactive refresh fires on that nominal level
reaching the ceiling; a sum keeps the deeper operand; a plaintext add realizes; conjugation, `mult_i` and
rotations are free; a bootstrap lands at degree 2 at the dense landing of `quirks.fakecap_landings`; a hint
fires on the effective level (`tests/test_fhe_ssm_fake.py::test_runtime_rules_are_the_shadows_fakecap_mode`).
The one rule `FakeFhe` leaves out, the reactive check after a ciphertext subtraction, the subclass adds.

The landing. The fork's fake landed every refresh at 34 with the rescale pending (effective 36), and the
fork's GPU runs on its C++ runtime counted the fake's 1,968 refreshes per tick (research record
`MEASURE_NIGHT_20261012.md`), so `--levels fork` (46 / 34) is the session validated for this model on that
runtime, and it reproduces the fork's counts here (below). This branch's n32 preset says ceiling 44 and
`fakecap_landings` "0:32" (effective 34): `--levels preset`. That is the convention of this branch's own
route table (`crates/pc-shadow/src/route.rs:93-103`: a dense refresh in the shipped order lands at 34,
entry 46, ceiling 44); the fork's 36 is the C++ runtime's. The two settings give identical counts (the whole
chain shifts by one level: out level 38 under `fork`, 36 under `preset`, state 34 vs 32). A
GPU session on this runtime must measure its own landing before any count here is quoted as the device's.

## Measured on the fake (published weights)

24 blocks, ring 2^16, 64 split pairs, shipped scales, the runtime's rules, 2 ticks each (every count
identical on both ticks), one process per arm under a 3 GB guard on a 16 GB Mac (`model_shape_runs.sh`;
reports in the research record `speedrun_port_20261027/runs_A/`). Counts per tick. First the axes against
the default, which is the reference circuit:

| arm | refreshes | reactive / hints / carry / PCMM | by step | ct×ct | ct×pt | rotations | conj | vs mirror | vs exact (worst conv.) | out lvl | s/tick | RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **default** (every axis off: the fork's reference circuit) | **1,968** | 1,832 / 88 / 48 / 0 | tm.norm 808, cm.norm 808, tm.gate 142, cm.gate 40, cm.v 88, carry 48, tail 34 | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 (1.3e-14) | 38 | 25.9 | 0.83 |
| `--rules stock` | 1,973 | 1,837 / 88 / 48 / 0 | tm.norm 812, tm.gate 143, rest as the default | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 | 38 | 24.8 | 0.88 |
| `--levels preset` (44 / 32) | 1,968 | 1,832 / 88 / 48 / 0 | as the default | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 (1.3e-14) | 36 | 29.1 | 0.76 |
| `--newton-form depth2` | **1,708** (−260, −13 %) | 1,660 / 0 / 48 / 0 | tm.norm 764, cm.norm 768, tm.gate 96, tail 32 | 4,294 | 345,265 | 14,372 | 265 | 5.9e-15 | 5.8e-15 | 36 | 33.9 | 0.73 |
| `--newton-sites per-site` | 2,071 (+103) | 2,000 / 23 / 48 / 0 | tm.norm 574, cm.norm 576, cm.gate 768, tm.gate 52, tm.out 23, tail 30 | 3,430 | 345,265 | 14,372 | 265 | 5.8e-15 | 5.9e-15 | 44 | 36.1 | 0.82 |
| `--pack-stats halves` | **1,423** (−545, −28 %) | 1,375 / 0 / 48 / 0 | tm.norm 500, cm.norm 502, tm.gate 168, cm.gate 184, tail 21 | 2,677 | 345,461 | 14,470 | 265 | 6.0e-15 | 5.8e-15 (1.1e-14) | 40 | 30.4 | 0.83 |
| `--linear pcmm-trunk --pcmm-wbits 24` | 1,710 (72 external) | 1,470 / 120 / 48 / 72 | tm.norm 720, cm.norm 720, tm.out 24, cm.v 96, tail 30 | 4,294 | 197,809 | 7,892 | 265 | 4.1e-15 | 2.1e-6 (6.2e-6) | 38 | 25.2 | 1.01 |
| `--linear pcmm` (16 bits) | 1,806 (264 external) | 1,470 / 24 / 48 / 264 | tm.norm 720, cm.norm 720, tm.out 24, tail 30 | 4,294 | 1,201 | 1,436 | 265 | 3.6e-15 | 6.3e-4 (2.1e-3) | 38 | 8.6 | 0.90 |
| `--lane-mix random` | 1,968 (+0) | as the default | as the default | 4,294 | 346,801 | 14,780 | 289 | 5.3e-15 | 0.12 (meaningless) | 38 | 33.7 | 0.86 |

The same arms with `--diag-hint 2` on (upstream's `diag_apply` headroom rule), the default's own levels:

| arm (`--diag-hint 2`) | refreshes | reactive / hints / carry / PCMM | by step | ct×ct | ct×pt | rotations | conj | vs mirror | vs exact | out lvl |
|---|---|---|---|---|---|---|---|---|---|---|
| `--diag-hint 2` alone | 1,713 (−255) | 1,522 / 143 / 48 / 0 | tm.norm 720, cm.norm 720, tm.gate 52, cm.kr 24, cm.v 96, tm.in 23, tail 30, carry 48 | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 | 44 |
| + `--levels preset` | 1,713 | as above | as above | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 | 42 |
| + depth2 | 1,804 | 1,660 / 96 / 48 / 0 | tm.norm 764, cm.norm 768, tm.gate 96, cm.v 96, tail 32 | 4,294 | 345,265 | 14,372 | 265 | 5.9e-15 | 5.8e-15 | 36 |
| + per-site | 2,071 | 2,000 / 23 / 48 / 0 | tm.norm 574, cm.norm 576, cm.gate 768, tm.gate 52, tm.in 23, tail 30 | 3,430 | 345,265 | 14,372 | 265 | 5.8e-15 | 5.9e-15 | 44 |
| + pack-stats | **1,172** | 1,028 / 96 / 48 / 0 | tm.norm 454, cm.norm 456, tm.gate 99, cm.v 96, tail 19 | 2,677 | 345,461 | 14,470 | 265 | 6.0e-15 | 5.8e-15 | 36 |
| + pcmm-trunk 24 | 1,710 | 1,470 / 120 / 48 / 72 | as at hint 1 | 4,294 | 197,809 | 7,892 | 265 | 4.1e-15 | 2.1e-6 | 38 |
| + pcmm 16 | 1,806 | 1,470 / 24 / 48 / 264 | as at hint 1 | 4,294 | 1,201 | 1,436 | 265 | 3.6e-15 | 6.3e-4 | 38 |
| + lane-mix | 1,713 | as `--diag-hint 2` alone | as `--diag-hint 2` alone | 4,294 | 346,801 | 14,780 | 289 | 5.3e-15 | 0.12 (meaningless) | 44 |

Arithmetic behind the deltas (all from the reports): the fork's formulas hold on the default — rotations
24 × (558 linear + 40 norm) + 20 = 14,372, conjugations 24 × 11 + 1 = 265, products 336 blocks × 1,024 =
344,064 of the 345,265 (the rest: 24 × (6 + 2 × 22) + 1 = 1,201 per-channel constants); per-site saves
3 steps × 3 ct×ct at 96 chains (4,294 − 3,430 = 864); pack-stats saves 33 ct×ct at 49 chains (1,617), adds
4 masks (196) and 2 rotations (98) per site; pcmm-trunk removes 6 blocks × 1,024 × 24 = 147,456 products and
24 × (62 + 62 + 155 − 9) = 6,480 rotations and adds 3 × 24 external refreshes; pcmm leaves the 1,201
constants and 24 × (40 + 19) + 20 rotations; the lane mix adds 24 × 64 products, 24 × 17 rotations and 24
conjugations and no refresh.

What the counts mean. The eager refresh count is the fake's reactive count under the runtime's rules,
not what a plan places. The depth2 form saves 13 % on the reference circuit (the figure the research record
measured on the fork, ROUTE1 §4) and loses 5 % under upstream's hint rule (the norm outputs land at 36
instead of 44 and the gates refresh more); the per-site Newton saves 864 ct×ct but moves the channel-mix
norm's output level, after which the gate fires 32 reactive refreshes per block under either rule (ROUTE1 §5
saw the same sensitivity and predicts −31 % planned); pack-stats cuts the count under both rules (−28 %,
−32 %), as the record's route-5 check measured per site (30 → 17); upstream's hint rule alone moves the
count 1,968 → 1,713 by landing every linear's output one level lower, after which the channel-mix gate fits
without its 40 reactive refreshes; PCMM's count does not depend on the hint rule (its own cap hints the
inputs); the lane-mix term never adds a refresh. The planned count per setting needs one capture
(`--capture`, below) and one plan.

## How each piece is verified

Two independent ways on edge cases (`tests/`; `VERIFICATION` of the fork, carried over and extended):

| code | edge cases | check 1 | check 2 | test |
|---|---|---|---|---|
| `pairs.split` / `repack`, `PairLaneOps.add_pt` | complex, real-only, imaginary-only, zero payloads; a pending rescale | numpy, exactly | the round trip; each axis | `test_filled_model.py` |
| `geometry.park`, rows in / out | 1, 2, t − 1, t, t + 1, 2t rows; bad shapes | the copies; the round trip | the layout as a transpose | `test_filled_model.py` |
| `LazyDiagonals`, `apply_diag` | 1→1, 1→4, 4→1 blocks; the hint at 1 and 2 | `DiagLinear.slot_vectors`, `x @ W` | `prefill.diag_apply`'s op stream and values, the textbook diagonal method | `test_filling_fake.py`, `test_filled_model.py` |
| the norm, the gates, a block, ticks, the three arms, partial batches, parking | both ends of the seeds' window; 1, t + 1, 2t − 1 conversations | the mirror | the closed form | `test_filled_model.py`, `test_fhe_ssm_fake.py` |
| the runtime's rules | every rule, the hint table, the landing, the strict error | the Shadow's recipes | the fork's documented rules | `test_fhe_ssm_fake.py` |
| every axis off | — | the default's op stream equals the explicit-default one | the fork's count formulas | `test_axes.py` |
| `depth2` | — | the mirror in the same form | the perseus form (one polynomial); the depth per step | `test_axes.py` |
| `per-site` | in-block and `ln_out` | the site constants | the mirror / the closed form; the ct×ct saved | `test_axes.py` |
| `pack_stats` | both halves, k = 1; parked lanes; both forms | the separate norms (≤ 1e-14) | the closed form; the merged period; the levels and refreshes of the record | `test_axes.py` |
| PCMM | 1→1, 1→4, 4→1 at d = 1,024; 16 and 24 bits; trunk and all | `x @ W_q` bitwise | `x @ W`; the quantized mirror vs the plain one; the external refreshes; the keys; the refusals | `test_pcmm.py`, `test_axes.py` |
| lane mix | the no-source lanes; a wrong t | slots vs rows (5e-15) | the count formulas; the band | `test_axes.py` |
| the capture | every stage at tick 1 only | the graphs' op census equals the fake's counts | the carried-state names; the refreshes and levels equal the fake's | `test_capture.py` |
| `--diag-source generated` | every plaintext of 1→1, 1→4, 4→1 at the synthetic shape; 64 random jobs at 2^15 slots, B = 1,024 | `LazyDiagonals`, `DiagLinear.slot_vectors`, one element by hand | the fake's reply bit-identical to `values` (a flipped roll breaks it); the capture equal to the values capture but `args.pt`; the stock plan binds with one sum per group and 808 launches fewer per block | `test_diag_source.py` |
| the driver | the four arms; the axes in the report; refusals | the errors it computes | its counts against independent tallies; the hashes | `test_run_fake.py` |
| the scale folds | exponents 0, mid, 14; brute force | bit-for-bit factors | the whole model with folds vs the mirror and the unfolded closed form | `test_scales.py` |

156 tests (140 at the synthetic shape, 16 marked `slow_shape` at ring 2^16), plus the planned driver's: 14
CPU (`tests/test_planned_contract.py`, one of them `slow_shape`: a 1-block PCMM capture at the model's width
planned and bound) and 3 marked `gpu` (`tests/test_planned_gpu.py`, skipped without the device build); the
device diagonal encoder's CPU half adds 14 (`tests/test_diag_source.py`) and its device half 3 marked `gpu`
(`tests/test_diag_gpu.py`). The whole light suite
(`-m "not slow_shape and not gpu"`) is 184 tests, the `slow_shape` set 19. The
fork's tests of its GPU runner, planner wrapper, Colab build and tick meter are not carried over: those modules
belong to the fork's C++ runtime; on this branch the device path is `impl.driver.PlannedRun` (below). Two
further checks run at the model's shape with the published checkpoint (`tests/test_model_shape.py`, skipped
without `FHE_SSM_WEIGHTS` / `FHE_SSM_SEEDS` / `FHE_SSM_CONFIG`): the default's 1,968 split and op counts,
pinned, and a PCMM capture whose every finish names its linear.

## What needs a GPU

1. **Capture → plan.** `run_fake.py --capture DIR` writes `block_0..24/graph.json` (and `capture_env.json`,
   `capture_config.json`) at tick 1 on the capture recorder at the fake's levels; the planner then runs on
   them (`tools/fhe_ssm/make_plan.sh`). The reference capture at the default settings and `--levels preset`
   (research record `runs_A/cap_ref_preset/`, 25 stages, 716,245 nodes) holds the fake's counts to the op:
   1,968 refreshes (808/808/142/40/88/48/34), 345,265 ct×pt, 14,372 rotations, 265 conjugations, 4,294 ct×ct;
   the recorder takes 538 s per tick against the fake's 29 s. Nothing here has run on a device: the magnitudes
   in a fake capture are the fake's float64 values, and the landing convention is the fake's.
2. **A planned run.** The device path on this branch is `impl.driver.PlannedRun` / `Recorder`
   (`driver.py:444-598`): every stage recorded once on the CPU twin, compiled with its plan, replayed;
   `FheSsmModel` supplies the stages (`token_stages`), its carried state is named for the planner
   (`cache.<name>-lvl=<level>`) and handed over as its K/V (`stage_caches`), `rotation_audit` lists the
   keys the band must hold. The driver is `run_planned.py` / `planned.py` (next section); its setup is
   CPU-verified, its session is not.
3. **PCMM's landing, cap and band shift**, and the refresh landing of this runtime, are measured there;
   until then every level in the table is the fake's.
4. **The planned refresh counts per axis** (the question the axes exist for) are plan outputs, not eager
   counts.

## Running under a plan on a GPU

`run_planned.py` runs the model under a plan on the device path (`planned.py`: `PlannedSsmRun`, an
`impl.driver.PlannedRun`; `SsmRecorder`, its CPU twin's recorder). The model is built on the capture
recorder with the same flags as the capture the plan was cut from (run_fake's model and axis flags, the
same spellings; `--levels preset` is the default here), the probe ticks are recorded on the twin, the
reference tick's stream (`--ref-token 1`: it reads a carried state, as the capture's tick 1 did) is
compiled against the plan with `--schedule program|siblings`, the carried state of every block becomes
two registers (`cache.s`, `cache.n_prev`), the stream is checked static across the probes (no dynamic
step, equal plaintexts), the session opens, the PCMM planes are uploaded by name, every plaintext is
staged once, one untimed warm-up tick runs on random public tokens (it captures the CUDA graphs), then
`--ticks T` timed ticks from zero state, each decrypted and scored against `ref.ExactModel` (`relErrRms`
of the reply, the worst conversation, the tied head's logits and top-1 agreement), the record rewritten
after every tick. `tools/fhe_ssm/run_gpu.sh run ARM` invokes it as
`--weights W --seeds S --config C --levels preset --plan DIR --ticks T --profile --keys inproc|bundle
<the arm's capture flags> [--schedule siblings] --out record.json`.

```bash
# the CPU part on any build: record the twin, compile (check) against the plan, report; no session
python -m perseus.examples.fhe_ssm.run_planned <model flags> --plan PLAN --prepare-only [--lean] --out prep.json
# the key request (the model's needs; on the device build the session's exact pc_client KeySpec too)
python -m perseus.examples.fhe_ssm.run_planned <model flags> --plan PLAN --keyspec keyspec.json
# the client role (device build): a secret and the bundle DIR/keys.pckb; client_secret.bin stays beside it
python -m perseus.examples.fhe_ssm.run_planned --keygen DIR --keyspec-file keyspec.json   # no model flags: the client role
# a timing run with in-process keys (INSECURE: the session holds the secret), the published tokens
python -m perseus.examples.fhe_ssm.run_planned <model flags> --linear pcmm --pcmm-wbits 24 --plan PLAN \
    --ticks 148 --tokens tokens.json [--free-run] --profile --insecure-test-keys --out record.json
# a run on a client's bundle: no secret in the session; the reply decrypted by the client's secret, or exported
python -m perseus.examples.fhe_ssm.run_planned <model flags> --linear pcmm --plan PLAN --ticks 8 --keys DIR --out record.json
```

**Keys.** A device run names its keys or does not run. `--keys DIR` opens the session on the client's
bundle `DIR/keys.pckb` (`Session.from_bundle`: no secret on the server; the bundle must answer this
session's own key request, which the runtime checks key for key at import; the driver checks the plan
contract, the chain and the key kinds first: the layout's 62 rotation steps (`filling.rotation_steps(2^15,
1024)`: 31 baby steps `j t`, 31 giant steps `k b t`, the norm ladder's `t … N/2` among them) plus the
conjugation and relinearization keys, and PCMM's ring-switch key set for `t = 32` when `--linear pcmm*`).
With `DIR/client_secret.bin` (written by `--keygen`) the reply is exported as a frame and decrypted by
`_pc.Client` on the client's side of the process, so a bundle run is scored too; without it the frames
land in `<out>.replies/` and the fidelity fields are absent. `--keys bundle` reads `DIR` from
`$FHE_SSM_KEYS`. `--insecure-test-keys` (= `--keys inproc`, what `run_gpu.sh` passes for `KEYS=inproc`; `KEYS` has no
default there, and its `keyspec` / `keygen` steps make a bundle)
generates the keys in process and keeps the secret in the session: a timing run only, loudly marked in the
log and the record (`keys.mode = "insecure-inproc"`). `--keyspec` writes the request: the model's own
needs on every build, and on the device build the session's `Session.key_manifest` (the exact
`pc_client::KeySpec`, after the programs are compiled), which `--keygen` answers.

**What `--profile` measures.** A device fence (`fhe.device_sync()`) after every stage's `Program.run`, so
each stage has a wall (`stages[].wall_s`, with the Rust replayer's `host_us` / `launch_us` and whether the
stage was captured, replayed or run eagerly); the tick's `seconds` then includes the fences, an
attribution and not the production wall (issue is serialized with device work). `--profile-every K`
fences every K-th tick only, so one run holds both kinds (`fenced` per tick). Every tick also records the
deltas of `Session.planned_stats()` (runs, host and launch microseconds, graph captures / replays / eager
runs, refresh pairs as one or split, fused product-adds, StC-first cache misses), the free device memory
(`fhe.mem_info()`) and the reply's level; the refresh count per tick is the compiled programs'
(`n_bootstraps`, the plan's count: the runtime fires every site of a program it runs and keeps no
separate counter). `impl.profile.TimedOps` times Python-issued ops and so measures nothing on this path.
The header holds the build id (commit, the extension's digest, nvcc, the arch), the plan's lines, the
capture md5 check, every flag, the environment's switches, the file hashes and, after setup, the session's
`mem_report`, `setup_seconds`, the registers, the uploaded PCMM linears.

**Refusals.** `--schedule lockstep` and `--streams N > 1`: this port captures one ciphertext set per
stage (k = 1), a lockstep compile would reorder nothing. No `--keys` and no `--insecure-test-keys`. A
bundle directory without `keys.pckb`, or a bundle for another plan contract or chain. A plan whose capture
md5 (`PLAN_CMD.txt`'s `graph=` line) differs from the capture at hand (`--capture DIR`, else the plan's
own path when it exists). A plan stamped under another Config than the twin's (another `--levels`). A
plan whose Config names a chain other than n32 (`CHAIN_PRESET` on its `runtime=` line, e.g. the lean
n32a4: the session, its shadow and its key manifest would build that chain, while the model's plaintext
rows and levels are n32's; `run_planned.session_config`). A model whose stream is not static (never this model). **Diagonal linears on the device path under
`--diag-source values`** (`--linear diagonal` and `pcmm-trunk`): `PlannedRun` keeps one resident plaintext
slot per distinct plaintext (`pt_slots = len(pt_levels) + 8`, `driver.py:731`; a slot is `pt_rows` rows of N
32-bit words, 7 MiB at 28 rows) and the reference circuit reads 344,064 diagonal plaintexts per tick (about
2.3 TiB), so the driver refuses above `--pt-budget-gb` (40) with the count; `--diag-source generated` runs
them on the device diagonal encoder (below). `--diag-source generated` on a device build without the
encoder (`Session.register_tile`). The `--linear pcmm` arms are resident (about 1,200 per-channel
constants per tick) and run.

**What the CPU verifies and what is GPU-only.** On the CPU-only `perseus._pc` (no `Session` /
`Program`), `--prepare-only` records the twin, writes the reference stream as a graph dir and compiles it
through `tools/fhe_ssm/compile_check` (the same `compile_dir` call as `Program.compile`, without the
driver's `live_out` / `input_metas`: the carried inputs take the compiler's seeds), standing
`CheckedProgram`s in for the programs; the report holds the check errors per program, the registers, the
carried names, the static-stream verdict, the plaintext budget, the PCMM declarations against the
finishes, the rotation steps against the plan's scope. `tests/test_planned_contract.py` runs that at the
synthetic shape against a plan the stock recipe makes of a synthetic capture (program and siblings
schedules, lean), the pair packing on the fake against the twin's, the key needs, the recorder's
registration of the named state against upstream's, and the refusals. GPU-only (`tests/test_planned_gpu.py`,
`@pytest.mark.gpu`): the session, `Session.key_manifest`, the bundle import, the PCMM service, the
reply's error on a device, every timing. The twin's recording at the model shape takes minutes per tick
(structure only; the capture with values took 538 s): the setup records two ticks. On a small machine
`--lean` (with `--prepare-only`) keeps plaintext digests instead of values, streams each block's graph of
the reference tick to the work dir at stage end and reduces the other tick's blocks to a stream digest
(`stream_info`: the same identity `diff_block` compares): a block of the model holds about 280 MB of node
dicts, 25 stages and two ticks would not fit; a lean prepare of the reference plan (the capture of
`runs_A/cap_ref_preset`, its stock plan of 637 refreshes) on a 16 GB Mac: 25 programs, 716,245 ops (the
capture's node count), 637 refresh sites, 0 check errors, 0 dynamic steps, 73 registers, 886 s under a 1.5 GB
footprint. A PCMM arm records with values on every path (`impl.pcmm`'s fake computes its products on the
slot vectors).

## The device diagonal encoder

Under `--diag-source values` every diagonal plaintext is a `PtStore` slot: 14 tiles x 1,024 diagonals = 14,336 per
block, 344,064 per tick, each (54 - level) rows of 2^16 32-bit words: 70.0 GiB per block at 20 rows, 2.30 TiB at the
driver's 28 [derived; `BATCH3_ENCODER_BRIEF_20261029.md` §0-1 in the research record]. No store policy fixes that:
the keys are per plaintext. Upstream's answer for its streamed GPT-2 weights is the fused op `CoeffSum` (the twin
records the loop `mult` + `add_inplace` over named rows, `examples/llama/model.py` `stream_sum_recorder`; the compiler
fuses it, `crates/pc-program/src/compile/stream.rs`; the device expands float32 rows inside the op). The encoder is
that op's sibling: the plaintext's source is the linear's float64 matrix resident on the device (the tile) instead
of host rows, and each diagonal is gathered from it, encoded and multiplied inside the sum. Bytes over PCIe per tick:
none; device memory: the tiles (24 x 14 x 1,024^2 x 8 B = 2.625 GiB; 1.5 GiB on `pcmm-trunk`) and a scratch of at
most 128 plaintexts in flight.

**The contract.** A generated plaintext is `<linear>#<n>.<m>.<gg>.<jj>`: `<linear>` the port's `b<b>.<kind>`, `(n, m)`
the `B x B` block of the linear's `(d_in, d_out)` matrix (`B = n_slots / t`), `gg` the giant and `jj` the baby index,
`k = gg b + jj`. Slot `v` holds `W[n B + ((i + k) mod B)][m B + i]` with `i = ((v - gg b t) mod n_slots) / t`
(`DiagLinear.slot_vectors`' `np.roll(np.repeat(blk[(i + k) % B, i], t), gg b t)`: the roll moves elements to higher
indices), a real payload. `filling.diag_slot_index` / `diag_gather_reference` are that arithmetic in numpy (and
`crates/pc-program/src/compile/diag.rs` `TileInfo::slot` / `addr` in Rust); both are held to `LazyDiagonals` and to
upstream's `slot_vectors` (`tests/test_diag_source.py`, and the crate's unit test).

**The twin and the capture.** `diaggen.use_source(inf, "generated")` installs the hook `filling.mult_diag_sum` hands
each `(m, gg)` group to. On the capture recorder (`diaggen.diag_sum_recorder`, the sibling of `stream_sum_recorder`)
it records the loop `FheOps.mult_pt_sum` records (a `mult` per term, n outer and jj inner, `add_inplace` into the
first) with each plaintext a `Pt(src=("diag", name))` at the ciphertext's effective level, degree 1, tagged from its
full vector in every mode (`pt_tag` reads the values), holding the vector only when the recorder keeps values. So a
generated capture equals the values capture in every op, var name, level, tag and magnitude, and differs only in
each diagonal's `args.pt` (`kind = "diag"`, `name`); the values capture is unchanged (md5 at the synthetic shape:
the same before and after this change). The planned twin keeps a `diaggen.GeneratedPt` (the name) for each, never
its 256 KiB vector. On the numpy fake the hook builds the group's vectors with `diag_gather_reference` and multiplies
them through `ops.mult_pt_sum` as `DiagLinear._sum` does: the reply is bit-identical under both sources only if the
index arithmetic is right (a flipped roll changes it; tested).

**The compiler** (upstream files, `tools/fhe_ssm/UPSTREAM_CHANGES.md`): `args.pt.kind = "diag"` is
`PtForm::DiagGen`, its name refused unless it parses; the fuse pass turns every generated product into a one-term
`CoeffSum` and merges partial sums of one form only (one `CoeffSum` per group: 352 per block at the model shape, 320
of 32 terms and `W_v`'s 32 of 128; 44 per block at the synthetic shape, 40 of 8 and 4 of 32); the check refuses a
sum mixing sources and any generated plaintext outside a sum. The bind keeps generated plaintexts out of the store
(`BoundPt.in_sum`), binds a diagonal sum only when the session declares every tile it reads, and parses each term
into a gather job; a stream-only program is bound exactly as before. `tools/fhe_ssm/compile_check` prints the census
per block (sums by source, term histograms, fused vars, plaintext forms, store references, store references named
`<linear>#...`).

**The driver.** `--diag-source generated`: the plaintext budget, `pt_levels` / `pt_slots` and `stream_info` leave
generated plaintexts out (as `StreamedPlannedRun` leaves streamed rows out); the session is built with
`diag_tiles={name: (d_in, d_out, t)}` (`diaggen.diag_tiles`: five linears per block on the all-diagonal arms, `W_k` and
`W_r` on `pcmm-trunk`; the lane count rides with the shape because the tile side `B = n_slots / t` is the layout's, not
the matrix's), and after `complete_setup` each tile's folded matrix (`LazyDiagonals.W`, C-contiguous float64) is
uploaded with `Session.register_tile`. `key_needs` reads no plaintext and is the same under both sources. A build
without `Session.register_tile` refuses `generated` (`run`, `--keyspec`) before any session opens; `values` and the
PCMM arms run on it as before.

**The runtime** (upstream files, listed). Two kernels (`crates/pc-kernels`, ABI 17): `diag_gather` writes job j's
N/2 slots `(W[at(v)], 0)` (thread = slot, coalesced writes, the address `(n B + row) d_out + m B + i` in 64 bits, the
mod by `n_slots` a mask and the mod by `B` one subtract) from a row of a device job table (the tile's address and
geometry, `(n, m, gg, jj)`), and `encode_many` is the `encode` kernels with the job on grid.y, so `encode` is
`encode_many` at batch 1 and a batch equals its single encodes word for word. pc-gpu's `DiagGen` (sibling of
`CoeffRowsDevice`) runs a sum in chunks of `host.diag_batch` jobs (default 32, at most 128): gather, encode_many, one
NTT of the chunk's rows (a per-top row list carved at setup), then `Ops::mult_pt` for the first term and
`Ops::mult_add_many` for the rest, `PtStore::sum_with`'s order. It pushes nothing through the constant ring (the
degree factor is a resident table of ones; every term is degree 1), so a graph capture leases no slot; the job rows
are written at bind (`Program.bind` after the bind proper) and a captured sum bakes fixed addresses only. The session
declares the tiles at construction (`Session(diag_tiles=...)`, also `from_bundle` and `key_manifest`), carves them
with the scratch, the job table and the row lists in `setup()` (`mem_report`: `diag.tiles`, `diag.scratch`,
`diag.jobs`, `diag.lists`), uploads each matrix with `register_tile` (C-contiguous float64 `(d_in, d_out)` only, legal
after `complete_setup`), and refuses a run while a declared tile was never uploaded. `Session.diag_stats()` counts the
issued sums and the jobs whose 2^61 split fired (0 expected: `|W| < 1` keeps every coefficient below 2^55.5);
`Session.diag_probe(names, level, values)` returns generated rows beside single and oracle encodes (tests). Device
traffic per plaintext is about 36 + 1.25 L MiB (61 MiB at 20 limbs) [derived, `kernels/diag.cuh`].

**On a device.** Nothing of the encoder's CUDA has been compiled or run. Its first device contact is
`tools/fhe_ssm/run_gpu.sh diag-tests` (`RUNSHEET_GPU.md` §2a): the kernel selftest's `encode_many` / `diag_gather`
lines, the crates' GPU tests (`crates/pc-program/tests/diag_gen_gpu.rs`: generated rows against the eager encode of
the host-built vector, `encode_many` against single encodes, a fused sum against `PtStore::mult_pt_sum`, word for
word; then the timings), the port's (`tests/test_diag_gpu.py`: the probe against `filling.diag_gather_reference` over
the registered folded matrices; one block and the model against the exact forward; a perturbed tile, which must leave
the tolerance). `run_planned` takes `--diag-batch` (the chunk), `--diag-probe K` (the probe after setup, recorded in
`setup.diag_probe`) and, for that negative control only, `--diag-perturb TILE:FACTOR` (the tile uploaded scaled, the
exact forward unchanged); the record carries `setup.diag` and each tick's `diag` counters.
Environment the GPU tests and the driver read: `FHE_SSM_REQUIRE_DEVICE=1` makes `tests/test_diag_gpu.py` fail
instead of skipping when the `perseus._pc` build has no `Session.register_tile` (`diag-tests` exports it, so a step
cannot pass with no test run); `FHE_SSM_WEIGHTS` / `FHE_SSM_SEEDS` / `FHE_SSM_CONFIG` and `FHE_SSM_PLAN` (the
model's files and its whole-model stock plan) select the model-shape test (e), which writes its record to
`FHE_SSM_E_OUT` (default: pytest's temporary directory); `FHE_SSM_COMPILE_CHECK` points the CPU build of the driver
at a `compile_check` binary other than the one `tools/fhe_ssm/compile_check/build.sh` builds (`planned.py`
`default_check_bin`). `run_gpu.sh`'s `diag-tests` step exports `FHE_SSM_REQUIRE_DEVICE=1` and passes the model's
files, `plan_ref` and `FHE_SSM_E_OUT` from `DATA` / `OUT`; it never sets `FHE_SSM_COMPILE_CHECK` (its `verify` step
builds the binary where the default looks). By hand, set them as that step does.

## Upstream APIs this package depends on (at d97a6f7)

| what | where |
|---|---|
| the diagonal linear | `perseus/impl/prefill.py:1128-1200` `DiagLinear` (`split` :1137, `slot_vectors` :1142, `_sum` :1192), `diag_apply` :1223-1248, `diag_rot_steps` :1263 |
| PCMM | `perseus/impl/pcmm.py:29` `Shape`, :262-297 `PcmmLinear.encode`, :414 `run`, :432 `Pcmm` (`apply` :481, `desc` :508, `_fake` :514), :773-774 the fake's cap / landing, :777 `rot_steps`; the legacy in-process keys :156-254 (refused) |
| the fake | `perseus/impl/fake.py:22` `FakeCt`, :65 `fake_config`, :75 `FakeFhe` (`MODE` :76, `_product` :105, `bootstrap` :204, `maybe_bootstrap` :244, `bootstrap_hint` :248, `record_external_bootstrap` :259), :264 `FakeInf` (`set_slot_pt` :322, `tag_reduce` :334), :420 `pack_tokens`, :475 `make_fake` |
| the capture recorder | `perseus/impl/fakecap.py:132` `CapFhe` (`MODE` :134, `inner_bootstrap` :291, `bootstrap_hint` :336, `record_external_bootstrap` :358), :617 `CapInf` (`name_ct` :639), :754 `CapCore` (`begin_subgraph_capture` :783, `end_subgraph_capture` :799), :841 `make_session` |
| the driver | `perseus/impl/driver.py:26` `ImplModel` (`set_capture` :124, `load_plans` :135, `enter_stage` :199, `exit_stage` :215, `run_stages` :222, `token_begin` :324, `token_tail` :339, `token_end` :361); `Recorder` :444, `PlannedRun` :598 |
| the ops protocol | `perseus/impl/ops.py:46` `FheOps` (`mult_pt` :226, `mult_pt_sum` :242, `mult_i` :279, `decrypt_slots_complex` :293, `rotate_many` :298, `tag_reduce` :350, `headroom` :374, `bootstrap` :376, `bootstrap_hint` :415, `realize` :422, `pack_tokens` :434), :441 `NumpyOps` |
| Newton and the lane route | `perseus/impl/poly.py:515` `inv_sqrt_newton`, :745 `LanePackedOps` (`lane_square` :719, `lane_mult` :733) |
| the decode layout | `perseus/impl/layout.py:23` `Dims`, :104 `linear_rot_steps`, :236 `encode_bias_vector`, :280 `lane_vec`, :289 `stride_mask`, :458 `ladder_radix_bits`; `perseus/impl/linear.py:15` `EncodedLinear`, :142 `linear`, :147 `linear_multi` |
| the session bundle | `perseus/impl/rt.py:22` `Rt` (`mult_mask` :97, `step` :194) |
| generated diagonals (this branch) | `crates/pc-program/src/compile/diag.rs` (`parse_diag_name`, `TileInfo`, `DiagSource`, `SumSrc`); `perseus/impl/fakecap.py:105` `Pt` (`src`), :33 `pt_tag`, :646 `CapInf._like`; `perseus/impl/driver.py:477-482` `rec_pt_var`; the session's `diag_tiles=` / `register_tile` (the GPU half) |
| the capture contract, the presets | `perseus/plan/contract.py:72` `capture_contract`; `perseus/config/__init__.py:29` `preset` (`_pc.Config.preset("n32_gpt2_decode")`) |
| the PCMM key path on a device | `crates/pc-py/src/session.rs:189, 751-770, 868`; `crates/pc-py/src/matmul.rs:303`; `crates/pc-client/src/client.rs:190-200` |

The port adds files under `python/perseus/examples/fhe_ssm/` and `tools/fhe_ssm/`; the upstream files it
changes as fixes to the base are listed with their reasons in `tools/fhe_ssm/UPSTREAM_CHANGES.md` (among
them `impl/fakecap.py`: the structure-only recorder's one kept slot is a copy, not a slice of the full
product vector, which the slice kept alive at 256 KB per op).
`LADDER_RADIX_BITS=0` is exported before a session or a fake is made (`port_env`): the rotate-and-sum
ladders run plain on both routes (the fake would otherwise hoist radix groups the C++ does not).

## License

This directory is under the MIT license (`LICENSE` here; each file carries `SPDX-License-Identifier: MIT`).
It runs on Perseus, whose own license (the repository's) applies to running this example on it.
