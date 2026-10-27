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
| `--diag-hint` | `2`, `1` | headroom of the diagonal linear's input hint: 2 = upstream's `diag_apply` rule, 1 = the fork's reference circuit |
| `--lane-mix` | `off`, `random` | the cost-only lane-mixing term per block (`lanemix.py`); no trained kernel, so count only |
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
realized once (`prefill.py:1127-1259`). `DiagLinear.slot_vectors` materializes every plaintext as a full slot
vector — 1.07 GB for one `1024 x 4096` matrix, 3.75 GB per block — so `encode_diag` hands it
`filling.LazyDiagonals`, a table that expands one giant group (8 MB) on `pts[m, g]`, the only access
`DiagLinear._sum` makes (`prefill.py:1191-1199`; `tests/test_filling_fake.py` holds the table to
`slot_vectors`). `apply_diag` is `prefill.diag_apply` (`prefill.py:1222-1247`) with the input hint's headroom
as a parameter: at 2 it records `diag_apply`'s op stream exactly (tested); at 1 it is the fork's circuit (a
hint of `headroom(1)` before each linear, the linear's one product fitting under the ceiling). The call to
`DiagLinear._sum` is the port's one use of an upstream private; the identity test guards it. Weights are
streamed (unkeyed `mult_pt_sum`: nothing cached on the fake; on a device `DiagLinear.encode(pinned=…)` and
`mult_coeff_sum` encode them there).

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
from the client's key bundle (`crates/pc-py/src/session.rs:745-764`: the key manifest's `pcmm_t`;
`:862` `Entry::Pcmm`; `crates/pc-client/src/client.rs:190-200`: the client generates them). The
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
`fakecap_landings` "0:32" (effective 34): `--levels preset`. The two give identical counts (the whole chain
shifts by one level; out level 42 vs 38, state 32 vs 34). A GPU session on this runtime must measure its
own landing (agent B's reading of `crates/pc-program` route.rs: dense shipped-order refreshes realize at
34 here, i.e. the preset's convention) before any count here is quoted as the device's.

## Measured on the fake (published weights)

24 blocks, ring 2^16, 64 split pairs, shipped scales, the runtime's rules, 2 ticks each (every count
identical on both ticks), one process per arm under a 3 GB guard on a 16 GB Mac (`model_shape_runs.sh`;
reports in the research record `speedrun_port_20261027/runs_A/`). Counts per tick:

| arm | refreshes | reactive / hints / carry / PCMM | by step | ct×ct | ct×pt | rotations | conj | vs mirror | vs exact (worst conv.) | out lvl | s/tick | RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **reference** (`--diag-hint 1`, the fork's circuit) | **1,968** | 1,832 / 88 / 48 / 0 | tm.norm 808, cm.norm 808, tm.gate 142, cm.gate 40, cm.v 88, carry 48, tail 34 | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 (1.3e-14) | 38 | 25.9 | 0.83 |
| `--rules stock` | 1,973 | 1,837 / 88 / 48 / 0 | tm.norm 812, tm.gate 143, rest as above | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 | 38 | 24.8 | 0.88 |
| default (`--diag-hint 2`) | 1,713 | 1,522 / 143 / 48 / 0 | tm.norm 720, cm.norm 720, tm.gate 52, cm.kr 24, cm.v 96, tm.in 23, tail 30, carry 48 | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 (1.3e-14) | 44 | 26.3 | 0.81 |
| `--levels preset` (44 / 32) | 1,713 | as default | as default | 4,294 | 345,265 | 14,372 | 265 | 6.0e-15 | 5.9e-15 | 42 | 24.9 | 0.84 |
| `--newton-form depth2` | 1,804 | 1,660 / 96 / 48 / 0 | tm.norm 764, cm.norm 768, tm.gate 96, cm.v 96, tail 32 | 4,294 | 345,265 | 14,372 | 265 | 5.9e-15 | 5.8e-15 | 36 | 24.1 | 0.91 |
| `--newton-sites per-site` | 2,071 | 2,000 / 23 / 48 / 0 | tm.norm 574, cm.norm 576, cm.gate 768, tm.gate 52, tm.in 23, tail 30 | 3,430 | 345,265 | 14,372 | 265 | 5.8e-15 | 5.9e-15 | 44 | 23.9 | 0.87 |
| `--pack-stats halves` | **1,172** | 1,028 / 96 / 48 / 0 | tm.norm 454, cm.norm 456, tm.gate 99, cm.v 96, tail 19 | 2,677 | 345,461 | 14,470 | 265 | 6.0e-15 | 5.8e-15 (1.1e-14) | 36 | 23.9 | 0.85 |
| `--linear pcmm-trunk --pcmm-wbits 24` | 1,710 | 1,470 / 120 / 48 / 72 | tm.norm 720, cm.norm 720, tm.out 24, cm.v 96, tail 30 | 4,294 | 197,809 | 7,892 | 265 | 4.1e-15 | 2.1e-6 (6.2e-6) | 38 | 17.0 | 0.99 |
| `--linear pcmm` (16 bits) | 1,806 | 1,470 / 24 / 48 / 264 | tm.norm 720, cm.norm 720, tm.out 24, tail 30 | 4,294 | 1,201 | 1,436 | 265 | 3.6e-15 | 6.3e-4 (2.1e-3) | 38 | 6.0 | 0.97 |
| `--lane-mix random` | 1,713 | as default | as default | 4,294 | 346,801 | 14,780 | 289 | 5.3e-15 | 0.12 (meaningless) | 44 | 25.5 | 0.76 |

The same axes under the fork's hint rule (`--diag-hint 1`), against the 1,968 reference:

| arm | refreshes | reactive / hints / carry / PCMM | by step | ct×ct | ct×pt | rotations | conj | vs mirror | vs exact | out lvl |
|---|---|---|---|---|---|---|---|---|---|---|
| `--newton-form depth2 --diag-hint 1` | **1,708** (−260, −13 %) | 1,660 / 0 / 48 / 0 | tm.norm 764, cm.norm 768, tm.gate 96, tail 32 | 4,294 | 345,265 | 14,372 | 265 | 5.9e-15 | 5.8e-15 | 36 |
| `--newton-sites per-site --diag-hint 1` | 2,071 (+103) | 2,000 / 23 / 48 / 0 | tm.norm 574, cm.norm 576, cm.gate 768, tm.gate 52, tm.out 23, tail 30 | 3,430 | 345,265 | 14,372 | 265 | 5.8e-15 | 5.9e-15 | 44 |
| `--pack-stats halves --diag-hint 1` | **1,423** (−545, −28 %) | 1,375 / 0 / 48 / 0 | tm.norm 500, cm.norm 502, tm.gate 168, cm.gate 184, tail 21 | 2,677 | 345,461 | 14,470 | 265 | 6.0e-15 | 5.8e-15 | 40 |
| `--linear pcmm-trunk --pcmm-wbits 24 --diag-hint 1` | 1,710 | 1,470 / 120 / 48 / 72 | as with hint 2 | 4,294 | 197,809 | 7,892 | 265 | 4.1e-15 | 2.1e-6 | 38 |
| `--linear pcmm --diag-hint 1` | 1,806 | 1,470 / 24 / 48 / 264 | as with hint 2 | 4,294 | 1,201 | 1,436 | 265 | 3.6e-15 | 6.3e-4 | 38 |
| `--lane-mix random --diag-hint 1` | 1,968 (+0) | as the reference | as the reference | 4,294 | 346,801 | 14,780 | 289 | 5.3e-15 | 0.12 (meaningless) | 38 |

The depth2 saving under the fork's circuit (−13 %) is the one the research record measured on the fork
(ROUTE1 §4: "cuts the eager count 13 %"); the PCMM arms do not depend on the hint rule (PCMM hints its own
inputs at the StC-first cap); the lane-mix term never adds a refresh.

Arithmetic behind the deltas (all from the reports): the fork's formulas hold on the reference arm —
rotations 24 × (558 linear + 40 norm) + 20 = 14,372, conjugations 24 × 11 + 1 = 265, products 336 blocks ×
1,024 = 344,064 of the 345,265 (the rest: 24 × (6 + 2 × 22) + 1 = 1,201 per-channel constants); per-site
saves 3 steps × 3 ct×ct at 96 chains (4,294 − 3,430 = 864); pack-stats saves 33 ct×ct at 49 chains
(1,617), adds 4 masks (196) and 2 rotations (98) per site; pcmm-trunk removes 6 blocks × 1,024 × 24 =
147,456 products and 24 × (62 + 62 + 155 − 9) = 6,480 rotations and adds 3 × 24 external refreshes; pcmm
leaves the 1,201 constants and 24 × (40 + 19) + 20 rotations; the lane mix adds 24 × 64 products,
24 × 17 rotations and 24 conjugations and no refresh.

What the counts mean. The eager refresh count is the fake's reactive count under the runtime's rules,
not what a plan places: upstream's hint rule alone moves it 1,968 → 1,713 by landing every linear's output
one level lower, after which the channel-mix gate fits without its 40 reactive refreshes; the per-site
Newton moves the channel-mix norm's output level and the gate then fires 32 reactive refreshes per block
under either hint rule (the research record's ROUTE1 §5 saw the same sensitivity and predicts −31 %
planned); depth2 saves 13 % under the fork's rule and loses 5 % under upstream's (the norm outputs land at
36 instead of 44 and the gates refresh more). Pack-stats cuts the eager count under both rules (−28 %,
−32 %), as the record's route-5 check measured per site (30 → 17). The planned count per setting needs one
capture (`--capture`, below) and one plan.

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
| the driver | the four arms; the axes in the report; refusals | the errors it computes | its counts against independent tallies; the hashes | `test_run_fake.py` |
| the scale folds | exponents 0, mid, 14; brute force | bit-for-bit factors | the whole model with folds vs the mirror and the unfolded closed form | `test_scales.py` |

149 tests (136 at the synthetic shape, 13 marked `slow_shape` at ring 2^16). The fork's tests of its
GPU runner, planner wrapper, Colab build, decode policy and tick meter are not carried over: those modules
belong to the fork's C++ runtime; on this branch the device path is `impl.driver.PlannedRun` (below).

## What needs a GPU

1. **Capture → plan.** `run_fake.py --capture DIR` writes `block_0..24/graph.json` (and `capture_env.json`,
   `capture_config.json`) at tick 1 on the capture recorder at the fake's levels; the planner then runs on
   them (`tools/fhe_ssm/make_plan.sh` of this session, agent B). Nothing here has run on a device: the
   magnitudes in a fake capture are the fake's float64 values, and the landing convention is the fake's.
2. **A planned run.** The device path on this branch is `impl.driver.PlannedRun` / `Recorder`
   (`driver.py:444-598`): every stage recorded once on the CPU twin, compiled with its plan, replayed;
   `FheSsmModel` supplies the stages (`token_stages`), its carried state is named for the planner
   (`cache.<name>-lvl=<level>`), `rotation_audit` lists the keys the band must hold. Wiring the model into
   `PlannedRun` (its `Recorder.token` contract, the client key bundle with the layout's rotation keys and,
   for PCMM, the ring-switch keys) is the GPU session's first step.
3. **PCMM's landing, cap and band shift**, and the refresh landing of this runtime, are measured there;
   until then every level in the table is the fake's.
4. **The planned refresh counts per axis** (the question the axes exist for) are plan outputs, not eager
   counts.

## Upstream APIs this package depends on (at a3dff85)

| what | where |
|---|---|
| the diagonal linear | `perseus/impl/prefill.py:1127-1199` `DiagLinear` (`split` :1136, `slot_vectors` :1141, `_sum` :1191), `diag_apply` :1222-1247, `diag_rot_steps` :1262 |
| PCMM | `perseus/impl/pcmm.py:29` `Shape`, :262-297 `PcmmLinear.encode`, :414 `run`, :432 `Pcmm` (`apply` :481, `desc` :508, `_fake` :514), :773-774 the fake's cap / landing, :777 `rot_steps`; the legacy in-process keys :156-254 (refused) |
| the fake | `perseus/impl/fake.py:22` `FakeCt`, :65 `fake_config`, :75 `FakeFhe` (`MODE` :76, `_product` :105, `bootstrap` :204, `maybe_bootstrap` :244, `bootstrap_hint` :248, `record_external_bootstrap` :259), :264 `FakeInf` (`set_slot_pt` :322, `tag_reduce` :334), :420 `pack_tokens`, :475 `make_fake` |
| the capture recorder | `perseus/impl/fakecap.py:132` `CapFhe` (`MODE` :134, `inner_bootstrap` :291, `bootstrap_hint` :336, `record_external_bootstrap` :358), :617 `CapInf` (`name_ct` :639), :754 `CapCore` (`begin_subgraph_capture` :783, `end_subgraph_capture` :799), :841 `make_session` |
| the driver | `perseus/impl/driver.py:26` `ImplModel` (`set_capture` :124, `load_plans` :135, `enter_stage` :199, `exit_stage` :215, `run_stages` :222, `token_begin` :324, `token_tail` :339, `token_end` :361); `Recorder` :444, `PlannedRun` :598 |
| the ops protocol | `perseus/impl/ops.py:46` `FheOps` (`mult_pt` :226, `mult_pt_sum` :242, `mult_i` :279, `decrypt_slots_complex` :293, `rotate_many` :298, `tag_reduce` :350, `headroom` :374, `bootstrap` :376, `bootstrap_hint` :415, `realize` :422, `pack_tokens` :434), :441 `NumpyOps` |
| Newton and the lane route | `perseus/impl/poly.py:515` `inv_sqrt_newton`, :745 `LanePackedOps` (`lane_square` :719, `lane_mult` :733) |
| the decode layout | `perseus/impl/layout.py:23` `Dims`, :104 `linear_rot_steps`, :236 `encode_bias_vector`, :280 `lane_vec`, :289 `stride_mask`, :458 `ladder_radix_bits`; `perseus/impl/linear.py:15` `EncodedLinear`, :142 `linear`, :147 `linear_multi` |
| the session bundle | `perseus/impl/rt.py:22` `Rt` (`mult_mask` :97, `step` :194) |
| the capture contract, the presets | `perseus/plan/contract.py:72` `capture_contract`; `perseus/config/__init__.py:29` `preset` (`_pc.Config.preset("n32_gpt2_decode")`) |
| the PCMM key path on a device | `crates/pc-py/src/session.rs:189, 745-764, 862`; `crates/pc-py/src/matmul.rs:303`; `crates/pc-client/src/client.rs:190-200` |

The port adds files under `python/perseus/examples/fhe_ssm/` only; no upstream file is edited.
`LADDER_RADIX_BITS=0` is exported before a session or a fake is made (`port_env`): the rotate-and-sum
ladders run plain on both routes (the fake would otherwise hoist radix groups the C++ does not).

## License

This directory is under the MIT license (`LICENSE` here; each file carries `SPDX-License-Identifier: MIT`).
It runs on Perseus, whose own license (the repository's) applies to running this example on it.
