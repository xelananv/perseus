# ROBUST_NORMS — RWKV-7's 1/√x sites as Newton from a constant seed (P6, 2026-11-05)

## The failure this fixes

Every 1/√x site (`ln1.rsqrt`, `ln2.rsqrt`, `att.kk_rsqrt`, `att.gn_rsqrt`, 4 per block) was `method: newton`
(`harness/fit.py` `newton_fit`): a degree-3 Chebyshev **seed** fitted on u ∈ [seed_lo, 1], then 1–14 Newton steps.
Outside that interval the seed polynomial is meaningless (negative, or far too large), and Newton from a bad seed
diverges. A ciphertext cannot clamp first. On the box, token 17 (`,`) of `CALIB_TEXT_LONG_V1` drove block 7's GroupNorm
variance to 15.82 against [0.000984, 10.46], and the capture carried NaN (commit 87fe1ab2, `common.py:27`).

## The mechanism of the fix

Newton for 1/√u is y ← y(1.5 − 0.5·u·y²). Write t = u·y² (t = 1 at the answer). One step maps
√t → √t·(3 − t)/2, which is ≤ 1 for every t ≥ 0. So after one step every iterate lies *below* the answer, and from
below it rises monotonically to it. The only way to fail is a first step with t ≥ 3. Here y₀ is a constant, so t₀ = u·y₀²:
**the iteration converges for every u in (0, 3/y₀²)**. Inputs far below the range just need more steps: y grows by
≤ 1.5× per step, so it takes about log₁.₅(1/√u_lo) steps to climb, then a few quadratic steps to finish. It never
explodes. This is how our own model's norm works (`perseus-speedrun/.../fhe_ssm/norm.py`: a public pre-scale, a
constant seed, `inv_sqrt_newton`).

## Design (`method: newton_const`, `results/vector-d-20261104/harness/rsqrt_const.py`)

For each site, with calibrated argument range [m, M] (the recorded min/max in `ranges_ctx4096.json`, x units before
eps) and widening factor R:

- **Accuracy range** [lo, hi] = [m/R, M·R]. The JSON's `lo`/`hi` mean this now.
- **Pre-scale** `scale` = M·R + eps, so u = (x + eps)/scale lies in [u_lo, 1] over the accuracy range. The seed is
  y₀ = 1. A constant y₀ is the same as folding y₀² into `scale`.
- **Divergence bound**: u = 3, i.e. x = 3·scale − eps = 3× the widened top = 3R× the calibrated top (`conv_hi`;
  `conv_lo` = −eps). This is 30× the calibrated top at R = 10.
- **Steps**: the fewest with max relative error ≤ tol = 1e-3 over [u_lo, 1], found on a 4001-point log grid. The
  error is largest at u_lo. Plain Newton from y₀ = 1 needs 14 steps for a range ratio of 1e4, 17 for 1e5 and 20 for
  1e6.
- **Output convention is unchanged**: y·scale^−1/2 = 1/√(x + eps). `model.py` already folds scale^−1/2 into γ / k_k /
  ln_x_w, so nothing downstream changed.
- For `ln` and `gn`, eps (1e-5, 6.4e-4) puts a floor under the argument: a zero variance still gives u ≥ eps/scale.
  For those two kinds the bottom widening is often below the eps floor and costs nothing.

R = 10 at every site (the brief's minimum). Larger R costs about log₁.₅(√R) ≈ 2.8 more steps per decade of R², so
R = 100 would add about 6 levels per site. Not taken.

### Depth per site (port: step 1 affine = 1 level, then 2 levels/step; old = ⌈log₂4⌉ = 2 seed levels + 2/step)

From `harness/nc_coeffs/<tag>/convert_report.json` (min, max, mean over blocks):

| model | site | old steps | new steps | old depth | new depth |
|---|---|---|---|---|---|
| 0.1B | ln1.rsqrt | 2–5 (3.3) | 11–14 (12.1) | 6–12 (8.7) | 21–27 (23.2) |
| 0.1B | att.kk_rsqrt | 4–12 (6.4) | 13–21 (15.4) | 10–26 (14.8) | 25–41 (29.8) |
| 0.1B | att.gn_rsqrt | 7–14 (9.7) | 14–21 (17.3) | 16–30 (21.3) | 27–41 (33.7) |
| 0.1B | ln2.rsqrt | 1–5 (3.3) | 10–14 (11.9) | 4–12 (8.7) | 19–27 (22.8) |
| 0.4B | ln1.rsqrt | 2–4 (3.1) | 10–13 (12.0) | 6–10 (8.2) | 19–25 (22.9) |
| 0.4B | att.kk_rsqrt | 4–10 (6.5) | 13–19 (15.6) | 10–22 (15.0) | 25–37 (30.3) |
| 0.4B | att.gn_rsqrt | 3–14 (9.3) | 10–21 (16.5) | 8–30 (20.5) | 19–41 (32.1) |
| 0.4B | ln2.rsqrt | 1–4 (3.0) | 10–13 (11.8) | 4–10 (8.0) | 19–25 (22.5) |

Sum of the four 1/√x depths per block (0.1B, mean): old ≈ 53.5 levels, new ≈ 109.5. The norms were cheap because the
fitted seed already did most of the work on a narrow interval. Widening 100× in ratio (R both ways) is what costs.

Example, block 7 `att.gn_rsqrt` (0.1B): old interval [0.000984, 10.46] with 9 steps. New accuracy range
[0.000197, 95.09] with 17 steps, convergent up to 285. The 15.82 that broke P5 sits 6× below the new top.

### Refresh cost (planner, measured)

1-block decode capture of 0.1B + plan (LANDING=high, `test_capture.HIGH`, default levers), census of block 0
(`harness/nc_coeffs/0.1b/plan_cost_block0_decode.json`, from `tests/test_newton_const.py::test_plan_cost_old_vs_nc`):

| coeffs | refreshes / token, block 0 | by step |
|---|---|---|
| coeffs_mixed.json | **14** | tm.kk 7, tm.gn 4, tm.norm 1, tm.lora 1, cm.norm 1 |
| coeffs_mixed_nc.json | **21** | tm.kk 9, tm.norm 4, tm.gn 4, cm.norm 3, tm.lora 1 |

That is +7 refreshes per token in block 0 (+50%). Block 0's new steps (11/21/14/10) are near the 0.1B means
(12.1/15.4/17.3/11.9). Expect a similar +6 to +9 in other blocks, but only block 0 was planned.

### The next lever: `method: newton_scaled` (built 2026-11-05, default off; the speed-up map's T1)

`perseus.impl.poly.scaled_newton_steps` keeps the constant seed but gives each step its own (a, b): the first step is
newton_const's (1.5, 0.5), with its zero at u = 3 (the same divergence bound), and every later step is the pair that
maps the band of t = u y² onto the widest image that stays ≤ 1 (q peaks at exactly 1 inside the band), so one step covers
about 6.7× of range ratio instead of 2.25×. Because every step keeps t ≤ 1, an input anywhere in (0, 3·scale) stays finite,
positive and under-normalised at worst: the monotone-from-below argument above holds unchanged, and the accuracy range,
pre-scale, eps and output convention are the newton_const site's own.

- **Converter:** `tools/rwkv7/convert_newton_scaled.py --coeffs coeffs_final.json --out coeffs_final_sc.json` (every
  non-1/√x site byte-equal; per site: max relative error ≤ tol on [lo, hi], finite / positive / from below on
  (hi, 0.98·conv_hi), finite at 0). Site fields: `steps [[a, b], …]`, `n_steps`, `depth` = 1 + 2(n_steps − 1), plus
  newton_const's range fields and `const_steps` / `const_depth`.
- **Port:** `model.py Circuit.newton` (the affine map of u folded onto x; the d2 product order: one scalar level for
  step 1, two levels per later step, the same shape as newton_const), `ref.py newton_scaled_rsqrt` (the mirror), the
  harness's `rwkv7_model.py newton_scaled_eval` and `scan_ranges.py` (the gate and the scans accept the method).
- **Depth, measured by the converter** (tol 1e-3, zero 3.0): 0.1B 48 sites, the four chains' mean depth per block
  109.5 → 60.2 levels (ln1 23.2 → 13.5, kk 29.8 → 16.0, gn 33.7 → 17.3, ln2 22.8 → 13.3); 0.4B 96 sites, 107.8 → 58.7
  (ln1 22.9 → 12.9, kk 30.2 → 15.9, gn 32.1 → 16.8, ln2 22.5 → 13.0). Max relative error over the accuracy ranges ≤ 9.5e-4
  (`tests/test_newton_scaled.py`).
- **Not yet done:** the gate (WikiText-103 / LAMBADA, clamped and unclamped, plus the full scans) on the converted
  files, and any device run. The planner's refresh count on a 1-block decode capture is `test_newton_scaled.py::
  test_plan_cost_const_vs_scaled` (`--rwkv7-plan`).

## What is built

- Harness (`results/vector-d-20261104/harness/`): `rsqrt_const.py` (design + numpy eval). `rwkv7_model.py`
  (`newton_const_eval`; `load_poly` and `_poly` know `newton_const`; `--clamp` clamps to the accuracy range).
  `fit.py --rsqrt newton-const --rsqrt-range-factor R`. `convert_newton_const.py` (the converter). `scan_ranges.py`
  (the plaintext range scan).
- Converted files (only the 1/√x sites differ; every other site is byte-equal as JSON):
  `harness/nc_coeffs/0.1b/coeffs_mixed_nc.json` (also `.cache/rwkv7/coeffs_mixed_nc.json`) and
  `harness/nc_coeffs/0.4b/coeffs_mixed_nc.json`, each with `convert_report.json`.
- Port: `model.py Circuit.newton` evaluates `newton_const` (affine first step, then `inv_sqrt_newton_d2`). The SiteGuard
  judges a nc site by its accuracy range [lo, hi], which lies inside the convergence region (−eps, conv_hi). `ref.py`
  mirrors it. `run_scan` uses the same lo/hi. `tests/conftest.py --rwkv7-coeffs FILE` runs any test on another
  coefficients file.

## CPU checks (this Mac, 2026-11-05)

- `tests/test_newton_const.py`: the converter for 0.1B and 0.4B (all 48 / 96 1/√x sites converted, the others
  unchanged, lo ≤ m/R, hi ≥ M·R, conv_hi ≥ 3·hi, rel err ≤ 1e-3). Exact agreement across every widened range: worst
  9.42e-4. Positive, finite and below 1/√x between hi and 0.98·conv_hi, and at x = 0.
- `CALIB_TEXT_LONG_V1`, unclamped poly scan of all 12 blocks over 80 tokens with the nc file: **every site inside,
  the non-norm sites included**. The same scan with the old file (`test_scan_text_info[genesis_v1]`) still
  fails as in P5. `blocks.7.att.gn_rsqrt` leaves at position 17 (15.82 vs [0.0009843, 10.46]). From position 18 the
  non-finite values cascade into `blocks.7.ln2.rsqrt`, `blocks.7.ffn.relu2`, `blocks.8.ln1.rsqrt`, and on.
- P5's failing capture (V1, 8 blocks, T = 32, SiteGuard on) with the nc file: **passes**
  (`test_genesis_v1_capture_guard_passes_nc`).
- Fake vs mirror and mirror vs poly reference, 2 blocks, on the nc file (`test_fake`, `test_mirror`, `test_ref`
  with `--rwkv7-coeffs .cache/rwkv7/coeffs_mixed_nc.json`): 7 passed. The mirror's logits relErrRms against the poly
  reference is at most 6.29e-4 at wbits 24.
- Whole suite: 55 passed, 5 skipped. The skips are the existing 3 plus the plan test (needs `--rwkv7-plan`) and the
  scan-script smoke test, which needs torch, absent from this venv. `scan_ranges.py` is therefore **untested here
  beyond a compile check**. Its first run is on the box.
- Not done here: WikiText / LAMBADA quality (GPU).
