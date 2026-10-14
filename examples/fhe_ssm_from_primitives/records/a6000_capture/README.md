# The capture: host decrypt vs device magnitude probe (one RTX A6000)

Branch `fhe-ssm-port` (the main-based port; the `.meta` files record each run's environment, not its commit),
FIDESlib at the tree of 374f6df with the arch-guarded patch of the example README, ring 2^16, the n32 chain, 64
conversations, the published run's tokens.

**The capture tick.** Host decrypt (Perseus's capture path): tick 1, the captured one, **26,974.62 s**
(`capture.log`:280). Device probe (`FHE_MAG_GPU=1`): **619.89 s** (`capture_gpumag.log`:305) — **43.5×**. The probe's own
time is the cumulative `[mag_gpu]` counter at the last stage, `enqueue_ms=354550 fetch_ms=5943`
(`capture_gpumag.log`:303): 360.5 s, 58 % of the tick. Same node counts on both paths (`[mag_gpu] items=…` per stage).

**One ciphertext at a time** (`gpu_mag_probe_check.out`, the ctest `gpu_mag_probe_check`): the five statistics from
the host path and from the probe agree to 2.20e-16 … 7.60e-16 relative on a dense ciphertext, one scaled ×1000, one with
a pending rescale (noise degree 2) and a 512-slot one (`"worst"` per case); verdict `worst_rel 7.598e-16, pass`.

**A whole capture, node by node** (`compare_captures.py`; outputs `compare_*.json`). Two captures are two keyed runs, so
their decrypted magnitudes differ by the FHE error of each; the script reports the distribution and gates only at 0.1.
- host decrypt vs device probe (`compare_captures_ml48.json`, `graphs`): 25 blocks, **716,245 nodes aligned**, no
  misaligned block; `output_max_abs` relative difference median **2.58e-4**, 99.9th percentile 4.48e-3, max **1.34e-2**.
  Its `"pass": false` comes from the 0.1 gate on the two fields that measure a remainder (`output_max_dev` max 0.521,
  `output_max_coeff_ac` max 0.134): a deviation from a mean or a coefficient without its constant term, relative to
  itself.
- device probe vs a second device probe (`compare_gpumag_runs_ml48.json`): median 2.59e-4, max 1.65e-2, `"pass": true` —
  the same envelope as host vs device.

**The plans made from each** (hard envelope, ceilings 48 and 50; `plans` in the same JSONs and the planner logs):

| capture | ceiling 48 | ceiling 50 |
|---|---|---|
| host decrypt | 1,297 (`plan_v2_ml48.log`:270) | 957 (`plan_v2_ml50.log`:383) |
| device probe | 1,297 (`plan_gpumag_ml48.log`:271) | 959 (`plan_gpumag_ml50.log`:373) |
| second device probe | 1,295 (`plan_gpumag2_ml48.log`:270) | 956 (`plan_gpumag2_ml50.log`:383) |

(refreshes per tick). The plan files differ in equal-cost site choices (`plans.identical` 0 of 25 at 48;
`decision_identical` 22 of 25 at 50); the costs agree.

**Eight planned ticks** with the device capture's ceiling-50 plan (`planned8.log`, `planned8.json`, `planned8.meta`):
959 runtime refreshes every tick, logits relative error vs the closed form 0.000265–0.000958, top-1 agreement 1.0 on
all 8 ticks.

**The fresh-clone capture** (`v7_chain.log`): the pushed head of `fhe-ssm-port`, 5d82074 (line 2), cloned and built
from scratch with FIDESlib 374f6df fetched fresh (line 5), then run end to end on the RTX A6000 (`capture_v7.log`:31).
Its capture, with the device probe by default (line 60): tick 1 **620.7 s** (`capture_v7.log`:305; `capture_v7.json`),
top-1 agreement 1.0. Its 25 block graphs are the capture `../dev_planner/` plans from (and so the A6000 side of
`../rtx_pro_6000/plan/`):
[`graphs_fhe_ssm_v7_head5d82074_20261012.tar.gz`](https://huggingface.co/datasets/xelananv/Perseus-graphs/resolve/main/graphs_fhe_ssm_v7_head5d82074_20261012.tar.gz),
46,851,607 bytes, sha256 `49a8f0d3e462d4457b9153a568ad189cc38753163107676c964ba3c1b701313a` (on Hugging Face, not in
git; it unpacks to `graphs/fhe_ssm/`). The graphs of the three captures above are not published.
