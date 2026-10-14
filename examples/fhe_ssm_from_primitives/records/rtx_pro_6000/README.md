# The port on upstream dev, on Blackwell (2× RTX PRO 6000)

Two NVIDIA RTX PRO 6000 Blackwell Workstation Edition cards (compute capability 12.0, 97,887 MiB, driver 595.84),
CUDA 13.0 with `include/cccl` on the host compiler's path (`chain/d3_run.log`:2–4). Branch `fhe-ssm-port-dev`, upstream
dev 225e49a merged, FIDESlib 33d9cd5 (`chain/d3_run.log`:8–9). The chain ran at 7892b47; the device probes again at
941162c; the free runs and the timing pair with one core build of 7892b47 (sha256 `bd3a154c…`, every `.meta`).

## The verification chain (`chain/d3_run.log`, one script, fresh clone)

| step | result | line |
|---|---|---|
| build (OpenFHE, FIDESlib, core) | ok | 11 |
| device probes | not built at 7892b47: the probe targets did not link GTest (fixed in 93e0962; run at 941162c below) | 15–28 |
| Python suites | 608 passed, 4 failed: three need the run environment or torch (`test_env_helpers` ×2, `test_import_without_core`), one a missing module (`test_serialization`) | 36–41 |
| lint (ruff, ruff format, mypy) | clean | 42–46 |
| device-route check and bench | ok | 48, 52 |
| one-block smoke | ok | 57 |
| capture, 2 ticks, device magnitude probe, `FIDESLIB_ERRCHECK=1` | ok; tick 1 292.98 s (`chain/d3_capture.json`) | 62–67 |
| plan (hard envelope, dev recipe with the re-cut fix) | 25 blocks, 953 bootstraps per tick, 658 sparse-routed | 70–76 |
| planned, 2 ticks (`FIDESLIB_ERRCHECK=1`) | logits vs closed form 0.000398 / 0.000518, top-1 1.0, 953 runtime refreshes = the plan's (`chain/d3_planned.json`) | 78–80 |
| end | `CHAIN_EXIT=0` | 83 |

## The device probes at 941162c (`probes/`)

Build ok (`p_run.log`:2), `ctest -L gpu` passes (`p_run.log`:17; `p_ctest_gpu.log`). Every probe passes stock and under
`FIDESLIB_ERRCHECK=1`: the layout probe's plaintext tag equals the CPU encode's on every field
(`p_probe_gpu_encode_layout_probe_34.out`:43), the device encoder at levels 34 and 44, the magnitude probe with worst
relative difference 8.904e-16 stock and 7.420e-16 under the error check (`p_probe_gpu_mag_probe_check_34_44_5.out`:41,
`p_probe_errcheck_gpu_mag_probe_check_34_44_5.out`:41).

## The plan, and the same plan from another card (`plan/`)

`chain/d3_plan.log`:423: 25 blocks, 953 bootstraps; 11 terminal-exit retries, every second block (lines 40–390); the
runtime switch the plan declares is in `PLAN_CMD.txt`. Compared block by block with the plan made from the A6000 capture
(`plan_compare.py`, `plan_compare.out`; both plans as archives beside it): placements, correction factors, rescales,
sparse routes, weight and mask levels, hint decisions and rules are equal in all 25 blocks; they differ only in
`capture_env` (the newer capture records more names) and in `real_route`, present in the Blackwell plan only (4 sites in
each even block: the newer capture's tags carry the payload field).

## Two free-running runs, 148 ticks (`free_run/`)

`gpu_run.py --free-run --ticks 148` with the published run's tokens and decode policy (norepeat), planned on GPU 0 and
eager on GPU 1 at the same time. Fidelity is the pick agreement with the float64 closed form on the same history; the
tables are `fr_planned148_trend.md` and `fr_eager148_trend.md` (`fr_trend.py`), the flips with their margins
`fr_*148_flips.json` (`fr_flips.py`, which replays the closed form and matches each run's recorded agreement on all 148
ticks).

| | planned | eager |
|---|---|---|
| top-1 agreement | 9,468 / 9,472 | 9,396 / 9,472 |
| decode-policy pick agreement | 9,465 / 9,472 | 9,399 / 9,472 |
| pick flips (margin range, logits) | 7 (0.000071–0.0343) | 73 (0.00069–0.0968) |
| logits vs closed form, block mean, ticks 0–9 → 130–139 | 0.000534 → 0.001453 | 0.004494 → 0.005178 |
| runtime refreshes per tick | 953 on every tick | 1,968 on every tick |
| free device memory from tick 1 | 59.9 GB on every tick | 60.9 GB on every tick |
| median seconds per tick | 46.34 | 74.34 |

**Where the runs leave the published record.** Teacher-forced through the closed form on the record's own tokens, the
decode policy differs from the record at 18 generated positions in 16 conversations (`record_departures.out`,
`record_departures.py`). They are exactly the published run's own encrypted pick flips (`cross_check_record.out`,
`cross_check_record.py` against that run's `fidelity.json` in the fhe-ssm repository): the same 18 (tick, lane), the
record's token = its encrypted pick and the closed form's pick = its plaintext pick at all 18. So a run that picks as the
closed form does must leave the record at those 16 first points and nowhere else. Planned: 17 departures, 14 of them
predicted and 3 its own flips; the other 2 predicted ones not made because it flipped the same near-tie the published run
did. Eager: 41 departures, most of them its own flips (`fr_*148_trend.md`, last lines).

**Slow ticks.** Both runs slowed at the same tick numbers, the delay about doubling each time, with the refresh count
unchanged (`"seconds"` and `"runtime_bootstraps"` per tick in `fr_*148.json`):

| tick | planned (median 46.34 s) | eager (median 74.34 s) |
|---|---|---|
| 16 | 47.55 s | — |
| 34 | 48.27 s | 76.23 s |
| 70 | 50.40 s | 78.13 s |
| 142 | 54.83 s | 81.94 s |

## Eager vs planned on one idle card (`timing/`)

GPU 0, nothing else running on the box, the same core build, stock (no `FIDESLIB_ERRCHECK`), the record's tokens
teacher-forced, alternating eager, planned, eager, planned, 3 ticks each (`ab_*.json`, `"seconds"`):

| arm | tick 0 | tick 1 | tick 2 |
|---|---|---|---|
| eager 1 | 87.01 | 77.48 | 77.96 |
| planned 1 | 58.34 | 46.23 | 46.20 |
| eager 2 | 87.89 | 77.80 | 78.05 |
| planned 2 | 58.63 | 45.99 | 46.25 |

Steady ticks: eager 77.82 s, planned 46.17 s, **1.69×**. 64 conversations per tick, so the planned tick is
1.39 tokens/s across them. The eager free run on GPU 1 ticked at 74.3 s; on GPU 0 eager took 77.8 s: the two cards differ,
which is why only the same-card pair is a comparison.

## The capture

The 25 block graphs of the Blackwell capture: `graphs_fhe_ssm_d3_7892b47.tar.gz`, 46,924,875 bytes, sha256
`d2a36cb81cf306cf030bf47e29fd19a33adccafbda86188529ede902980beab1` (published with the issue, not in git).
