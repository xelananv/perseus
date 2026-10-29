# RUNSHEET_GPU — the fhe-ssm port on Perseus's `speedrun` branch, one GPU session from a fresh box

Every step is one subcommand of `tools/fhe_ssm/run_gpu.sh`. The script logs to `$OUT/session.log` and writes every
product under `$OUT` (default `<checkout>/.cache/fhe_ssm/gpu_<UTC date>`, gitignored). A duration or count carries a
source tag [Sn] when it was measured (the sources are listed in the research record of this runsheet); otherwise it
says **unmeasured**. Paths are relative to the checkout.

Status: the plan tooling (`make_plan.sh`), the CPU compile check (`compile_check/`), the change check and the sync
script work on the CPU; the port's fake and capture run (`python/perseus/examples/fhe_ssm/run_fake.py`). The planned
driver (`python/perseus/examples/fhe_ssm/run_planned.py`, contract §4) exists and its CPU part (`--prepare-only`: the
recorded stream, the compile check, the report) runs on a CPU-only build [S12]; **nothing of it has run on a device**.
On the device path it runs only the arms whose every linear is PCMM (`pcmm24`, `best`); the others run their CPU part
(§3).

## 0. Rules for every timed step

- One binary for every number that is compared. Build once and never rebuild mid-matrix. Wall-clock belongs to a build,
  so a time measured on another build (another runtime, another toolkit, another link) is never a denominator here.
  `run_gpu.sh` writes the commit, the `_pc` .so hash, nvcc's version and the load average beside every run.
- An idle GPU and an idle host. `run` refuses a GPU with other compute processes. The planned path stages plaintexts
  from the host, so a CPU-heavy neighbour moves the timing. Captures and plans are CPU jobs: run them before the timed
  block, or on another machine at the same commit.
- A control inside the block: time the arm's control (§3) first and again last. Quote an arm's gain only against the
  control of the same block and binary; a difference inside the control's own spread is noise.
- Fidelity next to speed for every arm: the reply's RMS relative error against the exact forward, and top-1 agreement
  over the conversations. An arm that moves fidelity is reported as such, not as a free speedup.

## 1. The box

| need | why | source |
|---|---|---|
| Linux x86-64; an NVIDIA driver for CUDA 13.2; the CUDA 13.2 toolkit at `/usr/local/cuda-13.2` (or `CUDA_HOME`) | the kernels are built with nvcc from `CUDA_HOME` | `crates/pc-kernels/build.rs:8`; `speedrun/MAINTAINING.md:13` |
| GPU: **96 GB recommended** (RTX PRO 6000; sm_120 is the build's default arch). 48 GB is upstream's floor for GPT-2 | the lockstep schedule grows the slab 4.7× on GPT-2 prefill [S7]; this runtime's footprint for the model is unmeasured (§6) | `README.md:33,173`; `crates/pc-kernels/build.rs:2-3` |
| host RAM: ≥ 128 GB for the PCMM arms. ≥ 192 GB if any diagonal arm runs: those stream ≈ 84 GiB of float32 weight rows per tick from pinned host memory (344,064 diagonals × 2^16 × 4 B) | GPT-2 Prefill-128's server peaks at 39.6 GB | `speedrun/runner/entry.sh:10-11`; `python/perseus/impl/driver.py:941-959` |
| rustup + cargo (`rust-toolchain.toml` pins 1.99.1; rustup installs it on first use), `uv`, git | the build | `README.md:22-27` |
| ≥ 50 GB disk | venv, cargo target, captures (the model's 24-block capture: 472 MB [S4]) and plans | — |

## 2. Steps

| # | step | command | expected output | duration |
|---|---|---|---|---|
| 0 | record the box | `bash tools/fhe_ssm/run_gpu.sh box` | `$OUT/box.txt`: the GPU's name, memory, driver and compute capability; `nvidia-smi topo -m`; nvcc 13.2; RAM; disk; load; commit | seconds |
| 1 | the code | `git clone -b fhe-ssm-speedrun <the fork's URL> perseus && cd perseus && git remote add upstream https://github.com/gladia-research-group/perseus && git fetch upstream speedrun`. Without a pushed branch: `git bundle create fhe-ssm.bundle origin/speedrun..fhe-ssm-speedrun` where the branch lives, copy the bundle, then `git clone -b speedrun https://github.com/gladia-research-group/perseus perseus && git -C perseus fetch <bundle> fhe-ssm-speedrun:fhe-ssm-speedrun && git -C perseus switch fhe-ssm-speedrun` | `bash tools/fhe_ssm/check_additive.sh upstream/speedrun` → `... : OK` (every change is the port's own or listed in `tools/fhe_ssm/UPSTREAM_CHANGES.md`) | unmeasured |
| 2 | build | `bash tools/fhe_ssm/run_gpu.sh build`: uv venv 3.12, `uv pip install -e ".[hf,examples,dev]"`, `maturin develop --release`; `PERSEUS_CORE_ARCH` from the GPU's compute capability, `PC_NVCC_JOBS=8` | `build: done in N s; commit=… pc_so=… nvcc=13.2 arch=120` | unmeasured |
| 3 | verify | `bash tools/fhe_ssm/run_gpu.sh verify` | `perseus._pc loaded; device classes missing: none` (Session, Program, KeySource, Client, PtStore: `crates/pc-py/src/lib.rs:31-48`); the GPU's used / free / total memory from `nvidia-smi`; upstream's CPU tests (`test_fake_shadow`, `test_pairing_cpu`, `test_rmsnorm_example`, `test_prefill_cpu`): 70 passed, 4 skipped [S1]; the port's tests pass; the change check `OK`; the compile check's binary path. `_pc` has no free-memory function: a session's memory is its `mem_report` and `fhe.mem_info()` = (free, total) bytes (`crates/pc-py/src/fhe.rs:1307-1310`), which the driver records (§4) | the four upstream files: 1.76 s on a CPU-only build [S1]; this box unmeasured |
| 4 | the model's data | the pbd430a checkpoint's weights `.npz` (sha256 begins `3ec3f1d8`, ends `784b` [S3]), seeds `.json` and config `.json` in one directory; `export DATA=<that dir>` | `sha256sum` matches | — |
| 5 | this build's refresh prices | `bash tools/fhe_ssm/run_gpu.sh bench` = `cargo run --release -p pc-tools --bin pc-bench -- bts --all --k 1,8 --reps 20 --keys packed` (the command in `README.md` of upstream `main`) | `$OUT/bench_bts.txt`: ms per refresh per route at K = 1 and 8. Set `PLAN_BTS_MS` from it before planning with `ILP_OBJECTIVE=ms` | unmeasured |
| 6 | the arms | `bash tools/fhe_ssm/run_gpu.sh arms` | the table of §3 | — |
| 7 | captures (CPU) | `bash tools/fhe_ssm/run_gpu.sh capture <arm>` per distinct capture (arms that differ only in plan or run share one); or make them elsewhere at the same commit and copy `$OUT/cap_*` | `$OUT/cap_<landing>_<flags>/`: `block_<b>/graph.json` × 25 stages, `capture_env.json`, `capture_config.json`. A `LANDING=high` arm's capture is checked after it is written: its `capture_config.json` must carry `stc_first=2` and `raise_drop=512:1` (§3) | 24 blocks at `--levels preset`: 11 min 29 s on a 16 GB laptop CPU [S4] (17 min 7 s for the high switches' capture, with a planner running beside it [S10]); every linear PCMM: 31 s [S11]; the plain fake run of the same model: 17-25 s [S2] |
| 8 | plans (CPU) | `bash tools/fhe_ssm/run_gpu.sh plan <arm>` = `tools/fhe_ssm/make_plan.sh` with the arm's axes, then the compile check | `plan_<arm>/summary.txt`: refreshes per stage @ exit level, the total, `compile check … check errors 0`. A planner failure (exit ≠ 0) or a compile check error stops the step | 25 stages, min-cut: 168-183 s on the same laptop [S5]; `LANDING=high`: 992 s (every block retried once; another planner beside part of it) [S10]; ILP: 55-58 s for 2 GPT-2 stages [S5] |
| 8a | the driver's CPU part | `bash tools/fhe_ssm/run_gpu.sh prepare <arm>` = `$DRIVER … --plan P --prepare-only` (`PREPARE_EXTRA=--lean` on a host with little RAM: plaintext digests instead of values) | `prepare_<arm>/record.json`: the recorded stream's op count = the capture's node count, the plan's refresh count, 0 check errors, the stream static across ticks | the reference at the model shape, `--lean`: 885.6 s, contended [S12]; `pcmm24` at `BLOCKS=1`: 3 s; the 24-block `pcmm24` prepare needs more than 3 GB of host memory [S13] |
| 9 | keys | `KEYS` has no default: `run`, `smoke` and `matrix` refuse without it. `KEYS=inproc` = the driver's `--insecure-test-keys`: the session derives every key from the compiled programs and holds the secret for the whole run (`crates/pc-py/src/session.rs:617-737`), a timing run only, recorded as `keys.mode = "insecure-inproc"`. The deployment split: `bash tools/fhe_ssm/run_gpu.sh keyspec <arm>` = `$DRIVER … --plan P --keyspec $OUT/keys_<arm>/keyspec.json` (the key request: the model's rotation steps, the conjugation, the PCMM `t`s, and on the device build the session's own `Session.key_manifest` after the programs compile), then `bash tools/fhe_ssm/run_gpu.sh keygen <arm>` = `$DRIVER --keygen $OUT/keys_<arm> --keyspec-file …/keyspec.json` (the client role, no model flags: a fresh secret, `keys.pckb`, and `client_secret.bin` kept beside it so the run can decode its replies; a real client runs this elsewhere and ships `keys.pckb` only), then `KEYS=bundle FHE_SSM_KEYS=$OUT/keys_<arm>` (or `KEYS=<that dir>`) for `run`. The driver checks the bundle before the session opens (plan contract, PCMM keys, switching-key count, public key) and opens it with `Session.from_bundle` without a secret; the reply is decrypted by `_pc.Client` on the client's side of the process | `keyspec.json`; `keygen.log`: the key count, the bundle size; the run logs the PCMM `t`s | unmeasured |
| 10 | 1-block smoke | `bash tools/fhe_ssm/run_gpu.sh smoke` (capture, plan and run of `pcmm24` at `BLOCKS=1`: a diagonal arm cannot run on the device path, §3) | the driver's per-tick line, `mem_report`, `nvsmi.csv` | unmeasured |
| 11 | the matrix | `bash tools/fhe_ssm/run_gpu.sh matrix pcmm24 best pcmm24` on the device; `bash tools/fhe_ssm/run_gpu.sh matrix ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24` runs those arms' CPU part (capture, plan, prepare: `matrix` calls `prepare` instead of `run` for an arm that is not all PCMM). The lockstep arms stay blocked (§3). One arm at a time, the control first and last | per arm `run_<arm>/record.json` (§4), `run.log`, `nvsmi.csv`, `build_id.txt` | unmeasured on this runtime |
| 12 | bring it home | `rsync -a $OUT/ <the records' home>/gpu_<date>/` | — | — |

## 3. The axis matrix

> The `LANDING=high` arms (high, pair, sbk16, emfuse, pcmmtrunk24, pcmm24, best) run. The port's session Config takes
> the landing's runtime switches from the environment (`LANDING_SWITCHES` in `python/perseus/examples/fhe_ssm/fakes.py`),
> so a capture made under them writes a `capture_config.json` carrying `bts.stc_first=2` and `bts.raise_drop=512:1`
> and records its own dense refreshes StC-first; `run_gpu.sh capture` checks the stamp after the capture, and
> `HIGH_CAPTURE_OK=0` blocks these arms. Their plans need the planner fix of upstream PR #10, which this branch carries
> (`tools/fhe_ssm/UPSTREAM_CHANGES.md`). The lockstep arms stay blocked (below).


Columns: the capture flags (`run_fake.py`), the plan axes (`make_plan.sh`), the run flags (the driver), what the arm
isolates, and its control. The counts are plans made on the CPU [S5, S6, S10, S11]; no time on this runtime is measured.

| arm | capture | plan | run | isolates | control |
|---|---|---|---|---|---|
| `ref` | — (every axis at its default: the reference circuit, diagonal hint 1) | `LANDING=stock` | — | the reference circuit; this runtime's landings without StC-first: dense shipped order 34, SparseB-512 24; diagonal linears | itself (first and last) |
| `depth2` | `--newton-form depth2` | stock | — | 2 products per Newton step instead of 3 | `ref` |
| `persite` | `--newton-sites per-site` | stock | — | per-site Newton counts | `ref` |
| `packstats` | `--pack-stats halves` | stock | — | one Newton chain for both halves' statistics | `ref` |
| `high` | — | `LANDING=high` | — | dense refreshes StC-first at 28, SparseB-512 at 26 | `ref` |
| `pair` | as `high` | `LANDING=high PAIR=1` | `--schedule siblings` | refresh pairs on the real-payload dense StC-first sites | `high` |
| `sbk16` | as `high` | `LANDING=high SPARSE_B_K=16` | — | SparseB's EvalMod at K = 16, replay only | `high` |
| `emfuse` | as `high` | `LANDING=high RUNTIME_EXTRA=PC_EM_FUSE_RESCALE=1` | — | EvalMod's relinearization and rescale fused (a kernel switch) | `high` |
| `pcmmtrunk24` | `--linear pcmm-trunk --pcmm-wbits 24` | `LANDING=high` (PCMM needs StC-first: `python/perseus/impl/pcmm.py:3`) | — | PCMM on the trunk's linears, 24-bit weights | `high` |
| `pcmm24` | `--linear pcmm --pcmm-wbits 24` | `LANDING=high` | — | every linear PCMM | `high` |
| `best` | depth2 + per-site + pack-stats + pcmm 24 | `LANDING=high PAIR=1` | `--schedule siblings` | the stack | `ref` and `high` |
| `lockstep2`, `lockstep4` | `--streams K` (**does not exist**) | `LANDING=high` | `--schedule lockstep` | K batches of conversations through one stream | `high` |

Planned counts so far [S6, S10, S11] (`summary.total_bootstraps` per tick; the landings are effective levels, a raw
landing plus the pending rescale):
- **the reference (`ref`, diagonal hint 1)**: LANDING=stock **637** refreshes per tick (446 placed, 143 fired hints, 48
  deliberate; block 0 29, blocks 1-23 26 each, the head 10), 0 check errors, 152 s of planning;
- **`high`** (the same circuit, captured under the high switches): **609** per tick (560 placed, 1 fired hint, 48
  deliberate; block 0 24, blocks 1-23 25 each, the head 10), 0 check errors. Routes: dense StC-first landing at 28: 95;
  dense in the shipped order at 34: 24 (one per block, the exit refresh of the planner's retry, its input past the
  StC-first entry 46); SparseB-512 at 26: 490. The capture's own eager refreshes: 1,968 without the switches, 1,344 with
  them; every one StC-first, none past the entry [S10];
- **`pcmm24`** (every linear PCMM at 24 bits, captured under the high switches): **802** per tick = 490 placed
  (SparseB-512 at 26) + 48 deliberate carry refreshes (dense StC-first at 28) + 264 PCMM output finishes (11 per
  block); blocks 0-23 33 each, the head 10; no fired hint; 0 check errors (the compile counts 538 refresh sites plus
  120 PCMM finish ops, one per linear call). The PCMM input cap (44) never fires: every input arrives at 40 or below
  [S11];
- a LANDING=high plan of a capture made WITHOUT the switches: 610 planned, **120 compile-check errors** (does not bind)
  [S10];
- a capture at diagonal hint 2 (the upstream `diag_apply` rule): the same 637, stage for stage, although its graphs
  differ (half its linear-input hints have threshold 40 instead of 42). The planner decides which hints fire, and at
  this runtime's landings the thresholds do not bind. LANDING=fork (the fork's landings, dense and 512 at 36) on that
  capture: **900**.

What blocks which arms:
- **The `LANDING=high` arms are no longer blocked**; what they needed, and what still goes wrong without it:
  1. A capture under the high switches. The port's capture now takes them from the environment; a capture made without
     them records its own refreshes in the shipped order, and a LANDING=high plan of it does not bind (the compile
     check's level errors) [S10].
  2. The planner fix of upstream PR #10 (forced exit refreshes survive the re-cut and the prune), on this branch as a
     listed change. Under the high recipe the cheapest plan of each block leaves its exit at 48/d2, the next block is
     refused there, and the planner retries the block with a forced exit refresh; without the fix a re-cut trial drops
     that refresh and block 1 is infeasible [S6, S10]. With it the 24 stages after block 0 each refuse their entry once
     and blocks 0-23 are each retried once.
- **Lockstep**: `Program.compile(schedule="lockstep")` reorders what was captured (`crates/pc-py/src/program.rs:68-84`).
  K streams need K ciphertext sets in each captured stage and registers per stream; `PlannedRun.prepare` does not pass
  a schedule (`python/perseus/impl/driver.py:653-655`). `run_gpu.sh` refuses these arms.
- **Diagonal linears on the device** (every arm but `pcmm24` and `best`: `ref`, `depth2`, `persite`, `packstats`,
  `high`, `pair`, `sbk16`, `emfuse`, and `pcmmtrunk24`'s `W_k` / `W_r`): a planned session keeps every plaintext its
  programs read resident (`driver.py:731`; 28 rows × 2^16 × 4 B = 7 MiB each). For the model's 344,064 diagonals that
  is ≈ 2.3 TiB. The driver refuses such an arm above `--pt-budget-gb` (default 40) with the count, and `run_gpu.sh run`
  refuses it before the driver starts; `prepare` runs its CPU part. Running it needs the streamed-weights path
  (`StreamedPlannedRun`, `driver.py:941-965`; `Session(streams=, stream_frames=)`, `session.rs:1441-1444`) or a device
  diagonal encoder. Until then the device control of `best` is `pcmm24`, and the diagonal arms are compared by their
  planned counts only.
- **Captures at the runtime's levels**: `run_gpu.sh` passes `--levels preset` (ceiling 44, landing 32 + a pending
  rescale). A capture at the fork's levels records its own refreshes where this runtime has no route.
- **`SPARSE_SLOTS=512,32,1`**: the runtime lists no landing for 32 slots (`crates/pc-program/src/compile/mod.rs:250-262`);
  expect the compile check to refuse it. It is not in the matrix.
- **Placer arms** (`PLACER=ilp`, `ILP_OBJECTIVE=ms` with `PLAN_BTS_MS` from step 5) are CPU-only. Compare their refresh
  counts and the planner's ms cost, and run on the GPU only the one that wins on paper.

## 4. The driver's contract

`DRIVER` (default `perseus.examples.fhe_ssm.run_planned`) is invoked as

```
python -m $DRIVER --weights W --seeds S --config C --levels preset [--blocks N] [run_fake's axis flags]
                  --plan PLAN_DIR --ticks T --profile --profile-every K --keys inproc|bundle|DIR
                  [--schedule program|siblings|lockstep] --out record.json
python -m $DRIVER … --plan PLAN_DIR --prepare-only [--lean] … --out record.json        (the CPU part, no session)
```

The landing's runtime switches are in the environment. The driver must:
- build the model on the CPU twin with the same axis flags;
- run `PlannedRun` (decode-shaped: the carried state as registers, the same stream every tick), compiled with
  `--schedule` (an override of `PlannedRun.prepare`), on `config.plan_runtime(PLAN_DIR, "n32",
  base=config.preset("n32_gpt2_decode"))`;
- call `precompute`, run one untimed warm-up tick on public random tokens (it also captures the CUDA graphs), then T
  timed ticks;
- with `--profile`, fence the device after every stage so each stage has a wall. That wall is an attribution, not the
  production wall: the fence serializes host issue with device work. `impl/profile.TimedOps` times Python-issued ops
  and so times nothing on this path. `--profile-every K` fences every K-th tick only, so one run holds fenced and
  unfenced ticks (each tick's record says which); `run_gpu.sh` passes `PROFILE_EVERY` (default 2);
- refuse diagonal linears above `--pt-budget-gb` (§3).

`record.json`, one object per arm:
- `build_id`, and the plan's `axes=` and `runtime=` lines;
- per program: `n_bootstraps`, `bts_batches`, `n_pairs`, `n_finishes`, `mem()`;
- after setup: `sess.mem_report`, `sess.setup_seconds()` (setup, keygen), `sess.fhe.mem_info()` (free, total bytes);
- per tick: `wall_s` (issue to the reply's decrypt); `stage_wall_s` (with `--profile`); `sess.planned_stats()` (host_us,
  launch_us, captures and replays, pairs as one or split, fused product-adds, StC-first cache misses);
  `sess.fhe.mem_info()`; the reply's RMS relative error vs the exact forward and vs the mirror; its worst conversation;
  top-1 agreement over the conversations;
- the nvidia-smi peak comes from `run_gpu.sh`'s `nvsmi.csv`.

## 5. What each run writes

- `$OUT/box.txt`, `$OUT/session.log` (every step, with the build id), `$OUT/build.log`, `$OUT/bench_bts.txt`
- `$OUT/cap_<…>/`: the capture (`graph.json` per stage, `capture_env.json`, `capture_config.json`), with
  `cap_<…>.capture.log` and `cap_<…>.fake.json` (the fake's report of the capture tick) beside it
- `$OUT/bootstrap_placements/plan_<arm>/`: `block_<b>_placement.json` (stamped), `PLAN_CMD.txt`, `plan_rot_steps.json`,
  `summary.txt`, `compile_check.txt` / `.json`, `plan.log`
- `$OUT/prepare_<arm>/`: `record.json`, `prepare.log` (the driver's CPU part)
- `$OUT/run_<arm>/`: `record.json`, `run.log`, `nvsmi.csv` (500 ms samples of memory, utilisation and power),
  `build_id.txt`

## 6. VRAM: what is known

- **The fork's C++ runtime** (another binary): the planned plateau was 35,961 MiB on an RTX PRO 6000 and 35,118 MiB on
  an RTX A6000, leaving 13.1 GiB free on that 48 GB card [S8].
- **This runtime: unmeasured.** Every term differs:
  - no device weight tiles: PCMM keeps int8 digit planes, ≈ 0.7-1.05 GB for the model (≈ 349 M weights × 2-3 planes);
    diagonal arms stream weights into `stream_frames` device frames;
  - a ciphertext slab sized by the compiled programs. The compile check prints it per stage on the CPU: 1,711 MB for
    the model's largest stage under `program` [S6]; GPT-2 Prefill-128's record plan, 1,421 MB under `program` and
    6,705 MB under `lockstep` [S7];
  - keys derived from the plan (the packed layout regenerates `a`);
  - a `PtStore` of every plaintext the programs read.

  The session prints its own breakdown at setup (`sess.mem_report`; `driver.py:718`). Record it on the first arm, and
  decide then whether 48 GB would have been enough.

## 7. Not runnable yet

- a device run of the planned driver (§4: written, its CPU part verified [S12]);
- the lockstep arms: K-stream captures (§3);
- the diagonal arms on the GPU: the streamed-weights path in the driver (§3);
- `SPARSE_SLOTS=512,32,1`: no 32-slot landing in the runtime (§3).

## 8. After the first runs

- [ ] **Lockstep decision (pending, deferred).** With the per-stage split of the first planned runs in hand, report the
  norm chains' share of the pass and whether a device diagonal encoder exists, and put the lockstep-k-streams decision
  to the maintainer of this port: projected ≈ 1.1–1.25× aggregate at k = 4 beyond the k = 1 statistic packing; up to
  ≈ 1.4× on the diagonal-linear arms with the encoder; k ≤ 3–4 on a 96 GB card (projections [S9]; the mechanism and what
  lockstep needs from the driver: §3).
