# RUNSHEET_GPU — the fhe-ssm port on Perseus's `speedrun` branch, one GPU session from a fresh box

Every step is one subcommand of `tools/fhe_ssm/run_gpu.sh`. The script logs to `$OUT/session.log` and writes every
product under `$OUT` (default `<checkout>/.cache/fhe_ssm/gpu_<UTC date>`, gitignored). A duration or count carries a
source tag [Sn] when it was measured (the sources are listed in the research record of this runsheet); otherwise it
says **unmeasured**. Paths are relative to the checkout.

This session prices **a pass in any lane configuration** (the pass kernels: `python/perseus/examples/fhe_ssm/README.md`
"A pass in any lane configuration") against today's tick on one binary, accepts the device diagonal encoder (§3a), and
tries the deployment split. The plans and captures are cut on the CPU before the rental and shipped (§2): the box
captures and plans nothing. The steps are §3, in order; a session script in the research record runs them start to
finish and checks every record.

Status: the plan tooling, the CPU compile check, the change check, the port's fake and capture, and the planned driver's
CPU part (`run_planned.py --prepare-only`: the recorded stream, the compile check, the plaintext budget both ways, the
per-pass staging's dry run on the CPU stand-ins) run on a CPU-only build [S12, S18, S19]; `--prepare-only` compiles
through the CPU compile check on any build, the GPU build's included (`compiled_by cpu-check`: the dry run reads the
stand-ins' keys, which a device program no session has bound does not have), so step 1a runs on the box as it ran
here. **Nothing of the driver has run on a device**: the session, the graphs, the in-place restaging, every timing and every fidelity on a device are
this session's to measure.

## 0. Rules for every timed step

- One binary for every number that is compared. Build once and never rebuild mid-session. Wall-clock belongs to a
  build, so a time measured on another build (another runtime, another toolkit, another link) is never a denominator
  here. `run_gpu.sh` writes the commit, the `_pc` .so hash, nvcc's version and the load average beside every run; the
  numbers table prints them per run of a bracket and withholds the overhead line when a bracket spans two builds.
- An idle GPU and an idle host. `run` refuses a GPU with other compute processes. The planned path stages plaintexts
  from the host, so a CPU-heavy neighbour moves the timing. Prepares are CPU jobs: run them before the timed block.
- One `KEYS` mode across a bracket (`KEYS=inproc` for the timing brackets); the bundle run is last and never compared
  with them.
- A control inside every bracket: the arm's control first and again last (`RUN_TAG=ctlA`, `RUN_TAG=ctlB`, `TICKS=6`:
  three unfenced ticks each under `PROFILE_EVERY=2`). Quote an arm's difference only against the controls' mean and only
  when it falls outside their spread (max − min over both controls' unfenced ticks).
- `RUN_TAG` names every run: `run_<arm>[_bN][_<tag>]`; without it a second run of an arm overwrites the first.
- Fidelity next to speed for every row: the reply's RMS relative error against the exact reference over the live
  lane-slots, the worst live lane-slot, and top-1 agreement at every next-token lane-slot.

## 1. The box

| need | why | source |
|---|---|---|
| Linux x86-64; an NVIDIA driver ≥ 580; the CUDA 13.2 toolkit at `/usr/local/cuda-13.2` (or `CUDA_HOME`) | the kernels are built with nvcc from `CUDA_HOME`; a CUDA 13.x toolkit needs driver ≥ 580 | `crates/pc-kernels/build.rs:8`; `speedrun/MAINTAINING.md:13` |
| GPU: **96 GB** (RTX PRO 6000; sm_120 is the build's default arch) | this runtime's footprint for the model is unmeasured (§8); the pass kernels add store slots and 20 rotation keys | `README.md:33,173`; `crates/pc-kernels/build.rs:2-3` |
| host RAM: **≥ 128 GB** | the 24-block PCMM prepare needs more than 3 GB of host memory (a 3 GB guard stopped it without the pass kernels [S13]); the diagonal arms under `DIAG_SOURCE=generated` hold 2.625 GiB of float64 tiles; every configuration's masks are computed at setup (≈ 0.9 GB per configuration at the model shape [derived: 1,761 vectors of 256-512 KiB]) | `speedrun/runner/entry.sh:10-11` |
| rustup + cargo (`rust-toolchain.toml` pins 1.99.1), `uv`, git, tmux | the build | `README.md:22-27` |
| ≥ 50 GB disk | venv, cargo target, the staged tree (two diagonal captures of 472-475 MB each [S4]) | — |

## 2. Before renting: the staged tree

The box captures and plans nothing: `run_gpu.sh plan` reuses a plan already in `$OUT` after checking its `graph=` md5
against the capture there (`REPLAN=1` re-cuts it; `cfg-pcmm24-tick` is never cut on the box), and `capture` reuses a
directory holding `capture_config.json`. The tree below is laid out as the box's `$OUT` and built on the machine that cut
the plans; a manifest holds every capture's md5 (`cat block_*/graph.json | md5`, the value `make_plan.sh` writes into
`PLAN_CMD.txt`'s `graph=` line) and every plan file's, checked on the box before anything runs. `run`, `prepare` and
`keyspec` pass the driver `--capture $OUT/<the capture the arm's plan was cut from>` (a plan's `graph=` line names the
machine it was cut on, so without that flag the driver's md5 check would not run), and refuse to run without that
capture unless `ALLOW_UNCHECKED=1` (recorded as `capture.checked false`). What `capture.checked true` proves: the
shipped capture's md5 is the shipped plan's `graph=` md5, i.e. the staged tree is consistent, which the manifest also
proves; that the box's twin records the stream the plan was cut on is `prepare.check_errors {}`.

| box path under `$OUT` | what it is | capture md5 / plan |
|---|---|---|
| `cap_high__linear_pcmm_pcmm_wbits_24/` | the PCMM-24 capture under the high switches, `--pass-kernels off` [S11] | d67559a3d0eb2b4440528b14c32993f9 |
| `bootstrap_placements/plan_pcmm24/` | its LANDING=high plan: 802 refreshes per tick = 538 sites + 264 PCMM finishes, launches 15,190 [S11, S16] | cut on d67559a3… |
| `cap_high__linear_pcmm_pcmm_wbits_24_pass_kernels_on/` | the PCMM-24 arm under the pass kernels: six configurations' captures merged, every magnitude statistic its maximum over the six (`passmerge --rule max_each`) [S15, S16] | e60f2166c2a7fc49a5efdc2828e7dc22 |
| `cap_high__linear_pcmm_pcmm_wbits_24_pass_kernels_on__{tick,seqfresh,seqcarry,mixed,longdoc,longtick}/` | the six configurations' own captures (0 structural differences between them [S15]); `__tick` is what `cfg-pcmm24-tick` was cut on | the manifest |
| `bootstrap_placements/plan_cfg-pcmm24/` | the merged capture's LANDING=high plan: 802, the same routes as `plan_pcmm24`, launches 20,976 [S16] | cut on e60f2166… |
| `bootstrap_placements/plan_cfg-pcmm24-tick/` | the tick configuration's capture alone, LANDING=high: 802, launches 20,976 (a pure-kernel control for the tick) [S16] | cut on cf79cdaf7462a65bfdf156c1ec6a3224 |
| `cap_stock/` | the reference capture, diagonal linears, the stock landing [S4, S6] | 3f3c555c2e100acf5bc5cfc6575b4b55 |
| `bootstrap_placements/plan_ref/` | its stock plan, re-cut on a copy named `cap_stock` (diag-tests (e) checks the basename): 637, placements byte-identical to the first stock plan of this capture [S16] | cut on 3f3c555c… |
| `cap_stock__pass_kernels_on/` | the reference under the pass kernels, six configurations merged (`max_each`) [S16] | b3f021a99df258c0d10d1c5ae44d2288 |
| `bootstrap_placements/plan_cfg-ref/` | its stock plan: 637, launches 720,418 [S16] | cut on b3f021a9… |
| `cap_high__linear_pcmm_pcmm_wbits_24_b1/`, `bootstrap_placements/plan_pcmm24_b1/` | the 1-block smokes, `BLOCKS=1 run_gpu.sh capture / plan pcmm24` [S20] | the manifest |
| `cap_high__linear_pcmm_pcmm_wbits_24_pass_kernels_on_b1/`, `bootstrap_placements/plan_cfg-pcmm24_b1/` | `BLOCKS=1 … cfg-pcmm24`: the four capture configurations at one block (`passmasks.capture_configurations`: tick, one sequence fresh, the same carried, mixed; the lane-config files in `cfgs/`, which `CFG_CONFIGS=$OUT/cfgs` names to `capture`), kept beside as `…_b1__<label>` and merged under `max_each` (made one configuration per call under `CFG_ONLY=<label>`, each under its own memory guard, then merged by a call without it); the plan cut on the merge [S20] | the manifest |
| `cap_stock_b1/`, `bootstrap_placements/plan_ref_b1/` | `BLOCKS=1 … ref` (`smoke-diag`'s) [S20] | the manifest |

Also before renting: a fresh `--pass-kernels off` PCMM capture at the tip has the md5 of `plan_pcmm24`'s capture
(d67559a3…): the control plan binds the twin the box records [S17]. `KEYS` reaches the box on the session's start
command (`KEYS=…` on the tmux line); the session refuses to start without it.

## 3. Steps on the box

`served` is one letter per stage, `e` eager, `c` captured, `r` replayed (`crates/pc-py/src/program.rs` `run`); every
`run` below passes `--capture`, a `RUN_TAG` and, for the `cfg-*` arms, `--scenario $SCENARIO` (`--pass-kernels on` is
in their capture flags, which the run repeats). `PROFILE_EVERY=2` fences passes 0, 2, 4, …; every scenario holds an
unfenced pass whose configuration equals the previous pass's (`tick`: 1, 3, 5; `prefill:320`: 3 and 7; `doc:258:64`: 3
and 7; `mixed`: 3; `churn`: 5), so a steady-state wall exists without a fence.

| # | step | command | pass |
|---|---|---|---|
| 1 | setup and the build | the clone at the tip (upstream `speedrun` plus the branch's bundle; HEAD = the tip's full SHA; the weights' sha256 3ec3f1d8…293e784b), the data, the staged tree checked against its manifest; then `run_gpu.sh box`, `build`, `verify`, `bench` | stop on any failure; `verify`: `device classes missing: none`, the compile check built first, upstream's CPU tests and the port's light suite (`-m 'not gpu and not slow_shape'`, `-rs`) pass with no skipped test, the change check `OK` |
| 1a | the PCMM arm's CPU part under the pass kernels | `run_gpu.sh prepare cfg-pcmm24` (the 24-block PCMM prepare: more than 3 GB of host memory [S13]; `--prepare-only` compiles through the CPU compile check on this build too) | `prepare_cfg-pcmm24/record.json`: 25 programs, `compiled_by cpu-check`, `check_errors {}`, `static_stream true`, `capture.checked true`, the budget by references and by aliases, `staging_dry_run` |
| 2 | first device contact | `BLOCKS=1 RUN_TAG=smoke run pcmm24` | exit 0, `served` all `r` on the timed ticks, relErrRms < 1e-3 |
| 2a | the pass kernels' device tests | `run_gpu.sh pass-tests` (`tests/test_pass_gpu.py`: one block at d = 1,024, PCMM-24, LANDING=high, the `tick:4`, `mixed` and `doc:130:64` scenarios; `FHE_SSM_REQUIRE_DEVICE=1`, `CUDA_MODULE_LOADING=EAGER`) | `3 passed` |
| 2b | the per-pass staging's contacts | `BLOCKS=1 SCENARIO=tick TICKS=4 RUN_TAG=smoke_tick run cfg-pcmm24`, then `BLOCKS=1 SCENARIO=mixed RUN_TAG=smoke_mixed run cfg-pcmm24` | every timed pass: `served` all `r`; on every restaging pass (`staged` > 0): `counters.captures == 0` and `counters.invalidations == 0` (the in-place re-encode keeps the graphs: `crates/pc-py/src/store.rs:2-5`, `crates/pc-program/src/replay.rs:879`); relErrRms < 1e-3 over the live lane-slots and on the worst one (a non-finite value is the worst); `mem_free_bytes` equal on every pass. A failed contact withholds the `cfg-*` runs of step 3 (the controls run) |
| 3 | the PCMM bracket (whole model, one binary, `KEYS=inproc`, idle GPU and host, `PROFILE_EVERY=2`) | `TICKS=6 RUN_TAG=ctlA run pcmm24` → `TICKS=6 SCENARIO=tick RUN_TAG=tick run cfg-pcmm24` (the modularity overhead) → `SCENARIO=prefill:320 RUN_TAG=prefill run cfg-pcmm24` → `SCENARIO=doc:258:64 RUN_TAG=doc run cfg-pcmm24` → `SCENARIO=mixed RUN_TAG=mixed run cfg-pcmm24` → `TICKS=6 RUN_TAG=ctlB run pcmm24` → `TICKS=6 SCENARIO=tick RUN_TAG=tick run cfg-pcmm24-tick` (the pure-kernel control plan) | every record: `capture.checked true`, `prepare.check_errors {}`; the numbers table (§4) with every run's build identity (commit, dirty, `_pc` .so) and load beside it |
| 4 | the device diagonal encoder's acceptance | `run_gpu.sh diag-tests` (checks 1-3) → `RUN_TAG=smoke_diag run_gpu.sh smoke-diag` → `DIAG_E=1 DIAG_F=1 run_gpu.sh diag-tests` (on the shipped `plan_ref`) | §3a |
| 5 | the diagonal bracket under the encoder | `DIAG_SOURCE=generated TICKS=6 RUN_TAG=ctlA run ref` → `DIAG_SOURCE=generated TICKS=6 SCENARIO=tick RUN_TAG=tick run cfg-ref` → `DIAG_SOURCE=generated TICKS=6 RUN_TAG=ctlB run ref`; only once checks 1-3 left their `.passed` markers and `smoke-diag` passed | as step 3 |
| 6 | the deployment split | `run_gpu.sh keyspec cfg-pcmm24` → `run_gpu.sh keygen cfg-pcmm24` → `KEYS=bundle FHE_SSM_KEYS=$OUT/keys_cfg-pcmm24 TICKS=6 SCENARIO=tick RUN_TAG=bundle run cfg-pcmm24` | `keys.mode bundle`, `setup.has_secret false`; no timing comparison with step 3 |
| 7 | wrap up | the checks over every record, the numbers table, a tar of the records (without the staged tree and the `.work` directories); pull them home | — |

Never run `capture` or `plan` for a shipped arm expecting a new product: both reuse what is in `$OUT`. A `cfg-*` arm's
`capture` is four configuration captures merged under `max_each`: `CFG_CONFIGS=<dir>` names the directory of the four
lane-config files (`tick`, `seqfresh`, `seqcarry`, `mixed.passes.json`; default `passmasks.capture_configurations`
written to `<capname>.cfgs`), `CFG_ONLY=<label>` makes one of the four and stops (a small machine's memory guard; a call
without it reuses them and merges). A configuration capture is reused only when its `capture_pass.json` is the last pass
of its lane-config file, and a merged capture under `CFG_CONFIGS` only when its four are (`passmerge stamp`); a mismatch
stops the call, naming the capture and what differs. `TICKS` is the
pass count of `SCENARIO=tick`; the other scenarios set their own (`prefill:320` 8 passes, `doc:258:64` 8, `mixed` 5,
`churn` 6). Durations on a device: **unmeasured**. The twin's recording at the model shape: 437 s for the diagonal
arm's lean prepare on this machine's 16 GB laptop [S19]; it is part of every `run`'s setup.

### 3a. The device diagonal encoder's acceptance (step 4), in order

Nothing of the encoder's CUDA has been compiled or run. Each check names what a pass means; a failure stops the
sequence and prints the log's `FAILED` / `panicked at` / error lines (`run_gpu.sh diag-tests`; logs in
`$OUT/diag_tests/`). `diag-tests` builds with the same `PERSEUS_CORE_ARCH` / `PC_NVCC_JOBS` as `build`, sets
`CUDA_MODULE_LOADING=EAGER` and `FHE_SSM_REQUIRE_DEVICE=1` (a test that cannot find the encoder fails, never skips),
requires each step's pass line, leaves `<step>.passed` (skipped on the next call unless `DIAG_RERUN=1`), and runs the
steps `DIAG_ONLY` lists (default 1,2,3, plus e with `DIAG_E=1` and f with `DIAG_F=1`).

| # | check | command (inside `diag-tests`) | pass |
|---|---|---|---|
| 1 | the kernels against host references, every built instance | `cargo run --release -p pc-kernels --bin pc-kernels-selftest -- --no-bench` | exit 0 and the final `[selftest] instances ...: total mismatches 0` line; per instance 3 `encode`, 4 `encode_many` and 2 `diag_gather` lines, each at `0 mismatches` (18 lines on the default w32 build) |
| 2 | (a) generated slots and rows == the eager encode of the host vector; (b) `encode_many` == single encodes over B = 1, 32, 128; (c) a fused sum == `PtStore::mult_pt_sum` | `cargo test --release -p pc-program --features gpu --test diag_gen_gpu -- --ignored --test-threads=1 --nocapture --skip f_device_ms_per_plaintext` | `test result: ok. 3 passed`; every `[diag_gen]` line at 0 words differing |
| 3 | (a, b) the session's probe at the model's width against the port's oracle; (d) one block under `generated` against the exact forward, 3 ticks; a perturbed tile (negative control); chunks of 1 and 128 | `pytest -q -rs -x -m gpu python/perseus/examples/fhe_ssm/tests/test_diag_gpu.py -k "not model_shape"` | `3 passed`: probe mismatches 0, 0, 0; `relErrRms_vs_exact` < 1e-3 per tick; the perturbed run NOT below 1e-3; `splits` 0; free memory equal on every timed tick |
| 4 | the first planned run | `smoke-diag` (`ref` at `BLOCKS=1` under `generated`; the shipped `cap_stock_b1`, `plan_ref_b1`) | exit 0; `setup.diag` tiles = uploaded = 5, `missing` empty |
| 5 | (e) the model, 2 ticks on the shipped whole-model `plan_ref`, `--diag-probe 32` | `DIAG_E=1` (needs `DATA` and `$OUT/bootstrap_placements/plan_ref`, cut on a capture named `cap_stock`, 25 block plans) | `1 passed`: 120 tiles uploaded, 8,448 diagonal sums, probe mismatches 0, `splits` 0, free memory equal on both ticks, top-1 > 0.95 |
| 6 | (f) device ms per plaintext, B = 1, 32, 128, 18 and 10 limbs | `DIAG_F=1` | `test result: ok. 1 passed` and the `[diag_gen] (f)` lines |

A values run and a generated run never agree word for word (host vs device encoders, ±1 in rare coefficients), so (d)
and (e) are tolerance scores against the exact forward; the word-for-word checks are 1-3, inside one encoder.

## 4. The numbers table

Computed only from **unfenced passes that restaged nothing** (`fenced false`, `staged 0`), each row named by its run
and pass index:
- time per pass: the pass's `seconds` (issue to the reply read, the restaging inside the clock);
- token-values per pass: `token_values`, the live lane-slots (sequence positions plus live conversations; padding and
  parked lane-slots excluded); conversations per pass: `conversations`;
- tok/s aggregate = token-values / seconds; tok/s per sequence = that sequence's positions (`sequences`) / seconds;
  tok/s per conversation = 1 / seconds;
- the modularity overhead = the `cfg-pcmm24` (`cfg-ref`) tick's unfenced passes against the two controls' mean, quoted
  only when the difference falls outside the controls' spread (max − min over both controls' unfenced ticks);
- the restage cost: a restaging pass's `seconds` minus the next steady pass's in the same configuration (`stage_s`
  is host issue time only: `PtStore.stage` encodes on the compute lane, asynchronously to the host, so the device part
  of a restage lands in the pass's `seconds`);
- fidelity beside every time: `relErrRms_vs_exact`, its worst live lane-slot, `top1`.

The per-stage split comes from the fenced passes (`stages[].wall_s`). There is **no per-step split on this path**:
`PF_NVTX_STEPS` is a torch NVTX range around Python `rt.step` scopes (`crates/pc-config/src/fields.rs:502-504`,
`python/perseus/impl/rt.py:208-249`) and the Rust replayer emits none (`git grep -i nvtx -- crates` finds `fields.rs`
only), so the norm chains' share of a pass stays unmeasured here; the per-stage split is what the first runs give (§10).

## 5. The axis matrix

Columns: the capture flags (`run_fake.py`), the plan axes (`make_plan.sh`), the run flags (the driver), what the arm
isolates, and its control. `run_gpu.sh arms` prints it with each arm's capture name.

| arm | capture | plan | run | isolates | control |
|---|---|---|---|---|---|
| `ref` | — (every axis at its default: the reference circuit, diagonal hint 1) | `LANDING=stock` | — | the reference circuit; this runtime's landings without StC-first; diagonal linears | itself (first and last) |
| `depth2` | `--newton-form depth2` | stock | — | 2 products per Newton step instead of 3 | `ref` |
| `persite` | `--newton-sites per-site` | stock | — | per-site Newton counts | `ref` |
| `packstats` | `--pack-stats halves` | stock | — | one Newton chain for both halves' statistics | `ref` |
| `high` | — | `LANDING=high` | — | dense refreshes StC-first at 28, SparseB-512 at 26 | `ref` |
| `pair` | as `high` | `LANDING=high PAIR=1` | `--schedule siblings` | refresh pairs on the real-payload dense StC-first sites | `high` |
| `sbk16` | as `high` | `LANDING=high SPARSE_B_K=16` | — | SparseB's EvalMod at K = 16, replay only | `high` |
| `emfuse` | as `high` | `LANDING=high RUNTIME_EXTRA=PC_EM_FUSE_RESCALE=1` | — | EvalMod's relinearization and rescale fused (a kernel switch) | `high` |
| `pcmmtrunk24` | `--linear pcmm-trunk --pcmm-wbits 24` | `LANDING=high` | — | PCMM on the trunk's linears, 24-bit weights | `high` |
| `pcmm24` | `--linear pcmm --pcmm-wbits 24` | `LANDING=high` | — | every linear PCMM | `high` |
| `best` | depth2 + per-site + pack-stats + pcmm 24 | `LANDING=high PAIR=1` | `--schedule siblings` | the stack | `ref` and `high` |
| `cfg-pcmm24` | `--linear pcmm --pcmm-wbits 24 --pass-kernels on` | `LANDING=high` (shipped: the merged capture's) | `--scenario $SCENARIO` | the pass kernels: one op stream for every lane configuration, the masks restaged per pass | `pcmm24` |
| `cfg-pcmm24-tick` | as `cfg-pcmm24` | `LANDING=high` (shipped: the tick configuration's capture alone) | `--scenario $SCENARIO` | the merged magnitudes' share of the overhead (its plan's correction factors are the tick's) | `cfg-pcmm24` |
| `cfg-ref` | `--pass-kernels on` | `LANDING=stock` (shipped: the merged capture's) | `--scenario $SCENARIO` | the pass kernels on the diagonal arm (a device arm under `DIAG_SOURCE=generated`) | `ref` under `generated` |
| `lockstep2`, `lockstep4` | `--streams K` (**does not exist**) | `LANDING=high` | `--schedule lockstep` | K sets of 64 conversations through one stream | `high` |

Planned counts [S6, S10, S11, S16] (`summary.total_bootstraps` per tick): `ref` stock **637** (446 placed, 143 fired
hints, 48 deliberate); `high` **609**; `pcmm24` **802** = 538 refresh sites + 264 PCMM output finishes; `cfg-pcmm24`
**802** (the same 538 sites and routes, `dense_stcf@26:48 s512_sb@24:490`; launches 20,976 against `pcmm24`'s 15,190,
+5,786 = +38.1 % [S16]); `cfg-pcmm24-tick` 802 (launches 20,976); `cfg-ref` **637** (`dense@32:191 s512_sb@22:446`;
launches 720,418 against 714,610, +5,808 = +0.81 % [S16]). The kernels add +2,448 ct×pt, +672 rotations, +192
conjugations and +0 refreshes per pass, measured in the model-shape captures [S15] (+102 / +28 / +8 / +0 per block on the
fake [S14]): the overhead is in launches, not in
refreshes; its time is this session's to measure.

What blocks which arms:
- **Lockstep**: `Program.compile(schedule="lockstep")` reorders what was captured (`crates/pc-py/src/program.rs:68-84`);
  K streams need K ciphertext sets in each captured stage and registers per stream. `run_gpu.sh` refuses these arms.
- **Diagonal linears on the device** (`ref`, `depth2`, `persite`, `packstats`, `high`, `pair`, `sbk16`, `emfuse`,
  `cfg-ref`, and `pcmmtrunk24`'s `W_k` / `W_r`) run only under `DIAG_SOURCE=generated` (the device diagonal encoder):
  under `values` a planned session keeps every plaintext its programs read resident, 344,064 diagonals per tick
  (2,544,444,112,896 B = 2.31 TiB at 28 rows, one slot per distinct value [S19]), refused above `--pt-budget-gb` (40).
- **Captures at the runtime's levels**: `run_gpu.sh` passes `--levels preset` (ceiling 44, landing 32).
- **`SPARSE_SLOTS=512,32,1`**: the runtime lists no landing for 32 slots; not in the matrix.

## 6. The driver's contract

`DRIVER` (default `perseus.examples.fhe_ssm.run_planned`) is invoked as

```
python -m $DRIVER --weights W --seeds S --config C --levels preset [--blocks N] [run_fake's axis flags]
                  --plan PLAN_DIR [--capture DIR] --ticks T --profile --profile-every K --keys inproc|bundle|DIR
                  [--schedule program|siblings] [--diag-source values|generated]
                  [--pass-kernels on --scenario tick[:K]|prefill:T|doc:T:READERS|mixed|churn | --lane-config FILE]
                  --out record.json
python -m $DRIVER … --plan PLAN_DIR --prepare-only [--lean] … --out record.json        (the CPU part, no session)
```

The landing's runtime switches are in the environment. The driver must:
- build the model on the CPU twin with the same axis flags (`--pass-kernels on` among them);
- under the pass kernels, record the probes in one fixed configuration (the first pass's), refuse a stream that is not
  static across them, keep each pass-mask product as its mask id and values, alias the store keys of the masks by mask
  id and the static ones by value digest once before any graph capture, size the store by the distinct (alias, level)
  pairs, stage the first pass's masks with the static plaintexts, compute every configuration's masks at setup, and
  before every pass re-encode in place only the keys whose values change; pack block 0's input and unpack the reply in
  lane-slot order (one permutation on each end);
- run one untimed warm-up (the first pass's configuration on random public tokens; it captures the CUDA graphs), then
  the timed passes (ticks), each scored per live lane-slot against the exact reference;
- with `--profile`, fence the device after every stage so each stage has a wall (an attribution, not the production
  wall); `--profile-every K` fences every K-th pass only;
- refuse diagonal linears above `--pt-budget-gb` under `--diag-source values`.

`record.json`, one object per run:
- per run: `build_id` (commit, dirty, `_pc` .so, nvcc, arch; `run_gpu.sh` adds the load average in `build_id.txt`),
  `keys.mode`, the plan's `axes` / `recipe` / `runtime` lines, `capture` (`checked`: the shipped capture's md5 equals
  the plan's `graph=` md5, the staged tree's consistency; `md5`, `plan_md5`), `prepare`
  (`check_errors`, `static_stream`, `refreshes_planned`, per program `n_ops` / `n_launches` / `n_bootstraps` /
  `bts_batches` / `n_pairs` / `n_finishes` / `mem`, `plaintexts`: by references and `by_alias`, `sums`), `pass_list`
  (the scenario or file, the configurations, `config_ids`, the list, the generated tokens resolved), `setup`
  (`mem_report`, free / total memory, `plaintexts_staged`, `pass_masks`), the nvidia-smi peak (`nvsmi.csv`);
- per pass (`ticks[]` under the pass kernels): `pass`, `config`, `config_id`, `segments`, `describe`, `fenced`,
  `staged`, `stage_s`, `masks_s`, `restaged`, `seconds`, `issue_s`, `served`, `counters` (the deltas of
  `planned_stats`: runs, captures, replays, eager, invalidations, host_us, launch_us, …), `mem_free_bytes`, `stages`
  (fenced passes), `token_values`, `sequences` (positions per sequence), `conversations`, `parked`, `padding`,
  `relErrRms_vs_exact` (over the live lane-slots), `relErrRms_vs_exact_worst_lane_slot`, `logits_relErrRms_vs_exact`,
  `top1` (agree / of at every next-token lane-slot, which ones disagree).

## 7. What each run writes

- `$OUT/box.txt`, `$OUT/session.log` (every step, with the build id), `$OUT/build.log`, `$OUT/bench_bts.txt`
- `$OUT/cap_<…>/`: a capture (`graph.json` per stage, `capture_env.json`, `capture_config.json`)
- `$OUT/bootstrap_placements/plan_<arm>[_bN]/`: `block_<b>_placement.json` (stamped), `PLAN_CMD.txt`,
  `plan_rot_steps.json`, `summary.txt`, `compile_check.txt` / `.json`
- `$OUT/prepare_<arm>[_bN][_<tag>]/`: `record.json`, `prepare.log`
- `$OUT/run_<arm>[_bN][_<tag>]/`: `record.json`, `run.log`, `nvsmi.csv`, `build_id.txt`; `$OUT/keys_<arm>[_bN][_<tag>]/`
  (keyspec, keygen)
- `$OUT/diag_tests/`, `$OUT/pass_tests/`: one log per step and its `.passed` marker

## 8. VRAM: what is known

- **The fork's C++ runtime** (another binary): the planned plateau was 35,961 MiB on an RTX PRO 6000 [S8].
- **This runtime: unmeasured.** The terms: PCMM's int8 digit planes (≈ 0.7-1.05 GB for the model), or under
  `generated` the diagonal tiles (2.625 GiB) and the encoder's scratch; a ciphertext slab sized by the compiled
  programs (1,711 MB for the model's largest stage under `program` [S6]); the keys (the pass kernels add 20 rotation
  steps: `−1..−7, −8, −16, −24` and `+1..+7, +8, +16, +24`; key bytes per step unmeasured); the plaintext store, one slot
  per distinct (alias, level) pair: on `cfg-ref` 1,796 pass-mask pairs (1,761 mask ids) plus the static plaintexts
  [S19]; the driver's budget prices every slot at `max(28, 54 − min level)` rows × 2^16 × 4 B (17.7 GiB for the 2,589
  slots of `cfg-ref` under `generated` [S19]), an **upper bound**: the store carves `54 − level` rows per key
  (`crates/pc-py/src/session.rs` `level_classes`, `crates/pc-py/src/pt_store.rs` `slot_caps`), so the `--pt-budget-gb`
  gate is conservative and the measured figure is `setup.mem_report`.

  The session prints its own breakdown at setup (`sess.mem_report`). Record it on the first arm.

## 9. Not measured on this path yet

- anything on a device (§3: written, its CPU part verified);
- the lockstep arms (§5);
- the program-variant route for passes with no sequence (today's program for those, the kernel program otherwise):
  designed, not built; the bracket decides whether it is needed.

## 10. After the first runs

- [ ] **Lockstep decision (pending, deferred).** With the per-stage split of the first planned runs in hand, report the
  norm chains' share of the pass (on this path: not measurable below the stage, §4) and whether a device diagonal
  encoder exists, and put the lockstep-k-streams decision to the maintainer of this port: projected ≈ 1.1–1.25×
  aggregate at k = 4 beyond the k = 1 statistic packing; up to ≈ 1.4× on the diagonal-linear arms with the encoder;
  k ≤ 3–4 on a 96 GB card (projections [S9]).
