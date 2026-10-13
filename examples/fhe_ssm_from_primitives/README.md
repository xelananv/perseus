# fhe-ssm on `perseus.impl`

`examples/fhe_ssm_from_primitives/` is the fhe-ssm recurrent language model (tag `pbd430a`: 24
blocks, d = 1,024, d_ffn = 4,096, learned polynomial gates, Newton RMSNorm, two carried values per
block) written on Perseus's leaf-primitive ops protocol, in the shape of
`examples/gpt2_from_primitives`. The model, its training and its published FIDESlib run live in the
fhe-ssm repository (github.com/xelananv/fhe-ssm); paths such as `spec_decode/plain_recurrent.py`
below are relative to it. The checkpoint, the Newton seeds and the config are on Hugging Face
(huggingface.co/xelananv/fhe-ssm-pbd430a: `pbd430a_weights.npz`, `pbd430a_seeds.json`,
`pbd430a_config.json`).

By default it serves 64 conversations per ciphertext at ring 2^16: the dev branch's filled layout
with one conversation on each axis of a complex slot (`layout="filled"`, `payload="pairs"`).
32 conversations on real slots (`payload="real"`) and one sequence in the cachemir decode
packing (`layout="decode"`) are the other two arms.

It runs on Perseus's numpy fake session (CPU, no GPU, the runtime's level accounting); the same
code is meant to run on a live session through `ImplModel`.

## Run on the fake

From the repository root, in an environment with this checkout's `perseus` importable:

```bash
python -m pytest -q examples/fhe_ssm_from_primitives/tests
python -m examples.fhe_ssm_from_primitives.run_fake \
    --weights pbd430a_weights.npz --seeds pbd430a_seeds.json --config pbd430a_config.json \
    --tokens 4 --blocks 24 --form perseus \
    [--layout filled|decode] [--payload pairs|real] [--pair-route split|lane] [--conversations N]
```

Each conversation gets its own random token per tick; the references run all of them as rows.

Per tick the driver prints the reply's relative RMS error against two references and the
operation counts:

| field | meaning |
|---|---|
| `relErrRms_vs_mirror` | the same kernels on plain numpy (`ref.MirrorModel`): a non-zero value is a layout or protocol error |
| `relErrRms_vs_exact` | the trainer's forward with the closed-form rsqrt (`ref.ExactModel`): the Newton approximation |
| `relErrRms_vs_exact_worst_conversation` | the same, for the conversation that fares worst |
| `bootstraps` = `reactive` + `hints` + `carry` | refreshes fired at the ceiling, at the port's hint sites, and the canonical carry (2 per block) |
| `bootstraps_by_step` | the same refreshes by step scope (`tm.norm`, `cm.v`, `carry`, ...) |
| `mult_cc`, `mult_pt`, `mult_scalar` | ciphertext products (incl. squares), plaintext products, scalar products |
| `rotations`, `conjugations` | rotations; conjugations (the pair route's splits) |
| `out_level`, `state_level_max` | the reply's level and the deepest carried value (after the carry: the landing level) |
| `model_s`, `refs_s`, `peak_rss_gb` | the fake's own time for the tick, the references' time, peak memory: CPU numbers, not GPU ones |

The report also records the level rules, the hint convention, the norm constants, Perseus's commit
and the sha256 of the weights, seeds and config.

`--rules runtime` (the default) runs Perseus's fake with the runtime's level rules
(`runtime_fake.py`: a product lands at its operand's effective level with the rescale pending, a
bootstrap lands unrealized); `--rules stock` runs the fake as shipped, which charges each rescale at
once and so refreshes one product earlier. Only the runtime rules give eager counts a GPU session
would show. Both default to the n32 decode preset (ceiling 46, landing 34, two primes per level);
`--limit --bts-out --unit` change them, `--no-carry` with a huge `--limit` turns the levels
into a depth measurement. `--form depth2` evaluates the Newton step as `1.5 y + ((-0.5 v) y) y^2`
(two levels per step) instead of Perseus's `inv_sqrt_newton` (three); the polynomial is the same.

## Measured on the fake (published weights)

Three ticks of the 24-block model at ring 2^16, every lane a live conversation, the runtime's level
rules (ceiling 46, landing 34, two primes per level), eager refreshes (no plan). Counts per tick:

| arm | conversations | relErrRms vs mirror | vs the trainer's forward (worst conversation) | bootstraps | plaintext products | rotations | conjugations |
|---|---|---|---|---|---|---|---|
| pairs, split (default) | 64 | 6.1e-15 – 7.1e-15 | 5.7e-15 – 6.0e-15 (1.9e-14) | 1,968 | 345,265 | 14,372 | 265 |
| pairs, lane | 64 | 6.0e-15 – 6.9e-15 | 5.7e-15 – 6.1e-15 (2.1e-14) | 2,817 | 344,737 | 13,882 | 5,637 |
| real | 32 | 5.4e-15 – 8.2e-15 | 5.3e-15 – 1.2e-14 (5.7e-14) | 1,052 | 344,737 | 13,882 | 0 |

The errors are float64 rounding: the fake computes in the clear, so these check the layout, the
packing, the split and the kernels, not CKKS noise. The bootstrap counts are the fake's reactive
refreshes plus the hints and the carry; a plan from Perseus's planner places them instead.

## Run on a GPU: eager, capture, plan, planned

`RUNS.md` has the same chain from a fresh clone to a 148-tick planned run, with the inputs, the
expected durations and what each run answers.

Needs a GPU of compute capability 8.0 or newer (FIDESlib32bits's `CKKS/Discard.cu` uses a PTX
instruction older GPUs lack) and, per Perseus's README, 48 GB or more; the build is
`colab_build.py` on any Ubuntu machine with root (it is not Colab-specific: `--logs DIR`).

```bash
source scripts/local_env.sh                    # the n32 chain's defaults
M="--weights pbd430a_weights.npz --seeds pbd430a_seeds.json --config pbd430a_config.json"
python -m examples.fhe_ssm_from_primitives.gpu_smoke $M --blocks 1 --out smoke.json
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode eager --ticks 2 --out eager.json
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode capture --graph-dir graphs/fhe_ssm \
    --tokens tokens.json --ticks 2 --out capture.json
PLAN_HARD_ENV_CAP=1 python -m examples.fhe_ssm_from_primitives.plan --graph-dir graphs/fhe_ssm --name fhe_ssm  # CPU only
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode planned --plan-dir bootstrap_placements/fhe_ssm \
    --tokens tokens.json --ticks 8 --out planned.json
```

`gpu_run.py` checks every tick against the mirror and the closed form (reply, logits, top-1 on
the tied head) and rewrites its report after each tick. The capture records tick 1 (tick 0's
carried state is zero), so two ticks are enough; its magnitudes come from the device probe
(`FHE_MAG_GPU=0` selects the host decrypt, which takes hours at ring 2^16). `plan.py` runs
Perseus's placer with the n32 recipe of `examples/gpt2_from_primitives/make_plan.sh` (read back
from that file by the tests) and stamps the capture's contract into every plan, so a planned run
in a different environment refuses it. It also writes the plan's `PLAN_CMD.txt` (graph, recipe, and
the switches the plan needs at run time: `FIDESLIB_BTS_REAL=1` where a refresh takes the real-payload
route, `FIDESLIB_BTS_RAISE_DROPS` for the raise variants it uses), which `gpu_run.py --mode planned`
exports before the session opens unless they are already set. `--tokens` teacher-forces a serve run's token record, e.g.
the published 148-tick run's (fhe-ssm repository,
`results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/tokens.json`). `--free-run`
keeps each conversation's prompt teacher-forced and then feeds back the model's own top-1 on the
tied head, which is how that run generated its tokens (its record says `freeRun`); each tick then
records the tokens fed, the top-1s, and the share that picked the token the record has next.

The timings and plan sizes quoted in this section and in `RUNS.md` were measured before this branch
existed, on one RTX A6000 with FIDESlib at the tree of 374f6df built with the arch-guarded patch
described below: the eight-tick runs with one build of the `fhe-ssm-port` branch at 471a93e, the same
binary for every run, and the step-by-step chain of `RUNS.md` with a fresh clone of 5d82074. One
figure, named where it appears, is an RTX PRO 5000's, where FIDESlib builds without the patch. This
branch tracks upstream dev, with its FIDESlib pin and planner recipe: its planner (dev's, with the
re-cut passes keeping the forced refreshes) plans that clone's capture at the same 953 refreshes per
tick, and the timings are to be measured anew.

Two switches matter on this model's graphs, whose residual stream crosses the blocks without a
refresh. `PLAN_HARD_ENV_CAP=1` keeps the refresh envelope absolute: a block the placer cannot
plan inside it is refused and the previous block is re-planned with a terminal-exit refresh (the
default relaxes the envelope instead and plants refreshes where a bootstrap returns garbage). The
ceiling, `--set MAX_LEVEL=48` or the recipe's 50: at 48 the plan carries no warning; at 50 the
placer warns about values predicted at level 50 with no refresh, the planned runtime accepts them
(a predicted ceiling level is not a divergence), and the plan has fewer refreshes and the lower
error of the two. Measured on one RTX A6000, eight ticks of the same 64 conversations in one
binary: eager 1,839.5 s, planned 1,345.0 s at 48, 1,143.8 s at 50, with the reply error 4–8×
lower under either plan and top-1 1.0 on every planned tick.

The weight diagonals are encoded on the device at their use (`FHE_GPU_ENCODE=0` selects the
library's host encode in the same binary, 7.3× slower per block on the RTX A6000 and 13.4× on an RTX
PRO 5000, each within one build of 471a93e; FIDESlib's own device encode,
`FIDESLIB_GPU_ENCODE`, is a separate switch that this example leaves alone): a tile registered on the device keeps no host
copy, each giant group is one batched encode per input level, and a group the encoder cannot
serve (a pending rescale on a one-prime chain) takes the CPU route from the tile fetched back. On
a GPU below compute capability 9.0 the pinned FIDESlib32bits does not build as is (one kernel uses
thread-block clusters); the fork's measurement notes carry an arch-guarded patch for that file.
The encoder's and the probe's own checks are ctest tests labelled `gpu` (`ctest -L gpu` in the
tests build directory, no further environment); the repository's CI has no GPU runner, so they
run by hand. `python examples/fhe_ssm_from_primitives/lint.py --check` runs ruff and mypy over
this directory with its own, stricter configuration.

One known limit of the device route's bookkeeping, noted rather than changed for now: a tile
fetched back off the device (`EncodedTile.with_rows`) is fetched on every `apply_tiles` call that
cannot take the device route, not once per tile; on the composite chain this example ships on, no
call takes that path.

## Colab

`colab_check.ipynb` runs, on a Colab runtime: the tests and the lint, the published weights
from Hugging Face (`colab.fetch_model`, hash-checked), and two ticks of the full model on the fake
(`colab.fake_check`, about a minute per tick). On a GPU runtime it then builds Perseus's CUDA
extension (`colab_build.py`: installs what is missing, runs the repository's build scripts, one
log per step) and runs `gpu_smoke.py` on a live session sized for this model: an encryption round
trip, one linear tile and two ticks of one block, against the same references.

## Layout

**Filled** (`filling.py`, `geometry.FilledGeometry`): channel `i` of conversation `k` sits at slot
`i*t + k`, `t = N / d = 32` at ring 2^16 (the dev branch's `diagonal::encode_linear_input`). A
rotation by `j*t` moves every conversation `j` channels along and wraps where each conversation
wraps, so the diagonal method needs no lo/hi halves. Each linear is cut into `d x d` tiles (in one
piece a `1024 -> 4096` matrix would cap the batch at `N/4096 = 8`); a `d_ffn`-wide value is four
ciphertexts. The baby rotations of an input are made once for every tile that reads it, and each
giant group sums all input tiles before its one rotation. The norm's statistic is a per-conversation
rotate-and-sum over steps `t, 2t, ..., N/2`.

**Pairs** (`pairs.py`, `geometry.PairGeometry`): conversation `k` on the real axis of those slots and
`k + t` on the imaginary axis, 64 per ciphertext. Linears, plaintext products and bootstraps act on
both axes at once. A ciphertext product does not, so each nonlinear section (the norms, the time-mix
gate, the channel-mix activation times gate) takes one of two routes:

- `split` (default; the dev branch's `norm_token_pair` / `gelu_token_pair` pattern): one
  conjugation gives the real halves `x + conj x = 2 Re x` and `i (conj x - x) = 2 Im x`, the real
  kernels run on each, and `a + i b` repacks them. No level: the factor 2 rides the norm's masks
  (`gain = 0.5`) and the pre-halved `c`, `dd`, `W_k`, `W_r` that make the gates' inputs.
- `lane`: Perseus's `LanePackedOps` (`lane_mult` / `lane_square`), with plaintext adds landing on
  both axes. Each ciphertext product costs an extra level for its `1/4`.

Per tick of the 24-block model at ring 2^16 with 64 conversations as split pairs (derived from the
code, checked at the full shape in `tests/test_filled_model.py`): 336 tiles x 1,024 = 344,064
plaintext products for the linears; 598 rotations per block (558 for the linears, 40 for two norms
on two halves) plus 20 for the final norm; 11 conjugations per block plus 1. The norms' ladders are
the plain ten rungs on both routes: the example exports `LADDER_RADIX_BITS=0` before a session or a
fake is made (`port_env`). The two routes read that variable with different defaults (Perseus's fake
hoists radix groups, the C++ `rotate_and_sum` does not), and either hoists a group wherever the
session holds its rotation keys, which this layout's own steps do for two of the norm ladder's three
groups; one value on both keeps the fake's counts a GPU session's, and the plain ladder is what was
measured. The C++ reads the variable once per process, so the example runs in a process of its own.

**Parking**: fewer conversations than the capacity are filled with copies of conversation 0
(`geometry.park`). A lane left at zero enters every norm below the Newton seed's window and its
intermediates grow (8x the full batch's peak on the test model); a bootstrap needs every slot in
range. The parked outputs are dropped.

**Decode** (`geometry.SlotGeometry`): one token per ciphertext in the cachemir decode packing
(channel `i` at slot `i*t`), Perseus's GPT-2 decode layout.

## Scale folds (`scales.py`, `calibrate.py`)

A CKKS bootstrap adds a roughly constant absolute error per slot, about `2.2e-5 * 2^CF` on the
n32 chain (`perseus/plan/btserr.py`); a value refreshed while it is much smaller than its band
loses its precision. In this model the time-mix gate's pieces (`p1 z`, `p2 z^2`, the gate's
output; typical size 1e-3 to 1e-2) and the channel-mix activation times gate before `W_v`
(typical size 1e-3) are refreshed far below the default CF 6 band. Three exponents per block
lift them, each folded into constants the block multiplies anyway and undone in the next linear,
all exact powers of two:

| exponent | folds | limited by the products it scales |
|---|---|---|
| `gate_tm` = g | `p0, p1, p2` times `2^g`, `W_out` divided by `2^g` | `p1 z`, `p2 z^2`, the gate's output |
| `act_cm` = a | `q0..q3` times `2^a` | `q1 k`, `q2 k^2`, `q3 k`, `(q3 k) k^2` |
| `gate_cm` = b | `r0, r1, r2` times `2^b`; `W_v` divided by `2^(a+b)` | `r1 r`, `r2 r^2`; and the product before `W_v` by `a + b` |

Each exponent is the largest with `kappa * 2^e * max <= 32` (kappa = 2, Perseus's planner margin;
32, the amplitude up to which CF 6 stays near its floor in Perseus's measured table), the maxima
taken with the closed-form forward over token streams. `scales_pbd430a.json` is calibrated on the
published 148-tick run's 64 prompts and generations (fhe-ssm repository,
`results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/tokens.json`), ticks 0-73,
and checked on ticks 74-148: no held-out maximum exceeds its target (smallest margin 1.68 of 2).
The drivers use it by default (`--scales shipped`; `--scales none` turns the folds off):

```bash
python -m examples.fhe_ssm_from_primitives.calibrate --weights pbd430a_weights.npz \
    --seeds pbd430a_seeds.json --config pbd430a_config.json --tokens tokens.json --out scales.json
```

Predicted with Perseus's measured accuracy table (noise of the table's size added at every
bootstrap of the fake; random tokens, 3 ticks, 64 conversations; a prediction, not a GPU
measurement):

| bootstrap setting | logits relErrRms vs the closed form | top-1 agreement |
|---|---|---|
| CF 6 everywhere (no plan), no folds | 0.33 - 0.41 | 38 - 44 % |
| CF 6 everywhere (no plan), folds | 0.015 - 0.020 | 97 - 98 % |
| CF chosen per refresh, folds | 0.0028 - 0.0034 | 100 % |

So run this model with the folds; for its full fidelity, also under a plan.

## What is folded (all exact)

- The norm gains `g1`, `g2` into the rows of `W_in`, `W_k`, `W_r`; `alpha_res` into `W_out`;
  `g_out` stays a per-channel plaintext on the reply (`weights.py`).
- The norm's frame constant rides two lane masks, both one product from the input: `sqrt(sigma/d)`
  for the statistic (its all-slot sum of squares is `sigma mean(x^2)` directly) and `sqrt(sigma)`
  for the output, so the normalized value comes out at unit rms (`norm.py`).
- The first Newton step from the constant seed is affine in the statistic (no ciphertext product).
- On split pairs, the split's factor 2: `c`, `dd`, `W_k`, `W_r` are halved and the norm's masks carry
  `gain = 0.5` (`weights.encode_block`, `block.norm`); a power of two, so exact.

## Lint

`python lint.py` runs ruff (the repository's rules plus the stricter families in `ruff.toml`; it fixes
what it can and formats) and mypy (`mypy.ini`), and fails on anything left. `python lint.py --check`
changes nothing.

## License

This directory (`examples/fhe_ssm_from_primitives/`) is under the MIT license (`LICENSE` here;
each file carries `SPDX-License-Identifier: MIT`). It runs on Perseus, which is not: Perseus is
under the repository's Business Source License 1.1 (`../../LICENSE`; free for non-commercial use,
GPLv2 from 2030-08-18), and its terms apply to running this example on it.
