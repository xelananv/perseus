# Runs: from a fresh clone to the 148-tick chain

The five commands of the README's GPU section, with the tick counts of the three runs that extend
what was measured on one RTX A6000 (48 GB). Everything they need is public: the checkpoint comes
from Hugging Face (hash-pinned), the token record from the fhe-ssm repository. The durations and
plan sizes below were measured before this branch existed, on that card with FIDESlib at the tree of
374f6df (built with the README's arch-guarded patch): the steps with a fresh clone of the
`fhe-ssm-port` branch at 5d82074, the eight-tick pair with one build of it at 471a93e. This branch
tracks upstream dev, with its FIDESlib pin and planner recipe, and is measured anew.

## Setup

```bash
git clone --recurse-submodules -b fhe-ssm-port-dev https://github.com/xelananv/perseus && cd perseus
#   or from a bundle of the branch (git bundle create perseus-fhe-ssm-port-dev.bundle fhe-ssm-port-dev):
#   git clone -b fhe-ssm-port-dev perseus-fhe-ssm-port-dev.bundle perseus && cd perseus && git submodule update --init
export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"   # CUDA 13.0 or newer (12.x cannot compile this FIDESlib)
uv sync && NATIVE_SIZE=32 bash scripts/install_deps.sh && CHAIN=n32 bash scripts/local_build_core.sh
source scripts/local_env.sh                       # the n32 chain's defaults
python -c "from examples.fhe_ssm_from_primitives import colab; print(colab.fetch_model('.'))"
curl -L -o tokens.json https://raw.githubusercontent.com/xelananv/fhe-ssm/main/results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/tokens.json
M="--weights pbd430a_weights.npz --seeds pbd430a_seeds.json --config pbd430a_config.json"
```

`tokens.json` is the published 148-tick run's record: 64 conversations, 149 tokens each (the prompt,
then the generation). `gpu_run --tokens` teacher-forces it; without `--tokens` every conversation
gets random tokens, which is fine for timing and useless for comparing errors with that run.

## The sanity gate, the capture and the plan (once)

```bash
python -m examples.fhe_ssm_from_primitives.gpu_smoke $M --blocks 1 --out smoke.json
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode capture --graph-dir graphs/fhe_ssm \
    --tokens tokens.json --ticks 2 --out capture.json
PLAN_HARD_ENV_CAP=1 python -m examples.fhe_ssm_from_primitives.plan --graph-dir graphs/fhe_ssm --name fhe_ssm
```

On the A6000, with the fresh clone of 5d82074: the smoke took 88 s, a 58 s session open included (an
encryption round trip, one linear tile, two block ticks); the capture about 15 minutes (tick 0 at
236 s, then the recorded tick at 621 s with the device magnitude probe, which is on by default); the
plan about 6 minutes on the CPU. The plan at the recipe's ceiling (MAX_LEVEL 50) has 953 refreshes
per tick on that capture (959 on an earlier capture of the same model), with 36 planner warnings
about values predicted at level 50 and no refresh; the planned runtime accepts a predicted ceiling
level and every planned tick ran with top-1 agreement 1.0. This branch's planner plans the same
capture at 953 refreshes per tick.

## Run 1: 148 ticks under the plan

```bash
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode planned --plan-dir bootstrap_placements/fhe_ssm \
    --tokens tokens.json --ticks 148 --out planned148.json
```

A planned tick took 141 s on the A6000 after a first one of 149 s (the fresh clone), about 5.8 h for
the run; the report is rewritten after every tick, so a stopped run keeps what it did. What it
answers: the published 148-tick run (eager OpenFHE, two cards, ring 2^17) had its logit error grow
like tick^0.24; the planned ticks of the eight-tick runs were 4 to 8 times more accurate per tick, and the open question is whether the growth exponent moves or only
the constant. Compare `relErrRms_vs_mirror`, `logits_relErrRms_vs_exact` and `top1_agree_vs_exact`
per tick with the fhe-ssm repository's `FIDELITY_TREND.md`. Teacher-forced, the run replays the
published run's tokens; with `--free-run` it picks its own next tokens after each prompt under that
run's decode policy, as that run did. The fidelity to compare is then `pick_agree_vs_exact` (the same
policy on the closed form's logits, same history); `pick_agree_vs_record` counts agreement with the
published tokens only while a conversation's history still matches them.

## Run 2: the same-binary pair

```bash
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode eager   --tokens tokens.json --ticks 8 --out eager8.json
python -m examples.fhe_ssm_from_primitives.gpu_run $M --mode planned --plan-dir bootstrap_placements/fhe_ssm \
    --tokens tokens.json --ticks 8 --out planned8.json
```

Eager 229 s and planned 142 s per steady tick on the A6000, from one build (471a93e); a timing is a
property of the build and the card as much as of the configuration, so the pair is only meaningful
from one build on one card. Beside each run, `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu
--format=csv -l 5 > nvsmi.csv &` gives the memory column: the device peak was 35,118 MiB with a flat
pool (34,094 MiB over the fresh clone's two planned ticks), so a 48 GB card has room; smaller cards
are untested.

## Run 3: the ILP placer on the same capture

```bash
PLAN_HARD_ENV_CAP=1 python -m examples.fhe_ssm_from_primitives.plan --graph-dir graphs/fhe_ssm \
    --name fhe_ssm_ilp --set PLAN_PLACER=ilp
```

CPU only, needs scipy. The min-cut placer gave 1,297 refreshes per tick at ceiling 48 and 953 to 959
at 50 on captures of this model with the fork's planner, and this branch's planner gives 953 at 50
on the fresh clone's capture (48 not run); the ILP was not run on them. A 24-block graph whose residual stream crosses
blocks without a refresh is new ground for the placer comparison, and whether the ILP needs the hard
envelope at all is part of the answer. A planned run with `--plan-dir bootstrap_placements/fhe_ssm_ilp`
then measures what the plan predicts.

## What the reports hold

Every `gpu_run` report lists, per tick, `seconds`, `relErrRms_vs_mirror` (the reply against the same
kernels on numpy), `relErrRms_vs_exact` and `logits_relErrRms_vs_exact` (against the trainer's
closed-form forward, with the worst conversation), `top1_agree_vs_exact`, the refreshes (`bootstraps`
deliberate, `runtime_bootstraps` in total), `out_level` and `free_gb`; the header records Perseus's
commit and the sha256 of the weights, seeds and config. The README's table explains each field.
