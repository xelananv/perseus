# fhe-ssm on `perseus.impl`

`examples/fhe_ssm_from_primitives/` is the fhe-ssm recurrent language model (tag `pbd430a`: 24
blocks, d = 1,024, d_ffn = 4,096, learned polynomial gates, Newton RMSNorm, two carried values per
block) written on Perseus's leaf-primitive ops protocol, in the shape of
`examples/gpt2_from_primitives`. The model, its training and its published FIDESlib run live in the
fhe-ssm repository (github.com/xelananv/fhe-ssm); paths such as `spec_decode/plain_recurrent.py`
below are relative to it. The checkpoint, the Newton seeds and the config are on Hugging Face
(huggingface.co/xelananv/fhe-ssm-pbd430a: `pbd430a_weights.npz`, `pbd430a_seeds.json`,
`pbd430a_config.json`).

It runs on Perseus's numpy fake session (CPU, no GPU, the runtime's level accounting); the same
code is meant to run on a live session through `ImplModel`.

## Run on the fake

From the repository root, in an environment with this checkout's `perseus` importable:

```bash
python -m pytest -q examples/fhe_ssm_from_primitives/tests
python -m examples.fhe_ssm_from_primitives.run_fake \
    --weights pbd430a_weights.npz --seeds pbd430a_seeds.json --config pbd430a_config.json \
    --tokens 4 --blocks 24 --form perseus
```

Per tick the driver prints the reply's relative RMS error against two references and the
operation counts:

| field | meaning |
|---|---|
| `relErrRms_vs_mirror` | the same kernels on plain numpy (`ref.MirrorModel`): a non-zero value is a layout or protocol error |
| `relErrRms_vs_exact` | the trainer's forward with the closed-form rsqrt (`ref.ExactModel`): the Newton approximation |
| `bootstraps` = `reactive` + `hints` + `carry` | refreshes fired at the ceiling, at the port's hint sites, and the canonical carry (2 per block) |
| `bootstraps_by_step` | the same refreshes by step scope (`tm.norm`, `cm.v`, `carry`, ...) |
| `mult_cc`, `mult_pt`, `rotations` | ciphertext products (incl. squares), plaintext products, rotations |
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

## Layout

The model runs one token per ciphertext in the cachemir decode packing (channel `i` at slot `i*t`,
`t = 32` at ring 2^16), so one sequence per ciphertext. `filling.py` has the filled layout (32
conversations per ciphertext, 64 as real/imaginary pairs): its linear and the norm's
per-conversation statistic are tested exactly.

## What is folded (all exact)

- The norm gains `g1`, `g2` into the rows of `W_in`, `W_k`, `W_r`; `alpha_res` into `W_out`;
  `g_out` stays a per-channel plaintext on the reply (`weights.py`).
- The norm's frame constant rides two lane masks, both one product from the input: `sqrt(sigma/d)`
  for the statistic (its all-slot sum of squares is `sigma mean(x^2)` directly) and `sqrt(sigma)`
  for the output, so the normalized value comes out at unit rms (`norm.py`).
- The first Newton step from the constant seed is affine in the statistic (no ciphertext product).

## Lint

`./lint.sh` runs ruff (the repository's rules plus the stricter families in `ruff.toml`; it fixes
what it can and formats) and mypy (`mypy.ini`), and fails on anything left. `./lint.sh --check`
changes nothing.
