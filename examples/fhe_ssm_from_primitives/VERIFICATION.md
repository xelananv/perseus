# How each piece is verified

Every piece of this example is checked two independent ways, on edge cases. The tests are in
`tests/`; `python -m pytest -q examples/fhe_ssm_from_primitives/tests` runs all of them (CPU, a few
seconds), and `python lint.py --check` the static checks (ruff with `ruff.toml`, mypy with
`mypy.ini`).

| code | edge cases | check 1 | check 2 | test |
|---|---|---|---|---|
| `pairs.split`, `pairs.repack` | complex, real-only, imaginary-only and zero payloads; an operand with its rescale pending | `2 Re x`, `2 Im x` in numpy, exactly | the round trip `repack(split(x)) = 2x`; no level spent, one conjugation, no product | `test_filled_model.py::test_split_gives_twice_each_axis_without_a_level` |
| `pairs.PairLaneOps.add_pt` | random pair and plaintext | `x + v (1 + i)` | each axis on its own (`Re + v`, `Im + v`); the stock `LanePackedOps` reaches the real axis only | `test_lane_route_adds_a_plaintext_to_both_axes` |
| `geometry.park`, row checks | 1, 2, t - 1, t rows; empty, too many, 1-D, wrong width | the copies, explicitly | the bootstrap-range test below | `test_park_fills_with_conversation_zero`, `test_rows_outside_the_layout_are_refused` |
| `encode_rows`, `decode_rows` | real 1 and t; pairs 1, t, t + 1, 2t | the round trip | the slot layout written independently, as the transpose of the rows | `test_rows_round_trip_and_sit_where_the_layout_says` |
| `filling` linears (`linear_multi`, `linear_tiled`, the sharing) | real and pairs; 2 weights x 4 output tiles; 4 input tiles | `X @ W` | the textbook diagonal method (no baby/giant split); rotation and product counts against their formulas | `test_up_and_down_match_numpy_and_the_naive_diagonal_method` |
| the norm as a section | the three arms; conversations at both ends of the seeds' window | the same kernels on numpy | the closed-form RMSNorm | `test_norm_section_matches_mirror_and_closed_form` |
| the levels claim | the three arms with no ceiling | a split section spends the real kernel's depth | the lane route spends more | `test_split_spends_the_real_depth_and_lane_spends_more` |
| the gate sections | the three arms; the time-mix gate and the four channel-mix tiles | the same kernels on numpy | the polynomials in closed form | `test_gate_sections_match_mirror_and_closed_form` |
| the model | the three arms, three ticks, the carry | the mirror (`ref.MirrorModel`) | the trainer's forward (`ref.ExactModel`) | `test_ticks_match_mirror_and_exact` |
| the three arms | 64 pairs split, 64 pairs lane-wise, two ciphertexts of 32 real | — | three circuits, one answer | `test_routes_agree_on_the_same_conversations` |
| partial batches | 1, t + 1, 2t - 1 conversations | the mirror on just those conversations | the trainer's forward on them | `test_parked_lanes_leave_the_conversations_alone` |
| parking's purpose | 1 and t + 1 conversations | the largest bootstrap input stays within 1.5x of a full batch's | with zero lanes instead it exceeds 4x (measured: 8x) | `test_parking_keeps_every_bootstrap_input_in_range` |
| the full shape | one block at ring 2^16, d = 1,024, d_ffn = 4,096, 64 pairs | the mirror and the trainer's forward | the rotation count against its formula | `test_full_shape_block_on_pairs` |
| `run_fake` | the four arms (with the decode layout) | the errors it reports | its counts against independent tallies (by-step sums, parts, file hashes) | `test_run_fake.py` |
| `colab.fetch_model` | a good file, a flipped byte, other seeds | the recorded hash | the copy's own hash | `test_colab_and_smoke.py` |
| `colab.fake_check` | pairs and real | its verdict | its thresholds against the report | `test_fake_check_passes_on_a_consistent_checkpoint` |
| `gpu_smoke.run_checks` | the three arms; a band without the layout's rotations; a failing step | every check passes on the fake | it stops on a missing rotation and records a failure without hiding the other checks | `test_colab_and_smoke.py` |
| `colab_build.plan` | a fresh runtime, a ready one, CUDA installed earlier, no Python headers; no GPU, no root, unknown distribution | the expected step lists | invariants every plan keeps | `test_colab_build.py` |
| `scales.exponent` | 0, negative, NaN, infinity, tiny (capped at 14), huge (never negative), the exact boundary, 200 random values | the rule's closed form | a brute-force search over 0..14 | `test_scales.py` |
| `scales.choose` | values already past the target, tiny products whose lift exceeds one exponent's cap, nothing observed | the rule (each family's limit, the product's limit on a + b) | a brute-force search over all (a, b), 300 random cases | `test_choose_matches_brute_force_on_the_channel_mix` |
| the folds | exponents 0, mid, 14 | the folded kernels equal the unfolded ones times 2^e, bit for bit | the whole model with folds against the mirror with the same folds and against the unfolded closed form, all three arms | `test_folded_kernels_are_the_unfolded_ones_times_two_to_the_e`, `test_folded_model_matches_mirror_and_unfolded_exact` |
| `scales.measure` | a window after tick 0; empty, negative and too-long windows | a hand-rolled loop over the closed form | every observed family present | `test_measure_window_matches_a_hand_rolled_loop` |
| the shipped scales | — | recomputed from the file's own recorded maxima | held-out margin at least 1; no local paths | `test_shipped_scales_are_sane_and_public` |
| the decode layout | one sequence | the mirror | the trainer's forward | `test_fhe_ssm_fake.py` |

## Mutation checks

The tests were run against deliberately broken code, one break at a time:

| break | caught by |
|---|---|
| the split's halves swapped | the norm, gate, tick and three-arm tests |
| `repack` with the wrong sign | the round-trip, gate, lane-tick and three-arm tests; the split model's ticks still match, because the sign flips in pairs: every value in the flipped form passes an odd function (the norm) before anything else, so the circuit stays correct |
| the split gain forgotten | the norm, gate, tick, three-arm and partial-batch tests |
| zero lanes instead of parking | the bootstrap-range test only (the fake has no range limit, so the replies stay right) |
| the lane route's plaintext add on one axis | the lane-tick and three-arm tests |

## What the fake cannot check

Noise, timing and memory: the fake computes in float64 and counts levels, products, rotations and
bootstraps; it never encrypts. `gpu_smoke.py` runs the same checks on a live session, and only
`gpu_smoke.open_session` and `colab_build`'s execution are untested here (they need a GPU).
