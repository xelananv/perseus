# Upstream dev's planner on this model's capture, before and after the re-cut fix (CPU)

The capture: the 25 block graphs of the A6000 capture (`../a6000_capture/`), archive
`graphs_fhe_ssm_v7_head5d82074.tar.gz`, 46,851,607 bytes, sha256
`49a8f0d3e462d4457b9153a568ad189cc38753163107676c964ba3c1b701313a` (published with the issue, not in git). Every run:
`PLAN_HARD_ENV_CAP=1 CUDA_VISIBLE_DEVICES= python -m examples.fhe_ssm_from_primitives.plan --graph-dir <capture>/graphs/fhe_ssm --name <name>`
with the example's recipe (dev's `PLAN_REAL_ROUTE=1 PLAN_RAISE_DROP_SET=1,3,5 PLAN_SHARED_REALIZE=1` included unless
stated).

**Dev's planner, as it is** (`plan_dev_unfixed.log`, planner at the merge of upstream dev 225e49a): blocks 0–2 plan
(lines 2–30); block 3 is infeasible at entry 46/d2 and block 2 is re-planned with a forced
terminal-exit refresh (line 40); the retried block places 51 refreshes, one more (line 42), then
`shared_realize: … bootstraps 53 -> 52` (line 43) removes it again; block 3 re-enters at 46/d2 (line 51) and the planner
stops: `PlanInfeasible: every S->T path crosses only unrefreshable sites` (lines 88–89).

**Dev's planner, its three new recipe defaults off** (`plan_dev_nofeat.log`): the retry holds; 25 blocks,
953 bootstraps (line 387), 11 terminal-exit retries (`grep -c retrying`).

**With the fix** (`plan_dev2.log`; `perseus/plan/placer/planner.py` on `fhe-ssm-port-dev`: the three re-cut trial
placers — `rescale_opt`, `shared_realize`, `ks_realize` — also start from `placed=set(forced)`): the full recipe plans all
25 blocks, 953 bootstraps (line 423), 11 terminal-exit retries, no relaxed envelope. Unit test:
`tests/test_plan_placer.py::RetryKeepsForcedSiteTest`.

**The shipped plans** (`PERSEUS_ALL_PLANS=1 pytest -q tests/test_paper_plans.py -rs`, same Python, each side in its own
worktree): with the fix 22 failed, 4 passed, 7 skipped (`battery_ours_661cc8a.log`, last line); dev's own planner 22
failed, 4 passed, 7 skipped (`battery_dev_225e49a.log`, last line), the same 22 with the same first difference; one of them
run alone on dev: `battery_devside_n32_dense.log`. The four that regenerate on both sides include the main plan
(`gpt2_decode_python_n32`). Regenerations of every plan by the two planners compared with
`scripts/utils/plan_equiv.py` (`pair_regen.py`, `pair_compare_fix_vs_dev.txt`): no plan's runtime content differs
between them (`runtime content differs` occurs nowhere in the file); where a pair is not equal, the first difference is
the summary diagnostic `summary.bts_quality.env_cap.placed_past_cap`, which the runtime does not read.
