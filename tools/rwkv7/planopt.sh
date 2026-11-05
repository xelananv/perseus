# shellcheck shell=bash disable=SC2034
# planopt.sh — the PLANOPT plan variants (python/perseus/examples/rwkv7/PLAN_B2.md), sourced by run_box.sh and
# tools/rwkv7/b2_plan.sh. A variant changes where the planner may put refreshes, never the circuit: the capture, the
# fake and the recorder are the same for every variant.
#   planopt_env NAME   sets the array PLANOPT_ENV (KEY=VALUE tokens added to make_plan.sh's environment)
# base | ''  the planner's own choice
# b2         no planned refresh inside one conversation's own state work in a batched pass (the step scopes
#            tm.state.bcast, tm.state.update, tm.state.readout), so the cut must cover that work from shared values
#            (the three token-side sources, the GroupNorm after the readout); the planner's default forbidden steps
#            (.var, .mean, .qkv: perseus/plan/placer/planner.py PlanConfig.forbid_steps) are kept. Decode, prefill
#            and join captures have none of these scopes: the variant leaves their plans as they are
# st         (experiment, PLAN_B2.md §4) no planned refresh anywhere in tm.state (decode's and prefill's whole
#            state step; in a batched pass also the shared sources)
PLANOPT_DEFAULT_FORBID=".var,.mean,.qkv"
planopt_env() {
    PLANOPT_ENV=()
    case "${1:-base}" in
        ""|base) ;;
        b2) PLANOPT_ENV=("FORBID_STEPS=$PLANOPT_DEFAULT_FORBID,tm.state.bcast,tm.state.update,tm.state.readout") ;;
        st) PLANOPT_ENV=("FORBID_STEPS=$PLANOPT_DEFAULT_FORBID,tm.state") ;;
        *) echo "PLANOPT=$1: base | b2 | st" >&2; return 2 ;;
    esac
}
