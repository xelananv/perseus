//! Generated diagonals through the whole compiler: a captured group's `mult` + `add_inplace` loop over `diag`
//! plaintexts fuses into one `CoeffSum` of `DiagGen` terms, a sum never mixes a streamed row with a diagonal, a diag
//! plaintext whose name is not `<linear>#<n>.<m>.<gg>.<jj>` is refused, and the same loop over `values` plaintexts
//! stays the unfused at-use products it is today.
use pc_config::{Config, PlanConfig};
use pc_program::compile::diag::{census, DiagSource, SumKind};
use pc_program::compile::{compile_block, CompileOpts};
use pc_program::graph::{BlockGraph, GraphFile};
use pc_program::plan::{BlockPlan, PlacementFile};
use pc_program::program::PtForm;
use pc_program::{Kind, Program};
use pc_shadow::graph::LoadOpts;
use serde_json::json;
use std::collections::BTreeMap;

fn cfg() -> Config {
    let env = [("CHAIN", "n32"), ("LOGN", "16"), ("AUTO_BTS_LEVEL", "44"), ("BTS_MAX_INPUT_LEVEL", "48")];
    Config::from_legacy_env(env.iter().map(|(k, v)| (*k, Some(*v)))).unwrap().0
}

/// One term: `out = in * pt` with the plaintext's v2 `args.pt` (`kind`, `name`).
fn term(input: &str, pt: &str, out: &str, kind: &str, name: Option<&str>) -> serde_json::Value {
    let mut a = json!({"kind": kind, "level": 30, "deg": 1});
    if let Some(n) = name {
        a["name"] = json!(n);
    }
    json!({"op_type": "mult", "inputs": [input, pt], "input_levels": [30, 30], "output": out, "step": "blk0.cm.wk",
           "output_level": 32, "output_noise_level": 2, "args": {"pt": a}})
}

fn add(a: &str, b: &str, out: &str) -> serde_json::Value {
    json!({"op_type": "add_inplace", "inputs": [a, b], "input_levels": [32, 32], "output": out, "step": "blk0.cm.wk",
           "output_level": 32, "output_noise_level": 2})
}

/// The loop `ops.mult_pt_sum` records for three terms (n outer, jj inner): mult, mult, add, mult, add.
fn group(kinds: [(&str, Option<&str>); 3]) -> Vec<serde_json::Value> {
    vec![
        term("anon_1", "pt_1", "v_1", kinds[0].0, kinds[0].1),
        term("anon_2", "pt_2", "v_2", kinds[1].0, kinds[1].1),
        add("v_1", "v_2", "v_3"),
        term("anon_3", "pt_3", "v_4", kinds[2].0, kinds[2].1),
        add("v_3", "v_4", "v_5"),
    ]
}

fn plan() -> serde_json::Value {
    let names = ["v_1", "v_2", "v_3", "v_4", "v_5"];
    let lv: BTreeMap<&str, u32> = names.iter().map(|n| (*n, 32)).collect();
    let dg: BTreeMap<&str, u32> = names.iter().map(|n| (*n, 2)).collect();
    json!({
        // inputs seed at their first consumer's level (30) with degree 1 when that is not past bootstrap_level: the
        // realized baby rotations a diagonal linear's products read
        "version": 2, "rules": {"bootstrap_level": 30, "max_level": 50, "source_level": 36, "cache_read_level": 36},
        "placements": [], "hint_fire": [], "rescale_after": [], "sparse_slots": {}, "correction_factor": {}, "real_route": [],
        "summary": {"final_named_levels": lv, "final_named_degs": dg, "exit_var": "v_5", "exit_level": 32, "exit_deg": 2}
    })
}

fn compile(nodes: Vec<serde_json::Value>) -> pc_program::Result<Program> {
    let gf: GraphFile = serde_json::from_value(json!({"version": 2, "nodes": nodes})).unwrap();
    let bg = BlockGraph::from_file(gf, "t", &LoadOpts { level_unit: 2, delib_landing_remap: BTreeMap::new() })?;
    let pf: PlacementFile = serde_json::from_value(plan()).unwrap();
    let bp = BlockPlan::from_file(pf, "t.json").unwrap();
    let opts = CompileOpts { keep_going: true, ..Default::default() };
    compile_block("t", &bg, &bp, &cfg(), &PlanConfig::default(), &BTreeMap::new(), None, &opts)
}

fn sums(p: &Program) -> Vec<(usize, PtForm, usize)> {
    p.ops
        .iter()
        .filter_map(|o| match &o.kind {
            Kind::CoeffSum { pts, fused } => Some((pts.len(), p.pt_refs[pts[0] as usize].form, fused.len())),
            _ => None,
        })
        .collect()
}

#[test]
fn a_diag_group_fuses_into_one_sum() {
    let d = |j: u32| format!("b0.wk#0.1.2.{j}");
    let (n0, n1, n2) = (d(0), d(1), d(2));
    let p = compile(group([("diag", Some(&n0)), ("diag", Some(&n1)), ("diag", Some(&n2))])).unwrap();
    assert!(p.check.errors.is_empty(), "{:?}", p.check.errors);
    let kinds: Vec<&str> = p.ops.iter().map(|o| o.kind.name()).collect();
    assert_eq!(kinds, ["coeff_sum"]);
    // three terms, the four intermediates folded (two products, the partial sum, the third product)
    assert_eq!(sums(&p), vec![(3, PtForm::DiagGen, 4)]);
    assert_eq!(p.pt_refs.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), [n0.as_str(), n1.as_str(), n2.as_str()]);
    assert!(p.pt_refs.iter().all(|r| (r.level, r.deg) == (30, 1)));
    assert_eq!(p.accounting.fused, 4);
    assert!(p.accounting.balanced());
    let c = census(&p);
    assert_eq!((c.diag_sums, c.stream_sums, c.fused_vars, c.store_refs, c.form_errors), (1, 0, 4, 0, 0));
    assert_eq!(c.diag_terms, BTreeMap::from([(3, 1)]));
    // the bind side: every term's job on a declared tile (B = 2^15 / 32 = 1024, b = g = 32)
    let src = DiagSource::new(1 << 15, &[("b0.wk".into(), 1024, 4096, 32)]).unwrap();
    let Kind::CoeffSum { pts, .. } = &p.ops[0].kind else { unreachable!() };
    assert_eq!(pc_program::compile::diag::sum_kind(&p, pts), Ok(SumKind::Diag));
    let jobs: Vec<_> = pts.iter().map(|&i| src.job(&p.pt_refs[i as usize].name).unwrap()).collect();
    assert_eq!(jobs.iter().map(|j| (j.tile, j.n, j.m, j.gg, j.jj)).collect::<Vec<_>>(), [(0, 0, 1, 2, 0), (0, 0, 1, 2, 1), (0, 0, 1, 2, 2)]);
    assert_eq!(pc_program::compile::diag::diag_layout(&p, &src).unwrap().tiles, vec![0]);
}

#[test]
fn a_streamed_row_and_a_diagonal_never_share_a_sum() {
    let p = compile(group([("diag", Some("b0.wk#0.0.0.0")), ("stream", Some("b0.up#7")), ("diag", Some("b0.wk#0.0.0.1"))])).unwrap();
    assert!(p.check.errors.is_empty(), "{:?}", p.check.errors);
    // every product is a one-term sum of its own form; the adds stay
    let kinds: Vec<&str> = p.ops.iter().map(|o| o.kind.name()).collect();
    assert_eq!(kinds, ["coeff_sum", "coeff_sum", "add", "coeff_sum", "add"]);
    assert_eq!(sums(&p), vec![(1, PtForm::DiagGen, 0), (1, PtForm::CoeffStream, 0), (1, PtForm::DiagGen, 0)]);
    let c = census(&p);
    assert_eq!((c.diag_sums, c.stream_sums, c.form_errors), (2, 1, 0));
}

#[test]
fn a_diag_plaintext_needs_a_diagonal_name() {
    for bad in [None, Some("b0.wk#0.1.2"), Some("b0.wk#7"), Some("b0.wk")] {
        let e = compile(group([("diag", bad), ("diag", Some("b0.wk#0.0.0.1")), ("diag", Some("b0.wk#0.0.0.2"))])).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("a diag plaintext without args.pt.name = <linear>#<n>.<m>.<gg>.<jj>"), "{bad:?}: {msg}");
    }
}

#[test]
fn the_values_loop_stays_at_use_products() {
    // the port's values capture: the same loop over at-use plaintexts (`args.pt.kind = values`, no name)
    let p = compile(group([("values", None), ("values", None), ("values", None)])).unwrap();
    assert!(p.check.errors.is_empty(), "{:?}", p.check.errors);
    let kinds: Vec<&str> = p.ops.iter().map(|o| o.kind.name()).collect();
    assert_eq!(kinds, ["mult_pt", "mult_pt", "add", "mult_pt", "add"]);
    let c = census(&p);
    assert_eq!((c.diag_sums, c.stream_sums, c.store_refs, c.store_named_generated), (0, 0, 3, 0));
    assert_eq!(c.pt_forms, BTreeMap::from([("at_use", 3)]));
}
