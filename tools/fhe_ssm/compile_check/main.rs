//! compile_check — compile a plan directory against its graph directory with upstream's own compiler (pc-program,
//! CUDA-free), on any machine, and print what a planned GPU session of this branch would get from it.
//!
//!     compile_check <plan_dir> <graph_dir> [--schedule program|siblings|lockstep] [--no-batch] [--json <out.json>]
//!
//! It runs the same call the planned driver makes through `_pc.Program.compile` (crates/pc-py/src/program.rs:78-87:
//! `compile_dir(plan_dir, graph_dir, &CompileOpts { schedule, no_batch, .. })`), so the Config is the plan's own
//! (each block plan's stamp / capture_env overlaid by PLAN_CMD.txt's `runtime=` line, the PlanConfig from its
//! `recipe=` line: crates/pc-program/src/plan.rs:226-259). Not checked here: the contract check against a live
//! session's Config, and anything the device does (routes, keys, memory). Without `live_out` / `input_metas` (the
//! driver's K/V and carried-state hints) a block's carried inputs take the compiler's own seeds.
//!
//! Per block: refresh sites by kind (placed / fired hint / deliberate) and by route (dense StC-first, dense shipped
//! order, real payload, each sparse slot count), launches after K-batching, refresh launches, refresh pairs formed
//! (PC_BTS_PAIR on the plan's runtime= line), the level check's pins (exact / richer) and its errors.
//! Exit 0 when every block compiles with 0 check errors, 1 otherwise, 2 on usage.
use pc_program::compile::batch::Schedule;
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::SiteKind;
use std::collections::BTreeMap;
use std::path::Path;

fn usage() -> ! {
    eprintln!("usage: compile_check <plan_dir> <graph_dir> [--schedule program|siblings|lockstep] [--no-batch] [--json <out.json>]");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut pos, mut schedule, mut no_batch, mut json_out) = (vec![], Schedule::Program, false, None::<String>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--schedule" => {
                i += 1;
                schedule = match args.get(i).map(String::as_str) {
                    Some("program") => Schedule::Program,
                    Some("siblings") => Schedule::Siblings,
                    Some("lockstep") => Schedule::Lockstep,
                    _ => usage(),
                };
            }
            "--no-batch" => no_batch = true,
            "--json" => {
                i += 1;
                json_out = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            a if a.starts_with("--") => usage(),
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    if pos.len() != 2 {
        usage();
    }
    let (pd, gd) = (Path::new(&pos[0]), Path::new(&pos[1]));
    let t = std::time::Instant::now();
    let opts = CompileOpts { keep_going: true, no_batch, schedule, ..Default::default() };
    let dc = match compile_dir(pd, gd, &opts) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("compile_check: {e}");
            std::process::exit(1)
        }
    };
    println!("compile_check {} x {} (schedule {schedule:?}, no_batch {no_batch}): {} programs", pd.display(), gd.display(), dc.programs.len());
    println!("  block | sites placed/hint/delib | routes | launches bts_launches pairs k_max | pins exact richer | arena MB | errors");
    let mut tot: BTreeMap<&str, usize> = BTreeMap::new();
    let mut routes_tot: BTreeMap<String, usize> = BTreeMap::new();
    let mut rows = vec![];
    let mut failed = false;
    for (label, r) in &dc.programs {
        let p = match r {
            Ok(p) => p,
            Err(e) => {
                println!("  {label} | COMPILE ERROR: {e}");
                failed = true;
                continue;
            }
        };
        let (mut placed, mut hint, mut delib) = (0usize, 0usize, 0usize);
        let mut routes: BTreeMap<String, usize> = BTreeMap::new();
        for s in &p.bts_sites {
            match s.kind {
                SiteKind::Placement => placed += 1,
                SiteKind::Hint => hint += 1,
                SiteKind::Deliberate => delib += 1,
            }
            let rt = &s.route;
            let name = if rt.slots == 0 {
                format!("dense{}{}@{}", if rt.stc_first { "_stcf" } else { "" }, if rt.real { "_real" } else { "" }, s.landing.level)
            } else {
                format!("s{}{}@{}", rt.slots, if rt.spru_h > 0 { "_spru" } else if rt.sparse_b { "_sb" } else { "" }, s.landing.level)
            };
            *routes.entry(name.clone()).or_default() += 1;
            *routes_tot.entry(name).or_default() += 1;
        }
        let b = &p.batch_report;
        let c = &p.check;
        let rs: Vec<String> = routes.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        println!(
            "  {label} | {} {}/{}/{} | {} | {} {} {} {} | {} {} {} | {:.1} | {}",
            p.bts_sites.len(), placed, hint, delib, rs.join(" "), b.launches, b.bootstrap_launches, b.pairs, b.k_max,
            c.pinned, c.level_exact, c.level_richer, p.mem.bytes as f64 / 1e6, c.errors.len()
        );
        for e in c.errors.iter().take(4) {
            println!("      {e}");
        }
        failed |= !c.errors.is_empty();
        for (k, v) in [("sites", p.bts_sites.len()), ("placed", placed), ("hint", hint), ("delib", delib), ("launches", b.launches),
                       ("bts_launches", b.bootstrap_launches), ("pairs", b.pairs), ("errors", c.errors.len()),
                       ("richer", c.level_richer), ("finishes", p.accounting.finishes)] {
            *tot.entry(k).or_default() += v;
        }
        rows.push(serde_json::json!({
            "block": label, "sites": p.bts_sites.len(), "placed": placed, "hint": hint, "deliberate": delib,
            "routes": routes, "launches": b.launches, "bootstrap_launches": b.bootstrap_launches, "pairs": b.pairs,
            "k_max": b.k_max, "bootstrap_hist": b.bootstrap_hist, "pins": c.pinned, "level_exact": c.level_exact,
            "level_richer": c.level_richer, "arena_bytes": p.mem.bytes, "live_peak_bytes": p.mem.live_peak,
            "finishes": p.accounting.finishes, "errors": c.errors,
        }));
    }
    let rs: Vec<String> = routes_tot.iter().map(|(k, v)| format!("{k}:{v}")).collect();
    println!("TOTAL refresh sites {} (placed {}, fired hints {}, deliberate {}), PCMM finishes {}; launches {}, refresh launches {}, refresh pairs {}; level richer {}; check errors {}; {:.1} s",
        tot.get("sites").unwrap_or(&0), tot.get("placed").unwrap_or(&0), tot.get("hint").unwrap_or(&0), tot.get("delib").unwrap_or(&0),
        tot.get("finishes").unwrap_or(&0), tot.get("launches").unwrap_or(&0), tot.get("bts_launches").unwrap_or(&0), tot.get("pairs").unwrap_or(&0),
        tot.get("richer").unwrap_or(&0), tot.get("errors").unwrap_or(&0), t.elapsed().as_secs_f64());
    println!("ROUTES (route@landing level: sites) {}", rs.join(" "));
    if let Some(out) = json_out {
        let doc = serde_json::json!({"plan_dir": pos[0], "graph_dir": pos[1], "schedule": format!("{schedule:?}"),
                                     "no_batch": no_batch, "blocks": rows, "routes": routes_tot, "totals": tot});
        std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("write --json");
    }
    std::process::exit(if failed { 1 } else { 0 })
}
