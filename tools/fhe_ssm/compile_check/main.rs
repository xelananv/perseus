//! compile_check — compile a plan directory against its graph directory with pc-program's CUDA-free compiler and
//! print what a planned session would get from it.
//!
//!     compile_check <plan_dir> <graph_dir> [--schedule program|siblings|lockstep] [--no-batch] [--json <out.json>]
//!
//! The call is the one `_pc.Program.compile` makes (crates/pc-py/src/program.rs:78-87), so the Config is the plan's
//! own (stamp + PLAN_CMD.txt `runtime=`, PlanConfig from `recipe=`: crates/pc-program/src/plan.rs:226-259); without
//! the driver's `live_out` / `input_metas`, carried inputs take the compiler's seeds. Per block: refresh sites by kind
//! and route, launches, refresh launches, pairs, the level check's pins and errors. Exit 0 when every block compiles
//! with 0 check errors, 1 otherwise, 2 on usage. Not checked: the contract against a live session, anything on device.
use pc_config::Config;
use pc_program::compile::batch::Schedule;
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::plan::PlanDir;
use pc_program::program::SiteKind;
use std::collections::BTreeMap;
use std::path::Path;

/// The chain preset a plan names: the last `CHAIN_PRESET=` token of its `runtime=` line, else a block stamp's
/// `chain.preset` (the plan's own Config, as `PlanDir::config` builds it), else None (n32, the runtime's default).
fn plan_chain(pd: &Path) -> Option<String> {
    let dir = match PlanDir::open(pd) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("compile_check: {e}");
            std::process::exit(1)
        }
    };
    if let Some((_, v)) = dir.cmd.runtime.iter().rev().find(|(k, _)| k.as_str() == "CHAIN_PRESET") {
        return Some(v.clone());
    }
    for (_, _, path) in &dir.blocks {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
        if let Some(stamp) = v.get("config") {
            if let Ok(cfg) = serde_json::from_value::<Config>(stamp.clone()) {
                if let Some(p) = cfg.chain.preset.clone() {
                    return Some(p);
                }
            }
        }
    }
    None
}

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
    // the chain: this model's plaintext rows and levels are the n32 chain's (54 primes); a plan whose stamp or
    // runtime= line names another preset would compile on that chain (51 limbs on n32a4) and report counts that
    // bind nothing here, so it is refused before anything is compiled
    let chain = plan_chain(pd);
    if let Some(c) = chain.as_deref().filter(|c| *c != "n32") {
        eprintln!("compile_check: the plan names chain preset {c:?}; this model runs on n32 only");
        std::process::exit(2)
    }
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
    println!("  chain preset: {}", chain.as_deref().unwrap_or("n32 (the default)"));
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
