//! Streamed-weight and generated-diagonal sums. A streamed linear's baby-step inner sum is recorded as the loop it
//! stands for (`mult` of each term's plaintext, `add_inplace` into the first: the stream perseus `mult_coeff_sum`
//! records); the plaintexts are float32 coefficient rows the runtime expands inside the sum (`PtForm::CoeffStream`),
//! or diagonals it gathers from a resident matrix and encodes there (`PtForm::DiagGen`, `compile::diag`); a sum
//! merges terms of one form only. This pass,
//! after the lowering (plan decisions applied), realizes once every term the products read at degree 2 (their
//! Rescale prelude, hoisted), turns every product with a streamed plaintext into a `Kind::CoeffSum` of one term and
//! merges `Add(sum_a, sum_b)` into one sum wherever both partial sums are read only by that add, every term sits at
//! one level (degree 1) and no prelude is needed: one fused kernel loop.
//! A plan refresh or realize on an intermediate name reads it a second time, so the sum stops there (the C++ fused
//! path refused that case too: `mult_pt_sum_usable`). The merged-away vars stay in `fused` with their lowered
//! metadata for the check (the plan pins every name).
use super::apply_plan::Lowered;
use super::diag::is_generated;
use crate::program::{Kind, Op, Var, VarId};
use pc_shadow::{Operand, Prelude, Shadow};
use std::collections::HashMap;

/// Fuse in place (`keep`: vars that must stay materialized, e.g. live-out names); returns the ops folded away.
pub fn fuse(lw: &mut Lowered, sh: &Shadow, keep: &[VarId]) -> usize {
    let streamed = |k: &Kind| matches!(k, Kind::MultPt(p) if lw.pt_refs.get(*p as usize).is_some_and(|r| is_generated(r.form)));
    let ops: Vec<Option<Op>> = std::mem::take(&mut lw.ops).into_iter().map(Some).collect();
    // a term read at degree 2 (the loop's product realizes it first: a Rescale prelude on the copy): realize the
    // input once, before its first streamed product, and let the products read the realized var (same words: the
    // rescale is deterministic; the plan never names the copy)
    let mut realized: HashMap<VarId, VarId> = HashMap::new();
    let mut with_rs: Vec<Option<Op>> = Vec::with_capacity(ops.len());
    for op in ops.into_iter().flatten() {
        if streamed(&op.kind) && op.pre == [Prelude::Rescale(Operand::A)] && op.ins.len() == 1 {
            let x = op.ins[0];
            let xr = match realized.get(&x) {
                Some(&v) => v,
                None => {
                    let m = lw.vars[x as usize].meta;
                    let Ok(r) = sh.realize(&m) else {
                        with_rs.push(Some(op));
                        continue;
                    };
                    lw.vars.push(Var { name: format!("{}~rs", lw.vars[x as usize].name), meta: r.out, slot: None });
                    let v = (lw.vars.len() - 1) as VarId;
                    realized.insert(x, v);
                    with_rs.push(Some(Op { kind: Kind::Rescale, ins: vec![x], outs: vec![v], pre: r.pre, origin: op.origin.clone(), inplace: false }));
                    lw.acct.rescales += 1;
                    v
                }
            };
            with_rs.push(Some(Op { ins: vec![xr], pre: vec![], ..op }));
        } else {
            with_rs.push(Some(op));
        }
    }
    let mut ops = with_rs;
    for op in ops.iter_mut().flatten() {
        if streamed(&op.kind) && op.pre.is_empty() {
            if let Kind::MultPt(p) = op.kind {
                op.kind = Kind::CoeffSum { pts: vec![p], fused: vec![] };
            }
        }
    }
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    // a var the driver keeps (live out) is read once more: never folded away
    for &v in keep {
        *uses.entry(v).or_default() += 1;
    }
    for op in ops.iter().flatten() {
        for v in op.all_ins() {
            *uses.entry(v).or_default() += 1;
        }
    }
    // var -> index of the CoeffSum op producing it
    let mut sum_of: HashMap<VarId, usize> = HashMap::new();
    let mut folded = 0;
    for i in 0..ops.len() {
        let Some(op) = ops[i].as_ref() else { continue };
        if let Kind::CoeffSum { .. } = op.kind {
            sum_of.insert(op.outs[0], i);
            continue;
        }
        if !(matches!(op.kind, Kind::Add) && op.pre.is_empty() && op.ins.len() == 2 && op.ins[0] != op.ins[1]) {
            continue;
        }
        let (a, b) = (op.ins[0], op.ins[1]);
        let (Some(&ia), Some(&ib)) = (sum_of.get(&a), sum_of.get(&b)) else { continue };
        if uses.get(&a) != Some(&1) || uses.get(&b) != Some(&1) {
            continue;
        }
        let (Some(oa), Some(ob)) = (ops[ia].as_ref(), ops[ib].as_ref()) else { continue };
        let lv = |v: VarId| (lw.vars[v as usize].meta.level, lw.vars[v as usize].meta.deg);
        let l0 = lv(oa.ins[0]);
        if l0.1 != 1 || !oa.ins.iter().chain(&ob.ins).all(|&v| lv(v) == l0) {
            continue;
        }
        let (Kind::CoeffSum { pts: pa, fused: fa }, Kind::CoeffSum { pts: pb, fused: fb }) = (&oa.kind, &ob.kind) else { continue };
        // one source per sum: a streamed and a generated-diagonal partial stay apart (every sum starts as one term)
        let form = |ids: &Vec<u32>| ids.first().and_then(|&i| lw.pt_refs.get(i as usize)).map(|r| r.form);
        if form(pa).is_none() || form(pa) != form(pb) {
            continue;
        }
        let mut pts = pa.clone();
        pts.extend(pb);
        let mut fused = fa.clone();
        fused.push(a);
        fused.extend(fb);
        fused.push(b);
        let mut ins = oa.ins.clone();
        ins.extend(&ob.ins);
        let out = op.outs[0];
        let origin = op.origin.clone();
        ops[ia] = None;
        ops[ib] = None;
        sum_of.remove(&a);
        sum_of.remove(&b);
        ops[i] = Some(Op { kind: Kind::CoeffSum { pts, fused }, ins, outs: vec![out], pre: vec![], origin, inplace: false });
        sum_of.insert(out, i);
        folded += 2;
    }
    lw.ops = ops.into_iter().flatten().collect();
    lw.acct.fused += folded;
    lw.acct.ops = lw.ops.len();
    folded
}
