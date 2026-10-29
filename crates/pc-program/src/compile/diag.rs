//! Generated plaintexts, the CUDA-free half: what the compiler, the bind and the runtime agree on about the
//! plaintexts a fused sum (`Kind::CoeffSum`) makes on the device instead of reading them from the `PtStore`: streamed
//! weight rows (`PtForm::CoeffStream`, `<linear>#<row>`) and generated diagonals (`PtForm::DiagGen`,
//! `<linear>#<n>.<m>.<gg>.<jj>`: gathered from a resident float64 matrix, the tile, then encoded). One source per sum
//! (`sum_kind`); a generated plaintext outside a sum has no runtime path (`form_errors`).
//!
//! Plaintext `(n, m, gg, jj)` of a `DiagLinear` (`B x B` blocks over `n_slots = B t` slots, `b x g` steps,
//! `k = gg b + jj`) holds at slot `v`: `W[n B + ((i + k) mod B)][m B + i]`, `i = ((v - gg b t) mod n_slots) / t`
//! (`np.roll(np.repeat(blk[(i + k) % B, i], t), gg b t)`: the roll moves elements to higher indices).
//! `TileInfo::slot` / `TileInfo::addr` are that arithmetic.
use crate::program::{Kind, Program, PtForm, PtId};
use std::collections::BTreeMap;

/// A plaintext form the runtime generates inside a fused sum (never a `PtStore` slot).
pub fn is_generated(f: PtForm) -> bool {
    matches!(f, PtForm::CoeffStream | PtForm::DiagGen)
}

/// The graph's name for a form (`args.pt.kind`; pc-py's `Program.pt_refs`).
pub fn form_name(f: PtForm) -> &'static str {
    match f {
        PtForm::AtUse => "at_use",
        PtForm::Slot => "slot",
        PtForm::Cached => "cached",
        PtForm::CoeffStream => "stream",
        PtForm::DiagGen => "diag",
    }
}

/// Which source a fused sum reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumKind {
    Stream,
    Diag,
}

/// A bound fused sum's plaintext source (bind.rs `CoeffSumStep::src`): per term, its row in the block's weight frame
/// (rows of N floats) or its gather job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SumSrc {
    Stream { rows: Vec<u32> },
    Diag { jobs: Vec<DiagJob> },
}

impl SumSrc {
    pub fn kind(&self) -> SumKind {
        match self {
            SumSrc::Stream { .. } => SumKind::Stream,
            SumSrc::Diag { .. } => SumKind::Diag,
        }
    }
    pub fn terms(&self) -> usize {
        match self {
            SumSrc::Stream { rows } => rows.len(),
            SumSrc::Diag { jobs } => jobs.len(),
        }
    }
}

/// One generated diagonal: `tile` = the index of its matrix in the session's `DiagSource::tiles`, `(n, m)` the
/// `B x B` block, `gg` the giant and `jj` the baby index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagJob {
    pub tile: u32,
    pub n: u32,
    pub m: u32,
    pub gg: u32,
    pub jj: u32,
}

/// The source of the sum over plaintexts `pts` of `p`: every term streamed, or every term a generated diagonal.
/// Err: no term, a term out of range, a term of another form, or the two sources mixed.
pub fn sum_kind(p: &Program, pts: &[PtId]) -> Result<SumKind, String> {
    let mut kind: Option<SumKind> = None;
    for &id in pts {
        let r = p.pt_refs.get(id as usize).ok_or_else(|| format!("plaintext {id} out of range"))?;
        let k = match r.form {
            PtForm::CoeffStream => SumKind::Stream,
            PtForm::DiagGen => SumKind::Diag,
            f => return Err(format!("term {} is a {} plaintext, not a generated one", r.name, form_name(f))),
        };
        match kind {
            None => kind = Some(k),
            Some(k0) if k0 != k => return Err(format!("mixes streamed rows and generated diagonals (term {})", r.name)),
            Some(_) => {}
        }
    }
    kind.ok_or_else(|| "a sum of no terms".to_string())
}

/// `<linear>#<n>.<m>.<gg>.<jj>` -> (`<linear>`, [n, m, gg, jj]); None when the name is not of that shape (syntax only:
/// the ranges are the tile's, `TileInfo::check`).
pub fn parse_diag_name(name: &str) -> Option<(&str, [u32; 4])> {
    let (lin, rest) = name.rsplit_once('#')?;
    if lin.is_empty() {
        return None;
    }
    let mut out = [0u32; 4];
    let mut it = rest.split('.');
    for x in out.iter_mut() {
        let s = it.next()?;
        // digits only: no sign, no blank, no leading '+' (u32::from_str takes one)
        if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        *x = s.parse().ok()?;
    }
    if it.next().is_some() {
        return None;
    }
    Some((lin, out))
}

/// `perseus.impl.prefill.DiagLinear.split(B)`: (b, g) baby x giant steps, `b` the power of two near sqrt(B), b >= g.
pub fn split(side: u32) -> (u32, u32) {
    let bits = 32 - side.leading_zeros();
    let b = 1u32 << (bits / 2);
    (b, side / b)
}

/// One generated linear's geometry: its `(d_in, d_out)` float64 matrix in `B x B` blocks (`n_in x n_out` of them,
/// `B = side = n_slots / t`), `b x g` baby x giant steps, `t` lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileInfo {
    pub d_in: u32,
    pub d_out: u32,
    pub n_in: u32,
    pub n_out: u32,
    /// B, the block side.
    pub side: u32,
    pub b: u32,
    pub g: u32,
    pub t: u32,
}

impl TileInfo {
    /// The geometry of a `(d_in, d_out)` matrix over `n_slots` slots at `t` lanes; Err names what does not tile.
    pub fn new(d_in: u32, d_out: u32, n_slots: u32, t: u32) -> Result<Self, String> {
        if !n_slots.is_power_of_two() {
            return Err(format!("{n_slots} slots: not a power of two"));
        }
        if t == 0 || !t.is_power_of_two() || t > n_slots {
            return Err(format!("t = {t} lanes: not a power of two up to the {n_slots} slots"));
        }
        let side = n_slots / t;
        let (b, g) = split(side);
        if d_in == 0 || d_out == 0 || !d_in.is_multiple_of(side) || !d_out.is_multiple_of(side) {
            return Err(format!("a ({d_in}, {d_out}) matrix does not tile into {side} x {side} blocks (n_slots {n_slots} / t {t})"));
        }
        Ok(TileInfo { d_in, d_out, n_in: d_in / side, n_out: d_out / side, side, b, g, t })
    }
    /// The plaintexts the linear generates: `n_in n_out B` (every (n, m) block, `b g = B` diagonals each).
    pub fn diagonals(&self) -> u64 {
        self.n_in as u64 * self.n_out as u64 * self.side as u64
    }
    /// The matrix's resident bytes (float64).
    pub fn bytes(&self) -> u64 {
        self.d_in as u64 * self.d_out as u64 * 8
    }
    /// Err when `(n, m, gg, jj)` is outside the linear.
    pub fn check(&self, [n, m, gg, jj]: [u32; 4]) -> Result<(), String> {
        if n >= self.n_in || m >= self.n_out || gg >= self.g || jj >= self.b {
            return Err(format!(
                "(n, m, gg, jj) = ({n}, {m}, {gg}, {jj}) outside n_in {} x n_out {} x g {} x b {}",
                self.n_in, self.n_out, self.g, self.b
            ));
        }
        Ok(())
    }
    /// (row, col) of the matrix element slot `v` of job `j` holds (module doc): `i = ((v - gg b t) mod n_slots) / t`,
    /// row `n B + ((i + gg b + jj) mod B)`, col `m B + i`; the mod by n_slots is a mask (power of two), the mod by B
    /// one conditional subtract (`i + gg b + jj < 2 B`).
    pub fn slot(&self, v: u32, j: &DiagJob) -> (u32, u32) {
        let n_slots = self.side * self.t;
        let s = j.gg * self.b * self.t;
        let i = (v.wrapping_sub(s) & (n_slots - 1)) / self.t;
        let mut r = i + j.gg * self.b + j.jj;
        if r >= self.side {
            r -= self.side;
        }
        (j.n * self.side + r, j.m * self.side + i)
    }
    /// The element's index in the row-major `(d_in, d_out)` matrix: `row d_out + col`, in 64 bits.
    pub fn addr(&self, v: u32, j: &DiagJob) -> u64 {
        let (r, c) = self.slot(v, j);
        r as u64 * self.d_out as u64 + c as u64
    }
}

/// The tiles a session declares (`Session(diag_tiles={name: (d_in, d_out, t)})`), in declaration order: a
/// `DiagJob::tile` indexes `tiles`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagSource {
    pub n_slots: u32,
    pub tiles: Vec<(String, TileInfo)>,
}

impl DiagSource {
    /// `decl` = (name, d_in, d_out, t) per tile; Err on a repeated name or a matrix that does not tile.
    pub fn new(n_slots: u32, decl: &[(String, u32, u32, u32)]) -> Result<Self, String> {
        let mut tiles: Vec<(String, TileInfo)> = Vec::with_capacity(decl.len());
        for (name, d_in, d_out, t) in decl {
            if name.is_empty() || name.contains('#') {
                return Err(format!("tile name {name:?}: empty or holding '#'"));
            }
            if tiles.iter().any(|(n, _)| n == name) {
                return Err(format!("tile {name} declared twice"));
            }
            let ti = TileInfo::new(*d_in, *d_out, n_slots, *t).map_err(|e| format!("tile {name}: {e}"))?;
            tiles.push((name.clone(), ti));
        }
        Ok(DiagSource { n_slots, tiles })
    }
    pub fn index(&self, name: &str) -> Option<usize> {
        self.tiles.iter().position(|(n, _)| n == name)
    }
    /// The resident bytes of every declared tile (float64).
    pub fn bytes(&self) -> u64 {
        self.tiles.iter().map(|(_, t)| t.bytes()).sum()
    }
    /// The job of plaintext `name` (`<linear>#<n>.<m>.<gg>.<jj>`), parsed and range-checked against its tile.
    pub fn job(&self, name: &str) -> Result<DiagJob, String> {
        let (lin, [n, m, gg, jj]) = parse_diag_name(name).ok_or_else(|| format!("{name:?} is not <linear>#<n>.<m>.<gg>.<jj>"))?;
        let tile = self.index(lin).ok_or_else(|| format!("{name}: tile {lin} not declared"))?;
        self.tiles[tile].1.check([n, m, gg, jj]).map_err(|e| format!("{name}: {e}"))?;
        Ok(DiagJob { tile: tile as u32, n, m, gg, jj })
    }
}

/// The tiles a program generates from (indices into `DiagSource::tiles`, first-use order).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagLayout {
    pub tiles: Vec<u32>,
}

/// The tiles `p`'s diagonal sums read. Err = the linears `src` does not declare (first-use order).
pub fn diag_layout(p: &Program, src: &DiagSource) -> Result<DiagLayout, Vec<String>> {
    let mut out = DiagLayout::default();
    let mut missing: Vec<String> = vec![];
    for op in &p.ops {
        for m in op.members() {
            let Kind::CoeffSum { pts, .. } = &m.kind else { continue };
            for &id in pts {
                let Some(r) = p.pt_refs.get(id as usize).filter(|r| r.form == PtForm::DiagGen) else { continue };
                let Some((lin, _)) = parse_diag_name(&r.name) else { continue };
                match src.index(lin) {
                    Some(t) if !out.tiles.contains(&(t as u32)) => out.tiles.push(t as u32),
                    Some(_) => {}
                    None if !missing.iter().any(|x| x == lin) => missing.push(lin.to_string()),
                    None => {}
                }
            }
        }
    }
    if missing.is_empty() {
        Ok(out)
    } else {
        Err(missing)
    }
}

/// Limbs (top + 1) of the deepest-reaching sum of `kind` in `p` (the scratch rows a sum of that source needs), 0 =
/// none.
pub fn rows_needed(p: &Program, shape: pc_shadow::ChainShape, kind: SumKind) -> usize {
    let mut r = 0;
    for op in &p.ops {
        for m in op.members() {
            let Kind::CoeffSum { pts, .. } = &m.kind else { continue };
            if sum_kind(p, pts) != Ok(kind) {
                continue;
            }
            if let Some(&v) = m.ins.first() {
                if let Ok(t) = pc_shadow::meta::top_limb(shape, p.vars[v as usize].meta.level) {
                    r = r.max(t as usize + 1);
                }
            }
        }
    }
    r
}

/// The program-level form errors (the check appends them): a fused sum whose terms are not one generated source,
/// and a plaintext product or sum (`MultPt` / `AddPt`) of a generated plaintext, which no runtime path serves (the
/// replay reads `MultPt` operands from the store).
pub fn form_errors(p: &Program) -> Vec<String> {
    let mut out = vec![];
    for (oi, op) in p.ops.iter().enumerate() {
        for m in op.members() {
            match &m.kind {
                Kind::CoeffSum { pts, .. } => {
                    if let Err(e) = sum_kind(p, pts) {
                        out.push(format!("[pt_form_error] op {oi} coeff_sum: {e}"));
                    }
                }
                Kind::MultPt(id) | Kind::AddPt(id) => {
                    if let Some(r) = p.pt_refs.get(*id as usize).filter(|r| is_generated(r.form)) {
                        out.push(format!(
                            "[pt_form_error] op {oi} {}: {} plaintext {} outside a fused sum (it has no runtime path)",
                            m.kind.name(),
                            form_name(r.form),
                            r.name
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// What a program's fused sums and plaintexts are (compile_check's census).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SumCensus {
    pub stream_sums: usize,
    pub diag_sums: usize,
    /// Terms per sum -> sums, per source.
    pub stream_terms: BTreeMap<usize, usize>,
    pub diag_terms: BTreeMap<usize, usize>,
    /// The vars folded into the sums (`Kind::CoeffSum::fused`).
    pub fused_vars: usize,
    /// Plaintext references by form (`form_name`).
    pub pt_forms: BTreeMap<&'static str, usize>,
    /// References the `PtStore` would hold (every non-generated form).
    pub store_refs: usize,
    /// Store references named like a generated plaintext (`<linear>#...`): 0 when every such name reached a sum.
    pub store_named_generated: usize,
    /// `form_errors(p).len()`.
    pub form_errors: usize,
}

pub fn census(p: &Program) -> SumCensus {
    let mut c = SumCensus::default();
    for op in &p.ops {
        for m in op.members() {
            let Kind::CoeffSum { pts, fused } = &m.kind else { continue };
            c.fused_vars += fused.len();
            match sum_kind(p, pts) {
                Ok(SumKind::Stream) => {
                    c.stream_sums += 1;
                    *c.stream_terms.entry(pts.len()).or_default() += 1;
                }
                Ok(SumKind::Diag) => {
                    c.diag_sums += 1;
                    *c.diag_terms.entry(pts.len()).or_default() += 1;
                }
                Err(_) => {}
            }
        }
    }
    for r in &p.pt_refs {
        *c.pt_forms.entry(form_name(r.form)).or_default() += 1;
        if !is_generated(r.form) {
            c.store_refs += 1;
            if r.name.contains('#') {
                c.store_named_generated += 1;
            }
        }
    }
    c.form_errors = form_errors(p).len();
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{Op, Origin, PtRef, Var};
    use pc_shadow::CtMeta;

    fn pt(name: &str, form: PtForm) -> PtRef {
        PtRef { name: name.into(), level: 30, deg: 1, form, captured_level: None }
    }

    /// A program of one sum over `refs` (term i reads var i), plus a product of `extra` when given.
    fn prog(refs: Vec<PtRef>, extra: Option<PtRef>) -> Program {
        let n = refs.len();
        let mut p = Program { block: "t".into(), ..Default::default() };
        for i in 0..=n {
            p.vars.push(Var { name: format!("v_{i}"), meta: CtMeta::new(30, 1), slot: None });
        }
        p.pt_refs = refs;
        let ins: Vec<u32> = (0..n as u32).collect();
        p.ops.push(Op { kind: Kind::CoeffSum { pts: ins.clone(), fused: vec![] }, ins, outs: vec![n as u32], pre: vec![], origin: Origin::Node(0), inplace: false });
        if let Some(r) = extra {
            p.pt_refs.push(r);
            p.vars.push(Var { name: "v_x".into(), meta: CtMeta::new(30, 2), slot: None });
            let id = (p.pt_refs.len() - 1) as u32;
            p.ops.push(Op { kind: Kind::MultPt(id), ins: vec![0], outs: vec![(p.vars.len() - 1) as u32], pre: vec![], origin: Origin::Node(1), inplace: false });
        }
        p
    }

    #[test]
    fn names_parse_with_their_ranges() {
        assert_eq!(parse_diag_name("b3.wk#0.2.31.7"), Some(("b3.wk", [0, 2, 31, 7])));
        assert_eq!(parse_diag_name("a#b#1.2.3.4"), Some(("a#b", [1, 2, 3, 4])));
        for bad in ["b3.wk#0.2.31", "b3.wk#0.2.31.7.1", "b3.wk#0..31.7", "b3.wk#-1.2.3.4", "b3.wk#+1.2.3.4", "b3.wk", "#1.2.3.4", "b3.wk#12", "b3.wk# 1.2.3.4"] {
            assert_eq!(parse_diag_name(bad), None, "{bad}");
        }
        // the model's tiles at ring 2^16: B = 2^15 / 32 = 1024, b = g = 32
        let src = DiagSource::new(1 << 15, &[("b0.wk".into(), 1024, 4096, 32), ("b0.wv".into(), 4096, 1024, 32)]).unwrap();
        let (_, wk) = &src.tiles[0];
        assert_eq!((wk.side, wk.b, wk.g, wk.n_in, wk.n_out), (1024, 32, 32, 1, 4));
        assert_eq!(wk.diagonals(), 4096);
        assert_eq!(src.bytes(), 2 * 1024 * 4096 * 8);
        assert_eq!(src.job("b0.wk#0.3.31.31").unwrap(), DiagJob { tile: 0, n: 0, m: 3, gg: 31, jj: 31 });
        assert_eq!(src.job("b0.wv#3.0.0.0").unwrap(), DiagJob { tile: 1, n: 3, m: 0, gg: 0, jj: 0 });
        for (bad, why) in [("b0.wk#1.0.0.0", "outside"), ("b0.wk#0.4.0.0", "outside"), ("b0.wk#0.0.32.0", "outside"), ("b0.wk#0.0.0.32", "outside"), ("b0.wr#0.0.0.0", "not declared"), ("b0.wk#0.0.0", "is not")] {
            let e = src.job(bad).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
        // the synthetic shape: 512 slots, t = 16, B = 32, b = 8, g = 4 (DiagLinear.split(32))
        let s = TileInfo::new(32, 128, 512, 16).unwrap();
        assert_eq!((s.side, s.b, s.g, s.n_in, s.n_out), (32, 8, 4, 1, 4));
        assert_eq!(split(1024), (32, 32));
        assert_eq!(split(2048), (64, 32));
        assert!(DiagSource::new(512, &[("x".into(), 32, 32, 16), ("x".into(), 32, 32, 16)]).unwrap_err().contains("twice"));
        assert!(DiagSource::new(512, &[("x#1".into(), 32, 32, 16)]).is_err());
        assert!(TileInfo::new(48, 32, 512, 16).unwrap_err().contains("does not tile"));
        assert!(TileInfo::new(32, 32, 512, 3).is_err());
        assert!(TileInfo::new(32, 32, 500, 4).is_err());
    }

    #[test]
    fn the_slot_index_is_the_rolled_repeated_diagonal() {
        // DiagLinear.slot_vectors written out: v[s] = blk[(i + k) % B, i] for s in [i t, (i+1) t), rolled by gg b t
        let ti = TileInfo::new(64, 128, 256, 8).unwrap(); // B = 32, b = 8, g = 4, n_in 2, n_out 4
        let n_slots = 256u32;
        for (n, m, gg, jj) in [(0, 0, 0, 0), (1, 3, 3, 7), (1, 0, 2, 5), (0, 2, 1, 0)] {
            let j = DiagJob { tile: 0, n, m, gg, jj };
            let k = gg * ti.b + jj;
            let mut want = vec![(0u32, 0u32); n_slots as usize];
            for i in 0..ti.side {
                for l in 0..ti.t {
                    let s = (i * ti.t + l + gg * ti.b * ti.t) % n_slots; // np.roll by gg b t
                    want[s as usize] = (n * ti.side + (i + k) % ti.side, m * ti.side + i);
                }
            }
            for v in 0..n_slots {
                assert_eq!(ti.slot(v, &j), want[v as usize], "job {j:?} slot {v}");
                let (r, c) = want[v as usize];
                assert_eq!(ti.addr(v, &j), r as u64 * 128 + c as u64);
            }
        }
    }

    #[test]
    fn a_sum_takes_one_generated_source() {
        let p = prog(vec![pt("a#0.0.0.0", PtForm::DiagGen), pt("a#0.0.0.1", PtForm::DiagGen)], None);
        assert_eq!(sum_kind(&p, &[0, 1]), Ok(SumKind::Diag));
        assert!(form_errors(&p).is_empty());
        let p = prog(vec![pt("w#3", PtForm::CoeffStream), pt("w#4", PtForm::CoeffStream)], None);
        assert_eq!(sum_kind(&p, &[0, 1]), Ok(SumKind::Stream));
        let p = prog(vec![pt("w#3", PtForm::CoeffStream), pt("a#0.0.0.1", PtForm::DiagGen)], None);
        assert!(sum_kind(&p, &[0, 1]).unwrap_err().contains("mixes"));
        assert_eq!(form_errors(&p).len(), 1);
        let p = prog(vec![pt("a#0.0.0.0", PtForm::DiagGen), pt("pt_3", PtForm::AtUse)], None);
        assert!(sum_kind(&p, &[0, 1]).unwrap_err().contains("at_use"));
        assert!(sum_kind(&p, &[]).is_err());
        assert!(sum_kind(&p, &[7]).is_err());
        // a generated plaintext in a plain product: an error naming it; a store plaintext there is fine
        let p = prog(vec![pt("a#0.0.0.0", PtForm::DiagGen)], Some(pt("a#0.0.1.0", PtForm::DiagGen)));
        let e = form_errors(&p);
        assert_eq!(e.len(), 1);
        assert!(e[0].contains("diag plaintext a#0.0.1.0 outside a fused sum"), "{e:?}");
        let p = prog(vec![pt("a#0.0.0.0", PtForm::DiagGen)], Some(pt("pt_9", PtForm::AtUse)));
        assert!(form_errors(&p).is_empty());
    }

    #[test]
    fn layout_lists_tiles_in_first_use_order_and_names_the_missing() {
        let mut p = prog(vec![pt("b1.wr#0.0.0.0", PtForm::DiagGen), pt("b1.wk#0.1.0.0", PtForm::DiagGen), pt("b1.wr#0.0.0.1", PtForm::DiagGen)], None);
        let src = DiagSource::new(512, &[("b1.wk".into(), 32, 128, 16), ("b1.wr".into(), 32, 128, 16)]).unwrap();
        assert_eq!(diag_layout(&p, &src), Ok(DiagLayout { tiles: vec![1, 0] }));
        let only_wk = DiagSource::new(512, &[("b1.wk".into(), 32, 128, 16)]).unwrap();
        assert_eq!(diag_layout(&p, &only_wk), Err(vec!["b1.wr".to_string()]));
        // stream terms are not tiles
        p.pt_refs.push(pt("b1.up#4", PtForm::CoeffStream));
        assert_eq!(diag_layout(&p, &src).unwrap().tiles, vec![1, 0]);
    }

    #[test]
    fn census_counts_sums_terms_and_forms() {
        let mut p = prog(vec![pt("a#0.0.0.0", PtForm::DiagGen), pt("a#0.0.0.1", PtForm::DiagGen), pt("a#0.0.0.2", PtForm::DiagGen)], None);
        if let Kind::CoeffSum { fused, .. } = &mut p.ops[0].kind {
            *fused = vec![0, 1, 2, 3];
        }
        p.pt_refs.push(pt("pt_5", PtForm::AtUse));
        p.pt_refs.push(pt("mask.x", PtForm::Cached));
        let c = census(&p);
        assert_eq!((c.diag_sums, c.stream_sums, c.fused_vars), (1, 0, 4));
        assert_eq!(c.diag_terms, BTreeMap::from([(3, 1)]));
        assert_eq!(c.pt_forms, BTreeMap::from([("diag", 3), ("at_use", 1), ("cached", 1)]));
        assert_eq!((c.store_refs, c.store_named_generated, c.form_errors), (2, 0, 0));
        // a generated name read from the store would be counted
        p.pt_refs.push(pt("a#0.0.1.0", PtForm::AtUse));
        assert_eq!(census(&p).store_named_generated, 1);
    }

    #[test]
    fn rows_needed_reads_each_source_alone() {
        let shape = pc_shadow::ChainShape::N32;
        let p = prog(vec![pt("a#0.0.0.0", PtForm::DiagGen)], None);
        let want = pc_shadow::meta::top_limb(shape, 30).unwrap() as usize + 1;
        assert_eq!(rows_needed(&p, shape, SumKind::Diag), want);
        assert_eq!(rows_needed(&p, shape, SumKind::Stream), 0);
    }
}
