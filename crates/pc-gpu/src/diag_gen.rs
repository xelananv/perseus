//! The device diagonal encoder (`DiagGen`), the sibling of `encode_dev::CoeffRowsDevice` for generated diagonals:
//! `out = sum_j cts[j] * pt_j` where `pt_j` is diagonal `(n, m, gg, jj)` of a linear's resident float64 matrix (its
//! tile), made inside the sum in chunks of `batch` terms: `diag_gather` (the slot vectors from the tiles), `encode_many`
//! (the `encode` kernels over the chunk), one forward NTT of the chunk's rows, then the loop's products and bookkeeping
//! (`Ops::mult_pt` for the first term, `Ops::mult_add_many` for the rest: `PtStore::sum_with`'s order, so the words are
//! a store-encoded `DeviceFft` sum's). Nothing crosses PCIe per sum and nothing goes through the constant ring (a graph
//! capture leases no slot): the degree factor is a resident table of ones (every term is degree 1) and the jobs are
//! rows of a device table written at bind (`register`), so a captured sum bakes fixed addresses only. Layout and sizes:
//! `DiagShape`; the kernels' arithmetic: pc-kernels `kernels/diag.cuh`, `kernels/encode.cuh`.
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::encode_dev::{CoeffRowsDevice, DevEncoder, EncodeAt, EncoderKind, Payload2, PtForm};
use crate::error::{shape, Result};
use crate::executor::Executor;
use crate::mem::MemPlan;
use crate::ntt::{self, Dir, RowList};
use crate::ops::{ConstRing, Ops, Pt};
use crate::poly::Layout;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Stream};
use pc_kernels::{DiagGatherArgs, DiagJobRow, EncodeArgs, EncodeManyArgs};
use pc_shadow::meta::top_limb;
use pc_shadow::{CtMeta, PtMeta, Shadow};
use std::ffi::c_void;

/// Jobs per chunk at most (`host.diag_batch` is capped here; no sum of the fhe-ssm model exceeds 128 terms).
pub const DIAG_MAXB: usize = 128;

/// A declared tile's geometry as the gather reads it (pc-program `compile::diag::TileInfo`): a row-major `(d_in,
/// d_out)` float64 matrix in `side x side` blocks over `side * t = N/2` slots, `b` baby steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGeom {
    pub d_in: u32,
    pub d_out: u32,
    pub side: u32,
    pub b: u32,
    pub t: u32,
}

/// What a `DiagGen` is carved for (setup).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagShape {
    /// The declared tiles in declaration order (pc-program `DiagJob::tile` indexes them).
    pub tiles: Vec<TileGeom>,
    /// Jobs per chunk, `1..=DIAG_MAXB`; 0 = no scratch (no program generates a diagonal).
    pub batch: usize,
    /// Limbs (top + 1) of the deepest-reaching diagonal sum (pc-program `bind::diag_rows_needed`); 0 = none.
    pub max_rows: usize,
    /// Job-table rows for the bound sums (the sum of their terms); `batch` more rows follow for `probe`.
    pub jobs: usize,
}

impl DiagShape {
    /// Arena bytes of the tiles (MemPlan section "diag.tiles"): one carve of `d_in d_out` doubles each.
    pub fn tile_bytes(&self) -> usize {
        self.tiles.iter().map(|t| Arena::bytes_for::<f64>(t.d_in as usize * t.d_out as usize)).sum()
    }
    /// Arena bytes of the job table (section "diag.jobs"): the bound sums' rows and the probe's.
    pub fn job_bytes(&self) -> usize {
        Arena::bytes_for::<DiagJobRow>(self.jobs + self.batch)
    }
    /// Arena bytes of the NTT row lists (section "diag.lists"): per top `t < max_rows` the `batch (t + 1)` rows
    /// `(j (t + 1) + l, l)` (offset, prime id).
    pub fn list_bytes(&self) -> usize {
        if self.batch == 0 {
            return 0;
        }
        (0..self.max_rows).map(|t| Arena::bytes_for::<u32>(self.batch * (t + 1)) + Arena::bytes_for::<i32>(self.batch * (t + 1))).sum()
    }
    /// Reserve the four sections ("diag.tiles", "diag.scratch", "diag.jobs", "diag.lists": `mem_report` names them).
    pub fn reserve<W: DevWord>(&self, plan: &mut MemPlan) {
        let lay = plan.lay;
        plan.reserve("diag.tiles", self.tile_bytes());
        plan.reserve("diag.scratch", DiagGen::<W>::bytes(&lay, self.batch, self.max_rows));
        plan.reserve("diag.jobs", self.job_bytes());
        plan.reserve("diag.lists", self.list_bytes());
    }
}

/// A run of rows of the job table: the terms of one bound diagonal sum, in term order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobRange {
    pub at: u32,
    pub len: u32,
}

/// Host counters (no device read): what the host issued, in eager runs and graph captures (a replayed graph adds none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiagStats {
    pub sums: u64,
    pub terms: u64,
    pub chunks: u64,
}

pub struct DiagGen<'a, W: DevWord> {
    lay: Layout,
    tiles: Vec<(TileGeom, DBuf<'a, Compute, f64>)>,
    uploaded: Vec<bool>,
    /// per job: N/2 double2 slots (v), the FFT ping-pong (a, b: N double2), the coefficients (c: N doubles), max |c|
    v: DBuf<'a, Compute, f64>,
    a: DBuf<'a, Compute, f64>,
    b: DBuf<'a, Compute, f64>,
    c: DBuf<'a, Compute, f64>,
    cmax: DBuf<'a, Compute, u64>,
    /// per job: rows `0..L` of its plaintext, job j at `j L N` words (L = the sum's limbs; `lists[L - 1]`)
    rows: DBuf<'a, Compute, W>,
    /// jobs whose 2^61 split was nonzero, counted per launch since setup (a replayed graph counts again; the encode's
    /// counter; the diagonal bound keeps it 0)
    nsplit: DBuf<'a, Compute, u64>,
    /// `deg_factor` of a degree-1 plaintext: one word per Q prime, all 1
    ones: Option<ConstBuf<'a, W>>,
    jobs: DBuf<'a, Compute, DiagJobRow>,
    /// rows written for bound sums (the probe's rows follow `shape.jobs`)
    filled: usize,
    lists: Vec<RowList<'a>>,
    shape: DiagShape,
    pub stats: DiagStats,
}

impl<'a, W: DevWord> DiagGen<'a, W> {
    /// Arena bytes of the per-chunk scratch (MemPlan section "diag.scratch"): per job v N, a and b 2N each, c N doubles
    /// (48 N bytes) and one max word, plus `max_rows` rows of N words; the split counter; the table of ones (when
    /// `batch` > 0). At N = 2^16, W = u32: 3,145,736 + 262,144 L bytes per job.
    pub fn bytes(lay: &Layout, batch: usize, max_rows: usize) -> usize {
        let n = lay.n;
        let f = |k: usize| Arena::bytes_for::<f64>(k);
        let ones = if batch > 0 { Arena::bytes_for::<W>(lay.nq) } else { 0 };
        f(batch * n) + 2 * f(batch * 2 * n) + f(batch * n) + Arena::bytes_for::<u64>(batch) + Arena::bytes_for::<W>(batch * max_rows * n) + Arena::bytes_for::<u64>(1) + ones
    }

    /// Carve everything `shape` names (setup; `DiagShape::reserve` sized it), zero the split counter, write the
    /// table of ones and the row lists. The tiles are uploaded later (`upload_tile`), the job rows at bind (`register`).
    pub fn new(arena: &'a Arena, s: &Stream<Compute>, lay: Layout, shape: DiagShape) -> Result<Self> {
        let n = lay.n;
        if shape.batch > DIAG_MAXB || (shape.batch > 0) != (shape.max_rows > 0) || shape.max_rows > lay.nq {
            return Err(shape_err(format!("batch {} (0..={DIAG_MAXB}), max_rows {} (0..={}, 0 iff batch 0)", shape.batch, shape.max_rows, lay.nq)));
        }
        for (i, t) in shape.tiles.iter().enumerate() {
            let ok = t.t.is_power_of_two() && t.side as usize * t.t as usize == n / 2 && t.b > 0 && t.side.is_multiple_of(t.b) && t.d_in.is_multiple_of(t.side) && t.d_out.is_multiple_of(t.side) && t.d_in > 0 && t.d_out > 0;
            if !ok {
                return Err(shape_err(format!("tile {i}: {t:?} does not tile N/2 = {} slots", n / 2)));
            }
        }
        let tiles = shape.tiles.iter().map(|t| Ok((*t, arena.alloc("diag.tiles", t.d_in as usize * t.d_out as usize)?))).collect::<Result<Vec<_>>>()?;
        let (bt, mr) = (shape.batch, shape.max_rows);
        let mut nsplit = arena.alloc("diag.scratch", 1)?;
        nsplit.zero(s)?;
        let v = arena.alloc("diag.scratch", bt * n)?;
        let a = arena.alloc("diag.scratch", bt * 2 * n)?;
        let b = arena.alloc("diag.scratch", bt * 2 * n)?;
        let c = arena.alloc("diag.scratch", bt * n)?;
        let cmax = arena.alloc("diag.scratch", bt)?;
        let rows = arena.alloc("diag.scratch", bt * mr * n)?;
        let ones = if bt > 0 { Some(ConstBuf::new(arena, "diag.scratch", s, &vec![W::from_u64(1); lay.nq])?) } else { None };
        let jobs = arena.alloc("diag.jobs", shape.jobs + bt)?;
        let lists = if bt > 0 {
            (0..mr).map(|t| RowList::new(arena, s, &(0..bt).flat_map(|j| (0..=t).map(move |l| ((j * (t + 1) + l) as u32, l as i32))).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        let uploaded = vec![false; tiles.len()];
        Ok(DiagGen { lay, tiles, uploaded, v, a, b, c, cmax, rows, nsplit, ones, jobs, filled: 0, lists, shape, stats: DiagStats::default() })
    }

    pub fn shape(&self) -> &DiagShape {
        &self.shape
    }
    pub fn batch(&self) -> usize {
        self.shape.batch
    }
    pub fn max_rows(&self) -> usize {
        self.shape.max_rows
    }
    /// Job-table rows written for bound sums.
    pub fn filled(&self) -> usize {
        self.filled
    }
    pub fn uploaded(&self, tile: usize) -> bool {
        self.uploaded.get(tile).copied().unwrap_or(false)
    }
    /// Declared tiles never uploaded, by index.
    pub fn missing(&self) -> Vec<usize> {
        self.uploaded.iter().enumerate().filter(|(_, &u)| !u).map(|(i, _)| i).collect()
    }

    /// Upload tile `tile`'s matrix (`d_in d_out` doubles, row-major). Synchronous on `s`; the buffer was carved at
    /// setup, so this is legal after the arena froze (and re-uploading replaces the values in place).
    pub fn upload_tile(&mut self, s: &Stream<Compute>, tile: usize, w: &[f64]) -> Result<()> {
        let (g, buf) = self.tiles.get_mut(tile).ok_or_else(|| shape_err(format!("tile {tile} not declared")))?;
        if w.len() != g.d_in as usize * g.d_out as usize {
            return Err(shape_err(format!("tile {tile}: {} values for a ({}, {}) matrix", w.len(), g.d_in, g.d_out)));
        }
        buf.upload_at(s, 0, w)?;
        self.uploaded[tile] = true;
        Ok(())
    }

    /// The table row of diagonal `(n, m, gg, jj)` of tile `tile`, range-checked against the tile (the bind checked
    /// it against the same geometry; checked again because the gather reads the matrix without bounds).
    pub fn job_row(&self, tile: usize, n: u32, m: u32, gg: u32, jj: u32) -> Result<DiagJobRow> {
        let (g, buf) = self.tiles.get(tile).ok_or_else(|| shape_err(format!("tile {tile} not declared")))?;
        if n >= g.d_in / g.side || m >= g.d_out / g.side || gg >= g.side / g.b || jj >= g.b {
            return Err(shape_err(format!("tile {tile} {g:?}: (n, m, gg, jj) = ({n}, {m}, {gg}, {jj}) outside it")));
        }
        Ok(DiagJobRow { w: buf.dptr() as u64, d_out: g.d_out, side: g.side, b: g.b, log_t: g.t.trailing_zeros(), n, m, gg, jj })
    }

    /// Append one bound sum's job rows (term order) to the table; returns its range (`register_all` of one sum).
    pub fn register(&mut self, s: &Stream<Compute>, rows: &[DiagJobRow]) -> Result<JobRange> {
        Ok(self.register_all(s, &[rows.to_vec()])?[0])
    }

    /// Append the job rows of every diagonal sum of one bound program (term order), checked against the capacity first
    /// and uploaded once, so a refused program consumes no row; returns each sum's range. The capacity
    /// (`DiagShape::jobs`) is every diagonal term of the programs the session was sized for: it assumes each of them
    /// binds once. Setup (bind) only: a synchronous upload on `s`; the rows stay put once a graph reads them.
    pub fn register_all(&mut self, s: &Stream<Compute>, sums: &[Vec<DiagJobRow>]) -> Result<Vec<JobRange>> {
        let total: usize = sums.iter().map(|r| r.len()).sum();
        if sums.iter().any(|r| r.is_empty()) || self.filled + total > self.shape.jobs {
            return Err(shape_err(format!(
                "{} sums of {total} job rows: {} of {} filled (DiagShape::jobs: every diagonal term of the programs the session was sized for, each bound once)",
                sums.len(),
                self.filled,
                self.shape.jobs
            )));
        }
        let mut ranges = Vec::with_capacity(sums.len());
        let mut at = self.filled;
        for r in sums {
            ranges.push(JobRange { at: at as u32, len: r.len() as u32 });
            at += r.len();
        }
        if total > 0 {
            let flat: Vec<DiagJobRow> = sums.concat();
            self.jobs.upload_at(s, self.filled, &flat)?;
        }
        self.filled += total;
        Ok(ranges)
    }

    fn deg_factor(&self, deg: u8) -> Result<*const c_void> {
        if deg != 1 {
            return Err(shape_err(format!("degree {deg}: a generated diagonal is encoded like its ciphertext's effective level at degree 1 (the resident deg_factor table holds 1s)")));
        }
        self.ones.as_ref().map(|o| o.vptr()).ok_or_else(|| shape_err("no scratch (DiagShape::batch 0): no diagonal sum was declared"))
    }

    /// Gather, encode and NTT rows `[at, at + nb)` of the table into the scratch (job j's rows `0..=top` at `j (top +
    /// 1) N` words), at `scale`. Asynchronous on `s`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, at: usize, nb: usize, top: usize, scale: f64, deg: u8) -> Result<()> {
        let n = self.lay.n;
        let l = top + 1;
        if nb == 0 || nb > self.shape.batch || l > self.shape.max_rows || at + nb > self.shape.jobs + self.shape.batch {
            return Err(shape_err(format!("{nb} jobs at row {at}, {l} limbs (batch {}, max_rows {}, {} table rows)", self.shape.batch, self.shape.max_rows, self.shape.jobs + self.shape.batch)));
        }
        let df = self.deg_factor(deg)?;
        // SAFETY (pointer): at + nb <= the table's rows (checked above)
        let g = DiagGatherArgs { jobs: unsafe { self.jobs.dptr().add(at) }, v: self.v.vptr_mut(), v_stride: n as u64, njobs: nb as i32, pad: 0 };
        // SAFETY: the rows were written by `register` / `probe` from `job_row` (every address inside its tile, carved
        // at setup); v holds batch x N doubles >= nb x N
        launch_result("diag_gather", unsafe { (ctx.table.diag_gather)(&g, s.raw()) })?;
        let e = EncodeManyArgs {
            e: EncodeArgs {
                v: self.v.vptr(),
                a: self.a.vptr_mut(),
                b: self.b.vptr_mut(),
                c: self.c.vptr_mut(),
                cmax: self.cmax.dptr_mut(),
                out: self.rows.vptr_mut(),
                deg_factor: df,
                pr: ctx.pr(),
                scale,
                nl: l as i32,
                pad: 0,
            },
            v_stride: n as u64,
            ab_stride: 2 * n as u64,
            c_stride: n as u64,
            out_stride: (l * n) as u64,
            nsplit: self.nsplit.dptr_mut(),
            batch: nb as i32,
            pad: 0,
        };
        // SAFETY: per job v N, a / b 2N, c N doubles and one max word, rows nb L N <= batch max_rows N words (checked);
        // deg_factor holds nq >= l words
        launch_result("encode_many", unsafe { (ctx.table.encode_many)(&e, s.raw()) })?;
        // the list of `top` holds (j l + i, i) for j < batch, i < l: its first nb l rows are this chunk's
        ntt::transform_rows_prefix(ctx, s, Dir::Forward, &mut self.rows, &self.lists[top], nb * l)?;
        self.stats.chunks += 1;
        Ok(())
    }

    /// Device address of job `j`'s rows after `generate` at `top` (rows `0..=top`, N words apart); Err unless `j <
    /// batch` and `top < max_rows`.
    pub fn rows_ptr(&self, j: usize, top: usize) -> Result<*const c_void> {
        if j >= self.shape.batch || top >= self.shape.max_rows {
            return Err(shape_err(format!("rows of job {j} at top {top} (batch {}, max_rows {})", self.shape.batch, self.shape.max_rows)));
        }
        // SAFETY: j (top + 1) N + (top + 1) N <= batch max_rows N words, inside `rows` (checked above)
        Ok(unsafe { self.rows.dptr().add(j * (top + 1) * self.lay.n) } as *const c_void)
    }
    /// Device address of job `j`'s gathered slots (N/2 double2) after `generate` (`Payload2::DeviceSlots`); Err unless
    /// `j < batch`.
    pub fn slots_ptr(&self, j: usize) -> Result<*const c_void> {
        if j >= self.shape.batch {
            return Err(shape_err(format!("slots of job {j} (batch {})", self.shape.batch)));
        }
        // SAFETY: (j + 1) N <= batch N doubles, inside `v` (checked above)
        Ok(unsafe { self.v.dptr().add(j * self.lay.n) } as *const c_void)
    }
    /// Host copy of job `j`'s rows `0..=top` (tests, `Session.diag_probe`). Synchronous on `s`.
    pub fn download_rows(&self, s: &Stream<Compute>, j: usize, top: usize) -> Result<Vec<W>> {
        let l = top + 1;
        if j >= self.shape.batch || l > self.shape.max_rows {
            return Err(shape_err(format!("job {j}, {l} limbs (batch {}, max_rows {})", self.shape.batch, self.shape.max_rows)));
        }
        Ok(self.rows.download_range(s, j * l * self.lay.n, l * self.lay.n)?)
    }
    /// Host copy of job `j`'s gathered slots as (re, im) doubles. Synchronous on `s`.
    pub fn download_slots(&self, s: &Stream<Compute>, j: usize) -> Result<Vec<f64>> {
        if j >= self.shape.batch {
            return Err(shape_err(format!("job {j} (batch {})", self.shape.batch)));
        }
        Ok(self.v.download_range(s, j * self.lay.n, self.lay.n)?)
    }

    /// `generate` of ad-hoc rows (at most `batch`, written to the probe rows after the bound sums'): the
    /// tests' and `Session.diag_probe`'s entry. Synchronous upload, asynchronous generation on `s`.
    pub fn probe<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, rows: &[DiagJobRow], top: usize, scale: f64) -> Result<()> {
        if rows.is_empty() || rows.len() > self.shape.batch {
            return Err(shape_err(format!("{} probe rows (1..={})", rows.len(), self.shape.batch)));
        }
        let at = self.shape.jobs;
        self.jobs.upload_at(s, at, rows)?;
        self.generate(ctx, s, at, rows.len(), top, scale, 1)
    }

    /// The probe's comparison: `what` through `enc` (one `encode` launch, `DeviceFft`, degree 1, then the forward NTT)
    /// into job 0's scratch rows at `top`, returned as rows `0..=top`. It overwrites job 0's generated rows (download
    /// them first); `what` may be `Payload2::DeviceSlots` of a gathered vector (`slots_ptr`), which it does not touch.
    /// Synchronous on `s`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_with<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, enc: &mut DevEncoder<'_, W>, ring: &mut ConstRing<'_, W>, what: Payload2<'_>, top: usize, scale: f64) -> Result<Vec<W>> {
        let l = top + 1;
        if self.shape.batch == 0 || l > self.shape.max_rows {
            return Err(shape_err(format!("{l} limbs (batch {}, max_rows {})", self.shape.batch, self.shape.max_rows)));
        }
        enc.encode(ctx, s, ring, EncoderKind::DeviceFft, what, EncodeAt { top, scale, deg: 1, form: PtForm::Eval }, &mut self.rows)?;
        Ok(self.rows.download_range(s, 0, l * self.lay.n)?)
    }

    /// Jobs whose 2^61 split was nonzero, per launch since setup (a replayed graph counts again; any nonzero count = a
    /// coefficient past 2^61: |W| times the scale broke the bound the encoder relies on). Synchronous on `s`.
    pub fn splits(&self, s: &Stream<Compute>) -> Result<u64> {
        Ok(self.nsplit.download(s)?.first().copied().unwrap_or(0))
    }

    /// `out = sum_j cts[j] * pt_j`, `pt_j` the diagonal of table row `jobs.at + j`, encoded like `cts[j]` (its
    /// effective level, degree 1, the level's scaling factor: `PtStore::encode_like`). Every term at one level, degree
    /// 1 (`CoeffRowsDevice::sum_meta`'s check); `out` must not be one of `cts`. Bookkeeping and launches as
    /// `PtStore::sum_with`: `mult_pt` of the first term, then `mult_add_many`. Asynchronous on `ex.compute`; the scratch
    /// is rewritten per chunk after the previous chunk's products in stream order.
    #[allow(clippy::too_many_arguments)]
    pub fn sum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, cts: &[&Ct<'_, W>], jobs: JobRange) -> Result<()> {
        let nt = jobs.len as usize;
        if cts.is_empty() || cts.len() != nt {
            return Err(shape_err(format!("{} ciphertexts, {nt} jobs", cts.len())));
        }
        if jobs.at as usize + nt > self.filled {
            return Err(shape_err(format!("jobs {}..{} outside the {} rows registered at bind", jobs.at, jobs.at as usize + nt, self.filled)));
        }
        let metas: Vec<CtMeta> = cts.iter().map(|c| c.meta).collect();
        let (meta, scale) = CoeffRowsDevice::<W>::sum_meta(sh, &metas)?;
        // what PtStore::encode_like gives every term: its effective level, degree 1
        let (level, deg) = (metas[0].effective_level(sh.unit()), 1u8);
        let top = top_limb(ctx.shape, level)? as usize;
        if top + 1 > self.shape.max_rows {
            return Err(shape_err(format!("{} limbs, the scratch holds {} (DiagShape::max_rows)", top + 1, self.shape.max_rows)));
        }
        let ob = out.buf.w().base;
        if cts.iter().any(|c| c.buf.r().base == ob) {
            return Err(shape_err("the output aliases a term"));
        }
        let pm = PtMeta { level, deg, nf: scale, tag: Default::default() };
        let bt = self.shape.batch;
        for (ci, cc) in cts.chunks(bt).enumerate() {
            self.generate(ctx, &ex.compute, jobs.at as usize + ci * bt, cc.len(), top, scale, deg)?;
            // SAFETY: job j's rows 0..=top were just written by `generate` (stream order) and are not rewritten before
            // the products below run (the next chunk's generate is issued after them on the same lane)
            let pts: Vec<Pt<'_>> = (0..cc.len()).map(|j| Ok(unsafe { Pt::from_raw(pm, self.rows_ptr(j, top)?) })).collect::<Result<_>>()?;
            if ci == 0 {
                ops.mult_pt(ctx, ex, sh, std::slice::from_mut(out), &cc[..1], &pts[..1])?;
                if cc.len() > 1 {
                    ops.mult_add_many(ctx, ex, sh, out, &cc[1..], &pts[1..])?;
                }
            } else {
                ops.mult_add_many(ctx, ex, sh, out, cc, &pts)?;
            }
        }
        if (out.meta.level, out.meta.deg) != (meta.level, meta.deg) || out.meta.nf != meta.nf {
            return Err(shape_err(format!("the sum landed at level {} deg {} nf 2^{:.3}, its terms' bookkeeping says level {} deg {} nf 2^{:.3}", out.meta.level, out.meta.deg, out.meta.nf.log2(), meta.level, meta.deg, meta.nf.log2())));
        }
        self.stats.sums += 1;
        self.stats.terms += nt as u64;
        Ok(())
    }
}

fn shape_err(msg: impl Into<String>) -> crate::Error {
    shape("DiagGen", msg)
}
