//! The device diagonal encoder (pc-gpu `DiagGen`) on a GPU, feature `gpu`, run through tools/gpu.sh in this order:
//! `cargo test --release -p pc-program --features gpu --test diag_gen_gpu -- --ignored --test-threads=1 --nocapture`.
//! (a) a generated plaintext's slots equal its host-built slot vector (`TileInfo::slot`, the arithmetic pinned to the
//! port's `filling.diag_slot_index` by the CPU tests) and its rows equal the eager `DeviceFft` encode of that vector,
//! every limb; (b) `encode_many` over B = 1, 32, 128 jobs equals B single encodes; (c) a fused sum equals
//! `PtStore::mult_pt_sum` of the same terms, word for word (one chunk, and four); (f) device ms per plaintext at B = 1,
//! 32, 128 and 18 / 10 limbs. (d) and (e) are the port's (python/perseus/examples/fhe_ssm/tests/test_diag_gpu.py).
#![cfg(feature = "gpu")]
use pc_core::encode::{Complex, Payload};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Device, Event, EventPool};
use pc_gpu::diag_gen::{DiagGen, DiagShape, TileGeom};
use pc_gpu::encode_dev::Payload2;
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::pt_store::{PtStore, PtStoreCfg};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_kernels::DiagJobRow;
use pc_program::compile::diag::DiagSource;
use pc_shadow::meta::level_from_top_limb;
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use std::sync::Arc;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, k: usize) -> usize {
        ((self.next() as u128 * k as u128) >> 64) as usize
    }
    /// uniform in [-1, 1): |W| < 1 keeps every coefficient under 2^61 (split 0), as the model's matrices do
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Two tiles at t = 32 lanes over 2^15 slots (B = 1,024, b = g = 32): `t.wk` 1,024 x 4,096 (32-term groups), `t.wv`
/// 4,096 x 1,024 (128-term groups), random entries in [-1, 1).
fn tiles() -> Res<(DiagSource, Vec<Vec<f64>>)> {
    let decl = vec![("t.wk".to_string(), 1024, 4096, 32), ("t.wv".to_string(), 4096, 1024, 32)];
    let src = DiagSource::new(1 << 15, &decl)?;
    let mut rng = Rng(0x7115_d1a6);
    let mats = src.tiles.iter().map(|(_, t)| (0..t.d_in as usize * t.d_out as usize).map(|_| rng.unit()).collect()).collect();
    Ok((src, mats))
}

struct Env<'a, 'e> {
    ctx: Context<'a, W, M>,
    ex: Executor<'a, W>,
    ops: Ops<'a, W>,
    store: PtStore<'a, 'e, W>,
    dg: DiagGen<'a, W>,
    pool: CtPool<'a, W>,
    sh: Shadow,
    src: DiagSource,
    mats: Vec<Vec<f64>>,
    t0: Event,
    t1: Event,
}

/// A runtime with the two tiles uploaded and a `DiagGen` of `batch` jobs per chunk, `max_rows` limbs, `jobs` table rows.
fn with_env(batch: usize, max_rows: usize, jobs: usize, f: impl FnOnce(&mut Env<'_, '_>) -> Res<()>) -> Res<()> {
    let dev = Device::open()?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let mut plan = MemPlan::build::<W, M>(&chain, MemCfg { k_max: 4, pool: vec![PoolClass { k: 1, count: 12 }], events: 16, scratch_cts: 2 })?;
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut plan, &ocfg);
    let pcfg = PtStoreCfg { slots: 2, ring: 2, encode_threads: 2, ..PtStoreCfg::default() };
    PtStore::<W>::reserve(&mut plan, &pcfg);
    let (src, mats) = tiles()?;
    let geoms = src.tiles.iter().map(|(_, t)| TileGeom { d_in: t.d_in, d_out: t.d_out, side: t.side, b: t.b, t: t.t }).collect();
    let shape = DiagShape { tiles: geoms, batch, max_rows, jobs };
    shape.reserve::<W>(&mut plan);
    let arena = plan.allocate(&dev)?;
    let events = EventPool::new(&dev, 16)?;
    let ex = Executor::<W>::new(&dev, &arena, &plan)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let store = PtStore::new(&arena, &dev, &ex.compute, &ctx, &events, pcfg)?;
    let mut dg = DiagGen::new(&arena, &ex.compute, ctx.lay, shape)?;
    for (i, w) in mats.iter().enumerate() {
        dg.upload_tile(&ex.compute, i, w)?;
    }
    assert!(dg.missing().is_empty());
    ex.compute.sync()?;
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let (t0, t1) = (Event::timing(&dev)?, Event::timing(&dev)?);
    let mut env = Env { ctx, ex, ops, store, dg, pool, sh, src, mats, t0, t1 };
    f(&mut env)
}

/// `<tile>#<n>.<m>.<gg>.<jj>` of a random diagonal.
fn random_name(e: &Env<'_, '_>, rng: &mut Rng) -> String {
    let (name, t) = &e.src.tiles[rng.below(e.src.tiles.len())];
    format!("{name}#{}.{}.{}.{}", rng.below(t.n_in as usize), rng.below(t.n_out as usize), rng.below(t.g as usize), rng.below(t.b as usize))
}

/// Per tile `(0, 0, 0, 0)`, `(n_in - 1, n_out - 1, g - 1, b - 1)` and `(n_in - 1, 0, 1, 0)`.
fn corners(e: &Env<'_, '_>) -> Vec<String> {
    let mut out = vec![];
    for (name, t) in &e.src.tiles {
        for (n, m, gg, jj) in [(0, 0, 0, 0), (t.n_in - 1, t.n_out - 1, t.g - 1, t.b - 1), (t.n_in - 1, 0, 1, 0)] {
            out.push(format!("{name}#{n}.{m}.{gg}.{jj}"));
        }
    }
    out
}

/// Group `(m, gg)` of tile `tile` in the order the capture records it (n outer, jj inner).
fn group(e: &Env<'_, '_>, tile: usize, m: u32, gg: u32) -> Vec<String> {
    let (name, t) = &e.src.tiles[tile];
    (0..t.n_in).flat_map(|n| (0..t.b).map(move |jj| format!("{name}#{n}.{m}.{gg}.{jj}"))).collect()
}

fn row(e: &Env<'_, '_>, name: &str) -> Res<DiagJobRow> {
    let j = e.src.job(name)?;
    Ok(e.dg.job_row(j.tile as usize, j.n, j.m, j.gg, j.jj)?)
}

/// The slot vector of `name` built on the host from the matrix (`TileInfo::addr`): (W element, 0) per slot.
fn oracle(e: &Env<'_, '_>, name: &str) -> Res<Vec<Complex>> {
    let j = e.src.job(name)?;
    let (t, w) = (&e.src.tiles[j.tile as usize].1, &e.mats[j.tile as usize]);
    Ok((0..1u32 << 15).map(|v| Complex::new(w[t.addr(v, &j) as usize], 0.0)).collect())
}

fn diff(a: &[W], b: &[W]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// A ciphertext of random residues on rows 0..=top of both parts, at `level`, degree 1, the level's scale (it borrows
/// the pool only, so the runtime's other parts stay free for `&mut`).
fn random_ct<'p, 'a>(pool: &'p CtPool<'a, W>, ctx: &Context<'a, W, M>, ex: &Executor<'a, W>, rng: &mut Rng, level: u32, top: usize) -> Res<pc_gpu::PoolCt<'p, 'a, W>> {
    let n = ctx.lay.n;
    let meta = CtMeta { level, deg: 1, nf: ctx.scale.sf(top as i32)?, slots: (n / 2) as u32, tag: Default::default() };
    let mut c = pool.acquire(1, meta)?;
    let rows: Vec<usize> = (0..=top).collect();
    let cm = c.get_mut().ok_or("shared ciphertext")?;
    for part in 0..2 {
        let mut data: Vec<W> = Vec::with_capacity((top + 1) * n);
        for r in 0..=top {
            let p = ctx.prime(r) as usize;
            data.extend((0..n).map(|_| rng.below(p) as W));
        }
        cm.buf.upload_rows(&ex.compute, 0, part, &rows, &data)?;
    }
    Ok(c)
}

fn ct_rows(e: &Env<'_, '_>, c: &Ct<'_, W>, top: usize) -> Res<Vec<W>> {
    let rows: Vec<usize> = (0..=top).collect();
    let mut v = c.buf.download_rows(&e.ex.compute, 0, 0, &rows)?;
    v.extend(c.buf.download_rows(&e.ex.compute, 0, 1, &rows)?);
    Ok(v)
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <diag_gen_gpu test binary> --ignored --test-threads=1 --nocapture"]
fn a_generated_rows_equal_the_eager_device_encode_of_the_host_vector() -> Res<()> {
    with_env(32, 20, 0, |e| {
        let mut rng = Rng(0xa11ce);
        // the corners of each tile ((0, 0, 0, 0), the last block's last diagonal, the first rolled giant step) and 26 random
        let mut names: Vec<String> = corners(e);
        names.extend((0..32 - names.len()).map(|_| random_name(e, &mut rng)));
        for l in [20usize, 10] {
            let top = l - 1;
            let scale = e.ctx.scale.sf(top as i32)?;
            let rows = names.iter().map(|nm| row(e, nm)).collect::<Res<Vec<_>>>()?;
            e.dg.probe(&e.ctx, &e.ex.compute, &rows, top, scale)?;
            let mut generated = vec![];
            let (mut bad_slots, mut bad) = (0usize, 0usize);
            for (j, nm) in names.iter().enumerate() {
                generated.push(e.dg.download_rows(&e.ex.compute, j, top)?);
                let got = e.dg.download_slots(&e.ex.compute, j)?;
                let want: Vec<f64> = oracle(e, nm)?.iter().flat_map(|c| [c.re, c.im]).collect();
                bad_slots += got.iter().zip(&want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            }
            for (j, nm) in names.iter().enumerate() {
                let v = oracle(e, nm)?;
                let what = Payload2::Slots { values: &v, slots: 1 << 15, payload: Payload::Real };
                let want = e.dg.encode_with(&e.ctx, &e.ex.compute, &mut e.store.enc, &mut e.ops.ring, what, top, scale)?;
                bad += diff(&generated[j], &want);
            }
            println!("[diag_gen] (a) {} jobs at {l} limbs: slots != host vector {bad_slots} of {}; rows != eager DeviceFft {bad} of {}", names.len(), names.len() << 16, (names.len() * l) << 16);
            assert_eq!((bad_slots, bad), (0, 0));
        }
        assert_eq!(e.dg.splits(&e.ex.compute)?, 0);
        Ok(())
    })
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <diag_gen_gpu test binary> --ignored --test-threads=1 --nocapture"]
fn b_encode_many_equals_single_encodes() -> Res<()> {
    with_env(128, 20, 0, |e| {
        let mut rng = Rng(0xb0b);
        let top = 19;
        let scale = e.ctx.scale.sf(top as i32)?;
        for nb in [1usize, 32, 128] {
            let names: Vec<String> = (0..nb).map(|_| random_name(e, &mut rng)).collect();
            let rows = names.iter().map(|nm| row(e, nm)).collect::<Res<Vec<_>>>()?;
            e.dg.probe(&e.ctx, &e.ex.compute, &rows, top, scale)?;
            let generated = (0..nb).map(|j| Ok(e.dg.download_rows(&e.ex.compute, j, top)?)).collect::<Res<Vec<_>>>()?;
            let mut bad = 0;
            for (j, g) in generated.iter().enumerate() {
                let what = Payload2::DeviceSlots { ptr: e.dg.slots_ptr(j)?, slots: 1 << 15, payload: Payload::Real };
                bad += diff(g, &e.dg.encode_with(&e.ctx, &e.ex.compute, &mut e.store.enc, &mut e.ops.ring, what, top, scale)?);
            }
            println!("[diag_gen] (b) encode_many over B = {nb}: rows != {nb} single encodes {bad} of {}", (nb * 20) << 16);
            assert_eq!(bad, 0);
        }
        Ok(())
    })
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <diag_gen_gpu test binary> --ignored --test-threads=1 --nocapture"]
fn c_fused_sum_equals_the_store_loop() -> Res<()> {
    // a 32-term group of t.wk (one chunk) and a 128-term group of t.wv (four chunks of 32)
    with_env(32, 20, 32 + 128, |e| {
        let groups = [group(e, 0, 1, 3), group(e, 1, 0, 5)];
        let mut ranges = vec![];
        for g in &groups {
            let rows = g.iter().map(|nm| row(e, nm)).collect::<Res<Vec<_>>>()?;
            ranges.push(e.dg.register(&e.ex.compute, &rows)?);
        }
        let mut rng = Rng(0xc0ffee);
        for l in [20usize, 10] {
            let top = l - 1;
            let level = level_from_top_limb(e.ctx.shape, top as u32)?;
            let cts = (0..8).map(|_| random_ct(&e.pool, &e.ctx, &e.ex, &mut rng, level, top)).collect::<Res<Vec<_>>>()?;
            for (g, r) in groups.iter().zip(&ranges) {
                let terms: Vec<&Ct<'_, W>> = (0..g.len()).map(|j| &*cts[j % cts.len()]).collect();
                let values = g.iter().map(|nm| oracle(e, nm)).collect::<Res<Vec<_>>>()?;
                let meta = terms[0].meta;
                let (mut o1, mut o2) = (e.pool.acquire(1, meta)?, e.pool.acquire(1, meta)?);
                let (m1, m2) = (o1.get_mut().ok_or("shared")?, o2.get_mut().ok_or("shared")?);
                e.dg.sum(&e.ctx, &mut e.ex, &mut e.ops, &e.sh, m1, &terms, *r)?;
                e.store.mult_pt_sum(&e.ctx, &mut e.ex, &mut e.ops, &e.sh, m2, &terms, &values, 1 << 15, Payload::Real)?;
                let (out_top, a, b) = (m2.top(e.ctx.shape)?, m1.meta, m2.meta);
                let bad = diff(&ct_rows(e, m1, out_top)?, &ct_rows(e, m2, out_top)?);
                println!("[diag_gen] (c) {}-term sum at {l} limbs: words != PtStore::mult_pt_sum {bad} of {}; meta {:?} vs {:?}", g.len(), (2 * (out_top + 1)) << 16, (a.level, a.deg, a.nf), (b.level, b.deg, b.nf));
                assert_eq!(bad, 0);
                assert_eq!((a.level, a.deg, a.nf.to_bits()), (b.level, b.deg, b.nf.to_bits()));
            }
        }
        assert_eq!(e.dg.splits(&e.ex.compute)?, 0);
        Ok(())
    })
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <diag_gen_gpu test binary> --ignored --test-threads=1 --nocapture"]
fn f_device_ms_per_plaintext() -> Res<()> {
    let reps = 10;
    for bt in [1usize, 32, 128] {
        with_env(bt, 18, 128, |e| {
            let mut rng = Rng(0xf00d + bt as u64);
            let g = group(e, 1, 0, 7);
            let rows = g.iter().map(|nm| row(e, nm)).collect::<Res<Vec<_>>>()?;
            let r = e.dg.register(&e.ex.compute, &rows)?;
            for l in [18usize, 10] {
                let top = l - 1;
                let scale = e.ctx.scale.sf(top as i32)?;
                let level = level_from_top_limb(e.ctx.shape, top as u32)?;
                // the encoder alone: gather, encode_many, NTT of bt jobs
                let probe = (0..bt).map(|_| row(e, &random_name(e, &mut rng))).collect::<Res<Vec<_>>>()?;
                e.dg.probe(&e.ctx, &e.ex.compute, &probe, top, scale)?;
                let at = e.dg.shape().jobs;
                e.t0.record(&e.ex.compute)?;
                for _ in 0..reps {
                    e.dg.generate(&e.ctx, &e.ex.compute, at, bt, top, scale, 1)?;
                }
                e.t1.record(&e.ex.compute)?;
                e.t1.sync()?;
                let enc = e.t0.elapsed_ms(&e.t1)? as f64 / (reps * bt) as f64;
                // the fused sum: a 128-term group (encoder + products), chunks of bt
                let cts = (0..8).map(|_| random_ct(&e.pool, &e.ctx, &e.ex, &mut rng, level, top)).collect::<Res<Vec<_>>>()?;
                let terms: Vec<&Ct<'_, W>> = (0..g.len()).map(|j| &*cts[j % cts.len()]).collect();
                let mut o = e.pool.acquire(1, terms[0].meta)?;
                let om = o.get_mut().ok_or("shared")?;
                e.dg.sum(&e.ctx, &mut e.ex, &mut e.ops, &e.sh, om, &terms, r)?;
                e.t0.record(&e.ex.compute)?;
                for _ in 0..reps {
                    e.dg.sum(&e.ctx, &mut e.ex, &mut e.ops, &e.sh, om, &terms, r)?;
                }
                e.t1.record(&e.ex.compute)?;
                e.t1.sync()?;
                let sum = e.t0.elapsed_ms(&e.t1)? as f64 / (reps * g.len()) as f64;
                println!("[diag_gen] (f) B = {bt}, {l} limbs: encoder {enc:.4} ms per plaintext; fused sum {sum:.4} ms per term ({} terms, {reps} reps, CUDA events on the compute lane)", g.len());
            }
            Ok(())
        })?;
    }
    println!("[diag_gen] (f) for scale only (another binary, another card): the fork's device encoder on an A6000, 20 limbs, 0.119 ms (B = 32), 0.083 ms (B = 128) per plaintext");
    Ok(())
}
