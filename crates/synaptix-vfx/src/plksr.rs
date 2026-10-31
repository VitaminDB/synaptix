//! RealPLKSR (neosr, «Partial Large Kernel CNNs for Efficient Super-Resolution»): покадровая модель.
//! Блок: [LayerNorm] → DCCM (3×3 → Mish → 3×3) → большая свёртка K×K на первых `pdim` каналах →
//! EA `x·σ(conv3×3(x))` → 1×1 → [GroupNorm по кадру] + вход. Выход — свёртка в 3·r² каналов + вход
//! (повтор по r×r) и pixel shuffle. DySample не поддерживается.
//! Веса — `tools/plksr_convert.py` (`first`, `b{i}.{cm1,cm2,lk,ea,refine}`, `b{i}.norm.{g,b}` или
//! `b{i}.ln.{g,b}`, `last`; метаданные `dim`, `blocks`, `k`, `pdim`, `ea`, `groups`, `layer_norm`, `scale`).

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{pad16, Act, Conv, Gpu, In, Out, Res, Src};
use crate::weights::Weights;
use crate::{Error, Frame, Result, Upscaler};

struct Block {
    ln: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    cm1: Conv,
    cm2: Conv,
    lk: Conv,
    ea: Option<Conv>,
    refine: Conv,
    gn: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
}

struct Bufs {
    w: usize,
    h: usize,
    src: CudaSlice<u8>,
    x16: CudaSlice<f16>,
    x: CudaSlice<f16>,
    y: CudaSlice<f16>,
    t0: CudaSlice<f16>,
    t1: CudaSlice<f16>,
    t2: CudaSlice<f16>,
    t3: CudaSlice<f16>,
    t4: CudaSlice<f16>,
    stats: CudaSlice<f32>,
    out: CudaSlice<u8>,
    host: cudarc::driver::PinnedHostSlice<u8>,
}

pub struct RealPlksr {
    g: std::sync::Arc<Gpu>,
    first: Conv,
    blocks: Vec<Block>,
    last: Conv,
    dim: usize,
    pdim: usize,
    groups: usize,
    scale: usize,
    bufs: Option<Bufs>,
    name: String,
}

impl RealPlksr {
    pub fn load(g: std::sync::Arc<Gpu>, wts: &Weights) -> Result<Self> {
        let dim = wts.meta_usize("dim")?;
        let nb = wts.meta_usize("blocks")?;
        let k = wts.meta_usize("k")?;
        let pdim = wts.meta_usize("pdim")?;
        let scale = wts.meta_usize("scale")?;
        let groups = wts.meta_usize("groups").unwrap_or(4);
        let ea = wts.meta.get("ea").is_none_or(|v| v == "1");
        let layer_norm = wts.meta.get("layer_norm").is_some_and(|v| v == "1");
        let conv = |n: &str, cout: usize, splits: &[usize], k: usize, sh: usize| -> Result<Conv> {
            Conv::new(&g, wts.get(&format!("{n}.w"))?, Some(wts.get(&format!("{n}.b"))?), cout, k, splits, sh)
        };
        let pair = |n: &str| -> Result<(CudaSlice<f32>, CudaSlice<f32>)> { Ok((g.upload_f32(wts.get(&format!("{n}.g"))?)?, g.upload_f32(wts.get(&format!("{n}.b"))?)?)) };
        let first = conv("first", dim, &[3], 3, 1)?;
        let mut blocks = Vec::new();
        for i in 0..nb {
            let p = format!("b{i}");
            blocks.push(Block {
                ln: if layer_norm { Some(pair(&format!("{p}.ln"))?) } else { None },
                cm1: conv(&format!("{p}.cm1"), 2 * dim, &[dim], 3, 1)?,
                cm2: conv(&format!("{p}.cm2"), dim, &[2 * dim], 3, 1)?,
                lk: conv(&format!("{p}.lk"), pdim, &[pdim], k, 1)?,
                ea: if ea { Some(conv(&format!("{p}.ea"), dim, &[dim], 3, 1)?) } else { None },
                refine: conv(&format!("{p}.refine"), dim, &[dim], 1, 1)?,
                gn: if layer_norm { None } else { Some(pair(&format!("{p}.norm"))?) },
            });
        }
        let last = conv("last", 3 * scale * scale, &[dim], 3, scale)?;
        let name = wts.meta.get("name").cloned().unwrap_or_else(|| format!("RealPLKSR ×{scale}"));
        Ok(Self { g, first, blocks, last, dim, pdim, groups, scale, bufs: None, name })
    }

    fn ensure(&mut self, w: usize, h: usize) -> Result<()> {
        if self.bufs.as_ref().is_some_and(|b| b.w == w && b.h == h) {
            return Ok(());
        }
        let g = &self.g;
        let px = w * h;
        let cs = pad16(self.dim);
        let r2 = self.scale * self.scale;
        self.bufs = Some(Bufs {
            w,
            h,
            src: g.alloc_u8(px * 4)?,
            x16: g.alloc_f16(px * 16)?,
            x: g.alloc_f16(px * cs)?,
            y: g.alloc_f16(px * cs)?,
            t0: g.alloc_f16(px * cs)?,
            t1: g.alloc_f16(px * pad16(2 * self.dim))?,
            t2: g.alloc_f16(px * cs)?,
            t3: g.alloc_f16(px * cs)?,
            t4: g.alloc_f16(px * cs)?,
            stats: g.upload_f32(&vec![0.0; self.groups * 2])?,
            out: g.alloc_u8(px * r2 * 4)?,
            host: g.alloc_pinned(px * r2 * 4)?,
        });
        Ok(())
    }
}

impl Upscaler for RealPlksr {
    fn name(&self) -> &str {
        &self.name
    }

    fn scale(&self) -> u32 {
        self.scale as u32
    }

    fn push(&mut self, rgba: &[u8], width: u32, height: u32, emit: &mut dyn FnMut(Frame)) -> Result<()> {
        let (w, h) = (width as usize, height as usize);
        if rgba.len() < w * h * 4 {
            return Err(Error::Unsupported("кадр меньше w×h×4".into()));
        }
        self.ensure(w, h)?;
        let g = self.g.clone();
        let (dim, pdim, groups) = (self.dim, self.pdim, self.groups);
        let cs = pad16(dim);
        let cs2 = pad16(2 * dim);
        let px = w * h;
        let b = self.bufs.as_mut().unwrap();
        g.stream.memcpy_htod(&rgba[..px * 4], &mut b.src).map_err(|e| Error::Cuda(format!("upload: {e:?}")))?;
        g.rgba_to_nhwc(&b.src, &mut b.x16, px, 16)?;
        g.run(&self.first, In::one(&b.x16), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.x, cs })?;
        for blk in &self.blocks {
            let inp: &CudaSlice<f16> = match &blk.ln {
                Some((lg, lb)) => {
                    g.layer_norm(&b.x, &mut b.t0, lg, lb, px, cs, dim, 1e-6)?;
                    &b.t0
                }
                None => &b.x,
            };
            g.run(&blk.cm1, In::one(inp), h, w, Act::Mish, Res::None, 1, Out::F16 { buf: &mut b.t1, cs: cs2 })?;
            g.run(&blk.cm2, In::one(&b.t1), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.t2, cs })?;
            // большая свёртка на первых pdim каналах t2 → те же каналы t3, остальные — копией
            g.run(&blk.lk, In::srcs(vec![Src { buf: &b.t2, off: 0, ps: cs }]), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.t3, cs })?;
            g.copy_channels(&b.t2, &mut b.t3, px, cs, pdim, dim)?;
            let att: &CudaSlice<f16> = match &blk.ea {
                Some(ea) => {
                    g.run(ea, In::one(&b.t3), h, w, Act::Gate, Res::Before(&b.t3), 1, Out::F16 { buf: &mut b.t4, cs })?;
                    &b.t4
                }
                None => &b.t3,
            };
            match &blk.gn {
                Some((ng, nbias)) => {
                    g.run(&blk.refine, In::one(att), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.t2, cs })?;
                    g.group_norm(&b.t2, &b.x, &mut b.y, &mut b.stats, ng, nbias, px, cs, dim, groups, 1e-5)?;
                }
                None => g.run(&blk.refine, In::one(att), h, w, Act::None, Res::After(&b.x), 1, Out::F16 { buf: &mut b.y, cs })?,
            }
            std::mem::swap(&mut b.x, &mut b.y);
        }
        g.run(&self.last, In::one(&b.x), h, w, Act::None, Res::None, self.scale, Out::Rgba { buf: &mut b.out, base: Some((&b.src, w, h)), nearest: true })?;
        let n = w * h * self.scale * self.scale * 4;
        let rgba = g.download(&b.out, &mut b.host, n)?;
        emit(Frame { rgba, width: (w * self.scale) as u32, height: (h * self.scale) as u32 });
        Ok(())
    }

    fn reset(&mut self) {}
}
