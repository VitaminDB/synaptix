//! SPAN (Swift Parameter-free Attention Network, NTIRE 2024 ESR): покадровая модель. Conv3XC после
//! слияния — свёртка 3×3; блок SPAB: c1 → SiLU → c2 → SiLU → c3, внимание `(c3 + x)·(σ(c3) − ½)`;
//! выход — 1×1 над [признаки, conv_2(b6), b1, c1 шестого блока] и pixel shuffle ×r.
//! Веса — `tools/span_convert.py` (`conv_1`, `block{1..6}.c{1..3}`, `conv_cat`, `conv_2`, `up`;
//! метаданные `feat`, `scale`, `norm`).

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{pad16, Act, Conv, Gpu, In, Out, Res};
use crate::weights::Weights;
use crate::{Error, Frame, Result, Upscaler};

const RGB_MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];

struct Bufs {
    w: usize,
    h: usize,
    src: CudaSlice<u8>,
    x16: CudaSlice<f16>,
    f0: CudaSlice<f16>,
    b1: CudaSlice<f16>,
    a: CudaSlice<f16>,
    b: CudaSlice<f16>,
    o1: CudaSlice<f16>,
    o2: CudaSlice<f16>,
    out: CudaSlice<u8>,
    host: cudarc::driver::PinnedHostSlice<u8>,
}

pub struct Span {
    g: std::sync::Arc<Gpu>,
    conv1: Conv,
    blocks: Vec<[Conv; 3]>,
    conv_cat: Conv,
    conv2: Conv,
    up: Conv,
    c: usize,
    scale: usize,
    norm: bool,
    bufs: Option<Bufs>,
    name: String,
}

impl Span {
    pub fn load(g: std::sync::Arc<Gpu>, wts: &Weights) -> Result<Self> {
        let c = wts.meta_usize("feat")?;
        let scale = wts.meta_usize("scale")?;
        let norm = wts.meta.get("norm").is_some_and(|v| v == "1");
        let conv = |n: &str, cout: usize, splits: &[usize], k: usize, sh: usize| -> Result<Conv> {
            Conv::new(&g, wts.get(&format!("{n}.w"))?, Some(wts.get(&format!("{n}.b"))?), cout, k, splits, sh)
        };
        let conv1 = conv("conv_1", c, &[3], 3, 1)?;
        let mut blocks = Vec::new();
        for i in 1..=6 {
            blocks.push([conv(&format!("block{i}.c1"), c, &[c], 3, 1)?, conv(&format!("block{i}.c2"), c, &[c], 3, 1)?, conv(&format!("block{i}.c3"), c, &[c], 3, 1)?]);
        }
        let conv_cat = conv("conv_cat", c, &[c, c, c, c], 1, 1)?;
        let conv2 = conv("conv_2", c, &[c], 3, 1)?;
        let up = conv("up", 3 * scale * scale, &[c], 3, scale)?;
        let name = wts.meta.get("name").cloned().unwrap_or_else(|| format!("SPAN ×{scale}"));
        Ok(Self { g, conv1, blocks, conv_cat, conv2, up, c, scale, norm, bufs: None, name })
    }

    fn ensure(&mut self, w: usize, h: usize) -> Result<()> {
        if self.bufs.as_ref().is_some_and(|b| b.w == w && b.h == h) {
            return Ok(());
        }
        let g = &self.g;
        let px = w * h;
        let cs = pad16(self.c);
        let r2 = self.scale * self.scale;
        self.bufs = Some(Bufs {
            w,
            h,
            src: g.alloc_u8(px * 4)?,
            x16: g.alloc_f16(px * 16)?,
            f0: g.alloc_f16(px * cs)?,
            b1: g.alloc_f16(px * cs)?,
            a: g.alloc_f16(px * cs)?,
            b: g.alloc_f16(px * cs)?,
            o1: g.alloc_f16(px * cs)?,
            o2: g.alloc_f16(px * cs)?,
            out: g.alloc_u8(px * r2 * 4)?,
            host: g.alloc_pinned(px * r2 * 4)?,
        });
        Ok(())
    }
}

/// Блок SPAB: `x` → `dst`; c1 и c2 — с SiLU в `o1`/`o2`. В исходнике SiLU(inplace=True) портит
/// выход c1 на месте, поэтому и в конкатенацию шестого блока идёт уже активированный c1.
#[allow(clippy::too_many_arguments)]
fn spab(g: &Gpu, cv: &[Conv; 3], x: &CudaSlice<f16>, o1: &mut CudaSlice<f16>, o2: &mut CudaSlice<f16>, dst: &mut CudaSlice<f16>, w: usize, h: usize, cs: usize) -> Result<()> {
    g.run(&cv[0], In::one(x), h, w, Act::Silu, Res::None, 1, Out::F16 { buf: o1, cs })?;
    g.run(&cv[1], In::one(o1), h, w, Act::Silu, Res::None, 1, Out::F16 { buf: o2, cs })?;
    g.run(&cv[2], In::one(o2), h, w, Act::SpabAttention, Res::Before(x), 1, Out::F16 { buf: dst, cs })
}

impl Upscaler for Span {
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
        let cs = pad16(self.c);
        let b = self.bufs.as_mut().unwrap();
        g.stream.memcpy_htod(&rgba[..w * h * 4], &mut b.src).map_err(|e| Error::Cuda(format!("upload: {e:?}")))?;
        if self.norm {
            // (x − mean)·255 при x = v/255 → v − mean·255
            let off = RGB_MEAN.map(|m| -m * 255.0);
            g.rgba_to_nhwc_norm(&b.src, &mut b.x16, w * h, 16, 1.0, off)?;
        } else {
            g.rgba_to_nhwc(&b.src, &mut b.x16, w * h, 16)?;
        }
        g.run(&self.conv1, In::one(&b.x16), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.f0, cs })?;
        spab(&g, &self.blocks[0], &b.f0, &mut b.o1, &mut b.o2, &mut b.b1, w, h, cs)?;
        // блоки 2..5 — по очереди через a/b; шестой — в b, его c1 остаётся в o1
        spab(&g, &self.blocks[1], &b.b1, &mut b.o1, &mut b.o2, &mut b.a, w, h, cs)?;
        spab(&g, &self.blocks[2], &b.a, &mut b.o1, &mut b.o2, &mut b.b, w, h, cs)?;
        spab(&g, &self.blocks[3], &b.b, &mut b.o1, &mut b.o2, &mut b.a, w, h, cs)?;
        spab(&g, &self.blocks[4], &b.a, &mut b.o1, &mut b.o2, &mut b.b, w, h, cs)?;
        spab(&g, &self.blocks[5], &b.b, &mut b.o1, &mut b.o2, &mut b.a, w, h, cs)?;
        g.run(&self.conv2, In::one(&b.a), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.b, cs })?;
        // a свободен после conv_2 — в него конкатенация
        g.run(&self.conv_cat, In::cat(&[&b.f0, &b.b, &b.b1, &b.o1]), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut b.a, cs })?;
        g.run(&self.up, In::one(&b.a), h, w, Act::None, Res::None, self.scale, Out::Rgba { buf: &mut b.out, base: None, nearest: false })?;
        let n = w * h * self.scale * self.scale * 4;
        let rgba = g.download(&b.out, &mut b.host, n)?;
        emit(Frame { rgba, width: (w * self.scale) as u32, height: (h * self.scale) as u32 });
        Ok(())
    }

    fn reset(&mut self) {}
}
