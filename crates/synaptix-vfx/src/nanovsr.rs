//! NanoVSR (ECCV 2026, MIT): видео ×4, двунаправленная рекуррентная сеть на RepVGG-блоках (после
//! слияния — свёртки 3×3 + LeakyReLU 0,1), два pixel shuffle ×2 с PReLU, финальная свёртка + билинейная
//! база. Веса — `tools/nanovsr_convert.py` (слитые, ключи `feat`, `fwd.N`, `bwd.N`, `fusion`, `up1`,
//! `up2`, `last`).
//!
//! Поток кадров: копится окно `t + o` кадров; обратный проход идёт по всему окну от нуля, прямой —
//! непрерывно через окна (состояние не сбрасывается), отдаются первые `t` кадров, последние `o` —
//! начало следующего окна (у них будет «будущее»). Задержка — `t + o` кадров.

use std::collections::VecDeque;

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{pad16, Act, Conv, Gpu, In, Out, Res};
use crate::weights::Weights;
use crate::{Error, Frame, Result, Upscaler};

const LRELU: Act = Act::LeakyRelu(0.1);

struct Net {
    feat: Conv,
    fwd: Vec<Conv>,
    bwd: Vec<Conv>,
    fusion: Conv,
    up1: Conv,
    up2: Conv,
    last: Conv,
    c: usize,
    /// Наклоны PReLU после pixel shuffle (по одному на слой).
    a1: f32,
    a2: f32,
}

struct Slot {
    rgba: CudaSlice<u8>,
    feat: CudaSlice<f16>,
}

/// Буферы под размер кадра.
struct Bufs {
    w: usize,
    h: usize,
    x16: CudaSlice<f16>,
    a: CudaSlice<f16>,
    b: CudaSlice<f16>,
    fprop: CudaSlice<f16>,
    fnext: CudaSlice<f16>,
    fused: CudaSlice<f16>,
    u1: CudaSlice<f16>,
    u2: CudaSlice<f16>,
    out: CudaSlice<u8>,
    bw: Vec<CudaSlice<f16>>,
    host: cudarc::driver::PinnedHostSlice<u8>,
}

pub struct NanoVsr {
    g: std::sync::Arc<Gpu>,
    net: Net,
    t: usize,
    o: usize,
    pending: VecDeque<Slot>,
    pool: Vec<Slot>,
    bufs: Option<Bufs>,
    /// Прямое состояние есть (иначе — ноль: начало ролика, перемотка).
    fvalid: bool,
    name: String,
}

impl NanoVsr {
    pub fn load(g: std::sync::Arc<Gpu>, wts: &Weights) -> Result<Self> {
        let c = wts.meta_usize("num_feat")?;
        let nb = wts.meta_usize("num_blocks")?;
        let conv = |n: &str, cout: usize, splits: &[usize], k: usize, shuffle: usize| -> Result<Conv> {
            Conv::new(&g, wts.get(&format!("{n}.w"))?, Some(wts.get(&format!("{n}.b"))?), cout, k, splits, shuffle)
        };
        let feat = conv("feat", c, &[3], 3, 1)?;
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();
        for i in 0..nb {
            fwd.push(conv(&format!("fwd.{i}"), c, &[c], 3, 1)?);
            bwd.push(conv(&format!("bwd.{i}"), c, &[c], 3, 1)?);
        }
        let fusion = conv("fusion", c, &[c, c], 1, 1)?;
        let up1 = conv("up1", 4 * c, &[c], 3, 2)?;
        let up2_out = wts.shape("up2.w")?[0];
        let up2 = conv("up2", up2_out, &[c], 3, 2)?;
        let last_in = up2_out / 4;
        let last = conv("last", 3, &[last_in], 3, 1)?;
        let a1 = wts.get("up1.prelu")?[0];
        let a2 = wts.get("up2.prelu")?[0];
        let name = format!("NanoVSR-{}", wts.meta.get("name").cloned().unwrap_or_else(|| format!("{c}x{nb}")));
        let net = Net { feat, fwd, bwd, fusion, up1, up2, last, c, a1, a2 };
        Ok(Self { g, net, t: 8, o: 2, pending: VecDeque::new(), pool: Vec::new(), bufs: None, fvalid: false, name })
    }

    /// Окно: отдавать по `t` кадров, `o` кадров «будущего» сверху.
    pub fn with_window(mut self, t: usize, o: usize) -> Self {
        self.t = t.max(1);
        self.o = o;
        self.reset();
        self.bufs = None;
        self
    }

    fn ensure(&mut self, w: usize, h: usize) -> Result<()> {
        if self.bufs.as_ref().is_some_and(|b| b.w == w && b.h == h) {
            return Ok(());
        }
        self.reset();
        self.pool.clear();
        let g = &self.g;
        let cs = pad16(self.net.c);
        let px = w * h;
        let up2_cs = pad16(self.net.up2.cout / 4);
        let mut bw = Vec::new();
        for _ in 0..self.t + self.o {
            bw.push(g.alloc_f16(px * cs)?);
        }
        self.bufs = Some(Bufs {
            w,
            h,
            x16: g.alloc_f16(px * 16)?,
            a: g.alloc_f16(px * cs)?,
            b: g.alloc_f16(px * cs)?,
            fprop: g.alloc_f16(px * cs)?,
            fnext: g.alloc_f16(px * cs)?,
            fused: g.alloc_f16(px * cs)?,
            u1: g.alloc_f16(px * 4 * cs)?,
            u2: g.alloc_f16(px * 16 * up2_cs)?,
            out: g.alloc_u8(px * 16 * 4)?,
            bw,
            host: g.alloc_pinned(px * 16 * 4)?,
        });
        Ok(())
    }

    /// Блоки направления над `feat (+ prop)` → `dst`.
    fn blocks(g: &Gpu, convs: &[Conv], feat: &CudaSlice<f16>, prop: Option<&CudaSlice<f16>>, a: &mut CudaSlice<f16>, b: &mut CudaSlice<f16>, dst: &mut CudaSlice<f16>, w: usize, h: usize, cs: usize) -> Result<()> {
        let n = convs.len();
        for (i, c) in convs.iter().enumerate() {
            let last = i + 1 == n;
            // вход: feat(+prop) или прошлый блок (a/b по очереди)
            match (i, last) {
                (0, true) => g.run(c, In::sum(feat, prop), h, w, LRELU, Res::None, 1, Out::F16 { buf: dst, cs })?,
                (0, false) => g.run(c, In::sum(feat, prop), h, w, LRELU, Res::None, 1, Out::F16 { buf: a, cs })?,
                (i, true) if i % 2 == 1 => g.run(c, In::one(a), h, w, LRELU, Res::None, 1, Out::F16 { buf: dst, cs })?,
                (_, true) => g.run(c, In::one(b), h, w, LRELU, Res::None, 1, Out::F16 { buf: dst, cs })?,
                (i, false) if i % 2 == 1 => g.run(c, In::one(a), h, w, LRELU, Res::None, 1, Out::F16 { buf: b, cs })?,
                (_, false) => g.run(c, In::one(b), h, w, LRELU, Res::None, 1, Out::F16 { buf: a, cs })?,
            }
        }
        Ok(())
    }

    /// Окно набрано (или `all` — конец потока): обработать и отдать первые `t` (или все) кадры.
    fn run_window(&mut self, all: bool, emit: &mut dyn FnMut(Frame)) -> Result<()> {
        let n = self.pending.len();
        if n == 0 {
            return Ok(());
        }
        let out_n = if all { n } else { self.t.min(n) };
        let g = self.g.clone();
        let net = &self.net;
        let cs = pad16(net.c);
        let up2_cs = pad16(net.up2.cout / 4);
        let bufs = self.bufs.as_mut().ok_or_else(|| Error::Unsupported("нет буферов".into()))?;
        let (w, h) = (bufs.w, bufs.h);
        // обратный проход от нуля по окну
        for i in (0..n).rev() {
            let (head, tail) = bufs.bw.split_at_mut(i + 1);
            let prop = tail.first();
            Self::blocks(&g, &net.bwd, &self.pending[i].feat, prop, &mut bufs.a, &mut bufs.b, &mut head[i], w, h, cs)?;
        }
        for i in 0..out_n {
            let prop = self.fvalid.then_some(&bufs.fprop);
            Self::blocks(&g, &net.fwd, &self.pending[i].feat, prop, &mut bufs.a, &mut bufs.b, &mut bufs.fnext, w, h, cs)?;
            std::mem::swap(&mut bufs.fprop, &mut bufs.fnext);
            self.fvalid = true;
            g.run(&net.fusion, In::cat(&[&bufs.fprop, &bufs.bw[i]]), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut bufs.fused, cs })?;
            g.run(&net.up1, In::one(&bufs.fused), h, w, Act::PRelu(net.a1), Res::None, 2, Out::F16 { buf: &mut bufs.u1, cs })?;
            g.run(&net.up2, In::one(&bufs.u1), 2 * h, 2 * w, Act::PRelu(net.a2), Res::None, 2, Out::F16 { buf: &mut bufs.u2, cs: up2_cs })?;
            let src = &self.pending[i].rgba;
            g.run(&net.last, In::one(&bufs.u2), 4 * h, 4 * w, Act::None, Res::None, 1, Out::Rgba { buf: &mut bufs.out, base: Some((src, w, h)), nearest: false })?;
            let rgba = g.download(&bufs.out, &mut bufs.host, w * h * 16 * 4)?;
            emit(Frame { rgba, width: 4 * w as u32, height: 4 * h as u32 });
        }
        for _ in 0..out_n {
            if let Some(s) = self.pending.pop_front() {
                self.pool.push(s);
            }
        }
        Ok(())
    }
}

impl Upscaler for NanoVsr {
    fn name(&self) -> &str {
        &self.name
    }

    fn scale(&self) -> u32 {
        4
    }

    fn latency(&self) -> usize {
        self.t + self.o - 1
    }

    fn push(&mut self, rgba: &[u8], width: u32, height: u32, emit: &mut dyn FnMut(Frame)) -> Result<()> {
        let (w, h) = (width as usize, height as usize);
        if rgba.len() < w * h * 4 {
            return Err(Error::Unsupported("кадр меньше w×h×4".into()));
        }
        self.ensure(w, h)?;
        let g = self.g.clone();
        let cs = pad16(self.net.c);
        let mut slot = match self.pool.pop() {
            Some(s) => s,
            None => Slot { rgba: g.alloc_u8(w * h * 4)?, feat: g.alloc_f16(w * h * cs)? },
        };
        g.stream.memcpy_htod(&rgba[..w * h * 4], &mut slot.rgba).map_err(|e| Error::Cuda(format!("upload: {e:?}")))?;
        let bufs = self.bufs.as_mut().ok_or_else(|| Error::Unsupported("нет буферов".into()))?;
        g.rgba_to_nhwc(&slot.rgba, &mut bufs.x16, w * h, 16)?;
        g.run(&self.net.feat, In::one(&bufs.x16), h, w, LRELU, Res::None, 1, Out::F16 { buf: &mut slot.feat, cs })?;
        self.pending.push_back(slot);
        if self.pending.len() >= self.t + self.o {
            self.run_window(false, emit)?;
        }
        Ok(())
    }

    fn flush(&mut self, emit: &mut dyn FnMut(Frame)) -> Result<()> {
        self.run_window(true, emit)
    }

    fn reset(&mut self) {
        while let Some(s) = self.pending.pop_front() {
            self.pool.push(s);
        }
        self.fvalid = false;
    }
}
