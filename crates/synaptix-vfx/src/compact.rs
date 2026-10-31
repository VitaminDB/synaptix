//! SRVGGNetCompact (Real-ESRGAN): свёртка 3→F + PReLU по каналам, N × (3×3 F→F + PReLU), свёртка в
//! 3·r² каналов, pixel shuffle + ближайший пиксель входа. Аниме: AnimeJaNai V3 (Compact,
//! UltraCompact, SuperUltraCompact; ×2) и realesr-animevideov3 (×4). Веса — `tools/compact_convert.py`
//! (`conv{j}.{w,b}`, `act{j}`; метаданные `feat`, `convs`, `scale`).

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{pad16, Act, Conv, Gpu, In, Out, Res};
use crate::weights::Weights;
use crate::{Error, Frame, Result, Upscaler};

struct Bufs {
    w: usize,
    h: usize,
    src: CudaSlice<u8>,
    x16: CudaSlice<f16>,
    a: CudaSlice<f16>,
    b: CudaSlice<f16>,
    out: CudaSlice<u8>,
    host: cudarc::driver::PinnedHostSlice<u8>,
}

pub struct Compact {
    g: std::sync::Arc<Gpu>,
    convs: Vec<Conv>,
    feat: usize,
    scale: usize,
    bufs: Option<Bufs>,
    name: String,
}

impl Compact {
    pub fn load(g: std::sync::Arc<Gpu>, wts: &Weights) -> Result<Self> {
        let feat = wts.meta_usize("feat")?;
        let n = wts.meta_usize("convs")?;
        let scale = wts.meta_usize("scale")?;
        let mut convs = Vec::new();
        for j in 0..n + 2 {
            let (cout, cin, sh) = match j {
                0 => (feat, 3, 1),
                j if j == n + 1 => (3 * scale * scale, feat, scale),
                _ => (feat, feat, 1),
            };
            let mut c = Conv::new(&g, wts.get(&format!("conv{j}.w"))?, Some(wts.get(&format!("conv{j}.b"))?), cout, 3, &[cin], sh)?;
            if j <= n {
                c = c.with_alpha(&g, wts.get(&format!("act{j}"))?)?;
            }
            convs.push(c);
        }
        let name = wts.meta.get("name").cloned().unwrap_or_else(|| format!("Compact ×{scale}"));
        Ok(Self { g, convs, feat, scale, bufs: None, name })
    }

    fn ensure(&mut self, w: usize, h: usize) -> Result<()> {
        if self.bufs.as_ref().is_some_and(|b| b.w == w && b.h == h) {
            return Ok(());
        }
        let g = &self.g;
        let px = w * h;
        let cs = pad16(self.feat);
        let r2 = self.scale * self.scale;
        self.bufs = Some(Bufs {
            w,
            h,
            src: g.alloc_u8(px * 4)?,
            x16: g.alloc_f16(px * 16)?,
            a: g.alloc_f16(px * cs)?,
            b: g.alloc_f16(px * cs)?,
            out: g.alloc_u8(px * r2 * 4)?,
            host: g.alloc_pinned(px * r2 * 4)?,
        });
        Ok(())
    }
}

impl Upscaler for Compact {
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
        let cs = pad16(self.feat);
        let px = w * h;
        let b = self.bufs.as_mut().unwrap();
        g.stream.memcpy_htod(&rgba[..px * 4], &mut b.src).map_err(|e| Error::Cuda(format!("upload: {e:?}")))?;
        g.rgba_to_nhwc(&b.src, &mut b.x16, px, 16)?;
        let n = self.convs.len();
        g.run(&self.convs[0], In::one(&b.x16), h, w, Act::PReluCh, Res::None, 1, Out::F16 { buf: &mut b.a, cs })?;
        for (j, c) in self.convs.iter().enumerate().take(n - 1).skip(1) {
            if j % 2 == 1 {
                g.run(c, In::one(&b.a), h, w, Act::PReluCh, Res::None, 1, Out::F16 { buf: &mut b.b, cs })?;
            } else {
                g.run(c, In::one(&b.b), h, w, Act::PReluCh, Res::None, 1, Out::F16 { buf: &mut b.a, cs })?;
            }
        }
        let last_in = if (n - 2) % 2 == 1 { &b.b } else { &b.a };
        g.run(&self.convs[n - 1], In::one(last_in), h, w, Act::None, Res::None, self.scale, Out::Rgba { buf: &mut b.out, base: Some((&b.src, w, h)), nearest: true })?;
        let total = px * self.scale * self.scale * 4;
        let rgba = g.download(&b.out, &mut b.host, total)?;
        emit(Frame { rgba, width: (w * self.scale) as u32, height: (h * self.scale) as u32 });
        Ok(())
    }

    fn reset(&mut self) {}
}
