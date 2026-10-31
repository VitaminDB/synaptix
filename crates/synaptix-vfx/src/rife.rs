//! RIFE (Practical-RIFE, IFNet 4.25 / 4.25.lite / 4.26; MIT): промежуточный кадр между двумя.
//! Head даёт 4 канала признаков на кадр; пять IFBlock от грубого к точному уточняют двусторонний
//! поток (4 канала), маску и 8 каналов признаков; итог — смесь кадров, сдвинутых потоком, по маске.
//! Всё в f32 (NCHW): ядра `rife.cu`. Веса — `tools/rife_convert.py` (ключи как в flownet.pkl без
//! `module.`; метаданные `version`, `modulo`, `scales`).
//!
//! Поток: [`Rife::push`] — новый кадр (признаки считаются сразу, прошлый становится первым);
//! [`Rife::interpolate`] — кадр в момент `t ∈ (0, 1)` между двумя последними.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PinnedHostSlice, PushKernelArg};

use crate::gpu::Gpu;
use crate::weights::Weights;
use crate::{Error, Result};

fn cuda<E: std::fmt::Debug>(what: &'static str) -> impl FnOnce(E) -> Error {
    move |e| Error::Cuda(format!("{what}: {e:?}"))
}

struct K {
    conv: CudaFunction,
    deconv: CudaFunction,
    resize: CudaFunction,
    warp: CudaFunction,
    fill: CudaFunction,
    to_chw: CudaFunction,
    blend: CudaFunction,
    diff: CudaFunction,
    _m: Arc<CudaModule>,
}

/// Размеры под кадр.
struct State {
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    rgba: [CudaSlice<u8>; 2],
    img: [CudaSlice<f32>; 2],
    feat: [CudaSlice<f32>; 2],
    /// Индекс последнего кадра в парах выше; кадров получено.
    cur: usize,
    have: u32,
    flow: CudaSlice<f32>,
    /// mask (1) + feat (8) из последнего блока, полное разрешение.
    mf: CudaSlice<f32>,
    /// warped_img0, warped_img1, wf0, wf1 — 14 каналов, полное разрешение.
    warped: CudaSlice<f32>,
    xlow: CudaSlice<f32>,
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    out: CudaSlice<u8>,
    host: PinnedHostSlice<u8>,
    acc: CudaSlice<f32>,
}

pub struct Rife {
    g: Arc<Gpu>,
    k: K,
    w: HashMap<String, (Vec<usize>, CudaSlice<f32>)>,
    /// Каналы блоков (c) и масштабы (делители разрешения).
    chans: [usize; 5],
    scales: [f32; 5],
    modulo: usize,
    /// Делитель обработки потока: 1 — как есть, 0,5 — для 4K (блоки на вдвое меньшем разрешении).
    flow_scale: f32,
    st: Option<State>,
    name: String,
}

impl Rife {
    pub fn load(g: Arc<Gpu>, wts: &Weights) -> Result<Self> {
        let module = synaptix_kernels_cuda::kernels::compile::compile_module_with_opts(&g.ctx, include_str!("rife.cu"), "synaptix-vfx/rife.cu", &[], None)
            .map_err(|e| Error::Cuda(e.to_string()))?;
        let f = |n: &str| module.load_function(n).map_err(|e| Error::Cuda(format!("{n}: {e:?}")));
        let k = K {
            conv: f("conv_f32")?,
            deconv: f("deconv4_f32")?,
            resize: f("resize_f32")?,
            warp: f("warp_f32")?,
            fill: f("fill_f32")?,
            to_chw: f("rgba_to_chw")?,
            blend: f("blend_rgba")?,
            diff: f("frame_diff")?,
            _m: module.clone(),
        };
        for func in [&k.conv, &k.deconv] {
            func.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 99 * 1024).map_err(cuda("smem"))?;
        }
        let mut w = HashMap::new();
        for name in wts.names() {
            if name.starts_with("block") || name.starts_with("encode.") {
                w.insert(name.to_string(), (wts.shape(name)?.to_vec(), g.upload_f32(wts.get(name)?)?));
            }
        }
        let mut chans = [0usize; 5];
        for (i, c) in chans.iter_mut().enumerate() {
            *c = w.get(&format!("block{i}.convblock.0.conv.bias")).map(|t| t.0[0]).ok_or_else(|| Error::Weights(format!("нет block{i}")))?;
        }
        let scales: Vec<f32> = wts.meta.get("scales").map(String::as_str).unwrap_or("16,8,4,2,1").split(',').filter_map(|s| s.parse().ok()).collect();
        let scales: [f32; 5] = scales.try_into().map_err(|_| Error::Weights("scales: нужно 5 значений".into()))?;
        let modulo = wts.meta_usize("modulo").unwrap_or(64);
        let name = wts.meta.get("name").cloned().unwrap_or_else(|| "RIFE".into());
        Ok(Self { g, k, w, chans, scales, modulo, flow_scale: 1.0, st: None, name })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Обработка потока на меньшем разрешении (0,5 — для 4K, как scale у vs-rife).
    pub fn set_flow_scale(&mut self, s: f32) {
        if (s - self.flow_scale).abs() > 1e-6 {
            self.flow_scale = s;
            self.st = None;
        }
    }

    fn wt(&self, n: &str) -> Result<&CudaSlice<f32>> {
        self.w.get(n).map(|t| &t.1).ok_or_else(|| Error::Weights(format!("нет {n}")))
    }

    fn shape(&self, n: &str) -> Result<&[usize]> {
        self.w.get(n).map(|t| t.0.as_slice()).ok_or_else(|| Error::Weights(format!("нет {n}")))
    }

    fn ensure(&mut self, w: usize, h: usize) -> Result<()> {
        if self.st.as_ref().is_some_and(|s| s.w == w && s.h == h) {
            return Ok(());
        }
        let m = ((self.modulo as f32 / self.flow_scale) as usize).max(self.modulo);
        let (pw, ph) = (w.div_ceil(m) * m, h.div_ceil(m) * m);
        let px = pw * ph;
        let g = &self.g;
        let f32b = |n: usize| -> Result<CudaSlice<f32>> { g.stream.alloc_zeros::<f32>(n.max(1)).map_err(cuda("alloc")) };
        // буферы свёрток: наибольшее «каналы × пиксели» по блокам и Head
        // вход и выход блоков — на разрешении самого точного блока
        let smin = self.scales.iter().copied().fold(f32::MAX, f32::min) / self.flow_scale;
        let low = ((px as f32) / (smin * smin)).ceil() as usize;
        let mut big = 16 * px / 4;
        for (i, &c) in self.chans.iter().enumerate() {
            let s = self.scales[i] / self.flow_scale;
            let p2 = (px as f32 / (4.0 * s * s)) as usize;
            big = big.max(c / 2 * p2).max(c * p2 / 4);
        }
        self.st = Some(State {
            w,
            h,
            pw,
            ph,
            rgba: [g.alloc_u8(w * h * 4)?, g.alloc_u8(w * h * 4)?],
            img: [f32b(3 * px)?, f32b(3 * px)?],
            feat: [f32b(4 * px)?, f32b(4 * px)?],
            cur: 0,
            have: 0,
            flow: f32b(4 * px)?,
            mf: f32b(9 * px)?,
            warped: f32b(14 * px)?,
            xlow: f32b(28 * low)?,
            a: f32b(big)?,
            b: f32b(big)?,
            tmp: f32b(13 * low)?,
            out: g.alloc_u8(w * h * 4)?,
            host: g.alloc_pinned(w * h * 4)?,
            acc: f32b(1)?,
        });
        Ok(())
    }

    fn lc(n: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// Свёртка: `name` (weight [co,ci,k,k]), вход [ci,h,w] → выход [co,ho,wo].
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, name: &str, beta: Option<&str>, res: Option<&CudaSlice<f32>>, inp: &CudaSlice<f32>, h: usize, w: usize, out: &mut CudaSlice<f32>, stride: usize, act: bool) -> Result<(usize, usize)> {
        let sh = self.shape(&format!("{name}.weight"))?;
        let (co, ci, k) = (sh[0], sh[1], sh[2]);
        let pad = k / 2;
        let (ho, wo) = ((h + 2 * pad - k) / stride + 1, (w + 2 * pad - k) / stride + 1);
        let cfg = LaunchConfig { grid_dim: (((ho * wo) as u32).div_ceil(256), 1, (co as u32).div_ceil(8)), block_dim: (256, 1, 1), shared_mem_bytes: (8 * ci * k * k * 4) as u32 };
        let wt = self.wt(&format!("{name}.weight"))?;
        let bias = self.wt(&format!("{name}.bias"))?;
        let beta_p = match beta {
            Some(b) => self.g.ptr(self.wt(b)?),
            None => 0,
        };
        let res_p = res.map(|r| self.g.ptr(r)).unwrap_or(0);
        let (ci_, h_, w_, co_, ho_, wo_, k_, s_, p_) = (ci as i32, h as i32, w as i32, co as i32, ho as i32, wo as i32, k as i32, stride as i32, pad as i32);
        let (slope, a) = (0.2f32, i32::from(act));
        let mut b = self.g.stream.launch_builder(&self.k.conv);
        b.arg(inp).arg(&ci_).arg(&h_).arg(&w_).arg(wt).arg(bias).arg(&beta_p).arg(&res_p).arg(out).arg(&co_).arg(&ho_).arg(&wo_).arg(&k_).arg(&s_).arg(&p_).arg(&slope).arg(&a);
        unsafe { b.launch(cfg) }.map_err(cuda("conv_f32"))?;
        Ok((ho, wo))
    }

    fn deconv(&self, name: &str, inp: &CudaSlice<f32>, h: usize, w: usize, out: &mut CudaSlice<f32>, shuffle: bool) -> Result<()> {
        let sh = self.shape(&format!("{name}.weight"))?;
        let (ci, co) = (sh[0], sh[1]);
        let cfg = LaunchConfig { grid_dim: (((4 * h * w) as u32).div_ceil(256), 1, (co as u32).div_ceil(4)), block_dim: (256, 1, 1), shared_mem_bytes: (ci * 64 * 4) as u32 };
        let (ci_, h_, w_, co_, s_) = (ci as i32, h as i32, w as i32, co as i32, i32::from(shuffle));
        let mut b = self.g.stream.launch_builder(&self.k.deconv);
        b.arg(inp).arg(&ci_).arg(&h_).arg(&w_).arg(self.wt(&format!("{name}.weight"))?).arg(self.wt(&format!("{name}.bias"))?).arg(out).arg(&co_).arg(&s_);
        unsafe { b.launch(cfg) }.map_err(cuda("deconv4_f32"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn resize(&self, src: &CudaSlice<f32>, hs: usize, ws: usize, c0: usize, nc: usize, dst: &mut CudaSlice<f32>, hd: usize, wd: usize, dc: usize, mul: f32, add: bool) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (((hd * wd) as u32).div_ceil(256), 1, nc as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (hs_, ws_, c0_, nc_, hd_, wd_, dc_, add_) = (hs as i32, ws as i32, c0 as i32, nc as i32, hd as i32, wd as i32, dc as i32, i32::from(add));
        let mut b = self.g.stream.launch_builder(&self.k.resize);
        b.arg(src).arg(&hs_).arg(&ws_).arg(&c0_).arg(&nc_).arg(dst).arg(&hd_).arg(&wd_).arg(&dc_).arg(&mul).arg(&add_);
        unsafe { b.launch(cfg) }.map_err(cuda("resize_f32"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn warp(&self, src: &CudaSlice<f32>, nc: usize, flow: &CudaSlice<f32>, fo: usize, dst: &mut CudaSlice<f32>, dc: usize, h: usize, w: usize) -> Result<()> {
        let (nc_, fo_, dc_, h_, w_) = (nc as i32, fo as i32, dc as i32, h as i32, w as i32);
        let mut b = self.g.stream.launch_builder(&self.k.warp);
        b.arg(src).arg(&nc_).arg(flow).arg(&fo_).arg(dst).arg(&dc_).arg(&h_).arg(&w_);
        unsafe { b.launch(Self::lc(h * w)) }.map_err(cuda("warp_f32"))?;
        Ok(())
    }

    /// Head: признаки кадра [4, ph, pw] из [3, ph, pw].
    fn encode(&self, img: &CudaSlice<f32>, feat: &mut CudaSlice<f32>, a: &mut CudaSlice<f32>, b: &mut CudaSlice<f32>, ph: usize, pw: usize) -> Result<()> {
        let (h2, w2) = self.conv("encode.cnn0", None, None, img, ph, pw, a, 2, true)?;
        self.conv("encode.cnn1", None, None, a, h2, w2, b, 1, true)?;
        self.conv("encode.cnn2", None, None, b, h2, w2, a, 1, true)?;
        self.deconv("encode.cnn3", a, h2, w2, feat, false)
    }

    /// Новый кадр RGBA8: признаки сразу; прошлый становится первым кадром пары.
    pub fn push(&mut self, rgba: &[u8], w: usize, h: usize) -> Result<()> {
        if rgba.len() < w * h * 4 {
            return Err(Error::Unsupported("кадр меньше w×h×4".into()));
        }
        self.ensure(w, h)?;
        let mut st = self.st.take().unwrap();
        let i = if st.have == 0 { 0 } else { 1 - st.cur };
        let r = (|| -> Result<()> {
            self.g.stream.memcpy_htod(&rgba[..w * h * 4], &mut st.rgba[i]).map_err(cuda("upload"))?;
            let (pw_, ph_, w_, h_) = (st.pw as i32, st.ph as i32, w as i32, h as i32);
            let mut b = self.g.stream.launch_builder(&self.k.to_chw);
            b.arg(&st.rgba[i]).arg(&w_).arg(&h_).arg(&mut st.img[i]).arg(&pw_).arg(&ph_);
            unsafe { b.launch(Self::lc(st.pw * st.ph)) }.map_err(cuda("rgba_to_chw"))?;
            let (img, feat) = (&st.img[i], &mut st.feat[i]);
            self.encode(img, feat, &mut st.a, &mut st.b, st.ph, st.pw)
        })();
        st.cur = i;
        st.have = (st.have + 1).min(2);
        self.st = Some(st);
        r
    }

    /// Есть пара кадров.
    pub fn ready(&self) -> bool {
        self.st.as_ref().is_some_and(|s| s.have >= 2)
    }

    /// Средняя разница яркости двух последних кадров (0..255) — смена сцены, если велика.
    pub fn scene_diff(&mut self) -> Result<f32> {
        let st = self.st.as_mut().ok_or_else(|| Error::Unsupported("нет кадров".into()))?;
        if st.have < 2 {
            return Ok(0.0);
        }
        self.g.stream.memset_zeros(&mut st.acc).map_err(cuda("memset"))?;
        let step = 8usize;
        let (gw, gh) = (st.w.div_ceil(step), st.h.div_ceil(step));
        let cfg = LaunchConfig { grid_dim: ((gw as u32).div_ceil(128), gh as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (w_, h_, s_) = (st.w as i32, st.h as i32, step as i32);
        let mut b = self.g.stream.launch_builder(&self.k.diff);
        b.arg(&st.rgba[0]).arg(&st.rgba[1]).arg(&w_).arg(&h_).arg(&s_).arg(&mut st.acc);
        unsafe { b.launch(cfg) }.map_err(cuda("frame_diff"))?;
        let mut v = [0f32];
        self.g.stream.memcpy_dtoh(&st.acc, &mut v).map_err(cuda("download"))?;
        self.g.sync()?;
        Ok(v[0] / (gw * gh) as f32)
    }

    /// Промежуточный кадр в момент `t` между двумя последними (RGBA8 исходного размера).
    pub fn interpolate(&mut self, t: f32) -> Result<&[u8]> {
        let mut st = self.st.take().ok_or_else(|| Error::Unsupported("нет кадров".into()))?;
        let r = self.run(&mut st, t);
        self.st = Some(st);
        r?;
        let st = self.st.as_mut().unwrap();
        let n = st.w * st.h * 4;
        self.g.download(&st.out, &mut st.host, n)
    }

    fn run(&self, st: &mut State, t: f32) -> Result<()> {
        if st.have < 2 {
            return Err(Error::Unsupported("нужно два кадра".into()));
        }
        let (i0, i1) = (1 - st.cur, st.cur);
        let (ph, pw) = (st.ph, st.pw);
        for blk in 0..5 {
            let s = self.scales[blk] / self.flow_scale;
            let (hl, wl) = ((ph as f32 / s) as usize, (pw as f32 / s) as usize);
            // вход блока на его разрешении: кадры (или сдвинутые), признаки, время, [маска, признаки, поток]
            let mut c = 0;
            if blk == 0 {
                self.resize(&st.img[i0], ph, pw, 0, 3, &mut st.xlow, hl, wl, 0, 1.0, false)?;
                self.resize(&st.img[i1], ph, pw, 0, 3, &mut st.xlow, hl, wl, 3, 1.0, false)?;
                self.resize(&st.feat[i0], ph, pw, 0, 4, &mut st.xlow, hl, wl, 6, 1.0, false)?;
                self.resize(&st.feat[i1], ph, pw, 0, 4, &mut st.xlow, hl, wl, 10, 1.0, false)?;
                c = 14;
            } else {
                self.resize(&st.warped, ph, pw, 0, 14, &mut st.xlow, hl, wl, 0, 1.0, false)?;
                c += 14;
            }
            // время — постоянный канал
            {
                let n = (hl * wl) as i32;
                let off = (c * hl * wl) as u64 * 4;
                let ptr = self.g.ptr(&st.xlow) + off;
                let mut b = self.g.stream.launch_builder(&self.k.fill);
                b.arg(&ptr).arg(&n).arg(&t);
                unsafe { b.launch(Self::lc(hl * wl)) }.map_err(cuda("fill"))?;
                c += 1;
            }
            if blk > 0 {
                self.resize(&st.mf, ph, pw, 0, 9, &mut st.xlow, hl, wl, c, 1.0, false)?;
                c += 9;
                self.resize(&st.flow, ph, pw, 0, 4, &mut st.xlow, hl, wl, c, 1.0 / s, false)?;
            }
            let p = format!("block{blk}");
            let (h1, w1) = self.conv(&format!("{p}.conv0.0.0"), None, None, &st.xlow, hl, wl, &mut st.a, 2, true)?;
            let (h2, w2) = self.conv(&format!("{p}.conv0.1.0"), None, None, &st.a, h1, w1, &mut st.b, 2, true)?;
            // 8 ResConv: b → a → b … (чётное число — итог снова в b)
            for r in 0..8 {
                let (src, dst) = if r % 2 == 0 { (&st.b, &mut st.a) } else { (&st.a, &mut st.b) };
                let name = format!("{p}.convblock.{r}");
                self.conv(&format!("{name}.conv"), Some(&format!("{name}.beta")), Some(src), src, h2, w2, dst, 1, true)?;
            }
            // lastconv: ×2 и PixelShuffle(2) → 13 каналов на разрешении блока
            self.deconv(&format!("{p}.lastconv.0"), &st.b, h2, w2, &mut st.tmp, true)?;
            // на полное разрешение: поток ×s (первый блок — присвоить, дальше — прибавить), маска и признаки
            self.resize(&st.tmp, hl, wl, 0, 4, &mut st.flow, ph, pw, 0, s, blk > 0)?;
            self.resize(&st.tmp, hl, wl, 4, 9, &mut st.mf, ph, pw, 0, 1.0, false)?;
            // сдвинутые потоком кадры и признаки — для следующего блока и итога
            self.warp(&st.img[i0], 3, &st.flow, 0, &mut st.warped, 0, ph, pw)?;
            self.warp(&st.img[i1], 3, &st.flow, 2, &mut st.warped, 3, ph, pw)?;
            if blk < 4 {
                self.warp(&st.feat[i0], 4, &st.flow, 0, &mut st.warped, 6, ph, pw)?;
                self.warp(&st.feat[i1], 4, &st.flow, 2, &mut st.warped, 10, ph, pw)?;
            }
        }
        // итог: warped0·σ(m) + warped1·(1 − σ(m))
        let px = ph * pw;
        let w1p = self.g.ptr(&st.warped) + (3 * px) as u64 * 4;
        let (pw_, ph_, w_, h_) = (pw as i32, ph as i32, st.w as i32, st.h as i32);
        let mut b = self.g.stream.launch_builder(&self.k.blend);
        b.arg(&st.warped).arg(&w1p).arg(&st.mf).arg(&pw_).arg(&ph_).arg(&mut st.out).arg(&w_).arg(&h_);
        unsafe { b.launch(Self::lc(st.w * st.h)) }.map_err(cuda("blend"))?;
        Ok(())
    }

    /// Перемотка: пары кадров больше нет.
    pub fn reset(&mut self) {
        if let Some(st) = self.st.as_mut() {
            st.have = 0;
        }
    }
}
