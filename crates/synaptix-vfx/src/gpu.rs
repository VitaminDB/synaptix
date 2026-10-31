//! GPU: контекст, поток, ядра `vfx.cu`, буферы активаций (NHWC f16) и запуск свёртки со слитым
//! эпилогом.

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use half::f16;

use crate::{Error, Result};

/// Шаг каналов в памяти: кратно 16 (K-шаг mma).
pub fn pad16(c: usize) -> usize {
    c.div_ceil(16) * 16
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// Свёртки: [накопление f32, f16][строк на блок 4, 2][каналов на блок 16, 32, 48, 64].
    conv: [[[CudaFunction; 4]; 2]; 2],
    /// Накопление в f16 (вдвое быстрее на GeForce): `SYNAPTIX_VFX_ACC=f32` — выключить.
    pub half_acc: bool,
    rgba2nhwc: CudaFunction,
    copy_ch: CudaFunction,
    layer_norm: CudaFunction,
    group_stats: CudaFunction,
    group_apply: CudaFunction,
    _module: Arc<CudaModule>,
}

fn cuda<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> Error + '_ {
    move |e| Error::Cuda(format!("{what}: {e:?}"))
}

/// Общая память блока свёртки на устройстве (байт) — больше не просим.
const MAX_SMEM: u32 = 99 * 1024;

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let ctx = CudaContext::new(ordinal).map_err(cuda("CudaContext"))?;
        let stream = ctx.new_stream().map_err(cuda("new_stream"))?;
        let module = synaptix_kernels_cuda::kernels::compile::compile_module_with_opts(&ctx, include_str!("vfx.cu"), "synaptix-vfx/vfx.cu", &[], None)
            .map_err(|e| Error::Cuda(e.to_string()))?;
        let load = |n: &str| module.load_function(n).map_err(|e| Error::Cuda(format!("{n}: {e:?}")));
        let set = |p: &str| -> Result<[[CudaFunction; 4]; 2]> {
            Ok([
                [load(&format!("{p}16_4"))?, load(&format!("{p}32_4"))?, load(&format!("{p}48_4"))?, load(&format!("{p}64_4"))?],
                [load(&format!("{p}16_2"))?, load(&format!("{p}32_2"))?, load(&format!("{p}48_2"))?, load(&format!("{p}64_2"))?],
            ])
        };
        let conv = [set("conv_mma")?, set("conv_mmah")?];
        let half_acc = std::env::var("SYNAPTIX_VFX_ACC").map(|v| v != "f32").unwrap_or(true);
        for f in conv.iter().flatten().flatten() {
            f.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, MAX_SMEM as i32)
                .map_err(cuda("smem"))?;
        }
        let rgba2nhwc = load("rgba_to_nhwc")?;
        let copy_ch = load("copy_channels")?;
        let layer_norm = load("layer_norm_px")?;
        let group_stats = load("group_stats")?;
        let group_apply = load("group_apply")?;
        Ok(Arc::new(Self { ctx, stream, conv, half_acc, rgba2nhwc, copy_ch, layer_norm, group_stats, group_apply, _module: module }))
    }

    pub fn alloc_f16(&self, n: usize) -> Result<CudaSlice<f16>> {
        self.stream.alloc_zeros::<f16>(n.max(1)).map_err(cuda("alloc f16"))
    }

    pub fn alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>> {
        self.stream.alloc_zeros::<u8>(n.max(1)).map_err(cuda("alloc u8"))
    }

    pub fn upload_f16(&self, v: &[f32]) -> Result<CudaSlice<f16>> {
        let h: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
        self.stream.clone_htod(&h).map_err(cuda("upload f16"))
    }

    pub fn upload_f32(&self, v: &[f32]) -> Result<CudaSlice<f32>> {
        self.stream.clone_htod(v).map_err(cuda("upload f32"))
    }

    /// Закреплённая память хоста под выгрузку кадра (вдвое-втрое быстрее обычной).
    pub fn alloc_pinned(&self, n: usize) -> Result<cudarc::driver::PinnedHostSlice<u8>> {
        // SAFETY: память только пишется копией с устройства и читается после синхронизации.
        // флаги 0 — обычная кэшируемая закреплённая память: процессор её читает (write-combined —
        // в десятки раз медленнее на чтение)
        unsafe { self.ctx.alloc_pinned_with_flags::<u8>(n.max(1), 0) }.map_err(cuda("alloc pinned"))
    }

    /// Кадр RGBA с устройства в закреплённую память и срез на него (после синхронизации).
    pub fn download<'a>(&self, src: &CudaSlice<u8>, dst: &'a mut cudarc::driver::PinnedHostSlice<u8>, n: usize) -> Result<&'a [u8]> {
        self.stream.memcpy_dtoh(&src.slice(..n), &mut *dst).map_err(cuda("download"))?;
        self.sync()?;
        Ok(&dst.as_slice().map_err(cuda("pinned"))?[..n])
    }

    pub fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        let (p, _g) = s.device_ptr(&self.stream);
        p
    }

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().map_err(cuda("sync"))
    }

    /// RGBA8 (на устройстве) → NHWC f16 с шагом `cs` (RGB/255, остальное — нули).
    pub fn rgba_to_nhwc(&self, src: &CudaSlice<u8>, dst: &mut CudaSlice<f16>, pixels: usize, cs: usize) -> Result<()> {
        self.rgba_to_nhwc_norm(src, dst, pixels, cs, 1.0 / 255.0, [0.0; 3])
    }

    /// То же с нормализацией: канал = значение·`scale` + `off[c]` (значение 0..255).
    pub fn rgba_to_nhwc_norm(&self, src: &CudaSlice<u8>, dst: &mut CudaSlice<f16>, pixels: usize, cs: usize, scale: f32, off: [f32; 3]) -> Result<()> {
        let n = pixels as i32;
        let cs = cs as i32;
        let cfg = LaunchConfig { grid_dim: ((pixels as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.rgba2nhwc);
        b.arg(src).arg(dst).arg(&n).arg(&cs).arg(&scale).arg(&off[0]).arg(&off[1]).arg(&off[2]);
        unsafe { b.launch(cfg) }.map_err(cuda("rgba_to_nhwc"))?;
        Ok(())
    }

    /// Каналы `[c0, c1)` из `src` в `dst` (шаг пикселя `ps`).
    pub fn copy_channels(&self, src: &CudaSlice<f16>, dst: &mut CudaSlice<f16>, pixels: usize, ps: usize, c0: usize, c1: usize) -> Result<()> {
        let total = (pixels * (c1 - c0)) as u32;
        let cfg = LaunchConfig { grid_dim: (total.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (n, ps, c0, c1) = (pixels as i32, ps as i32, c0 as i32, c1 as i32);
        let mut b = self.stream.launch_builder(&self.copy_ch);
        b.arg(src).arg(dst).arg(&n).arg(&ps).arg(&c0).arg(&c1);
        unsafe { b.launch(cfg) }.map_err(cuda("copy_channels"))?;
        Ok(())
    }

    /// LayerNorm по каналам пикселя.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(&self, x: &CudaSlice<f16>, y: &mut CudaSlice<f16>, g: &CudaSlice<f32>, bta: &CudaSlice<f32>, pixels: usize, cs: usize, c: usize, eps: f32) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: ((pixels as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (n, cs, c) = (pixels as i32, cs as i32, c as i32);
        let mut b = self.stream.launch_builder(&self.layer_norm);
        b.arg(x).arg(y).arg(g).arg(bta).arg(&n).arg(&cs).arg(&c).arg(&eps);
        unsafe { b.launch(cfg) }.map_err(cuda("layer_norm"))?;
        Ok(())
    }

    /// GroupNorm по всему кадру + остаток: `y = norm(x)·g + b + skip`. `stats` — [groups·2] f32.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm(&self, x: &CudaSlice<f16>, skip: &CudaSlice<f16>, y: &mut CudaSlice<f16>, stats: &mut CudaSlice<f32>, g: &CudaSlice<f32>, bta: &CudaSlice<f32>, pixels: usize, cs: usize, c: usize, groups: usize, eps: f32) -> Result<()> {
        self.stream.memset_zeros(stats).map_err(cuda("memset"))?;
        let (n, csi, ci, gi) = (pixels as i32, cs as i32, c as i32, groups as i32);
        let total = (pixels * c) as u32;
        let blocks = total.div_ceil(256).min(1024);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (groups * 2 * 4) as u32 };
        let mut b = self.stream.launch_builder(&self.group_stats);
        b.arg(x).arg(&*stats).arg(&n).arg(&csi).arg(&ci).arg(&gi);
        unsafe { b.launch(cfg) }.map_err(cuda("group_stats"))?;
        let cfg = LaunchConfig { grid_dim: (total.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.group_apply);
        b.arg(x).arg(skip).arg(y).arg(&*stats).arg(g).arg(bta).arg(&n).arg(&csi).arg(&ci).arg(&gi).arg(&eps);
        unsafe { b.launch(cfg) }.map_err(cuda("group_apply"))?;
        Ok(())
    }

    pub fn conv(&self, a: &ConvArgs) -> Result<()> {
        let k = a.k as u32;
        let p = k / 2;
        let cin = a.cs.iter().sum::<i32>() as u32;
        let si = cin + 8;
        let (nb, ni) = match a.cout_pad {
            0..=16 => (16u32, 0),
            17..=32 => (32, 1),
            33..=48 => (48, 2),
            _ => (64, 3),
        };
        // блок 4 строки, если ореол и два буфера отвода помещаются, иначе 2
        // у 1×1 один отвод — второй буфер весов не нужен
        let nbuf = if k == 1 { 1 } else { 2 };
        let smem_for = |th: u32| ((th + 2 * p) * (64 + 2 * p) * si + nbuf * nb * si) * 2;
        // блок в 2 строки быстрее (больше блоков на SM); 4 — только по SYNAPTIX_VFX_TH=4 для замеров
        let force4 = std::env::var("SYNAPTIX_VFX_TH").is_ok_and(|v| v == "4");
        let (th, ti) = if force4 && smem_for(4) <= MAX_SMEM { (4u32, 0) } else { (2, 1) };
        let smem = smem_for(th);
        if smem > MAX_SMEM {
            return Err(Error::Unsupported(format!("свёртка {k}×{k}: {cin} входных каналов не помещаются в общую память")));
        }
        let f = &self.conv[usize::from(self.half_acc)][ti][ni];
        let cfg = LaunchConfig {
            grid_dim: ((a.w_ as u32).div_ceil(64), (a.h as u32).div_ceil(th), (a.cout_pad as u32).div_ceil(nb)),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        let mut b = self.stream.launch_builder(f);
        b.arg(a);
        unsafe { b.launch(cfg) }.map_err(cuda("conv_mma"))?;
        Ok(())
    }
}

/// Аргументы `conv_mma` — раскладка как у `struct ConvArgs` в vfx.cu.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ConvArgs {
    pub inp: [u64; 4],
    pub in0b: u64,
    pub w: u64,
    pub bias: u64,
    pub res: u64,
    pub alpha: u64,
    pub out: u64,
    pub base: u64,
    pub cs: [i32; 4],
    pub ps: [i32; 4],
    pub h: i32,
    pub w_: i32,
    pub cout_pad: i32,
    pub cso: i32,
    pub k: i32,
    pub in_act: i32,
    pub act: i32,
    pub slope: f32,
    pub res_mode: i32,
    pub shuffle: i32,
    pub out_kind: i32,
    pub base_mode: i32,
    pub base_w: i32,
    pub base_h: i32,
    pub out_scale: f32,
}

// SAFETY: plain-old-data, раскладка совпадает с C-структурой ядра.
unsafe impl DeviceRepr for ConvArgs {}

/// Активация эпилога.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    None,
    Relu,
    LeakyRelu(f32),
    /// PReLU с одним наклоном на все каналы.
    PRelu(f32),
    /// PReLU по каналам (наклоны в [`Conv::alpha`]).
    PReluCh,
    Silu,
    Sigmoid,
    Mish,
    /// Внимание SPAB: `(v + остаток) · (sigmoid(v) − 0,5)`; остаток — [`Res::Before`].
    SpabAttention,
    /// Ворота: `остаток · sigmoid(v)` (EA в RealPLKSR); остаток — [`Res::Before`].
    Gate,
}

impl Act {
    fn code(self) -> (i32, f32) {
        match self {
            Act::None => (0, 0.0),
            Act::Relu => (1, 0.0),
            Act::LeakyRelu(s) => (2, s),
            Act::PRelu(s) => (3, s),
            Act::PReluCh => (4, 0.0),
            Act::Silu => (5, 0.0),
            Act::Sigmoid => (6, 0.0),
            Act::Mish => (7, 0.0),
            Act::SpabAttention => (8, 0.0),
            Act::Gate => (9, 0.0),
        }
    }
}

/// Веса свёртки на устройстве: [K*K][cout_pad][cin_pad] f16, bias f32.
pub struct Conv {
    pub w: CudaSlice<f16>,
    pub bias: CudaSlice<f32>,
    pub alpha: Option<CudaSlice<f32>>,
    pub k: usize,
    /// Шаги каналов входов конкатенации, кратны 16.
    pub cs: Vec<usize>,
    pub cout: usize,
    pub cout_pad: usize,
}

impl Conv {
    /// `weight` — [cout, cin, k, k] (PyTorch), `splits` — каналы входов конкатенации по порядку (их
    /// сумма — `cin`). Шаги каналов в памяти — кратно 16; при pixel shuffle ×r выход дополняется
    /// целыми группами r².
    pub fn new(g: &Gpu, weight: &[f32], bias: Option<&[f32]>, cout: usize, k: usize, splits: &[usize], shuffle: usize) -> Result<Self> {
        let cin: usize = splits.iter().sum();
        if weight.len() != cout * cin * k * k || bias.is_some_and(|b| b.len() != cout) || splits.is_empty() || splits.len() > 4 {
            return Err(Error::Weights(format!("свёртка {cout}×{cin}×{k}×{k}: размер весов {}", weight.len())));
        }
        let cs: Vec<usize> = splits.iter().map(|c| pad16(*c)).collect();
        let cin_pad: usize = cs.iter().sum();
        // входной канал i → место в конкатенации с дополнением
        let mut place = Vec::with_capacity(cin);
        let mut base = 0;
        for (c, s) in splits.iter().zip(&cs) {
            place.extend(base..base + c);
            base += s;
        }
        // при shuffle канал выхода n = c·r² + sub: дополнять целыми группами r²
        let rr = shuffle * shuffle;
        let cout_pad = if rr > 1 { cout.div_ceil(rr).max(1) * rr } else { cout }.div_ceil(8) * 8;
        let mut w = vec![0f32; k * k * cout_pad * cin_pad];
        for o in 0..cout {
            for i in 0..cin {
                let pi = place[i];
                for t in 0..k * k {
                    w[(t * cout_pad + o) * cin_pad + pi] = weight[((o * cin + i) * k * k) + t];
                }
            }
        }
        let mut b = vec![0f32; cout_pad];
        if let Some(bias) = bias {
            b[..cout].copy_from_slice(bias);
        }
        Ok(Self { w: g.upload_f16(&w)?, bias: g.upload_f32(&b)?, alpha: None, k, cs, cout, cout_pad })
    }

    pub fn with_alpha(mut self, g: &Gpu, alpha: &[f32]) -> Result<Self> {
        let mut a = alpha.to_vec();
        a.resize(self.cout_pad.max(a.len()), 0.0);
        self.alpha = Some(g.upload_f32(&a)?);
        Ok(self)
    }
}

/// Куда пишет свёртка.
pub enum Out<'a> {
    /// NHWC f16 с шагом каналов `cs`.
    F16 { buf: &'a mut CudaSlice<f16>, cs: usize },
    /// RGBA8; `base` — исходник для добавки (его ширина, высота): билинейно или ближайшим пикселем.
    Rgba { buf: &'a mut CudaSlice<u8>, base: Option<(&'a CudaSlice<u8>, usize, usize)>, nearest: bool },
}

/// Один вход конкатенации: буфер, сдвиг (половинок) и шаг пикселя (0 — равен числу каналов).
#[derive(Clone, Copy)]
pub struct Src<'a> {
    pub buf: &'a CudaSlice<f16>,
    pub off: usize,
    pub ps: usize,
}

impl<'a> From<&'a CudaSlice<f16>> for Src<'a> {
    fn from(buf: &'a CudaSlice<f16>) -> Self {
        Self { buf, off: 0, ps: 0 }
    }
}

/// Вход свёртки: тензоры конкатенации по порядку, слагаемое к первому и активация первого.
pub struct In<'a> {
    pub xs: Vec<Src<'a>>,
    pub add: Option<&'a CudaSlice<f16>>,
    pub act: Act,
}

impl<'a> In<'a> {
    pub fn one(x: &'a CudaSlice<f16>) -> Self {
        Self { xs: vec![x.into()], add: None, act: Act::None }
    }

    pub fn cat(xs: &[&'a CudaSlice<f16>]) -> Self {
        Self { xs: xs.iter().map(|x| (*x).into()).collect(), add: None, act: Act::None }
    }

    pub fn srcs(xs: Vec<Src<'a>>) -> Self {
        Self { xs, add: None, act: Act::None }
    }

    pub fn sum(x: &'a CudaSlice<f16>, add: Option<&'a CudaSlice<f16>>) -> Self {
        Self { xs: vec![x.into()], add, act: Act::None }
    }

    /// Активация первого входа при загрузке (например, SiLU над сохранённым до активации выходом).
    pub fn activated(x: &'a CudaSlice<f16>, act: Act) -> Self {
        Self { xs: vec![x.into()], add: None, act }
    }
}

/// Остаток: прибавить до или после активации (тензор в раскладке выхода свёртки до shuffle).
pub enum Res<'a> {
    None,
    Before(&'a CudaSlice<f16>),
    After(&'a CudaSlice<f16>),
}

impl Gpu {
    /// Свёртка `c` над входом `h×w` с эпилогом.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, c: &Conv, input: In, h: usize, w: usize, act: Act, res: Res, shuffle: usize, out: Out) -> Result<()> {
        let (act, slope) = act.code();
        let (res_mode, res_p) = match res {
            Res::None => (0, 0),
            Res::Before(r) => (1, self.ptr(r)),
            Res::After(r) => (2, self.ptr(r)),
        };
        if input.xs.len() != c.cs.len() {
            return Err(Error::Unsupported(format!("свёртка ждёт {} входов, дано {}", c.cs.len(), input.xs.len())));
        }
        let mut inp = [0u64; 4];
        let mut cs = [0i32; 4];
        let mut ps = [0i32; 4];
        for (i, x) in input.xs.iter().enumerate() {
            inp[i] = self.ptr(x.buf) + (x.off * 2) as u64;
            cs[i] = c.cs[i] as i32;
            ps[i] = if x.ps == 0 { c.cs[i] } else { x.ps } as i32;
        }
        let mut a = ConvArgs {
            inp,
            cs,
            ps,
            in_act: input.act.code().0,
            in0b: input.add.map(|t| self.ptr(t)).unwrap_or(0),
            w: self.ptr(&c.w),
            bias: self.ptr(&c.bias),
            res: res_p,
            alpha: c.alpha.as_ref().map(|t| self.ptr(t)).unwrap_or(0),
            h: h as i32,
            w_: w as i32,
            cout_pad: c.cout_pad as i32,
            k: c.k as i32,
            act,
            slope,
            res_mode,
            shuffle: shuffle as i32,
            out_scale: 1.0,
            ..Default::default()
        };
        match out {
            Out::F16 { buf, cs } => {
                a.out = self.ptr(buf);
                a.cso = cs as i32;
            }
            Out::Rgba { buf, base, nearest } => {
                a.out = self.ptr(buf);
                a.out_kind = 1;
                a.base_mode = i32::from(nearest);
                if let Some((b, bw, bh)) = base {
                    a.base = self.ptr(b);
                    a.base_w = bw as i32;
                    a.base_h = bh as i32;
                }
            }
        }
        self.conv(&a)
    }
}
