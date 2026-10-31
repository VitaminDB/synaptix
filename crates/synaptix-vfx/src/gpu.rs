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
    conv: CudaFunction,
    rgba2nhwc: CudaFunction,
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
        let conv = load("conv_mma")?;
        conv.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, MAX_SMEM as i32)
            .map_err(cuda("smem"))?;
        let rgba2nhwc = load("rgba_to_nhwc")?;
        Ok(Arc::new(Self { ctx, stream, conv, rgba2nhwc, _module: module }))
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

    pub fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        let (p, _g) = s.device_ptr(&self.stream);
        p
    }

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().map_err(cuda("sync"))
    }

    /// RGBA8 (на устройстве) → NHWC f16 с шагом `cs` (RGB/255, остальное — нули).
    pub fn rgba_to_nhwc(&self, src: &CudaSlice<u8>, dst: &mut CudaSlice<f16>, pixels: usize, cs: usize) -> Result<()> {
        let n = pixels as i32;
        let cs = cs as i32;
        let cfg = LaunchConfig { grid_dim: ((pixels as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.rgba2nhwc);
        b.arg(src).arg(dst).arg(&n).arg(&cs);
        unsafe { b.launch(cfg) }.map_err(cuda("rgba_to_nhwc"))?;
        Ok(())
    }

    pub fn conv(&self, a: &ConvArgs) -> Result<()> {
        let k = a.k as u32;
        let p = k / 2;
        let cin = (a.cs0 + a.cs1) as u32;
        let si = cin + 8;
        let smem = ((2 + 2 * p) * (64 + 2 * p) * si + 64 * si) * 2;
        if smem > MAX_SMEM {
            return Err(Error::Unsupported(format!("свёртка: {cin} входных каналов не помещаются в общую память")));
        }
        let cfg = LaunchConfig {
            grid_dim: ((a.w_ as u32).div_ceil(64), (a.h as u32).div_ceil(2), (a.cout_pad as u32).div_ceil(64)),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        let mut b = self.stream.launch_builder(&self.conv);
        b.arg(a);
        unsafe { b.launch(cfg) }.map_err(cuda("conv_mma"))?;
        Ok(())
    }
}

/// Аргументы `conv_mma` — раскладка как у `struct ConvArgs` в vfx.cu.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ConvArgs {
    pub in0: u64,
    pub in0b: u64,
    pub in1: u64,
    pub w: u64,
    pub bias: u64,
    pub res: u64,
    pub alpha: u64,
    pub out: u64,
    pub base: u64,
    pub h: i32,
    pub w_: i32,
    pub cs0: i32,
    pub cs1: i32,
    pub cout_pad: i32,
    pub cso: i32,
    pub k: i32,
    pub act: i32,
    pub slope: f32,
    pub res_mode: i32,
    pub shuffle: i32,
    pub out_kind: i32,
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
        }
    }
}

/// Веса свёртки на устройстве: [K*K][cout_pad][cin_pad] f16, bias f32.
pub struct Conv {
    pub w: CudaSlice<f16>,
    pub bias: CudaSlice<f32>,
    pub alpha: Option<CudaSlice<f32>>,
    pub k: usize,
    /// Шаги каналов входа: основной и присоединённый (конкатенация), кратны 16.
    pub cs0: usize,
    pub cs1: usize,
    pub cout: usize,
    pub cout_pad: usize,
}

impl Conv {
    /// `weight` — [cout, cin, k, k] (PyTorch), `in_split` — сколько входных каналов у первого тензора
    /// конкатенации (`cin` — один тензор). Выходных каналов в памяти — кратно 16 (pixel shuffle ×r —
    /// кратно 16·r²).
    pub fn new(g: &Gpu, weight: &[f32], bias: &[f32], cout: usize, cin: usize, k: usize, in_split: usize, shuffle: usize) -> Result<Self> {
        if weight.len() != cout * cin * k * k || bias.len() != cout {
            return Err(Error::Weights(format!("свёртка {cout}×{cin}×{k}×{k}: размер весов {} / {}", weight.len(), bias.len())));
        }
        let c0 = in_split;
        let c1 = cin - c0;
        let cs0 = pad16(c0);
        let cs1 = if c1 > 0 { pad16(c1) } else { 0 };
        let cin_pad = cs0 + cs1;
        // при shuffle канал выхода n = c·r² + sub: дополнять целыми группами r²
        let rr = shuffle * shuffle;
        let cout_pad = if rr > 1 { cout.div_ceil(rr).max(1) * rr } else { cout }.div_ceil(8) * 8;
        let mut w = vec![0f32; k * k * cout_pad * cin_pad];
        for o in 0..cout {
            for i in 0..cin {
                let pi = if i < c0 { i } else { cs0 + (i - c0) };
                for t in 0..k * k {
                    w[(t * cout_pad + o) * cin_pad + pi] = weight[((o * cin + i) * k * k) + t];
                }
            }
        }
        let mut b = vec![0f32; cout_pad];
        b[..cout].copy_from_slice(bias);
        Ok(Self { w: g.upload_f16(&w)?, bias: g.upload_f32(&b)?, alpha: None, k, cs0, cs1, cout, cout_pad })
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
    /// RGBA8; `base` — исходник для билинейной добавки (его ширина, высота).
    Rgba { buf: &'a mut CudaSlice<u8>, base: Option<(&'a CudaSlice<u8>, usize, usize)> },
}

/// Вход свёртки: основной тензор, слагаемое к нему и присоединённый по каналам.
pub struct In<'a> {
    pub x: &'a CudaSlice<f16>,
    pub add: Option<&'a CudaSlice<f16>>,
    pub cat: Option<&'a CudaSlice<f16>>,
}

impl<'a> In<'a> {
    pub fn one(x: &'a CudaSlice<f16>) -> Self {
        Self { x, add: None, cat: None }
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
        let mut a = ConvArgs {
            in0: self.ptr(input.x),
            in0b: input.add.map(|t| self.ptr(t)).unwrap_or(0),
            in1: input.cat.map(|t| self.ptr(t)).unwrap_or(0),
            w: self.ptr(&c.w),
            bias: self.ptr(&c.bias),
            res: res_p,
            alpha: c.alpha.as_ref().map(|t| self.ptr(t)).unwrap_or(0),
            h: h as i32,
            w_: w as i32,
            cs0: c.cs0 as i32,
            cs1: c.cs1 as i32,
            cout_pad: c.cout_pad as i32,
            k: c.k as i32,
            act,
            slope,
            res_mode,
            shuffle: shuffle as i32,
            out_scale: 1.0,
            ..Default::default()
        };
        if c.cs1 > 0 && input.cat.is_none() {
            return Err(Error::Unsupported("свёртка ждёт присоединённый вход".into()));
        }
        match out {
            Out::F16 { buf, cs } => {
                a.out = self.ptr(buf);
                a.cso = cs as i32;
            }
            Out::Rgba { buf, base } => {
                a.out = self.ptr(buf);
                a.out_kind = 1;
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
