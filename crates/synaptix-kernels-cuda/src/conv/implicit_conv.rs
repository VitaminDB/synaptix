use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 16;
const WARP_TILE_K: u32 = 2;
const THREADS: u32 = 256;
const SWIZZLE_STRIDE: u32 = 2048;
const K_STEP: u32 = BK * WARP_TILE_K;

fn smem_bytes(stages: u32) -> u32 {
    stages * (BM + BN) * BK * WARP_TILE_K * 2
}

pub struct ImplicitConvKernels {
    _module: Arc<CudaModule>,
    bf16: [Option<CudaFunction>; 2],
    bf16_part: Option<CudaFunction>,
    f16: [Option<CudaFunction>; 2],
    f16_part: Option<CudaFunction>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ImplicitConvKernels>)>>> = OnceLock::new();

fn load_set(
    module: &Arc<CudaModule>,
    prefix: &str,
) -> Result<([Option<CudaFunction>; 2], Option<CudaFunction>)> {
    let mut fns: [Option<CudaFunction>; 2] = std::array::from_fn(|_| None);
    for (ci, (suffix, stages)) in [("s3", 3u32), ("s4", 4u32)].iter().enumerate() {
        let name = format!("{prefix}_swz_{suffix}");
        let f = load_fn(module, &name)?;
        f.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes(*stages) as i32,
        )
        .map_err(|e| SynaptixError::Cuda(format!("set smem {name}: {e:?}")))?;
        fns[ci] = Some(f);
    }
    let part = load_fn(module, &format!("{prefix}_part"))?;
    part.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes(3) as i32,
    )
    .map_err(|e| SynaptixError::Cuda(format!("set smem {prefix}_part: {e:?}")))?;
    Ok((fns, Some(part)))
}

impl ImplicitConvKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return Ok(v.clone());
                }
            }
        }
        let src = include_str!("../cu/conv/implicit_conv.cu");
        let module = compile_module_with_opts(ctx, src, "implicit_conv.cu", &[], Some("sm_120a"))?;
        let (bf16, bf16_part) = load_set(&module, "implicit_conv_bf16")?;
        let (f16, f16_part) = load_set(&module, "implicit_conv_f16")?;
        let new = Arc::new(Self {
            bf16,
            bf16_part,
            f16,
            f16_part,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn select(
        &self,
        dtype: DType,
        m: u32,
        n: u32,
        num_k_tiles: u32,
    ) -> Result<(&CudaFunction, u32)> {
        let partial = m % BM != 0 || n % BN != 0;
        let (set, part) = match dtype {
            DType::BF16 => (&self.bf16, &self.bf16_part),
            DType::F16 => (&self.f16, &self.f16_part),
            _ => return Err(SynaptixError::Unsupported("implicit_conv: dtype")),
        };
        if partial {
            let f = part
                .as_ref()
                .ok_or(SynaptixError::Unsupported("implicit_conv: нет part-ядра"))?;
            Ok((f, smem_bytes(3)))
        } else {
            let desired = if m <= 768 { 4u32 } else { 3u32 };
            let stages = desired.min(num_k_tiles + 1).max(3);
            let idx = if stages >= 4 { 1 } else { 0 };
            let f = set[idx]
                .as_ref()
                .ok_or(SynaptixError::Unsupported("implicit_conv: нет swz-ядра"))?;
            Ok((f, smem_bytes(stages)))
        }
    }
}

fn launch_for(m: u32, n: u32, partial: bool, smem: u32) -> LaunchConfig {
    if partial {
        LaunchConfig {
            grid_dim: (n.div_ceil(BN), m.div_ceil(BM), 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: smem,
        }
    } else {
        let n_swz = n.div_ceil(SWIZZLE_STRIDE).max(1);
        LaunchConfig {
            grid_dim: (n.div_ceil(BN).div_ceil(n_swz), m.div_ceil(BM), n_swz),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: smem,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_implicit_nhwc_u8(
    kernels: &ImplicitConvKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u8>,
    input_off: usize,
    filter: &CudaSlice<u8>,
    filter_off: usize,
    output: &mut CudaSlice<u8>,
    output_off: usize,
    nb: u32,
    h: u32,
    w: u32,
    c: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    p: u32,
    q: u32,
    pad_h: u32,
    pad_w: u32,
    stride_h: u32,
    stride_w: u32,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<u8>>,
    temb: Option<&CudaSlice<u8>>,
    out_nhwc: bool,
    dtype: DType,
) -> Result<()> {
    let m = nb * p * q;
    let n = cout;
    let k = kh * kw * c;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k % K_STEP != 0 {
        return Err(SynaptixError::Unsupported(
            "implicit_conv: K=Kh*Kw*Cin не кратно 32",
        ));
    }
    let num_k_tiles = k / K_STEP;
    if num_k_tiles < 2 {
        return Err(SynaptixError::Unsupported(
            "implicit_conv: K<64 (пайплайн требует >=2 K-тайла)",
        ));
    }
    let (kfn, smem) = kernels.select(dtype, m, n, num_k_tiles)?;
    let esz = (dtype.size_in_bits() / 8) as usize;
    let n_in = (nb * c * h * w) as usize;
    let n_f = (cout * c * kh * kw) as usize;
    let n_out = (m * n) as usize;
    let launch = launch_for(m, n, m % BM != 0 || n % BN != 0, smem);
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let (nbi, hi, wi, ci) = (nb as i32, h as i32, w as i32, c as i32);
    let (khi, kwi, pi, qi) = (kh as i32, kw as i32, p as i32, q as i32);
    let (shi, swi, phi, pwi) = (stride_h as i32, stride_w as i32, pad_h as i32, pad_w as i32);

    macro_rules! go {
        ($t:ty) => {{
            let in_v = unsafe {
                input
                    .slice(input_off..input_off + n_in * esz)
                    .transmute::<$t>(n_in)
                    .ok_or(SynaptixError::Cuda("implicit_conv: transmute input".into()))?
            };
            let f_v = unsafe {
                filter
                    .slice(filter_off..filter_off + n_f * esz)
                    .transmute::<$t>(n_f)
                    .ok_or(SynaptixError::Cuda("implicit_conv: transmute filter".into()))?
            };
            let mut out_s = output.slice_mut(output_off..output_off + n_out * esz);
            let mut out_v = unsafe {
                out_s
                    .transmute_mut::<$t>(n_out)
                    .ok_or(SynaptixError::Cuda("implicit_conv: transmute out".into()))?
            };
            let nbias = n as usize;
            let ntemb = (nb * cout) as usize;
            let bias_v = match bias {
                Some(b) => unsafe { b.slice(..nbias * esz).transmute::<$t>(nbias) }
                    .ok_or(SynaptixError::Cuda("implicit_conv: transmute bias".into()))?,
                None => unsafe { input.slice(input_off..input_off + nbias * esz).transmute::<$t>(nbias) }.unwrap(),
            };
            let res_v = match residual {
                Some(r) => unsafe { r.slice(..n_out * esz).transmute::<$t>(n_out) }
                    .ok_or(SynaptixError::Cuda("implicit_conv: transmute residual".into()))?,
                None => unsafe { input.slice(input_off..input_off + n_in.min(n_out) * esz).transmute::<$t>(n_in.min(n_out)) }.unwrap(),
            };
            let temb_v = match temb {
                Some(t) => unsafe { t.slice(..ntemb * esz).transmute::<$t>(ntemb) }
                    .ok_or(SynaptixError::Cuda("implicit_conv: transmute temb".into()))?,
                None => unsafe { input.slice(input_off..input_off + ntemb.min(n_in) * esz).transmute::<$t>(ntemb.min(n_in)) }.unwrap(),
            };
            let hb: i32 = if bias.is_some() { 1 } else { 0 };
            let hr: i32 = if residual.is_some() { 1 } else { 0 };
            let ht: i32 = if temb.is_some() { 1 } else { 0 };
            let onhwc: i32 = if out_nhwc { 1 } else { 0 };
            let mut bld = stream.launch_builder(kfn);
            bld.arg(&in_v)
                .arg(&f_v)
                .arg(&mut out_v)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&nbi)
                .arg(&hi)
                .arg(&wi)
                .arg(&ci)
                .arg(&khi)
                .arg(&kwi)
                .arg(&pi)
                .arg(&qi)
                .arg(&shi)
                .arg(&swi)
                .arg(&phi)
                .arg(&pwi)
                .arg(&bias_v)
                .arg(&hb)
                .arg(&res_v)
                .arg(&hr)
                .arg(&temb_v)
                .arg(&ht)
                .arg(&onhwc);
            unsafe {
                bld.launch(launch)
                    .map_err(|e| SynaptixError::Cuda(format!("launch implicit_conv: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::BF16 => go!(bf16),
        DType::F16 => go!(f16),
        _ => return Err(SynaptixError::Unsupported("implicit_conv: dtype")),
    }
    Ok(())
}
