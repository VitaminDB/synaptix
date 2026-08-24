use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

/// GEMV kernels (M=1 decode path) для всех dtype. Один warp = один output
/// элемент, K-axis между 32 lanes, warp-reduce через __shfl_xor_sync.
/// Drop-in замена cuBLAS-Lt matmul для M=1.
pub struct MmaGemvKernels {
    _module: Arc<CudaModule>,
    gemv_f16: CudaFunction,
    gemv_bf16: CudaFunction,
    gemv_f32: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<MmaGemvKernels>)>>> = OnceLock::new();

impl MmaGemvKernels {
    pub fn bf16_fn(&self) -> &CudaFunction {
        &self.gemv_bf16
    }

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
        let src = include_str!("mma_gemv.cu");
        // FP8 PTX cvt.f16x2.e4m3x2 требует sm_89+. F16/BF16/F32 GEMV — обычные.
        let module = compile_module_with_opts(ctx, src, "mma_gemv.cu", &[], Some("sm_89"))?;
        let gemv_f16 = load_fn(&module, "mma_gemv_f16")?;
        let gemv_bf16 = load_fn(&module, "mma_gemv_bf16")?;
        let gemv_f32 = load_fn(&module, "mma_gemv_f32")?;
        let new = Arc::new(Self {
            gemv_f16,
            gemv_bf16,
            gemv_f32,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// F16 GEMV: y = W @ x, W (N, K), x (K,), y (N,).
pub fn gemv_f16(
    kernels: &MmaGemvKernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<f16>,
    x: &CudaSlice<f16>,
    y: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if k % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "gemv_f16: K={k} must be even (для half2 vec load)"
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let launch = LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kernels.gemv_f16);
    b.arg(w).arg(x).arg(&mut *y).arg(&n).arg(&k);
    unsafe {
        b.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch gemv_f16: {e:?}")))?;
    }
    Ok(())
}

/// BF16 GEMV.
pub fn gemv_bf16(
    kernels: &MmaGemvKernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<bf16>,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if k % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "gemv_bf16: K={k} must be even"
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let launch = LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kernels.gemv_bf16);
    b.arg(w).arg(x).arg(&mut *y).arg(&n).arg(&k);
    unsafe {
        b.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch gemv_bf16: {e:?}")))?;
    }
    Ok(())
}

/// F32 GEMV.
pub fn gemv_f32(
    kernels: &MmaGemvKernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let launch = LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kernels.gemv_f32);
    b.arg(w).arg(x).arg(&mut *y).arg(&n).arg(&k);
    unsafe {
        b.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch gemv_f32: {e:?}")))?;
    }
    Ok(())
}

/// Linear GEMV из untyped `u8`-storage (для `Backend::linear`, M=1 decode path):
/// `y[N] = W[N,K] @ x[K]`, всё в `dtype` (F16/BF16/F32). Принимает byte-offset'ы
/// тензоров (x может быть narrow-строкой с offset≠0, напр. lm_head в prefill).
/// W читается row-major contiguous начиная с `w_off`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_linear_u8(
    kernels: &MmaGemvKernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    w_off: usize,
    x: &CudaSlice<u8>,
    x_off: usize,
    y: &mut CudaSlice<u8>,
    y_off: usize,
    n: u32,
    k: u32,
    dtype: DType,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if (dtype == DType::F16 || dtype == DType::BF16) && k % 2 != 0 {
        return Err(SynaptixError::Cuda(format!("gemv_linear_u8: K={k} odd")));
    }
    let cfg = LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nk = (n as usize) * (k as usize);
    let kk = k as usize;
    let nn = n as usize;

    macro_rules! launch {
        ($t:ty, $func:expr, $esz:expr) => {{
            let w_v = unsafe {
                w.slice(w_off..w_off + nk * $esz)
                    .transmute::<$t>(nk)
                    .ok_or_else(|| SynaptixError::Cuda("gemv: transmute w".into()))?
            };
            let x_v = unsafe {
                x.slice(x_off..x_off + kk * $esz)
                    .transmute::<$t>(kk)
                    .ok_or_else(|| SynaptixError::Cuda("gemv: transmute x".into()))?
            };
            let mut y_slice = y.slice_mut(y_off..y_off + nn * $esz);
            let mut y_v = unsafe {
                y_slice
                    .transmute_mut::<$t>(nn)
                    .ok_or_else(|| SynaptixError::Cuda("gemv: transmute y".into()))?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&w_v).arg(&x_v).arg(&mut y_v).arg(&n).arg(&k);
            unsafe {
                b.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch gemv_linear_u8: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::F16 => launch!(f16, &kernels.gemv_f16, 2),
        DType::BF16 => launch!(bf16, &kernels.gemv_bf16, 2),
        DType::F32 => launch!(f32, &kernels.gemv_f32, 4),
        _ => return Err(SynaptixError::Unsupported("gemv_linear_u8: dtype")),
    }
    Ok(())
}
