//! Fused Q/K/V проекции одним launch на pre-shuffled NVFP4 weights.
//!
//! Делает три GEMV (Q = W_q · x, K = W_k · x, V = W_v · x) в одном CUDA
//! kernel, экономя 2/3 чтений X из gmem за счёт shared smem-буфера. Caller
//! обязан pre-shuffle W_q/W_k/W_v через `nvfp4_w_repack` один раз при
//! weight loading и передавать сюда shuffled буферы.

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct Nvfp4QkvProjShufKernels {
    _module: Arc<CudaModule>,
    w4: CudaFunction,
    w8: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4QkvProjShufKernels>)>>> = OnceLock::new();

const SMEM_OPT_IN_BYTES: i32 = 99 * 1024;
const W4_M_TILE: u32 = 64;
const W4_THREADS: u32 = 128;
const W8_M_TILE: u32 = 128;
const W8_THREADS: u32 = 256;

impl Nvfp4QkvProjShufKernels {
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
        let src = include_str!("../cu/fused/projection/nvfp4_qkv_proj_shuf.cu");
        let module =
            compile_module_with_opts(ctx, src, "nvfp4_qkv_proj_shuf.cu", &[], Some("sm_120a"))?;
        let w4 = load_fn(&module, "nvfp4_qkv_proj_shuf_f16_w4")?;
        let w8 = load_fn(&module, "nvfp4_qkv_proj_shuf_f16_w8")?;
        for f in [&w4, &w8] {
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_OPT_IN_BYTES,
            )
            .map_err(|e| {
                SynaptixError::Cuda(format!("set_attribute nvfp4_qkv_proj_shuf shared: {e:?}"))
            })?;
        }
        let new = Arc::new(Self {
            w4,
            w8,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn sf_inner_dim(k: u32) -> u32 {
    k.div_ceil(64) * 4
}

/// Fused Q/K/V GEMV. Все три W должны быть pre-shuffled через `nvfp4_w_repack`.
/// `K` (hidden_size) одинаков для Q/K/V. N_q/N_k/N_v могут отличаться (GQA).
/// Каждое N_* должно быть кратно 16 (требование shuffled layout).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_qkv_proj_shuf_f16(
    kernels: &Nvfp4QkvProjShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_q: &CudaSlice<u8>,
    scales_w_q: &CudaSlice<u8>,
    packed_w_k: &CudaSlice<u8>,
    scales_w_k: &CudaSlice<u8>,
    packed_w_v: &CudaSlice<u8>,
    scales_w_v: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out_q: &mut CudaSlice<f16>,
    out_k: &mut CudaSlice<f16>,
    out_v: &mut CudaSlice<f16>,
    n_q: u32,
    n_k: u32,
    n_v: u32,
    k: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_qkv_proj_shuf_f16: K={k} must be multiple of 64"
        )));
    }
    for (label, n) in [("N_q", n_q), ("N_k", n_k), ("N_v", n_v)] {
        if n == 0 || n % 16 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "nvfp4_qkv_proj_shuf_f16: {label}={n} must be positive and multiple of 16"
            )));
        }
    }
    let n_max = n_q.max(n_k).max(n_v);
    let want_w8 = n_q % W8_M_TILE == 0 && n_k % W8_M_TILE == 0 && n_v % W8_M_TILE == 0;
    let (kfn, threads, m_tile) = if want_w8 {
        (&kernels.w8, W8_THREADS, W8_M_TILE)
    } else {
        if n_q % W4_M_TILE != 0 || n_k % W4_M_TILE != 0 || n_v % W4_M_TILE != 0 {
            return Err(SynaptixError::Cuda(format!(
                "nvfp4_qkv_proj_shuf_f16: each N must be multiple of 64 (W4) or 128 (W8); got N_q={n_q} N_k={n_k} N_v={n_v}"
            )));
        }
        (&kernels.w4, W4_THREADS, W4_M_TILE)
    };
    let grid = n_max.div_ceil(m_tile);
    let sf_inner_w = sf_inner_dim(k);
    let smem_bytes = (k / 2) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(packed_w_q)
        .arg(scales_w_q)
        .arg(packed_w_k)
        .arg(scales_w_k)
        .arg(packed_w_v)
        .arg(scales_w_v)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&mut *out_q)
        .arg(&mut *out_k)
        .arg(&mut *out_v)
        .arg(&n_q)
        .arg(&n_k)
        .arg(&n_v)
        .arg(&k)
        .arg(&sf_inner_w);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_qkv_proj_shuf: {e:?}")))?;
    }
    Ok(())
}
