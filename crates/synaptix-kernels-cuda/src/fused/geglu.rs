//! Fused GEGLU FFN на pre-shuffled NVFP4 weights.
//!
//! `out = gelu_tanh(W_gate · x) * (W_up · x)`. `gelu_tanh` — pytorch_tanh
//! approximation (Gemma, T5-style). Один CUDA launch, X reuse в smem.

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct Nvfp4GegluShufKernels {
    _module: Arc<CudaModule>,
    w4: CudaFunction,
    w8: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4GegluShufKernels>)>>> = OnceLock::new();

const SMEM_OPT_IN_BYTES: i32 = 99 * 1024;
const W4_M_TILE: u32 = 64;
const W4_THREADS: u32 = 128;
const W8_M_TILE: u32 = 128;
const W8_THREADS: u32 = 256;

impl Nvfp4GegluShufKernels {
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
        let src = include_str!("../cu/fused/mlp/nvfp4_geglu_shuf.cu");
        let module =
            compile_module_with_opts(ctx, src, "nvfp4_geglu_shuf.cu", &[], Some("sm_120a"))?;
        let w4 = load_fn(&module, "nvfp4_geglu_shuf_f16_w4")?;
        let w8 = load_fn(&module, "nvfp4_geglu_shuf_f16_w8")?;
        for f in [&w4, &w8] {
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_OPT_IN_BYTES,
            )
            .map_err(|e| {
                SynaptixError::Cuda(format!("set_attribute nvfp4_geglu_shuf shared: {e:?}"))
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

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_geglu_shuf_f16(
    kernels: &Nvfp4GegluShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_gate: &CudaSlice<u8>,
    scales_w_gate: &CudaSlice<u8>,
    packed_w_up: &CudaSlice<u8>,
    scales_w_up: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_geglu_shuf_f16: K={k} must be multiple of 64"
        )));
    }
    if n == 0 || n % 16 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_geglu_shuf_f16: N={n} must be positive and multiple of 16"
        )));
    }
    let (kfn, threads, m_tile) = if n % W8_M_TILE == 0 {
        (&kernels.w8, W8_THREADS, W8_M_TILE)
    } else if n % W4_M_TILE == 0 {
        (&kernels.w4, W4_THREADS, W4_M_TILE)
    } else {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_geglu_shuf_f16: N={n} must be multiple of 64 (W4) or 128 (W8)"
        )));
    };
    let grid = n / m_tile;
    let sf_inner_w = sf_inner_dim(k);
    let smem_bytes = (k / 2) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(packed_w_gate)
        .arg(scales_w_gate)
        .arg(packed_w_up)
        .arg(scales_w_up)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_geglu_shuf: {e:?}")))?;
    }
    Ok(())
}
