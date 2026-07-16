//! Pointwise fused `out = silu(gate) * up`. Один kernel-launch вместо двух
//! (silu unary + binary mul), 3 trip'а памяти вместо 4.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct SiluAndMulKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<SiluAndMulKernels>)>>> = OnceLock::new();

impl SiluAndMulKernels {
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
        let src = include_str!("../cu/fused/mlp/silu_and_mul.cu");
        let module = compile_module(ctx, src, "silu_and_mul.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "silu_and_mul_f32")?,
            f16: load_fn(&module, "silu_and_mul_f16")?,
            bf16: load_fn(&module, "silu_and_mul_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

const BLOCK: u32 = 256;

fn cfg_for(total: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn silu_and_mul_f32(
    kernels: &SiluAndMulKernels,
    stream: &Arc<CudaStream>,
    gate: &CudaView<f32>,
    up: &CudaView<f32>,
    out: &mut CudaViewMut<f32>,
    total: u32,
) -> Result<()> {
    let mut b = stream.launch_builder(&kernels.f32);
    b.arg(gate).arg(up).arg(&mut *out).arg(&total);
    unsafe {
        b.launch(cfg_for(total))
            .map_err(|e| SynaptixError::Cuda(format!("launch silu_and_mul_f32: {e:?}")))?;
    }
    Ok(())
}

pub fn silu_and_mul_f16(
    kernels: &SiluAndMulKernels,
    stream: &Arc<CudaStream>,
    gate: &CudaView<f16>,
    up: &CudaView<f16>,
    out: &mut CudaViewMut<f16>,
    total: u32,
) -> Result<()> {
    let mut b = stream.launch_builder(&kernels.f16);
    b.arg(gate).arg(up).arg(&mut *out).arg(&total);
    unsafe {
        b.launch(cfg_for(total))
            .map_err(|e| SynaptixError::Cuda(format!("launch silu_and_mul_f16: {e:?}")))?;
    }
    Ok(())
}

pub fn silu_and_mul_bf16(
    kernels: &SiluAndMulKernels,
    stream: &Arc<CudaStream>,
    gate: &CudaView<bf16>,
    up: &CudaView<bf16>,
    out: &mut CudaViewMut<bf16>,
    total: u32,
) -> Result<()> {
    let mut b = stream.launch_builder(&kernels.bf16);
    b.arg(gate).arg(up).arg(&mut *out).arg(&total);
    unsafe {
        b.launch(cfg_for(total))
            .map_err(|e| SynaptixError::Cuda(format!("launch silu_and_mul_bf16: {e:?}")))?;
    }
    Ok(())
}
