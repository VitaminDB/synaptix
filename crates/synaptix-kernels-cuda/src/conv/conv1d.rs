//! Conv1d direct CUDA kernel (один thread = один output element).
//!
//! Naive baseline вариант — без im2col, без Tensor Cores. Грузит каждый input
//! элемент C_in×K раз → подходит как функциональный baseline, но не как
//! production fast path. Production fast path = im2col + cuBLASLt GEMM
//! (см. отдельную задачу). Tests: F16/BF16/F32 vs CPU conv1d.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 128;

pub struct Conv1dKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Conv1dKernels>)>>> = OnceLock::new();

impl Conv1dKernels {
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
        let src = include_str!("../cu/conv/conv1d.cu");
        let module = compile_module(ctx, src, "conv1d.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "conv1d_direct_f32")?,
            f16: load_fn(&module, "conv1d_direct_f16")?,
            bf16: load_fn(&module, "conv1d_direct_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

pub fn l_out(l: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    let l_padded = l + 2 * padding;
    if l_padded < kernel {
        return 0;
    }
    (l_padded - kernel) / stride + 1
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d<T: DeviceRepr>(
    kernels: &Conv1dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    bias: Option<&CudaSlice<T>>,
    output: &mut CudaSlice<T>,
    b: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    padding: u32,
    dtype: DType,
) -> Result<()> {
    let l_o = l_out(l as usize, k as usize, stride as usize, padding as usize) as u32;
    if l_o == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "conv1d: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * c_out, l_o.div_ceil(BLOCK), 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let l_i = l as i32;
    let c_out_i = c_out as i32;
    let k_i = k as i32;
    let stride_i = stride as i32;
    let pad_i = padding as i32;
    let l_out_i = l_o as i32;
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let mut bld = stream.launch_builder(func);
    // bias может быть None — для kernel'я нам нужен валидный pointer; используем
    // input в качестве placeholder (kernel читает только если has_bias==1).
    let bias_ptr = bias.unwrap_or(input);
    bld.arg(input)
        .arg(weight)
        .arg(bias_ptr)
        .arg(&has_bias_i)
        .arg(&mut *output)
        .arg(&b_i)
        .arg(&c_in_i)
        .arg(&l_i)
        .arg(&c_out_i)
        .arg(&k_i)
        .arg(&stride_i)
        .arg(&pad_i)
        .arg(&l_out_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch conv1d: {e:?}")))?;
    }
    Ok(())
}

pub fn conv1d_f32(
    kernels: &Conv1dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    output: &mut CudaSlice<f32>,
    b: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    padding: u32,
) -> Result<()> {
    conv1d::<f32>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        l,
        c_out,
        k,
        stride,
        padding,
        DType::F32,
    )
}

pub fn conv1d_f16(
    kernels: &Conv1dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: Option<&CudaSlice<f16>>,
    output: &mut CudaSlice<f16>,
    b: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    padding: u32,
) -> Result<()> {
    conv1d::<f16>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        l,
        c_out,
        k,
        stride,
        padding,
        DType::F16,
    )
}

pub fn conv1d_bf16(
    kernels: &Conv1dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    output: &mut CudaSlice<bf16>,
    b: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    padding: u32,
) -> Result<()> {
    conv1d::<bf16>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        l,
        c_out,
        k,
        stride,
        padding,
        DType::BF16,
    )
}
