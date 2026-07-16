//! Depthwise conv1d (groups == C, симметричный padding):
//! `out[b,c,i] = sum_ki w[c,ki] * x[b,c, i*stride - padding + ki]` + bias[c].
//! `x` [B,C,L], `weight` [C,1,K], `bias` [C], `out` [B,C,L_out],
//! `L_out = (L + 2*padding - K)/stride + 1`. F32/F16/BF16, f32-аккумулятор.
//! Naive baseline. Семантика совпадает с `synaptix_ops::conv::depthwise_conv`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::conv::conv1d::l_out;
use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 128;

pub struct DepthwiseConv1dKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<DepthwiseConv1dKernels>)>>> = OnceLock::new();

impl DepthwiseConv1dKernels {
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
        let src = include_str!("../cu/conv/depthwise.cu");
        let module = compile_module(ctx, src, "depthwise.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "depthwise_conv1d_f32")?,
            f16: load_fn(&module, "depthwise_conv1d_f16")?,
            bf16: load_fn(&module, "depthwise_conv1d_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv1d<T: DeviceRepr>(
    kernels: &DepthwiseConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    bias: Option<&CudaSlice<T>>,
    out: &mut CudaSlice<T>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
    padding: u32,
    dtype: DType,
) -> Result<()> {
    let stride = stride.max(1);
    let lo = l_out(l as usize, k as usize, stride as usize, padding as usize) as u32;
    if lo == 0 || b == 0 || c == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "depthwise_conv1d: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * c, lo.div_ceil(BLOCK), 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (b_i, c_i, l_i, k_i, s_i, p_i, lo_i) = (
        b as i32,
        c as i32,
        l as i32,
        k as i32,
        stride as i32,
        padding as i32,
        lo as i32,
    );
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let bias_ptr = bias.unwrap_or(x);
    let mut bld = stream.launch_builder(func);
    bld.arg(x)
        .arg(weight)
        .arg(bias_ptr)
        .arg(&has_bias_i)
        .arg(&mut *out)
        .arg(&b_i)
        .arg(&c_i)
        .arg(&l_i)
        .arg(&k_i)
        .arg(&s_i)
        .arg(&p_i)
        .arg(&lo_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch depthwise_conv1d: {e:?}")))?;
    }
    Ok(())
}

pub fn depthwise_conv1d_f32(
    kernels: &DepthwiseConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    out: &mut CudaSlice<f32>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
    padding: u32,
) -> Result<()> {
    depthwise_conv1d::<f32>(
        kernels,
        stream,
        x,
        weight,
        bias,
        out,
        b,
        c,
        l,
        k,
        stride,
        padding,
        DType::F32,
    )
}

pub fn depthwise_conv1d_f16(
    kernels: &DepthwiseConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: Option<&CudaSlice<f16>>,
    out: &mut CudaSlice<f16>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
    padding: u32,
) -> Result<()> {
    depthwise_conv1d::<f16>(
        kernels,
        stream,
        x,
        weight,
        bias,
        out,
        b,
        c,
        l,
        k,
        stride,
        padding,
        DType::F16,
    )
}

pub fn depthwise_conv1d_bf16(
    kernels: &DepthwiseConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    out: &mut CudaSlice<bf16>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
    padding: u32,
) -> Result<()> {
    depthwise_conv1d::<bf16>(
        kernels,
        stream,
        x,
        weight,
        bias,
        out,
        b,
        c,
        l,
        k,
        stride,
        padding,
        DType::BF16,
    )
}
