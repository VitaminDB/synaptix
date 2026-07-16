//! Causal Conv3d (LTX-VAE style): causal-padding по T, VALID по H/W. Один thread =
//! один output voxel. F32-accumulator всегда (даже для F16/BF16).

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

const W_BLOCK: u32 = 32;

pub struct Conv3dCausalKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Conv3dCausalKernels>)>>> = OnceLock::new();

impl Conv3dCausalKernels {
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
        let src = include_str!("../cu/conv/conv3d_causal.cu");
        let module = compile_module(ctx, src, "conv3d_causal.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "conv3d_causal_f32")?,
            f16: load_fn(&module, "conv3d_causal_f16")?,
            bf16: load_fn(&module, "conv3d_causal_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `T_out = (T - 1) / stride + 1` (после causal-padding слева на `Kt-1` нулей).
pub fn t_out(t: usize, stride: usize) -> usize {
    let s = stride.max(1);
    (t.saturating_sub(1)) / s + 1
}

/// `H_out = (H - Kh) / stride + 1` (VALID по пространству).
pub fn spatial_out(in_size: usize, kernel: usize, stride: usize) -> usize {
    let s = stride.max(1);
    if in_size < kernel {
        return 0;
    }
    (in_size - kernel) / s + 1
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_causal<T: DeviceRepr>(
    kernels: &Conv3dCausalKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    bias: Option<&CudaSlice<T>>,
    output: &mut CudaSlice<T>,
    b: u32,
    c_in: u32,
    t: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    dtype: DType,
) -> Result<()> {
    let stride_u = stride.max(1);
    let to = t_out(t as usize, stride_u as usize) as u32;
    let ho = spatial_out(h as usize, kh as usize, stride_u as usize) as u32;
    let wo = spatial_out(w as usize, kw as usize, stride_u as usize) as u32;
    if to == 0 || ho == 0 || wo == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "conv3d_causal: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * c_out, to * ho, wo.div_ceil(W_BLOCK)),
        block_dim: (W_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let t_i = t as i32;
    let h_i = h as i32;
    let w_i = w as i32;
    let c_out_i = c_out as i32;
    let kt_i = kt as i32;
    let kh_i = kh as i32;
    let kw_i = kw as i32;
    let stride_i = stride_u as i32;
    let to_i = to as i32;
    let ho_i = ho as i32;
    let wo_i = wo as i32;
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let bias_ptr = bias.unwrap_or(input);
    let mut bld = stream.launch_builder(func);
    bld.arg(input)
        .arg(weight)
        .arg(bias_ptr)
        .arg(&has_bias_i)
        .arg(&mut *output)
        .arg(&b_i)
        .arg(&c_in_i)
        .arg(&t_i)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&c_out_i)
        .arg(&kt_i)
        .arg(&kh_i)
        .arg(&kw_i)
        .arg(&stride_i)
        .arg(&to_i)
        .arg(&ho_i)
        .arg(&wo_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch conv3d_causal: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_causal_f32(
    kernels: &Conv3dCausalKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    output: &mut CudaSlice<f32>,
    b: u32,
    c_in: u32,
    t: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    stride: u32,
) -> Result<()> {
    conv3d_causal::<f32>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        t,
        h,
        w,
        c_out,
        kt,
        kh,
        kw,
        stride,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_causal_f16(
    kernels: &Conv3dCausalKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: Option<&CudaSlice<f16>>,
    output: &mut CudaSlice<f16>,
    b: u32,
    c_in: u32,
    t: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    stride: u32,
) -> Result<()> {
    conv3d_causal::<f16>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        t,
        h,
        w,
        c_out,
        kt,
        kh,
        kw,
        stride,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_causal_bf16(
    kernels: &Conv3dCausalKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    output: &mut CudaSlice<bf16>,
    b: u32,
    c_in: u32,
    t: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    stride: u32,
) -> Result<()> {
    conv3d_causal::<bf16>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        t,
        h,
        w,
        c_out,
        kt,
        kh,
        kw,
        stride,
        DType::BF16,
    )
}
