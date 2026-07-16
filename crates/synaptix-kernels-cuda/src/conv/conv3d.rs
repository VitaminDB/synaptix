//! Conv3d direct CUDA kernel (один thread = один output voxel).
//!
//! Naive baseline вариант — по тому же паттерну что [`super::conv1d`] и
//! [`super::conv2d`]. Production fast path = im2col + cuBLASLt GEMM.

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

pub struct Conv3dKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Conv3dKernels>)>>> = OnceLock::new();

impl Conv3dKernels {
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
        let src = include_str!("../cu/conv/conv3d.cu");
        let module = compile_module(ctx, src, "conv3d.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "conv3d_direct_f32")?,
            f16: load_fn(&module, "conv3d_direct_f16")?,
            bf16: load_fn(&module, "conv3d_direct_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

pub fn out_dim(in_size: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    let padded = in_size + 2 * padding;
    if padded < kernel {
        return 0;
    }
    (padded - kernel) / stride + 1
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d<T: DeviceRepr>(
    kernels: &Conv3dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    bias: Option<&CudaSlice<T>>,
    output: &mut CudaSlice<T>,
    b: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dtype: DType,
) -> Result<()> {
    let d_out = out_dim(d as usize, kd as usize, sd as usize, pd as usize) as u32;
    let h_out = out_dim(h as usize, kh as usize, sh as usize, ph as usize) as u32;
    let w_out = out_dim(w as usize, kw as usize, sw as usize, pw as usize) as u32;
    if d_out == 0 || h_out == 0 || w_out == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "conv3d: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * c_out * d_out, h_out, w_out.div_ceil(W_BLOCK)),
        block_dim: (W_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let d_i = d as i32;
    let h_i = h as i32;
    let w_i = w as i32;
    let c_out_i = c_out as i32;
    let kd_i = kd as i32;
    let kh_i = kh as i32;
    let kw_i = kw as i32;
    let sd_i = sd as i32;
    let sh_i = sh as i32;
    let sw_i = sw as i32;
    let pd_i = pd as i32;
    let ph_i = ph as i32;
    let pw_i = pw as i32;
    let do_i = d_out as i32;
    let ho_i = h_out as i32;
    let wo_i = w_out as i32;
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
        .arg(&d_i)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&c_out_i)
        .arg(&kd_i)
        .arg(&kh_i)
        .arg(&kw_i)
        .arg(&sd_i)
        .arg(&sh_i)
        .arg(&sw_i)
        .arg(&pd_i)
        .arg(&ph_i)
        .arg(&pw_i)
        .arg(&do_i)
        .arg(&ho_i)
        .arg(&wo_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch conv3d: {e:?}")))?;
    }
    Ok(())
}

/// u8-обёртка (как [`super::conv2d::run_conv2d_u8`]): raw-слайсы + byte-offset +
/// dtype-диспатч. Питает `Backend::conv3d`. dilation=1 (ядро без dilation).
#[allow(clippy::too_many_arguments)]
pub fn run_conv3d_u8(
    kernels: &Conv3dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u8>,
    input_off: usize,
    weight: &CudaSlice<u8>,
    weight_off: usize,
    bias: Option<(&CudaSlice<u8>, usize)>,
    output: &mut CudaSlice<u8>,
    output_off: usize,
    b: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dtype: DType,
) -> Result<()> {
    let d_out = out_dim(d as usize, kd as usize, sd as usize, pd as usize) as u32;
    let h_out = out_dim(h as usize, kh as usize, sh as usize, ph as usize) as u32;
    let w_out = out_dim(w as usize, kw as usize, sw as usize, pw as usize) as u32;
    if d_out == 0 || h_out == 0 || w_out == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let n_in = (b * c_in * d * h * w) as usize;
    let n_w = (c_out * c_in * kd * kh * kw) as usize;
    let n_out = (b * c_out * d_out * h_out * w_out) as usize;
    let n_b = c_out as usize;

    let cfg = LaunchConfig {
        grid_dim: (b * c_out * d_out, h_out, w_out.div_ceil(W_BLOCK)),
        block_dim: (W_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let (b_i, c_in_i, d_i, h_i, w_i) = (b as i32, c_in as i32, d as i32, h as i32, w as i32);
    let (c_out_i, kd_i, kh_i, kw_i) = (c_out as i32, kd as i32, kh as i32, kw as i32);
    let (sd_i, sh_i, sw_i) = (sd as i32, sh as i32, sw as i32);
    let (pd_i, ph_i, pw_i) = (pd as i32, ph as i32, pw as i32);
    let (do_i, ho_i, wo_i) = (d_out as i32, h_out as i32, w_out as i32);

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let in_v = unsafe {
                input
                    .slice(input_off..input_off + n_in * esz)
                    .transmute::<$t>(n_in)
                    .ok_or_else(|| SynaptixError::Cuda("conv3d: transmute input".into()))?
            };
            let w_v = unsafe {
                weight
                    .slice(weight_off..weight_off + n_w * esz)
                    .transmute::<$t>(n_w)
                    .ok_or_else(|| SynaptixError::Cuda("conv3d: transmute weight".into()))?
            };
            // bias=None → дублируем input-view (ядро не читает при has_bias=0).
            let bias_v = match bias {
                Some((bs, bo)) => unsafe {
                    bs.slice(bo..bo + n_b * esz)
                        .transmute::<$t>(n_b)
                        .ok_or_else(|| SynaptixError::Cuda("conv3d: transmute bias".into()))?
                },
                None => unsafe {
                    input
                        .slice(input_off..input_off + n_in * esz)
                        .transmute::<$t>(n_in)
                        .ok_or_else(|| SynaptixError::Cuda("conv3d: transmute bias-dummy".into()))?
                },
            };
            let mut out_s = output.slice_mut(output_off..output_off + n_out * esz);
            let mut out_v = unsafe {
                out_s
                    .transmute_mut::<$t>(n_out)
                    .ok_or_else(|| SynaptixError::Cuda("conv3d: transmute output".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&in_v).arg(&w_v).arg(&bias_v).arg(&has_bias_i).arg(&mut out_v)
                .arg(&b_i).arg(&c_in_i).arg(&d_i).arg(&h_i).arg(&w_i)
                .arg(&c_out_i).arg(&kd_i).arg(&kh_i).arg(&kw_i)
                .arg(&sd_i).arg(&sh_i).arg(&sw_i).arg(&pd_i).arg(&ph_i).arg(&pw_i)
                .arg(&do_i).arg(&ho_i).arg(&wo_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch conv3d_u8: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "conv3d_u8: unsupported dtype {other:?}"
            )))
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_f32(
    kernels: &Conv3dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    output: &mut CudaSlice<f32>,
    b: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
) -> Result<()> {
    conv3d::<f32>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        d,
        h,
        w,
        c_out,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_f16(
    kernels: &Conv3dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: Option<&CudaSlice<f16>>,
    output: &mut CudaSlice<f16>,
    b: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
) -> Result<()> {
    conv3d::<f16>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        d,
        h,
        w,
        c_out,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_bf16(
    kernels: &Conv3dKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    output: &mut CudaSlice<bf16>,
    b: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
) -> Result<()> {
    conv3d::<bf16>(
        kernels,
        stream,
        input,
        weight,
        bias,
        output,
        b,
        c_in,
        d,
        h,
        w,
        c_out,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        DType::BF16,
    )
}
