//! Standalone 1D contiguous activations (F32/F16/BF16) + bias-add variants.
//!
//! Дополняют strided `UnaryOp` opcodes в `elementwise.cu`. Эти kernels
//! ассуминуют contiguous 1D layout — без unravel + per-axis stride, что
//! даёт ~30-50% speedup на больших векторах. Bias-add варианты покрывают
//! паттерн `FC.bias + activation` за один pass.

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

const BLOCK: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluExact,
    GeluTanh,
    QuickGelu,
    Softplus,
    Mish,
    Softsign,
}

pub struct ActivationsKernels {
    _module: Arc<CudaModule>,
    silu: [CudaFunction; 3],
    gelu_exact: [CudaFunction; 3],
    gelu_tanh: [CudaFunction; 3],
    quick_gelu: [CudaFunction; 3],
    softplus: [CudaFunction; 3],
    mish: [CudaFunction; 3],
    softsign: [CudaFunction; 3],
    swish_beta: [CudaFunction; 3],
    snake: [CudaFunction; 3],
    bias_silu: [CudaFunction; 3],
    bias_gelu_tanh: [CudaFunction; 3],
    bias_relu: [CudaFunction; 3],
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ActivationsKernels>)>>> = OnceLock::new();

fn load_triplet(module: &Arc<CudaModule>, prefix: &str) -> Result<[CudaFunction; 3]> {
    Ok([
        load_fn(module, &format!("{prefix}_f32"))?,
        load_fn(module, &format!("{prefix}_f16"))?,
        load_fn(module, &format!("{prefix}_bf16"))?,
    ])
}

fn dtype_idx(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(SynaptixError::Unsupported(
            "activations: dtype must be F32/F16/BF16",
        )),
    }
}

impl ActivationsKernels {
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
        let src = include_str!("../cu/elementwise/activations.cu");
        let module = compile_module(ctx, src, "activations.cu")?;
        let new = Arc::new(Self {
            silu: load_triplet(&module, "silu_act")?,
            gelu_exact: load_triplet(&module, "gelu_exact_act")?,
            gelu_tanh: load_triplet(&module, "gelu_tanh_act")?,
            quick_gelu: load_triplet(&module, "quick_gelu_act")?,
            softplus: load_triplet(&module, "softplus_act")?,
            mish: load_triplet(&module, "mish_act")?,
            softsign: load_triplet(&module, "softsign_act")?,
            swish_beta: load_triplet(&module, "swish_beta_act")?,
            snake: load_triplet(&module, "snake_act")?,
            bias_silu: load_triplet(&module, "bias_silu")?,
            bias_gelu_tanh: load_triplet(&module, "bias_gelu_tanh")?,
            bias_relu: load_triplet(&module, "bias_relu")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn fn_of(&self, act: Activation) -> &[CudaFunction; 3] {
        match act {
            Activation::Silu => &self.silu,
            Activation::GeluExact => &self.gelu_exact,
            Activation::GeluTanh => &self.gelu_tanh,
            Activation::QuickGelu => &self.quick_gelu,
            Activation::Softplus => &self.softplus,
            Activation::Mish => &self.mish,
            Activation::Softsign => &self.softsign,
        }
    }
}

fn grid_for(n: u32) -> u32 {
    let blocks = n.div_ceil(BLOCK);
    blocks.min(65535)
}

pub fn run<T: DeviceRepr>(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    n: u32,
    act: Activation,
    dtype: DType,
) -> Result<()> {
    let idx = dtype_idx(dtype)?;
    let func = &kernels.fn_of(act)[idx];
    let cfg = LaunchConfig {
        grid_dim: (grid_for(n), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(&mut *y).arg(&n_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch activation: {e:?}")))?;
    }
    Ok(())
}

pub fn run_swish_beta<T: DeviceRepr>(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    n: u32,
    beta: f32,
    dtype: DType,
) -> Result<()> {
    let idx = dtype_idx(dtype)?;
    let func = &kernels.swish_beta[idx];
    let cfg = LaunchConfig {
        grid_dim: (grid_for(n), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(&mut *y).arg(&n_i).arg(&beta);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch swish_beta: {e:?}")))?;
    }
    Ok(())
}

/// Fused Snake (untyped byte-срезы): `y[i] = x[i] + sin(a[c]*x[i])^2 * binv[c]`,
/// channel `c = (i / t_len) % chans`. `a`/`binv` — предвычисленные per-channel
/// `[C]` f32 (a = exp(alpha), binv = 1/(exp(beta)+eps)). `x`/`y` — dtype `dtype`.
#[allow(clippy::too_many_arguments)]
pub fn run_snake_u8(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    a: &CudaSlice<u8>,
    a_off: usize,
    binv: &CudaSlice<u8>,
    binv_off: usize,
    y: &mut CudaSlice<u8>,
    y_off: usize,
    n: usize,
    chans: u32,
    t_len: u32,
    dtype: DType,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let idx = dtype_idx(dtype)?;
    let func = &kernels.snake[idx];
    let esz = (dtype.size_in_bits() / 8) as usize;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(n as u32), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, c_i, t_i) = (n as i32, chans as i32, t_len as i32);
    let cc = chans as usize;
    let a_v = unsafe {
        a.slice(a_off..a_off + cc * 4)
            .transmute::<f32>(cc)
            .ok_or_else(|| SynaptixError::Cuda("snake: transmute a".into()))?
    };
    let binv_v = unsafe {
        binv.slice(binv_off..binv_off + cc * 4)
            .transmute::<f32>(cc)
            .ok_or_else(|| SynaptixError::Cuda("snake: transmute binv".into()))?
    };
    macro_rules! go {
        ($ty:ty) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + n * esz)
                    .transmute::<$ty>(n)
                    .ok_or_else(|| SynaptixError::Cuda("snake: transmute x".into()))?
            };
            let mut y_s = y.slice_mut(y_off..y_off + n * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$ty>(n)
                    .ok_or_else(|| SynaptixError::Cuda("snake: transmute y".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&x_v).arg(&a_v).arg(&binv_v).arg(&mut y_v).arg(&n_i).arg(&c_i).arg(&t_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch snake: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32),
        DType::F16 => go!(f16),
        DType::BF16 => go!(bf16),
        other => return Err(SynaptixError::Cuda(format!("snake: dtype {other:?}"))),
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasActivation {
    Silu,
    GeluTanh,
    Relu,
}

impl ActivationsKernels {
    fn bias_fn(&self, act: BiasActivation) -> &[CudaFunction; 3] {
        match act {
            BiasActivation::Silu => &self.bias_silu,
            BiasActivation::GeluTanh => &self.bias_gelu_tanh,
            BiasActivation::Relu => &self.bias_relu,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_bias_act<T: DeviceRepr>(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    bias: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    rows: u32,
    cols: u32,
    act: BiasActivation,
    dtype: DType,
) -> Result<()> {
    let idx = dtype_idx(dtype)?;
    let func = &kernels.bias_fn(act)[idx];
    let total = rows * cols;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(total), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let r_i = rows as i32;
    let c_i = cols as i32;
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(bias).arg(&mut *y).arg(&r_i).arg(&c_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch bias_act: {e:?}")))?;
    }
    Ok(())
}

pub fn run_f32(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    n: u32,
    act: Activation,
) -> Result<()> {
    run::<f32>(kernels, stream, x, y, n, act, DType::F32)
}

pub fn run_f16(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    y: &mut CudaSlice<f16>,
    n: u32,
    act: Activation,
) -> Result<()> {
    run::<f16>(kernels, stream, x, y, n, act, DType::F16)
}

pub fn run_bf16(
    kernels: &ActivationsKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    n: u32,
    act: Activation,
) -> Result<()> {
    run::<bf16>(kernels, stream, x, y, n, act, DType::BF16)
}
