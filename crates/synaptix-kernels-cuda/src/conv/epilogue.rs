//! Fused conv2d-эпилог: `out2d[M, Cout]` (NHWC-flat from im2col-GEMM) +
//! optional `bias[Cout]` → `out[B, Cout, H, W]` (NCHW). Заменяет
//! `broadcast_add(bias) + permute([0,3,1,2]).contiguous()` (два прохода
//! по памяти → один). Помогает каждому conv2d_im2col.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct ConvEpilogueKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ConvEpilogueKernels>)>>> = OnceLock::new();

impl ConvEpilogueKernels {
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
        let src = include_str!("../cu/fused/conv/conv_epilogue.cu");
        let module = compile_module(ctx, src, "conv_epilogue.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "conv_epilogue_f32")?,
            f16: load_fn(&module, "conv_epilogue_f16")?,
            bf16: load_fn(&module, "conv_epilogue_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv_epilogue_u8(
    kernels: &ConvEpilogueKernels,
    stream: &Arc<CudaStream>,
    out2d: &CudaSlice<u8>,
    out2d_off: usize,
    bias: Option<(&CudaSlice<u8>, usize)>,
    residual: Option<(&CudaSlice<u8>, usize)>,
    temb_bc: Option<(&CudaSlice<u8>, usize)>,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    dtype: DType,
) -> Result<()> {
    let n_total = (b as usize) * (c as usize) * (h as usize) * (w as usize);
    if n_total == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let n_c = c as usize;
    const BLOCK: u32 = 256;
    let grid = ((n_total as u64).div_ceil(BLOCK as u64).min(65535) as u32).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let has_res_i: i32 = if residual.is_some() { 1 } else { 0 };
    let has_temb_i: i32 = if temb_bc.is_some() { 1 } else { 0 };
    let n_temb = (b as usize) * (c as usize);
    let b_i = b as i32;
    let c_i = c as i32;
    let h_i = h as i32;
    let w_i = w as i32;

    macro_rules! go {
        ($ty:ty, $func:expr) => {{
            let in_v = unsafe {
                out2d
                    .slice(out2d_off..out2d_off + n_total * esz)
                    .transmute::<$ty>(n_total)
                    .ok_or_else(|| SynaptixError::Cuda("conv_epilogue: transmute out2d".into()))?
            };
            let bias_v = match bias {
                Some((bb, bo)) => unsafe {
                    bb.slice(bo..bo + n_c * esz)
                        .transmute::<$ty>(n_c)
                        .ok_or_else(|| {
                            SynaptixError::Cuda("conv_epilogue: transmute bias".into())
                        })?
                },
                None => unsafe {
                    out2d
                        .slice(out2d_off..out2d_off + n_total * esz)
                        .transmute::<$ty>(n_total)
                        .ok_or_else(|| {
                            SynaptixError::Cuda("conv_epilogue: dummy bias view".into())
                        })?
                },
            };
            let res_v = match residual {
                Some((rb, ro)) => unsafe {
                    rb.slice(ro..ro + n_total * esz)
                        .transmute::<$ty>(n_total)
                        .ok_or_else(|| SynaptixError::Cuda("conv_epilogue: transmute res".into()))?
                },
                None => unsafe {
                    out2d
                        .slice(out2d_off..out2d_off + n_total * esz)
                        .transmute::<$ty>(n_total)
                        .ok_or_else(|| {
                            SynaptixError::Cuda("conv_epilogue: dummy res view".into())
                        })?
                },
            };
            let temb_v = match temb_bc {
                Some((tb, to_off)) => unsafe {
                    tb.slice(to_off..to_off + n_temb * esz)
                        .transmute::<$ty>(n_temb)
                        .ok_or_else(|| {
                            SynaptixError::Cuda("conv_epilogue: transmute temb".into())
                        })?
                },
                None => unsafe {
                    out2d
                        .slice(out2d_off..out2d_off + n_temb * esz)
                        .transmute::<$ty>(n_temb)
                        .ok_or_else(|| {
                            SynaptixError::Cuda("conv_epilogue: dummy temb view".into())
                        })?
                },
            };
            let mut out_s = out.slice_mut(out_off..out_off + n_total * esz);
            let mut out_v = unsafe {
                out_s
                    .transmute_mut::<$ty>(n_total)
                    .ok_or_else(|| SynaptixError::Cuda("conv_epilogue: transmute out".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&in_v)
                .arg(&bias_v)
                .arg(&has_bias_i)
                .arg(&res_v)
                .arg(&has_res_i)
                .arg(&temb_v)
                .arg(&has_temb_i)
                .arg(&mut out_v)
                .arg(&b_i)
                .arg(&c_i)
                .arg(&h_i)
                .arg(&w_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch conv_epilogue: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "conv_epilogue: dtype {other:?}"
            )))
        }
    }
    Ok(())
}
