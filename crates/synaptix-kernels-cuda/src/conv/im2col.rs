//! im2col CUDA kernel: input `[B,C_in,H,W]` → col `[M,K]`,
//! `M = B*H_out*W_out`, `K = C_in*Kh*Kw`. Питает conv2d-через-GEMM
//! (`out[M,C_out] = col[M,K] @ Wᵀ[K,C_out]`) — на порядки быстрее direct-conv
//! на больших каналах/spatial (VAE/UNet), т.к. GEMM идёт через cutlass.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct Im2colKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Im2colKernels>)>>> = OnceLock::new();

impl Im2colKernels {
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
        let src = include_str!("../cu/conv/im2col.cu");
        let module = compile_module(ctx, src, "im2col.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "im2col_f32")?,
            f16: load_fn(&module, "im2col_f16")?,
            bf16: load_fn(&module, "im2col_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// u8-вход (для Backend::im2col). `input`/`col` — untyped storage-слайсы (offset в
/// БАЙТАХ), транзмутируются по `dtype`. `col` — `[m_count, K]` (логическая строка
/// `r` ↔ глобальная `m = m_offset + r`); `b/h/w` дают полный размер `input`.
#[allow(clippy::too_many_arguments)]
pub fn run_im2col_u8(
    kernels: &Im2colKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u8>,
    input_off: usize,
    col: &mut CudaSlice<u8>,
    col_off: usize,
    b: u32,
    c_in: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    h_out: u32,
    w_out: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    m_offset: u64,
    m_count: u64,
    dtype: DType,
) -> Result<()> {
    let kcols = (c_in as usize) * (kh as usize) * (kw as usize);
    let total = (m_count as usize) * kcols;
    if total == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let n_in = (b as usize) * (c_in as usize) * (h as usize) * (w as usize);

    const BLOCK: u32 = 256;
    let grid = ((total as u64).div_ceil(BLOCK as u64).min(65535) as u32).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let c_in_i = c_in as i32;
    let h_i = h as i32;
    let w_i = w as i32;
    let kh_i = kh as i32;
    let kw_i = kw as i32;
    let ho_i = h_out as i32;
    let wo_i = w_out as i32;
    let sh_i = stride_h as i32;
    let sw_i = stride_w as i32;
    let ph_i = pad_h as i32;
    let pw_i = pad_w as i32;
    let moff_i = m_offset as i64;
    let mcnt_i = m_count as i64;

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let in_v = unsafe {
                input
                    .slice(input_off..input_off + n_in * esz)
                    .transmute::<$t>(n_in)
                    .ok_or_else(|| SynaptixError::Cuda("im2col: transmute input".into()))?
            };
            let mut col_s = col.slice_mut(col_off..col_off + total * esz);
            let mut col_v = unsafe {
                col_s
                    .transmute_mut::<$t>(total)
                    .ok_or_else(|| SynaptixError::Cuda("im2col: transmute col".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&in_v)
                .arg(&mut col_v)
                .arg(&c_in_i)
                .arg(&h_i)
                .arg(&w_i)
                .arg(&kh_i)
                .arg(&kw_i)
                .arg(&ho_i)
                .arg(&wo_i)
                .arg(&sh_i)
                .arg(&sw_i)
                .arg(&ph_i)
                .arg(&pw_i)
                .arg(&moff_i)
                .arg(&mcnt_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch im2col: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "im2col: unsupported dtype {other:?}"
            )))
        }
    }
    Ok(())
}
