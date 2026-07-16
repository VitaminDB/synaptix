//! Depthwise conv1d / conv_transpose1d (groups == C) — настоящие ядра вместо
//! Rust-цикла по каналам (C×K микро-launch'ей; вокодер LTX жёг 14s на 5с аудио).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::{CudaBuf, Storage};

use super::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;

pub struct Dwconv1dKernels {
    _module: Arc<CudaModule>,
    conv_f32: CudaFunction,
    conv_f16: CudaFunction,
    conv_bf16: CudaFunction,
    convt_f32: CudaFunction,
    convt_f16: CudaFunction,
    convt_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Dwconv1dKernels>)>>> = OnceLock::new();

impl Dwconv1dKernels {
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
        let src = include_str!("../cu/kernels/dwconv1d.cu");
        let module = compile_module(ctx, src, "dwconv1d.cu")?;
        let new = Arc::new(Self {
            conv_f32: load_fn(&module, "dwconv1d_f32")?,
            conv_f16: load_fn(&module, "dwconv1d_f16")?,
            conv_bf16: load_fn(&module, "dwconv1d_bf16")?,
            convt_f32: load_fn(&module, "dwconvt1d_f32")?,
            convt_f16: load_fn(&module, "dwconvt1d_f16")?,
            convt_bf16: load_fn(&module, "dwconvt1d_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `input [B,C,L]`, `weight [C,1,K]`, `bias [C]?`, `out [B,C,Lo]` — все
/// contiguous, offset 0. `transpose=false`: stride-conv (вход предпаддан либо
/// `pad`); `true`: convT полной длины `(L-1)·s+K` (кроп у вызывающего).
#[allow(clippy::too_many_arguments)]
pub fn run_dwconv1d(
    kernels: &Dwconv1dKernels,
    input: (&Storage, &Layout),
    weight: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    stride: usize,
    pad: usize,
    transpose: bool,
) -> Result<()> {
    let (x_st, x_lo) = input;
    let (w_st, w_lo) = weight;
    let (o_st, o_lo) = out;
    let dtype = x_lo.dtype();
    if w_lo.dtype() != dtype || o_lo.dtype() != dtype {
        return Err(SynaptixError::Unsupported("dwconv1d: dtype mismatch"));
    }
    if !x_lo.is_contiguous() || x_lo.offset() != 0 || !w_lo.is_contiguous() || w_lo.offset() != 0 {
        return Err(SynaptixError::Unsupported("dwconv1d: non-contiguous"));
    }
    let (c, l) = (x_lo.dims()[1], x_lo.dims()[2]);
    let k = w_lo.dims()[2];
    let lo = o_lo.dims()[2];
    let total = o_lo.numel() as i64;
    let x_buf = x_st.as_cuda().ok_or(SynaptixError::Unsupported("dwconv1d: x non-cuda"))?;
    let w_buf = w_st.as_cuda().ok_or(SynaptixError::Unsupported("dwconv1d: w non-cuda"))?;
    let o_buf = o_st.as_cuda_mut().ok_or(SynaptixError::Unsupported("dwconv1d: out non-cuda"))?;
    let b_buf = match &bias {
        Some((b_st, b_lo)) => {
            if b_lo.dtype() != dtype || !b_lo.is_contiguous() || b_lo.offset() != 0 {
                return Err(SynaptixError::Unsupported("dwconv1d: bias layout"));
            }
            Some(b_st.as_cuda().ok_or(SynaptixError::Unsupported("dwconv1d: bias non-cuda"))?)
        }
        None => None,
    };
    let func = match (transpose, dtype) {
        (false, DType::F32) => &kernels.conv_f32,
        (false, DType::F16) => &kernels.conv_f16,
        (false, DType::BF16) => &kernels.conv_bf16,
        (true, DType::F32) => &kernels.convt_f32,
        (true, DType::F16) => &kernels.convt_f16,
        (true, DType::BF16) => &kernels.convt_bf16,
        _ => return Err(SynaptixError::Unsupported("dwconv1d: dtype")),
    };
    let grid = ((total as u64).div_ceil(BLOCK as u64)).min(u32::MAX as u64) as u32;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
    let stream = x_buf.stream().clone();
    let (ci, li, ki, loi, si, pi) = (c as i32, l as i32, k as i32, lo as i32, stride as i32, pad as i32);
    let x_v = x_buf.slice().as_view();
    let w_v = w_buf.slice().as_view();
    let mut o_v = o_buf.slice_mut().as_view_mut();
    // bias: нулевой указатель при отсутствии — передаём пустой вью того же буфера x
    // нельзя; используем отдельные ветки launch.
    let mut builder = stream.launch_builder(func);
    builder.arg(&x_v).arg(&w_v);
    let b_v;
    let null_ptr: u64 = 0;
    if let Some(b) = b_buf {
        b_v = b.slice().as_view();
        builder.arg(&b_v);
    } else {
        builder.arg(&null_ptr);
    }
    builder.arg(&mut o_v).arg(&ci).arg(&li).arg(&ki).arg(&loi);
    builder.arg(&si);
    if !transpose {
        builder.arg(&pi);
    }
    let total_ll: i64 = total;
    builder.arg(&total_ll);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch dwconv1d: {e:?}")))?;
    }
    Ok(())
}
