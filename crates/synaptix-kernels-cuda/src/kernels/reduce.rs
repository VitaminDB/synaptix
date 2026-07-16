use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::backend::ReduceOp;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::{CudaBuf, Storage};

use super::compile::{compile_module, load_fn};

const MAX_RANK: usize = 8;
const BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReduceParams {
    op_code: i32,
    rank: i32,
    n_reduce: i32,
    _pad: i32,
    out_numel: i64,
    inner_size: i64,
    in_offset: i64,
    dims_out: [i32; MAX_RANK],
    dims_red: [i32; MAX_RANK],
    red_axes: [i32; MAX_RANK],
    strides_in: [i32; MAX_RANK],
}
unsafe impl DeviceRepr for ReduceParams {}

pub struct ReduceKernels {
    _module: Arc<CudaModule>,
    reduce_f32: CudaFunction,
    reduce_f16: CudaFunction,
    reduce_bf16: CudaFunction,
    argmax_f32: CudaFunction,
    argmax_f16: CudaFunction,
    argmax_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ReduceKernels>)>>> = OnceLock::new();

impl ReduceKernels {
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
        let src = include_str!("../cu/kernels/reduce.cu");
        let module = compile_module(ctx, src, "reduce.cu")?;
        let new = Arc::new(Self {
            reduce_f32: load_fn(&module, "reduce_f32")?,
            reduce_f16: load_fn(&module, "reduce_f16")?,
            reduce_bf16: load_fn(&module, "reduce_bf16")?,
            argmax_f32: load_fn(&module, "argmax_f32")?,
            argmax_f16: load_fn(&module, "argmax_f16")?,
            argmax_bf16: load_fn(&module, "argmax_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn launch_cfg(numel: i64) -> LaunchConfig {
    let n = numel.max(1) as u64;
    let grid = (n + (BLOCK as u64) - 1) / (BLOCK as u64);
    LaunchConfig {
        grid_dim: (grid.min(u32::MAX as u64) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn pad_i32(src: &[i32]) -> Result<[i32; MAX_RANK]> {
    if src.len() > MAX_RANK {
        return Err(SynaptixError::Unsupported("rank > MAX_RANK"));
    }
    let mut out = [0i32; MAX_RANK];
    for (i, v) in src.iter().enumerate() {
        out[i] = *v;
    }
    Ok(out)
}

pub fn run_reduce(
    kernels: &ReduceKernels,
    op: ReduceOp,
    src: (&Storage, &Layout),
    dst: (&mut Storage, &Layout),
    dims_reduced: &[usize],
) -> Result<()> {
    let (src_st, src_lo) = src;
    let (dst_st, dst_lo) = dst;
    let src_buf = src_st
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("cuda reduce: src non-cuda"))?;
    let dst_buf = dst_st
        .as_cuda_mut()
        .ok_or(SynaptixError::Unsupported("cuda reduce: dst non-cuda"))?;

    let rank_in = src_lo.dims().len();
    let mut is_reduced = vec![false; rank_in];
    for &d in dims_reduced {
        is_reduced[d] = true;
    }
    let dims_in = src_lo.dims();
    let mut dims_out_kd = Vec::with_capacity(rank_in);
    for (i, &d) in dims_in.iter().enumerate() {
        dims_out_kd.push(if is_reduced[i] { 1i32 } else { d as i32 });
    }
    let mut dims_red: Vec<i32> = Vec::with_capacity(dims_reduced.len());
    let mut red_axes: Vec<i32> = Vec::with_capacity(dims_reduced.len());
    let mut inner_size: i64 = 1;
    for &d in dims_reduced {
        red_axes.push(d as i32);
        dims_red.push(dims_in[d] as i32);
        inner_size *= dims_in[d] as i64;
    }
    let out_numel: i64 = dims_out_kd.iter().map(|x| *x as i64).product();
    let strides_in: Vec<i32> = src_lo
        .strides()
        .as_slice()
        .iter()
        .map(|s| *s as i32)
        .collect();

    let params = ReduceParams {
        op_code: op_code(op),
        rank: rank_in as i32,
        n_reduce: dims_reduced.len() as i32,
        _pad: 0,
        out_numel,
        inner_size,
        in_offset: src_lo.offset() as i64,
        dims_out: pad_i32(&dims_out_kd)?,
        dims_red: pad_i32(&dims_red)?,
        red_axes: pad_i32(&red_axes)?,
        strides_in: pad_i32(&strides_in)?,
    };

    let stream = src_buf.stream().clone();
    let cfg = launch_cfg(out_numel);
    let dtype = src_lo.dtype();
    let is_argmax = matches!(op, ReduceOp::ArgMax);

    if is_argmax && dst_lo.dtype() != DType::U32 {
        return Err(SynaptixError::dtype_mismatch(DType::U32, dst_lo.dtype()));
    }
    if !is_argmax && dst_lo.dtype() != dtype {
        return Err(SynaptixError::dtype_mismatch(dtype, dst_lo.dtype()));
    }

    let func = if is_argmax {
        match dtype {
            DType::F32 => &kernels.argmax_f32,
            DType::F16 => &kernels.argmax_f16,
            DType::BF16 => &kernels.argmax_bf16,
            _ => return Err(SynaptixError::Unsupported("cuda argmax: dtype")),
        }
    } else {
        match dtype {
            DType::F32 => &kernels.reduce_f32,
            DType::F16 => &kernels.reduce_f16,
            DType::BF16 => &kernels.reduce_bf16,
            _ => return Err(SynaptixError::Unsupported("cuda reduce: dtype")),
        }
    };

    unsafe {
        if is_argmax {
            launch_argmax(&stream, func, src_buf, dst_buf, params, cfg, dtype)?;
        } else {
            match dtype {
                DType::F32 => launch_reduce::<f32>(&stream, func, src_buf, dst_buf, params, cfg)?,
                DType::F16 => {
                    launch_reduce::<half::f16>(&stream, func, src_buf, dst_buf, params, cfg)?
                }
                DType::BF16 => {
                    launch_reduce::<half::bf16>(&stream, func, src_buf, dst_buf, params, cfg)?
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

unsafe fn launch_reduce<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    src_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    params: ReduceParams,
    cfg: LaunchConfig,
) -> Result<()> {
    let elem = std::mem::size_of::<T>();
    let src_v = src_buf.slice().as_view();
    let src_t = src_v
        .transmute::<T>(src_buf.slice().len() / elem)
        .ok_or_else(|| SynaptixError::Cuda("reduce: transmute src".into()))?;
    let mut dst_v = dst_buf.slice_mut().as_view_mut();
    let mut dst_t = dst_v
        .transmute_mut::<T>(dst_v.len() / elem)
        .ok_or_else(|| SynaptixError::Cuda("reduce: transmute dst".into()))?;
    let mut builder = stream.launch_builder(func);
    builder.arg(&src_t).arg(&mut dst_t).arg(&params);
    builder
        .launch(cfg)
        .map_err(|e| SynaptixError::Cuda(format!("launch reduce: {e:?}")))?;
    Ok(())
}

unsafe fn launch_argmax(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    src_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    params: ReduceParams,
    cfg: LaunchConfig,
    dtype: DType,
) -> Result<()> {
    let src_v = src_buf.slice().as_view();
    let mut dst_v = dst_buf.slice_mut().as_view_mut();
    let mut dst_t = dst_v
        .transmute_mut::<u32>(dst_v.len() / 4)
        .ok_or_else(|| SynaptixError::Cuda("argmax: transmute dst".into()))?;
    match dtype {
        DType::F32 => {
            let src_t = src_v
                .transmute::<f32>(src_buf.slice().len() / 4)
                .ok_or_else(|| SynaptixError::Cuda("argmax: transmute src f32".into()))?;
            let mut builder = stream.launch_builder(func);
            builder.arg(&src_t).arg(&mut dst_t).arg(&params);
            builder
                .launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch argmax f32: {e:?}")))?;
        }
        DType::F16 => {
            let src_t = src_v
                .transmute::<half::f16>(src_buf.slice().len() / 2)
                .ok_or_else(|| SynaptixError::Cuda("argmax: transmute src f16".into()))?;
            let mut builder = stream.launch_builder(func);
            builder.arg(&src_t).arg(&mut dst_t).arg(&params);
            builder
                .launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch argmax f16: {e:?}")))?;
        }
        DType::BF16 => {
            let src_t = src_v
                .transmute::<half::bf16>(src_buf.slice().len() / 2)
                .ok_or_else(|| SynaptixError::Cuda("argmax: transmute src bf16".into()))?;
            let mut builder = stream.launch_builder(func);
            builder.arg(&src_t).arg(&mut dst_t).arg(&params);
            builder
                .launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch argmax bf16: {e:?}")))?;
        }
        _ => return Err(SynaptixError::Unsupported("argmax dtype")),
    }
    Ok(())
}

fn op_code(op: ReduceOp) -> i32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::ArgMax => 3,
    }
}
