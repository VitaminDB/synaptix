use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::backend::{BinaryOp, UnaryOp};
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::{CudaBuf, Storage};

use super::compile::{compile_module, load_fn};

const MAX_RANK: usize = 8;
const BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UnaryParams {
    op_code: i32,
    scalar_a: f32,
    scalar_b: f32,
    rank: i32,
    numel: i64,
    in_offset: i64,
    dims: [i32; MAX_RANK],
    in_strides: [i32; MAX_RANK],
}
unsafe impl DeviceRepr for UnaryParams {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BinaryParams {
    op_code: i32,
    rank: i32,
    numel: i64,
    a_offset: i64,
    b_offset: i64,
    dims: [i32; MAX_RANK],
    a_strides: [i32; MAX_RANK],
    b_strides: [i32; MAX_RANK],
}
unsafe impl DeviceRepr for BinaryParams {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CastParams {
    numel: i64,
    in_offset: i64,
    rank: i32,
    dims: [i32; MAX_RANK],
    in_strides: [i32; MAX_RANK],
}
unsafe impl DeviceRepr for CastParams {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RowsParams {
    op_code: i32,
    scalar_a: f32,
    scalar_b: f32,
    rank_outer: i32,
    d: i32,
    numel: i64,
    in_offset: i64,
    dims: [i32; MAX_RANK],
    strides: [i32; MAX_RANK],
}
unsafe impl DeviceRepr for RowsParams {}

pub struct ElementwiseKernels {
    _module: Arc<CudaModule>,
    unary_f32: CudaFunction,
    unary_f16: CudaFunction,
    unary_bf16: CudaFunction,
    unary_flat_f32: CudaFunction,
    unary_flat_f16: CudaFunction,
    unary_flat_bf16: CudaFunction,
    unary_rows_f32: CudaFunction,
    unary_rows_f16: CudaFunction,
    unary_rows_bf16: CudaFunction,
    binary_f32: CudaFunction,
    binary_f16: CudaFunction,
    binary_bf16: CudaFunction,
    binary_flat_f32: CudaFunction,
    binary_flat_f16: CudaFunction,
    binary_flat_bf16: CudaFunction,
    binary_rowb_f32: CudaFunction,
    binary_rowb_f16: CudaFunction,
    binary_rowb_bf16: CudaFunction,
    binary_colb_f32: CudaFunction,
    binary_colb_f16: CudaFunction,
    binary_colb_bf16: CudaFunction,
    fma_flat_f32: CudaFunction,
    fma_flat_f16: CudaFunction,
    fma_flat_bf16: CudaFunction,
    fma_rowb_f32: CudaFunction,
    fma_rowb_f16: CudaFunction,
    fma_rowb_bf16: CudaFunction,
    mod_rowb_f32: CudaFunction,
    mod_rowb_f16: CudaFunction,
    mod_rowb_bf16: CudaFunction,
    cast_flat_f16_bf16: CudaFunction,
    cast_flat_bf16_f16: CudaFunction,
    cast_flat_f32_bf16: CudaFunction,
    cast_flat_bf16_f32: CudaFunction,
    cast_flat_f32_f16: CudaFunction,
    cast_flat_f16_f32: CudaFunction,
    cast_f32_f16: CudaFunction,
    cast_f32_bf16: CudaFunction,
    cast_f32_f64: CudaFunction,
    cast_f32_u32: CudaFunction,
    cast_f16_f32: CudaFunction,
    cast_f16_bf16: CudaFunction,
    cast_bf16_f32: CudaFunction,
    cast_bf16_f16: CudaFunction,
    cast_f64_f32: CudaFunction,
    cast_u32_f32: CudaFunction,
    cast_u32_i64: CudaFunction,
    cast_i64_u32: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ElementwiseKernels>)>>> = OnceLock::new();

impl ElementwiseKernels {
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
        let src = include_str!("../cu/kernels/elementwise.cu");
        let module = compile_module(ctx, src, "elementwise.cu")?;
        let new = Arc::new(Self {
            unary_f32: load_fn(&module, "unary_f32")?,
            unary_f16: load_fn(&module, "unary_f16")?,
            unary_bf16: load_fn(&module, "unary_bf16")?,
            unary_flat_f32: load_fn(&module, "unary_flat_f32")?,
            unary_flat_f16: load_fn(&module, "unary_flat_f16")?,
            unary_flat_bf16: load_fn(&module, "unary_flat_bf16")?,
            unary_rows_f32: load_fn(&module, "unary_rows_f32")?,
            unary_rows_f16: load_fn(&module, "unary_rows_f16")?,
            unary_rows_bf16: load_fn(&module, "unary_rows_bf16")?,
            binary_f32: load_fn(&module, "binary_f32")?,
            binary_f16: load_fn(&module, "binary_f16")?,
            binary_bf16: load_fn(&module, "binary_bf16")?,
            binary_flat_f32: load_fn(&module, "binary_flat_f32")?,
            binary_flat_f16: load_fn(&module, "binary_flat_f16")?,
            binary_flat_bf16: load_fn(&module, "binary_flat_bf16")?,
            binary_rowb_f32: load_fn(&module, "binary_rowb_f32")?,
            binary_rowb_f16: load_fn(&module, "binary_rowb_f16")?,
            binary_rowb_bf16: load_fn(&module, "binary_rowb_bf16")?,
            binary_colb_f32: load_fn(&module, "binary_colb_f32")?,
            binary_colb_f16: load_fn(&module, "binary_colb_f16")?,
            binary_colb_bf16: load_fn(&module, "binary_colb_bf16")?,
            fma_flat_f32: load_fn(&module, "fma_flat_f32")?,
            fma_flat_f16: load_fn(&module, "fma_flat_f16")?,
            fma_flat_bf16: load_fn(&module, "fma_flat_bf16")?,
            fma_rowb_f32: load_fn(&module, "fma_rowb_f32")?,
            fma_rowb_f16: load_fn(&module, "fma_rowb_f16")?,
            fma_rowb_bf16: load_fn(&module, "fma_rowb_bf16")?,
            mod_rowb_f32: load_fn(&module, "mod_rowb_f32")?,
            mod_rowb_f16: load_fn(&module, "mod_rowb_f16")?,
            mod_rowb_bf16: load_fn(&module, "mod_rowb_bf16")?,
            cast_flat_f16_bf16: load_fn(&module, "cast_flat_f16_bf16")?,
            cast_flat_bf16_f16: load_fn(&module, "cast_flat_bf16_f16")?,
            cast_flat_f32_bf16: load_fn(&module, "cast_flat_f32_bf16")?,
            cast_flat_bf16_f32: load_fn(&module, "cast_flat_bf16_f32")?,
            cast_flat_f32_f16: load_fn(&module, "cast_flat_f32_f16")?,
            cast_flat_f16_f32: load_fn(&module, "cast_flat_f16_f32")?,
            cast_f32_f16: load_fn(&module, "cast_f32_f16")?,
            cast_f32_bf16: load_fn(&module, "cast_f32_bf16")?,
            cast_f32_f64: load_fn(&module, "cast_f32_f64")?,
            cast_f32_u32: load_fn(&module, "cast_f32_u32")?,
            cast_f16_f32: load_fn(&module, "cast_f16_f32")?,
            cast_f16_bf16: load_fn(&module, "cast_f16_bf16")?,
            cast_bf16_f32: load_fn(&module, "cast_bf16_f32")?,
            cast_bf16_f16: load_fn(&module, "cast_bf16_f16")?,
            cast_f64_f32: load_fn(&module, "cast_f64_f32")?,
            cast_u32_f32: load_fn(&module, "cast_u32_f32")?,
            cast_u32_i64: load_fn(&module, "cast_u32_i64")?,
            cast_i64_u32: load_fn(&module, "cast_i64_u32")?,
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

fn pad_to<const N: usize>(src: &[i64]) -> Result<[i32; N]> {
    if src.len() > N {
        return Err(SynaptixError::Unsupported("rank > MAX_RANK"));
    }
    let mut out = [0i32; N];
    for (i, v) in src.iter().enumerate() {
        let cast: i32 = (*v)
            .try_into()
            .map_err(|_| SynaptixError::Cuda(format!("dim/stride does not fit in i32: {v}")))?;
        out[i] = cast;
    }
    Ok(out)
}

pub fn run_unary(
    kernels: &ElementwiseKernels,
    op: UnaryOp,
    src: (&Storage, &Layout),
    dst: (&mut Storage, &Layout),
) -> Result<()> {
    let (src_st, src_lo) = src;
    let (dst_st, dst_lo) = dst;
    if src_lo.dims() != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), src_lo.dims()));
    }
    if src_lo.dtype() != dst_lo.dtype() {
        return Err(SynaptixError::dtype_mismatch(
            src_lo.dtype(),
            dst_lo.dtype(),
        ));
    }
    let dtype = src_lo.dtype();
    let src_buf = src_st
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("cuda unary: src non-cuda"))?;
    let dst_buf = dst_st
        .as_cuda_mut()
        .ok_or(SynaptixError::Unsupported("cuda unary: dst non-cuda"))?;

    let (op_code, scalar_a, scalar_b) = unary_code(op);

    // Flat-путь (16Б-вектор, generic strided оставлен фоллбэком): contiguous src,
    // выровненный offset; математика та же f32 → бит-в-бит.
    if matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        && dst_lo.offset() == 0
        && src_lo.is_contiguous()
        && (src_lo.offset() * dtype.bytes_for_numel(1)) % 16 == 0
    {
        let vec_n = 16 / dtype.bytes_for_numel(1).max(2);
        let numel = src_lo.numel();
        let func = match dtype {
            DType::F32 => &kernels.unary_flat_f32,
            DType::F16 => &kernels.unary_flat_f16,
            _ => &kernels.unary_flat_bf16,
        };
        let cfg = launch_cfg(numel.div_ceil(vec_n) as i64);
        let stream = synaptix_core::device::cuda::compute_stream_for(src_buf.stream(), src_buf.ordinal())?;
        unsafe {
            return launch_unary_fast(
                &stream, func, src_buf, dst_buf, dtype,
                numel as i64, src_lo.offset() as i64, op_code, scalar_a, scalar_b, cfg,
            );
        }
    }

    // ROWS-путь: src strided, но последняя ось плотная и строки выровнены 16Б
    // (transpose+contiguous в attention) — векторное чтение строк.
    if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) && dst_lo.offset() == 0 {
        let elem = dtype.bytes_for_numel(1);
        let vec_n = 16 / elem.max(2);
        let dims = src_lo.dims();
        let ss = src_lo.strides();
        let sss = ss.as_slice();
        let rank = dims.len();
        let d = *dims.last().unwrap_or(&0);
        let rows_ok = rank >= 2
            && rank - 1 <= MAX_RANK
            && d % vec_n == 0
            && sss.last() == Some(&1)
            && (src_lo.offset() * elem) % 16 == 0
            && sss[..rank - 1].iter().all(|&s| (s.unsigned_abs() * elem) % 16 == 0 && s >= 0);
        if rows_ok {
            let numel = src_lo.numel();
            let mut pdims = [0i32; MAX_RANK];
            let mut pstr = [0i32; MAX_RANK];
            for j in 0..rank - 1 {
                pdims[j] = dims[j] as i32;
                pstr[j] = sss[j] as i32;
            }
            let params = RowsParams {
                op_code,
                scalar_a,
                scalar_b,
                rank_outer: (rank - 1) as i32,
                d: d as i32,
                numel: numel as i64,
                in_offset: src_lo.offset() as i64,
                dims: pdims,
                strides: pstr,
            };
            let func = match dtype {
                DType::F32 => &kernels.unary_rows_f32,
                DType::F16 => &kernels.unary_rows_f16,
                _ => &kernels.unary_rows_bf16,
            };
            let cfg = launch_cfg(numel.div_ceil(vec_n) as i64);
            let stream = synaptix_core::device::cuda::compute_stream_for(src_buf.stream(), src_buf.ordinal())?;
            unsafe {
                return launch_unary_rows(&stream, func, src_buf, dst_buf, dtype, params, cfg);
            }
        }
    }

    let dims_i64: Vec<i64> = src_lo.dims().iter().map(|d| *d as i64).collect();
    let strides_i64: Vec<i64> = src_lo
        .strides()
        .as_slice()
        .iter()
        .map(|s| *s as i64)
        .collect();
    let params = UnaryParams {
        op_code,
        scalar_a,
        scalar_b,
        rank: src_lo.dims().len() as i32,
        numel: src_lo.numel() as i64,
        in_offset: src_lo.offset() as i64,
        dims: pad_to::<MAX_RANK>(&dims_i64)?,
        in_strides: pad_to::<MAX_RANK>(&strides_i64)?,
    };

    let func = match dtype {
        DType::F32 => &kernels.unary_f32,
        DType::F16 => &kernels.unary_f16,
        DType::BF16 => &kernels.unary_bf16,
        _ => return Err(SynaptixError::Unsupported("cuda unary: dtype")),
    };

    let stream = synaptix_core::device::cuda::compute_stream_for(src_buf.stream(), src_buf.ordinal())?;
    let cfg = launch_cfg(params.numel);

    unsafe {
        match dtype {
            DType::F32 => launch_unary::<f32>(&stream, func, src_buf, dst_buf, params, cfg)?,
            DType::F16 => launch_unary::<half::f16>(&stream, func, src_buf, dst_buf, params, cfg)?,
            DType::BF16 => {
                launch_unary::<half::bf16>(&stream, func, src_buf, dst_buf, params, cfg)?
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Launch flat-unary (in/out contiguous, 16Б-вектор).
#[allow(clippy::too_many_arguments)]
unsafe fn launch_unary_fast(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    src_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    dtype: DType,
    numel: i64,
    in_off: i64,
    op_code: i32,
    sa: f32,
    sb: f32,
    cfg: LaunchConfig,
) -> Result<()> {
    unsafe fn go<T: cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
        func: &CudaFunction,
        src_buf: &CudaBuf,
        dst_buf: &mut CudaBuf,
        numel: i64,
        in_off: i64,
        op_code: i32,
        sa: f32,
        sb: f32,
        cfg: LaunchConfig,
    ) -> Result<()> {
        let elem = std::mem::size_of::<T>();
        let src_v = src_buf.slice().as_view();
        let src_t = src_v
            .transmute::<T>(src_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("unary fast: transmute src".into()))?;
        let mut dst_v = dst_buf.slice_mut().as_view_mut();
        let mut dst_t = dst_v
            .transmute_mut::<T>(dst_v.len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("unary fast: transmute dst".into()))?;
        let mut builder = stream.launch_builder(func);
        builder.arg(&src_t).arg(&mut dst_t).arg(&numel).arg(&in_off).arg(&op_code).arg(&sa).arg(&sb);
        builder
            .launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch unary fast: {e:?}")))?;
        Ok(())
    }
    match dtype {
        DType::F32 => go::<f32>(stream, func, src_buf, dst_buf, numel, in_off, op_code, sa, sb, cfg),
        DType::F16 => go::<half::f16>(stream, func, src_buf, dst_buf, numel, in_off, op_code, sa, sb, cfg),
        DType::BF16 => go::<half::bf16>(stream, func, src_buf, dst_buf, numel, in_off, op_code, sa, sb, cfg),
        _ => Err(SynaptixError::Unsupported("unary fast: dtype")),
    }
}

/// Launch rows-unary (strided внешние оси, плотная последняя).
unsafe fn launch_unary_rows(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    src_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    dtype: DType,
    params: RowsParams,
    cfg: LaunchConfig,
) -> Result<()> {
    unsafe fn go<T: cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
        func: &CudaFunction,
        src_buf: &CudaBuf,
        dst_buf: &mut CudaBuf,
        params: RowsParams,
        cfg: LaunchConfig,
    ) -> Result<()> {
        let elem = std::mem::size_of::<T>();
        let src_v = src_buf.slice().as_view();
        let src_t = src_v
            .transmute::<T>(src_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("unary rows: transmute src".into()))?;
        let mut dst_v = dst_buf.slice_mut().as_view_mut();
        let mut dst_t = dst_v
            .transmute_mut::<T>(dst_v.len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("unary rows: transmute dst".into()))?;
        let mut builder = stream.launch_builder(func);
        builder.arg(&src_t).arg(&mut dst_t).arg(&params);
        builder
            .launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch unary rows: {e:?}")))?;
        Ok(())
    }
    match dtype {
        DType::F32 => go::<f32>(stream, func, src_buf, dst_buf, params, cfg),
        DType::F16 => go::<half::f16>(stream, func, src_buf, dst_buf, params, cfg),
        DType::BF16 => go::<half::bf16>(stream, func, src_buf, dst_buf, params, cfg),
        _ => Err(SynaptixError::Unsupported("unary rows: dtype")),
    }
}

unsafe fn launch_unary<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    src_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    params: UnaryParams,
    cfg: LaunchConfig,
) -> Result<()> {
    let src_v = src_buf.slice().as_view();
    let src_t = src_v
        .transmute::<T>(src_buf.slice().len() / std::mem::size_of::<T>())
        .ok_or_else(|| SynaptixError::Cuda("unary: transmute src".into()))?;
    let mut dst_v = dst_buf.slice_mut().as_view_mut();
    let mut dst_t = dst_v
        .transmute_mut::<T>(dst_v.len() / std::mem::size_of::<T>())
        .ok_or_else(|| SynaptixError::Cuda("unary: transmute dst".into()))?;
    let mut builder = stream.launch_builder(func);
    builder.arg(&src_t).arg(&mut dst_t).arg(&params);
    builder
        .launch(cfg)
        .map_err(|e| SynaptixError::Cuda(format!("launch unary: {e:?}")))?;
    Ok(())
}

pub fn run_binary(
    kernels: &ElementwiseKernels,
    op: BinaryOp,
    a: (&Storage, &Layout),
    b: (&Storage, &Layout),
    dst: (&mut Storage, &Layout),
) -> Result<()> {
    let (a_st, a_lo) = a;
    let (b_st, b_lo) = b;
    let (dst_st, dst_lo) = dst;
    if a_lo.dtype() != b_lo.dtype() || a_lo.dtype() != dst_lo.dtype() {
        return Err(SynaptixError::dtype_mismatch(a_lo.dtype(), dst_lo.dtype()));
    }
    if a_lo.dims() != dst_lo.dims() || b_lo.dims() != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), a_lo.dims()));
    }
    let dtype = a_lo.dtype();
    let a_buf = a_st
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("cuda binary: a non-cuda"))?;
    let b_buf = b_st
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("cuda binary: b non-cuda"))?;
    let dst_buf = dst_st
        .as_cuda_mut()
        .ok_or(SynaptixError::Unsupported("cuda binary: dst non-cuda"))?;

    let op_code = binary_code(op);

    // ── Быстрые пути (16Б-вектор, та же f32-математика → бит-в-бит с generic):
    // FLAT — все contiguous same-shape; ROWB — a/dst contiguous [.., D], b
    // broadcast-строка [D] (внешние strides 0, внутренний 1). Generic strided
    // (~60-90GB/s) оставлен фоллбэком для остального.
    if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) && dst_lo.offset() == 0 {
        let elem = dtype.bytes_for_numel(1);
        let vec_n: usize = 16 / elem.max(2); // f32→4, f16/bf16→8
        let numel = dst_lo.numel();
        let aligned = |lo: &Layout| (lo.offset() * elem) % 16 == 0;
        let func_flat = match dtype {
            DType::F32 => &kernels.binary_flat_f32,
            DType::F16 => &kernels.binary_flat_f16,
            _ => &kernels.binary_flat_bf16,
        };
        let func_rowb = match dtype {
            DType::F32 => &kernels.binary_rowb_f32,
            DType::F16 => &kernels.binary_rowb_f16,
            _ => &kernels.binary_rowb_bf16,
        };
        let d = *dst_lo.dims().last().unwrap_or(&0);
        let b_strides = b_lo.strides();
        let bs = b_strides.as_slice();
        // строка-broadcast: внутренняя ось плотная, внешние не двигаются (stride 0
        // ЛИБО ось размера 1 — у них stride произволен и не влияет).
        let b_is_row = d > 0
            && d % vec_n == 0
            && *b_lo.dims().last().unwrap() == d
            && bs.last() == Some(&1)
            && bs[..bs.len() - 1]
                .iter()
                .zip(dst_lo.dims()[..bs.len() - 1].iter())
                .all(|(&s, &dim)| s == 0 || dim == 1);
        if a_lo.is_contiguous() && aligned(a_lo) && aligned(b_lo) {
            let cfg = launch_cfg(numel.div_ceil(vec_n) as i64);
            let stream = synaptix_core::device::cuda::compute_stream_for(a_buf.stream(), a_buf.ordinal())?;
            if b_lo.is_contiguous() && b_lo.dims() == a_lo.dims() {
                unsafe {
                    return launch_binary_fast(
                        &stream, func_flat, a_buf, b_buf, dst_buf, dtype,
                        numel as i64, a_lo.offset() as i64, b_lo.offset() as i64, None, op_code, cfg,
                    );
                }
            }
            if b_is_row {
                unsafe {
                    return launch_binary_fast(
                        &stream, func_rowb, a_buf, b_buf, dst_buf, dtype,
                        numel as i64, a_lo.offset() as i64, b_lo.offset() as i64,
                        Some(d as i32), op_code, cfg,
                    );
                }
            }
            // COLB: b broadcast по последней оси ([..,G,1]→[..,G,D]) и плотный по
            // внешним (b_idx = i/D). Гейт: last-stride 0, внешние оси row-major
            // плотные (dim==1 → stride произволен).
            let b_is_col = d > 0 && d % vec_n == 0 && bs.last() == Some(&0) && {
                let dims = dst_lo.dims();
                let mut expected = 1isize;
                let mut ok = true;
                for j in (0..bs.len() - 1).rev() {
                    if dims[j] != 1 {
                        if bs[j] != expected {
                            ok = false;
                            break;
                        }
                        expected *= dims[j] as isize;
                    }
                }
                ok
            };
            if b_is_col {
                let func_colb = match dtype {
                    DType::F32 => &kernels.binary_colb_f32,
                    DType::F16 => &kernels.binary_colb_f16,
                    _ => &kernels.binary_colb_bf16,
                };
                unsafe {
                    return launch_binary_fast(
                        &stream, func_colb, a_buf, b_buf, dst_buf, dtype,
                        numel as i64, a_lo.offset() as i64, b_lo.offset() as i64,
                        Some(d as i32), op_code, cfg,
                    );
                }
            }
        }
    }

    let dims_i64: Vec<i64> = dst_lo.dims().iter().map(|d| *d as i64).collect();
    let a_strides_i64: Vec<i64> = a_lo
        .strides()
        .as_slice()
        .iter()
        .map(|s| *s as i64)
        .collect();
    let b_strides_i64: Vec<i64> = b_lo
        .strides()
        .as_slice()
        .iter()
        .map(|s| *s as i64)
        .collect();
    let params = BinaryParams {
        op_code,
        rank: dst_lo.dims().len() as i32,
        numel: dst_lo.numel() as i64,
        a_offset: a_lo.offset() as i64,
        b_offset: b_lo.offset() as i64,
        dims: pad_to::<MAX_RANK>(&dims_i64)?,
        a_strides: pad_to::<MAX_RANK>(&a_strides_i64)?,
        b_strides: pad_to::<MAX_RANK>(&b_strides_i64)?,
    };

    let func = match dtype {
        DType::F32 => &kernels.binary_f32,
        DType::F16 => &kernels.binary_f16,
        DType::BF16 => &kernels.binary_bf16,
        _ => return Err(SynaptixError::Unsupported("cuda binary: dtype")),
    };

    let stream = synaptix_core::device::cuda::compute_stream_for(a_buf.stream(), a_buf.ordinal())?;
    let cfg = launch_cfg(params.numel);

    unsafe {
        match dtype {
            DType::F32 => launch_binary::<f32>(&stream, func, a_buf, b_buf, dst_buf, params, cfg)?,
            DType::F16 => {
                launch_binary::<half::f16>(&stream, func, a_buf, b_buf, dst_buf, params, cfg)?
            }
            DType::BF16 => {
                launch_binary::<half::bf16>(&stream, func, a_buf, b_buf, dst_buf, params, cfg)?
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Launch быстрых binary-ядер (flat/rowb): `d=Some` → rowb (доп. аргумент D).
#[allow(clippy::too_many_arguments)]
/// Fused ternary: `FmaFlat` out=x+y*g (формы равны), `FmaRowb` g=[D]-строка,
/// `ModRowb` out=x*(1+s)+sh (s/sh=[D]-строки). Раунды повторяют decomposed →
/// бит-в-бит. Жёсткие гейты contiguous/выравнивания — иначе Err (вызывающий
/// падает на decomposed-цепочку).
pub enum TernaryFusedKind {
    FmaFlat,
    FmaRowb,
    ModRowb,
}

pub fn run_ternary_fused(
    kernels: &ElementwiseKernels,
    kind: TernaryFusedKind,
    x: (&Storage, &Layout),
    b: (&Storage, &Layout),
    c: (&Storage, &Layout),
    dst: (&mut Storage, &Layout),
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (b_st, b_lo) = b;
    let (c_st, c_lo) = c;
    let (dst_st, dst_lo) = dst;
    let dtype = x_lo.dtype();
    if b_lo.dtype() != dtype || c_lo.dtype() != dtype || dst_lo.dtype() != dtype {
        return Err(SynaptixError::Unsupported("ternary fused: dtype"));
    }
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("ternary fused: dtype"));
    }
    if x_lo.dims() != dst_lo.dims() || dst_lo.offset() != 0 {
        return Err(SynaptixError::Unsupported("ternary fused: формы/offset"));
    }
    let elem = dtype.bytes_for_numel(1);
    let vec_n: usize = 16 / elem.max(2);
    let numel = dst_lo.numel();
    let d = *dst_lo.dims().last().unwrap_or(&0);
    let aligned = |lo: &Layout| (lo.offset() * elem) % 16 == 0;
    let dense = |lo: &Layout| lo.strides().is_contiguous(lo.shape());
    let row_ok = |lo: &Layout| dense(lo) && lo.numel() == d && aligned(lo);
    if !x_lo.is_contiguous() || !aligned(x_lo) || numel % vec_n != 0 {
        return Err(SynaptixError::Unsupported("ternary fused: x"));
    }
    let (func, need_d) = match (&kind, dtype) {
        (TernaryFusedKind::FmaFlat, DType::F32) => (&kernels.fma_flat_f32, false),
        (TernaryFusedKind::FmaFlat, DType::F16) => (&kernels.fma_flat_f16, false),
        (TernaryFusedKind::FmaFlat, _) => (&kernels.fma_flat_bf16, false),
        (TernaryFusedKind::FmaRowb, DType::F32) => (&kernels.fma_rowb_f32, true),
        (TernaryFusedKind::FmaRowb, DType::F16) => (&kernels.fma_rowb_f16, true),
        (TernaryFusedKind::FmaRowb, _) => (&kernels.fma_rowb_bf16, true),
        (TernaryFusedKind::ModRowb, DType::F32) => (&kernels.mod_rowb_f32, true),
        (TernaryFusedKind::ModRowb, DType::F16) => (&kernels.mod_rowb_f16, true),
        (TernaryFusedKind::ModRowb, _) => (&kernels.mod_rowb_bf16, true),
    };
    match kind {
        TernaryFusedKind::FmaFlat => {
            if !(b_lo.is_contiguous() && aligned(b_lo) && b_lo.dims() == dst_lo.dims()
                && c_lo.is_contiguous() && aligned(c_lo) && c_lo.dims() == dst_lo.dims())
            {
                return Err(SynaptixError::Unsupported("fma flat: b/c"));
            }
        }
        TernaryFusedKind::FmaRowb => {
            if !(b_lo.is_contiguous() && aligned(b_lo) && b_lo.dims() == dst_lo.dims()
                && row_ok(c_lo) && d > 0 && d % vec_n == 0)
            {
                return Err(SynaptixError::Unsupported("fma rowb: b/c"));
            }
        }
        TernaryFusedKind::ModRowb => {
            if !(row_ok(b_lo) && row_ok(c_lo) && d > 0 && d % vec_n == 0) {
                return Err(SynaptixError::Unsupported("mod rowb: s/sh"));
            }
        }
    }
    let x_buf = x_st.as_cuda().ok_or(SynaptixError::Unsupported("ternary fused: x cuda"))?;
    let b_buf = b_st.as_cuda().ok_or(SynaptixError::Unsupported("ternary fused: b cuda"))?;
    let c_buf = c_st.as_cuda().ok_or(SynaptixError::Unsupported("ternary fused: c cuda"))?;
    let dst_buf = dst_st.as_cuda_mut().ok_or(SynaptixError::Unsupported("ternary fused: dst cuda"))?;
    let cfg = launch_cfg(numel.div_ceil(vec_n) as i64);
    let stream = synaptix_core::device::cuda::compute_stream_for(x_buf.stream(), x_buf.ordinal())?;
    unsafe {
        launch_ternary_fast(
            &stream, func, x_buf, b_buf, c_buf, dst_buf, dtype,
            numel as i64, x_lo.offset() as i64, b_lo.offset() as i64, c_lo.offset() as i64,
            need_d.then_some(d as i32), cfg,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_ternary_fast(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    x_buf: &CudaBuf,
    b_buf: &CudaBuf,
    c_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    dtype: DType,
    numel: i64,
    x_off: i64,
    b_off: i64,
    c_off: i64,
    d: Option<i32>,
    cfg: LaunchConfig,
) -> Result<()> {
    #[allow(clippy::too_many_arguments)]
    unsafe fn go<T: cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
        func: &CudaFunction,
        x_buf: &CudaBuf,
        b_buf: &CudaBuf,
        c_buf: &CudaBuf,
        dst_buf: &mut CudaBuf,
        numel: i64,
        x_off: i64,
        b_off: i64,
        c_off: i64,
        d: Option<i32>,
        cfg: LaunchConfig,
    ) -> Result<()> {
        let elem = std::mem::size_of::<T>();
        let x_v = x_buf.slice().as_view();
        let x_t = x_v
            .transmute::<T>(x_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("ternary fast: transmute x".into()))?;
        let b_v = b_buf.slice().as_view();
        let b_t = b_v
            .transmute::<T>(b_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("ternary fast: transmute b".into()))?;
        let c_v = c_buf.slice().as_view();
        let c_t = c_v
            .transmute::<T>(c_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("ternary fast: transmute c".into()))?;
        let mut dst_v = dst_buf.slice_mut().as_view_mut();
        let mut dst_t = dst_v
            .transmute_mut::<T>(dst_v.len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("ternary fast: transmute dst".into()))?;
        let mut builder = stream.launch_builder(func);
        builder.arg(&x_t).arg(&b_t).arg(&c_t).arg(&mut dst_t)
            .arg(&numel).arg(&x_off).arg(&b_off).arg(&c_off);
        if let Some(dd) = &d {
            builder.arg(dd);
        }
        builder
            .launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch ternary fast: {e:?}")))?;
        Ok(())
    }
    match dtype {
        DType::F32 => go::<f32>(stream, func, x_buf, b_buf, c_buf, dst_buf, numel, x_off, b_off, c_off, d, cfg),
        DType::F16 => go::<half::f16>(stream, func, x_buf, b_buf, c_buf, dst_buf, numel, x_off, b_off, c_off, d, cfg),
        DType::BF16 => go::<half::bf16>(stream, func, x_buf, b_buf, c_buf, dst_buf, numel, x_off, b_off, c_off, d, cfg),
        _ => Err(SynaptixError::Unsupported("ternary fast: dtype")),
    }
}

unsafe fn launch_binary_fast(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    a_buf: &CudaBuf,
    b_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    dtype: DType,
    numel: i64,
    a_off: i64,
    b_off: i64,
    d: Option<i32>,
    op_code: i32,
    cfg: LaunchConfig,
) -> Result<()> {
    unsafe fn go<T: cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
        func: &CudaFunction,
        a_buf: &CudaBuf,
        b_buf: &CudaBuf,
        dst_buf: &mut CudaBuf,
        numel: i64,
        a_off: i64,
        b_off: i64,
        d: Option<i32>,
        op_code: i32,
        cfg: LaunchConfig,
    ) -> Result<()> {
        let elem = std::mem::size_of::<T>();
        let a_v = a_buf.slice().as_view();
        let a_t = a_v
            .transmute::<T>(a_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("binary fast: transmute a".into()))?;
        let b_v = b_buf.slice().as_view();
        let b_t = b_v
            .transmute::<T>(b_buf.slice().len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("binary fast: transmute b".into()))?;
        let mut dst_v = dst_buf.slice_mut().as_view_mut();
        let mut dst_t = dst_v
            .transmute_mut::<T>(dst_v.len() / elem)
            .ok_or_else(|| SynaptixError::Cuda("binary fast: transmute dst".into()))?;
        let mut builder = stream.launch_builder(func);
        builder.arg(&a_t).arg(&b_t).arg(&mut dst_t).arg(&numel).arg(&a_off).arg(&b_off);
        if let Some(dd) = &d {
            builder.arg(dd);
        }
        builder.arg(&op_code);
        builder
            .launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch binary fast: {e:?}")))?;
        Ok(())
    }
    match dtype {
        DType::F32 => go::<f32>(stream, func, a_buf, b_buf, dst_buf, numel, a_off, b_off, d, op_code, cfg),
        DType::F16 => go::<half::f16>(stream, func, a_buf, b_buf, dst_buf, numel, a_off, b_off, d, op_code, cfg),
        DType::BF16 => go::<half::bf16>(stream, func, a_buf, b_buf, dst_buf, numel, a_off, b_off, d, op_code, cfg),
        _ => Err(SynaptixError::Unsupported("binary fast: dtype")),
    }
}

unsafe fn launch_binary<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<cudarc::driver::CudaStream>,
    func: &CudaFunction,
    a_buf: &CudaBuf,
    b_buf: &CudaBuf,
    dst_buf: &mut CudaBuf,
    params: BinaryParams,
    cfg: LaunchConfig,
) -> Result<()> {
    let elem = std::mem::size_of::<T>();
    let a_v = a_buf.slice().as_view();
    let a_t = a_v
        .transmute::<T>(a_buf.slice().len() / elem)
        .ok_or_else(|| SynaptixError::Cuda("binary: transmute a".into()))?;
    let b_v = b_buf.slice().as_view();
    let b_t = b_v
        .transmute::<T>(b_buf.slice().len() / elem)
        .ok_or_else(|| SynaptixError::Cuda("binary: transmute b".into()))?;
    let mut dst_v = dst_buf.slice_mut().as_view_mut();
    let mut dst_t = dst_v
        .transmute_mut::<T>(dst_v.len() / elem)
        .ok_or_else(|| SynaptixError::Cuda("binary: transmute dst".into()))?;
    let mut builder = stream.launch_builder(func);
    builder.arg(&a_t).arg(&b_t).arg(&mut dst_t).arg(&params);
    builder
        .launch(cfg)
        .map_err(|e| SynaptixError::Cuda(format!("launch binary: {e:?}")))?;
    Ok(())
}

pub fn run_cast(
    kernels: &ElementwiseKernels,
    src: (&Storage, &Layout),
    dst: (&mut Storage, &Layout),
) -> Result<()> {
    let (src_st, src_lo) = src;
    let (dst_st, dst_lo) = dst;
    if src_lo.dims() != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), src_lo.dims()));
    }
    let src_buf = src_st
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("cuda cast: src non-cuda"))?;
    let dst_buf = dst_st
        .as_cuda_mut()
        .ok_or(SynaptixError::Unsupported("cuda cast: dst non-cuda"))?;

    // Flat-путь (8 эл/поток векторно, маршрут in→f32→out бит-эквивалентен
    // generic in→f64→out для пар {f32,f16,bf16}): src/dst contiguous,
    // offset выровнен 16Б. Главный клиент — BF16↔F16 вокруг квант-GEMM.
    {
        let s_dt = src_lo.dtype();
        let d_dt = dst_lo.dtype();
        let func_flat = match (s_dt, d_dt) {
            (DType::F16, DType::BF16) => Some(&kernels.cast_flat_f16_bf16),
            (DType::BF16, DType::F16) => Some(&kernels.cast_flat_bf16_f16),
            (DType::F32, DType::BF16) => Some(&kernels.cast_flat_f32_bf16),
            (DType::BF16, DType::F32) => Some(&kernels.cast_flat_bf16_f32),
            (DType::F32, DType::F16) => Some(&kernels.cast_flat_f32_f16),
            (DType::F16, DType::F32) => Some(&kernels.cast_flat_f16_f32),
            _ => None,
        };
        if let Some(func) = func_flat {
            let elem_in = s_dt.bytes_for_numel(1);
            if dst_lo.offset() == 0
                && src_lo.is_contiguous()
                && (src_lo.offset() * elem_in) % 16 == 0
            {
                let numel = src_lo.numel();
                let cfg = launch_cfg(numel.div_ceil(8) as i64);
                let stream = synaptix_core::device::cuda::compute_stream_for(src_buf.stream(), src_buf.ordinal())?;
                let in_off: i64 = src_lo.offset() as i64;
                let numel_ll: i64 = numel as i64;
                let src_v = src_buf.slice().as_view();
                let mut dst_v = dst_buf.slice_mut().as_view_mut();
                let mut builder = stream.launch_builder(func);
                builder.arg(&src_v).arg(&mut dst_v).arg(&numel_ll).arg(&in_off);
                unsafe {
                    builder
                        .launch(cfg)
                        .map_err(|e| SynaptixError::Cuda(format!("launch cast flat: {e:?}")))?;
                }
                return Ok(());
            }
        }
    }

    let dims_i64: Vec<i64> = src_lo.dims().iter().map(|d| *d as i64).collect();
    let strides_i64: Vec<i64> = src_lo
        .strides()
        .as_slice()
        .iter()
        .map(|s| *s as i64)
        .collect();
    let params = CastParams {
        numel: src_lo.numel() as i64,
        in_offset: src_lo.offset() as i64,
        rank: src_lo.dims().len() as i32,
        dims: pad_to::<MAX_RANK>(&dims_i64)?,
        in_strides: pad_to::<MAX_RANK>(&strides_i64)?,
    };

    let stream = synaptix_core::device::cuda::compute_stream_for(src_buf.stream(), src_buf.ordinal())?;
    let cfg = launch_cfg(params.numel);

    let pair = (src_lo.dtype(), dst_lo.dtype());
    let func = match pair {
        (DType::F32, DType::F16) => &kernels.cast_f32_f16,
        (DType::F32, DType::BF16) => &kernels.cast_f32_bf16,
        (DType::F32, DType::F64) => &kernels.cast_f32_f64,
        (DType::F32, DType::U32) => &kernels.cast_f32_u32,
        (DType::F16, DType::F32) => &kernels.cast_f16_f32,
        (DType::F16, DType::BF16) => &kernels.cast_f16_bf16,
        (DType::BF16, DType::F32) => &kernels.cast_bf16_f32,
        (DType::BF16, DType::F16) => &kernels.cast_bf16_f16,
        (DType::F64, DType::F32) => &kernels.cast_f64_f32,
        (DType::U32, DType::F32) => &kernels.cast_u32_f32,
        (DType::U32, DType::I64) => &kernels.cast_u32_i64,
        (DType::I64, DType::U32) => &kernels.cast_i64_u32,
        _ => return Err(SynaptixError::Unsupported("cuda cast: dtype pair")),
    };

    unsafe {
        let src_p = src_buf.slice().as_view();
        let mut dst_p = dst_buf.slice_mut().as_view_mut();
        let mut builder = stream.launch_builder(func);
        builder.arg(&src_p).arg(&mut dst_p).arg(&params);
        builder
            .launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch cast: {e:?}")))?;
    }
    Ok(())
}

fn unary_code(op: UnaryOp) -> (i32, f32, f32) {
    match op {
        UnaryOp::Neg => (0, 0.0, 0.0),
        UnaryOp::Abs => (1, 0.0, 0.0),
        UnaryOp::Sqrt => (2, 0.0, 0.0),
        UnaryOp::Sqr => (3, 0.0, 0.0),
        UnaryOp::Recip => (4, 0.0, 0.0),
        UnaryOp::Exp => (5, 0.0, 0.0),
        UnaryOp::Log => (6, 0.0, 0.0),
        UnaryOp::Sin => (7, 0.0, 0.0),
        UnaryOp::Cos => (8, 0.0, 0.0),
        UnaryOp::Silu => (9, 0.0, 0.0),
        UnaryOp::GeluTanh => (10, 0.0, 0.0),
        UnaryOp::Tanh => (11, 0.0, 0.0),
        UnaryOp::GeluExact => (22, 0.0, 0.0),
        UnaryOp::Clamp(lo, hi) => (12, lo, hi),
        UnaryOp::Powf(e) => (13, e, 0.0),
        UnaryOp::Affine(mul, add) => (14, mul, add),
        UnaryOp::Erf => (15, 0.0, 0.0),
        UnaryOp::Sigmoid => (16, 0.0, 0.0),
        UnaryOp::Relu => (17, 0.0, 0.0),
        UnaryOp::Relu2 => (18, 0.0, 0.0),
        UnaryOp::LeakyRelu(a) => (19, a, 0.0),
        UnaryOp::Sign => (20, 0.0, 0.0),
        UnaryOp::StepGtZero => (21, 0.0, 0.0),
        UnaryOp::Round => (23, 0.0, 0.0),
        UnaryOp::Floor => (24, 0.0, 0.0),
        UnaryOp::Ceil => (25, 0.0, 0.0),
    }
}

fn binary_code(op: BinaryOp) -> i32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
    }
}
