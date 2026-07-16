//! Fast 4D permute NCHW↔NHWC через shmem-tile. Заменяет generic
//! `permute().contiguous()` для входов implicit-GEMM (без strided-copy).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct NchwNhwcKernels {
    _module: Arc<CudaModule>,
    nchw_to_nhwc_f32: CudaFunction,
    nchw_to_nhwc_f16: CudaFunction,
    nchw_to_nhwc_bf16: CudaFunction,
    nhwc_to_nchw_f32: CudaFunction,
    nhwc_to_nchw_f16: CudaFunction,
    nhwc_to_nchw_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<NchwNhwcKernels>)>>> = OnceLock::new();

impl NchwNhwcKernels {
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
        let src = include_str!("../cu/conv/nchw_nhwc.cu");
        let module = compile_module(ctx, src, "nchw_nhwc.cu")?;
        let new = Arc::new(Self {
            nchw_to_nhwc_f32: load_fn(&module, "nchw_to_nhwc_f32")?,
            nchw_to_nhwc_f16: load_fn(&module, "nchw_to_nhwc_f16")?,
            nchw_to_nhwc_bf16: load_fn(&module, "nchw_to_nhwc_bf16")?,
            nhwc_to_nchw_f32: load_fn(&module, "nhwc_to_nchw_f32")?,
            nhwc_to_nchw_f16: load_fn(&module, "nhwc_to_nchw_f16")?,
            nhwc_to_nchw_bf16: load_fn(&module, "nhwc_to_nchw_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

const TILE: u32 = 32;

#[allow(clippy::too_many_arguments)]
pub fn run_nchw_to_nhwc_u8(
    kernels: &NchwNhwcKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    dtype: DType,
) -> Result<()> {
    run_transpose(
        kernels, stream, src, src_off, dst, dst_off, b, c, h, w, dtype, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_nhwc_to_nchw_u8(
    kernels: &NchwNhwcKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    dtype: DType,
) -> Result<()> {
    run_transpose(
        kernels, stream, src, src_off, dst, dst_off, b, c, h, w, dtype, false,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_transpose(
    kernels: &NchwNhwcKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    dtype: DType,
    to_nhwc: bool,
) -> Result<()> {
    let n = (b as usize) * (c as usize) * (h as usize) * (w as usize);
    if n == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let hw = h * w;
    let grid_x = hw.div_ceil(TILE);
    let grid_y = c.div_ceil(TILE);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, b),
        block_dim: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    };
    let c_i = c as i32;
    let h_i = h as i32;
    let w_i = w as i32;

    macro_rules! go {
        ($ty:ty, $func:expr) => {{
            let s_v = unsafe {
                src.slice(src_off..src_off + n * esz)
                    .transmute::<$ty>(n)
                    .ok_or_else(|| SynaptixError::Cuda("nchw_nhwc: transmute src".into()))?
            };
            let mut d_s = dst.slice_mut(dst_off..dst_off + n * esz);
            let mut d_v = unsafe {
                d_s.transmute_mut::<$ty>(n)
                    .ok_or_else(|| SynaptixError::Cuda("nchw_nhwc: transmute dst".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&s_v).arg(&mut d_v).arg(&c_i).arg(&h_i).arg(&w_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch nchw_nhwc: {e:?}")))?;
            }
        }};
    }

    match (dtype, to_nhwc) {
        (DType::F32, true) => go!(f32, &kernels.nchw_to_nhwc_f32),
        (DType::F16, true) => go!(f16, &kernels.nchw_to_nhwc_f16),
        (DType::BF16, true) => go!(bf16, &kernels.nchw_to_nhwc_bf16),
        (DType::F32, false) => go!(f32, &kernels.nhwc_to_nchw_f32),
        (DType::F16, false) => go!(f16, &kernels.nhwc_to_nchw_f16),
        (DType::BF16, false) => go!(bf16, &kernels.nhwc_to_nchw_bf16),
        (other, _) => return Err(SynaptixError::Cuda(format!("nchw_nhwc: dtype {other:?}"))),
    }
    Ok(())
}
