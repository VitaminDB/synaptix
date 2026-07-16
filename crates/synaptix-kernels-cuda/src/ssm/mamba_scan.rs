//! Mamba selective state-space scan (S6).
//!
//! Sequential по time dim (L), parallel по (batch, dim). One block = one
//! (b, d) pair; block_dim = N (state size, typically 16). Each thread держит
//! h[n] в register. Reduction `sum(C * h)` через warp shuffle.

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

pub struct MambaScanKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<MambaScanKernels>)>>> = OnceLock::new();

impl MambaScanKernels {
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
        let src = include_str!("../cu/fused/ssm/mamba_scan.cu");
        let module = compile_module(ctx, src, "mamba_scan.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "mamba_scan_f32")?,
            f16: load_fn(&module, "mamba_scan_f16")?,
            bf16: load_fn(&module, "mamba_scan_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// Selective scan forward.
/// - `u`     shape `(B, L, D)` row-major
/// - `delta` shape `(B, L, D)`
/// - `a`     shape `(D, N)` — state transition (обычно negative)
/// - `b_in`  shape `(B, L, N)` — selective input projection
/// - `c_in`  shape `(B, L, N)` — selective output projection
/// - `d_skip` shape `(D,)` optional (None ⟹ no skip)
/// - `y`     shape `(B, L, D)` row-major
///
/// `n_state` (N) ограничен 32 (один warp для reduction).
#[allow(clippy::too_many_arguments)]
pub fn scan<T: DeviceRepr>(
    kernels: &MambaScanKernels,
    stream: &Arc<CudaStream>,
    u: &CudaSlice<T>,
    delta: &CudaSlice<T>,
    a: &CudaSlice<T>,
    b_in: &CudaSlice<T>,
    c_in: &CudaSlice<T>,
    d_skip: Option<&CudaSlice<T>>,
    y: &mut CudaSlice<T>,
    b: u32,
    l: u32,
    d: u32,
    n_state: u32,
    dtype: DType,
) -> Result<()> {
    if n_state > 32 {
        return Err(SynaptixError::Cuda(format!(
            "mamba_scan: n_state={n_state} > 32 (warp reduction limit)"
        )));
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "mamba_scan: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * d, 1, 1),
        block_dim: (32, 1, 1), // запускаем фиксированно 32 thread'а (warp), реальный N учитывается через guard внутри kernel'я.
        shared_mem_bytes: 0,
    };
    let has_d_i: i32 = if d_skip.is_some() { 1 } else { 0 };
    let b_i = b as i32;
    let l_i = l as i32;
    let d_i = d as i32;
    let n_i = n_state as i32;
    let d_skip_ptr = d_skip.unwrap_or(u);
    let mut bld = stream.launch_builder(func);
    bld.arg(u)
        .arg(delta)
        .arg(a)
        .arg(b_in)
        .arg(c_in)
        .arg(d_skip_ptr)
        .arg(&has_d_i)
        .arg(&mut *y)
        .arg(&b_i)
        .arg(&l_i)
        .arg(&d_i)
        .arg(&n_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch mamba_scan: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn scan_f32(
    kernels: &MambaScanKernels,
    stream: &Arc<CudaStream>,
    u: &CudaSlice<f32>,
    delta: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b_in: &CudaSlice<f32>,
    c_in: &CudaSlice<f32>,
    d_skip: Option<&CudaSlice<f32>>,
    y: &mut CudaSlice<f32>,
    b: u32,
    l: u32,
    d: u32,
    n_state: u32,
) -> Result<()> {
    scan::<f32>(
        kernels,
        stream,
        u,
        delta,
        a,
        b_in,
        c_in,
        d_skip,
        y,
        b,
        l,
        d,
        n_state,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn scan_f16(
    kernels: &MambaScanKernels,
    stream: &Arc<CudaStream>,
    u: &CudaSlice<f16>,
    delta: &CudaSlice<f16>,
    a: &CudaSlice<f16>,
    b_in: &CudaSlice<f16>,
    c_in: &CudaSlice<f16>,
    d_skip: Option<&CudaSlice<f16>>,
    y: &mut CudaSlice<f16>,
    b: u32,
    l: u32,
    d: u32,
    n_state: u32,
) -> Result<()> {
    scan::<f16>(
        kernels,
        stream,
        u,
        delta,
        a,
        b_in,
        c_in,
        d_skip,
        y,
        b,
        l,
        d,
        n_state,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn scan_bf16(
    kernels: &MambaScanKernels,
    stream: &Arc<CudaStream>,
    u: &CudaSlice<bf16>,
    delta: &CudaSlice<bf16>,
    a: &CudaSlice<bf16>,
    b_in: &CudaSlice<bf16>,
    c_in: &CudaSlice<bf16>,
    d_skip: Option<&CudaSlice<bf16>>,
    y: &mut CudaSlice<bf16>,
    b: u32,
    l: u32,
    d: u32,
    n_state: u32,
) -> Result<()> {
    scan::<bf16>(
        kernels,
        stream,
        u,
        delta,
        a,
        b_in,
        c_in,
        d_skip,
        y,
        b,
        l,
        d,
        n_state,
        DType::BF16,
    )
}
