use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const WARPS: u32 = 8;
const THREADS: u32 = WARPS * 32;

pub struct GemvMxfp8Kernels {
    _module: Arc<CudaModule>,
    gemv: CudaFunction,
    grouped: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GemvMxfp8Kernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<GemvMxfp8Kernels>)>>> = OnceLock::new();

impl GemvMxfp8Kernels {
    /// Выход F16.
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), &[], "gemv_mxfp8.cu")
    }

    /// Выход BF16 (тот же модуль, собранный с `-DSYN_OUT_BF16`): декод в BF16
    /// пишет проекцию сразу в рабочем dtype, без cast-ядра после GEMV.
    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["-DSYN_OUT_BF16"],
            "gemv_mxfp8_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<GemvMxfp8Kernels>)>>,
        opts: &[&str],
        name: &'static str,
    ) -> Result<Arc<Self>> {
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return Ok(v.clone());
                }
            }
        }
        let src = include_str!("gemv_mxfp8.cu");
        let module = compile_module_with_opts(ctx, src, name, opts, Some("sm_120a"))?;
        let gemv = load_fn(&module, "gemv_mxfp8_e4m3")?;
        let grouped = load_fn(&module, "gemv_mxfp8_e4m3_grouped")?;
        let new = Arc::new(Self {
            gemv,
            grouped,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

// MXFP8 GEMV (decode, M=1): y[N] = W[N,K] @ x[K]. w/x — E4M3 байты (natural
// [.,K]); sw/sx — E8M0 per-32-block scales (natural [., K/32]). out — f16 [N]
// (или bf16 той же ширины, если `kernels` собраны `for_context_bf16`; вьюха
// типизирована f16 лишь ради размера элемента).
#[allow(clippy::too_many_arguments)]
pub fn gemv_mxfp8(
    kernels: &GemvMxfp8Kernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    sw: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    sx: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let grid = n.div_ceil(WARPS);
    let (ni, ki) = (n as i32, k as i32);
    let mut bld = stream.launch_builder(&kernels.gemv);
    bld.arg(w)
        .arg(sw)
        .arg(x)
        .arg(sx)
        .arg(&mut *out)
        .arg(&ni)
        .arg(&ki);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch gemv_mxfp8: {e:?}")))?;
    }
    Ok(())
}

/// Группа для [`gemv_mxfp8_grouped`]: device-адреса веса (e4m3 natural),
/// масштабов (E8M0 natural) и выхода плюс N.
#[derive(Clone, Copy)]
pub struct MxGemvGroup {
    pub w: u64,
    pub scales: u64,
    pub out: u64,
    pub n: u32,
}

/// До трёх матриц одной K с общей MXFP8-активацией одним запуском (q/k/v).
/// Выход — f16 или bf16 по модулю.
pub fn gemv_mxfp8_grouped(
    kernels: &GemvMxfp8Kernels,
    stream: &Arc<CudaStream>,
    groups: &[MxGemvGroup],
    x: &CudaView<u8>,
    sx: &CudaView<u8>,
    k: u32,
) -> Result<()> {
    if groups.is_empty() || groups.len() > 3 {
        return Err(SynaptixError::Cuda("gemv_mxfp8_grouped: 1..3 группы".into()));
    }
    let mut g = [MxGemvGroup { w: 0, scales: 0, out: 0, n: 0 }; 3];
    let mut rows = 0u32;
    for (i, grp) in groups.iter().enumerate() {
        g[i] = *grp;
        rows += grp.n;
    }
    if rows == 0 {
        return Ok(());
    }
    let grid = rows.div_ceil(WARPS);
    let ki = k as i32;
    let n = [g[0].n as i32, g[1].n as i32, g[2].n as i32];
    let mut bld = stream.launch_builder(&kernels.grouped);
    bld.arg(&g[0].w).arg(&g[0].scales).arg(&g[0].out).arg(&n[0])
        .arg(&g[1].w).arg(&g[1].scales).arg(&g[1].out).arg(&n[1])
        .arg(&g[2].w).arg(&g[2].scales).arg(&g[2].out).arg(&n[2])
        .arg(x)
        .arg(sx)
        .arg(&ki);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch gemv_mxfp8_grouped: {e:?}")))?;
    }
    Ok(())
}
