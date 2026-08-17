use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaViewMut,
    DevicePtr, LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use cudarc::driver::sys::CUtensorMapSwizzle;

use crate::kernels::compile::{compile_module_with_opts, load_fn};
use crate::tma::{make_tma_desc_2d_u8_swz, make_tma_desc_3d_u8};

// Кэш TMA-дескрипторов (bf16-урок: encode+htod на КАЖДЫЙ вызов = H2D-копия,
// сериализующаяся со стримом, ~1µs с дескриптора; full-путь их 5 → львиная
// доля фикс-цены запуска ~10µs vs ~5µs у qutlass). Ключ = адрес + геометрия;
// переиспользование mempool-адреса с той же геометрией даёт корректный хит
// (дескриптор кодирует только адрес+layout). Память: 128Б на запись.
type DescKey = (u64, u32, u32, u32, u32, u32, u64, u64);
static DESC_CACHE: OnceLock<Mutex<std::collections::HashMap<DescKey, Arc<CudaSlice<u8>>>>> =
    OnceLock::new();

/// Сбросить кэш TMA-дескрипторов. Ключ записи — АДРЕС тензора, поэтому после
/// выгрузки модели весь кэш мёртв: 128-байтовые дескрипторы висят живыми
/// аллокациями, рассыпанными по сегментам mempool'а, и не дают
/// `cuMemPoolTrimTo` вернуть драйверу зарезервированное. Дескрипторы
/// восстанавливаются лениво на первом же вызове.
pub fn clear_desc_cache() -> usize {
    match DESC_CACHE.get() {
        Some(c) => {
            let mut g = c.lock();
            let n = g.len();
            g.clear();
            n
        }
        None => 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cached_desc_2d(
    stream: &Arc<CudaStream>,
    ptr: u64,
    rows: u32,
    cols_bytes: u32,
    box_rows: u32,
    box_cols_bytes: u32,
    swz: CUtensorMapSwizzle,
) -> Result<Arc<CudaSlice<u8>>> {
    let key = (ptr, rows, cols_bytes, box_rows, box_cols_bytes, swz as u32, 0u64, 0u64);
    let cache = DESC_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(d) = cache.lock().get(&key) {
        return Ok(d.clone());
    }
    let d = Arc::new(make_tma_desc_2d_u8_swz(
        stream, ptr, rows, cols_bytes, box_rows, box_cols_bytes, swz,
    )?);
    cache.lock().insert(key, d.clone());
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn cached_desc_3d(
    stream: &Arc<CudaStream>,
    ptr: u64,
    dim0_bytes: u32,
    dim1_count: u32,
    dim2_count: u32,
    stride1: u64,
    stride2: u64,
    box0_bytes: u32,
    box1: u32,
    box2: u32,
) -> Result<Arc<CudaSlice<u8>>> {
    let key = (
        ptr,
        dim0_bytes ^ (box0_bytes << 16),
        dim1_count ^ (box1 << 16),
        dim2_count ^ (box2 << 16),
        0,
        0,
        stride1,
        stride2,
    );
    let cache = DESC_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(d) = cache.lock().get(&key) {
        return Ok(d.clone());
    }
    let d = Arc::new(make_tma_desc_3d_u8(
        stream, ptr, dim0_bytes, dim1_count, dim2_count, stride1, stride2, box0_bytes, box1, box2,
    )?);
    cache.lock().insert(key, d.clone());
    Ok(d)
}

pub struct Nvfp4MmaGemmShufKernels {
    _module: Arc<CudaModule>,
    w4: CudaFunction,
    w8: CudaFunction,
    n8_w4: CudaFunction,
    n8_w8: CudaFunction,

    d_2x2: CudaFunction,
    d_4x2: CudaFunction,
    d_4x4: CudaFunction,
    d_8x4: CudaFunction,
    d_4x8: CudaFunction,

    r_4x4_m2n2: CudaFunction,
    r_4x2_m2n4: CudaFunction,
    r_2x2_m2n2: CudaFunction,
    r_2x2_m4n4: CudaFunction,
    r_4x2_m2n8: CudaFunction,
    r_2x2_m4n8: CudaFunction,

}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4MmaGemmShufKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4MmaGemmShufKernels>)>>> = OnceLock::new();

const SMEM_OPT_IN_BYTES: i32 = 99 * 1024;
const W4_M_TILE: u32 = 64;
const W4_THREADS: u32 = 128;
const W8_M_TILE: u32 = 128;
const W8_THREADS: u32 = 256;

impl Nvfp4MmaGemmShufKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), &[], "gemm_nvfp4.cu")
    }

    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["-DSYN_OUT_BF16"],
            "gemm_nvfp4_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<Nvfp4MmaGemmShufKernels>)>>,
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
        let src = include_str!("gemm_nvfp4.cu");
        let module = compile_module_with_opts(ctx, src, name, opts, Some("sm_120a"))?;
        let w4 = load_fn(&module, "nvfp4_mma_gemm_shuf_f16_w4")?;
        let w8 = load_fn(&module, "nvfp4_mma_gemm_shuf_f16_w8")?;
        let n8_w4 = load_fn(&module, "nvfp4_mma_gemm_shuf_n8_f16_w4")?;
        let n8_w8 = load_fn(&module, "nvfp4_mma_gemm_shuf_n8_f16_w8")?;
        let d_2x2 = load_fn(&module, "nvfp4_mma_gemm_shuf_2d_f16_2x2")?;
        let d_4x2 = load_fn(&module, "nvfp4_mma_gemm_shuf_2d_f16_4x2")?;
        let d_4x4 = load_fn(&module, "nvfp4_mma_gemm_shuf_2d_f16_4x4")?;
        let d_8x4 = load_fn(&module, "nvfp4_mma_gemm_shuf_2d_f16_8x4")?;
        let d_4x8 = load_fn(&module, "nvfp4_mma_gemm_shuf_2d_f16_4x8")?;
        let r_4x4_m2n2 = load_fn(&module, "nvfp4_mma_gemm_shuf_2dr_f16_4x4_m2n2")?;
        let r_4x2_m2n4 = load_fn(&module, "nvfp4_mma_gemm_shuf_2dr_f16_4x2_m2n4")?;
        let r_2x2_m2n2 = load_fn(&module, "nvfp4_mma_gemm_shuf_2dr_f16_2x2_m2n2")?;
        let r_2x2_m4n4 = load_fn(&module, "nvfp4_mma_gemm_shuf_2dr_f16_2x2_m4n4")?;
        let r_4x2_m2n8 = load_fn(&module, "nvfp4_mma_gemm_shuf_2dr_f16_4x2_m2n8")?;
        let r_2x2_m4n8 = load_fn(&module, "nvfp4_mma_gemm_shuf_2dr_f16_2x2_m4n8")?;
        for f in [
            &w4,
            &w8,
            &n8_w4,
            &n8_w8,
            &d_2x2,
            &d_4x2,
            &d_4x4,
            &d_8x4,
            &d_4x8,
            &r_4x4_m2n2,
            &r_4x2_m2n4,
            &r_2x2_m2n2,
            &r_2x2_m4n4,
            &r_4x2_m2n8,
            &r_2x2_m4n8,
        ] {
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_OPT_IN_BYTES,
            )
            .map_err(|e| {
                SynaptixError::Cuda(format!("set_attribute nvfp4_mma_gemm_shuf shared: {e:?}"))
            })?;
        }
        let new = Arc::new(Self {
            w4,
            w8,
            n8_w4,
            n8_w8,
            d_2x2,
            d_4x2,
            d_4x4,
            d_8x4,
            d_4x8,
            r_4x4_m2n2,
            r_4x2_m2n4,
            r_2x2_m2n2,
            r_2x2_m4n4,
            r_4x2_m2n8,
            r_2x2_m4n8,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn sf_inner_dim(k: u32) -> u32 {
    k.div_ceil(64) * 4
}

pub fn nvfp4_mma_gemm_shuf_f16(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemm_shuf_f16_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
        batch,
    )
}

pub fn nvfp4_mma_gemm_shuf_f16_view(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_f16: K={k} must be multiple of 64"
        )));
    }
    if batch == 0 {
        return Ok(());
    }
    let (kfn, threads, m_tile) = if n % W8_M_TILE == 0 {
        (&kernels.w8, W8_THREADS, W8_M_TILE)
    } else if n % W4_M_TILE == 0 {
        (&kernels.w4, W4_THREADS, W4_M_TILE)
    } else {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_f16: N={n} must be multiple of 64"
        )));
    };
    let sf_inner_w = sf_inner_dim(k);
    let sf_inner_x = sf_inner_dim(k);
    let smem_bytes = (k / 2) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n / m_tile, batch, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(packed_w_shuf)
        .arg(scales_w)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w)
        .arg(&sf_inner_x);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_mma_gemm_shuf: {e:?}")))?;
    }
    Ok(())
}

pub fn nvfp4_mma_gemm_shuf_n8_f16(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemm_shuf_n8_f16_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
        batch,
    )
}

pub fn nvfp4_mma_gemm_shuf_n8_f16_view(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_n8_f16: K={k} must be multiple of 64"
        )));
    }
    if batch % 8 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_n8_f16: batch={batch} must be multiple of 8 (use _f16 fallback)"
        )));
    }
    if batch == 0 {
        return Ok(());
    }
    let (kfn, threads, m_tile) = if n % W8_M_TILE == 0 {
        (&kernels.n8_w8, W8_THREADS, W8_M_TILE)
    } else if n % W4_M_TILE == 0 {
        (&kernels.n8_w4, W4_THREADS, W4_M_TILE)
    } else {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_n8_f16: N={n} must be multiple of 64"
        )));
    };
    let sf_inner_w = sf_inner_dim(k);
    let sf_inner_x = sf_inner_dim(k);
    let cfg = LaunchConfig {
        grid_dim: (n / m_tile, batch / 8, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(packed_w_shuf)
        .arg(scales_w)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w)
        .arg(&sf_inner_x);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_mma_gemm_shuf_n8: {e:?}")))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct Gemm2dConfig {
    pub warps_m: u32,
    pub warps_n: u32,
}

impl Gemm2dConfig {
    pub const D_2X2: Self = Self {
        warps_m: 2,
        warps_n: 2,
    };
    pub const D_4X2: Self = Self {
        warps_m: 4,
        warps_n: 2,
    };
    pub const D_4X4: Self = Self {
        warps_m: 4,
        warps_n: 4,
    };
    pub const D_8X4: Self = Self {
        warps_m: 8,
        warps_n: 4,
    };
    pub const D_4X8: Self = Self {
        warps_m: 4,
        warps_n: 8,
    };

    fn block_m(&self) -> u32 {
        self.warps_m * 16
    }
    fn block_n(&self) -> u32 {
        self.warps_n * 8
    }
    fn threads(&self) -> u32 {
        self.warps_m * self.warps_n * 32
    }
}

fn pick_2d_config(n: u32, batch: u32) -> Option<Gemm2dConfig> {
    let candidates = [
        Gemm2dConfig::D_8X4,
        Gemm2dConfig::D_4X8,
        Gemm2dConfig::D_4X4,
        Gemm2dConfig::D_4X2,
        Gemm2dConfig::D_2X2,
    ];
    candidates
        .into_iter()
        .find(|c| n % c.block_m() == 0 && batch % c.block_n() == 0)
}

pub fn nvfp4_mma_gemm_shuf_2d_f16(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemm_shuf_2d_f16_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
        batch,
    )
}

pub fn nvfp4_mma_gemm_shuf_2d_f16_view(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_2d_f16: K={k} must be multiple of 64"
        )));
    }
    if batch == 0 {
        return Ok(());
    }
    let cfg = pick_2d_config(n, batch).ok_or_else(|| {
        SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_2d_f16: no 2d config for N={n}, batch={batch}"
        ))
    })?;
    nvfp4_mma_gemm_shuf_2d_f16_with_cfg_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        out,
        n,
        k,
        batch,
        cfg,
    )
}

pub fn nvfp4_mma_gemm_shuf_2d_f16_with_cfg(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
    batch: u32,
    cfg: Gemm2dConfig,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemm_shuf_2d_f16_with_cfg_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
        batch,
        cfg,
    )
}

pub fn nvfp4_mma_gemm_shuf_2d_f16_with_cfg_view(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
    cfg: Gemm2dConfig,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_2d_f16: K={k} must be multiple of 64"
        )));
    }
    if n % cfg.block_m() != 0 || batch % cfg.block_n() != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_2d_f16: N={n}/{}_m or batch={batch}/{}_n misaligned",
            cfg.block_m(),
            cfg.block_n()
        )));
    }
    let kfn = match (cfg.warps_m, cfg.warps_n) {
        (2, 2) => &kernels.d_2x2,
        (4, 2) => &kernels.d_4x2,
        (4, 4) => &kernels.d_4x4,
        (8, 4) => &kernels.d_8x4,
        (4, 8) => &kernels.d_4x8,
        _ => {
            return Err(SynaptixError::Cuda(format!(
                "nvfp4_mma_gemm_shuf_2d_f16: unsupported cfg {cfg:?}"
            )))
        }
    };
    let sf_inner_w = sf_inner_dim(k);
    let sf_inner_x = sf_inner_dim(k);
    let launch_cfg = LaunchConfig {
        grid_dim: (n / cfg.block_m(), batch / cfg.block_n(), 1),
        block_dim: (cfg.threads(), 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(packed_w_shuf)
        .arg(scales_w)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w)
        .arg(&sf_inner_x);
    unsafe {
        b.launch(launch_cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_mma_gemm_shuf_2d: {e:?}")))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemm2drConfig {
    pub warps_m: u32,
    pub warps_n: u32,
    pub mu: u32,
    pub nu: u32,
}

impl Gemm2drConfig {
    pub const R_4X4_M2N2: Self = Self {
        warps_m: 4,
        warps_n: 4,
        mu: 2,
        nu: 2,
    };
    pub const R_4X4_M2N4: Self = Self {
        warps_m: 4,
        warps_n: 4,
        mu: 2,
        nu: 4,
    };
    pub const R_4X2_M2N4: Self = Self {
        warps_m: 4,
        warps_n: 2,
        mu: 2,
        nu: 4,
    };
    pub const R_2X2_M2N2: Self = Self {
        warps_m: 2,
        warps_n: 2,
        mu: 2,
        nu: 2,
    };
    pub const R_2X2_M4N4: Self = Self {
        warps_m: 2,
        warps_n: 2,
        mu: 4,
        nu: 4,
    };
    pub const R_4X2_M2N8: Self = Self {
        warps_m: 4,
        warps_n: 2,
        mu: 2,
        nu: 8,
    };
    pub const R_2X2_M4N8: Self = Self {
        warps_m: 2,
        warps_n: 2,
        mu: 4,
        nu: 8,
    };

    fn block_m(&self) -> u32 {
        self.warps_m * self.mu * 16
    }
    fn block_n(&self) -> u32 {
        self.warps_n * self.nu * 8
    }
    fn threads(&self) -> u32 {
        self.warps_m * self.warps_n * 32
    }
}

pub fn nvfp4_mma_gemm_shuf_2dr_f16(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemm_shuf_2dr_f16_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
        batch,
    )
}

pub fn nvfp4_mma_gemm_shuf_2dr_f16_view(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<()> {
    if batch == 0 {
        return Ok(());
    }

    let candidates = [
        Gemm2drConfig::R_2X2_M4N8,
        Gemm2drConfig::R_2X2_M4N4,
        Gemm2drConfig::R_4X2_M2N4,
        Gemm2drConfig::R_4X4_M2N2,
        Gemm2drConfig::R_2X2_M2N2,
    ];
    let cfg = candidates
        .into_iter()
        .find(|c| n % c.block_m() == 0 && batch % c.block_n() == 0)
        .ok_or_else(|| {
            SynaptixError::Cuda(format!(
                "nvfp4_mma_gemm_shuf_2dr_f16: no config for N={n}, batch={batch}"
            ))
        })?;
    nvfp4_mma_gemm_shuf_2dr_f16_with_cfg_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        out,
        n,
        k,
        batch,
        cfg,
    )
}

pub fn nvfp4_mma_gemm_shuf_2dr_f16_with_cfg(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
    batch: u32,
    cfg: Gemm2drConfig,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemm_shuf_2dr_f16_with_cfg_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
        batch,
        cfg,
    )
}

pub fn nvfp4_mma_gemm_shuf_2dr_f16_with_cfg_view(
    kernels: &Nvfp4MmaGemmShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
    cfg: Gemm2drConfig,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_2dr_f16: K={k} must be multiple of 64"
        )));
    }
    if n % cfg.block_m() != 0 || batch % cfg.block_n() != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemm_shuf_2dr_f16: N={n}/{} or batch={batch}/{} misaligned",
            cfg.block_m(),
            cfg.block_n()
        )));
    }
    let kfn = match (cfg.warps_m, cfg.warps_n, cfg.mu, cfg.nu) {
        (4, 4, 2, 2) => &kernels.r_4x4_m2n2,
        (4, 2, 2, 4) => &kernels.r_4x2_m2n4,
        (2, 2, 2, 2) => &kernels.r_2x2_m2n2,
        (2, 2, 4, 4) => &kernels.r_2x2_m4n4,
        (4, 2, 2, 8) => &kernels.r_4x2_m2n8,
        (2, 2, 4, 8) => &kernels.r_2x2_m4n8,
        _ => {
            return Err(SynaptixError::Cuda(format!(
                "nvfp4_mma_gemm_shuf_2dr_f16: unsupported cfg {cfg:?}"
            )))
        }
    };
    let sf_inner_w = sf_inner_dim(k);
    let sf_inner_x = sf_inner_dim(k);
    let launch_cfg = LaunchConfig {
        grid_dim: (n / cfg.block_m(), batch / cfg.block_n(), 1),
        block_dim: (cfg.threads(), 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(packed_w_shuf)
        .arg(scales_w)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w)
        .arg(&sf_inner_x);
    unsafe {
        b.launch(launch_cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_mma_gemm_shuf_2dr: {e:?}")))?;
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nvfp4FullCfg {
    pub wm: u32,
    pub wn: u32,
    pub mu: u32,
    pub nu: u32,
    pub stages: u32,

    pub swz: u32,

    pub persistent: bool,
    prod_warps: u32,
    fname: &'static str,
}

const KCH: u32 = 2;
const ROWB: u32 = 32 * KCH;

impl Nvfp4FullCfg {
    pub const C_128_128_S2: Self = Self {
        wm: 2,
        wn: 2,
        mu: 4,
        nu: 8,
        stages: 2,
        swz: 0,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x128_s2",
    };
    pub const C_128_128_C256_S4: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 8,
        stages: 4,
        swz: 0,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x128_c256_s4",
    };

    pub const C_128_128_C256_S4_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 8,
        stages: 4,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x128_c256_s4_swz",
    };
    pub const C_128_128_C256_S3_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 8,
        stages: 3,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x128_c256_s3_swz",
    };

    // Батч-тайл 64 (bf16-урок 64×128): средние M — 128-тайл давал 32-64 CTA.
    pub const C_128_64_S3_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 4,
        stages: 3,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x64_s3_swz",
    };
    pub const C_128_64_S4_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 4,
        stages: 4,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x64_s4_swz",
    };

    // P1: 1 producer-варп (288 потоков) — ptxas-бюджет 227 рег, без спиллов.
    pub const C_128_256_S3_SWZ: Self = Self {
        wm: 2,
        wn: 4,
        mu: 4,
        nu: 8,
        stages: 3,
        swz: 64,
        persistent: false,
        prod_warps: 0,
        fname: "gn_nvfp4_full_128x256_s3_swz",
    };
    pub const C_256_128_S3_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 4,
        nu: 8,
        stages: 3,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_256x128_s3_swz",
    };

    pub const C_PERSIST_C256_S4_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 8,
        stages: 4,
        swz: 64,
        persistent: true,
        prod_warps: 4,
        fname: "gn_nvfp4_full_persist_c256_s4_swz",
    };
    pub const C_PERSIST_C256_S3_SWZ: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 8,
        stages: 3,
        swz: 64,
        persistent: true,
        prod_warps: 4,
        fname: "gn_nvfp4_full_persist_c256_s3_swz",
    };

    // ROT: k64-конвейер по схеме CUTLASS (sm120_blockscaled_mma_tma.hpp) —
    // double-buffer фрагментов + ранний release + wait перед последней gemm-пачкой.
    pub const C_128_256_S3_SWZ_ROT: Self = Self {
        wm: 2,
        wn: 4,
        mu: 4,
        nu: 8,
        stages: 3,
        swz: 64,
        persistent: false,
        prod_warps: 0,
        fname: "gn_nvfp4_full_128x256_s3_swz_rot",
    };
    pub const C_128_128_C256_S4_SWZ_DROT: Self = Self {
        wm: 4,
        wn: 2,
        mu: 2,
        nu: 8,
        stages: 4,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x128_c256_s4_swz_drot",
    };
    // Структура qutlass: 384 потока, producer-warpgroup, setmaxnreg 240/32
    // (работает после ::cta-фикса TMA).
    pub const C_128_256_S3_SWZ_DROT: Self = Self {
        wm: 2,
        wn: 4,
        mu: 4,
        nu: 8,
        stages: 3,
        swz: 64,
        persistent: false,
        prod_warps: 4,
        fname: "gn_nvfp4_full_128x256_s3_swz_drot",
    };


    pub const ALL: [Self; 13] = [
        Self::C_128_64_S3_SWZ,
        Self::C_128_64_S4_SWZ,
        Self::C_128_128_S2,
        Self::C_128_128_C256_S4,
        Self::C_128_128_C256_S4_SWZ,
        Self::C_128_128_C256_S3_SWZ,
        Self::C_128_256_S3_SWZ,
        Self::C_256_128_S3_SWZ,
        Self::C_PERSIST_C256_S4_SWZ,
        Self::C_PERSIST_C256_S3_SWZ,
        Self::C_128_256_S3_SWZ_ROT,
        Self::C_128_128_C256_S4_SWZ_DROT,
        Self::C_128_256_S3_SWZ_DROT,
    ];

    pub fn fname(&self) -> &'static str {
        self.fname
    }
    pub fn block_m(&self) -> u32 {
        self.wm * self.mu * 16
    }
    pub fn block_n(&self) -> u32 {
        self.wn * self.nu * 8
    }
    fn threads(&self) -> u32 {
        (self.wm * self.wn + self.prod_warps) * 32
    }
    fn num_w_tiles(&self) -> u32 {
        self.block_m().div_ceil(128)
    }

    fn num_x_tiles(&self) -> u32 {
        self.block_n().div_ceil(128)
    }

    fn smem(&self) -> u32 {
        let w_sz = self.block_m() * ROWB;
        let x_sz = self.block_n() * ROWB;
        let sf_sz = KCH * 512;
        let sfa_sz = self.num_w_tiles() * sf_sz;
        let sfb_sz = self.num_x_tiles() * sf_sz;
        self.stages * (w_sz + x_sz + sfa_sz + sfb_sz) + 2 * self.stages * 8
    }
    pub fn fits(&self, m: u32, n: u32, k: u32) -> bool {
        // батч m ЛЮБОЙ >=1 у non-persist (TMA OOB-нули + BATCH-гард эпилогов);
        // persist-ядра без гарда — батч-кратность держим.
        let m_ok = if self.persistent { m % self.block_n() == 0 } else { m >= 1 };
        n % self.block_m() == 0
            && m_ok
            && k % 128 == 0
            && (k / 64) / KCH >= self.stages
            && self.smem() <= 101376
    }
}

pub struct GemmNvfp4FullKernels {
    _module: Arc<CudaModule>,
    fns: Vec<(Nvfp4FullCfg, CudaFunction)>,
    num_sms: u32,
}

fn query_sm_count(ctx: &Arc<CudaContext>) -> Result<u32> {
    let dev = ctx.cu_device();
    let mut v: i32 = 0;
    unsafe {
        cudarc::driver::sys::cuDeviceGetAttribute(
            &mut v as *mut i32,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            dev,
        )
        .result()
        .map_err(|e| SynaptixError::Cuda(format!("cuDeviceGetAttribute SM_COUNT: {e:?}")))?;
    }
    Ok(v as u32)
}

static FULL_CACHE: OnceLock<Mutex<Vec<(usize, Arc<GemmNvfp4FullKernels>)>>> = OnceLock::new();
static FULL_CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<GemmNvfp4FullKernels>)>>> = OnceLock::new();

impl GemmNvfp4FullKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, FULL_CACHE.get_or_init(|| Mutex::new(Vec::new())), &[], "gemm_nvfp4.cu")
    }

    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            FULL_CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["-DSYN_OUT_BF16"],
            "gemm_nvfp4_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<GemmNvfp4FullKernels>)>>,
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
        let src = include_str!("gemm_nvfp4.cu");
        let module = compile_module_with_opts(ctx, src, name, opts, Some("sm_120a"))?;
        let mut fns = Vec::new();
        for cfg in Nvfp4FullCfg::ALL {
            let f = load_fn(&module, cfg.fname)?;
            if cfg.smem() > 48 * 1024 {
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.smem() as i32,
                )
                .map_err(|e| {
                    SynaptixError::Cuda(format!("set_attr nvfp4_full {}: {e:?}", cfg.fname))
                })?;
            }
            fns.push((cfg, f));
        }
        let num_sms = query_sm_count(ctx)?;
        let new = Arc::new(Self {
            fns,
            _module: module,
            num_sms,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn fnref(&self, cfg: Nvfp4FullCfg) -> Option<&CudaFunction> {
        self.fns.iter().find(|(c, _)| *c == cfg).map(|(_, f)| f)
    }

}

#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4_full_cfg_view(
    kernels: &GemmNvfp4FullKernels,
    stream: &Arc<CudaStream>,
    packed_w: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    batch: u32,
    cfg: Nvfp4FullCfg,
) -> Result<()> {
    if !cfg.fits(batch, n, k) {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_full {}: shape/smem",
            cfg.fname
        )));
    }
    let kfn = kernels
        .fnref(cfg)
        .ok_or_else(|| SynaptixError::Cuda(format!("nvfp4_full: нет ядра {}", cfg.fname)))?;

    let sf_inner_w = sf_inner_dim(k);
    let sf_inner_x = sf_inner_dim(k);
    let k_half = k / 2;
    let (w_ptr, _rw) = packed_w.device_ptr(stream);
    let (x_ptr, _rx) = packed_x.device_ptr(stream);
    let (sw_ptr, _rsw) = scales_w.device_ptr(stream);
    let (sx_ptr, _rsx) = scales_x.device_ptr(stream);

    let x_swizzle = if cfg.swz == 64 {
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
    } else {
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
    };

    const W_CHUNK_BYTES: u32 = 512;
    let m_block_stride = ((k / 64) * W_CHUNK_BYTES) as u64;
    let w_dim1_count = (k / 64) * 2;
    let w_dim2_count = n / 16;
    let w_box1 = KCH * 2;
    let w_box2 = cfg.block_m() / 16;
    let w_desc = cached_desc_3d(
        stream,
        w_ptr,
        256,
        w_dim1_count,
        w_dim2_count,
        256,
        m_block_stride,
        256,
        w_box1,
        w_box2,
    )?;
    let x_desc =
        cached_desc_2d(stream, x_ptr, batch, k_half, cfg.block_n(), 64, x_swizzle)?;
    let sfa_rows = n.div_ceil(128) * (sf_inner_w * 128 / 256);
    let sfb_rows = batch.div_ceil(128) * (sf_inner_x * 128 / 256);
    const NOSWZ: CUtensorMapSwizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    let sfa_desc = cached_desc_2d(stream, sw_ptr, sfa_rows, 256, KCH * 2, 256, NOSWZ)?;
    let sfb_desc = cached_desc_2d(stream, sx_ptr, sfb_rows, 256, KCH * 2, 256, NOSWZ)?;

    if cfg.persistent {
        let total_tiles = (n / cfg.block_m()) * (batch / cfg.block_n());
        let blocks_per_sm = 1u32;
        let grid = (kernels.num_sms * blocks_per_sm).min(total_tiles).max(1);
        let launch = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (cfg.threads(), 1, 1),
            shared_mem_bytes: cfg.smem(),
        };
        let mut b = stream.launch_builder(kfn);
        b.arg(&*w_desc)
            .arg(&*x_desc)
            .arg(&*sfa_desc)
            .arg(&*sfb_desc)
            .arg(&mut *out)
            .arg(&n)
            .arg(&k)
            .arg(&batch)
            .arg(&sf_inner_w)
            .arg(&sf_inner_x);
        unsafe {
            b.launch(launch).map_err(|e| {
                SynaptixError::Cuda(format!("launch nvfp4_full {}: {e:?}", cfg.fname))
            })?;
        }
        return Ok(());
    }

    // out-дескриптор TMA-store эпилога (ROT-семейство): варп-бокс
    // [nu*8 батч-строк × mu*16 колонок f16]; OOB-строки клипает сам TMA
    // (BATCH-гард бесплатно). Non-ROT эпилоги пишут через out-указатель.
    let out_ptr = {
        let (p, _ro) = out.device_ptr(stream);
        p
    };
    // Свизл по ширине варп-строки эпилога (банк-конфликты stmatrix), TMA-store
    // раскручивает по дескриптору: 128Б → SWIZZLE_128B, 64Б → SWIZZLE_64B.
    let out_swz = match cfg.mu * 16 * 2 {
        128 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        64 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        _ => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    };
    let out_desc = cached_desc_2d(
        stream,
        out_ptr,
        batch,
        n * 2,
        cfg.nu * 8,
        cfg.mu * 16 * 2,
        out_swz,
    )?;
    let launch = LaunchConfig {
        // батч с ceil: TMA зануляет OOB-чтения хвостового ряда, эпилоги гардят
        // сторы по BATCH → невыровненный m без out_pad-скретча и 218MB-копии.
        grid_dim: (n / cfg.block_m(), batch.div_ceil(cfg.block_n()), 1),
        block_dim: (cfg.threads(), 1, 1),
        shared_mem_bytes: cfg.smem(),
    };
    let mut b = stream.launch_builder(kfn);
    b.arg(&*w_desc)
        .arg(&*x_desc)
        .arg(&*sfa_desc)
        .arg(&*sfb_desc)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w)
        .arg(&sf_inner_x)
        .arg(&batch)
        .arg(&*out_desc);
    unsafe {
        b.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_full {}: {e:?}", cfg.fname)))?;
    }
    Ok(())
}
