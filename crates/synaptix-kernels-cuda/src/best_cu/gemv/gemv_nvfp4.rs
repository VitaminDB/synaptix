use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig,
    PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct Nvfp4MmaGemvShufKernels {
    _module: Arc<CudaModule>,
    repack: CudaFunction,
    w4: CudaFunction,
    w8: CudaFunction,
    w8_persistent: CudaFunction,
    w8_batched: CudaFunction,
    ptr_gather: CudaFunction,
    grouped_splitk: CudaFunction,
    indexed_splitk: CudaFunction,
    num_sms: u32,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4MmaGemvShufKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4MmaGemvShufKernels>)>>> = OnceLock::new();

const SMEM_OPT_IN_BYTES: i32 = 99 * 1024;
const W4_M_TILE: u32 = 64;
const W4_THREADS: u32 = 128;
const W8_M_TILE: u32 = 128;
const W8_THREADS: u32 = 256;

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

impl Nvfp4MmaGemvShufKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), &[], "gemv_nvfp4.cu")
    }

    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["-DSYN_OUT_BF16"],
            "gemv_nvfp4_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<Nvfp4MmaGemvShufKernels>)>>,
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
        let src = include_str!("gemv_nvfp4.cu");
        let module = compile_module_with_opts(ctx, src, name, opts, Some("sm_120a"))?;
        let repack = load_fn(&module, "nvfp4_w_repack")?;
        let w4 = load_fn(&module, "nvfp4_mma_gemv_shuf_f16_w4")?;
        let w8 = load_fn(&module, "nvfp4_mma_gemv_shuf_f16_w8")?;
        let w8p = load_fn(&module, "nvfp4_mma_gemv_shuf_f16_w8_persistent")?;
        let w8b = load_fn(&module, "nvfp4_mma_gemv_shuf_f16_w8_batched")?;
        let ptr_gather = load_fn(&module, "nvfp4_expert_ptr_gather")?;
        let grouped_splitk = load_fn(&module, "nvfp4_mma_gemv_shuf_grouped_splitk")?;
        let indexed_splitk = load_fn(&module, "nvfp4_mma_gemv_shuf_indexed_splitk")?;
        for f in [&w4, &w8, &w8p, &w8b] {
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_OPT_IN_BYTES,
            )
            .map_err(|e| {
                SynaptixError::Cuda(format!("set_attribute nvfp4_mma_gemv_shuf shared: {e:?}"))
            })?;
        }
        let num_sms = query_sm_count(ctx)?;
        let new = Arc::new(Self {
            repack,
            w4,
            w8,
            w8_persistent: w8p,
            w8_batched: w8b,
            ptr_gather,
            grouped_splitk,
            indexed_splitk,
            _module: module,
            num_sms,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn sf_inner_dim(k: u32) -> u32 {
    k.div_ceil(64) * 4
}

/// Батч GEMV: `out[e]` = `W_e · x_e`, по одному блоку grid.z на эксперта.
/// Все веса обязаны быть одной формы `[n, k]`; `out` — плотный `[e, n]`.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_mma_gemv_shuf_f16_batched(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    w_ptrs: &CudaSlice<u64>,
    sw_ptrs: &CudaSlice<u64>,
    xp_ptrs: &CudaSlice<u64>,
    xs_ptrs: &CudaSlice<u64>,
    x_sf_offs: &CudaSlice<u32>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    experts: u32,
) -> Result<()> {
    nvfp4_mma_gemv_shuf_f16_batched_views(
        kernels,
        stream,
        &w_ptrs.slice(..),
        &sw_ptrs.slice(..),
        &xp_ptrs.slice(..),
        &xs_ptrs.slice(..),
        &x_sf_offs.slice(..),
        out,
        n,
        k,
        experts,
    )
}

/// То же, но по view'ам: указатели могут лежать в скретче, собранном ядром.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_mma_gemv_shuf_f16_batched_views(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    w_ptrs: &CudaView<u64>,
    sw_ptrs: &CudaView<u64>,
    xp_ptrs: &CudaView<u64>,
    xs_ptrs: &CudaView<u64>,
    x_sf_offs: &CudaView<u32>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
    experts: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemv_shuf_f16_batched: K={k} must be multiple of 64"
        )));
    }
    if n % W8_M_TILE != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemv_shuf_f16_batched: N={n} must be multiple of {W8_M_TILE}"
        )));
    }
    if experts == 0 {
        return Ok(());
    }
    let sf_inner_w = sf_inner_dim(k);
    let cfg = LaunchConfig {
        grid_dim: (n / W8_M_TILE, 1, experts),
        block_dim: (W8_THREADS, 1, 1),
        shared_mem_bytes: (k / 2) as u32,
    };
    let mut b = stream.launch_builder(&kernels.w8_batched);
    b.arg(w_ptrs)
        .arg(sw_ptrs)
        .arg(xp_ptrs)
        .arg(xs_ptrs)
        .arg(x_sf_offs)
        .arg(&mut *out)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w);
    unsafe { b.launch(cfg) }
        .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_mma_gemv_shuf_f16_batched: {e:?}")))?;
    Ok(())
}

pub fn nvfp4_w_repack(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_in: &CudaSlice<u8>,
    packed_w_out: &mut CudaSlice<u8>,
    n: u32,
    k: u32,
) -> Result<()> {
    if n % 16 != 0 || k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_w_repack: N={n} % 16, K={k} % 64 required"
        )));
    }
    let cfg = LaunchConfig {
        grid_dim: (n / 16, k / 64, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kernels.repack);
    b.arg(packed_w_in).arg(&mut *packed_w_out).arg(&n).arg(&k);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_w_repack: {e:?}")))?;
    }
    Ok(())
}

pub fn nvfp4_mma_gemv_shuf_f16(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaSlice<f16>,
    n: u32,
    k: u32,
) -> Result<()> {
    let mut ov = out.as_view_mut();
    nvfp4_mma_gemv_shuf_f16_view(
        kernels,
        stream,
        packed_w_shuf,
        scales_w,
        packed_x,
        scales_x,
        &mut ov,
        n,
        k,
    )
}

pub fn nvfp4_mma_gemv_shuf_f16_view(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    packed_w_shuf: &CudaSlice<u8>,
    scales_w: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_mma_gemv_shuf_f16: K={k} must be multiple of 64"
        )));
    }
    let num_tiles_w8 = if n % W8_M_TILE == 0 { n / W8_M_TILE } else { 0 };
    let want_w8 = num_tiles_w8 >= kernels.num_sms;
    let want_persistent = want_w8 && num_tiles_w8 >= kernels.num_sms * 4;
    let (kfn, threads, grid) = if want_persistent {
        (&kernels.w8_persistent, W8_THREADS, kernels.num_sms * 4)
    } else if want_w8 {
        (&kernels.w8, W8_THREADS, n / W8_M_TILE)
    } else {
        if n % W4_M_TILE != 0 {
            return Err(SynaptixError::Cuda(format!(
                "nvfp4_mma_gemv_shuf_f16: N={n} must be multiple of 64"
            )));
        }
        (&kernels.w4, W4_THREADS, n / W4_M_TILE)
    };
    let sf_inner_w = sf_inner_dim(k);
    let smem_bytes = (k / 2) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
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
        .arg(&sf_inner_w);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_mma_gemv_shuf: {e:?}")))?;
    }
    Ok(())
}

/// Собрать массивы указателей пакетного GEMV по device-индексам экспертов.
/// Все буферы уже на карте — хост в выборе не участвует, поэтому вызов
/// захватывается CUDA-графом.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_expert_ptr_gather(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    w_table: &CudaView<u64>,
    s_table: &CudaView<u64>,
    idx: &CudaView<u32>,
    w_out: &mut CudaViewMut<u64>,
    s_out: &mut CudaViewMut<u64>,
    xp_out: &mut CudaViewMut<u64>,
    xs_out: &mut CudaViewMut<u64>,
    off_out: &mut CudaViewMut<u32>,
    xp_base: u64,
    xs_base: u64,
    experts: u32,
    pairs: u32,
    rows_per_pair: bool,
    row_bytes: u32,
) -> Result<()> {
    if pairs == 0 {
        return Ok(());
    }
    let threads = 128u32;
    let cfg = LaunchConfig {
        grid_dim: (pairs.div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let xp = xp_base as i64;
    let xs = xs_base as i64;
    let e = experts as i32;
    let p = pairs as i32;
    let rpp = i32::from(rows_per_pair);
    let rb = row_bytes as i32;
    let mut b = stream.launch_builder(&kernels.ptr_gather);
    b.arg(w_table)
        .arg(s_table)
        .arg(idx)
        .arg(&mut *w_out)
        .arg(&mut *s_out)
        .arg(&mut *xp_out)
        .arg(&mut *xs_out)
        .arg(&mut *off_out)
        .arg(&xp)
        .arg(&xs)
        .arg(&e)
        .arg(&p)
        .arg(&rpp)
        .arg(&rb);
    unsafe { b.launch(cfg) }
        .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_expert_ptr_gather: {e:?}")))?;
    Ok(())
}

/// Группа для [`nvfp4_gemv_grouped_splitk`]: device-адреса перемешанного
/// веса, его масштабов и выхода плюс N. Выход — f16 или bf16 по модулю
/// (`for_context` / `for_context_bf16`).
#[derive(Clone, Copy)]
pub struct GemvGroup {
    pub w_shuf: u64,
    pub scales: u64,
    pub out: u64,
    pub n: u32,
}

const SPLITK_THREADS: u32 = 256;

/// До трёх весов одной K с общей квант-активацией (строка 0) одним запуском;
/// каждый блок — 16-строчный тайл со split-K по 8 варпам. N кратны 16.
pub fn nvfp4_gemv_grouped_splitk(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    groups: &[GemvGroup],
    packed_x: &CudaView<u8>,
    scales_x: &CudaView<u8>,
    k: u32,
) -> Result<()> {
    if groups.is_empty() || groups.len() > 3 {
        return Err(SynaptixError::Cuda("nvfp4_gemv_grouped_splitk: 1..3 группы".into()));
    }
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_gemv_grouped_splitk: K={k} must be multiple of 64"
        )));
    }
    let mut g = [GemvGroup { w_shuf: 0, scales: 0, out: 0, n: 0 }; 3];
    let mut tiles = 0u32;
    for (i, grp) in groups.iter().enumerate() {
        if grp.n % 16 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "nvfp4_gemv_grouped_splitk: N={} must be multiple of 16",
                grp.n
            )));
        }
        g[i] = *grp;
        tiles += grp.n / 16;
    }
    if tiles == 0 {
        return Ok(());
    }
    let sf_inner_w = sf_inner_dim(k);
    let cfg = LaunchConfig {
        grid_dim: (tiles, 1, 1),
        block_dim: (SPLITK_THREADS, 1, 1),
        shared_mem_bytes: k / 2,
    };
    let mut b = stream.launch_builder(&kernels.grouped_splitk);
    b.arg(&g[0].w_shuf).arg(&g[0].scales).arg(&g[0].out).arg(&g[0].n)
        .arg(&g[1].w_shuf).arg(&g[1].scales).arg(&g[1].out).arg(&g[1].n)
        .arg(&g[2].w_shuf).arg(&g[2].scales).arg(&g[2].out).arg(&g[2].n)
        .arg(packed_x)
        .arg(scales_x)
        .arg(&k)
        .arg(&sf_inner_w);
    unsafe { b.launch(cfg) }
        .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_gemv_grouped_splitk: {e:?}")))?;
    Ok(())
}

/// Куда кладёт индексный GEMV: строки `[pairs, N]` (f16/bf16 по модулю) или
/// взвешенная f32-сумма `acc[N] += wts[p]·y_p` атомиками.
#[derive(Clone, Copy)]
pub enum IndexedEpilogue {
    Rows { out: u64 },
    Accumulate { acc: u64, wts: u64 },
}

/// Пакетный GEMV с выбором эксперта индексом на карте; блок — 16-строчный
/// тайл со split-K, grid.z — пара. `row_bytes` = K/2 — шаг строки активации.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemv_indexed_splitk(
    kernels: &Nvfp4MmaGemvShufKernels,
    stream: &Arc<CudaStream>,
    w_table: &CudaView<u64>,
    s_table: &CudaView<u64>,
    idx: &CudaView<u32>,
    x_packed: &CudaView<u8>,
    x_scales: &CudaView<u8>,
    rows_per_pair: bool,
    epilogue: IndexedEpilogue,
    n: u32,
    k: u32,
    experts: u32,
    pairs: u32,
) -> Result<()> {
    if k % 64 != 0 || n % 16 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_gemv_indexed_splitk: N={n} % 16, K={k} % 64 required"
        )));
    }
    if pairs == 0 {
        return Ok(());
    }
    let sf_inner_w = sf_inner_dim(k);
    let cfg = LaunchConfig {
        grid_dim: (n / 16, 1, pairs),
        block_dim: (SPLITK_THREADS, 1, 1),
        shared_mem_bytes: k / 2,
    };
    let (out_u, acc_u, wts_u) = match epilogue {
        IndexedEpilogue::Rows { out } => (out, 0u64, 0u64),
        IndexedEpilogue::Accumulate { acc, wts } => (0u64, acc, wts),
    };
    let row_bytes = k / 2;
    let rpp: i32 = i32::from(rows_per_pair);
    let mut b = stream.launch_builder(&kernels.indexed_splitk);
    b.arg(w_table)
        .arg(s_table)
        .arg(idx)
        .arg(x_packed)
        .arg(x_scales)
        .arg(&row_bytes)
        .arg(&rpp)
        .arg(&out_u)
        .arg(&acc_u)
        .arg(&wts_u)
        .arg(&n)
        .arg(&k)
        .arg(&sf_inner_w)
        .arg(&experts);
    unsafe { b.launch(cfg) }
        .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_gemv_indexed_splitk: {e:?}")))?;
    Ok(())
}
