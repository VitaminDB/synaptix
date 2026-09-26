//! Деквант одноблобных форматов (SQ, типы ggml) на карте: `[rows, k]` → f16/bf16.
//!
//! Модуль `blockq_dequant.cu` собирается с `--fmad=false` и повторяет порядок
//! операций CPU-эталона `synaptix_core::quant` — результат совпадает бит в бит
//! (тест `cuda_blockq_dequant`). Один поток считает под-блок из 32 значений.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::quant::GgmlType;

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct BlockqDequantKernels {
    _module: Arc<CudaModule>,
    fns: HashMap<&'static str, CudaFunction>,
    /// `deqg_<suffix>` — gather строк таблицы по индексам с деквантом
    /// (только одноблобные форматы).
    gather_fns: HashMap<&'static str, CudaFunction>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<BlockqDequantKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<BlockqDequantKernels>)>>> = OnceLock::new();

/// Суффикс точки входа для формата (`deq_<suffix>` / `gemv_<suffix>`);
/// `None` — формат не квантованный вес.
pub fn entry_suffix(dtype: DType) -> Option<&'static str> {
    Some(match dtype {
        DType::NVFP4 => "nvfp4_syn",
        DType::MXFP8 => "mxfp8_syn",
        DType::Sq { bits: 1 } => "sq1",
        DType::Sq { bits: 2 } => "sq2",
        DType::Sq { bits: 3 } => "sq3",
        DType::Sq { bits: 4 } => "sq4",
        DType::Sq { bits: 5 } => "sq5",
        DType::Sq { bits: 6 } => "sq6",
        DType::Sq { bits: 7 } => "sq7",
        DType::Sq { bits: 8 } => "sq8",
        DType::Ggml(t) => {
            use GgmlType::*;
            match t {
                Q4_0 => "q4_0",
                Q4_1 => "q4_1",
                Q5_0 => "q5_0",
                Q5_1 => "q5_1",
                Q8_0 => "q8_0",
                Q8_1 => "q8_1",
                Q8K => "q8_k",
                Q1_0 => "q1_0",
                Q2_0 => "q2_0",
                Mxfp4 => "mxfp4",
                Nvfp4 => "nvfp4",
                Iq4Nl => "iq4_nl",
                Q2K => "q2_k",
                Q3K => "q3_k",
                Q4K => "q4_k",
                Q5K => "q5_k",
                Q6K => "q6_k",
                Iq4Xs => "iq4_xs",
                Iq2Xxs => "iq2_xxs",
                Iq2Xs => "iq2_xs",
                Iq2S => "iq2_s",
                Iq3Xxs => "iq3_xxs",
                Iq3S => "iq3_s",
                Iq1S => "iq1_s",
                Iq1M => "iq1_m",
                Tq1_0 => "tq1_0",
                Tq2_0 => "tq2_0",
                _ => return None,
            }
        }
        _ => return None,
    })
}

const SUFFIXES: &[&str] = &[
    "sq1", "sq2", "sq3", "sq4", "sq5", "sq6", "sq7", "sq8", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q1_0", "q2_0", "mxfp4", "nvfp4", "iq4_nl", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq4_xs",
    "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m", "tq1_0", "tq2_0",
];
/// Только деквант (форматы активаций dp4a и NVFP4/MXFP8 движка).
const DEQ_ONLY_SUFFIXES: &[&str] = &["q8_1", "q8_k", "nvfp4_syn", "mxfp8_syn"];

/// Исходник модуля деквантования: таблицы ggml + декодеры + точки входа.
pub fn module_source() -> String {
    format!(
        "{}\n{}\n{}",
        include_str!("../cu/elementwise/ggml_tables.cuh"),
        include_str!("../cu/elementwise/blockq_decode.cuh"),
        include_str!("../cu/elementwise/blockq_dequant.cu")
    )
}

/// Исходник модуля GEMV: таблицы ggml + декодеры + ядра GEMV.
pub fn gemv_module_source() -> String {
    format!(
        "{}\n{}\n{}",
        include_str!("../cu/elementwise/ggml_tables.cuh"),
        include_str!("../cu/elementwise/blockq_decode.cuh"),
        include_str!("../cu/elementwise/blockq_gemv.cu")
    )
}

/// Опции NVRTC модуля: без слияния в FMA — контракт бит в бит с CPU-эталоном.
pub const MODULE_OPTS: &[&str] = &["--fmad=false"];

impl BlockqDequantKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), MODULE_OPTS, "blockq_dequant.cu")
    }

    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["--fmad=false", "-DSYN_OUT_BF16"],
            "blockq_dequant_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<Self>)>>,
        opts: &[&str],
        tag: &'static str,
    ) -> Result<Arc<Self>> {
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock().unwrap();
            if let Some((_, v)) = g.iter().find(|(k, _)| *k == key) {
                return Ok(v.clone());
            }
        }
        let src = module_source();
        let module = compile_module_with_opts(ctx, &src, tag, opts, None)?;
        let mut fns = HashMap::new();
        let mut gather_fns = HashMap::new();
        for suf in SUFFIXES.iter().chain(DEQ_ONLY_SUFFIXES) {
            let name = format!("deq_{suf}");
            fns.insert(*suf, load_fn(&module, &name)?);
            if !matches!(*suf, "nvfp4_syn" | "mxfp8_syn") {
                gather_fns.insert(*suf, load_fn(&module, &format!("deqg_{suf}"))?);
            }
        }
        let new = Arc::new(Self { _module: module, fns, gather_fns });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }

    fn func(&self, dtype: DType) -> Result<&CudaFunction> {
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("blockq_dequant: формат не квантованный вес"))?;
        self.fns.get(suf).ok_or(SynaptixError::Unsupported("blockq_dequant: нет точки входа"))
    }

    fn gather_func(&self, dtype: DType) -> Result<&CudaFunction> {
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("blockq_gather_dequant: формат не квантованный вес"))?;
        self.gather_fns.get(suf).ok_or(SynaptixError::Unsupported("blockq_gather_dequant: только одноблобные форматы"))
    }
}

/// Gather эмбеддингов из упакованной таблицы `[vocab, k]` одноблобного
/// формата: `out[i, :] = dequant(table[ids[i], :])`, `out` — `[n_ids, k]` в
/// типе модуля. Индекс ≥ `vocab` даёт строку нулей.
#[allow(clippy::too_many_arguments)]
pub fn blockq_gather_dequant(
    kernels: &BlockqDequantKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    table: &CudaView<'_, u8>,
    ids: &CudaView<'_, u32>,
    out: &mut CudaViewMut<'_, u8>,
    n_ids: u32,
    vocab: u32,
    k: u32,
) -> Result<()> {
    let rb = row_bytes(dtype, k as usize)? as u32;
    if table.len() < vocab as usize * rb as usize {
        return Err(SynaptixError::Unsupported("blockq_gather_dequant: таблица короче vocab × row_bytes"));
    }
    if ids.len() < n_ids as usize {
        return Err(SynaptixError::Unsupported("blockq_gather_dequant: ids короче n_ids"));
    }
    if out.len() < n_ids as usize * k as usize * 2 {
        return Err(SynaptixError::Unsupported("blockq_gather_dequant: out короче n_ids × k"));
    }
    if n_ids == 0 {
        return Ok(());
    }
    let f = kernels.gather_func(dtype)?;
    let total = n_ids * (k / 32);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    let mut bld = stream.launch_builder(f);
    bld.arg(table).arg(ids).arg(&mut *out).arg(&n_ids).arg(&vocab).arg(&k).arg(&rb);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_gather_dequant {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

/// Проверка формы и байт на строку. `k` кратен 32 и блоку формата.
pub fn row_bytes(dtype: DType, k: usize) -> Result<usize> {
    if k % 32 != 0 {
        return Err(SynaptixError::Unsupported("blockq_dequant: K должно быть кратно 32"));
    }
    synaptix_core::quant::block_row_bytes(dtype, k)
        .ok_or(SynaptixError::Unsupported("blockq_dequant: K не кратен блоку формата"))
}

/// `src` — `rows × row_bytes(dtype, k)` байт подряд; `out` — `rows × k` в
/// типе модуля (f16 или bf16 — по тому, какой набор ядер передан).
#[allow(clippy::too_many_arguments)]
pub fn blockq_dequant_raw(
    kernels: &BlockqDequantKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    src: &CudaView<'_, u8>,
    out_ptr: &mut CudaViewMut<'_, u8>,
    rows: u32,
    k: u32,
) -> Result<()> {
    let rb = row_bytes(dtype, k as usize)? as u32;
    if src.len() < rows as usize * rb as usize {
        return Err(SynaptixError::Unsupported("blockq_dequant: src короче rows × row_bytes"));
    }
    if out_ptr.len() < rows as usize * k as usize * 2 {
        return Err(SynaptixError::Unsupported("blockq_dequant: out короче rows × k"));
    }
    let f = kernels.func(dtype)?;
    let total = rows * (k / 32);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    let mut bld = stream.launch_builder(f);
    bld.arg(src).arg(&mut *out_ptr).arg(&rows).arg(&k).arg(&rb);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_dequant {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

/// Удобная обёртка над целыми буферами.
pub fn blockq_dequant(
    kernels: &BlockqDequantKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    src: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    rows: u32,
    k: u32,
) -> Result<()> {
    let src_v = src.as_view();
    let mut out_v = out.as_view_mut();
    blockq_dequant_raw(kernels, stream, dtype, &src_v, &mut out_v, rows, k)
}

/// Полоса строк `[row_off, row_off+rows)` веса NVFP4/MXFP8 движка `[N, K]`
/// → `out` `[rows, K]` (f16/bf16 по модулю). `packed`/`scales` — целые
/// буферы веса; смещение полосы считает ядро.
#[allow(clippy::too_many_arguments)]
pub fn blockq_dequant_band_syn(
    kernels: &BlockqDequantKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    packed: &CudaView<'_, u8>,
    scales: &CudaView<'_, u8>,
    out: &mut CudaViewMut<'_, u8>,
    rows: u32,
    k: u32,
    row_off: u32,
) -> Result<()> {
    if !matches!(dtype, DType::NVFP4 | DType::MXFP8) {
        return Err(SynaptixError::Unsupported("blockq_dequant_band_syn: только NVFP4/MXFP8"));
    }
    if k % 32 != 0 {
        return Err(SynaptixError::Unsupported("blockq_dequant_band_syn: K должно быть кратно 32"));
    }
    if out.len() < rows as usize * k as usize * 2 {
        return Err(SynaptixError::Unsupported("blockq_dequant_band_syn: out короче rows × k"));
    }
    let f = kernels.func(dtype)?;
    let total = rows * (k / 32);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    let mut bld = stream.launch_builder(f);
    bld.arg(packed).arg(scales).arg(&mut *out).arg(&rows).arg(&k).arg(&row_off);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_dequant_band {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

// ───────────────────────────────── GEMV ─────────────────────────────────

pub struct BlockqGemvKernels {
    _module: Arc<CudaModule>,
    fns: HashMap<&'static str, CudaFunction>,
    batched: HashMap<&'static str, CudaFunction>,
    indexed: HashMap<&'static str, CudaFunction>,
}

static GEMV_CACHE: OnceLock<Mutex<Vec<(usize, Arc<BlockqGemvKernels>)>>> = OnceLock::new();
static GEMV_CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<BlockqGemvKernels>)>>> = OnceLock::new();

const GEMV_WARPS: u32 = 4;
const GEMV_THREADS: u32 = GEMV_WARPS * 32;
pub const GEMV_MAX_M: usize = 8;
/// Опции модуля GEMV (порядок операций здесь не контракт, FMA разрешён).
pub const GEMV_MODULE_OPTS: &[&str] = &[];

impl BlockqGemvKernels {
    /// Активация и выход F16.
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, GEMV_CACHE.get_or_init(|| Mutex::new(Vec::new())), &[], "blockq_gemv.cu")
    }

    /// Активация и выход BF16.
    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            GEMV_CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["-DSYN_ACT_BF16"],
            "blockq_gemv_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<Self>)>>,
        opts: &[&str],
        tag: &'static str,
    ) -> Result<Arc<Self>> {
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock().unwrap();
            if let Some((_, v)) = g.iter().find(|(k, _)| *k == key) {
                return Ok(v.clone());
            }
        }
        let src = gemv_module_source();
        let module = compile_module_with_opts(ctx, &src, tag, opts, None)?;
        let mut fns = HashMap::new();
        let mut batched = HashMap::new();
        let mut indexed = HashMap::new();
        for suf in SUFFIXES.iter().chain(["nvfp4_syn", "mxfp8_syn"].iter()) {
            fns.insert(*suf, load_fn(&module, &format!("gemv_{suf}"))?);
            batched.insert(*suf, load_fn(&module, &format!("gemv_{suf}_batched"))?);
            indexed.insert(*suf, load_fn(&module, &format!("gemv_{suf}_indexed"))?);
        }
        let new = Arc::new(Self { _module: module, fns, batched, indexed });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }

    fn func(&self, dtype: DType) -> Result<&CudaFunction> {
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("blockq_gemv: формат не квантованный вес"))?;
        self.fns.get(suf).ok_or(SynaptixError::Unsupported("blockq_gemv: нет ядра для формата"))
    }

    fn func_batched(&self, dtype: DType) -> Result<&CudaFunction> {
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("blockq_gemv: формат не квантованный вес"))?;
        self.batched.get(suf).ok_or(SynaptixError::Unsupported("blockq_gemv: нет батчевого ядра"))
    }

    fn func_indexed(&self, dtype: DType) -> Result<&CudaFunction> {
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("blockq_gemv: формат не квантованный вес"))?;
        self.indexed.get(suf).ok_or(SynaptixError::Unsupported("blockq_gemv: нет индексного ядра"))
    }
}

/// Байт на строку веса для ядра: одноблобные — по формату, NVFP4/MXFP8 — не
/// используется (адрес считает декодер), но проверяем кратность K.
fn gemv_row_bytes(dtype: DType, k: usize) -> Result<u32> {
    if k % 32 != 0 {
        return Err(SynaptixError::Unsupported("blockq_gemv: K должно быть кратно 32"));
    }
    match dtype {
        DType::NVFP4 | DType::MXFP8 => Ok(0),
        _ => synaptix_core::quant::block_row_bytes(dtype, k)
            .map(|b| b as u32)
            .ok_or(SynaptixError::Unsupported("blockq_gemv: K не кратен блоку формата")),
    }
}

/// `out[m, N] = x[m, K] · W[N, K]ᵀ`, `m ≤ 8`. `w` — блоб веса (одноблобный —
/// `N·row_bytes`; NVFP4 — packed `[N, K/2]`; MXFP8 — `[N, K]`), `sw` — масштабы
/// NVFP4/MXFP8 (у одноблобных любой буфер, не читается). `x`/`out` — байты
/// f16 или bf16 по набору ядер; строки активации через `x_stride` элементов,
/// выхода — `out_stride`.
#[allow(clippy::too_many_arguments)]
pub fn blockq_gemv(
    kernels: &BlockqGemvKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    w: &CudaView<'_, u8>,
    sw: &CudaView<'_, u8>,
    x: &CudaView<'_, u8>,
    out: &mut CudaViewMut<'_, u8>,
    n: u32,
    k: u32,
    m: u32,
    x_stride: u32,
    out_stride: u32,
) -> Result<()> {
    if m == 0 || m as usize > GEMV_MAX_M {
        return Err(SynaptixError::Unsupported("blockq_gemv: M должно быть в 1..=8"));
    }
    let rb = gemv_row_bytes(dtype, k as usize)?;
    if n == 0 {
        return Ok(());
    }
    let f = kernels.func(dtype)?;
    let grid = n.div_ceil(GEMV_WARPS);
    let mi = m as i32;
    let mut bld = stream.launch_builder(f);
    bld.arg(w).arg(sw).arg(x).arg(&mut *out).arg(&n).arg(&k).arg(&mi).arg(&x_stride).arg(&out_stride).arg(&rb);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (GEMV_THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_gemv {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

/// Батч экспертов: `out_e[N] = x_e[K] · W_e[N, K]ᵀ` для `e` в таблицах
/// device-адресов (`w_ptrs`, `sw_ptrs`, `x_ptrs`, `out_ptrs` — по одному
/// `u64` на эксперта). Одна форма `[N, K]` у всех.
#[allow(clippy::too_many_arguments)]
pub fn blockq_gemv_batched(
    kernels: &BlockqGemvKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    w_ptrs: &CudaSlice<u64>,
    sw_ptrs: &CudaSlice<u64>,
    x_ptrs: &CudaSlice<u64>,
    out_ptrs: &CudaSlice<u64>,
    experts: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let rb = gemv_row_bytes(dtype, k as usize)?;
    if n == 0 || experts == 0 {
        return Ok(());
    }
    let f = kernels.func_batched(dtype)?;
    let grid = n.div_ceil(GEMV_WARPS);
    let mut bld = stream.launch_builder(f);
    bld.arg(w_ptrs).arg(sw_ptrs).arg(x_ptrs).arg(out_ptrs).arg(&n).arg(&k).arg(&rb);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, experts, 1), block_dim: (GEMV_THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_gemv_batched {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

/// Индексный батч экспертов: `out[p, N] = x[r_p, K] · W_{idx[p]}[N, K]ᵀ`, где
/// `idx` — U32 НА КАРТЕ, а `w_table`/`sw_table` — u64-адреса весов и масштабов
/// всех экспертов. `r_p = p` при `rows_per_pair`, иначе 0. Ни одной выгрузки
/// на хост: вызов захватывается CUDA-графом. `idx[p] ≥ experts` — строка
/// выхода не пишется.
#[allow(clippy::too_many_arguments)]
pub fn blockq_gemv_indexed(
    kernels: &BlockqGemvKernels,
    stream: &Arc<CudaStream>,
    dtype: DType,
    w_table: &CudaView<'_, u64>,
    sw_table: &CudaView<'_, u64>,
    idx: &CudaView<'_, u32>,
    x: &CudaView<'_, u8>,
    out: &mut CudaViewMut<'_, u8>,
    experts: u32,
    pairs: u32,
    n: u32,
    k: u32,
    rows_per_pair: bool,
) -> Result<()> {
    let rb = gemv_row_bytes(dtype, k as usize)?;
    if n == 0 || pairs == 0 {
        return Ok(());
    }
    let f = kernels.func_indexed(dtype)?;
    let grid = n.div_ceil(GEMV_WARPS);
    let rpp: i32 = rows_per_pair as i32;
    let mut bld = stream.launch_builder(f);
    bld.arg(w_table).arg(sw_table).arg(idx).arg(x).arg(&mut *out).arg(&n).arg(&k).arg(&rb).arg(&experts).arg(&rpp);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, pairs, 1), block_dim: (GEMV_THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_gemv_indexed {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

// ─────────────────────────── групповой GEMM экспертов ───────────────────────────

pub const GG_BM: usize = 64;
const GG_BN: u32 = 64;
const GG_THREADS: u32 = 128;

/// Ядро группового GEMM под формат веса и тип активации. Модуль собирается на
/// один формат (NVRTC на все три десятка форматов с mma-телом — секунды),
/// строка инстанцирования берётся из списка GEMV-ядер того же формата.
pub struct GroupedGemmKernel {
    _module: Arc<CudaModule>,
    func: CudaFunction,
}

type GgKey = (usize, &'static str, bool);
static GG_CACHE: OnceLock<Mutex<Vec<(GgKey, Arc<GroupedGemmKernel>)>>> = OnceLock::new();

/// `GG_BLOB(gg, FN, BB, SPB)` / `GG_SYN(gg, FN)` для суффикса — из строки
/// `GEMV_BLOB(gemv_<suf>, …)` / `GEMV_SYN(gemv_<suf>, …)` исходника GEMV.
fn gg_instantiation(suf: &str) -> Option<String> {
    let src = include_str!("../cu/elementwise/blockq_gemv.cu");
    for (from, to) in [("GEMV_BLOB(", "GG_BLOB("), ("GEMV_SYN(", "GG_SYN(")] {
        let head = format!("{from}gemv_{suf},");
        if let Some(line) = src.lines().find(|l| l.trim_start().starts_with(&head)) {
            let rest = &line.trim_start()[head.len()..];
            return Some(format!("{to}gg,{rest}\n"));
        }
    }
    None
}

impl GroupedGemmKernel {
    pub fn for_context(ctx: &Arc<CudaContext>, dtype: DType, bf16: bool) -> Result<Arc<Self>> {
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("grouped_gemm: формат не квантованный вес"))?;
        let key = (Arc::as_ptr(ctx) as usize, suf, bf16);
        let cache = GG_CACHE.get_or_init(|| Mutex::new(Vec::new()));
        if let Some((_, k)) = cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
            return Ok(k.clone());
        }
        let inst = gg_instantiation(suf).ok_or(SynaptixError::Unsupported("grouped_gemm: нет ядра для формата"))?;
        let src = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../cu/elementwise/ggml_tables.cuh"),
            include_str!("../cu/elementwise/blockq_decode.cuh"),
            include_str!("../cu/elementwise/blockq_gemm_grouped.cu"),
            inst
        );
        let opts: &[&str] = if bf16 { &["-DSYN_ACT_BF16"] } else { &[] };
        let module = compile_module_with_opts(ctx, &src, "blockq_gemm_grouped.cu", opts, None)?;
        let func = load_fn(&module, "gg")?;
        let new = Arc::new(Self { _module: module, func });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }
}

/// Тайлы строк: `(эксперт, начало, конец)` по `bm` строк (`GG_BM` у BF16-ядра,
/// `GG8_BM` у FP8) из сегментов
/// `(эксперт, начало, конец)`; пустые сегменты пропускаются.
pub fn grouped_tiles(segments: &[(u32, u32, u32)], bm: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for &(e, s, t) in segments {
        let mut r = s;
        while r < t {
            let end = (r + bm as u32).min(t);
            out.extend_from_slice(&[e, r, end, 0]);
            r = end;
        }
    }
    out
}

/// `Y[r] = X[src(r)] · W_{e(r)}ᵀ` по тайлам (`[e, r0, r1, 0]` на тайл, u32
/// на карте); `src(r) = x_rows[r]` или `r`. `x`/`y` — байты f16/bf16
/// `[·, K]` / `[R, N]`, `K % 32 == 0`, `N` чётное.
#[allow(clippy::too_many_arguments)]
pub fn blockq_gemm_grouped(
    kernel: &GroupedGemmKernel,
    stream: &Arc<CudaStream>,
    dtype: DType,
    w_table: &CudaView<'_, u64>,
    s_table: &CudaView<'_, u64>,
    tiles: &CudaSlice<u32>,
    n_tiles: u32,
    x: &CudaView<'_, u8>,
    x_rows: Option<&CudaView<'_, u32>>,
    y: &mut CudaViewMut<'_, u8>,
    n: u32,
    k: u32,
) -> Result<()> {
    if k % 32 != 0 || n % 2 != 0 {
        return Err(SynaptixError::Unsupported("grouped_gemm: K кратно 32, N чётное"));
    }
    let rb = gemv_row_bytes(dtype, k as usize)?;
    if n_tiles == 0 || n == 0 {
        return Ok(());
    }
    let grid = n.div_ceil(GG_BN);
    let mut bld = stream.launch_builder(&kernel.func);
    let null_rows: u64 = 0;
    bld.arg(w_table).arg(s_table).arg(tiles).arg(x);
    match x_rows {
        Some(r) => bld.arg(r),
        None => bld.arg(&null_rows),
    };
    bld.arg(&mut *y).arg(&n).arg(&k).arg(&rb);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, n_tiles, 1), block_dim: (GG_THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_gemm_grouped {dtype:?}: {e:?}")))?;
    }
    Ok(())
}

// ──────────────────── групповой GEMM экспертов на FP8 MMA (sm_89+) ────────────────────

/// Ядро FP8-варианта под формат веса и тип выхода (см. blockq_gemm_grouped_fp8.cu).
pub struct GroupedGemmFp8Kernel {
    _module: Arc<CudaModule>,
    func: CudaFunction,
    bm: usize,
}

/// Высота тайла FP8-ядра по сегментам: 128 (вдвое меньше перечитываний веса)
/// или 64, когда сегменты мелкие (хвостовой чанк префилла, редкие эксперты) и
/// половина тайла 128 простаивала бы.
pub fn gg8_bm(segments: &[(u32, u32, u32)]) -> usize {
    let (rows, segs) = segments
        .iter()
        .filter(|s| s.2 > s.1)
        .fold((0u64, 0u64), |(r, n), s| (r + (s.2 - s.1) as u64, n + 1));
    if segs > 0 && rows / segs >= 96 { 128 } else { 64 }
}

/// Динамическая smem FP8-ядра — `G8_SMEM_BYTES` из .cu: 3 стадии × (A по `bm`
/// строк + B по 128, × 80 байт) + масштабы.
fn gg8_smem(bm: usize) -> u32 {
    (3 * (bm * 80 + 128 * 80) + 3 * (bm + 128) * 2 * 4) as u32
}
const GG8_BN: u32 = 128;
const GG8_THREADS: u32 = 256;

type Gg8Key = (usize, &'static str, bool, usize);
static GG8_CACHE: OnceLock<Mutex<Vec<(Gg8Key, Arc<GroupedGemmFp8Kernel>)>>> = OnceLock::new();

fn gg8_instantiation(suf: &str) -> Option<String> {
    if suf == "mxfp8_syn" {
        return Some("GG8_MX(gg8)\n".into());
    }
    gg_instantiation(suf).map(|s| s.replacen("GG_BLOB(gg,", "GG8_BLOB(gg8,", 1).replacen("GG_SYN(gg,", "GG8_SYN(gg8,", 1))
}

impl GroupedGemmFp8Kernel {
    pub fn for_context(ctx: &Arc<CudaContext>, dtype: DType, bf16: bool, bm: usize) -> Result<Arc<Self>> {
        crate::caps::DeviceCaps::for_context(ctx).require(crate::caps::Feature::Fp8Mma)?;
        if bm != 64 && bm != 128 {
            return Err(SynaptixError::Unsupported("grouped_gemm_fp8: высота тайла 64 или 128"));
        }
        let suf = entry_suffix(dtype).ok_or(SynaptixError::Unsupported("grouped_gemm_fp8: формат не квантованный вес"))?;
        let key = (Arc::as_ptr(ctx) as usize, suf, bf16, bm);
        let cache = GG8_CACHE.get_or_init(|| Mutex::new(Vec::new()));
        if let Some((_, k)) = cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
            return Ok(k.clone());
        }
        let inst = gg8_instantiation(suf).ok_or(SynaptixError::Unsupported("grouped_gemm_fp8: нет ядра для формата"))?;
        let src = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../cu/elementwise/ggml_tables.cuh"),
            include_str!("../cu/elementwise/blockq_decode.cuh"),
            include_str!("../cu/elementwise/blockq_gemm_grouped_fp8.cu"),
            inst
        );
        let bm_def = format!("-DG8_BM={bm}");
        let mut opts: Vec<&str> = vec![&bm_def];
        if bf16 {
            opts.push("-DSYN_ACT_BF16");
        }
        let module = compile_module_with_opts(ctx, &src, "blockq_gemm_grouped_fp8.cu", &opts, None)?;
        let func = load_fn(&module, "gg8")?;
        func.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            gg8_smem(bm) as i32,
        )
        .map_err(|e| SynaptixError::Cuda(format!("grouped_gemm_fp8: лимит smem: {e:?}")))?;
        let new = Arc::new(Self { _module: module, func, bm });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }
}

/// FP8-вариант [`blockq_gemm_grouped`]: `xq`/`xs` — активация в MXFP8
/// (natural: байты `[·, K]`, E8M0 `[·, K/32]`).
#[allow(clippy::too_many_arguments)]
pub fn blockq_gemm_grouped_fp8(
    kernel: &GroupedGemmFp8Kernel,
    stream: &Arc<CudaStream>,
    dtype: DType,
    w_table: &CudaView<'_, u64>,
    s_table: &CudaView<'_, u64>,
    tiles: &CudaSlice<u32>,
    n_tiles: u32,
    xq: &CudaView<'_, u8>,
    xs: &CudaView<'_, u8>,
    x_rows: Option<&CudaView<'_, u32>>,
    y: &mut CudaViewMut<'_, u8>,
    n: u32,
    k: u32,
) -> Result<()> {
    if k % 32 != 0 || n % 2 != 0 {
        return Err(SynaptixError::Unsupported("grouped_gemm_fp8: K кратно 32, N чётное"));
    }
    let rb = gemv_row_bytes(dtype, k as usize)?;
    if n_tiles == 0 || n == 0 {
        return Ok(());
    }
    let grid = n.div_ceil(GG8_BN);
    let mut bld = stream.launch_builder(&kernel.func);
    let null_rows: u64 = 0;
    bld.arg(w_table).arg(s_table).arg(tiles).arg(xq).arg(xs);
    match x_rows {
        Some(r) => bld.arg(r),
        None => bld.arg(&null_rows),
    };
    bld.arg(&mut *y).arg(&n).arg(&k).arg(&rb);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, n_tiles, 1), block_dim: (GG8_THREADS, 1, 1), shared_mem_bytes: gg8_smem(kernel.bm) })
            .map_err(|e| SynaptixError::Cuda(format!("launch blockq_gemm_grouped_fp8 {dtype:?}: {e:?}")))?;
    }
    Ok(())
}
