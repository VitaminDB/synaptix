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
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<BlockqDequantKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<BlockqDequantKernels>)>>> = OnceLock::new();

/// Имя точки входа для формата; `None` — формат не одноблобный или не
/// является форматом весов.
pub fn entry_name(dtype: DType) -> Option<&'static str> {
    Some(match dtype {
        DType::Sq { bits: 1 } => "deq_sq1",
        DType::Sq { bits: 2 } => "deq_sq2",
        DType::Sq { bits: 3 } => "deq_sq3",
        DType::Sq { bits: 4 } => "deq_sq4",
        DType::Sq { bits: 5 } => "deq_sq5",
        DType::Sq { bits: 6 } => "deq_sq6",
        DType::Sq { bits: 7 } => "deq_sq7",
        DType::Sq { bits: 8 } => "deq_sq8",
        DType::Ggml(t) => {
            use GgmlType::*;
            match t {
                Q4_0 => "deq_q4_0",
                Q4_1 => "deq_q4_1",
                Q5_0 => "deq_q5_0",
                Q5_1 => "deq_q5_1",
                Q8_0 => "deq_q8_0",
                Q8_1 => "deq_q8_1",
                Q8K => "deq_q8_k",
                Q1_0 => "deq_q1_0",
                Q2_0 => "deq_q2_0",
                Mxfp4 => "deq_mxfp4",
                Nvfp4 => "deq_nvfp4",
                Iq4Nl => "deq_iq4_nl",
                Q2K => "deq_q2_k",
                Q3K => "deq_q3_k",
                Q4K => "deq_q4_k",
                Q5K => "deq_q5_k",
                Q6K => "deq_q6_k",
                Iq4Xs => "deq_iq4_xs",
                Iq2Xxs => "deq_iq2_xxs",
                Iq2Xs => "deq_iq2_xs",
                Iq2S => "deq_iq2_s",
                Iq3Xxs => "deq_iq3_xxs",
                Iq3S => "deq_iq3_s",
                Iq1S => "deq_iq1_s",
                Iq1M => "deq_iq1_m",
                Tq1_0 => "deq_tq1_0",
                Tq2_0 => "deq_tq2_0",
                _ => return None,
            }
        }
        _ => return None,
    })
}

const ALL_ENTRIES: &[&str] = &[
    "deq_sq1", "deq_sq2", "deq_sq3", "deq_sq4", "deq_sq5", "deq_sq6", "deq_sq7", "deq_sq8",
    "deq_q4_0", "deq_q4_1", "deq_q5_0", "deq_q5_1", "deq_q8_0", "deq_q8_1", "deq_q8_k",
    "deq_q1_0", "deq_q2_0", "deq_mxfp4", "deq_nvfp4", "deq_iq4_nl", "deq_q2_k", "deq_q3_k",
    "deq_q4_k", "deq_q5_k", "deq_q6_k", "deq_iq4_xs", "deq_iq2_xxs", "deq_iq2_xs", "deq_iq2_s",
    "deq_iq3_xxs", "deq_iq3_s", "deq_iq1_s", "deq_iq1_m", "deq_tq1_0", "deq_tq2_0",
];

/// Исходник модуля: таблицы ggml + ядра (для матрицы архитектур в тестах).
pub fn module_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../cu/elementwise/ggml_tables.cuh"),
        include_str!("../cu/elementwise/blockq_dequant.cu")
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
        for name in ALL_ENTRIES {
            fns.insert(*name, load_fn(&module, name)?);
        }
        let new = Arc::new(Self { _module: module, fns });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }

    fn func(&self, dtype: DType) -> Result<&CudaFunction> {
        let name = entry_name(dtype).ok_or(SynaptixError::Unsupported("blockq_dequant: формат не одноблобный"))?;
        self.fns.get(name).ok_or(SynaptixError::Unsupported("blockq_dequant: нет точки входа"))
    }
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
