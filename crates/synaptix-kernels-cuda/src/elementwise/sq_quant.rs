//! Энкодер SQ на карте: `[rows, k]` f16/bf16 → блоб SQ`b`. Модуль
//! `sq_quant.cu` собирается с `--fmad=false` и повторяет CPU-эталон
//! `synaptix_core::quant::sq::quantize_matrix` бит в бит (тест
//! `cuda_sq_quant`). Поток — под-блок из 32 значений, восемь соседних —
//! супер-блок.

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::quant::sq;

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct SqQuantKernels {
    _module: Arc<CudaModule>,
    quant: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<SqQuantKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<SqQuantKernels>)>>> = OnceLock::new();

impl SqQuantKernels {
    /// Вход F16.
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), &["--fmad=false"], "sq_quant.cu")
    }

    /// Вход BF16.
    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["--fmad=false", "-DSYN_IN_BF16"],
            "sq_quant_bf16.cu",
        )
    }

    fn build(ctx: &Arc<CudaContext>, cache: &Mutex<Vec<(usize, Arc<Self>)>>, opts: &[&str], tag: &'static str) -> Result<Arc<Self>> {
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock().unwrap();
            if let Some((_, v)) = g.iter().find(|(k, _)| *k == key) {
                return Ok(v.clone());
            }
        }
        let src = include_str!("../cu/elementwise/sq_quant.cu");
        let module = compile_module_with_opts(ctx, src, tag, opts, None)?;
        let new = Arc::new(Self { quant: load_fn(&module, "sq_quant")?, _module: module });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }
}

/// `input` — `rows × k` значений f16/bf16 (по модулю) подряд; `out` —
/// `rows × row_bytes(bits, k)` байт.
pub fn sq_quant(
    kernels: &SqQuantKernels,
    stream: &Arc<CudaStream>,
    input: &CudaView<'_, u8>,
    out: &mut CudaViewMut<'_, u8>,
    bits: u8,
    rows: u32,
    k: u32,
) -> Result<()> {
    sq::check_bits(bits)?;
    if k % sq::SUB_BLOCK as u32 != 0 {
        return Err(SynaptixError::Unsupported("sq_quant: K должно быть кратно 32"));
    }
    let need = rows as usize * sq::row_bytes(bits, k as usize);
    if out.len() < need {
        return Err(SynaptixError::Unsupported("sq_quant: out короче rows × row_bytes"));
    }
    if input.len() < rows as usize * k as usize * 2 {
        return Err(SynaptixError::Unsupported("sq_quant: вход короче rows × k"));
    }
    if rows == 0 {
        return Ok(());
    }
    let supers = (k as usize).div_ceil(sq::SUPER_BLOCK) as u32;
    let total = rows * supers * sq::SUBS as u32;
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    let bits_u = bits as u32;
    let mut bld = stream.launch_builder(&kernels.quant);
    bld.arg(input).arg(&mut *out).arg(&rows).arg(&k).arg(&bits_u);
    unsafe {
        bld.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| SynaptixError::Cuda(format!("launch sq_quant: {e:?}")))?;
    }
    Ok(())
}
