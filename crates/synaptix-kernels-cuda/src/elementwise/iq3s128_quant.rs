//! Кодировщик IQ3_S-128 на карте (`GgmlType::Iq3S128`): `[rows, k]` f16/bf16 →
//! блоб по 56 байт на 128 значений. Модуль собирается с `--fmad=false` и
//! повторяет CPU-эталон `quantize_iq3_s128` бит в бит (тест `cuda_iq3s128_quant`).

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::quant::ggml::IQ3S128_BYTES;

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct Iq3s128QuantKernels {
    _module: Arc<CudaModule>,
    quant: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Iq3s128QuantKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<Iq3s128QuantKernels>)>>> = OnceLock::new();

impl Iq3s128QuantKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), &["--fmad=false"], "iq3s128_quant.cu")
    }

    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())), &["--fmad=false", "-DSYN_IN_BF16"], "iq3s128_quant_bf16.cu")
    }

    fn build(ctx: &Arc<CudaContext>, cache: &Mutex<Vec<(usize, Arc<Self>)>>, opts: &[&str], tag: &'static str) -> Result<Arc<Self>> {
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock().unwrap();
            if let Some((_, v)) = g.iter().find(|(k, _)| *k == key) {
                return Ok(v.clone());
            }
        }
        let src = format!(
            "{}\n{}",
            include_str!("../cu/elementwise/ggml_tables.cuh"),
            include_str!("../cu/elementwise/iq3s128_quant.cu")
        );
        let module = compile_module_with_opts(ctx, &src, tag, opts, None)?;
        let new = Arc::new(Self { quant: load_fn(&module, "iq3s128_quant")?, _module: module });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }
}

/// `input` — `rows × k` значений f16/bf16 подряд, `imp` — важность столбцов
/// (`k` f32) или `None`, `out` — `rows × k/128 × 56` байт.
pub fn iq3s128_quant(
    kernels: &Iq3s128QuantKernels,
    stream: &Arc<CudaStream>,
    input: &CudaView<'_, u8>,
    imp: Option<&CudaSlice<f32>>,
    out: &mut CudaViewMut<'_, u8>,
    rows: u32,
    k: u32,
) -> Result<()> {
    if k % 128 != 0 {
        return Err(SynaptixError::Unsupported("iq3s128_quant: K должно быть кратно 128"));
    }
    let need = rows as usize * (k as usize / 128) * IQ3S128_BYTES;
    if out.len() < need || input.len() < rows as usize * k as usize * 2 {
        return Err(SynaptixError::Unsupported("iq3s128_quant: буферы короче формы"));
    }
    if rows == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (rows * (k / 128), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let null: u64 = 0;
    let mut b = stream.launch_builder(&kernels.quant);
    b.arg(input);
    match imp {
        Some(h) => b.arg(h),
        None => b.arg(&null),
    };
    b.arg(out).arg(&rows).arg(&k);
    unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch iq3s128_quant: {e:?}")))?;
    Ok(())
}
