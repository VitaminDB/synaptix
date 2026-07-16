//! Fused «adaLN-модуляция + NVFP4-квант» (эпилог нормы):
//! `y = rms(x)·(1+scale)+shift` (бит-в-бит со старой цепочкой rms_no_gain →
//! add_scalar → broadcast_mul → broadcast_add) + `(packed, scales) = quant(f16(y))`
//! (бит-в-бит с quantize_f16_to_nvfp4_fast). Один launch вместо ~6, 3r+3w
//! DRAM-проходов вместо ~10. Для LTX DiT (scale/shift по-токенные [B,T,D]).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RmsModQuantParams {
    pub batch: i32,
    pub batch_cov: i32,
    pub hidden: i32,
    pub eps: f32,
    pub sf_inner_dim: i32,
}
unsafe impl DeviceRepr for RmsModQuantParams {}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormQuantKind {
    /// rms_no_gain·(1+scale)+shift (LTX adaLN, по-токенная модуляция)
    RmsMod,
    /// LN·(1+scale)+shift (FLUX adaLN, per-batch модуляция)
    LnMod,
    /// rms·w (LLM prefill, без модуляции; mod_div игнор; qwen → 1+w)
    RmsW { qwen: bool },
}

pub struct RmsModQuantKernels {
    _module: Arc<CudaModule>,
    f16: CudaFunction,
    bf16: CudaFunction,
    ln_f16: CudaFunction,
    ln_bf16: CudaFunction,
    w_f16: CudaFunction,
    w_bf16: CudaFunction,
    mx_f16: CudaFunction,
    mx_bf16: CudaFunction,
    mx_ln_f16: CudaFunction,
    mx_ln_bf16: CudaFunction,
    mx_w_f16: CudaFunction,
    mx_w_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<RmsModQuantKernels>)>>> = OnceLock::new();

impl RmsModQuantKernels {
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
        let src = include_str!("../cu/fused/norm/rms_mod_quant.cu");
        let module = compile_module_with_opts(ctx, src, "rms_mod_quant.cu", &[], Some("sm_80"))?;
        let new = Arc::new(Self {
            f16: load_fn(&module, "rms_mod_quant_nvfp4_f16")?,
            bf16: load_fn(&module, "rms_mod_quant_nvfp4_bf16")?,
            ln_f16: load_fn(&module, "ln_mod_quant_nvfp4_f16")?,
            ln_bf16: load_fn(&module, "ln_mod_quant_nvfp4_bf16")?,
            w_f16: load_fn(&module, "rms_w_quant_nvfp4_f16")?,
            w_bf16: load_fn(&module, "rms_w_quant_nvfp4_bf16")?,
            mx_f16: load_fn(&module, "rms_mod_quant_mxfp8_f16")?,
            mx_bf16: load_fn(&module, "rms_mod_quant_mxfp8_bf16")?,
            mx_ln_f16: load_fn(&module, "ln_mod_quant_mxfp8_f16")?,
            mx_ln_bf16: load_fn(&module, "ln_mod_quant_mxfp8_bf16")?,
            mx_w_f16: load_fn(&module, "rms_w_quant_mxfp8_f16")?,
            mx_w_bf16: load_fn(&module, "rms_w_quant_mxfp8_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `x/scale/shift/y` — байтовые буферы dtype (F16|BF16) длины m·k элементов.
/// NVFP4 (`mxfp8=false`): `packed` u8 `[m·k/2]`, `scales` u8
/// `[nvfp4_scale_buffer_size(m,k)]`. MXFP8 (`mxfp8=true`): `packed` u8 `[m·k]`,
/// `scales` natural u8 `[m·k/32]`.
#[allow(clippy::too_many_arguments)]
pub fn run_rms_mod_quant_u8(
    kernels: &RmsModQuantKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    scale: &CudaSlice<u8>,
    scale_off: usize,
    shift: &CudaSlice<u8>,
    shift_off: usize,
    y: &mut CudaSlice<u8>,
    packed: &mut CudaSlice<u8>,
    scales_out: &mut CudaSlice<u8>,
    m: u32,
    k: u32,
    eps: f32,
    dtype: DType,
    kind: NormQuantKind,
    // строк на один вектор scale/shift (broadcast по батчу; по-токенно = 1).
    mod_div: u32,
    mxfp8: bool,
) -> Result<()> {
    if m == 0 || k == 0 {
        return Ok(());
    }
    if mxfp8 {
        if k % 32 != 0 {
            return Err(SynaptixError::Unsupported("rms_mod_quant: K%32 != 0 (mxfp8)"));
        }
    } else if k % 16 != 0 {
        return Err(SynaptixError::Unsupported("rms_mod_quant: K%16 != 0"));
    }
    let m_cov = m.div_ceil(128) * 128;
    let params = RmsModQuantParams {
        batch: m as i32,
        batch_cov: m_cov as i32,
        hidden: k as i32,
        eps,
        sf_inner_dim: (k.div_ceil(64) * 4) as i32,
    };
    // Блок = бит-контракт редукции базового ядра: rms_norm поднимает блок при
    // m<=8 (launch_cfg), layernorm — всегда 256.
    let block = if !matches!(kind, NormQuantKind::LnMod) && m <= 8 {
        k.next_multiple_of(32).clamp(BLOCK, 1024)
    } else {
        BLOCK
    };
    // mxfp8: natural-scales без 128-хвоста → грид ровно m строк.
    let grid = if mxfp8 { m } else { m_cov };
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        // динамический smem: f16-копия строки для квант-фазы
        shared_mem_bytes: k * 2,
    };
    let esz = (dtype.size_in_bits() / 8) as usize;
    let n = (m as usize) * (k as usize);

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + n * esz)
                    .transmute::<$t>(n)
                    .ok_or_else(|| SynaptixError::Cuda("rms_mod_quant: transmute x".into()))?
            };
            let n_mod = if matches!(kind, NormQuantKind::RmsW { .. }) {
                k as usize
            } else {
                n / (mod_div as usize)
            };
            let s_v = unsafe {
                scale
                    .slice(scale_off..scale_off + n_mod * esz)
                    .transmute::<$t>(n_mod)
                    .ok_or_else(|| SynaptixError::Cuda("rms_mod_quant: transmute scale".into()))?
            };
            let b_v = unsafe {
                shift
                    .slice(shift_off..shift_off + n_mod * esz)
                    .transmute::<$t>(n_mod)
                    .ok_or_else(|| SynaptixError::Cuda("rms_mod_quant: transmute shift".into()))?
            };
            let mut y_s = y.slice_mut(0..n * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$t>(n)
                    .ok_or_else(|| SynaptixError::Cuda("rms_mod_quant: transmute y".into()))?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&x_v)
                .arg(&s_v)
                .arg(&b_v)
                .arg(&mut y_v)
                .arg(&mut *packed)
                .arg(&mut *scales_out)
                .arg(&params);
            let md: i32 = match kind {
                NormQuantKind::RmsMod => 0,
                NormQuantKind::LnMod => mod_div as i32,
                NormQuantKind::RmsW { qwen } => qwen as i32,
            };
            if !matches!(kind, NormQuantKind::RmsMod) {
                b.arg(&md);
            }
            unsafe {
                b.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch rms_mod_quant: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => go!(
            half::f16,
            match (kind, mxfp8) {
                (NormQuantKind::RmsMod, false) => &kernels.f16,
                (NormQuantKind::LnMod, false) => &kernels.ln_f16,
                (NormQuantKind::RmsW { .. }, false) => &kernels.w_f16,
                (NormQuantKind::RmsMod, true) => &kernels.mx_f16,
                (NormQuantKind::LnMod, true) => &kernels.mx_ln_f16,
                (NormQuantKind::RmsW { .. }, true) => &kernels.mx_w_f16,
            }
        ),
        DType::BF16 => go!(
            half::bf16,
            match (kind, mxfp8) {
                (NormQuantKind::RmsMod, false) => &kernels.bf16,
                (NormQuantKind::LnMod, false) => &kernels.ln_bf16,
                (NormQuantKind::RmsW { .. }, false) => &kernels.w_bf16,
                (NormQuantKind::RmsMod, true) => &kernels.mx_bf16,
                (NormQuantKind::LnMod, true) => &kernels.mx_ln_bf16,
                (NormQuantKind::RmsW { .. }, true) => &kernels.mx_w_bf16,
            }
        ),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "rms_mod_quant: dtype {other:?} не поддержан"
            )))
        }
    }
    Ok(())
}
