//! Gated DeltaNet рекуррентный шаг (decode T=1) + fused RmsNormGated.
//!
//! Портировано из `ai-quant/src/kernels/fla.rs` (валидировано bit-exact в проде
//! Qwen3.6). Один block = (batch, head); block_dim = hk; каждый thread держит
//! один key-канал. State `(B, H, HK, HV)` обновляется in-place.
//!
//! Рекуррентность: `S_t = g_t·S_{t-1} + β_t·k_t·(v_t − (g_t·S_{t-1})ᵀk_t)ᵀ`,
//! `o_t = S_tᵀ·q_t`, где q/k нормализованы L2 по hk, q·=q_scale, g_t=exp(g).
//!
//! Исходник CUDA: `src/cu/fused/ssm/gated_delta_rule.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct GatedDeltaRuleKernels {
    _module: Arc<CudaModule>,
    step: CudaFunction,
    step_fused_rms: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GatedDeltaRuleKernels>)>>> = OnceLock::new();

impl GatedDeltaRuleKernels {
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
        let src = include_str!("../cu/fused/ssm/gated_delta_rule.cu");
        let module = compile_module(ctx, src, "gated_delta_rule.cu")?;
        let new = Arc::new(Self {
            step: load_fn(&module, "gated_delta_rule_step_f32")?,
            step_fused_rms: load_fn(&module, "gated_delta_rule_step_fused_rms_norm_f32_to_f16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// Сырой handle fused-rms-ядра (для оркестратора linear-decode, который
    /// строит launch напрямую поверх device-резидентных views).
    pub(crate) fn step_fused_rms_fn(&self) -> &CudaFunction {
        &self.step_fused_rms
    }

    /// Один рекуррентный шаг gated delta rule. `state` `(B,H,HK,HV)` in/out,
    /// `out` `(B,H,HV)`. q/k/v/g/beta — F32. Block layout: grid `(B,H)`,
    /// block `(hk)`, shared `(3*hk+4)` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_step(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q_scale: f32,
        b: u32,
        h: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        let shared_bytes = ((3 * hk + 4) as usize * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (b, h, 1),
            block_dim: (hk, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&self.step);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(g)
            .arg(beta)
            .arg(state)
            .arg(out)
            .arg(&q_scale)
            .arg(&b)
            .arg(&h)
            .arg(&hk)
            .arg(&hv);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch gated_delta_rule_step: {e:?}")))?;
        }
        Ok(())
    }

    /// Fused: `gated_delta_rule_step` + RmsNormGated в один launch. SSM-выход
    /// (F32) хранится в shared, RMS-фаза тем же block:
    /// `out_f16 = weight · x / sqrt(mean(x²)+eps) · silu(gate_f16)`.
    /// Требует `hk == hv`. Shared `(3*hk + hv + 4)` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_step_fused_rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        gate_f16: &CudaSlice<f16>,
        weight_f16: &CudaSlice<f16>,
        out_f16: &mut CudaSlice<f16>,
        q_scale: f32,
        eps: f32,
        b: u32,
        h: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        debug_assert_eq!(hk, hv, "fused gdr+rms требует hk == hv");
        let shared_bytes = ((3 * hk + hv + 4) as usize * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (b, h, 1),
            block_dim: (hk, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&self.step_fused_rms);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(g)
            .arg(beta)
            .arg(state)
            .arg(gate_f16)
            .arg(weight_f16)
            .arg(out_f16)
            .arg(&q_scale)
            .arg(&eps)
            .arg(&b)
            .arg(&h)
            .arg(&hk)
            .arg(&hv);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!(
                    "launch gated_delta_rule_step_fused_rms_norm_f32_to_f16: {e:?}"
                ))
            })?;
        }
        Ok(())
    }
}
