//! DeltaNet рекуррентный шаг (без гейта).
//!
//! Частный случай gated delta rule с α≡1 (нет decay), без L2-нормализации и
//! q_scale: `S_t = S_{t-1} + β_t·k_t·(v_t − S_{t-1}ᵀk_t)ᵀ`, `o_t = S_tᵀ·q_t`.
//! Один block = (batch, head); block_dim = hk.
//!
//! Исходник CUDA: `src/cu/fused/ssm/delta_rule.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct DeltaRuleKernels {
    _module: Arc<CudaModule>,
    step: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<DeltaRuleKernels>)>>> = OnceLock::new();

impl DeltaRuleKernels {
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
        let src = include_str!("../cu/fused/ssm/delta_rule.cu");
        let module = compile_module(ctx, src, "delta_rule.cu")?;
        let new = Arc::new(Self {
            step: load_fn(&module, "delta_rule_step_f32")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// Один рекуррентный шаг DeltaNet. `state` `(B,H,HK,HV)` in/out,
    /// `out` `(B,H,HV)`. Block layout: grid `(B,H)`, block `(hk)`, shared
    /// `(hk + 2)` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_rule_step(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        b: u32,
        h: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        let shared_bytes = ((hk + 2) as usize * std::mem::size_of::<f32>()) as u32;
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
            .arg(beta)
            .arg(state)
            .arg(out)
            .arg(&b)
            .arg(&h)
            .arg(&hk)
            .arg(&hv);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch delta_rule_step: {e:?}")))?;
        }
        Ok(())
    }
}
