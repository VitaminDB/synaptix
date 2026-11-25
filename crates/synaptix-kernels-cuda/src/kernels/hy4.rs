//! Ядра Hy4-flash (`cu/fused/hy4.cu`): запуск по имени с общей структурой
//! аргументов [`synaptix_core::backend::Ink2Launch`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const NAMES: &[&str] = &[
    "hy4_hc_pre", "hy4_hc_post", "hy4_rms", "hy4_rope", "hy4_mla_attn", "hy4_cache_put", "hy4_idx_pool", "hy4_idx_score", "hy4_engram_gate", "hy4_gate_mul", "hy4_head_gemv",
];

pub struct Hy4Kernels {
    _module: Arc<CudaModule>,
    fns: HashMap<&'static str, CudaFunction>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Hy4Kernels>)>>> = OnceLock::new();

impl Hy4Kernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let module = compile_module(ctx, include_str!("../cu/fused/hy4.cu"), "hy4.cu")?;
        let mut fns = HashMap::new();
        for n in NAMES {
            fns.insert(*n, load_fn(&module, n)?);
        }
        let new = Arc::new(Self { _module: module, fns });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        kernel: &str,
        a: &synaptix_core::backend::Ink2Launch,
        grid: (u32, u32, u32),
        block: u32,
    ) -> Result<()> {
        #[repr(transparent)]
        struct Arg(synaptix_core::backend::Ink2Launch);
        unsafe impl cudarc::driver::DeviceRepr for Arg {}
        let f = self
            .fns
            .get(kernel)
            .ok_or_else(|| SynaptixError::Cuda(format!("hy4: нет ядра {kernel}")))?;
        let arg = Arg(*a);
        let mut b = stream.launch_builder(f);
        b.arg(&arg);
        let cfg = LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch {kernel}: {e:?}")))?;
        Ok(())
    }
}
