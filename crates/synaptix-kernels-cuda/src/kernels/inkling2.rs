//! Ядра Inkling-2 (`cu/fused/inkling2.cu`): запуск по имени с общей структурой
//! аргументов [`synaptix_core::backend::Ink2Launch`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const NAMES: &[&str] = &[
    "ink2_rms", "ink2_mix", "ink2_conv", "ink2_qkv", "ink2_attn", "ink2_act", "ink2_route", "ink2_combine", "ink2_logits",
];

pub struct Inkling2Kernels {
    _module: Arc<CudaModule>,
    fns: HashMap<&'static str, CudaFunction>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Inkling2Kernels>)>>> = OnceLock::new();

impl Inkling2Kernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let module = compile_module(ctx, include_str!("../cu/fused/inkling2.cu"), "inkling2.cu")?;
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
            .ok_or_else(|| SynaptixError::Cuda(format!("inkling2: нет ядра {kernel}")))?;
        let arg = Arg(*a);
        let mut b = stream.launch_builder(f);
        b.arg(&arg);
        let cfg = LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch {kernel}: {e:?}")))?;
        Ok(())
    }
}
