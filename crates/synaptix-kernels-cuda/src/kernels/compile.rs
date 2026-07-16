use std::path::Path;
use std::sync::{Arc, RwLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};
use synaptix_core::error::{Result, SynaptixError};

static CUDA_INCLUDE: RwLock<Option<String>> = RwLock::new(None);

pub fn set_cuda_include(path: Option<String>) {
    *CUDA_INCLUDE.write().unwrap() = path;
}

pub fn compile_module(
    ctx: &Arc<CudaContext>,
    src: &str,
    tag: &'static str,
) -> Result<Arc<CudaModule>> {
    compile_module_with_opts(ctx, src, tag, &[], Some("sm_80"))
}

pub fn compile_module_with_opts(
    ctx: &Arc<CudaContext>,
    src: &str,
    tag: &'static str,
    extra_options: &[&str],
    arch: Option<&'static str>,
) -> Result<Arc<CudaModule>> {
    let mut include_paths = Vec::new();
    if let Some(p) = CUDA_INCLUDE.read().unwrap().clone() {
        include_paths.push(p);
    }
    for candidate in &[
        "/opt/cuda/include",
        "/usr/local/cuda/include",
        "/usr/include/cuda",
    ] {
        if Path::new(candidate).join("cuda_fp16.h").is_file() {
            include_paths.push((*candidate).to_string());
        }
    }
    let mut options: Vec<String> = extra_options.iter().map(|s| (*s).to_string()).collect();
    options.push("-lineinfo".to_string());
    let opts = CompileOptions {
        arch: Some(arch.unwrap_or("sm_80")),
        include_paths,
        options,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, opts).map_err(|e| {
        let log_str = match &e {
            CompileError::CompileError { log, .. } => log.to_string_lossy().to_string(),
            other => format!("{other:?}"),
        };
        SynaptixError::Cuda(format!("nvrtc {tag}: {log_str}"))
    })?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| SynaptixError::Cuda(format!("load_module {tag}: {e:?}")))?;
    Ok(module)
}

pub fn load_fn(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction> {
    module
        .load_function(name)
        .map_err(|e| SynaptixError::Cuda(format!("load_function {name}: {e:?}")))
}
