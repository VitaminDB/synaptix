//! NVRTC runtime-компиляция + кеш модулей.
//!
//! Единая точка входа NVRTC живёт в [`crate::kernels::compile`]
//! (`compile_module` / `compile_module_with_opts` / `load_fn`); здесь — её
//! публичный ре-экспорт, чтобы `synaptix_kernels_cuda::nvrtc::*` был
//! каноничным путём к компиляции `.cu` через NVRTC.

pub use crate::kernels::compile::{compile_module, compile_module_with_opts, load_fn};
