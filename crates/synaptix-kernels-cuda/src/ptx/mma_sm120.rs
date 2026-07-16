//! sm_120 NVFP4 warp-level `mma.sync` — каноничная ссылка на инструкцию.
//!
//! На RTX 5090 (sm_120, GB202) нет tcgen05/TMEM, поэтому native FP4 Tensor Cores
//! доступны через **warp-level** `mma.sync.aligned.kind::mxf4nvf4.block_scale`.
//! Сама инструкция эмитится как inline-PTX в device-коде (`.cu`), не из Rust;
//! этот модуль документирует её форму и перечисляет ядра-эмиттеры, чтобы
//! `ptx::mma_sm120` был единой точкой для понимания FP4-MMA пути.
//!
//! Инструкция (m16n8k64, FP4 e2m1 × e2m1, F32-аккумулятор, ue4m3 scales):
//! ```ptx
//! mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//! ```
//! Требует `compute_120f` (CUDA 13.0+) для полного feature set.
//!
//! Ядра-эмиттеры (inline-asm): `src/cu/gemm/nvfp4_mma_gemm_shuf.cu`, `src/cu/gemm/nvfp4_mma_gemv_shuf.cu`,
//! `src/cu/fused/projection/nvfp4_qkv_proj_shuf.cu`, `src/cu/fused/mlp/nvfp4_swiglu_shuf.cu`, `src/cu/fused/mlp/nvfp4_geglu_shuf.cu`.

/// Полная PTX-строка инструкции NVFP4 `mma.sync` для sm_120a (m16n8k64,
/// block-scale `scale_vec::4X`, F32-аккумулятор). Совпадает с inline-asm в
/// `.cu`-ядрах; держим здесь как единый эталон для сверки/документации.
pub const MMA_MXF4NVF4_M16N8K64: &str =
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3";
