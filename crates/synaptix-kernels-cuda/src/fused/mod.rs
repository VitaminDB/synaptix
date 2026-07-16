//! Fused kernels: QKV, attn+RoPE, LN+residual, SwiGLU.

pub mod attn_rope;
pub mod cross_entropy;
pub mod geglu;
pub mod geglu_split;
pub mod layernorm_residual;
pub mod moe_dispatch;
pub mod qkv_proj;
pub mod rms_mod_quant;
pub mod rmsnorm_residual;
pub mod silu_and_mul;
pub mod swiglu;
