//! Dispatch attention kernels.

pub mod chunk_fla;
pub mod flash_bf16;
pub mod flash_blocks;
pub mod flash_decode;
pub mod flash_decode_mxfp8_v2;
pub mod flash_mxfp8_prefill;
pub mod flash_mxfp8_splitq;
pub mod flash_splitq;
pub mod linear_attn_raw;
pub mod linear_decode;
pub mod linear_prefill;
pub mod sdpa_f32_acc;
