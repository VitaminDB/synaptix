//! Fused QKV projection + RoPE rotation для Q и K (F16, NVFP4 weights).
//!
//! Public API объединяет три последовательных CUDA launch'а на одном stream:
//! сначала `nvfp4_qkv_proj_shuf_f16` (Q/K/V одной триадой mma на shuffled
//! weights), потом `rope_apply_partial_f16` дважды (Q и K). На decode (T=1)
//! это устраняет промежуточные host-sync'и между QKV и RoPE и упаковывает
//! пять host-вызовов в один Rust entry point.
//!
//! Caller передаёт раздельные pre-RoPE и post-RoPE буферы для Q и K, чтобы
//! RoPE-kernel читал и писал в разные буферы (in-place вызвал бы race).
//! V не вращается и пишется сразу финально.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use synaptix_core::error::Result;

use crate::elementwise::rope::{apply_partial_f16, RopeKernels};
use crate::fused::qkv_proj::{nvfp4_qkv_proj_shuf_f16, Nvfp4QkvProjShufKernels};

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_rope_f16(
    qkv_kernels: &Nvfp4QkvProjShufKernels,
    rope_kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    packed_w_q: &CudaSlice<u8>,
    scales_w_q: &CudaSlice<u8>,
    packed_w_k: &CudaSlice<u8>,
    scales_w_k: &CudaSlice<u8>,
    packed_w_v: &CudaSlice<u8>,
    scales_w_v: &CudaSlice<u8>,
    packed_x: &CudaSlice<u8>,
    scales_x: &CudaSlice<u8>,
    out_q_proj: &mut CudaSlice<f16>,
    out_q_roped: &mut CudaSlice<f16>,
    out_k_proj: &mut CudaSlice<f16>,
    out_k_roped: &mut CudaSlice<f16>,
    out_v: &mut CudaSlice<f16>,
    cos_table: &CudaSlice<f16>,
    sin_table: &CudaSlice<f16>,
    start_pos_dev: &CudaSlice<u32>,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    rotary_dim: u32,
    k: u32,
) -> Result<()> {
    let n_q = h_q * head_dim;
    let n_kv = h_kv * head_dim;
    nvfp4_qkv_proj_shuf_f16(
        qkv_kernels,
        stream,
        packed_w_q,
        scales_w_q,
        packed_w_k,
        scales_w_k,
        packed_w_v,
        scales_w_v,
        packed_x,
        scales_x,
        out_q_proj,
        out_k_proj,
        out_v,
        n_q,
        n_kv,
        n_kv,
        k,
    )?;
    apply_partial_f16(
        rope_kernels,
        stream,
        out_q_proj,
        out_q_roped,
        cos_table,
        sin_table,
        start_pos_dev,
        1,
        h_q,
        1,
        head_dim,
        rotary_dim,
    )?;
    apply_partial_f16(
        rope_kernels,
        stream,
        out_k_proj,
        out_k_roped,
        cos_table,
        sin_table,
        start_pos_dev,
        1,
        h_kv,
        1,
        head_dim,
        rotary_dim,
    )?;
    Ok(())
}
