use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;
use crate::attention::softmax::sliding_window::sliding_window_attention;

pub struct NsaConfig {
    pub block_size: usize,
    pub window_size: usize,
}

impl Default for NsaConfig {
    fn default() -> Self {
        Self { block_size: 32, window_size: 128 }
    }
}

fn block_pool_mean(t: &Tensor, block_size: usize) -> Result<Tensor> {
    let dims = t.dims();
    if dims.len() != 4 {
        return Err(SynaptixError::Unsupported("nsa pool: rank-4"));
    }
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    if s % block_size != 0 {
        return Err(SynaptixError::Unsupported(
            "nsa pool: seq must divide by block_size",
        ));
    }
    let nb = s / block_size;
    let reshaped = t.reshape(vec![b, h, nb, block_size, d])?;
    let summed = reshaped.sum_keepdim(3)?.squeeze(3)?;
    summed.affine(1.0 / block_size as f32, 0.0)
}

fn block_causal_mask(s_q: usize, s_kv: usize, block_size: usize, device: Device) -> Result<Tensor> {
    let mut data = vec![0.0_f32; s_q * s_kv];
    for i in 0..s_q {
        let qi_block = i / block_size;
        for j in 0..s_kv {
            if j > qi_block {
                data[i * s_kv + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec::<_, f32>(data, vec![s_q, s_kv], device)
}

pub fn nsa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    config: &NsaConfig,
    gates: Option<&Tensor>,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("nsa: requires rank-4 [B,H,S,D]"));
    }
    let (b, h, s_q, _) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let s_kv = k.dims()[2];
    if config.block_size == 0 || s_kv % config.block_size != 0 {
        return Err(SynaptixError::Unsupported(
            "nsa: s_kv must divide by block_size",
        ));
    }

    let k_cmp = block_pool_mean(k, config.block_size)?;
    let v_cmp = block_pool_mean(v, config.block_size)?;
    let mask_cmp = block_causal_mask(s_q, k_cmp.dims()[2], config.block_size, q.device())?;
    let y_cmp = scaled_dot_attention(q, &k_cmp, &v_cmp, scale, Some(&mask_cmp))?;

    let y_win = sliding_window_attention(q, k, v, scale, config.window_size, None)?;

    if let Some(g) = gates {
        if g.dims() != [b, h, s_q, 2] {
            return Err(SynaptixError::Unsupported("nsa: gates must be [B,H,S,2]"));
        }
        let g_f = g.to_dtype(DType::F32)?;
        let g_cmp = g_f.narrow(3, 0, 1)?;
        let g_win = g_f.narrow(3, 1, 1)?;
        let y_cmp_f = y_cmp.to_dtype(DType::F32)?;
        let y_win_f = y_win.to_dtype(DType::F32)?;
        let mixed = g_cmp.broadcast_mul(&y_cmp_f)?.add(&g_win.broadcast_mul(&y_win_f)?)?;
        mixed.to_dtype(q.dtype())
    } else {
        let y_cmp_f = y_cmp.to_dtype(DType::F32)?;
        let y_win_f = y_win.to_dtype(DType::F32)?;
        let mixed = y_cmp_f.add(&y_win_f)?.affine(0.5, 0.0)?;
        mixed.to_dtype(q.dtype())
    }
}
