use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, softmax_inplace, to_f32_vec};

/// Synthesizer (Random): синтетические attention-логиты вместо `QKᵀ`.
/// `synth` формы `[H, S, S]` (broadcast по batch). `O = softmax(synth(+mask)) @ V`;
/// при `causal` верхний треугольник маскируется. q,k не участвуют (random-форма),
/// но остаются в сигнатуре для единообразия интерфейса.
pub fn synthesizer_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    synth: &Tensor,
    causal: bool,
) -> Result<Tensor> {
    let (b, h, s, _dk, dv) = linear_dims(q, k, v)?;
    let _ = (q, k); // не используются by design
    if synth.dims() != [h, s, s] {
        return Err(SynaptixError::Unsupported("synthesizer: synth must be [H, S, S]"));
    }
    let dtype_in = v.dtype();
    let vf = to_f32_vec(v)?;
    let sf = to_f32_vec(synth)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    let mut row = vec![0.0f32; s];
    for bi in 0..b {
        for hi in 0..h {
            for i in 0..s {
                let syn_off = (hi * s + i) * s;
                for j in 0..s {
                    row[j] = if causal && j > i {
                        f32::NEG_INFINITY
                    } else {
                        sf[syn_off + j]
                    };
                }
                softmax_inplace(&mut row);
                let o_off = ((bi * h + hi) * s + i) * dv;
                for j in 0..s {
                    let p = row[j];
                    if p == 0.0 {
                        continue;
                    }
                    let v_off = ((bi * h + hi) * s + j) * dv;
                    for c in 0..dv {
                        out[o_off + c] += p * vf[v_off + c];
                    }
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], v.device())?.to_dtype(dtype_in)
}
