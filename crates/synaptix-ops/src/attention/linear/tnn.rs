use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, to_f32_vec};

/// TNN (Toeplitz Neural Network) — это не q,k,v-attention, а смешение V по
/// последовательности нижне-треугольной матрицей Тёплица из относительных
/// позиционных коэффициентов. `rel_kernel` формы `[H, S]` задаёт `kernel[h, lag]`:
///   `O[h,i,:] = Σ_{j≤i} kernel[h, i−j] · V[h,j,:]` (causal).
/// q и k не участвуют в вычислении (семантика варианта), но остаются в сигнатуре
/// для единообразия интерфейса семейства.
pub fn tnn_attention(q: &Tensor, k: &Tensor, v: &Tensor, rel_kernel: &Tensor) -> Result<Tensor> {
    let (b, h, s, _dk, dv) = linear_dims(q, k, v)?;
    let _ = (q, k); // не используются by design
    if rel_kernel.dims() != [h, s] {
        return Err(SynaptixError::Unsupported("tnn: rel_kernel must be [H, S]"));
    }
    let dtype_in = v.dtype();
    let vf = to_f32_vec(v)?;
    let rf = to_f32_vec(rel_kernel)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    for bi in 0..b {
        for hi in 0..h {
            let kern = &rf[hi * s..hi * s + s];
            for i in 0..s {
                let o_off = ((bi * h + hi) * s + i) * dv;
                for j in 0..=i {
                    let coef = kern[i - j];
                    let v_off = ((bi * h + hi) * s + j) * dv;
                    for c in 0..dv {
                        out[o_off + c] += coef * vf[v_off + c];
                    }
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], v.device())?.to_dtype(dtype_in)
}
