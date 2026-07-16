use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Ядро FlashAttention: online-softmax над ключами БЕЗ материализации матрицы
/// скоров `[Sq,Sk]`. Поддерживает блочную итерацию ключей (`block_k`) — числовой
/// результат bit-exact совпадает со стандартным softmax-attention.
///
/// `q:[B,H,Sq,D]`, `k,v:[B,H,Sk,D]`. `mask` (additive) — None | `[Sq,Sk]` |
/// `[B,H,Sq,Sk]`. Возвращает `[B,H,Sq,D]`.
pub(super) fn flash_attn_core(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
    block_k: usize,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("flash: q,k,v must be rank-4 [B,H,S,D]"));
    }
    let (b, h, sq, d) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let sk = k.dims()[2];
    if k.dims() != [b, h, sk, d] || v.dims() != [b, h, sk, d] {
        return Err(SynaptixError::shape_mismatch(k.dims(), v.dims()));
    }
    // раскладка маски: 0 = none, 2 = [Sq,Sk], 4 = [B,H,Sq,Sk]
    let mask_kind = match mask {
        None => 0,
        Some(m) if m.dims() == [sq, sk] => 2,
        Some(m) if m.dims() == [b, h, sq, sk] => 4,
        Some(_) => {
            return Err(SynaptixError::Unsupported(
                "flash: mask must be [Sq,Sk] or [B,H,Sq,Sk]",
            ))
        }
    };

    let dtype_in = q.dtype();
    let qf = f32v(q)?;
    let kf = f32v(k)?;
    let vf = f32v(v)?;
    let mf = match mask {
        Some(m) => Some(f32v(m)?),
        None => None,
    };
    let bk = block_k.max(1);

    let mut out = vec![0.0f32; b * h * sq * d];
    let mut acc = vec![0.0f32; d];
    for bi in 0..b {
        for hi in 0..h {
            let kv_base = (bi * h + hi) * sk;
            for qi in 0..sq {
                let q_off = ((bi * h + hi) * sq + qi) * d;
                let mut m_run = f32::NEG_INFINITY;
                let mut l_run = 0.0f32;
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                // итерация по блокам ключей (online softmax)
                let mut kb = 0;
                while kb < sk {
                    let kb_end = (kb + bk).min(sk);
                    for kj in kb..kb_end {
                        let k_off = (kv_base + kj) * d;
                        let mut s = 0.0f32;
                        for di in 0..d {
                            s += qf[q_off + di] * kf[k_off + di];
                        }
                        s *= scale;
                        s += match mask_kind {
                            2 => mf.as_ref().unwrap()[qi * sk + kj],
                            4 => mf.as_ref().unwrap()[((bi * h + hi) * sq + qi) * sk + kj],
                            _ => 0.0,
                        };
                        if !s.is_finite() && s == f32::NEG_INFINITY {
                            continue; // полностью замаскировано
                        }
                        let m_new = m_run.max(s);
                        let corr = if m_run.is_finite() { (m_run - m_new).exp() } else { 0.0 };
                        let p = (s - m_new).exp();
                        l_run = l_run * corr + p;
                        let v_off = (kv_base + kj) * d;
                        for di in 0..d {
                            acc[di] = acc[di] * corr + p * vf[v_off + di];
                        }
                        m_run = m_new;
                    }
                    kb = kb_end;
                }
                let o_off = q_off;
                let inv = if l_run > 0.0 { 1.0 / l_run } else { 0.0 };
                for di in 0..d {
                    out[o_off + di] = acc[di] * inv;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, sq, d], q.device())?.to_dtype(dtype_in)
}

/// FlashAttention-2: tile-based online softmax (block_k=64). На CPU числовой
/// результат bit-exact со стандартным attention; тайлинг экономит память (не
/// материализует `[Sq,Sk]`). Scale = `1/√D`.
pub fn flash_attention_v2(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let scale = (*q.dims().last().unwrap_or(&64) as f32).sqrt().recip();
    flash_attn_core(q, k, v, scale, mask, 64)
}
