use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Causal 3D свёртка (стиль LTX VAE): causal-паддинг по времени (слева `kt−1`
/// нулей), VALID по пространству. `x:[B,C_in,T,H,W]`, `weight:[C_out,C_in,kt,kh,kw]`,
/// `bias:[C_out]`. `stride` применяется ко всем трём осям равномерно.
///   `T_out = (T−1)/stride + 1`, `H_out = (H−kh)/stride + 1`, `W_out = (W−kw)/stride + 1`.
/// Эквивалент `conv3d(pad(x, T_left=kt−1), stride)` с VALID-паддингом.
pub fn causal_conv3d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    if x.rank() != 5 || weight.rank() != 5 {
        return Err(SynaptixError::Unsupported(
            "causal_conv3d: x [B,C_in,T,H,W], weight [C_out,C_in,kt,kh,kw]",
        ));
    }
    let (b, c_in, t, h, w) = (
        x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3], x.dims()[4],
    );
    let (c_out, c_in_w, kt, kh, kw) = (
        weight.dims()[0], weight.dims()[1], weight.dims()[2], weight.dims()[3], weight.dims()[4],
    );
    if c_in_w != c_in {
        return Err(SynaptixError::shape_mismatch(x.dims(), weight.dims()));
    }
    let stride = stride.max(1);
    // паддированное по времени T' = T + (kt-1); пространство без паддинга
    let t_pad = t + kt - 1;
    if h < kh || w < kw {
        return Err(SynaptixError::Unsupported("causal_conv3d: H<kh или W<kw"));
    }
    let t_out = (t_pad - kt) / stride + 1; // = (t-1)/stride + 1
    let h_out = (h - kh) / stride + 1;
    let w_out = (w - kw) / stride + 1;

    if let Some(bt) = bias {
        if bt.dims() != [c_out] {
            return Err(SynaptixError::Unsupported("causal_conv3d: bias must be [C_out]"));
        }
    }

    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let wf = f32v(weight)?;
    let bf = match bias {
        Some(bt) => Some(f32v(bt)?),
        None => None,
    };

    // индексация x: (((bi*C_in + ci)*T + ti)*H + hi)*W + wi
    let x_idx = |bi: usize, ci: usize, ti: usize, hi: usize, wi: usize| {
        (((bi * c_in + ci) * t + ti) * h + hi) * w + wi
    };
    // индексация weight: (((co*C_in + ci)*kt + dt)*kh + dh)*kw + dw
    let w_idx = |co: usize, ci: usize, dt: usize, dh: usize, dw: usize| {
        (((co * c_in + ci) * kt + dt) * kh + dh) * kw + dw
    };

    let mut out = vec![0.0f32; b * c_out * t_out * h_out * w_out];
    for bi in 0..b {
        for co in 0..c_out {
            for to in 0..t_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc = bf.as_ref().map_or(0.0f32, |bv| bv[co]);
                        for ci in 0..c_in {
                            for dt in 0..kt {
                                // позиция во времени в паддированном тензоре
                                let tp = to * stride + dt;
                                // вычитаем левый causal-pad (kt-1); вне реального диапазона → 0
                                if tp < kt - 1 {
                                    continue;
                                }
                                let ti = tp - (kt - 1);
                                if ti >= t {
                                    continue;
                                }
                                for dh in 0..kh {
                                    let hi = ho * stride + dh;
                                    for dw in 0..kw {
                                        let wi = wo * stride + dw;
                                        acc += xf[x_idx(bi, ci, ti, hi, wi)]
                                            * wf[w_idx(co, ci, dt, dh, dw)];
                                    }
                                }
                            }
                        }
                        let o = ((((bi * c_out + co) * t_out + to) * h_out + ho) * w_out) + wo;
                        out[o] = acc;
                    }
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, c_out, t_out, h_out, w_out], x.device())?
        .to_dtype(dtype_in)
}
