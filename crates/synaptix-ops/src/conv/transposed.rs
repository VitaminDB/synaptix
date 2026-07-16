use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Транспонированная (upsampling) свёртка 1D — как `torch.nn.ConvTranspose1d`.
/// `input:[B,C_in,L]`, `weight:[C_in,C_out,K]`, `bias:[C_out]`.
/// `L_out = (L−1)·stride + K − 2·padding`. Каждая входная позиция «разбрасывает»
/// вклад `input[b,c_in,i]·weight[c_in,c_out,k]` в `out[b,c_out, i·stride + k − padding]`.
pub fn transposed_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    if input.rank() != 3 || weight.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "transposed_conv: input [B,C_in,L], weight [C_in,C_out,K]",
        ));
    }
    let (b, c_in, l) = (input.dims()[0], input.dims()[1], input.dims()[2]);
    let (c_in_w, c_out, k) = (weight.dims()[0], weight.dims()[1], weight.dims()[2]);
    if c_in_w != c_in {
        return Err(SynaptixError::shape_mismatch(input.dims(), weight.dims()));
    }
    let stride = stride.max(1);
    let l_out_signed = (l as isize - 1) * stride as isize + k as isize - 2 * padding as isize;
    if l_out_signed <= 0 {
        return Err(SynaptixError::Unsupported("transposed_conv: вычисленная длина <= 0"));
    }
    let l_out = l_out_signed as usize;

    if let Some(bt) = bias {
        if bt.dims() != [c_out] {
            return Err(SynaptixError::Unsupported("transposed_conv: bias must be [C_out]"));
        }
    }

    let dtype_in = input.dtype();
    let xf = f32v(input)?;
    let wf = f32v(weight)?;
    let bf = match bias {
        Some(bt) => Some(f32v(bt)?),
        None => None,
    };

    let mut out = vec![0.0f32; b * c_out * l_out];
    for bi in 0..b {
        for i in 0..l {
            for ki in 0..k {
                let p = (i * stride + ki) as isize - padding as isize;
                if p < 0 || p as usize >= l_out {
                    continue;
                }
                let pos = p as usize;
                for co in 0..c_out {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        acc += xf[(bi * c_in + ci) * l + i] * wf[(ci * c_out + co) * k + ki];
                    }
                    out[(bi * c_out + co) * l_out + pos] += acc;
                }
            }
        }
    }
    if let Some(bvec) = &bf {
        for bi in 0..b {
            for co in 0..c_out {
                let base = (bi * c_out + co) * l_out;
                for pos in 0..l_out {
                    out[base + pos] += bvec[co];
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, c_out, l_out], input.device())?.to_dtype(dtype_in)
}
