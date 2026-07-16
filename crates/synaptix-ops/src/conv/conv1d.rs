use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    conv1d_dilated(input, weight, bias, stride, padding, 1)
}

pub fn conv1d_dilated(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Tensor> {
    if input.rank() != 3 || weight.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "conv1d: input [B,C_in,L], weight [C_out,C_in,K]",
        ));
    }
    let (b, _c_in, l) = (input.dims()[0], input.dims()[1], input.dims()[2]);
    let (_c_out, _c_in_w, k) = (weight.dims()[0], weight.dims()[1], weight.dims()[2]);
    let stride = stride.max(1);
    let dilation = dilation.max(1);
    let l_padded = l + 2 * padding;
    let span_k = dilation * (k - 1) + 1;
    let out_len = (l_padded.saturating_sub(span_k)) / stride + 1;

    // Fast path (dilation==1): a single fused 2D conv (cutlass im2col+GEMM, one
    // im2col launch) by treating [B,Cin,L] as [B,Cin,1,L]. Avoids the per-tap
    // im2col `.contiguous()` copy swarm that dominates VAE decode. Falls back to
    // the loop below where the fused path is unavailable (CPU / NonContiguous).
    if dilation == 1 && input.is_contiguous() && weight.is_contiguous() {
        let x4 = input.reshape(vec![b, _c_in, 1, l])?;
        let w4 = weight.reshape(vec![_c_out, _c_in_w, 1, k])?;
        match x4.conv2d(&w4, bias, (1, stride), (0, padding)) {
            Ok(o) => {
                let d = o.dims().to_vec();
                return o.reshape(vec![d[0], d[1], d[3]]);
            }
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }

    let x = if padding > 0 {
        let pad = Tensor::zeros(vec![b, _c_in, padding], input.dtype(), input.device())?;
        Tensor::cat(&[&pad, input, &pad], 2)?
    } else {
        input.clone()
    };

    if stride == 1 {
        let x_lc = x.permute(vec![0, 2, 1])?.contiguous()?;
        let mut cols: Vec<Tensor> = Vec::with_capacity(k);
        for ki in 0..k {
            cols.push(x_lc.narrow(1, ki * dilation, out_len)?.contiguous()?);
        }
        let refs: Vec<&Tensor> = cols.iter().collect();
        let im2col = Tensor::cat(&refs, 2)?;
        let w2 = weight
            .permute(vec![0, 2, 1])?
            .contiguous()?
            .reshape(vec![_c_out, k * _c_in])?
            .transpose(0, 1)?
            .contiguous()?;
        let out_lc = im2col.matmul(&w2)?;
        let mut out = out_lc.permute(vec![0, 2, 1])?.contiguous()?;
        if let Some(b_t) = bias {
            out = out.broadcast_add(&b_t.unsqueeze(0)?.unsqueeze(2)?)?;
        }
        return Ok(out);
    }

    let span = stride * out_len;
    let reshape_fits = span <= l_padded;

    let mut out = Tensor::zeros(vec![b, _c_out, out_len], input.dtype(), input.device())?;
    for ki in 0..k {
        let off = ki * dilation;
        let w_ki = weight.narrow(2, ki, 1)?.squeeze(2)?;
        let w_ki_t = w_ki.transpose(0, 1)?.contiguous()?;

        let x_slice = if stride == 1 {
            x.narrow(2, off, out_len)?
        } else if reshape_fits && off + span <= l_padded {
            x.narrow(2, off, span)?
                .contiguous()?
                .reshape(vec![b, _c_in, out_len, stride])?
                .narrow(3, 0, 1)?
                .squeeze(3)?
        } else {
            let idx: Vec<u32> = (0..out_len).map(|i| (i * stride) as u32).collect();
            let idx = Tensor::from_vec(idx, (out_len,), input.device())?;
            x.narrow(2, off, (out_len - 1) * stride + 1)?.index_select(2, &idx)?
        };

        let x_t = x_slice.permute(vec![0, 2, 1])?.contiguous()?;
        let proj = x_t.matmul(&w_ki_t)?;
        let proj_t = proj.permute(vec![0, 2, 1])?.contiguous()?;
        out = out.add(&proj_t)?;
    }

    if let Some(b_t) = bias {
        let b_shaped = b_t.unsqueeze(0)?.unsqueeze(2)?;
        out = out.broadcast_add(&b_shaped)?;
    }
    Ok(out)
}
