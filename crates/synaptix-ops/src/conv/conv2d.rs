use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err(SynaptixError::Unsupported(
            "conv2d: input [B,C_in,H,W], weight [C_out,C_in,KH,KW]",
        ));
    }
    let (b, c_in, h, w) = (input.dims()[0], input.dims()[1], input.dims()[2], input.dims()[3]);
    let (c_out, c_in_w, kh, kw) = (
        weight.dims()[0],
        weight.dims()[1],
        weight.dims()[2],
        weight.dims()[3],
    );
    if c_in != c_in_w {
        return Err(SynaptixError::shape_mismatch(input.dims(), weight.dims()));
    }
    let (sh, sw) = (stride.0.max(1), stride.1.max(1));
    let (ph, pw) = padding;
    let (dh, dw) = (dilation.0.max(1), dilation.1.max(1));

    let h_padded = h + 2 * ph;
    let w_padded = w + 2 * pw;
    let kh_eff = (kh - 1) * dh + 1;
    let kw_eff = (kw - 1) * dw + 1;
    if h_padded < kh_eff || w_padded < kw_eff {
        return Err(SynaptixError::Unsupported(
            "conv2d: input too small for kernel+padding",
        ));
    }
    let out_h = (h_padded - kh_eff) / sh + 1;
    let out_w = (w_padded - kw_eff) / sw + 1;

    // Fast-path: direct conv через backend (CUDA — один launch). Только dilation=1.
    // На CPU / неподдержке backend падаем в generic im2col-через-cat ниже.
    if dh == 1 && dw == 1 {
        match input.conv2d(weight, bias, (sh, sw), (ph, pw)) {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }

    let x = if ph > 0 || pw > 0 {
        let pad_h = Tensor::zeros(vec![b, c_in, ph, w], input.dtype(), input.device())?;
        let with_h = Tensor::cat(&[&pad_h, input, &pad_h], 2)?;
        let pad_w = Tensor::zeros(vec![b, c_in, h + 2 * ph, pw], input.dtype(), input.device())?;
        Tensor::cat(&[&pad_w, &with_h, &pad_w], 3)?
    } else {
        input.clone()
    };

    let mut out = Tensor::zeros(vec![b, c_out, out_h, out_w], input.dtype(), input.device())?;
    for ki in 0..kh {
        for kj in 0..kw {
            let w_kk = weight.narrow(2, ki, 1)?.narrow(3, kj, 1)?.squeeze(3)?.squeeze(2)?;
            let w_t = w_kk.transpose(0, 1)?.contiguous()?;

            let mut row_parts: Vec<Tensor> = Vec::with_capacity(out_h);
            for i in 0..out_h {
                let pos_h = ki * dh + i * sh;
                let row = x.narrow(2, pos_h, 1)?.contiguous()?;
                let mut col_parts: Vec<Tensor> = Vec::with_capacity(out_w);
                for j in 0..out_w {
                    let pos_w = kj * dw + j * sw;
                    col_parts.push(row.narrow(3, pos_w, 1)?.contiguous()?);
                }
                let refs: Vec<&Tensor> = col_parts.iter().collect();
                row_parts.push(Tensor::cat(&refs, 3)?);
            }
            let refs: Vec<&Tensor> = row_parts.iter().collect();
            let x_slice = Tensor::cat(&refs, 2)?;

            let x_perm = x_slice
                .permute(vec![0, 2, 3, 1])?
                .contiguous()?
                .reshape(vec![b * out_h * out_w, c_in])?;
            let proj = x_perm.matmul(&w_t)?;
            let proj_b = proj
                .reshape(vec![b, out_h, out_w, c_out])?
                .permute(vec![0, 3, 1, 2])?
                .contiguous()?;
            out = out.add(&proj_b)?;
        }
    }

    if let Some(b_t) = bias {
        let b_shaped = b_t.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        out = out.broadcast_add(&b_shaped)?;
    }
    Ok(out)
}
