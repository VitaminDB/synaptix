use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
) -> Result<Tensor> {
    if input.rank() != 5 || weight.rank() != 5 {
        return Err(SynaptixError::Unsupported(
            "conv3d: input [B,C_in,D,H,W], weight [C_out,C_in,KD,KH,KW]",
        ));
    }
    let (b, c_in, dz, h, w) = (
        input.dims()[0], input.dims()[1], input.dims()[2], input.dims()[3], input.dims()[4],
    );
    let (c_out, c_in_w, kd, kh, kw) = (
        weight.dims()[0],
        weight.dims()[1],
        weight.dims()[2],
        weight.dims()[3],
        weight.dims()[4],
    );
    if c_in != c_in_w {
        return Err(SynaptixError::shape_mismatch(input.dims(), weight.dims()));
    }
    let (sd, sh, sw) = (stride.0.max(1), stride.1.max(1), stride.2.max(1));
    let (pd, ph, pw) = padding;
    let (dd, dh, dw) = (dilation.0.max(1), dilation.1.max(1), dilation.2.max(1));

    let kd_eff = (kd - 1) * dd + 1;
    let kh_eff = (kh - 1) * dh + 1;
    let kw_eff = (kw - 1) * dw + 1;
    let dz_padded = dz + 2 * pd;
    let h_padded = h + 2 * ph;
    let w_padded = w + 2 * pw;
    if dz_padded < kd_eff || h_padded < kh_eff || w_padded < kw_eff {
        return Err(SynaptixError::Unsupported(
            "conv3d: input too small for kernel+padding",
        ));
    }
    let out_d = (dz_padded - kd_eff) / sd + 1;
    let out_h = (h_padded - kh_eff) / sh + 1;
    let out_w = (w_padded - kw_eff) / sw + 1;

    // Быстрый путь: direct conv3d-ядро через backend (dilation=1; CUDA F32/F16/BF16).
    // При Unsupported (CPU / dilation>1 не дошли сюда) / NonContiguous — decomposed ниже.
    if dd == 1 && dh == 1 && dw == 1 {
        match input.conv3d(weight, bias, (sd, sh, sw), (pd, ph, pw)) {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }

    let x = {
        let mut t = input.clone();
        if pd > 0 {
            let z = Tensor::zeros(vec![b, c_in, pd, h, w], input.dtype(), input.device())?;
            t = Tensor::cat(&[&z, &t, &z], 2)?;
        }
        if ph > 0 {
            let cur = t.dims().to_vec();
            let z = Tensor::zeros(vec![cur[0], cur[1], cur[2], ph, cur[4]], input.dtype(), input.device())?;
            t = Tensor::cat(&[&z, &t, &z], 3)?;
        }
        if pw > 0 {
            let cur = t.dims().to_vec();
            let z = Tensor::zeros(vec![cur[0], cur[1], cur[2], cur[3], pw], input.dtype(), input.device())?;
            t = Tensor::cat(&[&z, &t, &z], 4)?;
        }
        t
    };

    let mut out = Tensor::zeros(
        vec![b, c_out, out_d, out_h, out_w],
        input.dtype(),
        input.device(),
    )?;

    for ki in 0..kd {
        for kj in 0..kh {
            for kl in 0..kw {
                let w_kk = weight
                    .narrow(2, ki, 1)?
                    .narrow(3, kj, 1)?
                    .narrow(4, kl, 1)?
                    .squeeze(4)?
                    .squeeze(3)?
                    .squeeze(2)?;
                let w_t = w_kk.transpose(0, 1)?.contiguous()?;

                let mut d_parts: Vec<Tensor> = Vec::with_capacity(out_d);
                for di in 0..out_d {
                    let pos_d = ki * dd + di * sd;
                    let plane = x.narrow(2, pos_d, 1)?.contiguous()?;
                    let mut h_parts: Vec<Tensor> = Vec::with_capacity(out_h);
                    for hi in 0..out_h {
                        let pos_h = kj * dh + hi * sh;
                        let row = plane.narrow(3, pos_h, 1)?.contiguous()?;
                        let mut w_parts: Vec<Tensor> = Vec::with_capacity(out_w);
                        for wi in 0..out_w {
                            let pos_w = kl * dw + wi * sw;
                            w_parts.push(row.narrow(4, pos_w, 1)?.contiguous()?);
                        }
                        let refs: Vec<&Tensor> = w_parts.iter().collect();
                        h_parts.push(Tensor::cat(&refs, 4)?);
                    }
                    let refs: Vec<&Tensor> = h_parts.iter().collect();
                    d_parts.push(Tensor::cat(&refs, 3)?);
                }
                let refs: Vec<&Tensor> = d_parts.iter().collect();
                let x_slice = Tensor::cat(&refs, 2)?;

                let x_perm = x_slice
                    .permute(vec![0, 2, 3, 4, 1])?
                    .contiguous()?
                    .reshape(vec![b * out_d * out_h * out_w, c_in])?;
                let proj = x_perm.matmul(&w_t)?;
                let proj_b = proj
                    .reshape(vec![b, out_d, out_h, out_w, c_out])?
                    .permute(vec![0, 4, 1, 2, 3])?
                    .contiguous()?;
                out = out.add(&proj_b)?;
            }
        }
    }

    if let Some(b_t) = bias {
        let b_shaped = b_t
            .unsqueeze(0)?
            .unsqueeze(2)?
            .unsqueeze(3)?
            .unsqueeze(4)?;
        out = out.broadcast_add(&b_shaped)?;
    }
    Ok(out)
}
