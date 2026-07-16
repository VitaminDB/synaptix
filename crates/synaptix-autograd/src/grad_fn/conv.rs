//! Backward для `conv1d` / `conv2d` через те же im2col-loop'ы, что используются в
//! `synaptix_ops::conv::{conv1d,conv2d}` для forward.
//!
//! Формулы (стандартный конв-backward):
//!  - `grad_weight[c_out, c_in, k] = Σ_{b,i} grad_out[b, c_out, i] · x_padded[b, c_in, i·s + k]`
//!  - `grad_input` = full-correlation grad_output с весом = `w[c_out, c_in, k]` (rot180 для
//!    full conv). На практике для произвольных stride/padding делаем явный scatter-add по ki.
//!  - `grad_bias[c_out] = Σ_{b,i} grad_out[b, c_out, i]`.

use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

/// Backward для `conv1d(input, weight, bias, stride, padding)`.
///
/// `grad_output: [B, C_out, out_len]`, `input: [B, C_in, L]`, `weight: [C_out, C_in, K]`.
pub fn conv1d_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    if input.rank() != 3 || weight.rank() != 3 || grad_output.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "conv1d_backward: input [B,C_in,L], weight [C_out,C_in,K], grad_output [B,C_out,L_out]",
        ));
    }
    let stride = stride.max(1);
    let (b, c_in, l) = (input.dims()[0], input.dims()[1], input.dims()[2]);
    let (c_out, c_in_w, k) = (weight.dims()[0], weight.dims()[1], weight.dims()[2]);
    if c_in_w != c_in {
        return Err(SynaptixError::shape_mismatch(input.dims(), weight.dims()));
    }
    let (bg, cg, out_len) = (grad_output.dims()[0], grad_output.dims()[1], grad_output.dims()[2]);
    if (bg, cg) != (b, c_out) {
        return Err(SynaptixError::shape_mismatch(grad_output.dims(), &[b, c_out, out_len]));
    }
    let l_pad = l + 2 * padding;
    let expect_out_len = l_pad.saturating_sub(k) / stride + 1;
    if expect_out_len != out_len {
        return Err(SynaptixError::Unsupported(
            "conv1d_backward: forward out_len mismatch with grad_output",
        ));
    }

    let dtype = input.dtype();
    let dev = input.device();

    // x_padded: [B, C_in, L+2P]
    let x_padded = if padding > 0 {
        let pad = Tensor::zeros(vec![b, c_in, padding], dtype, dev)?;
        Tensor::cat(&[&pad, input, &pad], 2)?
    } else {
        input.clone()
    };

    // grad_w[c_out, c_in, ki] = Σ_b Σ_i grad_out[b, co, i] · x_padded[b, ci, i·s + ki].
    // Эффективно через matmul на плоских (B·out_len) × (C_*).
    let go_perm = grad_output.permute(vec![0, 2, 1])?.contiguous()?
        .reshape(vec![b * out_len, c_out])?; // [B·out_len, C_out]
    let go_t = go_perm.transpose(0, 1)?.contiguous()?; // [C_out, B·out_len]

    let mut grad_w_slices: Vec<Tensor> = Vec::with_capacity(k);
    for ki in 0..k {
        // x_slice[ki]: [B, C_in, out_len] — позиции (ki, ki+s, ..., ki+(out_len-1)·s) в L_pad.
        let mut parts: Vec<Tensor> = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = ki + i * stride;
            parts.push(x_padded.narrow(2, pos, 1)?.contiguous()?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let x_slice = Tensor::cat(&refs, 2)?; // [B, C_in, out_len]
        let x_slice_flat = x_slice
            .permute(vec![0, 2, 1])?
            .contiguous()?
            .reshape(vec![b * out_len, c_in])?; // [B·out_len, C_in]

        // grad_w_ki[c_out, c_in] = go_t @ x_slice_flat
        let grad_w_ki = go_t.matmul(&x_slice_flat)?; // [C_out, C_in]
        grad_w_slices.push(grad_w_ki.unsqueeze(2)?); // [C_out, C_in, 1]
    }
    let refs: Vec<&Tensor> = grad_w_slices.iter().collect();
    let grad_weight = Tensor::cat(&refs, 2)?.to_dtype(dtype)?; // [C_out, C_in, K]

    // grad_x_padded: zeros + Σ_ki cattributions.
    // contrib_ki[b, c_in, i] = Σ_co grad_out[b, co, i] · w[co, c_in, ki]
    //                       = (grad_out_perm[B, out_len, C_out]) @ (w[:, :, ki])  → [B, out_len, C_in]
    // Затем эти контрибуции «вставляются» в x_padded по позициям ki + i·s.
    let mut grad_x_padded = Tensor::zeros(vec![b, c_in, l_pad], dtype, dev)?;
    let go_full = grad_output
        .permute(vec![0, 2, 1])?
        .contiguous()?; // [B, out_len, C_out]
    for ki in 0..k {
        let w_ki = weight.narrow(2, ki, 1)?.squeeze(2)?; // [C_out, C_in]
        let contrib = go_full.matmul(&w_ki)?; // [B, out_len, C_in]
        let contrib_perm = contrib
            .permute(vec![0, 2, 1])?
            .contiguous()?; // [B, C_in, out_len]
        // Расставить contrib в позиции (ki, ki+s, ...) в L_pad через построение
        // padded-tensor: left zeros [0..ki], interleaved with stride-1 zeros, right zeros.
        let scattered = scatter_1d(&contrib_perm, ki, stride, l_pad)?;
        grad_x_padded = grad_x_padded.add(&scattered)?;
    }

    // Снимаем padding: grad_input = grad_x_padded[:, :, padding..padding+L]
    let grad_input = if padding > 0 {
        grad_x_padded.narrow(2, padding, l)?.contiguous()?
    } else {
        grad_x_padded
    };

    // grad_bias[c_out] = grad_output.sum(dim=0,2)
    let grad_bias = grad_output
        .to_dtype(DType::F32)?
        .sum_keepdim(0)?
        .sum_keepdim(2)?
        .squeeze(2)?
        .squeeze(0)?
        .to_dtype(dtype)?;

    Ok((grad_input, grad_weight, Some(grad_bias)))
}

/// Backward для `conv2d(input, weight, bias, stride, padding, dilation)` с dilation=1.
/// Если dilation != (1,1) — Err Unsupported.
///
/// `grad_output: [B, C_out, out_h, out_w]`, `input: [B, C_in, H, W]`,
/// `weight: [C_out, C_in, KH, KW]`.
pub fn conv2d_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    if dilation.0 != 1 || dilation.1 != 1 {
        return Err(SynaptixError::Unsupported(
            "conv2d_backward: dilation != 1 not supported yet",
        ));
    }
    if input.rank() != 4 || weight.rank() != 4 || grad_output.rank() != 4 {
        return Err(SynaptixError::Unsupported(
            "conv2d_backward: 4D input/weight/grad_output required",
        ));
    }
    let (sh, sw) = (stride.0.max(1), stride.1.max(1));
    let (ph, pw) = padding;
    let (b, c_in, h, w_) = (input.dims()[0], input.dims()[1], input.dims()[2], input.dims()[3]);
    let (c_out, c_in_w, kh, kw) = (
        weight.dims()[0], weight.dims()[1], weight.dims()[2], weight.dims()[3],
    );
    if c_in_w != c_in {
        return Err(SynaptixError::shape_mismatch(input.dims(), weight.dims()));
    }
    let h_pad = h + 2 * ph;
    let w_pad = w_ + 2 * pw;
    let out_h = (h_pad.saturating_sub(kh)) / sh + 1;
    let out_w = (w_pad.saturating_sub(kw)) / sw + 1;
    if grad_output.dims() != [b, c_out, out_h, out_w] {
        return Err(SynaptixError::shape_mismatch(
            grad_output.dims(),
            &[b, c_out, out_h, out_w],
        ));
    }

    let dtype = input.dtype();
    let dev = input.device();

    // x_padded: [B, C_in, H+2P_h, W+2P_w]
    let x_padded = pad_h_w(input, ph, pw)?;

    // grad_out_flat[B·oh·ow, C_out]
    let go_perm = grad_output.permute(vec![0, 2, 3, 1])?.contiguous()?
        .reshape(vec![b * out_h * out_w, c_out])?;
    let go_t = go_perm.transpose(0, 1)?.contiguous()?; // [C_out, B·oh·ow]

    let mut grad_w_rows: Vec<Tensor> = Vec::with_capacity(kh * kw);
    for ki in 0..kh {
        for kj in 0..kw {
            // x_slice[ki,kj]: [B, C_in, out_h, out_w]
            let x_slice = collect_window(&x_padded, ki, kj, sh, sw, out_h, out_w)?;
            let x_slice_flat = x_slice
                .permute(vec![0, 2, 3, 1])?
                .contiguous()?
                .reshape(vec![b * out_h * out_w, c_in])?;
            let grad_w_kk = go_t.matmul(&x_slice_flat)?; // [C_out, C_in]
            // → [C_out, C_in, 1, 1]
            grad_w_rows.push(grad_w_kk.unsqueeze(2)?.unsqueeze(3)?);
        }
    }
    // Соберём в [C_out, C_in, KH, KW] через cat по dim 3 (KW), затем по dim 2 (KH).
    let mut by_row: Vec<Tensor> = Vec::with_capacity(kh);
    for ki in 0..kh {
        let slice = &grad_w_rows[ki * kw..ki * kw + kw];
        let refs: Vec<&Tensor> = slice.iter().collect();
        by_row.push(Tensor::cat(&refs, 3)?); // [C_out, C_in, 1, KW]
    }
    let refs: Vec<&Tensor> = by_row.iter().collect();
    let grad_weight = Tensor::cat(&refs, 2)?.to_dtype(dtype)?; // [C_out, C_in, KH, KW]

    // grad_x_padded: Σ_{ki,kj} scatter(grad_out @ w[:, :, ki, kj])
    let mut grad_x_padded = Tensor::zeros(vec![b, c_in, h_pad, w_pad], dtype, dev)?;
    let go_full = grad_output
        .permute(vec![0, 2, 3, 1])?
        .contiguous()?; // [B, out_h, out_w, C_out]
    let go_full_flat = go_full.reshape(vec![b * out_h * out_w, c_out])?;
    for ki in 0..kh {
        for kj in 0..kw {
            let w_kk = weight.narrow(2, ki, 1)?.narrow(3, kj, 1)?.squeeze(3)?.squeeze(2)?; // [C_out, C_in]
            let contrib_flat = go_full_flat.matmul(&w_kk)?; // [B·out_h·out_w, C_in]
            let contrib = contrib_flat
                .reshape(vec![b, out_h, out_w, c_in])?
                .permute(vec![0, 3, 1, 2])?
                .contiguous()?; // [B, C_in, out_h, out_w]
            let scattered = scatter_2d(&contrib, ki, kj, sh, sw, h_pad, w_pad)?;
            grad_x_padded = grad_x_padded.add(&scattered)?;
        }
    }
    // Снять padding.
    let grad_input = unpad_h_w(&grad_x_padded, ph, pw, h, w_)?;

    // grad_bias = grad_output.sum(B, out_h, out_w)
    let grad_bias = grad_output
        .to_dtype(DType::F32)?
        .sum_keepdim(0)?
        .sum_keepdim(2)?
        .sum_keepdim(3)?
        .squeeze(3)?
        .squeeze(2)?
        .squeeze(0)?
        .to_dtype(dtype)?;

    Ok((grad_input, grad_weight, Some(grad_bias)))
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// «Раскидать» `src: [B, C, out_len]` в `[B, C, target_len]` по позициям `off + i·stride`.
fn scatter_1d(src: &Tensor, off: usize, stride: usize, target_len: usize) -> Result<Tensor> {
    let (b, c, out_len) = (src.dims()[0], src.dims()[1], src.dims()[2]);
    let dtype = src.dtype();
    let dev = src.device();
    // Шаг 1: interleave src с (stride-1) нулей между элементами по последнему dim.
    let interleaved = if stride == 1 {
        src.contiguous()?
    } else {
        // src → [B, C, out_len, 1]; concat with zeros [B, C, out_len, stride-1] → [B, C, out_len, stride]
        //  → reshape [B, C, out_len*stride]
        let src_4 = src.unsqueeze(3)?; // [B, C, out_len, 1]
        let zeros = Tensor::zeros(vec![b, c, out_len, stride - 1], dtype, dev)?;
        let cat = Tensor::cat(&[&src_4, &zeros], 3)?; // [B, C, out_len, stride]
        cat.reshape(vec![b, c, out_len * stride])?
    };
    // Шаг 2: левый pad нулями [0..off), правый pad нулями [off + len_interleaved..target_len).
    let cur_len = interleaved.dims()[2];
    let need_right = target_len.saturating_sub(off + cur_len);
    if off + cur_len > target_len {
        // Это не должно случиться при корректных входах; усечём.
        return interleaved.narrow(2, 0, target_len - off);
    }
    let mut parts: Vec<Tensor> = Vec::new();
    if off > 0 {
        parts.push(Tensor::zeros(vec![b, c, off], dtype, dev)?);
    }
    parts.push(interleaved);
    if need_right > 0 {
        parts.push(Tensor::zeros(vec![b, c, need_right], dtype, dev)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

fn scatter_2d(
    src: &Tensor,
    off_h: usize,
    off_w: usize,
    sh: usize,
    sw: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor> {
    // src: [B, C, out_h, out_w]. Сначала scatter по dim 2 (H), потом по dim 3 (W).
    let (b, c, out_h, out_w) = (src.dims()[0], src.dims()[1], src.dims()[2], src.dims()[3]);
    let dtype = src.dtype();
    let dev = src.device();

    // Interleave по W (dim 3).
    let w_inter = if sw == 1 {
        src.contiguous()?
    } else {
        let src5 = src.unsqueeze(4)?; // [B, C, out_h, out_w, 1]
        let z = Tensor::zeros(vec![b, c, out_h, out_w, sw - 1], dtype, dev)?;
        Tensor::cat(&[&src5, &z], 4)?.reshape(vec![b, c, out_h, out_w * sw])?
    };
    // Pad по W.
    let w_padded = pad_dim(&w_inter, 3, off_w, target_w)?;

    // Interleave по H (dim 2).
    let (bb, cc, oh2, tw2) = (
        w_padded.dims()[0], w_padded.dims()[1], w_padded.dims()[2], w_padded.dims()[3],
    );
    let h_inter = if sh == 1 {
        w_padded
    } else {
        let s5 = w_padded.unsqueeze(3)?; // [B, C, out_h, 1, target_w]
        let z = Tensor::zeros(vec![bb, cc, oh2, sh - 1, tw2], dtype, dev)?;
        Tensor::cat(&[&s5, &z], 3)?.reshape(vec![bb, cc, oh2 * sh, tw2])?
    };
    // Pad по H.
    pad_dim(&h_inter, 2, off_h, target_h)
}

/// Раскинуть в `target_len` по dim `dim`: left zero-pad `off`, потом src, потом right zero-pad.
fn pad_dim(src: &Tensor, dim: usize, off: usize, target_len: usize) -> Result<Tensor> {
    let cur = src.dims()[dim];
    if off + cur > target_len {
        return src.narrow(dim, 0, target_len - off);
    }
    let need_right = target_len - off - cur;
    let mut parts: Vec<Tensor> = Vec::new();
    let dtype = src.dtype();
    let dev = src.device();
    if off > 0 {
        let mut left_dims: Vec<usize> = src.dims().to_vec();
        left_dims[dim] = off;
        parts.push(Tensor::zeros(left_dims, dtype, dev)?);
    }
    parts.push(src.contiguous()?);
    if need_right > 0 {
        let mut right_dims: Vec<usize> = src.dims().to_vec();
        right_dims[dim] = need_right;
        parts.push(Tensor::zeros(right_dims, dtype, dev)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dim)
}

/// Pad H и W нулями (analog F.pad для conv2d входа).
fn pad_h_w(x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
    let (b, c, _h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
    let dtype = x.dtype();
    let dev = x.device();
    let with_h = if ph > 0 {
        let p = Tensor::zeros(vec![b, c, ph, w], dtype, dev)?;
        Tensor::cat(&[&p, x, &p], 2)?
    } else {
        x.clone()
    };
    let h_total = with_h.dims()[2];
    if pw > 0 {
        let p = Tensor::zeros(vec![b, c, h_total, pw], dtype, dev)?;
        Tensor::cat(&[&p, &with_h, &p], 3)
    } else {
        Ok(with_h)
    }
}

fn unpad_h_w(x: &Tensor, ph: usize, pw: usize, h: usize, w: usize) -> Result<Tensor> {
    let stripped_h = if ph > 0 { x.narrow(2, ph, h)?.contiguous()? } else { x.clone() };
    if pw > 0 {
        stripped_h.narrow(3, pw, w)?.contiguous()
    } else {
        Ok(stripped_h)
    }
}

/// Собрать window `x_padded[:, :, ki + i·sh, kj + j·sw]` → `[B, C_in, out_h, out_w]`.
fn collect_window(
    x_padded: &Tensor,
    ki: usize,
    kj: usize,
    sh: usize,
    sw: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    // Build by iterating dim H, then dim W (как в conv2d forward).
    let mut row_parts: Vec<Tensor> = Vec::with_capacity(out_h);
    for i in 0..out_h {
        let pos_h = ki + i * sh;
        let row = x_padded.narrow(2, pos_h, 1)?.contiguous()?; // [B, C_in, 1, W_pad]
        let mut col_parts: Vec<Tensor> = Vec::with_capacity(out_w);
        for j in 0..out_w {
            let pos_w = kj + j * sw;
            col_parts.push(row.narrow(3, pos_w, 1)?.contiguous()?);
        }
        let refs: Vec<&Tensor> = col_parts.iter().collect();
        row_parts.push(Tensor::cat(&refs, 3)?); // [B, C_in, 1, out_w]
    }
    let refs: Vec<&Tensor> = row_parts.iter().collect();
    Tensor::cat(&refs, 2) // [B, C_in, out_h, out_w]
}
