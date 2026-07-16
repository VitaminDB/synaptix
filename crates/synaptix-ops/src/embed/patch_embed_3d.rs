use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::conv::{Im2ColParams3d, im2col_3d_dispatch};

pub fn patch_embed_3d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    patch_t: usize,
    patch_h: usize,
    patch_w: usize,
    stride_t: Option<usize>,
    stride_h: Option<usize>,
    stride_w: Option<usize>,
) -> Result<Tensor> {
    if x.rank() != 5 {
        return Err(SynaptixError::Unsupported("patch_embed_3d: x must be (B, C, T, H, W)"));
    }
    if weight.rank() != 5 {
        return Err(SynaptixError::Unsupported(
            "patch_embed_3d: weight must be (D, C, Pt, Ph, Pw)",
        ));
    }
    let dims_x = x.dims().to_vec();
    let (batch, channels, in_t, in_h, in_w) =
        (dims_x[0], dims_x[1], dims_x[2], dims_x[3], dims_x[4]);
    let dims_w = weight.dims().to_vec();
    let out_features = dims_w[0];
    if dims_w[1] != channels {
        return Err(SynaptixError::shape_mismatch(&dims_x, &dims_w));
    }
    let st = stride_t.unwrap_or(patch_t);
    let sh = stride_h.unwrap_or(patch_h);
    let sw = stride_w.unwrap_or(patch_w);
    let p = Im2ColParams3d {
        batch,
        channels,
        in_t,
        in_h,
        in_w,
        kernel_t: patch_t,
        kernel_h: patch_h,
        kernel_w: patch_w,
        stride_t: st,
        stride_h: sh,
        stride_w: sw,
    };
    let ot = p.out_t();
    let oh = p.out_h();
    let ow = p.out_w();
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;
    let src_bytes = match x_f32.storage() {
        synaptix_core::tensor::storage::Storage::Cpu(b) => b.as_bytes().to_vec(),
        _ => return Err(SynaptixError::Unsupported("patch_embed_3d: CPU storage only")),
    };
    let elem = 4usize;
    let patch_elems = channels * patch_t * patch_h * patch_w;
    let patches = batch * ot * oh * ow;
    let mut dst = vec![0u8; patches * patch_elems * elem];
    im2col_3d_dispatch(DType::F32, &src_bytes, &mut dst, p)?;
    let patch_tensor = Tensor::from_raw_bytes(dst, (patches, patch_elems), DType::F32, x.device())?;
    let w_f32 = weight.to_dtype(DType::F32)?.reshape((out_features, patch_elems))?;
    let w_t = w_f32.transpose(0, 1)?.contiguous()?;
    let out = patch_tensor.matmul(&w_t)?;
    let out = match bias {
        Some(b) => {
            let b_f32 = b.to_dtype(DType::F32)?.reshape((1usize, out_features))?;
            out.broadcast_add(&b_f32)?
        }
        None => out,
    };
    let reshaped = out.reshape((batch, ot, oh, ow, out_features))?;
    reshaped.to_dtype(dtype_in)
}
