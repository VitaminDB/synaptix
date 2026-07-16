use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::conv::{Im2ColParams2d, im2col_2d_dispatch};

pub fn patch_embed_2d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    patch_size: usize,
    stride: Option<usize>,
) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(SynaptixError::Unsupported("patch_embed_2d: x must be (B, C, H, W)"));
    }
    if weight.rank() != 4 {
        return Err(SynaptixError::Unsupported("patch_embed_2d: weight must be (D, C, P, P)"));
    }
    let dims_x = x.dims().to_vec();
    let dims_w = weight.dims().to_vec();
    let (batch, channels, in_h, in_w) = (dims_x[0], dims_x[1], dims_x[2], dims_x[3]);
    let out_features = dims_w[0];
    if dims_w[1] != channels {
        return Err(SynaptixError::shape_mismatch(&dims_x, &dims_w));
    }
    if dims_w[2] != patch_size || dims_w[3] != patch_size {
        return Err(SynaptixError::Other(format!(
            "patch_embed_2d: weight patch ({}, {}) mismatch patch_size {patch_size}",
            dims_w[2], dims_w[3]
        )));
    }
    let stride = stride.unwrap_or(patch_size);
    let p = Im2ColParams2d {
        batch,
        channels,
        in_h,
        in_w,
        kernel_h: patch_size,
        kernel_w: patch_size,
        stride_h: stride,
        stride_w: stride,
        pad_h: 0,
        pad_w: 0,
        dilation_h: 1,
        dilation_w: 1,
    };
    let oh = p.out_h();
    let ow = p.out_w();
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;
    let elem = 4usize;
    let src_bytes = match x_f32.storage() {
        synaptix_core::tensor::storage::Storage::Cpu(b) => b.as_bytes().to_vec(),
        _ => return Err(SynaptixError::Unsupported("patch_embed_2d: CPU storage only")),
    };
    let patch_elems = channels * patch_size * patch_size;
    let patches = batch * oh * ow;
    let mut dst = vec![0u8; patches * patch_elems * elem];
    im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p)?;
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
    let reshaped = out.reshape((batch, oh, ow, out_features))?;
    reshaped.to_dtype(dtype_in)
}
