use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

#[derive(Debug, Clone, Copy)]
pub struct Im2ColParams2d {
    pub batch: usize,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

impl Im2ColParams2d {
    pub fn out_h(&self) -> usize {
        let eff = self.dilation_h * (self.kernel_h - 1) + 1;
        (self.in_h + 2 * self.pad_h - eff) / self.stride_h + 1
    }
    pub fn out_w(&self) -> usize {
        let eff = self.dilation_w * (self.kernel_w - 1) + 1;
        (self.in_w + 2 * self.pad_w - eff) / self.stride_w + 1
    }
    pub fn check(&self) -> Result<()> {
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(SynaptixError::Unsupported("im2col: zero kernel"));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(SynaptixError::Unsupported("im2col: zero stride"));
        }
        Ok(())
    }
}

pub fn im2col_2d_dispatch(
    dtype: DType,
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams2d,
) -> Result<(usize, usize)> {
    p.check()?;
    let oh = p.out_h();
    let ow = p.out_w();
    let patches_per_image = oh * ow;
    let patch_size = p.channels * p.kernel_h * p.kernel_w;
    let needed = p.batch * patches_per_image * patch_size * elem_size(dtype)?;
    if dst.len() < needed {
        return Err(SynaptixError::Other(format!(
            "im2col: dst too small, expected {} bytes, got {}",
            needed,
            dst.len()
        )));
    }
    match dtype {
        DType::F32 => im2col_2d_typed::<f32>(src, dst, p, oh, ow),
        DType::F64 => im2col_2d_typed::<f64>(src, dst, p, oh, ow),
        DType::F16 => im2col_2d_typed::<half::f16>(src, dst, p, oh, ow),
        DType::BF16 => im2col_2d_typed::<half::bf16>(src, dst, p, oh, ow),
        _ => Err(SynaptixError::Unsupported("im2col: dtype")),
    }?;
    Ok((p.batch * patches_per_image, patch_size))
}

fn im2col_2d_typed<T: bytemuck::Pod + bytemuck::Zeroable + Copy>(
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams2d,
    oh: usize,
    ow: usize,
) -> Result<()> {
    let src_t: &[T] = bytemuck::cast_slice(src);
    let dst_t: &mut [T] = bytemuck::cast_slice_mut(dst);
    let patch_size = p.channels * p.kernel_h * p.kernel_w;
    let patches_per_image = oh * ow;
    let zero = T::zeroed();
    for n in 0..p.batch {
        for ph in 0..oh {
            for pw in 0..ow {
                let patch_row = (n * patches_per_image + ph * ow + pw) * patch_size;
                let mut idx = 0usize;
                for c in 0..p.channels {
                    for ki in 0..p.kernel_h {
                        for kj in 0..p.kernel_w {
                            let h = ph * p.stride_h + ki * p.dilation_h;
                            let w = pw * p.stride_w + kj * p.dilation_w;
                            let in_h_pad = h as isize - p.pad_h as isize;
                            let in_w_pad = w as isize - p.pad_w as isize;
                            let v = if in_h_pad < 0
                                || in_w_pad < 0
                                || in_h_pad as usize >= p.in_h
                                || in_w_pad as usize >= p.in_w
                            {
                                zero
                            } else {
                                let src_off = ((n * p.channels + c) * p.in_h + in_h_pad as usize)
                                    * p.in_w
                                    + in_w_pad as usize;
                                src_t[src_off]
                            };
                            dst_t[patch_row + idx] = v;
                            idx += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct Im2ColParams3d {
    pub batch: usize,
    pub channels: usize,
    pub in_t: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_t: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_t: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

impl Im2ColParams3d {
    pub fn out_t(&self) -> usize { (self.in_t - self.kernel_t) / self.stride_t + 1 }
    pub fn out_h(&self) -> usize { (self.in_h - self.kernel_h) / self.stride_h + 1 }
    pub fn out_w(&self) -> usize { (self.in_w - self.kernel_w) / self.stride_w + 1 }
}

pub fn im2col_3d_dispatch(
    dtype: DType,
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams3d,
) -> Result<(usize, usize)> {
    if p.kernel_t == 0 || p.kernel_h == 0 || p.kernel_w == 0 {
        return Err(SynaptixError::Unsupported("im2col3d: zero kernel"));
    }
    if p.stride_t == 0 || p.stride_h == 0 || p.stride_w == 0 {
        return Err(SynaptixError::Unsupported("im2col3d: zero stride"));
    }
    let ot = p.out_t();
    let oh = p.out_h();
    let ow = p.out_w();
    let patches_per_image = ot * oh * ow;
    let patch_size = p.channels * p.kernel_t * p.kernel_h * p.kernel_w;
    let needed = p.batch * patches_per_image * patch_size * elem_size(dtype)?;
    if dst.len() < needed {
        return Err(SynaptixError::Other(format!(
            "im2col3d: dst too small, expected {} bytes, got {}",
            needed,
            dst.len()
        )));
    }
    match dtype {
        DType::F32 => im2col_3d_typed::<f32>(src, dst, p, ot, oh, ow),
        DType::F64 => im2col_3d_typed::<f64>(src, dst, p, ot, oh, ow),
        DType::F16 => im2col_3d_typed::<half::f16>(src, dst, p, ot, oh, ow),
        DType::BF16 => im2col_3d_typed::<half::bf16>(src, dst, p, ot, oh, ow),
        _ => Err(SynaptixError::Unsupported("im2col3d: dtype")),
    }?;
    Ok((p.batch * patches_per_image, patch_size))
}

fn im2col_3d_typed<T: bytemuck::Pod + bytemuck::Zeroable + Copy>(
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams3d,
    ot: usize,
    oh: usize,
    ow: usize,
) -> Result<()> {
    let src_t: &[T] = bytemuck::cast_slice(src);
    let dst_t: &mut [T] = bytemuck::cast_slice_mut(dst);
    let patches_per_image = ot * oh * ow;
    let patch_size = p.channels * p.kernel_t * p.kernel_h * p.kernel_w;
    for n in 0..p.batch {
        for pt in 0..ot {
            for ph in 0..oh {
                for pw in 0..ow {
                    let patch_row =
                        (n * patches_per_image + (pt * oh + ph) * ow + pw) * patch_size;
                    let mut idx = 0usize;
                    for c in 0..p.channels {
                        for kt in 0..p.kernel_t {
                            for ki in 0..p.kernel_h {
                                for kj in 0..p.kernel_w {
                                    let t = pt * p.stride_t + kt;
                                    let h = ph * p.stride_h + ki;
                                    let w = pw * p.stride_w + kj;
                                    let src_off = (((n * p.channels + c) * p.in_t + t) * p.in_h
                                        + h)
                                        * p.in_w
                                        + w;
                                    dst_t[patch_row + idx] = src_t[src_off];
                                    idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn elem_size(dtype: DType) -> Result<usize> {
    if dtype.is_sub_byte() || dtype.is_quantized() {
        return Err(SynaptixError::Unsupported("im2col: sub-byte/quant dtype"));
    }
    Ok((dtype.size_in_bits() / 8).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn im2col_2d_identity_patch1() {
        let p = Im2ColParams2d {
            batch: 1,
            channels: 1,
            in_h: 2,
            in_w: 2,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        };
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src_bytes = bytemuck::cast_slice(&src).to_vec();
        let mut dst = vec![0u8; 4 * 4];
        let (rows, cols) = im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 1);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert_eq!(out, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn im2col_2d_patch2_stride2() {
        let p = Im2ColParams2d {
            batch: 1,
            channels: 1,
            in_h: 4,
            in_w: 4,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        };
        let src: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let src_bytes = bytemuck::cast_slice(&src).to_vec();
        let mut dst = vec![0u8; 4 * 4 * 4];
        let (rows, cols) = im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 4);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert_eq!(&out[0..4], &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(&out[4..8], &[3.0, 4.0, 7.0, 8.0]);
        assert_eq!(&out[8..12], &[9.0, 10.0, 13.0, 14.0]);
        assert_eq!(&out[12..16], &[11.0, 12.0, 15.0, 16.0]);
    }

    #[test]
    fn im2col_2d_padding() {
        let p = Im2ColParams2d {
            batch: 1,
            channels: 1,
            in_h: 2,
            in_w: 2,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
        };
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src_bytes = bytemuck::cast_slice(&src).to_vec();
        let mut dst = vec![0u8; 4 * 9 * 4];
        let (rows, cols) = im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 9);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert_eq!(&out[0..9], &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);
    }
}
