use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::{IoError, Result};
use super::png::{hwc_to_chw, chw_to_hwc, f32_to_bytes};

pub fn normalize(tensor: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() < 3 {
        return Err(IoError::Image("normalize expects [..., C, H, W]".into()));
    }
    let c = dims[dims.len() - 3];
    if mean.len() != c || std.len() != c {
        return Err(IoError::Image(format!("mean/std length {}/{} != channels {c}", mean.len(), std.len())));
    }
    let flat = tensor.flatten_all().map_err(IoError::Core)?
        .to_vec1::<f32>().map_err(IoError::Core)?;
    let hw = dims[dims.len() - 2] * dims[dims.len() - 1];
    let mut out = flat.clone();
    for ch in 0..c {
        let m = mean[ch];
        let s = std[ch].max(1e-12);
        for i in 0..hw {
            out[ch * hw + i] = (flat[ch * hw + i] - m) / s;
        }
    }
    Tensor::from_raw_bytes(f32_to_bytes(&out), dims.to_vec(), DType::F32, tensor.device())
        .map_err(IoError::Core)
}

pub fn resize_bilinear(tensor: &Tensor, new_h: usize, new_w: usize) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(IoError::Image("resize expects [C, H, W]".into()));
    }
    let (c, src_h, src_w) = (dims[0], dims[1], dims[2]);
    let flat = tensor.flatten_all().map_err(IoError::Core)?
        .to_vec1::<f32>().map_err(IoError::Core)?;
    let mut out = vec![0.0f32; c * new_h * new_w];
    let scale_h = src_h as f32 / new_h as f32;
    let scale_w = src_w as f32 / new_w as f32;
    for ch in 0..c {
        for ny in 0..new_h {
            let sy = (ny as f32 + 0.5) * scale_h - 0.5;
            let y0 = (sy.floor() as i64).clamp(0, src_h as i64 - 1) as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let dy = sy - sy.floor();
            for nx in 0..new_w {
                let sx = (nx as f32 + 0.5) * scale_w - 0.5;
                let x0 = (sx.floor() as i64).clamp(0, src_w as i64 - 1) as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let dx = sx - sx.floor();
                let p00 = flat[ch * src_h * src_w + y0 * src_w + x0];
                let p01 = flat[ch * src_h * src_w + y0 * src_w + x1];
                let p10 = flat[ch * src_h * src_w + y1 * src_w + x0];
                let p11 = flat[ch * src_h * src_w + y1 * src_w + x1];
                let v = p00 * (1.0 - dy) * (1.0 - dx)
                    + p01 * (1.0 - dy) * dx
                    + p10 * dy * (1.0 - dx)
                    + p11 * dy * dx;
                out[ch * new_h * new_w + ny * new_w + nx] = v;
            }
        }
    }
    Tensor::from_raw_bytes(f32_to_bytes(&out), vec![c, new_h, new_w], DType::F32, tensor.device())
        .map_err(IoError::Core)
}

pub fn random_crop(tensor: &Tensor, crop_h: usize, crop_w: usize, seed: u64) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(IoError::Image("random_crop expects [C, H, W]".into()));
    }
    let (c, h, w) = (dims[0], dims[1], dims[2]);
    if crop_h > h || crop_w > w {
        return Err(IoError::Image(format!("crop {crop_h}x{crop_w} > image {h}x{w}")));
    }
    let flat = tensor.flatten_all().map_err(IoError::Core)?
        .to_vec1::<f32>().map_err(IoError::Core)?;
    let rng = splitmix64(seed);
    let top = (rng[0] % (h - crop_h + 1) as u64) as usize;
    let left = (rng[1] % (w - crop_w + 1) as u64) as usize;
    let mut out = vec![0.0f32; c * crop_h * crop_w];
    for ch in 0..c {
        for ny in 0..crop_h {
            for nx in 0..crop_w {
                out[ch * crop_h * crop_w + ny * crop_w + nx] =
                    flat[ch * h * w + (top + ny) * w + (left + nx)];
            }
        }
    }
    Tensor::from_raw_bytes(f32_to_bytes(&out), vec![c, crop_h, crop_w], DType::F32, tensor.device())
        .map_err(IoError::Core)
}

pub fn random_hflip(tensor: &Tensor, seed: u64) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(IoError::Image("random_hflip expects [C, H, W]".into()));
    }
    let (c, h, w) = (dims[0], dims[1], dims[2]);
    let rng = splitmix64(seed);
    if rng[0] % 2 == 0 {
        return Ok(tensor.clone());
    }
    let flat = tensor.flatten_all().map_err(IoError::Core)?
        .to_vec1::<f32>().map_err(IoError::Core)?;
    let mut out = flat.clone();
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                out[ch * h * w + row * w + col] =
                    flat[ch * h * w + row * w + (w - 1 - col)];
            }
        }
    }
    Tensor::from_raw_bytes(f32_to_bytes(&out), dims.to_vec(), DType::F32, tensor.device())
        .map_err(IoError::Core)
}

fn splitmix64(mut x: u64) -> [u64; 2] {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let a = {
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let b = {
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    [a, b]
}
