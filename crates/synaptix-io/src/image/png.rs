use std::path::Path;
use image::ImageReader;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::{IoError, Result};

pub fn load_image(path: impl AsRef<Path>, device: Device) -> Result<Tensor> {
    let img = ImageReader::open(path)
        .map_err(|e| IoError::Image(e.to_string()))?
        .decode()
        .map_err(|e| IoError::Image(e.to_string()))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<f32> = img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
    let (w, h) = (w as usize, h as usize);
    let chw = hwc_to_chw(&pixels, h, w, 3);
    Tensor::from_raw_bytes(f32_to_bytes(&chw), vec![3, h, w], DType::F32, device)
        .map_err(IoError::Core)
}

pub fn load_image_rgba(path: impl AsRef<Path>, device: Device) -> Result<Tensor> {
    let img = ImageReader::open(path)
        .map_err(|e| IoError::Image(e.to_string()))?
        .decode()
        .map_err(|e| IoError::Image(e.to_string()))?
        .into_rgba8();
    let (w, h) = img.dimensions();
    let pixels: Vec<f32> = img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
    let (w, h) = (w as usize, h as usize);
    let chw = hwc_to_chw(&pixels, h, w, 4);
    Tensor::from_raw_bytes(f32_to_bytes(&chw), vec![4, h, w], DType::F32, device)
        .map_err(IoError::Core)
}

pub fn save_image(tensor: &Tensor, path: impl AsRef<Path>) -> Result<()> {
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(IoError::Image(format!("expected CHW tensor, got {:?}", dims)));
    }
    let (c, h, w) = (dims[0], dims[1], dims[2]);
    let flat = tensor.flatten_all().map_err(IoError::Core)?
        .to_vec1::<f32>().map_err(IoError::Core)?;
    let hwc = chw_to_hwc(&flat, c, h, w);
    let bytes: Vec<u8> = hwc.into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    match c {
        3 => {
            image::DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(w as u32, h as u32, bytes)
                    .ok_or_else(|| IoError::Image("buffer mismatch".into()))?,
            ).save(path).map_err(|e| IoError::Image(e.to_string()))
        }
        4 => {
            image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(w as u32, h as u32, bytes)
                    .ok_or_else(|| IoError::Image("buffer mismatch".into()))?,
            ).save(path).map_err(|e| IoError::Image(e.to_string()))
        }
        _ => Err(IoError::Image(format!("unsupported channels {c}"))),
    }
}

pub fn hwc_to_chw(data: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * h * w];
    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                out[ch * h * w + row * w + col] = data[(row * w + col) * c + ch];
            }
        }
    }
    out
}

pub fn chw_to_hwc(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * h * w];
    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                out[(row * w + col) * c + ch] = data[ch * h * w + row * w + col];
            }
        }
    }
    out
}

pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 4];
    for (i, &f) in v.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    out
}
