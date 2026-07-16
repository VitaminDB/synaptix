use crate::error::{Result, VisionError};
use crate::image_buf::RgbImage;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub fn normalize(img: &RgbImage, mean: &[f32], std: &[f32]) -> Result<RgbImage> {
    if mean.len() != img.channels || std.len() != img.channels {
        return Err(VisionError::invalid_arg(format!(
            "normalize: mean/std len {}/{} != channels {}",
            mean.len(),
            std.len(),
            img.channels
        )));
    }
    let hwc = img.to_hwc();
    let mut out = hwc.clone();
    for y in 0..hwc.height {
        for x in 0..hwc.width {
            for c in 0..hwc.channels {
                let v = hwc.pixel(x, y, c);
                out.set_pixel(x, y, c, (v - mean[c]) / std[c].max(1e-12));
            }
        }
    }
    Ok(out)
}
