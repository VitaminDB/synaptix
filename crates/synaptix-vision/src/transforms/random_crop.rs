use rand::Rng;

use crate::error::{Result, VisionError};
use crate::image_buf::RgbImage;
use crate::transforms::center_crop::crop_region;

pub fn random_crop<R: Rng>(img: &RgbImage, crop_w: usize, crop_h: usize, rng: &mut R) -> Result<RgbImage> {
    if crop_w > img.width || crop_h > img.height {
        return Err(VisionError::invalid_arg(format!(
            "random_crop: target {crop_w}x{crop_h} > image {}x{}",
            img.width, img.height
        )));
    }
    let src = img.to_hwc();
    let max_x = src.width - crop_w;
    let max_y = src.height - crop_h;
    let x0 = if max_x == 0 { 0 } else { rng.gen_range(0..=max_x) };
    let y0 = if max_y == 0 { 0 } else { rng.gen_range(0..=max_y) };
    crop_region(&src, x0, y0, crop_w, crop_h)
}
