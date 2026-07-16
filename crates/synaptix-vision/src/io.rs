use std::path::Path;

use image::{ImageBuffer, Rgb};

use crate::error::{Result, VisionError};
use crate::image_buf::{ChannelOrder, RgbImage};

pub fn load_rgb_image(path: impl AsRef<Path>) -> Result<RgbImage> {
    let p = path.as_ref();
    let img = image::open(p).map_err(VisionError::Image)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for px in img.pixels() {
        data.push(px[0] as f32 / 255.0);
        data.push(px[1] as f32 / 255.0);
        data.push(px[2] as f32 / 255.0);
    }
    Ok(RgbImage { data, width: w, height: h, channels: 3, order: ChannelOrder::Hwc })
}

pub fn save_rgb_image(img: &RgbImage, path: impl AsRef<Path>) -> Result<()> {
    if img.channels != 3 {
        return Err(VisionError::invalid_arg("save_rgb_image: expected 3 channels"));
    }
    let hwc = img.to_hwc();
    let mut buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(hwc.width as u32, hwc.height as u32);
    for y in 0..hwc.height {
        for x in 0..hwc.width {
            let r = (hwc.pixel(x, y, 0).clamp(0.0, 1.0) * 255.0) as u8;
            let g = (hwc.pixel(x, y, 1).clamp(0.0, 1.0) * 255.0) as u8;
            let b = (hwc.pixel(x, y, 2).clamp(0.0, 1.0) * 255.0) as u8;
            buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    let p = path.as_ref();
    buf.save(p).map_err(VisionError::Image)?;
    Ok(())
}
