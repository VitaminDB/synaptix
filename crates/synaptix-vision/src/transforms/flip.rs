use crate::error::Result;
use crate::image_buf::RgbImage;

pub fn flip_horizontal(img: &RgbImage) -> Result<RgbImage> {
    let src = img.to_hwc();
    let mut out = RgbImage::zeros_hwc(src.width, src.height, src.channels);
    for y in 0..src.height {
        for x in 0..src.width {
            for c in 0..src.channels {
                out.set_pixel(src.width - 1 - x, y, c, src.pixel(x, y, c));
            }
        }
    }
    Ok(out)
}

pub fn flip_vertical(img: &RgbImage) -> Result<RgbImage> {
    let src = img.to_hwc();
    let mut out = RgbImage::zeros_hwc(src.width, src.height, src.channels);
    for y in 0..src.height {
        for x in 0..src.width {
            for c in 0..src.channels {
                out.set_pixel(x, src.height - 1 - y, c, src.pixel(x, y, c));
            }
        }
    }
    Ok(out)
}
