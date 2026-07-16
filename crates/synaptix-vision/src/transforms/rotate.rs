use crate::error::{Result, VisionError};
use crate::image_buf::RgbImage;

pub fn rotate90(img: &RgbImage, quarter_turns: u32) -> Result<RgbImage> {
    let src = img.to_hwc();
    match quarter_turns % 4 {
        0 => Ok(src),
        1 => {
            let mut out = RgbImage::zeros_hwc(src.height, src.width, src.channels);
            for y in 0..src.height {
                for x in 0..src.width {
                    for c in 0..src.channels {
                        out.set_pixel(src.height - 1 - y, x, c, src.pixel(x, y, c));
                    }
                }
            }
            Ok(out)
        }
        2 => {
            let mut out = RgbImage::zeros_hwc(src.width, src.height, src.channels);
            for y in 0..src.height {
                for x in 0..src.width {
                    for c in 0..src.channels {
                        out.set_pixel(src.width - 1 - x, src.height - 1 - y, c, src.pixel(x, y, c));
                    }
                }
            }
            Ok(out)
        }
        3 => {
            let mut out = RgbImage::zeros_hwc(src.height, src.width, src.channels);
            for y in 0..src.height {
                for x in 0..src.width {
                    for c in 0..src.channels {
                        out.set_pixel(y, src.width - 1 - x, c, src.pixel(x, y, c));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(VisionError::invalid_arg("rotate90: unreachable")),
    }
}
