use crate::error::Result;
use crate::image_buf::RgbImage;

#[derive(Debug, Clone, Copy)]
pub enum PadFill {
    Zero,
    Value(f32),
}

pub fn pad_to_multiple(img: &RgbImage, multiple: usize, fill: PadFill) -> Result<RgbImage> {
    let src = img.to_hwc();
    if multiple == 0 {
        return Ok(src);
    }
    let new_w = ((src.width + multiple - 1) / multiple) * multiple;
    let new_h = ((src.height + multiple - 1) / multiple) * multiple;
    let mut out = match fill {
        PadFill::Zero => RgbImage::zeros_hwc(new_w, new_h, src.channels),
        PadFill::Value(v) => {
            let mut o = RgbImage::zeros_hwc(new_w, new_h, src.channels);
            for px in o.data.iter_mut() {
                *px = v;
            }
            o
        }
    };
    for y in 0..src.height {
        for x in 0..src.width {
            for c in 0..src.channels {
                out.set_pixel(x, y, c, src.pixel(x, y, c));
            }
        }
    }
    Ok(out)
}
