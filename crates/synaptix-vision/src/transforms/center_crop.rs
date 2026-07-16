use crate::error::Result;
use crate::image_buf::RgbImage;

pub fn center_crop(img: &RgbImage, crop_w: usize, crop_h: usize) -> Result<RgbImage> {
    let src = img.to_hwc();
    let cw = crop_w.min(src.width);
    let ch = crop_h.min(src.height);
    let x0 = (src.width - cw) / 2;
    let y0 = (src.height - ch) / 2;
    crop_region(&src, x0, y0, cw, ch)
}

pub(crate) fn crop_region(src: &RgbImage, x0: usize, y0: usize, cw: usize, ch: usize) -> Result<RgbImage> {
    let mut out = RgbImage::zeros_hwc(cw, ch, src.channels);
    for y in 0..ch {
        for x in 0..cw {
            for c in 0..src.channels {
                out.set_pixel(x, y, c, src.pixel(x0 + x, y0 + y, c));
            }
        }
    }
    Ok(out)
}
