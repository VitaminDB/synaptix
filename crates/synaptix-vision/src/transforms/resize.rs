use crate::error::Result;
use crate::image_buf::RgbImage;

pub fn resize_bilinear(img: &RgbImage, new_w: usize, new_h: usize) -> Result<RgbImage> {
    let src = img.to_hwc();
    let mut out = RgbImage::zeros_hwc(new_w, new_h, src.channels);
    if new_w == 0 || new_h == 0 || src.width == 0 || src.height == 0 {
        return Ok(out);
    }
    let scale_x = src.width as f32 / new_w as f32;
    let scale_y = src.height as f32 / new_h as f32;
    for y_out in 0..new_h {
        let sy = (y_out as f32 + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().clamp(0.0, (src.height - 1) as f32) as usize;
        let y1 = (y0 + 1).min(src.height - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x_out in 0..new_w {
            let sx = (x_out as f32 + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().clamp(0.0, (src.width - 1) as f32) as usize;
            let x1 = (x0 + 1).min(src.width - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..src.channels {
                let v00 = src.pixel(x0, y0, c);
                let v01 = src.pixel(x1, y0, c);
                let v10 = src.pixel(x0, y1, c);
                let v11 = src.pixel(x1, y1, c);
                let v0 = v00 * (1.0 - fx) + v01 * fx;
                let v1 = v10 * (1.0 - fx) + v11 * fx;
                out.set_pixel(x_out, y_out, c, v0 * (1.0 - fy) + v1 * fy);
            }
        }
    }
    Ok(out)
}

pub fn resize_nearest(img: &RgbImage, new_w: usize, new_h: usize) -> Result<RgbImage> {
    let src = img.to_hwc();
    let mut out = RgbImage::zeros_hwc(new_w, new_h, src.channels);
    if new_w == 0 || new_h == 0 || src.width == 0 || src.height == 0 {
        return Ok(out);
    }
    let scale_x = src.width as f32 / new_w as f32;
    let scale_y = src.height as f32 / new_h as f32;
    for y_out in 0..new_h {
        let sy = ((y_out as f32 + 0.5) * scale_y).floor() as usize;
        let sy = sy.min(src.height - 1);
        for x_out in 0..new_w {
            let sx = ((x_out as f32 + 0.5) * scale_x).floor() as usize;
            let sx = sx.min(src.width - 1);
            for c in 0..src.channels {
                out.set_pixel(x_out, y_out, c, src.pixel(sx, sy, c));
            }
        }
    }
    Ok(out)
}
