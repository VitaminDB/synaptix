use crate::error::{Result, VisionError};
use crate::image_buf::RgbImage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl BBox {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

pub fn nms_iou(boxes: &[BBox], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        boxes[b].score.partial_cmp(&boxes[a].score).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<usize> = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        for &j in &order {
            if i == j || suppressed[j] {
                continue;
            }
            if boxes[i].iou(&boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

pub fn roi_pool_bilinear(
    feature: &RgbImage,
    roi: &BBox,
    out_w: usize,
    out_h: usize,
) -> Result<RgbImage> {
    if out_w == 0 || out_h == 0 {
        return Err(VisionError::invalid_arg("roi_pool_bilinear: out dims must be > 0"));
    }
    let src = feature.to_hwc();
    let w_f = (roi.x2 - roi.x1).max(1e-6);
    let h_f = (roi.y2 - roi.y1).max(1e-6);
    let mut out = RgbImage::zeros_hwc(out_w, out_h, src.channels);
    for y_out in 0..out_h {
        for x_out in 0..out_w {
            let fx = roi.x1 + (x_out as f32 + 0.5) * w_f / out_w as f32 - 0.5;
            let fy = roi.y1 + (y_out as f32 + 0.5) * h_f / out_h as f32 - 0.5;
            let x0 = fx.floor().clamp(0.0, (src.width - 1) as f32) as usize;
            let y0 = fy.floor().clamp(0.0, (src.height - 1) as f32) as usize;
            let x1 = (x0 + 1).min(src.width - 1);
            let y1 = (y0 + 1).min(src.height - 1);
            let dx = (fx - x0 as f32).clamp(0.0, 1.0);
            let dy = (fy - y0 as f32).clamp(0.0, 1.0);
            for c in 0..src.channels {
                let v00 = src.pixel(x0, y0, c);
                let v01 = src.pixel(x1, y0, c);
                let v10 = src.pixel(x0, y1, c);
                let v11 = src.pixel(x1, y1, c);
                let v0 = v00 * (1.0 - dx) + v01 * dx;
                let v1 = v10 * (1.0 - dx) + v11 * dx;
                out.set_pixel(x_out, y_out, c, v0 * (1.0 - dy) + v1 * dy);
            }
        }
    }
    Ok(out)
}
