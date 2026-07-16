use std::path::Path;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use crate::error::Result;
use super::ffmpeg::VideoReader;

pub fn extract_frames(path: impl AsRef<Path>, fps: Option<f32>, device: Device) -> Result<Vec<Tensor>> {
    let reader = VideoReader::open(path.as_ref())?;
    let all = reader.into_tensors(device)?;
    match fps {
        None => Ok(all),
        Some(target_fps) => Ok(subsample_frames(all, reader.fps_num, reader.fps_den, target_fps)),
    }
}

fn subsample_frames(frames: Vec<Tensor>, src_fps_num: i32, src_fps_den: i32, target_fps: f32) -> Vec<Tensor> {
    if frames.is_empty() { return frames; }
    let src_fps = src_fps_num as f32 / src_fps_den.max(1) as f32;
    if src_fps <= 0.0 || (src_fps - target_fps).abs() < 0.01 { return frames; }
    let step = src_fps / target_fps;
    let mut out = Vec::new();
    let mut cursor = 0.0f32;
    for (i, frame) in frames.into_iter().enumerate() {
        if i as f32 >= cursor {
            out.push(frame);
            cursor += step;
        }
    }
    out
}
