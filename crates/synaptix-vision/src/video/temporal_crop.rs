use crate::error::{Result, VisionError};

pub fn temporal_crop<T: Clone>(frames: &[T], start: usize, len: usize) -> Result<Vec<T>> {
    if start + len > frames.len() {
        return Err(VisionError::invalid_arg(format!(
            "temporal_crop: start={start}+len={len} > frames {}",
            frames.len()
        )));
    }
    Ok(frames[start..start + len].to_vec())
}
