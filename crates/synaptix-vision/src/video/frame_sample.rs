use crate::error::{Result, VisionError};

pub fn uniform_sample(total_frames: usize, n: usize) -> Result<Vec<usize>> {
    if total_frames == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![total_frames / 2]);
    }
    if n > total_frames {
        return Err(VisionError::invalid_arg(format!(
            "uniform_sample: n {n} > total_frames {total_frames}"
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let idx = ((i as f64) * (total_frames - 1) as f64 / (n - 1) as f64).round() as usize;
        out.push(idx);
    }
    Ok(out)
}

pub fn dense_sample(total_frames: usize, stride: usize) -> Result<Vec<usize>> {
    if total_frames == 0 || stride == 0 {
        return Ok(Vec::new());
    }
    Ok((0..total_frames).step_by(stride).collect())
}
