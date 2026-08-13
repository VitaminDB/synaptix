use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::H3Error;

pub const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
pub const FRAME_RESCALE: f64 = 5.0 / 3.0;
pub const AXIS_SCALE: f64 = 32.0;
pub const SPATIAL_PATCH: usize = 2;

pub fn axis_from_sqrt_area(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / patch;
    let base = (1.0 - ratio) / 2.0;
    let step = ratio / n as f64;
    (0..n).map(|i| (i as f64 * step + base) * AXIS_SCALE).collect()
}

pub struct FrameGrid {
    pub rows: Vec<[f64; 2]>,
    pub w_axis: Vec<f64>,
}

impl FrameGrid {
    pub fn new(latent_h: usize, latent_w: usize) -> Self {
        let area = ((latent_h * latent_w) as f64).sqrt();
        let h_axis = axis_from_sqrt_area(latent_h, SPATIAL_PATCH, area);
        let w_axis = axis_from_sqrt_area(latent_w, SPATIAL_PATCH, area);
        let mut rows = Vec::with_capacity(h_axis.len() * w_axis.len());
        for &hh in &h_axis {
            for &ww in &w_axis {
                rows.push([hh, ww]);
            }
        }
        Self { rows, w_axis }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn w_bounds(&self) -> (f64, f64) {
        let lo = self.w_axis.first().copied().unwrap_or(0.0);
        let hi = self.w_axis.last().copied().unwrap_or(0.0);
        (lo, hi)
    }
}

pub fn video_t_spans(n: usize) -> Vec<f64> {
    (0..n).map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % FRAME_PER_TOKEN.len()]).collect()
}

pub fn video_t_span_total(n: usize) -> f64 {
    video_t_spans(n).iter().sum()
}

pub fn video_t_grid(n: usize, origin: f64) -> Vec<f64> {
    let spans = video_t_spans(n);
    let mut out = Vec::with_capacity(n);
    let mut acc = 0.0;
    for (i, s) in spans.iter().enumerate() {
        out.push(origin + acc);
        if i + 1 < n {
            acc += s;
        }
    }
    out
}

pub fn video_positions(latent_t: usize, frame: &FrameGrid, cursor: f64) -> Vec<[f64; 3]> {
    let t_grid = video_t_grid(latent_t, cursor);
    let mut out = Vec::with_capacity(latent_t * frame.len());
    for &t in &t_grid {
        for hw in &frame.rows {
            out.push([t, hw[0], hw[1]]);
        }
    }
    out
}

pub fn audio_positions(cursor: f64, latent_t: usize, w_low: f64, w_high: f64) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(latent_t * 2);
    for i in 0..latent_t {
        out.push([cursor + i as f64, 0.0, w_low]);
    }
    for i in 0..latent_t {
        out.push([cursor + i as f64, 0.0, w_high]);
    }
    out
}

pub fn text_positions(len: usize) -> Vec<[f64; 3]> {
    (0..len).map(|i| [i as f64, 0.0, 0.0]).collect()
}

pub struct RopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
    pub rot_dim: usize,
    pub seq_len: usize,
}

impl RopeTables {
    pub fn build(
        positions: &[[f64; 3]],
        inv_freq: &[f32],
        device: Device,
    ) -> Result<Self, H3Error> {
        let n = inv_freq.len();
        let half = n * 3;
        let s = positions.len();
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        fill_angles(&mut cos, &mut sin, positions, inv_freq);
        let cos_t = Tensor::from_vec(cos, vec![s, half], device)?;
        let sin_t = Tensor::from_vec(sin, vec![s, half], device)?;
        Ok(Self { cos: cos_t, sin: sin_t, rot_dim: half * 2, seq_len: s })
    }

    pub fn from_angles(
        angles: Vec<f32>,
        seq_len: usize,
        half: usize,
        device: Device,
    ) -> Result<Self, H3Error> {
        let mut cos = vec![0f32; angles.len()];
        let mut sin = vec![0f32; angles.len()];
        for (i, a) in angles.iter().enumerate() {
            cos[i] = a.cos();
            sin[i] = a.sin();
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, vec![seq_len, half], device)?,
            sin: Tensor::from_vec(sin, vec![seq_len, half], device)?,
            rot_dim: half * 2,
            seq_len,
        })
    }

    pub fn apply_bshd(&self, x: &Tensor) -> Result<Tensor, H3Error> {
        let dims = x.dims();
        if dims.len() != 4 || dims[1] != self.seq_len {
            return Err(H3Error::Layout(format!(
                "rope bshd: ожидалось [1,{},H,D], получено {:?}",
                self.seq_len, dims
            )));
        }
        Ok(x.rope_split_partial_fused(&self.cos, &self.sin, self.rot_dim)?)
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor, H3Error> {
        let dims = x.dims();
        let rank = dims.len();
        if rank < 2 || dims[rank - 2] != self.seq_len {
            return Err(H3Error::Layout(format!(
                "rope: ожидалась ось S={}, получено {:?}",
                self.seq_len, dims
            )));
        }
        if matches!(x.device(), Device::Cuda(_)) {
            if let Ok(y) = x.rope_split_partial_fused(&self.cos, &self.sin, self.rot_dim) {
                return Ok(y);
            }
        }
        self.apply_fallback(x)
    }

    fn apply_fallback(&self, x: &Tensor) -> Result<Tensor, H3Error> {
        let dims = x.dims().to_vec();
        let d = dims[dims.len() - 1];
        let half = self.rot_dim / 2;
        let dt = x.dtype();
        let cos = self.cos.to_dtype(dt)?;
        let sin = self.sin.to_dtype(dt)?;
        let rot = x.narrow(dims.len() - 1, 0, self.rot_dim)?.contiguous()?;
        let x0 = rot.narrow(dims.len() - 1, 0, half)?.contiguous()?;
        let x1 = rot.narrow(dims.len() - 1, half, half)?.contiguous()?;
        let out0 = x0.mul(&cos)?.sub(&x1.mul(&sin)?)?;
        let out1 = x1.mul(&cos)?.add(&x0.mul(&sin)?)?;
        if self.rot_dim == d {
            return Ok(Tensor::cat(&[&out0, &out1], dims.len() - 1)?);
        }
        let pass = x
            .narrow(dims.len() - 1, self.rot_dim, d - self.rot_dim)?
            .contiguous()?;
        Ok(Tensor::cat(&[&out0, &out1, &pass], dims.len() - 1)?)
    }
}

fn fill_angles(cos: &mut [f32], sin: &mut [f32], positions: &[[f64; 3]], inv_freq: &[f32]) {
    let n = inv_freq.len();
    let half = n * 3;
    let work = |cos: &mut [f32], sin: &mut [f32], pos: &[[f64; 3]]| {
        for (s, p) in pos.iter().enumerate() {
            for axis in 0..3 {
                let pv = p[axis] as f32;
                for i in 0..n {
                    let ang = pv * inv_freq[i];
                    let o = s * half + axis * n + i;
                    cos[o] = ang.cos();
                    sin[o] = ang.sin();
                }
            }
        }
    };
    if positions.len() >= 4096 {
        let nthr = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(16);
        let chunk = positions.len().div_ceil(nthr);
        std::thread::scope(|sp| {
            for ((pc, cc), sc) in positions
                .chunks(chunk)
                .zip(cos.chunks_mut(chunk * half))
                .zip(sin.chunks_mut(chunk * half))
            {
                sp.spawn(move || work(cc, sc, pc));
            }
        });
    } else {
        work(cos, sin, positions);
    }
}

pub fn read_inv_freq(t: &Tensor) -> Result<Vec<f32>, H3Error> {
    let host = t.to_device(Device::Cpu)?.to_dtype(DType::F32)?;
    Ok(host.to_vec1::<f32>()?)
}
