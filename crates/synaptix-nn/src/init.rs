use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::{Philox4x32, fill_normal_f32, fill_uniform_f32};

#[derive(Debug, Clone, Copy)]
pub enum InitMethod {
    Zeros,
    Ones,
    Constant(f32),
    Uniform { low: f32, high: f32 },
    Normal { mean: f32, std: f32 },
    KaimingUniform { fan_in: usize, a: f32 },
    KaimingNormal { fan_in: usize, a: f32 },
    XavierUniform { fan_in: usize, fan_out: usize },
    XavierNormal { fan_in: usize, fan_out: usize },
    Orthogonal { gain: f32 },
}

pub fn init_tensor(
    shape: &[usize],
    method: InitMethod,
    dtype: DType,
    seed: u64,
    device: Device,
) -> Result<Tensor> {
    if shape.is_empty() {
        return Err(SynaptixError::Unsupported("init_tensor: empty shape"));
    }
    let numel: usize = shape.iter().product();
    let mut rng = Philox4x32::new(seed);
    let data_f32 = match method {
        InitMethod::Zeros => vec![0.0_f32; numel],
        InitMethod::Ones => vec![1.0_f32; numel],
        InitMethod::Constant(c) => vec![c; numel],
        InitMethod::Uniform { low, high } => {
            let mut buf = vec![0.0_f32; numel];
            fill_uniform_f32(&mut rng, &mut buf, low, high);
            buf
        }
        InitMethod::Normal { mean, std } => {
            let mut buf = vec![0.0_f32; numel];
            fill_normal_f32(&mut rng, &mut buf);
            for v in buf.iter_mut() {
                *v = *v * std + mean;
            }
            buf
        }
        InitMethod::KaimingUniform { fan_in, a } => {
            if fan_in == 0 {
                return Err(SynaptixError::Unsupported("kaiming: fan_in=0"));
            }
            let bound = (6.0_f32 / ((1.0 + a * a) * fan_in as f32)).sqrt();
            let mut buf = vec![0.0_f32; numel];
            fill_uniform_f32(&mut rng, &mut buf, -bound, bound);
            buf
        }
        InitMethod::KaimingNormal { fan_in, a } => {
            if fan_in == 0 {
                return Err(SynaptixError::Unsupported("kaiming: fan_in=0"));
            }
            let std = (2.0_f32 / ((1.0 + a * a) * fan_in as f32)).sqrt();
            let mut buf = vec![0.0_f32; numel];
            fill_normal_f32(&mut rng, &mut buf);
            for v in buf.iter_mut() {
                *v *= std;
            }
            buf
        }
        InitMethod::XavierUniform { fan_in, fan_out } => {
            if fan_in + fan_out == 0 {
                return Err(SynaptixError::Unsupported("xavier: fan_in+fan_out=0"));
            }
            let bound = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            let mut buf = vec![0.0_f32; numel];
            fill_uniform_f32(&mut rng, &mut buf, -bound, bound);
            buf
        }
        InitMethod::XavierNormal { fan_in, fan_out } => {
            if fan_in + fan_out == 0 {
                return Err(SynaptixError::Unsupported("xavier: fan_in+fan_out=0"));
            }
            let std = (2.0_f32 / (fan_in + fan_out) as f32).sqrt();
            let mut buf = vec![0.0_f32; numel];
            fill_normal_f32(&mut rng, &mut buf);
            for v in buf.iter_mut() {
                *v *= std;
            }
            buf
        }
        InitMethod::Orthogonal { gain } => {
            if shape.len() < 2 {
                return Err(SynaptixError::Unsupported("orthogonal: needs rank >= 2"));
            }
            let rows = shape[0];
            let cols: usize = shape[1..].iter().product();
            let mut a = vec![0.0_f32; rows * cols];
            fill_normal_f32(&mut rng, &mut a);
            let q = gram_schmidt_qr(&a, rows, cols);
            let mut out: Vec<f32> = q.into_iter().map(|v| v * gain).collect();
            if out.len() < numel {
                out.resize(numel, 0.0);
            }
            out
        }
    };
    let t_f32 = Tensor::from_vec(data_f32, shape.to_vec(), device)?;
    if dtype == DType::F32 {
        Ok(t_f32)
    } else {
        t_f32.to_dtype(dtype)
    }
}

fn gram_schmidt_qr(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    if rows >= cols {
        let mut q = vec![0.0_f32; rows * cols];
        for j in 0..cols {
            for r in 0..rows {
                q[r * cols + j] = a[r * cols + j];
            }
            for k in 0..j {
                let mut dot = 0.0_f64;
                for r in 0..rows {
                    dot += (q[r * cols + k] as f64) * (q[r * cols + j] as f64);
                }
                for r in 0..rows {
                    q[r * cols + j] -= (dot * q[r * cols + k] as f64) as f32;
                }
            }
            let mut norm = 0.0_f64;
            for r in 0..rows {
                norm += (q[r * cols + j] as f64).powi(2);
            }
            let norm = norm.sqrt() as f32;
            if norm > 1e-8 {
                for r in 0..rows {
                    q[r * cols + j] /= norm;
                }
            }
        }
        q
    } else {
        let mut q = vec![0.0_f32; rows * cols];
        for i in 0..rows {
            for c in 0..cols {
                q[i * cols + c] = a[i * cols + c];
            }
            for k in 0..i {
                let mut dot = 0.0_f64;
                for c in 0..cols {
                    dot += (q[i * cols + c] as f64) * (q[k * cols + c] as f64);
                }
                for c in 0..cols {
                    q[i * cols + c] -= (dot * q[k * cols + c] as f64) as f32;
                }
            }
            let mut norm = 0.0_f64;
            for c in 0..cols {
                norm += (q[i * cols + c] as f64).powi(2);
            }
            let norm = norm.sqrt() as f32;
            if norm > 1e-8 {
                for c in 0..cols {
                    q[i * cols + c] /= norm;
                }
            }
        }
        q
    }
}
