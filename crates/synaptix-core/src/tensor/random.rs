use crate::device::Device;
use crate::dtype::{DType, SynaptixScalar};
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::shape::IntoShape;

use rand::distributions::Distribution;
use rand::SeedableRng;

static GLOBAL_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0xDEADBEEF);

pub fn set_global_seed(seed: u64) {
    GLOBAL_SEED.store(seed, std::sync::atomic::Ordering::Relaxed);
}

fn base_seed() -> u64 {
    GLOBAL_SEED.load(std::sync::atomic::Ordering::Relaxed)
}

impl Tensor {
    pub fn rand_uniform<S: IntoShape>(
        shape: S,
        low: f32,
        high: f32,
        device: Device,
    ) -> Result<Self> {
        if !device.is_cpu() {
            return Err(SynaptixError::Unsupported("rand on non-cpu device"));
        }
        let shape = shape.into_shape();
        let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed());
        let dist = rand::distributions::Uniform::new(low, high);
        let data: Vec<f32> = (0..shape.numel()).map(|_| dist.sample(&mut rng)).collect();
        Tensor::from_vec(data, shape, device)
    }

    pub fn randn<S: IntoShape>(shape: S, device: Device) -> Result<Self> {
        Self::randn_seeded(shape, base_seed(), device)
    }

    pub fn randn_seeded<S: IntoShape>(shape: S, seed: u64, device: Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(SynaptixError::Unsupported("randn on non-cpu device"));
        }
        let shape = shape.into_shape();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = rand_distr_normal();
        let data: Vec<f32> = (0..shape.numel()).map(|_| dist(&mut rng)).collect();
        Tensor::from_vec(data, shape, device)
    }

    pub fn rand_seeded<S: IntoShape, T: SynaptixScalar>(
        _shape: S,
        _seed: u64,
        _device: Device,
    ) -> Result<Self> {
        Err(SynaptixError::Unsupported("rand_seeded: not implemented in MVP"))
    }
}

fn rand_distr_normal() -> impl Fn(&mut rand::rngs::StdRng) -> f32 {
    move |rng: &mut rand::rngs::StdRng| {
        use rand::Rng;
        let u1: f32 = rng.gen_range(1e-10_f32..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z
    }
}

#[allow(dead_code)]
fn _suppress_unused_dtype() { let _ = DType::F32; }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randn_seeded_matches_randn_bitwise() {
        let a = Tensor::randn(vec![2usize, 3, 5], Device::Cpu).unwrap();
        let b = Tensor::randn_seeded(vec![2usize, 3, 5], base_seed(), Device::Cpu).unwrap();
        let av = a.reshape(vec![30usize]).unwrap().to_vec1::<f32>().unwrap();
        let bv = b.reshape(vec![30usize]).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            av.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            bv.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn randn_seeded_distinct_seeds_differ() {
        let a = Tensor::randn_seeded(vec![64usize], 1, Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        let b = Tensor::randn_seeded(vec![64usize], 2, Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(a, b);
    }
}
