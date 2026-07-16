use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::{Philox4x32, bernoulli_mask};

use crate::module::Module;

pub struct Dropout {
    p: f32,
    seed: u64,
    counter: AtomicU64,
    training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self::with_seed(p, 0xDEADBEEF_DEADBEEF)
    }

    pub fn with_seed(p: f32, seed: u64) -> Self {
        Self {
            p,
            seed,
            counter: AtomicU64::new(0),
            training: AtomicBool::new(true),
        }
    }

    pub fn p(&self) -> f32 { self.p }
    pub fn is_training(&self) -> bool { self.training.load(Ordering::Acquire) }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Acquire) || self.p <= 0.0 {
            return Ok(x.clone());
        }
        if !(self.p < 1.0) {
            return Err(SynaptixError::Unsupported("Dropout: p must be < 1.0"));
        }
        let n = x.numel();
        let blocks = ((n as u64) + 3) / 4;
        let start = self.counter.fetch_add(blocks, Ordering::AcqRel);
        let mut rng = Philox4x32::new(self.seed);
        rng.advance(start);
        let mask = bernoulli_mask(&mut rng, 1.0 - self.p, n);
        let scale = 1.0 / (1.0 - self.p);
        let m_f32 = Tensor::from_vec(mask, x.dims().to_vec(), x.device())?;
        let m = if x.dtype() != DType::F32 { m_f32.to_dtype(x.dtype())? } else { m_f32 };
        let dtype_in = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let mf = m.to_dtype(DType::F32)?;
        let out = xf.mul(&mf)?.mul_scalar(scale)?;
        out.to_dtype(dtype_in)
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Release);
    }
}

pub struct AlphaDropout {
    p: f32,
    seed: u64,
    counter: AtomicU64,
    training: AtomicBool,
}

impl AlphaDropout {
    pub fn new(p: f32) -> Self {
        Self::with_seed(p, 0xC0FFEE_C0FFEE)
    }

    pub fn with_seed(p: f32, seed: u64) -> Self {
        Self { p, seed, counter: AtomicU64::new(0), training: AtomicBool::new(true) }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Acquire) || self.p <= 0.0 {
            return Ok(x.clone());
        }
        let n = x.numel();
        let blocks = ((n as u64) + 3) / 4;
        let start = self.counter.fetch_add(blocks, Ordering::AcqRel);
        let mut rng = Philox4x32::new(self.seed);
        rng.advance(start);
        let mask = bernoulli_mask(&mut rng, 1.0 - self.p, n);
        let alpha = -1.7580993408473766_f32;
        let q = 1.0 - self.p;
        let a = (q + alpha * alpha * q * (1.0 - q)).recip().sqrt();
        let b = -a * alpha * (1.0 - q);
        let dtype_in = x.dtype();
        let xf = x.to_dtype(DType::F32)?.contiguous()?;
        let mut data = xf.reshape((n,))?.to_vec1::<f32>()?;
        for (i, val) in data.iter_mut().enumerate() {
            if mask[i] > 0.5 {
                *val = a * (*val) + b;
            } else {
                *val = a * alpha + b;
            }
        }
        let out = Tensor::from_vec(data, x.dims().to_vec(), x.device())?;
        out.to_dtype(dtype_in)
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Release);
    }
}
