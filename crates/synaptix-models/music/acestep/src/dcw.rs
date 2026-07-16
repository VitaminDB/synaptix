
use synaptix_core::tensor::Tensor;

use crate::AceError;

const INV_SQRT2: f32 = 0.707_106_77;

fn pad_even(x: &Tensor) -> Result<(Tensor, usize), AceError> {
    let d = x.dims().to_vec();
    let t = d[1];
    if t % 2 == 0 {
        return Ok((x.clone(), t));
    }
    let z = Tensor::zeros(vec![d[0], 1, d[2]], x.dtype(), x.device())?;
    Ok((Tensor::cat(&[x, &z], 1)?, t))
}

fn haar_dwt(x: &Tensor) -> Result<(Tensor, Tensor), AceError> {
    let d = x.dims().to_vec();
    let (b, t, c) = (d[0], d[1], d[2]);
    let xr = x.reshape(vec![b, t / 2, 2, c])?;
    let even = xr.narrow(2, 0, 1)?.contiguous()?.reshape(vec![b, t / 2, c])?;
    let odd = xr.narrow(2, 1, 1)?.contiguous()?.reshape(vec![b, t / 2, c])?;
    let low = even.broadcast_add(&odd)?.affine(INV_SQRT2, 0.0)?;
    let high = even.broadcast_add(&odd.affine(-1.0, 0.0)?)?.affine(INV_SQRT2, 0.0)?;
    Ok((low, high))
}

fn haar_idwt(low: &Tensor, high: &Tensor) -> Result<Tensor, AceError> {
    let d = low.dims().to_vec();
    let (b, th, c) = (d[0], d[1], d[2]);
    let even = low.broadcast_add(high)?.affine(INV_SQRT2, 0.0)?;
    let odd = low.broadcast_add(&high.affine(-1.0, 0.0)?)?.affine(INV_SQRT2, 0.0)?;
    let stacked = Tensor::cat(&[&even.reshape(vec![b, th, 1, c])?, &odd.reshape(vec![b, th, 1, c])?], 2)?;
    Ok(stacked.reshape(vec![b, th * 2, c])?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcwMode {
    Low,
    High,
    Double,
    Pix,
}

#[derive(Debug, Clone)]
pub struct DcwCorrector {
    pub enabled: bool,
    pub mode: DcwMode,
    pub scaler: f32,
    pub high_scaler: f32,
}

impl Default for DcwCorrector {
    fn default() -> Self {
        Self { enabled: false, mode: DcwMode::Double, scaler: 0.05, high_scaler: 0.02 }
    }
}

impl DcwCorrector {
    pub fn is_active(&self) -> bool {
        self.enabled
            && match self.mode {
                DcwMode::Double => self.scaler != 0.0 || self.high_scaler != 0.0,
                _ => self.scaler != 0.0,
            }
    }

    fn correct_band(xb: &Tensor, yb: &Tensor, lam: f32) -> Result<Tensor, AceError> {
        let diff = xb.broadcast_add(&yb.affine(-1.0, 0.0)?)?;
        Ok(xb.broadcast_add(&diff.affine(lam, 0.0)?)?)
    }

    pub fn apply(&self, x: &Tensor, y: &Tensor, t: f32) -> Result<Tensor, AceError> {
        if !self.is_active() {
            return Ok(x.clone());
        }
        if self.mode == DcwMode::Pix {
            let diff = x.broadcast_add(&y.affine(-1.0, 0.0)?)?;
            return Ok(x.broadcast_add(&diff.affine(self.scaler, 0.0)?)?);
        }
        let out_t = x.dims()[1];
        let (xp, _) = pad_even(x)?;
        let (yp, _) = pad_even(y)?;
        let (xl, xh) = haar_dwt(&xp)?;
        let (yl, yh) = haar_dwt(&yp)?;
        let (xl, xh) = match self.mode {
            DcwMode::Low => (Self::correct_band(&xl, &yl, t * self.scaler)?, xh),
            DcwMode::High => (xl, Self::correct_band(&xh, &yh, (1.0 - t) * self.scaler)?),
            DcwMode::Double => (
                Self::correct_band(&xl, &yl, t * self.scaler)?,
                Self::correct_band(&xh, &yh, (1.0 - t) * self.high_scaler)?,
            ),
            DcwMode::Pix => unreachable!(),
        };
        let x_new = haar_idwt(&xl, &xh)?;
        Ok(x_new.narrow(1, 0, out_t)?.contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::{device::Device, dtype::DType};

    #[test]
    fn haar_roundtrip() {
        synaptix_kernels_cpu::ensure_registered();
        let x = Tensor::randn_seeded(vec![1usize, 8, 4], 7, Device::Cpu).unwrap();
        let (l, h) = haar_dwt(&x).unwrap();
        assert_eq!(l.dims(), &[1, 4, 4]);
        let r = haar_idwt(&l, &h).unwrap();
        let xv: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let rv: Vec<f32> = r.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in xv.iter().zip(rv.iter()) {
            assert!((a - b).abs() < 1e-4, "roundtrip {a} vs {b}");
        }
    }

    #[test]
    fn pix_and_disabled() {
        synaptix_kernels_cpu::ensure_registered();
        let x = Tensor::randn_seeded(vec![1usize, 6, 4], 1, Device::Cpu).unwrap();
        let y = Tensor::zeros(vec![1usize, 6, 4], DType::F32, Device::Cpu).unwrap();
        let off = DcwCorrector::default();
        assert!(!off.is_active());
        let same: Vec<f32> = off.apply(&x, &y, 0.5).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let xv: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(same, xv);
        let dbl = DcwCorrector { enabled: true, mode: DcwMode::Double, scaler: 0.05, high_scaler: 0.02 };
        let out = dbl.apply(&x, &y, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 6, 4]);
    }
}
