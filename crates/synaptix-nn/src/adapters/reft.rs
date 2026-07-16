use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::parameter::Parameter;

/// LoReFT — Low-Rank Representation Fine-Tuning.
///
/// На hidden state `h ∈ [..., hidden]` накладывается аддитивная интервенция
/// в подпространстве, заданном `R ∈ [r, hidden]`:
///
/// `h' = h + Rᵀ · ((W · h + b) − R · h)`
///
/// Параметр `R` интерпретируется как (низкоранговая) проекция; `W ∈ [r, hidden]`
/// и `b ∈ [r]` — линейный source для нового представления в этом подпространстве.
pub struct ReftAdapter {
    pub r_proj: Parameter,
    pub w: Parameter,
    pub b: Parameter,
    pub hidden_size: usize,
    pub r: usize,
}

impl ReftAdapter {
    pub fn new(hidden_size: usize, r: usize, device: Device, dtype: DType) -> Result<Self> {
        let r_proj = crate::init::init_tensor(
            &[r, hidden_size],
            InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
            dtype, 0, device,
        )?;
        let w = crate::init::init_tensor(
            &[r, hidden_size],
            InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
            dtype, 1, device,
        )?;
        let b = crate::init::init_tensor(&[r], InitMethod::Zeros, dtype, 2, device)?;
        Ok(Self {
            r_proj: Parameter::new(r_proj),
            w: Parameter::new(w),
            b: Parameter::new(b),
            hidden_size,
            r,
        })
    }

    pub fn from_weights(r_proj: Tensor, w: Tensor, b: Tensor) -> Result<Self> {
        let r = r_proj.dims()[0];
        let hidden_size = r_proj.dims()[1];
        Ok(Self {
            r_proj: Parameter::new(r_proj),
            w: Parameter::new(w),
            b: Parameter::new(b),
            hidden_size,
            r,
        })
    }

    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let r_p = self.r_proj.tensor();
        let r_t = r_p.transpose(0, 1)?.contiguous()?;
        let rh = h.matmul(&r_t)?;
        let w_w = self.w.tensor();
        let w_t = w_w.transpose(0, 1)?.contiguous()?;
        let wh = h.matmul(&w_t)?;
        let wh_b = wh.broadcast_add(&self.b.tensor())?;
        let diff = wh_b.sub(&rh)?;
        let r_full = r_p.contiguous()?;
        let delta = diff.matmul(&r_full)?;
        h.add(&delta)
    }
}
