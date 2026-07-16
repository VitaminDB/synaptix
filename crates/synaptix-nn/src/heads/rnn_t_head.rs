use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct RnnTHead {
    pub enc_proj: Linear,
    pub pred_proj: Linear,
    pub out: Linear,
    pub vocab_size: usize,
    pub joint_dim: usize,
}

impl RnnTHead {
    pub fn new(
        enc_dim: usize,
        pred_dim: usize,
        joint_dim: usize,
        vocab_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            enc_proj: Linear::from_init(enc_dim, joint_dim, true, InitMethod::KaimingUniform { fan_in: enc_dim, a: 0.0 }, InitMethod::Zeros, device, dtype, 0)?,
            pred_proj: Linear::from_init(pred_dim, joint_dim, true, InitMethod::KaimingUniform { fan_in: pred_dim, a: 0.0 }, InitMethod::Zeros, device, dtype, 1)?,
            out: Linear::from_init(joint_dim, vocab_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2)?,
            vocab_size,
            joint_dim,
        })
    }

    pub fn from_weights(
        enc_w: Tensor, enc_b: Option<Tensor>,
        pred_w: Tensor, pred_b: Option<Tensor>,
        out_w: Tensor, out_b: Option<Tensor>,
    ) -> Result<Self> {
        let enc_proj = Linear::new(enc_w, enc_b)?;
        let pred_proj = Linear::new(pred_w, pred_b)?;
        let out_layer = Linear::new(out_w, out_b)?;
        if enc_proj.out_features() != pred_proj.out_features() {
            return Err(SynaptixError::shape_mismatch(
                &[enc_proj.out_features()],
                &[pred_proj.out_features()],
            ));
        }
        let joint_dim = enc_proj.out_features();
        let vocab_size = out_layer.out_features();
        Ok(Self { enc_proj, pred_proj, out: out_layer, vocab_size, joint_dim })
    }

    pub fn forward(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor> {
        if enc.rank() != 3 || pred.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "rnn_t_head: enc/pred должны быть [B, T, *] / [B, U, *]",
            ));
        }
        let f = self.enc_proj.forward(enc)?;
        let g = self.pred_proj.forward(pred)?;
        let f_b = f.unsqueeze(2)?;
        let g_b = g.unsqueeze(1)?;
        let joint = f_b.broadcast_add(&g_b)?.tanh()?;
        self.out.forward(&joint)
    }

    pub fn forward_paired(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor> {
        let f = self.enc_proj.forward(enc)?;
        let g = self.pred_proj.forward(pred)?;
        let joint = f.broadcast_add(&g)?.tanh()?;
        self.out.forward(&joint)
    }
}
