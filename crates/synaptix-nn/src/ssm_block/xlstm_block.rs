use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XLstmKind {
    SLstm,
    MLstm,
}

pub struct XLstmBlock {
    pub gate_proj: Linear,
    pub out_proj: Linear,
    pub hidden_size: usize,
    pub kind: XLstmKind,
}

impl XLstmBlock {
    pub fn new(hidden_size: usize, kind: XLstmKind, device: Device, dtype: DType) -> Result<Self> {
        let gate_out = match kind {
            XLstmKind::SLstm => 4 * hidden_size,
            XLstmKind::MLstm => 3 * hidden_size,
        };
        Ok(Self {
            gate_proj: Linear::from_init(
                hidden_size, gate_out, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            out_proj: Linear::from_init(
                hidden_size, hidden_size, false,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            hidden_size,
            kind,
        })
    }

    pub fn from_weights(
        gate_w: Tensor, gate_b: Option<Tensor>,
        out_w: Tensor, out_b: Option<Tensor>,
        kind: XLstmKind,
    ) -> Result<Self> {
        let gate_proj = Linear::new(gate_w, gate_b)?;
        let out_proj = Linear::new(out_w, out_b)?;
        let factor = match kind {
            XLstmKind::SLstm => 4,
            XLstmKind::MLstm => 3,
        };
        let hidden_size = gate_proj.in_features();
        if gate_proj.out_features() != factor * hidden_size {
            return Err(SynaptixError::shape_mismatch(
                &[factor * hidden_size],
                &[gate_proj.out_features()],
            ));
        }
        if out_proj.in_features() != hidden_size || out_proj.out_features() != hidden_size {
            return Err(SynaptixError::shape_mismatch(
                &[hidden_size, hidden_size],
                &[out_proj.in_features(), out_proj.out_features()],
            ));
        }
        Ok(Self { gate_proj, out_proj, hidden_size, kind })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use synaptix_ops::ssm::xlstm::{mlstm_step, slstm_step};

        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported("XLstmBlock: x must be [B, L, hidden]"));
        }
        let dims = x.dims();
        let (b_sz, l, h) = (dims[0], dims[1], dims[2]);
        if h != self.hidden_size {
            return Err(SynaptixError::shape_mismatch(&[self.hidden_size], &[h]));
        }
        let gates = self.gate_proj.forward(x)?;

        let mut state_h = Tensor::zeros(vec![b_sz, h], x.dtype(), x.device())?;
        let mut state_c = match self.kind {
            XLstmKind::SLstm => Tensor::zeros(vec![b_sz, h], x.dtype(), x.device())?,
            XLstmKind::MLstm => Tensor::zeros(vec![b_sz, h * h], x.dtype(), x.device())?,
        };

        let mut ys: Vec<Tensor> = Vec::with_capacity(l);
        for t in 0..l {
            let gate_t = gates.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let out_step = match self.kind {
                XLstmKind::SLstm => slstm_step(&gate_t, &state_h, &state_c)?,
                XLstmKind::MLstm => mlstm_step(&gate_t, &state_h, &state_c)?,
            };
            match self.kind {
                XLstmKind::SLstm => {
                    let z_act = gate_t.narrow(1, 0, h)?.contiguous()?.tanh()?;
                    let i_gate = gate_t.narrow(1, h, h)?.contiguous()?.sigmoid()?;
                    let f_gate = gate_t.narrow(1, 2 * h, h)?.contiguous()?.sigmoid()?;
                    state_c = f_gate.broadcast_mul(&state_c)?.add(&i_gate.broadcast_mul(&z_act)?)?;
                    state_h = out_step;
                }
                XLstmKind::MLstm => {
                    let k = gate_t.narrow(1, h, h)?.contiguous()?;
                    let v = gate_t.narrow(1, 2 * h, h)?.contiguous()?;
                    let c_mat = state_c.reshape(vec![b_sz, h, h])?;
                    let outer = v.unsqueeze(2)?.broadcast_mul(&k.unsqueeze(1)?)?;
                    let c_new = c_mat.add(&outer)?;
                    state_c = c_new.reshape(vec![b_sz, h * h])?;
                    state_h = state_h.add(&k)?;
                    let _ = out_step.clone();
                    ys.push(out_step.unsqueeze(1)?);
                    continue;
                }
            }
            ys.push(state_h.unsqueeze(1)?);
        }
        let refs: Vec<&Tensor> = ys.iter().collect();
        let y_seq = Tensor::cat(&refs, 1)?;
        self.out_proj.forward(&y_seq)
    }
}
