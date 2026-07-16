use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::parameter::Parameter;

pub struct RwkvBlock {
    pub time_mix_r: Parameter,
    pub time_mix_k: Parameter,
    pub time_mix_v: Parameter,
    pub key: Linear,
    pub value: Linear,
    pub receptance: Linear,
    pub output: Linear,
    pub hidden_size: usize,
}

impl RwkvBlock {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let tm_r = crate::init::init_tensor(&[hidden_size], InitMethod::Normal { mean: 0.0, std: 0.02 }, dtype, 0, device)?;
        let tm_k = crate::init::init_tensor(&[hidden_size], InitMethod::Normal { mean: 0.0, std: 0.02 }, dtype, 1, device)?;
        let tm_v = crate::init::init_tensor(&[hidden_size], InitMethod::Normal { mean: 0.0, std: 0.02 }, dtype, 2, device)?;
        Ok(Self {
            time_mix_r: Parameter::new(tm_r),
            time_mix_k: Parameter::new(tm_k),
            time_mix_v: Parameter::new(tm_v),
            key: Linear::from_init(hidden_size, hidden_size, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            value: Linear::from_init(hidden_size, hidden_size, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1)?,
            receptance: Linear::from_init(hidden_size, hidden_size, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2)?,
            output: Linear::from_init(hidden_size, hidden_size, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 3)?,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, L, H]
        use synaptix_ops::ssm::rwkv::{rwkv_time_mix, rwkv_wkv};
        use crate::module::Module;

        let dims = x.dims();
        let (b_sz, l, h) = (dims[0], dims[1], dims[2]);
        let tmr = self.time_mix_r.tensor();
        let tmk = self.time_mix_k.tensor();
        let tmv = self.time_mix_v.tensor();
        let mut ys: Vec<Tensor> = Vec::with_capacity(l);
        for t in 0..l {
            let xt = x.narrow(1, t, 1)?.squeeze(1)?;
            let x_prev = if t == 0 {
                Tensor::zeros(vec![b_sz, h], x.dtype(), x.device())?
            } else {
                x.narrow(1, t - 1, 1)?.squeeze(1)?
            };
            let mixed = rwkv_time_mix(&xt, &x_prev, &tmk, &tmv, &tmr)?; // [B, 3H]
            let xk = mixed.narrow(1, 0, h)?;
            let xv = mixed.narrow(1, h, h)?;
            let xr = mixed.narrow(1, h * 2, h)?;
            let k = Module::forward(&self.key, &xk)?.unsqueeze(1)?;
            let v = Module::forward(&self.value, &xv)?.unsqueeze(1)?;
            let r = Module::forward(&self.receptance, &xr)?.unsqueeze(1)?;
            let td = Tensor::zeros(vec![b_sz, h], x.dtype(), x.device())?;
            let tf = Tensor::zeros(vec![b_sz, h], x.dtype(), x.device())?;
            let y = rwkv_wkv(&k, &v, &r, &td, &tf)?.squeeze(1)?;
            ys.push(Module::forward(&self.output, &y)?.unsqueeze(1)?);
        }
        let refs: Vec<&Tensor> = ys.iter().collect();
        Tensor::cat(&refs, 1)
    }
}
