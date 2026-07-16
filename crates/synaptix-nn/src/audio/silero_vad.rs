use std::collections::BTreeMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::conv::conv1d;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::{Module, join_path};
use crate::parameter::Parameter;

#[derive(Debug, Clone, Copy)]
pub struct SileroVadConfig {
    pub spec_bins: usize,
    pub stem_channels: usize,
    pub hidden_size: usize,
    pub num_conv_blocks: usize,
    pub conv_kernel: usize,
}

impl Default for SileroVadConfig {
    fn default() -> Self {
        Self {
            spec_bins: 129,
            stem_channels: 64,
            hidden_size: 64,
            num_conv_blocks: 4,
            conv_kernel: 3,
        }
    }
}

pub struct LstmCell {
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias: Parameter,
    input_size: usize,
    hidden_size: usize,
}

impl LstmCell {
    pub fn new(weight_ih: Tensor, weight_hh: Tensor, bias: Tensor) -> Result<Self> {
        if weight_ih.rank() != 2 || weight_hh.rank() != 2 || bias.rank() != 1 {
            return Err(SynaptixError::Unsupported(
                "LstmCell: weight_ih [4H, I], weight_hh [4H, H], bias [4H]",
            ));
        }
        let four_h = weight_ih.dims()[0];
        if four_h % 4 != 0 {
            return Err(SynaptixError::Unsupported("LstmCell: out_features % 4 != 0"));
        }
        let hidden = four_h / 4;
        let input = weight_ih.dims()[1];
        if weight_hh.dims() != [four_h, hidden] || bias.dims() != [four_h] {
            return Err(SynaptixError::shape_mismatch(&[four_h, hidden], weight_hh.dims()));
        }
        Ok(Self {
            weight_ih: Parameter::new(weight_ih).with_name("weight_ih"),
            weight_hh: Parameter::new(weight_hh).with_name("weight_hh"),
            bias: Parameter::new(bias).with_name("bias"),
            input_size: input,
            hidden_size: hidden,
        })
    }

    pub fn from_init(
        input_size: usize,
        hidden_size: usize,
        device: Device,
        dtype: DType,
        seed: u64,
    ) -> Result<Self> {
        let four_h = 4 * hidden_size;
        let w_ih = crate::init::init_tensor(
            &[four_h, input_size],
            InitMethod::KaimingUniform { fan_in: input_size, a: 0.0 },
            dtype,
            seed,
            device,
        )?;
        let w_hh = crate::init::init_tensor(
            &[four_h, hidden_size],
            InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
            dtype,
            seed.wrapping_add(1),
            device,
        )?;
        let bias = Tensor::zeros(vec![four_h], dtype, device)?;
        Self::new(w_ih, w_hh, bias)
    }

    pub fn input_size(&self) -> usize { self.input_size }
    pub fn hidden_size(&self) -> usize { self.hidden_size }

    pub fn step(
        &self,
        x: &Tensor,
        h_prev: &Tensor,
        c_prev: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if x.rank() != 2 || x.dims()[1] != self.input_size {
            return Err(SynaptixError::shape_mismatch(
                &[x.dims().first().copied().unwrap_or(0), self.input_size],
                x.dims(),
            ));
        }
        if h_prev.rank() != 2 || h_prev.dims()[1] != self.hidden_size
            || c_prev.rank() != 2 || c_prev.dims()[1] != self.hidden_size
        {
            return Err(SynaptixError::Unsupported(
                "LstmCell::step: h/c must be [B, hidden_size]",
            ));
        }

        let w_ih_t = self.weight_ih.tensor().transpose(0, 1)?.contiguous()?;
        let w_hh_t = self.weight_hh.tensor().transpose(0, 1)?.contiguous()?;

        let gx = x.matmul(&w_ih_t)?;
        let gh = h_prev.matmul(&w_hh_t)?;
        let gates_pre = gx.add(&gh)?.broadcast_add(&self.bias.tensor())?;

        let h = self.hidden_size;
        let i_gate = gates_pre.narrow(1, 0, h)?.sigmoid()?;
        let f_gate = gates_pre.narrow(1, h, h)?.sigmoid()?;
        let g_gate = gates_pre.narrow(1, 2 * h, h)?.tanh()?;
        let o_gate = gates_pre.narrow(1, 3 * h, h)?.sigmoid()?;

        let c_new = f_gate.mul(c_prev)?.add(&i_gate.mul(&g_gate)?)?;
        let h_new = o_gate.mul(&c_new.tanh()?)?;
        Ok((h_new, c_new))
    }

    pub fn zero_state(&self, batch: usize, device: Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let h = Tensor::zeros(vec![batch, self.hidden_size], dtype, device)?;
        let c = Tensor::zeros(vec![batch, self.hidden_size], dtype, device)?;
        Ok((h, c))
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight_ih, &self.weight_hh, &self.bias]
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        vec![
            (join_path(prefix, "weight_ih"), &self.weight_ih),
            (join_path(prefix, "weight_hh"), &self.weight_hh),
            (join_path(prefix, "bias"), &self.bias),
        ]
    }
}

pub struct ConvBlock {
    pub weight: Parameter,
    pub bias: Parameter,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub padding: usize,
}

impl ConvBlock {
    pub fn new(weight: Tensor, bias: Tensor) -> Result<Self> {
        if weight.rank() != 3 {
            return Err(SynaptixError::Unsupported("ConvBlock: weight must be [C_out, C_in, K]"));
        }
        let dims = weight.dims().to_vec();
        let (out_channels, in_channels, kernel) = (dims[0], dims[1], dims[2]);
        if bias.rank() != 1 || bias.dims()[0] != out_channels {
            return Err(SynaptixError::shape_mismatch(&[out_channels], bias.dims()));
        }
        Ok(Self {
            weight: Parameter::new(weight).with_name("weight"),
            bias: Parameter::new(bias).with_name("bias"),
            in_channels,
            out_channels,
            kernel,
            padding: kernel / 2,
        })
    }

    pub fn from_init(
        in_channels: usize, out_channels: usize, kernel: usize,
        device: Device, dtype: DType, seed: u64,
    ) -> Result<Self> {
        let w = crate::init::init_tensor(
            &[out_channels, in_channels, kernel],
            InitMethod::KaimingUniform { fan_in: in_channels * kernel, a: 0.0 },
            dtype, seed, device,
        )?;
        let b = Tensor::zeros(vec![out_channels], dtype, device)?;
        Self::new(w, b)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = conv1d(x, &self.weight.tensor(), Some(&self.bias.tensor()), 1, self.padding)?;
        y.relu()
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        vec![
            (join_path(prefix, "weight"), &self.weight),
            (join_path(prefix, "bias"), &self.bias),
        ]
    }
}

pub struct SileroVadModel {
    pub config: SileroVadConfig,
    pub stem: ConvBlock,
    pub blocks: Vec<ConvBlock>,
    pub lstm: LstmCell,
    pub head: Linear,
}

impl SileroVadModel {
    pub fn new(config: SileroVadConfig, device: Device, dtype: DType) -> Result<Self> {
        let stem = ConvBlock::from_init(
            config.spec_bins, config.stem_channels, config.conv_kernel,
            device, dtype, 0,
        )?;
        let mut blocks = Vec::with_capacity(config.num_conv_blocks);
        for i in 0..config.num_conv_blocks {
            blocks.push(ConvBlock::from_init(
                config.stem_channels, config.stem_channels, config.conv_kernel,
                device, dtype, (10 + i as u64) * 7,
            )?);
        }
        let lstm = LstmCell::from_init(
            config.stem_channels, config.hidden_size, device, dtype, 1001,
        )?;
        let head = Linear::from_init(
            config.hidden_size, 1, true,
            InitMethod::KaimingUniform { fan_in: config.hidden_size, a: 0.0 },
            InitMethod::Zeros, device, dtype, 2002,
        )?;
        Ok(Self { config, stem, blocks, lstm, head })
    }

    pub fn zero_state(&self, batch: usize, device: Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        self.lstm.zero_state(batch, device, dtype)
    }

    pub fn forward_spec(
        &self,
        spec: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if spec.rank() != 3 || spec.dims()[1] != self.config.spec_bins {
            return Err(SynaptixError::shape_mismatch(
                &[
                    spec.dims().first().copied().unwrap_or(0),
                    self.config.spec_bins,
                    spec.dims().last().copied().unwrap_or(0),
                ],
                spec.dims(),
            ));
        }

        let mut feat = self.stem.forward(spec)?;
        for block in &self.blocks {
            let y = block.forward(&feat)?;
            feat = feat.add(&y)?;
        }

        let batch = feat.dims()[0];
        let chan = feat.dims()[1];
        let t = feat.dims()[2];
        let feat_t = feat.permute(vec![0, 2, 1])?.contiguous()?;
        let feat_flat = feat_t.reshape(vec![batch * t, chan])?;

        let h_per_step = h.clone();
        let c_per_step = c.clone();

        let mut h_state = h_per_step;
        let mut c_state = c_per_step;
        let mut logits_per_t: Vec<Tensor> = Vec::with_capacity(t);
        for ti in 0..t {
            let xi = feat_flat.narrow(0, ti * batch, batch)?.contiguous()?;
            let (h_new, c_new) = self.lstm.step(&xi, &h_state, &c_state)?;
            let l = self.head.forward(&h_new)?;
            logits_per_t.push(l);
            h_state = h_new;
            c_state = c_new;
        }
        let refs: Vec<&Tensor> = logits_per_t.iter().collect();
        let stacked = Tensor::cat(&refs, 0)?;
        let logits = stacked.reshape(vec![batch, t])?;
        Ok((logits, h_state, c_state))
    }

    pub fn forward_last(
        &self,
        spec: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (logits, h_new, c_new) = self.forward_spec(spec, h, c)?;
        let t = logits.dims()[1];
        if t == 0 {
            return Err(SynaptixError::Unsupported(
                "SileroVadModel::forward_last: time dimension is zero",
            ));
        }
        let last = logits.narrow(1, t - 1, 1)?.squeeze(1)?;
        let prob = last.sigmoid()?;
        Ok((prob, h_new, c_new))
    }
}

impl Module for SileroVadModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dims().first().copied().unwrap_or(0);
        let device = x.device();
        let dtype = x.dtype();
        let (h0, c0) = self.zero_state(batch, device, dtype)?;
        let (prob, _, _) = self.forward_last(x, &h0, &c0)?;
        Ok(prob)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut out = Vec::new();
        out.push(&self.stem.weight);
        out.push(&self.stem.bias);
        for b in &self.blocks {
            out.push(&b.weight);
            out.push(&b.bias);
        }
        out.extend(self.lstm.parameters());
        out.extend(self.head.parameters());
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut out = Vec::new();
        let stem_prefix = join_path(prefix, "stem");
        out.extend(self.stem.named_parameters(&stem_prefix));
        for (i, b) in self.blocks.iter().enumerate() {
            let p = join_path(prefix, &format!("blocks.{i}"));
            out.extend(b.named_parameters(&p));
        }
        out.extend(self.lstm.named_parameters(&join_path(prefix, "lstm")));
        out.extend(self.head.named_parameters(&join_path(prefix, "head")));
        out
    }

    fn load_state_dict(&self, dict: &BTreeMap<String, Tensor>) -> Result<()> {
        for (name, param) in self.named_parameters("") {
            let value = dict.get(&name).ok_or_else(|| {
                SynaptixError::Other(format!("SileroVadModel::load_state_dict: missing key '{name}'"))
            })?;
            param.set(value.clone())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_kernels_cpu::ensure_registered;

    fn cfg_small() -> SileroVadConfig {
        SileroVadConfig {
            spec_bins: 32,
            stem_channels: 16,
            hidden_size: 16,
            num_conv_blocks: 2,
            conv_kernel: 3,
        }
    }

    fn flat_f32(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn lstm_step_shape_and_finite() {
        ensure_registered();
        let cell = LstmCell::from_init(8, 4, Device::Cpu, DType::F32, 42).unwrap();
        let x = Tensor::ones(vec![2, 8], DType::F32, Device::Cpu).unwrap();
        let (h0, c0) = cell.zero_state(2, Device::Cpu, DType::F32).unwrap();
        let (h1, c1) = cell.step(&x, &h0, &c0).unwrap();
        assert_eq!(h1.dims(), &[2, 4]);
        assert_eq!(c1.dims(), &[2, 4]);
        let h_data = flat_f32(&h1);
        let c_data = flat_f32(&c1);
        assert!(h_data.iter().all(|v| v.is_finite()));
        assert!(c_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lstm_state_is_persistent() {
        ensure_registered();
        let cell = LstmCell::from_init(4, 4, Device::Cpu, DType::F32, 123).unwrap();
        let x = Tensor::ones(vec![1, 4], DType::F32, Device::Cpu).unwrap();
        let (h0, c0) = cell.zero_state(1, Device::Cpu, DType::F32).unwrap();
        let (h1, c1) = cell.step(&x, &h0, &c0).unwrap();
        let (h2, _) = cell.step(&x, &h1, &c1).unwrap();
        let v0 = flat_f32(&h0);
        let v1 = flat_f32(&h1);
        let v2 = flat_f32(&h2);
        let d01: f32 = v0.iter().zip(&v1).map(|(a, b)| (a - b).abs()).sum();
        let d12: f32 = v1.iter().zip(&v2).map(|(a, b)| (a - b).abs()).sum();
        assert!(d01 > 1e-4, "first step should change hidden state");
        assert!(d12 > 1e-6, "second step should depend on prior state");
    }

    #[test]
    fn silero_vad_forward_last_shape() {
        ensure_registered();
        let cfg = cfg_small();
        let m = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        let (h, c) = m.zero_state(1, Device::Cpu, DType::F32).unwrap();
        let spec = Tensor::ones(vec![1, cfg.spec_bins, 6], DType::F32, Device::Cpu).unwrap();
        let (prob, h_new, c_new) = m.forward_last(&spec, &h, &c).unwrap();
        assert_eq!(prob.dims(), &[1]);
        assert_eq!(h_new.dims(), &[1, cfg.hidden_size]);
        assert_eq!(c_new.dims(), &[1, cfg.hidden_size]);
        let p = flat_f32(&prob)[0];
        assert!(p.is_finite() && (0.0..=1.0).contains(&p), "prob must be in [0,1], got {p}");
    }

    #[test]
    fn silero_vad_state_dict_round_trip() {
        ensure_registered();
        let cfg = cfg_small();
        let m1 = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        let sd = m1.state_dict();
        assert!(sd.contains_key("stem.weight"));
        assert!(sd.contains_key("stem.bias"));
        assert!(sd.contains_key("blocks.0.weight"));
        assert!(sd.contains_key(&format!("blocks.{}.weight", cfg.num_conv_blocks - 1)));
        assert!(sd.contains_key("lstm.weight_ih"));
        assert!(sd.contains_key("lstm.weight_hh"));
        assert!(sd.contains_key("lstm.bias"));
        assert!(sd.contains_key("head.weight"));

        let m2 = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        m2.load_state_dict(&sd).unwrap();
        let (h, c) = m1.zero_state(1, Device::Cpu, DType::F32).unwrap();
        let spec = Tensor::ones(vec![1, cfg.spec_bins, 4], DType::F32, Device::Cpu).unwrap();
        let (p1, _, _) = m1.forward_last(&spec, &h, &c).unwrap();
        let (p2, _, _) = m2.forward_last(&spec, &h, &c).unwrap();
        let v1 = flat_f32(&p1);
        let v2 = flat_f32(&p2);
        let diff: f32 = v1.iter().zip(&v2).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-5, "state-dict round-trip should be bit-exact, got diff={diff}");
    }
}
