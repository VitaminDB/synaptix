use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::attention::linear::{
    gated_delta_decay_beta, gated_delta_net_recurrent, GatedDeltaNetState,
};
use synaptix_ops::conv::causal_conv1d::causal_conv1d_stateful;
use synaptix_ops::norm::rms_norm::rms_norm;

use crate::config::Qwen4ExpConfig;
use crate::norm::coerr;

const CHUNK: usize = 64;

pub struct LinearAttn {
    in_proj_qkv: QLinear,
    in_proj_a: QLinear,
    in_proj_b: QLinear,
    in_proj_z: QLinear,
    out_proj: QLinear,
    conv_w: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_weight: Tensor,
    conv_w_dev: Option<Tensor>,
    a_log_dev: Option<Tensor>,
    dt_bias_dev: Option<Tensor>,
    num_k_heads: usize,
    num_v_heads: usize,
    dk: usize,
    dv: usize,
    conv_k: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    q_scale: f32,
    rms_eps: f32,
    gate_sigmoid: bool,
    device: Device,
    compute: DType,
}

impl LinearAttn {
    pub fn load(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, ModelError> {
        let lc = &cfg.linear;
        let lin = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{prefix}.{name}.weight");
            if let Some(prequant) = weights.quant(&key, device) {
                return Ok(QLinear::Quant(prequant?));
            }
            let w = weights.tensor(&key, device, if quant.is_quantized() { DType::F16 } else { compute })?;
            QLinear::build(w, quant, compute)
        };
        let host_f32 = |name: &str| -> Result<Vec<f32>, ModelError> {
            weights
                .tensor(&format!("{prefix}.{name}"), Device::Cpu, DType::F32)?
                .flatten_all()
                .and_then(|x| x.to_vec1::<f32>())
                .map_err(|e| ModelError::Load(e.to_string()))
        };

        let conv_w = host_f32("conv1d.weight")?;
        let a_log = host_f32("A_log")?;
        let dt_bias = host_f32("dt_bias")?;
        let norm_weight = weights.tensor(&format!("{prefix}.norm.weight"), device, compute)?;
        let (conv_dim, ck, nv) = (lc.conv_dim(), lc.conv_kernel, lc.num_value_heads);
        let (conv_w_dev, a_log_dev, dt_bias_dev) = if device.is_cpu() {
            (None, None, None)
        } else {
            (
                Some(coerr(coerr(Tensor::from_vec(conv_w.clone(), vec![conv_dim, ck], device))?
                    .to_dtype(compute))?),
                Some(coerr(Tensor::from_vec(a_log.clone(), vec![nv], device))?),
                Some(coerr(Tensor::from_vec(dt_bias.clone(), vec![nv], device))?),
            )
        };

        Ok(Self {
            in_proj_qkv: lin("in_proj_qkv")?,
            in_proj_a: lin("in_proj_a")?,
            in_proj_b: lin("in_proj_b")?,
            in_proj_z: lin("in_proj_z")?,
            out_proj: lin("out_proj")?,
            conv_w,
            a_log,
            dt_bias,
            norm_weight,
            conv_w_dev,
            a_log_dev,
            dt_bias_dev,
            num_k_heads: lc.num_key_heads,
            num_v_heads: lc.num_value_heads,
            dk: lc.key_head_dim,
            dv: lc.value_head_dim,
            conv_k: lc.conv_kernel,
            key_dim: lc.key_dim(),
            value_dim: lc.value_dim(),
            conv_dim: lc.conv_dim(),
            q_scale: 1.0 / (lc.key_head_dim as f32).sqrt(),
            rms_eps: cfg.rms_norm_eps,
            gate_sigmoid: cfg.output_gate_sigmoid,
            device,
            compute,
        })
    }

    pub fn make_state(&self) -> GatedDeltaNetState {
        GatedDeltaNetState {
            conv_state: vec![0.0; (self.conv_k - 1) * self.conv_dim],
            ssm_state: vec![0.0; self.num_v_heads * self.dk * self.dv],
            conv_state_dev: None,
            ssm_state_dev: None,
        }
    }

    fn gate(&self, core: &Tensor, z: &Tensor) -> Result<Tensor, ModelError> {
        let normed = coerr(rms_norm(core, &self.norm_weight, self.rms_eps))?;
        let act = if self.gate_sigmoid {
            coerr(z.sigmoid())?
        } else {
            coerr(z.silu())?
        };
        coerr(normed.mul(&act))
    }

    pub fn forward(
        &self,
        h: &Tensor,
        state: &mut GatedDeltaNetState,
        s: usize,
    ) -> Result<Tensor, ModelError> {
        let core = match self.core_on_device(h, state, s) {
            Some(Ok(core)) => core,
            Some(Err(e)) => return Err(e),
            None => self.core_host(h, state, s)?,
        };
        let z = coerr(self
            .in_proj_z
            .forward(h)?
            .reshape(vec![1, s, self.num_v_heads, self.dv]))?;
        let normed = self.gate(&core, &z)?;
        let normed = coerr(normed.reshape(vec![s, self.value_dim]))?;
        self.out_proj.forward(&normed)
    }

    /// CUDA-путь, если он применим к этой длине чанка: фьюз
    /// `conv1d + prep + chunk_gated_delta_rule` требует кратности чанку скана,
    /// иначе считаем host-цепочкой.
    fn core_on_device(
        &self,
        h: &Tensor,
        state: &mut GatedDeltaNetState,
        s: usize,
    ) -> Option<Result<Tensor, ModelError>> {
        if !matches!(self.device, Device::Cuda(_)) || self.conv_w_dev.is_none() {
            return None;
        }
        match self.core_device(h, state, s) {
            Ok(core) => Some(Ok(core)),
            Err(ModelError::Forward(msg)) if unsupported(&msg) => {
                state.conv_state_dev = None;
                state.ssm_state_dev = None;
                None
            }
            Err(e) => Some(Err(e)),
        }
    }

    fn core_host(
        &self,
        h: &Tensor,
        state: &mut GatedDeltaNetState,
        s: usize,
    ) -> Result<Tensor, ModelError> {
        let (dk, dv, h_v, h_k) = (self.dk, self.dv, self.num_v_heads, self.num_k_heads);
        let group = h_v / h_k;
        let qkv = self.in_proj_qkv.forward(h)?;
        let qkv_v = host_vec(&qkv)?;
        let mut conv_out = causal_conv1d_stateful(
            &mut state.conv_state,
            &qkv_v,
            &self.conv_w,
            s,
            self.conv_dim,
            self.conv_k,
        );
        for x in conv_out.iter_mut() {
            *x /= 1.0 + (-*x).exp();
        }
        let a_v = host_vec(&self.in_proj_a.forward(h)?)?;
        let b_v = host_vec(&self.in_proj_b.forward(h)?)?;
        let (g, beta) = gated_delta_decay_beta(&a_v, &b_v, &self.a_log, &self.dt_bias, s, h_v);

        let mut qe = vec![0.0f32; h_v * s * dk];
        let mut ke = vec![0.0f32; h_v * s * dk];
        let mut vv = vec![0.0f32; h_v * s * dv];
        let v_off0 = self.key_dim * 2;
        for hi in 0..h_v {
            let kh = hi / group;
            for t in 0..s {
                let row = t * self.conv_dim;
                let qsrc = row + kh * dk;
                let ksrc = row + self.key_dim + kh * dk;
                let vsrc = row + v_off0 + hi * dv;
                let qdst = (hi * s + t) * dk;
                let vdst = (hi * s + t) * dv;
                qe[qdst..qdst + dk].copy_from_slice(&conv_out[qsrc..qsrc + dk]);
                ke[qdst..qdst + dk].copy_from_slice(&conv_out[ksrc..ksrc + dk]);
                vv[vdst..vdst + dv].copy_from_slice(&conv_out[vsrc..vsrc + dv]);
            }
        }
        let core = gated_delta_net_recurrent(
            &mut state.ssm_state,
            &qe,
            &ke,
            &vv,
            &g,
            &beta,
            h_v,
            s,
            dk,
            dv,
            self.q_scale,
        );
        let mut shaped = vec![0.0f32; s * h_v * dv];
        for hi in 0..h_v {
            for t in 0..s {
                let src = (hi * s + t) * dv;
                let dst = (t * h_v + hi) * dv;
                shaped[dst..dst + dv].copy_from_slice(&core[src..src + dv]);
            }
        }
        coerr(coerr(Tensor::from_vec(shaped, vec![1, s, h_v, dv], self.device))?.to_dtype(self.compute))
    }

    fn core_device(
        &self,
        h: &Tensor,
        state: &mut GatedDeltaNetState,
        s: usize,
    ) -> Result<Tensor, ModelError> {
        let (dk, dv, h_v) = (self.dk, self.dv, self.num_v_heads);
        let qkv = self.in_proj_qkv.forward(h)?;
        let qkv = coerr(qkv.reshape(vec![1, s, self.conv_dim]))?;
        let a = self.in_proj_a.forward(h)?;
        let b = self.in_proj_b.forward(h)?;
        let a = coerr(a.to_dtype(DType::F16))?;
        let b = coerr(b.to_dtype(DType::F16))?;
        let conv_w = self.conv_w_dev.as_ref().unwrap();
        let a_log = self.a_log_dev.as_ref().unwrap();
        let dt_bias = self.dt_bias_dev.as_ref().unwrap();

        if state.conv_state_dev.is_none() {
            let cs = coerr(coerr(Tensor::from_vec(
                state.conv_state.clone(),
                vec![self.conv_k - 1, self.conv_dim],
                self.device,
            ))?
            .to_dtype(self.compute))?;
            state.conv_state_dev = Some(cs);
        }
        if state.ssm_state_dev.is_none() {
            let ss = coerr(Tensor::from_vec(
                state.ssm_state.clone(),
                vec![h_v, dk, dv],
                self.device,
            ))?;
            state.ssm_state_dev = Some(ss);
        }
        let conv_w_c;
        let conv_w = if conv_w.dtype() == self.compute {
            conv_w
        } else {
            conv_w_c = coerr(conv_w.to_dtype(self.compute))?;
            &conv_w_c
        };
        let out = {
            let cs = state.conv_state_dev.as_mut().unwrap();
            let ss = state.ssm_state_dev.as_mut().unwrap();
            coerr(qkv.linear_attn_chunk_prefill(
                conv_w,
                &a,
                &b,
                dt_bias,
                a_log,
                cs,
                ss,
                self.num_k_heads,
                h_v,
                dk,
                dv,
                self.conv_k,
                CHUNK,
                self.q_scale,
                true,
            ))?
        };
        coerr(coerr(coerr(coerr(out.transpose(0, 1))?.contiguous())?
            .reshape(vec![1, s, h_v, dv]))?
            .to_dtype(self.compute))
    }
}

fn unsupported(msg: &str) -> bool {
    let m = msg.to_lowercase();
    m.contains("unsupported") || m.contains("не поддерж") || m.contains("chunk")
}

fn host_vec(t: &Tensor) -> Result<Vec<f32>, ModelError> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .map_err(|e| ModelError::Forward(e.to_string()))
}
