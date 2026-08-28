use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};

use crate::config::Qwen4ExpConfig;
use crate::ngram::NGramEmbedding;
use crate::norm::{coerr, group_rms, load_one_plus};

#[derive(Clone)]
pub struct PleState {
    pub tokens: Vec<u32>,
    pub conv: Vec<f32>,
}

impl PleState {
    pub fn new(context_len: usize, eos: u32, state_len: usize, width: usize) -> Self {
        Self {
            tokens: vec![eos; context_len],
            conv: vec![0.0; state_len * width],
        }
    }
}

pub struct PleLayer {
    embedding: NGramEmbedding,
    key_proj: QLinear,
    value_proj: QLinear,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    conv_w: Tensor,
    hidden: usize,
    hc_count: usize,
    conv_kernel: usize,
    dilation: usize,
    state_len: usize,
    eps: f32,
    device: Device,
    compute: DType,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        embedding: NGramEmbedding,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, ModelError> {
        let ple = cfg
            .ple
            .as_ref()
            .ok_or_else(|| ModelError::Build("PLE-слой без ple-конфига".into()))?;
        let lin = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{prefix}.{name}.weight");
            if let Some(prequant) = weights.quant(&key, device) {
                return Ok(QLinear::Quant(prequant?));
            }
            let w = weights.tensor(&key, device, if quant.is_quantized() { DType::F16 } else { compute })?;
            QLinear::build(w, quant, compute)
        };
        let hc_hidden = cfg.hc_hidden();
        let conv_w = weights
            .tensor(&format!("{prefix}.conv1d.weight"), device, compute)?
            .reshape(vec![hc_hidden, ple.conv_kernel_size])
            .map_err(|e| ModelError::Load(e.to_string()))?;

        Ok(Self {
            embedding,
            key_proj: lin("key_proj")?,
            value_proj: lin("value_proj")?,
            norm_key: load_one_plus(weights, &format!("{prefix}.norm_key.weight"), device, compute)?,
            norm_query: load_one_plus(weights, &format!("{prefix}.norm_query.weight"), device, compute)?,
            norm_conv: load_one_plus(weights, &format!("{prefix}.norm_conv.weight"), device, compute)?,
            conv_w,
            hidden: cfg.hidden_size,
            hc_count: cfg.hc_count,
            conv_kernel: ple.conv_kernel_size,
            dilation: ple.ngram_size,
            state_len: ple.conv_state_len(),
            eps: cfg.rms_norm_eps,
            device,
            compute,
        })
    }

    pub fn state(&self) -> PleState {
        PleState::new(
            self.embedding.context_len(),
            self.embedding.eos(),
            self.state_len,
            self.hc_count * self.hidden,
        )
    }

    pub fn conv_width(&self) -> usize {
        self.hc_count * self.hidden
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        tokens: &[u32],
        state: &mut PleState,
    ) -> Result<Tensor, ModelError> {
        let t = tokens.len();
        let width = self.conv_width();
        let mut history = state.tokens.clone();
        history.extend_from_slice(tokens);
        let embeddings = self.embedding.forward(&history, t)?;

        let key = self.key_proj.forward(&embeddings)?;
        let key = group_rms(&key, &self.norm_key, self.hidden, self.eps)?;
        let value = self.value_proj.forward(&embeddings)?;
        let query = group_rms(hidden, &self.norm_query, self.hidden, self.eps)?;

        let split = vec![t, self.hc_count, self.hidden];
        let key = coerr(key.reshape(split.clone()))?;
        let query = coerr(coerr(query.contiguous())?.reshape(split.clone()))?;
        let gate = coerr(coerr(key.mul(&query))?.sum_keepdim(2))?;
        let gate = coerr(gate.mul_scalar(1.0 / (self.hidden as f32).sqrt()))?;
        let mag = coerr(coerr(coerr(gate.abs())?.clamp(1e-6, f32::MAX))?.sqrt())?;
        let gate = coerr(mag.mul(&coerr(gate.sign())?))?;

        let value = coerr(coerr(value.contiguous())?.reshape(vec![t, 1, self.hidden]))?;
        let gated = coerr(coerr(gate.sigmoid())?.broadcast_mul(&value))?;
        let gated = coerr(coerr(gated.contiguous())?.reshape(vec![t, width]))?;
        let gated_normed = group_rms(&gated, &self.norm_conv, self.hidden, self.eps)?;

        let conv = self.short_conv(&gated_normed, state)?;
        state.tokens = {
            let ctx = self.embedding.context_len();
            let mut all = history;
            let keep = all.len().saturating_sub(ctx);
            all.drain(..keep);
            all
        };
        coerr(gated.add(&conv))
    }

    fn short_conv(&self, x: &Tensor, state: &mut PleState) -> Result<Tensor, ModelError> {
        let t = x.dims()[0];
        let width = self.conv_width();
        let state_t = Tensor::from_vec(state.conv.clone(), vec![self.state_len, width], self.device)
            .and_then(|s| s.to_dtype(self.compute))
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let padded = coerr(Tensor::cat(&[&state_t, x], 0))?;

        let mut acc: Option<Tensor> = None;
        for j in 0..self.conv_kernel {
            let offset = j * self.dilation;
            let slice = coerr(coerr(padded.narrow(0, offset, t))?.contiguous())?;
            let w = coerr(coerr(coerr(self.conv_w.narrow(1, j, 1))?.contiguous())?.reshape(vec![1, width]))?;
            let term = coerr(slice.broadcast_mul(&w))?;
            acc = Some(match acc {
                Some(a) => coerr(a.add(&term))?,
                None => term,
            });
        }
        let out = acc.ok_or_else(|| ModelError::Forward("PLE: пустое ядро свёртки".into()))?;

        let tail_start = padded.dims()[0].saturating_sub(self.state_len);
        let tail = coerr(coerr(padded.narrow(0, tail_start, self.state_len))?.contiguous())?;
        state.conv = tail
            .to_device(Device::Cpu)
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())
            .map_err(|e| ModelError::Forward(e.to_string()))?;

        coerr(out.silu())
    }
}
