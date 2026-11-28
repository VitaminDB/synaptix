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
    /// Состояние свёртки на карте (`[state_len, width]` в `compute`): убирает
    /// H2D-загрузку и синхронную D2H-выгрузку на каждом шаге декода и префилла.
    pub conv_dev: Option<Tensor>,
    /// Чем заполнен контекст n-грамм до первого токена — к этому `reset`.
    eos: u32,
    /// Размер свёртки на хосте (`conv` бывает пуст, пока истина на карте).
    conv_len: usize,
}

impl PleState {
    pub fn new(context_len: usize, eos: u32, state_len: usize, width: usize) -> Self {
        Self {
            tokens: vec![eos; context_len],
            conv: vec![0.0; state_len * width],
            conv_dev: None,
            eos,
            conv_len: state_len * width,
        }
    }

    pub fn snapshot(&self) -> Result<Self, ModelError> {
        let conv_dev = match &self.conv_dev {
            Some(src) => {
                let mut dst = coerr(Tensor::empty_uninit(
                    src.dims().to_vec(),
                    src.dtype(),
                    src.device(),
                ))?;
                coerr(dst.copy_from(src))?;
                Some(dst)
            }
            None => None,
        };
        let conv = if conv_dev.is_some() { Vec::new() } else { self.conv.clone() };
        Ok(Self { tokens: self.tokens.clone(), conv, conv_dev, eos: self.eos, conv_len: self.conv_len })
    }

    pub fn restore_from(&mut self, snap: &Self) -> Result<(), ModelError> {
        self.tokens.clone_from(&snap.tokens);
        match (&mut self.conv_dev, &snap.conv_dev) {
            (Some(dst), Some(src)) => coerr(dst.copy_from(src))?,
            (None, Some(src)) => {
                let mut dst = coerr(Tensor::empty_uninit(
                    src.dims().to_vec(),
                    src.dtype(),
                    src.device(),
                ))?;
                coerr(dst.copy_from(src))?;
                self.conv_dev = Some(dst);
            }
            (_, None) => {
                self.conv.clone_from(&snap.conv);
                self.conv_dev = None;
            }
        }
        Ok(())
    }

    pub fn park_to_host(&mut self) -> usize {
        let Some(t) = self.conv_dev.take() else { return 0 };
        let bytes = t.dtype().bytes_for_numel(t.numel());
        if let Ok(v) = t
            .to_device(Device::Cpu)
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())
        {
            self.conv = v;
        }
        bytes
    }

    pub fn device_bytes(&self) -> usize {
        self.conv_dev
            .as_ref()
            .map(|t| t.dtype().bytes_for_numel(t.numel()))
            .unwrap_or(0)
    }

    /// Состояние до первого токена. Контекст n-грамм — тоже: иначе полный
    /// префилл по старой сессии видел последние токены прошлого хода перед
    /// своим первым (ответ расходился со свежей сессией).
    pub fn reset(&mut self) {
        self.tokens.fill(self.eos);
        self.conv.clear();
        self.conv.resize(self.conv_len, 0.0);
        self.conv_dev = None;
    }
}

pub struct PleLayer {
    embedding: NGramEmbedding,
    key_proj: QLinear,
    value_proj: QLinear,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    conv_taps: Vec<Tensor>,
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
        let mut conv_taps = Vec::with_capacity(ple.conv_kernel_size);
        for j in 0..ple.conv_kernel_size {
            let tap = conv_w
                .narrow(1, j, 1)
                .and_then(|t| t.contiguous())
                .and_then(|t| t.reshape(vec![1, hc_hidden]))
                .map_err(|e| ModelError::Load(e.to_string()))?;
            conv_taps.push(tap);
        }

        Ok(Self {
            embedding,
            key_proj: lin("key_proj")?,
            value_proj: lin("value_proj")?,
            norm_key: load_one_plus(weights, &format!("{prefix}.norm_key.weight"), device, compute)?,
            norm_query: load_one_plus(weights, &format!("{prefix}.norm_query.weight"), device, compute)?,
            norm_conv: load_one_plus(weights, &format!("{prefix}.norm_conv.weight"), device, compute)?,
            conv_taps,
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
        let state_t = if self.device != Device::Cpu {
            if state.conv_dev.is_none() {
                let s = Tensor::from_vec(state.conv.clone(), vec![self.state_len, width], self.device)
                    .and_then(|s| s.to_dtype(self.compute))
                    .map_err(|e| ModelError::Forward(e.to_string()))?;
                state.conv_dev = Some(s);
            }
            state.conv_dev.as_ref().expect("инициализирован выше").clone()
        } else {
            Tensor::from_vec(state.conv.clone(), vec![self.state_len, width], self.device)
                .and_then(|s| s.to_dtype(self.compute))
                .map_err(|e| ModelError::Forward(e.to_string()))?
        };
        let padded = coerr(Tensor::cat(&[&state_t, x], 0))?;

        let mut acc: Option<Tensor> = None;
        for j in 0..self.conv_kernel {
            let offset = j * self.dilation;
            let slice = coerr(coerr(padded.narrow(0, offset, t))?.contiguous())?;
            let term = coerr(slice.broadcast_mul(&self.conv_taps[j]))?;
            acc = Some(match acc {
                Some(a) => coerr(a.add(&term))?,
                None => term,
            });
        }
        let out = acc.ok_or_else(|| ModelError::Forward("PLE: пустое ядро свёртки".into()))?;

        let tail_start = padded.dims()[0].saturating_sub(self.state_len);
        let tail = coerr(coerr(padded.narrow(0, tail_start, self.state_len))?.contiguous())?;
        if self.device != Device::Cpu {
            state.conv_dev = Some(tail);
        } else {
            state.conv = tail
                .to_device(Device::Cpu)
                .and_then(|x| x.to_dtype(DType::F32))
                .and_then(|x| x.flatten_all())
                .and_then(|x| x.to_vec1::<f32>())
                .map_err(|e| ModelError::Forward(e.to_string()))?;
        }

        coerr(out.silu())
    }
}
