use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// P-Tuning v2: обучаемые префикс-эмбеддинги `embeddings ∈ [prefix_len, hidden]`
/// прогоняются через MLP-репараметризацию и развёртываются в per-layer
/// (K, V) префикс-тензоры.
///
/// Выход `forward` имеет форму `[num_layers, 2, prefix_len, hidden]`:
/// `[layer, 0, :, :]` — ключи, `[layer, 1, :, :]` — значения.
pub struct PTuningV2 {
    pub embeddings: Parameter,
    pub reparameterize: Linear,
    pub prefix_len: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
}

impl PTuningV2 {
    pub fn new(
        prefix_len: usize,
        num_layers: usize,
        hidden_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let emb = crate::init::init_tensor(
            &[prefix_len, hidden_size],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype, 0, device,
        )?;
        Ok(Self {
            embeddings: Parameter::new(emb),
            reparameterize: Linear::from_init(
                hidden_size,
                num_layers * 2 * hidden_size,
                false,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros,
                device,
                dtype,
                0,
            )?,
            prefix_len,
            num_layers,
            hidden_size,
        })
    }

    pub fn from_weights(
        embeddings: Tensor,
        reparam_w: Tensor,
        num_layers: usize,
    ) -> Result<Self> {
        if embeddings.rank() != 2 {
            return Err(SynaptixError::Unsupported("PTuningV2: embeddings must be [prefix_len, hidden]"));
        }
        let prefix_len = embeddings.dims()[0];
        let hidden_size = embeddings.dims()[1];
        if reparam_w.rank() != 2 || reparam_w.dims()[1] != hidden_size {
            return Err(SynaptixError::shape_mismatch(&[num_layers * 2 * hidden_size, hidden_size], reparam_w.dims()));
        }
        if reparam_w.dims()[0] != num_layers * 2 * hidden_size {
            return Err(SynaptixError::shape_mismatch(&[num_layers * 2 * hidden_size, hidden_size], reparam_w.dims()));
        }
        Ok(Self {
            embeddings: Parameter::new(embeddings),
            reparameterize: Linear::new(reparam_w, None)?,
            prefix_len,
            num_layers,
            hidden_size,
        })
    }

    /// Возвращает префикс-K/V форму `[num_layers, 2, prefix_len, hidden]`.
    pub fn forward(&self) -> Result<Tensor> {
        let emb = self.embeddings.tensor();
        let proj = self.reparameterize.forward(&emb)?;
        let four = proj.reshape(vec![self.prefix_len, self.num_layers, 2, self.hidden_size])?;
        four.permute(&[1, 2, 0, 3])?.contiguous()
    }

    /// Достаёт `(k, v)` для конкретного слоя; формы `[prefix_len, hidden]`.
    pub fn layer_kv(&self, full: &Tensor, layer: usize) -> Result<(Tensor, Tensor)> {
        let lyr = full.narrow(0, layer, 1)?.squeeze(0)?;
        let k = lyr.narrow(0, 0, 1)?.squeeze(0)?;
        let v = lyr.narrow(0, 1, 1)?.squeeze(0)?;
        Ok((k.contiguous()?, v.contiguous()?))
    }
}
