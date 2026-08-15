use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm;

use crate::config::{DecoderConfig, LayerKind, NormGain};
use crate::model::{DecoderModel, KvCache, ModelError};
use crate::weights::{QLinear, WeightSource};

pub const MTP_PREFIX: &str = "mtp";

fn remap(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.layers.") {
        format!("{MTP_PREFIX}.layers.{rest}")
    } else if key == "model.norm.weight" {
        format!("{MTP_PREFIX}.norm.weight")
    } else {
        key.to_string()
    }
}

struct MtpWeights<'a> {
    inner: &'a dyn WeightSource,
    hidden: usize,
}

impl WeightSource for MtpWeights<'_> {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        if key == "model.embed_tokens.weight" || key == "lm_head.weight" {
            return Tensor::zeros(vec![1usize, self.hidden], dtype, device)
                .map_err(|e| ModelError::Load(e.to_string()));
        }
        self.inner.tensor(&remap(key), device, dtype)
    }
    fn contains(&self, key: &str) -> bool {
        if key == "model.embed_tokens.weight" || key == "lm_head.weight" {
            return true;
        }
        self.inner.contains(&remap(key))
    }
}

pub fn present(weights: &dyn WeightSource) -> bool {
    weights.contains("mtp.fc.weight")
        && weights.contains("mtp.layers.0.self_attn.q_proj.weight")
        && weights.contains("mtp.norm.weight")
}

pub struct MtpModule {
    pub layers: DecoderModel,
    pub num_layers: usize,
    fc: QLinear,
    norm_hidden: Tensor,
    norm_embedding: Tensor,
    rms_eps: f32,
    device: Device,
    compute: DType,
}

impl MtpModule {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        cfg: &DecoderConfig,
        num_layers: usize,
        weights: &dyn WeightSource,
        device: Device,
        compute: DType,
        attn_w: DType,
        mlp_w: DType,
        lm_head_dtype: DType,
        embed_dtype: DType,
        rope_capacity: usize,
    ) -> Result<Self, ModelError> {
        if num_layers == 0 {
            return Err(ModelError::Build("mtp_num_hidden_layers = 0".into()));
        }
        let mut mcfg = cfg.clone();
        mcfg.num_hidden_layers = num_layers;
        mcfg.layer_kinds = vec![LayerKind::Full; num_layers];
        mcfg.linear = None;
        mcfg.sliding_window = None;
        mcfg.rope_local = None;
        mcfg.vocab_size = 1;
        mcfg.tie_word_embeddings = true;

        let src = MtpWeights { inner: weights, hidden: cfg.hidden_size };
        let layers = DecoderModel::build(
            &mcfg,
            &src,
            device,
            compute,
            attn_w,
            mlp_w,
            lm_head_dtype,
            embed_dtype,
            rope_capacity,
        )?;

        let one_plus = cfg.norm_gain == NormGain::OnePlus;
        let norm = |key: &str| -> Result<Tensor, ModelError> {
            let w = weights.tensor(key, device, if one_plus { DType::F32 } else { compute })?;
            if one_plus {
                w.add_scalar(1.0)
                    .and_then(|t| t.to_dtype(compute))
                    .map_err(|e| ModelError::Load(e.to_string()))
            } else {
                Ok(w)
            }
        };
        let fc_dtype = if matches!(device, Device::Cuda(_)) && attn_w.is_quantized() {
            attn_w
        } else {
            compute
        };
        let wdt = if fc_dtype.is_quantized() { DType::F16 } else { compute };
        let fc = QLinear::build(
            weights.tensor("mtp.fc.weight", device, wdt)?,
            fc_dtype,
            compute,
        )?;

        Ok(Self {
            layers,
            num_layers,
            fc,
            norm_hidden: norm("mtp.pre_fc_norm_hidden.weight")?,
            norm_embedding: norm("mtp.pre_fc_norm_embedding.weight")?,
            rms_eps: cfg.rms_norm_eps,
            device,
            compute,
        })
    }

    pub fn rope_capacity(&self) -> usize {
        self.layers.rope_capacity()
    }

    pub fn make_kv_cache(&self, batch: usize, max_seq: usize) -> Result<KvCache, ModelError> {
        self.layers.make_kv_cache(batch, max_seq)
    }

    pub fn project(
        &self,
        parent: &DecoderModel,
        trunk_hidden: &Tensor,
        next_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let e = parent.embed_ids(next_ids)?;
        let h = rms_norm(trunk_hidden, &self.norm_hidden, self.rms_eps)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let e = rms_norm(&e, &self.norm_embedding, self.rms_eps)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let cat = Tensor::cat(&[&e, &h], 2).map_err(|e| ModelError::Forward(e.to_string()))?;
        let cat = cat
            .to_dtype(self.compute)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        self.fc.forward(&cat)
    }

    pub fn advance(
        &self,
        parent: &DecoderModel,
        trunk_hidden: &Tensor,
        next_ids: &Tensor,
        kv: &mut KvCache,
    ) -> Result<(), ModelError> {
        let x = self.project(parent, trunk_hidden, next_ids)?;
        self.layers.forward_from_hidden(&x, kv)?;
        Ok(())
    }

    pub fn draft_logits(
        &self,
        parent: &DecoderModel,
        trunk_hidden: &Tensor,
        next_ids: &Tensor,
        kv: &mut KvCache,
    ) -> Result<Tensor, ModelError> {
        let x = self.project(parent, trunk_hidden, next_ids)?;
        let out = self.layers.forward_from_hidden(&x, kv)?;
        let row = self.layers.normed_at(&out, out.dims()[1] - 1)?;
        parent.lm_head_forward(&row)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_layer_and_norm_keys_only() {
        assert_eq!(remap("model.layers.0.self_attn.q_proj.weight"), "mtp.layers.0.self_attn.q_proj.weight");
        assert_eq!(remap("model.norm.weight"), "mtp.norm.weight");
        assert_eq!(remap("model.embed_tokens.weight"), "model.embed_tokens.weight");
        assert_eq!(remap("lm_head.weight"), "lm_head.weight");
    }
}
