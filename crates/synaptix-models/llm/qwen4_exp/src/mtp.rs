use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::moe::{ExpertCache, ExpertSource, MoeFfn};
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::pos::rope_cache::RopeCache;
use std::sync::Arc;

use crate::attention::{KvLayer, QsaAttention};
use crate::config::Qwen4ExpConfig;
use crate::gated_residual::GatedResidual;
use crate::norm::{coerr, group_rms, load_one_plus, rms};
use crate::qsa::IndexerCache;

pub const MTP_PREFIX: &str = "mtp";

/// Голова многотокенного предсказания: один слой той же архитектуры поверх
/// потока последнего слоя и эмбеддинга уже выбранного токена.
///
/// В transformers эта голова не реализована, поэтому схема восстановлена по
/// раскладке весов: `pre_fc_norm_hidden` нормирует поток на 10240 группами по
/// ширине модели, `fc_hidden` [2560, 2560] применяется к каждой из четырёх
/// ветвей, `fc_embedding` проецирует нормированный эмбеддинг и складывается со
/// всеми ветвями. Дальше — обычный блок и свой сводящий gated-residual.
/// Проверять её стоит по доле совпадений с основной моделью.
pub struct MtpHead {
    norm_hidden: Tensor,
    norm_embedding: Tensor,
    fc_hidden: QLinear,
    fc_embedding: QLinear,
    attn: QsaAttention,
    moe: MoeFfn,
    attn_hc: GatedResidual,
    mlp_hc: GatedResidual,
    mixer: GatedResidual,
    hidden: usize,
    hc_count: usize,
    eps: f32,
}

pub struct MtpCache {
    pub kv: KvLayer,
    pub indexer: IndexerCache,
    pub seq_len: usize,
}

pub fn present(weights: &dyn WeightSource) -> bool {
    weights.contains(&format!("{MTP_PREFIX}.fc_hidden.weight"))
        && weights.contains(&format!("{MTP_PREFIX}.layers.0.self_attn.q_proj.weight"))
}

impl MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &dyn WeightSource,
        cfg: &Qwen4ExpConfig,
        device: Device,
        compute: DType,
        quant: DType,
        expert_cache: Option<Arc<ExpertCache>>,
        expert_source: Option<Arc<dyn ExpertSource>>,
        layer_id: usize,
    ) -> Result<Self, ModelError> {
        let lin = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{MTP_PREFIX}.{name}.weight");
            if let Some(prequant) = weights.quant(&key, device) {
                return Ok(QLinear::Quant(prequant?));
            }
            let w = weights.tensor(&key, device, if quant.is_quantized() { DType::F16 } else { compute })?;
            QLinear::build(w, quant, compute)
        };

        let moe = match (&expert_cache, &expert_source) {
            (Some(cache), Some(source)) => MoeFfn::load_lazy(
                weights,
                &format!("{MTP_PREFIX}.layers.0.mlp"),
                cfg.moe.clone(),
                device,
                compute,
                quant,
                cache.clone(),
                layer_id,
                source.clone(),
            )?,
            (Some(cache), None) => MoeFfn::load_offloaded(
                weights,
                &format!("{MTP_PREFIX}.layers.0.mlp"),
                cfg.moe.clone(),
                device,
                compute,
                quant,
                cache.clone(),
                layer_id,
            )?,
            (None, _) => MoeFfn::load(
                weights,
                &format!("{MTP_PREFIX}.layers.0.mlp"),
                cfg.moe.clone(),
                device,
                compute,
                quant,
            )?,
        };

        Ok(Self {
            norm_hidden: load_one_plus(
                weights,
                &format!("{MTP_PREFIX}.pre_fc_norm_hidden.weight"),
                device,
                compute,
            )?,
            norm_embedding: load_one_plus(
                weights,
                &format!("{MTP_PREFIX}.pre_fc_norm_embedding.weight"),
                device,
                compute,
            )?,
            fc_hidden: lin("fc_hidden")?,
            fc_embedding: lin("fc_embedding")?,
            attn: QsaAttention::load(
                weights,
                &format!("{MTP_PREFIX}.layers.0.self_attn"),
                cfg,
                device,
                compute,
                quant,
            )?,
            moe,
            attn_hc: GatedResidual::load(
                weights,
                &format!("{MTP_PREFIX}.layers.0.attn_hyper_connection"),
                cfg,
                device,
                compute,
                quant,
                true,
            )?,
            mlp_hc: GatedResidual::load(
                weights,
                &format!("{MTP_PREFIX}.layers.0.mlp_hyper_connection"),
                cfg,
                device,
                compute,
                quant,
                true,
            )?,
            mixer: GatedResidual::load(
                weights,
                &format!("{MTP_PREFIX}.hyper_connection_mixer"),
                cfg,
                device,
                compute,
                quant,
                false,
            )?,
            hidden: cfg.hidden_size,
            hc_count: cfg.hc_count,
            eps: cfg.rms_norm_eps,
        })
    }

    pub fn make_cache(
        &self,
        cfg: &Qwen4ExpConfig,
        max_seq: usize,
        device: Device,
        compute: DType,
    ) -> Result<MtpCache, ModelError> {
        Ok(MtpCache {
            kv: KvLayer::new(cfg.num_key_value_heads, cfg.head_dim, max_seq, device, compute)?,
            indexer: self.attn.indexer.make_cache(max_seq)?,
            seq_len: 0,
        })
    }

    /// `hidden` — поток последнего слоя `[T, hc·H]` (до сводящего миксера
    /// основной модели), `embeds` — эмбеддинги уже выбранных токенов `[T, H]`.
    /// Возвращает скрытое состояние `[T, H]` под общую голову словаря.
    pub fn forward(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        cache: &mut MtpCache,
        rope: &RopeCache,
    ) -> Result<Tensor, ModelError> {
        let t = hidden.dims()[0];
        let normed = group_rms(hidden, &self.norm_hidden, self.hidden, self.eps)?;
        let branches = coerr(coerr(normed.contiguous())?.reshape(vec![t * self.hc_count, self.hidden]))?;
        let projected = self.fc_hidden.forward(&branches)?;
        let projected = coerr(projected.reshape(vec![t, self.hc_count, self.hidden]))?;

        let e = rms(embeds, &self.norm_embedding, self.eps)?;
        let e = self.fc_embedding.forward(&e)?;
        let e = coerr(coerr(e.contiguous())?.reshape(vec![t, 1, self.hidden]))?;

        let mut stream = coerr(coerr(projected.broadcast_add(&e))?
            .reshape(vec![t, self.hc_count * self.hidden]))?;

        let past = cache.seq_len;
        let mixed = self.attn_hc.forward(&stream)?;
        let (out, _) = self
            .attn
            .forward(&mixed.mixed, &mut cache.kv, &mut cache.indexer, past, t, rope)?;
        stream = self.attn_hc.inject(
            &mixed.hyper,
            &out,
            mixed
                .inject_weights
                .as_ref()
                .ok_or_else(|| ModelError::Forward("mtp: attn hc без inject".into()))?,
        )?;

        let mixed = self.mlp_hc.forward(&stream)?;
        let out = self.moe.forward(&mixed.mixed)?;
        stream = self.mlp_hc.inject(
            &mixed.hyper,
            &out,
            mixed
                .inject_weights
                .as_ref()
                .ok_or_else(|| ModelError::Forward("mtp: mlp hc без inject".into()))?,
        )?;

        cache.seq_len = past + t;
        Ok(self.mixer.forward(&stream)?.mixed)
    }
}
