use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use std::sync::Arc;

use synaptix_llm_common::moe::{ExpertCache, ExpertCacheStats, ExpertSource, MoeFfn};
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::attention::linear::GatedDeltaNetState;
use synaptix_ops::embed::token_embedding;
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::attention::{KvLayer, QsaAttention};
use crate::config::{LayerType, Qwen4ExpConfig};
use crate::gated_residual::GatedResidual;
use crate::linear_attn::LinearAttn;
use crate::ngram::{NGramEmbedding, NGramRows};
use crate::norm::{coerr, ctx};
use crate::ple::{PleLayer, PleState};
use crate::qsa::IndexerCache;

pub const LM_PREFIX: &str = "model.language_model";

pub enum Mixer {
    Linear(LinearAttn),
    Qsa(QsaAttention),
}

pub struct Block {
    mixer: Mixer,
    moe: MoeFfn,
    attn_hc: GatedResidual,
    mlp_hc: GatedResidual,
    ple: Option<PleLayer>,
}

pub enum LayerState {
    Linear(GatedDeltaNetState),
    Qsa(Box<(KvLayer, IndexerCache)>),
}

pub struct ModelCache {
    pub layers: Vec<LayerState>,
    pub ple: Vec<PleState>,
    pub seq_len: usize,
    pub max_seq: usize,
}

impl ModelCache {
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for l in self.layers.iter_mut() {
            match l {
                LayerState::Linear(s) => {
                    s.conv_state.iter_mut().for_each(|x| *x = 0.0);
                    s.ssm_state.iter_mut().for_each(|x| *x = 0.0);
                    s.conv_state_dev = None;
                    s.ssm_state_dev = None;
                }
                LayerState::Qsa(b) => b.1.reset(),
            }
        }
    }
}

pub type NGramTableFactory<'a> = dyn Fn(usize) -> Result<Box<dyn NGramRows>, ModelError> + 'a;

/// Таблица токен-эмбеддингов: в квантованном бандле она лежит MXFP8 и
/// читается gather-ядром, в плотном — обычным `index_select`.
pub enum EmbedTable {
    Dense(Tensor),
    Quant(QuantWeight),
}

pub struct Qwen4ExpModel {
    pub config: Qwen4ExpConfig,
    pub device: Device,
    pub compute: DType,
    embed: EmbedTable,
    blocks: Vec<Block>,
    mixer_hc: GatedResidual,
    lm_head: QLinear,
    rope: RopeCache,
    ple_layers: Vec<usize>,
    expert_cache: Option<Arc<ExpertCache>>,
}

impl Qwen4ExpModel {
    pub fn build(
        cfg: &Qwen4ExpConfig,
        weights: &dyn WeightSource,
        device: Device,
        compute: DType,
        quant: DType,
        rope_capacity: usize,
        ngram_table: &NGramTableFactory<'_>,
    ) -> Result<Self, ModelError> {
        Self::build_with_cache(
            cfg,
            weights,
            device,
            compute,
            quant,
            rope_capacity,
            ngram_table,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_with_cache(
        cfg: &Qwen4ExpConfig,
        weights: &dyn WeightSource,
        device: Device,
        compute: DType,
        quant: DType,
        rope_capacity: usize,
        ngram_table: &NGramTableFactory<'_>,
        expert_cache: Option<Arc<ExpertCache>>,
        expert_source: Option<Arc<dyn ExpertSource>>,
    ) -> Result<Self, ModelError> {
        let embed_key = format!("{LM_PREFIX}.embed_tokens.weight");
        let embed = match weights.quant(&embed_key, device) {
            Some(q) => EmbedTable::Quant(q?),
            None => EmbedTable::Dense(weights.tensor(&embed_key, device, compute)?),
        };
        let lm_head = if cfg.tie_word_embeddings {
            match &embed {
                EmbedTable::Dense(t) => QLinear::build(t.clone(), compute, compute)?,
                EmbedTable::Quant(_) => {
                    return Err(ModelError::Build(
                        "tie_word_embeddings с квантованными эмбеддингами не поддержан".into(),
                    ))
                }
            }
        } else if let Some(prequant) = weights.quant("lm_head.weight", device) {
            QLinear::Quant(prequant?)
        } else {
            let w = weights.tensor(
                "lm_head.weight",
                device,
                if quant.is_quantized() { DType::F16 } else { compute },
            )?;
            QLinear::build(w, quant, compute)?
        };

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let mut ple_layers = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            let prefix = format!("{LM_PREFIX}.layers.{l}");
            let mixer = match cfg.layer_type(l) {
                LayerType::Linear => Mixer::Linear(LinearAttn::load(
                    weights,
                    &format!("{prefix}.linear_attn"),
                    cfg,
                    device,
                    compute,
                    quant,
                )?),
                LayerType::Qsa => Mixer::Qsa(QsaAttention::load(
                    weights,
                    &format!("{prefix}.self_attn"),
                    cfg,
                    device,
                    compute,
                    quant,
                )?),
            };
            let moe = match (&expert_cache, &expert_source) {
                (Some(cache), Some(source)) => MoeFfn::load_lazy(
                    weights,
                    &format!("{prefix}.mlp"),
                    cfg.moe.clone(),
                    device,
                    compute,
                    quant,
                    cache.clone(),
                    l,
                    source.clone(),
                )?,
                (Some(cache), None) => MoeFfn::load_offloaded(
                    weights,
                    &format!("{prefix}.mlp"),
                    cfg.moe.clone(),
                    device,
                    compute,
                    quant,
                    cache.clone(),
                    l,
                )?,
                (None, _) => MoeFfn::load(
                    weights,
                    &format!("{prefix}.mlp"),
                    cfg.moe.clone(),
                    device,
                    compute,
                    quant,
                )?,
            };
            let ple = match cfg.ple_at(l) {
                Some(index) => {
                    let ple_cfg = cfg.ple.as_ref().unwrap();
                    let eos = cfg.eos_token_ids.first().copied().unwrap_or(0);
                    let table = ngram_table(l)?;
                    let buffers = read_ngram_buffers(weights, &format!("{prefix}.ple.ple_embedding"));
                    let embedding = NGramEmbedding::new(
                        ple_cfg,
                        index,
                        cfg.vocab_size,
                        eos,
                        table,
                        buffers,
                        device,
                        compute,
                    )?;
                    ple_layers.push(l);
                    Some(PleLayer::load(
                        weights,
                        &format!("{prefix}.ple"),
                        cfg,
                        embedding,
                        device,
                        compute,
                        quant,
                    )?)
                }
                None => None,
            };
            blocks.push(Block {
                mixer,
                moe,
                attn_hc: GatedResidual::load(
                    weights,
                    &format!("{prefix}.attn_hyper_connection"),
                    cfg,
                    device,
                    compute,
                    quant,
                    true,
                )?,
                mlp_hc: GatedResidual::load(
                    weights,
                    &format!("{prefix}.mlp_hyper_connection"),
                    cfg,
                    device,
                    compute,
                    quant,
                    true,
                )?,
                ple,
            });
        }

        let mixer_hc = GatedResidual::load(
            weights,
            &format!("{LM_PREFIX}.hyper_connection_mixer"),
            cfg,
            device,
            compute,
            quant,
            false,
        )?;
        let rope = RopeCache::new(
            cfg.rope.rotary_dim.max(2),
            rope_capacity.max(1),
            cfg.rope.theta,
            device,
        )
        .map_err(|e| ModelError::Build(e.to_string()))?;

        Ok(Self {
            config: cfg.clone(),
            device,
            compute,
            embed,
            blocks,
            mixer_hc,
            lm_head,
            rope,
            ple_layers,
            expert_cache,
        })
    }

    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.expert_cache.as_ref().map(|c| c.stats())
    }

    pub fn make_cache(&self, max_seq: usize) -> Result<ModelCache, ModelError> {
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            layers.push(match &block.mixer {
                Mixer::Linear(la) => LayerState::Linear(la.make_state()),
                Mixer::Qsa(qa) => LayerState::Qsa(Box::new((
                    KvLayer::new(
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                        max_seq,
                        self.device,
                        self.compute,
                    )?,
                    qa.indexer.make_cache(max_seq)?,
                ))),
            });
        }
        let ple = self
            .blocks
            .iter()
            .filter_map(|b| b.ple.as_ref().map(|p| p.state()))
            .collect();
        Ok(ModelCache { layers, ple, seq_len: 0, max_seq })
    }

    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor, ModelError> {
        let ids = Tensor::from_vec(tokens.to_vec(), vec![tokens.len()], self.device)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        match &self.embed {
            EmbedTable::Dense(t) => coerr(token_embedding(&ids, t)),
            EmbedTable::Quant(q) => {
                let rows = coerr(q.embed_gather(&ids))?;
                coerr(rows.to_dtype(self.compute))
            }
        }
    }

    pub fn forward_hidden(
        &self,
        tokens: &[u32],
        cache: &mut ModelCache,
    ) -> Result<Tensor, ModelError> {
        self.forward_traced(tokens, cache, false).map(|(h, _)| h)
    }

    pub fn forward_traced(
        &self,
        tokens: &[u32],
        cache: &mut ModelCache,
        trace: bool,
    ) -> Result<(Tensor, Vec<(String, Tensor)>), ModelError> {
        let s = tokens.len();
        if s == 0 {
            return Err(ModelError::Forward("пустой вход".into()));
        }
        let past = cache.seq_len;
        let embeds = self.embed_tokens(tokens)?;
        let hc = self.config.hc_count;
        let hidden = coerr(embeds.reshape(vec![s, 1, self.config.hidden_size]))?;
        let ones = coerr(Tensor::zeros(
            vec![s, hc, self.config.hidden_size],
            self.compute,
            self.device,
        ))?;
        let mut hidden = coerr(coerr(hidden.broadcast_add(&ones))?
            .reshape(vec![s, hc * self.config.hidden_size]))?;

        let mut traced = Vec::new();
        let mut ple_slot = 0usize;
        for (l, block) in self.blocks.iter().enumerate() {
            if trace {
                traced.push((format!("in_{l}"), hidden.clone()));
            }
            if let Some(ple) = &block.ple {
                let state = &mut cache.ple[ple_slot];
                ple_slot += 1;
                let delta = ctx(ple.forward(&hidden, tokens, state), &format!("слой {l} ple"))?;
                hidden = coerr(hidden.add(&delta))?;
            }

            let mixed = ctx(block.attn_hc.forward(&hidden), &format!("слой {l} attn_hc"))?;
            let out = match (&block.mixer, &mut cache.layers[l]) {
                (Mixer::Linear(la), LayerState::Linear(state)) => {
                    ctx(la.forward(&mixed.mixed, state, s), &format!("слой {l} linear_attn"))?
                }
                (Mixer::Qsa(qa), LayerState::Qsa(state)) => {
                    let (kv, idx) = state.as_mut();
                    let (out, selected) = ctx(
                        qa.forward(&mixed.mixed, kv, idx, past, s, &self.rope),
                        &format!("слой {l} qsa"),
                    )?;
                    if trace {
                        let kv_len = past + s;
                        let mut mask = vec![0f32; s * kv_len];
                        match &selected {
                            Some(sel) => {
                                for (i, row) in sel.iter().enumerate() {
                                    for t in row {
                                        mask[i * kv_len + *t as usize] = 1.0;
                                    }
                                }
                            }
                            None => {
                                for i in 0..s {
                                    for j in 0..=(past + i) {
                                        mask[i * kv_len + j] = 1.0;
                                    }
                                }
                            }
                        }
                        traced.push((
                            format!("index_mask_{l}"),
                            coerr(Tensor::from_vec(mask, vec![s, kv_len], self.device))?,
                        ));
                    }
                    out
                }
                _ => return Err(ModelError::Shape(format!("слой {l}: кэш не того типа"))),
            };
            if trace {
                traced.push((format!("mixer_out_{l}"), out.clone()));
            }
            hidden = ctx(block.attn_hc.inject(
                &mixed.hyper,
                &out,
                mixed
                    .inject_weights
                    .as_ref()
                    .ok_or_else(|| ModelError::Forward("attn hc без inject".into()))?,
            ), &format!("слой {l} attn_inject"))?;

            let mixed = ctx(block.mlp_hc.forward(&hidden), &format!("слой {l} mlp_hc"))?;
            let out = ctx(block.moe.forward(&mixed.mixed), &format!("слой {l} moe"))?;
            hidden = ctx(block.mlp_hc.inject(
                &mixed.hyper,
                &out,
                mixed
                    .inject_weights
                    .as_ref()
                    .ok_or_else(|| ModelError::Forward("mlp hc без inject".into()))?,
            ), &format!("слой {l} mlp_inject"))?;
            if trace {
                traced.push((format!("layer_out_{l}"), hidden.clone()));
            }
        }

        cache.seq_len = past + s;
        let out = self.mixer_hc.forward(&hidden)?.mixed;
        if trace {
            traced.push(("final".to_string(), out.clone()));
        }
        Ok((out, traced))
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut ModelCache) -> Result<Tensor, ModelError> {
        let hidden = self.forward_hidden(tokens, cache)?;
        self.lm_head.forward(&hidden)
    }

    pub fn forward_last(&self, tokens: &[u32], cache: &mut ModelCache) -> Result<Tensor, ModelError> {
        let hidden = self.forward_hidden(tokens, cache)?;
        let s = hidden.dims()[0];
        let last = coerr(coerr(hidden.narrow(0, s - 1, 1))?.contiguous())?;
        self.lm_head.forward(&last)
    }

    pub fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        self.lm_head.forward(hidden)
    }

    pub fn ple_layers(&self) -> &[usize] {
        &self.ple_layers
    }
}

fn read_ngram_buffers(
    weights: &dyn WeightSource,
    prefix: &str,
) -> Option<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    let read = |name: &str| -> Option<Vec<i64>> {
        let key = format!("{prefix}.{name}");
        if !weights.contains(&key) {
            return None;
        }
        let t = weights.tensor(&key, Device::Cpu, DType::I64).ok()?;
        t.flatten_all().ok()?.to_vec1::<i64>().ok()
    };
    Some((
        read("layer_multipliers")?,
        read("ngram_heads_vocab_sizes")?,
        read("ngram_heads_offsets")?,
    ))
}
