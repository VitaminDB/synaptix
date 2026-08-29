use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use std::sync::Arc;

use synaptix_llm_common::model::RopePositions;
use synaptix_llm_common::moe::{ExpertCache, ExpertCacheStats, ExpertSource, MoeFfn};
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::attention::linear::GatedDeltaNetState;
use synaptix_ops::embed::token_embedding;
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::attention::{KvLayer, QsaAttention};
use crate::config::{LayerType, Qwen4ExpConfig};
use crate::gated_residual::GatedResidual;
use crate::linear_attn::{GdnSnap, LinearAttn};
use crate::ngram::{NGramEmbedding, NGramRows};
use crate::norm::{coerr, ctx, stage};
use crate::ple::{PleLayer, PleState};
use crate::qsa::{park_tensor, unpark_tensor, IndexerCache};

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
    /// Снять состояние целиком: рекуррентное — копией, KV и ключи индексатора —
    /// меткой позиции, дальше их просто перепишут.
    pub fn snapshot(&mut self) -> Result<CacheSnapshot, ModelError> {
        let mut linear = Vec::new();
        let mut qsa = Vec::new();
        for layer in self.layers.iter_mut() {
            match layer {
                LayerState::Linear(s) => linear.push(GdnSnap::take(s)?),
                LayerState::Qsa(b) => qsa.push(b.1.mark()),
            }
        }
        Ok(CacheSnapshot {
            seq_len: self.seq_len,
            linear,
            qsa,
            ple: self.ple.clone(),
        })
    }

    /// Вернуть состояние: драфт не подтвердился.
    pub fn restore(&mut self, snap: &CacheSnapshot) -> Result<(), ModelError> {
        let mut linear = snap.linear.iter();
        let mut qsa = snap.qsa.iter();
        for layer in self.layers.iter_mut() {
            match layer {
                LayerState::Linear(s) => {
                    if let Some(saved) = linear.next() {
                        saved.restore(s)?;
                    }
                }
                LayerState::Qsa(b) => {
                    if let Some(mark) = qsa.next() {
                        b.1.rewind(*mark);
                    }
                }
            }
        }
        self.ple.clone_from(&snap.ple);
        self.seq_len = snap.seq_len;
        Ok(())
    }

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

    /// Переселить содержимое кэша в host-RAM, освободив VRAM. Позиции
    /// (`seq_len`, метки индексатора) не трогаются — переезжают только
    /// данные, так что после [`Self::unpark_to`] ход продолжается с той же
    /// точки.
    ///
    /// Зачем: пока идёт вложенная генерация (субагент чата), посчитанный
    /// контекст диалога держит гигабайты, а вложенному прогону их не хватает.
    /// Перевоз через PCIe дешевле, чем потерять префикс и префиллить историю
    /// заново.
    pub fn park_to_host(&mut self) -> Result<usize, ModelError> {
        let mut moved = 0;
        for l in self.layers.iter_mut() {
            match l {
                LayerState::Linear(s) => {
                    // Истина GDN — host-векторы, зеркала пересеет
                    // `sync_to_device`; но сперва считываем то, что дописал
                    // graph-декод.
                    s.sync_to_host().map_err(|e| ModelError::Forward(e.to_string()))?;
                    moved += dev_bytes(s.conv_state_dev.as_ref())
                        + dev_bytes(s.ssm_state_dev.as_ref());
                    s.conv_state_dev = None;
                    s.ssm_state_dev = None;
                }
                LayerState::Qsa(b) => {
                    let (kv, idx) = b.as_mut();
                    moved += park_tensor(&mut kv.k)?;
                    moved += park_tensor(&mut kv.v)?;
                    if let Some(t) = kv.k_scale.as_mut() {
                        moved += park_tensor(t)?;
                    }
                    if let Some(t) = kv.v_scale.as_mut() {
                        moved += park_tensor(t)?;
                    }
                    moved += idx.park_to_host()?;
                }
            }
        }
        Ok(moved)
    }

    pub fn unpark_to(&mut self, device: Device) -> Result<usize, ModelError> {
        let mut moved = 0;
        for l in self.layers.iter_mut() {
            let LayerState::Qsa(b) = l else { continue };
            let (kv, idx) = b.as_mut();
            moved += unpark_tensor(&mut kv.k, device)?;
            moved += unpark_tensor(&mut kv.v, device)?;
            if let Some(t) = kv.k_scale.as_mut() {
                moved += unpark_tensor(t, device)?;
            }
            if let Some(t) = kv.v_scale.as_mut() {
                moved += unpark_tensor(t, device)?;
            }
            moved += idx.unpark_to(device)?;
        }
        Ok(moved)
    }

    pub fn is_parked(&self) -> bool {
        self.layers.iter().any(|l| match l {
            LayerState::Qsa(b) => b.0.k.device() == Device::Cpu,
            LayerState::Linear(_) => false,
        })
    }

    /// Сколько VRAM держит кэш прямо сейчас.
    pub fn device_bytes(&self) -> usize {
        let mut total = 0;
        for l in &self.layers {
            match l {
                LayerState::Qsa(b) => {
                    let (kv, idx) = (&b.0, &b.1);
                    for t in [Some(&kv.k), Some(&kv.v), kv.k_scale.as_ref(), kv.v_scale.as_ref()] {
                        if let Some(t) = t.filter(|t| t.device() != Device::Cpu) {
                            total += dev_bytes(Some(t));
                        }
                    }
                    total += idx.device_bytes();
                }
                LayerState::Linear(s) => {
                    total += dev_bytes(s.conv_state_dev.as_ref())
                        + dev_bytes(s.ssm_state_dev.as_ref());
                }
            }
        }
        total
    }
}

fn dev_bytes(t: Option<&Tensor>) -> usize {
    t.map(|t| t.dtype().bytes_for_numel(t.numel())).unwrap_or(0)
}

/// Снимок состояний, которые нельзя переписать задним числом. KV и ключи
/// индексатора достаточно усечь по позиции, а рекуррентное состояние линейного
/// внимания, свёртка PLE и его история токенов копируются: спекулятивный шаг
/// впитывает драфт, и при отказе всё это надо вернуть.
pub struct CacheSnapshot {
    seq_len: usize,
    linear: Vec<GdnSnap>,
    qsa: Vec<(usize, usize)>,
    ple: Vec<PleState>,
}

impl CacheSnapshot {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Отпустить device-копии снимка (см. [`ModelCache::park_to_host`]):
    /// `GdnSnap` дублирует состояние в host-векторах, поэтому переезд для
    /// него — это просто освобождение зеркал.
    pub fn park_to_host(&mut self) -> usize {
        self.linear.iter_mut().map(|s| s.park_to_host()).sum()
    }

    pub fn device_bytes(&self) -> usize {
        self.linear.iter().map(|s| s.device_bytes()).sum()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Trace {
    Off,
    Stream,
    Full,
}

#[derive(Clone, Copy)]
struct RunOpts<'a> {
    trace: Trace,
    split: Option<usize>,
    rope: RopePositions<'a>,
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
    kv_dtype: DType,
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
            kv_dtype: compute,
        })
    }

    /// Чем хранить KV: `MXFP8` кладёт его квантованным — вдвое меньше и
    /// занятой памяти, и трафика внимания.
    pub fn set_kv_dtype(&mut self, dtype: DType) {
        self.kv_dtype = dtype;
    }

    pub fn kv_quantized(&self) -> bool {
        (self.kv_dtype == DType::MXFP8 || crate::attention::kv_fp8()) && self.device != Device::Cpu
    }

    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.expert_cache.as_ref().map(|c| c.stats())
    }

    pub fn expert_cache(&self) -> Option<&Arc<ExpertCache>> {
        self.expert_cache.as_ref()
    }

    pub fn make_cache(&self, max_seq: usize) -> Result<ModelCache, ModelError> {
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            layers.push(match &block.mixer {
                Mixer::Linear(la) => LayerState::Linear(la.make_state()),
                Mixer::Qsa(qa) => LayerState::Qsa(Box::new((
                    if self.kv_quantized() {
                        KvLayer::new_mxfp8(
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            max_seq,
                            self.device,
                        )?
                    } else {
                        KvLayer::new(
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            max_seq,
                            self.device,
                            self.compute,
                        )?
                    },
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

    /// Эмбеддинги промпта с подставленными строками медиа: `media` — пары
    /// «id заполнителя → матрица `[n, H]`», строки расходуются по порядку
    /// появления заполнителя в промпте.
    pub fn embed_with_media(
        &self,
        tokens: &[u32],
        media: &[(u32, Tensor)],
    ) -> Result<Tensor, ModelError> {
        let embeds = self.embed_tokens(tokens)?;
        if media.is_empty() {
            return Ok(embeds);
        }
        let hidden = self.config.hidden_size;
        let mut rows = embeds
            .to_device(Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| ModelError::Forward(e.to_string()))?;

        for (pad, feats) in media {
            let dims = feats.dims().to_vec();
            if dims.len() != 2 || dims[1] != hidden {
                return Err(ModelError::Shape(format!(
                    "медиа-эмбеддинги: форма {dims:?}, ожидалось [n, {hidden}]"
                )));
            }
            let src = feats
                .to_device(Device::Cpu)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| ModelError::Forward(e.to_string()))?;
            let slots: Vec<usize> = tokens
                .iter()
                .enumerate()
                .filter(|(_, t)| *t == pad)
                .map(|(i, _)| i)
                .collect();
            if slots.len() != dims[0] {
                return Err(ModelError::Shape(format!(
                    "медиа: заполнителей {} против {} строк эмбеддингов",
                    slots.len(),
                    dims[0]
                )));
            }
            for (row, slot) in slots.into_iter().enumerate() {
                let from = row * hidden;
                rows[slot * hidden..(slot + 1) * hidden]
                    .copy_from_slice(&src[from..from + hidden]);
            }
        }

        coerr(
            Tensor::from_vec(rows, vec![tokens.len(), hidden], self.device)
                .and_then(|t| t.to_dtype(self.compute)),
        )
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
        self.forward_traced_media(tokens, &[], cache, trace)
    }

    pub fn forward_traced_media(
        &self,
        tokens: &[u32],
        media: &[(u32, Tensor)],
        cache: &mut ModelCache,
        trace: bool,
    ) -> Result<(Tensor, Vec<(String, Tensor)>), ModelError> {
        let opts = RunOpts {
            trace: if trace { Trace::Full } else { Trace::Off },
            split: None,
            rope: RopePositions::Sequential,
        };
        let (out, traced, _) = self.run(tokens, media, cache, opts)?;
        Ok((out, traced))
    }

    /// Прогон пары «токен + драфт» одним проходом. Кроме скрытых состояний
    /// обеих позиций отдаёт снимок кэша на границе между ними: если драфт не
    /// подтвердился, откатывать надо только его.
    pub fn forward_pair(
        &self,
        tokens: &[u32],
        cache: &mut ModelCache,
        pos: RopePositions,
    ) -> Result<(Tensor, Tensor, CacheSnapshot), ModelError> {
        if tokens.len() < 2 {
            return Err(ModelError::Forward("пара короче двух токенов".into()));
        }
        let opts = RunOpts { trace: Trace::Stream, split: Some(1), rope: pos };
        let (out, traced, snap) = self.run(tokens, &[], cache, opts)?;
        let stream = pick_stream(traced)?;
        let snap = snap.ok_or_else(|| ModelError::Forward("снимок пары не снят".into()))?;
        Ok((out, stream, snap))
    }

    /// Префилл слой за слоем: внешний цикл идёт по слоям, внутренний — по
    /// чанкам промпта. Эксперты слоя тогда поднимаются на карту один раз на
    /// весь промпт, а не на каждый чанк, — при чтении их из бандла это главная
    /// статья расхода. Цена — держать поток всех токенов целиком.
    ///
    /// Возвращает скрытое состояние последней позиции и её поток (последний
    /// нужен голове многотокенного предсказания).
    pub fn prefill_by_layers(
        &self,
        tokens: &[u32],
        media: &[(u32, Tensor)],
        cache: &mut ModelCache,
        chunk: usize,
        pos: RopePositions,
    ) -> Result<(Tensor, Tensor), ModelError> {
        let s = tokens.len();
        if s == 0 {
            return Err(ModelError::Forward("пустой вход".into()));
        }
        let chunk = chunk.clamp(1, s);
        let past = cache.seq_len;
        let hc = self.config.hc_count;
        let width = hc * self.config.hidden_size;

        let embeds = self.embed_with_media(tokens, media)?;
        let hidden = coerr(embeds.reshape(vec![s, 1, self.config.hidden_size]))?;
        let ones = coerr(Tensor::zeros(
            vec![s, hc, self.config.hidden_size],
            self.compute,
            self.device,
        ))?;
        let mut stream = coerr(coerr(hidden.broadcast_add(&ones))?.reshape(vec![1, s, width]))?;

        let mut ple_slot = 0usize;
        for (l, block) in self.blocks.iter().enumerate() {
            let ple_here = block.ple.as_ref().map(|ple| (ple, ple_slot));
            if ple_here.is_some() {
                ple_slot += 1;
            }
            let mut start = 0usize;
            while start < s {
                let len = chunk.min(s - start);
                // Копия обязана быть настоящей: срез потока делит storage с
                // ним самим, а запись результата обратно требует уникального
                // владения.
                let view = coerr(stream.narrow(1, start, len))?;
                let mut piece = coerr(Tensor::empty_uninit(
                    vec![1, len, width],
                    self.compute,
                    self.device,
                ))?;
                coerr(piece.copy_from(&view))?;
                drop(view);
                let mut piece = coerr(piece.reshape(vec![len, width]))?;

                if let Some((ple, slot)) = ple_here {
                    let state = &mut cache.ple[slot];
                    let delta = stage("ple", || {
                        ctx(
                            ple.forward(&piece, &tokens[start..start + len], state),
                            &format!("слой {l} ple"),
                        )
                    })?;
                    piece = coerr(piece.add(&delta))?;
                }

                let mixed = stage("hc", || {
                    ctx(block.attn_hc.forward(&piece), &format!("слой {l} attn_hc"))
                })?;
                let out = match (&block.mixer, &mut cache.layers[l]) {
                    (Mixer::Linear(la), LayerState::Linear(state)) => stage("linear_attn", || {
                        ctx(la.forward(&mixed.mixed, state, len), &format!("слой {l} linear_attn"))
                    })?,
                    (Mixer::Qsa(qa), LayerState::Qsa(state)) => {
                        let (kv, idx) = state.as_mut();
                        let (out, _) = stage("qsa", || {
                            ctx(
                                qa.forward(
                                    &mixed.mixed,
                                    kv,
                                    idx,
                                    past + start,
                                    len,
                                    &self.rope,
                                    pos,
                                ),
                                &format!("слой {l} qsa"),
                            )
                        })?;
                        out
                    }
                    _ => return Err(ModelError::Shape(format!("слой {l}: кэш не того типа"))),
                };
                let injected = ctx(block.attn_hc.inject(
                    &mixed.hyper,
                    &out,
                    mixed
                        .inject_weights
                        .as_ref()
                        .ok_or_else(|| ModelError::Forward("attn hc без inject".into()))?,
                ), &format!("слой {l} attn_inject"))?;

                let mixed = ctx(block.mlp_hc.forward(&injected), &format!("слой {l} mlp_hc"))?;
                let out = stage("moe", || {
                    ctx(block.moe.forward(&mixed.mixed), &format!("слой {l} moe"))
                })?;
                let done = ctx(block.mlp_hc.inject(
                    &mixed.hyper,
                    &out,
                    mixed
                        .inject_weights
                        .as_ref()
                        .ok_or_else(|| ModelError::Forward("mlp hc без inject".into()))?,
                ), &format!("слой {l} mlp_inject"))?;

                let row = coerr(done.reshape(vec![1, len, width]))?;
                coerr(stream.copy_rows_from(start, &row))?;
                start += len;
            }
        }

        cache.seq_len = past + s;
        let stream = coerr(stream.reshape(vec![s, width]))?;
        let last_stream = coerr(coerr(stream.narrow(0, s - 1, 1))?.contiguous())?;
        let out = self.mixer_hc.forward(&last_stream)?.mixed;
        Ok((out, last_stream))
    }

    fn run(
        &self,
        tokens: &[u32],
        media: &[(u32, Tensor)],
        cache: &mut ModelCache,
        opts: RunOpts,
    ) -> Result<(Tensor, Vec<(String, Tensor)>, Option<CacheSnapshot>), ModelError> {
        let s = tokens.len();
        if s == 0 {
            return Err(ModelError::Forward("пустой вход".into()));
        }
        let split = match opts.split {
            Some(k) if k == 0 || k >= s => {
                return Err(ModelError::Forward(format!("разрыв {k} вне длины {s}")))
            }
            other => other,
        };
        let trace = opts.trace == Trace::Full;
        let past = cache.seq_len;
        let embeds = self.embed_with_media(tokens, media)?;
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
        let mut snap_linear = Vec::new();
        let mut snap_qsa = Vec::new();
        let mut snap_ple = Vec::new();
        for (l, block) in self.blocks.iter().enumerate() {
            if trace {
                traced.push((format!("in_{l}"), hidden.clone()));
            }
            if let Some(ple) = &block.ple {
                let state = &mut cache.ple[ple_slot];
                ple_slot += 1;
                let delta = stage("ple", || match split {
                    None => ctx(ple.forward(&hidden, tokens, state), &format!("слой {l} ple")),
                    Some(k) => {
                        let head = coerr(coerr(hidden.narrow(0, 0, k))?.contiguous())?;
                        let first = ctx(
                            ple.forward(&head, &tokens[..k], state),
                            &format!("слой {l} ple"),
                        )?;
                        snap_ple.push(state.clone());
                        let tail = coerr(coerr(hidden.narrow(0, k, s - k))?.contiguous())?;
                        let second = ctx(
                            ple.forward(&tail, &tokens[k..], state),
                            &format!("слой {l} ple"),
                        )?;
                        coerr(Tensor::cat(&[&first, &second], 0))
                    }
                })?;
                hidden = coerr(hidden.add(&delta))?;
            }

            let mixed = stage("hc", || {
                ctx(block.attn_hc.forward(&hidden), &format!("слой {l} attn_hc"))
            })?;
            let out = match (&block.mixer, &mut cache.layers[l]) {
                (Mixer::Linear(la), LayerState::Linear(state)) => stage("linear_attn", || {
                    match split {
                        None => ctx(la.forward(&mixed.mixed, state, s), &format!("слой {l} linear_attn")),
                        Some(k) => {
                            let (out, snap) = ctx(
                                la.forward_split(&mixed.mixed, state, s, k),
                                &format!("слой {l} linear_attn"),
                            )?;
                            snap_linear.push(snap);
                            Ok(out)
                        }
                    }
                })?,
                (Mixer::Qsa(qa), LayerState::Qsa(state)) => {
                    let (kv, idx) = state.as_mut();
                    if let Some(k) = split {
                        snap_qsa.push(idx.mark_after(k));
                    }
                    let (out, selected) = stage("qsa", || {
                        ctx(
                            qa.forward(&mixed.mixed, kv, idx, past, s, &self.rope, opts.rope),
                            &format!("слой {l} qsa"),
                        )
                    })?;
                    if trace {
                        let kv_len = past + s;
                        let mut mask = vec![0f32; s * kv_len];
                        match &selected {
                            Some(sel) => {
                                for i in 0..sel.len() {
                                    for t in sel.positions(i)? {
                                        mask[i * kv_len + t as usize] = 1.0;
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
            let out = stage("moe", || {
                ctx(block.moe.forward(&mixed.mixed), &format!("слой {l} moe"))
            })?;
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
        if opts.trace != Trace::Off {
            traced.push(("stream".to_string(), hidden.clone()));
        }
        let out = self.mixer_hc.forward(&hidden)?.mixed;
        if trace {
            traced.push(("final".to_string(), out.clone()));
        }
        let snap = split.map(|k| CacheSnapshot {
            seq_len: past + k,
            linear: snap_linear,
            qsa: snap_qsa,
            ple: snap_ple,
        });
        Ok((out, traced, snap))
    }

    /// Скрытое состояние под голову словаря и поток последнего слоя `[T, hc·H]`
    /// до сводящего миксера — второй нужен голове многотокенного предсказания.
    pub fn forward_with_stream(
        &self,
        tokens: &[u32],
        cache: &mut ModelCache,
    ) -> Result<(Tensor, Tensor), ModelError> {
        let opts = RunOpts { trace: Trace::Stream, split: None, rope: RopePositions::Sequential };
        let (out, traced, _) = self.run(tokens, &[], cache, opts)?;
        Ok((out, pick_stream(traced)?))
    }

    pub fn forward_media_with_stream(
        &self,
        tokens: &[u32],
        media: &[(u32, Tensor)],
        cache: &mut ModelCache,
        pos: RopePositions,
    ) -> Result<(Tensor, Tensor), ModelError> {
        let opts = RunOpts { trace: Trace::Stream, split: None, rope: pos };
        let (out, traced, _) = self.run(tokens, media, cache, opts)?;
        Ok((out, pick_stream(traced)?))
    }

    pub fn forward_media_last(
        &self,
        tokens: &[u32],
        media: &[(u32, Tensor)],
        cache: &mut ModelCache,
        pos: RopePositions,
    ) -> Result<Tensor, ModelError> {
        let opts = RunOpts { trace: Trace::Off, split: None, rope: pos };
        let (hidden, _, _) = self.run(tokens, media, cache, opts)?;
        let s = hidden.dims()[0];
        let last = coerr(coerr(hidden.narrow(0, s - 1, 1))?.contiguous())?;
        self.lm_head.forward(&last)
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut ModelCache) -> Result<Tensor, ModelError> {
        let hidden = self.forward_hidden(tokens, cache)?;
        self.lm_head.forward(&hidden)
    }

    pub fn forward_last(&self, tokens: &[u32], cache: &mut ModelCache) -> Result<Tensor, ModelError> {
        self.forward_last_pos(tokens, cache, RopePositions::Sequential)
    }

    pub fn forward_last_pos(
        &self,
        tokens: &[u32],
        cache: &mut ModelCache,
        pos: RopePositions,
    ) -> Result<Tensor, ModelError> {
        let opts = RunOpts { trace: Trace::Off, split: None, rope: pos };
        let (hidden, _, _) = self.run(tokens, &[], cache, opts)?;
        let s = hidden.dims()[0];
        let last = coerr(coerr(hidden.narrow(0, s - 1, 1))?.contiguous())?;
        self.lm_head.forward(&last)
    }

    pub fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        self.lm_head.forward(hidden)
    }

    pub fn rope(&self) -> &RopeCache {
        &self.rope
    }

    pub fn ple_layers(&self) -> &[usize] {
        &self.ple_layers
    }
}

fn pick_stream(traced: Vec<(String, Tensor)>) -> Result<Tensor, ModelError> {
    traced
        .into_iter()
        .find(|(name, _)| name == "stream")
        .map(|(_, t)| t)
        .ok_or_else(|| ModelError::Forward("поток последнего слоя не снят".into()))
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
