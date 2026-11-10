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
use crate::qsa::{park_tensor, unpark_tensor, IndexerCache, IndexerTail};

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
    /// CUDA-графы шага декода по отрезкам слоёв линейного внимания.
    pub seg: crate::seg_graph::SegGraphs,
}

impl ModelCache {
    /// Снять состояние целиком: рекуррентное — копией, KV — меткой позиции
    /// (дальше его просто перепишут), индексатор — меткой ПЛЮС содержимым
    /// хвоста. Метки одной мало: между этим снимком и рестором следующего
    /// хода декод сворачивает `pending` десятки раз, и ключи хвоста промпта
    /// затираются — сессия обязана унести их с собой.
    pub fn snapshot(&mut self) -> Result<CacheSnapshot, ModelError> {
        let mut linear = Vec::new();
        let mut qsa = Vec::new();
        let mut qsa_tails = Vec::new();
        for layer in self.layers.iter_mut() {
            match layer {
                LayerState::Linear(s) => linear.push(GdnSnap::take(s)?),
                LayerState::Qsa(b) => {
                    qsa.push(b.1.mark());
                    qsa_tails.push(b.1.tail_snapshot()?);
                }
            }
        }
        Ok(CacheSnapshot {
            seq_len: self.seq_len,
            linear,
            qsa,
            qsa_tails: Some(qsa_tails),
            ple: self.ple.clone(),
        })
    }

    /// Вернуть состояние: драфт не подтвердился либо ход продолжает сессию.
    pub fn restore(&mut self, snap: &CacheSnapshot) -> Result<(), ModelError> {
        let mut linear = snap.linear.iter();
        let mut qsa = snap.qsa.iter();
        let mut tails = snap.qsa_tails.as_ref().map(|t| t.iter());
        for layer in self.layers.iter_mut() {
            match layer {
                LayerState::Linear(s) => {
                    if let Some(saved) = linear.next() {
                        saved.restore(s)?;
                    }
                }
                LayerState::Qsa(b) => {
                    let mark = qsa.next();
                    match tails.as_mut().and_then(|t| t.next()) {
                        Some(tail) => b.1.restore_tail(tail)?,
                        None => {
                            if let Some(mark) = mark {
                                b.1.rewind(*mark);
                            }
                        }
                    }
                }
            }
        }
        self.ple.clone_from(&snap.ple);
        self.seq_len = snap.seq_len;
        Ok(())
    }

    /// Расширить кэш до `max_seq` позиций без потери содержимого: KV и ключи
    /// индексатора дорастают нулями, рекуррентное состояние и PLE от длины
    /// не зависят. Позиции и снимки, снятые до роста, остаются годными.
    pub fn grow(&mut self, max_seq: usize) -> Result<(), ModelError> {
        if max_seq <= self.max_seq {
            return Ok(());
        }
        for l in self.layers.iter_mut() {
            if let LayerState::Qsa(b) = l {
                let (kv, idx) = b.as_mut();
                kv.grow(max_seq)?;
                idx.grow(max_seq)?;
            }
        }
        self.max_seq = max_seq;
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
        // Графы шага и буферы снимка пары держат VRAM и адреса прежнего состояния.
        self.seg.clear();
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
    /// Содержимое хвостов индексаторов. Есть у снимка сессии
    /// ([`ModelCache::snapshot`]); снимок спекулятивной пары живёт без него —
    /// пара откатывается тут же, пока хвост ниже метки цел, а снимать D2H
    /// посреди прогона было бы синхронизацией на каждый шаг.
    qsa_tails: Option<Vec<IndexerTail>>,
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
    /// Плотная таблица в RAM: строки токенов выбираются на хосте и едут на
    /// карту. Малая карта — таблица (248k×2560, 0,64 ГБ в MXFP8) там лишняя.
    Host(Tensor),
}

/// Держать ли таблицу эмбеддингов в RAM: на карте, где до загрузки свободно
/// меньше 16 ГБ (модель там и так едва помещается рядом с кэшем экспертов).
/// `SYN_QWEN4EXP_EMBED_HOST=1|0` — принудительно.
fn embed_on_host(device: Device) -> bool {
    if let Ok(v) = std::env::var("SYN_QWEN4EXP_EMBED_HOST") {
        return v.trim() == "1";
    }
    match device {
        Device::Cuda(o) => synaptix_core::device::cuda::mem_info(o)
            .map(|(free, _)| free < (16usize << 30))
            .unwrap_or(false),
        _ => false,
    }
}

/// Квантованная таблица → плотная в RAM: строки деквантуются тем же
/// gather-ядром на карте, порциями (пик — порция, а не таблица).
fn dequant_to_host(q: &QuantWeight, rows: usize, device: Device, compute: DType) -> Result<Tensor, ModelError> {
    const CHUNK: usize = 8192;
    let mut parts = Vec::with_capacity(rows.div_ceil(CHUNK));
    let mut start = 0usize;
    while start < rows {
        let n = CHUNK.min(rows - start);
        let ids: Vec<u32> = (start as u32..(start + n) as u32).collect();
        let idx = coerr(Tensor::from_vec(ids, vec![n], device))?;
        let part = coerr(q.embed_gather(&idx).and_then(|t| t.to_dtype(compute)).and_then(|t| t.to_device(Device::Cpu)))?;
        parts.push(part);
        start += n;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    coerr(Tensor::cat(&refs, 0))
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
        let host = embed_on_host(device) && !cfg.tie_word_embeddings;
        let embed = match weights.quant(&embed_key, device) {
            Some(q) if host => {
                let q = q?;
                let t = dequant_to_host(&q, cfg.vocab_size, device, compute)?;
                drop(q);
                eprintln!("[qwen4_exp] таблица эмбеддингов в RAM ({:.2} ГБ): на карте для неё мало места", (t.numel() * 2) as f64 / 1e9);
                EmbedTable::Host(t)
            }
            Some(q) => EmbedTable::Quant(q?),
            None if host => EmbedTable::Host(weights.tensor(&embed_key, Device::Cpu, compute)?),
            None => EmbedTable::Dense(weights.tensor(&embed_key, device, compute)?),
        };
        let lm_head = if cfg.tie_word_embeddings {
            match &embed {
                EmbedTable::Dense(t) => QLinear::build(t.clone(), compute, compute)?,
                EmbedTable::Quant(_) | EmbedTable::Host(_) => {
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

    /// `SYN_QWEN4EXP_TRACE_MEM=1`: раскладка VRAM перед каждым слоем
    /// послойного префилла — кэш экспертов по своему учёту, арена, пулы,
    /// свободное по драйверу. Нужна, когда OOM ловится «посреди слоя» и по
    /// сводке аллокатора не видно, кто держит память.
    fn trace_prefill_mem(&self, layer: usize) {
        let Device::Cuda(ord) = self.device else { return };
        let gb = |x: u64| x as f64 / (1u64 << 30) as f64;
        let (cache_used, cache_cap, resident) = match self.expert_cache() {
            Some(c) => {
                let st = c.stats();
                (st.bytes as u64, c.capacity_bytes() as u64, st.resident)
            }
            None => (0, 0, 0),
        };
        let arena = synaptix_core::memory::expert_arena::stats();
        let (ex_rsv, ex_used) =
            synaptix_core::device::cuda::experts_pool_stats(ord).unwrap_or((0, 0));
        let (act_rsv, act_used) =
            synaptix_core::device::cuda::activations_pool_stats(ord).unwrap_or((0, 0));
        let (free, _) = synaptix_core::device::cuda::mem_info(ord).unwrap_or((0, 0));
        eprintln!(
            "[qwen4_exp] слой {layer}: кэш {resident} шт {:.2}/{:.2} ГБ; арена {} slab'ов {:.2} ГБ (живых {:.2}); \
             пул экспертов {:.2}/{:.2} ГБ, пул активаций {:.2}/{:.2} ГБ; свободно {:.2} ГБ",
            gb(cache_used),
            gb(cache_cap),
            arena.slabs,
            gb(arena.reserved as u64),
            gb(arena.live as u64),
            gb(ex_used),
            gb(ex_rsv),
            gb(act_used),
            gb(act_rsv),
            gb(free as u64),
        );
    }

    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.expert_cache.as_ref().map(|c| c.stats())
    }

    /// Подключить декод MoE без синхронизации (zero-copy) ко всем слоям.
    pub fn attach_zc(&mut self, pool: &Arc<synaptix_llm_common::moe_zc::ZcPool>) {
        for b in &mut self.blocks {
            b.moe.attach_zc(pool.clone());
        }
    }

    /// Освободить слоты zero-copy всех слоёв (перед длинным префиллом).
    pub fn zc_release(&self) {
        for b in &self.blocks {
            b.moe.zc_release();
        }
    }

    /// Сводка zero-copy по слоям: (попадания в слоты, выборы, резидентов).
    pub fn zc_stats(&self) -> Option<(u64, u64, usize)> {
        let mut acc: Option<(u64, u64, usize)> = None;
        for b in &self.blocks {
            if let Some((h, p, r)) = b.moe.zc_stats() {
                let a = acc.get_or_insert((0, 0, 0));
                a.0 += h;
                a.1 += p;
                a.2 += r;
            }
        }
        acc
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
        Ok(ModelCache { layers, ple, seq_len: 0, max_seq, seg: Default::default() })
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
        if let EmbedTable::Host(t) = &self.embed {
            let ids = coerr(Tensor::from_vec(tokens.to_vec(), vec![tokens.len()], Device::Cpu))?;
            return coerr(t.index_select(0, &ids).and_then(|r| r.to_device(self.device)));
        }
        let ids = Tensor::from_vec(tokens.to_vec(), vec![tokens.len()], self.device)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        match &self.embed {
            EmbedTable::Dense(t) => coerr(token_embedding(&ids, t)),
            EmbedTable::Quant(q) => {
                let rows = coerr(q.embed_gather(&ids))?;
                coerr(rows.to_dtype(self.compute))
            }
            EmbedTable::Host(_) => unreachable!("разобран выше"),
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

    /// Пара «токен + драфт» с послойной трассировкой (отладка расхождений
    /// парного шага с декодом по одному токену, пример `pair_trace`).
    pub fn forward_pair_traced(&self, tokens: &[u32], cache: &mut ModelCache) -> Result<(Tensor, Vec<(String, Tensor)>), ModelError> {
        let opts = RunOpts { trace: Trace::Full, split: Some(1), rope: RopePositions::Sequential };
        let (out, traced, _) = self.run(tokens, &[], cache, opts)?;
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

        // Поток hyper-connections на весь промпт — единственный буфер «на
        // токен» этого прохода (20 КБ на токен у Flash-Next). Собираем его
        // чанками: прежний `zeros [s, hc, hidden]` + результат broadcast_add
        // жили до конца префилла вместе с эмбеддингами — три копии на весь
        // промпт вместо одной, 3,75 ГБ лишних на 146k токенов, и ход упирался
        // в OOM на первом же слое (журнал 10.09.2026).
        let mut stream = {
            let embeds = self.embed_with_media(tokens, media)?;
            let mut stream = coerr(Tensor::empty_uninit(
                vec![1, s, width],
                self.compute,
                self.device,
            ))?;
            let zeros = coerr(Tensor::zeros(
                vec![chunk, hc, self.config.hidden_size],
                self.compute,
                self.device,
            ))?;
            let mut start = 0usize;
            while start < s {
                let len = chunk.min(s - start);
                let piece = coerr(coerr(coerr(embeds.narrow(0, start, len))?.contiguous())?
                    .reshape(vec![len, 1, self.config.hidden_size]))?;
                let z = if len == chunk { zeros.clone() } else { coerr(zeros.narrow(0, 0, len))? };
                let row = coerr(coerr(piece.broadcast_add(&z))?.reshape(vec![1, len, width]))?;
                coerr(stream.copy_rows_from(start, &row))?;
                start += len;
            }
            stream
        };

        let mut ple_slot = 0usize;
        let trace_mem = std::env::var("SYN_QWEN4EXP_TRACE_MEM").is_ok();
        for (l, block) in self.blocks.iter().enumerate() {
            if trace_mem {
                self.trace_prefill_mem(l);
            }
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

                // Внимание — в своём блоке: его буферы (смесь, выход, впрыск —
                // по 335 МБ на чанк 16k) обязаны уйти до MoE, иначе они
                // складываются с перестановками экспертов в один пик, а
                // shadowing в Rust ничего не освобождает до конца итерации.
                let mixed = {
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
                    drop(out);
                    drop(mixed);
                    drop(piece);
                    ctx(block.mlp_hc.forward(&injected), &format!("слой {l} mlp_hc"))?
                };
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

    /// Конец отрезка слоёв линейного внимания, начатого на `l0` (не включая):
    /// до слоя разреженного внимания или слоя с PLE.
    fn segment_end(&self, l0: usize) -> usize {
        let mut e = l0 + 1;
        while e < self.blocks.len() && matches!(self.blocks[e].mixer, Mixer::Linear(_)) && self.blocks[e].ple.is_none() {
            e += 1;
        }
        e
    }

    /// Слой линейного внимания целиком (HC, GDN, HC, MoE) — без трассировки,
    /// проб и упреждения: то, что пишется в CUDA-граф.
    fn gdn_layer(
        &self,
        l: usize,
        hidden: &Tensor,
        state: &mut LayerState,
        s: usize,
        split: Option<usize>,
        snaps: &mut Vec<crate::linear_attn::GdnSnap>,
        pair_buf: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor, ModelError> {
        let block = &self.blocks[l];
        let (Mixer::Linear(la), LayerState::Linear(state)) = (&block.mixer, state) else {
            return Err(ModelError::Shape(format!("слой {l}: не линейное внимание")));
        };
        let no_inject = || ModelError::Forward(format!("слой {l}: hc без inject"));
        let mixed = block.attn_hc.forward(hidden)?;
        let out = match split {
            None => la.forward(&mixed.mixed, state, s)?,
            Some(k) => {
                let (out, snap) = la.forward_split(&mixed.mixed, state, s, k, pair_buf)?;
                snaps.push(snap);
                out
            }
        };
        let hidden = block.attn_hc.inject(&mixed.hyper, &out, mixed.inject_weights.as_ref().ok_or_else(no_inject)?)?;
        let mixed = block.mlp_hc.forward(&hidden)?;
        let out = block.moe.forward(&mixed.mixed)?;
        block.mlp_hc.inject(&mixed.hyper, &out, mixed.inject_weights.as_ref().ok_or_else(no_inject)?)
    }

    /// Слои `l0..l1` (линейное внимание) графом. `None` — отрезок ещё не готов
    /// (прогрев, ленивые части не достроены) или захват не удался: считать
    /// обычным путём. См. `seg_graph`.
    #[allow(clippy::too_many_arguments)]
    fn run_segment(
        &self,
        l0: usize,
        l1: usize,
        hidden: &Tensor,
        cache: &mut ModelCache,
        s: usize,
        split: Option<usize>,
        snaps: &mut Vec<crate::linear_attn::GdnSnap>,
    ) -> Result<Option<Tensor>, ModelError> {
        use crate::seg_graph::{capture, copy_into_fixed, like_fixed, SegGraph, SegSlot};
        let Device::Cuda(ord) = self.device else { return Ok(None) };
        let pair = split.is_some();
        for l in l0..l1 {
            let block = &self.blocks[l];
            let (Mixer::Linear(la), LayerState::Linear(st)) = (&block.mixer, &cache.layers[l]) else { return Ok(None) };
            if !la.decode_ready(st) || !block.moe.zc_built() {
                return Ok(None);
            }
        }
        let addr = |t: &Tensor| t.cuda_region().map(|r| r.0).unwrap_or(0);
        if pair {
            if cache.seg.pair.len() < self.blocks.len() {
                cache.seg.pair.resize_with(self.blocks.len(), || None);
            }
            for l in l0..l1 {
                let LayerState::Linear(st) = &cache.layers[l] else { return Ok(None) };
                let (Some(c), Some(ss)) = (st.conv_state_dev.as_ref(), st.ssm_state_dev.as_ref()) else { return Ok(None) };
                let fresh = match &cache.seg.pair[l] {
                    Some((bc, bs)) => bc.dims() != c.dims() || bs.dims() != ss.dims(),
                    None => true,
                };
                if fresh {
                    cache.seg.pair[l] = Some((like_fixed(c)?, like_fixed(ss)?));
                }
            }
        }
        let mut stamp: Vec<u64> = Vec::with_capacity(4 * (l1 - l0) + 2);
        for l in l0..l1 {
            let LayerState::Linear(st) = &cache.layers[l] else { return Ok(None) };
            stamp.push(st.conv_state_dev.as_ref().map(addr).unwrap_or(0));
            stamp.push(st.ssm_state_dev.as_ref().map(addr).unwrap_or(0));
            if pair {
                let (bc, bs) = cache.seg.pair[l].as_ref().expect("создан выше");
                stamp.push(addr(bc));
                stamp.push(addr(bs));
            }
        }
        stamp.push(synaptix_llm_common::moe_zc::ZC_GENERATION.load(std::sync::atomic::Ordering::Relaxed));
        stamp.push(l1 as u64);

        let key = (l0, s);
        let mut slot = cache.seg.slots.remove(&key).unwrap_or_default();
        if slot.stamp != stamp {
            slot = SegSlot { stamp, ..Default::default() };
        }
        let push_pair_snaps = |cache: &ModelCache, snaps: &mut Vec<crate::linear_attn::GdnSnap>| {
            for l in l0..l1 {
                snaps.push(crate::linear_attn::GdnSnap::from_bufs(cache.seg.pair[l].as_ref().expect("создан выше")));
            }
        };
        let launch = |g: &SegGraph| -> Result<(), ModelError> {
            g.graph.0.launch().map_err(|e| ModelError::Forward(format!("граф декода, слои {l0}..{l1}: запуск: {e:?}")))
        };
        if let Some(g) = &slot.graph {
            copy_into_fixed(&g.input, hidden)?;
            launch(g)?;
            crate::seg_graph::LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let out = g.output.clone();
            if pair {
                push_pair_snaps(cache, snaps);
            }
            cache.seg.slots.insert(key, slot);
            return Ok(Some(out));
        }
        if slot.failed || slot.warm < crate::seg_graph::WARM_STEPS {
            crate::seg_graph::EAGER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            slot.warm += 1;
            cache.seg.slots.insert(key, slot);
            return Ok(None);
        }
        // Захват: ядра только записываются, шаг исполнит первый запуск графа.
        let input = like_fixed(hidden)?;
        let output = like_fixed(hidden)?;
        copy_into_fixed(&input, hidden)?;
        let captured = {
            let (layers, pairs) = (&mut cache.layers, &cache.seg.pair);
            let mut scratch = Vec::new();
            capture(ord, || {
                let mut h = input.clone();
                for l in l0..l1 {
                    let buf = if pair { pairs[l].as_ref() } else { None };
                    h = self.gdn_layer(l, &h, &mut layers[l], s, split, &mut scratch, buf)?;
                }
                copy_into_fixed(&output, &h)
            })
        };
        match captured {
            Ok(graph) => {
                let g = SegGraph { graph: crate::seg_graph::GraphHandle(graph), input, output };
                launch(&g)?;
                crate::seg_graph::CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let out = g.output.clone();
                slot.graph = Some(g);
                if pair {
                    push_pair_snaps(cache, snaps);
                }
                cache.seg.slots.insert(key, slot);
                Ok(Some(out))
            }
            Err(e) => {
                eprintln!("[qwen4_exp] граф декода, слои {l0}..{l1} (строк {s}): захват не удался, обычный путь: {e}");
                slot.failed = true;
                cache.seg.slots.insert(key, slot);
                Ok(None)
            }
        }
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
        let mut pending_prefetch: Option<synaptix_llm_common::moe::AsyncPrefetch> = None;
        let seg_ok = crate::seg_graph::enabled()
            && matches!(self.device, Device::Cuda(_))
            && self.compute == DType::F16
            && ((s == 1 && split.is_none()) || (s == 2 && split == Some(1)))
            && opts.trace != Trace::Full
            && !crate::norm::profiling()
            && !predict_probe::on()
            && prefetch_k() == 0
            && !synaptix_llm_common::moe::MoeFfn::zc_host_flags();
        let mut l = 0usize;
        while l < self.blocks.len() {
            let block = &self.blocks[l];
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

            // Отрезок слоёв линейного внимания — одним CUDA-графом.
            if seg_ok && matches!(block.mixer, Mixer::Linear(_)) {
                let end = self.segment_end(l);
                if let Some(h) = self.run_segment(l, end, &hidden, cache, s, split, &mut snap_linear)? {
                    hidden = h;
                    l = end;
                    continue;
                }
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
                                la.forward_split(&mixed.mixed, state, s, k, None),
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
            let probe = s == 1 && predict_probe::on() && l + 1 < self.blocks.len();
            let predicted = if probe {
                let next = &self.blocks[l + 1].moe;
                let ids = next.predict_topk(&mixed.mixed, predict_probe::k())?;
                let miss: Vec<bool> = ids.iter().map(|e| !next.resident(*e as usize)).collect();
                Some((ids, miss))
            } else {
                None
            };
            let resident_before: Vec<bool> = if s == 1 && predict_probe::on() {
                (0..512).map(|e| block.moe.resident(e)).collect()
            } else {
                Vec::new()
            };
            // Асинхронная подкачка: эксперты, предсказанные для этого слоя на
            // прошлом, должны быть дождены до роутинга.
            if let Some(p) = pending_prefetch.take() {
                if p.layer() == block.moe.layer_id() {
                    block.moe.adopt_prefetch(p);
                }
            }
            let predicted_next = if s == 1 && prefetch_k() > 0 && l + 1 < self.blocks.len() {
                self.blocks[l + 1].moe.predict_topk(&mixed.mixed, prefetch_k()).ok()
            } else {
                None
            };
            // Zero-copy: эксперты следующего слоя подкачиваются вторым стримом
            // по этому входу, пока считается MoE этого слоя и внимание следующего.
            if s <= 4 && l + 1 < self.blocks.len() {
                ctx(self.blocks[l + 1].moe.zc_prefetch(&mixed.mixed), &format!("слой {l} упреждение"))?;
            }
            let out = stage("moe", || {
                ctx(block.moe.forward(&mixed.mixed), &format!("слой {l} moe"))
            })?;
            if let Some(ids) = predicted_next {
                pending_prefetch = self.blocks[l + 1].moe.prefetch_async(&ids, prefetch_k());
            }
            if s == 1 && predict_probe::on() {
                predict_probe::check(l, &synaptix_llm_common::moe::last_route(), &resident_before);
            }
            if let Some((ids, miss)) = predicted {
                predict_probe::stash(l + 1, ids, miss);
            }
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
            l += 1;
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
            qsa_tails: None,
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
        self.lm_head_forward(&last)
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut ModelCache) -> Result<Tensor, ModelError> {
        let hidden = self.forward_hidden(tokens, cache)?;
        self.lm_head_forward(&hidden)
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
        self.lm_head_forward(&last)
    }

    pub fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        // Декод (одна строка): NVFP4-голова без кванта активации — точнее
        // MMA-пути (W4A4): KL к эталону 0,128 → 0,115 ценой 0,27 мс на вызов.
        // `SYN_QWEN4EXP_HEAD_A16=0` — прежний путь.
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var("SYN_QWEN4EXP_HEAD_A16").as_deref() != Ok("0"))
            && hidden.rank() == 2
            && hidden.dims()[0] == 1
            && hidden.dtype() == DType::F16
        {
            if let Some(w) = self.lm_head.quant_weight() {
                match w.gemv_a16(hidden) {
                    Ok(y) => return Ok(y),
                    Err(synaptix_core::error::SynaptixError::Unsupported(_)) => {}
                    Err(e) => return Err(ModelError::Forward(format!("lm_head: {e}"))),
                }
            }
        }
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

/// Замер предсказания экспертов слоя L+1 по входу MoE слоя L
/// (`SYN_QWEN4EXP_PREDICT_PROBE=<k>`): сколько выбранных экспертов было в
/// предсказанных top-k и сколько из настоящих промахов кэша угадано.
mod predict_probe {
    use std::sync::Mutex;

    struct St {
        pending: Option<(usize, Vec<u32>, Vec<bool>)>,
        chosen: u64,
        hit: u64,
        actual_miss: u64,
        miss_predicted: u64,
        predicted_miss: u64,
        steps: u64,
    }
    static ST: Mutex<St> = Mutex::new(St {
        pending: None, chosen: 0, hit: 0, actual_miss: 0, miss_predicted: 0, predicted_miss: 0, steps: 0,
    });

    pub fn on() -> bool {
        k() > 0
    }
    pub fn k() -> usize {
        static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *K.get_or_init(|| {
            std::env::var("SYN_QWEN4EXP_PREDICT_PROBE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
        })
    }
    pub fn stash(layer: usize, ids: Vec<u32>, miss: Vec<bool>) {
        if let Ok(mut st) = ST.lock() {
            st.pending = Some((layer, ids, miss));
        }
    }
    /// Сверка после MoE слоя `layer`: `actual` — выбор роутера, промах
    /// определяется по состоянию кэша в момент предсказания.
    pub fn check(layer: usize, actual: &[u32], resident: &[bool]) {
        let Ok(mut st) = ST.lock() else { return };
        let Some((pl, ids, miss)) = st.pending.take() else { return };
        if pl != layer {
            return;
        }
        st.steps += 1;
        for (e, m) in ids.iter().zip(&miss) {
            if *m {
                st.predicted_miss += 1;
            }
            let _ = e;
        }
        for a in actual {
            st.chosen += 1;
            let pos = ids.iter().position(|e| e == a);
            if pos.is_some() {
                st.hit += 1;
            }
            // промах: эксперта не было на карте в момент предсказания
            let _ = &miss;
            let was_miss = !resident.get(*a as usize).copied().unwrap_or(true);
            if was_miss {
                st.actual_miss += 1;
                if pos.is_some() {
                    st.miss_predicted += 1;
                }
            }
        }
        if st.steps % 4800 == 0 {
            eprintln!(
                "[predict] k={} шагов {}: выбранных в предсказании {:.1}%, промахов {:.2}/шаг, из них угадано {:.1}%, подкачка предсказанных {:.2}/шаг",
                k(),
                st.steps,
                100.0 * st.hit as f64 / st.chosen.max(1) as f64,
                st.actual_miss as f64 / st.steps as f64,
                100.0 * st.miss_predicted as f64 / st.actual_miss.max(1) as f64,
                st.predicted_miss as f64 / st.steps as f64,
            );
        }
    }
}

/// Сколько экспертов слоя L+1 предсказывать и подкачивать асинхронно на
/// декоде (`SYN_QWEN4EXP_PREFETCH`, 0 — выключено).
fn prefetch_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("SYN_QWEN4EXP_PREFETCH").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    })
}
