use std::path::Path;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use std::sync::Arc;

use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink, TokenSampler};
use synaptix_llm_common::model::RopePositions;
use synaptix_llm_common::moe::{ExpertCache, ExpertCacheStats, ExpertSource};
use synaptix_llm_common::mrope;
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::config::Qwen4ExpConfig;
use crate::loader::{BundleExperts, Qwen4ExpWeights};
use crate::model::{CacheSnapshot, ModelCache, Qwen4ExpModel};
use crate::mtp::{present as mtp_present, MtpCache, MtpHead};

/// Чанк префилла. Чем он крупнее, тем меньше проходов по всей стопке
/// экспертов: любой чанк длиннее сотни токенов задевает почти все 512
/// экспертов слоя, так что стоимость префилла — это число чанков, умноженное
/// на объём экспертов. На промпте в 35k токенов чанк 16384 вместо 8192 срезает
/// подкачку с 290 до 186 ГБ, а префилл с 560 до 718 tok/s.
/// Меняется `SYN_QWEN4EXP_PREFILL_CHUNK`.
pub const DEFAULT_PREFILL_CHUNK: usize = 16384;

/// Вложение одной модальности: id заполнителя, строки эмбеддингов и
/// merged-сетки блоков в порядке появления в промпте. Сетки нужны M-RoPE:
/// без них позиции картинки одномерны.
pub struct MediaInput {
    pub pad: u32,
    pub embeds: Tensor,
    pub grids: Vec<mrope::Grid3>,
}

/// Разметка видео в промпте: сколько групп кадров, сколько токенов на
/// группу и таймкод каждой.
#[derive(Debug, Clone)]
pub struct VideoPromptInfo {
    pub groups: usize,
    pub tokens_per_group: usize,
    pub timestamps: Vec<f32>,
    /// Merged-сетка `(h, w)` одной группы кадров — для M-RoPE.
    pub grid_hw: (usize, usize),
}

impl VideoPromptInfo {
    /// Всего vision-токенов видео.
    pub fn tokens(&self) -> usize {
        self.groups * self.tokens_per_group
    }

    /// Блок промпта как у HF-процессора Qwen3-VL: на каждую группу кадров
    /// `<{t:.1} seconds><|vision_start|><|video_pad|>…<|vision_end|>`.
    /// Таймкод — обычный текст, так модель видит временну́ю шкалу и без
    /// M-RoPE по оси времени.
    pub fn prompt_block(&self) -> String {
        let mut s = String::new();
        for g in 0..self.groups {
            let ts = self.timestamps.get(g).copied().unwrap_or(0.0);
            s.push_str(&format!("<{ts:.1} seconds><|vision_start|>"));
            s.push_str(&"<|video_pad|>".repeat(self.tokens_per_group));
            s.push_str("<|vision_end|>");
        }
        s
    }
}

pub struct Qwen4ExpPipeline {
    /// Pinned-зеркало стопок экспертов в RAM (см. [`host_mirror_for`]).
    /// Первое поле — гард умирает раньше модели, а с ней и mmap-диапазонов,
    /// на которые он ссылается.
    _host_mirror: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    pub model: Qwen4ExpModel,
    pub vision: Option<synaptix_vlm_qwen3::VisionTower>,
    pub config: Qwen4ExpConfig,
    pub chat_template: Option<String>,
    pub mtp: Option<MtpHead>,
    tokenizer: Option<HfTokenizer>,
    max_seq: usize,
}

/// Сессия префикс-KV: посчитанный контекст диалога живёт между ходами, и ход
/// дописывает в него только новый хвост промпта.
///
/// Точка возврата стоит на КОНЦЕ ПРОМПТА, а не хода: chat-шаблон
/// перерисовывает реплику ассистента, поэтому сгенерированные токены новому
/// промпту префиксом не являются, а весь промпт прошлого хода — является.
/// Снимок в этой точке снимается тем же механизмом, что откатывает
/// непринятый спекулятивный драфт (`ModelCache::snapshot`/`restore`):
/// рекуррентное состояние GDN и PLE копируются, KV и ключи индексатора
/// откатываются по позиции.
pub struct Qwen4ExpSession {
    cache: ModelCache,
    /// Токены промпта, на конце которого стоит точка возврата.
    ids: Vec<u32>,
    /// Состояние в этой точке. `None` — сессия пустая.
    snap: Option<CacheSnapshot>,
    /// Медиа, вошедшие в кэшированный префикс. Токены заполнителей у двух
    /// разных картинок одинаковы, так что промпт с той же разметкой, но
    /// другим вложением на её месте префиксом считаться не должен.
    media: Vec<MediaSig>,
}

/// Отпечаток строк медиа-эмбеддингов, попавших в кэшированный префикс.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MediaSig {
    pad: u32,
    rows: usize,
    hash: u64,
}

/// Отпечатки медиа по префиксу `ids`: у каждой модальности берутся ровно те
/// строки, чьи заполнители стоят в `ids`. Хэш по битам f32 — эмбеддинги
/// приходят из кэша башни, так что одна и та же картинка даёт одни и те же
/// байты.
fn media_sigs(inputs: &[MediaInput], ids: &[u32]) -> Result<Vec<MediaSig>, PipelineError> {
    use std::hash::{Hash, Hasher};
    let mut out = Vec::new();
    for m in inputs {
        let rows = ids.iter().filter(|t| **t == m.pad).count().min(m.embeds.dims()[0]);
        if rows == 0 {
            continue;
        }
        let host = m
            .embeds
            .narrow(0, 0, rows)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.to_device(Device::Cpu))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| PipelineError::Model(format!("медиа: отпечаток эмбеддингов: {e}")))?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        rows.hash(&mut h);
        for v in &host {
            v.to_bits().hash(&mut h);
        }
        out.push(MediaSig { pad: m.pad, rows, hash: h.finish() });
    }
    Ok(out)
}

impl Qwen4ExpSession {
    /// Сколько токенов нового промпта уже посчитано. Ноль — префикс не
    /// совпал (сжали историю, поправили сообщение, сменили системный
    /// prompt): частичное совпадение бесполезно, GDN-состояние на середину
    /// не откатить.
    pub fn reusable(&self, prompt_ids: &[u32]) -> usize {
        let n = self.ids.len();
        if n == 0 || self.snap.is_none() || prompt_ids.len() <= n {
            return 0;
        }
        if prompt_ids[..n] != self.ids[..] {
            return 0;
        }
        n
    }

    /// То же для промпта с медиа: помимо токенов должны совпасть и строки
    /// вложений, вошедшие в кэшированный префикс.
    pub fn reusable_media(&self, prompt_ids: &[u32], inputs: &[MediaInput]) -> usize {
        let n = self.reusable(prompt_ids);
        if n == 0 {
            return 0;
        }
        match media_sigs(inputs, &self.ids) {
            Ok(sigs) if sigs == self.media => n,
            _ => 0,
        }
    }

    pub fn ctx_tokens(&self) -> usize {
        self.cache.max_seq
    }

    pub fn cached_tokens(&self) -> usize {
        self.ids.len()
    }

    /// Забыть префикс (смена чата/модели, сжатие истории).
    pub fn invalidate(&mut self) {
        self.ids.clear();
        self.snap = None;
        self.media.clear();
        self.cache.reset();
    }

    /// Переселить кэш сессии в host-RAM и освободить VRAM — на время
    /// вложенной генерации (см. `ModelCache::park_to_host`).
    pub fn park_to_host(&mut self) -> Result<usize, PipelineError> {
        let mut moved = self.cache.park_to_host()?;
        if let Some(snap) = self.snap.as_mut() {
            moved += snap.park_to_host();
        }
        Ok(moved)
    }

    /// Обратный переезд. Снимок GDN device-копий не восстанавливает: их
    /// пересеет из host-векторов первый же `restore`.
    pub fn unpark_to(&mut self, device: Device) -> Result<usize, PipelineError> {
        Ok(self.cache.unpark_to(device)?)
    }

    pub fn is_parked(&self) -> bool {
        self.cache.is_parked()
    }

    /// Сколько VRAM держит сессия прямо сейчас.
    pub fn device_bytes(&self) -> usize {
        self.cache.device_bytes()
            + self.snap.as_ref().map(|s| s.device_bytes()).unwrap_or(0)
    }
}

/// Состояние спекулятивного декода: голова со своим кэшем и хвост потока
/// последних впитанных позиций вместе с токенами, которые голова ещё не
/// видела. Голова обязана пройти по каждой позиции ровно один раз, иначе её
/// собственные позиции разъезжаются с позициями модели.
struct Speculation<'a> {
    head: &'a MtpHead,
    cache: MtpCache,
    stream: Tensor,
    tokens: Vec<u32>,
}

impl Qwen4ExpPipeline {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        Self::load_with_precision(path, device, PrecisionConfig::dense(dtype), None)
    }

    pub fn load_with_precision(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let weights = Arc::new(
            Qwen4ExpWeights::open(path, device, precision.compute)
                .map_err(|e| PipelineError::Load(e.to_string()))?,
        );
        let mut config = weights.config.clone();
        let tokenizer = if weights.tokenizer_json.is_empty() {
            None
        } else {
            Some(
                HfTokenizer::from_bytes(&weights.tokenizer_json)
                    .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?,
            )
        };
        let cap = max_seq.unwrap_or_else(|| config.max_position_embeddings.min(4096));
        // MoE считает вход своими под-чанками; если они мельче чанка префилла,
        // каждый под-чанк заново поднимает почти всех экспертов слоя — на
        // длинном промпте это кратный перечит всей стопки.
        config.moe.chunk = config.moe.chunk.max(prefill_chunk());
        let expert_cache = expert_cache_for(&config, device);
        let lazy = expert_cache.is_some() && weights.has_lazy_experts(0);
        let expert_source: Option<Arc<dyn ExpertSource>> = lazy
            .then(|| Arc::new(BundleExperts::new(weights.clone())) as Arc<dyn ExpertSource>);
        let host_mirror = if lazy { host_mirror_for(&weights) } else { None };
        if let Some(cache) = &expert_cache {
            eprintln!(
                "[qwen4_exp] эксперты {}, на карте кэш {:.1} ГБ",
                if lazy { "читаются из бандла по одному" } else { "в системной памяти" },
                cache.capacity_bytes() as f64 / (1 << 30) as f64
            );
        }
        let kv_reserve = model_kv_reserve(&config, cap, precision.kv);
        let mut model = Qwen4ExpModel::build_with_cache(
            &config,
            &*weights,
            device,
            precision.compute,
            precision.mlp_w,
            cap,
            &|layer| weights.ngram_rows(layer),
            expert_cache,
            expert_source,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?;
        model.set_kv_dtype(precision.kv);
        if model.kv_quantized() {
            eprintln!("[qwen4_exp] KV-кэш квантованный: mxfp8 с масштабом на 32 элемента");
        }
        if let Some(cache) = model.expert_cache() {
            fit_cache_to_vram(cache, &config, device, kv_reserve);
        }
        let chat_template = weights.chat_template.clone();
        let mtp = if speculation_on() && mtp_present(&*weights) {
            let cache = model.expert_cache().cloned();
            let source: Option<Arc<dyn ExpertSource>> = (cache.is_some()
                && weights.has_lazy_mtp_experts())
            .then(|| Arc::new(BundleExperts::new(weights.clone())) as Arc<dyn ExpertSource>);
            let head = MtpHead::load(
                &*weights,
                &config,
                device,
                precision.compute,
                precision.mlp_w,
                cache,
                source,
                config.num_hidden_layers,
            )
            .map_err(|e| PipelineError::Load(format!("mtp: {e}")))?;
            eprintln!("[qwen4_exp] спекулятивный декод: голова многотокенного предсказания поднята");
            Some(head)
        } else {
            None
        };
        if let Some(m) = &host_mirror {
            m.resume();
        }
        Ok(Self {
            _host_mirror: host_mirror,
            model,
            vision: None,
            config,
            chat_template,
            mtp,
            tokenizer,
            max_seq: cap,
        })
    }

    /// Поднять vision-башню из того же бандла. `false` — компонента нет.
    pub fn load_vision(
        &mut self,
        path: impl AsRef<Path>,
        dtype: DType,
    ) -> Result<bool, PipelineError> {
        let path = path.as_ref();
        if !synaptix_vlm_qwen3::bundle_has_vision(path) {
            return Ok(false);
        }
        let tower = synaptix_vlm_qwen3::load_from_bundle(path, self.model.device, dtype)
            .map_err(|e| PipelineError::Load(format!("vision: {e}")))?;
        self.vision = Some(tower);
        Ok(true)
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Есть ли в бандле башня — без подъёма весов (читается только заголовок).
    pub fn bundle_has_vision(path: impl AsRef<Path>) -> bool {
        synaptix_vlm_qwen3::bundle_has_vision(path)
    }

    /// Выгружает башню и возвращает её память пулу устройства.
    pub fn release_vision(&mut self) {
        if self.vision.take().is_some() {
            if let Device::Cuda(o) = self.model.device {
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
            }
        }
    }

    /// Лимиты препроцессинга под потолок vision-токенов: один токен после
    /// merge накрывает `size_factor²` пикселей, так что `max_tokens`
    /// пересчитывается в `max_pixels` для `smart_resize`. `None`/0 — дефолт.
    pub fn image_limits(
        &self,
        max_tokens: Option<usize>,
    ) -> Result<synaptix_vlm_qwen3::PreprocessLimits, PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let mut limits = synaptix_vlm_qwen3::PreprocessLimits::default();
        if let Some(n) = max_tokens.filter(|n| *n > 0) {
            let f = tower.config.size_factor();
            limits.max_pixels = limits.max_pixels.min(n * f * f);
            limits.min_pixels = limits.min_pixels.min(limits.max_pixels);
        }
        Ok(limits)
    }

    /// Картинка → строки эмбеддингов `[n, hidden]` и merged-сетка `(h, w)`.
    pub fn encode_image(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::PreprocessLimits,
    ) -> Result<(Tensor, (usize, usize)), PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let prepared =
            synaptix_vlm_qwen3::prepare_image(path, &tower.config, limits, self.model.device)
                .map_err(|e| PipelineError::Load(format!("image: {e}")))?;
        let merge = tower.config.spatial_merge_size.max(1);
        let grid = (prepared.grid.h / merge, prepared.grid.w / merge);
        let feats = no_grad(|| tower.forward(&prepared.patches, prepared.grid))
            .map_err(|e| PipelineError::Model(format!("vision forward: {e}")))?;
        let feats = feats
            .to_dtype(self.model.compute)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        Ok((feats, grid))
    }

    /// Видео → строки эмбеддингов `[n, hidden]` и merged-сетка `(t, h, w)`
    /// всего видео одним блоком заполнителей (путь CLI: промпт без таймкодов).
    pub fn encode_video(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::VideoLimits,
    ) -> Result<(Tensor, mrope::Grid3), PipelineError> {
        let (feats, info) = self.encode_video_limited(path, limits)?;
        let grid =
            mrope::Grid3 { t: info.groups, h: info.grid_hw.0, w: info.grid_hw.1 };
        Ok((feats, grid))
    }

    /// Видео → эмбеддинги и разметка промпта по группам кадров: каждая группа
    /// идёт своим блоком заполнителей с таймкодом, как у HF-процессора
    /// Qwen3-VL (см. [`VideoPromptInfo::prompt_block`]).
    pub fn encode_video_with_info(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Tensor, VideoPromptInfo), PipelineError> {
        self.encode_video_limited(path, synaptix_vlm_qwen3::VideoLimits::default())
    }

    /// Кадры группируются по `temporal_patch_size`: группа кадров даёт один
    /// набор патчей и один блок заполнителей в промпте.
    pub fn encode_video_limited(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::VideoLimits,
    ) -> Result<(Tensor, VideoPromptInfo), PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let synaptix_vlm_qwen3::PreparedVideo { patches, grid, group_timestamps } =
            synaptix_vlm_qwen3::prepare_video(path, &tower.config, limits, self.model.device)
                .map_err(|e| PipelineError::Load(format!("video: {e}")))?;
        let feats = no_grad(|| tower.forward(&patches, grid))
            .map_err(|e| PipelineError::Model(format!("vision forward: {e}")))?;
        let feats = feats
            .to_dtype(self.model.compute)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let merge = tower.config.spatial_merge_size.max(1);
        let info = VideoPromptInfo {
            groups: grid.t,
            tokens_per_group: (grid.h * grid.w) / tower.config.merge_unit(),
            timestamps: group_timestamps,
            grid_hw: (grid.h / merge, grid.w / merge),
        };
        Ok((feats, info))
    }

    /// Сколько токенов-заполнителей займёт картинка в промпте.
    pub fn image_token_count(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::PreprocessLimits,
    ) -> Result<usize, PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let prepared =
            synaptix_vlm_qwen3::prepare_image(path, &tower.config, limits, Device::Cpu)
                .map_err(|e| PipelineError::Load(format!("image: {e}")))?;
        let merge = tower.config.spatial_merge_size.max(1);
        Ok((prepared.grid.h / merge) * (prepared.grid.w / merge))
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PipelineError::Tokenize("бандл без tokenizer.json".into()))?;
        let enc = tokenizer
            .encode(prompt, false)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PipelineError::Tokenize("бандл без tokenizer.json".into()))?;
        tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    /// Отдать память карты, которую держат резидентные эксперты. Нужно при
    /// выгрузке модели: кэш живёт в самой модели, но её `Drop` может
    /// задержаться, пока кто-то держит ссылку, а память освободить надо сразу.
    pub fn release_device_caches(&self) {
        if let Some(cache) = self.model.expert_cache() {
            cache.clear();
        }
    }

    fn draft(&self, spec: &mut Speculation<'_>) -> Result<u32, ModelError> {
        let embeds = self.model.embed_tokens(&spec.tokens)?;
        let hidden = spec
            .head
            .forward(&spec.stream, &embeds, &mut spec.cache, self.model.rope())?;
        let logits = self.model.lm_head_forward(&last_row(&hidden)?)?;
        let best = logits
            .flatten_all()
            .and_then(|t| t.argmax(0))
            .and_then(|t| t.to_device(Device::Cpu))
            .and_then(|t| t.to_dtype(DType::U32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<u32>())
            .map_err(|e| ModelError::Forward(format!("драфт: {e}")))?;
        best.first()
            .copied()
            .ok_or_else(|| ModelError::Forward("драфт: пустой argmax".into()))
    }

    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.model.expert_cache_stats()
    }

    pub fn rope_capacity(&self) -> usize {
        self.max_seq
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        let cfg = &self.config;
        let elem = (self.model.compute.size_in_bits() / 8).max(1);
        let qsa_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, crate::config::LayerType::Qsa))
            .count();
        let kv = if self.model.kv_quantized() {
            2 * cfg.num_key_value_heads * cfg.head_dim * 33 / 32
        } else {
            2 * cfg.num_key_value_heads * cfg.head_dim * elem
        };
        let indexer = if cfg.indexer.compress_ratio > 0 {
            cfg.indexer.head_dim * elem / cfg.indexer.compress_ratio
        } else {
            0
        };
        qsa_layers * (kv + indexer)
    }

    pub fn kv_fixed_bytes(&self, batch: usize, max_seq: usize) -> usize {
        let cfg = &self.config;
        let elem = 4;
        let linear_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, crate::config::LayerType::Linear))
            .count();
        let state = cfg.linear.num_value_heads * cfg.linear.key_head_dim * cfg.linear.value_head_dim * elem;
        let conv = (cfg.linear.conv_kernel - 1) * cfg.linear.conv_dim() * elem;
        batch * (linear_layers * (state + conv) + max_seq * self.kv_bytes_per_token())
    }

    pub fn make_cache(&self, tokens: usize) -> Result<ModelCache, PipelineError> {
        self.model
            .make_cache(tokens.max(1).min(self.max_seq))
            .map_err(PipelineError::from)
    }

    fn prepare(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_ids = self.config.eos_token_ids.clone();
        }
        // Цена префилла здесь — число чанков, умноженное на объём экспертов:
        // любой чанк длиннее сотни токенов задевает почти все 512 экспертов
        // слоя, поэтому мелкий чанк означает лишний полный прогон весов через
        // шину. Внешнюю настройку (её ставят ради пика VRAM у плотных
        // моделей) поднимаем до своего минимума, но не опускаем ниже неё.
        let want = prefill_chunk();
        if cfg.prefill_batch < want {
            if cfg.prefill_batch > 0 {
                eprintln!(
                    "[qwen4_exp] чанк префилла поднят с {} до {want}: на мелких чанках \
                     эксперты перечитываются целиком на каждый",
                    cfg.prefill_batch
                );
            }
            cfg.prefill_batch = want;
        }
        cfg
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        self.generate_streaming(prompt_ids, cfg, &mut |_: u32| true)
    }

    pub fn generate_text(&self, prompt: &str, cfg: GenerationConfig) -> Result<String, PipelineError> {
        let ids = self.encode(prompt)?;
        let (out, _) = self.generate(&ids, cfg)?;
        self.decode(&out)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        self.generate_media_streaming(prompt_ids, &[], cfg, sink)
    }

    /// Таблицы cos/sin M-RoPE на промпт и на `tail` позиций после него.
    /// Хвост нужен декоду: у токенов картинки позиции трёхмерны, поэтому
    /// после блока счётчик позиций отстаёт от индекса токена, и брать их
    /// приходится по той же таблице — в том числе индексатору, который
    /// поворачивает ключи блоков по индексу их первого токена.
    ///
    /// `None` — позиции одномерные: конфиг без `mrope_section`, промпт без
    /// медиа, у вложения нет сеток либо `SYN_QWEN4EXP_MROPE=0`.
    fn mrope_tables(
        &self,
        prompt_ids: &[u32],
        inputs: &[MediaInput],
        tail: usize,
    ) -> Result<Option<(Tensor, Tensor)>, PipelineError> {
        let Some(section) = self.config.rope.mrope_section else {
            return Ok(None);
        };
        if inputs.is_empty() || !mrope_on() || inputs.iter().any(|m| m.grids.is_empty()) {
            return Ok(None);
        }
        let runs: Vec<mrope::MediaRuns> = inputs
            .iter()
            .map(|m| mrope::MediaRuns { pad: m.pad, grids: &m.grids })
            .collect();
        let mut positions = mrope::positions_3d(prompt_ids, &runs)
            .map_err(|e| PipelineError::Model(format!("mrope: {e}")))?;
        let next = positions.max_pos + 1;
        positions.pos.extend((0..tail as u32).map(|k| [next + k, next + k, next + k]));
        let inv = self.config.rope.inv_freqs();
        let (cos, sin) = mrope::rope_tables(
            &positions.pos,
            &inv,
            &section,
            self.config.rope.mrope_interleaved,
        );
        let rows = positions.pos.len();
        let half = inv.len();
        let device = self.model.device;
        let make = |v: Vec<f32>| {
            Tensor::from_vec(v, vec![rows, half], device)
                .map_err(|e| PipelineError::Model(e.to_string()))
        };
        Ok(Some((make(cos)?, make(sin)?)))
    }

    /// Пустая сессия префикс-KV на `ctx_tokens` токенов контекста.
    pub fn new_session(&self, ctx_tokens: usize) -> Result<Qwen4ExpSession, PipelineError> {
        Ok(Qwen4ExpSession {
            cache: self.make_cache(ctx_tokens)?,
            ids: Vec::new(),
            snap: None,
            media: Vec::new(),
        })
    }

    /// Как [`Self::generate_streaming`], но с префикс-KV: всё, что уже лежит
    /// в `session`, не считается заново. Возвращает, сколько токенов промпта
    /// удалось переиспользовать.
    pub fn generate_cached_streaming(
        &self,
        session: &mut Qwen4ExpSession,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        self.run_stream(prompt_ids, &[], cfg, sink, Some(session))
    }

    /// Как [`Self::generate_cached_streaming`], но с медиа-вложениями:
    /// строки заполнителей, посчитанные прошлым ходом, живут в кэше вместе
    /// с остальным префиксом — префиллится только хвост.
    pub fn generate_cached_media_streaming(
        &self,
        session: &mut Qwen4ExpSession,
        prompt_ids: &[u32],
        inputs: &[MediaInput],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        self.run_stream(prompt_ids, inputs, cfg, sink, Some(session))
    }

    /// Генерация с медиа-вложениями: `media` — пары «id заполнителя → строки
    /// эмбеддингов», строки расходуются по порядку появления заполнителя.
    pub fn generate_media_streaming(
        &self,
        prompt_ids: &[u32],
        inputs: &[MediaInput],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        self.run_stream(prompt_ids, inputs, cfg, sink, None)
            .map(|(out, stats, _)| (out, stats))
    }

    /// Общее тело генерации. `session` задан — ход идёт по префикс-KV:
    /// префиллится только хвост промпта, а на его конце снимается новая
    /// точка возврата.
    fn run_stream(
        &self,
        prompt_ids: &[u32],
        inputs: &[MediaInput],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
        mut session: Option<&mut Qwen4ExpSession>,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let media: Vec<(u32, Tensor)> =
            inputs.iter().map(|m| (m.pad, m.embeds.clone())).collect();
        let media = media.as_slice();
        let cfg = self.prepare(cfg);
        let budget = prompt_ids.len() + cfg.max_new_tokens;
        if budget > self.max_seq {
            return Err(PipelineError::Model(format!(
                "промпт {} + {} новых токенов больше окна {}",
                prompt_ids.len(),
                cfg.max_new_tokens,
                self.max_seq
            )));
        }
        let tables = self.mrope_tables(prompt_ids, inputs, cfg.max_new_tokens + 8)?;
        let pos = match &tables {
            Some((cos, sin)) => RopePositions::Tables { cos, sin },
            None => RopePositions::Sequential,
        };
        let (prefill_pos, decode_pos) = (pos, pos);

        // Кэш хода: свой одноразовый или тот, что живёт в сессии между
        // ходами. Медиа-промпт сессию не выключает: состояние заполнителей
        // лежит в кэше как у любых токенов, а подмену вложения при тех же
        // токенах ловит отпечаток строк (`reusable_media`).
        if let Some(s) = session.as_deref_mut() {
            // Ход не влезает в кэш сессии — пересоздаём его под новый
            // размер, префикс при этом теряется.
            if budget > s.cache.max_seq {
                s.cache = self.make_cache(budget)?;
                s.ids.clear();
                s.snap = None;
                s.media.clear();
            }
        }
        let reuse = session
            .as_deref()
            .map(|s| s.reusable_media(prompt_ids, inputs))
            .unwrap_or(0);
        // Флаг снимаем заранее: дальше `session` занята заимствованием кэша.
        let want_session = session.is_some();
        let mut owned_cache;
        let cache: &mut ModelCache = match session.as_deref_mut() {
            Some(s) => {
                match (reuse > 0).then(|| s.snap.as_ref()).flatten() {
                    Some(snap) => s.cache.restore(snap).map_err(PipelineError::from)?,
                    None => s.cache.reset(),
                }
                &mut s.cache
            }
            None => {
                owned_cache = self.make_cache(budget)?;
                &mut owned_cache
            }
        };
        let mut sampler = TokenSampler::new(&cfg, prompt_ids);
        let eos: Vec<u32> = if cfg.eos_token_ids.is_empty() {
            cfg.eos_token_id.into_iter().collect()
        } else {
            cfg.eos_token_ids.clone()
        };

        // Префилл обходит почти всю стопку экспертов: пусть поднятое им живёт
        // отдельно и уходит по окончании, а прогретое прошлым ходом остаётся —
        // декоду оно и пригодится. И ужимаем кэш ПОД длину промпта: активации
        // префилла считаются после экспертов, и ловить на них OOM дороже, чем
        // подкачать пару сотен экспертов заново.
        // Промпт длиннее чанка выгоднее считать слой за слоем: эксперты слоя
        // тогда поднимаются на карту один раз на весь промпт, а не на каждый
        // чанк. Цена — поток всех токенов в памяти, поэтому короткие промпты
        // идут прежним путём.
        // Хвост промпта: всё до `reuse` уже посчитано прошлым ходом и лежит
        // в кэше сессии. Позиции при этом абсолютные — `cache.seq_len`
        // указывает на границу.
        let fresh = &prompt_ids[reuse..];
        // Хвосту достаются только те строки медиа, чьи заполнители в нём.
        let tail_media = media_for_chunk(prompt_ids, media, reuse, fresh.len())?;
        let by_layers = fresh.len() > cfg.prefill_batch
            && self.model.expert_cache().is_some()
            && layer_major();
        // Чанк послойного прохода — по свободной VRAM: поток всех токенов
        // уже заказан, и если под активации чанка 16k места не остаётся,
        // чанк уменьшается, а не кэш экспертов ужимается ниже слоя.
        let chunk = if by_layers {
            layer_major_chunk(&self.config, self.model.device, cfg.prefill_batch, fresh.len())
        } else {
            cfg.prefill_batch
        };
        if let Some(cache) = self.model.expert_cache() {
            // Послойный префилл держит поток всех токенов хвоста; если его не
            // учесть, аллокатор упрётся в OOM посреди слоя и `reclaim` ужмёт
            // кэш до минимума — дальше каждый чанк перечитывает всю стопку.
            let stream_tokens = if by_layers { fresh.len() } else { 0 };
            cache.fit_to_vram(activation_reserve_prefill(
                &self.config,
                fresh.len().min(chunk),
                stream_tokens,
            ));
            cache.set_scratch_mode(true);
        }
        let prefill_start = Instant::now();
        let want_stream = self.mtp.is_some();
        let mut tail_stream: Option<Tensor> = None;
        // Страховка послойного прохода: оценка чанка по свободной VRAM — лишь
        // оценка, а OOM посреди слоя иначе роняет весь ход (synthos повторяет
        // его с тем же чанком). Откатываем кэш к состоянию до префилла и
        // идём чанком вдвое меньше.
        let mut chunk = chunk;
        let mut logits = if by_layers {
            let pre_prefill = cache.snapshot().map_err(PipelineError::from)?;
            loop {
                let attempt = no_grad(|| -> Result<_, ModelError> {
                    let (hidden, stream) = self.model.prefill_by_layers(
                        fresh,
                        &tail_media,
                        cache,
                        chunk,
                        prefill_pos,
                    )?;
                    if want_stream {
                        tail_stream = Some(stream);
                    }
                    self.model.lm_head_forward(&hidden)
                });
                match attempt {
                    Ok(l) => break l,
                    Err(e) if chunk > MIN_LAYER_CHUNK && is_oom(&e) => {
                        cache.restore(&pre_prefill).map_err(PipelineError::from)?;
                        tail_stream = None;
                        let next = (chunk / 2).max(MIN_LAYER_CHUNK);
                        eprintln!(
                            "[qwen4_exp] OOM в послойном префилле на чанке {chunk} ({e}); \
                             откат кэша, повтор чанком {next}"
                        );
                        if let Some(ec) = self.model.expert_cache() {
                            ec.clear_scratch();
                            if let Device::Cuda(ord) = self.model.device {
                                let _ = synaptix_core::device::cuda::synchronize_all(ord);
                                let _ = synaptix_core::device::cuda::trim_activations_pool(ord);
                            }
                            ec.fit_to_vram(activation_reserve_prefill(
                                &self.config,
                                fresh.len().min(next),
                                fresh.len(),
                            ));
                        }
                        chunk = next;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        } else {
            no_grad(|| -> Result<_, ModelError> {
                let mut last = None;
                let mut offset = reuse;
                while offset < prompt_ids.len() {
                    let take = cfg.prefill_batch.min(prompt_ids.len() - offset);
                    let chunk = &prompt_ids[offset..offset + take];
                    let slice = media_for_chunk(prompt_ids, media, offset, take)?;
                    let done = offset + take >= prompt_ids.len();
                    if want_stream && done {
                        let (hidden, stream) = self.model.forward_media_with_stream(
                            chunk,
                            &slice,
                            cache,
                            prefill_pos,
                        )?;
                        tail_stream = Some(last_row(&stream)?);
                        last = Some(self.model.lm_head_forward(&last_row(&hidden)?)?);
                    } else {
                        last = Some(self.model.forward_media_last(
                            chunk,
                            &slice,
                            cache,
                            prefill_pos,
                        )?);
                    }
                    offset += take;
                }
                last.ok_or_else(|| ModelError::Forward("пустой префилл".into()))
            })?
        };
        // Точка возврата для следующего хода — ровно здесь, до декода.
        let boundary = match want_session {
            true => Some(cache.snapshot().map_err(PipelineError::from)?),
            false => None,
        };
        let prefill_ms = prefill_start.elapsed().as_millis();
        if let Some(cache) = self.model.expert_cache() {
            cache.set_scratch_mode(false);
            cache.clear_scratch();
            // Пул активаций после длинного префилла держит гигабайты слабины,
            // и `fit_to_vram` видел бы их как занятые: после 80k кэш замирал
            // на 5 ГБ при потолке 12. Отдаём слабину драйверу до подгонки.
            if let Device::Cuda(ord) = self.model.device {
                let _ = synaptix_core::device::cuda::synchronize_all(ord);
                let _ = synaptix_core::device::cuda::trim_activations_pool(ord);
            }
            // Пик хода позади: декод считает по горстке токенов, и кэшу можно
            // вернуться к потолку — на нём и держится скорость генерации.
            cache.fit_to_vram(activation_reserve(&self.config, DECODE_TOKENS));
        }

        let decode_start = Instant::now();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        let mut spec = match (&self.mtp, tail_stream) {
            (Some(head), Some(stream)) => Some(Speculation {
                head,
                cache: head
                    .make_cache(&self.config, budget + 8, self.model.device, self.model.compute)
                    .map_err(PipelineError::from)?,
                stream,
                tokens: Vec::new(),
            }),
            _ => None,
        };
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        let mut runs = 0usize;

        let mut token = sampler.sample(&logits)?;
        loop {
            out.push(token);
            if !sink.on_token(token) || eos.contains(&token) || out.len() >= cfg.max_new_tokens {
                break;
            }
            if cache.seq_len >= self.max_seq {
                break;
            }
            if cache.seq_len + 2 > self.max_seq {
                spec = None;
            }
            let draft = match spec.as_mut() {
                Some(spec) => {
                    if spec.tokens.len() < spec.stream.dims()[0] {
                        spec.tokens.push(token);
                    }
                    Some(no_grad(|| self.draft(spec))?)
                }
                None => None,
            };
            let Some(draft) = draft else {
                runs += 1;
                logits = no_grad(|| self.model.forward_last_pos(&[token], cache, decode_pos))?;
                token = sampler.sample(&logits)?;
                continue;
            };

            drafted += 1;
            runs += 1;
            let (hidden, stream, snap) =
                no_grad(|| self.model.forward_pair(&[token, draft], cache, decode_pos))?;
            let first = no_grad(|| self.model.lm_head_forward(&row(&hidden, 0)?))?;
            let next = sampler.sample(&first)?;
            let spec = spec.as_mut().expect("драфт без спекуляции");
            if next != draft {
                cache.restore(&snap).map_err(PipelineError::from)?;
                spec.stream = row(&stream, 0)?;
                spec.tokens = vec![next];
                token = next;
                continue;
            }

            accepted += 1;
            out.push(next);
            if !sink.on_token(next) || eos.contains(&next) || out.len() >= cfg.max_new_tokens {
                break;
            }
            let second = no_grad(|| self.model.lm_head_forward(&row(&hidden, 1)?))?;
            token = sampler.sample(&second)?;
            spec.stream = stream;
            spec.tokens = vec![next, token];
        }
        let decode_ms = decode_start.elapsed().as_millis();
        if let (Some(cache), Some(st)) = (self.model.expert_cache(), self.model.expert_cache_stats())
        {
            let gb = |x: usize| x as f64 / (1u64 << 30) as f64;
            let (rsv, used) = match self.model.device {
                Device::Cuda(o) => {
                    synaptix_core::device::cuda::experts_pool_stats(o).unwrap_or((0, 0))
                }
                _ => (0, 0),
            };
            eprintln!(
                "[qwen4_exp] кэш экспертов: {} шт / {:.2} ГБ при потолке {:.2} ГБ (пул {:.2}/{:.2} ГБ),                  попаданий {}, промахов {}, подкачано {} за {} мс",
                st.resident,
                gb(st.bytes),
                gb(cache.capacity_bytes()),
                gb(used as usize),
                gb(rsv as usize),
                st.hits,
                st.misses,
                st.fetched,
                st.fetch_millis,
            );
        }
        if drafted > 0 {
            eprintln!(
                "[qwen4_exp] спекуляция: принято {accepted} из {drafted} ({:.0}%), токенов за прогон {:.2}",
                100.0 * accepted as f32 / drafted as f32,
                out.len() as f32 / runs.max(1) as f32
            );
        }

        // Кэш больше не нужен как заимствование — записываем точку возврата
        // в сессию. Порядок важен: `boundary` снят до декода, а декод успел
        // дописать в кэш свои токены; сравнение префикса на следующем ходу
        // идёт по `ids`, откат — по снимку.
        if let (Some(s), Some(snap)) = (session, boundary) {
            s.ids = prompt_ids.to_vec();
            s.snap = Some(snap);
            s.media = media_sigs(inputs, prompt_ids)?;
        }

        Ok((
            out.clone(),
            GenerationStats {
                prompt_tokens: prompt_ids.len(),
                new_tokens: out.len(),
                prefill_ms,
                decode_ms,
            },
            reuse,
        ))
    }
}

fn row(t: &Tensor, i: usize) -> Result<Tensor, ModelError> {
    t.narrow(0, i, 1)
        .and_then(|x| x.contiguous())
        .map_err(|e| ModelError::Forward(format!("строка {i}: {e}")))
}

fn last_row(t: &Tensor) -> Result<Tensor, ModelError> {
    row(t, t.dims()[0] - 1)
}

/// Считать ли длинный префилл слой за слоем. `SYN_QWEN4EXP_LAYER_MAJOR=0`
/// возвращает прежний порядок (чанк целиком через все слои).
fn layer_major() -> bool {
    std::env::var("SYN_QWEN4EXP_LAYER_MAJOR").map(|v| v.trim() != "0").unwrap_or(true)
}

fn speculation_on() -> bool {
    std::env::var("SYN_QWEN4EXP_SPEC").map(|v| v.trim() != "0").unwrap_or(false)
}

/// Строки медиа-эмбеддингов, попадающие в чанк префилла `[offset, offset+len)`.
/// Заполнители нумеруются по всему промпту, поэтому при нарезке нужно взять
/// ровно те строки, чьи слоты попали в чанк.
fn media_for_chunk(
    prompt: &[u32],
    media: &[(u32, Tensor)],
    offset: usize,
    len: usize,
) -> Result<Vec<(u32, Tensor)>, ModelError> {
    if media.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(media.len());
    for (pad, feats) in media {
        let before = prompt[..offset].iter().filter(|t| *t == pad).count();
        let inside = prompt[offset..offset + len].iter().filter(|t| *t == pad).count();
        if inside == 0 {
            continue;
        }
        let rows = feats
            .narrow(0, before, inside)
            .and_then(|t| t.contiguous())
            .map_err(|e| ModelError::Forward(format!("медиа: срез строк: {e}")))?;
        out.push((*pad, rows));
    }
    Ok(out)
}

/// Мультимодальные позиции RoPE. `SYN_QWEN4EXP_MROPE=0` возвращает
/// одномерные — для сверки с прежним поведением.
fn mrope_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYN_QWEN4EXP_MROPE").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

fn prefill_chunk() -> usize {
    std::env::var("SYN_QWEN4EXP_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_PREFILL_CHUNK)
}

/// Сколько VRAM уйдёт под KV и ключи индексатора при полном окне.
fn model_kv_reserve(cfg: &Qwen4ExpConfig, max_seq: usize, kv: DType) -> usize {
    let qsa = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, crate::config::LayerType::Qsa))
        .count();
    // Квантованный KV — байт на элемент плюс масштаб на каждые 32.
    let kv_elem = if kv == DType::MXFP8 { 33 } else { 32 * 2 };
    let per_token = 2 * cfg.num_key_value_heads * cfg.head_dim * kv_elem / 32
        + if cfg.indexer.compress_ratio > 0 {
            cfg.indexer.head_dim * 2 / cfg.indexer.compress_ratio
        } else {
            0
        };
    qsa * max_seq * per_token
}

/// Сколько VRAM обязано остаться свободным под активации хода на `tokens`
/// токенов разом.
///
/// Префилл держит поток скрытых состояний на ВЕСЬ промпт сразу: `[T,
/// hc_count, hidden]` в f32 — у Qwen3.8-Flash-Next это 40 КБ на токен, и ещё
/// столько же уходит на копии между слоями. MoE поверх этого собирает и
/// разбирает `T × top_k` строк. Всё это считается ПОСЛЕ того, как эксперты
/// уже заняли своё, поэтому кэш обязан ужаться заранее, а не ловить OOM.
/// Сколько токенов разом считает декод (шаг + спекулятивный черновик).
const DECODE_TOKENS: usize = 8;

/// Нижняя граница чанка послойного префилла: мельче — групповому GEMM
/// экспертов достаётся по три десятка строк, и префилл встаёт на запусках.
const MIN_LAYER_CHUNK: usize = 4096;

/// Резерв под префилл: активации чанка плюс поток hyper-connections всех
/// `stream_tokens` токенов послойного прохода (одна копия F16, чанки
/// копируются в него и из него) и эмбеддинги промпта, живущие, пока поток
/// собирается. Раньше здесь закладывались две копии потока — и ровно
/// столько `prefill_by_layers` держал по факту, так что на 146k токенов
/// кэшу экспертов оставалось меньше группы.
fn activation_reserve_prefill(cfg: &Qwen4ExpConfig, chunk_tokens: usize, stream_tokens: usize) -> usize {
    let per_token = cfg.hc_count.max(1) * cfg.hidden_size * 2 + cfg.hidden_size * 2;
    let stream = stream_tokens.saturating_mul(per_token);
    activation_reserve(cfg, chunk_tokens).saturating_add(stream)
}

/// Чанк послойного префилла под свободную VRAM. На 24 ГБ при промпте у cap
/// модели (262k) поток hyper-connections и KV забирают ≈9 ГБ, и на
/// активации чанка 16k (≈7 ГБ с запасом) места уже нет: кэш экспертов
/// ужимался до минимума, а пул активаций всё равно упирался в OOM на слое
/// 4 (журнал 11.09.2026). Чанк делим пополам, пока активации чанка вместе
/// с потоком не влезают в свободное за вычетом слоя экспертов (меньше слоя
/// в кэше — и каждый чанк перечитывает стопку). Замер 250k: чанк 8192
/// проходит с 0,1 ГБ запаса, 16384 падает; по этой формуле выходит 4096.
/// Нижняя граница 4096: мельче — групповому GEMM экспертов достаётся по
/// три десятка строк, и префилл встаёт на запусках ядер.
fn layer_major_chunk(cfg: &Qwen4ExpConfig, device: Device, want: usize, stream_tokens: usize) -> usize {
    // `SYN_QWEN4EXP_LAYER_CHUNK` — чанк в обход оценки (для A/B и проверки
    // отката по OOM).
    if let Some(forced) = std::env::var("SYN_QWEN4EXP_LAYER_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        return forced;
    }
    if want <= MIN_LAYER_CHUNK {
        return want;
    }
    let Device::Cuda(ord) = device else { return want };
    let Ok((free, _)) = synaptix_core::device::cuda::mem_info(ord) else {
        return want;
    };
    // Кэш экспертов может ужаться до своего минимума — его slab'ы тоже наши.
    let cache_min: usize = 1 << 30;
    let room =
        free + synaptix_core::memory::expert_arena::reserved_bytes().saturating_sub(cache_min);
    // Слой экспертов NVFP4: три проекции по полбайта на вес плюс масштабы.
    let layer_experts =
        cfg.moe.num_experts * (cfg.hidden_size * cfg.moe.moe_intermediate_size * 3 / 2) * 17 / 16;
    // Поток на все токены (эмбеддинги живут только до цикла по слоям, к
    // чанкам они уже отпущены).
    let stream = stream_tokens.saturating_mul(cfg.hc_count.max(1) * cfg.hidden_size * 2);
    // Накладные, которых нет в учёте: арена держит slab'ы под огрызки и под
    // группу GEMM, вынутую из кэша по `Arc` (2,5 ГБ slab'ов при кэше в 1 ГБ
    // на 250k), пул активаций не переиспользует часть слабины.
    let overhead: usize = 3 << 30;
    let budget = room.saturating_sub(layer_experts + stream + overhead);
    let mut chunk = want;
    while chunk > MIN_LAYER_CHUNK && activation_reserve(cfg, chunk) > budget {
        chunk /= 2;
    }
    let chunk = chunk.max(MIN_LAYER_CHUNK);
    if chunk < want {
        eprintln!(
            "[qwen4_exp] чанк послойного префилла {want} → {chunk}: на активации чанка \
             остаётся {:.1} ГБ (свободно {:.1} ГБ, поток {} ток {:.1} ГБ)",
            budget as f64 / (1u64 << 30) as f64,
            room as f64 / (1u64 << 30) as f64,
            stream_tokens,
            stream as f64 / (1u64 << 30) as f64,
        );
    }
    chunk
}

/// Ошибка аллокатора CUDA о нехватке памяти (в любой из его формулировок).
fn is_oom<E: std::fmt::Display>(e: &E) -> bool {
    let s = e.to_string();
    s.contains("OUT_OF_MEMORY") || s.contains("out of memory") || s.contains(": OOM")
}

fn activation_reserve(cfg: &Qwen4ExpConfig, tokens: usize) -> usize {
    let hidden = cfg.hidden_size;
    // поток hyper-connections (f32) с копиями + перестановки MoE: сбор строк,
    // выходы экспертов, их сборка, взвешивание и обратная перестановка —
    // пять живых копий `T × top_k × hidden` разом.
    let per_token = cfg.hc_count.max(1) * hidden * 4 * 3
        + cfg.moe.num_experts_per_tok * hidden * 2 * 5;
    const FLOOR: usize = 2 << 30;
    FLOOR + tokens.saturating_mul(per_token)
}

/// Подогнать ёмкость кэша экспертов под то, что реально осталось на карте
/// после подъёма весов.
///
/// Ужимается именно ЁМКОСТЬ, а не потолок: расклад «свободно минус KV на всё
/// окно минус активации самого длинного чанка префилла» — это худший случай
/// префилла, а на декоде свободно втрое больше, и кэш там и есть источник
/// скорости. Пока это опускало потолок навсегда, кэш оставался ужатым на всю
/// жизнь модели: в чате он замирал на 7 ГБ при девяти свободных, эксперты
/// перечитывались с хоста каждый шаг, и декод шёл вдвое медленнее замеров
/// CLI на том же бандле. Потолок остаётся тем, что заказан настройкой, а
/// пофазный `fit_to_vram` в `generate_media_streaming` двигает ёмкость в обе
/// стороны; на OOM аллокатор всё равно заберёт своё через `Reclaimable`.
fn fit_cache_to_vram(cache: &Arc<ExpertCache>, cfg: &Qwen4ExpConfig, device: Device, kv_reserve: usize) {
    let Device::Cuda(ordinal) = device else { return };
    let Ok((free, _total)) = synaptix_core::device::cuda::mem_info(ordinal) else {
        return;
    };
    let reserve = kv_reserve + activation_reserve(cfg, prefill_chunk());
    let capacity = cache.fit_to_vram(reserve);
    if capacity >= cache.ceiling_bytes() {
        return;
    }
    let gb = |x: usize| x as f64 / (1u64 << 30) as f64;
    eprintln!(
        "[qwen4_exp] кэш экспертов на старте {:.1} ГБ из потолка {:.1} ГБ: свободно {:.1} ГБ, \
         под KV всего окна и активации префилла нужно {:.1} ГБ (на декоде ёмкость вернётся)",
        gb(capacity),
        gb(cache.ceiling_bytes()),
        gb(free),
        gb(reserve),
    );
}

/// Кэш резидентных экспертов: на CUDA держим часть экспертов на карте, всё
/// остальное — в системной памяти. Размер задаётся
/// Сколько байт RAM свободно под зеркало (MemAvailable из /proc/meminfo).
fn mem_available_bytes() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Pinned-зеркало стопок экспертов в RAM. Подкачка эксперта из mmap идёт
/// «страничный кэш → копия в pinned-буфер → DMA» и упирается в 8–12 ГБ/с;
/// из зеркала DMA идёт напрямую (~40 ГБ/с). Регистрировать страницы mmap
/// как pinned (`cuMemHostRegister`) драйвер на этой платформе не даёт, потому
/// копия. Бюджет — `SYN_QWEN4EXP_HOST_MIRROR_GB` (0 — выключить), по умолчанию
/// всё, что влезает в MemAvailable минус запас 10 ГБ; скопированные страницы
/// mmap отдаются системе, так что RAM в сумме растёт лишь на то, чего в
/// страничном кэше не было. Копировщики работают фоном: пока диапазон не
/// догнан, подкачка идёт прежним путём.
fn host_mirror_for(
    weights: &Qwen4ExpWeights,
) -> Option<synaptix_core::device::cuda::OffloadPinCacheGuard> {
    use synaptix_core::device::cuda::{offload_pin_cache_active, OffloadPinCacheGuard};
    const GB: f64 = (1u64 << 30) as f64;
    let ranges = weights.expert_blob_ranges();
    if ranges.is_empty() || offload_pin_cache_active() {
        return None;
    }
    let total: usize = ranges.iter().map(|r| r.bytes.len()).sum();
    let avail = mem_available_bytes().unwrap_or(0);
    let budget = match std::env::var("SYN_QWEN4EXP_HOST_MIRROR_GB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        Some(gb) if gb <= 0.0 => return None,
        Some(gb) => (gb * GB) as usize,
        None => avail.saturating_sub(10 << 30),
    };
    let budget = budget.min(total);
    if budget < (2usize << 30) {
        eprintln!(
            "[qwen4_exp] зеркало экспертов в RAM не поднято: доступно {:.1} ГБ, стопки {:.1} ГБ",
            avail as f64 / GB,
            total as f64 / GB
        );
        return None;
    }
    let workers = std::env::var("SYN_QWEN4EXP_HOST_MIRROR_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, 16);
    eprintln!(
        "[qwen4_exp] зеркало экспертов в RAM: до {:.1} ГБ из {:.1} ГБ (RAM доступно {:.1} ГБ), \
         копировщиков {workers}",
        budget as f64 / GB,
        total as f64 / GB,
        avail as f64 / GB
    );
    // Старт на паузе: заполнение читает весь бандл и отняло бы NVMe у
    // загрузки весов (3 → 20 с). Продолжаем, когда модель поднята.
    Some(OffloadPinCacheGuard::new_pooled_paused(&ranges, workers, budget, true))
}

/// `SYN_QWEN4EXP_EXPERT_CACHE_GB` (0 — грузить эксперты целиком на
/// устройство, как раньше); по умолчанию 12 ГБ, и только для моделей, чьи
/// эксперты заведомо не влезают в память карты.
fn expert_cache_for(cfg: &Qwen4ExpConfig, device: Device) -> Option<Arc<ExpertCache>> {
    if !matches!(device, Device::Cuda(_)) {
        return None;
    }
    let requested = std::env::var("SYN_QWEN4EXP_EXPERT_CACHE_GB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok());
    let big = cfg.moe.num_experts * cfg.num_hidden_layers >= 1024;
    // Потолок — верхняя граница, а не бронь: каждая фаза хода зовёт
    // `fit_to_vram` и опускает ёмкость под реально свободную VRAM, а на OOM
    // аллокатор забирает своё через `Reclaimable`. Замеры на 125B (RTX 5090
    // Laptop 24 ГБ) дают плато скорости декода как раз на 16–18 ГБ кэша, и
    // прежние 12 ГБ отсекали его без всякой пользы.
    let gb = match requested {
        Some(v) if v <= 0.0 => return None,
        Some(v) => v,
        None if big => 16.0,
        None => return None,
    };
    Some(ExpertCache::new(device, (gb * (1u64 << 30) as f64) as usize))
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("model: {0}")]
    Model(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        PipelineError::Model(e.to_string())
    }
}
