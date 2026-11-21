//! Генерация Inkling-2: токенизатор, шаблон чата, сессия с префиксным кэшем.
//!
//! Состояние коротких свёрток назад не откатывается, поэтому кэш сессии
//! годен, только если новый промпт целиком продолжает уже посчитанные токены
//! (как у LFM3); иначе ход считается с нуля.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_llm_common::generate::{eos_set, GenerationConfig, GenerationStats, StreamSink, TokenSampler, INTERRUPTED};
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Inkling2Config;
use crate::loader::{Inkling2Weights, LoadError};
use synaptix_llm_common::expert_mirror::{self as em, ExpertStacks};
use synaptix_llm_common::moe::ExpertSource;
use synaptix_llm_common::moe_zc::{ExpertBlobs, ZcLayer, ZcPool};
use crate::model::{EmbedPatch, Inkling2Model, KvFormat, ModelCache};
use crate::preprocess::{load_image, ImageLimits};
use crate::vision::{expand_image_tokens, prefill_with_images, ImageSpan, VisionTower, IMAGE_TOKEN_ID};
use synaptix_core::tensor::Tensor;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("model: {0}")]
    Model(#[from] ModelError),
}

pub struct Inkling2Pipeline {
    pub model: Inkling2Model,
    pub tokenizer: HfTokenizer,
    pub chat_template: Option<String>,
    pub config: Inkling2Config,
    pub path: PathBuf,
    /// Закреплённое зеркало стопок экспертов в RAM (режим zero-copy). Поле
    /// раньше `_stacks`: гард джойнит копировщиков, пока mmap ещё жив.
    _host_mirror: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    _stacks: Option<Arc<ExpertStacks>>,
    /// Башня зрения: грузится по требованию ([`Self::load_vision`]) и
    /// отпускается после кодирования картинок ([`Self::release_vision`]).
    pub vision: Option<VisionTower>,
}

/// Где держать экспертов MoE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertMode {
    /// Все на карте (как было).
    Resident,
    /// В закреплённом зеркале RAM, на карте — кэш слотов (zero-copy).
    Zc,
}

/// Ручной выбор: `SYN_INKLING2_EXPERTS=resident|zc` (иначе — автоматически).
fn expert_mode_env() -> Option<ExpertMode> {
    match std::env::var("SYN_INKLING2_EXPERTS").ok()?.trim() {
        "resident" | "vram" | "gpu" => Some(ExpertMode::Resident),
        "zc" | "ram" | "zero-copy" => Some(ExpertMode::Zc),
        _ => None,
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// Ёмкость контекста, под которую считается VRAM, если загрузчику её не
/// назвали: `SYN_INKLING2_CTX`, иначе 32K.
fn planned_ctx() -> usize {
    env_usize("SYN_INKLING2_CTX").unwrap_or(32768)
}

/// Запас VRAM сверх весов и KV: активации чанка префилла, граф декода,
/// фрагментация пулов. `SYN_INKLING2_ACT_RESERVE_MB` (1536).
fn activation_reserve() -> usize {
    env_usize("SYN_INKLING2_ACT_RESERVE_MB").unwrap_or(1536) << 20
}

/// Место соседям по карте в режиме zero-copy (малые модели службы),
/// `SYN_INKLING2_ZC_HEADROOM_MB` (512).
fn zc_headroom() -> usize {
    env_usize("SYN_INKLING2_ZC_HEADROOM_MB").unwrap_or(512) << 20
}

const GB: f64 = (1u64 << 30) as f64;

/// Кэш хода между репликами чата.
pub struct Inkling2Session {
    pub cache: ModelCache,
    /// Токены, чьё состояние лежит в `cache` (промпт и ответ прошлого хода).
    pub ids: Vec<u32>,
    /// Картинки в `ids`: (первая строка, отпечаток признаков от вызывающего).
    /// Та же разметка с другой картинкой кэш не берёт.
    pub images: Vec<(usize, u64)>,
    /// Байт KV на позицию контекста (для учёта VRAM фасадом).
    kv_bytes_per_token: usize,
}

impl Inkling2Session {
    pub fn ctx_tokens(&self) -> usize {
        self.cache.capacity
    }

    pub fn cached_tokens(&self) -> usize {
        self.ids.len()
    }

    /// Сколько токенов `prompt_ids` возьмётся из кэша: всё посчитанное, если
    /// промпт его целиком продолжает (и хоть один токен остаётся на
    /// префилл), иначе 0 — состояние свёрток назад не откатить.
    pub fn reusable(&self, prompt_ids: &[u32]) -> usize {
        let n = self.ids.len();
        if n > 0 && n < prompt_ids.len() && prompt_ids[..n] == self.ids[..] && self.cache.seq_len == n {
            n
        } else {
            0
        }
    }

    /// Забыть посчитанное: следующий ход считается с нуля (обнуление
    /// состояний — в начале хода).
    pub fn invalidate(&mut self) {
        self.ids.clear();
        self.images.clear();
        self.cache.seq_len = 0;
    }

    /// VRAM под KV сессии (состояния свёрток — копейки, не в счёт); 0 — кэш в RAM.
    pub fn device_bytes(&self) -> usize {
        if self.cache.is_parked() {
            return 0;
        }
        self.cache.capacity * self.kv_bytes_per_token
    }

    /// Переселить KV и состояния свёрток в host-RAM (на время вложенной
    /// генерации): позиция и токены сессии сохраняются, следующий ход после
    /// [`Self::unpark_to`] продолжает кэш. Возвращает байты, ушедшие с карты.
    pub fn park_to_host(&mut self) -> Result<usize, PipelineError> {
        Ok(self.cache.park_to_host()?)
    }

    /// Обратный переезд; граф декода переснимется на первом шаге.
    pub fn unpark_to(&mut self, device: Device) -> Result<usize, PipelineError> {
        Ok(self.cache.unpark_to(device)?)
    }

    pub fn is_parked(&self) -> bool {
        self.cache.is_parked()
    }
}

/// Длина чанка префилла: внимание считается по всему контексту чанка.
fn prefill_chunk() -> usize {
    std::env::var("SYN_INKLING2_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512)
}

impl Inkling2Pipeline {
    /// Загрузка под контекст по умолчанию ([`planned_ctx`]).
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self, PipelineError> {
        Self::load_with_ctx(path, device, None)
    }

    /// Загрузка под ёмкость контекста `ctx` (токенов): если эксперты вместе с
    /// KV на `ctx` и запасом под активации помещаются на карту — они
    /// резидентны, иначе — zero-copy (зеркало в RAM + слоты VRAM на всё, что
    /// осталось). `SYN_INKLING2_EXPERTS=resident|zc` — выбрать вручную.
    pub fn load_with_ctx(path: impl AsRef<Path>, device: Device, ctx: Option<usize>) -> Result<Self, PipelineError> {
        let path = path.as_ref().to_path_buf();
        let le = |e: LoadError| PipelineError::Load(e.to_string());
        let w = Inkling2Weights::open(&path).map_err(le)?;
        let weights_guard = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let mut model = Inkling2Model::load_base(&w, device).map_err(le)?;
        let tokenizer = HfTokenizer::from_bytes(&w.tokenizer_json).map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let config = w.config.clone();
        let chat_template = w.chat_template.clone();
        let finish = |model, host_mirror, stacks| Self { model, tokenizer, chat_template, config: w.config.clone(), path, _host_mirror: host_mirror, _stacks: stacks, vision: None };
        let ctx = ctx.unwrap_or_else(planned_ctx).max(2);
        let stacks = w.expert_stacks().map(Arc::new);
        let experts_bytes: usize = stacks.as_deref().map_or(0, |s| em::mirror_ranges_natural(s, device).iter().map(|r| r.bytes.len()).sum());
        let free = match device {
            Device::Cuda(ord) => {
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
                synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0)
            }
            _ => usize::MAX,
        };
        let zc_ok = stacks.as_deref().is_some_and(|s| em::experts_zc_capable(s)) && device.is_cuda();
        // Байт слота zero-copy (gate_up + down всех слоёв) — нужен и для выбора формата KV.
        let t = &config.text;
        let (hidden, inter, ne) = (t.hidden_size, t.intermediate_size, t.n_routed_experts);
        let mut dtypes = Vec::with_capacity(t.num_hidden_layers);
        let mut per_slot = 0usize;
        for l in 0..t.num_hidden_layers {
            let dt = w.expert_dtype(l);
            if let Some(dt) = dt {
                let b = |n, k| synaptix_core::quant::matrix_blob_bytes(dt, n, k).map(|(p, s)| p + s).unwrap_or(0);
                per_slot += b(2 * inter, hidden) + b(hidden, inter);
            }
            dtypes.push(dt);
        }
        // Формат KV полных слотов: BF16, пока он помещается (с экспертами в RAM и минимумом
        // слотов); иначе MXFP8. `SYN_INKLING2_KV=bf16|mxfp8|int8` — вручную.
        let min_slots = env_usize("SYN_INKLING2_ZC_MIN_SLOTS").unwrap_or(8);
        let need = |f: KvFormat| {
            kv_fixed_bytes(&config) + ctx * kv_bytes_per_token(&config, f) + activation_reserve() + zc_headroom() + min_slots * per_slot
        };
        let kv_fmt = match std::env::var("SYN_INKLING2_KV").ok().as_deref().and_then(KvFormat::parse) {
            Some(f) => f,
            None if free < need(KvFormat::Bf16) => KvFormat::Mxfp8,
            None => KvFormat::Bf16,
        };
        model.set_kv_format(kv_fmt);
        let reserve = kv_fixed_bytes(&config) + ctx * kv_bytes_per_token(&config, kv_fmt) + activation_reserve();
        let mode = match expert_mode_env() {
            Some(ExpertMode::Zc) if zc_ok => ExpertMode::Zc,
            Some(ExpertMode::Zc) => return Err(PipelineError::Load("zero-copy экспертов: нужен .syn-бандл NVFP4/MXFP8 и CUDA".into())),
            Some(m) => m,
            None if zc_ok && free < reserve + experts_bytes => ExpertMode::Zc,
            None => ExpertMode::Resident,
        };
        eprintln!(
            "[inkling2] эксперты {:.1} ГБ, KV {} на {ctx} ток. + запас {:.1} ГБ, свободно {:.1} ГБ → {}",
            experts_bytes as f64 / GB,
            kv_fmt.name(),
            reserve as f64 / GB,
            free as f64 / GB,
            if mode == ExpertMode::Zc { "zero-copy (эксперты в RAM, слоты VRAM)" } else { "эксперты на карте" }
        );
        if mode == ExpertMode::Resident {
            model.load_resident_experts(&w).map_err(le)?;
            return Ok(finish(model, None, None));
        }
        let stacks = stacks.expect("zc_ok");
        // Зеркало стопок: NVFP4 — в натуральной раскладке (её читают ядра GEMV_SYN/группового GEMM).
        let (mirror, budget) = {
            let ranges = em::mirror_ranges_natural(&*stacks, device);
            em::host_mirror(&ranges, "inkling2").ok_or_else(|| PipelineError::Load("zero-copy: зеркало экспертов в RAM не поднято".into()))?
        };
        if budget < experts_bytes {
            return Err(PipelineError::Load(format!(
                "zero-copy: RAM под зеркало {:.1} ГБ < стопок {:.1} ГБ (SYN_HOST_MIRROR_GB / SYN_MIRROR_RESERVE_GB)",
                budget as f64 / GB,
                experts_bytes as f64 / GB
            )));
        }
        mirror.resume();
        wait_mirror()?;
        // Слоты: всё, что свободно после весов, KV на `ctx`, активаций и запаса соседям.
        let dtypes: Vec<_> = dtypes
            .into_iter()
            .enumerate()
            .map(|(l, dt)| dt.ok_or_else(|| PipelineError::Load(format!("слой {l}: формат экспертов"))))
            .collect::<Result<_, _>>()?;
        let avail = free.saturating_sub(reserve + zc_headroom());
        let slots = env_usize("SYN_INKLING2_ZC_SLOTS").unwrap_or(avail / per_slot.max(1)).min(ne);
        let pool = ZcPool::new(device);
        pool.set_min_slots(min_slots);
        pool.set_slots_per_layer(slots);
        let src = em::BundleExperts::new(stacks.clone());
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for (l, &dt) in dtypes.iter().enumerate() {
            let blobs: Vec<ExpertBlobs> = (0..ne)
                .map(|e| src.blob_addrs4(l, e))
                .collect::<Option<_>>()
                .ok_or_else(|| PipelineError::Load(format!("zero-copy: слой {l}: эксперт не в зеркале")))?;
            let z = ZcLayer::build_from_blobs(&blobs, dt, (2 * inter, hidden), (hidden, inter), &pool, None)?
                .ok_or_else(|| PipelineError::Load(format!("zero-copy: слой {l}: слотов {slots} мало (свободно {:.1} ГБ)", avail as f64 / GB)))?;
            layers.push(z);
        }
        model.attach_zc(pool, layers).map_err(le)?;
        drop(weights_guard);
        eprintln!(
            "[inkling2] zero-copy: слотов {slots} из {ne} на слой ({:.1} ГБ VRAM), зеркало {:.1} ГБ",
            (slots * per_slot) as f64 / GB,
            experts_bytes as f64 / GB
        );
        Ok(finish(model, Some(mirror), Some(stacks)))
    }

    /// Режим экспертов загруженной модели.
    pub fn expert_mode(&self) -> ExpertMode {
        if self.model.zc_active() { ExpertMode::Zc } else { ExpertMode::Resident }
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        self.config.eos_ids.clone()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, PipelineError> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| PipelineError::Tokenize(e.to_string()))?.ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer.decode(ids, false).map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    /// Байт KV на токен контекста: растут только слоты полного внимания (у SWA — кольцо
    /// фиксированного размера, см. [`Self::kv_fixed_bytes`]).
    pub fn kv_bytes_per_token(&self) -> usize {
        kv_bytes_per_token(&self.config, self.model.kv_format())
    }

    /// Формат KV слотов полного внимания (выбран при загрузке).
    pub fn kv_format(&self) -> KvFormat {
        self.model.kv_format()
    }

    /// Слотов полного внимания (физический полный слой × его проходы).
    pub fn full_slots(&self) -> usize {
        full_slots(&self.config)
    }

    /// Фиксированные байты KV: кольца SWA-слотов (окно + чанк префилла).
    pub fn kv_fixed_bytes(&self) -> usize {
        kv_fixed_bytes(&self.config)
    }

    pub fn new_session(&self, ctx: usize) -> Result<Inkling2Session, PipelineError> {
        Ok(Inkling2Session {
            cache: self.model.make_cache(ctx.max(2))?,
            ids: Vec::new(),
            images: Vec::new(),
            kv_bytes_per_token: self.kv_bytes_per_token(),
        })
    }

    /// Дорастить сессию: KV копируется в больший кэш, позиции сохраняются.
    pub fn grow_session(&self, s: &mut Inkling2Session, ctx: usize) -> Result<(), PipelineError> {
        if ctx <= s.cache.capacity {
            return Ok(());
        }
        self.model.grow_cache(&mut s.cache, ctx)?;
        Ok(())
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Загрузить башню зрения. `Ok(false)` — у модели нет `vision_config`.
    pub fn load_vision(&mut self) -> Result<bool, PipelineError> {
        if self.vision.is_some() {
            return Ok(true);
        }
        if self.config.vision.is_none() {
            return Ok(false);
        }
        let w = Inkling2Weights::open(&self.path).map_err(|e| PipelineError::Load(e.to_string()))?;
        self.vision = Some(VisionTower::load(&w, self.model.device).map_err(|e| PipelineError::Load(e.to_string()))?);
        Ok(true)
    }

    pub fn release_vision(&mut self) {
        self.vision = None;
    }

    /// Признаки картинки `[N, hidden]` (N — клеток 40 px, по токену на клетку);
    /// `max_tokens` — потолок клеток на картинку.
    pub fn encode_image(&self, path: &Path, max_tokens: Option<usize>) -> Result<Tensor, PipelineError> {
        let tower = self.vision.as_ref().ok_or_else(|| PipelineError::Load("башня зрения не загружена".into()))?;
        let vcfg = self.config.vision.as_ref().ok_or_else(|| PipelineError::Load("нет vision_config".into()))?;
        let mut lim = ImageLimits::default();
        if let Some(n) = max_tokens {
            lim.max_tokens = n;
        }
        let img = load_image(path, vcfg, lim).map_err(|e| PipelineError::Load(e.to_string()))?;
        Ok(no_grad(|| tower.encode(&img))?)
    }

    /// Разовая генерация по промпту с картинками. В `prompt_ids` на картинку —
    /// один плейсхолдер [`IMAGE_TOKEN_ID`] (в порядке `images`); если он стоит
    /// внутри текстовой части реплики (`<|content_text|>…`), картинка выносится
    /// в свою часть `<|content_image|>` перед этой текстовой — как шаблон
    /// рендерит сообщение с частями `[картинки…, текст]`. Затем плейсхолдер
    /// разворачивается в N строк признаков.
    pub fn generate_media_streaming(
        &self,
        prompt_ids: &[u32],
        images: &[&Tensor],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let ids = self.image_parts(prompt_ids)?;
        let counts: Vec<usize> = images.iter().map(|t| t.dims()[0]).collect();
        let (ids, starts) = expand_image_tokens(&ids, IMAGE_TOKEN_ID, &counts)?;
        let spans: Vec<ImageSpan> =
            starts.into_iter().zip(images).map(|(start, t)| ImageSpan { start, embeds: (*t).clone() }).collect();
        let ctx = cfg.max_seq.unwrap_or(0).max(ids.len() + cfg.max_new_tokens + 1);
        let mut s = self.new_session(ctx)?;
        let eos = eos_set(cfg);
        let mut sampler = TokenSampler::new(cfg, &ids);
        let t0 = Instant::now();
        let logits = no_grad(|| prefill_with_images(&self.model, &mut s.cache, &ids, &spans))?;
        synaptix_llm_common::prefill_progress::add(ids.len());
        let prefill_ms = t0.elapsed().as_millis();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        let t1 = Instant::now();
        if cfg.max_new_tokens > 0 {
            let mut tok = sampler.sample(&logits)?;
            loop {
                out.push(tok);
                if !sink.on_token(tok) || eos.contains(&tok) || out.len() >= cfg.max_new_tokens {
                    break;
                }
                let l = no_grad(|| self.model.decode(tok, &mut s.cache))?;
                tok = sampler.sample(&l)?;
            }
        }
        let st = GenerationStats { prompt_tokens: ids.len(), new_tokens: out.len(), prefill_ms, decode_ms: t1.elapsed().as_millis() };
        Ok((out, st))
    }

    /// Вынести плейсхолдеры картинок из текстовых частей в свои части
    /// `<роль><|content_image|><img><|end_message|>` перед текстовой.
    fn image_parts(&self, ids: &[u32]) -> Result<Vec<u32>, PipelineError> {
        let single = |s: &str| -> Result<u32, PipelineError> {
            match self.encode(s)?.as_slice() {
                [id] => Ok(*id),
                _ => Err(PipelineError::Tokenize(format!("{s}: не один токен"))),
            }
        };
        let (text, image, end) = (single("<|content_text|>")?, single("<|content_image|>")?, single("<|end_message|>")?);
        let mut out = Vec::with_capacity(ids.len() + 8);
        let mut i = 0;
        while i < ids.len() {
            // Текстовая часть: `<роль><|content_text|>` … `<|end_message|>`.
            if i + 1 < ids.len() && ids[i + 1] == text {
                let body_end = ids[i + 2..].iter().position(|&t| t == end).map_or(ids.len(), |p| i + 2 + p);
                let body = &ids[i + 2..body_end];
                let n_img = body.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
                if n_img > 0 {
                    for _ in 0..n_img {
                        out.extend_from_slice(&[ids[i], image, IMAGE_TOKEN_ID, end]);
                    }
                    out.extend_from_slice(&[ids[i], text]);
                    out.extend(body.iter().copied().filter(|&t| t != IMAGE_TOKEN_ID));
                    i = body_end;
                    continue;
                }
            }
            out.push(ids[i]);
            i += 1;
        }
        Ok(out)
    }

    /// Разовая генерация с нуля.
    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let ctx = cfg.max_seq.unwrap_or(prompt_ids.len() + cfg.max_new_tokens + 1);
        let mut s = self.new_session(ctx)?;
        let (out, st, _) = self.generate_cached_streaming(&mut s, prompt_ids, cfg, sink)?;
        Ok((out, st))
    }

    /// Ход в сессии: переиспользует кэш, если промпт продолжает посчитанные
    /// токены. Возвращает (ответ, статистика, сколько токенов взято из кэша).
    pub fn generate_cached_streaming(
        &self,
        s: &mut Inkling2Session,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        self.cached_turn(s, prompt_ids, &[], cfg, sink)
    }

    /// Ход в сессии по промпту с картинками (как
    /// [`Self::generate_media_streaming`]: по плейсхолдеру на картинку, в
    /// порядке `images`; у картинки — признаки и отпечаток от вызывающего).
    /// Кэш берётся, если развёрнутый промпт целиком продолжает посчитанное и
    /// картинки в посчитанном префиксе — те же. В статистике `prompt_tokens`
    /// — длина развёрнутого промпта (строки картинок), из кэша — в тех же
    /// единицах.
    pub fn generate_cached_media_streaming(
        &self,
        s: &mut Inkling2Session,
        prompt_ids: &[u32],
        images: &[(&Tensor, u64)],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        let ids = self.image_parts(prompt_ids)?;
        let counts: Vec<usize> = images.iter().map(|(t, _)| t.dims()[0]).collect();
        let (ids, starts) = expand_image_tokens(&ids, IMAGE_TOKEN_ID, &counts)?;
        let spans: Vec<(usize, &Tensor, u64)> = starts.into_iter().zip(images).map(|(at, (t, fp))| (at, *t, *fp)).collect();
        self.cached_turn(s, &ids, &spans, cfg, sink)
    }

    /// Общий ход в сессии; `images` — (строка, признаки, отпечаток) картинок
    /// в `prompt_ids` (уже развёрнутых).
    fn cached_turn(
        &self,
        s: &mut Inkling2Session,
        prompt_ids: &[u32],
        images: &[(usize, &Tensor, u64)],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Model(ModelError::Forward("пустой промпт".into())));
        }
        let need = prompt_ids.len() + cfg.max_new_tokens + 1;
        if need > s.cache.capacity {
            self.grow_session(s, need)?;
        }
        // Кэш годен, только если промпт продолжает его целиком и хоть один токен остаётся на префилл,
        // а картинки посчитанного префикса — те же (та же разметка — ещё не та же картинка).
        let mut reuse = s.reusable(prompt_ids);
        let same_images = s.images.iter().filter(|(at, _)| *at < reuse).enumerate().all(|(k, &(at, fp))| {
            images.get(k).is_some_and(|&(at2, _, fp2)| at2 == at && fp2 == fp)
        });
        if !same_images {
            reuse = 0;
        }
        if reuse == 0 {
            s.cache.reset()?;
            s.ids.clear();
            s.images.clear();
        }
        s.images.retain(|(at, _)| *at < reuse);
        let eos = eos_set(cfg);
        let mut sampler = TokenSampler::new(cfg, prompt_ids);
        let t0 = Instant::now();
        let chunk = prefill_chunk().max(1);
        let mut off = reuse;
        let mut logits = None;
        s.ids.truncate(reuse);
        while off < prompt_ids.len() {
            if sink.interrupted() {
                s.ids.clear();
                s.cache.seq_len = 0;
                return Err(PipelineError::Model(ModelError::Forward(INTERRUPTED.into())));
            }
            let end = (off + chunk).min(prompt_ids.len());
            // Строки картинок этого куска — готовые признаки.
            let mut patches = Vec::new();
            for &(at, t, _) in images {
                let (a, b) = (at.max(off), (at + t.dims()[0]).min(end));
                if a < b {
                    patches.push(EmbedPatch { row: a - off, embeds: t.narrow(0, a - at, b - a).and_then(|x| x.contiguous()).map_err(|e| ModelError::Forward(e.to_string()))? });
                }
            }
            let res = no_grad(|| self.model.forward_with_embeds(&prompt_ids[off..end], &patches, &mut s.cache, false));
            match res {
                Ok(l) => logits = Some(l),
                Err(e) => {
                    s.ids.clear();
                    s.cache.seq_len = 0;
                    return Err(e.into());
                }
            }
            s.ids.extend_from_slice(&prompt_ids[off..end]);
            synaptix_llm_common::prefill_progress::add(end - off);
            off = end;
        }
        s.images.extend(images.iter().filter(|(at, _, _)| *at >= reuse).map(|&(at, _, fp)| (at, fp)));
        let prefill_ms = t0.elapsed().as_millis();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        let stats = |out: &Vec<u32>, decode_ms| GenerationStats { prompt_tokens: prompt_ids.len(), new_tokens: out.len(), prefill_ms, decode_ms };
        if cfg.max_new_tokens == 0 {
            let st = stats(&out, 0);
            return Ok((out, st, reuse));
        }
        let logits = logits.expect("префилл непустой");
        let t1 = Instant::now();
        let mut tok = sampler.sample(&logits)?;
        loop {
            out.push(tok);
            if !sink.on_token(tok) || eos.contains(&tok) || out.len() >= cfg.max_new_tokens {
                break;
            }
            let l = no_grad(|| self.model.decode(tok, &mut s.cache))?;
            s.ids.push(tok);
            tok = sampler.sample(&l)?;
        }
        Ok((out.clone(), stats(&out, t1.elapsed().as_millis()), reuse))
    }
}

/// Дождаться, пока зеркало экспертов заполнится целиком: слои zero-copy
/// собираются по адресам в нём.
fn wait_mirror() -> Result<(), PipelineError> {
    let t0 = Instant::now();
    let mut last = 0usize;
    loop {
        let Some((ready, total)) = synaptix_core::device::cuda::offload_pin_cache_progress() else {
            return Err(PipelineError::Load("zero-copy: зеркало пропало".into()));
        };
        if ready >= total {
            eprintln!("[inkling2] зеркало экспертов: {:.1} ГБ за {:.1} с", total as f64 / GB, t0.elapsed().as_secs_f64());
            return Ok(());
        }
        if t0.elapsed().as_secs() > 900 {
            return Err(PipelineError::Load(format!("zero-copy: зеркало {:.1}/{:.1} ГБ за 15 мин", ready as f64 / GB, total as f64 / GB)));
        }
        if ready >= last + (2usize << 30) {
            eprintln!("[inkling2] зеркало экспертов: {:.1}/{:.1} ГБ", ready as f64 / GB, total as f64 / GB);
            last = ready;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn kv_bytes_per_token(cfg: &Inkling2Config, f: KvFormat) -> usize {
    let t = &cfg.text;
    full_slots(cfg) * 2 * t.num_key_value_heads * t.head_dim * (f.bytes_per_elem() * 32.0) as usize / 32
}

fn full_slots(cfg: &Inkling2Config) -> usize {
    let n = cfg.text.num_hidden_layers;
    let (cs, ce) = cfg.core();
    (0..cfg.num_slots())
        .filter(|&s| {
            let phys = if s < n { s } else { cs + (s - n) % (ce - cs) };
            !cfg.is_swa(phys)
        })
        .count()
}

fn kv_fixed_bytes(cfg: &Inkling2Config) -> usize {
    let t = &cfg.text;
    let swa = cfg.num_slots() - full_slots(cfg);
    swa * 2 * t.num_key_value_heads * t.head_dim * 2 * (t.sliding_window_size + crate::model::MAX_CHUNK)
}
