//! Генерация Inkling-2: токенизатор, шаблон чата, сессия с префиксным кэшем.
//!
//! Состояние коротких свёрток и кольца SWA назад не откатываются, поэтому
//! кэш сессии берётся либо целиком (новый промпт продолжает всё посчитанное),
//! либо от точки возврата — снимка этих состояний на конце промпта прошлого
//! хода (или на границе, названной вызывающим), как у Qwen4Exp. Иначе ход
//! считается с нуля.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_llm_common::generate::{eos_set, GenerationConfig, GenerationStats, StreamSink, TokenSampler, INTERRUPTED};
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Inkling2Config;
use crate::experts::{env_usize, kv_format_env, ExpertHub, KvNeed};
pub use crate::experts::ExpertMode;
use crate::loader::{Inkling2Weights, LoadError};
use crate::model::{CacheSnapshot, EmbedPatch, Inkling2Model, KvFormat, ModelCache};
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
    /// Режим экспертов (на карте / zero-copy) по длине сессий на карте;
    /// держит зеркало стопок в RAM и mmap бандла.
    pub hub: Arc<ExpertHub>,
    /// Башня зрения: грузится по требованию ([`Self::load_vision`]) и
    /// отпускается после кодирования картинок ([`Self::release_vision`]).
    pub vision: Option<VisionTower>,
}

/// Ёмкость первой сессии, под которую при загрузке выбирается режим
/// экспертов: `SYN_INKLING2_CTX`, иначе 32K. Названная загрузчику ёмкость —
/// только предел роста, она режим не задаёт.
fn planned_ctx() -> usize {
    env_usize("SYN_INKLING2_CTX").unwrap_or(32768)
}

const GB: f64 = (1u64 << 30) as f64;

/// Кэш хода между репликами чата.
///
/// Точка возврата стоит на КОНЦЕ ПРОМПТА хода (или на границе, названной
/// [`Self::set_resume_at`]), как у Qwen4Exp: клиент убирает из истории
/// размышления прошлых ответов, и новый промпт продолжает прошлый промпт, но
/// не сгенерированный ответ. Снимок ([`CacheSnapshot`]) держит то, что ход
/// после точки портит безвозвратно (состояния свёрток, кольца SWA); KV полных
/// слотов ниже точки цел и возвращается уменьшением `seq_len`.
pub struct Inkling2Session {
    pub cache: ModelCache,
    /// Токены, чьё состояние лежит в `cache` (промпт и ответ прошлого хода).
    pub ids: Vec<u32>,
    /// Картинки в `ids`: (первая строка, отпечаток признаков от вызывающего).
    /// Та же разметка с другой картинкой кэш не берёт.
    pub images: Vec<(usize, u64)>,
    /// Точка возврата: снимок, токены промпта до неё и картинки в них.
    /// Строки полных слотов ниже точки в `cache` обязаны быть целы: сброс кэша
    /// снимок выбрасывает.
    snap: Option<SessionSnap>,
    /// Где поставить точку возврата этого хода, если не на конце промпта.
    /// Действует на один ход.
    resume_at: Option<usize>,
    /// Байт KV полных слотов на позицию: BF16 и квантованный (учёт VRAM).
    rate_bf16: usize,
    rate_q: usize,
    /// Кольца SWA (возвращаются на карту вместе с кэшем).
    kv_fixed: usize,
    /// Режим экспертов модели: возврат из RAM освобождает под кэш место.
    /// Слабая ссылка — сессия может пережить выгрузку модели.
    hub: Weak<ExpertHub>,
}

struct SessionSnap {
    ids: Vec<u32>,
    images: Vec<(usize, u64)>,
    cache: CacheSnapshot,
}

impl Inkling2Session {
    pub fn ctx_tokens(&self) -> usize {
        self.cache.capacity
    }

    /// Токенов в кэше: всё посчитанное либо промпт до точки возврата.
    pub fn cached_tokens(&self) -> usize {
        self.ids.len().max(self.snap.as_ref().map_or(0, |s| s.ids.len()))
    }

    /// Всё посчитанное, если промпт его целиком продолжает (и хоть один токен
    /// остаётся на префилл), иначе 0.
    fn reusable_all(&self, prompt_ids: &[u32]) -> usize {
        let n = self.ids.len();
        if n > 0 && n < prompt_ids.len() && prompt_ids[..n] == self.ids[..] && self.cache.seq_len == n {
            n
        } else {
            0
        }
    }

    /// Промпт до точки возврата, если новый промпт его продолжает, иначе 0.
    fn reusable_snap(&self, prompt_ids: &[u32]) -> usize {
        match &self.snap {
            Some(s) if !s.ids.is_empty() && s.ids.len() < prompt_ids.len() && prompt_ids[..s.ids.len()] == s.ids[..] => s.ids.len(),
            _ => 0,
        }
    }

    /// Сколько токенов `prompt_ids` возьмётся из кэша: всё посчитанное, если
    /// промпт его целиком продолжает; иначе промпт до точки возврата, если
    /// промпт продолжает его; иначе 0 — середину свёрток не откатить. Хоть
    /// один токен всегда остаётся на префилл.
    pub fn reusable(&self, prompt_ids: &[u32]) -> usize {
        self.reusable_all(prompt_ids).max(self.reusable_snap(prompt_ids))
    }

    /// Поставить точку возврата следующего хода после `at` токенов промпта,
    /// а не на его конце (как у Qwen4Exp). Вызывающий называет конец истории
    /// перед приглашением к ответу: следующий промпт может разойтись с этим
    /// сразу за историей. Граница вне хвоста, который ход префиллит, не
    /// действует (если на ней уже стоит точка — она остаётся). На один ход.
    pub fn set_resume_at(&mut self, at: Option<usize>) {
        self.resume_at = at;
    }

    /// Граница точки возврата следующего хода (см. [`Self::set_resume_at`]).
    pub fn resume_at(&self) -> Option<usize> {
        self.resume_at
    }

    /// Позиция точки возврата и байты её снимка (на карте ли).
    pub fn snapshot_info(&self) -> Option<(usize, usize, bool)> {
        self.snap.as_ref().map(|s| (s.cache.pos(), s.cache.bytes(), s.cache.on_device()))
    }

    /// Забыть посчитанное: следующий ход считается с нуля (обнуление
    /// состояний — в начале хода).
    pub fn invalidate(&mut self) {
        self.ids.clear();
        self.images.clear();
        self.snap = None;
        self.cache.seq_len = 0;
    }

    /// VRAM под KV сессии (полные слоты в своём формате и кольца SWA;
    /// состояния свёрток — копейки, не в счёт) и снимок точки возврата на
    /// карте; 0 — кэш в RAM.
    pub fn device_bytes(&self) -> usize {
        if self.cache.is_parked() {
            return 0;
        }
        self.cache.full_kv_bytes() + self.kv_fixed + self.snap.as_ref().map_or(0, |s| s.cache.device_bytes())
    }

    /// Формат KV полных слотов сессии (BF16, после перехода по памяти — MXFP8).
    pub fn kv_format(&self) -> KvFormat {
        self.cache.full_format().unwrap_or(KvFormat::Bf16)
    }


    /// Переселить KV и состояния свёрток в host-RAM (на время вложенной
    /// генерации): позиция и токены сессии сохраняются, следующий ход после
    /// [`Self::unpark_to`] продолжает кэш. Снимок точки возврата уезжает в
    /// закреплённую RAM и там остаётся (восстановление идёт прямо оттуда).
    /// Возвращает байты, ушедшие с карты.
    pub fn park_to_host(&mut self) -> Result<usize, PipelineError> {
        let mut moved = 0;
        if let Some(s) = self.snap.as_mut() {
            moved += s.cache.park_to_host()?;
        }
        Ok(moved + self.cache.park_to_host()?)
    }

    /// Обратный переезд; граф декода переснимется на первом шаге. Снимок,
    /// уехавший с карты, возвращается туда же, если там есть место. Не
    /// помещается рядом с экспертами — режим экспертов ужимается (меньше
    /// слотов, zero-copy), формат KV сессии не меняется.
    pub fn unpark_to(&mut self, device: Device) -> Result<usize, PipelineError> {
        let mut moved = match (self.cache.is_parked(), self.hub.upgrade()) {
            (true, Some(hub)) => {
                let fmt = self.kv_format();
                let (cap, fixed) = (self.cache.capacity, self.kv_fixed);
                let (rb, rq) = (self.rate_bf16, self.rate_q);
                let need = move |f: KvFormat| cap * if f == KvFormat::Bf16 { rb } else { rq } + fixed;
                let cache = &mut self.cache;
                hub.arrange(KvNeed { held: 0, need: &need, fixed: Some(fmt), upgrade: false }, |_| Ok(cache.unpark_to(device)?))?
            }
            _ => self.cache.unpark_to(device)?,
        };
        if let Some(s) = self.snap.as_mut() {
            moved += s.cache.unpark()?;
        }
        Ok(moved)
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

    /// Загрузка. `ctx` — предел роста сессий (окно службы), режим он не
    /// задаёт: эксперты поднимаются под первую сессию на [`planned_ctx`]
    /// (на карту, если помещаются с KV BF16 и запасом), дальше режим следует
    /// за сессиями ([`crate::experts`]). `SYN_INKLING2_EXPERTS=resident|zc`,
    /// `SYN_INKLING2_KV=bf16|mxfp8|int8` — выбрать вручную;
    /// `SYN_INKLING2_MIRROR=eager` — сразу заполнять зеркало стопок в фоне.
    pub fn load_with_ctx(path: impl AsRef<Path>, device: Device, ctx: Option<usize>) -> Result<Self, PipelineError> {
        let path = path.as_ref().to_path_buf();
        let le = |e: LoadError| PipelineError::Load(e.to_string());
        let w = Inkling2Weights::open(&path).map_err(le)?;
        let weights_guard = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let model = Inkling2Model::load_base(&w, device).map_err(le)?;
        drop(weights_guard);
        let tokenizer = HfTokenizer::from_bytes(&w.tokenizer_json).map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let config = w.config.clone();
        let hub = Arc::new(ExpertHub::new(&w, model.xrt.clone(), path.clone(), device));
        let first = planned_ctx().min(ctx.unwrap_or(usize::MAX)).max(2);
        let (rate_b, rate_q, fixed) =
            (kv_bytes_per_token(&config, KvFormat::Bf16), kv_bytes_per_token(&config, KvFormat::Mxfp8), kv_fixed_bytes(&config));
        let need = |f: KvFormat| first * if f == KvFormat::Bf16 { rate_b } else { rate_q } + fixed;
        let fmt = hub.initial(&need)?;
        eprintln!(
            "[inkling2] эксперты {:.1} ГБ: {}; KV {} под первую сессию {first} ток. (предел {})",
            hub.experts_bytes() as f64 / GB,
            match hub.mode() {
                Some(ExpertMode::Zc) => format!("zero-copy, {} слотов", hub.slots()),
                _ => "на карте".into(),
            },
            fmt.name(),
            ctx.map_or("—".into(), |c| c.to_string())
        );
        if std::env::var("SYN_INKLING2_MIRROR").is_ok_and(|v| v.trim() == "eager") {
            hub.prefetch_mirror();
        }
        Ok(Self { model, tokenizer, chat_template: w.chat_template.clone(), config, path, hub, vision: None })
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

    /// Байт KV на токен контекста у новой сессии: растут только слоты полного
    /// внимания (у SWA — кольцо фиксированного размера, см.
    /// [`Self::kv_fixed_bytes`]). Сессия, переведённая в MXFP8, — вдвое меньше
    /// ([`Inkling2Session::device_bytes`] считает по факту).
    pub fn kv_bytes_per_token(&self) -> usize {
        kv_bytes_per_token(&self.config, self.kv_format())
    }

    /// Формат KV полных слотов новой сессии: BF16 (или `SYN_INKLING2_KV`).
    pub fn kv_format(&self) -> KvFormat {
        kv_format_env().unwrap_or(KvFormat::Bf16)
    }

    /// Слотов zero-copy на слой (0 — эксперты на карте).
    pub fn zc_slots(&self) -> usize {
        self.hub.slots()
    }

    /// Слотов полного внимания (физический полный слой × его проходы).
    pub fn full_slots(&self) -> usize {
        full_slots(&self.config)
    }

    /// Фиксированные байты KV: кольца SWA-слотов (окно + чанк префилла).
    pub fn kv_fixed_bytes(&self) -> usize {
        kv_fixed_bytes(&self.config)
    }

    fn rate(&self, f: KvFormat) -> usize {
        kv_bytes_per_token(&self.config, f)
    }

    /// Новая сессия на `ctx` позиций. Режим экспертов подстраивается под неё:
    /// если она с KV BF16 помещается рядом с экспертами на карте — они
    /// возвращаются на карту (после длинного разговора), иначе zero-copy со
    /// слотами на всё, что осталось; только если и этого мало — KV MXFP8.
    pub fn new_session(&self, ctx: usize) -> Result<Inkling2Session, PipelineError> {
        self.new_session_fmt(ctx, None)
    }

    /// Как [`Self::new_session`], но KV полных слотов сразу в формате `fmt`
    /// (сверки; `None` — по памяти).
    pub fn new_session_fmt(&self, ctx: usize, fmt: Option<KvFormat>) -> Result<Inkling2Session, PipelineError> {
        let ctx = ctx.max(2);
        let fixed = self.kv_fixed_bytes();
        let need = |f: KvFormat| ctx * self.rate(f) + fixed;
        let cache = self.hub.arrange(KvNeed { held: 0, need: &need, fixed: fmt, upgrade: true }, |f| Ok(self.model.make_cache_fmt(ctx, f)?))?;
        Ok(Inkling2Session {
            cache,
            ids: Vec::new(),
            images: Vec::new(),
            snap: None,
            resume_at: None,
            rate_bf16: self.rate(KvFormat::Bf16),
            rate_q: self.rate(KvFormat::Mxfp8),
            kv_fixed: fixed,
            hub: Arc::downgrade(&self.hub),
        })
    }

    /// Дорастить сессию: KV копируется в больший кэш, позиции, точка
    /// возврата и граф декода (переснимется) сохраняются. Новому KV не
    /// хватает карты — по порядку: эксперты в zero-copy (слотов — сколько
    /// влезает после KV и запаса, по мере роста меньше), затем KV полных
    /// слотов в MXFP8 (записанный BF16 переквантуется на карте).
    pub fn grow_session(&self, s: &mut Inkling2Session, ctx: usize) -> Result<(), PipelineError> {
        if ctx <= s.cache.capacity {
            return Ok(());
        }
        let fmt = s.kv_format();
        let need = |f: KvFormat| ctx * self.rate(f);
        let kv = KvNeed { held: s.cache.full_kv_bytes(), need: &need, fixed: (fmt != KvFormat::Bf16).then_some(fmt), upgrade: false };
        let cache = &mut s.cache;
        self.hub.arrange(kv, |f| Ok(self.model.grow_cache_fmt(cache, ctx, Some(f))?))
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
        // Граница точки возврата названа в токенах `prompt_ids` — перевести в строки
        // развёрнутого промпта: развернуть её префикс так же и проверить, что он префикс
        // развёрнутого целого (граница не посреди текстовой части с картинкой).
        if let Some(at) = s.resume_at.filter(|&at| at > 0 && at < prompt_ids.len()) {
            let head = &prompt_ids[..at];
            let k = head.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count().min(counts.len());
            let mapped = self
                .image_parts(head)
                .ok()
                .and_then(|h| expand_image_tokens(&h, IMAGE_TOKEN_ID, &counts[..k]).ok())
                .map(|(h, _)| h)
                .filter(|h| h.len() < ids.len() && ids[..h.len()] == h[..])
                .map(|h| h.len());
            s.resume_at = mapped;
        }
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
        let eos = eos_set(cfg);
        let mut sampler = TokenSampler::new(cfg, prompt_ids);
        let t0 = Instant::now();
        let (logits, reuse) = self.prefill_turn(s, prompt_ids, images, cfg.max_new_tokens, sink, None)?;
        let prefill_ms = t0.elapsed().as_millis();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        let stats = |out: &Vec<u32>, decode_ms| GenerationStats { prompt_tokens: prompt_ids.len(), new_tokens: out.len(), prefill_ms, decode_ms };
        if cfg.max_new_tokens == 0 {
            let st = stats(&out, 0);
            return Ok((out, st, reuse));
        }
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

    /// Только префилл хода в сессии (кэш, точка возврата — как у
    /// [`Self::generate_cached_streaming`]): логиты последней позиции промпта
    /// `[1, V]` F32 и сколько токенов взято из кэша. Для проверок.
    pub fn prefill_cached(&self, s: &mut Inkling2Session, prompt_ids: &[u32]) -> Result<(Tensor, usize), PipelineError> {
        self.prefill_turn(s, prompt_ids, &[], 1, &mut |_: u32| true, None)
    }

    /// Как [`Self::prefill_cached`], но логиты всех строк хвоста `[T, V]` F32
    /// отдаются по чанкам в `rows` (сверка KL). Возвращает, сколько взято из кэша.
    pub fn prefill_cached_rows(
        &self,
        s: &mut Inkling2Session,
        prompt_ids: &[u32],
        rows: &mut dyn FnMut(&Tensor) -> Result<(), PipelineError>,
    ) -> Result<usize, PipelineError> {
        Ok(self.prefill_turn(s, prompt_ids, &[], 1, &mut |_: u32| true, Some(rows))?.1)
    }

    /// Префилл хода: выбор кэша (целиком / от точки возврата / с нуля),
    /// прогон хвоста с границей чанка на новой точке возврата. Логиты
    /// последней позиции промпта и сколько токенов взято из кэша.
    fn prefill_turn(
        &self,
        s: &mut Inkling2Session,
        prompt_ids: &[u32],
        images: &[(usize, &Tensor, u64)],
        max_new: usize,
        sink: &mut dyn StreamSink,
        mut rows: Option<&mut dyn FnMut(&Tensor) -> Result<(), PipelineError>>,
    ) -> Result<(Tensor, usize), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Model(ModelError::Forward("пустой промпт".into())));
        }
        let need = prompt_ids.len() + max_new + 1;
        if need > s.cache.capacity {
            self.grow_session(s, need)?;
        }
        // Картинки посчитанного префикса — те же (та же разметка — ещё не та же картинка).
        let same_images = |upto: usize, have: &[(usize, u64)]| {
            have.iter().filter(|(at, _)| *at < upto).enumerate().all(|(k, &(at, fp))| {
                images.get(k).is_some_and(|&(at2, _, fp2)| at2 == at && fp2 == fp)
            })
        };
        // Кэш берётся целиком, если промпт продолжает всё посчитанное; иначе — от точки
        // возврата, если промпт продолжает её промпт; иначе — с нуля.
        let all = s.reusable_all(prompt_ids);
        let at_snap = s.reusable_snap(prompt_ids);
        let (reuse, restore) = if all > 0 && same_images(all, &s.images) {
            (all, false)
        } else if at_snap > 0 && s.snap.as_ref().is_some_and(|sn| same_images(at_snap, &sn.images)) {
            (at_snap, true)
        } else {
            (0, false)
        };
        let resume = s.resume_at.take();
        if reuse == 0 {
            s.cache.reset()?;
            s.ids.clear();
            s.images.clear();
            s.snap = None;
        } else if restore {
            let sn = s.snap.as_ref().expect("точка возврата");
            if let Err(e) = self.model.restore(&mut s.cache, &sn.cache) {
                s.ids.clear();
                s.images.clear();
                s.snap = None;
                s.cache.seq_len = 0;
                return Err(e.into());
            }
            s.ids = sn.ids.clone();
            s.images = sn.images.clone();
        }
        s.images.retain(|(at, _)| *at < reuse);
        // Где снять новую точку возврата: на границе вызывающего внутри хвоста, иначе на конце
        // промпта. Граница, на которой точка уже стоит (второй проход того же хода, пересчёт
        // после закрытия размышлений по бюджету), оставляет её как есть.
        let split = resume.filter(|&at| at > reuse && at < prompt_ids.len());
        let keep = split.is_none() && resume.is_some_and(|at| at <= reuse && s.snap.as_ref().is_some_and(|sn| sn.ids.len() == at));
        let snap_at = if keep { None } else { Some(split.unwrap_or(prompt_ids.len())) };
        let chunk = prefill_chunk().max(1);
        let mut off = reuse;
        let mut logits = None;
        s.ids.truncate(reuse);
        let fail = |s: &mut Inkling2Session| {
            s.ids.clear();
            s.images.clear();
            s.cache.seq_len = 0;
        };
        while off < prompt_ids.len() {
            if sink.interrupted() {
                fail(s);
                return Err(PipelineError::Model(ModelError::Forward(INTERRUPTED.into())));
            }
            let mut end = (off + chunk).min(prompt_ids.len());
            // Граница чанка — точно на точке возврата.
            if let Some(p) = snap_at.filter(|&p| p > off) {
                end = end.min(p);
            }
            // Строки картинок этого куска — готовые признаки.
            let mut patches = Vec::new();
            for &(at, t, _) in images {
                let (a, b) = (at.max(off), (at + t.dims()[0]).min(end));
                if a < b {
                    patches.push(EmbedPatch { row: a - off, embeds: t.narrow(0, a - at, b - a).and_then(|x| x.contiguous()).map_err(|e| ModelError::Forward(e.to_string()))? });
                }
            }
            let res = no_grad(|| self.model.forward_with_embeds(&prompt_ids[off..end], &patches, &mut s.cache, rows.is_some()));
            match res {
                Ok(l) => {
                    if let Some(f) = rows.as_mut() {
                        f(&l)?;
                    }
                    logits = Some(l)
                }
                Err(e) => {
                    fail(s);
                    return Err(e.into());
                }
            }
            s.ids.extend_from_slice(&prompt_ids[off..end]);
            synaptix_llm_common::prefill_progress::add(end - off);
            off = end;
            if snap_at == Some(end) {
                match self.model.snapshot(&s.cache) {
                    Ok(cache) => {
                        let imgs = s.images.iter().copied().chain(images.iter().filter(|(at, _, _)| *at >= reuse).map(|&(at, _, fp)| (at, fp))).filter(|(at, _)| *at < end).collect();
                        s.snap = Some(SessionSnap { ids: prompt_ids[..end].to_vec(), images: imgs, cache });
                    }
                    // Без снимка ход идёт дальше, следующий — по всему посчитанному либо с нуля.
                    Err(e) => {
                        eprintln!("[inkling2] точка возврата на {end} не снята: {e}");
                        s.snap = None;
                    }
                }
            }
        }
        s.images.extend(images.iter().filter(|(at, _, _)| *at >= reuse).map(|&(at, _, fp)| (at, fp)));
        Ok((logits.expect("префилл непустой"), reuse))
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
