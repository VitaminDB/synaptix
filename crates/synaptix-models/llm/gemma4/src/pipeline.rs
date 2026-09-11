//! Пайплайн Gemma-4: токенайзер + общий декодер с расширенным профилем.

use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_core::grad::no_grad;
use synaptix_llm_common::model::RopePositions;
use synaptix_llm_common::{
    generate, DecoderModel, GenerationConfig, GenerationStats, KvCache, ModelError, StreamSink,
};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Gemma4Config;
use crate::loader::{is_bundle, read_aux, Gemma4Weights};
use crate::vision::{aspect_preserving_size, VisionConfig, VisionTower};

pub struct Gemma4Pipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: Gemma4Config,
    /// Шаблон чата из чекпойнта (`chat_template.jinja`).
    pub chat_template: Option<String>,
    /// Конфиг башни зрения; `None` — text-only чекпойнт.
    pub vision_config: Option<VisionConfig>,
    /// Башня зрения. Грузится по требованию: она стоит около гигабайта VRAM,
    /// а нужна только на ходе с картинкой.
    vision: Option<VisionTower>,
    model_path: PathBuf,
}

/// Готовые эмбеддинги картинок для префилла: `embeds` — все мягкие токены
/// подряд, `pad` — id токена-заполнителя в промпте.
pub struct MediaInput {
    pub pad: u32,
    pub embeds: Tensor,
}

fn load_tokenizer(path: &Path) -> Result<HfTokenizer, String> {
    if is_bundle(path) {
        let bytes = read_aux(path, "tokenizer.json").map_err(|e| e.to_string())?;
        return HfTokenizer::from_bytes(&bytes).map_err(|e| e.to_string());
    }
    HfTokenizer::from_file(path.join("tokenizer.json")).map_err(|e| e.to_string())
}

impl Gemma4Pipeline {
    pub fn load(
        model: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, PipelineError> {
        Self::load_with_precision(model, device, PrecisionConfig::dense(dtype), None)
    }

    /// Основная точка входа: точность по компонентам, `max_seq` — ёмкость
    /// RoPE-кэша (весь `max_position_embeddings` в 256k никто не аллоцирует).
    pub fn load_with_precision(
        model: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций: смешанный пул за
        // длинный префилл деградирует в «решето».
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let path: PathBuf = model.as_ref().to_path_buf();
        // tokenizer.json на 262k-вокаб разбирается дольше, чем грузятся веса,
        // и ни от чего не зависит — читаем его параллельно сборке модели.
        let tok_path = path.clone();
        let tok_handle = std::thread::spawn(move || load_tokenizer(&tok_path));

        // Активации Gemma переваливают за 65504 (massive activations), поэтому
        // резидуальный поток обязан быть BF16 даже при квантованных весах —
        // каст bf16↔f16 вокруг квант-ядра делает сам `QLinear::Quant`.
        let mut precision = precision;
        if precision.compute == DType::F16 {
            precision.compute = DType::BF16;
        }
        let weights = Gemma4Weights::open(&path, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let chat_template = weights.chat_template.clone();
        let dcfg = config.to_decoder_config();
        let rope_capacity = max_seq
            .unwrap_or(config.max_position_embeddings)
            .min(config.max_position_embeddings);
        let model = DecoderModel::build(
            &dcfg,
            &weights,
            device,
            precision.compute,
            precision.attn_w,
            precision.mlp_w,
            precision.lm_head,
            precision.embed,
            rope_capacity,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        .with_kv_cache_dtype(precision.kv);

        let cfg_bytes = read_aux(&path, "config.json")
            .map_err(|e| PipelineError::Load(format!("config.json: {e}")))?;
        let vision_config = VisionConfig::from_hf_bytes(&cfg_bytes)
            .filter(|_| weights.has_vision_tower());

        let tokenizer = tok_handle
            .join()
            .map_err(|_| PipelineError::Load("поток токенайзера упал".into()))?
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        Ok(Self {
            model,
            tokenizer,
            config,
            chat_template,
            vision_config,
            vision: None,
            model_path: path,
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        self.tokenizer
            .encode(prompt, true)
            .map(|e| e.ids)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    fn cfg_with_eos(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        // Ход Gemma-4 закрывается не только `<eos>`: `<turn|>` (и его вариант
        // из generation_config) — такие же стоп-токены, одного первого мало.
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_id = self.config.eos_token_ids.first().copied();
            cfg.eos_token_ids = self.config.eos_token_ids.clone();
        }
        cfg
    }

    pub fn make_kv_cache(&self, max_seq: usize) -> Result<KvCache, PipelineError> {
        self.model
            .make_kv_cache(1, max_seq)
            .map_err(|e| PipelineError::Model(e.to_string()))
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        generate::generate(&self.model, prompt_ids, &cfg).map_err(PipelineError::from)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        generate::generate_streaming(&self.model, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    /// Стрим с переиспользованием префикса: `kv.seq_len` токенов промпта уже
    /// посчитаны, префиллится только хвост.
    pub fn generate_streaming_resume(
        &self,
        kv: &mut KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        gen_cfg: GenerationConfig,
    ) -> Result<(String, GenerationStats), PipelineError> {
        let ids = self.encode(prompt)?;
        let (new_ids, stats) = self.generate(&ids, gen_cfg)?;
        Ok((self.decode(&new_ids)?, stats))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("model: {0}")]
    Model(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("generate: {0}")]
    Generate(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        PipelineError::Generate(e.to_string())
    }
}

// ── Зрение ──────────────────────────────────────────────────────────────────

impl Gemma4Pipeline {
    /// Есть ли в чекпойнте башня зрения (по конфигу и по наличию весов).
    pub fn has_vision_config(&self) -> bool {
        self.vision_config.is_some()
    }

    /// Загружена ли башня в память устройства.
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Идемпотентно догружает башню зрения. `Ok(false)` — её нет в чекпойнте.
    pub fn ensure_vision(&mut self, dtype: DType) -> Result<bool, PipelineError> {
        if self.vision.is_some() {
            return Ok(true);
        }
        let Some(cfg) = self.vision_config.clone() else { return Ok(false) };
        let _weights =
            synaptix_core::device::cuda::WeightsAllocGuard::for_device(self.model.device);
        let weights = Gemma4Weights::open(&self.model_path, self.model.device, dtype)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let tower = VisionTower::load(&weights, cfg, self.model.device, dtype)
            .map_err(|e| PipelineError::Load(format!("башня зрения: {e}")))?;
        self.vision = Some(tower);
        Ok(true)
    }

    /// Отпустить башню (её VRAM нужна контексту между ходами с картинками).
    pub fn release_vision(&mut self) {
        self.vision = None;
    }

    /// Сколько мягких токенов займёт картинка такого размера.
    pub fn soft_tokens_for(&self, height: usize, width: usize, max_soft: Option<usize>) -> usize {
        let Some(cfg) = &self.vision_config else { return 0 };
        let max_soft = max_soft.unwrap_or(cfg.max_soft_tokens).min(cfg.max_soft_tokens);
        let pool = cfg.pooling_kernel_size;
        match aspect_preserving_size(height, width, cfg.patch_size, max_soft * pool * pool, pool) {
            Ok((th, tw)) => (th / cfg.patch_size) * (tw / cfg.patch_size) / (pool * pool),
            Err(_) => 0,
        }
    }

    /// Картинка с диска → мягкие токены `[S, hidden]` в пространстве LM.
    ///
    /// Ресайз — билинейный: HF берёт бикубический с антиалиасингом, точного
    /// совпадения тут нет (как и у остальных vision-башен этого движка).
    pub fn encode_image(
        &self,
        path: &Path,
        max_soft: Option<usize>,
    ) -> Result<Tensor, PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("башня зрения не загружена".into()))?;
        let cfg = &tower.cfg;
        let img = synaptix_io::image::png::load_image(path, Device::Cpu)
            .map_err(|e| PipelineError::Load(format!("картинка: {e}")))?;
        let dims = img.dims().to_vec();
        if dims.len() != 3 || dims[0] < 3 {
            return Err(PipelineError::Model(format!("ожидался [C>=3, H, W], а не {dims:?}")));
        }
        let (h, w) = (dims[1], dims[2]);
        let max_soft = max_soft.unwrap_or(cfg.max_soft_tokens).min(cfg.max_soft_tokens);
        let pool = cfg.pooling_kernel_size;
        let (th, tw) = aspect_preserving_size(h, w, cfg.patch_size, max_soft * pool * pool, pool)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let rgb = if dims[0] == 3 {
            img
        } else {
            img.narrow(0, 0, 3)
                .and_then(|t| t.contiguous())
                .map_err(|e| PipelineError::Model(e.to_string()))?
        };
        let resized = if (th, tw) == (h, w) {
            rgb
        } else {
            synaptix_io::image::augment::resize_bilinear(&rgb, th, tw)
                .map_err(|e| PipelineError::Model(format!("ресайз: {e}")))?
        };
        let chw = resized
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        tower
            .encode_chw(&chw, th, tw)
            .map_err(|e| PipelineError::Model(format!("башня зрения: {e}")))
    }

    /// Стрим по промпту с картинками: эмбеддинги встают на места прогонов
    /// токена-заполнителя, а внимание внутри каждого такого прогона —
    /// двустороннее (`use_bidirectional_attention = "vision"`).
    pub fn generate_media_streaming(
        &self,
        prompt_ids: &[u32],
        media: &[MediaInput],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        self.generate_media_resume(&mut kv, prompt_ids, media, gen_cfg, sink)
    }

    /// Как [`Self::generate_media_streaming`], но по ГОТОВОМУ кэшу: префилл
    /// стартует с `kv.seq_len` (префикс-KV), эмбеддинги вложений берутся
    /// только для заполнителей хвоста. Вызывающий отвечает за то, что
    /// `prompt_ids[..kv.seq_len]` — ровно те токены (и те же эмбеддинги на
    /// местах заполнителей), что лежат в кэше. Граница префикса — конец
    /// прошлого промпта, внутрь блока картинки она не попадает, так что
    /// двусторонние участки хвоста целы.
    pub fn generate_media_resume(
        &self,
        kv: &mut KvCache,
        prompt_ids: &[u32],
        media: &[MediaInput],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        let device = self.model.device;
        let l = prompt_ids.len();
        if l > kv.max_seq {
            return Err(PipelineError::Model(format!(
                "промпт {l} ток не влезает в KV-кэш на {} ток",
                kv.max_seq
            )));
        }
        // Префикс-KV: всё до `kv.seq_len` уже посчитано. Минус один токен —
        // логиты нужно получить хотя бы из одного forward'а.
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;
        let tail = &prompt_ids[prefix..];

        let ids = Tensor::from_vec(tail.to_vec(), vec![1usize, tail.len()], device)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let mut hidden = self
            .model
            .embed_ids(&ids)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        // Участки двустороннего внимания — в АБСОЛЮТНЫХ позициях контекста.
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for input in media {
            let Some(tail_input) = media_rows_for(prompt_ids, input, prefix, l - prefix)? else {
                continue;
            };
            let (patched, runs) = splice_media(&hidden, tail, &tail_input)?;
            hidden = patched;
            spans.extend(runs.into_iter().map(|(a, b)| (a + prefix, b + prefix)));
        }

        // Префилл чанками: пик активаций MoE растёт с длиной чанка. Границы
        // чанков произвольны — двусторонние участки заданы абсолютными
        // позициями и переживают разрез.
        let chunk = match cfg.prefill_batch {
            0 => l,
            n => n.max(1),
        };
        let chunk = self.model.max_prefill_chunk().map_or(chunk, |cap| chunk.min(cap));
        let t0 = std::time::Instant::now();
        let mut last_hidden = None;
        let mut off = prefix;
        while off < l {
            let step = chunk.min(l - off);
            let part = hidden
                .narrow(1, off - prefix, step)
                .and_then(|t| t.contiguous())
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            let h = no_grad(|| {
                self.model.forward_from_hidden_spans(
                    &part,
                    kv,
                    RopePositions::Sequential,
                    &spans,
                )
            })
            .map_err(|e| PipelineError::Model(e.to_string()))?;
            last_hidden = Some(h);
            off += step;
        }
        let last = last_hidden.ok_or_else(|| PipelineError::Model("пустой префилл".into()))?;
        let mut logits = self
            .model
            .head_at(&last, last.dims()[1] - 1)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let eos = generate::eos_set(&cfg);
        let mut sampler = generate::TokenSampler::new(&cfg, prompt_ids);
        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let dec_t0 = std::time::Instant::now();
        loop {
            let tok = sampler.sample(&logits).map_err(PipelineError::from)?;
            out.push(tok);
            if !sink.on_token(tok) || out.len() >= cfg.max_new_tokens || eos.contains(&tok) {
                break;
            }
            if kv.seq_len >= kv.max_seq {
                break;
            }
            let step = Tensor::from_vec(vec![tok], vec![1usize, 1], device)
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            logits = no_grad(|| self.model.forward(&step, kv))
                .map_err(|e| PipelineError::Model(e.to_string()))?;
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        let new_tokens = out.len();
        Ok((
            out,
            GenerationStats { prompt_tokens: l, new_tokens, prefill_ms, decode_ms },
        ))
    }
}

/// Строки эмбеддингов, чьи заполнители попадают в `[offset, offset+len)`
/// промпта: заполнители нумеруются по всему промпту, поэтому при префилле
/// хвоста берутся ровно строки его слотов. `None` — в хвосте слотов этой
/// модальности нет.
fn media_rows_for(
    prompt: &[u32],
    input: &MediaInput,
    offset: usize,
    len: usize,
) -> Result<Option<MediaInput>, PipelineError> {
    let before = prompt[..offset].iter().filter(|t| **t == input.pad).count();
    let inside = prompt[offset..offset + len].iter().filter(|t| **t == input.pad).count();
    if inside == 0 {
        return Ok(None);
    }
    let have = input.embeds.dims()[0];
    if before + inside > have {
        return Err(PipelineError::Model(format!(
            "заполнителей {}, а эмбеддингов {have}",
            before + inside
        )));
    }
    let embeds = input
        .embeds
        .narrow(0, before, inside)
        .and_then(|t| t.contiguous())
        .map_err(|e| PipelineError::Model(format!("медиа: срез строк: {e}")))?;
    Ok(Some(MediaInput { pad: input.pad, embeds }))
}

/// Подставляет эмбеддинги медиа на места прогонов токена-заполнителя.
/// Возвращает новый `hidden` и участки `[начало, конец)` каждого прогона.
fn splice_media(
    hidden: &Tensor,
    prompt_ids: &[u32],
    input: &MediaInput,
) -> Result<(Tensor, Vec<(usize, usize)>), PipelineError> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < prompt_ids.len() {
        if prompt_ids[i] != input.pad {
            i += 1;
            continue;
        }
        let start = i;
        while i < prompt_ids.len() && prompt_ids[i] == input.pad {
            i += 1;
        }
        runs.push((start, i));
    }
    let slots: usize = runs.iter().map(|(a, b)| b - a).sum();
    let have = input.embeds.dims()[0];
    if slots != have {
        return Err(PipelineError::Model(format!(
            "заполнителей {slots}, а эмбеддингов {have}"
        )));
    }
    if runs.is_empty() {
        return Ok((hidden.clone(), runs));
    }
    let dims = hidden.dims().to_vec();
    let hsz = dims[2];
    let embeds = input
        .embeds
        .to_dtype(hidden.dtype())
        .and_then(|t| t.to_device(hidden.device()))
        .map_err(|e| PipelineError::Model(e.to_string()))?;

    // Собираем последовательность заново кусками: текст берём из `hidden`,
    // блоки — из `embeds`. Кусков мало (по два на картинку), а in-place записи
    // среза у тензора нет.
    let mut parts: Vec<Tensor> = Vec::new();
    let mut cursor = 0usize;
    let mut taken = 0usize;
    for (a, b) in &runs {
        if *a > cursor {
            parts.push(
                hidden
                    .narrow(1, cursor, a - cursor)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| PipelineError::Model(e.to_string()))?,
            );
        }
        let len = b - a;
        parts.push(
            embeds
                .narrow(0, taken, len)
                .and_then(|t| t.contiguous())
                .and_then(|t| t.reshape(vec![1usize, len, hsz]))
                .map_err(|e| PipelineError::Model(e.to_string()))?,
        );
        taken += len;
        cursor = *b;
    }
    if cursor < dims[1] {
        parts.push(
            hidden
                .narrow(1, cursor, dims[1] - cursor)
                .and_then(|t| t.contiguous())
                .map_err(|e| PipelineError::Model(e.to_string()))?,
        );
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let out = Tensor::cat(&refs, 1).map_err(|e| PipelineError::Model(e.to_string()))?;
    Ok((out, runs))
}

// ── Графовый декод ──────────────────────────────────────────────────────────

impl Gemma4Pipeline {
    /// Можно ли захватить шаг декода в CUDA-граф.
    ///
    /// Про KV спрашиваем не политику, а ФАКТ: у Gemma-4 квантованный кэш не
    /// достаётся ни одному слою (sliding отсекает окно, global — голова 512),
    /// поэтому профиль с `kv = mxfp8` графу не мешает. Голова в MXFP8 тоже не
    /// мешает: её скретчи прогреваются тремя прогонами до захвата. А вот
    /// MXFP8-таблица эмбеддингов ходит своим ядром — с ней граф не берём.
    pub fn graph_decode_supported(&self) -> bool {
        matches!(self.model.device, Device::Cuda(_))
            && matches!(self.model.dtype, DType::F16 | DType::BF16)
            && self.model.kv_all_dense()
            && !self.model.has_mxfp8_embed()
            && self.model.graph_decode_ready()
    }

    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, prompt_ids, gen_cfg, sink)
    }

    /// Как [`Self::generate_with_graph_streaming`], но префилл стартует с
    /// `kv.seq_len` — кэш переиспользуется между ходами чата.
    pub fn generate_with_graph_resume(
        &self,
        kv: &mut KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        let eos = generate::eos_set(&cfg);
        let mut sampler = generate::TokenSampler::new(&cfg, prompt_ids);
        let device = self.model.device;
        let Device::Cuda(ord) = device else {
            return Err(PipelineError::Model("графовый декод требует CUDA".into()));
        };
        let l = prompt_ids.len();
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;

        // Префилл идёт обычным путём: он упирается в счёт, а не в запуски ядер,
        // и захватывать его смысла нет.
        let suffix = &prompt_ids[prefix..];
        // Чанк — как можно длиннее: MoE считает экспертов групповым GEMM, и
        // его цена на слой почти не зависит от числа токенов в чанке; предел
        // ставит кольцевой KV sliding-слоёв (2048 у Gemma-4).
        let ring_cap = self.model.max_prefill_chunk().unwrap_or(usize::MAX);
        let chunk = if cfg.prefill_batch > 0 { cfg.prefill_batch } else { 2048 }.min(ring_cap);
        let t0 = std::time::Instant::now();
        let mut logits_opt: Option<Tensor> = None;
        let mut off = 0usize;
        while off < suffix.len() {
            let end = (off + chunk).min(suffix.len());
            let part = Tensor::from_vec(suffix[off..end].to_vec(), vec![1usize, end - off], device)
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            let lg = no_grad(|| self.model.forward(&part, kv))
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            logits_opt = Some(lg);
            off = end;
        }
        let logits = logits_opt.ok_or_else(|| PipelineError::Model("пустой хвост промпта".into()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let tok0 = sampler.sample(&logits).map_err(PipelineError::from)?;
        out.push(tok0);
        let mut cancelled = !sink.on_token(tok0);

        let mut state = self
            .model
            .make_decode_state()
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        // Кольцевой KV sliding-слоёв: сдвиг окна делает хост ДО запуска графа,
        // а в граф уходят уже device-резидентные позиция и длина.
        let start0 = self
            .model
            .ring_prepare_decode(kv, l)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        state
            .update_ring(tok0, l as u32, start0 as u32)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| PipelineError::Model(format!("stream: {e}")))?;

        let mut capturer = GraphCapturer::new(3);
        let graph = {
            let model = &self.model;
            let state_ref = &mut state;
            let kv_ref = &mut *kv;
            no_grad(|| {
                capturer.capture_with(&stream, |_s| {
                    model
                        .forward_decode_dev(state_ref, kv_ref)
                        .map_err(|e| InferError::Other(e.to_string()))
                })
            })
        }
        .map_err(|e| PipelineError::Model(format!("захват графа: {e}")))?;
        let _ = graph.upload();

        let dec_t0 = std::time::Instant::now();
        // Шаг захвата уже посчитал логиты для следующего токена.
        if !cancelled && out.len() < cfg.max_new_tokens && !eos.contains(&tok0) {
            let tok1 = sampler.sample(&state.logits).map_err(PipelineError::from)?;
            out.push(tok1);
            cancelled = !sink.on_token(tok1);
        }
        while !cancelled && out.len() < cfg.max_new_tokens {
            let last = *out.last().unwrap();
            if eos.contains(&last) {
                break;
            }
            let pos = l + out.len() - 1;
            if pos >= kv.max_seq {
                break;
            }
            let start = self
                .model
                .ring_prepare_decode(kv, pos)
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            state
                .update_ring(last, pos as u32, start as u32)
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            graph
                .launch()
                .map_err(|e| PipelineError::Model(format!("запуск графа: {e:?}")))?;
            // Запись логитов графом не отслеживается событиями — без барьера
            // выгрузка на хост в сэмплере обогнала бы граф.
            stream
                .synchronize()
                .map_err(|e| PipelineError::Model(format!("sync после запуска: {e:?}")))?;
            let tok = sampler.sample(&state.logits).map_err(PipelineError::from)?;
            out.push(tok);
            cancelled = !sink.on_token(tok);
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        kv.seq_len = (l + out.len() - 1).min(kv.max_seq);
        let new_tokens = out.len();
        Ok((
            out,
            GenerationStats { prompt_tokens: l, new_tokens, prefill_ms, decode_ms },
        ))
    }
}
