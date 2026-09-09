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
        let cfg = self.cfg_with_eos(gen_cfg);
        let device = self.model.device;
        let kv_max = cfg.max_seq.unwrap_or(prompt_ids.len() + cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let ids = Tensor::from_vec(prompt_ids.to_vec(), vec![1usize, prompt_ids.len()], device)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let mut hidden = self
            .model
            .embed_ids(&ids)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let mut spans: Vec<(usize, usize)> = Vec::new();
        for input in media {
            let (patched, runs) = splice_media(&hidden, prompt_ids, input)?;
            hidden = patched;
            spans.extend(runs);
        }

        // Префилл чанками: пик активаций MoE растёт с длиной чанка. Границы
        // чанков произвольны — двусторонние участки задаются АБСОЛЮТНЫМИ
        // позициями и переживают разрез.
        let l = prompt_ids.len();
        let chunk = match cfg.prefill_batch {
            0 => l,
            n => n.max(1),
        };
        let t0 = std::time::Instant::now();
        let mut last_hidden = None;
        let mut off = 0usize;
        while off < l {
            let step = chunk.min(l - off);
            let part = hidden
                .narrow(1, off, step)
                .and_then(|t| t.contiguous())
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            let h = no_grad(|| {
                self.model.forward_from_hidden_spans(
                    &part,
                    &mut kv,
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
            logits = no_grad(|| self.model.forward(&step, &mut kv))
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
