use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::DecoderModel;
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::MuseConfig;
use crate::config::VisionConfig;
use crate::dflash::{self, DFlashConfig, DFlashModule};
use crate::loader::MuseWeights;
use crate::preprocess::{prepare_image, prepare_video, PreparedImage, PreparedVideo};
use crate::vision::{BundleVisionWeights, VisionTower, VIS_PREFIX};

pub use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink};

pub fn set_offload_mode_for_tests() {
    synaptix_llm_common::model::set_offload_mode(synaptix_llm_common::model::OffloadMode::Offload);
}

/// Запас VRAM поверх весов башни: активации forward'а и фрагментация пула.
/// Меньше — и резидентная загрузка проходит впритык, а падает уже на первой
/// картинке.
///
/// 2 ГБ — не про загрузку, а про сам forward: видео идёт в башню одним
/// батчем на 96 кадров (13 824 патча), и промежуточные тензоры MLP на
/// `intermediate_size` весят под гигабайт. С прежними 512 МБ на 24 ГБ рядом
/// с 19 ГБ весов LLM резидентная загрузка ровно проходила по порогу — а
/// кодирование потом падало с OOM на 85-мегабайтной активации. Порог должен
/// выбирать послойную загрузку раньше, чем впритык: она медленнее по диску,
/// но помещается всегда.
const VISION_VRAM_RESERVE: usize = 2048 * 1024 * 1024;

/// Сколько весят трансформерные блоки башни в выбранном dtype.
/// На блок: attn (q/k/v/proj с bias) + mlp (fc1/fc2 с bias) + две нормы.
fn tower_block_bytes(cfg: &VisionConfig, dtype: DType) -> usize {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let per_layer = 4 * h * h + 4 * h + 2 * h * i + i + h + 4 * h;
    per_layer * cfg.num_hidden_layers * dtype.size_in_bits() / 8
}

/// Память устройства, доступная под новые аллокации; `None` — для CPU и
/// когда драйвер не ответил.
///
/// Это не только «свободно» по `cuMemGetInfo`: свободные блоки, которые уже
/// держит пул аллокатора, драйвер считает занятыми, а башня садится ровно в
/// них. После генерации `reserved` пула выше `used` на размер отработавшего
/// KV-ринга — по одному лишь `cuMemGetInfo` башня выглядит невлезающей там,
/// где на деле места вдвое больше.
fn free_vram(device: Device) -> Option<usize> {
    let Device::Cuda(ordinal) = device else {
        return None;
    };
    let (free, _) = synaptix_core::device::cuda::mem_info(ordinal).ok()?;
    let slack = |(reserved, used): (u64, u64)| reserved.saturating_sub(used) as usize;
    let pool = synaptix_core::memory::cuda_pool::cuda_mempool_stats(ordinal)
        .map(slack)
        .unwrap_or(0);
    Some(free + pool)
}

pub struct MusePipeline {
    pub model: DecoderModel,
    pub vision: Option<VisionTower>,
    pub dflash: Option<DFlashModule>,
    pub tokenizer: HfTokenizer,
    pub config: MuseConfig,
    pub add_bos: bool,
}

pub struct VideoPromptInfo {
    pub groups: usize,
    pub tokens_per_group: usize,
    pub timestamps: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LookupStats {
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
}

impl LookupStats {
    pub fn acceptance(&self) -> f32 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f32 / self.drafted as f32
        }
    }
}

/// Черновик усечён до 8 токенов: замеры на RTX 5090 (NVFP4) дают 61.6 tok/s
/// против 56.8 на полном блоке из 15 — хвост блока почти не принимается,
/// а verify-чанк дорожает.
const DFLASH_DEFAULT_SPAN: usize = 8;

const LOOKUP_NGRAM: usize = 3;
const LOOKUP_SPAN: usize = 12;

fn lookup_draft(ids: &[u32], max_ngram: usize, span: usize) -> Vec<u32> {
    if span == 0 || ids.len() < 2 {
        return Vec::new();
    }
    for n in (1..=max_ngram.min(ids.len() - 1)).rev() {
        let needle = &ids[ids.len() - n..];
        let hay_end = ids.len() - 1;
        let mut i = hay_end;
        while i >= n {
            let start = i - n;
            if &ids[start..i] == needle {
                let cont_start = i;
                let cont_end = (cont_start + span).min(ids.len());
                if cont_end > cont_start {
                    return ids[cont_start..cont_end].to_vec();
                }
                break;
            }
            i -= 1;
        }
    }
    Vec::new()
}

impl VideoPromptInfo {
    pub fn prompt_block(&self) -> String {
        let mut s = String::from("<|vid_start|>");
        for g in 0..self.groups {
            let ts = self.timestamps.get(g).copied().unwrap_or(0.0);
            s.push_str(&format!("Time: {ts:.1}s"));
            s.push_str(&"<|video|>".repeat(self.tokens_per_group));
            if g + 1 < self.groups {
                s.push_str("<|vid_frame_separator|>");
            } else {
                s.push_str("<|vid_end|>");
            }
        }
        s
    }
}

impl MusePipeline {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        Self::load_with_precision(path, device, PrecisionConfig::dense(dtype), None)
    }

    pub fn load_with_precision(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let weights = MuseWeights::load(path, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer = HfTokenizer::from_bytes(&weights.tokenizer_json)
            .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?;
        let cap = max_seq.unwrap_or_else(|| config.max_position_embeddings.min(4096));
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build_auto(
            &dcfg,
            &weights,
            device,
            precision.compute,
            precision.attn_w,
            precision.mlp_w,
            precision.lm_head,
            precision.embed,
            cap,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        .with_kv_cache_dtype(precision.kv);
        Ok(Self { model, vision: None, dflash: None, tokenizer, config, add_bos: false })
    }

    /// Подключить DFlash-драфтер из того же бандла (компонент `dflash`).
    /// Возвращает `false`, если драфтера в бандле нет.
    pub fn load_dflash(
        &mut self,
        path: impl AsRef<Path>,
        precision: PrecisionConfig,
    ) -> Result<bool, PipelineError> {
        let path = path.as_ref();
        let weights = dflash::BundleDFlashWeights::open(path, self.model.device)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        if !dflash::present(&weights) {
            return Ok(false);
        }
        let cfg_bytes = synaptix_bundle::Bundle::open(path)
            .and_then(|b| b.read_file("dflash_config.json").map(|c| c.into_owned()))
            .map_err(|e| PipelineError::Load(format!("dflash_config.json: {e}")))?;
        let dcfg = DFlashConfig::from_hf_bytes(&cfg_bytes)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        if dcfg.hidden_size != self.config.hidden_size {
            return Err(PipelineError::Load(format!(
                "dflash hidden {} != target hidden {}",
                dcfg.hidden_size, self.config.hidden_size
            )));
        }
        if let Some(&max_tap) = dcfg.target_layer_ids.iter().max() {
            if max_tap >= self.config.num_hidden_layers {
                return Err(PipelineError::Load(format!(
                    "dflash target_layer_ids содержит слой {max_tap} ≥ {} слоёв target'а",
                    self.config.num_hidden_layers
                )));
            }
        }
        // Драфтер маленький (2.3B), но его черновики напрямую определяют
        // приёмку — агрессивный квант съедает выигрыш. Дефолт MXFP8 (2.6 ГБ),
        // переопределяется `SYN_DFLASH_W=nvfp4|mxfp8|f16|bf16`.
        let dw = match std::env::var("SYN_DFLASH_W").as_deref() {
            Ok("nvfp4") => DType::NVFP4,
            Ok("f16") => DType::F16,
            Ok("bf16") => DType::BF16,
            Ok("mxfp8") => DType::MXFP8,
            _ => {
                if precision.attn_w.is_quantized() {
                    DType::MXFP8
                } else {
                    precision.attn_w
                }
            }
        };
        let module = DFlashModule::build(
            dcfg,
            &weights,
            self.model.device,
            precision.compute,
            dw,
            dw,
        )
        .map_err(|e| PipelineError::Load(format!("dflash: {e}")))?;
        self.dflash = Some(module);
        Ok(true)
    }

    pub fn load_vision(&mut self, path: impl AsRef<Path>, dtype: DType) -> Result<bool, PipelineError> {
        let Some(vcfg) = self.config.vision.clone() else {
            return Ok(false);
        };
        let path = path.as_ref();
        let weights = BundleVisionWeights::open(path, self.model.device)
            .map_err(|e| PipelineError::Load(format!("vision: {e}")))?;
        if !weights.has(&format!("{VIS_PREFIX}.ln_pre.weight")) {
            return Ok(false);
        }
        let eps = self.config.rms_norm_eps;
        let device = self.model.device;
        // Веса блоков башни (у 30B — 3.45 ГБ в BF16) кладутся поверх уже
        // поднятой LLM: если свободной VRAM меньше, чем нужно, резидентная
        // загрузка упирается в OOM на середине. В этом случае идём послойно —
        // медленнее по диску, но помещается всегда.
        let need = tower_block_bytes(&vcfg, dtype);
        let free = free_vram(device);
        let tower = if free.is_some_and(|f| f < need.saturating_add(VISION_VRAM_RESERVE)) {
            eprintln!(
                "[muse_glimmer] vision-башня {} MB не влезает в {} MB свободной VRAM — \
                 послойная загрузка",
                need / (1024 * 1024),
                free.unwrap_or(0) / (1024 * 1024)
            );
            VisionTower::build_streaming(vcfg, eps, weights, device, dtype)
        } else {
            VisionTower::build(vcfg, eps, &weights, device, dtype)
        }
        .map_err(|e| PipelineError::Load(format!("vision: {e}")))?;
        self.vision = Some(tower);
        Ok(true)
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn release_vision(&mut self) {
        if self.vision.take().is_some() {
            if let Device::Cuda(o) = self.model.device {
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
            }
        }
    }

    pub fn encode_image(&self, path: impl AsRef<Path>) -> Result<Tensor, PipelineError> {
        self.encode_image_limited(path, None)
    }

    /// То же, что [`Self::encode_image`], но с потолком на число vision-токенов.
    ///
    /// `max_image_tokens` из `processor_config.json` у Muse Glimmer — 4096, то
    /// есть одна крупная картинка съедает контекст сопоставимо с длинной
    /// статьёй и заметно удлиняет prefill. Интерактивным вызывающим (чат)
    /// нужен свой, более скромный лимит, поэтому здесь мы работаем на копии
    /// `VisionConfig`: башня резолюционно-агностична (window-attention +
    /// интерполяция pos-таблицы по grid), меняется только `smart_resize`
    /// в препроцессинге.
    pub fn encode_image_limited(
        &self,
        path: impl AsRef<Path>,
        max_image_tokens: Option<usize>,
    ) -> Result<Tensor, PipelineError> {
        use synaptix_core::grad::no_grad;
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let cfg = match max_image_tokens {
            Some(n) if n > 0 && n < tower.config.max_image_tokens => {
                let mut c = tower.config.clone();
                c.max_image_tokens = n;
                c
            }
            _ => tower.config.clone(),
        };
        let PreparedImage { patches, grid } = prepare_image(path, &cfg, self.model.device)
            .map_err(|e| PipelineError::Load(format!("image: {e}")))?;
        no_grad(|| tower.forward(&patches, grid))
            .map_err(|e| PipelineError::Forward(format!("vision forward: {e}")))
    }

    pub fn encode_video(&self, path: impl AsRef<Path>) -> Result<(Tensor, VideoPromptInfo), PipelineError> {
        use synaptix_core::grad::no_grad;
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let PreparedVideo { patches, grid, group_timestamps } =
            prepare_video(path, &tower.config, self.model.device)
                .map_err(|e| PipelineError::Load(format!("video: {e}")))?;
        let feats = no_grad(|| tower.forward(&patches, grid))
            .map_err(|e| PipelineError::Forward(format!("vision forward: {e}")))?;
        let info = VideoPromptInfo {
            groups: grid.t,
            tokens_per_group: (grid.h * grid.w) / tower.config.merge_unit(),
            timestamps: group_timestamps,
        };
        Ok((feats, info))
    }

    pub fn image_token_count(&self, path: impl AsRef<Path>) -> Result<usize, PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let img = synaptix_io::image::png::load_image(path, Device::Cpu)
            .map_err(|e| PipelineError::Load(format!("image: {e}")))?;
        let dims = img.dims();
        let cfg = &tower.config;
        let unit = cfg.patch_size * cfg.merge_size;
        let (nh, nw) = crate::preprocess::smart_resize(dims[1], dims[2], unit, cfg.max_image_tokens);
        Ok((nh / cfg.patch_size) * (nw / cfg.patch_size) / cfg.merge_unit())
    }

    /// Собирает входные эмбеддинги промпта, подменяя прогоны медиа-плейсхолдеров
    /// строками из `media`.
    ///
    /// `media` — список пар «id токена-заполнителя → матрица эмбеддингов
    /// `[tokens, hidden]`». Пар может быть несколько (картинки и видео в одном
    /// промпте живут под разными id: `image_token_id` / `video_token_id`), у
    /// каждой — свой курсор, поэтому модальности читаются независимо, но
    /// строго в порядке появления их токенов в промпте.
    fn embed_with_media_pads(
        &self,
        ids: &[u32],
        media: &[(u32, &Tensor)],
    ) -> Result<Tensor, PipelineError> {
        let device = self.model.device;
        let hidden = self.config.hidden_size;
        let is_pad = |t: u32| media.iter().any(|(p, _)| *p == t);
        // Курсор на каждую модальность — индекс следующей неиспользованной строки.
        let mut cursors = vec![0usize; media.len()];
        let mut segments: Vec<Tensor> = Vec::new();
        let mut i = 0usize;
        while i < ids.len() {
            let tok = ids[i];
            if let Some(slot) = media.iter().position(|(p, _)| *p == tok) {
                let start = i;
                while i < ids.len() && ids[i] == tok {
                    i += 1;
                }
                let run = i - start;
                let feats = media[slot].1;
                let total = feats.dims()[0];
                let cursor = cursors[slot];
                if cursor + run > total {
                    return Err(PipelineError::Forward(format!(
                        "медиа-токенов в промпте больше, чем строк эмбеддингов: {} > {total}",
                        cursor + run
                    )));
                }
                let e = feats
                    .narrow(0, cursor, run)
                    .and_then(|t| t.contiguous())
                    .and_then(|t| t.to_dtype(self.model.dtype))
                    .and_then(|t| t.reshape(vec![1usize, run, hidden]))
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                cursors[slot] = cursor + run;
                segments.push(e);
            } else {
                let start = i;
                while i < ids.len() && !is_pad(ids[i]) {
                    i += 1;
                }
                let chunk = Tensor::from_vec(ids[start..i].to_vec(), vec![1usize, i - start], device)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                let e = self
                    .model
                    .embed_ids(&chunk)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                segments.push(e);
            }
        }
        for (slot, (_, feats)) in media.iter().enumerate() {
            let total = feats.dims()[0];
            if cursors[slot] != total {
                return Err(PipelineError::Forward(format!(
                    "vision дал {total} эмбеддингов, а промпт использует {}",
                    cursors[slot]
                )));
            }
        }
        let refs: Vec<&Tensor> = segments.iter().collect();
        Tensor::cat(&refs, 1).map_err(|e| PipelineError::Forward(e.to_string()))
    }

    pub fn generate_with_images(
        &self,
        prompt_ids: &[u32],
        image_embeds: &[Tensor],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let refs: Vec<&Tensor> = image_embeds.iter().collect();
        let feats = Tensor::cat(&refs, 0).map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_mixed_media(prompt_ids, Some(&feats), None, gen_cfg, sink)
    }

    pub fn generate_with_video(
        &self,
        prompt_ids: &[u32],
        video_embeds: &Tensor,
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        self.generate_with_mixed_media(prompt_ids, None, Some(video_embeds), gen_cfg, sink)
    }

    /// Генерация по промпту, где встречаются картинки и/или видео.
    ///
    /// `image_embeds` / `video_embeds` — конкатенация эмбеддингов всех
    /// вложений своей модальности **в порядке их появления в промпте**;
    /// прогоны `image_token_id` и `video_token_id` разбираются независимыми
    /// курсорами, поэтому одно сообщение может нести и то, и другое.
    pub fn generate_with_mixed_media(
        &self,
        prompt_ids: &[u32],
        image_embeds: Option<&Tensor>,
        video_embeds: Option<&Tensor>,
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let mut media: Vec<(u32, &Tensor)> = Vec::new();
        if let Some(t) = image_embeds {
            let pad = self
                .config
                .image_token_id
                .ok_or_else(|| PipelineError::Model("config.json без image_token_id".into()))?;
            media.push((pad, t));
        }
        if let Some(t) = video_embeds {
            let pad = self
                .config
                .video_token_id
                .ok_or_else(|| PipelineError::Model("config.json без video_token_id".into()))?;
            media.push((pad, t));
        }
        if media.is_empty() {
            return Err(PipelineError::Model("generate_with_mixed_media без медиа".into()));
        }
        self.generate_with_media(prompt_ids, &media, gen_cfg, sink)
    }

    fn generate_with_media(
        &self,
        prompt_ids: &[u32],
        media: &[(u32, &Tensor)],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::grad::no_grad;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.prepare_cfg(gen_cfg);
        let device = self.model.device;
        let l = prompt_ids.len();
        let kv_max = cfg.max_seq.unwrap_or(l + cfg.max_new_tokens + 1);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let eos = synaptix_llm_common::generate::eos_set(&cfg);
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&cfg, prompt_ids);

        let t0 = std::time::Instant::now();
        let emb = self.embed_with_media_pads(prompt_ids, media)?;
        let chunk = match cfg.prefill_batch {
            0 => 512,
            n => n.min(synaptix_llm_common::model::RING_SLACK),
        };
        let mut off = 0usize;
        let mut last_hidden = None;
        while off < l {
            let step = chunk.min(l - off);
            let part = emb
                .narrow(1, off, step)
                .and_then(|t| t.contiguous())
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let h = no_grad(|| self.model.forward_from_hidden(&part, &mut kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            last_hidden = Some(h);
            off += step;
        }
        let hidden = last_hidden.ok_or_else(|| PipelineError::Forward("empty prefill".into()))?;
        let mut logits = self
            .model
            .head_at(&hidden, hidden.dims()[1] - 1)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let dec_t0 = std::time::Instant::now();
        if cfg.temperature == 0.0 {
            let first = sampler.sample(&logits).map_err(PipelineError::from)?;
            out.push(first);
            let mut all_ids: Vec<u32> = prompt_ids.to_vec();
            all_ids.push(first);
            if sink.on_token(first) && !eos.contains(&first) {
                self.lookup_loop(&mut kv, &mut all_ids, &mut out, &cfg, &eos, sink)?;
            }
        } else {
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
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                logits = no_grad(|| self.model.forward(&step, &mut kv))
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
            }
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        let new_tokens = out.len();
        Ok((
            out,
            GenerationStats { prompt_tokens: l, new_tokens, prefill_ms, decode_ms },
        ))
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    fn maybe_prepend_bos(&self, prompt_ids: &[u32]) -> Vec<u32> {
        match self.config.bos_token_id {
            Some(bos) if self.add_bos && prompt_ids.first() != Some(&bos) => {
                let mut v = Vec::with_capacity(prompt_ids.len() + 1);
                v.push(bos);
                v.extend_from_slice(prompt_ids);
                v
            }
            _ => prompt_ids.to_vec(),
        }
    }

    fn prepare_cfg(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_ids = self.config.eos_token_ids.clone();
        }
        let cap = synaptix_llm_common::model::RING_SLACK;
        cfg.prefill_batch = match cfg.prefill_batch {
            0 => 1024,
            n => n.min(cap),
        };
        cfg
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate(&self.model, &prompt, &cfg)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate_streaming(&self.model, &prompt, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        gen_cfg: GenerationConfig,
    ) -> Result<(String, GenerationStats), PipelineError> {
        let ids = self.encode(prompt)?;
        let (new_ids, stats) = self.generate(&ids, gen_cfg)?;
        let text = self.decode(&new_ids)?;
        Ok((text, stats))
    }

    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// Спекулятивный декод на DFlash-драфтере: блок из `draft_len` кандидатов за
    /// один forward драфтера, верификация одним чанком target'а. Greedy-путь
    /// lossless: принимаются только токены, совпавшие с argmax target'а.
    pub fn generate_dflash_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, LookupStats), PipelineError> {
        use synaptix_core::grad::no_grad;

        let dflash = self
            .dflash
            .as_ref()
            .ok_or_else(|| PipelineError::Model("DFlash-драфтер не загружен".into()))?;
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.prepare_cfg(gen_cfg);
        if cfg.temperature > 0.0 {
            return Err(PipelineError::Model(
                "DFlash-декод реализован для greedy (--temperature 0)".into(),
            ));
        }
        let device = self.model.device;
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let l = prompt.len();
        let kv_max = cfg.max_seq.unwrap_or(l + cfg.max_new_tokens + 1);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let mut dcache = dflash
            .make_cache()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let eos = synaptix_llm_common::generate::eos_set(&cfg);
        let taps = dflash.config.target_layer_ids.clone();
        // Драфтер денойзит весь блок сразу, поэтому хвост блока угадывается
        // хуже головы: усечение черновика удешевляет verify-чанк, почти не
        // теряя принятых токенов. Подбирается `SYN_DFLASH_SPAN`.
        let draft_len = std::env::var("SYN_DFLASH_SPAN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, dflash.config.draft_len()))
            .unwrap_or_else(|| DFLASH_DEFAULT_SPAN.min(dflash.config.draft_len()));

        let t0 = std::time::Instant::now();
        let chunk = cfg.prefill_batch.max(1);
        let mut off = 0usize;
        let mut last_hidden: Option<Tensor> = None;
        // Контекст драфтера после prefill — hidden ВСЕХ токенов промпта:
        // anchor'ом диффузионного окна становится первый сгенерированный токен.
        let mut tap_chunks: Vec<Vec<Tensor>> = vec![Vec::new(); taps.len()];
        let ctx_pos0 = 0usize;
        while off < l {
            let end = (off + chunk).min(l);
            let part = Tensor::from_vec(prompt[off..end].to_vec(), vec![1usize, end - off], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let (h, tapped) = no_grad(|| self.model.forward_trunk_tapped(&part, &mut kv, &taps))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            for (slot, t) in tap_chunks.iter_mut().zip(tapped.into_iter()) {
                slot.push(t);
            }
            last_hidden = Some(h);
            off = end;
        }
        let hidden = last_hidden.ok_or_else(|| PipelineError::Forward("empty prefill".into()))?;
        let mut ctx_pos = ctx_pos0;
        let mut ctx_taps: Vec<Tensor> = tap_chunks
            .iter()
            .map(|parts| {
                if parts.len() == 1 {
                    Ok(parts[0].clone())
                } else {
                    let refs: Vec<&Tensor> = parts.iter().collect();
                    Tensor::cat(&refs, 1)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let logits = self
            .model
            .head_at(&hidden, hidden.dims()[1] - 1)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let prefill_ms = t0.elapsed().as_millis();

        // CUDA-argmax по словарю 202k оказался в 2.5× медленнее host-пути
        // (generic-reduce без специализации), поэтому логиты уезжают в host.
        let argmax_rows = |t: &Tensor| -> Result<Vec<u32>, PipelineError> {
            let dims = t.dims().to_vec();
            let vocab = dims[dims.len() - 1];
            let rows = t.numel() / vocab;
            let v = t
                .to_dtype(DType::F32)
                .and_then(|x| x.flatten_all())
                .and_then(|x| x.to_vec1::<f32>())
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            Ok((0..rows)
                .map(|r| {
                    v[r * vocab..(r + 1) * vocab]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                })
                .collect())
        };

        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let mut anchor = argmax_rows(&logits)?[0];
        out.push(anchor);
        let mut cancelled = !sink.on_token(anchor);
        let mut stats = LookupStats::default();
        // Диагностика: пересобирать контекст драфтера с нуля каждый блок
        // (проверка инкрементального кэша). Дорого по памяти, только для отладки.
        let full_ctx = std::env::var("SYN_DFLASH_FULLCTX").is_ok();
        let mut all_taps: Vec<Vec<Tensor>> = ctx_taps.iter().map(|t| vec![t.clone()]).collect();

        let dec_t0 = std::time::Instant::now();
        'outer: while !cancelled && out.len() < cfg.max_new_tokens {
            if eos.contains(&anchor) {
                break;
            }
            let pos = kv.seq_len;
            if pos + 1 >= kv.max_seq {
                break;
            }
            let budget = (cfg.max_new_tokens - out.len()).min(kv.max_seq - pos - 1);
            if budget == 0 {
                break;
            }

            let (draft_ctx, draft_pos) = if full_ctx {
                dcache.reset();
                let joined: Vec<Tensor> = all_taps
                    .iter()
                    .map(|parts| {
                        let refs: Vec<&Tensor> = parts.iter().collect();
                        if refs.len() == 1 { Ok(refs[0].clone()) } else { Tensor::cat(&refs, 1) }
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                (joined, 0usize)
            } else {
                (ctx_taps.clone(), ctx_pos)
            };
            let dlogits = no_grad(|| {
                dflash.draft_logits(&self.model, &mut dcache, &draft_ctx, draft_pos, anchor)
            })
            .map_err(|e| PipelineError::Forward(format!("dflash draft: {e}")))?;
            let mut draft = argmax_rows(&dlogits)?;
            draft.truncate(draft_len.min(budget));
            stats.steps += 1;
            stats.drafted += draft.len();

            let mut chunk_ids = Vec::with_capacity(1 + draft.len());
            chunk_ids.push(anchor);
            chunk_ids.extend_from_slice(&draft);
            let s = chunk_ids.len();
            let t = Tensor::from_vec(chunk_ids, vec![1usize, s], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let (hh, tapped) = no_grad(|| self.model.forward_trunk_tapped(&t, &mut kv, &taps))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let lg = self
                .model
                .heads_all(&hh)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let preds = argmax_rows(&lg)?;

            let mut accepted = 0usize;
            while accepted < draft.len() && preds[accepted] == draft[accepted] {
                accepted += 1;
            }
            if std::env::var("SYN_DFLASH_DEBUG").is_ok() && stats.steps <= 4 {
                let n = draft.len().min(8);
                eprintln!(
                    "[dflash#{}] anchor={anchor} ctx_pos={ctx_pos} m={} draft={:?} preds={:?} accepted={accepted}",
                    stats.steps,
                    ctx_taps[0].dims()[1],
                    &draft[..n],
                    &preds[..n],
                );
            }
            stats.accepted += accepted;
            kv.seq_len = pos + 1 + accepted;

            // Контекст следующего блока — принятые токены этого чанка
            // (anchor + accepted), их hidden берём из tap-слоёв verify-прохода.
            let keep = accepted + 1;
            ctx_pos = pos;
            ctx_taps = tapped
                .iter()
                .map(|x| x.narrow(1, 0, keep).and_then(|y| y.contiguous()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            if full_ctx {
                for (slot, t) in all_taps.iter_mut().zip(ctx_taps.iter()) {
                    slot.push(t.clone());
                }
            }

            for &tok in preds.iter().take(keep) {
                out.push(tok);
                cancelled = !sink.on_token(tok);
                anchor = tok;
                if cancelled || out.len() >= cfg.max_new_tokens || eos.contains(&tok) {
                    break 'outer;
                }
            }
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        let new_tokens = out.len();
        Ok((
            out,
            GenerationStats { prompt_tokens: l, new_tokens, prefill_ms, decode_ms },
            stats,
        ))
    }

    pub fn generate_lookup_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, LookupStats), PipelineError> {
        use synaptix_core::grad::no_grad;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.prepare_cfg(gen_cfg);
        if cfg.temperature > 0.0 {
            return Err(PipelineError::Model(
                "lookup-декод реализован для greedy (--temperature 0)".into(),
            ));
        }
        let device = self.model.device;
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let l = prompt.len();
        let kv_max = cfg.max_seq.unwrap_or(l + cfg.max_new_tokens + 1);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let eos = synaptix_llm_common::generate::eos_set(&cfg);

        let argmax_rows = |logits: &Tensor| -> Result<Vec<u32>, PipelineError> {
            let dims = logits.dims().to_vec();
            let vocab = dims[dims.len() - 1];
            let rows = logits.numel() / vocab;
            let v = logits
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            Ok((0..rows)
                .map(|r| {
                    let row = &v[r * vocab..(r + 1) * vocab];
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                })
                .collect())
        };

        let t0 = std::time::Instant::now();
        let chunk = cfg.prefill_batch.max(1);
        let mut off = 0usize;
        let mut logits_opt = None;
        while off < l {
            let end = (off + chunk).min(l);
            let part = Tensor::from_vec(prompt[off..end].to_vec(), vec![1usize, end - off], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let lg = no_grad(|| self.model.forward(&part, &mut kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            logits_opt = Some(lg);
            off = end;
        }
        let logits = logits_opt.ok_or_else(|| PipelineError::Forward("empty prefill".into()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut all_ids: Vec<u32> = prompt.clone();
        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let first = argmax_rows(&logits)?[0];
        out.push(first);
        all_ids.push(first);
        let cancelled = !sink.on_token(first);

        let dec_t0 = std::time::Instant::now();
        let stats = if cancelled {
            LookupStats::default()
        } else {
            self.lookup_loop(&mut kv, &mut all_ids, &mut out, &cfg, &eos, sink)?
        };
        let decode_ms = dec_t0.elapsed().as_millis();
        let new_tokens = out.len();
        Ok((
            out,
            GenerationStats { prompt_tokens: l, new_tokens, prefill_ms, decode_ms },
            stats,
        ))
    }

    fn lookup_loop(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        all_ids: &mut Vec<u32>,
        out: &mut Vec<u32>,
        cfg: &GenerationConfig,
        eos: &std::collections::HashSet<u32>,
        sink: &mut dyn StreamSink,
    ) -> Result<LookupStats, PipelineError> {
        use synaptix_core::grad::no_grad;
        let device = self.model.device;
        let mut stats = LookupStats::default();
        let argmax_rows = |logits: &Tensor| -> Result<Vec<u32>, PipelineError> {
            let dims = logits.dims().to_vec();
            let vocab = dims[dims.len() - 1];
            let rows = logits.numel() / vocab;
            let v = logits
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            Ok((0..rows)
                .map(|r| {
                    let row = &v[r * vocab..(r + 1) * vocab];
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                })
                .collect())
        };
        let mut cancelled = false;
        'outer: while !cancelled && out.len() < cfg.max_new_tokens {
            let cur = *out.last().unwrap();
            if eos.contains(&cur) {
                break;
            }
            let pos = kv.seq_len;
            if pos + 1 > kv.max_seq {
                break;
            }
            let budget = (cfg.max_new_tokens - out.len()).min(kv.max_seq - pos - 1);
            let draft = lookup_draft(all_ids, LOOKUP_NGRAM, LOOKUP_SPAN.min(budget));
            let mut chunk_ids = Vec::with_capacity(1 + draft.len());
            chunk_ids.push(cur);
            chunk_ids.extend_from_slice(&draft);
            let s = chunk_ids.len();
            let t = Tensor::from_vec(chunk_ids, vec![1usize, s], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let hidden = no_grad(|| self.model.forward_trunk(&t, kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let lg = self
                .model
                .heads_all(&hidden)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let preds = argmax_rows(&lg)?;
            stats.steps += 1;
            stats.drafted += draft.len();

            let mut accepted = 0usize;
            while accepted < draft.len() && preds[accepted] == draft[accepted] {
                accepted += 1;
            }
            stats.accepted += accepted;
            kv.seq_len = pos + 1 + accepted;

            for &tok in preds.iter().take(accepted + 1) {
                out.push(tok);
                all_ids.push(tok);
                cancelled = !sink.on_token(tok);
                if cancelled || out.len() >= cfg.max_new_tokens || eos.contains(&tok) {
                    break 'outer;
                }
            }
        }
        Ok(stats)
    }

    pub fn graph_decode_supported(&self) -> bool {
        matches!(self.model.device, Device::Cuda(_))
            && matches!(self.model.dtype, DType::F16 | DType::BF16)
            && self.model.kv_dtype != DType::MXFP8
            && !self.model.has_mxfp8_head_or_embed()
    }

    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, &prompt, gen_cfg, sink)
    }

    pub fn generate_with_graph_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let device = self.model.device;
        let ord = match device {
            Device::Cuda(o) => o,
            _ => return Err(PipelineError::Forward("generate_with_graph требует CUDA".into())),
        };
        let l = prompt_ids.len();
        let cfg = self.prepare_cfg(gen_cfg);
        let eos = synaptix_llm_common::generate::eos_set(&cfg);
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&cfg, prompt_ids);
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;

        let suffix = &prompt_ids[prefix..];
        let chunk = cfg.prefill_batch.max(1);
        let t0 = std::time::Instant::now();
        let mut logits_opt = None;
        let mut off = 0usize;
        while off < suffix.len() {
            let end = (off + chunk).min(suffix.len());
            let part = Tensor::from_vec(suffix[off..end].to_vec(), vec![1usize, end - off], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let lg = no_grad(|| self.model.forward(&part, kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            logits_opt = Some(lg);
            off = end;
        }
        let logits = logits_opt.ok_or_else(|| PipelineError::Forward("empty prefill suffix".into()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let tok0 = sampler.sample(&logits).map_err(PipelineError::from)?;
        out.push(tok0);
        let mut cancelled = !sink.on_token(tok0);

        let mut state = self
            .model
            .make_decode_state()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let start0 = self
            .model
            .ring_prepare_decode(kv, l)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        state
            .update_ring(tok0, l as u32, start0 as u32)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| PipelineError::Forward(format!("stream: {e}")))?;

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
        .map_err(|e| PipelineError::Forward(format!("graph capture: {e}")))?;
        let _ = graph.upload();

        let dec_t0 = std::time::Instant::now();
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
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            state
                .update_ring(last, pos as u32, start as u32)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            graph
                .launch()
                .map_err(|e| PipelineError::Forward(format!("graph launch: {e:?}")))?;
            stream
                .synchronize()
                .map_err(|e| PipelineError::Forward(format!("sync post-launch: {e:?}")))?;
            let tok = sampler.sample(&state.logits).map_err(PipelineError::from)?;
            out.push(tok);
            cancelled = !sink.on_token(tok);
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        kv.seq_len = (l + out.len() - 1).min(kv.max_seq);
        let new_tokens = out.len();

        Ok((
            out,
            GenerationStats {
                prompt_tokens: l,
                new_tokens,
                prefill_ms,
                decode_ms,
            },
        ))
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
    #[error("forward: {0}")]
    Forward(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        Self::Model(e.to_string())
    }
}
