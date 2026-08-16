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
use crate::loader::MuseWeights;
use crate::preprocess::{prepare_image, prepare_video, PreparedImage, PreparedVideo};
use crate::vision::{BundleVisionWeights, VisionTower, VIS_PREFIX};

pub use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink};

pub fn set_offload_mode_for_tests() {
    synaptix_llm_common::model::set_offload_mode(synaptix_llm_common::model::OffloadMode::Offload);
}

pub struct MusePipeline {
    pub model: DecoderModel,
    pub vision: Option<VisionTower>,
    pub tokenizer: HfTokenizer,
    pub config: MuseConfig,
    pub add_bos: bool,
}

pub struct VideoPromptInfo {
    pub groups: usize,
    pub tokens_per_group: usize,
    pub timestamps: Vec<f32>,
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
        Self::load_with_precision(path, device, PrecisionConfig::dense(dtype), None)
    }

    pub fn load_with_precision(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
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
        Ok(Self { model, vision: None, tokenizer, config, add_bos: false })
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
        let tower = VisionTower::build(vcfg, self.config.rms_norm_eps, &weights, self.model.device, dtype)
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
        use synaptix_core::grad::no_grad;
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let PreparedImage { patches, grid } = prepare_image(path, &tower.config, self.model.device)
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

    fn embed_with_media(
        &self,
        ids: &[u32],
        pad: u32,
        feats: &Tensor,
    ) -> Result<Tensor, PipelineError> {
        let device = self.model.device;
        let hidden = self.config.hidden_size;
        let total = feats.dims()[0];
        let mut segments: Vec<Tensor> = Vec::new();
        let mut cursor = 0usize;
        let mut i = 0usize;
        while i < ids.len() {
            if ids[i] == pad {
                let start = i;
                while i < ids.len() && ids[i] == pad {
                    i += 1;
                }
                let run = i - start;
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
                cursor += run;
                segments.push(e);
            } else {
                let start = i;
                while i < ids.len() && ids[i] != pad {
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
        if cursor != total {
            return Err(PipelineError::Forward(format!(
                "vision дал {total} эмбеддингов, а промпт использует {cursor}"
            )));
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
        let pad = self
            .config
            .image_token_id
            .ok_or_else(|| PipelineError::Model("config.json без image_token_id".into()))?;
        let refs: Vec<&Tensor> = image_embeds.iter().collect();
        let feats = Tensor::cat(&refs, 0).map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_media(prompt_ids, pad, &feats, gen_cfg, sink)
    }

    pub fn generate_with_video(
        &self,
        prompt_ids: &[u32],
        video_embeds: &Tensor,
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let pad = self
            .config
            .video_token_id
            .ok_or_else(|| PipelineError::Model("config.json без video_token_id".into()))?;
        self.generate_with_media(prompt_ids, pad, video_embeds, gen_cfg, sink)
    }

    fn generate_with_media(
        &self,
        prompt_ids: &[u32],
        pad: u32,
        feats: &Tensor,
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
        let emb = self.embed_with_media(prompt_ids, pad, feats)?;
        let chunk = if cfg.prefill_batch == 0 { 512 } else { cfg.prefill_batch };
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
