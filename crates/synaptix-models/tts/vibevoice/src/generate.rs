use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::config::GenerationConfig;
use crate::model::VibeVoiceModel;
use crate::processor::{PromptEncoding, VibeVoiceProcessor};
use crate::schedule::DpmSolverMultistep;
use crate::{err, Result, VibeVoiceError};

pub struct NormalRng {
    state: u64,
    spare: Option<f32>,
    zeros: bool,
}

impl NormalRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
            spare: None,
            zeros: false,
        }
    }

    pub fn zeros() -> Self {
        Self {
            state: 0,
            spare: None,
            zeros: true,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        if self.zeros {
            return 0.0;
        }
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }

    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}

pub struct GenerationOutput {
    pub audio: Vec<f32>,
    pub tokens: Vec<i64>,
    pub reached_max: bool,
}

pub struct SpeechGenerator<'a> {
    model: &'a VibeVoiceModel,
    processor: &'a VibeVoiceProcessor,
    scheduler: DpmSolverMultistep,
    rng: NormalRng,
}

impl<'a> SpeechGenerator<'a> {
    pub fn new(
        model: &'a VibeVoiceModel,
        processor: &'a VibeVoiceProcessor,
        seed: u64,
    ) -> Result<Self> {
        Self::with_rng(model, processor, NormalRng::new(seed))
    }

    pub fn with_rng(
        model: &'a VibeVoiceModel,
        processor: &'a VibeVoiceProcessor,
        rng: NormalRng,
    ) -> Result<Self> {
        Ok(Self {
            scheduler: model.new_scheduler()?,
            model,
            processor,
            rng,
        })
    }

    pub fn randn(&mut self, dims: Vec<usize>) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data = self.rng.normals(n);
        Tensor::from_vec(data, dims, self.model.device)
            .map_err(err)?
            .to_dtype(self.model.dtype)
            .map_err(err)
    }

    pub fn encode_voice_latents(&mut self, prompt: &PromptEncoding) -> Result<Option<Tensor>> {
        if prompt.speech_tensors.is_empty() {
            return Ok(None);
        }
        let n = prompt.speech_tensors.len();
        let l = prompt.speech_tensors[0].len();
        let mut flat = Vec::with_capacity(n * l);
        for wav in &prompt.speech_tensors {
            flat.extend_from_slice(wav);
        }
        let audio = Tensor::from_vec(flat, vec![n, 1usize, l], self.model.device)
            .map_err(err)?
            .to_dtype(self.model.dtype)
            .map_err(err)?;
        let mean = self.model.acoustic.encode(&audio, None)?;
        let value = self.model.acoustic.fix_std / 0.8;
        let per_std: Vec<f32> = (0..n).map(|_| self.rng.normal() * value).collect();
        let noise = self.randn(mean.dims().to_vec())?;
        let latents = self
            .model
            .acoustic
            .sample_with_noise(&mean, &per_std, &noise)?;
        Ok(Some(latents))
    }

    pub fn build_prompt_embeds(
        &mut self,
        prompt: &PromptEncoding,
        voice_latents: Option<&Tensor>,
    ) -> Result<Tensor> {
        let embeds = self.model.lm.embed_tokens(&prompt.input_ids)?;
        let Some(latents) = voice_latents else {
            return Ok(embeds);
        };
        let features = self.model.scale_latents(latents)?;
        let connected = self.model.acoustic_connector.forward(&features)?;
        let dims = connected.dims().to_vec();
        let (n, t, h) = (dims[0], dims[1], dims[2]);
        let flat = connected.reshape(vec![n * t, h]).map_err(err)?;

        let mut idx: Vec<i64> = Vec::new();
        for (i, mask) in prompt.speech_masks.iter().enumerate() {
            for (j, on) in mask.iter().enumerate() {
                if *on {
                    idx.push((i * t + j) as i64);
                }
            }
        }
        let want = prompt.speech_input_mask.iter().filter(|v| **v).count();
        if idx.len() != want {
            return Err(VibeVoiceError::Inference(format!(
                "voice prompt: {} латентов против {} слотов",
                idx.len(),
                want
            )));
        }
        let idx_t = Tensor::from_vec(idx.clone(), vec![idx.len()], self.model.device).map_err(err)?;
        let sel = flat.index_select(0, &idx_t).map_err(err)?;

        let mut pieces: Vec<Tensor> = Vec::new();
        let mut pos = 0usize;
        let mut speech_off = 0usize;
        let s = prompt.speech_input_mask.len();
        while pos < s {
            let on = prompt.speech_input_mask[pos];
            let mut end = pos;
            while end < s && prompt.speech_input_mask[end] == on {
                end += 1;
            }
            let len = end - pos;
            if on {
                pieces.push(
                    sel.narrow(0, speech_off, len)
                        .and_then(|t| t.contiguous())
                        .and_then(|t| t.reshape(vec![1usize, len, h]))
                        .map_err(err)?,
                );
                speech_off += len;
            } else {
                pieces.push(
                    embeds
                        .narrow(1, pos, len)
                        .and_then(|t| t.contiguous())
                        .map_err(err)?,
                );
            }
            pos = end;
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        Tensor::cat(&refs, 1).map_err(err)
    }

    pub fn sample_latent(
        &mut self,
        positive: &Tensor,
        negative: &Tensor,
        cfg_scale: f32,
        steps: usize,
        init_noise: &Tensor,
    ) -> Result<Tensor> {
        self.scheduler.set_timesteps(steps);
        let condition = Tensor::cat(&[positive, negative], 0).map_err(err)?;
        let mut x = init_noise.clone();
        let batch = x.dims()[0];
        let timesteps = self.scheduler.timesteps.clone();
        for t in timesteps {
            let combined = Tensor::cat(&[&x, &x], 0).map_err(err)?;
            let ts = vec![t; batch * 2];
            let eps = self.model.head.forward(&combined, &ts, &condition)?;
            let cond_eps = eps.narrow(0, 0, batch).and_then(|t| t.contiguous()).map_err(err)?;
            let uncond_eps = eps
                .narrow(0, batch, batch)
                .and_then(|t| t.contiguous())
                .map_err(err)?;
            let guided = cond_eps
                .sub(&uncond_eps)
                .and_then(|d| d.mul_scalar(cfg_scale))
                .and_then(|d| d.add(&uncond_eps))
                .map_err(err)?;
            x = self.scheduler.step(&guided, &x)?;
        }
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        prompt: &PromptEncoding,
        cfg: &GenerationConfig,
        mut on_chunk: Option<&mut dyn FnMut(&[f32])>,
        mut on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<GenerationOutput> {
        let model = self.model;
        let proc = self.processor;
        let prompt_len = prompt.input_ids.len();
        let max_pos = model.config.decoder_config.max_position_embeddings;
        let max_new = cfg
            .max_new_tokens
            .unwrap_or_else(|| max_pos.saturating_sub(prompt_len));
        let max_length = prompt_len + max_new;
        let by_ratio = (cfg.max_length_times * prompt_len as f32) as usize;
        let max_steps = max_new.min(by_ratio);
        if max_steps == 0 {
            return Err(VibeVoiceError::Inference("пустой бюджет генерации".into()));
        }

        let cap = (prompt_len + max_steps + 1).min(max_pos.max(prompt_len + max_steps + 1));
        let mut pos_cache = model.lm.new_cache(cap)?;
        let neg_cap = (max_steps + 2).min(cap);
        let mut neg_cache = model.lm.new_cache(neg_cap)?;
        let mut acoustic_cache = model.acoustic.new_cache();
        let mut semantic_cache = model.semantic.new_cache();

        let voice = self.encode_voice_latents(prompt)?;
        let prompt_embeds = self.build_prompt_embeds(prompt, voice.as_ref())?;

        let valid_ids = [
            proc.speech_start_id,
            proc.speech_end_id,
            proc.speech_diffusion_id,
            proc.eos_id,
        ];
        let head_rows = self.constrained_head(&valid_ids)?;

        let mut tokens: Vec<i64> = Vec::new();
        let mut audio: Vec<f32> = Vec::new();
        let mut inputs_embeds: Option<Tensor> = None;
        let mut neg_pending_reset = true;
        let mut neg_started = false;
        let mut reached_max = false;
        let vae_dim = model.config.acoustic_vae_dim();

        for step in 0..max_steps {
            if pos_cache.len() >= max_length {
                reached_max = true;
                break;
            }
            let step_input = match &inputs_embeds {
                Some(e) => e.clone(),
                None => prompt_embeds.clone(),
            };
            let hidden = model.lm.forward(&step_input, &mut pos_cache)?;
            let last = hidden.dims()[1] - 1;
            let last_hidden = hidden
                .narrow(1, last, 1)
                .and_then(|t| t.contiguous())
                .and_then(|t| t.reshape(vec![1usize, model.lm.hidden_size()]))
                .map_err(err)?;
            let scores = last_hidden
                .linear(&head_rows)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(err)?;
            let mut best = 0usize;
            for (i, v) in scores.iter().enumerate() {
                if *v > scores[best] {
                    best = i;
                }
            }
            let next_token = valid_ids[best];
            tokens.push(next_token);

            if let Some(cb) = on_step.as_deref_mut() {
                cb(step + 1, max_steps);
            }

            if next_token == proc.eos_id {
                break;
            }
            if next_token == proc.speech_end_id {
                acoustic_cache.zero_all()?;
                semantic_cache.zero_all()?;
            }
            if next_token == proc.speech_start_id {
                neg_pending_reset = true;
            }

            let mut next_embed = model.lm.embed_tokens(&[next_token])?;

            if next_token == proc.speech_diffusion_id {
                if neg_pending_reset {
                    neg_cache.reset();
                    neg_started = false;
                    neg_pending_reset = false;
                }
                let neg_input = if !neg_started {
                    neg_started = true;
                    model.lm.embed_tokens(&[proc.speech_start_id])?
                } else {
                    match &inputs_embeds {
                        Some(e) => e.clone(),
                        None => model.lm.embed_tokens(&[proc.speech_start_id])?,
                    }
                };
                let neg_hidden = model.lm.forward(&neg_input, &mut neg_cache)?;
                let nlast = neg_hidden.dims()[1] - 1;
                let neg_last = neg_hidden
                    .narrow(1, nlast, 1)
                    .and_then(|t| t.contiguous())
                    .and_then(|t| t.reshape(vec![1usize, model.lm.hidden_size()]))
                    .map_err(err)?;

                let init = self.randn(vec![1usize, vae_dim])?;
                let latent = self.sample_latent(
                    &last_hidden,
                    &neg_last,
                    cfg.cfg_scale,
                    cfg.ddpm_inference_steps,
                    &init,
                )?;
                let latent3 = latent.reshape(vec![1usize, 1usize, vae_dim]).map_err(err)?;
                let scaled = model.unscale_latents(&latent3)?;
                let chunk = model
                    .acoustic
                    .decode(&scaled, Some(&mut acoustic_cache))?;
                let pcm = chunk
                    .to_dtype(DType::F32)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(err)?;
                if let Some(cb) = on_chunk.as_deref_mut() {
                    cb(&pcm);
                }
                audio.extend_from_slice(&pcm);

                let sem = model.semantic.encode(&chunk, Some(&mut semantic_cache))?;
                let a_emb = self.model.acoustic_connector.forward(&latent3)?;
                let s_emb = self.model.semantic_connector.forward(&sem)?;
                next_embed = a_emb.add(&s_emb).map_err(err)?;
            }

            inputs_embeds = Some(next_embed);
            if step + 1 == max_steps {
                reached_max = true;
            }
        }

        Ok(GenerationOutput {
            audio,
            tokens,
            reached_max,
        })
    }

    fn constrained_head(&self, ids: &[i64]) -> Result<Tensor> {
        self.model.lm.lm_head_rows(ids)
    }
}
