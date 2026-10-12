use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::audio_io::{self, PadMode};
use crate::model::{InferOptions, VoxCpmModel};
use crate::tokenizer::{TextTokenizer, AUDIO_START, REF_AUDIO_END, REF_AUDIO_START};
use crate::loader::VoxCheckpoint;
use crate::VoxError;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub min_len: usize,
    pub max_len: usize,
    pub n_timesteps: usize,
    pub cfg_value: f32,
    pub seed: u64,
    pub streaming_prefix_len: usize,
    pub retry_ratio_threshold: f32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            min_len: 2,
            max_len: 2000,
            n_timesteps: 10,
            cfg_value: 2.0,
            seed: 1988,
            streaming_prefix_len: 4,
            retry_ratio_threshold: 6.0,
        }
    }
}

pub struct Waveform {
    pub pcm: Vec<f32>,
    pub sample_rate: usize,
}

struct Segment {
    tokens: Vec<u32>,
    text_mask: Vec<f32>,
    audio_mask: Vec<f32>,
    feat: Tensor,
}

pub struct VoxCpmPipeline {
    model: VoxCpmModel,
    tokenizer: TextTokenizer,
    device: Device,
    /// Где собираются входы модели: на NPU — на хосте (сборка на NPU — граф
    /// на каждую длину фразы).
    host: Device,
    compute: DType,
}

impl VoxCpmPipeline {
    pub fn from_bundle(path: impl AsRef<Path>, device: Device, compute: DType) -> Result<Self, VoxError> {
        Self::from_bundle_quant(path, device, compute, None)
    }

    /// Линейные слои квантуются при загрузке в `quant` (NVFP4/MXFP8/Sq)
    /// независимо от формата весов; `compute` — точность счёта.
    pub fn from_bundle_quant(path: impl AsRef<Path>, device: Device, compute: DType, quant: Option<DType>) -> Result<Self, VoxError> {
        let ck = VoxCheckpoint::open_quant(path, device, compute, quant.unwrap_or(compute))?;
        let tok_bytes = ck.read_file("tokenizer.json")?;
        let tokenizer = TextTokenizer::from_bytes(&tok_bytes)?;
        let model = VoxCpmModel::load(&ck)?;
        let host = if matches!(device, Device::Npu(_)) { Device::Cpu } else { device };
        Ok(Self { model, tokenizer, device, host, compute })
    }

    fn zeros_patches(&self, n: usize) -> Result<Tensor, VoxError> {
        let cfg = &self.model.config;
        Ok(Tensor::zeros(
            vec![n, cfg.patch_size, cfg.feat_dim],
            self.compute,
            self.host,
        )?)
    }

    fn text_segment(&self, ids: &[u32]) -> Result<Segment, VoxError> {
        let n = ids.len();
        Ok(Segment {
            tokens: ids.to_vec(),
            text_mask: vec![1.0; n],
            audio_mask: vec![0.0; n],
            feat: self.zeros_patches(n)?,
        })
    }

    fn audio_segment(&self, feat: Tensor) -> Result<Segment, VoxError> {
        let n = feat.dims()[0];
        Ok(Segment {
            tokens: vec![0u32; n],
            text_mask: vec![0.0; n],
            audio_mask: vec![1.0; n],
            feat,
        })
    }

    fn encode_wav(&self, path: &str, mode: PadMode) -> Result<Tensor, VoxError> {
        let sr = self.model.config.audio_vae_config.sample_rate;
        let samples = audio_io::load_resampled(path, sr)?;
        self.encode_pcm(samples, sr as u32, mode)
    }

    /// Частота дискретизации, в которой работает кодек (и выходной звук).
    pub fn sample_rate(&self) -> usize {
        self.model.config.audio_vae_config.sample_rate
    }

    /// Признаки референса из PCM (`samples` в `sr` Гц; при несовпадении —
    /// ресэмплинг).
    fn encode_pcm(&self, samples: Vec<f32>, sr: u32, mode: PadMode) -> Result<Tensor, VoxError> {
        let cfg = &self.model.config.audio_vae_config;
        let patch = self.model.config.patch_size;
        let patch_len = patch * cfg.hop_length();
        let samples = audio_io::resampled(samples, sr, cfg.sample_rate)?;
        let samples = audio_io::pad_to_multiple(samples, patch_len, mode);
        let len = samples.len();
        let x = Tensor::from_vec(samples, vec![1usize, 1, len], self.host)?;
        let mu = self.model.audio_vae.encode(&x)?;
        let t_lat = mu.dims()[2];
        let t = t_lat / patch;
        let feat = mu
            .squeeze(0)?
            .reshape((cfg.latent_dim, t, patch))?
            .permute([1usize, 2, 0])?
            .contiguous()?
            .to_dtype(self.compute)?
            .to_device(self.host)?;
        Ok(feat)
    }

    fn assemble(&self, segments: Vec<Segment>) -> Result<(Tensor, Tensor, Tensor, Tensor), VoxError> {
        let mut tokens: Vec<u32> = Vec::new();
        let mut tmask: Vec<f32> = Vec::new();
        let mut amask: Vec<f32> = Vec::new();
        let mut feats: Vec<Tensor> = Vec::new();
        for seg in &segments {
            tokens.extend_from_slice(&seg.tokens);
            tmask.extend_from_slice(&seg.text_mask);
            amask.extend_from_slice(&seg.audio_mask);
            feats.push(seg.feat.clone());
        }
        let l = tokens.len();
        let token_t = Tensor::from_vec(tokens, vec![1usize, l], self.host)?;
        let tmask_t = Tensor::from_vec(tmask, vec![1usize, l], self.host)?.to_dtype(self.compute)?;
        let amask_t = Tensor::from_vec(amask, vec![1usize, l], self.host)?.to_dtype(self.compute)?;
        let feat_t = Tensor::cat(&feats.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        Ok((token_t, tmask_t, amask_t, feat_t))
    }

    fn run(
        &self,
        segments: Vec<Segment>,
        context_len: usize,
        target_text: &str,
        opts: &GenerateOptions,
    ) -> Result<Waveform, VoxError> {
        let (token_t, tmask_t, amask_t, feat_t) = self.assemble(segments)?;

        let target_len = self.tokenizer.encode(target_text)?.len();
        let max_len = ((target_len as f32 * opts.retry_ratio_threshold) as usize + 10).min(opts.max_len);

        let infer = InferOptions {
            min_len: opts.min_len,
            max_len,
            n_timesteps: opts.n_timesteps,
            cfg_value: opts.cfg_value,
            seed: opts.seed,
        };
        let (latent, ctx) = self
            .model
            .infer(&token_t, &tmask_t, &feat_t, &amask_t, context_len, &infer)?;

        let cfg = &self.model.config.audio_vae_config;
        let decoded = self.model.audio_vae.decode(&latent)?;
        let samples = decoded.dims()[2];
        let decode_patch_len = self.model.config.patch_size * cfg.decode_chunk_size();
        let trim = decode_patch_len * ctx;
        let pcm_t = if trim > 0 && trim < samples {
            decoded.narrow(2, trim, samples - trim)?.contiguous()?
        } else {
            decoded.clone()
        };
        let n = pcm_t.dims()[2];
        let pcm = pcm_t.reshape((n,))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(Waveform { pcm, sample_rate: cfg.out_sample_rate })
    }

    pub fn synthesize(&self, text: &str, opts: &GenerateOptions) -> Result<Waveform, VoxError> {
        let mut ids = self.tokenizer.encode(text)?;
        ids.push(AUDIO_START);
        let seg = self.text_segment(&ids)?;
        self.run(vec![seg], 0, text, opts)
    }

    pub fn synthesize_with_reference(
        &self,
        text: &str,
        reference_wav: &str,
        opts: &GenerateOptions,
    ) -> Result<Waveform, VoxError> {
        let ref_feat = self.encode_wav(reference_wav, PadMode::Right)?;
        self.synthesize_with_reference_feat(text, ref_feat, opts)
    }

    /// Как [`Self::synthesize_with_reference`], но референс — PCM в памяти
    /// (например, уже синтезированная речь: следующие фразы тем же голосом).
    pub fn synthesize_with_reference_pcm(
        &self,
        text: &str,
        reference: &[f32],
        sr: u32,
        opts: &GenerateOptions,
    ) -> Result<Waveform, VoxError> {
        let ref_feat = self.encode_pcm(reference.to_vec(), sr, PadMode::Right)?;
        self.synthesize_with_reference_feat(text, ref_feat, opts)
    }

    fn synthesize_with_reference_feat(&self, text: &str, ref_feat: Tensor, opts: &GenerateOptions) -> Result<Waveform, VoxError> {
        let mut ids = self.tokenizer.encode(text)?;
        ids.push(AUDIO_START);
        let segments = vec![
            self.text_segment(&[REF_AUDIO_START])?,
            self.audio_segment(ref_feat)?,
            self.text_segment(&[REF_AUDIO_END])?,
            self.text_segment(&ids)?,
        ];
        self.run(segments, 0, text, opts)
    }

    pub fn synthesize_continuation(
        &self,
        target_text: &str,
        prompt_text: &str,
        prompt_wav: &str,
        opts: &GenerateOptions,
    ) -> Result<Waveform, VoxError> {
        let prompt_feat = self.encode_wav(prompt_wav, PadMode::Left)?;
        self.synthesize_continuation_feat(target_text, prompt_text, prompt_feat, opts)
    }

    /// Как [`Self::synthesize_continuation`], но промпт-речь — PCM в памяти.
    pub fn synthesize_continuation_pcm(
        &self,
        target_text: &str,
        prompt_text: &str,
        prompt: &[f32],
        sr: u32,
        opts: &GenerateOptions,
    ) -> Result<Waveform, VoxError> {
        let prompt_feat = self.encode_pcm(prompt.to_vec(), sr, PadMode::Left)?;
        self.synthesize_continuation_feat(target_text, prompt_text, prompt_feat, opts)
    }

    fn synthesize_continuation_feat(&self, target_text: &str, prompt_text: &str, prompt_feat: Tensor, opts: &GenerateOptions) -> Result<Waveform, VoxError> {
        let a = prompt_feat.dims()[0];
        let context_len = opts.streaming_prefix_len.saturating_sub(1).min(a);
        let text = format!("{prompt_text}{target_text}");
        let mut ids = self.tokenizer.encode(&text)?;
        ids.push(AUDIO_START);
        let segments = vec![self.text_segment(&ids)?, self.audio_segment(prompt_feat)?];
        self.run(segments, context_len, target_text, opts)
    }

    pub fn synthesize_combined(
        &self,
        target_text: &str,
        prompt_text: &str,
        prompt_wav: &str,
        reference_wav: &str,
        opts: &GenerateOptions,
    ) -> Result<Waveform, VoxError> {
        let ref_feat = self.encode_wav(reference_wav, PadMode::Right)?;
        let prompt_feat = self.encode_wav(prompt_wav, PadMode::Left)?;
        let a = prompt_feat.dims()[0];
        let context_len = opts.streaming_prefix_len.saturating_sub(1).min(a);
        let text = format!("{prompt_text}{target_text}");
        let mut ids = self.tokenizer.encode(&text)?;
        ids.push(AUDIO_START);
        let segments = vec![
            self.text_segment(&[REF_AUDIO_START])?,
            self.audio_segment(ref_feat)?,
            self.text_segment(&[REF_AUDIO_END])?,
            self.text_segment(&ids)?,
            self.audio_segment(prompt_feat)?,
        ];
        self.run(segments, context_len, target_text, opts)
    }
}
