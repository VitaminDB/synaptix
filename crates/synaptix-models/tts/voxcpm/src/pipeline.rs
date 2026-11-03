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
    /// Защита от «убегания» (модель не ставит стоп и бормочет): звук длиннее
    /// `символы × max_sec_per_char + extra_sec` (символы — без описания
    /// голоса в скобках) — брак; сегмент синтезируется заново с другим
    /// зерном, до `retries` раз, иначе берётся самый короткий вариант.
    /// Генерация обрывается на полуторном пределе.
    pub max_sec_per_char: f32,
    pub extra_sec: f32,
    pub retries: usize,
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
            max_sec_per_char: 0.2,
            extra_sec: 1.5,
            retries: 3,
        }
    }
}

pub struct Waveform {
    pub pcm: Vec<f32>,
    pub sample_rate: usize,
}

/// Число произносимых символов: без описания голоса `(…)` в начале и пробелов
/// по краям.
pub fn spoken_chars(text: &str) -> usize {
    let mut t = text.trim_start();
    if t.starts_with('(') {
        if let Some(e) = t.find(')') {
            t = &t[e + 1..];
        }
    }
    t.trim().chars().count()
}

/// Тихий результат — громче: некоторые голоса (и все клоны с тихой опорной
/// записи) выходят с пиком около −30 дБ, и проверка «есть ли речь» принимает
/// их за тишину. Только усиление, до пика 0.9, не больше чем в 40 раз.
fn normalize_level(mut w: Waveform) -> Waveform {
    let peak = w.pcm.iter().fold(0f32, |m, x| m.max(x.abs()));
    if peak > 1e-6 && peak < 0.5 {
        let g = (0.9 / peak).min(40.0);
        w.pcm.iter_mut().for_each(|x| *x *= g);
    }
    w
}

/// Текст как у upstream (`VoxCPM._generate`): переводы строк — пробелы, подряд
/// идущие пробельные символы — один пробел; пробел в начале убирается
/// (LlamaTokenizerFast upstream его не кодирует, сырой tokenizer.json дал бы
/// «▁▁»). Без этого «\n», двойные и ведущие пробелы дают токены, которых
/// модель при обучении не видела.
pub fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.trim_start().chars() {
        if c.is_whitespace() {
            space = true;
        } else {
            if space {
                out.push(' ');
                space = false;
            }
            out.push(c);
        }
    }
    if space {
        out.push(' ');
    }
    out
}

/// Предел длительности фразы по тексту (секунды).
pub fn duration_limit(text: &str, opts: &GenerateOptions) -> f32 {
    spoken_chars(text) as f32 * opts.max_sec_per_char + opts.extra_sec
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

    fn assemble(&self, segments: &[Segment]) -> Result<(Tensor, Tensor, Tensor, Tensor), VoxError> {
        let mut tokens: Vec<u32> = Vec::new();
        let mut tmask: Vec<f32> = Vec::new();
        let mut amask: Vec<f32> = Vec::new();
        let mut feats: Vec<Tensor> = Vec::new();
        for seg in segments {
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
        let (token_t, tmask_t, amask_t, feat_t) = self.assemble(&segments)?;

        let cfg = &self.model.config.audio_vae_config;
        let patch_samples = self.model.config.patch_size * cfg.decode_chunk_size();
        let patch_sec = patch_samples as f32 / cfg.out_sample_rate as f32;
        let limit = duration_limit(target_text, opts);
        let target_len = self.tokenizer.encode(target_text)?.len();
        let max_len = ((target_len as f32 * opts.retry_ratio_threshold) as usize + 10)
            .min((limit * 1.5 / patch_sec) as usize + 4)
            .min(opts.max_len)
            .max(opts.min_len + 2);

        let mut best: Option<Waveform> = None;
        for attempt in 0..=opts.retries {
            let infer = InferOptions {
                min_len: opts.min_len,
                max_len,
                n_timesteps: opts.n_timesteps,
                cfg_value: opts.cfg_value,
                // Другое зерно на повторе; первая попытка — заданное.
                seed: opts.seed.wrapping_add(attempt as u64 * 0x9E37_79B9),
            };
            let (latent, ctx) = self.model.infer(&token_t, &tmask_t, &feat_t, &amask_t, context_len, &infer)?;
            let generated = latent.dims()[2] / self.model.config.patch_size - ctx;
            let w = self.decode(&latent, ctx)?;
            let dur = w.pcm.len() as f32 / w.sample_rate as f32;
            let runaway = generated >= max_len || dur > limit;
            if !runaway {
                return Ok(normalize_level(w));
            }
            tracing::warn!(
                "voxcpm: брак сегмента ({dur:.1} с при пределе {limit:.1} с{}), попытка {}/{}",
                if generated >= max_len { ", без стопа" } else { "" },
                attempt + 1,
                opts.retries + 1
            );
            if best.as_ref().is_none_or(|b| w.pcm.len() < b.pcm.len()) {
                best = Some(w);
            }
        }
        Ok(normalize_level(best.expect("хотя бы одна попытка")))
    }

    fn decode(&self, latent: &Tensor, ctx: usize) -> Result<Waveform, VoxError> {
        let cfg = &self.model.config.audio_vae_config;
        let decoded = self.model.audio_vae.decode(latent)?;
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
        let text = &clean_text(text);
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
        let text = &clean_text(text);
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
        let target_text = &clean_text(target_text);
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
        let target_text = &clean_text(target_text);
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

#[cfg(test)]
mod tests {
    use super::clean_text;

    #[test]
    fn clean_text_as_upstream() {
        // re.sub(r"\s+", " ", text.replace("\n", " ")); ведущий пробел upstream
        // съедает токенизатор, хвостовой остаётся.
        assert_eq!(clean_text("  double  spaces and\nnewline "), "double spaces and newline ");
        assert_eq!(clean_text("(A calm voice)Hello,\n\tworld."), "(A calm voice)Hello, world.");
        assert_eq!(clean_text("plain"), "plain");
    }
}
