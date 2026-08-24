use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::config::PreprocessorConfig;
use crate::{Result, VibeVoiceError};

pub const SPEECH_START_TOKEN: &str = "<|vision_start|>";
pub const SPEECH_END_TOKEN: &str = "<|vision_end|>";
pub const SPEECH_DIFFUSION_TOKEN: &str = "<|vision_pad|>";
pub const PAD_TOKEN: &str = "<|image_pad|>";
pub const EOS_TOKEN: &str = "<|endoftext|>";

pub const SYSTEM_PROMPT: &str =
    " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

pub struct AudioNormalizer {
    pub target_db_fs: f32,
    pub eps: f32,
}

impl AudioNormalizer {
    pub fn new(target_db_fs: f32, eps: f32) -> Self {
        Self { target_db_fs, eps }
    }

    pub fn normalize(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let mean_sq = audio.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / audio.len() as f64;
        let rms = mean_sq.sqrt();
        let scalar = 10f64.powf(self.target_db_fs as f64 / 20.0) / (rms + self.eps as f64);
        let mut out: Vec<f32> = audio.iter().map(|v| (*v as f64 * scalar) as f32).collect();
        let max_val = out.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        let clip = if max_val > 1.0 {
            max_val + self.eps
        } else {
            1.0
        };
        if clip != 1.0 {
            for v in out.iter_mut() {
                *v /= clip;
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct ScriptLine {
    pub speaker: usize,
    pub text: String,
}

pub fn parse_script(script: &str) -> Result<Vec<ScriptLine>> {
    let mut parsed: Vec<(i64, String)> = Vec::new();
    for line in script.trim().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match split_speaker(line) {
            Some((id, text)) => parsed.push((id, format!(" {}", text.trim()))),
            None => continue,
        }
    }
    if parsed.is_empty() {
        return Err(VibeVoiceError::Config(
            "script: не найдено ни одной строки вида 'Speaker N: текст'".into(),
        ));
    }
    let min_id = parsed.iter().map(|(id, _)| *id).min().unwrap_or(0);
    let shift = if min_id > 0 { 1 } else { 0 };
    Ok(parsed
        .into_iter()
        .map(|(id, text)| ScriptLine {
            speaker: (id - shift).max(0) as usize,
            text,
        })
        .collect())
}

fn split_speaker(line: &str) -> Option<(i64, String)> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("speaker") {
        return None;
    }
    let rest = &line["speaker".len()..];
    let rest_trim = rest.trim_start();
    let digits: String = rest_trim.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let after = &rest_trim[digits.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let id = digits.parse::<i64>().ok()?;
    Some((id, after.to_string()))
}

pub fn plain_text_to_script(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in text.trim().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match split_speaker(line) {
            Some((id, t)) => {
                let t = t.trim();
                if !t.is_empty() {
                    lines.push(format!("Speaker {id}: {t}"));
                }
            }
            None => lines.push(format!("Speaker 1: {line}")),
        }
    }
    lines.join("\n")
}

pub struct PromptEncoding {
    pub input_ids: Vec<i64>,
    pub speech_input_mask: Vec<bool>,
    pub speech_tensors: Vec<Vec<f32>>,
    pub speech_masks: Vec<Vec<bool>>,
    pub lines: Vec<ScriptLine>,
    pub speakers: usize,
}

pub struct VibeVoiceProcessor {
    tokenizer: HfTokenizer,
    pub speech_start_id: i64,
    pub speech_end_id: i64,
    pub speech_diffusion_id: i64,
    pub pad_id: i64,
    pub eos_id: i64,
    pub compress_ratio: usize,
    pub sampling_rate: u32,
    normalizer: Option<AudioNormalizer>,
}

impl VibeVoiceProcessor {
    pub fn new(tokenizer_json: &[u8], cfg: &PreprocessorConfig) -> Result<Self> {
        let tokenizer = HfTokenizer::from_bytes(tokenizer_json)
            .map_err(|e| VibeVoiceError::Load(format!("tokenizer.json: {e}")))?;
        let id_of = |tok: &str| -> Result<i64> {
            tokenizer
                .token_to_id(tok)
                .map(|v| v as i64)
                .ok_or_else(|| VibeVoiceError::Load(format!("tokenizer: нет токена {tok}")))
        };
        let normalizer = if cfg.db_normalize {
            Some(AudioNormalizer::new(
                cfg.audio_processor.target_db_fs,
                cfg.audio_processor.eps,
            ))
        } else {
            None
        };
        Ok(Self {
            speech_start_id: id_of(SPEECH_START_TOKEN)?,
            speech_end_id: id_of(SPEECH_END_TOKEN)?,
            speech_diffusion_id: id_of(SPEECH_DIFFUSION_TOKEN)?,
            pad_id: id_of(PAD_TOKEN)?,
            eos_id: id_of(EOS_TOKEN)?,
            compress_ratio: cfg.speech_tok_compress_ratio,
            sampling_rate: cfg.audio_processor.sampling_rate,
            normalizer,
            tokenizer,
        })
    }

    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| VibeVoiceError::Load(format!("encode '{text}': {e}")))?;
        Ok(enc.ids.into_iter().map(|v| v as i64).collect())
    }

    pub fn normalize_audio(&self, wav: &[f32]) -> Vec<f32> {
        match &self.normalizer {
            Some(n) => n.normalize(wav),
            None => wav.to_vec(),
        }
    }

    pub fn build_prompt(&self, script: &str, voices: &[Vec<f32>]) -> Result<PromptEncoding> {
        let lines = parse_script(script)?;
        let mut speaker_ids: Vec<usize> = lines.iter().map(|l| l.speaker).collect();
        speaker_ids.sort_unstable();
        speaker_ids.dedup();
        let speakers = speaker_ids.len();

        let mut input_ids = self.encode_text(SYSTEM_PROMPT)?;
        let mut speech_input_mask = vec![false; input_ids.len()];

        let mut speech_tensors: Vec<Vec<f32>> = Vec::new();
        let mut speech_masks: Vec<Vec<bool>> = Vec::new();

        let used = voices.len().min(speakers.max(1));
        if !voices.is_empty() {
            let head = self.encode_text(" Voice input:\n")?;
            speech_input_mask.extend(std::iter::repeat(false).take(head.len()));
            input_ids.extend(head);

            let newline = self.encode_text("\n")?;
            let mut lengths: Vec<usize> = Vec::with_capacity(used);
            for (speaker_id, raw) in voices.iter().take(used).enumerate() {
                let wav = self.normalize_audio(raw);
                let prefix = self.encode_text(&format!(" Speaker {speaker_id}:"))?;
                let vae_len = wav.len().div_ceil(self.compress_ratio);
                speech_input_mask.extend(std::iter::repeat(false).take(prefix.len()));
                input_ids.extend(prefix);

                input_ids.push(self.speech_start_id);
                speech_input_mask.push(false);
                for _ in 0..vae_len {
                    input_ids.push(self.speech_diffusion_id);
                    speech_input_mask.push(true);
                }
                input_ids.push(self.speech_end_id);
                speech_input_mask.push(false);
                input_ids.extend(newline.iter().copied());
                speech_input_mask.extend(std::iter::repeat(false).take(newline.len()));

                lengths.push(vae_len);
                speech_tensors.push(wav);
            }
            let max_len = speech_tensors.iter().map(|s| s.len()).max().unwrap_or(0);
            let max_tok = lengths.iter().copied().max().unwrap_or(0);
            for s in speech_tensors.iter_mut() {
                s.resize(max_len, 0.0);
            }
            for l in lengths {
                let mut m = vec![false; max_tok];
                for item in m.iter_mut().take(l) {
                    *item = true;
                }
                speech_masks.push(m);
            }
        }

        let text_head = self.encode_text(" Text input:\n")?;
        speech_input_mask.extend(std::iter::repeat(false).take(text_head.len()));
        input_ids.extend(text_head);

        for line in &lines {
            let toks = self.encode_text(&format!(" Speaker {}:{}\n", line.speaker, line.text))?;
            speech_input_mask.extend(std::iter::repeat(false).take(toks.len()));
            input_ids.extend(toks);
        }

        let out_head = self.encode_text(" Speech output:\n")?;
        speech_input_mask.extend(std::iter::repeat(false).take(out_head.len() + 1));
        input_ids.extend(out_head);
        input_ids.push(self.speech_start_id);

        Ok(PromptEncoding {
            input_ids,
            speech_input_mask,
            speech_tensors,
            speech_masks,
            lines,
            speakers,
        })
    }
}
