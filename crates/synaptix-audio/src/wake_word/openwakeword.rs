use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::audio::silero_vad::{SileroVadConfig, SileroVadModel};

use crate::error::{AudioError, Result};
use crate::mel::{log_mel_spectrogram, MelConfig, MelNorm, MelScale};
use crate::stft::{PadMode, StftConfig};
use crate::window::WindowKind;

pub struct OpenWakeWordConfig {
    pub model_path: std::path::PathBuf,
    pub threshold: f32,
    pub sample_rate: u32,
    pub n_mels: usize,
}

impl Default for OpenWakeWordConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::new(),
            threshold: 0.5,
            sample_rate: 16000,
            n_mels: 96,
        }
    }
}

pub struct OpenWakeWord {
    pub config: OpenWakeWordConfig,
    stft_cfg: StftConfig,
    mel_cfg: MelConfig,
    keywords: HashMap<String, KeywordSlot>,
    energy_threshold: f32,
}

struct KeywordSlot {
    model: SileroVadModel,
    h: Tensor,
    c: Tensor,
}

impl OpenWakeWord {
    pub fn new(config: OpenWakeWordConfig) -> Self {
        let mel_cfg = MelConfig {
            sample_rate: config.sample_rate,
            n_fft: 512,
            f_min: 0.0,
            f_max: config.sample_rate as f32 / 2.0,
            n_mels: config.n_mels,
            mel_scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
        };
        let stft_cfg = StftConfig {
            n_fft: mel_cfg.n_fft,
            hop_length: mel_cfg.n_fft / 4,
            win_length: mel_cfg.n_fft,
            window: WindowKind::Hann,
            center: false,
            pad_mode: PadMode::Zero,
        };
        let energy_threshold = (config.threshold.max(0.0) * 0.02).max(1.0e-4);
        Self {
            config,
            stft_cfg,
            mel_cfg,
            keywords: HashMap::new(),
            energy_threshold,
        }
    }

    pub fn default_keyword_config(&self) -> SileroVadConfig {
        SileroVadConfig {
            spec_bins: self.config.n_mels,
            stem_channels: 64,
            hidden_size: 64,
            num_conv_blocks: 3,
            conv_kernel: 3,
        }
    }

    pub fn add_keyword(&mut self, name: impl Into<String>, model: SileroVadModel) -> Result<()> {
        if model.config.spec_bins != self.config.n_mels {
            return Err(AudioError::invalid_arg(format!(
                "OpenWakeWord: keyword model.spec_bins={} != n_mels={}",
                model.config.spec_bins, self.config.n_mels
            )));
        }
        let (h, c) = model
            .zero_state(1, Device::Cpu, DType::F32)
            .map_err(|e| AudioError::other(e.to_string()))?;
        self.keywords.insert(name.into(), KeywordSlot { model, h, c });
        Ok(())
    }

    pub fn keyword_names(&self) -> Vec<&str> {
        self.keywords.keys().map(|s| s.as_str()).collect()
    }

    pub fn reset(&mut self) {
        for slot in self.keywords.values_mut() {
            if let Ok((h, c)) = slot.model.zero_state(1, Device::Cpu, DType::F32) {
                slot.h = h;
                slot.c = c;
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        !self.keywords.is_empty()
    }

    fn compute_mel(&self, frame: &[f32]) -> Result<Option<Tensor>> {
        let mel = log_mel_spectrogram(frame, &self.stft_cfg, &self.mel_cfg)?;
        if mel.is_empty() {
            return Ok(None);
        }
        let t = mel.len();
        let bins = self.config.n_mels;
        let mut data = vec![0.0f32; bins * t];
        for (ti, col) in mel.iter().enumerate() {
            for (mi, &v) in col.iter().enumerate() {
                data[mi * t + ti] = v;
            }
        }
        let spec = Tensor::from_vec(data, vec![1usize, bins, t], Device::Cpu)
            .map_err(|e| AudioError::other(e.to_string()))?;
        Ok(Some(spec))
    }

    fn fallback_score(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        if energy > self.energy_threshold {
            (energy / (self.energy_threshold * 50.0)).min(1.0)
        } else {
            0.0
        }
    }

    pub fn score(&mut self, frame: &[f32]) -> f32 {
        if !self.is_loaded() {
            return self.fallback_score(frame);
        }
        let spec = match self.compute_mel(frame) {
            Ok(Some(s)) => s,
            _ => return 0.0,
        };
        let mut best: f32 = 0.0;
        let keys: Vec<String> = self.keywords.keys().cloned().collect();
        for k in keys {
            if let Some(slot) = self.keywords.get_mut(&k) {
                let (prob, h_new, c_new) = match slot.model.forward_last(&spec, &slot.h, &slot.c) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                slot.h = h_new;
                slot.c = c_new;
                let p_vec = match prob.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let p = p_vec[0];
                if p > best {
                    best = p;
                }
            }
        }
        best
    }

    pub fn score_per_keyword(&mut self, frame: &[f32]) -> Vec<(String, f32)> {
        if !self.is_loaded() {
            return Vec::new();
        }
        let spec = match self.compute_mel(frame) {
            Ok(Some(s)) => s,
            _ => return Vec::new(),
        };
        let mut out = Vec::with_capacity(self.keywords.len());
        let keys: Vec<String> = self.keywords.keys().cloned().collect();
        for k in keys {
            if let Some(slot) = self.keywords.get_mut(&k) {
                let (prob, h_new, c_new) = match slot.model.forward_last(&spec, &slot.h, &slot.c) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                slot.h = h_new;
                slot.c = c_new;
                let p = match prob.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    Ok(v) => v[0],
                    Err(_) => continue,
                };
                out.push((k.clone(), p));
            }
        }
        out
    }

    pub fn detect(&mut self, frame: &[f32]) -> bool {
        self.score(frame) >= self.config.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_kernels_cpu::ensure_registered;

    #[test]
    fn oww_fallback_silence_is_zero() {
        ensure_registered();
        let mut o = OpenWakeWord::new(OpenWakeWordConfig::default());
        assert!(!o.is_loaded());
        let n = 1024;
        assert_eq!(o.score(&vec![0.0; n]), 0.0);
        assert!(!o.detect(&vec![0.0; n]));
    }

    #[test]
    fn oww_fallback_loud_signal_scores_above_zero() {
        ensure_registered();
        let mut o = OpenWakeWord::new(OpenWakeWordConfig { threshold: 0.1, ..Default::default() });
        let loud: Vec<f32> = (0..2048).map(|i| 0.5 * (i as f32 * 0.05).sin()).collect();
        let s = o.score(&loud);
        assert!(s > 0.0, "fallback should produce non-zero score for loud signal, got {s}");
    }

    #[test]
    fn oww_register_keyword_and_score() {
        ensure_registered();
        let mut o = OpenWakeWord::new(OpenWakeWordConfig::default());
        let cfg = o.default_keyword_config();
        let model = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        o.add_keyword("alexa", model).unwrap();
        assert!(o.is_loaded());
        assert!(o.keyword_names().contains(&"alexa"));

        let frame: Vec<f32> = (0..2048).map(|i| 0.3 * (i as f32 * 0.07).sin()).collect();
        let s = o.score(&frame);
        assert!((0.0..=1.0).contains(&s), "score must be a probability, got {s}");
    }

    #[test]
    fn oww_reset_returns_state_to_zero() {
        ensure_registered();
        let mut o = OpenWakeWord::new(OpenWakeWordConfig::default());
        let cfg = o.default_keyword_config();
        let model = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        o.add_keyword("hey-syn", model).unwrap();

        let frame: Vec<f32> = (0..2048).map(|i| 0.5 * (i as f32 * 0.04).sin()).collect();
        let s1 = o.score(&frame);

        let frame2: Vec<f32> = (0..2048).map(|i| 0.5 * (i as f32 * 0.04).sin()).collect();
        let s2 = o.score(&frame2);
        assert!(s1.is_finite() && s2.is_finite());

        o.reset();
        let s3 = o.score(&frame);
        assert!(s3.is_finite() && (0.0..=1.0).contains(&s3));
    }

    #[test]
    fn oww_rejects_spec_mismatch() {
        ensure_registered();
        let mut o = OpenWakeWord::new(OpenWakeWordConfig::default());
        let bad = SileroVadConfig { spec_bins: 40, ..SileroVadConfig::default() };
        let model = SileroVadModel::new(bad, Device::Cpu, DType::F32).unwrap();
        assert!(o.add_keyword("bad", model).is_err());
    }

    #[test]
    fn oww_multiple_keywords_and_per_keyword_scores() {
        ensure_registered();
        let mut o = OpenWakeWord::new(OpenWakeWordConfig::default());
        for kw in ["k1", "k2"] {
            let cfg = o.default_keyword_config();
            let m = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
            o.add_keyword(kw, m).unwrap();
        }
        let frame: Vec<f32> = (0..2048).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        let scores = o.score_per_keyword(&frame);
        assert_eq!(scores.len(), 2);
        for (_, s) in &scores {
            assert!((0.0..=1.0).contains(s));
        }
    }
}
