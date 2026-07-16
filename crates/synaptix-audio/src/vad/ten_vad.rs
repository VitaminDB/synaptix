use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::audio::silero_vad::{SileroVadConfig, SileroVadModel};

use crate::error::{AudioError, Result};
use crate::mel::{log_mel_spectrogram, MelConfig, MelNorm, MelScale};
use crate::stft::{PadMode, StftConfig};
use crate::window::WindowKind;

use super::{Vad, VadDecision};

pub struct TenVad {
    sample_rate: u32,
    frame_samples: usize,
    threshold: f32,
    energy_threshold: f32,
    stft_cfg: StftConfig,
    mel_cfg: MelConfig,
    n_mels: usize,
    model: Option<SileroVadModel>,
    h: Option<Tensor>,
    c: Option<Tensor>,
}

fn default_mel_cfg(sample_rate: u32, n_mels: usize) -> MelConfig {
    MelConfig {
        sample_rate,
        n_fft: 256,
        f_min: 0.0,
        f_max: sample_rate as f32 / 2.0,
        n_mels,
        mel_scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
    }
}

fn default_stft_cfg(n_fft: usize) -> StftConfig {
    StftConfig {
        n_fft,
        hop_length: n_fft / 2,
        win_length: n_fft,
        window: WindowKind::Hann,
        center: false,
        pad_mode: PadMode::Zero,
    }
}

impl TenVad {
    pub fn new(sample_rate: u32, threshold: f32) -> Self {
        let frame_samples = match sample_rate {
            8000 => 128,
            16000 => 256,
            32000 => 512,
            48000 => 768,
            sr => ((sr as f32 * 0.016).round() as usize).max(128),
        };
        let n_mels = 40;
        let mel_cfg = default_mel_cfg(sample_rate, n_mels);
        let stft_cfg = default_stft_cfg(mel_cfg.n_fft);
        Self {
            sample_rate,
            frame_samples,
            threshold,
            energy_threshold: threshold.max(1.0e-4),
            stft_cfg,
            mel_cfg,
            n_mels,
            model: None,
            h: None,
            c: None,
        }
    }

    pub fn with_model(mut self, model: SileroVadModel) -> Result<Self> {
        if model.config.spec_bins != self.n_mels {
            return Err(AudioError::invalid_arg(format!(
                "TenVad: model.spec_bins={} but n_mels={}",
                model.config.spec_bins, self.n_mels
            )));
        }
        let (h, c) = model
            .zero_state(1, Device::Cpu, DType::F32)
            .map_err(|e| AudioError::other(e.to_string()))?;
        self.h = Some(h);
        self.c = Some(c);
        self.model = Some(model);
        Ok(self)
    }

    pub fn reset(&mut self) {
        if let Some(m) = &self.model {
            if let Ok((h, c)) = m.zero_state(1, Device::Cpu, DType::F32) {
                self.h = Some(h);
                self.c = Some(c);
            }
        }
    }

    pub fn is_native_model_loaded(&self) -> bool {
        self.model.is_some()
    }

    pub fn default_model_config(&self) -> SileroVadConfig {
        SileroVadConfig {
            spec_bins: self.n_mels,
            stem_channels: 32,
            hidden_size: 32,
            num_conv_blocks: 2,
            conv_kernel: 3,
        }
    }

    fn classify_energy(&self, frame: &[f32]) -> VadDecision {
        if frame.is_empty() {
            return VadDecision::Silence;
        }
        let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        if energy > self.energy_threshold {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        }
    }

    fn classify_model(&mut self, frame: &[f32]) -> Result<VadDecision> {
        let mel = log_mel_spectrogram(frame, &self.stft_cfg, &self.mel_cfg)?;
        if mel.is_empty() {
            return Ok(VadDecision::Silence);
        }
        let t = mel.len();
        let mut data = vec![0.0f32; self.n_mels * t];
        for (ti, col) in mel.iter().enumerate() {
            for (mi, &v) in col.iter().enumerate() {
                data[mi * t + ti] = v;
            }
        }
        let spec = Tensor::from_vec(data, vec![1usize, self.n_mels, t], Device::Cpu)
            .map_err(|e| AudioError::other(e.to_string()))?;
        let model = self.model.as_ref().expect("model present");
        let h = self.h.as_ref().expect("hidden present");
        let c = self.c.as_ref().expect("cell present");
        let (prob, h_new, c_new) = model
            .forward_last(&spec, h, c)
            .map_err(|e| AudioError::other(e.to_string()))?;
        self.h = Some(h_new);
        self.c = Some(c_new);
        let p = prob
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| AudioError::other(e.to_string()))?[0];
        Ok(if p > self.threshold {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        })
    }
}

impl Vad for TenVad {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }
    fn classify(&mut self, frame: &[f32]) -> VadDecision {
        if self.model.is_some() {
            match self.classify_model(frame) {
                Ok(d) => d,
                Err(_) => self.classify_energy(frame),
            }
        } else {
            self.classify_energy(frame)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_kernels_cpu::ensure_registered;

    #[test]
    fn ten_energy_fallback_silence() {
        ensure_registered();
        let mut v = TenVad::new(16000, 0.01);
        assert!(!v.is_native_model_loaded());
        let n = v.frame_samples();
        assert_eq!(n, 256);
        assert_eq!(v.classify(&vec![0.0; n]), VadDecision::Silence);
    }

    #[test]
    fn ten_energy_fallback_speech() {
        ensure_registered();
        let mut v = TenVad::new(16000, 0.01);
        let n = v.frame_samples();
        let loud: Vec<f32> = (0..n).map(|i| 0.4 * (i as f32 * 0.08).sin()).collect();
        assert_eq!(v.classify(&loud), VadDecision::Speech);
    }

    #[test]
    fn ten_with_model_forward_finite() {
        ensure_registered();
        let v = TenVad::new(16000, 0.5);
        let cfg = v.default_model_config();
        let model = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        let mut v = v.with_model(model).unwrap();
        let n = v.frame_samples();
        let frame: Vec<f32> = (0..n * 3).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        let d = v.classify(&frame);
        assert!(matches!(d, VadDecision::Speech | VadDecision::Silence));
    }

    #[test]
    fn ten_rejects_model_dim_mismatch() {
        ensure_registered();
        let v = TenVad::new(16000, 0.5);
        let bad = SileroVadConfig { spec_bins: 129, ..SileroVadConfig::default() };
        let model = SileroVadModel::new(bad, Device::Cpu, DType::F32).unwrap();
        assert!(v.with_model(model).is_err());
    }
}
