use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::audio::silero_vad::{SileroVadConfig, SileroVadModel};

use crate::error::{AudioError, Result};
use crate::stft::{stft, PadMode, StftConfig};
use crate::window::WindowKind;

use super::{Vad, VadDecision};

pub struct SileroVad {
    sample_rate: u32,
    frame_samples: usize,
    threshold: f32,
    energy_threshold: f32,
    stft_cfg: StftConfig,
    model: Option<SileroVadModel>,
    h: Option<Tensor>,
    c: Option<Tensor>,
}

impl SileroVad {
    pub fn new(sample_rate: u32, threshold: f32) -> Self {
        let frame_samples = match sample_rate {
            8000 => 256,
            16000 => 512,
            32000 => 1024,
            48000 => 1536,
            sr => ((sr as f32 * 0.032).round() as usize).max(256),
        };
        let stft_cfg = StftConfig {
            n_fft: 256,
            hop_length: 128,
            win_length: 256,
            window: WindowKind::Hann,
            center: false,
            pad_mode: PadMode::Zero,
        };
        Self {
            sample_rate,
            frame_samples,
            threshold,
            energy_threshold: threshold.max(1.0e-4),
            stft_cfg,
            model: None,
            h: None,
            c: None,
        }
    }

    pub fn with_model(mut self, model: SileroVadModel) -> Result<Self> {
        if model.config.spec_bins != self.stft_cfg.num_freqs() {
            return Err(AudioError::invalid_arg(format!(
                "SileroVad: model.spec_bins={} but STFT yields {} bins (n_fft={})",
                model.config.spec_bins,
                self.stft_cfg.num_freqs(),
                self.stft_cfg.n_fft
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

    pub fn with_model_and_stft(mut self, model: SileroVadModel, stft_cfg: StftConfig) -> Result<Self> {
        if model.config.spec_bins != stft_cfg.num_freqs() {
            return Err(AudioError::invalid_arg(format!(
                "SileroVad: model.spec_bins={} but provided STFT yields {} bins",
                model.config.spec_bins,
                stft_cfg.num_freqs()
            )));
        }
        let (h, c) = model
            .zero_state(1, Device::Cpu, DType::F32)
            .map_err(|e| AudioError::other(e.to_string()))?;
        self.stft_cfg = stft_cfg;
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
            spec_bins: self.stft_cfg.num_freqs(),
            ..SileroVadConfig::default()
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
        let frames = stft(frame, &self.stft_cfg)?;
        if frames.is_empty() {
            return Ok(VadDecision::Silence);
        }
        let bins = self.stft_cfg.num_freqs();
        let t = frames.len();
        let mut mag = vec![0.0f32; bins * t];
        for (ti, f) in frames.iter().enumerate() {
            for (bi, c) in f.iter().enumerate() {
                mag[bi * t + ti] = (c.re * c.re + c.im * c.im).sqrt();
            }
        }
        let spec = Tensor::from_vec(mag, vec![1usize, bins, t], Device::Cpu)
            .map_err(|e| AudioError::other(e.to_string()))?;
        let model = self.model.as_ref().expect("model present");
        let h = self.h.as_ref().expect("hidden state present");
        let c = self.c.as_ref().expect("cell state present");
        let (prob, h_new, c_new) = model
            .forward_last(&spec, h, c)
            .map_err(|e| AudioError::other(e.to_string()))?;
        self.h = Some(h_new);
        self.c = Some(c_new);
        let p_vec = prob
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| AudioError::other(e.to_string()))?;
        let p = p_vec[0];
        Ok(if p > self.threshold {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        })
    }
}

impl Vad for SileroVad {
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
    fn silero_energy_fallback_silence() {
        ensure_registered();
        let mut v = SileroVad::new(16000, 0.01);
        assert!(!v.is_native_model_loaded());
        let n = v.frame_samples();
        assert_eq!(n, 512);
        assert_eq!(v.classify(&vec![0.0; n]), VadDecision::Silence);
    }

    #[test]
    fn silero_energy_fallback_speech() {
        ensure_registered();
        let mut v = SileroVad::new(16000, 0.01);
        let n = v.frame_samples();
        let loud: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32 * 0.05).sin()).collect();
        assert_eq!(v.classify(&loud), VadDecision::Speech);
    }

    #[test]
    fn silero_with_model_threshold() {
        ensure_registered();
        let v = SileroVad::new(16000, 0.5);
        let cfg = v.default_model_config();
        let model = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        let mut v = v.with_model(model).unwrap();
        assert!(v.is_native_model_loaded());
        let n = v.frame_samples();
        let frame: Vec<f32> = (0..n).map(|i| 0.3 * (i as f32 * 0.1).sin()).collect();
        let d = v.classify(&frame);
        assert!(matches!(d, VadDecision::Speech | VadDecision::Silence));
    }

    #[test]
    fn silero_with_model_state_persistence() {
        ensure_registered();
        let v = SileroVad::new(16000, 0.5);
        let cfg = v.default_model_config();
        let model = SileroVadModel::new(cfg, Device::Cpu, DType::F32).unwrap();
        let mut v = v.with_model(model).unwrap();

        let n = v.frame_samples();
        let s: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();

        let h0 = v.h.as_ref().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let _ = v.classify(&s);
        let h1 = v.h.as_ref().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let drift: f32 = h0.iter().zip(&h1).map(|(a, b)| (a - b).abs()).sum();
        assert!(drift > 1e-6, "lstm state must update after first classify");

        v.reset();
        let h2 = v.h.as_ref().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let zero_drift: f32 = h0.iter().zip(&h2).map(|(a, b)| (a - b).abs()).sum();
        assert!(zero_drift < 1e-9, "reset must return state to zero");
    }

    #[test]
    fn silero_rejects_model_spec_mismatch() {
        ensure_registered();
        let v = SileroVad::new(16000, 0.5);
        let bad_cfg = SileroVadConfig { spec_bins: 64, ..SileroVadConfig::default() };
        let model = SileroVadModel::new(bad_cfg, Device::Cpu, DType::F32).unwrap();
        let err = v.with_model(model);
        assert!(err.is_err(), "spec_bins mismatch must be rejected");
    }
}
