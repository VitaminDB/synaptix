pub mod silero;
pub mod ten_vad;
pub mod webrtc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    Speech,
    Silence,
}

pub trait Vad: Send + Sync {
    fn sample_rate(&self) -> u32;
    fn frame_samples(&self) -> usize;
    fn classify(&mut self, frame: &[f32]) -> VadDecision;
}

pub struct EnergyVad {
    sample_rate: u32,
    frame_samples: usize,
    threshold: f32,
}

impl EnergyVad {
    pub fn new(sample_rate: u32, frame_ms: u32, threshold_db: f32) -> Self {
        let frame_samples = ((sample_rate as f32 * frame_ms as f32) / 1000.0).round() as usize;
        let threshold = 10.0f32.powf(threshold_db / 10.0);
        Self { sample_rate, frame_samples, threshold }
    }
}

impl Vad for EnergyVad {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }
    fn classify(&mut self, frame: &[f32]) -> VadDecision {
        if frame.is_empty() {
            return VadDecision::Silence;
        }
        let power: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        if power > self.threshold {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_vad_classifies_silence() {
        let mut v = EnergyVad::new(16000, 20, -40.0);
        let silence = vec![0.0f32; 320];
        assert_eq!(v.classify(&silence), VadDecision::Silence);
    }

    #[test]
    fn energy_vad_classifies_speech() {
        let mut v = EnergyVad::new(16000, 20, -40.0);
        let loud: Vec<f32> = (0..320).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();
        assert_eq!(v.classify(&loud), VadDecision::Speech);
    }
}
