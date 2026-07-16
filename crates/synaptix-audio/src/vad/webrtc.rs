use super::{Vad, VadDecision};

const SUB_BANDS_HZ_8K: [(f32, f32); 6] = [
    (80.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 3000.0),
    (3000.0, 4000.0),
];

const BAND_SPEECH_WEIGHTS: [f32; 6] = [0.40, 0.85, 1.00, 1.10, 1.00, 0.70];

const AGGRESSIVENESS_BIAS: [f32; 4] = [-0.6, -0.2, 0.2, 0.6];

pub struct WebRtcVad {
    sample_rate: u32,
    frame_samples: usize,
    aggressiveness: u8,
    band_filters: Vec<Biquad>,
    noise_floor: [f32; 6],
    initialized: bool,
}

impl WebRtcVad {
    pub fn new(sample_rate: u32, aggressiveness: u8) -> Self {
        Self::with_frame_ms(sample_rate, aggressiveness, 10)
    }

    pub fn with_frame_ms(sample_rate: u32, aggressiveness: u8, frame_ms: u32) -> Self {
        assert!(
            matches!(frame_ms, 10 | 20 | 30),
            "WebRtcVad: frame_ms must be 10/20/30"
        );
        assert!(
            matches!(sample_rate, 8000 | 16000 | 32000 | 48000),
            "WebRtcVad: sample_rate must be 8000/16000/32000/48000"
        );
        let frame_samples = (sample_rate as usize * frame_ms as usize) / 1000;
        let band_filters = SUB_BANDS_HZ_8K
            .iter()
            .map(|(lo, hi)| {
                let center = 0.5 * (lo + hi);
                let bw = hi - lo;
                Biquad::band_pass(center, bw, sample_rate as f32)
            })
            .collect();
        Self {
            sample_rate,
            frame_samples,
            aggressiveness: aggressiveness.min(3),
            band_filters,
            noise_floor: [-60.0; 6],
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        for f in &mut self.band_filters {
            f.reset();
        }
        self.noise_floor = [-60.0; 6];
        self.initialized = false;
    }

    pub fn score(&mut self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return -120.0;
        }
        let inv_n = 1.0 / frame.len() as f32;
        let mut band_db = [0.0f32; 6];
        for (b, filter) in self.band_filters.iter_mut().enumerate() {
            let mut energy = 0.0f32;
            for &x in frame {
                let y = filter.process(x);
                energy += y * y;
            }
            energy *= inv_n;
            band_db[b] = 10.0 * (energy + 1.0e-12).log10();
        }

        if !self.initialized {
            self.noise_floor = band_db;
            self.initialized = true;
        }

        let mut weighted_snr = 0.0f32;
        let mut weight_sum = 0.0f32;
        for b in 0..6 {
            let snr_db = (band_db[b] - self.noise_floor[b]).max(0.0);
            let w = BAND_SPEECH_WEIGHTS[b];
            weighted_snr += w * snr_db;
            weight_sum += w;
        }
        let avg_snr_db = weighted_snr / weight_sum;

        for b in 0..6 {
            let delta = band_db[b] - self.noise_floor[b];
            let alpha = if delta < 6.0 { 0.05 } else { 0.001 };
            self.noise_floor[b] += alpha * delta;
        }

        avg_snr_db
    }

    fn threshold_db(&self) -> f32 {
        6.0 + AGGRESSIVENESS_BIAS[self.aggressiveness as usize] * 6.0
    }

    pub fn aggressiveness(&self) -> u8 {
        self.aggressiveness
    }
}

impl Vad for WebRtcVad {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }
    fn classify(&mut self, frame: &[f32]) -> VadDecision {
        let snr_db = self.score(frame);
        if snr_db >= self.threshold_db() {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        }
    }
}

#[derive(Debug, Clone)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn band_pass(center_hz: f32, bandwidth_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sample_rate;
        let q = center_hz / bandwidth_hz.max(1.0);
        let alpha = w0.sin() / (2.0 * q.max(0.05));
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        let b1 = 0.0;
        let b2 = -alpha / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;
        Self { b0, b1, b2, a1, a2, z1: 0.0, z2: 0.0 }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn sine_frame(freq_hz: f32, sr: u32, n: usize, amp: f32) -> Vec<f32> {
        let dt = 1.0 / sr as f32;
        (0..n).map(|i| amp * (2.0 * std::f32::consts::PI * freq_hz * i as f32 * dt).sin()).collect()
    }

    fn vowel_like_frame(sr: u32, n: usize, amp: f32) -> Vec<f32> {
        let f0 = 130.0;
        let formants = [500.0, 1500.0, 2500.0];
        let dt = 1.0 / sr as f32;
        (0..n)
            .map(|i| {
                let t = i as f32 * dt;
                let mut s = 0.6 * (2.0 * std::f32::consts::PI * f0 * t).sin();
                for f in formants.iter() {
                    s += 0.5 * (2.0 * std::f32::consts::PI * f * t).sin();
                }
                amp * s / 2.4
            })
            .collect()
    }

    #[test]
    fn webrtc_vad_silence_below_threshold() {
        let mut v = WebRtcVad::new(16000, 1);
        let n = v.frame_samples();
        let silence = vec![0.0f32; n];
        for _ in 0..5 {
            assert_eq!(v.classify(&silence), VadDecision::Silence);
        }
    }

    #[test]
    fn webrtc_vad_white_noise_below_speech() {
        let mut v = WebRtcVad::new(16000, 1);
        let n = v.frame_samples();
        let mut rng_state = 0x12345u32;
        let noise: Vec<f32> = (0..n * 10)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng_state >> 16) as f32 / 32768.0 - 1.0) * 0.02
            })
            .collect();
        for chunk in noise.chunks(n) {
            v.classify(chunk);
        }
        let last: Vec<f32> = noise[noise.len() - n..].to_vec();
        let snr = v.score(&last);
        assert!(snr < 6.0, "constant noise should adapt and not register as speech (snr={snr})");
    }

    #[test]
    fn webrtc_vad_vowel_above_silence_floor() {
        let mut v = WebRtcVad::new(16000, 1);
        let n = v.frame_samples();
        let silence = vec![0.0f32; n];
        for _ in 0..3 {
            v.classify(&silence);
        }
        let voiced = vowel_like_frame(16000, n, 0.4);
        let mut spoke = false;
        for _ in 0..3 {
            if v.classify(&voiced) == VadDecision::Speech {
                spoke = true;
            }
        }
        assert!(spoke, "vowel-like signal should be classified as speech");
    }

    #[test]
    fn webrtc_vad_aggressiveness_monotonic() {
        let n_samples = 160;
        let voiced = vowel_like_frame(16000, n_samples, 0.18);
        let mut counts = [0usize; 4];
        for ag in 0..4u8 {
            let mut v = WebRtcVad::new(16000, ag);
            for _ in 0..2 {
                v.classify(&vec![0.0; n_samples]);
            }
            for _ in 0..10 {
                if v.classify(&voiced) == VadDecision::Speech {
                    counts[ag as usize] += 1;
                }
            }
        }
        assert!(
            counts[0] >= counts[1] && counts[1] >= counts[2] && counts[2] >= counts[3],
            "speech detections must be non-increasing with aggressiveness: {:?}",
            counts
        );
    }

    #[test]
    fn webrtc_vad_dc_offset_not_speech() {
        let mut v = WebRtcVad::new(16000, 1);
        let n = v.frame_samples();
        let dc = vec![0.5f32; n];
        for _ in 0..10 {
            v.classify(&dc);
        }
        let mut speech_hits = 0;
        for _ in 0..10 {
            if v.classify(&dc) == VadDecision::Speech {
                speech_hits += 1;
            }
        }
        assert_eq!(
            speech_hits, 0,
            "stationary DC offset has no band-pass energy after settle; should not be speech"
        );
    }

    #[test]
    fn webrtc_vad_supported_rates_and_frames() {
        for sr in [8000u32, 16000, 32000, 48000] {
            for ms in [10u32, 20, 30] {
                let v = WebRtcVad::with_frame_ms(sr, 2, ms);
                let expected = (sr as usize * ms as usize) / 1000;
                assert_eq!(v.frame_samples(), expected);
            }
        }
    }

    #[test]
    #[should_panic(expected = "frame_ms")]
    fn webrtc_vad_rejects_invalid_frame_ms() {
        let _ = WebRtcVad::with_frame_ms(16000, 1, 15);
    }

    #[test]
    #[should_panic(expected = "sample_rate")]
    fn webrtc_vad_rejects_unsupported_sr() {
        let _ = WebRtcVad::with_frame_ms(22050, 1, 10);
    }
}
