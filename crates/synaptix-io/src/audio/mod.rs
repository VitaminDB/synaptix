#[cfg(feature = "audio")]
pub mod flac;
#[cfg(feature = "audio")]
pub mod mp3;
#[cfg(feature = "audio")]
pub mod ogg;
#[cfg(feature = "audio")]
pub mod resample;
#[cfg(feature = "audio")]
pub mod wav;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self { samples, sample_rate, channels }
    }

    pub fn num_frames(&self) -> usize {
        if self.channels == 0 { 0 } else { self.samples.len() / self.channels as usize }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 { 0.0 } else { self.num_frames() as f64 / self.sample_rate as f64 }
    }

    pub fn channel(&self, idx: usize) -> Vec<f32> {
        let ch = self.channels as usize;
        self.samples.iter().skip(idx).step_by(ch).copied().collect()
    }

    pub fn to_mono(&self) -> Vec<f32> {
        let ch = self.channels as usize;
        if ch == 0 { return vec![]; }
        if ch == 1 { return self.samples.clone(); }
        let frames = self.num_frames();
        (0..frames)
            .map(|i| {
                let sum: f32 = (0..ch).map(|c| self.samples[i * ch + c]).sum();
                sum / ch as f32
            })
            .collect()
    }
}

pub trait AudioDecoder {
    fn decode(path: &std::path::Path) -> crate::error::Result<AudioBuffer>;
}
