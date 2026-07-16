use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct LogMelConfig {
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub n_mels: usize,
    pub sample_rate: u32,
    pub fmin: f32,
    pub fmax: f32,
    pub log_offset: f32,
}

impl LogMelConfig {
    pub fn whisper_default() -> Self {
        Self {
            n_fft: 400,
            hop: 160,
            win: 400,
            n_mels: 80,
            sample_rate: 16000,
            fmin: 0.0,
            fmax: 8000.0,
            log_offset: 1e-10,
        }
    }
}

pub fn log_mel_spectrogram(waveform: &Tensor, cfg: LogMelConfig) -> Result<Tensor> {
    if waveform.rank() != 1 && waveform.rank() != 2 {
        return Err(SynaptixError::Unsupported("log_mel: waveform must be 1D or 2D (B, T)"));
    }
    if cfg.n_fft == 0 || cfg.hop == 0 || cfg.win == 0 || cfg.win > cfg.n_fft {
        return Err(SynaptixError::Unsupported("log_mel: invalid config"));
    }
    let (batch, samples, is_batched) = if waveform.rank() == 1 {
        let n = waveform.dims()[0];
        (1usize, n, false)
    } else {
        (waveform.dims()[0], waveform.dims()[1], true)
    };
    let device: Device = waveform.device();
    let window = hann_window(cfg.win);
    let mel_filters = mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax);
    let pad = (cfg.win - cfg.n_fft).min(0);
    let _ = pad;
    let n_frames = if samples >= cfg.win {
        (samples - cfg.win) / cfg.hop + 1
    } else {
        0
    };
    if n_frames == 0 {
        return Err(SynaptixError::Other("log_mel: waveform too short".to_string()));
    }
    let n_bins = cfg.n_fft / 2 + 1;
    let mut output = vec![0.0_f32; batch * cfg.n_mels * n_frames];
    let pcm_flat: Vec<f32> = if waveform.rank() == 1 {
        waveform
            .to_dtype(synaptix_core::dtype::DType::F32)?
            .to_vec1::<f32>()?
    } else {
        waveform
            .to_dtype(synaptix_core::dtype::DType::F32)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect()
    };
    for b in 0..batch {
        let pcm = &pcm_flat[b * samples..(b + 1) * samples];
        let mut frame_buf = vec![0.0_f32; cfg.n_fft];
        let mut power = vec![0.0_f32; n_bins];
        for f in 0..n_frames {
            let off = f * cfg.hop;
            for i in 0..cfg.win {
                frame_buf[i] = pcm[off + i] * window[i];
            }
            for i in cfg.win..cfg.n_fft {
                frame_buf[i] = 0.0;
            }
            dft_power(&frame_buf, cfg.n_fft, &mut power);
            for m in 0..cfg.n_mels {
                let mut acc = 0.0_f32;
                for k in 0..n_bins {
                    acc += mel_filters[m * n_bins + k] * power[k];
                }
                let log_val = (acc + cfg.log_offset).ln();
                output[(b * cfg.n_mels + m) * n_frames + f] = log_val;
            }
        }
    }
    if is_batched {
        Tensor::from_vec(output, (batch, cfg.n_mels, n_frames), device)
    } else {
        Tensor::from_vec(output, (cfg.n_mels, n_frames), device)
    }
}

fn hann_window(win: usize) -> Vec<f32> {
    let mut w = vec![0.0_f32; win];
    for n in 0..win {
        let x = (n as f32) / ((win - 1) as f32);
        w[n] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos());
    }
    w
}

fn dft_power(frame: &[f32], n_fft: usize, power: &mut [f32]) {
    let n_bins = n_fft / 2 + 1;
    for k in 0..n_bins {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for n in 0..n_fft {
            let angle = -2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (n_fft as f32);
            re += frame[n] * angle.cos();
            im += frame[n] * angle.sin();
        }
        power[k] = re * re + im * im;
    }
}

fn hz_to_mel(hz: f32) -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() }

fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) }

fn mel_filterbank(n_fft: usize, n_mels: usize, sr: u32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mut mel_points = vec![0.0_f32; n_mels + 2];
    for i in 0..(n_mels + 2) {
        mel_points[i] = mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32);
    }
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_resolution = (sr as f32) / (n_fft as f32);
    let bin_points: Vec<f32> = hz_points.iter().map(|&hz| hz / bin_resolution).collect();
    let mut filters = vec![0.0_f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..n_bins {
            let kf = k as f32;
            let val = if kf < left || kf > right {
                0.0
            } else if kf <= center {
                (kf - left) / (center - left).max(1e-6)
            } else {
                (right - kf) / (right - center).max(1e-6)
            };
            filters[m * n_bins + k] = val;
        }
    }
    filters
}
