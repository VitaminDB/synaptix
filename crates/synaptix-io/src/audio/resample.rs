use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use crate::error::{IoError, Result};
use super::AudioBuffer;

pub fn resample(buf: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    if buf.sample_rate == target_rate {
        return Ok(buf.clone());
    }
    let ratio = target_rate as f64 / buf.sample_rate as f64;
    let channels = buf.channels as usize;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024usize;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        chunk_size,
        channels,
    ).map_err(|e| IoError::Audio(e.to_string()))?;

    let frames = buf.num_frames();
    let mut deinterleaved: Vec<Vec<f32>> = (0..channels)
        .map(|c| buf.channel(c))
        .collect();

    let mut out_channels: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut pos = 0usize;

    while pos < frames {
        let end = (pos + chunk_size).min(frames);
        let actual = end - pos;
        let mut input: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                let mut ch = deinterleaved[c][pos..end].to_vec();
                ch.resize(chunk_size, 0.0);
                ch
            })
            .collect();

        let output = resampler.process(&input, None)
            .map_err(|e| IoError::Audio(e.to_string()))?;

        for (c, ch_out) in output.iter().enumerate() {
            out_channels[c].extend_from_slice(ch_out);
        }
        pos = end;
    }

    let total_frames = out_channels[0].len();
    let mut interleaved = vec![0.0f32; total_frames * channels];
    for (c, ch) in out_channels.iter().enumerate() {
        for (i, &s) in ch.iter().enumerate() {
            interleaved[i * channels + c] = s;
        }
    }

    Ok(AudioBuffer::new(interleaved, target_rate, buf.channels))
}
