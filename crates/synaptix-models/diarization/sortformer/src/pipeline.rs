//! High-level API: `SortformerPipeline::from_syn` + `diarize(samples, sample_rate)`.
//!
//! Открывает `.syn`-бандл, грузит `SortformerModel`, прогоняет PCM (ресэмпл→16кГц)
//! через batch-forward и постпроцессит per-frame probs в сегменты.

use std::path::Path;

use synaptix_audio::resample_linear;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

use crate::loader::SortformerWeights;
use crate::model::SortformerModel;
use crate::postprocess::{frames_to_segments, DiarizationResult, DiarizeSegment, PostprocessParams, RESULT_VERSION};
use crate::{Result, SortformerError};

pub struct SortformerPipeline {
    model: SortformerModel,
    target_sr: u32,
}

impl SortformerPipeline {
    pub fn from_syn(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self> {
        let w = SortformerWeights::open(path, device, dtype)?;
        let target_sr = w.config.sample_rate as u32;
        let model = SortformerModel::load(&w)?;
        Ok(Self { model, target_sr })
    }

    pub fn device(&self) -> Device {
        self.model.device()
    }

    /// Постпроцесс-параметры из конфига модели (frame_rate/max_speakers).
    pub fn default_params(&self) -> PostprocessParams {
        let cfg = self.model.config();
        PostprocessParams {
            frame_rate_hz: cfg.frame_rate_hz,
            max_speakers: cfg.max_speakers,
            ..PostprocessParams::default()
        }
    }

    fn to_target(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate == self.target_sr {
            Ok(samples.to_vec())
        } else {
            resample_linear(samples, sample_rate, self.target_sr)
                .map_err(|e| SortformerError::Audio(e.to_string()))
        }
    }

    /// PCM (любой sr) → сегменты диаризации (дефолтные постпроцесс-параметры).
    pub fn diarize(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<DiarizeSegment>> {
        self.diarize_with(samples, sample_rate, &self.default_params())
    }

    pub fn diarize_with(
        &self,
        samples: &[f32],
        sample_rate: u32,
        params: &PostprocessParams,
    ) -> Result<Vec<DiarizeSegment>> {
        let pcm = self.to_target(samples, sample_rate)?;
        // streaming-путь = поведение NeMo v2.1 (для ≤1 чанка ≡ batch full-attention).
        let preds = self.model.diarize_pcm_streaming(&pcm)?; // (1,T,n_spk)
        let dims = preds.dims().to_vec();
        let (n_frames, n_spk) = (dims[1], dims[2]);
        let flat = preds.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(frames_to_segments(&flat, n_frames, n_spk, params))
    }

    /// Полный результат с метаданными.
    pub fn diarize_result(&self, samples: &[f32], sample_rate: u32) -> Result<DiarizationResult> {
        let segments = self.diarize(samples, sample_rate)?;
        let num_speakers =
            segments.iter().map(|s| s.speaker as usize + 1).max().unwrap_or(0);
        Ok(DiarizationResult {
            version: RESULT_VERSION,
            sample_rate,
            duration_s: samples.len() as f32 / sample_rate as f32,
            num_speakers,
            segments,
        })
    }
}
