//! Диаризация (NeMo Streaming Sortformer 4spk v2.1) — нативный
//! `synaptix-diarization-sortformer` (batch/full-attention путь).

use std::path::PathBuf;

use synaptix_core::dtype::DType;
use synaptix_diarization_sortformer::{DiarizeSegment, SortformerPipeline};

use super::asr::{ComputeDType, Device, StorageDType};

#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    pub model_path: PathBuf,
    pub device: Device,
    pub storage_dtype: StorageDType,
    pub compute_dtype: ComputeDType,
    pub threshold: f32,
    pub allow_overlap: bool,
}

fn compute_to_dtype(c: ComputeDType) -> DType {
    match c {
        ComputeDType::BF16 => DType::BF16,
        ComputeDType::F32 => DType::F32,
        ComputeDType::F16 | ComputeDType::Fp8E4M3 | ComputeDType::Nvfp4 => DType::F16,
    }
}

pub struct DiarizationResult {
    segments: Vec<DiarizeSegment>,
    duration_s: f32,
}

impl DiarizationResult {
    pub fn to_pretty(&self) -> String {
        if self.segments.is_empty() {
            return "(речь не обнаружена)".to_string();
        }
        let mut s = String::new();
        for seg in &self.segments {
            s.push_str(&format!(
                "spk{}  {:>7.2}–{:<7.2}s  conf={:.3}\n",
                seg.speaker, seg.start_s, seg.end_s, seg.confidence
            ));
        }
        s
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.segments).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn duration_s(&self) -> f32 {
        self.duration_s
    }
}

pub struct Diarizer {
    pipe: SortformerPipeline,
    threshold: f32,
    allow_overlap: bool,
}

impl Diarizer {
    pub fn load(cfg: DiarizationConfig) -> Result<Self, String> {
        let dtype = compute_to_dtype(cfg.compute_dtype);
        let pipe = SortformerPipeline::from_syn(&cfg.model_path, cfg.device, dtype)
            .map_err(|e| e.to_string())?;
        Ok(Self { pipe, threshold: cfg.threshold, allow_overlap: cfg.allow_overlap })
    }

    pub fn model_name(&self) -> &str {
        "sortformer-4spk-v2.1"
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    pub fn set_allow_overlap(&mut self, allow: bool) {
        self.allow_overlap = allow;
    }

    pub fn diarize_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<DiarizationResult, String> {
        let mut params = self.pipe.default_params();
        params.threshold = self.threshold;
        let _ = self.allow_overlap; // per-speaker threshold уже допускает overlap
        let segments =
            self.pipe.diarize_with(pcm, sample_rate, &params).map_err(|e| e.to_string())?;
        Ok(DiarizationResult {
            segments,
            duration_s: pcm.len() as f32 / sample_rate.max(1) as f32,
        })
    }
}
