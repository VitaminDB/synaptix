//! ASR-фасад. Whisper и GigaAM — нативные synaptix (synaptix-asr-whisper /
//! synaptix-asr-gigaam). `Transcriber` диспетчит по [`AsrModelKind`]; submodule
//! [`stream`] даёт streaming-ASR поверх готового `Transcriber`.

use std::io::Cursor;
use std::path::PathBuf;

use synaptix_asr_gigaam::GigaAm;
use synaptix_asr_whisper::{Task, WhisperPipeline};
use synaptix_audio::resample_linear;
use synaptix_core::dtype::DType;

pub use synaptix_core::device::Device;
pub use synaptix_core::dtype::DType as StorageDType;

const WHISPER_SR: u32 = 16000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDType {
    F16,
    BF16,
    F32,
    Fp8E4M3,
    Nvfp4,
}

impl ComputeDType {
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" | "fp32" | "float" => Some(Self::F32),
            "fp8e4m3" | "fp8" | "mxfp8" => Some(Self::Fp8E4M3),
            "nvfp4" => Some(Self::Nvfp4),
            _ => None,
        }
    }

    pub fn to_dtype(self) -> DType {
        match self {
            ComputeDType::BF16 => DType::BF16,
            ComputeDType::F32 => DType::F32,
            ComputeDType::F16 | ComputeDType::Fp8E4M3 | ComputeDType::Nvfp4 => DType::F16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrModelKind {
    Whisper,
    GigaAm,
}

#[derive(Debug, Clone)]
pub struct AsrConfig {
    pub kind: AsrModelKind,
    pub model_path: PathBuf,
    pub language: Option<String>,
    pub device: Device,
    pub storage_dtype: StorageDType,
    pub compute_dtype: ComputeDType,
}

enum Engine {
    Whisper(WhisperPipeline),
    GigaAm(GigaAm),
}

pub struct Transcriber {
    kind: AsrModelKind,
    engine: Engine,
    language: Option<String>,
}

impl Transcriber {
    pub fn load(cfg: AsrConfig) -> Result<Self, String> {
        match cfg.kind {
            AsrModelKind::Whisper => {
                let whisper = WhisperPipeline::from_syn(
                    &cfg.model_path,
                    cfg.device,
                    cfg.compute_dtype.to_dtype(),
                )
                .map_err(|e| e.to_string())?;
                Ok(Self {
                    kind: cfg.kind,
                    engine: Engine::Whisper(whisper),
                    language: cfg.language,
                })
            }
            AsrModelKind::GigaAm => {
                // Каталог = распакованный HF-снапшот; файл = .syn-бандл.
                let gigaam = if cfg.model_path.is_dir() {
                    GigaAm::from_unpacked(&cfg.model_path, &cfg.device, cfg.compute_dtype.to_dtype())
                } else {
                    GigaAm::from_syn(&cfg.model_path, &cfg.device, cfg.compute_dtype.to_dtype())
                }
                .map_err(|e| e.to_string())?;
                Ok(Self {
                    kind: cfg.kind,
                    engine: Engine::GigaAm(gigaam),
                    language: cfg.language,
                })
            }
        }
    }

    pub fn model_name(&self) -> &str {
        match self.kind {
            AsrModelKind::Whisper => "whisper",
            AsrModelKind::GigaAm => "gigaam",
        }
    }

    pub fn transcribe_pcm_text(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String, String> {
        match &self.engine {
            Engine::Whisper(whisper) => {
                let resampled;
                let audio: &[f32] = if sample_rate == WHISPER_SR {
                    pcm
                } else {
                    resampled =
                        resample_linear(pcm, sample_rate, WHISPER_SR).map_err(|e| e.to_string())?;
                    &resampled
                };
                let lang = language.or(self.language.as_deref());
                whisper
                    .transcribe(audio, lang, Task::Transcribe)
                    .map_err(|e| e.to_string())
            }
            // GigaAM — русский CTC; язык фиксирован, ресэмпл внутри transcribe_pcm.
            Engine::GigaAm(gigaam) => {
                gigaam.transcribe_pcm(pcm, sample_rate).map_err(|e| e.to_string())
            }
        }
    }

    pub fn transcribe_wav(&mut self, wav: &[u8]) -> Result<String, String> {
        let (pcm, sr) = decode_wav_mono(wav)?;
        self.transcribe_pcm_text(&pcm, sr, None)
    }
}

fn decode_wav_mono(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let ch = (spec.channels as usize).max(1);
    let sr = spec.sample_rate;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    let mono = if ch <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, sr))
}

/// Streaming-ASR поверх готового [`Transcriber`]. Worker буферизует PCM-чанки,
/// раз в ~CADENCE_S секунд ре-транскрибирует весь буфер и шлёт `Delta` с новым
/// хвостом (word-prefix-diff). При закрытии pcm-канала — финальная транскрипция
/// → `Final`.
pub mod stream {
    use std::sync::mpsc::{channel, Receiver};
    use std::sync::{Arc, Mutex};

    use super::Transcriber;

    #[derive(Debug, Clone, Default)]
    pub struct StreamingAsrConfig {
        pub language: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub enum StreamingAsrEvent {
        Delta { text: String, partial: bool },
        Final { text: String },
        Error(String),
    }

    pub struct StreamingAsr {
        pub events_rx: Receiver<StreamingAsrEvent>,
    }

    /// Накопленного аудио (сек) между ре-транскрипциями live-превью.
    const CADENCE_S: f32 = 1.0;
    /// Минимум аудио (сек) для первой попытки транскрипции (короче — шум/пусто).
    const MIN_S: f32 = 0.4;

    /// Слова `new` за пределами общего префикса со `prev` (инкремент для аппенда).
    fn suffix_after_common_prefix(prev: &[&str], new: &[&str]) -> String {
        let mut i = 0;
        while i < prev.len() && i < new.len() && prev[i] == new[i] {
            i += 1;
        }
        new[i..].join(" ")
    }

    impl StreamingAsr {
        pub fn start(
            pcm_rx: Receiver<Vec<f32>>,
            sample_rate: u32,
            asr: Arc<Mutex<Option<Transcriber>>>,
            cfg: StreamingAsrConfig,
        ) -> Self {
            let (tx, rx) = channel();

            std::thread::Builder::new()
                .name("syn-streaming-asr".into())
                .spawn(move || {
                    let sr = sample_rate;
                    let cadence = (CADENCE_S * sr as f32) as usize;
                    let min_samples = (MIN_S * sr as f32) as usize;
                    let lang = cfg.language.as_deref();

                    let mut buf: Vec<f32> = Vec::new();
                    let mut last_len = 0usize;
                    let mut prev_text = String::new();

                    let run = |buf: &[f32]| -> Result<String, String> {
                        let mut guard =
                            asr.lock().map_err(|_| "asr mutex poisoned".to_string())?;
                        match guard.as_mut() {
                            Some(t) => t.transcribe_pcm_text(buf, sr, lang),
                            None => Err("ASR-модель не загружена".to_string()),
                        }
                    };

                    loop {
                        match pcm_rx.recv() {
                            Ok(chunk) => {
                                buf.extend_from_slice(&chunk);
                                if buf.len() >= min_samples && buf.len() - last_len >= cadence {
                                    match run(&buf) {
                                        Ok(full) => {
                                            let prev_w: Vec<&str> =
                                                prev_text.split_whitespace().collect();
                                            let new_w: Vec<&str> =
                                                full.split_whitespace().collect();
                                            let delta =
                                                suffix_after_common_prefix(&prev_w, &new_w);
                                            if !delta.trim().is_empty() {
                                                let _ = tx.send(StreamingAsrEvent::Delta {
                                                    text: delta,
                                                    partial: true,
                                                });
                                            }
                                            prev_text = full;
                                            last_len = buf.len();
                                        }
                                        Err(e) => {
                                            let _ = tx.send(StreamingAsrEvent::Error(e));
                                            return;
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                if buf.is_empty() {
                                    let _ = tx
                                        .send(StreamingAsrEvent::Final { text: String::new() });
                                    return;
                                }
                                match run(&buf) {
                                    Ok(full) => {
                                        let _ = tx.send(StreamingAsrEvent::Final { text: full });
                                    }
                                    Err(e) => {
                                        let _ = tx.send(StreamingAsrEvent::Error(e));
                                    }
                                }
                                return;
                            }
                        }
                    }
                })
                .ok();

            Self { events_rx: rx }
        }
    }
}
