//! Транскрипция целиком: звук → окна по 300 с → память энкодера → события под
//! грамматикой → склейка → ABC.
//!
//! Повторяет `Transcriber.analyze` релиза в дефолтном пресете: перекрытие
//! 200 с, из них 100 с правого контекста; окно, за которым идёт следующее,
//! генерирует до 200-й секунды, следующее продолжает с префикса уже принятых
//! событий перекрытия.

use std::path::Path;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType};

use crate::config::SheetSage2Config;
use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::export::{export_abc, AbcExport};
pub use crate::export::VoiceSelect;
use crate::generate::constrained_generate;
use crate::loader::{read_file, Weights};
use crate::stitch::{sliding_window_plan, slice_audio, Stitcher, Window};
use crate::vocab::{Event, Vocab, FULL_TASK_PROMPTS};
use crate::{SheetError, COMPONENT};

pub struct TranscribeOptions {
    /// Партитура без аккордов — план кавера для YuE2 (`cot = melody`).
    pub melody_only: bool,
    /// Какие мелодии оставить в ABC.
    pub voices: VoiceSelect,
    pub overlap_seconds: f64,
    pub lookahead_seconds: f64,
    /// Обрезать запись до этой длины.
    pub max_seconds: Option<f64>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            melody_only: true,
            voices: VoiceSelect::Both,
            overlap_seconds: 200.0,
            lookahead_seconds: 100.0,
            max_seconds: None,
        }
    }
}

/// Стадии для индикатора.
#[derive(Debug, Clone, Copy)]
pub enum Progress {
    Encoding { window: usize, windows: usize },
    Decoding { window: usize, windows: usize, tokens: usize },
    Notation,
}

#[derive(Debug, Clone)]
pub struct WindowReport {
    pub window: Window,
    pub prefix_tokens: usize,
    pub tokens: Vec<u32>,
    pub events: usize,
    pub accepted_events: usize,
    pub encode_seconds: f64,
    pub decode_seconds: f64,
}

pub struct Transcription {
    /// Партитура (ABC) или `None` — тогда причина в `abc_error`.
    pub abc: Option<String>,
    pub abc_error: Option<String>,
    pub duration_seconds: f64,
    pub windows: Vec<WindowReport>,
    pub events: Vec<Event>,
    pub warnings: Vec<String>,
    pub export: AbcExport,
    pub seconds: f64,
}

impl Transcription {
    /// Тональность и темп из заголовка ABC — для подписи на карточке.
    pub fn header_field(&self, field: &str) -> Option<String> {
        let abc = self.abc.as_ref()?;
        abc.lines().find_map(|l| l.strip_prefix(&format!("{field}:")).map(|s| s.to_string()))
    }
}

pub struct SheetSage2 {
    pub config: SheetSage2Config,
    pub vocab: Vocab,
    encoder: Encoder,
    decoder: Decoder,
    pub compute: DType,
    pub device: Device,
    pub load_seconds: f64,
}

impl SheetSage2 {
    /// Загрузить компонент `sheetsage2` из бандла (обычно `yue2-3b.syn`).
    pub fn open(path: impl AsRef<Path>, device: Device, compute: DType) -> Result<Self, SheetError> {
        let t0 = Instant::now();
        let path = path.as_ref();
        let weights = Weights::open(path, device)?;
        let config = SheetSage2Config::from_json(&read_file(path, &format!("{COMPONENT}/config.json"))?)?;
        let vocab = Vocab::new(config.input_audio_length, config.time_hz as u32);
        if vocab.n_tokens as usize != config.vocab_size {
            return Err(SheetError::Config(format!(
                "словарь схемы v1 — {} токенов, модель ждёт {}",
                vocab.n_tokens, config.vocab_size
            )));
        }
        if vocab.fingerprint() != config.tokenizer_fingerprint {
            return Err(SheetError::Config("отпечаток словаря не совпал с чекпойнтом".into()));
        }
        let encoder = Encoder::load(&weights, &config, compute)?;
        let decoder = Decoder::load(&weights, &config, compute)?;
        Ok(Self {
            config,
            vocab,
            encoder,
            decoder,
            compute,
            device,
            load_seconds: t0.elapsed().as_secs_f64(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    /// Память энкодера для окна (для сверки с эталоном и отладки).
    pub fn encode_window(&self, window_audio: &[f32]) -> Result<synaptix_core::tensor::Tensor, SheetError> {
        self.encoder.forward(window_audio, &|| false)
    }

    /// Нормированный лог-мел окна (для сверки с эталоном).
    pub fn mel_window(&self, window_audio: &[f32]) -> Result<synaptix_core::tensor::Tensor, SheetError> {
        self.encoder.mel.forward(window_audio)
    }

    /// Окно звука (дополняется тишиной до окна модели) → токены жадной
    /// генерации с дефолтными подсказками, до `stop` секунд.
    pub fn generate_window(&self, audio: &[f32], stop: f64) -> Result<Vec<u32>, SheetError> {
        let mut window = audio.to_vec();
        window.resize(self.config.window_samples(), 0.0);
        let memory = self.encoder.forward(&window, &|| false)?;
        let mut state = self.decoder.start(&memory)?;
        let prompts = Vocab::normalize_prompts(&FULL_TASK_PROMPTS)?;
        let generated = constrained_generate(
            &self.decoder,
            &mut state,
            &self.vocab,
            &Vocab::prompt_prefix(&prompts),
            self.config.max_output_seq_len,
            Some(stop),
            &|| false,
            &mut |_| {},
        )?;
        Ok(generated.tokens)
    }

    /// Транскрибировать моно-запись 24 кГц.
    pub fn transcribe(
        &self,
        audio: &[f32],
        options: &TranscribeOptions,
        progress: &mut dyn FnMut(Progress),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Transcription, SheetError> {
        let t0 = Instant::now();
        let sr = self.config.sampling_rate;
        let mut audio = audio;
        if let Some(max) = options.max_seconds {
            if !(max.is_finite() && max > 0.0) {
                return Err(SheetError::Config("max_seconds должен быть положительным".into()));
            }
            let n = (max * sr as f64).round() as usize;
            audio = &audio[..audio.len().min(n)];
        }
        if audio.len() < self.config.backbone.minimum_input_samples() || audio.iter().any(|x| !x.is_finite()) {
            return Err(SheetError::Config("нужно не меньше 1025 конечных сэмплов 24 кГц".into()));
        }
        let duration = audio.len() as f64 / sr as f64;
        let window_length = self.config.input_audio_length;
        let plan = sliding_window_plan(duration, window_length, options.overlap_seconds, options.lookahead_seconds)?;
        let prompts = Vocab::normalize_prompts(&FULL_TASK_PROMPTS)?;
        let max_len = self.config.max_output_seq_len;
        let mut stitcher = Stitcher::new(&self.vocab, &prompts, duration, window_length);
        let mut reports = Vec::with_capacity(plan.len());
        let mut warnings = Vec::new();
        let windows = plan.len();
        for (index, window) in plan.iter().enumerate() {
            progress(Progress::Encoding { window: index + 1, windows });
            let (prefix, base) = match stitcher.prefix_for(index, window)? {
                Some((prefix, base)) => {
                    if prefix.len() >= max_len - 128 {
                        return Err(SheetError::Config(
                            "префикс перекрытия заполняет окно декодера — уменьшите перекрытие".into(),
                        ));
                    }
                    (prefix, base)
                }
                None => (Vocab::prompt_prefix(&prompts), 0),
            };
            let t_enc = Instant::now();
            let mut segment = slice_audio(audio, sr, window.start, window_length);
            segment.resize(self.config.window_samples(), 0.0);
            let memory = self.encoder.forward(&segment, cancel)?;
            let encode_seconds = t_enc.elapsed().as_secs_f64();
            let t_dec = Instant::now();
            let mut state = self.decoder.start(&memory)?;
            drop(memory);
            let stop = window.generation_stop.unwrap_or_else(|| (duration - window.start).min(window_length));
            let mut on_token = |n: usize| {
                if n % 64 == 0 {
                    progress(Progress::Decoding { window: index + 1, windows, tokens: n });
                }
            };
            let generated = constrained_generate(
                &self.decoder,
                &mut state,
                &self.vocab,
                &prefix,
                max_len,
                Some(stop),
                cancel,
                &mut on_token,
            )?;
            drop(state);
            if generated.tokens.len() > max_len {
                warnings.push(format!("окно {}: упёрлись в предел токенов — проверьте покрытие", index + 1));
            }
            let (events, accepted) = stitcher.accept(index, window, &generated.tokens, base)?;
            reports.push(WindowReport {
                window: *window,
                prefix_tokens: if index == 0 { 0 } else { prefix.len() },
                tokens: generated.tokens,
                events,
                accepted_events: accepted,
                encode_seconds,
                decode_seconds: t_dec.elapsed().as_secs_f64(),
            });
        }
        progress(Progress::Notation);
        let (events, stitch_warnings) = stitcher.finish();
        warnings.extend(stitch_warnings);
        let export = export_abc(&events, duration, options.melody_only, options.voices);
        Ok(Transcription {
            abc: export.abc.clone(),
            abc_error: export.abc_error.clone(),
            duration_seconds: duration,
            windows: reports,
            events,
            warnings,
            export,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }
}

/// Каналы любой частоты → моно 24 кГц (как `ffmpeg -ac 1 -ar 24000` у релиза:
/// среднее каналов, затем ресэмплинг).
pub fn prepare_audio(channels: &[Vec<f32>], sample_rate: u32, target_rate: u32) -> Result<Vec<f32>, SheetError> {
    if channels.is_empty() || channels[0].is_empty() {
        return Err(SheetError::Config("пустая запись".into()));
    }
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    let k = channels.len() as f32;
    let mono: Vec<f32> = (0..n).map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / k).collect();
    resample(&mono, sample_rate, target_rate)
}

/// FFT-ресэмплинг (rubato) с учётом задержки: длина — `round(len · out/in)`.
pub fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>, SheetError> {
    use rubato::{FftFixedIn, Resampler};
    if from == to {
        return Ok(input.to_vec());
    }
    let err = |e: String| SheetError::Other(format!("ресэмплинг {from} → {to}: {e}"));
    let mut rs = FftFixedIn::<f32>::new(from as usize, to as usize, 4096, 2, 1).map_err(|e| err(e.to_string()))?;
    let delay = rs.output_delay();
    let want = (input.len() as f64 * to as f64 / from as f64).round() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(want + delay + 8192);
    let mut pos = 0usize;
    while input.len() - pos >= rs.input_frames_next() {
        let n = rs.input_frames_next();
        let chunk = rs.process(&[&input[pos..pos + n]], None).map_err(|e| err(e.to_string()))?;
        out.extend_from_slice(&chunk[0]);
        pos += n;
    }
    if pos < input.len() {
        let chunk = rs.process_partial(Some(&[&input[pos..]]), None).map_err(|e| err(e.to_string()))?;
        out.extend_from_slice(&chunk[0]);
    }
    while out.len() < want + delay {
        let chunk = rs.process_partial::<&[f32]>(None, None).map_err(|e| err(e.to_string()))?;
        if chunk[0].is_empty() {
            break;
        }
        out.extend_from_slice(&chunk[0]);
    }
    let end = (delay + want).min(out.len());
    Ok(out[delay.min(end)..end].to_vec())
}
