//! Voice-clone prompt: ref-аудио (+ ref-текст) → `VoiceClonePrompt`.
//!
//! Порт `create_voice_clone_prompt` + аудио-утилит из
//! `~/Temp/OmniVoice/omnivoice/{models/omnivoice.py, utils/audio.py, utils/text.py}`:
//!   load_audio → mono → resample→24к (torchaudio bit-faithful);
//!   ref_rms = sqrt(mean(x²)); если 0<rms<0.1 → x *= 0.1/rms;
//!   preprocess_prompt (ref_text задан → trim НЕ применяется): remove_silence
//!     (pydub split_on_silence + remove_silence_edges, mid=200/lead=100/trail=200);
//!   clip хвост до кратного hop_length; encode (CodecEncoder) → ref_audio_tokens [C,T];
//!   add_punctuation(ref_text).
//!
//! `remove_silence` — bit-faithful порт pydub: float→int16 (×32768, clip), dBFS
//! по audioop.rms (int16), ms-гранулярность срезов. См. SPEC.md «План порта» п.8.

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

use crate::audio_encode::{resample, CodecEncoder};
use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Audio(e.to_string())
}

/// Результат `create_voice_clone_prompt`: ref-коды `[C, T]` (i64, host), ref-текст
/// (с пунктуацией), ref_rms (для post-process громкости).
pub struct VoiceClonePrompt {
    pub ref_audio_tokens: Tensor,
    pub ref_text: String,
    pub ref_rms: f32,
}

// ── pydub-faithful AudioSegment (int16 mono) ──────────────────────────────────

/// Минимальный int16-mono AudioSegment (sample_width=2, channels=1) — достаточно
/// для silence-removal пути voice-clone (numpy_to_audiosegment даёт mono int16).
#[derive(Clone)]
struct Seg {
    /// int16-сэмплы.
    data: Vec<i16>,
    frame_rate: usize,
}

impl Seg {
    /// `numpy_to_audiosegment`: float (C,T) mono → int16. `(audio*32768).clip(-32768,32767)`.
    fn from_f32(samples: &[f32], frame_rate: usize) -> Self {
        let data: Vec<i16> = samples
            .iter()
            .map(|&x| {
                let v = (x * 32768.0).round_ties_even_like();
                v.clamp(-32768.0, 32767.0) as i16
            })
            .collect();
        Self { data, frame_rate }
    }

    /// `audiosegment_to_numpy`: int16 → float32 `/32768`.
    fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32 / 32768.0).collect()
    }

    fn silent(frame_rate: usize) -> Self {
        Self { data: Vec::new(), frame_rate }
    }

    /// len в ms = round(1000 * frame_count / frame_rate). Python `round` =
    /// banker's (round-half-to-even), не f64::round (half-away).
    fn len_ms(&self) -> usize {
        let v = 1000.0 * self.data.len() as f64 / self.frame_rate as f64;
        round_half_even(v) as usize
    }

    /// _parse_position(ms) = int(ms * frame_rate / 1000) (число фреймов).
    fn ms_to_frames(&self, ms: usize) -> usize {
        ((ms as f64) * (self.frame_rate as f64) / 1000.0) as usize
    }

    /// Срез по ms `[start, end)` (как AudioSegment.__getitem__ slice). Границы
    /// клампятся к len(self) в ms (Python: start/end = min(.., len(self))), затем
    /// _parse_position → фреймы. Если фактических данных меньше ожидаемого (хвост)
    /// — pydub добивает тишиной (missing_frames). Воспроизводим паддинг нулями.
    fn slice_ms(&self, start_ms: usize, end_ms: usize) -> Seg {
        let lm = self.len_ms();
        let s = start_ms.min(lm);
        let e = end_ms.min(lm);
        let sf = self.ms_to_frames(s);
        let ef = self.ms_to_frames(e);
        let expected = ef.saturating_sub(sf);
        let avail_end = ef.min(self.data.len());
        let avail_start = sf.min(self.data.len());
        let mut data = self.data[avail_start..avail_end.max(avail_start)].to_vec();
        let missing = expected.saturating_sub(data.len());
        if missing > 0 {
            data.extend(std::iter::repeat(0i16).take(missing));
        }
        Seg { data, frame_rate: self.frame_rate }
    }

    /// Конкатенация (append crossfade=0 → seg1._data + seg2._data).
    fn concat(&self, other: &Seg) -> Seg {
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        Seg { data, frame_rate: self.frame_rate }
    }

    fn reverse(&self) -> Seg {
        let mut data = self.data.clone();
        data.reverse();
        Seg { data, frame_rate: self.frame_rate }
    }

    /// audioop.rms(data, 2): CPython вычисляет `sqrt((double)(sum_sq/len))` и
    /// приводит к int усечением (`(int)sqrt(...)`); sum_sq аккумулируется в
    /// double, len — целое (целочисленное деление НЕ применяется, делится double).
    fn rms(&self) -> i64 {
        let n = self.data.len();
        if n == 0 {
            return 0;
        }
        let mut sum = 0.0f64;
        for &v in &self.data {
            let x = v as f64;
            sum += x * x;
        }
        (sum / n as f64).sqrt() as i64
    }

    /// dBFS = ratio_to_db(rms / max_possible_amplitude); rms==0 → -inf.
    /// max_possible_amplitude = 2^16/2 = 32768.
    fn dbfs(&self) -> f64 {
        let r = self.rms();
        if r == 0 {
            return f64::NEG_INFINITY;
        }
        20.0 * (r as f64 / 32768.0).log10()
    }
}

/// Округление как numpy .astype(int16) после умножения: numpy round-half-to-even
/// НЕ применяется — astype усекает к нулю. Но `(audio*32768)` без round → astype
/// усекает (truncate toward zero). Воспроизводим усечение к нулю.
trait RoundLike {
    fn round_ties_even_like(self) -> f32;
}
impl RoundLike for f32 {
    fn round_ties_even_like(self) -> f32 {
        // numpy astype(np.int16) усекает к нулю (truncation), НЕ округляет.
        self.trunc()
    }
}

// ── pydub silence detection ───────────────────────────────────────────────────

/// `detect_silence`: список [start_ms, end_ms] тихих участков.
fn detect_silence(seg: &Seg, min_silence_len: usize, silence_thresh_db: f64, seek_step: usize) -> Vec<(usize, usize)> {
    let seg_len = seg.len_ms();
    if seg_len < min_silence_len {
        return Vec::new();
    }
    // silence_thresh = db_to_float(thresh) * max_possible_amplitude (амплитудный).
    let silence_thresh = 10f64.powf(silence_thresh_db / 20.0) * 32768.0;

    let mut silence_starts: Vec<usize> = Vec::new();
    let last_slice_start = seg_len - min_silence_len;
    let mut slice_starts: Vec<usize> = (0..=last_slice_start).step_by(seek_step).collect();
    if last_slice_start % seek_step != 0 {
        slice_starts.push(last_slice_start);
    }
    for &i in &slice_starts {
        let sl = seg.slice_ms(i, i + min_silence_len);
        if (sl.rms() as f64) <= silence_thresh {
            silence_starts.push(i);
        }
    }
    if silence_starts.is_empty() {
        return Vec::new();
    }

    let mut silent_ranges: Vec<(usize, usize)> = Vec::new();
    let mut iter = silence_starts.iter();
    let mut prev_i = *iter.next().unwrap();
    let mut current_range_start = prev_i;
    for &si in iter {
        let continuous = si == prev_i + seek_step;
        let silence_has_gap = si > prev_i + min_silence_len;
        if !continuous && silence_has_gap {
            silent_ranges.push((current_range_start, prev_i + min_silence_len));
            current_range_start = si;
        }
        prev_i = si;
    }
    silent_ranges.push((current_range_start, prev_i + min_silence_len));
    silent_ranges
}

/// `detect_nonsilent`: список [start_ms, end_ms] звучащих участков.
fn detect_nonsilent(seg: &Seg, min_silence_len: usize, silence_thresh_db: f64, seek_step: usize) -> Vec<(usize, usize)> {
    let silent = detect_silence(seg, min_silence_len, silence_thresh_db, seek_step);
    let len_seg = seg.len_ms();
    if silent.is_empty() {
        return vec![(0, len_seg)];
    }
    if silent[0].0 == 0 && silent[0].1 == len_seg {
        return Vec::new();
    }
    let mut nonsilent: Vec<(usize, usize)> = Vec::new();
    let mut prev_end_i = 0usize;
    let mut last_end = 0usize;
    for &(start_i, end_i) in &silent {
        nonsilent.push((prev_end_i, start_i));
        prev_end_i = end_i;
        last_end = end_i;
    }
    if last_end != len_seg {
        nonsilent.push((prev_end_i, len_seg));
    }
    if !nonsilent.is_empty() && nonsilent[0] == (0, 0) {
        nonsilent.remove(0);
    }
    nonsilent
}

/// `split_on_silence`: список Seg, склейка диапазонов nonsilent ± keep_silence,
/// со сглаживанием пересечений (midpoint).
fn split_on_silence(
    seg: &Seg,
    min_silence_len: usize,
    silence_thresh_db: f64,
    keep_silence: usize,
    seek_step: usize,
) -> Vec<Seg> {
    let nonsilent = detect_nonsilent(seg, min_silence_len, silence_thresh_db, seek_step);
    // output_ranges = [start-keep, end+keep] (могут быть отрицательны → i64).
    let mut ranges: Vec<(i64, i64)> = nonsilent
        .iter()
        .map(|&(s, e)| (s as i64 - keep_silence as i64, e as i64 + keep_silence as i64))
        .collect();
    // pairwise: если next_start < last_end → midpoint.
    for i in 0..ranges.len().saturating_sub(1) {
        let last_end = ranges[i].1;
        let next_start = ranges[i + 1].0;
        if next_start < last_end {
            let mid = (last_end + next_start) / 2; // python // (floor для int)
            ranges[i].1 = mid;
            ranges[i + 1].0 = mid;
        }
    }
    let len_ms = seg.len_ms() as i64;
    ranges
        .iter()
        .map(|&(s, e)| {
            let start = s.max(0).min(len_ms) as usize;
            let end = e.max(0).min(len_ms) as usize;
            seg.slice_ms(start, end)
        })
        .collect()
}

/// `detect_leading_silence`: ms, на котором заканчивается ведущая тишина.
/// `while sound[t:t+chunk].dBFS < thresh and t < len(sound): t += chunk`.
fn detect_leading_silence(seg: &Seg, silence_threshold_db: f64, chunk_size: usize) -> usize {
    let mut trim_ms = 0usize;
    let len_ms = seg.len_ms();
    while seg.slice_ms(trim_ms, trim_ms + chunk_size).dbfs() < silence_threshold_db && trim_ms < len_ms {
        trim_ms += chunk_size;
    }
    trim_ms.min(len_ms)
}

/// `remove_silence_edges`: срезать края, оставив lead/trail ms.
fn remove_silence_edges(seg: &Seg, lead_sil: usize, trail_sil: usize, silence_threshold_db: f64) -> Seg {
    let mut audio = seg.clone();
    let start_idx = detect_leading_silence(&audio, silence_threshold_db, 10);
    let start_idx = start_idx.saturating_sub(lead_sil);
    audio = audio.slice_ms(start_idx, audio.len_ms());

    audio = audio.reverse();
    let start_idx = detect_leading_silence(&audio, silence_threshold_db, 10);
    let start_idx = start_idx.saturating_sub(trail_sil);
    audio = audio.slice_ms(start_idx, audio.len_ms());
    audio.reverse()
}

/// Порт `remove_silence` (utils/audio.py). `audio` — mono f32 [N], `sampling_rate`
/// = 24000. Возвращает mono f32 [N'].
pub fn remove_silence(
    audio: &[f32],
    sampling_rate: usize,
    mid_sil: usize,
    lead_sil: usize,
    trail_sil: usize,
) -> Vec<f32> {
    let mut wave = Seg::from_f32(audio, sampling_rate);

    if mid_sil > 0 {
        let segs = split_on_silence(&wave, mid_sil, -50.0, mid_sil, 10);
        let mut acc = Seg::silent(sampling_rate);
        for seg in &segs {
            acc = acc.concat(seg);
        }
        wave = acc;
    }

    let wave = remove_silence_edges(&wave, lead_sil, trail_sil, -50.0);
    wave.to_f32()
}

/// Порт `add_punctuation` (utils/text.py): добавить `.`/`。` если в конце нет
/// завершающего знака.
pub fn add_punctuation(text: &str) -> String {
    let t = text.trim();
    if t.is_empty() {
        return t.to_string();
    }
    let last = t.chars().last().unwrap();
    if END_PUNCTUATION.contains(&last) {
        return t.to_string();
    }
    let is_chinese = t.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    let mut out = t.to_string();
    out.push(if is_chinese { '\u{3002}' } else { '.' });
    out
}

/// END_PUNCTUATION (utils/text.py) — посимвольно. `……` многосимвольный, но
/// проверка идёт по последнему char, так что достаточно одиночных.
const END_PUNCTUATION: &[char] = &[
    ';', ':', ',', '.', '!', '?', '\u{2026}', ')', ']', '}', '"', '\'',
    '\u{201d}', '\u{2019}', '\u{ff1b}', '\u{ff1a}', '\u{ff0c}', '\u{3002}',
    '\u{ff01}', '\u{ff1f}', '\u{3001}', '\u{ff09}', '\u{3011}', '\u{201c}',
    '\u{2018}',
];

/// `create_voice_clone_prompt(ref_audio_path, ref_text)` — собрать ref-промпт.
///
/// `ref_text` ДОЛЖЕН быть задан (auto-transcribe = ASR-зависимость, не здесь).
/// `preprocess_prompt=true` → remove_silence (mid=200/lead=100/trail=200) и
/// add_punctuation; trim НЕ применяется (ref_text задан, как в upstream).
pub fn create_voice_clone_prompt(
    encoder: &CodecEncoder,
    ref_audio_path: impl AsRef<std::path::Path>,
    ref_text: &str,
    sampling_rate: usize,
    hop_length: usize,
    preprocess_prompt: bool,
) -> Result<VoiceClonePrompt> {
    // load_audio: read → mono → resample → sampling_rate.
    let (mono, sr) = synaptix_audio::read_wav_mono_f32(ref_audio_path.as_ref())
        .map_err(|e| OmniVoiceError::Audio(format!("read_wav: {e}")))?;
    let mut ref_wav = if sr as usize != sampling_rate {
        resample(&mono, sr as usize, sampling_rate)
    } else {
        mono
    };

    // ref_rms = sqrt(mean(x²)); если 0<rms<0.1 → x *= 0.1/rms.
    let ref_rms = rms_f32(&ref_wav);
    if ref_rms > 0.0 && ref_rms < 0.1 {
        let scale = 0.1 / ref_rms;
        for x in ref_wav.iter_mut() {
            *x *= scale;
        }
    }

    if preprocess_prompt {
        // ref_text задан → trim НЕ применяется. remove_silence (200/100/200).
        ref_wav = remove_silence(&ref_wav, sampling_rate, 200, 100, 200);
        if ref_wav.is_empty() {
            return Err(OmniVoiceError::Audio(
                "reference audio is empty after silence removal".into(),
            ));
        }
    }

    // clip хвост до кратного hop_length.
    let clip = ref_wav.len() % hop_length;
    if clip > 0 {
        ref_wav.truncate(ref_wav.len() - clip);
    }

    // encode → ref_audio_tokens [C, T].
    let n = ref_wav.len();
    let input = Tensor::from_vec(ref_wav, vec![n], Device::Cpu).map_err(err)?;
    let ref_audio_tokens = encoder.encode(&input)?;

    let ref_text_out = if preprocess_prompt {
        add_punctuation(ref_text)
    } else {
        ref_text.to_string()
    };

    Ok(VoiceClonePrompt {
        ref_audio_tokens,
        ref_text: ref_text_out,
        ref_rms,
    })
}

/// Python `round`: round-half-to-even (banker's).
fn round_half_even(v: f64) -> f64 {
    let f = v.floor();
    let diff = v - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else {
        // ровно .5 → к чётному.
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

/// rms = sqrt(mean(x²)) (как `np.sqrt(np.mean(ref_wav**2))`).
fn rms_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &v in x {
        let d = v as f64;
        sum += d * d;
    }
    (sum / x.len() as f64).sqrt() as f32
}
