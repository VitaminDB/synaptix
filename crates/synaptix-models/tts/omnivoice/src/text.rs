//! Текстовый фронтенд OmniVoice.
//!
//! Порт `_prepare_inference_inputs` + `_combine_text` + `_tokenize_with_nonverbal_tags`
//! + `RuleDurationEstimator` из `~/Temp/OmniVoice/omnivoice/{models/omnivoice.py,
//! utils/{text,duration}.py}`. Токенайзер — Qwen3 (`tokenizer.json`) через
//! `synaptix-tokenizer` (`HfTokenizer`). Special-токены резолвятся по содержимому
//! из added-vocab. См. SPEC.md «План порта» п.4.

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::{HfTokenizer, Tokenizer};

use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Other(e.to_string())
}

/// Список nonverbal-тегов (см. `_NONVERBAL_PATTERN`). Токенизируются
/// standalone, чтобы id были стабильны вне зависимости от контекста.
const NONVERBAL_TAGS: &[&str] = &[
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
];

/// Текстовый фронтенд: токенайзер + audio-конфиг (число кодбуков, MASK-id).
pub struct TextFrontend {
    tokenizer: HfTokenizer,
    num_audio_codebook: usize,
    audio_mask_id: i64,
}

/// Результат `prepare_inference_inputs`: cond `input_ids` [1, C, S] (I64) и
/// `audio_mask` [1, S] (U8).
pub struct PreparedInputs {
    pub input_ids: Tensor,
    pub audio_mask: Tensor,
    pub seq_len: usize,
}

impl TextFrontend {
    /// Собрать из `tokenizer.json`-байтов + audio-конфига.
    pub fn from_tokenizer_bytes(
        tokenizer_json: &[u8],
        num_audio_codebook: usize,
        audio_mask_id: i64,
    ) -> Result<Self> {
        let tokenizer = HfTokenizer::from_bytes(tokenizer_json).map_err(err)?;
        Ok(Self { tokenizer, num_audio_codebook, audio_mask_id })
    }

    /// Собрать из пути к `tokenizer.json` + audio-конфига.
    pub fn from_tokenizer_file(
        path: impl AsRef<std::path::Path>,
        num_audio_codebook: usize,
        audio_mask_id: i64,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(err)?;
        Self::from_tokenizer_bytes(&bytes, num_audio_codebook, audio_mask_id)
    }

    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

    /// Токенизация с обычными special-токенами (`tokenizer(text, return_tensors)`
    /// в Python = `add_special_tokens=True`; для Qwen3 без BOS/EOS-постпроцессора
    /// это не добавляет лишних токенов, но раскрывает `<|...|>`-теги).
    fn encode_with_specials(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.tokenizer.encode(text, true).map_err(err)?;
        Ok(enc.ids.iter().map(|&id| id as i64).collect())
    }

    /// Токенизация без special-токенов (`add_special_tokens=False`).
    fn encode_no_specials(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.tokenizer.encode(text, false).map_err(err)?;
        Ok(enc.ids.iter().map(|&id| id as i64).collect())
    }

    /// Порт `_tokenize_with_nonverbal_tags`: режет текст по nonverbal-тегам,
    /// каждый сегмент/тег токенизируется отдельно (`add_special_tokens=False`),
    /// конкатенация. Если тегов нет — fallback на `encode(text, add_special=True)`
    /// (как в upstream, где return_tensors-путь добавляет special).
    fn tokenize_with_nonverbal_tags(&self, text: &str) -> Result<Vec<i64>> {
        let matches = find_nonverbal(text);
        if matches.is_empty() {
            return self.encode_with_specials(text);
        }

        let mut combined: Vec<i64> = Vec::new();
        let mut last_end = 0usize;
        for (start, end) in matches {
            if start > last_end {
                let segment = &text[last_end..start];
                let ids = self.encode_no_specials(segment)?;
                combined.extend(ids);
            }
            let tag = &text[start..end];
            let tag_ids = self.encode_no_specials(tag)?;
            combined.extend(tag_ids);
            last_end = end;
        }
        if last_end < text.len() {
            let segment = &text[last_end..];
            let ids = self.encode_no_specials(segment)?;
            combined.extend(ids);
        }
        Ok(combined)
    }

    /// Порт `_prepare_inference_inputs` (B=1).
    ///
    /// Строит style-токены (`<|denoise|>`? + lang/instruct-обёртки), текст-токены
    /// (`<|text_start|>{combine(ref,text)}<|text_end|>` через nonverbal-путь),
    /// конкатенирует [style ; text ; ref_audio? ; MASK×T] по оси кодбуков (repeat
    /// по 8), audio_mask=true на хвосте (ref+target).
    ///
    /// `ref_audio_tokens` — опц. `[C, T_ref]` (I64) на `Device::Cpu`.
    pub fn prepare_inference_inputs(
        &self,
        text: &str,
        num_target_tokens: usize,
        ref_text: Option<&str>,
        ref_audio_tokens: Option<&Tensor>,
        lang: Option<&str>,
        instruct: Option<&str>,
        denoise: bool,
    ) -> Result<PreparedInputs> {
        let n_cb = self.num_audio_codebook;
        let device = Device::Cpu;

        // style_text = <|denoise|>? <|lang_start|>L<|lang_end|><|instruct_start|>I<|instruct_end|>
        let mut style_text = String::new();
        if denoise && ref_audio_tokens.is_some() {
            style_text.push_str("<|denoise|>");
        }
        let lang_str = lang.unwrap_or("None");
        let instruct_str = instruct.unwrap_or("None");
        style_text.push_str(&format!("<|lang_start|>{lang_str}<|lang_end|>"));
        style_text.push_str(&format!("<|instruct_start|>{instruct_str}<|instruct_end|>"));
        let style_tokens = self.encode_with_specials(&style_text)?;

        // full_text = _combine_text(ref_text, text); wrap <|text_start|>..<|text_end|>
        let full_text = combine_text(text, ref_text);
        let wrapped_text = format!("<|text_start|>{full_text}<|text_end|>");
        let text_tokens = self.tokenize_with_nonverbal_tags(&wrapped_text)?;

        let n_style = style_tokens.len();
        let n_text = text_tokens.len();
        let n_ref = ref_audio_tokens.map(|t| t.dims()[t.dims().len() - 1]).unwrap_or(0);
        let t = num_target_tokens;
        let total = n_style + n_text + n_ref + t;

        // ref_audio_tokens на host (I64) [C, T_ref].
        let ref_vals: Option<Vec<i64>> = match ref_audio_tokens {
            Some(rt) => {
                let v = rt
                    .to_dtype(synaptix_core::dtype::DType::I64)
                    .and_then(|x| x.flatten_all())
                    .and_then(|x| x.to_vec1::<i64>())
                    .map_err(err)?;
                Some(v)
            }
            None => None,
        };

        // cond_input_ids[1, C, total]: row-major по (c, s). style/text — одинаковы
        // для всех кодбуков (repeat); ref — по своим кодбукам; target — MASK.
        let mut ids = vec![0i64; n_cb * total];
        for c in 0..n_cb {
            let base = c * total;
            let mut s = 0usize;
            for &tok in &style_tokens {
                ids[base + s] = tok;
                s += 1;
            }
            for &tok in &text_tokens {
                ids[base + s] = tok;
                s += 1;
            }
            if let Some(rv) = &ref_vals {
                for j in 0..n_ref {
                    ids[base + s] = rv[c * n_ref + j];
                    s += 1;
                }
            }
            for _ in 0..t {
                ids[base + s] = self.audio_mask_id;
                s += 1;
            }
            debug_assert_eq!(s, total);
        }

        // audio_mask[1, total]: true на (ref+target)-хвосте = последние (n_ref+t).
        let audio_start = total - t - n_ref;
        let mut mask = vec![0u8; total];
        for m in mask.iter_mut().skip(audio_start) {
            *m = 1;
        }

        let input_ids = Tensor::from_vec(ids, vec![1, n_cb, total], device).map_err(err)?;
        let audio_mask = Tensor::from_vec(mask, vec![1, total], device).map_err(err)?;
        Ok(PreparedInputs { input_ids, audio_mask, seq_len: total })
    }
}

/// Найти все вхождения nonverbal-тегов `[tag]` → список (start, end) байтовых
/// смещений в порядке появления (как `re.finditer`).
fn find_nonverbal(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // ищем закрывающую ']' и проверяем содержимое на принадлежность списку
            if let Some(rel) = text[i + 1..].find(']') {
                let inner = &text[i + 1..i + 1 + rel];
                if NONVERBAL_TAGS.contains(&inner) {
                    let end = i + 1 + rel + 1;
                    out.push((i, end));
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// Порт `_combine_text(text, ref_text=None)`:
/// 1. ref ? `ref.strip()+" "+text.strip()` : `text.strip()`;
/// 2. убрать `\r\n` (схлопнуть в пусто);
/// 3. китайские скобки `（）`→`()`;
/// 4. схлопнуть пробелы/табы в один пробел;
/// 5. убрать пробелы вокруг китайских иероглифов.
pub fn combine_text(text: &str, ref_text: Option<&str>) -> String {
    let mut full = match ref_text {
        Some(r) if !r.is_empty() => format!("{} {}", r.trim(), text.trim()),
        _ => text.trim().to_string(),
    };

    // 2. убрать \r и \n (re.sub r"[\r\n]+" → "").
    full = full.chars().filter(|&c| c != '\r' && c != '\n').collect();

    // 3. китайские скобки → английские.
    full = full.replace('\u{ff08}', "(").replace('\u{ff09}', ")");

    // 4. схлопнуть последовательности пробелов/табов в один пробел.
    full = collapse_spaces(&full);

    // 5. убрать пробелы вокруг китайских иероглифов.
    full = strip_spaces_around_cjk(&full);

    full
}

/// re.sub(r"[ \t]+", " ", s).
fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// re.sub(r"(?<=CJK)\s+|\s+(?=CJK)", "", s): удалить whitespace, если он
/// примыкает (слева или справа) к китайскому иероглифу.
fn strip_spaces_around_cjk(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut keep = vec![true; n];
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            let prev_cjk = i > 0 && is_cjk(chars[i - 1]);
            let next_cjk = i + 1 < n && is_cjk(chars[i + 1]);
            if prev_cjk || next_cjk {
                keep[i] = false;
            }
        }
    }
    chars
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, &c)| c)
        .collect()
}

/// Порт `RuleDurationEstimator` (utils/duration.py): оценка длительности по
/// фонетическим весам символов.
pub struct DurationEstimator;

impl Default for DurationEstimator {
    fn default() -> Self {
        Self
    }
}

impl DurationEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Вес одного символа (порт `_get_char_weight`).
    fn char_weight(c: char) -> f64 {
        let code = c as u32;
        // ASCII-латиница A-Z/a-z.
        if (65..=90).contains(&code) || (97..=122).contains(&code) {
            return 1.0; // latin
        }
        if code == 32 {
            return 0.2; // space
        }
        // Arabic Tatweel → mark (0.0).
        if code == 0x0640 {
            return 0.0;
        }
        let cat = unicode_category(c);
        match cat {
            UCat::Mark => return 0.0,
            UCat::Punct => return 0.5,
            UCat::Separator => return 0.2,
            UCat::Number => return 3.5,
            UCat::Other => {}
        }
        // Binary search по unicode-блокам.
        if let Some(w) = block_weight(code) {
            return w;
        }
        if code > 0x20000 {
            return 3.0; // cjk (upper planes)
        }
        1.0 // default
    }

    pub fn total_weight(text: &str) -> f64 {
        text.chars().map(Self::char_weight).sum()
    }

    /// Порт `estimate_duration`. Возвращает оценённую длительность (в тех же
    /// единицах, что и `ref_duration` — у OmniVoice это число ref-аудио-токенов).
    pub fn estimate_duration(
        &self,
        target_text: &str,
        ref_text: &str,
        ref_duration: f64,
    ) -> f64 {
        let low_threshold = 50.0f64;
        let boost_strength = 3.0f64;

        if ref_duration <= 0.0 || ref_text.is_empty() {
            return 0.0;
        }
        let ref_weight = Self::total_weight(ref_text);
        if ref_weight == 0.0 {
            return 0.0;
        }
        let speed_factor = ref_weight / ref_duration;
        let target_weight = Self::total_weight(target_text);
        let estimated = target_weight / speed_factor;
        if estimated < low_threshold {
            let alpha = 1.0 / boost_strength;
            low_threshold * (estimated / low_threshold).powf(alpha)
        } else {
            estimated
        }
    }

    /// Порт `_estimate_target_tokens` (auto-fallback ref="Nice to meet you.",
    /// num_ref=25 если ref-текста/аудио нет). Возвращает `max(1, int(est/speed))`.
    pub fn estimate_target_tokens(
        &self,
        text: &str,
        ref_text: Option<&str>,
        num_ref_audio_tokens: Option<usize>,
        speed: f64,
    ) -> usize {
        let (rt, num_ref): (String, f64) = match (num_ref_audio_tokens, ref_text) {
            (Some(n), Some(r)) if !r.is_empty() => (r.to_string(), n as f64),
            _ => ("Nice to meet you.".to_string(), 25.0),
        };
        let mut est = self.estimate_duration(text, &rt, num_ref);
        if speed > 0.0 && speed != 1.0 {
            est /= speed;
        }
        (est as i64).max(1) as usize
    }
}

#[derive(PartialEq, Eq)]
enum UCat {
    Mark,
    Punct,
    Separator,
    Number,
    Other,
}

/// Грубая категоризация unicode для duration-весов. Покрывает категории, которые
/// проверяет upstream (`startswith("M"/"P"/"S"/"Z"/"N")`) для ASCII + общих
/// диапазонов; для не-ASCII-букв возвращает Other (далее идёт block-lookup).
fn unicode_category(c: char) -> UCat {
    let code = c as u32;
    // ASCII-пунктуация и символы.
    if code < 0x80 {
        if c.is_ascii_digit() {
            return UCat::Number;
        }
        if c == ' ' || c == '\t' {
            return UCat::Separator;
        }
        if c.is_ascii_punctuation() {
            return UCat::Punct;
        }
        return UCat::Other;
    }
    // Combining diacritical marks (0300-036F), и расширения.
    if (0x0300..=0x036F).contains(&code)
        || (0x1AB0..=0x1AFF).contains(&code)
        || (0x1DC0..=0x1DFF).contains(&code)
        || (0x20D0..=0x20FF).contains(&code)
        || (0xFE20..=0xFE2F).contains(&code)
    {
        return UCat::Mark;
    }
    // Прочие пробелы (Zs/Zl/Zp).
    if c.is_whitespace() {
        return UCat::Separator;
    }
    // Полноширинные/прочие цифры покрываются block-lookup'ом (digit=3.5) — но
    // upstream ловит их раньше через category "N"; для частых кейсов достаточно.
    UCat::Other
}

/// Вес unicode-блока (порт таблицы `self.ranges` + bisect). Возвращает None,
/// если код вне таблицы (выше 0xFFEF → fallback в caller).
fn block_weight(code: u32) -> Option<f64> {
    // (end_codepoint, weight) — упорядочены по возрастанию end (bisect_left).
    const RANGES: &[(u32, f64)] = &[
        (0x02AF, 1.0),  // latin
        (0x03FF, 1.0),  // greek
        (0x052F, 1.0),  // cyrillic
        (0x058F, 1.0),  // armenian
        (0x05FF, 1.5),  // hebrew
        (0x077F, 1.5),  // arabic
        (0x089F, 1.5),  // arabic
        (0x08FF, 1.5),  // arabic
        (0x097F, 1.8),  // indic devanagari
        (0x09FF, 1.8),  // bengali
        (0x0A7F, 1.8),  // gurmukhi
        (0x0AFF, 1.8),  // gujarati
        (0x0B7F, 1.8),  // oriya
        (0x0BFF, 1.8),  // tamil
        (0x0C7F, 1.8),  // telugu
        (0x0CFF, 1.8),  // kannada
        (0x0D7F, 1.8),  // malayalam
        (0x0DFF, 1.8),  // sinhala
        (0x0EFF, 1.5),  // thai_lao
        (0x0FFF, 1.8),  // tibetan→indic
        (0x109F, 1.8),  // myanmar→khmer_myanmar
        (0x10FF, 1.0),  // georgian
        (0x11FF, 2.5),  // hangul jamo
        (0x137F, 3.0),  // ethiopic
        (0x139F, 3.0),  // ethiopic supp
        (0x13FF, 1.0),  // cherokee→default
        (0x167F, 1.0),  // canadian→default
        (0x169F, 1.0),  // ogham→default
        (0x16FF, 1.0),  // runic→default
        (0x171F, 1.0),  // tagalog→default
        (0x173F, 1.0),  // hanunoo→default
        (0x175F, 1.0),  // buhid→default
        (0x177F, 1.0),  // tagbanwa→default
        (0x17FF, 1.8),  // khmer→khmer_myanmar
        (0x18AF, 1.0),  // mongolian→default
        (0x18FF, 1.0),  // canadian ext→default
        (0x194F, 1.8),  // limbu→indic
        (0x19DF, 1.8),  // tai le→indic
        (0x19FF, 1.8),  // khmer symbols→khmer_myanmar
        (0x1A1F, 1.8),  // buginese→indic
        (0x1AAF, 1.8),  // tai tham→indic
        (0x1B7F, 1.8),  // balinese→indic
        (0x1BBF, 1.8),  // sundanese→indic
        (0x1BFF, 1.8),  // batak→indic
        (0x1C4F, 1.8),  // lepcha→indic
        (0x1C7F, 1.8),  // ol chiki→indic
        (0x1C8F, 1.0),  // cyrillic ext-c
        (0x1CBF, 1.0),  // georgian ext
        (0x1CCF, 1.8),  // sundanese supp→indic
        (0x1CFF, 1.8),  // vedic→indic
        (0x1D7F, 1.0),  // phonetic ext→latin
        (0x1DBF, 1.0),  // phonetic ext supp→latin
        (0x1DFF, 1.0),  // combining diacritical supp→default
        (0x1EFF, 1.0),  // latin ext additional (vietnamese)
        (0x309F, 2.2),  // hiragana→kana
        (0x30FF, 2.2),  // katakana→kana
        (0x312F, 3.0),  // bopomofo→cjk
        (0x318F, 2.5),  // hangul compat jamo
        (0x9FFF, 3.0),  // cjk unified
        (0xA4CF, 3.0),  // yi syllables
        (0xA4FF, 1.0),  // lisu→default
        (0xA63F, 1.0),  // vai→default
        (0xA69F, 1.0),  // cyrillic ext-b
        (0xA6FF, 1.0),  // bamum→default
        (0xA7FF, 1.0),  // latin ext-d
        (0xA82F, 1.8),  // syloti nagri→indic
        (0xA87F, 1.0),  // phags-pa→default
        (0xA8DF, 1.8),  // saurashtra→indic
        (0xA8FF, 1.8),  // devanagari ext→indic
        (0xA92F, 1.8),  // kayah li→indic
        (0xA95F, 1.8),  // rejang→indic
        (0xA97F, 2.5),  // hangul jamo ext-a
        (0xA9DF, 1.8),  // javanese→indic
        (0xA9FF, 1.8),  // myanmar ext-b→khmer_myanmar
        (0xAA5F, 1.8),  // cham→indic
        (0xAA7F, 1.8),  // myanmar ext-a→khmer_myanmar
        (0xAADF, 1.8),  // tai viet→indic
        (0xAAFF, 1.8),  // meetei mayek ext→indic
        (0xAB2F, 3.0),  // ethiopic ext-a
        (0xAB6F, 1.0),  // latin ext-e
        (0xABBF, 1.0),  // cherokee supp→default
        (0xABFF, 1.8),  // meetei mayek→indic
        (0xD7AF, 2.5),  // hangul syllables
        (0xFAFF, 3.0),  // cjk compat
        (0xFDFF, 1.5),  // arabic pres forms-a
        (0xFE6F, 1.0),  // variation selectors→default
        (0xFEFF, 1.5),  // arabic pres forms-b
        (0xFFEF, 1.0),  // fullwidth latin
    ];
    // bisect_left по end-кодпоинтам: первый range с end >= code.
    for &(end, w) in RANGES {
        if code <= end {
            return Some(w);
        }
    }
    None
}
