
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::AceError;

pub const NUM_AUDIO_CODES: u32 = 64000;

// Optional fields are `None` until phase-1 / the LM CoT actually produces them.
// The phase-2 CoT (metadata_yaml) and the DiT metas block must NOT inject
// concrete defaults (120 BPM / C major / en / 4/4) for absent fields — Python
// emits only LM-produced keys (CoT) or "N/A" (DiT). Injecting defaults pinned
// every uncued track to a generic 120-BPM C-major 4/4 English feel.
#[derive(Debug, Clone)]
pub struct Metadata {
    pub bpm: Option<u32>,
    pub caption: String,
    pub duration: u32,
    pub genres: Option<String>,
    pub keyscale: Option<String>,
    pub language: Option<String>,
    pub timesignature: Option<String>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            bpm: None,
            caption: String::new(),
            duration: 30,
            genres: None,
            keyscale: None,
            language: None,
            timesignature: None,
        }
    }
}

pub struct AceTokenizer {
    hf: HfTokenizer,
    audio_base: u32,
    eos: u32,
    bos: u32,
}

pub fn parse_metadata_text(cot_text: &str, base: &Metadata) -> Metadata {
    let mut m = base.clone();
    let body = match (cot_text.find("<think>"), cot_text.find("</think>")) {
        (Some(a), Some(b)) if b > a => &cot_text[a + 7..b],
        _ => cot_text,
    };
    // YAML как у Python parse_lm_output: новое поле — строка без отступа с ':',
    // строки с отступом продолжают значение. LM переносит длинный caption по
    // ширине 80 — раньше продолжения терялись и caption обрывался на полуслове.
    let mut fields: Vec<(String, Vec<&str>)> = Vec::new();
    for line in body.lines() {
        if line.trim_start().starts_with('<') {
            continue;
        }
        if line.starts_with([' ', '\t']) {
            if let Some((_, vals)) = fields.last_mut() {
                vals.push(line);
            }
        } else if let Some((k, v)) = line.split_once(':') {
            fields.push((k.trim().to_lowercase(), vec![v]));
        }
    }
    for (k, vals) in fields {
        // Строки значения склеиваются через пробел (postprocess_caption).
        let joined = vals.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        // Strip a leading list dash + surrounding YAML quotes (match old port).
        let v = joined.trim_start_matches('-').trim().trim_matches('\'').trim_matches('"').trim();
        match k.as_str() {
            // No bpm clamp (Python parse_lm_output does none); only Some when present.
            "bpm" => {
                if let Ok(n) = v.parse::<u32>() {
                    m.bpm = Some(n);
                }
            }
            // duration clamped only to guard the codes/KV budget (OOM), not for fidelity.
            "duration" => {
                if let Ok(n) = v.parse::<f32>() {
                    m.duration = (n.round() as u32).clamp(1, 600);
                }
            }
            "caption" if !v.is_empty() => m.caption = v.to_string(),
            "keyscale" if !v.is_empty() => m.keyscale = Some(v.to_string()),
            "language" if !v.is_empty() => m.language = Some(v.to_string()),
            // Verbatim string — preserve the denominator ("6/8", "3/4", not just "/4").
            "timesignature" if !v.is_empty() => m.timesignature = Some(v.to_string()),
            "genres" if !v.is_empty() => m.genres = Some(v.to_string()),
            _ => {}
        }
    }
    m
}

/// Скаляр, который PyYAML пишет без кавычек (блочный контекст): иначе
/// `yaml.dump` берёт одинарные кавычки. Цифры — это int после `isdigit()`.
fn yaml_plain_ok(s: &str) -> bool {
    let Some(first) = s.chars().next() else { return false };
    if s != s.trim() || s.contains('\n') || s.contains(": ") || s.contains(" #") || s.ends_with(':') {
        return false;
    }
    if "#,[]{}&*!|>'\"%@`".contains(first) {
        return false;
    }
    if matches!(first, '-' | '?' | ':') && s.chars().nth(1).map_or(true, |c| c == ' ') {
        return false;
    }
    let l = s.to_lowercase();
    if matches!(l.as_str(), "yes" | "no" | "true" | "false" | "on" | "off" | "null" | "~") {
        return false;
    }
    let numeric = s.bytes().any(|b| b.is_ascii_digit())
        && s.bytes().all(|b| b.is_ascii_digit() || b"+-._eE".contains(&b));
    !numeric || s.bytes().all(|b| b.is_ascii_digit())
}

/// `key: value` как `yaml.dump(width=80)`: значение переносится по одиночным
/// пробелам, когда колонка ушла за 80, продолжение — с отступом 2. В таком
/// виде LM видела CoT на обучении и так печатает его сама.
fn yaml_field(key: &str, value: &str) -> String {
    let scalar =
        if yaml_plain_ok(value) { value.to_string() } else { format!("'{}'", value.replace('\'', "''")) };
    let mut out = format!("{key}:");
    let mut col = out.chars().count();
    for (i, word) in scalar.split(' ').enumerate() {
        if i > 0 && col > 80 {
            out.push_str("\n  ");
            col = 2;
        } else {
            out.push(' ');
            col += 1;
        }
        out.push_str(word);
        col += word.chars().count();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_CAPTION: &str = "A melancholic and romantic hip-hop track built around a clean, reverberant piano loop with deep 808 bass and crisp hi-hats, featuring an emotional male rap vocal in Russian.";

    /// Длинный caption LM печатает с YAML-переносами — склеивается целиком.
    #[test]
    fn parse_cot_multiline_caption() {
        let cot = "<think>\nbpm: 86\ncaption: A melancholic and romantic hip-hop track built around a clean, reverberant\n  piano loop with deep 808 bass and crisp hi-hats, featuring an emotional male rap\n  vocal in Russian.\nduration: 135\nkeyscale: C# major\nlanguage: ru\ntimesignature: 4\n</think>";
        let m = parse_metadata_text(cot, &Metadata::default());
        assert_eq!(m.caption, LONG_CAPTION);
        assert_eq!(m.bpm, Some(86));
        assert_eq!(m.duration, 135);
        assert_eq!(m.keyscale.as_deref(), Some("C# major"));
        assert_eq!(m.language.as_deref(), Some("ru"));
        assert_eq!(m.timesignature.as_deref(), Some("4"));
    }

    /// Эталон — вывод `yaml.dump(..., allow_unicode=True, sort_keys=True)`
    /// (PyYAML 6.0.3) для тех же метаданных; "4/4" → 4, как в Python.
    #[test]
    fn metadata_yaml_matches_pyyaml_folding() {
        let m = Metadata {
            bpm: Some(86),
            caption: LONG_CAPTION.into(),
            duration: 135,
            keyscale: Some("C# major".into()),
            language: Some("ru".into()),
            timesignature: Some("4/4".into()),
            genres: None,
        };
        let yaml = AceTokenizer::metadata_yaml(&m);
        assert_eq!(
            yaml,
            "bpm: 86\ncaption: A melancholic and romantic hip-hop track built around a clean, reverberant\n  piano loop with deep 808 bass and crisp hi-hats, featuring an emotional male rap\n  vocal in Russian.\nduration: 135\nkeyscale: C# major\nlanguage: ru\ntimesignature: 4"
        );
        let back = parse_metadata_text(&format!("<think>\n{yaml}\n</think>"), &Metadata::default());
        assert_eq!(back.caption, LONG_CAPTION);
    }

    /// Небезопасные для plain-скаляра значения — в одинарных кавычках, как PyYAML.
    #[test]
    fn yaml_field_quotes_like_pyyaml() {
        assert_eq!(yaml_field("language", "no"), "language: 'no'");
        assert_eq!(yaml_field("caption", "Intro: soft piano"), "caption: 'Intro: soft piano'");
        assert_eq!(yaml_field("timesignature", "6/8"), "timesignature: 6/8");
        assert_eq!(yaml_field("timesignature", "3"), "timesignature: 3");
    }

    #[test]
    fn parse_cot_metadata() {
        let base = Metadata::default();
        let cot = "<think>\nbpm: 128\ncaption: dreamy synthwave\nduration: 42\nkeyscale: A minor\nlanguage: en\ntimesignature: 6/8\n</think>";
        let m = parse_metadata_text(cot, &base);
        assert_eq!(m.bpm, Some(128));
        assert_eq!(m.duration, 42);
        assert_eq!(m.caption, "dreamy synthwave");
        assert_eq!(m.keyscale.as_deref(), Some("A minor"));
        // verbatim — denominator preserved (was collapsed to "/4" -> 4 before).
        assert_eq!(m.timesignature.as_deref(), Some("6/8"));
    }

    #[test]
    fn metadata_yaml_omits_absent_fields() {
        // No phase-1 metadata: only the always-known duration is emitted; NO
        // injected bpm: 120 / keyscale: C major / language: en / timesignature: 4.
        let m = Metadata { duration: 180, ..Metadata::default() };
        assert_eq!(AceTokenizer::metadata_yaml(&m), "duration: 180");
        // Partial: only the produced fields, alphabetical, no genres in the CoT.
        let m2 = Metadata {
            bpm: Some(90),
            duration: 120,
            keyscale: Some("A minor".into()),
            genres: Some("folk".into()),
            ..Metadata::default()
        };
        assert_eq!(AceTokenizer::metadata_yaml(&m2), "bpm: 90\nduration: 120\nkeyscale: A minor");
    }
}

const INSTRUCTION: &str = "Generate audio semantic tokens based on the given conditions:";

impl AceTokenizer {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AceError> {
        let hf = HfTokenizer::from_bytes(bytes).map_err(|e| AceError::Load(e.to_string()))?;
        let audio_base = hf
            .token_to_id("<|audio_code_0|>")
            .ok_or_else(|| AceError::Load("token <|audio_code_0|> not in vocab".into()))?;
        Ok(Self { hf, audio_base, eos: 151645, bos: 151643 })
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }
    pub fn bos(&self) -> u32 {
        self.bos
    }
    pub fn audio_base(&self) -> u32 {
        self.audio_base
    }

    pub fn code_to_id(&self, n: u32) -> u32 {
        self.audio_base + n
    }

    pub fn id_to_code(&self, id: u32) -> Option<u32> {
        if id >= self.audio_base && id < self.audio_base + NUM_AUDIO_CODES {
            Some(id - self.audio_base)
        } else {
            None
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AceError> {
        let enc = self
            .hf
            .encode(text, false)
            .map_err(|e| AceError::Other(format!("encode: {e}")))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, AceError> {
        self.hf
            .decode(ids, false)
            .map_err(|e| AceError::Other(format!("decode: {e}")))
    }

    // phase-2 <think> metadata: emit ONLY fields phase-1 actually produced
    // (Python _format_metadata_as_cot key set bpm/caption/duration/keyscale/
    // language/timesignature, alphabetical, NO genres, NO injected defaults).
    // duration is the always-known target.
    fn metadata_yaml(meta: &Metadata) -> String {
        let mut lines: Vec<String> = Vec::new();
        if let Some(b) = meta.bpm {
            lines.push(format!("bpm: {b}"));
        }
        if !meta.caption.is_empty() {
            lines.push(yaml_field("caption", &meta.caption));
        }
        lines.push(format!("duration: {}", meta.duration));
        if let Some(k) = &meta.keyscale {
            lines.push(yaml_field("keyscale", k));
        }
        if let Some(l) = &meta.language {
            lines.push(yaml_field("language", l));
        }
        if let Some(t) = &meta.timesignature {
            // Python: "4/4" → 4, "3/4" → 3 (знаменатель /4 отбрасывается), "6/8" как есть.
            lines.push(yaml_field("timesignature", t.strip_suffix("/4").unwrap_or(t)));
        }
        lines.join("\n")
    }

    pub fn build_codes_prompt(&self, caption: &str, lyrics: &str, meta: &Metadata) -> String {
        let yaml = Self::metadata_yaml(meta);
        format!(
            "<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n\
             <|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\n{yaml}\n</think>\n\n"
        )
    }

    pub fn build_codes_prompt_uncond(&self) -> String {
        format!(
            "<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n\
             <|im_start|>user\nNO USER INPUT<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    }

    pub fn build_cot_prompt(&self, caption: &str, lyrics: &str) -> String {
        format!(
            "<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n\
             <|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    pub fn build_cot_prompt_uncond(&self, lyrics: &str) -> String {
        format!(
            "<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n\
             <|im_start|>user\n# Lyric\n{lyrics}\n<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    pub fn think_end_id(&self) -> Option<u32> {
        let ids = self.encode("</think>").ok()?;
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    }

    pub fn parse_metadata(&self, cot_text: &str, base: &Metadata) -> Metadata {
        parse_metadata_text(cot_text, base)
    }
}
