
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
    for line in body.lines() {
        let Some((k, v)) = line.trim().split_once(':') else { continue };
        let k = k.trim().to_lowercase();
        // Strip a leading list dash + surrounding YAML quotes (match old port).
        let v = v.trim().trim_start_matches('-').trim().trim_matches('\'').trim_matches('"').trim();
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

#[cfg(test)]
mod tests {
    use super::*;

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
            lines.push(format!("caption: {}", meta.caption));
        }
        lines.push(format!("duration: {}", meta.duration));
        if let Some(k) = &meta.keyscale {
            lines.push(format!("keyscale: {k}"));
        }
        if let Some(l) = &meta.language {
            lines.push(format!("language: {l}"));
        }
        if let Some(t) = &meta.timesignature {
            lines.push(format!("timesignature: {t}"));
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
