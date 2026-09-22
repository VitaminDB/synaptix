//! Токенизатор из метаданных GGUF → `tokenizer.json` / `tokenizer_config.json`
//! в формате HF `tokenizers` (движок читает именно его через synaptix-tokenizer).
//!
//! Три семейства `tokenizer.ggml.model`:
//! * `gpt2` — byte-level BPE (Qwen, Llama-3, GPT-2…): токены и merges как есть,
//!   регэксп пре-токенизации — по `tokenizer.ggml.pre` из таблицы llama.cpp
//!   (`llm_tokenizer_bpe`); неизвестное значение `pre` — ошибка, а не тихий
//!   дефолт.
//! * `llama` — SentencePiece (Llama-2, Mistral, Gemma-3): токены + scores, merges
//!   не хранятся — восстанавливаются так же, как `LlamaConverter` в transformers
//!   (все разбиения куска на два куска словаря, сортировка по score куска);
//!   модель BPE с `byte_fallback`, нормализатор `▁`.
//! * `gemma4` — SentencePiece-стиль с готовыми merges (Gemma-4).
//!
//! Именованные шаблоны (`tokenizer.chat_templates` + `tokenizer.chat_template.<имя>`)
//! уезжают в `tokenizer_config.json` списком, как их пишет transformers.

use serde_json::{json, Map, Value as J};

use crate::error::{GgufError, Result};
use crate::reader::{Array, GgufFile};

pub const TOKEN_TYPE_NORMAL: i64 = 1;
pub const TOKEN_TYPE_UNKNOWN: i64 = 2;
pub const TOKEN_TYPE_CONTROL: i64 = 3;
pub const TOKEN_TYPE_USER_DEFINED: i64 = 4;
pub const TOKEN_TYPE_UNUSED: i64 = 5;
pub const TOKEN_TYPE_BYTE: i64 = 6;

// Регэкспы пре-токенизации byte-level BPE — таблица `llm_tokenizer_bpe` из
// llama-vocab.cpp (в оригинальной форме tokenizer.json там, где llama.cpp её
// приводит в комментарии; `tokenizers` понимает `(?i:…)` и `(?!…)`).
const RE_QWEN2: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_QWEN35: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_LLAMA3: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
const RE_DIGIT: &str = r"\p{N}";
const RE_DIGITS3: &str = r"\p{N}{1,3}";
const RE_CHATGLM4: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_LINES: &str = r"[^\n]+|[\n]+";
const RE_FALCON_P: &str = r"[\p{P}\$\+<=>\^~\|`]+";
const RE_3DIGITS: &str = r"[0-9][0-9][0-9]";
const RE_DEFAULT_P: &str = r"[\p{P}\$\+<=>\^~\|]+";
const RE_DIGITS_PLUS: &str = r"\p{N}+";
const RE_MINICPM5: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_SEED_CODER: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_UFAKZEKA: &str = r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_TEKKEN: &str =
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_GPT4O: &str =
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_DEEPSEEK_CODER: [&str; 5] = [r"[\r\n]", r"\s?\p{L}+", r"\s?\p{P}+", "[一-龥ࠀ-一가-퟿]+", r"\p{N}"];
const RE_BLOOM: &str = r" ?[^(\s|.,!?…。，、।۔،)]+";

/// Регэкспы пре-токенизации по `tokenizer.ggml.pre` (последовательность
/// `Split`, как `regex_exprs` llama.cpp). `None` — значение неизвестно.
pub fn pre_regexes(pre: &str) -> Option<Vec<&'static str>> {
    Some(match pre {
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2" | "hunyuan"
        | "solar-open" => vec![RE_QWEN2],
        "qwen35" => vec![RE_QWEN35],
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" | "smaug-bpe" | "dbrx" => vec![RE_LLAMA3],
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de"
        | "a.x-4.0" | "mellum" | "modern-bert" | "jina-v1-en" | "jina-v2-code" | "roberta-bpe"
        | "exaone4" | "mpt" | "olmo" | "jais" | "trillion" | "granite-docling" => vec![RE_GPT2],
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva-7b"
        | "mellum2" => vec![RE_DIGIT, RE_GPT2],
        "glm4" | "chatglm-bpe" => vec![RE_CHATGLM4],
        "gemma4" | "granite-embed-multi-311m" | "sarvam-moe" => vec![RE_LINES],
        "falcon" => vec![RE_FALCON_P, RE_GPT2, RE_3DIGITS],
        "deepseek-coder" => RE_DEEPSEEK_CODER.to_vec(),
        "minicpm5" => vec![RE_DIGITS3, RE_MINICPM5],
        "seed-coder" => vec![RE_SEED_CODER],
        "ufakzeka" => vec![RE_UFAKZEKA],
        "grok-2" => vec![RE_QWEN2],
        "tekken" => vec![RE_TEKKEN],
        "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2" => vec![RE_GPT4O],
        "poro-chat" | "bloom" | "gpt3-finnish" => vec![RE_BLOOM],
        "viking" => vec![RE_BLOOM, RE_DIGIT],
        "laguna" => vec![RE_LINES, RE_QWEN2],
        "default" => vec![RE_DEFAULT_P, RE_GPT2, RE_DIGITS_PLUS, RE_3DIGITS],
        _ => return None,
    })
}

/// Кусок SentencePiece: пробелы пользовательских токенов GGUF (`"  "`) — это
/// `▁▁` в словаре HF.
fn spm_piece(t: &str) -> String {
    t.replace(' ', "\u{2581}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabKind {
    /// byte-level BPE (`gpt2`).
    ByteBpe,
    /// SentencePiece с scores, merges восстанавливаются (`llama`).
    Spm,
    /// SentencePiece-стиль с готовыми merges (`gemma4`).
    SpmBpe,
}

pub struct GgufVocab {
    pub tokens: Vec<String>,
    pub types: Vec<i64>,
    pub scores: Vec<f32>,
    pub merges: Vec<(String, String)>,
    pub model: String,
    pub pre: Option<String>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
    pub eot: Option<u32>,
    pub add_bos: Option<bool>,
    pub add_eos: Option<bool>,
    pub add_space_prefix: Option<bool>,
    pub chat_template: Option<String>,
    /// Именованные шаблоны `(имя, текст)`.
    pub chat_templates: Vec<(String, String)>,
}

impl GgufVocab {
    pub fn read(f: &GgufFile) -> Result<Self> {
        let tokens = f
            .require("tokenizer.ggml.tokens")?
            .as_array()
            .and_then(|a| a.as_str_slice())
            .ok_or_else(|| GgufError::WrongKeyType {
                key: "tokenizer.ggml.tokens".into(),
                expected: "array<string>",
                actual: "other",
            })?
            .to_vec();

        let types = f
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_i64_vec())
            .unwrap_or_else(|| vec![TOKEN_TYPE_NORMAL; tokens.len()]);

        let scores = match f.get("tokenizer.ggml.scores").and_then(|v| v.as_array()) {
            Some(Array::F32(v)) => v.clone(),
            Some(Array::F64(v)) => v.iter().map(|x| *x as f32).collect(),
            _ => Vec::new(),
        };

        let merges = f
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_str_slice())
            .map(|s| {
                s.iter()
                    .filter_map(|m| m.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut chat_templates = Vec::new();
        if let Some(names) = f
            .get("tokenizer.chat_templates")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_str_slice())
        {
            for n in names {
                if let Some(t) = f.opt_str(&format!("tokenizer.chat_template.{n}")) {
                    chat_templates.push((n.clone(), t.to_string()));
                }
            }
        }

        Ok(Self {
            tokens,
            types,
            scores,
            merges,
            model: f.str_of("tokenizer.ggml.model")?.to_string(),
            pre: f.opt_str("tokenizer.ggml.pre").map(String::from),
            bos: f.opt_usize("tokenizer.ggml.bos_token_id").map(|v| v as u32),
            eos: f.opt_usize("tokenizer.ggml.eos_token_id").map(|v| v as u32),
            pad: f.opt_usize("tokenizer.ggml.padding_token_id").map(|v| v as u32),
            unk: f.opt_usize("tokenizer.ggml.unknown_token_id").map(|v| v as u32),
            eot: f.opt_usize("tokenizer.ggml.eot_token_id").map(|v| v as u32),
            add_bos: f.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()),
            add_eos: f.get("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()),
            add_space_prefix: f.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.as_bool()),
            chat_template: f.opt_str("tokenizer.chat_template").map(String::from),
            chat_templates,
        })
    }

    pub fn kind(&self) -> Result<VocabKind> {
        match self.model.as_str() {
            "gpt2" => Ok(VocabKind::ByteBpe),
            "llama" => Ok(VocabKind::Spm),
            "gemma4" => Ok(VocabKind::SpmBpe),
            other => Err(GgufError::UnsupportedArch(format!(
                "tokenizer.ggml.model = `{other}`: поддержаны gpt2 (byte-level BPE), llama (SentencePiece), gemma4"
            ))),
        }
    }

    pub fn kind_of(&self, id: usize) -> i64 {
        self.types.get(id).copied().unwrap_or(TOKEN_TYPE_NORMAL)
    }

    fn split_regexes(&self) -> Result<Vec<&'static str>> {
        match self.pre.as_deref() {
            None => Ok(vec![RE_DEFAULT_P, RE_GPT2, RE_DIGITS_PLUS, RE_3DIGITS]),
            Some(p) => pre_regexes(p).ok_or_else(|| {
                GgufError::UnsupportedArch(format!(
                    "tokenizer.ggml.pre = `{p}` не в таблице пре-токенизаторов llama.cpp"
                ))
            }),
        }
    }

    /// Добавленные токены (управляющие — special, пользовательские — нет).
    /// Дубликаты содержимого схлопываются к первому id: у Gemma-3 `"\n"` есть
    /// и как обычный кусок 107, и как пользовательский токен 255837 — HF и
    /// llama.cpp дают 107, added-токен с тем же текстом обязан вести туда же.
    pub fn added_tokens(&self) -> Vec<(u32, &str, bool)> {
        use std::collections::HashMap;
        let mut first: HashMap<&str, usize> = HashMap::with_capacity(self.tokens.len());
        for (i, t) in self.tokens.iter().enumerate() {
            if self.kind_of(i) != TOKEN_TYPE_UNUSED {
                first.entry(t.as_str()).or_insert(i);
            }
        }
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        self.tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let special = match self.kind_of(i) {
                    TOKEN_TYPE_CONTROL => true,
                    TOKEN_TYPE_USER_DEFINED => false,
                    _ => return None,
                };
                if !seen.insert(t.as_str()) {
                    return None;
                }
                let id = first.get(t.as_str()).copied().unwrap_or(i);
                Some((id as u32, t.as_str(), special))
            })
            .collect()
    }

    fn token_name(&self, id: Option<u32>) -> J {
        match id.and_then(|i| self.tokens.get(i as usize)) {
            Some(t) => J::from(t.as_str()),
            None => J::Null,
        }
    }

    /// Merges для SentencePiece-словаря без merges — как `SentencePieceExtractor`
    /// в transformers: каждый кусок словаря, делящийся на два куска словаря,
    /// даёт merge; порядок — по score куска (убыв.), затем длины частей (убыв.).
    pub fn spm_merges(&self) -> Vec<(String, String)> {
        use std::collections::HashMap;
        // Только куски модели BPE: управляющие и пользовательские токены —
        // added_tokens, их BPE не видит; байтовые — есть в словаре, но не
        // сливаются.
        // Куски модели BPE: обычные, байтовые и пользовательские (у Gemma
        // `▁▁`, `\n` — именно пользовательские куски sentencepiece с scores);
        // управляющие и неиспользуемые не сливаются.
        let in_model = |i: usize| matches!(self.kind_of(i), TOKEN_TYPE_NORMAL | TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_BYTE | TOKEN_TYPE_USER_DEFINED);
        let pieces: Vec<String> = self.tokens.iter().map(|t| spm_piece(t)).collect();
        let mut vocab: HashMap<&str, (usize, f32)> = HashMap::with_capacity(self.tokens.len());
        // Score куска: у пользовательских в sentencepiece он 0 (выше любого
        // обычного, те отрицательные), а llama.cpp пишет для них −1000 —
        // восстанавливаем 0, иначе их merges уходят в середину списка и
        // `▁▁` проигрывает `▁l`.
        let score_of = |i: usize| -> f32 {
            if self.kind_of(i) == TOKEN_TYPE_USER_DEFINED { 0.0 } else { self.scores.get(i).copied().unwrap_or(0.0) }
        };
        for (i, t) in pieces.iter().enumerate() {
            if !in_model(i) {
                continue;
            }
            vocab.entry(t.as_str()).or_insert((i, score_of(i)));
        }
        let mut merges: Vec<(&str, &str, f32)> = Vec::new();
        for (i, piece) in pieces.iter().enumerate() {
            if !matches!(self.kind_of(i), TOKEN_TYPE_NORMAL | TOKEN_TYPE_USER_DEFINED) {
                continue;
            }
            let sc = score_of(i);
            for (cut, _) in piece.char_indices().skip(1) {
                let (l, r) = piece.split_at(cut);
                if vocab.contains_key(l) && vocab.contains_key(r) {
                    merges.push((l, r, sc));
                }
            }
        }
        merges.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.chars().count().cmp(&a.0.chars().count()))
                .then(b.1.chars().count().cmp(&a.1.chars().count()))
        });
        merges.into_iter().map(|(l, r, _)| (l.to_string(), r.to_string())).collect()
    }

    pub fn to_tokenizer_json(&self) -> Result<Vec<u8>> {
        let kind = self.kind()?;
        let spm = kind != VocabKind::ByteBpe;
        // Словарь модели — ВСЕ токены с их id (как в tokenizer.json HF): иначе
        // `tokenizers` переназначит id добавленным токенам, которых нет в
        // словаре. У SentencePiece пробелы в пользовательских кусках — `▁`.
        let mut vocab = Map::new();
        for (i, t) in self.tokens.iter().enumerate() {
            let piece = if spm { spm_piece(t) } else { t.clone() };
            vocab.entry(piece).or_insert(J::from(i as u64));
        }
        let added: Vec<J> = self
            .added_tokens()
            .into_iter()
            .map(|(id, content, special)| {
                let content = if spm { spm_piece(content) } else { content.to_string() };
                json!({
                    "id": id, "content": content, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": special,
                })
            })
            .collect();
        let merges_src: Vec<(String, String)> = match kind {
            VocabKind::Spm if self.merges.is_empty() => self.spm_merges(),
            _ => self.merges.clone(),
        };
        let merges: Vec<J> = merges_src.iter().map(|(a, b)| json!([a, b])).collect();

        // Добавление BOS — TemplateProcessing, как в tokenizer.json Llama-3/Gemma-3.
        let bos_post = |bos: &str, id: u32| -> J {
            json!({
                "type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [
                    {"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": bos, "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {bos: {"id": bos, "ids": [id], "tokens": [bos]}}
            })
        };
        let bos_name = self.bos.and_then(|i| self.tokens.get(i as usize).cloned());
        let want_bos = self.add_bos.unwrap_or(false);

        let doc = match kind {
            VocabKind::ByteBpe => {
                let splits: Vec<J> = self
                    .split_regexes()?
                    .into_iter()
                    .map(|re| json!({"type": "Split", "pattern": {"Regex": re}, "behavior": "Isolated", "invert": false}))
                    .collect();
                let mut pre = splits;
                pre.push(json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}));
                let byte_post = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
                let post = match (&bos_name, self.bos, want_bos) {
                    (Some(b), Some(id), true) => json!({"type": "Sequence", "processors": [byte_post, bos_post(b, id)]}),
                    _ => byte_post,
                };
                json!({
                    "version": "1.0", "truncation": null, "padding": null,
                    "added_tokens": added,
                    "normalizer": {"type": "NFC"},
                    "pre_tokenizer": {"type": "Sequence", "pretokenizers": pre},
                    "post_processor": post,
                    "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
                    "model": {
                        "type": "BPE", "dropout": null, "unk_token": null,
                        "continuing_subword_prefix": "", "end_of_word_suffix": "",
                        "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                        "vocab": vocab, "merges": merges,
                    }
                })
            }
            VocabKind::Spm | VocabKind::SpmBpe => {
                let space_prefix = self.add_space_prefix.unwrap_or(kind == VocabKind::Spm);
                let replace = json!({"type": "Replace", "pattern": {"String": " "}, "content": "▁"});
                let normalizer = if space_prefix {
                    json!({"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, replace]})
                } else {
                    replace
                };
                let pre_tokenizer = if space_prefix {
                    J::Null
                } else {
                    json!({"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false})
                };
                let mut decoders = vec![
                    json!({"type": "Replace", "pattern": {"String": "▁"}, "content": " "}),
                    json!({"type": "ByteFallback"}),
                    json!({"type": "Fuse"}),
                ];
                if space_prefix {
                    decoders.push(json!({"type": "Strip", "content": " ", "start": 1, "stop": 0}));
                }
                let post = match (&bos_name, self.bos, want_bos) {
                    (Some(b), Some(id), true) => bos_post(b, id),
                    _ => J::Null,
                };
                let unk = self.token_name(self.unk);
                json!({
                    "version": "1.0", "truncation": null, "padding": null,
                    "added_tokens": added,
                    "normalizer": normalizer,
                    "pre_tokenizer": pre_tokenizer,
                    "post_processor": post,
                    "decoder": {"type": "Sequence", "decoders": decoders},
                    "model": {
                        "type": "BPE", "dropout": null, "unk_token": unk,
                        "continuing_subword_prefix": null, "end_of_word_suffix": null,
                        "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
                        "vocab": vocab, "merges": merges,
                    }
                })
            }
        };
        Ok(serde_json::to_vec(&doc)?)
    }

    pub fn to_tokenizer_config_json(&self) -> Result<Vec<u8>> {
        let mut decoder = Map::new();
        for (id, content, special) in self.added_tokens() {
            decoder.insert(
                id.to_string(),
                json!({"content": content, "lstrip": false, "normalized": false, "rstrip": false, "single_word": false, "special": special}),
            );
        }
        let class = match self.kind().ok() {
            Some(VocabKind::ByteBpe) => "PreTrainedTokenizerFast",
            Some(VocabKind::Spm) => "LlamaTokenizer",
            Some(VocabKind::SpmBpe) => "GemmaTokenizer",
            None => "PreTrainedTokenizerFast",
        };
        let mut doc = json!({
            "added_tokens_decoder": decoder,
            "bos_token": self.token_name(self.bos),
            "eos_token": self.token_name(self.eos),
            "pad_token": self.token_name(self.pad),
            "unk_token": self.token_name(self.unk),
            "add_bos_token": self.add_bos.unwrap_or(false),
            "add_eos_token": self.add_eos.unwrap_or(false),
            "clean_up_tokenization_spaces": false,
            "tokenizer_class": class,
        });
        match (&self.chat_template, self.chat_templates.is_empty()) {
            (Some(t), true) => doc["chat_template"] = J::from(t.as_str()),
            (default, false) => {
                let mut list: Vec<J> = Vec::new();
                if let Some(t) = default {
                    list.push(json!({"name": "default", "template": t}));
                }
                for (n, t) in &self.chat_templates {
                    list.push(json!({"name": n, "template": t}));
                }
                doc["chat_template"] = J::Array(list);
            }
            (None, true) => {}
        }
        Ok(serde_json::to_vec_pretty(&doc)?)
    }

    /// Все токены конца генерации: `eos`/`eot` из метаданных плюс
    /// управляющие токены с известными именами — тот же список, по которому
    /// llama.cpp собирает `special_eog_ids` (`<|eot_id|>`, `<|eom_id|>`,
    /// `<|im_end|>`, `<end_of_turn>`, …). Без него Llama-3 после вызова
    /// инструмента (`<|eom_id|>`) не останавливается.
    pub fn eos_ids(&self) -> Vec<u32> {
        const EOG_NAMES: &[&str] = &[
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
            "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
            "<|end_of_text|>", "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
            "<｜end▁of▁sentence｜>", "[e~[",
        ];
        let mut v = Vec::new();
        let mut push = |i: u32| {
            if !v.contains(&i) {
                v.push(i);
            }
        };
        for id in [self.eos, self.eot].into_iter().flatten() {
            push(id);
        }
        for (i, t) in self.tokens.iter().enumerate() {
            if EOG_NAMES.contains(&t.as_str()) && self.types.get(i).is_none_or(|ty| *ty != TOKEN_TYPE_NORMAL) {
                push(i as u32);
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> GgufVocab {
        GgufVocab {
            tokens: vec!["!".into(), "Ġa".into(), "<|im_start|>".into(), "<|unused_0|>".into()],
            types: vec![TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_UNUSED],
            scores: Vec::new(),
            merges: vec![("Ġ".into(), "a".into())],
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            bos: Some(2),
            eos: Some(2),
            pad: None,
            unk: None,
            eot: None,
            add_bos: Some(false),
            add_eos: None,
            add_space_prefix: None,
            chat_template: Some("{{ 'x' }}".into()),
            chat_templates: Vec::new(),
        }
    }

    fn doc(v: &GgufVocab) -> J {
        serde_json::from_slice(&v.to_tokenizer_json().unwrap()).unwrap()
    }

    #[test]
    fn control_tokens_go_to_added_not_vocab() {
        let d = doc(&vocab());
        let voc = d["model"]["vocab"].as_object().unwrap();
        // Словарь — все токены с их id (иначе tokenizers переназначит id
        // добавленным), а добавленные перечислены отдельно.
        assert_eq!(voc.len(), 4);
        assert!(voc.contains_key("!"));
        assert_eq!(voc["<|im_start|>"], 2);
        let added = d["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 1);
        assert_eq!(added[0]["id"], 2);
        assert_eq!(added[0]["special"], true);
    }

    #[test]
    fn pre_table_picks_regex_or_fails_loudly() {
        let d = doc(&vocab());
        let re = d["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].as_str().unwrap();
        assert!(re.contains(r"\p{M}"), "qwen35 — регэксп с \\p{{M}}: {re}");
        let mut v = vocab();
        v.pre = Some("qwen2".into());
        let re2 = doc(&v)["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].as_str().unwrap().to_string();
        assert!(!re2.contains(r"\p{M}"));
        v.pre = Some("no-such-pre".into());
        assert!(v.to_tokenizer_json().is_err(), "неизвестный pre — ошибка, не тихий дефолт");
        assert_eq!(pre_regexes("deepseek-coder").unwrap().len(), 5);
        assert_eq!(pre_regexes("gemma4").unwrap(), vec![RE_LINES]);
    }

    #[test]
    fn bos_is_added_when_model_asks() {
        let mut v = vocab();
        v.pre = Some("llama-bpe".into());
        v.add_bos = Some(true);
        let d = doc(&v);
        assert_eq!(d["post_processor"]["type"], "Sequence");
        assert_eq!(d["post_processor"]["processors"][1]["type"], "TemplateProcessing");
        assert_eq!(d["post_processor"]["processors"][1]["special_tokens"]["<|im_start|>"]["ids"][0], 2);
    }

    #[test]
    fn parses_with_tokenizers_crate() {
        let bytes = vocab().to_tokenizer_json().unwrap();
        let d: J = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(d["model"]["type"], "BPE");
        assert_eq!(d["normalizer"]["type"], "NFC");
    }

    #[test]
    fn spm_merges_follow_transformers_order() {
        let v = GgufVocab {
            tokens: vec!["<unk>".into(), "<s>".into(), "▁".into(), "a".into(), "b".into(), "ab".into(), "▁ab".into(), "<0x41>".into()],
            types: vec![TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_BYTE],
            scores: vec![0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, 0.0],
            merges: Vec::new(),
            model: "llama".into(),
            pre: None,
            bos: Some(1),
            eos: None,
            pad: None,
            unk: Some(0),
            eot: None,
            add_bos: Some(true),
            add_eos: None,
            add_space_prefix: None,
            chat_template: None,
            chat_templates: Vec::new(),
        };
        let m = v.spm_merges();
        // «ab» (score −4) раньше «▁ab» (−5); у «▁ab» два разбиения: ▁+ab и ▁a? (нет ▁a) → одно.
        assert_eq!(m, vec![("a".to_string(), "b".to_string()), ("▁".to_string(), "ab".to_string())]);
        let d = doc(&v);
        assert_eq!(d["model"]["byte_fallback"], true);
        assert_eq!(d["model"]["unk_token"], "<unk>");
        assert_eq!(d["normalizer"]["type"], "Sequence", "llama: Prepend ▁ + Replace");
        assert!(d["pre_tokenizer"].is_null());
        assert_eq!(d["post_processor"]["type"], "TemplateProcessing");
        assert!(d["model"]["vocab"].as_object().unwrap().contains_key("<0x41>"), "байтовые токены остаются в словаре");
        // Реальный парсер tokenizers: должен собраться и токенизировать.
        let tk = tokenizers::Tokenizer::from_bytes(v.to_tokenizer_json().unwrap()).expect("tokenizers");
        let enc = tk.encode("ab", true).unwrap();
        assert_eq!(enc.get_ids(), &[1, 6], "<s> + ▁ab");
    }

    #[test]
    fn gemma4_style_uses_given_merges_without_prefix() {
        let mut v = vocab();
        v.model = "gemma4".into();
        v.pre = Some("gemma4".into());
        v.add_space_prefix = Some(false);
        let d = doc(&v);
        assert_eq!(d["normalizer"]["type"], "Replace");
        assert_eq!(d["pre_tokenizer"]["type"], "Split");
        assert_eq!(d["model"]["merges"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn tokenizer_config_carries_chat_templates() {
        let v = vocab();
        let d: J = serde_json::from_slice(&v.to_tokenizer_config_json().unwrap()).unwrap();
        assert_eq!(d["eos_token"], "<|im_start|>");
        assert_eq!(d["chat_template"], "{{ 'x' }}");
        assert!(d["added_tokens_decoder"]["2"].is_object());
        let mut v = vocab();
        v.chat_templates = vec![("tool_use".into(), "{{ 't' }}".into())];
        let d: J = serde_json::from_slice(&v.to_tokenizer_config_json().unwrap()).unwrap();
        let list = d["chat_template"].as_array().unwrap();
        assert_eq!(list[0]["name"], "default");
        assert_eq!(list[1]["name"], "tool_use");
    }

    #[test]
    fn rejects_unknown_model() {
        let mut v = vocab();
        v.model = "bert".into();
        assert!(v.to_tokenizer_json().is_err());
    }
}
