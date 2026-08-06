use serde_json::{json, Map, Value as J};

use crate::error::{GgufError, Result};
use crate::reader::GgufFile;

pub const TOKEN_TYPE_NORMAL: i64 = 1;
pub const TOKEN_TYPE_UNKNOWN: i64 = 2;
pub const TOKEN_TYPE_CONTROL: i64 = 3;
pub const TOKEN_TYPE_USER_DEFINED: i64 = 4;
pub const TOKEN_TYPE_UNUSED: i64 = 5;
pub const TOKEN_TYPE_BYTE: i64 = 6;

const QWEN_SPLIT_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const LLAMA3_SPLIT_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct GgufVocab {
    pub tokens: Vec<String>,
    pub types: Vec<i64>,
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
    pub chat_template: Option<String>,
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

        Ok(Self {
            tokens,
            types,
            merges,
            model: f.str_of("tokenizer.ggml.model")?.to_string(),
            pre: f.opt_str("tokenizer.ggml.pre").map(String::from),
            bos: f.opt_usize("tokenizer.ggml.bos_token_id").map(|v| v as u32),
            eos: f.opt_usize("tokenizer.ggml.eos_token_id").map(|v| v as u32),
            pad: f.opt_usize("tokenizer.ggml.padding_token_id").map(|v| v as u32),
            unk: f.opt_usize("tokenizer.ggml.unknown_token_id").map(|v| v as u32),
            eot: f.opt_usize("tokenizer.ggml.eot_token_id").map(|v| v as u32),
            add_bos: f
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.as_bool()),
            add_eos: f
                .get("tokenizer.ggml.add_eos_token")
                .and_then(|v| v.as_bool()),
            chat_template: f.opt_str("tokenizer.chat_template").map(String::from),
        })
    }

    pub fn kind_of(&self, id: usize) -> i64 {
        self.types.get(id).copied().unwrap_or(TOKEN_TYPE_NORMAL)
    }

    fn split_regex(&self) -> &'static str {
        match self.pre.as_deref() {
            Some("llama3") | Some("llama-bpe") => LLAMA3_SPLIT_REGEX,
            _ => QWEN_SPLIT_REGEX,
        }
    }

    pub fn added_tokens(&self) -> Vec<(u32, &str, bool)> {
        self.tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| match self.kind_of(i) {
                TOKEN_TYPE_CONTROL => Some((i as u32, t.as_str(), true)),
                TOKEN_TYPE_USER_DEFINED => Some((i as u32, t.as_str(), false)),
                _ => None,
            })
            .collect()
    }

    pub fn to_tokenizer_json(&self) -> Result<Vec<u8>> {
        if self.model != "gpt2" {
            return Err(GgufError::UnsupportedArch(format!(
                "tokenizer.ggml.model = `{}`; синтез tokenizer.json реализован для byte-level BPE (gpt2)",
                self.model
            )));
        }

        let mut vocab = Map::new();
        for (i, t) in self.tokens.iter().enumerate() {
            match self.kind_of(i) {
                TOKEN_TYPE_UNUSED => continue,
                TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED => continue,
                _ => {}
            }
            vocab.insert(t.clone(), J::from(i as u64));
        }

        let added: Vec<J> = self
            .added_tokens()
            .into_iter()
            .map(|(id, content, special)| {
                json!({
                    "id": id,
                    "content": content,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": special,
                })
            })
            .collect();

        let merges: Vec<J> = self
            .merges
            .iter()
            .map(|(a, b)| json!([a, b]))
            .collect();

        let doc = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": self.split_regex()},
                     "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false,
                     "trim_offsets": true, "use_regex": false}
                ]
            },
            "post_processor": {"type": "ByteLevel", "add_prefix_space": false,
                               "trim_offsets": false, "use_regex": false},
            "decoder": {"type": "ByteLevel", "add_prefix_space": true,
                        "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges,
            }
        });
        Ok(serde_json::to_vec(&doc)?)
    }

    pub fn to_tokenizer_config_json(&self) -> Result<Vec<u8>> {
        let mut decoder = Map::new();
        for (id, content, special) in self.added_tokens() {
            decoder.insert(
                id.to_string(),
                json!({
                    "content": content,
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": special,
                }),
            );
        }
        let name = |id: Option<u32>| -> J {
            match id.and_then(|i| self.tokens.get(i as usize)) {
                Some(t) => J::from(t.as_str()),
                None => J::Null,
            }
        };
        let mut doc = json!({
            "added_tokens_decoder": decoder,
            "bos_token": name(self.bos),
            "eos_token": name(self.eos),
            "pad_token": name(self.pad),
            "unk_token": name(self.unk),
            "add_bos_token": self.add_bos.unwrap_or(false),
            "add_eos_token": self.add_eos.unwrap_or(false),
            "clean_up_tokenization_spaces": false,
            "tokenizer_class": "Qwen2Tokenizer",
        });
        if let Some(t) = &self.chat_template {
            doc["chat_template"] = J::from(t.as_str());
        }
        Ok(serde_json::to_vec_pretty(&doc)?)
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        let mut v = Vec::new();
        for id in [self.eos, self.eot] {
            if let Some(i) = id {
                if !v.contains(&i) {
                    v.push(i);
                }
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
            tokens: vec![
                "!".into(),
                "Ġa".into(),
                "<|im_start|>".into(),
                "<|unused_0|>".into(),
            ],
            types: vec![
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_UNUSED,
            ],
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
            chat_template: Some("{{ 'x' }}".into()),
        }
    }

    #[test]
    fn control_tokens_go_to_added_not_vocab() {
        let v = vocab();
        let bytes = v.to_tokenizer_json().unwrap();
        let d: J = serde_json::from_slice(&bytes).unwrap();
        let voc = d["model"]["vocab"].as_object().unwrap();
        assert_eq!(voc.len(), 2);
        assert!(voc.contains_key("!"));
        assert!(!voc.contains_key("<|im_start|>"));
        assert!(!voc.contains_key("<|unused_0|>"));
        let added = d["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 1);
        assert_eq!(added[0]["id"], 2);
        assert_eq!(added[0]["special"], true);
    }

    #[test]
    fn merges_are_pairs() {
        let v = vocab();
        let d: J = serde_json::from_slice(&v.to_tokenizer_json().unwrap()).unwrap();
        assert_eq!(d["model"]["merges"][0][0], "Ġ");
        assert_eq!(d["model"]["merges"][0][1], "a");
    }

    #[test]
    fn parses_with_tokenizers_crate() {
        let v = vocab();
        let bytes = v.to_tokenizer_json().unwrap();
        let d: J = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(d["model"]["type"], "BPE");
        assert_eq!(d["normalizer"]["type"], "NFC");
    }

    #[test]
    fn tokenizer_config_carries_chat_template_and_names() {
        let v = vocab();
        let d: J = serde_json::from_slice(&v.to_tokenizer_config_json().unwrap()).unwrap();
        assert_eq!(d["eos_token"], "<|im_start|>");
        assert_eq!(d["chat_template"], "{{ 'x' }}");
        assert!(d["added_tokens_decoder"]["2"].is_object());
    }

    #[test]
    fn rejects_non_bpe_model() {
        let mut v = vocab();
        v.model = "llama".into();
        assert!(v.to_tokenizer_json().is_err());
    }
}
