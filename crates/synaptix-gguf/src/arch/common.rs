//! Общее для мапперов: чтение ключей `{arch}.…`, стандартные тензоры,
//! `generation_config.json`, сборка файлов плана.

use serde_json::{json, Value as J};

use crate::error::{GgufError, Result};
use crate::plan::{Component, ConversionPlan, MappedFile, MappedTensor, Producer, Transform};
use crate::reader::GgufFile;
use crate::tokenizer::GgufVocab;

/// Доступ к ключам метаданных с префиксом архитектуры.
pub struct Keys<'a> {
    pub f: &'a GgufFile,
    pub arch: String,
}

impl<'a> Keys<'a> {
    pub fn new(f: &'a GgufFile) -> Result<Self> {
        Ok(Self { f, arch: f.architecture()?.to_string() })
    }
    pub fn k(&self, s: &str) -> String {
        format!("{}.{s}", self.arch)
    }
    pub fn usize(&self, s: &str) -> Result<usize> {
        self.f.usize_of(&self.k(s))
    }
    pub fn opt_usize(&self, s: &str) -> Option<usize> {
        self.f.opt_usize(&self.k(s))
    }
    pub fn opt_f32(&self, s: &str) -> Option<f32> {
        self.f.opt_f32(&self.k(s))
    }
    pub fn opt_str(&self, s: &str) -> Option<&str> {
        self.f.opt_str(&self.k(s))
    }
    pub fn opt_bool_vec(&self, s: &str) -> Option<Vec<bool>> {
        match self.f.get(&self.k(s))?.as_array()? {
            crate::reader::Array::Bool(v) => Some(v.clone()),
            other => other.as_i64_vec().map(|v| v.into_iter().map(|x| x != 0).collect()),
        }
    }
    /// Скаляр или массив (Gemma-4 хранит `head_count_kv` и
    /// `feed_forward_length` по слоям): возвращает вектор длины `n`.
    pub fn usize_per_layer(&self, s: &str, n: usize) -> Option<Vec<usize>> {
        let v = self.f.get(&self.k(s))?;
        if let Some(a) = v.as_array() {
            let vals = a.as_i64_vec()?;
            return Some(vals.into_iter().map(|x| x.max(0) as usize).collect());
        }
        v.as_u64().map(|x| vec![x as usize; n])
    }
    pub fn vocab_size(&self) -> usize {
        self.f
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0)
    }
    pub fn has_tensor(&self, name: &str) -> bool {
        self.f.tensor(name).is_some()
    }
    /// Число строк тензора (первое HF-измерение).
    pub fn rows(&self, name: &str) -> Option<usize> {
        self.f.tensor(name).map(|t| t.hf_shape()[0])
    }
    pub fn tied(&self) -> bool {
        !self.has_tensor("output.weight")
    }
    /// Размер словаря по таблице эмбеддингов (Gemma-3 обрезает OOV-строки).
    pub fn embed_rows(&self) -> Option<usize> {
        self.rows("token_embd.weight")
    }
}

pub fn direct(hf: impl Into<String>, gguf: impl Into<String>) -> MappedTensor {
    MappedTensor::direct(hf, gguf)
}

pub fn direct_sub1(hf: impl Into<String>, gguf: impl Into<String>) -> MappedTensor {
    MappedTensor::direct(hf, gguf).with_transform(Transform::SubOne)
}

/// Стопка экспертов из частей `[E, Nᵢ, K]`.
pub fn stack_concat(hf: impl Into<String>, parts: Vec<String>) -> MappedTensor {
    MappedTensor {
        hf_name: hf.into(),
        producer: Producer::StackConcat { parts },
        shape: None,
        transform: Transform::None,
    }
}

/// Проверить, что все источники есть в файле; иначе понятная ошибка.
pub fn check_sources(f: &GgufFile, tensors: &[MappedTensor]) -> Result<()> {
    let missing: Vec<&str> = tensors
        .iter()
        .flat_map(|m| m.producer.sources().iter().map(|s| s.as_str()))
        .filter(|n| f.tensor(n).is_none())
        .collect();
    if let Some(first) = missing.first() {
        return Err(GgufError::BadTensor {
            name: first.to_string(),
            reason: format!("не найден в GGUF (всего пропущено {})", missing.len()),
        });
    }
    Ok(())
}

pub fn generation_config_json(f: &GgufFile, vocab: &GgufVocab) -> Result<Vec<u8>> {
    let mut doc = json!({ "do_sample": true, "eos_token_id": vocab.eos_ids() });
    if let Some(b) = vocab.bos {
        doc["bos_token_id"] = J::from(b);
    }
    if let Some(p) = vocab.pad {
        doc["pad_token_id"] = J::from(p);
    }
    if let Some(t) = f.opt_f32("general.sampling.temp") {
        doc["temperature"] = J::from(t);
    }
    if let Some(t) = f.opt_f32("general.sampling.top_p") {
        doc["top_p"] = J::from(t);
    }
    if let Some(t) = f.opt_usize("general.sampling.top_k") {
        doc["top_k"] = J::from(t);
    }
    Ok(serde_json::to_vec_pretty(&doc)?)
}

/// Стандартный набор файлов плана для текстовой модели.
pub fn standard_files(f: &GgufFile, vocab: &GgufVocab, config: Vec<u8>) -> Result<Vec<MappedFile>> {
    let mut files = vec![
        MappedFile { path: "config.json".into(), bytes: config },
        MappedFile { path: "tokenizer.json".into(), bytes: vocab.to_tokenizer_json()? },
        MappedFile { path: "tokenizer_config.json".into(), bytes: vocab.to_tokenizer_config_json()? },
        MappedFile { path: "generation_config.json".into(), bytes: generation_config_json(f, vocab)? },
    ];
    if let Some(t) = &vocab.chat_template {
        files.push(MappedFile { path: "chat_template.jinja".into(), bytes: t.clone().into_bytes() });
    }
    Ok(files)
}

pub fn plan(bundle_id: &str, arch: &str, tensors: Vec<MappedTensor>, files: Vec<MappedFile>) -> ConversionPlan {
    ConversionPlan {
        bundle_id: bundle_id.to_string(),
        arch: arch.to_string(),
        components: vec![Component { name: "main".into(), tensors }],
        files,
    }
}

/// Множители частот RoPE Llama-3 (как `generate_extra_tensors` конвертера).
pub fn llama3_rope_factors(theta: f32, dim: usize, factor: f32, low: f32, high: f32, old_ctx: f32) -> Vec<f32> {
    let low_wl = old_ctx / low;
    let high_wl = old_ctx / high;
    (0..dim / 2)
        .map(|i| {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
            let wavelen = 2.0 * std::f32::consts::PI / freq;
            if wavelen < high_wl {
                1.0
            } else if wavelen > low_wl {
                factor
            } else {
                let smooth = (old_ctx / wavelen - low) / (high - low);
                1.0 / ((1.0 - smooth) / factor + smooth)
            }
        })
        .collect()
}

/// Прочитать `rope_freqs.weight` (F32) — множители частот.
pub fn rope_freqs(f: &GgufFile) -> Result<Option<Vec<f32>>> {
    let Some(info) = f.tensor("rope_freqs.weight") else { return Ok(None) };
    let n = info.elem_count();
    let mut v = vec![0f32; n];
    crate::dequant::dequantize(info.ty, f.tensor_bytes(info)?, n, &mut v)?;
    Ok(Some(v))
}
