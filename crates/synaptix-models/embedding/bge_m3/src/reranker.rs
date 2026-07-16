//! BGE-reranker-v2-m3 (cross-encoder): тот же XLM-RoBERTa-энкодер ([`BgeEncoder`]) +
//! `RobertaClassificationHead` (dense → tanh → out_proj→1) на CLS-токене.
//!
//! Источник истины: HF `BAAI/bge-reranker-v2-m3` (`XLMRobertaForSequenceClassification`).
//! Вход — пара (query, passage) → `<s> q </s></s> p </s>` (encode_pair) → encoder →
//! last_hidden[:,0] → голова → скаляр-релевантность (raw logit; sigmoid опционально).

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use synaptix_bundle::Bundle;

use crate::config::BgeConfig;
use crate::loader::{read_bundle_file, BgeWeights};
use crate::model::BgeEncoder;
use crate::BgeError;

/// `RobertaClassificationHead`: dense(H→H) → tanh → out_proj(H→1). Dropout = no-op (eval).
struct RerankerHead {
    dense_w: Tensor,
    dense_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
}

impl RerankerHead {
    fn load(w: &BgeWeights) -> Result<Self, BgeError> {
        Ok(Self {
            dense_w: w.get("classifier.dense.weight")?.clone(),
            dense_b: w.get("classifier.dense.bias")?.clone(),
            out_w: w.get("classifier.out_proj.weight")?.clone(),
            out_b: w.get("classifier.out_proj.bias")?.clone(),
        })
    }

    /// CLS `[N, H]` → логиты `[N]`.
    fn forward(&self, cls: &Tensor) -> Result<Tensor, BgeError> {
        let h = cls.linear(&self.dense_w)?.broadcast_add(&self.dense_b)?.tanh()?;
        let logit = h.linear(&self.out_w)?.broadcast_add(&self.out_b)?; // [N,1]
        let n = logit.dims()[0];
        Ok(logit.reshape(vec![n])?)
    }
}

pub struct BgeReranker {
    encoder: BgeEncoder,
    head: RerankerHead,
    tokenizer: HfTokenizer,
    cfg: BgeConfig,
    max_tokens: usize,
    device: Device,
}

impl BgeReranker {
    /// Из распакованного HF-снапшота (`config.json`, `tokenizer.json`, `model.safetensors`
    /// с префиксом `roberta.`).
    pub fn from_unpacked(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let dir = dir.as_ref();
        let cfg_bytes = std::fs::read(dir.join("config.json"))
            .map_err(|e| BgeError::Load(format!("config.json: {e}")))?;
        let cfg = BgeConfig::from_json_bytes(&cfg_bytes)?;
        let tokenizer = HfTokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| BgeError::Load(format!("tokenizer.json: {e}")))?;
        let weights =
            BgeWeights::load_safetensors_strip(dir.join("model.safetensors"), device, dtype, "roberta.")?;
        let encoder = BgeEncoder::build(&cfg, &weights)?;
        let head = RerankerHead::load(&weights)?;
        Ok(Self { encoder, head, tokenizer, cfg, max_tokens: 512, device })
    }

    /// Из `.syn`-бандла (config/tokenizer — file-чанки, веса — `tensors:main` с `roberta.`).
    pub fn from_syn(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| BgeError::Bundle(e.to_string()))?;
        let cfg = BgeConfig::from_json_bytes(&read_bundle_file(&bundle, "config.json")?)?;
        let tokenizer = HfTokenizer::from_bytes(&read_bundle_file(&bundle, "tokenizer.json")?)
            .map_err(|e| BgeError::Load(format!("tokenizer.json: {e}")))?;
        let weights = BgeWeights::load_bundle_strip(path, device, dtype, "roberta.")?;
        let encoder = BgeEncoder::build(&cfg, &weights)?;
        let head = RerankerHead::load(&weights)?;
        Ok(Self { encoder, head, tokenizer, cfg, max_tokens: 512, device })
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn set_max_tokens(&mut self, n: usize) {
        self.max_tokens = n;
    }

    /// Токенизация пар (query, passage) → (input_ids [N,S], attention_mask [N,S]).
    fn tokenize_pairs(&self, pairs: &[(&str, &str)]) -> Result<(Tensor, Tensor), BgeError> {
        if pairs.is_empty() {
            return Err(BgeError::Inference("rerank: empty batch".into()));
        }
        let pad = self.cfg.pad_token_id;
        let mut all: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        let mut max_len = 1usize;
        for (q, p) in pairs {
            let enc = self
                .tokenizer
                .encode_pair(q, p, true)
                .map_err(|e| BgeError::Inference(format!("tokenize pair: {e}")))?;
            let mut ids: Vec<i64> = enc.ids.iter().map(|&i| i as i64).collect();
            if ids.len() > self.max_tokens {
                ids.truncate(self.max_tokens);
            }
            if ids.is_empty() {
                ids.push(pad);
            }
            max_len = max_len.max(ids.len());
            all.push(ids);
        }
        let n = pairs.len();
        let mut ids_flat = vec![pad; n * max_len];
        let mut mask_flat = vec![0i64; n * max_len];
        for (bi, ids) in all.iter().enumerate() {
            for (si, &id) in ids.iter().enumerate() {
                ids_flat[bi * max_len + si] = id;
                mask_flat[bi * max_len + si] = 1;
            }
        }
        let input_ids = Tensor::from_vec(ids_flat, vec![n, max_len], self.device)?;
        let attention_mask = Tensor::from_vec(mask_flat, vec![n, max_len], self.device)?;
        Ok((input_ids, attention_mask))
    }

    /// Скоры релевантности для пар (raw logits). `[N]`.
    pub fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, BgeError> {
        let (input_ids, attention_mask) = self.tokenize_pairs(pairs)?;
        let last_hidden = self.encoder.forward(&input_ids, &attention_mask)?; // (N,S,H)
        let h = last_hidden.dims()[2];
        let n = last_hidden.dims()[0];
        let cls = last_hidden.narrow(1, 0, 1)?.contiguous()?.reshape(vec![n, h])?;
        let logits = self.head.forward(&cls)?;
        Ok(logits.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }

    /// Реранк: скоринг (query, doc) для всех docs → сорт по убыванию → top_k (idx, score).
    pub fn rerank(
        &self,
        query: &str,
        docs: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, BgeError> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<(&str, &str)> = docs.iter().map(|d| (query, *d)).collect();
        let scores = self.score_pairs(&pairs)?;
        let mut idx: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(top_k.min(docs.len()));
        Ok(idx)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
