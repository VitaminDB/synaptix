//! High-level BGE-M3: токенизация батча (padding + attention-mask) → encoder →
//! CLS-pool → L2-norm. Dense-эмбеддинги [N, hidden].

use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::config::BgeConfig;
use crate::loader::{read_bundle_file, BgeWeights};
use crate::model::BgeEncoder;
use crate::BgeError;

const DEFAULT_MAX_TOKENS: usize = 512;

pub struct BgeM3 {
    encoder: BgeEncoder,
    tokenizer: HfTokenizer,
    cfg: BgeConfig,
    device: Device,
    max_tokens: usize,
}

impl BgeM3 {
    /// Загрузить из распакованного HF-снапшота (директория c `config.json`,
    /// `tokenizer.json`, `model.safetensors`).
    pub fn from_unpacked(
        dir: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let dir = dir.as_ref();
        let cfg = BgeConfig::from_json_bytes(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| BgeError::Load(format!("read config.json: {e}")))?,
        )?;
        let tokenizer = HfTokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| BgeError::Load(format!("tokenizer.json: {e}")))?;
        let weights =
            BgeWeights::load_safetensors(dir.join("model.safetensors"), *device, dtype)?;
        let encoder = BgeEncoder::build(&cfg, &weights)?;
        Ok(Self {
            encoder,
            tokenizer,
            cfg,
            device: *device,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    /// Загрузить из `.syn`-бандла (config/tokenizer — file-чанки, веса —
    /// `tensors:main`).
    pub fn from_syn(
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| BgeError::Bundle(e.to_string()))?;
        let cfg = BgeConfig::from_bundle(&bundle)?;
        let tok_bytes = read_bundle_file(&bundle, "tokenizer.json")?;
        let tokenizer = HfTokenizer::from_bytes(&tok_bytes)
            .map_err(|e| BgeError::Load(format!("tokenizer.json: {e}")))?;
        let weights = BgeWeights::load_bundle(path, *device, dtype)?;
        let encoder = BgeEncoder::build(&cfg, &weights)?;
        Ok(Self {
            encoder,
            tokenizer,
            cfg,
            device: *device,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    pub fn dim(&self) -> usize {
        self.cfg.hidden_size
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn set_max_tokens(&mut self, n: usize) {
        self.max_tokens = n.max(1);
    }

    pub fn config(&self) -> &BgeConfig {
        &self.cfg
    }

    /// Токенизировать батч (add_special, truncation `max_tokens`, right-pad до
    /// max-len батча `pad_token_id`) → (input_ids [N,S], attention_mask [N,S]).
    fn tokenize_batch(&self, texts: &[&str]) -> Result<(Tensor, Tensor, usize, usize), BgeError> {
        if texts.is_empty() {
            return Err(BgeError::Inference("encode: empty batch".into()));
        }
        let pad = self.cfg.pad_token_id;

        let mut all_ids: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        let mut max_len = 1usize;
        for t in texts {
            let enc = self
                .tokenizer
                .encode(t, true)
                .map_err(|e| BgeError::Inference(format!("tokenize: {e}")))?;
            let mut ids: Vec<i64> = enc.ids.iter().map(|&i| i as i64).collect();
            if ids.len() > self.max_tokens {
                ids.truncate(self.max_tokens);
            }
            if ids.is_empty() {
                ids.push(pad);
            }
            max_len = max_len.max(ids.len());
            all_ids.push(ids);
        }

        let n = texts.len();
        let mut ids_flat = vec![pad; n * max_len];
        let mut mask_flat = vec![0i64; n * max_len];
        for (bi, ids) in all_ids.iter().enumerate() {
            for (si, &id) in ids.iter().enumerate() {
                ids_flat[bi * max_len + si] = id;
                mask_flat[bi * max_len + si] = 1;
            }
        }

        let input_ids =
            Tensor::from_vec(ids_flat, vec![n, max_len], self.device).map_err(map_tensor_err)?;
        let attention_mask =
            Tensor::from_vec(mask_flat, vec![n, max_len], self.device).map_err(map_tensor_err)?;
        Ok((input_ids, attention_mask, n, max_len))
    }

    /// Энкодить батч → dense-эмбеддинги (CLS + L2-norm). `Vec<Vec<f32>>` [N][hidden].
    pub fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BgeError> {
        let (input_ids, attention_mask, n, _s) = self.tokenize_batch(texts)?;
        let last_hidden = self.encoder.forward(&input_ids, &attention_mask)?;
        let dense = self.encoder.dense_embed(&last_hidden)?;
        let flat = dense
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(map_tensor_err)?;
        let hidden = self.cfg.hidden_size;
        let mut out = Vec::with_capacity(n);
        for bi in 0..n {
            out.push(flat[bi * hidden..(bi + 1) * hidden].to_vec());
        }
        Ok(out)
    }

    /// last_hidden_state [B,S,hidden] напрямую (для гейта/диагностики).
    pub fn forward_last_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, BgeError> {
        self.encoder.forward(input_ids, attention_mask)
    }

    pub fn encoder(&self) -> &BgeEncoder {
        &self.encoder
    }
}

fn map_tensor_err(e: synaptix_core::error::SynaptixError) -> BgeError {
    BgeError::Tensor(e)
}
