//! Нативные модули Whisper: attention (self + cross), энкодер-слой,
//! энкодер. Декодер и KV-cache добавляются в Фазе 2.
//!
//! Соглашения Whisper, реализованные здесь:
//! - `k_proj` без bias, `q/v/out_proj` — с bias;
//! - точный erf-`gelu` в conv-stem и FFN;
//! - обучаемые позиционные эмбеддинги (срез + add), не RoPE;
//! - scale внимания = `head_dim^-0.5`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::{Linear, Module};
use synaptix_ops::activation::gelu_exact;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d;
use synaptix_ops::mask::causal_mask;
use synaptix_ops::norm::layer_norm;

use crate::loader::{dec_layer, enc_layer, WhisperWeights};
use crate::Result;

/// Layer-norm параметры (gain + bias).
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), self.eps)?)
    }
}

/// Multi-head attention Whisper. Используется и как self-, и как cross-attention:
/// query берётся из `x`, key/value — из `kv_source` (для self-attn это тот же `x`).
pub struct WhisperAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl WhisperAttention {
    /// q-проекция с разбивкой на головы → `[B, H, S, Dh]`.
    fn q_heads(&self, x: &Tensor) -> Result<Tensor> {
        split_heads(&self.q.forward(x)?, self.num_heads, self.head_dim)
    }

    /// k/v-проекции с разбивкой на головы → (`[B, H, S, Dh]`, `[B, H, S, Dh]`).
    fn kv_heads(&self, src: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = split_heads(&self.k.forward(src)?, self.num_heads, self.head_dim)?;
        let v = split_heads(&self.v.forward(src)?, self.num_heads, self.head_dim)?;
        Ok((k, v))
    }

    /// scaled-dot-attention + merge + out-проекция.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let attn = scaled_dot_attention(q, k, v, self.scale, mask)?;
        let attn = merge_heads(&attn)?;
        Ok(self.out.forward(&attn)?)
    }

    fn forward(&self, x: &Tensor, kv_source: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let q = self.q_heads(x)?;
        let (k, v) = self.kv_heads(kv_source)?;
        self.attend(&q, &k, &v, mask)
    }

    /// Полное (бидиректное, без маски) self-attn энкодера. На CUDA F16/BF16 — через
    /// tensor-core FA-4 (`flash_attention`, hd=64 поддержан), что устраняет F32 SIMT
    /// GEMM + материализацию [S,S] scores в scaled_dot (узкое место профиля энкодера).
    /// На CPU / прочих dtype — обычный scaled_dot (bit-exact reference-путь).
    fn forward_full(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_heads(x)?;
        let (k, v) = self.kv_heads(x)?;
        if matches!(x.device(), Device::Cuda(_)) && matches!(x.dtype(), DType::F16 | DType::BF16) {
            let attn = q.flash_attention(&k, &v, self.scale, false)?;
            let attn = merge_heads(&attn)?;
            Ok(self.out.forward(&attn)?)
        } else {
            self.attend(&q, &k, &v, None)
        }
    }

    /// Device-резидентный self-attn шаг (CUDA-graph): пишет K/V в preallocated
    /// буфер по `pos_dev`, читает `[0..tcache_dev]` через flash_attention_dev.
    /// `h` — `[1, 1, d_model]`.
    fn decode_self_dev(
        &self,
        h: &Tensor,
        k_buf: &mut Tensor,
        v_buf: &mut Tensor,
        pos_dev: &Tensor,
        tcache_dev: &Tensor,
    ) -> Result<Tensor> {
        let q = self.q_heads(h)?; // [1, nh, 1, hd]
        let (k, v) = self.kv_heads(h)?;
        k_buf.kv_append_dev(&k, pos_dev)?;
        v_buf.kv_append_dev(&v, pos_dev)?;
        let attn = q.flash_attention_dev(k_buf, v_buf, tcache_dev, self.scale, true)?;
        let attn = merge_heads(&attn)?;
        Ok(self.out.forward(&attn)?)
    }

    /// Device-резидентный cross-attn шаг (CUDA-graph): фьюзед F16 flash-decode по
    /// фикс. cross-K/V (non-causal, активная длина = `cross_len` device-резидент).
    /// Заменяет `decode_cross`/`scaled_dot` (F32-upcast + F32 SIMT-GEMM фикс. K/V
    /// на каждом шаге) — узкое место decode-профиля.
    fn decode_cross_dev(
        &self,
        h: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        cross_len: &Tensor,
    ) -> Result<Tensor> {
        let q = self.q_heads(h)?;
        let attn = q.flash_attention_dev(cross_k, cross_v, cross_len, self.scale, false)?;
        let attn = merge_heads(&attn)?;
        Ok(self.out.forward(&attn)?)
    }
}

/// `[B, S, H*Dh]` → `[B, H, S, Dh]`.
fn split_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let d = x.dims();
    let (b, s) = (d[0], d[1]);
    Ok(x
        .reshape(vec![b, s, num_heads, head_dim])?
        .permute(vec![0, 2, 1, 3])?
        .contiguous()?)
}

/// `[B, H, S, Dh]` → `[B, S, H*Dh]`.
fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let d = x.dims();
    let (b, h, s, dh) = (d[0], d[1], d[2], d[3]);
    Ok(x
        .permute(vec![0, 2, 1, 3])?
        .contiguous()?
        .reshape(vec![b, s, h * dh])?)
}

pub struct EncoderLayer {
    self_attn_ln: LayerNorm,
    self_attn: WhisperAttention,
    final_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.self_attn_ln.forward(x)?;
        let h = self.self_attn.forward_full(&h)?;
        let x = residual.add(&h)?;

        let residual = &x;
        let h = self.final_ln.forward(&x)?;
        let h = self.fc2.forward(&gelu_exact(&self.fc1.forward(&h)?)?)?;
        Ok(residual.add(&h)?)
    }
}

pub struct WhisperEncoder {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    embed_positions: Tensor,
    layers: Vec<EncoderLayer>,
    final_ln: LayerNorm,
}

impl WhisperEncoder {
    /// mel `[B, num_mel_bins, T]` → `[B, T/2, d_model]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = gelu_exact(&conv1d(mel, &self.conv1_w, Some(&self.conv1_b), 1, 1)?)?;
        let x = gelu_exact(&conv1d(&x, &self.conv2_w, Some(&self.conv2_b), 2, 1)?)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let seq = x.dims()[1];
        let pos = self.embed_positions.narrow(0, 0, seq)?.unsqueeze(0)?;
        let mut h = x.broadcast_add(&pos)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        self.final_ln.forward(&h)
    }

    pub fn load(w: &WhisperWeights) -> Result<Self> {
        let cfg = &w.config;
        let heads = cfg.encoder_attention_heads;
        let hd = cfg.encoder_head_dim();
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(EncoderLayer {
                self_attn_ln: load_layer_norm(w, &enc_layer(i, "self_attn_layer_norm"))?,
                self_attn: load_attention(w, &enc_layer(i, "self_attn"), heads, hd)?,
                final_ln: load_layer_norm(w, &enc_layer(i, "final_layer_norm"))?,
                fc1: load_linear(w, &enc_layer(i, "fc1"), true)?,
                fc2: load_linear(w, &enc_layer(i, "fc2"), true)?,
            });
        }
        Ok(Self {
            conv1_w: w.get("model.encoder.conv1.weight")?,
            conv1_b: w.get("model.encoder.conv1.bias")?,
            conv2_w: w.get("model.encoder.conv2.weight")?,
            conv2_b: w.get("model.encoder.conv2.bias")?,
            embed_positions: w.get("model.encoder.embed_positions.weight")?,
            layers,
            final_ln: load_layer_norm(w, "model.encoder.layer_norm")?,
        })
    }
}

pub struct DecoderLayer {
    self_attn_ln: LayerNorm,
    self_attn: WhisperAttention,
    encoder_attn_ln: LayerNorm,
    encoder_attn: WhisperAttention,
    final_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl DecoderLayer {
    fn forward(&self, x: &Tensor, enc_out: &Tensor, causal: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.self_attn_ln.forward(x)?;
        let h = self.self_attn.forward(&h, &h, Some(causal))?;
        let x = residual.add(&h)?;

        let residual = &x;
        let h = self.encoder_attn_ln.forward(&x)?;
        let h = self.encoder_attn.forward(&h, enc_out, None)?;
        let x = residual.add(&h)?;

        let residual = &x;
        let h = self.final_ln.forward(&x)?;
        let h = self.fc2.forward(&gelu_exact(&self.fc1.forward(&h)?)?)?;
        Ok(residual.add(&h)?)
    }

    /// Device-резидентный decode-шаг слоя (CUDA-graph): self-attn через
    /// preallocated KV-буфер + device-позицию/длину, cross-attn по фикс. K/V.
    #[allow(clippy::too_many_arguments)]
    fn forward_decode_dev(
        &self,
        x: &Tensor,
        k_buf: &mut Tensor,
        v_buf: &mut Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        cross_len: &Tensor,
        pos_dev: &Tensor,
        tcache_dev: &Tensor,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.self_attn_ln.forward(x)?;
        let a = self.self_attn.decode_self_dev(&h, k_buf, v_buf, pos_dev, tcache_dev)?;
        let x = residual.add(&a)?;

        let residual = &x;
        let h = self.encoder_attn_ln.forward(&x)?;
        let c = self.encoder_attn.decode_cross_dev(&h, cross_k, cross_v, cross_len)?;
        let x = residual.add(&c)?;

        let residual = &x;
        let h = self.final_ln.forward(&x)?;
        let h = self.fc2.forward(&gelu_exact(&self.fc1.forward(&h)?)?)?;
        Ok(residual.add(&h)?)
    }
}

/// Per-layer KV-кэш декодера. self-K/V растут по шагам (cat вдоль позиций),
/// cross-K/V (`encoder_attn`) считаются один раз из энкодер-выхода на сегмент.
pub struct LayerKv {
    self_k: Option<Tensor>,
    self_v: Option<Tensor>,
    cross_k: Tensor,
    cross_v: Tensor,
}

pub struct DecoderCache {
    layers: Vec<LayerKv>,
}

/// Device-резидентное состояние decode-шага для CUDA-graph (стабильные адреса
/// буферов — обновляются HtoD перед replay, граф ссылается на них).
pub struct WhisperDecodeState {
    pub input: Tensor,      // U32[1,1] токен
    pub pos_dev: Tensor,    // U32[1] позиция (self-attn слот + pos-embedding)
    pub tcache_dev: Tensor, // U32[1] активная длина self-KV
    pub logits: Tensor,     // [1, vocab] выход
}

impl WhisperDecodeState {
    pub fn new(device: Device, dtype: DType, vocab: usize) -> Result<Self> {
        Ok(Self {
            input: Tensor::from_vec(vec![0u32], vec![1usize, 1], device)?,
            pos_dev: Tensor::from_vec(vec![0u32], vec![1usize], device)?,
            tcache_dev: Tensor::from_vec(vec![0u32], vec![1usize], device)?,
            logits: Tensor::zeros(vec![1, vocab], dtype, device)?,
        })
    }

    /// HtoD обновление токена/позиции (длина self-KV = pos+1) без реаллокации.
    pub fn update(&mut self, token: u32, pos: u32) -> Result<()> {
        self.input.write_host_u32(&[token])?;
        self.pos_dev.write_host_u32(&[pos])?;
        self.tcache_dev.write_host_u32(&[pos + 1])?;
        Ok(())
    }
}

/// Per-сегмент device KV-кэш для graph-decode: self-K/V preallocated на
/// max_target (пишется по позиции), cross-K/V — фикс. проекции энкодер-выхода.
pub struct WhisperDevCache {
    pub self_k: Vec<Tensor>,
    pub self_v: Vec<Tensor>,
    pub cross_k: Vec<Tensor>,
    pub cross_v: Vec<Tensor>,
    pub cross_len: Tensor,
}

pub struct WhisperDecoder {
    embed_tokens: Tensor,
    embed_positions: Tensor,
    layers: Vec<DecoderLayer>,
    final_ln: LayerNorm,
    lm_head: Linear,
}

impl WhisperDecoder {
    /// Teacher-forced прогон префикса: `token_ids` (batch=1) + энкодер-выход
    /// `[1, T_enc, d_model]` → логиты `[1, S, vocab]`.
    pub fn forward_prefix(&self, token_ids: &[u32], enc_out: &Tensor) -> Result<Tensor> {
        let device = enc_out.device();
        let seq = token_ids.len();
        let h = self.embed(token_ids, 0, device)?;
        let causal = causal_mask(seq, device)?;
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h, enc_out, &causal)?;
        }
        let h = self.final_ln.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }

    fn d_model(&self) -> usize {
        self.embed_tokens.dims()[1]
    }

    /// Device KV-кэш для graph-decode: cross-K/V из энкодер-выхода (фикс. на
    /// сегмент), self-K/V — preallocated нули `[1, nh, max_target, hd]`.
    pub fn make_dev_cache(
        &self,
        enc_out: &Tensor,
        max_target: usize,
        device: Device,
        dtype: DType,
    ) -> Result<WhisperDevCache> {
        let n = self.layers.len();
        let mut self_k = Vec::with_capacity(n);
        let mut self_v = Vec::with_capacity(n);
        let mut cross_k = Vec::with_capacity(n);
        let mut cross_v = Vec::with_capacity(n);
        for layer in &self.layers {
            let nh = layer.encoder_attn.num_heads;
            let hd = layer.encoder_attn.head_dim;
            let (ck, cv) = layer.encoder_attn.kv_heads(enc_out)?;
            cross_k.push(ck);
            cross_v.push(cv);
            self_k.push(Tensor::zeros(vec![1, nh, max_target, hd], dtype, device)?);
            self_v.push(Tensor::zeros(vec![1, nh, max_target, hd], dtype, device)?);
        }
        let cross_len = Tensor::from_vec(vec![enc_out.dims()[1] as u32], vec![1usize], device)?;
        Ok(WhisperDevCache { self_k, self_v, cross_k, cross_v, cross_len })
    }

    /// Device-резидентный decode-шаг (T=1) для CUDA-graph: embed_gather токена и
    /// позиции, слои через `forward_decode_dev`, логиты в `state.logits`.
    pub fn forward_decode_dev(
        &self,
        state: &mut WhisperDecodeState,
        cache: &mut WhisperDevCache,
    ) -> Result<()> {
        let ids = state.input.reshape(vec![1usize])?;
        let tok = self.embed_tokens.embed_gather(&ids)?; // [1, d]
        let pos = self.embed_positions.embed_gather(&state.pos_dev)?; // [1, d]
        let mut hidden = tok.add(&pos)?.reshape(vec![1usize, 1, self.d_model()])?;
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_decode_dev(
                &hidden,
                &mut cache.self_k[idx],
                &mut cache.self_v[idx],
                &cache.cross_k[idx],
                &cache.cross_v[idx],
                &cache.cross_len,
                &state.pos_dev,
                &state.tcache_dev,
            )?;
        }
        let normed = self.final_ln.forward(&hidden)?;
        let last = normed.narrow(1, 0, 1)?.squeeze(1)?; // [1, d]
        let logits = self.lm_head.forward(&last)?; // [1, vocab]
        state.logits.copy_from(&logits)?;
        Ok(())
    }

    /// Инициализировать KV-кэш для сегмента: cross-K/V из энкодер-выхода
    /// (считаются один раз), self-K/V пустые.
    pub fn init_cache(&self, enc_out: &Tensor) -> Result<DecoderCache> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (cross_k, cross_v) = layer.encoder_attn.kv_heads(enc_out)?;
            layers.push(LayerKv { self_k: None, self_v: None, cross_k, cross_v });
        }
        Ok(DecoderCache { layers })
    }

    /// Инкрементальный шаг: один токен на позиции `pos` → логиты `[1, 1, vocab]`.
    /// self-attn идёт против накопленного KV (без causal-маски: единственный
    /// query видит все прошлые ключи), cross-attn — против cross-K/V из кэша.
    pub fn decode_step(
        &self,
        token_id: u32,
        pos: usize,
        cache: &mut DecoderCache,
    ) -> Result<Tensor> {
        let device = cache.layers[0].cross_k.device();
        let mut h = self.embed(&[token_id], pos, device)?; // [1,1,d]
        for (layer, kv) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let residual = &h;
            let hn = layer.self_attn_ln.forward(&h)?;
            let q = layer.self_attn.q_heads(&hn)?;
            let (k, v) = layer.self_attn.kv_heads(&hn)?;
            let full_k = match &kv.self_k {
                Some(prev) => Tensor::cat(&[prev, &k], 2)?,
                None => k,
            };
            let full_v = match &kv.self_v {
                Some(prev) => Tensor::cat(&[prev, &v], 2)?,
                None => v,
            };
            let a = layer.self_attn.attend(&q, &full_k, &full_v, None)?;
            kv.self_k = Some(full_k);
            kv.self_v = Some(full_v);
            let x = residual.add(&a)?;

            let residual = &x;
            let hn = layer.encoder_attn_ln.forward(&x)?;
            let q = layer.encoder_attn.q_heads(&hn)?;
            let c = layer.encoder_attn.attend(&q, &kv.cross_k, &kv.cross_v, None)?;
            let x = residual.add(&c)?;

            let residual = &x;
            let hn = layer.final_ln.forward(&x)?;
            let m = layer.fc2.forward(&gelu_exact(&layer.fc1.forward(&hn)?)?)?;
            h = residual.add(&m)?;
        }
        let h = self.final_ln.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }

    /// Эмбеддинг токенов + позиционный эмбеддинг (срез `[past_len..past_len+S]`).
    fn embed(&self, token_ids: &[u32], past_len: usize, device: Device) -> Result<Tensor> {
        let seq = token_ids.len();
        // Один токен (decode-шаг) → narrow вместо index_select: последний на CUDA
        // делает host-roundtrip, что доминировало в decode-loop.
        let tok = if seq == 1 {
            self.embed_tokens.narrow(0, token_ids[0] as usize, 1)? // [1, d_model]
        } else {
            let idx = Tensor::from_vec(token_ids.to_vec(), (seq,), device)?;
            self.embed_tokens.index_select(0, &idx)? // [S, d_model]
        };
        let pos = self.embed_positions.narrow(0, past_len, seq)?;
        let h = tok.add(&pos)?;
        Ok(h.unsqueeze(0)?) // [1, S, d_model]
    }

    pub fn load(w: &WhisperWeights) -> Result<Self> {
        let cfg = &w.config;
        let heads = cfg.decoder_attention_heads;
        let hd = cfg.decoder_head_dim();
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            layers.push(DecoderLayer {
                self_attn_ln: load_layer_norm(w, &dec_layer(i, "self_attn_layer_norm"))?,
                self_attn: load_attention(w, &dec_layer(i, "self_attn"), heads, hd)?,
                encoder_attn_ln: load_layer_norm(w, &dec_layer(i, "encoder_attn_layer_norm"))?,
                encoder_attn: load_attention(w, &dec_layer(i, "encoder_attn"), heads, hd)?,
                final_ln: load_layer_norm(w, &dec_layer(i, "final_layer_norm"))?,
                fc1: load_linear(w, &dec_layer(i, "fc1"), true)?,
                fc2: load_linear(w, &dec_layer(i, "fc2"), true)?,
            });
        }
        let embed_tokens = w.get("model.decoder.embed_tokens.weight")?;
        // tied lm_head: проекция = embed_tokens (logits = h @ embed_tokensᵀ).
        let lm_head = Linear::new(embed_tokens.clone(), None)?;
        Ok(Self {
            embed_tokens,
            embed_positions: w.get("model.decoder.embed_positions.weight")?,
            layers,
            final_ln: load_layer_norm(w, "model.decoder.layer_norm")?,
            lm_head,
        })
    }
}

pub struct WhisperModel {
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
}

impl WhisperModel {
    pub fn load(w: &WhisperWeights) -> Result<Self> {
        Ok(Self {
            encoder: WhisperEncoder::load(w)?,
            decoder: WhisperDecoder::load(w)?,
        })
    }
}

fn load_linear(w: &WhisperWeights, prefix: &str, with_bias: bool) -> Result<Linear> {
    let weight = w.get(&format!("{prefix}.weight"))?;
    let bias = if with_bias {
        Some(w.get(&format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias)?)
}

fn load_layer_norm(w: &WhisperWeights, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: w.get(&format!("{prefix}.weight"))?,
        bias: w.get(&format!("{prefix}.bias"))?,
        eps: w.config.layer_norm_eps,
    })
}

fn load_attention(
    w: &WhisperWeights,
    prefix: &str,
    num_heads: usize,
    head_dim: usize,
) -> Result<WhisperAttention> {
    Ok(WhisperAttention {
        // k_proj без bias — характерно для Whisper.
        q: load_linear(w, &format!("{prefix}.q_proj"), true)?,
        k: load_linear(w, &format!("{prefix}.k_proj"), false)?,
        v: load_linear(w, &format!("{prefix}.v_proj"), true)?,
        out: load_linear(w, &format!("{prefix}.out_proj"), true)?,
        num_heads,
        head_dim,
        scale: (head_dim as f32).powf(-0.5),
    })
}
