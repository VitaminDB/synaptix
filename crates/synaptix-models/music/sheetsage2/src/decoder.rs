//! Декодер: BART на 6 слоёв (post-LN), выученные позиции со сдвигом 2,
//! LayerNorm эмбеддингов, выходная голова связана с таблицей токенов.
//!
//! Самовнимание пишет K/V в заранее выделенный буфер на всю длину окна
//! генерации (5 120 токенов), перекрёстное — считает K/V памяти энкодера один
//! раз на окно. Первый шаг прогоняет весь префикс (причинно), дальше — по
//! одному токену.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::config::SheetSage2Config;
use crate::loader::Weights;
use crate::nn::{add, attention, gelu, merge_heads, split_heads, Linear, Norm, R};
use crate::SheetError;

const POSITION_OFFSET: usize = 2;
const LN_EPS: f32 = 1e-5;

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Attn {
    fn load(w: &Weights, p: &str, compute: DType) -> R<Self> {
        Ok(Self {
            q: Linear::load(w, &format!("{p}.q_proj"), compute, true)?,
            k: Linear::load(w, &format!("{p}.k_proj"), compute, true)?,
            v: Linear::load(w, &format!("{p}.v_proj"), compute, true)?,
            o: Linear::load(w, &format!("{p}.out_proj"), compute, true)?,
        })
    }
}

struct Layer {
    self_attn: Attn,
    self_norm: Norm,
    cross_attn: Attn,
    cross_norm: Norm,
    fc1: Linear,
    fc2: Linear,
    final_norm: Norm,
}

pub struct Decoder {
    /// Таблица токенов в F32 (эмбеддинги у релиза не автокастятся).
    embed: Tensor,
    /// Позиции `[max_len + 2, d]` в F32.
    positions: Tensor,
    embed_norm: Norm,
    layers: Vec<Layer>,
    /// Та же таблица токенов в вычислительном типе — выходная голова.
    head: Tensor,
    heads: usize,
    head_dim: usize,
    pub max_len: usize,
    compute: DType,
    device: Device,
}

/// Состояние декодера одного окна.
pub struct DecoderState {
    self_k: Vec<Tensor>,
    self_v: Vec<Tensor>,
    cross_k: Vec<Tensor>,
    cross_v: Vec<Tensor>,
    pub len: usize,
}

impl Decoder {
    pub fn load(w: &Weights, cfg: &SheetSage2Config, compute: DType) -> R<Self> {
        let layers = (0..cfg.decoder_layers)
            .map(|i| {
                let p = format!("decoder.layers.{i}");
                Ok(Layer {
                    self_attn: Attn::load(w, &format!("{p}.self_attn"), compute)?,
                    self_norm: Norm::load(w, &format!("{p}.self_attn_layer_norm"), LN_EPS)?,
                    cross_attn: Attn::load(w, &format!("{p}.encoder_attn"), compute)?,
                    cross_norm: Norm::load(w, &format!("{p}.encoder_attn_layer_norm"), LN_EPS)?,
                    fc1: Linear::load(w, &format!("{p}.fc1"), compute, true)?,
                    fc2: Linear::load(w, &format!("{p}.fc2"), compute, true)?,
                    final_norm: Norm::load(w, &format!("{p}.final_layer_norm"), LN_EPS)?,
                })
            })
            .collect::<R<Vec<_>>>()?;
        Ok(Self {
            embed: w.f32("token_embedding.weight")?,
            positions: w.f32("decoder.embed_positions.weight")?,
            embed_norm: Norm::load(w, "decoder.layernorm_embedding", LN_EPS)?,
            layers,
            head: w.get("token_embedding.weight", compute)?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            max_len: cfg.max_output_seq_len,
            compute,
            device: w.device(),
        })
    }

    /// Новое окно: K/V памяти энкодера по слоям и пустой буфер самовнимания.
    pub fn start(&self, memory: &Tensor) -> R<DecoderState> {
        let mut cross_k = Vec::with_capacity(self.layers.len());
        let mut cross_v = Vec::with_capacity(self.layers.len());
        let mut self_k = Vec::with_capacity(self.layers.len());
        let mut self_v = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            cross_k.push(split_heads(&layer.cross_attn.k.forward(memory)?, self.heads)?);
            cross_v.push(split_heads(&layer.cross_attn.v.forward(memory)?, self.heads)?);
            let shape = vec![1usize, self.heads, self.max_len, self.head_dim];
            self_k.push(Tensor::zeros(shape.clone(), self.compute, self.device)?);
            self_v.push(Tensor::zeros(shape, self.compute, self.device)?);
        }
        Ok(DecoderState { self_k, self_v, cross_k, cross_v, len: 0 })
    }

    /// Прогнать токены `ids` (префикс или один новый) → логиты последней
    /// позиции (F32, на CPU).
    pub fn step(&self, state: &mut DecoderState, ids: &[u32]) -> R<Vec<f32>> {
        let n = ids.len();
        let past = state.len;
        if past + n > self.max_len {
            return Err(SheetError::Sequence("декодер: окно генерации переполнено".into()));
        }
        let ids_t = Tensor::from_vec(ids.to_vec(), vec![n], Device::Cpu)?.to_device(self.device)?;
        let tok = self.embed.index_select(0, &ids_t)?;
        let pos = self.positions.narrow(0, past + POSITION_OFFSET, n)?;
        let x = tok.add(&pos)?.reshape(vec![1usize, n, tok.dims()[1]])?;
        let mut x = self.embed_norm.forward(&x)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let q = split_heads(&layer.self_attn.q.forward(&x)?, self.heads)?;
            let k = split_heads(&layer.self_attn.k.forward(&x)?, self.heads)?;
            let v = split_heads(&layer.self_attn.v.forward(&x)?, self.heads)?;
            state.self_k[li].kv_append_inplace(&k, past)?;
            state.self_v[li].kv_append_inplace(&v, past)?;
            let k_all = state.self_k[li].narrow(2, 0, past + n)?;
            let v_all = state.self_v[li].narrow(2, 0, past + n)?;
            // Причинная маска нужна только префиксу (Tq = Tk); шаг в один токен видит всё.
            let attn = attention(&q, &k_all, &v_all, scale, n > 1)?;
            let o = layer.self_attn.o.forward(&merge_heads(&attn)?)?;
            x = layer.self_norm.forward(&add(&residual, &o)?)?;

            let residual = x.clone();
            let q = split_heads(&layer.cross_attn.q.forward(&x)?, self.heads)?;
            let attn = attention(&q, &state.cross_k[li], &state.cross_v[li], scale, false)?;
            let o = layer.cross_attn.o.forward(&merge_heads(&attn)?)?;
            x = layer.cross_norm.forward(&add(&residual, &o)?)?;

            let residual = x.clone();
            let y = layer.fc2.forward(&gelu(&layer.fc1.forward(&x)?)?)?;
            x = layer.final_norm.forward(&add(&residual, &y)?)?;
        }
        state.len = past + n;
        let last = x.narrow(1, n - 1, 1)?.contiguous()?.to_dtype(self.compute)?;
        let logits = last.linear(&self.head)?;
        Ok(logits.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1()?)
    }
}
