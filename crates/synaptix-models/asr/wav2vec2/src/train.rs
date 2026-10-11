//! Обучение: именованные параметры (имена HF), замена CTC-головы под новый
//! словарь с тёплым стартом, сохранение весов в safetensors.

use std::collections::BTreeMap;
use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::model::{ConvNorm, Linear, Ln, Wav2Vec2Model};
use crate::{Result, Wav2Vec2Error};

fn push_ln<'a>(out: &mut Vec<(String, &'a mut Tensor)>, prefix: String, ln: &'a mut Ln) {
    out.push((format!("{prefix}.weight"), &mut ln.weight));
    out.push((format!("{prefix}.bias"), &mut ln.bias));
}

fn push_lin<'a>(out: &mut Vec<(String, &'a mut Tensor)>, prefix: String, l: &'a mut Linear) {
    out.push((format!("{prefix}.weight"), &mut l.weight));
    out.push((format!("{prefix}.bias"), &mut l.bias));
}

impl Wav2Vec2Model {
    /// Все параметры с именами HF `Wav2Vec2ForCTC` (энкодер — с префиксом
    /// `wav2vec2.`). Позиционная свёртка хранится уже свёрнутой (`weight`).
    pub fn named_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let prefix = self.config.weight_prefix();
        let mut out: Vec<(String, &mut Tensor)> = Vec::new();
        for (i, c) in self.convs.iter_mut().enumerate() {
            let p = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            out.push((format!("{p}.conv.weight"), &mut c.weight));
            if let Some(b) = c.bias.as_mut() {
                out.push((format!("{p}.conv.bias"), b));
            }
            match &mut c.norm {
                ConvNorm::Layer(ln) | ConvNorm::Group(ln) => push_ln(&mut out, format!("{p}.layer_norm"), ln),
                ConvNorm::None => {}
            }
        }
        push_ln(&mut out, "wav2vec2.feature_projection.layer_norm".into(), &mut self.proj_norm);
        push_lin(&mut out, "wav2vec2.feature_projection.projection".into(), &mut self.proj);
        out.push(("wav2vec2.encoder.pos_conv_embed.conv.weight".into(), &mut self.pos_conv.weight));
        out.push(("wav2vec2.encoder.pos_conv_embed.conv.bias".into(), &mut self.pos_conv.bias));
        push_ln(&mut out, "wav2vec2.encoder.layer_norm".into(), &mut self.enc_norm);
        for (i, l) in self.layers.iter_mut().enumerate() {
            let p = format!("wav2vec2.encoder.layers.{i}");
            push_ln(&mut out, format!("{p}.layer_norm"), &mut l.attn_norm);
            push_lin(&mut out, format!("{p}.attention.q_proj"), &mut l.attn.q);
            push_lin(&mut out, format!("{p}.attention.k_proj"), &mut l.attn.k);
            push_lin(&mut out, format!("{p}.attention.v_proj"), &mut l.attn.v);
            push_lin(&mut out, format!("{p}.attention.out_proj"), &mut l.attn.out);
            push_ln(&mut out, format!("{p}.final_layer_norm"), &mut l.ff_norm);
            push_lin(&mut out, format!("{p}.feed_forward.intermediate_dense"), &mut l.ff_in);
            push_lin(&mut out, format!("{p}.feed_forward.output_dense"), &mut l.ff_out);
            if let Some(g) = l.attn.gate.as_mut() {
                push_lin(&mut out, format!("{p}.attention.gru_rel_pos_linear"), &mut g.linear);
                out.push((format!("{p}.attention.gru_rel_pos_const"), &mut g.constant));
            }
        }
        if let Some((_, t)) = self.rel_embed.as_mut() {
            out.push(("wav2vec2.encoder.layers.0.attention.rel_attn_embed.weight".into(), t));
        }
        if let Some(h) = self.lm_head.as_mut() {
            push_lin(&mut out, "lm_head".into(), h);
        }
        // Имена HF зависят от семейства модели.
        for (n, _) in out.iter_mut() {
            if prefix != "wav2vec2." {
                if let Some(rest) = n.strip_prefix("wav2vec2.") {
                    *n = format!("{prefix}{rest}");
                }
            }
        }
        out
    }

    /// Нормализованный PCM `[1, N]` → логиты `[1, T, V]` (с графом, если
    /// какие-то параметры требуют градиента).
    pub fn logits(&self, wav: &Tensor) -> Result<Tensor> {
        let features = self.extract_features(wav)?;
        let proj = self.project(&features)?;
        let (_, hidden) = self.encode_with_pos(&proj)?;
        self.head(&hidden)
    }

    /// Новая CTC-голова под словарь размера `vocab`: строка `i` копируется из
    /// старой строки `init[i]`, иначе — среднее старых строк (смещение — то же).
    pub fn replace_head(&mut self, vocab: usize, init: &[Option<usize>]) -> Result<()> {
        let d = self.config.hidden_size;
        // Базовая модель без головы: случайная инициализация N(0, 0.02), смещение 0.
        if self.lm_head.is_none() {
            let mut seed = 0x9e3779b97f4a7c15u64;
            let mut gauss = || {
                let mut u = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 11) as f64 + 1.0) / (1u64 << 53) as f64
                };
                let (a, b) = (u(), u());
                ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
            };
            let w: Vec<f32> = (0..vocab * d).map(|_| 0.02 * gauss()).collect();
            let dev = self.device();
            let dt = self.dtype();
            self.lm_head = Some(Linear {
                weight: Tensor::from_vec(w, vec![vocab, d], dev)?.to_dtype(dt)?,
                bias: Tensor::from_vec(vec![0f32; vocab], vec![vocab], dev)?.to_dtype(dt)?,
            });
            self.config.vocab_size = vocab;
            return Ok(());
        }
        let head = self.lm_head.as_ref().ok_or_else(|| Wav2Vec2Error::Config("нет lm_head".into()))?;
        let w = head.weight.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?;
        let b = head.bias.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?;
        let (old_v, d) = (w.dims()[0], w.dims()[1]);
        let wh = w.flatten_all()?.to_vec1::<f32>()?;
        let bh = b.flatten_all()?.to_vec1::<f32>()?;
        let mut mean_w = vec![0f32; d];
        for r in wh.chunks(d) {
            for (m, x) in mean_w.iter_mut().zip(r) {
                *m += x / old_v as f32;
            }
        }
        let mean_b = bh.iter().sum::<f32>() / old_v as f32;
        let mut nw = Vec::with_capacity(vocab * d);
        let mut nb = Vec::with_capacity(vocab);
        for i in 0..vocab {
            match init.get(i).copied().flatten().filter(|&j| j < old_v) {
                Some(j) => {
                    nw.extend_from_slice(&wh[j * d..(j + 1) * d]);
                    nb.push(bh[j]);
                }
                None => {
                    nw.extend_from_slice(&mean_w);
                    nb.push(mean_b);
                }
            }
        }
        let dev = self.device();
        let dt = self.dtype();
        self.lm_head = Some(Linear {
            weight: Tensor::from_vec(nw, vec![vocab, d], dev)?.to_dtype(dt)?,
            bias: Tensor::from_vec(nb, vec![vocab], dev)?.to_dtype(dt)?,
        });
        self.config.vocab_size = vocab;
        Ok(())
    }

    /// Сохранить все веса в safetensors (`dtype` — F32 или F16).
    pub fn save_safetensors(&mut self, path: &Path, dtype: DType) -> Result<()> {
        let mut blobs: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
        for (name, t) in self.named_parameters_mut() {
            let shape = t.dims().to_vec();
            let host = t.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let bytes: Vec<u8> = match dtype {
                DType::F16 => host.iter().flat_map(|&x| f32_to_f16_bits(x).to_le_bytes()).collect(),
                _ => host.iter().flat_map(|x| x.to_le_bytes()).collect(),
            };
            blobs.insert(name, (shape, bytes));
        }
        let st_dtype = if dtype == DType::F16 { safetensors::Dtype::F16 } else { safetensors::Dtype::F32 };
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = blobs
            .iter()
            .map(|(n, (s, b))| {
                safetensors::tensor::TensorView::new(st_dtype, s.clone(), b)
                    .map(|v| (n.clone(), v))
                    .map_err(|e| Wav2Vec2Error::Load(format!("{n}: {e}")))
            })
            .collect::<Result<_>>()?;
        let bytes = safetensors::serialize(views, None).map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
        std::fs::write(path, bytes).map_err(|e| Wav2Vec2Error::Load(format!("{}: {e}", path.display())))
    }
}

/// f32 → IEEE half (округление к ближайшему чётному).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        let round = (m >> 12) & 1 != 0 && ((m & 0xfff) != 0 || (m >> 13) & 1 != 0);
        return sign | ((m >> 13) as u16 + round as u16);
    }
    let mut h = sign as u32 | ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0) {
        h += 1;
    }
    h as u16
}

#[cfg(test)]
mod tests {
    use super::f32_to_f16_bits;

    #[test]
    fn f16_known_values() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(1e-8), 0);
    }
}
