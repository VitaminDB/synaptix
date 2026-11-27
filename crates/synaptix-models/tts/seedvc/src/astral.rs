//! Содержание речи: HuBERT-large (первые 18 слоёв, без финальной нормы) →
//! ASTRAL (`modules/astral_quantization`): ConvNeXtV2 (1024→512, 12 блоков) →
//! BSQ: `project_in` 512→log2(K), бит = знак, индекс — биты старшим вперёд.
//! Два токенизатора со своими весами: «широкий» (K = 2048) и «узкий» (K = 32).

use std::path::Path;

use synaptix_asr_wav2vec2::{normalize_audio, Wav2Vec2Model, Wav2Vec2Weights};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::conv::depthwise::depthwise_conv;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::loader::Comp;
use crate::nn::{host, Lin};
use crate::{SeedVcError, R};

/// Слой HuBERT, с которого берутся признаки.
pub const SSL_LAYER: usize = 18;
/// Сэмплов 16 кГц на токен (50 токенов в секунду).
pub const SSL_HOP: usize = 320;
/// Окно длинной записи и перекрытие, секунд (как в эталоне).
const WINDOW_S: usize = 30;
const OVERLAP_S: usize = 5;

pub struct Hubert {
    model: Wav2Vec2Model,
    device: Device,
}

impl Hubert {
    /// Компонент `hubert` бандла, конфиг — файл `hubert/config.json`.
    pub fn load(bundle: &Path, device: Device) -> R<Self> {
        let w = Wav2Vec2Weights::from_syn_component(bundle, "hubert", "hubert/config.json", device, DType::F32).map_err(|e| SeedVcError::Load(format!("hubert: {e}")))?;
        let model = Wav2Vec2Model::load(&w).map_err(|e| SeedVcError::Load(format!("hubert: {e}")))?;
        if model.config().num_hidden_layers < SSL_LAYER {
            return Err(SeedVcError::Load(format!("hubert: слоёв {}, нужно {SSL_LAYER}", model.config().num_hidden_layers)));
        }
        Ok(Self { model, device })
    }

    /// Волна 16 кГц (не длиннее 30 с) → признаки `[1, T, 1024]`, `T = min(кадров, N / 320)`.
    pub fn features(&self, wave_16k: &[f32]) -> R<Tensor> {
        let e = |e: synaptix_asr_wav2vec2::Wav2Vec2Error| SeedVcError::Other(format!("hubert: {e}"));
        let frames = self.model.config().frames_for(wave_16k.len());
        let t = frames.min(wave_16k.len() / SSL_HOP);
        if t == 0 {
            return Err(SeedVcError::Input("запись слишком короткая".into()));
        }
        let x = Tensor::from_vec(normalize_audio(wave_16k), vec![1, wave_16k.len()], self.device)?;
        let proj = self.model.project(&self.model.extract_features(&x).map_err(e)?).map_err(e)?;
        let h = self.model.encode_partial(&proj, SSL_LAYER).map_err(e)?;
        Ok(h.narrow(1, 0, t)?.contiguous()?)
    }
}

struct Block {
    dw_w: Tensor,
    dw_b: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: Lin,
    grn_g: Tensor,
    grn_b: Tensor,
    pw2: Lin,
}

pub struct Astral {
    input: Lin,
    blocks: Vec<Block>,
    project_in: Lin,
}

impl Astral {
    pub fn load(c: &Comp) -> R<Self> {
        let mut blocks = Vec::new();
        let mut i = 0;
        while c.has(&format!("encoder.blocks.{i}.dwconv.weight")) {
            let p = format!("encoder.blocks.{i}");
            let grn = |n: &str| -> R<Tensor> {
                let t = c.get(&format!("{p}.grn.{n}"), DType::F32)?;
                let ch = t.dims().iter().product::<usize>();
                Ok(t.reshape(vec![1, 1, ch])?)
            };
            blocks.push(Block {
                dw_w: c.get(&format!("{p}.dwconv.weight"), DType::F32)?,
                dw_b: c.get(&format!("{p}.dwconv.bias"), DType::F32)?,
                ln_w: c.get(&format!("{p}.norm.weight"), DType::F32)?,
                ln_b: c.get(&format!("{p}.norm.bias"), DType::F32)?,
                pw1: Lin::load(c, &format!("{p}.pwconv1"), true)?,
                grn_g: grn("gamma")?,
                grn_b: grn("beta")?,
                pw2: Lin::load(c, &format!("{p}.pwconv2"), true)?,
            });
            i += 1;
        }
        if blocks.is_empty() {
            return Err(SeedVcError::Load("в компоненте ASTRAL нет блоков".into()));
        }
        Ok(Self { input: Lin::load(c, "encoder.input_projection", true)?, blocks, project_in: Lin::load(c, "quantizer.project_in", true)? })
    }

    /// Число кодов (2^бит).
    pub fn codebook_size(&self) -> usize {
        1 << self.project_in.out_dim()
    }

    /// Признаки HuBERT `[1, T, 1024]` → скрытое состояние `[1, T, 512]`.
    pub fn encode(&self, feats: &Tensor) -> R<Tensor> {
        let mut x = self.input.fwd(feats)?; // [1, T, 512], каналы последними
        for b in &self.blocks {
            let c = x.dims()[2];
            let y = depthwise_conv(&x.transpose(1, 2)?.contiguous()?, &b.dw_w, Some(&b.dw_b), 1, 3, c)?;
            let y = layer_norm(&y.transpose(1, 2)?.contiguous()?, Some(&b.ln_w), Some(&b.ln_b), 1e-6)?;
            let y = b.pw1.fwd(&y)?.gelu_exact()?;
            // GRN: норма по времени на канал, делённая на её среднее по каналам.
            let g = y.sqr()?.sum_keepdim(1)?.sqrt()?;
            let n = g.broadcast_div(&g.mean_keepdim(2)?.add_scalar(1e-6)?)?;
            let y = y.broadcast_mul(&n)?.broadcast_mul(&b.grn_g)?.broadcast_add(&b.grn_b)?.add(&y)?;
            x = x.add(&b.pw2.fwd(&y)?)?;
        }
        Ok(x)
    }

    /// Скрытое состояние → индексы кодов (`T` штук).
    pub fn quantize(&self, hidden: &Tensor) -> R<Vec<u32>> {
        let bits = self.project_in.out_dim();
        let v = host(&self.project_in.fwd(hidden)?)?;
        Ok(v.chunks(bits).map(|row| row.iter().fold(0u32, |acc, &x| (acc << 1) | u32::from(x > 0.0))).collect())
    }

    pub fn tokens(&self, feats: &Tensor) -> R<Vec<u32>> {
        self.quantize(&self.encode(feats)?)
    }
}

/// Токены записи любой длины: до 30 с — одним проходом, дольше — окнами по
/// 30 с с перекрытием 5 с (у следующих окон первые 250 токенов отбрасываются),
/// как `_process_content_features` эталона. Возвращает по вектору токенов на
/// каждый токенизатор из `quantizers`.
pub fn content_tokens(hubert: &Hubert, quantizers: &[&Astral], wave_16k: &[f32]) -> R<Vec<Vec<u32>>> {
    let sr = 16_000usize;
    let mut out: Vec<Vec<u32>> = quantizers.iter().map(|_| Vec::new()).collect();
    let run = |chunk: &[f32], skip: usize, out: &mut Vec<Vec<u32>>| -> R<()> {
        let feats = hubert.features(chunk)?;
        for (q, o) in quantizers.iter().zip(out.iter_mut()) {
            let t = q.tokens(&feats)?;
            o.extend_from_slice(&t[skip.min(t.len())..]);
        }
        Ok(())
    };
    let n = wave_16k.len();
    if n <= sr * WINDOW_S {
        run(wave_16k, 0, &mut out)?;
        return Ok(out);
    }
    let mut pos = 0usize;
    let mut buffer: Vec<f32> = Vec::new();
    let mut first = true;
    while pos < n {
        if first {
            let end = (pos + sr * WINDOW_S).min(n);
            let chunk = &wave_16k[pos..end];
            run(chunk, 0, &mut out)?;
            buffer = chunk[chunk.len() - sr * OVERLAP_S..].to_vec();
            pos += sr * WINDOW_S;
            first = false;
        } else {
            let end = (pos + sr * (WINDOW_S - OVERLAP_S)).min(n);
            let mut chunk = buffer.clone();
            chunk.extend_from_slice(&wave_16k[pos..end]);
            run(&chunk, 50 * OVERLAP_S, &mut out)?;
            buffer = chunk[chunk.len() - sr * OVERLAP_S..].to_vec();
            pos += chunk.len() - sr * OVERLAP_S;
        }
    }
    Ok(out)
}

/// Схлопнуть подряд идущие повторы (`duration_reduction_func`, n_gram = 1).
pub fn dedup(tokens: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(tokens.len());
    for &t in tokens {
        if out.last() != Some(&t) {
            out.push(t);
        }
    }
    out
}
