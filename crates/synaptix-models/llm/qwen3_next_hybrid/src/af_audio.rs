//! Аудио-башня Audio Flamingo Next (`audio_tower.*`, `multi_modal_projector.*`,
//! `model_type = musicflamingo`) — как `modeling_musicflamingo.py`,
//! `modeling_audioflamingo3.py` и `processing_musicflamingo.py` transformers.
//!
//! Путь звука:
//! 1. 16 кГц моно (torchcodec у оригинала — ffmpeg, как у WeMM);
//! 2. окна по 30 с; у каждого — log-mel Whisper (128 полос, 3000 кадров, хвост
//!    нулями) и маска валидных кадров `ceil(n / 160)`;
//! 3. энкодер AF3: стем Whisper (conv1d ×2 + GELU, позиции) по всему окну,
//!    затем — только валидные кадры `(L-1)/2+1` (то же, что маска внимания
//!    оригинала), 32 слоя Whisper, AvgPool1d(2) и LayerNorm;
//! 4. RoTE (`MusicFlamingoRotaryEmbedding`): первые 512 измерений выхода
//!    поворачиваются парами — оси «номер окна» и «кадр в окне» (по 128
//!    частот `1200^(-2i/256)`), углы умножены на `-2π·t` абсолютного времени
//!    кадра `t = окно·30 + j·0,04` с; таблицы в f32, поворот в f64;
//! 5. проектор `linear_1` → GELU → `linear_2` в скрытый размер LLM;
//! 6. промпт: `<|sound_bos|>` + `<sound>` × N + `<|sound_eos|>`, N — сумма
//!    валидных строк окон.

use std::path::Path;

use synaptix_asr_whisper::mel::whisper_log_mel;
use synaptix_asr_whisper::{EncoderLayer, WhisperConfig, WhisperWeights};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::conv::conv1d;
use synaptix_ops::norm::layer_norm;

use crate::pipeline::PipelineError;

type Res<T> = Result<T, PipelineError>;

fn err<E: std::fmt::Display>(ctx: &'static str) -> impl Fn(E) -> PipelineError {
    move |e| PipelineError::Forward(format!("{ctx}: {e}"))
}

const SR: usize = 16_000;
const WINDOW: usize = 30 * SR;
const HOP: usize = 160;

/// Конфиг башни AF-Next из config.json.
#[derive(Debug, Clone)]
pub struct AfConfig {
    pub whisper: WhisperConfig,
    /// Шаг кадра mel, с (`audio_frame_step`).
    pub frame_step: f64,
    /// `rope_parameters.rope_theta` RoTE.
    pub rote_theta: f64,
    /// Число повёрнутых частот на ось: `head_dim·partial/2`.
    pub rote_freqs: usize,
    /// `max_position_embeddings` RoTE (1200).
    pub rote_max: f64,
}

impl AfConfig {
    pub fn from_config_json(bytes: &[u8]) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        if v.get("model_type").and_then(|m| m.as_str()) != Some("musicflamingo") {
            return None;
        }
        let a = v.get("audio_config")?;
        let mut whisper: WhisperConfig = serde_json::from_value(a.clone()).ok()?;
        let n = |o: &serde_json::Value, k: &str| o.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        // Конфиг энкодера AF3 — в именах `hidden_size`/`num_hidden_layers`/…
        whisper.d_model = n(a, "hidden_size").unwrap_or(whisper.d_model);
        whisper.encoder_layers = n(a, "num_hidden_layers").unwrap_or(whisper.encoder_layers);
        whisper.encoder_attention_heads = n(a, "num_attention_heads").unwrap_or(whisper.encoder_attention_heads);
        whisper.encoder_ffn_dim = n(a, "intermediate_size").unwrap_or(whisper.encoder_ffn_dim);
        let rp = v.get("rope_parameters")?;
        let head_dim = n(&v, "head_dim").unwrap_or(whisper.d_model) as f64;
        let partial = rp.get("partial_rotary_factor").and_then(|x| x.as_f64()).unwrap_or(1.0);
        Some(Self {
            whisper,
            frame_step: v.get("audio_frame_step").and_then(|x| x.as_f64()).unwrap_or(0.01),
            rote_theta: rp.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1200.0),
            rote_freqs: ((head_dim * partial) as usize) / 2,
            rote_max: v.get("max_position_embeddings").and_then(|x| x.as_f64()).unwrap_or(1200.0),
        })
    }
}

fn conv_len(l: usize) -> usize {
    (l - 1) / 2 + 1
}

fn pool_len(l: usize) -> usize {
    (l - 2) / 2 + 1
}

/// Аудиотокенов окна с `samples` отсчётами (`_get_audio_token_length` по
/// маске кадров `ceil(n/160)`).
pub fn window_tokens(samples: usize) -> usize {
    pool_len(conv_len(samples.div_ceil(HOP)))
}

/// `<|sound_bos|><sound>…<|sound_eos|>`.
pub fn prompt_block(tokens: usize) -> String {
    format!("<|sound_bos|>{}<|sound_eos|>", "<sound>".repeat(tokens))
}

/// Таблицы RoTE окна `w` на `seq` строк: `(cos, sin)` по `[seq, 4·freqs]`
/// (оси окна и кадра, частоты повторены парами), в f32, как у оригинала.
fn rote_tables(cfg: &AfConfig, w: usize, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let nf = cfg.rote_freqs;
    let dim = 2 * nf;
    let inv: Vec<f32> = (0..nf)
        .map(|i| 1.0 / (cfg.rote_theta as f32).powf((2 * i) as f32 / dim as f32))
        .collect();
    let step = (cfg.frame_step * 4.0) as f32;
    let max = cfg.rote_max as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    // Начало окна и его номер по оси (`round(start / duration) / max`).
    let start = (w * seq) as f32 * step;
    let duration = (cfg.frame_step * 4.0 * seq as f64) as f32;
    let wpos = (start / duration).round() / max;
    let rot = 2 * dim;
    let mut cos = vec![0f32; seq * rot];
    let mut sin = vec![0f32; seq * rot];
    for j in 0..seq {
        let t = start + j as f32 * step;
        let angle = -t * 2.0 * std::f32::consts::PI;
        let pos = j as f32 / max * two_pi;
        for i in 0..nf {
            let wf = wpos * inv[i];
            let tf = pos * inv[i];
            for (base, f) in [(0, wf), (dim, tf)] {
                let a = f * angle;
                let (c, s) = (a.cos(), a.sin());
                for k in 0..2 {
                    cos[j * rot + base + 2 * i + k] = c;
                    sin[j * rot + base + 2 * i + k] = s;
                }
            }
        }
    }
    (cos, sin)
}

/// `apply_rotary_time_emb`: поворот первых `cos.len()/seq` измерений парами
/// `(x0, x1) → (x0·c − x1·s, x1·c + x0·s)` в f64; остальное — как есть.
fn apply_rote(x: &mut [f32], d: usize, seq: usize, cos: &[f32], sin: &[f32]) {
    let rot = cos.len() / seq;
    for j in 0..seq {
        let row = &mut x[j * d..(j + 1) * d];
        for p in (0..rot).step_by(2) {
            let (a, b) = (row[p] as f64, row[p + 1] as f64);
            let (c0, s0) = (cos[j * rot + p] as f64, sin[j * rot + p] as f64);
            let (c1, s1) = (cos[j * rot + p + 1] as f64, sin[j * rot + p + 1] as f64);
            row[p] = (a * c0 - b * s0) as f32;
            row[p + 1] = (b * c1 + a * s1) as f32;
        }
    }
}

struct Tower {
    conv1: (Tensor, Tensor),
    conv2: (Tensor, Tensor),
    pos: Tensor,
    layers: Vec<EncoderLayer>,
    ln: (Tensor, Tensor),
    proj1: QuantLinear,
    proj2: QuantLinear,
    d: usize,
    device: Device,
    dtype: DType,
}

impl Tower {
    fn load(bundle: &Path, cfg: &AfConfig, device: Device, dtype: DType) -> Res<Self> {
        let w = WhisperWeights::open_encoder(bundle, cfg.whisper.clone(), "audio_tower.", device, dtype)
            .map_err(err("af encoder"))?;
        let g = |n: &str| w.get(n).map_err(err("af encoder"));
        let lin = |p: &str| -> Res<QuantLinear> {
            QuantLinear::dense(g(&format!("{p}.weight"))?, Some(g(&format!("{p}.bias"))?)).map_err(err("af projector"))
        };
        Ok(Self {
            conv1: (g("audio_tower.conv1.weight")?, g("audio_tower.conv1.bias")?),
            conv2: (g("audio_tower.conv2.weight")?, g("audio_tower.conv2.bias")?),
            pos: g("audio_tower.embed_positions.weight")?,
            layers: (0..cfg.whisper.encoder_layers)
                .map(|i| EncoderLayer::load_index(&w, i).map_err(err("af encoder")))
                .collect::<Res<Vec<_>>>()?,
            ln: (g("audio_tower.layer_norm.weight")?, g("audio_tower.layer_norm.bias")?),
            proj1: lin("multi_modal_projector.linear_1")?,
            proj2: lin("multi_modal_projector.linear_2")?,
            d: cfg.whisper.d_model,
            device,
            dtype,
        })
    }

    /// Окно mel `[n_mels, 3000]` с `valid` валидными кадрами → строки
    /// энкодера после пула и LayerNorm `[pool_len(conv_len(valid)), d]` и
    /// полная длина после пула (для RoTE).
    fn window(&self, mel: &[f32], n_mels: usize, frames: usize, valid: usize) -> Res<(Tensor, usize)> {
        let e = err("af encoder");
        let x = Tensor::from_vec(mel.to_vec(), vec![1, n_mels, frames], self.device)
            .and_then(|x| x.to_dtype(self.dtype))
            .map_err(&e)?;
        let x = conv1d(&x, &self.conv1.0, Some(&self.conv1.1), 1, 1).and_then(|y| y.gelu_exact()).map_err(&e)?;
        let x = conv1d(&x, &self.conv2.0, Some(&self.conv2.1), 2, 1).and_then(|y| y.gelu_exact()).map_err(&e)?;
        let x = x.transpose(1, 2).and_then(|x| x.contiguous()).map_err(&e)?;
        let full = x.dims()[1];
        let x = x.broadcast_add(&self.pos.narrow(0, 0, full).and_then(|p| p.unsqueeze(0)).map_err(&e)?).map_err(&e)?;
        // Маска внимания оригинала = только валидные кадры.
        let lv = conv_len(valid).min(full);
        let mut h = x.narrow(1, 0, lv).and_then(|x| x.contiguous()).map_err(&e)?;
        for l in &self.layers {
            h = l.run(&h).map_err(err("af encoder"))?;
        }
        // AvgPool1d(2, stride 2): средние соседних пар.
        let np = pool_len(lv);
        let pairs = h
            .narrow(1, 0, 2 * np)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![1, np, 2, self.d]))
            .map_err(&e)?;
        let a = pairs.narrow(2, 0, 1).and_then(|t| t.contiguous()).map_err(&e)?;
        let b = pairs.narrow(2, 1, 1).and_then(|t| t.contiguous()).map_err(&e)?;
        let pooled = a
            .add(&b)
            .and_then(|t| t.mul_scalar(0.5))
            .and_then(|t| t.reshape(vec![np, self.d]))
            .map_err(&e)?;
        let out = layer_norm(&pooled, Some(&self.ln.0), Some(&self.ln.1), 1e-5).map_err(&e)?;
        Ok((out, pool_len(full)))
    }

    fn project(&self, x: &Tensor) -> Res<Tensor> {
        let e = err("af projector");
        let h = self.proj1.forward(x).and_then(|t| t.gelu_exact()).map_err(&e)?;
        self.proj2.forward(&h).map_err(&e)
    }
}

/// Звук → строки `[N, hidden]` для `<sound>` и блок промпта.
pub struct AfEncoded {
    pub embeds: Tensor,
    pub prompt_block: String,
}

fn encode_on(bundle: &Path, cfg: &AfConfig, samples: &[f32], device: Device, dtype: DType) -> Res<Tensor> {
    use synaptix_core::grad::no_grad;
    let n_mels = cfg.whisper.num_mel_bins;
    let frames = cfg.whisper.max_source_positions * 2;
    let tower = Tower::load(bundle, cfg, device, dtype)?;
    let mut rows = Vec::new();
    let n_win = samples.len().div_ceil(WINDOW).max(1);
    for w in 0..n_win {
        let chunk = &samples[(w * WINDOW).min(samples.len())..((w + 1) * WINDOW).min(samples.len())];
        let mut seg = chunk.to_vec();
        seg.resize(WINDOW, 0.0);
        let (mel, _, t) = whisper_log_mel(&seg, n_mels, frames).map_err(err("mel"))?;
        let valid = chunk.len().div_ceil(HOP);
        let (h, seq) = no_grad(|| tower.window(&mel, n_mels, t, valid))?;
        // RoTE на хосте (f64), как у оригинала.
        let n = h.dims()[0];
        let mut v: Vec<f32> = h
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(err("af rote"))?;
        let (cos, sin) = rote_tables(cfg, w, seq);
        let rot = cos.len() / seq;
        apply_rote(&mut v, tower.d, n, &cos[..n * rot], &sin[..n * rot]);
        let h = Tensor::from_vec(v, vec![n, tower.d], device)
            .and_then(|t| t.to_dtype(dtype))
            .map_err(err("af rote"))?;
        rows.push(no_grad(|| tower.project(&h))?);
    }
    let r: Vec<&Tensor> = rows.iter().collect();
    Tensor::cat(&r, 0).map_err(err("af cat"))
}

/// Звук → эмбеддинги AF-Next. Башня грузится на время вызова; на CUDA при
/// нехватке памяти — CPU F32 и перенос на `device`.
pub fn encode(bundle: &Path, cfg: &AfConfig, audio: &Path, device: Device) -> Res<AfEncoded> {
    let samples = crate::wemm_audio::decode_audio(audio)?;
    let force_cpu = std::env::var_os("SYN_AF_AUDIO_CPU").is_some();
    let embeds = match device {
        // F32 и на карте: в BF16 у ~1 % строк выход расходился с эталоном до
        // cos 0,3 (выбросы проектора), в F32 — 2e-5; башня 0,65 млрд — 2,6 ГБ
        // на время кодирования, рядом с 7B-LLM места хватает.
        Device::Cuda(o) if !force_cpu => match encode_on(bundle, cfg, &samples, device, DType::F32) {
            Ok(v) => v,
            Err(e) if e.to_string().to_lowercase().contains("memory") || e.to_string().contains("alloc") => {
                eprintln!("[af-audio] не влезло в VRAM ({e}) — кодирую на CPU");
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
                encode_on(bundle, cfg, &samples, Device::Cpu, DType::F32)?
            }
            Err(e) => return Err(e),
        },
        _ => encode_on(bundle, cfg, &samples, Device::Cpu, DType::F32)?,
    };
    if let Device::Cuda(o) = device {
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
    }
    let embeds = embeds.to_device(device).map_err(err("af to device"))?;
    let tokens = embeds.dims()[0];
    let expect: usize = (0..samples.len().div_ceil(WINDOW).max(1))
        .map(|w| window_tokens(((w + 1) * WINDOW).min(samples.len()) - (w * WINDOW).min(samples.len())))
        .sum();
    if tokens != expect {
        return Err(PipelineError::Forward(format!("af: {tokens} строк вместо {expect}")));
    }
    if let Some(dump) = std::env::var_os("SYN_AF_AUDIO_DUMP") {
        let dump = std::path::PathBuf::from(dump);
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let emb: Vec<f32> = embeds
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(err("dump"))?;
        let _ = std::fs::write(dump.with_extension("pcm.f32"), bytes(&samples));
        let _ = std::fs::write(dump.with_extension("emb.f32"), bytes(&emb));
    }
    Ok(AfEncoded { embeds, prompt_block: prompt_block(tokens) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_like_processor() {
        // 30 с: 3000 кадров → 1500 → 750; 3,84 с (61395 отсч.) → 384 кадра → 192 → 96.
        assert_eq!(window_tokens(480_000), 750);
        assert_eq!(window_tokens(61_395), 96);
    }
}
