//! Аудио-энкодер AuT Qwen3-Omni (`thinker.audio_tower.*`) — как
//! `Qwen3OmniMoeAudioEncoder` и `Qwen3OmniMoeProcessor` transformers.
//!
//! Путь звука:
//! 1. 16 кГц моно (у оригинала `qwen_omni_utils` — librosa; здесь ffmpeg);
//! 2. log-mel Whisper (128 полос) по всей длине, `len / 160` кадров, без
//!    добивки (`padding=True` у одного звука), значения в bf16;
//! 3. куски по `2·n_window` = 100 кадров: три Conv2d 3×3/2 + GELU,
//!    `conv_out` 480·16 → 1280 (без смещения), синусоидальные позиции с нуля
//!    в каждом куске; валидные строки кусков подряд (13 на полный кусок);
//! 4. внимание — окнами по `13 · n_window_infer / 100` = 104 строки по всему
//!    звуку (`cu_seqlens` оригинала): окна независимы во всех слоях, поэтому
//!    каждое проходит 32 слоя Whisper отдельно; `k_proj` со смещением;
//! 5. `ln_post` → `proj1` → GELU → `proj2` в 2048 (скрытый размер LLM);
//! 6. промпт: `<|audio_start|>` + `<|audio_pad|>` × N + `<|audio_end|>`;
//!    позиции аудиотокенов — обычные (все три оси M-RoPE равны).

use std::path::Path;

use synaptix_asr_whisper::{EncoderLayer, WhisperConfig, WhisperWeights};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::conv::conv2d;
use synaptix_ops::norm::layer_norm;

use crate::pipeline::PipelineError;

type Res<T> = Result<T, PipelineError>;

fn err<E: std::fmt::Display>(ctx: &'static str) -> impl Fn(E) -> PipelineError {
    move |e| PipelineError::Forward(format!("{ctx}: {e}"))
}

const PREFIX: &str = "thinker.audio_tower.";

/// `thinker_config.audio_config` Qwen3-Omni.
#[derive(Debug, Clone)]
pub struct OmniAudioConfig {
    pub whisper: WhisperConfig,
    pub n_window: usize,
    pub n_window_infer: usize,
}

impl OmniAudioConfig {
    pub fn from_config_json(bytes: &[u8]) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        if v.get("model_type").and_then(|m| m.as_str()) != Some("qwen3_omni_moe") {
            return None;
        }
        let a = v.get("thinker_config")?.get("audio_config")?;
        // В audio_config Omni id токенов — `null`: для разбора их убираем.
        let mut clean = a.clone();
        if let Some(o) = clean.as_object_mut() {
            o.retain(|_, v| !v.is_null());
        }
        let whisper: WhisperConfig = serde_json::from_value(clean).ok()?;
        let n = |k: &str| a.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        Some(Self { whisper, n_window: n("n_window").unwrap_or(50), n_window_infer: n("n_window_infer").unwrap_or(800) })
    }
}

fn conv_len(l: usize) -> usize {
    (l - 1) / 2 + 1
}

/// Строк после стема у куска из `frames` кадров.
fn chunk_rows(frames: usize) -> usize {
    conv_len(conv_len(conv_len(frames)))
}

/// Аудиотокенов на `frames` кадров mel (`_get_feat_extract_output_lengths`).
pub fn tokens_for_frames(frames: usize, n_window: usize) -> usize {
    let chunk = 2 * n_window;
    let tail = frames % chunk;
    let full = frames / chunk;
    let tail_rows = if tail == 0 { 0 } else { chunk_rows(tail) };
    // У оригинала хвост 0 даёт (−1)//2+1 = 0 — та же сумма.
    full * chunk_rows(chunk) + tail_rows
}

pub fn prompt_block(tokens: usize) -> String {
    format!("<|audio_start|>{}<|audio_end|>", "<|audio_pad|>".repeat(tokens))
}

struct Tower {
    conv: Vec<(Tensor, Tensor)>,
    conv_out: QuantLinear,
    layers: Vec<EncoderLayer>,
    ln: (Tensor, Tensor),
    proj1: QuantLinear,
    proj2: QuantLinear,
    d: usize,
    device: Device,
    dtype: DType,
}

impl Tower {
    fn load(bundle: &Path, cfg: &OmniAudioConfig, device: Device, dtype: DType) -> Res<Self> {
        let w = WhisperWeights::open_encoder(bundle, cfg.whisper.clone(), PREFIX, device, dtype)
            .map_err(err("omni encoder"))?;
        let g = |n: &str| w.get(&format!("{PREFIX}{n}")).map_err(err("omni encoder"));
        let lin = |n: &str, bias: bool| -> Res<QuantLinear> {
            let b = if bias { Some(g(&format!("{n}.bias"))?) } else { None };
            QuantLinear::dense(g(&format!("{n}.weight"))?, b).map_err(err("omni encoder"))
        };
        Ok(Self {
            conv: (1..=3)
                .map(|i| Ok((g(&format!("conv2d{i}.weight"))?, g(&format!("conv2d{i}.bias"))?)))
                .collect::<Res<Vec<_>>>()?,
            conv_out: lin("conv_out", false)?,
            layers: (0..cfg.whisper.encoder_layers)
                .map(|i| EncoderLayer::load_index(&w, i).map_err(err("omni encoder")))
                .collect::<Res<Vec<_>>>()?,
            ln: (g("ln_post.weight")?, g("ln_post.bias")?),
            proj1: lin("proj1", true)?,
            proj2: lin("proj2", true)?,
            d: cfg.whisper.d_model,
            device,
            dtype,
        })
    }

    fn positions(&self, t: usize) -> Res<Tensor> {
        // `SinusoidsPositionEmbedding`: `[sin | cos]`, шаг log(10000)/(d/2−1).
        let d = self.d;
        let half = d / 2;
        let inc = (10000f64).ln() as f32 / (half as f32 - 1.0);
        let mut v = vec![0f32; t * d];
        for i in 0..t {
            for j in 0..half {
                let a = i as f32 * (-inc * j as f32).exp();
                v[i * d + j] = a.sin();
                v[i * d + half + j] = a.cos();
            }
        }
        Tensor::from_vec(v, vec![t, d], self.device).and_then(|x| x.to_dtype(self.dtype)).map_err(err("omni pos"))
    }

    /// Кусок mel `[n_mels, t]` → строки после стема и позиций `[rows, d]`.
    fn stem(&self, mel: &[f32], n_mels: usize, t: usize) -> Res<Tensor> {
        let e = err("omni stem");
        let mut x = Tensor::from_vec(mel.to_vec(), vec![1, 1, n_mels, t], self.device)
            .and_then(|x| x.to_dtype(self.dtype))
            .map_err(&e)?;
        for (w, b) in &self.conv {
            x = conv2d(&x, w, Some(b), (2, 2), (1, 1), (1, 1)).and_then(|y| y.gelu_exact()).map_err(&e)?;
        }
        let d = x.dims().to_vec();
        let (c, f, tt) = (d[1], d[2], d[3]);
        let x = x
            .permute(vec![0, 3, 1, 2])
            .and_then(|x| x.contiguous())
            .and_then(|x| x.reshape(vec![tt, c * f]))
            .map_err(&e)?;
        let h = self.conv_out.forward(&x).map_err(&e)?;
        h.add(&self.positions(tt)?).map_err(&e)
    }

    /// Окно строк `[n, d]` через все слои.
    fn window(&self, x: &Tensor) -> Res<Tensor> {
        let n = x.dims()[0];
        let mut h = x.reshape(vec![1, n, self.d]).map_err(err("omni encoder"))?;
        for l in &self.layers {
            h = l.run(&h).map_err(err("omni encoder"))?;
        }
        h.reshape(vec![n, self.d]).map_err(err("omni encoder"))
    }

    fn head(&self, x: &Tensor) -> Res<Tensor> {
        let e = err("omni head");
        let h = layer_norm(x, Some(&self.ln.0), Some(&self.ln.1), 1e-5).map_err(&e)?;
        let h = self.proj1.forward(&h).and_then(|t| t.gelu_exact()).map_err(&e)?;
        self.proj2.forward(&h).map_err(&e)
    }
}

pub struct OmniEncoded {
    pub embeds: Tensor,
    pub prompt_block: String,
}

fn encode_on(bundle: &Path, cfg: &OmniAudioConfig, mel: &[f32], frames: usize, device: Device, dtype: DType) -> Res<Tensor> {
    use synaptix_core::grad::no_grad;
    let n_mels = cfg.whisper.num_mel_bins;
    let tower = Tower::load(bundle, cfg, device, dtype)?;
    let chunk = 2 * cfg.n_window;
    // Стем по кускам, валидные строки подряд.
    let mut rows = Vec::new();
    let mut max_rows = 0usize;
    let mut start = 0usize;
    // Хвостовой кусок у оригинала добит нулями до самого длинного куска
    // (`pad_sequence`): за краем свёртки видят не ноль, а `gelu(bias)`
    // предыдущей свёртки по нулям — поэтому добиваем так же и берём только
    // валидные строки.
    let pad_to = if frames > chunk { chunk } else { frames };
    while start < frames {
        let len = chunk.min(frames - start);
        let mut part = vec![0f32; n_mels * pad_to];
        for m in 0..n_mels {
            part[m * pad_to..m * pad_to + len].copy_from_slice(&mel[m * frames + start..m * frames + start + len]);
        }
        let r = no_grad(|| tower.stem(&part, n_mels, pad_to))?;
        let valid = chunk_rows(len);
        max_rows = max_rows.max(r.dims()[0]);
        rows.push(r.narrow(0, 0, valid).and_then(|t| t.contiguous()).map_err(err("omni stem"))?);
        start += len;
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    let all = Tensor::cat(&refs, 0).map_err(err("omni cat"))?;
    let total = all.dims()[0];
    // Окна внимания: `max_len_after_cnn · n_window_infer / chunk` строк.
    let win = max_rows * (cfg.n_window_infer / chunk).max(1);
    let mut outs = Vec::new();
    let mut off = 0usize;
    while off < total {
        let n = win.min(total - off);
        let part = all.narrow(0, off, n).and_then(|t| t.contiguous()).map_err(err("omni window"))?;
        outs.push(no_grad(|| tower.window(&part))?);
        off += n;
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    let h = Tensor::cat(&refs, 0).map_err(err("omni cat"))?;
    no_grad(|| tower.head(&h))
}

/// Звук → эмбеддинги AuT. Башня грузится на время вызова (F32 на карте —
/// как у AF-Next, башня ~0,6 млрд); при нехватке памяти — CPU.
pub fn encode(bundle: &Path, cfg: &OmniAudioConfig, audio: &Path, device: Device) -> Res<OmniEncoded> {
    let samples = crate::wemm_audio::decode_audio(audio)?;
    let (m, frames) = crate::moss_audio::mel(&samples, cfg.whisper.num_mel_bins)?;
    let force_cpu = std::env::var_os("SYN_OMNI_AUDIO_CPU").is_some();
    let embeds = match device {
        Device::Cuda(o) if !force_cpu => match encode_on(bundle, cfg, &m, frames, device, DType::F32) {
            Ok(v) => v,
            Err(e) if e.to_string().to_lowercase().contains("memory") || e.to_string().contains("alloc") => {
                eprintln!("[omni-audio] не влезло в VRAM ({e}) — кодирую на CPU");
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
                encode_on(bundle, cfg, &m, frames, Device::Cpu, DType::F32)?
            }
            Err(e) => return Err(e),
        },
        _ => encode_on(bundle, cfg, &m, frames, Device::Cpu, DType::F32)?,
    };
    if let Device::Cuda(o) = device {
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
    }
    let embeds = embeds.to_device(device).map_err(err("omni to device"))?;
    let tokens = embeds.dims()[0];
    let expect = tokens_for_frames(frames, cfg.n_window);
    if tokens != expect {
        return Err(PipelineError::Forward(format!("omni: {tokens} строк вместо {expect}")));
    }
    if let Some(dump) = std::env::var_os("SYN_OMNI_AUDIO_DUMP") {
        let dump = std::path::PathBuf::from(dump);
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let emb: Vec<f32> = embeds
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(err("dump"))?;
        let _ = std::fs::write(dump.with_extension("pcm.f32"), bytes(&samples));
        let _ = std::fs::write(dump.with_extension("mel.f32"), bytes(&m));
        let _ = std::fs::write(dump.with_extension("emb.f32"), bytes(&emb));
    }
    Ok(OmniEncoded { embeds, prompt_block: prompt_block(tokens) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_like_processor() {
        // Пробы процессора: 383 кадра → 50, 1200 → 156, 3000 → 390.
        assert_eq!(tokens_for_frames(383, 50), 50);
        assert_eq!(tokens_for_frames(1200, 50), 156);
        assert_eq!(tokens_for_frames(3000, 50), 390);
    }
}
