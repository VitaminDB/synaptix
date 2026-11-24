//! Аудио-башня MOSS-Audio (`audio_encoder.*`, `audio_adapter.*`,
//! `deepstack_audio_merger_list.*`) — как `modeling_moss_audio.py` и
//! `processing_moss_audio.py` оригинала (github.com/OpenMOSS/MOSS-Audio).
//!
//! Путь звука:
//! 1. декодирование в исходной частоте, каналы — среднее, передискретизация
//!    в 16 кГц как `torchaudio.functional.resample` (`audio_io.load_audio`);
//! 2. log-mel Whisper (128 полос) по ВСЕЙ длине, без добивки до 30 с,
//!    `len / 160` кадров, затем округление в bf16 (`mel_dtype`);
//! 3. куски по `2·n_window` = 400 кадров (4 с), каждый отдельно: три
//!    Conv2d 3×3/2 + GELU (время и частоты ÷8), `stem_proj` 480·16 → 1280,
//!    синусоидальные позиции с нуля, 32 слоя Whisper, LayerNorm; хвостовой
//!    кусок короче — это то же, что паддинг с маской у оригинала;
//! 4. deepstack — выходы слоёв энкодера 8/16/24 (до LayerNorm), каждый —
//!    своим `GatedMLP` в скрытый размер LLM; прибавляются к аудиопозициям
//!    после блоков LLM 0/1/2; основной выход — `audio_adapter`;
//! 5. 12,5 токена в секунду; в промпте `<|audio_bos|>` (`<|system|>`),
//!    токены `<|AUDIO|>` (`<|vision_pad|>`) с метками секунд цифрами через
//!    каждые 2 с (`enable_time_marker`), `<|audio_eos|>` (`<|user|>`).
//!
//! Башня (~0,85 млрд параметров, BF16) грузится на время кодирования.

use std::path::Path;

use synaptix_asr_whisper::mel::whisper_log_mel;
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

const SR: u32 = 16_000;
/// Аудиотокенов в секунду (8 × 10 мс).
const TOKENS_PER_SECOND: f64 = 12.5;
/// Метка времени каждые 2 с.
const MARKER_EVERY: usize = 2;

/// `audio_config` MOSS-Audio.
#[derive(Debug, Clone)]
pub struct MossAudioConfig {
    pub whisper: WhisperConfig,
    pub downsample_hidden: usize,
    pub deepstack_layers: Vec<usize>,
    pub deepstack_inject: usize,
    pub n_window: usize,
    pub eps: f32,
}

impl MossAudioConfig {
    pub fn from_config_json(bytes: &[u8]) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        if v.get("model_type").and_then(|m| m.as_str()) != Some("moss_audio") {
            return None;
        }
        let a = v.get("audio_config")?;
        let whisper: WhisperConfig = serde_json::from_value(a.clone()).ok()?;
        let num = |k: &str| a.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let deepstack_layers: Vec<usize> = a
            .get("deepstack_encoder_layer_indexes")
            .and_then(|x| x.as_array())
            .map(|l| l.iter().filter_map(|i| i.as_u64().map(|i| i as usize)).collect())
            .unwrap_or_default();
        let inject = v
            .get("deepstack_num_inject_layers")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(deepstack_layers.len())
            .min(deepstack_layers.len());
        Some(Self {
            whisper,
            downsample_hidden: num("downsample_hidden_size")?,
            deepstack_layers,
            deepstack_inject: inject,
            n_window: num("n_window").unwrap_or(200),
            eps: a.get("layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
        })
    }
}

/// Звук → 16 кГц моно, как `audio_io.load_audio` (torchaudio): исходная
/// частота и каналы через ffmpeg (f32 без пересчёта), среднее каналов,
/// передискретизация torchaudio.
pub fn load_audio(path: &Path) -> Res<Vec<f32>> {
    let probe = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .map_err(|e| PipelineError::Load(format!("ffprobe: {e}")))?;
    let text = String::from_utf8_lossy(&probe.stdout);
    let mut it = text.trim().split(',');
    let rate: u32 = it.next().and_then(|v| v.trim().parse().ok()).ok_or_else(|| PipelineError::Load(format!("в {} нет звука", path.display())))?;
    let channels: usize = it.next().and_then(|v| v.trim().parse().ok()).unwrap_or(1).max(1);
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-vn", "-map", "0:a:0", "-f", "f32le", "-"])
        .output()
        .map_err(|e| PipelineError::Load(format!("ffmpeg: {e}")))?;
    if !out.status.success() {
        return Err(PipelineError::Load(format!("ffmpeg: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    let raw: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let mono: Vec<f32> = raw
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect();
    if mono.is_empty() {
        return Err(PipelineError::Load("звук пустой".into()));
    }
    synaptix_audio::resample_torchaudio(&mono, rate, SR).map_err(err("resample"))
}

/// Log-mel Whisper по всей длине (`_np_extract_fbank_features`): `[128, T]`,
/// `T = len / 160`, значения округлены до bf16.
pub fn mel(samples: &[f32], n_mels: usize) -> Res<(Vec<f32>, usize)> {
    let frames = samples.len() / 160;
    if frames == 0 {
        return Err(PipelineError::Load("звук короче 10 мс".into()));
    }
    let (mut m, _, t) = whisper_log_mel(samples, n_mels, frames).map_err(err("mel"))?;
    for v in &mut m {
        // Округление до bf16 к ближайшему чётному, как `.to(torch.bfloat16)`.
        let b = v.to_bits();
        *v = f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000);
    }
    Ok((m, t))
}

fn conv_len(l: usize) -> usize {
    (l - 1) / 2 + 1
}

/// Аудиотокенов на `frames` кадров mel (три свёртки с шагом 2).
pub fn tokens_for_frames(frames: usize) -> usize {
    conv_len(conv_len(conv_len(frames)))
}

/// `_build_audio_tokens_with_time_markers`: токены звука с метками секунд
/// (цифрами) через каждые 2 с — строка для промпта.
pub fn prompt_block(tokens: usize) -> String {
    let full = (tokens as f64 / TOKENS_PER_SECOND) as usize;
    let every = (TOKENS_PER_SECOND * MARKER_EVERY as f64) as usize;
    let mut s = String::from("<|system|>");
    let mut used = 0usize;
    let mut second = MARKER_EVERY;
    while second <= full {
        let at = (second / MARKER_EVERY) * every;
        if at > used {
            s.push_str(&"<|vision_pad|>".repeat(at - used));
            used = at;
        }
        s.push_str(&second.to_string());
        second += MARKER_EVERY;
    }
    if tokens > used {
        s.push_str(&"<|vision_pad|>".repeat(tokens - used));
    }
    s.push_str("<|user|>");
    s
}

struct Gated {
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl Gated {
    fn load(w: &WhisperWeights, p: &str) -> Res<Self> {
        let l = |n: &str| -> Res<QuantLinear> {
            let t = w.get(&format!("{p}.{n}.weight")).map_err(err("moss adapter"))?;
            QuantLinear::dense(t, None).map_err(err("moss adapter"))
        };
        Ok(Self { gate: l("gate_proj")?, up: l("up_proj")?, down: l("down_proj")? })
    }
    fn forward(&self, x: &Tensor) -> Res<Tensor> {
        let g = self.gate.forward(x).and_then(|t| t.silu()).map_err(err("moss adapter"))?;
        let u = self.up.forward(x).map_err(err("moss adapter"))?;
        self.down.forward(&g.mul(&u).map_err(err("moss adapter"))?).map_err(err("moss adapter"))
    }
}

struct Tower {
    conv: Vec<(Tensor, Tensor)>,
    stem: QuantLinear,
    layers: Vec<EncoderLayer>,
    ln: (Tensor, Tensor),
    eps: f32,
    adapter: Gated,
    mergers: Vec<Gated>,
    taps: Vec<usize>,
    d_model: usize,
    device: Device,
    dtype: DType,
}

impl Tower {
    fn load(bundle: &Path, cfg: &MossAudioConfig, device: Device, dtype: DType) -> Res<Self> {
        let w = WhisperWeights::open_encoder(bundle, cfg.whisper.clone(), "audio_encoder.", device, dtype)
            .map_err(err("moss encoder"))?;
        let g = |n: &str| w.get(n).map_err(err("moss encoder"));
        let conv = (1..=3)
            .map(|i| Ok((g(&format!("audio_encoder.conv{i}.weight"))?, g(&format!("audio_encoder.conv{i}.bias"))?)))
            .collect::<Res<Vec<_>>>()?;
        let stem = QuantLinear::dense(g("audio_encoder.stem_proj.weight")?, Some(g("audio_encoder.stem_proj.bias")?))
            .map_err(err("moss encoder"))?;
        let layers = (0..cfg.whisper.encoder_layers)
            .map(|i| EncoderLayer::load_index(&w, i).map_err(err("moss encoder")))
            .collect::<Res<Vec<_>>>()?;
        let ln = (g("audio_encoder.layer_norm.weight")?, g("audio_encoder.layer_norm.bias")?);
        let adapter = Gated::load(&w, "audio_adapter")?;
        let mergers = (0..cfg.deepstack_inject)
            .map(|i| Gated::load(&w, &format!("deepstack_audio_merger_list.{i}")))
            .collect::<Res<Vec<_>>>()?;
        Ok(Self {
            conv,
            stem,
            layers,
            ln,
            eps: cfg.eps,
            adapter,
            mergers,
            taps: cfg.deepstack_layers[..cfg.deepstack_inject].to_vec(),
            d_model: cfg.whisper.d_model,
            device,
            dtype,
        })
    }

    /// Синусоидальные позиции `SinusoidsPositionEmbedding`: `[sin | cos]`.
    fn positions(&self, t: usize) -> Res<Tensor> {
        let d = self.d_model;
        let half = d / 2;
        let inc = (10000f32).ln() / (half as f32 - 1.0);
        let mut v = vec![0f32; t * d];
        for i in 0..t {
            for j in 0..half {
                let a = i as f32 * (-inc * j as f32).exp();
                v[i * d + j] = a.sin();
                v[i * d + half + j] = a.cos();
            }
        }
        Tensor::from_vec(v, vec![1, t, d], self.device)
            .and_then(|x| x.to_dtype(self.dtype))
            .map_err(err("moss pos"))
    }

    /// Кусок mel `[n_mels, T]` → (выход `[T/8, d]`, deepstack-отводы).
    fn chunk(&self, mel: &[f32], n_mels: usize, t: usize) -> Res<(Tensor, Vec<Tensor>)> {
        let e = err("moss encoder");
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
            .and_then(|x| x.reshape(vec![1, tt, c * f]))
            .map_err(&e)?;
        let mut h = self.stem.forward(&x).map_err(&e)?;
        h = h.add(&self.positions(tt)?).map_err(&e)?;
        let mut taps = Vec::with_capacity(self.taps.len());
        for (i, l) in self.layers.iter().enumerate() {
            h = l.run(&h).map_err(err("moss encoder"))?;
            if self.taps.contains(&i) {
                taps.push(h.reshape(vec![tt, self.d_model]).map_err(&e)?);
            }
        }
        let h = layer_norm(&h, Some(&self.ln.0), Some(&self.ln.1), self.eps).map_err(&e)?;
        Ok((h.reshape(vec![tt, self.d_model]).map_err(&e)?, taps))
    }
}

/// Звук MOSS-Audio: эмбеддинги для заполнителей `[n, hidden]`, deepstack по
/// слою `[n, hidden]` и блок промпта.
pub struct MossAudioEncoded {
    pub embeds: Tensor,
    pub deepstack: Vec<Tensor>,
    pub tokens: usize,
    pub prompt_block: String,
}

fn encode_on(bundle: &Path, cfg: &MossAudioConfig, mel: &[f32], frames: usize, device: Device, dtype: DType) -> Res<(Tensor, Vec<Tensor>)> {
    use synaptix_core::grad::no_grad;
    let n_mels = cfg.whisper.num_mel_bins;
    let tower = Tower::load(bundle, cfg, device, dtype)?;
    let chunk = cfg.n_window * 2;
    let mut outs = Vec::new();
    let mut taps: Vec<Vec<Tensor>> = vec![Vec::new(); tower.taps.len()];
    let mut start = 0usize;
    while start < frames {
        let len = chunk.min(frames - start);
        let mut part = vec![0f32; n_mels * len];
        for m in 0..n_mels {
            part[m * len..(m + 1) * len].copy_from_slice(&mel[m * frames + start..m * frames + start + len]);
        }
        let (o, t) = no_grad(|| tower.chunk(&part, n_mels, len))?;
        outs.push(o);
        for (k, x) in t.into_iter().enumerate() {
            taps[k].push(x);
        }
        start += len;
    }
    let cat = |v: &[Tensor]| -> Res<Tensor> {
        let r: Vec<&Tensor> = v.iter().collect();
        Tensor::cat(&r, 0).map_err(err("moss cat"))
    };
    let main = no_grad(|| tower.adapter.forward(&cat(&outs)?))?;
    let mut ds = Vec::with_capacity(taps.len());
    for (k, t) in taps.iter().enumerate() {
        ds.push(no_grad(|| tower.mergers[k].forward(&cat(t)?))?);
    }
    Ok((main, ds))
}

/// Звук → эмбеддинги MOSS-Audio. Башня грузится из `bundle` на время
/// вызова; на CUDA при нехватке памяти — CPU F32 и перенос на `device`.
pub fn encode(bundle: &Path, cfg: &MossAudioConfig, audio: &Path, device: Device) -> Res<MossAudioEncoded> {
    let samples = load_audio(audio)?;
    let (m, frames) = mel(&samples, cfg.whisper.num_mel_bins)?;
    let force_cpu = std::env::var_os("SYN_MOSS_AUDIO_CPU").is_some();
    let (main, ds) = match device {
        Device::Cuda(o) if !force_cpu => match encode_on(bundle, cfg, &m, frames, device, DType::BF16) {
            Ok(v) => v,
            Err(e) if e.to_string().to_lowercase().contains("memory") || e.to_string().contains("alloc") => {
                eprintln!("[moss-audio] не влезло в VRAM ({e}) — кодирую на CPU");
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
    let main = main.to_device(device).map_err(err("moss to device"))?;
    let ds = ds.into_iter().map(|t| t.to_device(device).map_err(err("moss to device"))).collect::<Res<Vec<_>>>()?;
    let tokens = main.dims()[0];
    if tokens != tokens_for_frames(frames) {
        return Err(PipelineError::Forward(format!("moss: {tokens} токенов вместо {}", tokens_for_frames(frames))));
    }
    // Сверка с эталоном: `SYN_MOSS_AUDIO_DUMP=путь` → pcm, mel, emb, ds0..
    if let Some(dump) = std::env::var_os("SYN_MOSS_AUDIO_DUMP") {
        let dump = std::path::PathBuf::from(dump);
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let host = |t: &Tensor| -> Res<Vec<f32>> {
            t.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()).map_err(err("dump"))
        };
        let _ = std::fs::write(dump.with_extension("pcm.f32"), bytes(&samples));
        let _ = std::fs::write(dump.with_extension("mel.f32"), bytes(&m));
        let _ = std::fs::write(dump.with_extension("emb.f32"), bytes(&host(&main)?));
        for (k, t) in ds.iter().enumerate() {
            let _ = std::fs::write(dump.with_extension(format!("ds{k}.f32")), bytes(&host(t)?));
        }
    }
    Ok(MossAudioEncoded { embeds: main, deepstack: ds, tokens, prompt_block: prompt_block(tokens) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_like_reference() {
        // 5 с = 62 токена (12,5/с): метки «2» после 25, «4» после 50, хвост 12.
        let b = prompt_block(62);
        let pads = |n: usize| "<|vision_pad|>".repeat(n);
        assert_eq!(b, format!("<|system|>{}2{}4{}<|user|>", pads(25), pads(25), pads(12)));
        assert_eq!(tokens_for_frames(400), 50);
        assert_eq!(tokens_for_frames(401), 51);
    }
}
