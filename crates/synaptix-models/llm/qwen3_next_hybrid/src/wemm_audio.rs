//! Аудио-башня WeMM-Video (`model.audio.*`): энкодер Whisper-large и
//! Q-Former BERT поверх него — как `h5_modeling_whisper.WhisperEncoder` и
//! `processing_wemm.py` / `wemm_media.py` оригинала.
//!
//! Путь звука:
//! 1. декодирование ffmpeg в 16 кГц моно f32 (у оригинала — torchcodec
//!    `AudioDecoder(sample_rate=16000, num_channels=1)` на том же
//!    ffmpeg/swresample; стерео сводится в моно — панорамы модель не видит);
//!    отдельный звук ещё проходит через PCM16 (WAV), как в оригинале;
//! 2. короче 1 с — добивается нулями до 1 с; режется на куски по 30 с,
//!    каждый — log-mel Whisper (128 полос, 3000 кадров, хвост нулями);
//! 3. энкодер Whisper → `[1500, 1280]` на кусок, финальный LayerNorm;
//! 4. окна по `second_per_window` (0,1 с = 5 кадров энкодера, шаг столько
//!    же) → один запрос Q-Former на окно: self-attn по одному токену
//!    (softmax по одному ключу = 1, выход — value-проекция), cross-attn к
//!    кадрам окна, FFN `intermediate_query`/`output_query`; `audio_proj` →
//!    скрытый размер LLM. 300 токенов на 30-секундный кусок, хвост куска —
//!    тишина паддинга, её токены тоже идут в промпт (как у оригинала);
//! 5. блок промпта: `<|audio_start|>` + по группе на `timestamp_interval_sec`
//!    (1 с) `<{t:.1} seconds>` + её `<|audio_pad|>` + `<|audio_end|>`;
//!    токены паддинга дописываются к последней группе.
//!
//! Веса — BF16 (квантование бандла их не трогает). Башня грузится на время
//! кодирования и сразу выгружается: рядом с 35B-A3B на карте ~2,6 ГБ, а
//! энкодер и Q-Former по очереди занимают ≤1,3 ГБ. Не влезло — CPU F32.

use std::path::Path;

use synaptix_asr_whisper::mel::whisper_log_mel;
use synaptix_asr_whisper::{WhisperConfig, WhisperEncoder, WhisperWeights};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::activation::gelu_exact;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::pipeline::PipelineError;

const SR: usize = 16_000;
/// Кусок Whisper — 30 с.
const SEGMENT: usize = 30 * SR;
const PREFIX: &str = "model.audio.";

type Res<T> = Result<T, PipelineError>;

fn err<E: std::fmt::Display>(ctx: &'static str) -> impl Fn(E) -> PipelineError {
    move |e| PipelineError::Forward(format!("{ctx}: {e}"))
}

/// `audio_config` из config.json WeMM-Video.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub whisper: WhisperConfig,
    pub second_per_window: f64,
    pub second_stride: f64,
    pub q_hidden: usize,
    pub q_heads: usize,
    pub q_layers: usize,
    pub q_eps: f32,
    pub query_length: usize,
    /// Шаг меток времени в промпте, секунд (`timestamp_interval_sec`).
    pub timestamp_interval: f64,
}

impl AudioConfig {
    /// `None` — в config.json нет `audio_config` (не WeMM).
    pub fn from_config_json(bytes: &[u8]) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        let a = v.get("audio_config")?;
        let whisper: WhisperConfig = serde_json::from_value(a.clone()).ok()?;
        let q = a.get("qformer_config")?;
        let num = |o: &serde_json::Value, k: &str| o.get(k).and_then(|x| x.as_f64());
        Some(Self {
            second_per_window: num(a, "second_per_window").unwrap_or(0.1),
            second_stride: num(a, "second_stride").or(num(a, "second_per_window")).unwrap_or(0.1),
            q_hidden: num(q, "hidden_size")? as usize,
            q_heads: num(q, "num_attention_heads")? as usize,
            q_layers: num(q, "num_hidden_layers")? as usize,
            q_eps: num(q, "layer_norm_eps").unwrap_or(1e-12) as f32,
            query_length: num(q, "query_length").unwrap_or(1.0) as usize,
            timestamp_interval: 1.0,
            whisper,
        })
    }

    fn frames(&self) -> usize {
        self.whisper.max_source_positions
    }

    /// (окно, шаг) в кадрах энкодера — `round(1500·sec/30)`.
    fn kernel_step(&self) -> (usize, usize) {
        let f = self.frames() as f64;
        let k = (f * self.second_per_window / 30.0).round() as usize;
        let s = ((f * self.second_stride / 30.0).round() as usize).max(1);
        (k.max(1), s)
    }

    /// Токенов на 30-секундный кусок (`wemm_media.tokens_per_segment`).
    pub fn tokens_per_segment(&self) -> usize {
        let (k, s) = self.kernel_step();
        ((self.frames() - k) / s + 1) * self.query_length
    }
}

/// Раскладка аудиотокенов в промпте.
#[derive(Debug, Clone)]
pub struct AudioPromptInfo {
    /// Длительность звука, секунд.
    pub duration: f64,
    /// `<|audio_pad|>` в каждой группе меток времени.
    pub groups: Vec<usize>,
    pub times: Vec<f64>,
}

impl AudioPromptInfo {
    pub fn tokens(&self) -> usize {
        self.groups.iter().sum()
    }

    /// `<|audio_start|><0.0 seconds><|audio_pad|>…<1.0 seconds>…<|audio_end|>`.
    pub fn prompt_block(&self) -> String {
        let mut s = String::from("<|audio_start|>");
        for (t, n) in self.times.iter().zip(&self.groups) {
            s.push_str(&format!("<{t:.1} seconds>"));
            s.push_str(&"<|audio_pad|>".repeat(*n));
        }
        s.push_str("<|audio_end|>");
        s
    }
}

/// `wemm_media.audio_timestamp_groups`: метки через `interval` по реальной
/// длительности, действительные токены поровну по группам, паддинг куска —
/// в последнюю группу.
pub fn timestamp_groups(count: usize, window: f64, duration: f64, interval: f64) -> (Vec<usize>, Vec<f64>) {
    if count == 0 {
        return (Vec::new(), Vec::new());
    }
    let padded = count as f64 * window;
    let duration = duration.min(padded);
    let effective = ((duration / window - 1e-9).ceil() as usize).clamp(1, count);
    let fps = 1.0 / interval;
    let groups = ((duration * fps - 1e-9).ceil() as usize).max(1);
    let (base, rem) = (effective / groups, effective % groups);
    let mut sizes = vec![base; groups];
    if rem > 0 {
        let step = groups as f64 / rem as f64;
        for i in 1..=rem {
            let idx = ((i as f64 * step).floor() as usize).saturating_sub(1).min(groups - 1);
            sizes[idx] += 1;
        }
    }
    sizes[groups - 1] += count - effective;
    let times = (0..groups).map(|i| i as f64 / fps).collect();
    (sizes, times)
}

/// Звук любого формата → 16 кГц моно f32 (ffmpeg, как torchcodec у оригинала).
pub fn decode_audio(path: &Path) -> Res<Vec<f32>> {
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .output()
        .map_err(|e| PipelineError::Load(format!("ffmpeg: {e}")))?;
    if !out.status.success() {
        return Err(PipelineError::Load(format!(
            "ffmpeg: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let samples: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if samples.is_empty() || !samples.iter().all(|x| x.is_finite()) {
        return Err(PipelineError::Load("звук пустой или с NaN".into()));
    }
    Ok(samples)
}

/// Отдельный звук у оригинала проходит через WAV PCM16 (`pcm16_bytes` →
/// `sf.read`): libsndfile пишет `lrint(x·32767)` и читает `/32768`.
pub fn pcm16_roundtrip(samples: &mut [f32]) {
    for x in samples {
        let q = (*x * 32767.0).round().clamp(-32768.0, 32767.0);
        *x = q / 32768.0;
    }
}

/// Log-mel всех 30-секундных кусков `[n_seg][128·3000]`.
pub fn mel_segments(samples: &[f32], cfg: &AudioConfig) -> Res<Vec<Vec<f32>>> {
    let mut s = samples.to_vec();
    if s.len() < SR {
        s.resize(SR, 0.0);
    }
    let target = cfg.frames() * 2;
    let mut out = Vec::new();
    for chunk in s.chunks(SEGMENT) {
        let mut seg = chunk.to_vec();
        seg.resize(SEGMENT, 0.0);
        let (mel, _, frames) =
            whisper_log_mel(&seg, cfg.whisper.num_mel_bins, target).map_err(err("mel"))?;
        if frames != target {
            return Err(PipelineError::Forward(format!("mel: {frames} кадров вместо {target}")));
        }
        out.push(mel);
    }
    Ok(out)
}

struct Ln {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl Ln {
    fn load(wts: &WhisperWeights, p: &str, eps: f32) -> Res<Self> {
        let w = wts.get(&format!("{p}.gamma")).map_err(err("qformer"))?;
        let b = wts.get(&format!("{p}.beta")).map_err(err("qformer"))?;
        Ok(Self { w, b, eps })
    }
    fn fwd(&self, x: &Tensor) -> Res<Tensor> {
        layer_norm(x, Some(&self.w), Some(&self.b), self.eps).map_err(err("qformer ln"))
    }
}

fn linear(wts: &WhisperWeights, p: &str) -> Res<QuantLinear> {
    let w = wts.get(&format!("{p}.weight")).map_err(err("qformer"))?;
    let b = wts.get(&format!("{p}.bias")).map_err(err("qformer"))?;
    QuantLinear::dense(w, Some(b)).map_err(err("qformer"))
}

fn lin(l: &QuantLinear, x: &Tensor) -> Res<Tensor> {
    l.forward(x).map_err(err("qformer linear"))
}

struct QLayer {
    sa_v: QuantLinear,
    sa_o: QuantLinear,
    sa_ln: Ln,
    ca_q: QuantLinear,
    ca_k: QuantLinear,
    ca_v: QuantLinear,
    ca_o: QuantLinear,
    ca_ln: Ln,
    ff_in: QuantLinear,
    ff_out: QuantLinear,
    ff_ln: Ln,
}

struct QFormer {
    query: Tensor,
    emb_ln: Ln,
    layers: Vec<QLayer>,
    proj: QuantLinear,
    heads: usize,
    hidden: usize,
}

impl QFormer {
    fn load(w: &WhisperWeights, cfg: &AudioConfig) -> Res<Self> {
        let e = cfg.q_eps;
        let b = format!("{PREFIX}qformer.bert");
        let mut layers = Vec::with_capacity(cfg.q_layers);
        for i in 0..cfg.q_layers {
            let l = format!("{b}.encoder.layer.{i}");
            layers.push(QLayer {
                // query/key self-attn не нужны: один запрос, softmax ≡ 1.
                sa_v: linear(w, &format!("{l}.attention.self.value"))?,
                sa_o: linear(w, &format!("{l}.attention.output.dense"))?,
                sa_ln: Ln::load(w, &format!("{l}.attention.output.LayerNorm"), e)?,
                ca_q: linear(w, &format!("{l}.crossattention.self.query"))?,
                ca_k: linear(w, &format!("{l}.crossattention.self.key"))?,
                ca_v: linear(w, &format!("{l}.crossattention.self.value"))?,
                ca_o: linear(w, &format!("{l}.crossattention.output.dense"))?,
                ca_ln: Ln::load(w, &format!("{l}.crossattention.output.LayerNorm"), e)?,
                ff_in: linear(w, &format!("{l}.intermediate_query.dense"))?,
                ff_out: linear(w, &format!("{l}.output_query.dense"))?,
                ff_ln: Ln::load(w, &format!("{l}.output_query.LayerNorm"), e)?,
            });
        }
        if cfg.query_length != 1 {
            return Err(PipelineError::Model(format!(
                "Q-Former с query_length={} не поддержан (у WeMM — 1)",
                cfg.query_length
            )));
        }
        Ok(Self {
            query: w.get(&format!("{PREFIX}q_tokens")).map_err(err("qformer"))?,
            emb_ln: Ln::load(w, &format!("{b}.embeddings.LayerNorm"), e)?,
            layers,
            proj: linear(w, &format!("{PREFIX}audio_proj"))?,
            heads: cfg.q_heads,
            hidden: cfg.q_hidden,
        })
    }

    /// `windows` — `[N, k, d_enc]` кадров энкодера по окнам → `[N, out_hidden]`.
    fn forward(&self, windows: &Tensor) -> Res<Tensor> {
        let d = windows.dims().to_vec();
        let (n, k, de) = (d[0], d[1], d[2]);
        let (h, hd) = (self.heads, self.hidden / self.heads);
        let scale = (hd as f32).powf(-0.5);
        let q0 = self
            .emb_ln
            .fwd(&self.query.reshape(vec![1, self.hidden]).map_err(err("qformer"))?)?;
        let zeros = Tensor::zeros(vec![n, self.hidden], q0.dtype(), q0.device()).map_err(err("qformer"))?;
        let mut x = zeros.broadcast_add(&q0).map_err(err("qformer"))?;
        let enc = windows.reshape(vec![n * k, de]).map_err(err("qformer"))?;
        for l in &self.layers {
            let sa = lin(&l.sa_o, &lin(&l.sa_v, &x)?)?;
            let x1 = l.sa_ln.fwd(&sa.add(&x).map_err(err("qformer"))?)?;
            let q = lin(&l.ca_q, &x1)?
                .reshape(vec![n, h, 1, hd])
                .map_err(err("qformer"))?;
            let heads = |t: Tensor| -> Res<Tensor> {
                t.reshape(vec![n, k, h, hd])
                    .and_then(|t| t.permute(vec![0, 2, 1, 3]))
                    .and_then(|t| t.contiguous())
                    .map_err(err("qformer"))
            };
            let kk = heads(lin(&l.ca_k, &enc)?)?;
            let vv = heads(lin(&l.ca_v, &enc)?)?;
            let ctx = scaled_dot_attention(&q.contiguous().map_err(err("qformer"))?, &kk, &vv, scale, None)
                .map_err(err("qformer attn"))?
                .reshape(vec![n, self.hidden])
                .map_err(err("qformer"))?;
            let ca = lin(&l.ca_o, &ctx)?;
            let x2 = l.ca_ln.fwd(&ca.add(&x1).map_err(err("qformer"))?)?;
            let f = gelu_exact(&lin(&l.ff_in, &x2)?).map_err(err("qformer gelu"))?;
            let f = lin(&l.ff_out, &f)?;
            x = l.ff_ln.fwd(&f.add(&x2).map_err(err("qformer"))?)?;
        }
        lin(&self.proj, &x)
    }
}

/// Эмбеддинги звука `[tokens, hidden]` (F32) на `device`/`dtype`.
fn encode_on(
    bundle: &Path,
    cfg: &AudioConfig,
    mels: &[Vec<f32>],
    device: Device,
    dtype: DType,
) -> Res<Tensor> {
    use synaptix_core::grad::no_grad;
    let (kernel, step) = cfg.kernel_step();
    if kernel != step {
        return Err(PipelineError::Model(format!(
            "окна Q-Former с перекрытием (окно {kernel}, шаг {step}) не поддержаны"
        )));
    }
    let n_mels = cfg.whisper.num_mel_bins;
    let frames = cfg.frames() * 2;
    // 1) энкодер Whisper по кускам; веса выгружаются до Q-Former.
    let hidden: Vec<Tensor> = {
        let w = WhisperWeights::open_encoder(bundle, cfg.whisper.clone(), PREFIX, device, dtype)
            .map_err(err("whisper"))?;
        let enc = WhisperEncoder::load(&w).map_err(err("whisper"))?;
        let mut out = Vec::with_capacity(mels.len());
        for mel in mels {
            let t = Tensor::from_vec(mel.clone(), vec![1, n_mels, frames], device)
                .and_then(|t| t.to_dtype(dtype))
                .map_err(err("whisper"))?;
            let h = no_grad(|| enc.forward(&t)).map_err(err("whisper forward"))?;
            out.push(h);
        }
        out
    };
    // 2) Q-Former по окнам всех кусков разом.
    let w = WhisperWeights::open_encoder(bundle, cfg.whisper.clone(), PREFIX, device, dtype)
        .map_err(err("qformer"))?;
    let q = QFormer::load(&w, cfg)?;
    let per = cfg.frames() / kernel;
    let mut parts = Vec::with_capacity(hidden.len());
    for h in &hidden {
        let d = h.dims()[2];
        let win = h
            .narrow(1, 0, per * kernel)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![per, kernel, d]))
            .map_err(err("qformer"))?;
        let e = no_grad(|| q.forward(&win))?;
        parts.push(e.to_dtype(DType::F32).map_err(err("qformer"))?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0).map_err(err("qformer"))
}

/// Звук → эмбеддинги `[tokens, hidden]` и раскладка промпта. Башня
/// грузится из `bundle` на время вызова; на CUDA при нехватке памяти —
/// повтор на CPU в F32.
pub fn encode_audio(
    bundle: &Path,
    cfg: &AudioConfig,
    audio: &Path,
    device: Device,
) -> Res<(Tensor, AudioPromptInfo)> {
    let mut samples = decode_audio(audio)?;
    let duration = samples.len() as f64 / SR as f64;
    pcm16_roundtrip(&mut samples);
    let mels = mel_segments(&samples, cfg)?;
    // `SYN_WEMM_AUDIO_CPU=1` — кодировать на CPU (проверка запасного пути).
    let force_cpu = std::env::var_os("SYN_WEMM_AUDIO_CPU").is_some();
    let embeds = match device {
        Device::Cuda(o) if !force_cpu => match encode_on(bundle, cfg, &mels, device, DType::BF16) {
            Ok(e) => e,
            Err(e) if e.to_string().to_lowercase().contains("memory") || e.to_string().contains("alloc") => {
                eprintln!("[wemm-audio] не влезло в VRAM ({e}) — кодирую на CPU");
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
                encode_on(bundle, cfg, &mels, Device::Cpu, DType::F32)?
            }
            Err(e) => return Err(e),
        },
        _ => encode_on(bundle, cfg, &mels, Device::Cpu, DType::F32)?,
    };
    if let Device::Cuda(o) = device {
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
    }
    // После запасного пути на CPU эмбеддинги — на устройство LLM (иначе
    // префилл падал «device mismatch: lhs=Cuda(0), rhs=Cpu»).
    let embeds = embeds.to_device(device).map_err(err("audio to device"))?;
    // Сверка с эталоном: `SYN_WEMM_AUDIO_DUMP=путь` → `путь.pcm.f32`
    // (16 кГц после PCM16) и `путь.emb.f32` (`[tokens, hidden]`).
    if let Some(dump) = std::env::var_os("SYN_WEMM_AUDIO_DUMP") {
        let dump = std::path::PathBuf::from(dump);
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let emb: Vec<f32> = embeds.flatten_all().and_then(|t| t.to_vec1()).map_err(err("dump"))?;
        let _ = std::fs::write(dump.with_extension("pcm.f32"), bytes(&samples));
        let _ = std::fs::write(dump.with_extension("emb.f32"), bytes(&emb));
        eprintln!("[wemm-audio] дамп → {}.{{pcm,emb}}.f32", dump.display());
    }
    let count = mels.len() * cfg.tokens_per_segment();
    if embeds.dims()[0] != count {
        return Err(PipelineError::Forward(format!(
            "аудио: {} эмбеддингов вместо {count}",
            embeds.dims()[0]
        )));
    }
    let (groups, times) = timestamp_groups(count, cfg.second_per_window, duration, cfg.timestamp_interval);
    Ok((embeds, AudioPromptInfo { duration, groups, times }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_match_reference() {
        // wemm_media.audio_timestamp_groups(300, 0.1, 2.5, 1.0):
        // groups=3, effective=25 → [9, 8, 8] → балансировка остатка 1 → idx 2,
        // плюс 275 паддинга в последнюю.
        let (s, t) = timestamp_groups(300, 0.1, 2.5, 1.0);
        assert_eq!(t, vec![0.0, 1.0, 2.0]);
        assert_eq!(s.iter().sum::<usize>(), 300);
        assert_eq!(s, vec![8, 8, 9 + 275]);
        let (s, _) = timestamp_groups(300, 0.1, 0.8, 1.0);
        assert_eq!(s, vec![300]);
    }
}
