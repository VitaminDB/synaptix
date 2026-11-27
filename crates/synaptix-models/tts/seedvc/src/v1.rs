//! Seed-VC v1 `seed-uvit-whisper-small-wavenet` (офлайн-преобразование тембра,
//! 22,05 кГц; эталон — `inference.py`, `modules/diffusion_transformer.py`,
//! `modules/flow_matching.py`, `modules/wavenet.py`).
//!
//! Содержание — последний слой энкодера Whisper-small (768, 50 Гц) → регулятор
//! длины (Linear 768→512, растяжение, 4 × Conv/GroupNorm/Mish, Conv 1×1) →
//! DiT: вход кадра `[x | мел образца | условие | вектор диктора]` (864→512),
//! 13 блоков с модуляцией норм по времени и U-ViT-скипами (слои 0…5 отдают,
//! 7…12 принимают), длинный скип от входа, WaveNet-постнет (8 слоёв, k5) →
//! скорость `[B, 80, T]`. Сэмплер — Эйлер по равномерной сетке, CFG одной
//! безусловной веткой. Вектор диктора CAM++ и BigVGAN — те же, что в v2.

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_asr_whisper::mel::whisper_log_mel;
use synaptix_asr_whisper::{WhisperEncoder, WhisperWeights};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::activation::mish::mish;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d::conv1d;
use synaptix_ops::norm::group_norm::group_norm;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::campplus::{fbank, CamPlus, EMBED};
use crate::loader::Comp;
use crate::mel::mel_spectrogram;
use crate::nn::{apply_rope, host, rms_norm, swiglu, Lin, Rope};
use crate::pipeline::{resample, Progress, SeedVcOutput, Stitcher, Waves, DIT_CONTEXT_S, MAX_REFERENCE_S, OVERLAP_FRAMES};
use crate::vocoder::Vocoder;
use crate::{SeedVcError, HOP, N_MELS, R, SAMPLE_RATE};

const DIM: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const EPS: f32 = 1e-5;
const FREQ_DIM: usize = 256;
const WN_LAYERS: usize = 8;
/// Кадров Whisper в секунду и сэмплов 16 кГц на кадр.
const CONTENT_HOP: usize = 320;
const WINDOW_S: usize = 30;
const OVERLAP_S: usize = 5;

/// Энкодер Whisper-small: волна 16 кГц → признаки `[1, T, 768]`.
pub struct Content {
    enc: WhisperEncoder,
    n_mels: usize,
    frames: usize,
    device: Device,
}

impl Content {
    pub fn load(bundle: &Path, device: Device) -> R<Self> {
        let w = WhisperWeights::open_component(bundle, "whisper", "whisper/config.json", device, DType::F32).map_err(|e| SeedVcError::Load(format!("whisper: {e}")))?;
        let enc = WhisperEncoder::load(&w).map_err(|e| SeedVcError::Load(format!("whisper: {e}")))?;
        Ok(Self { enc, n_mels: w.config.num_mel_bins, frames: w.config.max_source_positions * 2, device })
    }

    /// Не длиннее 30 с: дополнить нулями до 30 с, энкодер, первые `N / 320 + 1` кадров.
    pub fn features(&self, wave_16k: &[f32]) -> R<Tensor> {
        let e = |e: synaptix_asr_whisper::WhisperError| SeedVcError::Other(format!("whisper: {e}"));
        let total = self.frames * 160;
        let mut a = wave_16k[..wave_16k.len().min(total)].to_vec();
        a.resize(total, 0.0);
        let (mel, n_mels, n_frames) = whisper_log_mel(&a, self.n_mels, self.frames).map_err(e)?;
        let h = self.enc.forward(&Tensor::from_vec(mel, vec![1, n_mels, n_frames], self.device)?).map_err(e)?;
        let keep = (wave_16k.len() / CONTENT_HOP + 1).min(h.dims()[1]);
        Ok(h.narrow(1, 0, keep)?.contiguous()?)
    }

    /// Любая длина: окна 30 с с перекрытием 5 с, как в `inference.py`.
    pub fn features_long(&self, wave_16k: &[f32]) -> R<Tensor> {
        let sr = 16_000usize;
        let n = wave_16k.len();
        if n <= sr * WINDOW_S {
            return self.features(wave_16k);
        }
        let mut parts: Vec<Tensor> = Vec::new();
        let mut pos = 0usize;
        let mut buffer: Vec<f32> = Vec::new();
        while pos < n {
            if parts.is_empty() {
                let chunk = &wave_16k[..sr * WINDOW_S];
                parts.push(self.features(chunk)?);
                buffer = chunk[chunk.len() - sr * OVERLAP_S..].to_vec();
                pos += sr * WINDOW_S;
            } else {
                let end = (pos + sr * (WINDOW_S - OVERLAP_S)).min(n);
                let mut chunk = buffer.clone();
                chunk.extend_from_slice(&wave_16k[pos..end]);
                let f = self.features(&chunk)?;
                let skip = (50 * OVERLAP_S).min(f.dims()[1]);
                if f.dims()[1] > skip {
                    parts.push(f.narrow(1, skip, f.dims()[1] - skip)?.contiguous()?);
                }
                buffer = chunk[chunk.len() - sr * OVERLAP_S..].to_vec();
                pos += chunk.len() - sr * OVERLAP_S;
            }
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Tensor::cat(&refs, 1)?.contiguous()?)
    }
}

struct Stage {
    w: Tensor,
    b: Tensor,
    gn_w: Tensor,
    gn_b: Tensor,
}

pub struct Regulator {
    proj: Lin,
    stages: Vec<Stage>,
    out_w: Tensor,
    out_b: Tensor,
    device: Device,
}

impl Regulator {
    pub fn load(c: &Comp) -> R<Self> {
        let p = "length_regulator";
        let mut stages = Vec::with_capacity(4);
        for i in 0..4 {
            let (cv, gn) = (3 * i, 3 * i + 1);
            stages.push(Stage {
                w: c.get(&format!("{p}.model.{cv}.weight"), DType::F32)?,
                b: c.get(&format!("{p}.model.{cv}.bias"), DType::F32)?,
                gn_w: c.get(&format!("{p}.model.{gn}.weight"), DType::F32)?,
                gn_b: c.get(&format!("{p}.model.{gn}.bias"), DType::F32)?,
            });
        }
        Ok(Self {
            proj: Lin::load(c, &format!("{p}.content_in_proj"), true)?,
            stages,
            out_w: c.get(&format!("{p}.model.12.weight"), DType::F32)?,
            out_b: c.get(&format!("{p}.model.12.bias"), DType::F32)?,
            device: c.device(),
        })
    }

    /// Признаки `[1, T, 768]` → условие `[1, frames, 512]`.
    pub fn forward(&self, feats: &Tensor, frames: usize) -> R<Tensor> {
        let x = self.proj.fwd(feats)?;
        let (n, d) = (x.dims()[1], x.dims()[2]);
        if n == 0 || frames == 0 {
            return Err(SeedVcError::Input("пустые признаки содержания".into()));
        }
        let v = host(&x)?;
        let scale = n as f32 / frames as f32;
        let mut y = vec![0f32; d * frames];
        for t in 0..frames {
            let src = ((t as f32 * scale).floor() as usize).min(n - 1);
            for ch in 0..d {
                y[ch * frames + t] = v[src * d + ch];
            }
        }
        let mut x = Tensor::from_vec(y, vec![1, d, frames], self.device)?;
        for s in &self.stages {
            x = conv1d(&x, &s.w, Some(&s.b), 1, 1)?;
            x = mish(&group_norm(&x, Some(&s.gn_w), Some(&s.gn_b), 1, 1e-5)?)?;
        }
        Ok(conv1d(&x, &self.out_w, Some(&self.out_b), 1, 0)?.transpose(1, 2)?.contiguous()?)
    }
}

/// `AdaptiveLayerNorm` v1: `weight · RMSNorm(x) + bias`, пара — Linear от эмбеддинга времени.
struct AdaNorm {
    project: Lin,
    norm: Tensor,
}

impl AdaNorm {
    fn load(c: &Comp, p: &str) -> R<Self> {
        Ok(Self { project: Lin::load(c, &format!("{p}.project_layer"), true)?, norm: c.get(&format!("{p}.norm.weight"), DType::F32)? })
    }

    fn fwd(&self, x: &Tensor, t: &Tensor) -> R<Tensor> {
        let m = self.project.fwd(t)?; // [1, 1024]
        let w = m.narrow(1, 0, DIM)?.contiguous()?.reshape(vec![1, 1, DIM])?;
        let b = m.narrow(1, DIM, DIM)?.contiguous()?.reshape(vec![1, 1, DIM])?;
        Ok(rms_norm(x, &self.norm, EPS)?.broadcast_mul(&w)?.broadcast_add(&b)?)
    }
}

struct Block {
    attn_norm: AdaNorm,
    ffn_norm: AdaNorm,
    wqkv: Lin,
    wo: Lin,
    w1: Lin,
    w3: Lin,
    w2: Lin,
    skip_in: Lin,
}

struct WnLayer {
    in_w: Tensor,
    in_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
}

pub struct Dit {
    cond_projection: Lin,
    merge: Lin,
    t1: (Lin, Lin),
    t2: (Lin, Lin),
    blocks: Vec<Block>,
    norm: AdaNorm,
    skip_linear: Lin,
    conv1: Lin,
    wn_cond: Lin,
    wn: Vec<WnLayer>,
    res_projection: Lin,
    final_mod: Lin,
    final_linear: Lin,
    conv2: Lin,
    rope: Rope,
    device: Device,
}

/// Зеркальное дополнение по времени на 2 кадра с каждой стороны (`SConv1d`, k5).
fn reflect_pad2(x: &Tensor) -> R<Tensor> {
    let t = x.dims()[2];
    if t < 3 {
        return Err(SeedVcError::Input("окно слишком короткое".into()));
    }
    let s = |i: usize| -> R<Tensor> { Ok(x.narrow(2, i, 1)?.contiguous()?) };
    let (l2, l1, r1, r2) = (s(2)?, s(1)?, s(t - 2)?, s(t - 3)?);
    Ok(Tensor::cat(&[&l2, &l1, &x.contiguous()?, &r1, &r2], 2)?.contiguous()?)
}

impl Dit {
    pub fn load(c: &Comp) -> R<Self> {
        let e = "estimator";
        let mut blocks = Vec::new();
        let mut i = 0;
        while c.has(&format!("{e}.transformer.layers.{i}.attention.wqkv.weight")) {
            let p = format!("{e}.transformer.layers.{i}");
            blocks.push(Block {
                attn_norm: AdaNorm::load(c, &format!("{p}.attention_norm"))?,
                ffn_norm: AdaNorm::load(c, &format!("{p}.ffn_norm"))?,
                wqkv: Lin::load(c, &format!("{p}.attention.wqkv"), false)?,
                wo: Lin::load(c, &format!("{p}.attention.wo"), false)?,
                w1: Lin::load(c, &format!("{p}.feed_forward.w1"), false)?,
                w3: Lin::load(c, &format!("{p}.feed_forward.w3"), false)?,
                w2: Lin::load(c, &format!("{p}.feed_forward.w2"), false)?,
                skip_in: Lin::load(c, &format!("{p}.skip_in_linear"), true)?,
            });
            i += 1;
        }
        if blocks.is_empty() {
            return Err(SeedVcError::Load("в компоненте cfm нет блоков трансформера".into()));
        }
        let mut wn = Vec::with_capacity(WN_LAYERS);
        for i in 0..WN_LAYERS {
            let w = |n: &str| c.get(&format!("{e}.wavenet.{n}.{i}.conv.conv.weight"), DType::F32);
            let b = |n: &str| c.get(&format!("{e}.wavenet.{n}.{i}.conv.conv.bias"), DType::F32);
            wn.push(WnLayer { in_w: w("in_layers")?, in_b: b("in_layers")?, out_w: w("res_skip_layers")?, out_b: b("res_skip_layers")? });
        }
        let tmlp = |n: &str| -> R<(Lin, Lin)> { Ok((Lin::load(c, &format!("{e}.{n}.mlp.0"), true)?, Lin::load(c, &format!("{e}.{n}.mlp.2"), true)?)) };
        Ok(Self {
            cond_projection: Lin::load(c, &format!("{e}.cond_projection"), true)?,
            merge: Lin::load(c, &format!("{e}.cond_x_merge_linear"), true)?,
            t1: tmlp("t_embedder")?,
            t2: tmlp("t_embedder2")?,
            blocks,
            norm: AdaNorm::load(c, &format!("{e}.transformer.norm"))?,
            skip_linear: Lin::load(c, &format!("{e}.skip_linear"), true)?,
            conv1: Lin::load(c, &format!("{e}.conv1"), true)?,
            wn_cond: Lin::load(c, &format!("{e}.wavenet.cond_layer.conv.conv"), true)?,
            wn,
            res_projection: Lin::load(c, &format!("{e}.res_projection"), true)?,
            final_mod: Lin::load(c, &format!("{e}.final_layer.adaLN_modulation.1"), true)?,
            final_linear: Lin::load(c, &format!("{e}.final_layer.linear"), true)?,
            conv2: Lin::load(c, &format!("{e}.conv2"), true)?,
            rope: Rope::compute(4096, HEAD_DIM, 10000.0),
            device: c.device(),
        })
    }

    fn time_embed(&self, t: f32, mlp: &(Lin, Lin)) -> R<Tensor> {
        let half = FREQ_DIM / 2;
        let mut emb = vec![0f32; FREQ_DIM];
        for i in 0..half {
            let a = 1000.0 * t * (-(10000f32.ln()) * i as f32 / half as f32).exp();
            emb[i] = a.cos();
            emb[half + i] = a.sin();
        }
        mlp.1.fwd(&mlp.0.fwd(&Tensor::from_vec(emb, vec![1, FREQ_DIM], self.device)?)?.silu()?)
    }

    /// Скорость потока. `x`, `prompt` — `[B, 80, T]`; `style` — `[B, 192]`; `cond` — `[B, T, 512]`.
    pub fn forward(&self, x: &Tensor, prompt: &Tensor, t: f32, style: &Tensor, cond: &Tensor) -> R<Tensor> {
        let (b, frames) = (x.dims()[0], x.dims()[2]);
        if frames > self.rope.max_positions() {
            return Err(SeedVcError::Input(format!("окно {frames} кадров длиннее предела {}", self.rope.max_positions())));
        }
        let t1 = self.time_embed(t, &self.t1)?;
        let xt = x.transpose(1, 2)?.contiguous()?;
        let pt = prompt.transpose(1, 2)?.contiguous()?;
        let style = style.reshape(vec![b, 1, EMBED])?.broadcast_as(vec![b, frames, EMBED])?.contiguous()?;
        let mut h = self.merge.fwd(&Tensor::cat(&[&xt, &pt, &self.cond_projection.fwd(cond)?, &style], 2)?.contiguous()?)?;
        let positions: Vec<usize> = (0..frames).collect();
        let rope = self.rope.tables(&positions, self.device)?;
        let heads = |t: Tensor| -> R<Tensor> { Ok(t.reshape(vec![b, frames, HEADS, HEAD_DIM])?) };
        let per_head = |t: Tensor| -> R<Tensor> { Ok(t.permute(vec![0, 2, 1, 3])?.contiguous()?) };
        let third = |t: &Tensor, i: usize| -> R<Tensor> { Ok(t.narrow(2, i * DIM, DIM)?.contiguous()?) };
        let half = self.blocks.len() / 2;
        let mut skips: Vec<Tensor> = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            if i > half {
                let s = skips.pop().ok_or_else(|| SeedVcError::Other("U-ViT: нет скипа".into()))?;
                h = blk.skip_in.fwd(&Tensor::cat(&[&h, &s], 2)?.contiguous()?)?;
            }
            let qkv = blk.wqkv.fwd(&blk.attn_norm.fwd(&h, &t1)?)?;
            let q = per_head(apply_rope(&heads(third(&qkv, 0)?)?, &rope)?)?;
            let k = per_head(apply_rope(&heads(third(&qkv, 1)?)?, &rope)?)?;
            let v = per_head(heads(third(&qkv, 2)?)?)?;
            let a = scaled_dot_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None)?;
            h = h.add(&blk.wo.fwd(&a.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, frames, DIM])?)?)?;
            h = h.add(&swiglu(&blk.ffn_norm.fwd(&h, &t1)?, &blk.w1, &blk.w3, &blk.w2)?)?;
            if i < half {
                skips.push(h.clone());
            }
        }
        let h = self.norm.fwd(&h, &t1)?;
        let x_res = self.skip_linear.fwd(&Tensor::cat(&[&h, &xt], 2)?.contiguous()?)?; // длинный скип от входа
        // WaveNet-постнет: условие — второй эмбеддинг времени.
        let g = self.wn_cond.fwd(&self.time_embed(t, &self.t2)?)?; // [1, 2·512·8]
        let mut w = self.conv1.fwd(&x_res)?.transpose(1, 2)?.contiguous()?; // [B, 512, T]
        let mut out: Option<Tensor> = None;
        for (i, l) in self.wn.iter().enumerate() {
            let x_in = conv1d(&reflect_pad2(&w)?, &l.in_w, Some(&l.in_b), 1, 0)?;
            let g_l = g.narrow(1, i * 2 * DIM, 2 * DIM)?.contiguous()?.reshape(vec![1, 2 * DIM, 1])?;
            let a = x_in.broadcast_add(&g_l)?;
            let acts = a.narrow(1, 0, DIM)?.contiguous()?.tanh()?.mul(&a.narrow(1, DIM, DIM)?.contiguous()?.sigmoid()?)?;
            let rs = conv1d(&acts, &l.out_w, Some(&l.out_b), 1, 0)?;
            let skip = if i + 1 < self.wn.len() {
                w = w.add(&rs.narrow(1, 0, DIM)?.contiguous()?)?;
                rs.narrow(1, DIM, DIM)?.contiguous()?
            } else {
                rs
            };
            out = Some(match out {
                Some(o) => o.add(&skip)?,
                None => skip,
            });
        }
        let y = out.expect("слои WaveNet").transpose(1, 2)?.contiguous()?.add(&self.res_projection.fwd(&x_res)?)?;
        // FinalLayer: LayerNorm без параметров, сдвиг и масштаб от времени, Linear.
        let m = self.final_mod.fwd(&t1.silu()?)?;
        let shift = m.narrow(1, 0, DIM)?.contiguous()?.reshape(vec![1, 1, DIM])?;
        let scale = m.narrow(1, DIM, DIM)?.contiguous()?.reshape(vec![1, 1, DIM])?;
        let y = layer_norm(&y, None, None, 1e-6)?.broadcast_mul(&scale.add_scalar(1.0)?)?.broadcast_add(&shift)?;
        let y = self.conv2.fwd(&self.final_linear.fwd(&y)?)?; // [B, T, 80]
        Ok(y.transpose(1, 2)?.contiguous()?)
    }
}

/// Сэмплер Эйлера v1: равномерная сетка, CFG `(1 + r)·v − r·v_пусто`.
pub fn sample(dit: &Dit, mu: &Tensor, prompt: &Tensor, style: &Tensor, noise: &Tensor, steps: usize, cfg_rate: f32, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<Tensor> {
    let dev = dit.device;
    let frames = mu.dims()[1];
    let plen = prompt.dims()[2].min(frames);
    let mask: Vec<f32> = (0..frames).map(|i| if i < plen { 0.0 } else { 1.0 }).collect();
    let mask = Tensor::from_vec(mask, vec![1, 1, frames], dev)?;
    let prompt_x = if plen == frames {
        prompt.narrow(2, 0, plen)?.contiguous()?
    } else {
        Tensor::cat(&[&prompt.narrow(2, 0, plen)?.contiguous()?, &Tensor::zeros(vec![1, N_MELS, frames - plen], DType::F32, dev)?], 2)?.contiguous()?
    };
    let mu = mu.contiguous()?;
    let (prompt2, style2, mu2) = if cfg_rate > 0.0 {
        (
            Tensor::cat(&[&prompt_x, &Tensor::zeros(vec![1, N_MELS, frames], DType::F32, dev)?], 0)?.contiguous()?,
            Tensor::cat(&[style, &Tensor::zeros(vec![1, EMBED], DType::F32, dev)?], 0)?.contiguous()?,
            Tensor::cat(&[&mu, &Tensor::zeros(vec![1, frames, mu.dims()[2]], DType::F32, dev)?], 0)?.contiguous()?,
        )
    } else {
        (prompt_x.clone(), style.clone(), mu.clone())
    };
    let steps = steps.max(1);
    // torch.linspace(0, 1, N + 1) в f32.
    let n = steps as f32;
    let grid: Vec<f32> = (0..=steps).map(|i| if i < (steps + 1) / 2 { i as f32 * (1.0 / n) } else { 1.0 - (steps - i) as f32 * (1.0 / n) }).collect();
    let mut x = noise.broadcast_mul(&mask)?;
    let mut t = grid[0];
    for step in 1..grid.len() {
        let dt = grid[step] - grid[step - 1];
        let v = if cfg_rate > 0.0 {
            let o = dit.forward(&Tensor::cat(&[&x, &x], 0)?.contiguous()?, &prompt2, t, &style2, &mu2)?;
            o.narrow(0, 0, 1)?.contiguous()?.mul_scalar(1.0 + cfg_rate)?.sub(&o.narrow(0, 1, 1)?.contiguous()?.mul_scalar(cfg_rate)?)?
        } else {
            dit.forward(&x, &prompt2, t, &style2, &mu2)?
        };
        x = x.add(&v.mul_scalar(dt)?)?.broadcast_mul(&mask)?;
        t += dt;
        if !progress(step, steps) {
            return Err(SeedVcError::Cancelled);
        }
    }
    Ok(x)
}

#[derive(Debug, Clone)]
pub struct SeedVcV1Request {
    pub source: Vec<f32>,
    pub source_rate: u32,
    /// Образец голоса (используются первые 25 с).
    pub reference: Vec<f32>,
    pub reference_rate: u32,
    pub steps: usize,
    pub length_adjust: f32,
    /// CFG (`inference_cfg_rate`).
    pub cfg_rate: f32,
    pub seed: u64,
}

impl Default for SeedVcV1Request {
    fn default() -> Self {
        Self { source: Vec::new(), source_rate: SAMPLE_RATE, reference: Vec::new(), reference_rate: SAMPLE_RATE, steps: 30, length_adjust: 1.0, cfg_rate: 0.7, seed: 0 }
    }
}

pub struct SeedVcV1 {
    bundle: PathBuf,
    device: Device,
    pub content: Content,
    pub campplus: CamPlus,
    pub regulator: Regulator,
    pub dit: Dit,
    pub vocoder: Vocoder,
}

impl SeedVcV1 {
    pub fn load(bundle: impl AsRef<Path>, device: Device) -> R<Self> {
        let bundle = bundle.as_ref().to_path_buf();
        let comp = |name: &str| Comp::open(&bundle, name, device);
        let cfm = comp("cfm")?;
        Ok(Self {
            content: Content::load(&bundle, device)?,
            campplus: CamPlus::load(&comp("campplus")?)?,
            regulator: Regulator::load(&cfm)?,
            dit: Dit::load(&cfm)?,
            vocoder: Vocoder::load(&comp("bigvgan")?)?,
            bundle,
            device,
        })
    }

    pub fn bundle(&self) -> &Path {
        &self.bundle
    }

    pub fn convert(&self, req: &SeedVcV1Request, progress: Progress<'_>) -> R<SeedVcOutput> {
        if req.source.is_empty() || req.reference.is_empty() {
            return Err(SeedVcError::Input("нужны исходная речь и образец голоса".into()));
        }
        let src_22k = resample(&req.source, req.source_rate, SAMPLE_RATE)?;
        let mut ref_22k = resample(&req.reference, req.reference_rate, SAMPLE_RATE)?;
        ref_22k.truncate(SAMPLE_RATE as usize * MAX_REFERENCE_S);
        // Эталон v1 переводит в 16 кГц через torchaudio — у нас она бит в бит.
        let e = |e: synaptix_audio::AudioError| SeedVcError::Input(format!("передискретизация: {e}"));
        let src_16k = synaptix_audio::resample_torchaudio(&src_22k, SAMPLE_RATE, 16_000).map_err(e)?;
        let ref_16k = synaptix_audio::resample_torchaudio(&ref_22k, SAMPLE_RATE, 16_000).map_err(e)?;
        self.convert_waves(&Waves { src_22k: &src_22k, ref_22k: &ref_22k, src_16k: &src_16k, ref_16k: &ref_16k }, req, None, progress)
    }

    /// То же по готовым волнам; `noise` — шум окон `[80 × T]` (сверка с эталоном).
    pub fn convert_waves(&self, w: &Waves<'_>, req: &SeedVcV1Request, noise: Option<&[Vec<f32>]>, progress: Progress<'_>) -> R<SeedVcOutput> {
        let dev = self.device;
        synaptix_core::grad::no_grad(|| -> R<SeedVcOutput> {
            let t0 = Instant::now();
            let (_, src_frames) = mel_spectrogram(w.src_22k)?;
            let (ref_mel, ref_frames) = mel_spectrogram(w.ref_22k)?;
            let window = SAMPLE_RATE as usize / HOP * DIT_CONTEXT_S;
            if ref_frames + 2 * OVERLAP_FRAMES >= window {
                return Err(SeedVcError::Input("образец голоса слишком длинный".into()));
            }
            let prompt = Tensor::from_vec(ref_mel, vec![1, N_MELS, ref_frames], dev)?;
            if !progress("содержание", 0, 2) {
                return Err(SeedVcError::Cancelled);
            }
            let s_alt = self.content.features_long(w.src_16k)?;
            if !progress("содержание", 1, 2) {
                return Err(SeedVcError::Cancelled);
            }
            let s_ori = self.content.features(w.ref_16k)?;
            let (f, fr) = fbank(w.ref_16k);
            let style = Tensor::from_vec(self.campplus.embed(&f, fr, false)?, vec![1, EMBED], dev)?;
            let frames = ((src_frames as f64 * req.length_adjust as f64) as usize).max(1);
            let cond = self.regulator.forward(&s_alt, frames)?;
            let prompt_cond = self.regulator.forward(&s_ori, ref_frames)?;
            if !progress("содержание", 2, 2) {
                return Err(SeedVcError::Cancelled);
            }
            let content_seconds = t0.elapsed().as_secs_f64();
            let (mut cfm_seconds, mut vocoder_seconds) = (0f64, 0f64);
            let mut stitch = Stitcher { out: Vec::new(), prev: None, overlap: OVERLAP_FRAMES * HOP };
            let max_source = window - ref_frames;
            let total = if frames <= max_source { 1 } else { 1 + (frames - max_source).div_ceil(max_source - OVERLAP_FRAMES) };
            let (mut processed, mut ci) = (0usize, 0usize);
            while processed < frames {
                let len = max_source.min(frames - processed);
                let last = processed + max_source >= frames;
                let t = Instant::now();
                let mu = Tensor::cat(&[&prompt_cond, &cond.narrow(1, processed, len)?.contiguous()?], 1)?.contiguous()?;
                let n = mu.dims()[1];
                let z = match noise {
                    Some(v) => Tensor::from_vec(v.get(ci).ok_or_else(|| SeedVcError::Other("не хватает заданного шума".into()))?.clone(), vec![1, N_MELS, n], dev)?,
                    None => Tensor::randn_seeded(vec![1, N_MELS, n], req.seed.wrapping_add(ci as u64 * 7919), Device::Cpu)?.to_dtype(DType::F32)?.to_device(dev)?,
                };
                let mel = sample(&self.dit, &mu, &prompt, &style, &z, req.steps, req.cfg_rate, &mut |i, k| progress("звук", ci * k + i, total.max(ci + 1) * k))?;
                let mel = mel.narrow(2, ref_frames, n - ref_frames)?.contiguous()?;
                cfm_seconds += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let wave = self.vocoder.decode(&mel)?;
                vocoder_seconds += t.elapsed().as_secs_f64();
                stitch.push(wave, last);
                if last {
                    break;
                }
                processed += len - OVERLAP_FRAMES;
                ci += 1;
            }
            Ok(SeedVcOutput { pcm: stitch.out, sample_rate: SAMPLE_RATE, ar_tokens: Vec::new(), content_seconds, ar_seconds: 0.0, cfm_seconds, vocoder_seconds })
        })
    }
}
