//! NAR-ветка YuE2: акустические латенты методом flow matching.
//!
//! У каждого слоя свой, второй комплект весов (`nar_*`). NAR-позиции видят
//! весь AR-префикс и друг друга — маски нет вовсе, причинность здесь не нужна.
//!
//! AR-контекст чанка не пересчитывается на каждом шаге ODE: префикс
//! прогоняется один раз обычным AR-проходом, а его K/V остаются в кэше. Больше
//! того, NAR пишет свои K/V в хвост того же буфера — поэтому ни склейки, ни
//! второй копии контекста на шаг не возникает.
//!
//! Решатель — midpoint, 32 шага, время идёт от 1 к 0.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::model::{partial_rope, RopePositions};
use synaptix_llm_common::{KvCache, LayerCache, QLinear};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope_cache::RopeCache;
use synaptix_ops::rng::{fill_normal_f32, Philox4x32};

use crate::ar::Yue2Ar;
use crate::config::Yue2Config;
use crate::loader::{read_bundle_file, CompLoader};
use crate::protocol::{chunk_ranges, CODEC_OFFSET, CODEC_SIZE, MUSIC_END};
use crate::YueError;

type R<T> = Result<T, YueError>;

/// Размерность синусоидального кодирования времени перед MLP.
const TIME_FREQ_DIM: usize = 256;

struct NarLayer {
    input_norm: Tensor,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    pre_mlp_norm: Tensor,
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl NarLayer {
    fn load(ck: &CompLoader, idx: usize, compute: DType, quant: Option<DType>) -> R<Self> {
        let p = format!("model.layers.{idx}");
        let dense = |name: &str| -> R<QLinear> {
            let w = ck.get(&format!("{p}.{name}.weight"), compute)?;
            QLinear::build(w, quant.unwrap_or(compute), compute)
                .map_err(|e| YueError::Load(e.to_string()))
        };
        Ok(Self {
            input_norm: ck.get(&format!("{p}.nar_input_layernorm.weight"), compute)?,
            q_proj: dense("nar_self_attn.q_proj")?,
            k_proj: dense("nar_self_attn.k_proj")?,
            v_proj: dense("nar_self_attn.v_proj")?,
            o_proj: dense("nar_self_attn.o_proj")?,
            q_norm: ck.get(&format!("{p}.nar_self_attn.q_norm.weight"), compute)?,
            k_norm: ck.get(&format!("{p}.nar_self_attn.k_norm.weight"), compute)?,
            pre_mlp_norm: ck.get(&format!("{p}.nar_pre_mlp_layernorm.weight"), compute)?,
            gate_proj: dense("nar_mlp.gate_proj")?,
            up_proj: dense("nar_mlp.up_proj")?,
            down_proj: dense("nar_mlp.down_proj")?,
        })
    }
}

pub struct Yue2Nar {
    layers: Vec<NarLayer>,
    final_norm: Tensor,
    /// Латент → скрытое состояние и обратно (со смещениями).
    vae2llm: Linear,
    llm2vae: Linear,
    time_mlp_in: Linear,
    time_mlp_out: Linear,
    /// Синусоидальные позиции NAR-ветки, `[max_latent_frames, hidden]`.
    pos_embed: Tensor,
    rope: RopeCache,
    pub config: Yue2Config,
    device: Device,
    compute: DType,
}

impl Yue2Nar {
    /// Открыть NAR-веса того же бандла, что и AR.
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant: Option<DType>,
    ) -> R<Self> {
        let path = path.as_ref();
        let ck = CompLoader::open(path, None, device)?;
        let config = match read_bundle_file(path, "config.json") {
            Ok(bytes) => Yue2Config::from_json(&bytes)?,
            Err(_) => Yue2Config::default(),
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            layers.push(NarLayer::load(&ck, idx, compute, quant)?);
        }
        let lin = |name: &str| -> R<Linear> {
            let w = ck.get(&format!("{name}.weight"), compute)?;
            let b = ck.get(&format!("{name}.bias"), compute)?;
            Linear::new(w, Some(b)).map_err(|e| YueError::Load(e.to_string()))
        };
        let rope = RopeCache::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;
        Ok(Self {
            final_norm: ck.get("model.norm.weight", compute)?,
            vae2llm: lin("vae2llm")?,
            llm2vae: lin("llm2vae")?,
            time_mlp_in: lin("time_embedder.mlp.0")?,
            time_mlp_out: lin("time_embedder.mlp.2")?,
            pos_embed: ck.get("latent_pos_embed.pe", compute)?,
            layers,
            rope,
            config,
            device,
            compute,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Время шага после сдвига шкалы: `shift·σ(t) / (1 + (shift−1)·σ(t))`.
    fn shifted_time(&self, raw: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-raw).exp());
        let shift = self.config.timestep_shift;
        shift * sigmoid / (1.0 + (shift - 1.0) * sigmoid)
    }

    /// Синусоидальное кодирование времени → MLP → вектор скрытого состояния.
    fn time_embedding(&self, t: f32) -> R<Tensor> {
        let half = TIME_FREQ_DIM / 2;
        let mut emb = vec![0f32; TIME_FREQ_DIM];
        for i in 0..half {
            let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let arg = t * freq;
            emb[i] = arg.cos();
            emb[half + i] = arg.sin();
        }
        let x = Tensor::from_vec(emb, vec![1usize, TIME_FREQ_DIM], self.device)?
            .to_dtype(self.compute)?;
        let h = self.time_mlp_in.forward(&x)?;
        let h = synaptix_ops::activation::silu::silu(&h)?;
        Ok(self.time_mlp_out.forward(&h)?.reshape(vec![1usize, 1usize, self.config.hidden_size])?)
    }

    /// Поле скоростей `v_θ(x_t, t)` для одного чанка. `state` — `[кадры, 64]`,
    /// `raw_t` — время до сдвига шкалы (`logit` шага решателя).
    pub fn velocity(&self, ctx: &mut ChunkContext, state: &Tensor, raw_t: f32) -> R<Tensor> {
        let hidden = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let eps = self.config.rms_norm_eps;
        let scale = 1.0 / (hd as f32).sqrt();
        let (frames, latent) = (state.dims()[0], state.dims()[1]);
        let nar_len = ctx.nar_len;
        let ar_len = ctx.ar_len;

        // Края NAR-окна (позиции LATENT_START/END) идут в модель нулями —
        // так же, как на обучении.
        let pad = Tensor::zeros(vec![1usize, 1usize, latent], self.compute, self.device)?;
        let body = state.to_dtype(self.compute)?.reshape(vec![1usize, frames, latent])?;
        let x_nar = Tensor::cat(&[&pad, &body, &pad], 1)?;

        let mut x = self.vae2llm.forward(&x_nar)?;
        let t_emb = self.time_embedding(self.shifted_time(raw_t))?;
        x = x.broadcast_add(&t_emb)?;
        let pos = self
            .pos_embed
            .narrow(0, 0, nar_len)?
            .reshape(vec![1usize, nar_len, hidden])?;
        x = x.broadcast_add(&pos)?;

        for (idx, layer) in self.layers.iter().enumerate() {
            let h = rms_norm(&x, &layer.input_norm, eps)?;
            let q = layer.q_proj.forward(&h).map_err(err)?;
            let k = layer.k_proj.forward(&h).map_err(err)?;
            let v = layer.v_proj.forward(&h).map_err(err)?;
            let q = q
                .reshape(vec![1usize, nar_len, heads, hd])?
                .permute(vec![0, 2, 1, 3])?
                .contiguous()?;
            let k = k
                .reshape(vec![1usize, nar_len, kv_heads, hd])?
                .permute(vec![0, 2, 1, 3])?
                .contiguous()?;
            let v = v
                .reshape(vec![1usize, nar_len, kv_heads, hd])?
                .permute(vec![0, 2, 1, 3])?
                .contiguous()?;
            let q = rms_norm(&q, &layer.q_norm, eps)?;
            let k = rms_norm(&k, &layer.k_norm, eps)?;
            // Позиции NAR-окна идут сразу за AR-префиксом.
            let q = partial_rope(&q, &self.rope, ar_len, nar_len, hd, hd, RopePositions::Sequential)?;
            let k = partial_rope(&k, &self.rope, ar_len, nar_len, hd, hd, RopePositions::Sequential)?;

            let (k_full, v_full) = ctx.write_nar_kv(idx, &k, &v)?;
            let attn = q
                .flash_attention(&k_full, &v_full, scale, false)
                .or_else(|_| fallback_attention(&q, &k_full, &v_full, scale))?;
            let attn = attn
                .permute(vec![0, 2, 1, 3])?
                .contiguous()?
                .reshape(vec![1usize, nar_len, heads * hd])?;
            x = x.broadcast_add(&layer.o_proj.forward(&attn).map_err(err)?)?;

            let h = rms_norm(&x, &layer.pre_mlp_norm, eps)?;
            let gate = synaptix_ops::activation::silu::silu(&layer.gate_proj.forward(&h).map_err(err)?)?;
            let up = layer.up_proj.forward(&h).map_err(err)?;
            let ffn = layer.down_proj.forward(&gate.mul(&up)?).map_err(err)?;
            x = x.broadcast_add(&ffn)?;
        }
        let out = self.llm2vae.forward(&rms_norm(&x, &self.final_norm, eps)?)?;
        // Края окна наружу не идут.
        Ok(out.narrow(1, 1, frames)?.contiguous()?.reshape(vec![frames, latent])?)
    }

    /// Решить ODE для одного чанка. `noise` — `[кадры, 64]`, ответ той же формы
    /// в F32 на CPU.
    pub fn solve_chunk(
        &self,
        ctx: &mut ChunkContext,
        noise: &Tensor,
        steps: usize,
        cancel: &dyn Fn() -> bool,
        on_progress: &dyn Fn(usize, usize),
    ) -> R<Tensor> {
        if steps < 1 {
            return Err(YueError::Config("шагов ODE должно быть хотя бы один".into()));
        }
        let mut state = noise.to_device(self.device)?.to_dtype(self.compute)?;
        let dt = 1.0 / steps as f32;
        for step in 0..steps {
            if cancel() {
                return Err(YueError::Cancelled("flow matching"));
            }
            let t = 1.0 - step as f32 * dt;
            let first = self.velocity(ctx, &state, logit(t))?;
            let mid = state.sub(&first.affine(dt / 2.0, 0.0)?)?;
            if cancel() {
                return Err(YueError::Cancelled("flow matching"));
            }
            let second = self.velocity(ctx, &mid, logit(t - dt / 2.0))?;
            state = state.sub(&second.affine(dt, 0.0)?)?;
            on_progress(step + 1, steps);
        }
        let out = state.to_dtype(DType::F32)?.to_device(Device::Cpu)?;
        let values: Vec<f32> = out.flatten_all()?.to_vec1()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(YueError::Other(
                "flow matching дал не-конечные латенты".into(),
            ));
        }
        Ok(out)
    }
}

fn err(e: synaptix_llm_common::ModelError) -> YueError {
    YueError::Other(e.to_string())
}

/// `logit(t)` с тем же ограничением, что у эталона: на концах шкалы ±20.
fn logit(t: f32) -> f32 {
    let t = t as f64;
    let raw = (t / (1.0 - t)).ln();
    raw.clamp(-20.0, 20.0) as f32
}

/// Запасной путь внимания (CPU и всё, где нет flash-ядра): честный
/// `softmax(QKᵀ·scale)·V` с размножением KV-голов под GQA.
fn fallback_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> R<Tensor> {
    let (heads, kv_heads) = (q.dims()[1], k.dims()[1]);
    let group = heads / kv_heads;
    let mut outs: Vec<Tensor> = Vec::with_capacity(heads);
    for h in 0..heads {
        let qh = q.narrow(1, h, 1)?.contiguous()?.squeeze(1)?;
        let kh = k.narrow(1, h / group, 1)?.contiguous()?.squeeze(1)?;
        let vh = v.narrow(1, h / group, 1)?.contiguous()?.squeeze(1)?;
        let scores = qh
            .matmul(&kh.transpose(1, 2)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let probs = synaptix_ops::attention::softmax_dim(&scores, 2)?;
        outs.push(probs.matmul(&vh)?.unsqueeze(1)?);
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

/// Один акустический чанк: AR-префикс в KV-кэше плюс место под NAR-хвост.
pub struct ChunkContext {
    kv: KvCache,
    ar_len: usize,
    nar_len: usize,
}

impl ChunkContext {
    /// Длина AR-части (префикс запроса, семантические токены чанка, `</music>`).
    pub fn ar_len(&self) -> usize {
        self.ar_len
    }

    /// Длина NAR-окна: кадры плюс два служебных края.
    pub fn nar_len(&self) -> usize {
        self.nar_len
    }

    /// Записать K/V NAR-позиций в хвост кэша и вернуть виды на «AR + NAR».
    fn write_nar_kv(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> R<(Tensor, Tensor)> {
        let total = self.ar_len + self.nar_len;
        let LayerCache::Full(cache) = &mut self.kv.layers[layer] else {
            return Err(YueError::Other("NAR: у слоя нелинейный кэш".into()));
        };
        if cache.k.dtype() != k.dtype() {
            return Err(YueError::Other(format!(
                "NAR: кэш в {:?}, а ключи в {:?} — ветки должны считаться в одном типе",
                cache.k.dtype(),
                k.dtype()
            )));
        }
        cache.k.kv_append_inplace(k, self.ar_len)?;
        cache.v.kv_append_inplace(v, self.ar_len)?;
        Ok((cache.k.narrow(2, 0, total)?, cache.v.narrow(2, 0, total)?))
    }
}

/// Прогнать AR-префикс чанка и оставить его K/V для NAR-ветки.
pub fn prefill_chunk(ar: &Yue2Ar, ar_tokens: &[u32], frames: usize) -> R<ChunkContext> {
    let ar_len = ar_tokens.len();
    let nar_len = frames + 2;
    if ar_len == 0 {
        return Err(YueError::Config("акустический чанк без AR-префикса".into()));
    }
    if ar_len + nar_len > ar.config.max_position_embeddings {
        return Err(YueError::Config(format!(
            "чанк не влезает в окно модели: {ar_len} + {nar_len} > {}",
            ar.config.max_position_embeddings
        )));
    }
    let mut session = crate::ar::ArSession::new(ar, ar_len + nar_len)?;
    session.feed(ar_tokens)?;
    Ok(ChunkContext { kv: session.into_kv(), ar_len, nar_len })
}

/// Как нарезаются чанки и что в каждом из них.
pub struct Chunk {
    /// AR-часть: префикс запроса + семантические токены чанка + `</music>`.
    pub ar_tokens: Vec<u32>,
    /// Диапазон кадров песни `[начало, конец)`.
    pub range: (usize, usize),
}

/// Разложить песню на чанки. Шум рисуется один раз на всю песню, чтобы
/// границы чанков не меняли результат.
pub fn song_chunks(
    prefix: &[u32],
    codec: &[u32],
    context: usize,
) -> R<Vec<Chunk>> {
    if codec.iter().any(|&c| c >= CODEC_SIZE) {
        return Err(YueError::Config("семантический токен вне словаря".into()));
    }
    let ranges = chunk_ranges(codec.len(), prefix.len(), context)?;
    Ok(ranges
        .into_iter()
        .map(|(a, b)| {
            let mut ar_tokens = Vec::with_capacity(prefix.len() + (b - a) + 1);
            ar_tokens.extend_from_slice(prefix);
            ar_tokens.extend(codec[a..b].iter().map(|&c| c + CODEC_OFFSET));
            ar_tokens.push(MUSIC_END);
            Chunk { ar_tokens, range: (a, b) }
        })
        .collect())
}

/// Гауссов шум на всю песню: `[кадры, 64]` в F32 на CPU.
pub fn song_noise(frames: usize, latent_dim: usize, seed: u64) -> R<Tensor> {
    let mut rng = Philox4x32::new(seed);
    let mut values = vec![0f32; frames * latent_dim];
    fill_normal_f32(&mut rng, &mut values);
    Ok(Tensor::from_vec(values, vec![frames, latent_dim], Device::Cpu)?)
}

/// Синтез акустических латентов всей песни: чанки решаются подряд, каждый —
/// со своим AR-контекстом. Ответ `[кадры, 64]` F32 на CPU.
#[allow(clippy::too_many_arguments)]
pub fn synthesize(
    ar: &Yue2Ar,
    nar: &Yue2Nar,
    prefix: &[u32],
    codec: &[u32],
    seed: u64,
    steps: usize,
    context: usize,
    cancel: &dyn Fn() -> bool,
    on_progress: &dyn Fn(usize, usize),
) -> R<Tensor> {
    let chunks = song_chunks(prefix, codec, context)?;
    let noise = song_noise(codec.len(), nar.config.latent_dim, seed)?;
    let total_steps = steps * chunks.len();
    let mut parts: Vec<Tensor> = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        if cancel() {
            return Err(YueError::Cancelled("акустический префилл"));
        }
        let (a, b) = chunk.range;
        let mut ctx = prefill_chunk(ar, &chunk.ar_tokens, b - a)?;
        let slice = noise.narrow(0, a, b - a)?.contiguous()?;
        let done = index * steps;
        let latents = nar.solve_chunk(&mut ctx, &slice, steps, cancel, &|step, _| {
            on_progress(done + step, total_steps)
        })?;
        parts.push(latents);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 0)?)
}
