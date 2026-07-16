//! Текст-кондишен LTX-2.3 (видео-путь): из 49 hidden states Gemma →
//! FeatureExtractorV2 → Embeddings1DConnector → `video_encoding [B,T,4096]`.
//!
//! Аудио-путь (тот же код, dims 2048/64) добавляется на Фазе 8. Примитивы
//! (SPLIT-RoPE, Attention с qk-norm+gated, gelu-tanh FF) — общие с DiT (Фаза 4).
//!
//! Веса: `text_embedding_projection.video_aggregate_embed.*` +
//! `model.diffusion_model.video_embeddings_connector.*`.

use std::f64::consts::PI;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::loader::{LtxCheckpoint, DIT_PREFIX, TEXT_PROJ_PREFIX};
use crate::LtxError;

type R<T> = Result<T, SynaptixError>;

const NORM_EPS: f64 = 1e-6;

/// Linear `y = x @ Wᵀ + b`, W `[out,in]`, x `[..,in]`.
pub(crate) fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> R<Tensor> {
    let dims = x.dims().to_vec();
    let in_d = *dims.last().unwrap();
    let n: usize = dims[..dims.len() - 1].iter().product();
    let out_d = w.dims()[0];
    let x2 = x.contiguous()?.reshape(vec![n, in_d])?;
    let wt = w.transpose(0, 1)?.contiguous()?; // [in,out]
    let mut y = x2.matmul(&wt)?; // [n,out]
    if let Some(b) = b {
        y = y.broadcast_add(b)?;
    }
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(out_d);
    y.reshape(out_shape)
}

/// RMSNorm по последней оси без gain (`utils.rms_norm(x)`).
// Кэш ones-веса [H] для no-gain fused-rms (variant=0 → scale=w; ones=чистый rms).
fn ones_weight(h: usize, dt: DType, dev: Device) -> R<Tensor> {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<(usize, DType, usize, Tensor)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let ord = dev.ordinal();
    if let Ok(g) = cache.lock() {
        for (hh, dd, oo, t) in g.iter() {
            if *hh == h && *dd == dt && *oo == ord {
                return Ok(t.clone());
            }
        }
    }
    let t = Tensor::ones(vec![h], dt, dev)?;
    if let Ok(mut g) = cache.lock() {
        g.push((h, dt, ord, t.clone()));
    }
    Ok(t)
}

pub(crate) fn rms_no_gain(x: &Tensor) -> R<Tensor> {
    // torch F.rms_norm считает в f32, кастует выход в dtype входа. bf16-редукция по
    // 4096 каналам теряет точность → дрейф velocity по 48 блокам (класс FLUX).
    // Fused-ядро rms_norm считает sumsq/mean/rsqrt в F32 (load_f32) — F32-faithful,
    // 1 launch вместо cast→sqr→reduce→div→cast→contiguous (~6 ядер, reduce 23× медленнее BW).
    let dt = x.dtype();
    if matches!(x.device(), Device::Cuda(_)) && matches!(dt, DType::BF16 | DType::F16) {
        let h = x.dims()[x.rank() - 1];
        if let Ok(ones) = ones_weight(h, dt, x.device()) {
            if let Ok(y) = x.rms_norm_fused(&ones, NORM_EPS as f32, false) {
                return Ok(y);
            }
        }
    }
    let xf = x.to_dtype(DType::F32)?;
    let last = xf.rank() - 1;
    let denom = xf.sqr()?.mean_keepdim(last)?.add_scalar(NORM_EPS as f32)?.sqrt()?;
    xf.broadcast_div(&denom)?.to_dtype(dt)?.contiguous()
}

/// RMSNorm с gain (`nn.RMSNorm`, вес-множитель, НЕ 1+w). F32-faithful через
/// fused-ядро (variant=0 = plain weight); fallback — decomposed f32.
pub(crate) fn rms_gain(x: &Tensor, w: &Tensor) -> R<Tensor> {
    let dt = x.dtype();
    if matches!(x.device(), Device::Cuda(_)) && matches!(dt, DType::BF16 | DType::F16) {
        let h = x.dims()[x.rank() - 1];
        if w.rank() == 1 && w.dims()[0] == h {
            let wc = if w.dtype() == dt { w.clone() } else { w.to_dtype(dt)? };
            if let Ok(y) = x.rms_norm_fused(&wc, NORM_EPS as f32, false) {
                return Ok(y);
            }
        }
    }
    let xf = x.to_dtype(DType::F32)?;
    let last = xf.rank() - 1;
    let denom = xf.sqr()?.mean_keepdim(last)?.add_scalar(NORM_EPS as f32)?.sqrt()?;
    let normed = xf.broadcast_div(&denom)?;
    normed.broadcast_mul(&w.to_dtype(DType::F32)?)?.to_dtype(dt)?.contiguous()
}

/// SPLIT-RoPE cos/sin для 1D-позиций (коннектор): `[1, heads, T, dim_head/2]`.
/// Сетка частот в f64 (`frequencies_precision="float64"`), как в reference.
fn split_rope_cos_sin(
    t: usize,
    heads: usize,
    dim_head: usize,
    theta: f64,
    max_pos: f64,
    device: Device,
    dtype: DType,
) -> R<(Tensor, Tensor)> {
    let half = dim_head / 2; // частот на голову
    let n_freq = heads * half; // = inner_dim/2
    // indices[j] = theta^(j/(n_freq-1)) * pi/2, j in 0..n_freq
    let mut indices = vec![0f64; n_freq];
    for (j, idx) in indices.iter_mut().enumerate() {
        let exponent = if n_freq > 1 { j as f64 / (n_freq - 1) as f64 } else { 0.0 };
        *idx = theta.powf(exponent) * PI / 2.0;
    }
    // layout [1, heads, T, half]: cos[h][p][f] = cos(indices[h*half+f] * s_p), s_p=(p/max_pos)*2-1
    let mut cos = vec![0f32; heads * t * half];
    let mut sin = vec![0f32; heads * t * half];
    for h in 0..heads {
        for p in 0..t {
            let s = (p as f64 / max_pos) * 2.0 - 1.0;
            for f in 0..half {
                let ang = indices[h * half + f] * s;
                let o = (h * t + p) * half + f;
                cos[o] = ang.cos() as f32;
                sin[o] = ang.sin() as f32;
            }
        }
    }
    // Таблицы в F32: fused-ядро (rope_split) ждёт F32 и ротация в F32 = как
    // python-эталон; decomposed-фолбэк кастует в dtype потока сам.
    let _ = dtype;
    let cos = Tensor::from_vec(cos, vec![1, heads, t, half], device)?;
    let sin = Tensor::from_vec(sin, vec![1, heads, t, half], device)?;
    Ok((cos, sin))
}

/// `apply_split_rotary_emb`: x `[1,H,T,D]`, cos/sin `[1,H,T,D/2]` (F32). Вращение
/// по ПОЛОВИНАМ (НЕ interleaved): out[:D/2]=x0·cos−x1·sin, out[D/2:]=x1·cos+x0·sin.
/// CUDA → fused rope_split (1 launch, F32-ротация — точнее bf16-цепочки и ближе
/// к python); иначе decomposed (9 ops).
pub(crate) fn apply_split_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> R<Tensor> {
    let dims = x.dims().to_vec();
    let (b, h, t, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half = d / 2;
    if matches!(x.device(), synaptix_core::device::Device::Cuda(_)) {
        let cos32 = if cos.dtype() == DType::F32 { cos.clone() } else { cos.to_dtype(DType::F32)? };
        let sin32 = if sin.dtype() == DType::F32 { sin.clone() } else { sin.to_dtype(DType::F32)? };
        let xv = x.reshape(vec![b, h * t, d])?;
        let cv = cos32.reshape(vec![h * t, half])?;
        let sv = sin32.reshape(vec![h * t, half])?;
        if let Ok(y) = xv.rope_split_fused(&cv, &sv) {
            return Ok(y.reshape(dims)?);
        }
    }
    let cos_t = if cos.dtype() == x.dtype() { cos.clone() } else { cos.to_dtype(x.dtype())? };
    let sin_t = if sin.dtype() == x.dtype() { sin.clone() } else { sin.to_dtype(x.dtype())? };
    let x0 = x.narrow(3, 0, half)?.contiguous()?;
    let x1 = x.narrow(3, half, half)?.contiguous()?;
    let out0 = x0.mul(&cos_t)?.sub(&x1.mul(&sin_t)?)?;
    let out1 = x1.mul(&cos_t)?.add(&x0.mul(&sin_t)?)?;
    Tensor::cat(&[&out0, &out1], 3)
}

struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}
impl Linear {
    fn load(ckpt: &LtxCheckpoint, prefix: &str, key: &str, bias: bool) -> Result<Self, LtxError> {
        let w = ckpt.get(&format!("{prefix}.{key}.weight"))?;
        let b = if bias { Some(ckpt.get(&format!("{prefix}.{key}.bias"))?) } else { None };
        Ok(Self { w, b })
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        linear(x, &self.w, self.b.as_ref())
    }
}

/// Один блок коннектора: pre-norm RMS → self-attn (qk-norm, SPLIT-RoPE, gated) →
/// residual → pre-norm RMS → gelu-tanh FF → residual.
struct ConnectorBlock {
    q_norm: Tensor,
    k_norm: Tensor,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_gate: Linear,
    to_out: Linear,
    ff0: Linear,
    ff2: Linear,
    heads: usize,
    dim_head: usize,
    scale: f32,
}

impl ConnectorBlock {
    fn load(ckpt: &LtxCheckpoint, prefix: &str, heads: usize, dim_head: usize) -> Result<Self, LtxError> {
        let a = format!("{prefix}.attn1");
        Ok(Self {
            q_norm: ckpt.get(&format!("{a}.q_norm.weight"))?,
            k_norm: ckpt.get(&format!("{a}.k_norm.weight"))?,
            to_q: Linear::load(ckpt, &a, "to_q", true)?,
            to_k: Linear::load(ckpt, &a, "to_k", true)?,
            to_v: Linear::load(ckpt, &a, "to_v", true)?,
            to_gate: Linear::load(ckpt, &a, "to_gate_logits", true)?,
            to_out: Linear::load(ckpt, &a, "to_out.0", true)?,
            ff0: Linear::load(ckpt, prefix, "ff.net.0.proj", true)?,
            ff2: Linear::load(ckpt, prefix, "ff.net.2", true)?,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f32).sqrt(),
        })
    }

    fn attn(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> R<Tensor> {
        let (b, t) = (x.dims()[0], x.dims()[1]);
        let (h, dh) = (self.heads, self.dim_head);
        let to_bhtd = |y: Tensor| -> R<Tensor> {
            y.reshape(vec![b, t, h, dh])?.transpose(1, 2)?.contiguous()
        };
        let q = rms_gain(&self.to_q.fwd(x)?, &self.q_norm)?;
        let k = rms_gain(&self.to_k.fwd(x)?, &self.k_norm)?;
        let v = self.to_v.fwd(x)?;
        let q = apply_split_rope(&to_bhtd(q)?, cos, sin)?;
        let k = apply_split_rope(&to_bhtd(k)?, cos, sin)?;
        let v = to_bhtd(v)?;
        // bidirectional (маска занулена register-substitution'ом) → mask=None.
        let attn = scaled_dot_attention(&q, &k, &v, self.scale, None)?; // [b,h,t,dh]
        let out = attn.transpose(1, 2)?.contiguous()?.reshape(vec![b, t, h * dh])?;
        // per-head gating: gates = 2·sigmoid(to_gate(x)).
        let gates = self.to_gate.fwd(x)?.sigmoid()?.mul_scalar(2.0)?; // [b,t,h]
        let out = out
            .reshape(vec![b, t, h, dh])?
            .broadcast_mul(&gates.reshape(vec![b, t, h, 1])?)?
            .contiguous()?
            .reshape(vec![b, t, h * dh])?;
        self.to_out.fwd(&out)
    }

    fn ff(&self, x: &Tensor) -> R<Tensor> {
        self.ff2.fwd(&self.ff0.fwd(x)?.gelu_tanh()?)
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> R<Tensor> {
        let x = x.add(&self.attn(&rms_no_gain(x)?, cos, sin)?)?;
        let x = x.add(&self.ff(&rms_no_gain(&x)?)?)?;
        Ok(x)
    }
}

/// Embeddings1DConnector (перцивер-реземплер): learnable-register замена пэдов
/// (маска занулена → bidirectional), SPLIT-RoPE, N блоков, финальная RMS.
struct Connector {
    registers: Tensor, // [num_reg, inner]
    blocks: Vec<ConnectorBlock>,
    heads: usize,
    dim_head: usize,
    inner: usize,
    num_registers: usize,
    theta: f64,
    max_pos: f64,
    device: Device,
    dtype: DType,
}

impl Connector {
    fn load(
        ckpt: &LtxCheckpoint,
        prefix: &str,
        heads: usize,
        dim_head: usize,
        num_layers: usize,
        num_registers: usize,
        theta: f64,
        max_pos: f64,
        device: Device,
        dtype: DType,
    ) -> Result<Self, LtxError> {
        let registers = ckpt.get(&format!("{prefix}.learnable_registers"))?;
        let mut blocks = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            blocks.push(ConnectorBlock::load(
                ckpt,
                &format!("{prefix}.transformer_1d_blocks.{l}"),
                heads,
                dim_head,
            )?);
        }
        Ok(Self {
            registers,
            blocks,
            heads,
            dim_head,
            inner: heads * dim_head,
            num_registers,
            theta,
            max_pos,
            device,
            dtype,
        })
    }

    /// `features` `[1,T,inner]` (уже right-pad-reordered), `valid` `[1,T,1]`
    /// (1.0 для валидных-первых, 0.0 для пэдов). Возвращает `[1,T,inner]`.
    fn forward(&self, features: &Tensor, valid: &Tensor) -> R<Tensor> {
        let t = features.dims()[1];
        assert_eq!(t % self.num_registers, 0, "seq_len must be divisible by num_registers");
        // registers tiled [1,T,inner], замена пэдов.
        let reps = t / self.num_registers;
        let reg = self.registers.reshape(vec![1, self.num_registers, self.inner])?;
        let reg = if reps > 1 {
            let parts: Vec<&Tensor> = (0..reps).map(|_| &reg).collect();
            Tensor::cat(&parts, 1)?
        } else {
            reg
        }; // [1,T,inner]
        let one_minus = valid.affine(-1.0, 1.0)?; // 1-valid
        let mut hidden = features.broadcast_mul(valid)?.add(&reg.broadcast_mul(&one_minus)?)?;

        let (cos, sin) = split_rope_cos_sin(
            t, self.heads, self.dim_head, self.theta, self.max_pos, self.device, self.dtype,
        )?;
        for blk in &self.blocks {
            hidden = blk.forward(&hidden, &cos, &sin)?;
        }
        rms_no_gain(&hidden)
    }
}

/// FeatureExtractorV2 (видео): stack 49 → per-token RMS(D) → reshape `[..,D*L]`
/// (L=fastest) → mask-zero → ×√(out/D) → video_aggregate_embed.
struct FeatureExtractorV2 {
    aggregate: Linear,
    embedding_dim: usize, // 3840
}

impl FeatureExtractorV2 {
    /// `proj_key` = `video_aggregate_embed` (→4096) или `audio_aggregate_embed` (→2048).
    fn load(ckpt: &LtxCheckpoint, proj_key: &str, embedding_dim: usize) -> Result<Self, LtxError> {
        Ok(Self {
            aggregate: Linear::load(ckpt, TEXT_PROJ_PREFIX, proj_key, true)?,
            embedding_dim,
        })
    }

    /// `states`: 49 тензоров `[1,T,D]`. `valid` `[1,T,1]` (1/0). → `[1,T,out]`.
    fn forward(&self, states: &[Tensor], valid: &Tensor) -> R<Tensor> {
        let (b, t, d) = (states[0].dims()[0], states[0].dims()[1], states[0].dims()[2]);
        let l = states.len();
        // stack по последней оси → [B,T,D,L]
        let expanded: Vec<Tensor> =
            states.iter().map(|s| s.reshape(vec![b, t, d, 1])).collect::<R<Vec<_>>>()?;
        let refs: Vec<&Tensor> = expanded.iter().collect();
        let encoded = Tensor::cat(&refs, 3)?; // [B,T,D,L]
        // per-token RMS по D (dim=2), keepdim → [B,T,1,L]
        let var = encoded.sqr()?.mean_keepdim(2)?;
        let denom = var.add_scalar(NORM_EPS as f32)?.sqrt()?;
        let normed = encoded.broadcast_div(&denom)?;
        // reshape [B,T,D*L] (L fastest, row-major)
        let normed = normed.reshape(vec![b, t, d * l])?;
        // mask-zero пэд-токены ([B,T,1] broadcast)
        let normed = normed.broadcast_mul(valid)?;
        // rescale ×√(out/D) и проекция
        let out_dim = self.aggregate.w.dims()[0];
        let scale = ((out_dim as f64 / self.embedding_dim as f64).sqrt()) as f32;
        let rescaled = normed.mul_scalar(scale)?;
        self.aggregate.fwd(&rescaled)
    }
}

/// Общая логика кондишена: FE → right-pad reorder (валидные-первые) → коннектор.
/// `states`: 49×`[1,T,3840]`, `mask`: `[T]` (1=valid/0=pad, left-pad).
fn condition(fe: &FeatureExtractorV2, connector: &Connector, device: Device, states: &[Tensor], mask: &[u32]) -> R<Tensor> {
    let t = mask.len();
    let mask_f: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
    let valid_orig = Tensor::from_vec(mask_f, vec![1, t, 1], device)?.to_dtype(states[0].dtype())?;
    let feats = fe.forward(states, &valid_orig)?;

    let mut order: Vec<u32> = (0..t as u32).filter(|&i| mask[i as usize] != 0).collect();
    order.extend((0..t as u32).filter(|&i| mask[i as usize] == 0));
    let n_valid = order.len() - mask.iter().filter(|&&m| m == 0).count();
    let idx = Tensor::from_vec(order, vec![t], device)?;
    let feats_reordered = feats.index_select(1, &idx)?;

    let mut rmask = vec![0f32; t];
    for v in rmask.iter_mut().take(n_valid) {
        *v = 1.0;
    }
    let valid_re = Tensor::from_vec(rmask, vec![1, t, 1], device)?.to_dtype(states[0].dtype())?;
    connector.forward(&feats_reordered, &valid_re)
}

/// Текст-кондишен аудио-путь (FeatureExtractorV2 audio_aggregate + аудио-коннектор).
/// → `audio_encoding [1,T,2048]`.
pub struct AudioTextConditioner {
    fe: FeatureExtractorV2,
    connector: Connector,
    device: Device,
}

impl AudioTextConditioner {
    pub fn load(ckpt: &LtxCheckpoint, device: Device, dtype: DType) -> Result<Self, LtxError> {
        let t = &ckpt.config.transformer;
        let fe = FeatureExtractorV2::load(ckpt, "audio_aggregate_embed", t.caption_channels)?;
        let connector = Connector::load(
            ckpt,
            &format!("{DIT_PREFIX}.audio_embeddings_connector"),
            t.audio_connector_num_attention_heads,
            t.audio_connector_attention_head_dim,
            t.connector_num_layers,
            t.connector_num_learnable_registers,
            10000.0,
            t.connector_positional_embedding_max_pos[0] as f64,
            device,
            dtype,
        )?;
        Ok(Self { fe, connector, device })
    }

    /// `states`: 49×`[1,T,3840]`, `mask`: `[T]` (1=valid/0=pad). → `audio_encoding [1,T,2048]`.
    pub fn forward(&self, states: &[Tensor], mask: &[u32]) -> Result<Tensor, LtxError> {
        Ok(condition(&self.fe, &self.connector, self.device, states, mask)?)
    }
}

/// Текст-кондишен видео-путь end-to-end (FeatureExtractorV2 + видео-коннектор).
pub struct VideoTextConditioner {
    fe: FeatureExtractorV2,
    connector: Connector,
    device: Device,
}

impl VideoTextConditioner {
    pub fn load(ckpt: &LtxCheckpoint, device: Device, dtype: DType) -> Result<Self, LtxError> {
        let t = &ckpt.config.transformer;
        let fe = FeatureExtractorV2::load(ckpt, "video_aggregate_embed", t.caption_channels)?;
        let connector = Connector::load(
            ckpt,
            &format!("{DIT_PREFIX}.video_embeddings_connector"),
            t.connector_num_attention_heads,
            t.connector_attention_head_dim,
            t.connector_num_layers,
            t.connector_num_learnable_registers,
            10000.0, // positional_embedding_theta коннектора (дефолт модуля)
            t.connector_positional_embedding_max_pos[0] as f64,
            device,
            dtype,
        )?;
        Ok(Self { fe, connector, device })
    }

    /// `states`: 49 тензоров `[1,T,3840]`. `mask`: `[T]` (1=valid/0=pad, left-pad).
    /// Возвращает `video_encoding [1,T,4096]` (в right-pad-reordered порядке, как
    /// эталон LTX).
    pub fn forward(&self, states: &[Tensor], mask: &[u32]) -> Result<Tensor, LtxError> {
        Ok(condition(&self.fe, &self.connector, self.device, states, mask)?)
    }
}
