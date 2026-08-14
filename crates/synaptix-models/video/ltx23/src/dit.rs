//! LTX-2.3 video-only DiT (`AVTransformer3DModel`, путь run_vx): denoiser,
//! предсказывает velocity. 48 блоков: adaLN-single self-attn (qk-norm + gated +
//! 3D SPLIT-RoPE) → text-cross-attn (cross_attention_adaln) → gelu-tanh FF.
//! X0: denoised = latent − velocity·timesteps.
//!
//! Аудио-поток и A2V/V2A cross-attn — на Фазе 8. Примитивы (linear/rms/rope-apply)
//! переиспользованы из [`crate::text_encoder`].

use std::f64::consts::PI;
use std::sync::Arc;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_nn::module::Module as _;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::loader::{LtxCheckpoint, DIT_PREFIX};
use crate::text_encoder::{apply_split_rope, rms_gain, rms_no_gain};
use crate::LtxError;

type R<T> = Result<T, SynaptixError>;
const EPS: f64 = 1e-6;

/// Синусоидальный timestep-эмбеддинг (PixArt time_proj): 256 каналов,
/// flip_sin_to_cos=True (→ [cos, sin]), downscale_freq_shift=0, max_period=10000.
/// `vals` — таймстепы (уже ×scale_multiplier). Возвращает host-вектор `[N*256]`.
fn timestep_sinusoidal(vals: &[f32]) -> Vec<f32> {
    let dim = 256usize;
    let half = dim / 2; // 128
    let max_period = 10000f64;
    let mut emb = vec![0f32; vals.len() * dim];
    fn fill(emb: &mut [f32], vals: &[f32], dim: usize, half: usize, max_period: f64) {
        for (n, &t) in vals.iter().enumerate() {
            for i in 0..half {
                let exponent = -max_period.ln() * (i as f64) / (half as f64); // downscale_freq_shift=0
                let freq = exponent.exp();
                let ang = t as f64 * freq;
                // flip_sin_to_cos → [cos(ang) (0..half), sin(ang) (half..dim)]
                emb[n * dim + i] = ang.cos() as f32;
                emb[n * dim + half + i] = ang.sin() as f32;
            }
        }
    }
    // Per-token timesteps (s2-refine: len==Tv): серийная CPU-тригонометрия была
    // дырой пролога; параллель по чанкам бит-в-бит (формулы поэлементны).
    if vals.len() >= 2048 {
        let nthr = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
        let chunk = vals.len().div_ceil(nthr);
        std::thread::scope(|sp| {
            for (vc, ec) in vals.chunks(chunk).zip(emb.chunks_mut(chunk * dim)) {
                sp.spawn(move || fill(ec, vc, dim, half, max_period));
            }
        });
    } else {
        fill(&mut emb, vals, dim, half, max_period);
    }
    emb
}

/// 3D SPLIT-RoPE для видео: positions `[3,T,2]` (start,end в пиксель-координатах,
/// host f64), middle-indices grid, max_pos[3], freq-grid f64, pad front.
/// → cos,sin `[1, heads, T, dim_head/2]`.
#[allow(clippy::too_many_arguments)]
fn rope3d(
    positions: &[f64], // flat [n_pos * T * 2], pos[(d*T+t)*2 + {0,1}]
    n_pos: usize,
    t: usize,
    heads: usize,
    dim_head: usize,
    theta: f64,
    max_pos: &[f64],
    device: Device,
    dtype: DType,
) -> R<(Tensor, Tensor)> {
    let inner = heads * dim_head;
    let half = inner / 2; // ожидаемое число частот (= dim_head/2 * heads)
    let n_elem = 2 * n_pos;
    let count = inner / n_elem; // частот на позиционную ось
    // indices[i] = theta^(i/(count-1)) * pi/2
    // freq-grid: f64 как в Python, НО Python кастует результат в f32 перед
    // умножением на frac (generate_freq_grid_np → torch.float32) — повторяем,
    // иначе на высоких частотах f64-угол расходится с f32-эталоном.
    let mut indices = vec![0f32; count];
    for (i, idx) in indices.iter_mut().enumerate() {
        let e = if count > 1 { i as f64 / (count - 1) as f64 } else { 0.0 };
        *idx = (theta.powf(e) * PI / 2.0) as f32;
    }
    // freqs[t][c*n_pos + d] = indices[c] * (frac[d][t]*2 - 1), frac = mid/max_pos
    let n_freq = count * n_pos;
    let pad = half - n_freq; // pad спереди (cos=1, sin=0)
    let head_half = dim_head / 2;
    let mut cos = vec![0f32; heads * t * head_half];
    let mut sin = vec![0f32; heads * t * head_half];
    struct SendPtr(*mut f32);
    // SAFETY: потоки пишут только свои p-диапазоны — выходные индексы
    // (h*t+p)*head_half+f не пересекаются между разными p.
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    let fill = |p0: usize, p1: usize, cos: &SendPtr, sin: &SendPtr| {
        for p in p0..p1 {
            // padded ряд длины `half`: [1.0*pad, cos(freqs)...]
            let mut row_cos = vec![1f32; half];
            let mut row_sin = vec![0f32; half];
            for c in 0..count {
                for d in 0..n_pos {
                    let start = positions[(d * t + p) * 2];
                    let end = positions[(d * t + p) * 2 + 1];
                    let mid = (start + end) / 2.0;
                    let frac = (mid / max_pos[d]) as f32; // f32 как в Python
                    let ang = indices[c] * (frac * 2.0 - 1.0); // f32
                    let j = pad + c * n_pos + d;
                    row_cos[j] = ang.cos();
                    row_sin[j] = ang.sin();
                }
            }
            // reshape [half] → [heads, head_half]: элемент (h,f) = row[h*head_half+f]
            for h in 0..heads {
                for f in 0..head_half {
                    let o = (h * t + p) * head_half + f;
                    // SAFETY: см. SendPtr — диапазоны p непересекающиеся.
                    unsafe {
                        *cos.0.add(o) = row_cos[h * head_half + f];
                        *sin.0.add(o) = row_sin[h * head_half + f];
                    }
                }
            }
        }
    };
    let cptr = SendPtr(cos.as_mut_ptr());
    let sptr = SendPtr(sin.as_mut_ptr());
    if t >= 2048 {
        let nthr = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
        let chunk = t.div_ceil(nthr);
        std::thread::scope(|sp| {
            let (fill, cptr, sptr) = (&fill, &cptr, &sptr);
            let mut p0 = 0usize;
            while p0 < t {
                let p1 = (p0 + chunk).min(t);
                sp.spawn(move || fill(p0, p1, cptr, sptr));
                p0 = p1;
            }
        });
    } else {
        fill(0, t, &cptr, &sptr);
    }
    let cos = Tensor::from_vec(cos, vec![1, heads, t, head_half], device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, vec![1, heads, t, head_half], device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Оценка резидентного VRAM-размера DiT (байты весов) при квантовании блочных
/// linear'ов в `quant` (остальное в `compute`). Для авто-решения резидент-vs-offload
/// в CLI: 22B в MXFP8 ≈ 22.8GB — не влезает в 24GB ни при каком разрешении.
/// Правила совместимости зеркалят `QuantLinear::build` (несовместимая форма → dense).
pub fn dit_resident_bytes(ckpt: &LtxCheckpoint, quant: DType, compute: DType) -> usize {
    ckpt.infos()
        .filter(|(name, _, _)| name.starts_with(DIT_PREFIX))
        .map(|(name, _, shape)| {
            let numel: usize = shape.iter().product();
            let quantizable = name.contains(".transformer_blocks.")
                && name.ends_with(".weight")
                && shape.len() == 2;
            if quantizable {
                let (n, k) = (shape[0], shape[1]);
                match quant {
                    DType::NVFP4 if n % 64 == 0 && k % 64 == 0 => numel / 2 + numel / 16,
                    DType::MXFP8 if k % 32 == 0 => numel + numel / 32,
                    _ => compute.bytes_for_numel(numel),
                }
            } else {
                compute.bytes_for_numel(numel)
            }
        })
        .sum()
}

/// Linear с опциональным квантом веса. `qdt`=MXFP8/NVFP4 → квантуется (большие
/// блочные linear'ы → влезает в 24GB); `qdt`=compute → плотный (мелкие
/// patchify/adaln/proj_out, важны для точности модуляции).
struct Lin(QuantLinear);
impl Lin {
    fn load(
        ckpt: &LtxCheckpoint,
        prefix: &str,
        key: &str,
        bias: bool,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, LtxError> {
        let mut w = ckpt.get_raw(&format!("{prefix}.{key}.weight"))?;
        // distilled-LoRA merge: W += (B·strength)@A (до кванта). None если LoRA не трогает.
        if let Some(d) = ckpt.lora_delta(&format!("{prefix}.{key}"), w.dtype())? {
            let d = if d.device() == w.device() { d } else { d.to_device(w.device()).map_err(LtxError::from)? };
            w = w.add(&d).map_err(LtxError::from)?;
        }
        let b = if bias { Some(ckpt.get_raw(&format!("{prefix}.{key}.bias"))?) } else { None };
        let ql = QuantLinear::build(w, b, qdt, compute)
            .map_err(|e| LtxError::Load(format!("qlinear {prefix}.{key}: {e}")))?;
        Ok(Self(ql))
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        // Чанк по T при длинных последовательностях (только квант-вес):
        // QuantLinear::forward держит f16-каст входа + f16-выход + bf16-каст
        // (по [T,·] каждый; cross-attn проекции 20s-refine шли этим путём —
        // x_pq=None — и добивали VRAM). Построчная независимость → бит-в-бит.
        let chunk_m = lin_chunk_m();
        if chunk_m > 0
            && matches!(&self.0, QuantLinear::Quant { .. })
            && x.rank() == 3
            && x.dims()[0] == 1
            && x.dims()[1] > chunk_m
        {
            let t = x.dims()[1];
            let chunk = chunk_m;
            // выход предвыделяется ДО чанков и заполняется D2D-кусками: cat в
            // конце требовал цельный [T,n]-кусок на фрагментированном пуле + 
            // копию всех частей (nvfp4-19s OOM на cat)
            let mut out: Option<Tensor> = None;
            let mut o = 0usize;
            while o < t {
                let n = chunk.min(t - o);
                let yc = self.0.forward(&x.narrow(1, o, n)?.contiguous()?)?;
                if out.is_none() {
                    out = Some(Tensor::empty_uninit(
                        vec![1, t, yc.dims()[2]], yc.dtype(), yc.device(),
                    )?);
                }
                out.as_mut().unwrap().copy_rows_from(o, &yc)?;
                o += n;
            }
            return Ok(out.expect("chunked fwd: t > 0"));
        }
        self.0.forward(x)
    }
    /// Формат квант-веса (NVFP4|MXFP8) для выбора формата prequant-пары; None = Dense.
    fn quant_dtype(&self) -> Option<DType> {
        match &self.0 {
            QuantLinear::Quant { w, .. } => Some(w.dtype()),
            QuantLinear::Dense(_) => None,
        }
    }
    /// Проекция из УЖЕ квантованной активации (packed, scales от
    /// rms_mod_quant_{nvfp4,mxfp8} / *_quantize_act): пропускает f16-каст и
    /// квант. Возвращает [m, n] в `out_dt`. Формат пары = формат веса.
    fn fwd_prequant(&self, packed: &Tensor, scales: &Tensor, m: usize, out_dt: DType) -> R<Tensor> {
        match &self.0 {
            QuantLinear::Quant { w, bias } => {
                // Чанк по строкам M при длинных T: GEMM-выход f16 [m,n] + bf16-каст
                // + bias-add держали по 2-3 полных буфера (440MB×3 на T=53.7k,
                // 20s-refine OOM); построчная независимость → бит-в-бит. Скейлы
                // m-блочные по 128 строк (nvfp4 swizzle/паддинг, mxfp8 row-major)
                // → чанк кратен 128.
                let chunk_m = lin_chunk_m();
                if chunk_m > 0 && m > chunk_m {
    let k = w.k();
                    let p_total = packed.numel();
                    let sc_total = scales.numel();
                    let p_row = p_total / m; // байт упаковки на строку (nvfp4 k/2, mxfp8 k)
                    // байт скейлов на строку: mxfp8 row-major k/32; nvfp4 k/16
                    // (k.div_ceil(64)·4 при k%64==0, наши k кратны)
                    let sc_row = match w.dtype() {
                        DType::MXFP8 => k / 32,
                        _ => k / 16,
                    };
                    let mut out: Option<Tensor> = None;
                    let mut m0 = 0usize;
                    while m0 < m {
                        let mc = chunk_m.min(m - m0);
                        let p_sl = packed.narrow(0, m0 * p_row, mc * p_row)?.contiguous()?;
                        let sc0 = m0 * sc_row;
                        let sc_len = (mc.div_ceil(128) * 128 * sc_row).min(sc_total - sc0);
                        let sc_sl = scales.narrow(0, sc0, sc_len)?.contiguous()?;
                        let y = p_sl.linear_quant_prequant(&sc_sl, w, mc, DType::F16)?;
                        let y = if out_dt == DType::F16 { y } else { y.to_dtype(out_dt)? };
                        let y = match bias {
                            Some(b) => y.broadcast_add(b)?,
                            None => y,
                        };
                        // сборка в предвыделенный выход без cat (см. Lin::fwd)
                        let y3 = y.reshape(vec![1, mc, y.dims()[1]])?;
                        if out.is_none() {
                            out = Some(Tensor::empty_uninit(
                                vec![1, m, y3.dims()[2]], y3.dtype(), y3.device(),
                            )?);
                        }
                        out.as_mut().unwrap().copy_rows_from(m0, &y3)?;
                        m0 += mc;
                    }
                    let out = out.expect("chunked prequant: m > 0");
                    let n_out = out.dims()[2];
                    return Ok(out.reshape(vec![m, n_out])?);
                }
                let y = packed.linear_quant_prequant(scales, w, m, DType::F16)?;
                let y = if out_dt == DType::F16 { y } else { y.to_dtype(out_dt)? };
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
            QuantLinear::Dense(_) => Err(SynaptixError::Unsupported("fwd_prequant: Dense")),
        }
    }
    fn to_device(&self, dev: Device) -> R<Self> {
        Ok(Self(self.0.to_device(dev)?))
    }
}

/// Порог чанков Lin::fwd/fwd_prequant по строкам (VRAM-пики vs дробление пула;
/// 0 = чанки выкл — крупные аллокации).
fn lin_chunk_m() -> usize {
    16384
}

/// Attention блока DiT (self или cross). qk-norm ×weight, gated 2·sigmoid, to_out.
struct Attn {
    q_norm: Tensor,
    k_norm: Tensor,
    to_q: Lin,
    to_k: Lin,
    to_v: Lin,
    to_gate: Lin,
    to_out: Lin,
    heads: usize,
    dim_head: usize,
    scale: f32,
}

impl Attn {
    fn load(
        ckpt: &LtxCheckpoint,
        prefix: &str,
        heads: usize,
        dim_head: usize,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, LtxError> {
        Ok(Self {
            q_norm: ckpt.get(&format!("{prefix}.q_norm.weight"))?,
            k_norm: ckpt.get(&format!("{prefix}.k_norm.weight"))?,
            to_q: Lin::load(ckpt, prefix, "to_q", true, qdt, compute)?,
            to_k: Lin::load(ckpt, prefix, "to_k", true, qdt, compute)?,
            to_v: Lin::load(ckpt, prefix, "to_v", true, qdt, compute)?,
            to_gate: Lin::load(ckpt, prefix, "to_gate_logits", true, qdt, compute)?,
            to_out: Lin::load(ckpt, prefix, "to_out.0", true, qdt, compute)?,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f32).sqrt(),
        })
    }

    fn to_device(&self, dev: Device) -> R<Self> {
        Ok(Self {
            q_norm: self.q_norm.to_device(dev)?,
            k_norm: self.k_norm.to_device(dev)?,
            to_q: self.to_q.to_device(dev)?,
            to_k: self.to_k.to_device(dev)?,
            to_v: self.to_v.to_device(dev)?,
            to_gate: self.to_gate.to_device(dev)?,
            to_out: self.to_out.to_device(dev)?,
            heads: self.heads,
            dim_head: self.dim_head,
            scale: self.scale,
        })
    }

    /// `x` `[1,Tq,Dq]` (query). `context` для cross (`[1,Tk,Dk]`) или None (self).
    /// `cos/sin` — RoPE на q,k (только self; для cross None).
    fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> R<Tensor> {
        let qr = cos.zip(sin);
        self.forward2(x, context, qr, qr, None)
    }

    /// Как [`Attn::forward`], но с prequant-активацией `x` (packed, scales) от
    /// rms_mod_quant_nvfp4 — q (и k/v при self-attn) идут без повторного кванта.
    fn forward_pq(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        x_pq: Option<(&Tensor, &Tensor)>,
    ) -> R<Tensor> {
        let qr = cos.zip(sin);
        self.forward2(x, context, qr, qr, x_pq)
    }

    /// Как [`Attn::forward`], но с РАЗДЕЛЬНЫМ RoPE для q (`q_rope`) и k (`k_rope`) —
    /// нужно для cross-modal A2V/V2A (q-поток и k-поток имеют свои позиции).
    /// Для self: `q_rope==k_rope`. Для text-cross: оба None.
    fn forward2(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        q_rope: Option<(&Tensor, &Tensor)>,
        k_rope: Option<(&Tensor, &Tensor)>,
        x_pq: Option<(&Tensor, &Tensor)>,
    ) -> R<Tensor> {
        let ctx = context.unwrap_or(x);
        let (b, tq) = (x.dims()[0], x.dims()[1]);
        let tk = ctx.dims()[1];
        let (h, dh) = (self.heads, self.dim_head);
        let dt = x.dtype();
        // под-фазная разбивка attention (sync-точки) — диагностика роя мелких
        // ядер вокруг flash.
        let aprof = crate::runtime::ltx_attn_prof();
        let mut tmarks: Vec<(&'static str, f64)> = Vec::new();
        let mut mark = |name: &'static str, t0: &mut std::time::Instant| {
            if aprof {
                let _ = synaptix_core::device::cuda::synchronize(0);
                tmarks.push((name, t0.elapsed().as_secs_f64()));
                *t0 = std::time::Instant::now();
            }
        };
        let mut t0 = std::time::Instant::now();
        // prequant x: q всегда; k/v — только self-attn (ctx == x). Формат пары
        // = формат to_q (производитель выбирал по нему) → гейт на совпадение.
        let pq_fmt = self.to_q.quant_dtype();
        let pq_proj = |lin: &Lin, t: usize| -> Option<R<Tensor>> {
            let (p, sc) = x_pq?;
            if pq_fmt.is_none() || lin.quant_dtype() != pq_fmt {
                return None;
            }
            // Форма вне prequant-пути (напр. to_gate n=32 у mxfp8) → None →
            // фолбэк на обычный fwd (бит-инвариант: fwd == fwd_prequant).
            match lin.fwd_prequant(p, sc, b * t, dt) {
                Ok(y) => Some(y.reshape(vec![b, t, y.dims()[1]])),
                Err(_) => None,
            }
        };
        let q_lin = match pq_proj(&self.to_q, tq) {
            Some(r) => r?,
            None => self.to_q.fwd(x)?,
        };
        let self_attn = context.is_none();
        let k_lin = match if self_attn { pq_proj(&self.to_k, tk) } else { None } {
            Some(r) => r?,
            None => self.to_k.fwd(ctx)?,
        };
        let v_lin = match if self_attn { pq_proj(&self.to_v, tk) } else { None } {
            Some(r) => r?,
            None => self.to_v.fwd(ctx)?,
        };
        mark("qkv_proj", &mut t0);
        let q = rms_gain(&q_lin, &self.q_norm)?
            .reshape(vec![b, tq, h, dh])?.transpose(1, 2)?.contiguous()?;
        let k = rms_gain(&k_lin, &self.k_norm)?
            .reshape(vec![b, tk, h, dh])?.transpose(1, 2)?.contiguous()?;
        let v = v_lin
            .reshape(vec![b, tk, h, dh])?.transpose(1, 2)?.contiguous()?;
        mark("norm_xpose", &mut t0);
        let q = match q_rope { Some((c, s)) => apply_split_rope(&q, c, s)?, None => q };
        let k = match k_rope { Some((c, s)) => apply_split_rope(&k, c, s)?, None => k };
        mark("rope", &mut t0);
        // flash (tensor-core) на bf16/f16 CUDA — O(T) память (критично для FullHD,
        // где наивный scores [b,h,T,T] не влезает); f32/CPU → наивный (bit-exact).
        let attn = match q.dtype() {
            DType::BF16 | DType::F16 => q
                .flash_attention(&k, &v, self.scale, false)
                .or_else(|_| scaled_dot_attention(&q, &k, &v, self.scale, None))?,
            _ => scaled_dot_attention(&q, &k, &v, self.scale, None)?,
        }; // [b,h,tq,dh]
        mark("flash", &mut t0);
        let out = attn.transpose(1, 2)?.contiguous()?.reshape(vec![b, tq, h * dh])?;
        let gates_lin = match pq_proj(&self.to_gate, tq) {
            Some(r) => r?,
            None => self.to_gate.fwd(x)?,
        };
        let gates = gates_lin.sigmoid()?.mul_scalar(2.0)?; // [b,tq,h]
        let out = out
            .reshape(vec![b, tq, h, dh])?
            .broadcast_mul(&gates.reshape(vec![b, tq, h, 1])?)?
            .contiguous()?
            .reshape(vec![b, tq, h * dh])?;
        mark("gate_xpose", &mut t0);
        let r = self.to_out.fwd(&out);
        mark("out_proj", &mut t0);
        if aprof {
            let s: Vec<String> = tmarks.iter().map(|(n, t)| format!("{n}={:.2}ms", t * 1e3)).collect();
            eprintln!("[ATTN_PROF tq={tq} tk={tk}] {}", s.join(" "));
        }
        r
    }
}

struct Block {
    sst: Tensor,        // [9, dim] F32
    prompt_sst: Tensor, // [2, dim] F32
    attn1: Attn,
    attn2: Attn,
    ff0: Lin,
    ff2: Lin,
}

impl Block {
    fn load(
        ckpt: &LtxCheckpoint,
        idx: usize,
        heads: usize,
        dim_head: usize,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, LtxError> {
        let p = format!("{DIT_PREFIX}.transformer_blocks.{idx}");
        Ok(Self {
            sst: ckpt.get_raw(&format!("{p}.scale_shift_table"))?.to_dtype(DType::F32)?,
            prompt_sst: ckpt.get_raw(&format!("{p}.prompt_scale_shift_table"))?.to_dtype(DType::F32)?,
            attn1: Attn::load(ckpt, &format!("{p}.attn1"), heads, dim_head, qdt, compute)?,
            attn2: Attn::load(ckpt, &format!("{p}.attn2"), heads, dim_head, qdt, compute)?,
            ff0: Lin::load(ckpt, &p, "ff.net.0.proj", true, qdt, compute)?,
            ff2: Lin::load(ckpt, &p, "ff.net.2", true, qdt, compute)?,
        })
    }

    /// Перенос блока между устройствами (host-stream квант-блоков: квантуем 1× на
    /// GPU → CPU-резидент → стрим на GPU по требованию в forward).
    fn to_device(&self, dev: Device) -> R<Self> {
        Ok(Self {
            sst: self.sst.to_device(dev)?,
            prompt_sst: self.prompt_sst.to_device(dev)?,
            attn1: self.attn1.to_device(dev)?,
            attn2: self.attn2.to_device(dev)?,
            ff0: self.ff0.to_device(dev)?,
            ff2: self.ff2.to_device(dev)?,
        })
    }
}

/// `AdaLayerNormSingle`: `emb.timestep_embedder.{linear_1,linear_2}` + `linear`.
/// `modulate(vals,n) → (modul [n,k·dim], embedded [n,dim])`.
struct AdaLN {
    te1: Lin,
    te2: Lin,
    lin: Lin,
}
impl AdaLN {
    fn load(ckpt: &LtxCheckpoint, prefix: &str, dt: DType) -> Result<Self, LtxError> {
        Ok(Self {
            te1: Lin::load(ckpt, prefix, "emb.timestep_embedder.linear_1", true, dt, dt)?,
            te2: Lin::load(ckpt, prefix, "emb.timestep_embedder.linear_2", true, dt, dt)?,
            lin: Lin::load(ckpt, prefix, "linear", true, dt, dt)?,
        })
    }
    fn to_device(&self, dev: Device) -> R<Self> {
        Ok(Self { te1: self.te1.to_device(dev)?, te2: self.te2.to_device(dev)?, lin: self.lin.to_device(dev)? })
    }
    fn modulate(&self, vals_scaled: &[f32], n: usize, device: Device, dtype: DType) -> R<(Tensor, Tensor)> {
        adaln_mod(vals_scaled, &self.te1, &self.te2, &self.lin, n, device, dtype)
    }
}

/// adaln-single: sinusoidal → timestep_embedder → linear → (modulation, embedded).
fn adaln_mod(vals_scaled: &[f32], te1: &Lin, te2: &Lin, lin: &Lin, n: usize, device: Device, dtype: DType) -> R<(Tensor, Tensor)> {
    let sinus = timestep_sinusoidal(vals_scaled); // [n*256]
    let proj = Tensor::from_vec(sinus, vec![n, 256], device)?.to_dtype(dtype)?;
    let emb = te2.fwd(&te1.fwd(&proj)?.silu()?)?; // [n, dim] embedded_timestep
    let modul = lin.fwd(&emb.silu()?)?; // [n, k*dim]
    Ok((modul, emb))
}

/// Fused «модуляция+квант» (vs decomposed-цепочка rms→+1→mul→add без prequant);
/// доказан бит-в-бит — всегда вкл.
fn norm_quant_on() -> bool {
    true
}

/// adaLN-модулированная норма + (опц.) prequant одним ядром (формат по весу
/// потребителя: NVFP4|MXFP8): возвращает (модулированный y, Some((packed,
/// scales)) если fused-путь сработал). Бит-в-бит с decomposed-цепочкой
/// (редукция/раунды повторены в ядре).
fn mod_norm_quant(
    vx: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    want: Option<DType>,
) -> R<(Tensor, Option<(Tensor, Tensor)>)> {
    use synaptix_core::dtype::DType as DT;
    if norm_quant_on()
        && matches!(vx.device(), Device::Cuda(_))
        && matches!(vx.dtype(), DT::BF16 | DT::F16)
    {
        let fused = match want {
            Some(DT::NVFP4) => vx.rms_mod_quant_nvfp4(scale, shift, 1e-6).ok().map(|(y, p, sc)| (y, Some((p, sc)))),
            Some(DT::MXFP8) => vx.rms_mod_quant_mxfp8(scale, shift, 1e-6).ok().map(|(y, p, sc)| (y, Some((p, sc)))),
            // Dense (bf16): fused норма+модуляция ТЕМ ЖЕ ядром (y бит-в-бит с
            // decomposed-цепочкой — гейт cuda_rms_mod_quant), квант-выход
            // выбрасываем: лишняя packed-запись копеечна против ~10 DRAM-проходов.
            _ => vx.rms_mod_quant_nvfp4(scale, shift, 1e-6).ok().map(|(y, _, _)| (y, None)),
        };
        if let Some((y, pq)) = fused {
            return Ok((y, pq));
        }
    }
    let y = crate::text_encoder::rms_no_gain(vx)?
        .broadcast_mul(&scale.add_scalar(1.0)?)?
        .broadcast_add(shift)?;
    Ok((y, None))
}

/// `[B,T,dim]` модуляция: для строк `lo..hi` таблицы → sst[r] + mods[r].
/// `mods` — предвырезанные ряды modul[:,:,r,:] (contiguous [B,T,dim]), режутся
/// ОДИН раз на шаг (см. split_modul) — strided-копии не повторяются на блок.
/// Gated-residual `x + y*g` одним ядром (fused бит-в-бит с decomposed;
/// фоллбэк — broadcast_mul→add).
fn gate_residual(x: &Tensor, y: &Tensor, g: &Tensor) -> R<Tensor> {
    if matches!(x.device(), Device::Cuda(_)) {
        if let Ok(o) = x.fused_gate_residual(y, g) {
            return Ok(o);
        }
    }
    x.add(&y.broadcast_mul(g)?)
}

/// adaLN-модуляция готовой нормы `x*(1+s)+sh` (`s`/`sh` строки [1,1,d]) одним
/// ядром (fused бит-в-бит с decomposed; фоллбэк — add_scalar→mul→add).
fn mod_row(x: &Tensor, s: &Tensor, sh: &Tensor) -> R<Tensor> {
    if matches!(x.device(), Device::Cuda(_)) {
        if let Ok(o) = x.fused_mod_row(s, sh) {
            return Ok(o);
        }
    }
    x.broadcast_mul(&s.add_scalar(1.0)?)?.broadcast_add(sh)
}

fn ada(sst: &Tensor, mods: &[Tensor], lo: usize, hi: usize, dtype: DType) -> R<Vec<Tensor>> {
    let dim = sst.dims()[1];
    let mut out = Vec::with_capacity(hi - lo);
    for r in lo..hi {
        let table = sst.narrow(0, r, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(dtype)?;
        out.push(mods[r].broadcast_add(&table)?);
    }
    Ok(out)
}

/// Разрез modul `[1,T,9,dim]` на 9 contiguous-рядов `[1,T,dim]` (раз на шаг).
fn split_modul(modul: &Tensor) -> R<Vec<Tensor>> {
    (0..modul.dims()[2])
        .map(|r| modul.narrow(2, r, 1)?.squeeze(2)?.contiguous())
        .collect()
}

/// LayerNorm по последней оси без affine (`norm_out`), biased var. Считается в
/// f32 (как torch), возвращается dtype входа.
fn layer_norm_no_affine(x: &Tensor) -> R<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let last = xf.rank() - 1;
    let mean = xf.mean_keepdim(last)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(last)?;
    xc.broadcast_div(&var.add_scalar(EPS as f32)?.sqrt()?)?.to_dtype(dt)
}

/// Video-only DiT поверх чекпойнта LTX-2.3.
pub struct VideoDit {
    patchify: Lin,
    // adaln_single
    ada_te1: Lin, // emb.timestep_embedder.linear_1 (256->dim)
    ada_te2: Lin, // emb.timestep_embedder.linear_2 (dim->dim)
    ada_lin: Lin, // linear (dim->9*dim)
    // prompt_adaln_single
    pada_te1: Lin,
    pada_te2: Lin,
    pada_lin: Lin, // (dim->2*dim)
    /// Резидентные блоки на `device` (offload=false). При offload пусто — блоки
    /// грузятся mmap→VRAM по требованию в forward через `stream`.
    blocks: Vec<Block>,
    sst_out: Tensor, // [2, dim] F32
    proj_out: Lin,
    heads: usize,
    dim_head: usize,
    dim: usize,
    theta: f64,
    max_pos: Vec<f64>,
    ts_mult: f64,
    device: Device, // compute-устройство (где идёт forward)
    dtype: DType,
    quant: DType,   // dtype квантования блочных linear'ов (=compute при offload)
    nblocks: usize,
    /// Pinned-кэш стримящихся весов при dense-offload (см. [`pin_ckpt_for_stream`]):
    /// первый своп копирует mmap→pinned, дальше H2D ~45GB/s без NVMe-перечиток.
    _host_pin: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    /// Pinned-зеркало host-stream квант-блоков (первый своп наполняет).
    _host_mirror: Option<synaptix_core::device::cuda::PinMirrorGuard>,
    /// `Some` → offload: блоки стримятся mmap→`device` поблочно в forward (dense
    /// bf16, best_cu float-acc → точно И влезает в 24GB), БЕЗ резидентной
    /// host-копии (`stream` — GPU-вью чекпойнта, общий mmap). `None` → блоки уже
    /// резидентны на `device` (либо на CPU при `host_stream`).
    stream: Option<Arc<LtxCheckpoint>>,
    /// Квант-offload: `blocks` КВАНТОВАННЫЕ на CPU, стримятся на `device`
    /// поблочно в forward (байты не меняются → bit-identical резидентному кванту).
    host_stream: bool,
}

/// Включить pinned-кэш offload-стрима для mmap-шардов ckpt (общее для
/// VideoDit/AvDit; no-op если CLI уже создал гард с prefill-форой). Host-RAM
/// цена ≈ размер ckpt (LTX 22B ~43GB).
fn pin_ckpt_for_stream(ckpt: &LtxCheckpoint) -> Option<synaptix_core::device::cuda::OffloadPinCacheGuard> {
    if synaptix_core::device::cuda::offload_pin_cache_active() {
        return None;
    }
    Some(synaptix_core::device::cuda::OffloadPinCacheGuard::new(&ckpt.shard_bytes()))
}

impl VideoDit {
    /// compute-dtype модели (касты ctx/латентов в pipeline).
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// `quant` — dtype квантования блочных linear'ов (MXFP8/NVFP4 для влезания в
    /// 24GB; compute → плотно). Мелкие patchify/adaln/proj_out всегда плотные.
    /// Изоляция adaLN: грузит только adaln-веса (без 19B блоков) и считает
    /// модуляцию video adaln + prompt adaln для `val_scaled` (=timestep×mult).
    #[doc(hidden)]
    pub fn _adaln_for_test(
        ckpt: &LtxCheckpoint, device: Device, dtype: DType, val_scaled: f32,
    ) -> Result<(Tensor, Tensor, Tensor), LtxError> {
        let te1 = Lin::load(ckpt, DIT_PREFIX, "adaln_single.emb.timestep_embedder.linear_1", true, dtype, dtype)?;
        let te2 = Lin::load(ckpt, DIT_PREFIX, "adaln_single.emb.timestep_embedder.linear_2", true, dtype, dtype)?;
        let lin = Lin::load(ckpt, DIT_PREFIX, "adaln_single.linear", true, dtype, dtype)?;
        let pte1 = Lin::load(ckpt, DIT_PREFIX, "prompt_adaln_single.emb.timestep_embedder.linear_1", true, dtype, dtype)?;
        let pte2 = Lin::load(ckpt, DIT_PREFIX, "prompt_adaln_single.emb.timestep_embedder.linear_2", true, dtype, dtype)?;
        let plin = Lin::load(ckpt, DIT_PREFIX, "prompt_adaln_single.linear", true, dtype, dtype)?;
        let (modul, emb) = adaln_mod(&[val_scaled], &te1, &te2, &lin, 1, device, dtype)?;
        let (pmod, _) = adaln_mod(&[val_scaled], &pte1, &pte2, &plin, 1, device, dtype)?;
        Ok((modul, emb, pmod))
    }

    #[doc(hidden)]
    pub fn _rope3d_for_test(
        positions: &[f64], t: usize, heads: usize, dim_head: usize, theta: f64, max_pos: &[f64], device: Device, dtype: DType,
    ) -> R<(Tensor, Tensor)> {
        rope3d(positions, 3, t, heads, dim_head, theta, max_pos, device, dtype)
    }

    pub fn load(ckpt: &LtxCheckpoint, device: Device, dtype: DType, quant: DType) -> Result<Self, LtxError> {
        Self::load_with(ckpt, device, dtype, quant, false)
    }

    /// `offload=true`: блоки стримятся на `device` поблочно в forward; мелкие веса
    /// резидентны на `device`. `quant==dtype` (dense) → стрим bf16 из mmap (ckpt на
    /// CPU). `quant!=dtype` → host-stream: блоки квантуются 1× на GPU при загрузке,
    /// живут на CPU и стримятся квантованными (байт меньше в 2-4×, bit-identical
    /// резидентному кванту; LoRA мерджится 1×, не на блок/шаг).
    pub fn load_with(
        ckpt: &LtxCheckpoint, device: Device, dtype: DType, quant: DType, offload: bool,
    ) -> Result<Self, LtxError> {
        let t = &ckpt.config.transformer;
        let host_stream = offload && quant != dtype;
        // Квант (резидентный или host-stream) требует CUDA-тензор (quantize_to_*) →
        // GPU-вью, чтобы веса материализовались на `device` ДО кванта. Dense-offload:
        // оставляем ckpt (CPU) — блоки стримятся из mmap, мелкие веса переносятся per-g.
        let gpu_view;
        let wckpt: &LtxCheckpoint = if offload && !host_stream {
            ckpt
        } else {
            gpu_view = ckpt.view_on(device);
            &gpu_view
        };
        // Мелкие dense-веса: грузим через wckpt, при offload переносим на compute.
        let g = |k: &str| -> Result<Lin, LtxError> {
            let l = Lin::load(wckpt, DIT_PREFIX, k, true, dtype, dtype)?;
            if offload { l.to_device(device).map_err(LtxError::from) } else { Ok(l) }
        };
        // runtime::set_dit_nblocks_cap — диагностический кап числа блоков (изоляция).
        let nblocks = crate::runtime::dit_nblocks_cap()
            .unwrap_or(t.num_layers)
            .min(t.num_layers);
        // dense-offload: НЕ материализуем блоки на host (24GB bf16!) — держим GPU-вью
        // чекпойнта и грузим каждый блок mmap→VRAM по требованию в forward.
        // host-stream: квантуем блоки 1× на GPU → CPU-резидент (квант-байты).
        // Иначе блоки резидентны на `device`.
        let host_pin = if offload && !host_stream { pin_ckpt_for_stream(ckpt) } else { None };
        let mut host_mirror: Option<synaptix_core::device::cuda::PinMirrorGuard> = None;
        let (blocks, stream) = if offload && !host_stream {
            (Vec::new(), Some(Arc::new(ckpt.view_on(device))))
        } else if host_stream {
            synaptix_core::device::cuda::set_offload_pinned(true);
            let mut blocks = Vec::with_capacity(nblocks);
            for i in 0..nblocks {
                let b = Block::load(wckpt, i, t.num_attention_heads, t.attention_head_dim, quant, dtype)?;
                blocks.push(b.to_device(Device::Cpu)?);
            }
            synaptix_core::device::cuda::set_offload_pinned(false);
            host_mirror = Some(synaptix_core::device::cuda::PinMirrorGuard::new());
            // прогрев зеркала: fetch→drop материализует pinned-копии блоков
            // (~1.5-2s в load) — иначе первый denoise-шаг платит ~3.7s wait.
            for b in &blocks {
                synaptix_core::device::cuda::set_pin_mirror(true);
                let warm = b.to_device(device);
                synaptix_core::device::cuda::set_pin_mirror(false);
                warm?;
            }
            (blocks, None)
        } else {
            let mut blocks = Vec::with_capacity(nblocks);
            for i in 0..nblocks {
                blocks.push(Block::load(wckpt, i, t.num_attention_heads, t.attention_head_dim, quant, dtype)?);
            }
            (blocks, None)
        };
        let sst_out = {
            let s = wckpt.get_raw(&format!("{DIT_PREFIX}.scale_shift_table"))?.to_dtype(DType::F32)?;
            if offload { s.to_device(device)? } else { s }
        };
        Ok(Self {
            patchify: g("patchify_proj")?,
            ada_te1: g("adaln_single.emb.timestep_embedder.linear_1")?,
            ada_te2: g("adaln_single.emb.timestep_embedder.linear_2")?,
            ada_lin: g("adaln_single.linear")?,
            pada_te1: g("prompt_adaln_single.emb.timestep_embedder.linear_1")?,
            pada_te2: g("prompt_adaln_single.emb.timestep_embedder.linear_2")?,
            pada_lin: g("prompt_adaln_single.linear")?,
            blocks,
            sst_out,
            proj_out: g("proj_out")?,
            heads: t.num_attention_heads,
            dim_head: t.attention_head_dim,
            dim: t.inner_dim(),
            theta: t.positional_embedding_theta,
            max_pos: t.positional_embedding_max_pos.iter().map(|&x| x as f64).collect(),
            ts_mult: t.timestep_scale_multiplier,
            device,
            dtype,
            quant,
            nblocks,
            _host_pin: host_pin,
            _host_mirror: host_mirror,
            stream,
            host_stream,
        })
    }

    fn adaln(&self, vals_scaled: &[f32], te1: &Lin, te2: &Lin, lin: &Lin, n: usize) -> R<(Tensor, Tensor)> {
        adaln_mod(vals_scaled, te1, te2, lin, n, self.device, self.dtype)
    }

    /// Один DiT-блок: self-attn (adaLN 0:3) → text cross-attn (6:9, prompt) → FF (3:6).
    /// `vx` обновляется на месте (residual). prof → пофазные таймеры (с sync).
    /// `skip_self_attn` (STG-пертурбация): пропустить вклад video self-attn1 в этом
    /// блоке — residual проходит без attn1 (как `SKIP_VIDEO_SELF_ATTN` в референсе).
    #[allow(clippy::too_many_arguments)]
    fn compute_block(
        &self,
        blk: &Block,
        vx: &mut Tensor,
        mods: &[Tensor],
        pmod: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        context: &Tensor,
        skip_self_attn: bool,
        dim: usize,
        prof: bool,
        ord: usize,
        t_a1: &mut f32,
        t_a2: &mut f32,
        t_ff: &mut f32,
    ) -> Result<(), LtxError> {
        let sync_t = |t: &mut f32, ti: std::time::Instant| {
            if prof {
                let _ = synaptix_core::device::cuda::synchronize(ord);
                *t += ti.elapsed().as_secs_f32();
            }
        };
        // self-attn: ada(0:3) = shift,scale,gate. STG: skip_self_attn → пропустить вклад.
        let tc = std::time::Instant::now();
        if !skip_self_attn {
            let m = ada(&blk.sst, mods, 0, 3, self.dtype)?;
            let (norm, pq) = mod_norm_quant(vx, &m[1], &m[0], blk.attn1.to_q.quant_dtype())?;
            let a = blk.attn1.forward_pq(
                &norm, None, Some(cos), Some(sin),
                pq.as_ref().map(|(p, s)| (p, s)),
            )?;
            *vx = vx.add(&a.broadcast_mul(&m[2])?)?;
        }
        sync_t(t_a1, tc);
        // text cross-attn (cross_attention_adaln): ada(6:9)=shift_q,scale_q,gate; prompt → shift_kv,scale_kv
        let tc2 = std::time::Instant::now();
        let mc = ada(&blk.sst, mods, 6, 9, self.dtype)?;
        let p_shift = blk.prompt_sst.narrow(0, 0, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?;
        let p_scale = blk.prompt_sst.narrow(0, 1, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?;
        let shift_kv = p_shift.broadcast_add(&pmod.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?)?;
        let scale_kv = p_scale.broadcast_add(&pmod.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?)?;
        let (attn_in, pq2) = mod_norm_quant(vx, &mc[1], &mc[0], blk.attn2.to_q.quant_dtype())?;
        let enc = context.broadcast_mul(&scale_kv.add_scalar(1.0)?)?.broadcast_add(&shift_kv)?;
        let c = blk.attn2.forward_pq(
            &attn_in, Some(&enc), None, None,
            pq2.as_ref().map(|(p, s)| (p, s)),
        )?;
        *vx = vx.add(&c.broadcast_mul(&mc[2])?)?;
        sync_t(t_a2, tc2);
        // FF: ada(3:6)=shift,scale,gate. Чанк по токенам при длинных T (см.
        // AvBlock: ff_mid [T,4·dim] на 53k токенов = 1.76GB×копии, stage2-refine
        // 20s HD не влезал). Модуляции broadcast'ные → чанк бит-в-бит.
        let tc3 = std::time::Instant::now();
        let mf = ada(&blk.sst, mods, 3, 6, self.dtype)?;
        // модуляции PER-TOKEN [1,T,dim] (split_modul) → чанк режет и их; gate
        // применяется по-чанково до cat. Бит-в-бит: построчные op'ы.
        let ff_one = |vc: &Tensor, sh: &Tensor, sc: &Tensor, gate: &Tensor| -> R<Tensor> {
            let (fn_in, pqf) = mod_norm_quant(vc, sc, sh, blk.ff0.quant_dtype())?;
            let ff_mid = match &pqf {
                Some((p, s)) => {
                    let dims = fn_in.dims();
                    let (b, t) = (dims[0], dims[1]);
                    let y = blk.ff0.fwd_prequant(p, s, b * t, fn_in.dtype())?;
                    y.reshape(vec![b, t, y.dims()[1]])?
                }
                None => blk.ff0.fwd(&fn_in)?,
            };
            blk.ff2.fwd(&ff_mid.gelu_tanh()?)?.broadcast_mul(gate)
        };
        let t_len = vx.dims()[1];
        let ff_chunk: usize = 16384;
        if ff_chunk > 0 && t_len > ff_chunk {
            let mut parts: Vec<Tensor> = Vec::with_capacity(t_len.div_ceil(ff_chunk));
            let mut o = 0usize;
            while o < t_len {
                let n = ff_chunk.min(t_len - o);
                let nar = |t: &Tensor| -> R<Tensor> { t.narrow(1, o, n)?.contiguous() };
                parts.push(ff_one(&nar(vx)?, &nar(&mf[0])?, &nar(&mf[1])?, &nar(&mf[2])?)?);
                o += n;
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            *vx = vx.add(&Tensor::cat(&refs, 1)?)?;
        } else {
            *vx = vx.add(&ff_one(vx, &mf[0], &mf[1], &mf[2])?)?;
        }
        sync_t(t_ff, tc3);
        Ok(())
    }

    /// Forward → velocity `[1,T,128]`. Входы (как в Modality):
    /// - `latent` `[1,T,128]`
    /// - `timesteps` `[T]` (per-token), `sigma` `f32`
    /// - `positions` host f64 `[3*T*2]` (pixel coords [start,end])
    /// - `context` `[1,T_txt,4096]` (video_encoding)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: &Tensor,
        timesteps: &[f32],
        sigma: f32,
        positions: &[f64],
        context: &Tensor,
    ) -> Result<Tensor, LtxError> {
        self.forward_perturbed(latent, timesteps, sigma, positions, context, &[])
    }

    /// Как [`forward`], но STG-пертурбация: в блоках `stg_blocks` пропускается
    /// вклад video self-attn1 (uncond_perturbed-проход multimodal guidance).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_perturbed(
        &self,
        latent: &Tensor,
        timesteps: &[f32],
        sigma: f32,
        positions: &[f64],
        context: &Tensor,
        stg_blocks: &[usize],
    ) -> Result<Tensor, LtxError> {
        let t = latent.dims()[1];
        let dim = self.dim;
        let mut vx = self.patchify.fwd(latent)?; // [1,T,dim]

        // adaLN-single: per-token timesteps*mult → [1,T,9,dim] + embedded [1,T,dim]
        let ts_scaled: Vec<f32> = timesteps.iter().map(|&x| x * self.ts_mult as f32).collect();
        let (modul, embedded) = self.adaln(&ts_scaled, &self.ada_te1, &self.ada_te2, &self.ada_lin, t)?;
        let modul = modul.reshape(vec![1, t, 9, dim])?; // [1,T,9,dim]
        let mods = split_modul(&modul)?; // 9× contiguous [1,T,dim] (раз на шаг)
        let embedded = embedded.reshape(vec![1, t, dim])?;
        // prompt_adaln: sigma*mult → [1,1,2,dim]
        let (pmod, _) = self.adaln(&[sigma * self.ts_mult as f32], &self.pada_te1, &self.pada_te2, &self.pada_lin, 1)?;
        let pmod = pmod.reshape(vec![1, 1, 2, dim])?;

        let (cos, sin) = rope3d(
            positions, 3, t, self.heads, self.dim_head, self.theta, &self.max_pos, self.device, self.dtype,
        )?;

        let prof = crate::runtime::ltx_prof();
        let ord = if let Device::Cuda(o) = self.device { o } else { 0 };
        let (mut t_stream, mut t_a1, mut t_a2, mut t_ff) = (0f32, 0f32, 0f32, 0f32);

        if self.stream.is_some() || self.host_stream {
            // offload: блоки стримятся в VRAM (mmap bf16 либо host-RAM квант — см.
            // `fetch`). ПРЕФЕТЧ (двойная буферизация): блок i+1 грузится на фоновом
            // host-потоке во время compute блока i → H2D перекрывается с GPU-compute
            // (те же веса/математика → bit-identical).
            let (heads, dim_head, quant, dtype) = (self.heads, self.dim_head, self.quant, self.dtype);
            let stream = self.stream.as_deref();
            let fetch = |i: usize| -> Result<Block, LtxError> {
                match stream {
                    Some(sc) => Block::load(sc, i, heads, dim_head, quant, dtype),
                    None => {
                        // host-stream: H2D через pinned-зеркало (ptr блок-Vec'ов
                        // стабильны) — без staging-копии на каждый своп.
                        synaptix_core::device::cuda::set_pin_mirror(true);
                        let r = self.blocks[i].to_device(self.device).map_err(LtxError::from);
                        synaptix_core::device::cuda::set_pin_mirror(false);
                        r
                    }
                }
            };
            let fetch = &fetch;
            // offload-загрузка весов через pinned staging (H2D 45 vs 3.6 GB/s pageable).
            synaptix_core::device::cuda::set_offload_pinned(true);
            let ls = synaptix_core::device::cuda::loader_stream(ord).map_err(LtxError::from)?;
            let mut cur = fetch(0)?;
            for i in 0..self.nblocks {
                let ti = std::time::Instant::now();
                let lsc = ls.clone();
                let next: Option<Block> = std::thread::scope(|s| -> Result<Option<Block>, LtxError> {
                    let h = if i + 1 < self.nblocks {
                        Some(s.spawn(move || -> Result<Block, LtxError> {
                            // H2D загрузки → loader-stream + pinned staging (overlap + 45 GB/s)
                            synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                            synaptix_core::device::cuda::set_offload_pinned(true);
                            let r = fetch(i + 1);
                            let _ = lsc.synchronize();
                            synaptix_core::device::cuda::set_offload_pinned(false);
                            synaptix_core::device::cuda::set_alloc_stream(None);
                            r
                        }))
                    } else {
                        None
                    };
                    self.compute_block(&cur, &mut vx, &mods, &pmod, &cos, &sin, context, stg_blocks.contains(&i), dim, prof, ord, &mut t_a1, &mut t_a2, &mut t_ff)?;
                    match h {
                        Some(h) => Ok(Some(h.join().expect("ltx prefetch thread panicked")?)),
                        None => Ok(None),
                    }
                })?;
                if prof { let _ = synaptix_core::device::cuda::synchronize(ord); t_stream += ti.elapsed().as_secs_f32(); }
                if let Some(nb) = next {
                    cur = nb;
                }
            }
        } else {
            for i in 0..self.nblocks {
                let blk = &self.blocks[i];
                self.compute_block(blk, &mut vx, &mods, &pmod, &cos, &sin, context, stg_blocks.contains(&i), dim, prof, ord, &mut t_a1, &mut t_a2, &mut t_ff)?;
            }
        }
        let t_compute = t_a1 + t_a2 + t_ff;
        if prof { eprintln!("[LTX_PROF]   stream={t_stream:.2}s compute={t_compute:.2}s (attn1={t_a1:.2} attn2={t_a2:.2} ff={t_ff:.2}) ({} блоков)", self.nblocks); }

        // head: scale_shift_table[None,None] + embedded[:,:,None]
        let shift = self.sst_out.narrow(0, 0, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?
            .broadcast_add(&embedded)?;
        let scale = self.sst_out.narrow(0, 1, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?
            .broadcast_add(&embedded)?;
        let x = layer_norm_no_affine(&vx)?;
        let x = x.broadcast_mul(&scale.add_scalar(1.0)?)?.broadcast_add(&shift)?;
        Ok(self.proj_out.fwd(&x)?)
    }

    /// Multi-stream forward: N потоков (cond/uncond_text/uncond_perturbed для
    /// guidance) через ОДИН стрим-свип блоков — блок грузится из mmap один раз и
    /// применяется ко всем потокам. Для offload это режет H2D-стриминг 46GB ×N
    /// (узкое место guided-генерации: GPU простаивал на загрузке). `perturb[s]` —
    /// STG-пертурбация потока s (skip self-attn в `stg_blocks`). Возвращает N
    /// velocity `[1,T,128]`. Все потоки делят `positions`/rope (один input-латент).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_multi(
        &self,
        latents: &[&Tensor],
        timesteps: &[Vec<f32>],
        sigmas: &[f32],
        positions: &[f64],
        contexts: &[&Tensor],
        perturb: &[bool],
        stg_blocks: &[usize],
    ) -> Result<Vec<Tensor>, LtxError> {
        let n = latents.len();
        let t = latents[0].dims()[1];
        let dim = self.dim;
        // per-stream pre-compute: vx, modul, pmod, embedded
        let mut vxs = Vec::with_capacity(n);
        let mut moduls = Vec::with_capacity(n);
        let mut pmods = Vec::with_capacity(n);
        let mut embs = Vec::with_capacity(n);
        for s in 0..n {
            let vx = self.patchify.fwd(latents[s])?;
            let ts_scaled: Vec<f32> = timesteps[s].iter().map(|&x| x * self.ts_mult as f32).collect();
            let (modul, embedded) = self.adaln(&ts_scaled, &self.ada_te1, &self.ada_te2, &self.ada_lin, t)?;
            moduls.push(split_modul(&modul.reshape(vec![1, t, 9, dim])?)?);
            embs.push(embedded.reshape(vec![1, t, dim])?);
            let (pmod, _) = self.adaln(&[sigmas[s] * self.ts_mult as f32], &self.pada_te1, &self.pada_te2, &self.pada_lin, 1)?;
            pmods.push(pmod.reshape(vec![1, 1, 2, dim])?);
            vxs.push(vx);
        }
        let (cos, sin) = rope3d(
            positions, 3, t, self.heads, self.dim_head, self.theta, &self.max_pos, self.device, self.dtype,
        )?;
        let prof = crate::runtime::ltx_prof();
        let ord = if let Device::Cuda(o) = self.device { o } else { 0 };
        let (mut t_a1, mut t_a2, mut t_ff) = (0f32, 0f32, 0f32);
        // streaming sweep: блок i грузится один раз → compute по всем потокам;
        // префетч блока i+1 на фоновом потоке перекрывает H2D с N×compute.
        if self.stream.is_some() || self.host_stream {
            let (heads, dim_head, quant, dtype) = (self.heads, self.dim_head, self.quant, self.dtype);
            let stream = self.stream.as_deref();
            let fetch = |i: usize| -> Result<Block, LtxError> {
                match stream {
                    Some(sc) => Block::load(sc, i, heads, dim_head, quant, dtype),
                    None => {
                        // host-stream: H2D через pinned-зеркало (ptr блок-Vec'ов
                        // стабильны) — без staging-копии на каждый своп.
                        synaptix_core::device::cuda::set_pin_mirror(true);
                        let r = self.blocks[i].to_device(self.device).map_err(LtxError::from);
                        synaptix_core::device::cuda::set_pin_mirror(false);
                        r
                    }
                }
            };
            let fetch = &fetch;
            synaptix_core::device::cuda::set_offload_pinned(true);
            let ls = synaptix_core::device::cuda::loader_stream(ord).map_err(LtxError::from)?;
            let mut cur = fetch(0)?;
            for i in 0..self.nblocks {
                let lsc = ls.clone();
                let next: Option<Block> = std::thread::scope(|sp| -> Result<Option<Block>, LtxError> {
                    let h = if i + 1 < self.nblocks {
                        Some(sp.spawn(move || -> Result<Block, LtxError> {
                            synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                            synaptix_core::device::cuda::set_offload_pinned(true);
                            let r = fetch(i + 1);
                            let _ = lsc.synchronize();
                            synaptix_core::device::cuda::set_offload_pinned(false);
                            synaptix_core::device::cuda::set_alloc_stream(None);
                            r
                        }))
                    } else {
                        None
                    };
                    for s in 0..n {
                        let skip = perturb[s] && stg_blocks.contains(&i);
                        self.compute_block(&cur, &mut vxs[s], &moduls[s], &pmods[s], &cos, &sin, contexts[s], skip, dim, prof, ord, &mut t_a1, &mut t_a2, &mut t_ff)?;
                    }
                    match h {
                        Some(h) => Ok(Some(h.join().expect("ltx prefetch thread panicked")?)),
                        None => Ok(None),
                    }
                })?;
                if let Some(nb) = next {
                    cur = nb;
                }
            }
        } else {
            for i in 0..self.nblocks {
                let blk = &self.blocks[i];
                for s in 0..n {
                    let skip = perturb[s] && stg_blocks.contains(&i);
                    self.compute_block(blk, &mut vxs[s], &moduls[s], &pmods[s], &cos, &sin, contexts[s], skip, dim, prof, ord, &mut t_a1, &mut t_a2, &mut t_ff)?;
                }
            }
        }
        // per-stream head
        let mut outs = Vec::with_capacity(n);
        for s in 0..n {
            let shift = self.sst_out.narrow(0, 0, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?
                .broadcast_add(&embs[s])?;
            let scale = self.sst_out.narrow(0, 1, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?
                .broadcast_add(&embs[s])?;
            let x = layer_norm_no_affine(&vxs[s])?;
            let x = x.broadcast_mul(&scale.add_scalar(1.0)?)?.broadcast_add(&shift)?;
            outs.push(self.proj_out.fwd(&x)?);
        }
        Ok(outs)
    }
}

// ───────────────────────── Фаза 8: joint A/V DiT ─────────────────────────

/// av_ca-модуляция (cross-modal): таблица `[5,dim]`, scale_shift-modul
/// (reshaped `[1,1,4,dim]`), gate-modul (`[1,1,1,dim]`), пара строк `lo,lo+1`.
/// → (scale=row[lo], shift=row[lo+1], gate=row[4]), каждый `[1,1,dim]` + modul.
fn av_ca_ada(table: &Tensor, ss_modul: &Tensor, gate_modul: &Tensor, lo: usize, dtype: DType) -> R<(Tensor, Tensor, Tensor)> {
    let dim = table.dims()[1];
    let row = |r: usize| -> R<Tensor> { table.narrow(0, r, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(dtype) };
    let scale = row(lo)?.broadcast_add(&ss_modul.narrow(2, lo, 1)?.squeeze(2)?.contiguous()?)?;
    let shift = row(lo + 1)?.broadcast_add(&ss_modul.narrow(2, lo + 1, 1)?.squeeze(2)?.contiguous()?)?;
    let gate = row(4)?.broadcast_add(&gate_modul.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?)?;
    Ok((scale, shift, gate))
}

/// Один блок joint A/V `BasicAVTransformerBlock`: video- и audio-потоки +
/// двунаправленный A2V/V2A cross-attn.
struct AvBlock {
    // video
    v_sst: Tensor,
    v_prompt_sst: Tensor,
    v_attn1: Attn,
    v_attn2: Attn,
    v_ff0: Lin,
    v_ff2: Lin,
    // audio
    a_sst: Tensor,
    a_prompt_sst: Tensor,
    a_attn1: Attn,
    a_attn2: Attn,
    a_ff0: Lin,
    a_ff2: Lin,
    // cross-modal (оба в audio-конфиге голов: 32×64=2048)
    a2v_attn: Attn, // Q=video, KV=audio
    v2a_attn: Attn, // Q=audio, KV=video
    ca_audio_table: Tensor, // [5,2048]
    ca_video_table: Tensor, // [5,4096]
    /// Слот-стриминг: готовые [1,1,dim]-строки sst-таблиц в compute-dtype —
    /// убирают tiny-обвязку (narrow/contiguous/cast) из ada-путей. `None` у
    /// легаси-блоков (резидент / host-stream).
    sst_dt: Option<SstDt>,
}

struct SstDt {
    v_rows: Vec<Tensor>,
    vp_rows: Vec<Tensor>,
    a_rows: Vec<Tensor>,
    ap_rows: Vec<Tensor>,
    ca_a_rows: Vec<Tensor>,
    ca_v_rows: Vec<Tensor>,
}

/// ada() на готовых строках: один broadcast_add на ряд (без tiny-обвязки).
fn ada_rows(rows: &[Tensor], mods: &[Tensor], lo: usize, hi: usize) -> R<Vec<Tensor>> {
    (lo..hi).map(|r| mods[r].broadcast_add(&rows[r])).collect()
}

fn av_ca_ada_rows(
    rows: &[Tensor], ss_modul: &Tensor, gate_modul: &Tensor, lo: usize,
) -> R<(Tensor, Tensor, Tensor)> {
    let scale = rows[lo].broadcast_add(&ss_modul.narrow(2, lo, 1)?.squeeze(2)?.contiguous()?)?;
    let shift = rows[lo + 1].broadcast_add(&ss_modul.narrow(2, lo + 1, 1)?.squeeze(2)?.contiguous()?)?;
    let gate = rows[4].broadcast_add(&gate_modul.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?)?;
    Ok((scale, shift, gate))
}

/// Общий контекст forward'а блока (module-level, считается один раз).
struct AvCtx<'a> {
    v_mods: &'a [Tensor],  // 9× [1,Tv,4096] (вырезы modul, contiguous)
    a_mods: &'a [Tensor],  // 9× [1,Ta,2048]
    v_pmod: &'a Tensor,    // [1,1,2,4096]
    a_pmod: &'a Tensor,    // [1,1,2,2048]
    v_cross_ss: &'a Tensor, // [1,1,4,4096]
    v_cross_gate: &'a Tensor, // [1,1,1,4096]
    a_cross_ss: &'a Tensor, // [1,1,4,2048]
    a_cross_gate: &'a Tensor, // [1,1,1,2048]
    v_ctx: &'a Tensor,     // [1,Ttv,4096]
    a_ctx: &'a Tensor,     // [1,Tta,2048]
    v_rope: (&'a Tensor, &'a Tensor),     // [1,32,Tv,64]
    a_rope: (&'a Tensor, &'a Tensor),     // [1,32,Ta,32]
    v_cross_pe: (&'a Tensor, &'a Tensor), // [1,32,Tv,32]
    a_cross_pe: (&'a Tensor, &'a Tensor), // [1,32,Ta,32]
    dtype: DType,
    /// NAG (Normalized Attention Guidance) на видео-text-cross (v_attn2):
    /// (neg-контекст [1,Ttv,4096], scale, alpha, tau). Только stage1 —
    /// distilled stage2-refine NAG не требует (рецепт убирания субтитров).
    v_nag: Option<(&'a Tensor, f32, f32, f32)>,
}

impl AvBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(ckpt: &LtxCheckpoint, idx: usize, qdt: DType, compute: DType) -> Result<Self, LtxError> {
        let p = format!("{DIT_PREFIX}.transformer_blocks.{idx}");
        let raw = |k: &str| -> Result<Tensor, LtxError> { Ok(ckpt.get_raw(&format!("{p}.{k}"))?.to_dtype(DType::F32)?) };
        Ok(Self {
            v_sst: raw("scale_shift_table")?,
            v_prompt_sst: raw("prompt_scale_shift_table")?,
            v_attn1: Attn::load(ckpt, &format!("{p}.attn1"), 32, 128, qdt, compute)?,
            v_attn2: Attn::load(ckpt, &format!("{p}.attn2"), 32, 128, qdt, compute)?,
            v_ff0: Lin::load(ckpt, &p, "ff.net.0.proj", true, qdt, compute)?,
            v_ff2: Lin::load(ckpt, &p, "ff.net.2", true, qdt, compute)?,
            a_sst: raw("audio_scale_shift_table")?,
            a_prompt_sst: raw("audio_prompt_scale_shift_table")?,
            a_attn1: Attn::load(ckpt, &format!("{p}.audio_attn1"), 32, 64, qdt, compute)?,
            a_attn2: Attn::load(ckpt, &format!("{p}.audio_attn2"), 32, 64, qdt, compute)?,
            a_ff0: Lin::load(ckpt, &p, "audio_ff.net.0.proj", true, qdt, compute)?,
            a_ff2: Lin::load(ckpt, &p, "audio_ff.net.2", true, qdt, compute)?,
            a2v_attn: Attn::load(ckpt, &format!("{p}.audio_to_video_attn"), 32, 64, qdt, compute)?,
            v2a_attn: Attn::load(ckpt, &format!("{p}.video_to_audio_attn"), 32, 64, qdt, compute)?,
            ca_audio_table: raw("scale_shift_table_a2v_ca_audio")?,
            ca_video_table: raw("scale_shift_table_a2v_ca_video")?,
            sst_dt: None,
        })
    }

    /// Перенос блока между устройствами (host-stream квант-блоков, см. [`Block::to_device`]).
    fn to_device(&self, dev: Device) -> R<Self> {
        Ok(Self {
            v_sst: self.v_sst.to_device(dev)?,
            v_prompt_sst: self.v_prompt_sst.to_device(dev)?,
            v_attn1: self.v_attn1.to_device(dev)?,
            v_attn2: self.v_attn2.to_device(dev)?,
            v_ff0: self.v_ff0.to_device(dev)?,
            v_ff2: self.v_ff2.to_device(dev)?,
            a_sst: self.a_sst.to_device(dev)?,
            a_prompt_sst: self.a_prompt_sst.to_device(dev)?,
            a_attn1: self.a_attn1.to_device(dev)?,
            a_attn2: self.a_attn2.to_device(dev)?,
            a_ff0: self.a_ff0.to_device(dev)?,
            a_ff2: self.a_ff2.to_device(dev)?,
            a2v_attn: self.a2v_attn.to_device(dev)?,
            v2a_attn: self.v2a_attn.to_device(dev)?,
            ca_audio_table: self.ca_audio_table.to_device(dev)?,
            ca_video_table: self.ca_video_table.to_device(dev)?,
            sst_dt: None,
        })
    }

    /// `prompt_kv(table[2,dim], pmod[1,1,2,dim]) → (shift_kv, scale_kv)` для
    /// cross_attention_adaln (модуляция текстовых K/V из sigma).
    fn prompt_kv_rows(rows: &[Tensor], pmod: &Tensor) -> R<(Tensor, Tensor)> {
        let shift_kv = rows[0].broadcast_add(&pmod.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?)?;
        let scale_kv = rows[1].broadcast_add(&pmod.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?)?;
        Ok((shift_kv, scale_kv))
    }

    fn prompt_kv(table: &Tensor, pmod: &Tensor, dim: usize, dtype: DType) -> R<(Tensor, Tensor)> {
        let p_shift = table.narrow(0, 0, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(dtype)?;
        let p_scale = table.narrow(0, 1, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(dtype)?;
        let shift_kv = p_shift.broadcast_add(&pmod.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?)?;
        let scale_kv = p_scale.broadcast_add(&pmod.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?)?;
        Ok((shift_kv, scale_kv))
    }

    /// `tm`: PROF-аккумулятор фаз `[v_attn1, v_attn2, audio, av_ca, ff]` (sync на
    /// границах фаз; только под runtime::set_ltx_prof — см. AvDit::forward).
    fn forward(&self, vx: &Tensor, ax: &Tensor, c: &AvCtx, mut tm: Option<&mut [f32; 5]>) -> R<(Tensor, Tensor)> {
        let ord = if let Device::Cuda(o) = vx.device() { o } else { 0 };
        let mut tphase = std::time::Instant::now();
        let mut lap = |slot: usize, tm: &mut Option<&mut [f32; 5]>| {
            if let Some(t) = tm {
                let _ = synaptix_core::device::cuda::synchronize(ord);
                t[slot] += tphase.elapsed().as_secs_f32();
                tphase = std::time::Instant::now();
            }
        };
        let dt = c.dtype;
        let mut vx = vx.clone();
        let mut ax = ax.clone();
        // суб-фазы блока (sync-метки) — охота на рой.
        let bprof = crate::runtime::ltx_blk_prof();
        let mut bm: Vec<(&'static str, f64)> = Vec::new();
        let mut bt = std::time::Instant::now();
        let mut bmark = |name: &'static str, bt: &mut std::time::Instant, bm: &mut Vec<(&'static str, f64)>| {
            if bprof {
                let _ = synaptix_core::device::cuda::synchronize(ord);
                bm.push((name, bt.elapsed().as_secs_f64()));
                *bt = std::time::Instant::now();
            }
        };

        // ── video: self-attn + text-cross ──
        // Fused «модуляция+квант» (рецепт 4496dfd1, как в видео-Block): эпилог
        // нормы выдаёт prequant-пару, q/k/v шарят её (3 кванта + decomposed
        // цепочка → 1 launch); бит-в-бит с rms_no_gain→mul→add→quantize.
        let mut m = match &self.sst_dt {
            Some(d) => ada_rows(&d.v_rows, c.v_mods, 0, 3)?,
            None => ada(&self.v_sst, c.v_mods, 0, 3, dt)?,
        };
        bmark("ada1", &mut bt, &mut bm);
        let (norm, pq) = mod_norm_quant(&vx, &m[1], &m[0], self.v_attn1.to_q.quant_dtype())?;
        bmark("nq1", &mut bt, &mut bm);
        // shift/scale (2×[1,T,dim], 880MB на 20s) потреблены нормой — на время
        // attention держим только gate (20s-refine VRAM впритык).
        let m2 = m.pop().ok_or(SynaptixError::Unsupported("ada m2"))?;
        m.clear();
        let attn1_out = self
            .v_attn1
            .forward_pq(&norm, None, Some(c.v_rope.0), Some(c.v_rope.1), pq.as_ref().map(|(p, s)| (p, s)))?;
        vx = gate_residual(&vx, &attn1_out, &m2)?;
        drop(m2);
        bmark("attn1", &mut bt, &mut bm);
        if bprof && !bm.is_empty() {
            let s: Vec<String> = bm.iter().map(|(n, t)| format!("{n}={:.1}ms", t * 1e3)).collect();
            eprintln!("[BLK_PROF] {}", s.join(" "));
        }
        lap(0, &mut tm);
        let mut mc = match &self.sst_dt {
            Some(d) => ada_rows(&d.v_rows, c.v_mods, 6, 9)?,
            None => ada(&self.v_sst, c.v_mods, 6, 9, dt)?,
        };
        let (shift_kv, scale_kv) = match &self.sst_dt {
            Some(d) => Self::prompt_kv_rows(&d.vp_rows, c.v_pmod)?,
            None => Self::prompt_kv(&self.v_prompt_sst, c.v_pmod, 4096, dt)?,
        };
        let (attn_in, pq2) = mod_norm_quant(&vx, &mc[1], &mc[0], self.v_attn2.to_q.quant_dtype())?;
        let mc2 = mc.pop().ok_or(SynaptixError::Unsupported("ada mc2"))?;
        mc.clear(); // shift/scale потреблены — см. m выше
        let enc = mod_row(c.v_ctx, &scale_kv, &shift_kv)?;
        let x_pos = self
            .v_attn2
            .forward_pq(&attn_in, Some(&enc), None, None, pq2.as_ref().map(|(p, s)| (p, s)))?;
        // NAG: x_neg тем же модулем на neg-контексте; экстраполяция
        // pos·scale − neg·(scale−1), L1-кламп по tau (per-token), бленд alpha.
        let attn2_out = match c.v_nag {
            Some((nag_ctx, scale, alpha, tau)) => {
                let enc_n = mod_row(nag_ctx, &scale_kv, &shift_kv)?;
                let x_neg = self.v_attn2.forward_pq(
                    &attn_in, Some(&enc_n), None, None,
                    pq2.as_ref().map(|(p, s)| (p, s)),
                )?;
                let g = x_pos
                    .mul_scalar(scale)?
                    .sub(&x_neg.mul_scalar(scale - 1.0)?)?;
                let last = g.rank() - 1;
                let l1_pos = x_pos.abs()?.sum_keepdim(last)?;
                let l1_g = g.abs()?.sum_keepdim(last)?;
                let ratio = l1_g.div(&l1_pos.add_scalar(1e-6)?)?;
                let factor = ratio.clamp(0.0, tau)?.div(&ratio.add_scalar(1e-6)?)?;
                let g = g.broadcast_mul(&factor)?;
                g.mul_scalar(alpha)?.add(&x_pos.mul_scalar(1.0 - alpha)?)?
            }
            None => x_pos,
        };
        vx = gate_residual(&vx, &attn2_out, &mc2)?;
        drop(mc2);
        lap(1, &mut tm);

        // ── audio: self-attn + text-cross ── (gate-only удержание, как video)
        let mut am = match &self.sst_dt {
            Some(d) => ada_rows(&d.a_rows, c.a_mods, 0, 3)?,
            None => ada(&self.a_sst, c.a_mods, 0, 3, dt)?,
        };
        let (anorm, _) = mod_norm_quant(&ax, &am[1], &am[0], None)?;
        let am2 = am.pop().ok_or(SynaptixError::Unsupported("ada am2"))?;
        am.clear();
        let a_attn1_out = self.a_attn1.forward(&anorm, None, Some(c.a_rope.0), Some(c.a_rope.1))?;
        ax = gate_residual(&ax, &a_attn1_out, &am2)?;
        drop(am2);
        let mut amc = match &self.sst_dt {
            Some(d) => ada_rows(&d.a_rows, c.a_mods, 6, 9)?,
            None => ada(&self.a_sst, c.a_mods, 6, 9, dt)?,
        };
        let (a_shift_kv, a_scale_kv) = match &self.sst_dt {
            Some(d) => Self::prompt_kv_rows(&d.ap_rows, c.a_pmod)?,
            None => Self::prompt_kv(&self.a_prompt_sst, c.a_pmod, 2048, dt)?,
        };
        let (a_attn_in, _) = mod_norm_quant(&ax, &amc[1], &amc[0], None)?;
        let amc2 = amc.pop().ok_or(SynaptixError::Unsupported("ada amc2"))?;
        amc.clear();
        let a_enc = mod_row(c.a_ctx, &a_scale_kv, &a_shift_kv)?;
        let a_attn2_out = self.a_attn2.forward(&a_attn_in, Some(&a_enc), None, None)?;
        ax = gate_residual(&ax, &a_attn2_out, &amc2)?;
        drop(amc2);
        lap(2, &mut tm);

        // ── A/V cross-attn (нормы обоих потоков считаются ОДИН раз, до апдейтов) ──
        let vx_norm3 = rms_no_gain(&vx)?;
        let ax_norm3 = rms_no_gain(&ax)?;
        // A2V (Q=video, KV=audio): модифицирует vx
        let (sc_v, sh_v, gate_a2v) = match &self.sst_dt {
            Some(d) => av_ca_ada_rows(&d.ca_v_rows, c.v_cross_ss, c.v_cross_gate, 0)?,
            None => av_ca_ada(&self.ca_video_table, c.v_cross_ss, c.v_cross_gate, 0, dt)?,
        };
        let (sc_a, sh_a, _) = match &self.sst_dt {
            Some(d) => av_ca_ada_rows(&d.ca_a_rows, c.a_cross_ss, c.a_cross_gate, 0)?,
            None => av_ca_ada(&self.ca_audio_table, c.a_cross_ss, c.a_cross_gate, 0, dt)?,
        };
        let vx_s = mod_row(&vx_norm3, &sc_v, &sh_v)?;
        let ax_s = mod_row(&ax_norm3, &sc_a, &sh_a)?;
        let a2v = self.a2v_attn.forward2(&vx_s, Some(&ax_s), Some(c.v_cross_pe), Some(c.a_cross_pe), None)?;
        vx = gate_residual(&vx, &a2v, &gate_a2v)?;
        // V2A (Q=audio, KV=video): модифицирует ax (использует vx_norm3/ax_norm3 «до A2V»)
        let (sc_a2, sh_a2, gate_v2a) = match &self.sst_dt {
            Some(d) => av_ca_ada_rows(&d.ca_a_rows, c.a_cross_ss, c.a_cross_gate, 2)?,
            None => av_ca_ada(&self.ca_audio_table, c.a_cross_ss, c.a_cross_gate, 2, dt)?,
        };
        let (sc_v2, sh_v2, _) = match &self.sst_dt {
            Some(d) => av_ca_ada_rows(&d.ca_v_rows, c.v_cross_ss, c.v_cross_gate, 2)?,
            None => av_ca_ada(&self.ca_video_table, c.v_cross_ss, c.v_cross_gate, 2, dt)?,
        };
        let ax_s2 = mod_row(&ax_norm3, &sc_a2, &sh_a2)?;
        let vx_s2 = mod_row(&vx_norm3, &sc_v2, &sh_v2)?;
        let v2a = self.v2a_attn.forward2(&ax_s2, Some(&vx_s2), Some(c.a_cross_pe), Some(c.v_cross_pe), None)?;
        ax = gate_residual(&ax, &v2a, &gate_v2a)?;
        lap(3, &mut tm);

        // ── FF (video, audio) ──
        // Видео-FF чанкуется по токенам при длинных T: ff_mid [T,4·dim] и его
        // gelu-копия на T=53k (20s HD stage2) — по 1.76GB транзиентов, весь
        // хвост FF не влезает в 24GB. Модуляции broadcast'ные → чанк бит-в-бит.
        let t_len = vx.dims()[1];
        let ff_chunk: usize = 16384;
        // ada по-чанково: mods[3..6]-чанк + sst-строка (бит-в-бит с полным ada —
        // поэлементный broadcast_add); полные mf 3×[1,T,dim] (1.3GB на 20s) не
        // материализуются. Gate применяется по-чанково до cat.
        let tb = |r: usize| -> R<Tensor> {
            self.v_sst.narrow(0, r, 1)?.contiguous()?.reshape(vec![1, 1, 4096])?.to_dtype(dt)
        };
        let (tb_sh, tb_sc, tb_g) = match &self.sst_dt {
            Some(d) => (d.v_rows[3].clone(), d.v_rows[4].clone(), d.v_rows[5].clone()),
            None => (tb(3)?, tb(4)?, tb(5)?),
        };
        let ff_one = |vc: &Tensor, sh_m: &Tensor, sc_m: &Tensor, gate_m: &Tensor| -> R<(Tensor, Tensor)> {
            let sh = sh_m.broadcast_add(&tb_sh)?;
            let sc = sc_m.broadcast_add(&tb_sc)?;
            let (fin, pqf) = mod_norm_quant(vc, &sc, &sh, self.v_ff0.quant_dtype())?;
            drop(sh);
            drop(sc);
            let ff_mid = match &pqf {
                Some((p, s)) => {
                    let dims = fin.dims();
                    let (fb, ft) = (dims[0], dims[1]);
                    let y = self.v_ff0.fwd_prequant(p, s, fb * ft, fin.dtype())?;
                    y.reshape(vec![fb, ft, y.dims()[1]])?
                }
                None => self.v_ff0.fwd(&fin)?,
            };
            let gate = gate_m.broadcast_add(&tb_g)?;
            Ok((self.v_ff2.fwd(&ff_mid.gelu_tanh()?)?, gate))
        };
        if ff_chunk > 0 && t_len > ff_chunk {
            let mut parts: Vec<Tensor> = Vec::with_capacity(t_len.div_ceil(ff_chunk));
            let mut o = 0usize;
            while o < t_len {
                let n = ff_chunk.min(t_len - o);
                let nar = |t: &Tensor| -> R<Tensor> { t.narrow(1, o, n)?.contiguous() };
                let (y, gate) = ff_one(
                    &nar(&vx)?, &nar(&c.v_mods[3])?, &nar(&c.v_mods[4])?, &nar(&c.v_mods[5])?,
                )?;
                parts.push(y.broadcast_mul(&gate)?);
                o += n;
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            vx = vx.add(&Tensor::cat(&refs, 1)?)?;
        } else {
            let (y, gate) = ff_one(&vx, &c.v_mods[3], &c.v_mods[4], &c.v_mods[5])?;
            vx = gate_residual(&vx, &y, &gate)?;
        }
        let amf = match &self.sst_dt {
            Some(d) => ada_rows(&d.a_rows, c.a_mods, 3, 6)?,
            None => ada(&self.a_sst, c.a_mods, 3, 6, dt)?,
        };
        let (afin, _) = mod_norm_quant(&ax, &amf[1], &amf[0], None)?;
        let a_ff_out = self.a_ff2.fwd(&self.a_ff0.fwd(&afin)?.gelu_tanh()?)?;
        ax = gate_residual(&ax, &a_ff_out, &amf[2])?;
        lap(4, &mut tm);

        Ok((vx, ax))
    }
}

/// Слот-стриминг dense-offload AvBlock'ов: 2 ping-pong набора долгоживущих
/// GPU-тензоров с СТАБИЛЬНЫМИ адресами (фундамент CUDA-graph replay на блок).
/// fill = регион-H2D сырых байт весов из pinned-зеркала mmap (без пер-блочных
/// аллокаций weights-пула и без QuantLinear::build на каждый своп) + D2D F32
/// sst-таблиц из резидентного стора. Байты и kernel-пути идентичны легаси
/// (тензоры contiguous) → бит-в-бит.
const SST_TABLES: [&str; 6] = [
    "scale_shift_table",
    "prompt_scale_shift_table",
    "audio_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "scale_shift_table_a2v_ca_audio",
    "scale_shift_table_a2v_ca_video",
];

fn av_weight_suffixes() -> Vec<String> {
    let mut v = Vec::new();
    for ap in ["attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"] {
        v.push(format!("{ap}.q_norm.weight"));
        v.push(format!("{ap}.k_norm.weight"));
        for lin in ["to_q", "to_k", "to_v", "to_gate_logits", "to_out.0"] {
            v.push(format!("{ap}.{lin}.weight"));
            v.push(format!("{ap}.{lin}.bias"));
        }
    }
    for f in ["ff.net.0.proj", "ff.net.2", "audio_ff.net.0.proj", "audio_ff.net.2"] {
        v.push(format!("{f}.weight"));
        v.push(format!("{f}.bias"));
    }
    v
}

fn block_prefix(idx: usize) -> String {
    format!("{DIT_PREFIX}.transformer_blocks.{idx}")
}

enum SlotSrc {
    /// Сырые байты файла по суффиксу имени (file-dtype == compute, проверено в build).
    File(String),
    /// Индекс в [`SST_TABLES`] (D2D из резидентного F32-стора).
    Table(usize),
    /// Строка `row` таблицы `t` из bf16-предкаст-стора (D2D): готовые
    /// [1,1,dim]-строки убирают tiny-обвязку (narrow/contiguous/cast) ada-путей
    /// из графа блока (~55% нод были <4µs-ядрами этой обвязки).
    TableRow { t: usize, row: usize },
}

struct SlotFill {
    src: SlotSrc,
    dptr: u64,
    bytes: usize,
}

struct SlotBlock {
    block: AvBlock,
    fills: Vec<SlotFill>,
    /// Запись слота (loader) только после завершения чтений предыдущего
    /// блока этого слота (compute) — write-after-read анти-гонка: легаси-путь
    /// получал это бесплатно от stream-ordered alloc пула, слоты — явно.
    ev_done: synaptix_core::device::cuda::SlotEvent,
}

struct SlotAlloc<'a> {
    ckpt: &'a LtxCheckpoint,
    device: Device,
    fills: Vec<SlotFill>,
}

impl<'a> SlotAlloc<'a> {
    fn w(&mut self, suffix: &str) -> Result<Tensor, LtxError> {
        let name = format!("{}.{suffix}", block_prefix(0));
        let (bytes, fdt, shape) = self
            .ckpt
            .raw_bytes(&name)
            .ok_or_else(|| LtxError::Load(format!("slot: нет тензора {name}")))?;
        let t = Tensor::empty_uninit(shape.to_vec(), fdt, self.device).map_err(LtxError::from)?;
        let (dptr, len) = t.cuda_region().map_err(LtxError::from)?;
        if len != bytes.len() {
            return Err(LtxError::Load(format!("slot: байты {name}: {len} != {}", bytes.len())));
        }
        self.fills.push(SlotFill { src: SlotSrc::File(suffix.to_string()), dptr, bytes: len });
        Ok(t)
    }

    fn lin(&mut self, prefix: &str, l: &str) -> Result<Lin, LtxError> {
        let w = self.w(&format!("{prefix}.{l}.weight"))?;
        let b = self.w(&format!("{prefix}.{l}.bias"))?;
        Ok(Lin(QuantLinear::dense(w, Some(b)).map_err(LtxError::from)?))
    }

    fn attn(&mut self, ap: &str, heads: usize, dim_head: usize) -> Result<Attn, LtxError> {
        Ok(Attn {
            q_norm: self.w(&format!("{ap}.q_norm.weight"))?,
            k_norm: self.w(&format!("{ap}.k_norm.weight"))?,
            to_q: self.lin(ap, "to_q")?,
            to_k: self.lin(ap, "to_k")?,
            to_v: self.lin(ap, "to_v")?,
            to_gate: self.lin(ap, "to_gate_logits")?,
            to_out: self.lin(ap, "to_out.0")?,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f32).sqrt(),
        })
    }

    fn table(&mut self, k: usize) -> Result<Tensor, LtxError> {
        let name = format!("{}.{}", block_prefix(0), SST_TABLES[k]);
        let info = self
            .ckpt
            .tensor_info(&name)
            .ok_or_else(|| LtxError::Load(format!("slot: нет таблицы {name}")))?;
        let t = Tensor::empty_uninit(info.shape.clone(), DType::F32, self.device).map_err(LtxError::from)?;
        let (dptr, len) = t.cuda_region().map_err(LtxError::from)?;
        self.fills.push(SlotFill { src: SlotSrc::Table(k), dptr, bytes: len });
        Ok(t)
    }

    /// Готовые [1,1,dim]-строки таблицы `k` в compute-dtype (см. [`SlotSrc::TableRow`]).
    fn table_rows(&mut self, k: usize, dt: DType) -> Result<Vec<Tensor>, LtxError> {
        let name = format!("{}.{}", block_prefix(0), SST_TABLES[k]);
        let info = self
            .ckpt
            .tensor_info(&name)
            .ok_or_else(|| LtxError::Load(format!("slot: нет таблицы {name}")))?;
        let (nrows, dim) = (info.shape[0], info.shape[1]);
        let mut rows = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let t = Tensor::empty_uninit(vec![1, 1, dim], dt, self.device).map_err(LtxError::from)?;
            let (dptr, len) = t.cuda_region().map_err(LtxError::from)?;
            self.fills.push(SlotFill { src: SlotSrc::TableRow { t: k, row: r }, dptr, bytes: len });
            rows.push(t);
        }
        Ok(rows)
    }
}

struct SlotState {
    slots: [SlotBlock; 2],
    /// `[блок][таблица] → (dptr, байты)` резидентного F32-стора sst-таблиц.
    sst_ptrs: Vec<Vec<(u64, usize)>>,
    /// `[блок][таблица] → (dptr, байты строки)` bf16-предкаст-стора (TableRow-fill).
    sst_dt_rows: Vec<Vec<(u64, usize)>>,
    _sst_store: Vec<Vec<Tensor>>,
    _sst_store_dt: Vec<Vec<Tensor>>,
}

impl SlotState {
    /// `None` → слоты неприменимы (LoRA / dtype-несовпадение / нестандартные
    /// формы) — вызывающий падает на легаси-карусель.
    fn build(ckpt: &LtxCheckpoint, nblocks: usize, device: Device, dtype: DType) -> Option<SlotState> {
        if ckpt.has_lora() || !matches!(device, Device::Cuda(_)) {
            return None;
        }
        let suffixes = av_weight_suffixes();
        let mut shapes0: Vec<Vec<usize>> = Vec::with_capacity(suffixes.len());
        for s in &suffixes {
            let (_, fdt, shape) = ckpt.raw_bytes(&format!("{}.{s}", block_prefix(0)))?;
            if fdt != dtype {
                return None;
            }
            shapes0.push(shape.to_vec());
        }
        for idx in 1..nblocks {
            let p = block_prefix(idx);
            for (s, sh0) in suffixes.iter().zip(&shapes0) {
                let (_, fdt, shape) = ckpt.raw_bytes(&format!("{p}.{s}"))?;
                if fdt != dtype || shape != sh0.as_slice() {
                    return None;
                }
            }
        }
        let build_block = || -> Result<SlotBlock, LtxError> {
            let ord = if let Device::Cuda(o) = device { o } else { 0 };
            let mut sa = SlotAlloc { ckpt, device, fills: Vec::new() };
            let block = AvBlock {
                v_sst: sa.table(0)?,
                v_prompt_sst: sa.table(1)?,
                v_attn1: sa.attn("attn1", 32, 128)?,
                v_attn2: sa.attn("attn2", 32, 128)?,
                v_ff0: sa.lin("ff.net.0", "proj")?,
                v_ff2: Lin(QuantLinear::dense(sa.w("ff.net.2.weight")?, Some(sa.w("ff.net.2.bias")?)).map_err(LtxError::from)?),
                a_sst: sa.table(2)?,
                a_prompt_sst: sa.table(3)?,
                a_attn1: sa.attn("audio_attn1", 32, 64)?,
                a_attn2: sa.attn("audio_attn2", 32, 64)?,
                a_ff0: sa.lin("audio_ff.net.0", "proj")?,
                a_ff2: Lin(QuantLinear::dense(sa.w("audio_ff.net.2.weight")?, Some(sa.w("audio_ff.net.2.bias")?)).map_err(LtxError::from)?),
                a2v_attn: sa.attn("audio_to_video_attn", 32, 64)?,
                v2a_attn: sa.attn("video_to_audio_attn", 32, 64)?,
                ca_audio_table: sa.table(4)?,
                ca_video_table: sa.table(5)?,
                sst_dt: Some(SstDt {
                    v_rows: sa.table_rows(0, dtype)?,
                    vp_rows: sa.table_rows(1, dtype)?,
                    a_rows: sa.table_rows(2, dtype)?,
                    ap_rows: sa.table_rows(3, dtype)?,
                    ca_a_rows: sa.table_rows(4, dtype)?,
                    ca_v_rows: sa.table_rows(5, dtype)?,
                }),
            };
            Ok(SlotBlock {
                block,
                fills: sa.fills,
                ev_done: synaptix_core::device::cuda::SlotEvent::new(ord).map_err(LtxError::from)?,
            })
        };
        let s0 = build_block().ok()?;
        let s1 = build_block().ok()?;
        let mut store: Vec<Vec<Tensor>> = Vec::with_capacity(nblocks);
        let mut ptrs: Vec<Vec<(u64, usize)>> = Vec::with_capacity(nblocks);
        let mut store_dt: Vec<Vec<Tensor>> = Vec::with_capacity(nblocks);
        let mut rows_dt: Vec<Vec<(u64, usize)>> = Vec::with_capacity(nblocks);
        for idx in 0..nblocks {
            let p = block_prefix(idx);
            let mut row = Vec::with_capacity(SST_TABLES.len());
            let mut prow = Vec::with_capacity(SST_TABLES.len());
            let mut row_dt = Vec::with_capacity(SST_TABLES.len());
            let mut prow_dt = Vec::with_capacity(SST_TABLES.len());
            for tname in SST_TABLES {
                let t = ckpt.get_raw(&format!("{p}.{tname}")).ok()?.to_dtype(DType::F32).ok()?;
                // bf16-предкаст таблицы: тот же per-element раунд f32→dt, что и
                // легаси-каст строки в ada() — бит-в-бит.
                let tdt = t.to_dtype(dtype).ok()?;
                let (base, total) = tdt.cuda_region().ok()?;
                let nrows = tdt.dims()[0];
                prow_dt.push((base, total / nrows));
                prow.push(t.cuda_region().ok()?);
                row.push(t);
                row_dt.push(tdt);
            }
            store.push(row);
            ptrs.push(prow);
            store_dt.push(row_dt);
            rows_dt.push(prow_dt);
        }
        if let Device::Cuda(o) = device {
            // одноразовый sync: alloc слотов/стора стрим-упорядочен на default,
            // первый fill пишет/читает с loader-стрима — без барьера это гонка.
            let _ = synaptix_core::device::cuda::synchronize(o);
        }
        Some(SlotState {
            slots: [s0, s1],
            sst_ptrs: ptrs,
            sst_dt_rows: rows_dt,
            _sst_store: store,
            _sst_store_dt: store_dt,
        })
    }

    /// Заливка блока `idx` в слот `slot` на `ls` (loader-стрим): wait ev_done →
    /// регион-копии. Асинхронна; вызывающий синкает `ls` перед использованием.
    fn fill(
        &self,
        ckpt: &LtxCheckpoint,
        idx: usize,
        slot: usize,
        ls: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(), LtxError> {
        let sb = &self.slots[slot];
        sb.ev_done.wait_on(ls).map_err(LtxError::from)?;
        let p = block_prefix(idx);
        for f in &sb.fills {
            match &f.src {
                SlotSrc::File(suffix) => {
                    let name = format!("{p}.{suffix}");
                    let (bytes, _, _) = ckpt
                        .raw_bytes(&name)
                        .ok_or_else(|| LtxError::Load(format!("slot fill: нет {name}")))?;
                    if bytes.len() != f.bytes {
                        return Err(LtxError::Load(format!("slot fill: байты {name}")));
                    }
                    synaptix_core::device::cuda::htod_into_region(ls, f.dptr, bytes)
                        .map_err(LtxError::from)?;
                }
                SlotSrc::Table(k) => {
                    let (src, len) = self.sst_ptrs[idx][*k];
                    if len != f.bytes {
                        return Err(LtxError::Load("slot fill: байты sst".into()));
                    }
                    synaptix_core::device::cuda::dtod_into_region(ls, f.dptr, src, len)
                        .map_err(LtxError::from)?;
                }
                SlotSrc::TableRow { t, row } => {
                    let (base, row_bytes) = self.sst_dt_rows[idx][*t];
                    if row_bytes != f.bytes {
                        return Err(LtxError::Load("slot fill: байты sst-row".into()));
                    }
                    let src = base + (*row as u64) * row_bytes as u64;
                    synaptix_core::device::cuda::dtod_into_region(ls, f.dptr, src, row_bytes)
                        .map_err(LtxError::from)?;
                }
            }
        }
        Ok(())
    }
}

/// D2D-копия `src` в долгоживущий фикс-буфер `dst` (та же форма/dtype) на
/// default-стриме — стрим-упорядочена с продюсерами/консьюмерами.
fn copy_into_fixed(dst: &Tensor, src: &Tensor) -> R<()> {
    if dst.dims() != src.dims() || dst.dtype() != src.dtype() {
        return Err(SynaptixError::Unsupported("copy_into_fixed: форма/dtype"));
    }
    let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
    let (d, dl) = dst.cuda_region()?;
    let (s, sl) = src_c.cuda_region()?;
    if dl != sl {
        return Err(SynaptixError::Unsupported("copy_into_fixed: байты"));
    }
    let ord = if let Device::Cuda(o) = dst.device() { o } else { 0 };
    let ds = synaptix_core::device::cuda::default_stream(ord)?;
    synaptix_core::device::cuda::dtod_into_region(&ds, d, s, dl)
}

fn like_fixed(src: &Tensor) -> R<Tensor> {
    Tensor::empty_uninit(src.dims().to_vec(), src.dtype(), src.device())
}

/// Захват одного блок-шага в CUDA-graph на `stream` (default). `step` уже
/// прогрет eager-проходом (NVRTC/JIT/пул); под capture ничего не исполняется —
/// только записываются ноды. Event-tracking выключен на время capture
/// (cross-capture wait → CAPTURE_ISOLATION, см. GraphCapturer LLM-decode).
fn capture_block<F: FnMut() -> R<()>>(
    stream: &Arc<cudarc::driver::CudaStream>,
    mut step: F,
) -> R<cudarc::driver::CudaGraph> {
    use cudarc::driver::sys::{CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum};
    let ctx = stream.context();
    let prev = ctx.is_event_tracking();
    unsafe { ctx.disable_event_tracking() };
    if let Err(e) = stream.begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED) {
        if prev {
            unsafe { ctx.enable_event_tracking() };
        }
        return Err(SynaptixError::Cuda(format!("begin_capture: {e:?}")));
    }
    let step_res = step();
    let end = stream
        .end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    if prev {
        unsafe { ctx.enable_event_tracking() };
    }
    step_res?;
    end.map_err(|e| SynaptixError::Cuda(format!("end_capture: {e:?}")))?
        .ok_or(SynaptixError::Cuda("end_capture: stream не в capture".into()))
}

/// Отпечаток позиционных сеток: rope3d детерминирован от (форма, positions) —
/// совпадение отпечатка позволяет переиспользовать static-часть GraphCtx
/// (CPU-тригонометрия ~58M элементов + сотни MB H2D на каждый шаг s2 были
/// главными дырами пролога).
fn rope_fingerprint(v_positions: &[f64], a_positions: &[f64]) -> Vec<f64> {
    let mut fp = Vec::with_capacity(10);
    for ps in [v_positions, a_positions] {
        let n = ps.len();
        fp.push(n as f64);
        for i in [0usize, n / 3, (2 * n) / 3, n.saturating_sub(1)] {
            fp.push(*ps.get(i).unwrap_or(&0.0));
        }
    }
    fp
}

/// Фикс-буферы шага под CUDA-graph replay: граф захватывается один раз на
/// (Tv,Ta)-форму и реплеится все шаги — всё, что блок читает/пишет (vx/ax,
/// per-step модуляции, rope, текст-кондиции), живёт по стабильным адресам;
/// пролог шага копируется сюда D2D, оригиналы дропаются (нетто-VRAM ≈ легаси:
/// фикс-буферы замещают per-step копии).
struct GraphCtx {
    tv: usize,
    ta: usize,
    vxb: Tensor,
    axb: Tensor,
    v_mods: Vec<Tensor>,
    a_mods: Vec<Tensor>,
    v_pmod: Tensor,
    a_pmod: Tensor,
    v_css: Tensor,
    v_cg: Tensor,
    a_css: Tensor,
    a_cg: Tensor,
    v_rope: (Tensor, Tensor),
    a_rope: (Tensor, Tensor),
    v_cpe: (Tensor, Tensor),
    a_cpe: (Tensor, Tensor),
    v_ctx: Tensor,
    a_ctx: Tensor,
    /// Отпечаток positions (см. [`rope_fingerprint`]) — валидность static-части.
    fp: Vec<f64>,
    /// Граф слота 0 / слота 1 (чётные/нечётные блоки).
    graphs: [std::sync::Mutex<Option<GraphHolder>>; 2],
}

struct GraphHolder(cudarc::driver::CudaGraph);
// SAFETY: CUDA driver API потокобезопасен (контекст в обёртке cudarc); граф
// запускается только на своём capture-стриме, доступ сериализован Mutex'ом.
// Send/Sync нужны лишь потому, что AvDit шарится в thread::scope префетча.
unsafe impl Send for GraphHolder {}
unsafe impl Sync for GraphHolder {}

impl GraphCtx {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vx: &Tensor, ax: &Tensor,
        v_mods: &[Tensor], a_mods: &[Tensor],
        v_pmod: &Tensor, a_pmod: &Tensor,
        v_css: &Tensor, v_cg: &Tensor, a_css: &Tensor, a_cg: &Tensor,
        v_rope: (&Tensor, &Tensor), a_rope: (&Tensor, &Tensor), v_cpe: (&Tensor, &Tensor),
        v_ctx: &Tensor, a_ctx: &Tensor, fp: Vec<f64>,
    ) -> R<Self> {
        let likes = |v: &[Tensor]| -> R<Vec<Tensor>> { v.iter().map(like_fixed).collect() };
        let a_rope_b = (like_fixed(a_rope.0)?, like_fixed(a_rope.1)?);
        // audio cross_pe == audio rope (та же ось) — шарим буферы.
        let a_cpe_b = (a_rope_b.0.clone(), a_rope_b.1.clone());
        Ok(Self {
            tv: vx.dims()[1],
            ta: ax.dims()[1],
            vxb: like_fixed(vx)?,
            axb: like_fixed(ax)?,
            v_mods: likes(v_mods)?,
            a_mods: likes(a_mods)?,
            v_pmod: like_fixed(v_pmod)?,
            a_pmod: like_fixed(a_pmod)?,
            v_css: like_fixed(v_css)?,
            v_cg: like_fixed(v_cg)?,
            a_css: like_fixed(a_css)?,
            a_cg: like_fixed(a_cg)?,
            v_rope: (like_fixed(v_rope.0)?, like_fixed(v_rope.1)?),
            a_rope: a_rope_b,
            v_cpe: (like_fixed(v_cpe.0)?, like_fixed(v_cpe.1)?),
            a_cpe: a_cpe_b,
            v_ctx: like_fixed(v_ctx)?,
            a_ctx: like_fixed(a_ctx)?,
            fp,
            graphs: [std::sync::Mutex::new(None), std::sync::Mutex::new(None)],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_step(
        &self,
        vx: &Tensor, ax: &Tensor,
        v_mods: &[Tensor], a_mods: &[Tensor],
        v_pmod: &Tensor, a_pmod: &Tensor,
        v_css: &Tensor, v_cg: &Tensor, a_css: &Tensor, a_cg: &Tensor,
        v_ctx: &Tensor, a_ctx: &Tensor,
    ) -> R<()> {
        copy_into_fixed(&self.vxb, vx)?;
        copy_into_fixed(&self.axb, ax)?;
        for (d, s) in self.v_mods.iter().zip(v_mods) {
            copy_into_fixed(d, s)?;
        }
        for (d, s) in self.a_mods.iter().zip(a_mods) {
            copy_into_fixed(d, s)?;
        }
        copy_into_fixed(&self.v_pmod, v_pmod)?;
        copy_into_fixed(&self.a_pmod, a_pmod)?;
        copy_into_fixed(&self.v_css, v_css)?;
        copy_into_fixed(&self.v_cg, v_cg)?;
        copy_into_fixed(&self.a_css, a_css)?;
        copy_into_fixed(&self.a_cg, a_cg)?;
        copy_into_fixed(&self.v_ctx, v_ctx)?;
        copy_into_fixed(&self.a_ctx, a_ctx)?;
        Ok(())
    }

    /// Static-часть (rope/cross-pe): только при (пере)создании GraphCtx —
    /// значения зависят лишь от (формы, positions), см. [`rope_fingerprint`].
    fn upload_static(
        &self,
        v_rope: (&Tensor, &Tensor), a_rope: (&Tensor, &Tensor), v_cpe: (&Tensor, &Tensor),
    ) -> R<()> {
        copy_into_fixed(&self.v_rope.0, v_rope.0)?;
        copy_into_fixed(&self.v_rope.1, v_rope.1)?;
        copy_into_fixed(&self.a_rope.0, a_rope.0)?;
        copy_into_fixed(&self.a_rope.1, a_rope.1)?;
        copy_into_fixed(&self.v_cpe.0, v_cpe.0)?;
        copy_into_fixed(&self.v_cpe.1, v_cpe.1)?;
        Ok(())
    }

    /// AvCtx из фикс-буферов (NAG в graph-режиме не поддержан — eager-фоллбэк).
    fn av_ctx(&self, dtype: DType) -> AvCtx<'_> {
        AvCtx {
            v_mods: &self.v_mods,
            a_mods: &self.a_mods,
            v_pmod: &self.v_pmod,
            a_pmod: &self.a_pmod,
            v_cross_ss: &self.v_css,
            v_cross_gate: &self.v_cg,
            a_cross_ss: &self.a_css,
            a_cross_gate: &self.a_cg,
            v_ctx: &self.v_ctx,
            a_ctx: &self.a_ctx,
            v_rope: (&self.v_rope.0, &self.v_rope.1),
            a_rope: (&self.a_rope.0, &self.a_rope.1),
            v_cross_pe: (&self.v_cpe.0, &self.v_cpe.1),
            a_cross_pe: (&self.a_cpe.0, &self.a_cpe.1),
            dtype,
            v_nag: None,
        }
    }

    /// Один блок-шаг на фикс-буферах: forward + copy-back vx/ax (тело графа).
    fn block_step(&self, blk: &AvBlock, ctx: &AvCtx) -> R<()> {
        let (nvx, nax) = blk.forward(&self.vxb, &self.axb, ctx, None)?;
        copy_into_fixed(&self.vxb, &nvx)?;
        copy_into_fixed(&self.axb, &nax)?;
        Ok(())
    }
}

/// Joint Audio/Video DiT поверх чекпойнта LTX-2.3. `forward → (velocity_video,
/// velocity_audio)`, оба `[1,T,128]`.
pub struct AvDit {
    // video module-level
    v_patchify: Lin,
    v_ada: AdaLN,
    v_pada: AdaLN,
    v_sst_out: Tensor,
    v_proj_out: Lin,
    // audio module-level
    a_patchify: Lin,
    a_ada: AdaLN,
    a_pada: AdaLN,
    a_sst_out: Tensor,
    a_proj_out: Lin,
    // av_ca module-level adaln
    avca_v_ss: AdaLN,
    avca_a_ss: AdaLN,
    avca_a2v_gate: AdaLN,
    avca_v2a_gate: AdaLN,
    /// Резидентные блоки на `device` (offload=false). При offload пусто — блоки
    /// грузятся mmap→VRAM по требованию в forward через `stream`.
    blocks: Vec<AvBlock>,
    theta: f64,
    v_max_pos: Vec<f64>,
    a_max_pos: Vec<f64>,
    cross_max_pos: f64,
    ts_mult: f64,
    avca_mult: f64,
    device: Device,
    dtype: DType,
    quant: DType,
    nblocks: usize,
    /// Pinned-кэш стримящихся весов при dense-offload (см. [`pin_ckpt_for_stream`]).
    _host_pin: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    /// Pinned-зеркало host-stream квант-блоков (первый своп наполняет).
    _host_mirror: Option<synaptix_core::device::cuda::PinMirrorGuard>,
    /// `Some` → offload (блоки стримятся mmap→`device` поблочно, без host-копии);
    /// `None` → блоки резидентны на `device` (либо на CPU при `host_stream`).
    stream: Option<Arc<LtxCheckpoint>>,
    /// Квант-offload: `blocks` КВАНТОВАННЫЕ на CPU, стримятся на `device`
    /// поблочно в forward (bit-identical резидентному кванту).
    host_stream: bool,
    /// Ленивый слот-стриминг dense-offload (`runtime::ltx_block_mode() >= 1`).
    /// Релизуемый ([`AvDit::release_slots`] перед VAE: слоты держат ~1.5GB —
    /// иначе VAE-бюджет видит меньше free VRAM и меняет тайлинг).
    slot_state: std::sync::Mutex<SlotCell>,
    /// Фикс-буферы + графы текущей (Tv,Ta)-формы (`ltx_block_mode() >= 2`);
    /// графы читают адреса слотов → при пересборке слотов сбрасывается.
    graph_ctx: std::sync::Mutex<Option<Arc<GraphCtx>>>,
}

enum SlotCell {
    Unbuilt,
    Na,
    Ready(Arc<SlotState>),
}

impl AvDit {
    /// compute-dtype модели (касты ctx/латентов в pipeline).
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    pub fn load(ckpt: &LtxCheckpoint, device: Device, dtype: DType, quant: DType) -> Result<Self, LtxError> {
        Self::load_with(ckpt, device, dtype, quant, false)
    }

    /// `offload=true`: блоки стримятся на `device` поблочно. `quant==dtype` → dense
    /// bf16 из mmap (ckpt на CPU); `quant!=dtype` → host-stream квант-блоков
    /// (см. [`VideoDit::load_with`]).
    pub fn load_with(
        ckpt: &LtxCheckpoint, device: Device, dtype: DType, quant: DType, offload: bool,
    ) -> Result<Self, LtxError> {
        let t = &ckpt.config.transformer;
        let host_stream = offload && quant != dtype;
        // Квант-путь требует CUDA-весов (quantize_*); ckpt на CPU →
        // GPU-вью для материализации на `device` ДО кванта (см. VideoDit::load_with).
        let gpu_view;
        let wckpt: &LtxCheckpoint = if offload && !host_stream {
            ckpt
        } else {
            gpu_view = ckpt.view_on(device);
            &gpu_view
        };
        let g = |k: &str| -> Result<Lin, LtxError> {
            let l = Lin::load(wckpt, DIT_PREFIX, k, true, dtype, dtype)?;
            if offload { l.to_device(device).map_err(LtxError::from) } else { Ok(l) }
        };
        let ga = |k: &str| -> Result<AdaLN, LtxError> {
            let a = AdaLN::load(wckpt, &format!("{DIT_PREFIX}.{k}"), dtype)?;
            if offload { a.to_device(device).map_err(LtxError::from) } else { Ok(a) }
        };
        let raw = |k: &str| -> Result<Tensor, LtxError> {
            let s = wckpt.get_raw(&format!("{DIT_PREFIX}.{k}"))?.to_dtype(DType::F32)?;
            if offload { s.to_device(device).map_err(LtxError::from) } else { Ok(s) }
        };
        let nblocks = crate::runtime::dit_nblocks_cap()
            .unwrap_or(t.num_layers).min(t.num_layers);
        // dense-offload: НЕ материализуем блоки на host (~42GB bf16 для A/V!) —
        // держим GPU-вью чекпойнта и грузим блок mmap→VRAM по требованию в forward.
        // host-stream: квантуем блоки 1× на GPU → CPU-резидент (квант-байты).
        let host_pin = if offload && !host_stream { pin_ckpt_for_stream(ckpt) } else { None };
        let mut host_mirror: Option<synaptix_core::device::cuda::PinMirrorGuard> = None;
        let (blocks, stream) = if offload && !host_stream {
            (Vec::new(), Some(Arc::new(ckpt.view_on(device))))
        } else if host_stream {
            synaptix_core::device::cuda::set_offload_pinned(true);
            let mut blocks = Vec::with_capacity(nblocks);
            for i in 0..nblocks {
                let b = AvBlock::load(wckpt, i, quant, dtype)?;
                blocks.push(b.to_device(Device::Cpu)?);
            }
            synaptix_core::device::cuda::set_offload_pinned(false);
            host_mirror = Some(synaptix_core::device::cuda::PinMirrorGuard::new());
            // прогрев зеркала: fetch→drop материализует pinned-копии блоков
            // (~1.5-2s в load) — иначе первый denoise-шаг платит ~3.7s wait.
            for b in &blocks {
                synaptix_core::device::cuda::set_pin_mirror(true);
                let warm = b.to_device(device);
                synaptix_core::device::cuda::set_pin_mirror(false);
                warm?;
            }
            (blocks, None)
        } else {
            let mut blocks = Vec::with_capacity(nblocks);
            for i in 0..nblocks {
                blocks.push(AvBlock::load(wckpt, i, quant, dtype)?);
            }
            (blocks, None)
        };
        let a_max_pos = t.audio_positional_embedding_max_pos.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let cross_max_pos = t.positional_embedding_max_pos[0]
            .max(t.audio_positional_embedding_max_pos[0]) as f64;
        Ok(Self {
            v_patchify: g("patchify_proj")?,
            v_ada: ga("adaln_single")?,
            v_pada: ga("prompt_adaln_single")?,
            v_sst_out: raw("scale_shift_table")?,
            v_proj_out: g("proj_out")?,
            a_patchify: g("audio_patchify_proj")?,
            a_ada: ga("audio_adaln_single")?,
            a_pada: ga("audio_prompt_adaln_single")?,
            a_sst_out: raw("audio_scale_shift_table")?,
            a_proj_out: g("audio_proj_out")?,
            avca_v_ss: ga("av_ca_video_scale_shift_adaln_single")?,
            avca_a_ss: ga("av_ca_audio_scale_shift_adaln_single")?,
            avca_a2v_gate: ga("av_ca_a2v_gate_adaln_single")?,
            avca_v2a_gate: ga("av_ca_v2a_gate_adaln_single")?,
            blocks,
            theta: t.positional_embedding_theta,
            v_max_pos: t.positional_embedding_max_pos.iter().map(|&x| x as f64).collect(),
            a_max_pos,
            cross_max_pos,
            ts_mult: t.timestep_scale_multiplier,
            avca_mult: t.av_ca_timestep_scale_multiplier,
            device,
            dtype,
            quant,
            nblocks,
            _host_pin: host_pin,
            _host_mirror: host_mirror,
            stream,
            host_stream,
            slot_state: std::sync::Mutex::new(SlotCell::Unbuilt),
            graph_ctx: std::sync::Mutex::new(None),
        })
    }

    /// Освободить слот-буферы весов (~1.5GB) и графы — вызывать после последнего
    /// DiT-прохода перед VAE-декодом (бюджет тайлинга). Следующий forward
    /// перестроит лениво. Drop стрим-упорядочен (cuMemFreeAsync).
    pub fn release_slots(&self) {
        *self.graph_ctx.lock().expect("graph_ctx lock") = None;
        let mut g = self.slot_state.lock().expect("slot_state lock");
        if matches!(&*g, SlotCell::Ready(_)) {
            *g = SlotCell::Unbuilt;
        }
    }

    fn slot_state_get(&self) -> Option<Arc<SlotState>> {
        if crate::runtime::ltx_block_mode() == 0 || self.host_stream {
            return None;
        }
        let sc = self.stream.as_ref()?;
        let mut g = self.slot_state.lock().expect("slot_state lock");
        match &*g {
            SlotCell::Ready(a) => Some(a.clone()),
            SlotCell::Na => None,
            SlotCell::Unbuilt => match SlotState::build(sc, self.nblocks, self.device, self.dtype) {
                Some(s) => {
                    // графы (если были) ссылались на адреса СТАРЫХ слотов.
                    *self.graph_ctx.lock().expect("graph_ctx lock") = None;
                    let a = Arc::new(s);
                    *g = SlotCell::Ready(a.clone());
                    Some(a)
                }
                None => {
                    *g = SlotCell::Na;
                    None
                }
            },
        }
    }

    /// head: `norm_out` (LN no-affine) → `(1+scale)·x+shift` (sst_out+embedded) → proj.
    /// Построчно по T → чанк при длинных T бит-в-бит (LN f32-каст полного
    /// [T,dim] добивал VRAM на 20s-refine после 48 блоков).
    fn head(&self, x: &Tensor, sst_out: &Tensor, embedded: &Tensor, proj: &Lin, dim: usize) -> R<Tensor> {
        let t_len = x.dims()[1];
        let chunk: usize = 16384;
        if chunk > 0 && t_len > chunk {
            let mut parts: Vec<Tensor> = Vec::with_capacity(t_len.div_ceil(chunk));
            let mut o = 0usize;
            while o < t_len {
                let n = chunk.min(t_len - o);
                let xs = x.narrow(1, o, n)?.contiguous()?;
                let es = embedded.narrow(1, o, n)?.contiguous()?;
                parts.push(self.head_one(&xs, sst_out, &es, proj, dim)?);
                o += n;
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            return Ok(Tensor::cat(&refs, 1)?);
        }
        self.head_one(x, sst_out, embedded, proj, dim)
    }

    fn head_one(&self, x: &Tensor, sst_out: &Tensor, embedded: &Tensor, proj: &Lin, dim: usize) -> R<Tensor> {
        let shift = sst_out.narrow(0, 0, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?.broadcast_add(embedded)?;
        let scale = sst_out.narrow(0, 1, 1)?.contiguous()?.reshape(vec![1, 1, dim])?.to_dtype(self.dtype)?.broadcast_add(embedded)?;
        let x = layer_norm_no_affine(x)?;
        let x = x.broadcast_mul(&scale.add_scalar(1.0)?)?.broadcast_add(&shift)?;
        proj.fwd(&x)
    }

    /// Forward → (velocity_video `[1,Tv,128]`, velocity_audio `[1,Ta,128]`).
    /// `positions_*` — host f64: video `[3*Tv*2]`, audio `[1*Ta*2]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        v_latent: &Tensor, v_timesteps: &[f32], v_sigma: f32, v_positions: &[f64], v_context: &Tensor,
        a_latent: &Tensor, a_timesteps: &[f32], a_sigma: f32, a_positions: &[f64], a_context: &Tensor,
        v_nag: Option<(&Tensor, f32, f32, f32)>,
    ) -> Result<(Tensor, Tensor), LtxError> {
        let (dev, dt) = (self.device, self.dtype);
        let tv = v_latent.dims()[1];
        let ta = a_latent.dims()[1];
        // Компакт ПЕРЕД подготовкой: на длинных T (T≈51-54k) split_modul/rope
        // аллоцируют десятки [1,T,dim]-тензоров — фрагментированный после
        // прошлой фазы пул не даёт куски (OOM 418MB при свободных GB).
        if let Device::Cuda(ord_c) = dev {
            if tv > 32768 {
                let _ = synaptix_core::device::cuda::synchronize_all(ord_c);
                let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord_c);
            }
        }
        let mut vx = self.v_patchify.fwd(v_latent)?; // [1,Tv,4096]
        let mut ax = self.a_patchify.fwd(a_latent)?; // [1,Ta,2048]

        // within-stream adaln (per-token). split СРАЗУ после modulate и drop
        // исходного [1,T,9,dim] (3.96GB на 20s) — иначе он жил через rope/pmod
        // подготовку и добивал VRAM (OOM_TOP: modul жив на OOM в подготовке).
        let vts: Vec<f32> = v_timesteps.iter().map(|&x| x * self.ts_mult as f32).collect();
        let (v_modul, v_emb) = self.v_ada.modulate(&vts, tv, dev, dt)?;
        let v_mods = split_modul(&v_modul.reshape(vec![1, tv, 9, 4096])?)?;
        drop(v_modul);
        // emb нужны только в head (после всех блоков) — паркуем на CPU
        // ([1,T,dim] = 440+220MB; 20s-refine VRAM на волоске), поднимаем перед head.
        let v_emb = v_emb.reshape(vec![1, tv, 4096])?.to_device(Device::Cpu)?;
        let ats: Vec<f32> = a_timesteps.iter().map(|&x| x * self.ts_mult as f32).collect();
        let (a_modul, a_emb) = self.a_ada.modulate(&ats, ta, dev, dt)?;
        let a_mods = split_modul(&a_modul.reshape(vec![1, ta, 9, 2048])?)?;
        drop(a_modul);
        let a_emb = a_emb.reshape(vec![1, ta, 2048])?.to_device(Device::Cpu)?;
        // prompt adaln (sigma)
        let (v_pmod, _) = self.v_pada.modulate(&[v_sigma * self.ts_mult as f32], 1, dev, dt)?;
        let v_pmod = v_pmod.reshape(vec![1, 1, 2, 4096])?;
        let (a_pmod, _) = self.a_pada.modulate(&[a_sigma * self.ts_mult as f32], 1, dev, dt)?;
        let a_pmod = a_pmod.reshape(vec![1, 1, 2, 2048])?;

        // av_ca cross-timesteps: scale_shift на (cross_sigma·ts_mult), gate на
        // (cross_sigma·ts_mult·avca_factor), avca_factor = avca_mult/ts_mult.
        let factor = self.avca_mult / self.ts_mult;
        // video args: cross_modality = audio → conditioned on a_sigma
        let v_css = self.avca_v_ss.modulate(&[a_sigma * self.ts_mult as f32], 1, dev, dt)?.0.reshape(vec![1, 1, 4, 4096])?;
        let v_cg = self.avca_a2v_gate.modulate(&[a_sigma * self.ts_mult as f32 * factor as f32], 1, dev, dt)?.0.reshape(vec![1, 1, 1, 4096])?;
        // audio args: cross_modality = video → conditioned on v_sigma
        let a_css = self.avca_a_ss.modulate(&[v_sigma * self.ts_mult as f32], 1, dev, dt)?.0.reshape(vec![1, 1, 4, 2048])?;
        let a_cg = self.avca_v2a_gate.modulate(&[v_sigma * self.ts_mult as f32 * factor as f32], 1, dev, dt)?.0.reshape(vec![1, 1, 1, 2048])?;

        // Решение по graph-кэшу ДО RoPE: при готовом GraphCtx той же формы и
        // positions rope3d (CPU-тригонометрия + H2D сотен MB) пропускается.
        let prof = crate::runtime::ltx_prof();
        let mut tm = [0f32; 5];
        let mut t_stream = 0f32;
        let slot_st = self.slot_state_get();
        let use_graph = slot_st.is_some()
            && crate::runtime::ltx_block_mode() >= 2
            && v_nag.is_none()
            && matches!(self.device, Device::Cuda(_));
        let fp = rope_fingerprint(v_positions, a_positions);
        let gc_ready: Option<Arc<GraphCtx>> = if use_graph {
            self.graph_ctx
                .lock()
                .expect("graph_ctx lock")
                .as_ref()
                .filter(|g| g.tv == tv && g.ta == ta && g.fp == fp)
                .cloned()
        } else {
            None
        };
        // RoPE: video 3D (32×128), audio 1D (32×64); cross_pe — temporal-only
        // 32×64; audio cross_pe == audio within-stream pe (та же ось/параметры).
        let rope = if gc_ready.is_none() {
            let (vc, vs) = rope3d(v_positions, 3, tv, 32, 128, self.theta, &self.v_max_pos, dev, dt)?;
            let (ac, as_) = rope3d(a_positions, 1, ta, 32, 64, self.theta, &self.a_max_pos, dev, dt)?;
            let v_temporal: Vec<f64> = v_positions[..tv * 2].to_vec();
            let (vcc, vcs) = rope3d(&v_temporal, 1, tv, 32, 64, self.theta, &[self.cross_max_pos], dev, dt)?;
            Some((vc, vs, ac, as_, vcc, vcs))
        } else {
            None
        };

        // Компакт после подготовки (mods/rope/emb уже размещены): дыры от
        // GEMM-транзиентов modulate/rope возвращаются драйверу до цикла блоков —
        // иначе на длинных T фрагментация сегментов копится с первого блока.
        if let Device::Cuda(ord_c) = dev {
            if tv > 32768 {
                let _ = synaptix_core::device::cuda::synchronize_all(ord_c);
                let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord_c);
            }
        }

        if use_graph {
            let st = slot_st.as_ref().expect("graph path: slots");
            let sc = self.stream.as_deref().expect("slot path: stream ckpt");
            let ord = if let Device::Cuda(o) = self.device { o } else { 0 };
            let gc = match gc_ready {
                Some(g) => g,
                None => {
                    let (vc, vs, ac, as_, vcc, vcs) = rope.as_ref().expect("rope для GraphCtx");
                    let g = Arc::new(GraphCtx::new(
                        &vx, &ax, &v_mods, &a_mods, &v_pmod, &a_pmod, &v_css, &v_cg, &a_css,
                        &a_cg, (vc, vs), (ac, as_), (vcc, vcs), v_context, a_context, fp,
                    )?);
                    g.upload_static((vc, vs), (ac, as_), (vcc, vcs))?;
                    *self.graph_ctx.lock().expect("graph_ctx lock") = Some(g.clone());
                    g
                }
            };
            gc.upload_step(
                &vx, &ax, &v_mods, &a_mods, &v_pmod, &a_pmod, &v_css, &v_cg, &a_css, &a_cg,
                v_context, a_context,
            )?;
            // пролог-оригиналы дропаются — их данные уже в фикс-буферах
            // (нетто-VRAM: фикс-буферы вместо per-step копий).
            drop(v_mods);
            drop(a_mods);
            drop(v_pmod);
            drop(a_pmod);
            drop(v_css);
            drop(v_cg);
            drop(a_css);
            drop(a_cg);
            drop(rope);
            let gctx = gc.av_ctx(dt);
            let ls = synaptix_core::device::cuda::loader_stream(ord).map_err(LtxError::from)?;
            let ds = synaptix_core::device::cuda::default_stream(ord).map_err(LtxError::from)?;
            st.fill(sc, 0, 0, &ls)?;
            let _ = ls.synchronize();
            for i in 0..self.nblocks {
                let s = i % 2;
                let need_capture = gc.graphs[s].lock().expect("graph lock").is_none();
                if need_capture {
                    // eager-проход = warmup (NVRTC/JIT/пул) И фактический compute
                    // блока i; затем capture записывает ноды БЕЗ исполнения.
                    // Префетч на время capture сериализован (одноразово на стадию).
                    gc.block_step(&st.slots[s].block, &gctx)?;
                    let _ = ds.synchronize();
                    let graph = capture_block(&ds, || gc.block_step(&st.slots[s].block, &gctx))
                        .map_err(LtxError::from)?;
                    *gc.graphs[s].lock().expect("graph lock") = Some(GraphHolder(graph));
                    st.slots[s].ev_done.record_default(ord).map_err(LtxError::from)?;
                    if i + 1 < self.nblocks {
                        st.fill(sc, i + 1, (i + 1) % 2, &ls)?;
                        let _ = ls.synchronize();
                    }
                } else {
                    let lsc = ls.clone();
                    let stc = &st;
                    let gcc = &gc;
                    std::thread::scope(|sp| -> Result<(), LtxError> {
                        let h = if i + 1 < self.nblocks {
                            Some(sp.spawn(move || -> Result<(), LtxError> {
                                stc.fill(sc, i + 1, (i + 1) % 2, &lsc)?;
                                let _ = lsc.synchronize();
                                Ok(())
                            }))
                        } else {
                            None
                        };
                        gcc.graphs[s]
                            .lock()
                            .expect("graph lock")
                            .as_ref()
                            .expect("captured graph")
                            .0
                            .launch()
                            .map_err(|e| LtxError::Load(format!("graph launch: {e:?}")))?;
                        stc.slots[s].ev_done.record_default(ord).map_err(LtxError::from)?;
                        let tj = std::time::Instant::now();
                        if let Some(h) = h {
                            h.join().expect("ltx graph prefetch thread panicked")?;
                        }
                        if prof {
                            t_stream += tj.elapsed().as_secs_f32();
                        }
                        Ok(())
                    })?;
                }
            }
            vx = gc.vxb.clone();
            ax = gc.axb.clone();
        } else {
        let (vc, vs, ac, as_, vcc, vcs) = rope.as_ref().expect("rope eager");
        let ctx = AvCtx {
            v_mods: &v_mods, a_mods: &a_mods, v_pmod: &v_pmod, a_pmod: &a_pmod,
            v_cross_ss: &v_css, v_cross_gate: &v_cg, a_cross_ss: &a_css, a_cross_gate: &a_cg,
            v_ctx: v_context, a_ctx: a_context,
            v_rope: (vc, vs), a_rope: (ac, as_),
            v_cross_pe: (vcc, vcs), a_cross_pe: (ac, as_),
            dtype: dt,
            v_nag: v_nag.map(|(t, s, a, tau)| (t, s, a, tau)),
        };
        if let Some(st) = slot_st {
            let sc = self.stream.as_deref().expect("slot path: stream ckpt");
            let ord = if let Device::Cuda(o) = self.device { o } else { 0 };
            let ls = synaptix_core::device::cuda::loader_stream(ord).map_err(LtxError::from)?;
            st.fill(sc, 0, 0, &ls)?;
            let _ = ls.synchronize();
            for i in 0..self.nblocks {
                let lsc = ls.clone();
                let stc = &st;
                std::thread::scope(|sp| -> Result<(), LtxError> {
                    let h = if i + 1 < self.nblocks {
                        Some(sp.spawn(move || -> Result<(), LtxError> {
                            stc.fill(sc, i + 1, (i + 1) % 2, &lsc)?;
                            let _ = lsc.synchronize();
                            Ok(())
                        }))
                    } else {
                        None
                    };
                    let (nvx, nax) = stc.slots[i % 2].block.forward(&vx, &ax, &ctx, prof.then_some(&mut tm))?;
                    vx = nvx;
                    ax = nax;
                    stc.slots[i % 2].ev_done.record_default(ord).map_err(LtxError::from)?;
                    let tj = std::time::Instant::now();
                    if let Some(h) = h {
                        h.join().expect("ltx av slot prefetch thread panicked")?;
                    }
                    if prof {
                        t_stream += tj.elapsed().as_secs_f32();
                    }
                    Ok(())
                })?;
            }
        } else if self.stream.is_some() || self.host_stream {
            // offload: стрим блоков (mmap bf16 / host-RAM квант) с префетчем i+1 на
            // фоновом потоке (loader-stream + pinned staging) — как VideoDit::forward.
            let (quant, dtype) = (self.quant, self.dtype);
            let stream = self.stream.as_deref();
            let fetch = |i: usize| -> Result<AvBlock, LtxError> {
                match stream {
                    Some(sc) => AvBlock::load(sc, i, quant, dtype),
                    None => {
                        // host-stream: H2D через pinned-зеркало (ptr блок-Vec'ов
                        // стабильны) — без staging-копии на каждый своп.
                        synaptix_core::device::cuda::set_pin_mirror(true);
                        let r = self.blocks[i].to_device(self.device).map_err(LtxError::from);
                        synaptix_core::device::cuda::set_pin_mirror(false);
                        r
                    }
                }
            };
            let fetch = &fetch;
            let ord = if let Device::Cuda(o) = self.device { o } else { 0 };
            synaptix_core::device::cuda::set_offload_pinned(true);
            let ls = synaptix_core::device::cuda::loader_stream(ord).map_err(LtxError::from)?;
            // Изолированный weights-пул убрал фрагментацию от вес-карусели —
            // parallel-префетч (overlap H2D с compute: bf16-s2 158.8→148.1s).
            let mut cur = fetch(0)?;
            for i in 0..self.nblocks {
                let lsc = ls.clone();
                let next: Option<AvBlock> = std::thread::scope(|sp| -> Result<Option<AvBlock>, LtxError> {
                    let h = if i + 1 < self.nblocks {
                        Some(sp.spawn(move || -> Result<AvBlock, LtxError> {
                            synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                            synaptix_core::device::cuda::set_offload_pinned(true);
                            let r = fetch(i + 1);
                            let _ = lsc.synchronize();
                            synaptix_core::device::cuda::set_offload_pinned(false);
                            synaptix_core::device::cuda::set_alloc_stream(None);
                            r
                        }))
                    } else {
                        None
                    };
                    let (nvx, nax) = cur.forward(&vx, &ax, &ctx, prof.then_some(&mut tm))?;
                    vx = nvx;
                    ax = nax;
                    let tj = std::time::Instant::now();
                    let r = match h {
                        Some(h) => Ok(Some(h.join().expect("ltx av prefetch thread panicked")?)),
                        None => Ok(None),
                    };
                    if prof { t_stream += tj.elapsed().as_secs_f32(); }
                    r
                })?;
                if let Some(nb) = next {
                    cur = nb;
                }
            }
        } else {
            for i in 0..self.nblocks {
                let (nvx, nax) = self.blocks[i].forward(&vx, &ax, &ctx, prof.then_some(&mut tm))?;
                vx = nvx;
                ax = nax;
            }
        }
        }
        if prof {
            eprintln!(
                "[LTX_PROF]   av blocks: v_attn1={:.2}s v_attn2={:.2}s audio={:.2}s av_ca={:.2}s ff={:.2}s | stream-wait={t_stream:.2}s ({} блоков)",
                tm[0], tm[1], tm[2], tm[3], tm[4], self.nblocks,
            );
        }

        let v_emb = v_emb.to_device(dev)?;
        let v_vel = self.head(&vx, &self.v_sst_out, &v_emb, &self.v_proj_out, 4096)?;
        drop(v_emb);
        drop(vx);
        let a_emb = a_emb.to_device(dev)?;
        let a_vel = self.head(&ax, &self.a_sst_out, &a_emb, &self.a_proj_out, 2048)?;
        Ok((v_vel, a_vel))
    }
}
