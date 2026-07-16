//! FluxTransformer2DModel (MMDiT) — ядро FLUX.1-dev, bit-exact к diffusers.
//!
//! 19 double-stream + 38 single-stream блоков, hidden=3072 (24×128). RoPE
//! axial (axes [16,56,56], theta 1e4) строится на host в f64. guidance-distilled
//! (timestep И guidance домножаются на 1000 перед sinusoidal-эмбеддингом).
//! Все Linear с bias; LayerNorm-ы без affine (eps 1e-6); QK-RMSNorm per-head.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::norm::{layer_norm, rms_norm};

const HEAD_DIM: usize = 128;
const NUM_HEADS: usize = 24;
const INNER: usize = NUM_HEADS * HEAD_DIM; // 3072
const EPS: f32 = 1e-6;
const AXES: [usize; 3] = [16, 56, 56];
const THETA: f64 = 10000.0;

thread_local! {
    static DBG: std::cell::RefCell<Option<Vec<(String, Tensor)>>> = const { std::cell::RefCell::new(None) };
}
fn dbg_push(name: &str, t: &Tensor) {
    DBG.with(|d| {
        if let Some(v) = d.borrow_mut().as_mut() {
            v.push((name.to_string(), t.clone()));
        }
    });
}
/// Тайминг под-операции под [`set_prof`] (synced; serial-distortion ок для split).
fn prof_t<T>(name: &'static str, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !prof_on() {
        return f();
    }
    prof_sync(dev);
    let t = std::time::Instant::now();
    let r = f()?;
    prof_sync(dev);
    prof_add(name, t.elapsed().as_secs_f64());
    Ok(r)
}

/// Включить сбор под-операций блока (для bit-exact-локализации). Между start/take
/// каждый блок дампит свои промежуточные тензоры в общий буфер.
pub fn dbg_start() {
    DBG.with(|d| *d.borrow_mut() = Some(Vec::new()));
}
pub fn dbg_take() -> Vec<(String, Tensor)> {
    DBG.with(|d| d.borrow_mut().take().unwrap_or_default())
}

thread_local! {
    static PROF: std::cell::RefCell<std::collections::BTreeMap<&'static str, f64>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
}
static PROF_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Включить/выключить фазовый профиль forward (диагностика, дефолт ВЫКЛ).
/// Дамп через [`prof_dump`].
pub fn set_prof(on: bool) {
    PROF_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}
fn prof_on() -> bool {
    PROF_ON.load(std::sync::atomic::Ordering::Relaxed)
}
fn prof_sync(dev: &Device) {
    if let Device::Cuda(o) = dev {
        let _ = synaptix_core::device::cuda::synchronize(*o);
    }
}
fn prof_add(k: &'static str, s: f64) {
    PROF.with(|p| *p.borrow_mut().entry(k).or_insert(0.0) += s);
}
/// Дамп+сброс фазовых таймингов forward (см. [`set_prof`]). Зовётся пайплайном.
pub fn prof_dump() {
    if !prof_on() {
        return;
    }
    PROF.with(|p| {
        let mut m = p.borrow_mut();
        let mut v: Vec<_> = m.iter().map(|(k, s)| (*k, *s)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (k, s) in v {
            eprintln!("[FLUX_PROF] {k:<10} {s:.2}s");
        }
        m.clear();
    });
}

#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub num_layers: usize,        // 19 double
    pub num_single_layers: usize, // 38 single
    pub in_channels: usize,       // 64
}

impl FluxConfig {
    pub fn dev() -> Self {
        Self { num_layers: 19, num_single_layers: 38, in_channels: 64 }
    }
}

thread_local! {
    static LOAD_PREC: std::cell::Cell<(DType, DType)> =
        const { std::cell::Cell::new((DType::BF16, DType::BF16)) };
}
/// (quant_dtype, compute) для последующих FluxTransformer::load. quant=BF16/F16/F32
/// → dense; NVFP4/MXFP8 → квантованные веса. Ставится пайплайном перед load.
pub fn set_load_precision(quant: DType, compute: DType) {
    LOAD_PREC.with(|c| c.set((quant, compute)));
}
fn load_prec() -> (DType, DType) {
    LOAD_PREC.with(|c| c.get())
}

fn lin<F: Fn(&str) -> Result<Tensor>>(get: &F, p: &str) -> Result<QuantLinear> {
    let (qd, comp) = load_prec();
    QuantLinear::build(get(&format!("{p}.weight"))?, Some(get(&format!("{p}.bias"))?), qd, comp)
}

/// Копия QuantLinear с весами на `dev` (для layer-streaming dense CPU→GPU).
fn lin_to(l: &QuantLinear, dev: Device) -> Result<QuantLinear> {
    l.to_device(dev)
}

fn t_to(t: &Tensor, dev: Device) -> Result<Tensor> {
    t.to_device(dev)
}

/// sinusoidal timestep-эмбеддинг (dim=256, flip_sin_to_cos=True, shift=0): на
/// выходе cat([cos(emb), sin(emb)]). `t` уже домножен на 1000 и приведён в f32.
fn timestep_embedding(t: &Tensor, dim: usize, device: Device) -> Result<Tensor> {
    let half = dim / 2;
    let ln_max = (10000.0_f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-ln_max * i as f64 / half as f64).exp() as f32)
        .collect();
    let freq = Tensor::from_vec(freqs, (1, half), device)?; // [1,half]
    let b = t.dims()[0];
    let t_f32 = t.to_dtype(DType::F32)?.reshape((b, 1))?;
    let emb = t_f32.broadcast_mul(&freq)?; // [B,half]
    let cos = emb.cos()?;
    let sin = emb.sin()?;
    Tensor::cat(&[&cos, &sin], 1) // flip → cos первым, [B,dim]
}

/// MLP-эмбеддер: Linear → SiLU → Linear (timestep/guidance/text embedders).
struct MlpEmbed {
    l1: QuantLinear,
    l2: QuantLinear,
}
impl MlpEmbed {
    fn load<F: Fn(&str) -> Result<Tensor>>(get: &F, p: &str) -> Result<Self> {
        Ok(Self { l1: lin(get, &format!("{p}.linear_1"))?, l2: lin(get, &format!("{p}.linear_2"))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.silu()?)
    }
    fn to_device(&self, dev: Device) -> Result<Self> {
        Ok(Self { l1: lin_to(&self.l1, dev)?, l2: lin_to(&self.l2, dev)? })
    }
}

/// `[S,128]` cos/sin RoPE для конкат-последовательности ids (txt первым).
fn build_rope(
    txt_seq: usize,
    img_h: usize,
    img_w: usize,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let img_seq = img_h * img_w;
    let seq = txt_seq + img_seq;
    let mut ids: Vec<[f64; 3]> = Vec::with_capacity(seq);
    for _ in 0..txt_seq {
        ids.push([0.0, 0.0, 0.0]);
    }
    for h in 0..img_h {
        for w in 0..img_w {
            ids.push([0.0, h as f64, w as f64]);
        }
    }
    let mut cos = vec![0f32; seq * HEAD_DIM];
    let mut sin = vec![0f32; seq * HEAD_DIM];
    for (s, id) in ids.iter().enumerate() {
        let mut col = 0usize;
        for ax in 0..3 {
            let dim_i = AXES[ax];
            let pos = id[ax];
            for j in 0..dim_i / 2 {
                let freq = 1.0 / THETA.powf((2 * j) as f64 / dim_i as f64);
                let ang = pos * freq;
                let (c, sn) = (ang.cos() as f32, ang.sin() as f32);
                cos[s * HEAD_DIM + col + 2 * j] = c;
                cos[s * HEAD_DIM + col + 2 * j + 1] = c;
                sin[s * HEAD_DIM + col + 2 * j] = sn;
                sin[s * HEAD_DIM + col + 2 * j + 1] = sn;
            }
            col += dim_i;
        }
    }
    let cos_t = Tensor::from_vec(cos, (1, seq, 1, HEAD_DIM), device)?;
    let sin_t = Tensor::from_vec(sin, (1, seq, 1, HEAD_DIM), device)?;
    Ok((cos_t, sin_t))
}

/// apply_rotary_emb (FLUX, use_real_unbind_dim=-1): out = x·cos + rotate(x)·sin,
/// rotate: (-x1,x0,-x3,x2,…). Вычисление в f32. `x:[B,S,H,128]`, cos/sin [1,S,1,128].
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Fused CUDA-ядро (один launch вместо ~10 decomposed elementwise-ops:
    // to_f32+narrow×2+cat+neg+mul×2+add+to_bf16). На CPU/неподдержке → decomposed.
    match x.rope_interleaved_fused(cos, sin) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let d = x.dims();
    let (b, s, h) = (d[0], d[1], d[2]);
    let xf = x.to_dtype(DType::F32)?;
    let pairs = xf.reshape((b, s, h, HEAD_DIM / 2, 2))?;
    let even = pairs.narrow(4, 0, 1)?.contiguous()?; // x_r [.,64,1]
    let odd = pairs.narrow(4, 1, 1)?.contiguous()?; // x_i
    let rot = Tensor::cat(&[&odd.neg()?, &even], 4)?.contiguous()?.reshape((b, s, h, HEAD_DIM))?;
    let out = xf.broadcast_mul(cos)?.add(&rot.broadcast_mul(sin)?)?;
    out.to_dtype(x.dtype())
}

/// SDPA: scale=1/sqrt(128), softmax в f32. q/k/v `[B,S,H,128]` → `[B,S,3072]`.
/// Flash-attention ([B,H,S,D], не материализует [S,S] — критично для VRAM на
/// высоком разрешении); fallback на manual softmax при Unsupported.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let prof = prof_on();
    if prof {
        prof_sync(&q.device()); // дренаж ПРЕДЫДУЩИХ ops ДО старта таймера (иначе врёт)
    }
    let t_attn = std::time::Instant::now();
    let d = q.dims();
    let (b, sq) = (d[0], d[1]);
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    // (bshd-путь [B,S,H,D] без транспоз пробовали — perf-нейтрально: ядро читает
    // K/V strided, что съедает экономию транспоза. Оставлен transpose→[B,H,S,D].)
    let qh = q.transpose(1, 2)?.contiguous()?; // [B,H,S,128]
    let kh = k.transpose(1, 2)?.contiguous()?;
    let vh = v.transpose(1, 2)?.contiguous()?;
    if prof {
        prof_sync(&q.device());
        prof_add("attn_xpose", t_attn.elapsed().as_secs_f64());
    }
    let t_flash = std::time::Instant::now();
    // fallback-путь (bshd Unsupported / CPU): transpose→[B,H,S,D]→flash или
    // manual f32-softmax.
    let flash_res = qh.flash_attention(&kh, &vh, scale as f32, false);
    let out = match flash_res {
        Ok(o) => o, // [B,H,Sq,128]
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
            let (qf, kf, vf) = (qh.to_dtype(DType::F32)?, kh.to_dtype(DType::F32)?, vh.to_dtype(DType::F32)?);
            let kt = kf.transpose(2, 3)?.contiguous()?; // [B,H,128,S]
            let scores = qf.matmul(&kt)?.mul_scalar(scale as f32)?;
            let attn = softmax_dim(&scores, 3)?;
            attn.matmul(&vf)?.to_dtype(q.dtype())?
        }
        Err(e) => return Err(e),
    };
    if prof {
        prof_sync(&q.device());
        prof_add("attn_flash", t_flash.elapsed().as_secs_f64());
    }
    let t_out = std::time::Instant::now();
    let r = out.transpose(1, 2)?.contiguous()?.reshape((b, sq, INNER));
    if prof {
        prof_sync(&q.device());
        prof_add("attn_outx", t_out.elapsed().as_secs_f64());
        prof_add("attn", t_attn.elapsed().as_secs_f64());
    }
    r
}

thread_local! {
    static LN_ONES: std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) };
}
/// Кэш единичного gamma `[INNER]` для fused-layer_norm-без-affine (gamma=1).
fn ln_ones(dtype: DType, dev: Device) -> Result<Tensor> {
    LN_ONES.with(|c| {
        if let Some(t) = c.borrow().as_ref() {
            if t.dtype() == dtype && t.device() == dev {
                return Ok(t.clone());
            }
        }
        let t = Tensor::from_vec(vec![1.0f32; INNER], (INNER,), dev)?.to_dtype(dtype)?;
        *c.borrow_mut() = Some(t.clone());
        Ok(t)
    })
}

/// adaLN: `LN(x)·(1+scale) + shift`. Bit-faithful к Python (округляет LN→bf16
/// ПЕРЕД affine): дорогую layer_norm-редукцию (~6 decomposed-ядер) фьюзим в 1
/// launch через layer_norm_fused с gamma=1 (== LN-без-affine, round в bf16), затем
/// decomposed modulate (как Python: bf16 mul/add). dtype-mismatch → полностью
/// decomposed.
fn ada_ln(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let cdt = scale.dtype();
    if x.dtype() == cdt {
        let ones = ln_ones(cdt, x.device())?;
        let n = layer_norm(x, Some(&ones), None, EPS)?; // 1 fused kernel, round→bf16
        return modulate(&n, scale, shift); // round ПЕРЕД affine — как Python
    }
    modulate(&layer_norm(x, None, None, EPS)?.to_dtype(cdt)?, scale, shift)
}

/// adaLN + (опц.) prequant одним ядром (формат по весу потребителя:
/// NVFP4|MXFP8): (y, Some((packed, scales, fmt))) если fused-путь сработал.
/// Бит-в-бит с ada_ln→to(F16)→quantize (гейт cuda_rms_mod_quant::ln).
fn ada_ln_quant(
    x: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    want: Option<DType>,
) -> Result<(Tensor, Option<(Tensor, Tensor, DType)>)> {
    let cdt = scale.dtype();
    if x.dtype() == cdt
        && matches!(x.device(), Device::Cuda(_))
    {
        let fused = match want {
            Some(DType::NVFP4) => x.ln_mod_quant_nvfp4(scale, shift, EPS).ok(),
            Some(DType::MXFP8) => x.ln_mod_quant_mxfp8(scale, shift, EPS).ok(),
            _ => None,
        };
        if let Some(r) = fused {
            return Ok((r.0, Some((r.1, r.2, want.unwrap()))));
        }
    }
    Ok((ada_ln(x, scale, shift)?, None))
}

/// Проекция с prequant-парой, если она есть и формат веса совпадает с форматом
/// пары; иначе обычный forward.
fn lin_pq(
    lin: &QuantLinear,
    x: &Tensor,
    pq: Option<&(Tensor, Tensor, DType)>,
) -> Result<Tensor> {
    if let Some((p, sc, fmt)) = pq {
        if lin.quant_dtype() == Some(*fmt) {
            let dims = x.dims();
            let (b, t) = (dims[0], dims[1]);
            let y = lin.forward_prequant(p, sc, b * t, x.dtype())?;
            return y.reshape((b, t, y.dims()[1]));
        }
    }
    lin.forward(x)
}

/// `LN(x)·(1+scale) + shift` (scale/shift `[B,3072]` → broadcast по seq).
fn modulate(norm_x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let b = scale.dims()[0];
    let sc = scale.add_scalar(1.0)?.reshape((b, 1, INNER))?;
    let sh = shift.reshape((b, 1, INNER))?;
    norm_x.broadcast_mul(&sc)?.broadcast_add(&sh)
}

fn chunk(t: &Tensor, i: usize, size: usize) -> Result<Tensor> {
    t.narrow(1, i * size, size)?.contiguous()
}

/// residual-сложение с кастом contribution в дтайп accumulator (f32-residual +
/// bf16-compute). no-op при совпадении дтайпов.
fn res_add(acc: &Tensor, contrib: &Tensor) -> Result<Tensor> {
    if acc.dtype() == contrib.dtype() {
        acc.add(contrib)
    } else {
        acc.add(&contrib.to_dtype(acc.dtype())?)
    }
}

struct Attn {
    to_q: QuantLinear,
    to_k: QuantLinear,
    to_v: QuantLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    // double-only:
    add_q: Option<QuantLinear>,
    add_k: Option<QuantLinear>,
    add_v: Option<QuantLinear>,
    norm_aq: Option<Tensor>,
    norm_ak: Option<Tensor>,
    to_out: Option<QuantLinear>,
    to_add_out: Option<QuantLinear>,
}

impl Attn {
    fn to_device(&self, dev: Device) -> Result<Self> {
        let opt_lin = |o: &Option<QuantLinear>| o.as_ref().map(|l| lin_to(l, dev)).transpose();
        let opt_t = |o: &Option<Tensor>| o.as_ref().map(|t| t_to(t, dev)).transpose();
        Ok(Self {
            to_q: lin_to(&self.to_q, dev)?,
            to_k: lin_to(&self.to_k, dev)?,
            to_v: lin_to(&self.to_v, dev)?,
            norm_q: t_to(&self.norm_q, dev)?,
            norm_k: t_to(&self.norm_k, dev)?,
            add_q: opt_lin(&self.add_q)?,
            add_k: opt_lin(&self.add_k)?,
            add_v: opt_lin(&self.add_v)?,
            norm_aq: opt_t(&self.norm_aq)?,
            norm_ak: opt_t(&self.norm_ak)?,
            to_out: opt_lin(&self.to_out)?,
            to_add_out: opt_lin(&self.to_add_out)?,
        })
    }
}

fn to_heads(t: &Tensor) -> Result<Tensor> {
    let d = t.dims();
    t.reshape((d[0], d[1], NUM_HEADS, HEAD_DIM))
}

fn qk_norm(t: &Tensor, w: &Tensor) -> Result<Tensor> {
    rms_norm(t, w, EPS)
}

struct DoubleBlock {
    norm1: QuantLinear,
    norm1_ctx: QuantLinear,
    attn: Attn,
    ff0: QuantLinear,
    ff2: QuantLinear,
    ffc0: QuantLinear,
    ffc2: QuantLinear,
}

impl DoubleBlock {
    fn to_device(&self, dev: Device) -> Result<Self> {
        Ok(Self {
            norm1: lin_to(&self.norm1, dev)?,
            norm1_ctx: lin_to(&self.norm1_ctx, dev)?,
            attn: self.attn.to_device(dev)?,
            ff0: lin_to(&self.ff0, dev)?,
            ff2: lin_to(&self.ff2, dev)?,
            ffc0: lin_to(&self.ffc0, dev)?,
            ffc2: lin_to(&self.ffc2, dev)?,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let st = txt.dims()[1];
        // модуляция
        let m = self.norm1.forward(&temb.silu()?)?; // [B,18432]
        let (sh_msa, sc_msa, g_msa) = (chunk(&m, 0, INNER)?, chunk(&m, 1, INNER)?, chunk(&m, 2, INNER)?);
        let (sh_mlp, sc_mlp, g_mlp) = (chunk(&m, 3, INNER)?, chunk(&m, 4, INNER)?, chunk(&m, 5, INNER)?);
        let cm = self.norm1_ctx.forward(&temb.silu()?)?;
        let (csh_msa, csc_msa, cg_msa) = (chunk(&cm, 0, INNER)?, chunk(&cm, 1, INNER)?, chunk(&cm, 2, INNER)?);
        let (csh_mlp, csc_mlp, cg_mlp) = (chunk(&cm, 3, INNER)?, chunk(&cm, 4, INNER)?, chunk(&cm, 5, INNER)?);

        let _cdt = temb.dtype();
        let (nh, pq_h) = ada_ln_quant(img, &sc_msa, &sh_msa, self.attn.to_q.quant_dtype())?;
        let (ne, pq_e) = ada_ln_quant(
            txt, &csc_msa, &csh_msa,
            self.attn.add_q.as_ref().and_then(|l| l.quant_dtype()),
        )?;
        dbg_push("nh", &nh);
        dbg_push("ne", &ne);

        // проекции + QK-norm (q/k/v шарят prequant nh; add_* — prequant ne)
        let q = qk_norm(&to_heads(&lin_pq(&self.attn.to_q, &nh, pq_h.as_ref())?)?, &self.attn.norm_q)?;
        let k = qk_norm(&to_heads(&lin_pq(&self.attn.to_k, &nh, pq_h.as_ref())?)?, &self.attn.norm_k)?;
        let v = to_heads(&lin_pq(&self.attn.to_v, &nh, pq_h.as_ref())?)?;
        let eq = qk_norm(&to_heads(&lin_pq(self.attn.add_q.as_ref().unwrap(), &ne, pq_e.as_ref())?)?, self.attn.norm_aq.as_ref().unwrap())?;
        let ek = qk_norm(&to_heads(&lin_pq(self.attn.add_k.as_ref().unwrap(), &ne, pq_e.as_ref())?)?, self.attn.norm_ak.as_ref().unwrap())?;
        let ev = to_heads(&lin_pq(self.attn.add_v.as_ref().unwrap(), &ne, pq_e.as_ref())?)?;

        // конкат txt первым + RoPE
        let q = apply_rope(&Tensor::cat(&[&eq, &q], 1)?, cos, sin)?;
        let k = apply_rope(&Tensor::cat(&[&ek, &k], 1)?, cos, sin)?;
        let v = Tensor::cat(&[&ev, &v], 1)?;
        let attn = attention(&q, &k, &v)?; // [B, St+Si, 3072]
        dbg_push("attn_raw", &attn);
        let ctx_attn = attn.narrow(1, 0, st)?.contiguous()?;
        let img_attn = attn.narrow(1, st, attn.dims()[1] - st)?.contiguous()?;
        let img_attn = self.attn.to_out.as_ref().unwrap().forward(&img_attn)?;
        let ctx_attn = self.attn.to_add_out.as_ref().unwrap().forward(&ctx_attn)?;
        dbg_push("img_attn", &img_attn);
        dbg_push("ctx_attn", &ctx_attn);

        // img: attn-residual + ff
        let img = res_add(&img, &g_msa.reshape((g_msa.dims()[0], 1, INNER))?.broadcast_mul(&img_attn)?)?;
        let (n2, pq_f) = ada_ln_quant(&img, &sc_mlp, &sh_mlp, self.ff0.quant_dtype())?;
        let ff = self.ff2.forward(&gelu_tanh(&lin_pq(&self.ff0, &n2, pq_f.as_ref())?)?)?;
        dbg_push("ff", &ff);
        let img = res_add(&img, &g_mlp.reshape((g_mlp.dims()[0], 1, INNER))?.broadcast_mul(&ff)?)?;

        // txt: attn-residual + ff
        let txt = res_add(&txt, &cg_msa.reshape((cg_msa.dims()[0], 1, INNER))?.broadcast_mul(&ctx_attn)?)?;
        let (n2c, pq_fc) = ada_ln_quant(&txt, &csc_mlp, &csh_mlp, self.ffc0.quant_dtype())?;
        let ffc = self.ffc2.forward(&gelu_tanh(&lin_pq(&self.ffc0, &n2c, pq_fc.as_ref())?)?)?;
        let txt = res_add(&txt, &cg_mlp.reshape((cg_mlp.dims()[0], 1, INNER))?.broadcast_mul(&ffc)?)?;

        Ok((img, txt))
    }
}

struct SingleBlock {
    norm: QuantLinear,
    attn: Attn,
    proj_mlp: QuantLinear,
    proj_out: QuantLinear,
}

impl SingleBlock {
    fn to_device(&self, dev: Device) -> Result<Self> {
        Ok(Self {
            norm: lin_to(&self.norm, dev)?,
            attn: self.attn.to_device(dev)?,
            proj_mlp: lin_to(&self.proj_mlp, dev)?,
            proj_out: lin_to(&self.proj_out, dev)?,
        })
    }

    fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
        let dev = img.device();
        let st = txt.dims()[1];
        let hidden = Tensor::cat(&[txt, img], 1)?; // [B, St+Si, 3072]
        let residual = hidden.clone();
        let m = self.norm.forward(&temb.silu()?)?; // [B,9216]
        let (sh, sc, gate) = (chunk(&m, 0, INNER)?, chunk(&m, 1, INNER)?, chunk(&m, 2, INNER)?);
        let (nh, pq) = prof_t("s_norm", &dev, || {
            ada_ln_quant(&hidden, &sc, &sh, self.attn.to_q.quant_dtype())
        })?;

        let mlp = prof_t("s_mlp", &dev, || gelu_tanh(&lin_pq(&self.proj_mlp, &nh, pq.as_ref())?))?; // [B,seq,12288]
        let qg = prof_t("s_gemm", &dev, || to_heads(&lin_pq(&self.attn.to_q, &nh, pq.as_ref())?))?;
        let qn = prof_t("s_qknorm", &dev, || qk_norm(&qg, &self.attn.norm_q))?;
        let q = prof_t("s_rope", &dev, || apply_rope(&qn, cos, sin))?;
        let kg = prof_t("s_gemm", &dev, || to_heads(&lin_pq(&self.attn.to_k, &nh, pq.as_ref())?))?;
        let kn = prof_t("s_qknorm", &dev, || qk_norm(&kg, &self.attn.norm_k))?;
        let k = prof_t("s_rope", &dev, || apply_rope(&kn, cos, sin))?;
        let v = prof_t("s_gemm", &dev, || to_heads(&lin_pq(&self.attn.to_v, &nh, pq.as_ref())?))?;
        let attn = attention(&q, &k, &v)?; // [B,seq,3072]

        let cat = Tensor::cat(&[&attn, &mlp], 2)?; // [B,seq,15360]
        let b = gate.dims()[0];
        let proj = prof_t("s_projout", &dev, || Ok(gate.reshape((b, 1, INNER))?.broadcast_mul(&self.proj_out.forward(&cat)?)?))?;
        let hidden = res_add(&residual, &proj)?;
        let txt_out = hidden.narrow(1, 0, st)?.contiguous()?;
        let img_out = hidden.narrow(1, st, hidden.dims()[1] - st)?.contiguous()?;
        Ok((img_out, txt_out))
    }
}

pub struct FluxTransformer {
    x_embedder: QuantLinear,
    context_embedder: QuantLinear,
    ts_embed: MlpEmbed,
    guid_embed: MlpEmbed,
    text_embed: MlpEmbed,
    blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out: QuantLinear,
    proj_out: QuantLinear,
    /// layer-streaming: блоки [resident_*..] на CPU, каждый перед forward
    /// копируется на этот device и освобождается после. Блоки [..resident_*]
    /// РЕЗИДЕНТНЫ на GPU (частичный offload — используем доступную VRAM). `None` =
    /// всё резидентно. Полный стриминг = Some + resident_*=0.
    stream: Option<Device>,
    resident_double: usize,
    resident_single: usize,
}

impl FluxTransformer {
    pub fn load<F>(cfg: &FluxConfig, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let x_embedder = lin(get, "x_embedder")?;
        let context_embedder = lin(get, "context_embedder")?;
        let ts_embed = MlpEmbed::load(get, "time_text_embed.timestep_embedder")?;
        let guid_embed = MlpEmbed::load(get, "time_text_embed.guidance_embedder")?;
        let text_embed = MlpEmbed::load(get, "time_text_embed.text_embedder")?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            let a = &format!("{p}.attn");
            blocks.push(DoubleBlock {
                norm1: lin(get, &format!("{p}.norm1.linear"))?,
                norm1_ctx: lin(get, &format!("{p}.norm1_context.linear"))?,
                attn: Attn {
                    to_q: lin(get, &format!("{a}.to_q"))?,
                    to_k: lin(get, &format!("{a}.to_k"))?,
                    to_v: lin(get, &format!("{a}.to_v"))?,
                    norm_q: get(&format!("{a}.norm_q.weight"))?,
                    norm_k: get(&format!("{a}.norm_k.weight"))?,
                    add_q: Some(lin(get, &format!("{a}.add_q_proj"))?),
                    add_k: Some(lin(get, &format!("{a}.add_k_proj"))?),
                    add_v: Some(lin(get, &format!("{a}.add_v_proj"))?),
                    norm_aq: Some(get(&format!("{a}.norm_added_q.weight"))?),
                    norm_ak: Some(get(&format!("{a}.norm_added_k.weight"))?),
                    to_out: Some(lin(get, &format!("{a}.to_out.0"))?),
                    to_add_out: Some(lin(get, &format!("{a}.to_add_out"))?),
                },
                ff0: lin(get, &format!("{p}.ff.net.0.proj"))?,
                ff2: lin(get, &format!("{p}.ff.net.2"))?,
                ffc0: lin(get, &format!("{p}.ff_context.net.0.proj"))?,
                ffc2: lin(get, &format!("{p}.ff_context.net.2"))?,
            });
        }

        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            let p = format!("single_transformer_blocks.{i}");
            let a = &format!("{p}.attn");
            single_blocks.push(SingleBlock {
                norm: lin(get, &format!("{p}.norm.linear"))?,
                attn: Attn {
                    to_q: lin(get, &format!("{a}.to_q"))?,
                    to_k: lin(get, &format!("{a}.to_k"))?,
                    to_v: lin(get, &format!("{a}.to_v"))?,
                    norm_q: get(&format!("{a}.norm_q.weight"))?,
                    norm_k: get(&format!("{a}.norm_k.weight"))?,
                    add_q: None, add_k: None, add_v: None,
                    norm_aq: None, norm_ak: None, to_out: None, to_add_out: None,
                },
                proj_mlp: lin(get, &format!("{p}.proj_mlp"))?,
                proj_out: lin(get, &format!("{p}.proj_out"))?,
            });
        }

        let norm_out = lin(get, "norm_out.linear")?;
        let proj_out = lin(get, "proj_out")?;
        Ok(Self {
            x_embedder, context_embedder, ts_embed, guid_embed, text_embed,
            blocks, single_blocks, norm_out, proj_out, stream: None,
            resident_double: 0, resident_single: 0,
        })
    }

    /// Включить layer-streaming: мелкие части (эмбеддеры, norm_out, proj_out)
    /// переезжают на `dev` (GPU), 57 блоков остаются на CPU и стримятся per-block
    /// в forward. Вызывать после `load` с весами на CPU. Пик VRAM ≈ активации +
    /// 1 блок (~1GB) → 1024²/2048² влезают в скромный VRAM (как ComfyUI).
    pub fn into_streaming(mut self, dev: Device) -> Result<Self> {
        self.x_embedder = lin_to(&self.x_embedder, dev)?;
        self.context_embedder = lin_to(&self.context_embedder, dev)?;
        self.ts_embed = self.ts_embed.to_device(dev)?;
        self.guid_embed = self.guid_embed.to_device(dev)?;
        self.text_embed = self.text_embed.to_device(dev)?;
        self.norm_out = lin_to(&self.norm_out, dev)?;
        self.proj_out = lin_to(&self.proj_out, dev)?;
        self.stream = Some(dev);
        self.resident_double = 0;
        self.resident_single = 0;
        Ok(self)
    }

    /// Частичный offload: мелкие части + СТОЛЬКО блоков, сколько влезает в
    /// доступную VRAM (двигаем блоки на GPU, пока free > `min_free_bytes`),
    /// остальные стримятся per-block. Использует ~всю свободную память вместо
    /// бинарного «не влезло целиком → всё на CPU». Вызывать после load на CPU.
    pub fn into_partial_streaming(mut self, dev: Device, min_free_bytes: u64) -> Result<Self> {
        self.x_embedder = lin_to(&self.x_embedder, dev)?;
        self.context_embedder = lin_to(&self.context_embedder, dev)?;
        self.ts_embed = self.ts_embed.to_device(dev)?;
        self.guid_embed = self.guid_embed.to_device(dev)?;
        self.text_embed = self.text_embed.to_device(dev)?;
        self.norm_out = lin_to(&self.norm_out, dev)?;
        self.proj_out = lin_to(&self.proj_out, dev)?;
        let ord = if let Device::Cuda(o) = dev { o } else { 0 };
        let free_ok = || {
            synaptix_core::device::cuda::mem_info(ord)
                .map(|(f, _)| f as u64 > min_free_bytes)
                .unwrap_or(false)
        };
        let mut rd = 0;
        for i in 0..self.blocks.len() {
            if !free_ok() {
                break;
            }
            self.blocks[i] = self.blocks[i].to_device(dev)?;
            rd += 1;
        }
        let mut rs = 0;
        for i in 0..self.single_blocks.len() {
            if !free_ok() {
                break;
            }
            self.single_blocks[i] = self.single_blocks[i].to_device(dev)?;
            rs += 1;
        }
        self.stream = Some(dev);
        self.resident_double = rd;
        self.resident_single = rs;
        eprintln!(
            "[FLUX] partial offload: резидентно {rd}/{} double + {rs}/{} single блоков, остальные streaming",
            self.blocks.len(), self.single_blocks.len()
        );
        Ok(self)
    }

    /// `hidden_states[B,img_seq,64]`, `encoder_hidden_states[B,txt_seq,4096]`,
    /// `pooled[B,768]`, `timestep[B]`, `guidance[B]`, сетка img (`img_h`×`img_w`).
    /// → `[B, img_seq, 64]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        pooled: &Tensor,
        timestep: &Tensor,
        guidance: &Tensor,
        img_h: usize,
        img_w: usize,
    ) -> Result<Tensor> {
        self.forward_cap(hidden_states, encoder_hidden_states, pooled, timestep, guidance, img_h, img_w, &mut None)
    }

    /// Как `forward`, но при `Some(cap)` складывает промежуточные тензоры
    /// (для bit-exact-локализации против diffusers-hooks).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cap(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        pooled: &Tensor,
        timestep: &Tensor,
        guidance: &Tensor,
        img_h: usize,
        img_w: usize,
        cap: &mut Option<Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        macro_rules! grab {
            ($name:expr, $t:expr) => {
                if let Some(v) = cap.as_mut() {
                    v.push(($name.to_string(), $t.clone()));
                }
            };
        }
        let dt = hidden_states.dtype();
        let dev = hidden_states.device();
        let txt_seq = encoder_hidden_states.dims()[1];

        let mut img = self.x_embedder.forward(hidden_states)?;
        let mut txt = self.context_embedder.forward(encoder_hidden_states)?;
        grab!("x_emb", img);
        grab!("ctx_emb", txt);

        // temb (timestep И guidance ×1000)
        let t1000 = timestep.to_dtype(dt)?.mul_scalar(1000.0)?;
        let g1000 = guidance.to_dtype(dt)?.mul_scalar(1000.0)?;
        let ts_proj = timestep_embedding(&t1000, 256, dev)?.to_dtype(dt)?;
        let g_proj = timestep_embedding(&g1000, 256, dev)?.to_dtype(dt)?;
        let temb = self
            .ts_embed
            .forward(&ts_proj)?
            .add(&self.guid_embed.forward(&g_proj)?)?
            .add(&self.text_embed.forward(pooled)?)?; // [B,3072]
        grab!("temb", temb);

        let (cos, sin) = build_rope(txt_seq, img_h, img_w, dev)?;

        let prof = prof_on();
        if prof {
            prof_sync(&dev);
        }
        let t_dbl = std::time::Instant::now();
        for (i, blk) in self.blocks.iter().enumerate() {
            let probe = i == 0 || i == 14;
            if cap.is_some() && probe {
                grab!(format!("db{i}in_img"), img);
                grab!(format!("db{i}in_txt"), txt);
                dbg_start();
            }
            let (i2, t2) = match self.stream {
                Some(d) if i >= self.resident_double => {
                    blk.to_device(d)?.forward(&img, &txt, &temb, &cos, &sin)? // streamed
                }
                _ => blk.forward(&img, &txt, &temb, &cos, &sin)?, // резидент (или stream=None)
            };
            img = i2;
            txt = t2;
            if probe {
                for (n, t) in dbg_take() {
                    grab!(format!("db{i}sub_{n}"), t);
                }
                grab!(format!("db{i}_img"), img);
                grab!(format!("db{i}_txt"), txt);
            }
            if i == 9 || i == 18 {
                grab!(format!("depthD{i}_img"), img);
            }
        }
        if prof {
            prof_sync(&dev);
            prof_add("double", t_dbl.elapsed().as_secs_f64());
        }
        let t_sgl = std::time::Instant::now();
        for (i, blk) in self.single_blocks.iter().enumerate() {
            let (i2, t2) = match self.stream {
                Some(d) if i >= self.resident_single => {
                    blk.to_device(d)?.forward(&img, &txt, &temb, &cos, &sin)? // streamed
                }
                _ => blk.forward(&img, &txt, &temb, &cos, &sin)?, // резидент (или stream=None)
            };
            img = i2;
            txt = t2;
            if i == 0 {
                grab!("sb0_img", img);
                grab!("sb0_txt", txt);
            }
            if i == 9 || i == 18 || i == 37 {
                grab!(format!("depthS{i}_img"), img);
            }
        }
        if prof {
            prof_sync(&dev);
            prof_add("single", t_sgl.elapsed().as_secs_f64());
        }

        // norm_out (AdaLayerNormContinuous: scale ПЕРВЫМ) + proj_out
        let m = self.norm_out.forward(&temb.silu()?)?; // [B,6144]
        let scale = chunk(&m, 0, INNER)?;
        let shift = chunk(&m, 1, INNER)?;
        let out = ada_ln(&img, &scale, &shift)?;
        self.proj_out.forward(&out)
    }

    /// Изолированный прогон double-блока `idx` на заданном входе (для bit-exact-
    /// бисекции CUDA-бага: скармливаем чистый Python-вход → расхождение под-
    /// операций = ЧИСТАЯ ошибка CUDA-ядра этого блока, без унаследованной).
    /// Возвращает захваченные под-операции + out_img/out_txt.
    pub fn dbg_double_block(
        &self,
        idx: usize,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        img_h: usize,
        img_w: usize,
    ) -> Result<Vec<(String, Tensor)>> {
        let txt_seq = txt.dims()[1];
        let (cos, sin) = build_rope(txt_seq, img_h, img_w, img.device())?;
        let blk = &self.blocks[idx];
        dbg_start();
        let (i2, t2) = match self.stream {
            Some(d) => blk.to_device(d)?.forward(img, txt, temb, &cos, &sin)?,
            None => blk.forward(img, txt, temb, &cos, &sin)?,
        };
        let mut out = dbg_take();
        out.push(("out_img".into(), i2));
        out.push(("out_txt".into(), t2));
        Ok(out)
    }
}
