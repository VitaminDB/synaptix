//! Слой декодера на обучении: прямой проход с сохранением всего, что нужно
//! обратному, и обратный проход по входу слоя с накоплением градиентов LoRA.
//!
//! Смешивание токенов — полное внимание (flash-ядро) или Gated DeltaNet
//! (рекуррентное ядро с контрольными точками состояния). Активации прямого
//! хода — в compute-типе модели (как у инференса), градиенты — в F32.
//! Последовательность одна (`[T, H]`), несколько примеров в ней разделены
//! номерами `seg`.

use synaptix_core::backend::train::TrainOp;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::train_access::{AttnRefs, BlockRefs, FfnRefs, LinearRefs, MixerRefs, MlpRefs};
use synaptix_llm_common::{Activation, QLinear};
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::error::other;
use crate::linear::Weights;
use crate::lora::LoraLayer;
use crate::Result;

/// Адаптеры одного слоя (`None` — проекция не учится) и постоянные слоя на
/// карте.
#[derive(Default)]
pub struct BlockLora {
    pub q: Option<LoraLayer>,
    pub k: Option<LoraLayer>,
    pub v: Option<LoraLayer>,
    pub o: Option<LoraLayer>,
    pub qkv: Option<LoraLayer>,
    pub z: Option<LoraLayer>,
    pub a: Option<LoraLayer>,
    pub b: Option<LoraLayer>,
    pub out: Option<LoraLayer>,
    pub gate: Option<LoraLayer>,
    pub up: Option<LoraLayer>,
    pub down: Option<LoraLayer>,
    pub lin: Option<LinConst>,
}

/// Постоянные GDN-слоя на карте (F32).
pub struct LinConst {
    /// `[conv_dim, conv_k]`.
    pub conv_w: Tensor,
    /// `−exp(A_log)` `[Hv]`.
    pub neg_a: Tensor,
    pub dt_bias: Tensor,
}

impl LinConst {
    pub fn new(l: &LinearRefs<'_>, dev: Device) -> Result<Self> {
        Ok(Self {
            conv_w: Tensor::from_vec(l.conv_w.to_vec(), vec![l.conv_dim, l.conv_k], dev)?,
            neg_a: Tensor::from_vec(l.a_log.iter().map(|a| -a.exp()).collect::<Vec<f32>>(), vec![l.num_v_heads], dev)?,
            dt_bias: Tensor::from_vec(l.dt_bias.to_vec(), vec![l.num_v_heads], dev)?,
        })
    }
}

impl BlockLora {
    fn all(&self) -> [&Option<LoraLayer>; 12] {
        [&self.q, &self.k, &self.v, &self.o, &self.qkv, &self.z, &self.a, &self.b, &self.out, &self.gate, &self.up, &self.down]
    }

    pub fn layers(&self) -> impl Iterator<Item = &LoraLayer> {
        self.all().into_iter().flatten()
    }

    pub fn layers_mut(&mut self) -> impl Iterator<Item = &mut LoraLayer> {
        [
            &mut self.q, &mut self.k, &mut self.v, &mut self.o, &mut self.qkv, &mut self.z, &mut self.a, &mut self.b,
            &mut self.out, &mut self.gate, &mut self.up, &mut self.down,
        ]
        .into_iter()
        .flatten()
    }
}

/// Общее для всех слоёв на одной последовательности.
pub struct SeqCtx {
    pub t: usize,
    /// Позиции RoPE `[T]` U32 (у каждого упакованного примера — с нуля).
    pub positions: Tensor,
    /// Номер примера `[T]` I32; `None` — один пример.
    pub seg: Option<Tensor>,
    /// Таблицы RoPE по позициям — общие для слоёв с одним кэшем
    /// (ключ — адрес кэша и число поворачиваемых каналов).
    pub rope_tabs: std::cell::RefCell<Vec<(usize, usize, Tensor, Tensor)>>,
}

impl SeqCtx {
    pub fn new(t: usize, positions: Tensor, seg: Option<Tensor>) -> Self {
        Self { t, positions, seg, rope_tabs: std::cell::RefCell::new(Vec::new()) }
    }
}

/// Что сохраняет ветвь полного внимания.
pub struct FullSaved {
    zq: Option<Tensor>,
    zk: Option<Tensor>,
    zv: Option<Tensor>,
    q_raw: Tensor,
    gate_raw: Option<Tensor>,
    k_raw: Tensor,
    core: AttnCore,
    /// Выход внимания до гейта `[T, nh, hd]` F32 — только при гейте.
    o: Option<Tensor>,
    attn_in: Option<Tensor>,
    zo: Option<Tensor>,
    cos: Tensor,
    sin: Tensor,
}

/// Что сохраняет ветвь Gated DeltaNet.
pub struct LinSaved {
    zqkv: Option<Tensor>,
    zz: Option<Tensor>,
    za: Option<Tensor>,
    zb: Option<Tensor>,
    /// Свёртка до silu `[T, conv_dim]` F32.
    pre: Tensor,
    /// q, k до L2-нормы `[T, Hk, dk]` F32 и их обратные нормы `[T, Hk, 1]`.
    qf: Tensor,
    kf: Tensor,
    rq: Tensor,
    rk: Tensor,
    qn: Tensor,
    kn: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    /// `a + dt_bias` `[T, Hv]` F32.
    a_dt: Tensor,
    ckpt: Tensor,
    /// Выход рекуррентности `[T, Hv, dv]` F32 и нормированный выход.
    o: Tensor,
    on: Tensor,
    z: Tensor,
    out_in: Option<Tensor>,
    zout: Option<Tensor>,
}

pub enum MixSaved {
    Full(FullSaved),
    Linear(LinSaved),
}

/// Всё, что прямой проход слоя оставляет обратному.
pub struct Saved {
    x: Tensor,
    n1: Tensor,
    mix: MixSaved,
    seg: Option<Tensor>,
    a: Option<Tensor>,
    h: Tensor,
    ffn: FfnSaved,
    d: Option<Tensor>,
}

/// Что сохраняет FFN.
enum FfnSaved {
    Dense {
        n2: Option<Tensor>,
        zg: Option<Tensor>,
        zu: Option<Tensor>,
        g: Tensor,
        u: Tensor,
        m: Option<Tensor>,
        zd: Option<Tensor>,
    },
    Moe(crate::moe::MoeSaved),
}

fn f32(t: &Tensor) -> Result<Tensor> {
    Ok(if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32)? })
}

fn same_dtype(w: &Tensor, as_: &Tensor) -> Result<Tensor> {
    Ok(if w.dtype() == as_.dtype() { w.clone() } else { w.to_dtype(as_.dtype())? })
}

/// Обратный RMSNorm по последней оси: `x` — сохранённый вход нормы.
pub fn rms_bwd(x: &Tensor, w: &Tensor, dy: &Tensor, eps: f32) -> Result<Tensor> {
    let w = same_dtype(w, x)?;
    let dy = f32(dy)?;
    crate::prof::prof(x.device(), "norm_bwd", || -> Result<Tensor> {
        Ok(Tensor::train_op(TrainOp::RmsNormBwd { eps }, &[x, &w, &dy], &[(x.dims().to_vec(), DType::F32)])?.remove(0))
    })
}

/// Проекция с адаптером: `(y в compute-типе x, z адаптера)`.
pub(crate) fn proj(w: &Weights, q: &QLinear, lora: Option<&LoraLayer>, x: &Tensor, bias: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
    let mut y = w.forward(q, x)?;
    if let Some(b) = bias {
        y = y.broadcast_add(&same_dtype(b, &y)?)?;
    }
    match lora {
        Some(l) => {
            let (z, y) = crate::prof::prof(x.device(), "lora_fwd", || l.forward_add(x, &y))?;
            Ok((y, Some(z)))
        }
        None => Ok((y, None)),
    }
}

/// `dx` проекции: замороженный вес плюс вклад адаптера (градиенты адаптера
/// копятся в нём).
pub(crate) fn proj_bwd(w: &Weights, q: &QLinear, lora: Option<&mut LoraLayer>, x32: Option<&Tensor>, z: Option<&Tensor>, dy: &Tensor) -> Result<Tensor> {
    let mut dx = w.backward_input(q, dy)?;
    if let Some(l) = lora {
        let (x32, z) = x32.zip(z).ok_or_else(|| other("proj_bwd: нет сохранённых входов LoRA"))?;
        dx = crate::prof::prof(dy.device(), "lora_bwd", || l.backward_add(x32, z, dy, &dx))?;
    }
    Ok(dx)
}

/// RoPE над `[heads, T, hd]` F32 (Split-раскладка, частичный поворот —
/// первые `rot` каналов). `inverse` — поворот назад (обратный проход:
/// матрица поворота ортогональна, градиент поворачивается на −θ).
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rot: usize, inverse: bool) -> Result<Tensor> {
    let (h, t, hd) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    if rot == 0 {
        return Ok(x.clone());
    }
    let sin = if inverse { sin.neg()? } else { sin.clone() };
    let x4 = x.reshape(vec![1, h, t, hd])?;
    let out = if rot == hd {
        apply_rope_with_cossin(&x4, cos, &sin, RopeLayout::Split)?
    } else {
        let xr = x4.narrow(3, 0, rot)?.contiguous()?;
        let xp = x4.narrow(3, rot, hd - rot)?.contiguous()?;
        let r = apply_rope_with_cossin(&xr, cos, &sin, RopeLayout::Split)?;
        Tensor::cat(&[&r, &xp], 3)?
    };
    Ok(out.reshape(vec![h, t, hd])?)
}

/// `[n_kv·group, T, hd]` → сумма по группе `[n_kv, T, hd]`.
fn sum_groups(x: &Tensor, nkv: usize, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (t, hd) = (x.dims()[1], x.dims()[2]);
    Ok(x.reshape(vec![nkv, group, t * hd])?.sum_keepdim(1)?.reshape(vec![nkv, t, hd])?)
}

fn rope_tables(attn: &AttnRefs<'_>, ctx: &SeqCtx) -> Result<(Tensor, Tensor)> {
    let key = (attn.rope as *const _ as usize, attn.rotary_dim);
    if let Some((_, _, c, s)) = ctx.rope_tabs.borrow().iter().find(|e| (e.0, e.1) == key) {
        return Ok((c.clone(), s.clone()));
    }
    let (c, s) = rope_tables_build(attn, ctx)?;
    ctx.rope_tabs.borrow_mut().push((key.0, key.1, c.clone(), s.clone()));
    Ok((c, s))
}

fn rope_tables_build(attn: &AttnRefs<'_>, ctx: &SeqCtx) -> Result<(Tensor, Tensor)> {
    let rot = attn.rotary_dim;
    let cos = attn.rope.cos().index_select(0, &ctx.positions)?;
    let sin = attn.rope.sin().index_select(0, &ctx.positions)?;
    // Таблицы кэша — на голову целиком; частичному повороту нужны первые rot/2.
    let half = cos.dims()[1];
    if rot / 2 != half {
        if rot / 2 > half {
            return Err(other(format!("RoPE: таблица на {half} частот, а поворачивается {rot} каналов")));
        }
        return Ok((f32(&cos.narrow(1, 0, rot / 2)?.contiguous()?)?, f32(&sin.narrow(1, 0, rot / 2)?.contiguous()?)?));
    }
    Ok((f32(&cos)?, f32(&sin)?))
}

/// Что ядро внимания оставляет обратному проходу.
enum AttnCore {
    /// Flash-ядро: входы в BF16, выход и log-sum-exp строк.
    Flash { q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor },
    /// Явная матрица вероятностей `[nh, T, T]` (размеры голов без flash-ядра).
    Naive { q: Tensor, k: Tensor, v: Tensor, p: Tensor },
}

fn flash_ok(hd: usize) -> bool {
    matches!(hd, 64 | 128 | 256) && std::env::var("SYN_TRAIN_FLASH").as_deref() != Ok("0")
}


/// `q` — `[nh, T, hd]` F32, `k`/`v` — `[nkv, T, hd]`. Выход `[T, nh, hd]` F32.
fn attn_forward(q: Tensor, k: Tensor, v: Tensor, group: usize, scale: f32, ctx: &SeqCtx) -> Result<(AttnCore, Tensor)> {
    let (nh, t, hd) = (q.dims()[0], q.dims()[1], q.dims()[2]);
    if flash_ok(hd) {
        let (q, k, v) = (q.to_dtype(DType::BF16)?, k.to_dtype(DType::BF16)?, v.to_dtype(DType::BF16)?);
        let has_seg = ctx.seg.is_some();
        let mut ins = vec![&q, &k, &v];
        if let Some(sg) = &ctx.seg {
            ins.push(sg);
        }
        let mut out = Tensor::train_op(TrainOp::FlashFwd { scale, has_seg }, &ins, &[(vec![nh, t, hd], DType::BF16), (vec![nh, t], DType::F32)])?;
        let lse = out.pop().unwrap();
        let o = out.pop().unwrap();
        let ot = f32(&o)?.permute(vec![1, 0, 2])?.contiguous()?;
        return Ok((AttnCore::Flash { q, k, v, o, lse }, ot));
    }
    let v = f32(&v)?;
    let krep = if group > 1 { k.repeat_interleave(0, group)? } else { k.clone() };
    let vrep = if group > 1 { v.repeat_interleave(0, group)? } else { v.clone() };
    let scores = q.matmul(&krep.t()?)?;
    let dims = scores.dims().to_vec();
    let p = match &ctx.seg {
        Some(seg) => Tensor::train_op(TrainOp::CausalSoftmax { scale, has_seg: true }, &[&scores, seg], &[(dims, DType::F32)])?,
        None => Tensor::train_op(TrainOp::CausalSoftmax { scale, has_seg: false }, &[&scores], &[(dims, DType::F32)])?,
    }
    .remove(0);
    drop(scores);
    let o = p.matmul(&vrep)?.permute(vec![1, 0, 2])?.contiguous()?;
    Ok((AttnCore::Naive { q, k, v, p }, o))
}

/// `d_o` — `[T, nh, hd]` F32. Результат — `(dq [nh,T,hd], dk, dv [nkv,T,hd])` F32.
fn attn_backward(core: AttnCore, d_o: &Tensor, nkv: usize, group: usize, scale: f32, seg: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
    let doh = d_o.permute(vec![1, 0, 2])?.contiguous()?; // [nh, T, hd]
    match core {
        AttnCore::Flash { q, k, v, o, lse } => {
            let (nh, t, hd) = (q.dims()[0], q.dims()[1], q.dims()[2]);
            let delta = doh.mul(&f32(&o)?)?.sum_keepdim(2)?.reshape(vec![nh, t])?;
            let do16 = doh.to_dtype(DType::BF16)?;
            let mut ins = vec![&q, &k, &v, &do16, &lse, &delta];
            if let Some(sg) = seg {
                ins.push(sg);
            }
            let tp = t.div_ceil(32) * 32;
            let mut g = Tensor::train_op(
                TrainOp::FlashBwd { scale, has_seg: seg.is_some() },
                &ins,
                &[(vec![nh, t, hd], DType::F32), (vec![nkv, tp, hd], DType::F32), (vec![nkv, tp, hd], DType::F32)],
            )?;
            let dv = g.pop().unwrap();
            let dk = g.pop().unwrap();
            let dq = g.pop().unwrap();
            let cut = |x: Tensor| -> Result<Tensor> { Ok(if tp == t { x } else { x.narrow(1, 0, t)?.contiguous()? }) };
            Ok((dq, cut(dk)?, cut(dv)?))
        }
        AttnCore::Naive { q, k, v, p } => {
            let krep = if group > 1 { k.repeat_interleave(0, group)? } else { k.clone() };
            let vrep = if group > 1 { v.repeat_interleave(0, group)? } else { v.clone() };
            let dp = doh.matmul(&vrep.t()?)?;
            let dvrep = p.t()?.matmul(&doh)?;
            drop(vrep);
            let pdims = p.dims().to_vec();
            let ds = Tensor::train_op(TrainOp::SoftmaxBwd { scale }, &[&p, &dp], &[(pdims, DType::F32)])?.remove(0);
            drop(dp);
            let dq = ds.matmul(&krep)?;
            let dkrep = ds.t()?.matmul(&q)?;
            Ok((dq, sum_groups(&dkrep, nkv, group)?, sum_groups(&dvrep, nkv, group)?))
        }
    }
}

/// Прямой проход слоя. `x` — `[T, H]` в compute-типе.
pub fn forward(b: &BlockRefs<'_>, lora: &BlockLora, x: &Tensor, ctx: &SeqCtx, w: &Weights) -> Result<(Tensor, Saved)> {
    if let FfnRefs::Dense(m) = &b.mlp {
        if !matches!(m.activation, Activation::Silu) {
            return Err(other("обучение: пока только SwiGLU-MLP"));
        }
    }
    let dev = x.device();
    let n1 = crate::prof::prof(dev, "norm_fwd", || rms_norm(x, b.pre_attn_norm, b.rms_eps))?;
    let (a, mix) = match &b.mixer {
        MixerRefs::Full(at) => {
            let (a, s) = full_fwd(at, lora, &n1, ctx, w)?;
            (a, MixSaved::Full(s))
        }
        MixerRefs::Linear(la) => {
            let (a, s) = crate::prof::prof(dev, "gdn_fwd", || lin_fwd(la, lora, &n1, ctx, w))?;
            (a, MixSaved::Linear(s))
        }
    };
    let a2 = match b.post_attn_norm {
        Some(w) => rms_norm(&a, w, b.post_eps)?,
        None => a.clone(),
    };
    let h = crate::prof::prof(dev, "resid_fwd", || -> Result<Tensor> { Ok(x.add(&same_dtype(&a2, x)?)?) })?;

    // ── MLP ──
    let n2 = crate::prof::prof(dev, "norm_fwd", || rms_norm(&h, b.pre_mlp_norm, b.rms_eps))?;
    let (d, ffn) = match &b.mlp {
        FfnRefs::Dense(mlp) => dense_fwd(mlp, lora, n2, w)?,
        FfnRefs::Moe(moe) => {
            let (d, s) = crate::prof::prof(dev, "moe_fwd", || crate::moe::forward(moe, lora, &n2, w))?;
            (d, FfnSaved::Moe(s))
        }
    };
    let d2 = match b.post_mlp_norm {
        Some(w) => rms_norm(&d, w, b.post_eps)?,
        None => d.clone(),
    };
    let out = crate::prof::prof(dev, "resid_fwd", || -> Result<Tensor> { Ok(h.add(&same_dtype(&d2, &h)?)?) })?;

    let saved = Saved {
        x: x.clone(),
        n1,
        mix,
        seg: ctx.seg.clone(),
        a: b.post_attn_norm.map(|_| a),
        h,
        ffn,
        d: b.post_mlp_norm.map(|_| d),
    };
    Ok((out, saved))
}

fn dense_fwd(mlp: &MlpRefs<'_>, lora: &BlockLora, n2: Tensor, w: &Weights) -> Result<(Tensor, FfnSaved)> {
    let dev = n2.device();
    let (g, zg) = proj(w, mlp.gate_proj, lora.gate.as_ref(), &n2, None)?;
    let (u, zu) = proj(w, mlp.up_proj, lora.up.as_ref(), &n2, None)?;
    let m = crate::prof::prof(dev, "act_fwd", || g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|s| s.mul(&u))))?;
    let (d, zd) = proj(w, mlp.down_proj, lora.down.as_ref(), &m, None)?;
    let saved = FfnSaved::Dense {
        n2: (lora.gate.is_some() || lora.up.is_some()).then_some(n2),
        zg,
        zu,
        g,
        u,
        m: lora.down.is_some().then_some(m),
        zd,
    };
    Ok((d, saved))
}

/// Обратный проход слоя: `dout` — `[T, H]` F32, результат — `dx` F32.
pub fn backward(b: &BlockRefs<'_>, lora: &mut BlockLora, s: Saved, dout: &Tensor, w: &Weights) -> Result<Tensor> {
    // ── MLP ──
    let dd = match (b.post_mlp_norm, &s.d) {
        (Some(w), Some(d)) => rms_bwd(d, w, dout, b.post_eps)?,
        _ => dout.clone(),
    };
    let dn2 = match (&b.mlp, s.ffn) {
        (FfnRefs::Dense(mlp), FfnSaved::Dense { n2, zg, zu, g, u, m, zd }) => {
            let dm = proj_bwd(w, mlp.down_proj, lora.down.as_mut(), m.as_ref(), zd.as_ref(), &dd)?;
            let shp = g.dims().to_vec();
            let gu = Tensor::train_op(TrainOp::SiluMulBwd, &[&g, &u, &dm], &[(shp.clone(), DType::F32), (shp, DType::F32)])?;
            let (dg, du) = (&gu[0], &gu[1]);
            proj_bwd(w, mlp.gate_proj, lora.gate.as_mut(), n2.as_ref(), zg.as_ref(), dg)?
                .add(&proj_bwd(w, mlp.up_proj, lora.up.as_mut(), n2.as_ref(), zu.as_ref(), du)?)?
        }
        (FfnRefs::Moe(moe), FfnSaved::Moe(ms)) => {
            crate::prof::prof(dout.device(), "moe_bwd", || crate::moe::backward(moe, lora, ms, &dd, w))?
        }
        _ => return Err(other("обучение: сохранённое FFN не совпадает с видом слоя")),
    };
    let dh = dout.add(&rms_bwd(&s.h, b.pre_mlp_norm, &dn2, b.rms_eps)?)?;

    // ── Смешивание ──
    let da = match (b.post_attn_norm, &s.a) {
        (Some(w), Some(a)) => rms_bwd(a, w, &dh, b.post_eps)?,
        _ => dh.clone(),
    };
    let dn1 = match (&b.mixer, s.mix) {
        (MixerRefs::Full(at), MixSaved::Full(fs)) => full_bwd(at, lora, fs, &s.n1, s.seg.as_ref(), &da, w)?,
        (MixerRefs::Linear(la), MixSaved::Linear(ls)) => {
            crate::prof::prof(dout.device(), "gdn_bwd", || lin_bwd(la, lora, ls, &s.n1, s.seg.as_ref(), &da, w))?
        }
        _ => return Err(other("обучение: сохранённое не совпадает с видом слоя")),
    };
    Ok(dh.add(&rms_bwd(&s.x, b.pre_attn_norm, &dn1, b.rms_eps)?)?)
}

// ── Полное внимание ──────────────────────────────────────────────────────────

fn full_fwd(at: &AttnRefs<'_>, lora: &BlockLora, n1: &Tensor, ctx: &SeqCtx, w: &Weights) -> Result<(Tensor, FullSaved)> {
    let (t, nh, nkv, hd) = (ctx.t, at.num_heads, at.num_kv_heads, at.head_dim);
    let group = nh / nkv;
    let v_proj = at.v_proj.ok_or_else(|| other("обучение: слой без v_proj пока не поддержан"))?;
    let dev = n1.device();
    let (qg, zq) = proj(w, at.q_proj, lora.q.as_ref(), n1, at.q_bias)?;
    let (k, zk) = proj(w, at.k_proj, lora.k.as_ref(), n1, at.k_bias)?;
    let (v, zv) = proj(w, v_proj, lora.v.as_ref(), n1, at.v_bias)?;
    let (q_raw, gate_raw) = if at.attn_output_gate {
        let qg = qg.reshape(vec![t, nh, 2 * hd])?;
        (qg.narrow(2, 0, hd)?.contiguous()?, Some(qg.narrow(2, hd, hd)?.contiguous()?))
    } else {
        (qg.reshape(vec![t, nh, hd])?, None)
    };
    let k_raw = k.reshape(vec![t, nkv, hd])?;
    let qn = match at.q_norm {
        Some(wn) => crate::prof::prof(dev, "qknorm_fwd", || rms_norm(&q_raw, wn, at.rms_eps))?,
        None => q_raw.clone(),
    };
    let kn = match at.k_norm {
        Some(wn) => crate::prof::prof(dev, "qknorm_fwd", || rms_norm(&k_raw, wn, at.rms_eps))?,
        None => k_raw.clone(),
    };
    let (cos, sin) = crate::prof::prof(dev, "rope_tab", || rope_tables(at, ctx))?;
    let (qr, kr) = crate::prof::prof(dev, "rope", || -> Result<(Tensor, Tensor)> {
        Ok((
            rope(&f32(&qn)?.permute(vec![1, 0, 2])?.contiguous()?, &cos, &sin, at.rotary_dim, false)?,
            rope(&f32(&kn)?.permute(vec![1, 0, 2])?.contiguous()?, &cos, &sin, at.rotary_dim, false)?,
        ))
    })?;
    let vh = v.reshape(vec![t, nkv, hd])?.permute(vec![1, 0, 2])?.contiguous()?;
    let (core, o) = crate::prof::prof(dev, "attn_fwd", || attn_forward(qr, kr, vh, group, at.attn_scale, ctx))?;
    let og = match &gate_raw {
        Some(g) => o.mul(&f32(g)?.sigmoid()?)?,
        None => o.clone(),
    };
    let attn_flat = og.reshape(vec![t, nh * hd])?.to_dtype(n1.dtype())?;
    let (a, zo) = proj(w, at.o_proj, lora.o.as_ref(), &attn_flat, None)?;
    let saved = FullSaved {
        zq,
        zk,
        zv,
        q_raw,
        k_raw,
        core,
        o: if at.attn_output_gate { Some(o) } else { None },
        gate_raw,
        attn_in: lora.o.is_some().then_some(attn_flat),
        zo,
        cos,
        sin,
    };
    Ok((a, saved))
}

fn full_bwd(at: &AttnRefs<'_>, lora: &mut BlockLora, s: FullSaved, n1: &Tensor, seg: Option<&Tensor>, da: &Tensor, w: &Weights) -> Result<Tensor> {
    let (t, nh, nkv, hd) = (n1.dims()[0], at.num_heads, at.num_kv_heads, at.head_dim);
    let group = nh / nkv;
    let v_proj = at.v_proj.ok_or_else(|| other("обучение: слой без v_proj"))?;
    let dev = da.device();
    let dattn = proj_bwd(w, at.o_proj, lora.o.as_mut(), s.attn_in.as_ref(), s.zo.as_ref(), da)?.reshape(vec![t, nh, hd])?;
    let (d_o, dgate) = match &s.gate_raw {
        Some(g) => {
            let sg = f32(g)?.sigmoid()?;
            let d_o = dattn.mul(&sg)?;
            // d sigmoid = σ(1 − σ)
            let dsg = sg.mul(&sg.affine(-1.0, 1.0)?)?;
            let o = s.o.as_ref().ok_or_else(|| other("гейт без сохранённого выхода внимания"))?;
            let dgate = dattn.mul(o)?.mul(&dsg)?;
            (d_o, Some(dgate))
        }
        None => (dattn, None),
    };
    let (dqr, dkr, dv) = crate::prof::prof(dev, "attn_bwd", || attn_backward(s.core, &d_o, nkv, group, at.attn_scale, seg))?;
    let (dqn, dkn) = crate::prof::prof(dev, "rope", || -> Result<(Tensor, Tensor)> {
        Ok((
            rope(&dqr, &s.cos, &s.sin, at.rotary_dim, true)?.permute(vec![1, 0, 2])?.contiguous()?,
            rope(&dkr, &s.cos, &s.sin, at.rotary_dim, true)?.permute(vec![1, 0, 2])?.contiguous()?,
        ))
    })?;
    let dq_raw = match at.q_norm {
        Some(wn) => rms_bwd(&s.q_raw, wn, &dqn, at.rms_eps)?,
        None => dqn,
    };
    let dk_raw = match at.k_norm {
        Some(wn) => rms_bwd(&s.k_raw, wn, &dkn, at.rms_eps)?,
        None => dkn,
    };
    let dq_full = match dgate {
        Some(dg) => Tensor::cat(&[&dq_raw, &dg], 2)?.reshape(vec![t, nh * 2 * hd])?,
        None => dq_raw.reshape(vec![t, nh * hd])?,
    };
    let dk = dk_raw.reshape(vec![t, nkv * hd])?;
    let dv = dv.permute(vec![1, 0, 2])?.contiguous()?.reshape(vec![t, nkv * hd])?;
    Ok(proj_bwd(w, at.q_proj, lora.q.as_mut(), Some(n1), s.zq.as_ref(), &dq_full)?
        .add(&proj_bwd(w, at.k_proj, lora.k.as_mut(), Some(n1), s.zk.as_ref(), &dk)?)?
        .add(&proj_bwd(w, v_proj, lora.v.as_mut(), Some(n1), s.zv.as_ref(), &dv)?)?)
}

// ── Gated DeltaNet ───────────────────────────────────────────────────────────

/// `x · (Σx² + eps)^(−½)` по последней оси: `(y, r)`, `r` — `[.., 1]`.
fn l2norm(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let last = x.rank() - 1;
    let r = x.sqr()?.sum_keepdim(last)?.add_scalar(1e-6)?.sqrt()?.recip()?;
    Ok((x.broadcast_mul(&r)?, r))
}

/// Обратный проход `y = s · x · r`: `dx = s·r·dy − s·r³·x·Σ(dy·x)`.
fn l2norm_bwd(x: &Tensor, r: &Tensor, dy: &Tensor, s: f32) -> Result<Tensor> {
    let last = x.rank() - 1;
    let dot = dy.mul(x)?.sum_keepdim(last)?;
    let r3 = r.mul(r)?.mul(r)?;
    Ok(dy.broadcast_mul(r)?.sub(&x.broadcast_mul(&r3.mul(&dot)?)?)?.mul_scalar(s)?)
}

/// `softplus(x) = max(x, 0) + log(1 + e^−|x|)`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok(x.relu()?.add(&x.abs()?.neg()?.exp()?.add_scalar(1.0)?.log()?)?)
}

fn lin_fwd(la: &LinearRefs<'_>, lora: &BlockLora, n1: &Tensor, ctx: &SeqCtx, w: &Weights) -> Result<(Tensor, LinSaved)> {
    let lc = lora.lin.as_ref().ok_or_else(|| other("GDN-слой без постоянных"))?;
    let (t, hk, hv, dk, dv) = (ctx.t, la.num_k_heads, la.num_v_heads, la.dk, la.dv);
    let (kd, vd) = (la.key_dim, la.value_dim);
    let (qkv, zqkv) = proj(w, la.in_proj_qkv, lora.qkv.as_ref(), n1, None)?;
    let has_seg = ctx.seg.is_some();
    let mut ins = vec![&qkv, &lc.conv_w];
    if let Some(sg) = &ctx.seg {
        ins.push(sg);
    }
    let pre = Tensor::train_op(TrainOp::GdnConvFwd { has_seg }, &ins, &[(vec![t, la.conv_dim], DType::F32)])?.remove(0);
    let c = pre.silu()?;
    let qf = c.narrow(1, 0, kd)?.contiguous()?.reshape(vec![t, hk, dk])?;
    let kf = c.narrow(1, kd, kd)?.contiguous()?.reshape(vec![t, hk, dk])?;
    let v = c.narrow(1, 2 * kd, vd)?.contiguous()?.reshape(vec![t, hv, dv])?;
    drop(c);
    let (qn, rq) = l2norm(&qf)?;
    let qn = qn.mul_scalar(la.q_scale)?;
    let (kn, rk) = l2norm(&kf)?;

    let (a, za) = proj(w, la.in_proj_a, lora.a.as_ref(), n1, None)?;
    let (bb, zb) = proj(w, la.in_proj_b, lora.b.as_ref(), n1, None)?;
    let beta = f32(&bb)?.sigmoid()?;
    let a_dt = f32(&a)?.broadcast_add(&lc.dt_bias)?;
    let g = softplus(&a_dt)?.broadcast_mul(&lc.neg_a)?;

    let nck = t.div_ceil(64);
    let mut ins = vec![&qn, &kn, &v, &g, &beta];
    if let Some(sg) = &ctx.seg {
        ins.push(sg);
    }
    let mut out = crate::prof::prof(n1.device(), "gdn_core_fwd", || Tensor::train_op(TrainOp::GdnFwd { has_seg }, &ins, &[(vec![t, hv, dv], DType::F32), (vec![hv, nck, dk, dv], DType::F32)]))?;
    let ckpt = out.pop().unwrap();
    let o = out.pop().unwrap();
    let on = rms_norm(&o, &f32(la.norm_weight)?, la.rms_eps)?;
    let (zt, zz) = proj(w, la.in_proj_z, lora.z.as_ref(), n1, None)?;
    let z = f32(&zt)?.reshape(vec![t, hv, dv])?;
    let y = on.mul(&z.silu()?)?.reshape(vec![t, vd])?.to_dtype(n1.dtype())?;
    let (outp, zout) = proj(w, la.out_proj, lora.out.as_ref(), &y, None)?;
    let saved = LinSaved {
        zqkv,
        zz,
        za,
        zb,
        pre,
        qf,
        kf,
        rq,
        rk,
        qn,
        kn,
        v,
        g,
        beta,
        a_dt,
        ckpt,
        o,
        on,
        z,
        out_in: lora.out.is_some().then_some(y),
        zout,
    };
    Ok((outp, saved))
}

fn lin_bwd(la: &LinearRefs<'_>, lora: &mut BlockLora, s: LinSaved, n1: &Tensor, seg: Option<&Tensor>, da: &Tensor, w: &Weights) -> Result<Tensor> {
    let (t, hk, hv, dk, dv) = (n1.dims()[0], la.num_k_heads, la.num_v_heads, la.dk, la.dv);
    let (kd, vd) = (la.key_dim, la.value_dim);
    let dy = proj_bwd(w, la.out_proj, lora.out.as_mut(), s.out_in.as_ref(), s.zout.as_ref(), da)?.reshape(vec![t, hv, dv])?;
    // y = silu(z) · on
    let sh = vec![t, hv, dv];
    let mut zg = Tensor::train_op(TrainOp::SiluMulBwd, &[&s.z, &s.on, &dy], &[(sh.clone(), DType::F32), (sh, DType::F32)])?;
    let d_on = zg.pop().unwrap();
    let dz = zg.pop().unwrap().reshape(vec![t, vd])?;
    let d_o = rms_bwd(&s.o, la.norm_weight, &d_on, la.rms_eps)?;

    let p = dv / 32;
    let has_seg = seg.is_some();
    let mut ins = vec![&s.qn, &s.kn, &s.v, &s.g, &s.beta, &s.ckpt, &d_o];
    if let Some(sg) = seg {
        ins.push(sg);
    }
    let mut gr = crate::prof::prof(n1.device(), "gdn_core_bwd", || Tensor::train_op(
        TrainOp::GdnBwd { has_seg },
        &ins,
        &[
            (vec![p, t, hv, dk], DType::F32),
            (vec![p, t, hv, dk], DType::F32),
            (vec![p, t, hv], DType::F32),
            (vec![p, t, hv], DType::F32),
            (vec![t, hv, dv], DType::F32),
            (vec![p * hv * 64 * dk * 32], DType::F32),
        ],
    ))?;
    gr.pop(); // scratch
    let dv_ = gr.pop().unwrap();
    let sum_p = |x: Tensor, rest: Vec<usize>| -> Result<Tensor> {
        Ok(if p == 1 { x.reshape(rest)? } else { x.sum_keepdim(0)?.reshape(rest)? })
    };
    let dbeta = sum_p(gr.pop().unwrap(), vec![t, hv])?;
    let dg = sum_p(gr.pop().unwrap(), vec![t, hv])?;
    let group = hv / hk;
    let to_k = |x: Tensor| -> Result<Tensor> {
        let x = sum_p(x, vec![t, hv, dk])?;
        Ok(if group == 1 { x } else { x.reshape(vec![t, hk, group, dk])?.sum_keepdim(2)?.reshape(vec![t, hk, dk])? })
    };
    let dkn = to_k(gr.pop().unwrap())?;
    let dqn = to_k(gr.pop().unwrap())?;

    // β = σ(b),  g = −e^{A_log} · softplus(a + dt)
    let db = dbeta.mul(&s.beta.mul(&s.beta.affine(-1.0, 1.0)?)?)?;
    let da_ = dg.mul(&s.a_dt.sigmoid()?)?.broadcast_mul(&lora.lin.as_ref().unwrap().neg_a)?;

    let dqf = l2norm_bwd(&s.qf, &s.rq, &dqn, la.q_scale)?;
    let dkf = l2norm_bwd(&s.kf, &s.rk, &dkn, 1.0)?;
    let dc = Tensor::cat(&[&dqf.reshape(vec![t, kd])?, &dkf.reshape(vec![t, kd])?, &dv_.reshape(vec![t, vd])?], 1)?;
    // c = silu(pre): та же формула, что у silu(gate)·1.
    let ones = Tensor::ones(vec![t, la.conv_dim], DType::F32, n1.device())?;
    let cdims = vec![t, la.conv_dim];
    let dpre = Tensor::train_op(TrainOp::SiluMulBwd, &[&s.pre, &ones, &dc], &[(cdims.clone(), DType::F32), (cdims.clone(), DType::F32)])?.remove(0);
    let lc = lora.lin.as_ref().unwrap();
    let mut ins = vec![&dpre, &lc.conv_w];
    if let Some(sg) = seg {
        ins.push(sg);
    }
    let dqkv = Tensor::train_op(TrainOp::GdnConvBwd { has_seg }, &ins, &[(cdims, DType::F32)])?.remove(0);

    Ok(proj_bwd(w, la.in_proj_qkv, lora.qkv.as_mut(), Some(n1), s.zqkv.as_ref(), &dqkv)?
        .add(&proj_bwd(w, la.in_proj_z, lora.z.as_mut(), Some(n1), s.zz.as_ref(), &dz)?)?
        .add(&proj_bwd(w, la.in_proj_a, lora.a.as_mut(), Some(n1), s.za.as_ref(), &da_)?)?
        .add(&proj_bwd(w, la.in_proj_b, lora.b.as_mut(), Some(n1), s.zb.as_ref(), &db)?)?)
}
