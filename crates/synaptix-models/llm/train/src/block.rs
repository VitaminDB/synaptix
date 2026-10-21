//! Слой декодера на обучении: прямой проход с сохранением всего, что нужно
//! обратному, и обратный проход по входу слоя с накоплением градиентов LoRA.
//!
//! Активации прямого хода — в compute-типе модели (как у инференса), ядро
//! внимания и все градиенты — в F32. Последовательность одна (`[T, H]`),
//! несколько примеров в ней разделены номерами `seg`.

use synaptix_core::backend::train::TrainOp;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::train_access::{AttnRefs, BlockRefs};
use synaptix_llm_common::{Activation, QLinear};
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::error::other;
use crate::linear::Weights;
use crate::lora::LoraLayer;
use crate::Result;

/// Адаптеры одного слоя (`None` — проекция не учится).
#[derive(Default)]
pub struct BlockLora {
    pub q: Option<LoraLayer>,
    pub k: Option<LoraLayer>,
    pub v: Option<LoraLayer>,
    pub o: Option<LoraLayer>,
    pub gate: Option<LoraLayer>,
    pub up: Option<LoraLayer>,
    pub down: Option<LoraLayer>,
}

impl BlockLora {
    pub fn layers(&self) -> impl Iterator<Item = &LoraLayer> {
        [&self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down].into_iter().flatten()
    }

    pub fn layers_mut(&mut self) -> impl Iterator<Item = &mut LoraLayer> {
        [&mut self.q, &mut self.k, &mut self.v, &mut self.o, &mut self.gate, &mut self.up, &mut self.down]
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

/// Всё, что прямой проход оставляет обратному.
pub struct Saved {
    x: Tensor,
    n1_32: Option<Tensor>,
    zq: Option<Tensor>,
    zk: Option<Tensor>,
    zv: Option<Tensor>,
    q_raw: Tensor,
    gate_raw: Option<Tensor>,
    k_raw: Tensor,
    core: AttnCore,
    seg: Option<Tensor>,
    /// Выход внимания до гейта `[T, nh, hd]` F32 — только при гейте.
    o: Option<Tensor>,
    attn32: Option<Tensor>,
    zo: Option<Tensor>,
    a: Option<Tensor>,
    h: Tensor,
    n2_32: Option<Tensor>,
    zg: Option<Tensor>,
    zu: Option<Tensor>,
    g: Tensor,
    u: Tensor,
    m32: Option<Tensor>,
    zd: Option<Tensor>,
    d: Option<Tensor>,
    cos: Tensor,
    sin: Tensor,
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
fn proj(w: &Weights, q: &QLinear, lora: Option<&LoraLayer>, x: &Tensor, x32: Option<&Tensor>, bias: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
    let mut y = w.forward(q, x)?;
    if let Some(b) = bias {
        y = y.broadcast_add(&same_dtype(b, &y)?)?;
    }
    match lora {
        Some(l) => {
            let x32 = x32.ok_or_else(|| other("proj: нет F32-входа для LoRA"))?;
            let (z, y) = crate::prof::prof(x.device(), "lora_fwd", || l.forward_add(x32, &y))?;
            Ok((y, Some(z)))
        }
        None => Ok((y, None)),
    }
}

/// `dx` проекции: замороженный вес плюс вклад адаптера (градиенты адаптера
/// копятся в нём).
fn proj_bwd(w: &Weights, q: &QLinear, lora: Option<&mut LoraLayer>, x32: Option<&Tensor>, z: Option<&Tensor>, dy: &Tensor) -> Result<Tensor> {
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

/// Прямой проход слоя. `x` — `[T, H]` в compute-типе.
pub fn forward(b: &BlockRefs<'_>, lora: &BlockLora, x: &Tensor, ctx: &SeqCtx, w: &Weights) -> Result<(Tensor, Saved)> {
    let at = &b.attn;
    let (t, nh, nkv, hd) = (ctx.t, at.num_heads, at.num_kv_heads, at.head_dim);
    let group = nh / nkv;
    let v_proj = at.v_proj.ok_or_else(|| other("обучение: слой без v_proj пока не поддержан"))?;
    if !matches!(b.mlp.activation, Activation::Silu) {
        return Err(other("обучение: пока только SwiGLU-MLP"));
    }

    // ── Внимание ──
    let n1 = crate::prof::prof(x.device(), "norm_fwd", || rms_norm(x, b.pre_attn_norm, b.rms_eps))?;
    let any_attn_lora = lora.q.is_some() || lora.k.is_some() || lora.v.is_some();
    let n1_32 = if any_attn_lora { Some(n1.clone()) } else { None };
    let (qg, zq) = proj(w, at.q_proj, lora.q.as_ref(), &n1, n1_32.as_ref(), at.q_bias)?;
    let (k, zk) = proj(w, at.k_proj, lora.k.as_ref(), &n1, n1_32.as_ref(), at.k_bias)?;
    let (v, zv) = proj(w, v_proj, lora.v.as_ref(), &n1, n1_32.as_ref(), at.v_bias)?;

    let (q_raw, gate_raw) = if at.attn_output_gate {
        let qg = qg.reshape(vec![t, nh, 2 * hd])?;
        (qg.narrow(2, 0, hd)?.contiguous()?, Some(qg.narrow(2, hd, hd)?.contiguous()?))
    } else {
        (qg.reshape(vec![t, nh, hd])?, None)
    };
    let k_raw = k.reshape(vec![t, nkv, hd])?;
    let qn = match at.q_norm {
        Some(w) => crate::prof::prof(x.device(), "qknorm_fwd", || rms_norm(&q_raw, w, at.rms_eps))?,
        None => q_raw.clone(),
    };
    let kn = match at.k_norm {
        Some(w) => crate::prof::prof(x.device(), "qknorm_fwd", || rms_norm(&k_raw, w, at.rms_eps))?,
        None => k_raw.clone(),
    };
    let (cos, sin) = crate::prof::prof(x.device(), "rope_tab", || rope_tables(at, ctx))?;
    let (qr, kr) = crate::prof::prof(x.device(), "rope", || -> Result<(Tensor, Tensor)> {
        Ok((
            rope(&f32(&qn)?.permute(vec![1, 0, 2])?.contiguous()?, &cos, &sin, at.rotary_dim, false)?,
            rope(&f32(&kn)?.permute(vec![1, 0, 2])?.contiguous()?, &cos, &sin, at.rotary_dim, false)?,
        ))
    })?;
    let vh = v.reshape(vec![t, nkv, hd])?.permute(vec![1, 0, 2])?.contiguous()?;

    let (core, o) = crate::prof::prof(x.device(), "attn_fwd", || attn_forward(qr, kr, vh, group, at.attn_scale, ctx))?;
    let og = match &gate_raw {
        Some(g) => o.mul(&f32(g)?.sigmoid()?)?,
        None => o.clone(),
    };
    let attn_flat = crate::prof::prof(x.device(), "o_cast", || og.reshape(vec![t, nh * hd]).and_then(|v| v.to_dtype(x.dtype())))?;
    let attn32 = if lora.o.is_some() { Some(attn_flat.clone()) } else { None };
    let (a, zo) = proj(w, at.o_proj, lora.o.as_ref(), &attn_flat, attn32.as_ref(), None)?;
    let a2 = match b.post_attn_norm {
        Some(w) => rms_norm(&a, w, b.post_eps)?,
        None => a.clone(),
    };
    let h = crate::prof::prof(x.device(), "resid_fwd", || -> Result<Tensor> { Ok(x.add(&same_dtype(&a2, x)?)?) })?;

    // ── MLP ──
    let n2 = crate::prof::prof(x.device(), "norm_fwd", || rms_norm(&h, b.pre_mlp_norm, b.rms_eps))?;
    let n2_32 = if lora.gate.is_some() || lora.up.is_some() { Some(n2.clone()) } else { None };
    let (g, zg) = proj(w, b.mlp.gate_proj, lora.gate.as_ref(), &n2, n2_32.as_ref(), None)?;
    let (u, zu) = proj(w, b.mlp.up_proj, lora.up.as_ref(), &n2, n2_32.as_ref(), None)?;
    let m = crate::prof::prof(x.device(), "act_fwd", || g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|s| s.mul(&u))))?;
    let m32 = if lora.down.is_some() { Some(m.clone()) } else { None };
    let (d, zd) = proj(w, b.mlp.down_proj, lora.down.as_ref(), &m, m32.as_ref(), None)?;
    let d2 = match b.post_mlp_norm {
        Some(w) => rms_norm(&d, w, b.post_eps)?,
        None => d.clone(),
    };
    let out = crate::prof::prof(x.device(), "resid_fwd", || -> Result<Tensor> { Ok(h.add(&same_dtype(&d2, &h)?)?) })?;

    let saved = Saved {
        x: x.clone(),
        n1_32,
        zq,
        zk,
        zv,
        q_raw,
        gate_raw,
        k_raw,
        core,
        seg: ctx.seg.clone(),
        o: if at.attn_output_gate { Some(o) } else { None },
        attn32,
        zo,
        a: b.post_attn_norm.map(|_| a),
        h,

        n2_32,
        zg,
        zu,
        g,
        u,
        m32,
        zd,
        d: b.post_mlp_norm.map(|_| d),
        cos,
        sin,
    };
    Ok((out, saved))
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

fn ctx_seg(seg: &Option<Tensor>) -> Option<&Tensor> {
    seg.as_ref()
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

/// Обратный проход слоя: `dout` — `[T, H]` F32, результат — `dx` F32.
pub fn backward(b: &BlockRefs<'_>, lora: &mut BlockLora, s: Saved, dout: &Tensor, w: &Weights) -> Result<Tensor> {
    let at = &b.attn;
    let (t, nh, nkv, hd) = (s.x.dims()[0], at.num_heads, at.num_kv_heads, at.head_dim);
    let group = nh / nkv;
    let v_proj = at.v_proj.ok_or_else(|| other("обучение: слой без v_proj"))?;

    // ── MLP ──
    let dd = match (b.post_mlp_norm, &s.d) {
        (Some(w), Some(d)) => rms_bwd(d, w, dout, b.post_eps)?,
        _ => dout.clone(),
    };
    let dm = proj_bwd(w, b.mlp.down_proj, lora.down.as_mut(), s.m32.as_ref(), s.zd.as_ref(), &dd)?;
    let shp = s.g.dims().to_vec();
    let gu = Tensor::train_op(TrainOp::SiluMulBwd, &[&s.g, &s.u, &dm], &[(shp.clone(), DType::F32), (shp, DType::F32)])?;
    let (dg, du) = (&gu[0], &gu[1]);
    let dn2 = proj_bwd(w, b.mlp.gate_proj, lora.gate.as_mut(), s.n2_32.as_ref(), s.zg.as_ref(), dg)?
        .add(&proj_bwd(w, b.mlp.up_proj, lora.up.as_mut(), s.n2_32.as_ref(), s.zu.as_ref(), du)?)?;
    drop(gu);
    let dh = dout.add(&rms_bwd(&s.h, b.pre_mlp_norm, &dn2, b.rms_eps)?)?;

    // ── Внимание ──
    let da = match (b.post_attn_norm, &s.a) {
        (Some(w), Some(a)) => rms_bwd(a, w, &dh, b.post_eps)?,
        _ => dh.clone(),
    };
    let dattn = proj_bwd(w, at.o_proj, lora.o.as_mut(), s.attn32.as_ref(), s.zo.as_ref(), &da)?.reshape(vec![t, nh, hd])?;
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
    let dev = dout.device();
    let (dqr, dkr, dv) = crate::prof::prof(dev, "attn_bwd", || attn_backward(s.core, &d_o, nkv, group, at.attn_scale, ctx_seg(&s.seg)))?;

    let (dqn, dkn) = crate::prof::prof(dev, "rope", || -> Result<(Tensor, Tensor)> {
        Ok((
            rope(&dqr, &s.cos, &s.sin, at.rotary_dim, true)?.permute(vec![1, 0, 2])?.contiguous()?,
            rope(&dkr, &s.cos, &s.sin, at.rotary_dim, true)?.permute(vec![1, 0, 2])?.contiguous()?,
        ))
    })?;
    let dq_raw = match at.q_norm {
        Some(w) => rms_bwd(&s.q_raw, w, &dqn, at.rms_eps)?,
        None => dqn,
    };
    let dk_raw = match at.k_norm {
        Some(w) => rms_bwd(&s.k_raw, w, &dkn, at.rms_eps)?,
        None => dkn,
    };
    let dq_full = match dgate {
        Some(dg) => Tensor::cat(&[&dq_raw, &dg], 2)?.reshape(vec![t, nh * 2 * hd])?,
        None => dq_raw.reshape(vec![t, nh * hd])?,
    };
    let dk = dk_raw.reshape(vec![t, nkv * hd])?;
    let dv = dv.permute(vec![1, 0, 2])?.contiguous()?.reshape(vec![t, nkv * hd])?;

    let dn1 = proj_bwd(w, at.q_proj, lora.q.as_mut(), s.n1_32.as_ref(), s.zq.as_ref(), &dq_full)?
        .add(&proj_bwd(w, at.k_proj, lora.k.as_mut(), s.n1_32.as_ref(), s.zk.as_ref(), &dk)?)?
        .add(&proj_bwd(w, v_proj, lora.v.as_mut(), s.n1_32.as_ref(), s.zv.as_ref(), &dv)?)?;
    Ok(dh.add(&rms_bwd(&s.x, b.pre_attn_norm, &dn1, b.rms_eps)?)?)
}
