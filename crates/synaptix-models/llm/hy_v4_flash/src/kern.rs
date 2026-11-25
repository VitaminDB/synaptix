//! Обёртки над ядрами `hy4.cu`: тензоры → адреса, сетка запуска.

use synaptix_core::backend::{registry, Ink2Launch};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;

pub type R<T> = Result<T, ModelError>;

pub fn fe(e: impl std::fmt::Display) -> ModelError {
    if std::env::var_os("HY4_TRACE").is_some() {
        eprintln!("hy4: {e}\n{}", std::backtrace::Backtrace::force_capture());
    }
    ModelError::Forward(e.to_string())
}

pub fn addr(t: &Tensor) -> R<u64> {
    if !t.is_contiguous() {
        return Err(ModelError::Forward("hy4: тензор не contiguous".into()));
    }
    t.device_address().ok_or_else(|| ModelError::Forward("hy4: тензор не на карте".into()))
}

fn launch(dev: Device, name: &str, a: &Ink2Launch, grid: (u32, u32, u32), block: u32) -> R<()> {
    registry::backend_for(dev).map_err(fe)?.hy4_launch(dev, name, a, grid, block).map_err(fe)
}

pub fn empty(shape: Vec<usize>, dt: DType, dev: Device) -> R<Tensor> {
    Tensor::empty_uninit(shape, dt, dev).map_err(fe)
}

pub fn zeros(shape: Vec<usize>, dt: DType, dev: Device) -> R<Tensor> {
    Tensor::zeros(shape, dt, dev).map_err(fe)
}

/// Код типа для ядер: 0 F32, 1 BF16, 2 F16.
pub fn kind(dt: DType) -> R<u32> {
    match dt {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F16 => Ok(2),
        o => Err(ModelError::Forward(format!("hy4: тип {o:?} ядрам не годится"))),
    }
}

pub fn contig(t: &Tensor) -> R<Tensor> {
    if t.is_contiguous() { Ok(t.clone()) } else { t.contiguous().map_err(fe) }
}

pub fn to(t: &Tensor, dt: DType) -> R<Tensor> {
    let t = if t.dtype() == dt { t.clone() } else { t.to_dtype(dt).map_err(fe)? };
    contig(&t)
}

pub fn f32c(t: &Tensor) -> R<Tensor> {
    to(t, DType::F32)
}

/// Потоки → вход подслоя. `x` `[T, M, H]` F32, `mixes` `[T, NM]` F32.
/// Обычный режим: `(h [T, H], Some(post [T, M]))`; голова (`scale` из одного
/// числа, `base` `[M]`) — `(h, None)`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(x: &Tensor, mixes: &Tensor, base: &Tensor, scale: &Tensor, head: bool, eps: f32, hc_eps: f32, mag: f32) -> R<(Tensor, Option<Tensor>)> {
    let (t, m, h) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let dev = x.device();
    let out = empty(vec![t, h], DType::F32, dev)?;
    let post = if head { None } else { Some(empty(vec![t, m], DType::F32, dev)?) };
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = addr(mixes)?;
    a.a[2] = addr(base)?;
    a.a[3] = addr(scale)?;
    a.a[4] = addr(&out)?;
    a.a[5] = match &post { Some(p) => addr(p)?, None => 0 };
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[2] = m as u32;
    a.u[3] = head as u32;
    a.f[0] = eps;
    a.f[1] = hc_eps;
    a.f[2] = mag;
    launch(dev, "hy4_hc_pre", &a, (t as u32, 1, 1), 256)?;
    Ok((out, post))
}

/// `x[t, i, :] += post[t, i] · y[t, :]` на месте.
pub fn hc_post(x: &Tensor, post: &Tensor, y: &Tensor) -> R<()> {
    let (t, m, h) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let y = contig(y)?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = addr(post)?;
    a.a[2] = addr(&y)?;
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[2] = m as u32;
    a.u[3] = kind(y.dtype())?;
    launch(x.device(), "hy4_hc_post", &a, (t as u32, 1, 1), 256)
}

/// RMS-норма строк `[T, H]` (любой из F32/BF16/F16) с весом F32 → `out_dt`.
pub fn rms(x: &Tensor, w: Option<&Tensor>, eps: f32, out_dt: DType) -> R<Tensor> {
    let x = contig(x)?;
    let h = *x.dims().last().expect("ранг ≥ 1");
    let t = x.numel() / h;
    let out = empty(x.dims().to_vec(), out_dt, x.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(&x)?;
    a.a[1] = match w { Some(w) => addr(w)?, None => 0 };
    a.a[2] = addr(&out)?;
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[3] = kind(x.dtype())?;
    a.u[4] = kind(out_dt)?;
    a.f[0] = eps;
    launch(x.device(), "hy4_rms", &a, (t as u32, 1, 1), 256)?;
    Ok(out)
}

/// RMS-норма первых `h` столбцов строк шириной `stride` → `[T, h]`.
pub fn rms_cols(x: &Tensor, h: usize, w: Option<&Tensor>, eps: f32, out_dt: DType) -> R<Tensor> {
    let stride = *x.dims().last().expect("ранг ≥ 1");
    let t = x.numel() / stride;
    let out = empty(vec![t, h], out_dt, x.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = match w { Some(w) => addr(w)?, None => 0 };
    a.a[2] = addr(&out)?;
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[2] = stride as u32;
    a.u[3] = kind(x.dtype())?;
    a.u[4] = kind(out_dt)?;
    a.f[0] = eps;
    launch(x.device(), "hy4_rms", &a, (t as u32, 1, 1), 256)?;
    Ok(out)
}

/// RoPE на месте: строка `t` шириной `row`, `heads` голов с шагом `head_stride`,
/// хвост от `off` шириной `width`.
#[allow(clippy::too_many_arguments)]
pub fn rope(x: &Tensor, pos: &Tensor, row: usize, heads: usize, head_stride: usize, off: usize, width: usize, theta: f64) -> R<()> {
    let t = x.numel() / row;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = addr(pos)?;
    a.u[0] = t as u32;
    a.u[1] = row as u32;
    a.u[2] = head_stride as u32;
    a.u[3] = off as u32;
    a.u[4] = heads as u32;
    a.u[5] = width as u32;
    a.f[0] = theta.log2() as f32;
    launch(x.device(), "hy4_rope", &a, (t as u32, heads as u32, 1), 32)
}

/// Записать строки `src [T, D]` F32 в кэш `[CAP, D]` по слотам `pos mod CAP`.
pub fn cache_put(src: &Tensor, cache: &Tensor, pos: &Tensor) -> R<()> {
    let (t, d) = (src.dims()[0], src.dims()[1]);
    let mut a = Ink2Launch::default();
    a.a[0] = addr(src)?;
    a.a[1] = addr(cache)?;
    a.a[2] = addr(pos)?;
    a.u[0] = t as u32;
    a.u[1] = d as u32;
    a.u[2] = cache.dims()[0] as u32;
    a.u[3] = kind(cache.dtype())?;
    launch(src.device(), "hy4_cache_put", &a, (t as u32, 1, 1), 256)
}

/// Выбор ключей внимания.
pub enum Keys<'a> {
    /// Окно: ключи `pos − w + 1 … pos`.
    Window(usize),
    /// Группы `[T, NG]` (u32::MAX — пусто) плюс своя группа запроса; `G` токенов в группе.
    Groups(&'a Tensor, usize),
}

/// MLA по латентному кэшу с синками. `qa` `[T, NH, L+R]` F32 → `o [T, NH, L]` F32.
pub fn mla_attn(qa: &Tensor, cache: &Tensor, sinks: &Tensor, pos: &Tensor, latent: usize, keys: Keys<'_>, scale: f32) -> R<Tensor> {
    let (t, nh, d) = (qa.dims()[0], qa.dims()[1], qa.dims()[2]);
    let o = empty(vec![t, nh, latent], DType::F32, qa.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(qa)?;
    a.a[1] = addr(cache)?;
    a.a[2] = addr(sinks)?;
    a.a[3] = addr(&o)?;
    a.a[4] = addr(pos)?;
    a.u[0] = t as u32;
    a.u[1] = nh as u32;
    a.u[2] = latent as u32;
    a.u[3] = (d - latent) as u32;
    a.u[4] = cache.dims()[0] as u32;
    a.u[7] = kind(cache.dtype())?;
    match keys {
        Keys::Window(w) => {
            a.u[5] = 0;
            a.u[6] = w.min(u32::MAX as usize) as u32;
        }
        Keys::Groups(g, size) => {
            a.u[5] = 1;
            a.a[5] = addr(g)?;
            a.u[8] = g.dims()[1] as u32;
            a.u[9] = size as u32;
        }
    }
    a.f[0] = scale;
    launch(qa.device(), "hy4_mla_attn", &a, (t as u32, nh as u32, 1), 256)?;
    Ok(o)
}

/// Пулинг групп `g0..g1` индексатора в кэш ключей групп.
#[allow(clippy::too_many_arguments)]
pub fn idx_pool(tok: &Tensor, wv: &Tensor, wg: &Tensor, ape: &Tensor, norm: &Tensor, out: &Tensor, g0: usize, g1: usize, group: usize, rope_w: usize, eps: f32, theta: f64) -> R<()> {
    if g1 <= g0 {
        return Ok(());
    }
    let c = tok.dims()[1];
    let mut a = Ink2Launch::default();
    a.a[0] = addr(tok)?;
    a.a[1] = addr(wv)?;
    a.a[2] = addr(wg)?;
    a.a[3] = addr(ape)?;
    a.a[4] = addr(norm)?;
    a.a[5] = addr(out)?;
    a.u[0] = g0 as u32;
    a.u[1] = c as u32;
    a.u[2] = group as u32;
    a.u[3] = tok.dims()[0] as u32;
    a.u[4] = out.dims()[0] as u32;
    a.u[5] = rope_w as u32;
    a.f[0] = eps;
    a.f[1] = theta.log2() as f32;
    launch(tok.device(), "hy4_idx_pool", &a, ((g1 - g0) as u32, 1, 1), 128)
}

/// Оценки групп `[T, ng]` (групп не дальше своей). Остальное — не определено.
pub fn idx_score(q: &Tensor, w: &Tensor, keys: &Tensor, pos: &Tensor, ng: usize, group: usize) -> R<Tensor> {
    let (t, nhi, c) = (q.dims()[0], q.dims()[1], q.dims()[2]);
    let out = zeros(vec![t, ng.max(1)], DType::F32, q.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(q)?;
    a.a[1] = addr(w)?;
    a.a[2] = addr(keys)?;
    a.a[3] = addr(&out)?;
    a.a[4] = addr(pos)?;
    a.u[0] = t as u32;
    a.u[1] = nhi as u32;
    a.u[2] = c as u32;
    a.u[3] = ng.max(1) as u32;
    a.u[4] = group as u32;
    a.u[5] = keys.dims()[0] as u32;
    launch(q.device(), "hy4_idx_score", &a, (t as u32, ng.max(1).div_ceil(256) as u32, 1), 256)?;
    Ok(out)
}

/// Гейт engram на месте по потокам `x [T, M, H]`.
#[allow(clippy::too_many_arguments)]
pub fn engram_gate(x: &Tensor, proj: &Tensor, kn: &Tensor, qn: &Tensor, pre: &Tensor, hcs: &Tensor, live: &Tensor, eps: f32) -> R<()> {
    let (t, m, h) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = addr(proj)?;
    a.a[2] = addr(kn)?;
    a.a[3] = addr(qn)?;
    a.a[4] = addr(pre)?;
    a.a[5] = addr(hcs)?;
    a.a[6] = addr(live)?;
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[2] = m as u32;
    a.f[0] = eps;
    launch(x.device(), "hy4_engram_gate", &a, ((t * m) as u32, 1, 1), 256)
}

/// `out = o · σ(gate)` → `out_dt`.
pub fn gate_mul(o: &Tensor, gate: &Tensor, out_dt: DType) -> R<Tensor> {
    let gate = contig(gate)?;
    let n = *o.dims().last().expect("ранг");
    let t = o.numel() / n;
    let out = empty(vec![t, n], out_dt, o.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(o)?;
    a.a[1] = addr(&gate)?;
    a.a[2] = addr(&out)?;
    a.u[0] = t as u32;
    a.u[1] = n as u32;
    a.u[2] = kind(gate.dtype())?;
    a.u[3] = kind(out_dt)?;
    launch(o.device(), "hy4_gate_mul", &a, (t as u32, 1, 1), 256)?;
    Ok(out)
}
