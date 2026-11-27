//! Обёртки над ядрами `inkling2.cu`: тензоры → адреса, сетка запуска.

use synaptix_core::backend::{registry, Ink2Launch};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;

pub type R<T> = Result<T, ModelError>;

pub fn fe(e: impl std::fmt::Display) -> ModelError {
    if std::env::var_os("INK2_TRACE").is_some() {
        eprintln!("inkling2: {e}\n{}", std::backtrace::Backtrace::force_capture());
    }
    ModelError::Forward(e.to_string())
}

pub fn addr(t: &Tensor) -> R<u64> {
    if !t.is_contiguous() {
        return Err(ModelError::Forward("inkling2: тензор не contiguous".into()));
    }
    t.device_address().ok_or_else(|| ModelError::Forward("inkling2: тензор не на карте".into()))
}

pub fn launch(dev: Device, name: &str, a: &Ink2Launch, grid: (u32, u32, u32), block: u32) -> R<()> {
    registry::backend_for(dev).map_err(fe)?.inkling2_launch(dev, name, a, grid, block).map_err(fe)
}

pub fn empty(shape: Vec<usize>, dt: DType, dev: Device) -> R<Tensor> {
    Tensor::empty_uninit(shape, dt, dev).map_err(fe)
}

pub fn f32c(t: &Tensor) -> R<Tensor> {
    let t = if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32).map_err(fe)? };
    if t.is_contiguous() { Ok(t) } else { t.contiguous().map_err(fe) }
}

pub fn bf16c(t: &Tensor) -> R<Tensor> {
    let t = if t.dtype() == DType::BF16 { t.clone() } else { t.to_dtype(DType::BF16).map_err(fe)? };
    if t.is_contiguous() { Ok(t) } else { t.contiguous().map_err(fe) }
}

/// RMS-норма строк `[T, H]` F32 с весом (или без) → F32 либо BF16.
pub fn rms(x: &Tensor, w: Option<&Tensor>, eps: f32, out_bf16: bool) -> R<Tensor> {
    let (t, h) = (x.dims()[0], x.dims()[1]);
    let out = empty(vec![t, h], if out_bf16 { DType::BF16 } else { DType::F32 }, x.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = match w { Some(w) => addr(w)?, None => 0 };
    a.a[2] = addr(&out)?;
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[2] = out_bf16 as u32;
    a.f[0] = eps;
    launch(x.device(), "ink2_rms", &a, (t as u32, 1, 1), 256)?;
    Ok(out)
}

/// RMS-норма `nz` входов одним запуском: вход `j` — строки `[T, H]` с `j·x_z·T·H` элемента `x`
/// (F32), вес — `ws[j]`. Выходы — отдельные `[T, H]` BF16.
pub fn rms_z(x: &Tensor, x_z: usize, t: usize, ws: &[&Tensor], eps: f32) -> R<Vec<Tensor>> {
    let h = *x.dims().last().expect("ранг ≥ 1");
    let nz = ws.len();
    if nz == 0 || nz > 3 {
        return Err(ModelError::Forward(format!("inkling2 rms_z: {nz} входов")));
    }
    let outs: Vec<Tensor> = (0..nz).map(|_| empty(vec![t, h], DType::BF16, x.device())).collect::<R<_>>()?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = addr(ws[0])?;
    a.a[2] = addr(&outs[0])?;
    for j in 1..nz {
        a.a[3 + j] = addr(ws[j])?;
        a.a[5 + j] = addr(&outs[j])?;
    }
    a.u[0] = t as u32;
    a.u[1] = h as u32;
    a.u[2] = 1;
    a.u[3] = (x_z * t * h) as u32;
    a.f[0] = eps;
    launch(x.device(), "ink2_rms", &a, (t as u32, 1, nz as u32), 256)?;
    Ok(outs)
}

/// Внимание по глубине: состояния (`[T, H]` F32), запросы `[nq, H]` → `[nq, T, H]`.
/// На декоде (`T = 1`, `H` кратен 4) — одним запуском `ink2_mix_dec` (без второго чтения
/// состояний); на префилле — два прохода `ink2_mix_part` + `ink2_mix_out`.
pub fn mix(states: &[&Tensor], queries: &Tensor) -> R<Tensor> {
    if states.is_empty() || states.len() > 8 {
        return Err(ModelError::Forward(format!("inkling2 mix: {} состояний", states.len())));
    }
    let (t, h) = (states[0].dims()[0], states[0].dims()[1]);
    let nq = queries.dims()[0];
    let dev = states[0].device();
    let out = empty(vec![nq, t, h], DType::F32, dev)?;
    let mut a = Ink2Launch::default();
    for (i, s) in states.iter().enumerate() {
        a.a[i] = addr(s)?;
    }
    a.a[8] = addr(queries)?;
    a.a[9] = addr(&out)?;
    a.u[0] = states.len() as u32;
    a.u[1] = t as u32;
    a.u[2] = h as u32;
    a.u[3] = nq as u32;
    if t == 1 && h % 4 == 0 && h <= 4096 {
        let block = ((h / 4) as u32).div_ceil(32) * 32;
        launch(dev, "ink2_mix_dec", &a, (1, 1, 1), block)?;
        return Ok(out);
    }
    let b = (h as u32).div_ceil(256);
    let part = empty(vec![t * b as usize * 40], DType::F32, dev)?;
    a.a[10] = addr(&part)?;
    a.u[4] = b;
    launch(dev, "ink2_mix_part", &a, (t as u32, b, 1), 256)?;
    launch(dev, "ink2_mix_out", &a, (t as u32, b, 1), 256)?;
    Ok(out)
}

/// Слитое `mix` (nq = 1) + `rms` → `[1, H]` BF16 на декоде (`T = 1`, один запуск вместо трёх).
pub fn mix_rms_dec(states: &[&Tensor], query: &Tensor, rms_w: &Tensor, eps: f32) -> R<Tensor> {
    let h = states[0].dims()[1];
    let dev = states[0].device();
    let out = empty(vec![1, h], DType::BF16, dev)?;
    let mut a = Ink2Launch::default();
    for (i, s) in states.iter().enumerate() {
        a.a[i] = addr(s)?;
    }
    a.a[8] = addr(query)?;
    a.a[9] = addr(&out)?;
    a.a[10] = addr(rms_w)?;
    a.u[0] = states.len() as u32;
    a.u[1] = 1;
    a.u[2] = h as u32;
    a.u[3] = 1;
    a.u[4] = 1;
    a.f[0] = eps;
    let block = ((h / 4) as u32).div_ceil(32) * 32;
    launch(dev, "ink2_mix_dec", &a, (1, 1, 1), block)?;
    Ok(out)
}

/// Строка `j` результата [`mix`] как `[T, H]`.
pub fn pick(mixed: &Tensor, j: usize) -> R<Tensor> {
    let t = mixed.narrow(0, j, 1).map_err(fe)?.contiguous().map_err(fe)?;
    let (tt, h) = (mixed.dims()[1], mixed.dims()[2]);
    t.reshape(vec![tt, h]).map_err(fe)
}

/// Столбцы `[off, …)` матрицы `t [T, S]` (F32 либо BF16, contiguous) для ядра: адрес первого
/// столбца, шаг строки S и признак BF16. Без копии: ядра читают срез с шагом строки.
pub struct Cols {
    pub addr: u64,
    pub stride: u32,
    pub bf16: u32,
}

pub fn cols(t: &Tensor, off: usize) -> R<Cols> {
    let bf16 = match t.dtype() {
        DType::BF16 => 1,
        DType::F32 => 0,
        d => return Err(ModelError::Forward(format!("inkling2: вход {d:?}"))),
    };
    Ok(Cols { addr: addr(t)? + (off * t.dtype().bytes_for_numel(1)) as u64, stride: t.dims()[1] as u32, bf16 })
}

/// Одна короткая свёртка: вход — столбцы матрицы (F32/BF16), веса `[C, K]` F32, состояние
/// `[K-1, C]`; динамические отводы — столбцы той же dtype, что вход, и число голов.
pub struct ConvJob<'a> {
    pub x: Cols,
    pub w: &'a Tensor,
    pub state: &'a Tensor,
    pub dynamic: Option<(Cols, usize)>,
}

/// Короткая свёртка с остатком по `c` каналам (одна или две одной формы — один запуск);
/// `res` — прибавить остаток F32 `[T, C]` к первой (слитое сложение с потоком). Декод (T = 1) —
/// поток на канал; префилл — блок на (канал, токен) и отдельный запуск обновления состояния.
pub fn conv(jobs: &[ConvJob], t: usize, c: usize, res: Option<&Tensor>, scale: f32, dev: Device) -> R<Vec<Tensor>> {
    if jobs.is_empty() || jobs.len() > 2 {
        return Err(ModelError::Forward(format!("inkling2 conv: {} свёрток", jobs.len())));
    }
    let k = jobs[0].w.dims()[1];
    let outs: Vec<Tensor> = jobs.iter().map(|_| empty(vec![t, c], DType::F32, dev)).collect::<R<_>>()?;
    let mut a = Ink2Launch::default();
    a.u[4] = jobs[0].x.bf16;
    a.u[5] = jobs[0].x.stride;
    for (z, (jb, out)) in jobs.iter().zip(&outs).enumerate() {
        if jb.x.bf16 != a.u[4] || jb.x.stride != a.u[5] {
            return Err(ModelError::Forward("inkling2 conv: входы пары разной раскладки".into()));
        }
        let o = if z == 0 { [0, 1, 2, 3, 4] } else { [6, 7, 8, 9, 10] };
        a.a[o[0]] = jb.x.addr;
        a.a[o[1]] = addr(out)?;
        a.a[o[2]] = addr(jb.state)?;
        a.a[o[3]] = addr(jb.w)?;
        if let Some((d, heads)) = &jb.dynamic {
            if d.bf16 != jb.x.bf16 {
                return Err(ModelError::Forward("inkling2 conv: dtype отводов ≠ dtype входа".into()));
            }
            a.a[o[4]] = d.addr;
            a.u[3] = *heads as u32;
            a.u[6] = d.stride;
        }
    }
    if let Some(r) = res {
        a.a[5] = addr(r)?;
    }
    a.u[0] = t as u32;
    a.u[1] = c as u32;
    a.u[2] = k as u32;
    a.f[0] = scale;
    let (gx, gz) = ((c as u32).div_ceil(256), jobs.len() as u32);
    if t == 1 {
        launch(dev, "ink2_conv", &a, (gx, 1, gz), 256)?;
    } else {
        launch(dev, "ink2_conv", &a, (gx, t as u32, gz), 256)?;
        launch(dev, "ink2_conv_state", &a, (gx, 1, gz), 256)?;
    }
    Ok(outs)
}

static PROF: std::sync::Mutex<Vec<(&'static str, f64, u64)>> = std::sync::Mutex::new(Vec::new());

fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("INK2_PROF").is_some())
}

fn sync(dev: Device) {
    if let Device::Cuda(o) = dev {
        if let Ok(s) = synaptix_core::device::cuda::default_stream(o) {
            let _ = s.synchronize();
        }
    }
}

/// Этап с замером (`INK2_PROF`): синхронизация карты до и после — время ядер этапа.
pub fn stage<T>(name: &'static str, dev: Device, f: impl FnOnce() -> T) -> T {
    if !prof_on() {
        return f();
    }
    sync(dev);
    let t = std::time::Instant::now();
    let out = f();
    sync(dev);
    let dt = t.elapsed().as_secs_f64();
    let mut p = PROF.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += dt;
            e.2 += 1;
        }
        None => p.push((name, dt, 1)),
    }
    out
}

pub fn prof_report() -> String {
    let mut p = PROF.lock().unwrap();
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = p.iter().map(|e| e.1).sum();
    let s = p.iter().map(|(n, t, c)| format!("  {n}: {:.1} мс ({:.1}%), {c}\n", t * 1e3, 100.0 * t / total.max(1e-12))).collect();
    p.clear();
    s
}

/// Число частей по ключам для `ink2_attn_fa` (префилл): `n` запросов, `nh` голов, `keys` —
/// ключей на блок запросов. Цель — ~1024 блока на карту, часть не короче 1024 ключей
/// (32 плитки), буфер частей `[n, nh, S, 130]` F32 — не больше 64 МБ.
pub fn attn_fa_splits(n: usize, nh: usize, keys: usize) -> usize {
    let base = n.div_ceil(64) * nh;
    let by_mem = (64usize << 20) / (n * nh * 130 * 4).max(1);
    1024usize.div_ceil(base.max(1)).min(keys.div_ceil(1024)).min(by_mem).clamp(1, 64)
}
