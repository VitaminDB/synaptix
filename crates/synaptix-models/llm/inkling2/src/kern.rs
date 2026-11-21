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

/// Внимание по глубине: состояния (`[T, H]` F32), запросы `[nq, H]` → `[nq, T, H]`.
pub fn mix(states: &[&Tensor], queries: &Tensor) -> R<Tensor> {
    if states.is_empty() || states.len() > 8 {
        return Err(ModelError::Forward(format!("inkling2 mix: {} состояний", states.len())));
    }
    let (t, h) = (states[0].dims()[0], states[0].dims()[1]);
    let nq = queries.dims()[0];
    let out = empty(vec![nq, t, h], DType::F32, states[0].device())?;
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
    let dev = states[0].device();
    let b = (h as u32).div_ceil(256);
    let part = empty(vec![t * b as usize * 40], DType::F32, dev)?;
    a.a[10] = addr(&part)?;
    a.u[4] = b;
    launch(dev, "ink2_mix_part", &a, (t as u32, b, 1), 256)?;
    launch(dev, "ink2_mix_out", &a, (t as u32, b, 1), 256)?;
    Ok(out)
}

/// Строка `j` результата [`mix`] как `[T, H]`.
pub fn pick(mixed: &Tensor, j: usize) -> R<Tensor> {
    let t = mixed.narrow(0, j, 1).map_err(fe)?.contiguous().map_err(fe)?;
    let (tt, h) = (mixed.dims()[1], mixed.dims()[2]);
    t.reshape(vec![tt, h]).map_err(fe)
}

/// Короткая свёртка с остатком: `x [T, C]` F32, `w [C, K]` F32, состояние `[K-1, C]`.
pub fn conv(x: &Tensor, w: &Tensor, state: &Tensor, dynamic: Option<(&Tensor, usize)>, scale: f32) -> R<Tensor> {
    let (t, c) = (x.dims()[0], x.dims()[1]);
    let k = w.dims()[1];
    let out = empty(vec![t, c], DType::F32, x.device())?;
    let mut a = Ink2Launch::default();
    a.a[0] = addr(x)?;
    a.a[1] = addr(&out)?;
    a.a[2] = addr(state)?;
    a.a[3] = addr(w)?;
    if let Some((d, heads)) = dynamic {
        a.a[4] = addr(d)?;
        a.u[3] = heads as u32;
    }
    a.u[0] = t as u32;
    a.u[1] = c as u32;
    a.u[2] = k as u32;
    a.f[0] = scale;
    launch(x.device(), "ink2_conv", &a, ((c as u32).div_ceil(256), 1, 1), 256)?;
    Ok(out)
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
