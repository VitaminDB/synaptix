use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;
use synaptix_ops::norm::rms_norm;

pub fn coerr<T>(r: synaptix_core::error::Result<T>) -> Result<T, ModelError> {
    r.map_err(|e| ModelError::Forward(e.to_string()))
}

static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static STAGES: std::sync::Mutex<Vec<(&'static str, u64, u64)>> = std::sync::Mutex::new(Vec::new());

pub fn profiling() -> bool {
    *PROFILE.get_or_init(|| std::env::var("SYN_QWEN4EXP_PROFILE").is_ok())
}

/// Замер этапа forward'а: включается `SYN_QWEN4EXP_PROFILE`. Замеры грубые —
/// очередь CUDA синхронизируется не здесь, а на ближайшей выгрузке на хост
/// (роутер MoE, top-k индексатора), так что доли стоит читать как порядок
/// величины.
pub fn stage<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !profiling() {
        return f();
    }
    let started = std::time::Instant::now();
    let out = f();
    let nanos = started.elapsed().as_nanos() as u64;
    if let Ok(mut stages) = STAGES.lock() {
        match stages.iter_mut().find(|(n, _, _)| *n == name) {
            Some(entry) => {
                entry.1 += nanos;
                entry.2 += 1;
            }
            None => stages.push((name, nanos, 1)),
        }
    }
    out
}

pub fn profile_report() -> String {
    let Ok(mut stages) = STAGES.lock() else {
        return String::new();
    };
    stages.sort_by_key(|(_, nanos, _)| std::cmp::Reverse(*nanos));
    let total: u64 = stages.iter().map(|(_, n, _)| *n).sum();
    let mut out = String::new();
    for (name, nanos, calls) in stages.iter() {
        out.push_str(&format!(
            "  {name}: {:.2} с ({:.1}%), вызовов {calls}\n",
            *nanos as f64 / 1e9,
            100.0 * *nanos as f64 / total.max(1) as f64,
        ));
    }
    stages.clear();
    out
}

pub fn ctx<T>(r: Result<T, ModelError>, what: &str) -> Result<T, ModelError> {
    r.map_err(|e| match e {
        ModelError::Forward(m) => ModelError::Forward(format!("{what}: {m}")),
        ModelError::Shape(m) => ModelError::Shape(format!("{what}: {m}")),
        ModelError::Load(m) => ModelError::Load(format!("{what}: {m}")),
        other => other,
    })
}

pub fn rms(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, ModelError> {
    coerr(rms_norm(x, weight, eps))
}

pub fn group_rms(x: &Tensor, weight: &Tensor, group: usize, eps: f32) -> Result<Tensor, ModelError> {
    let dims = x.dims().to_vec();
    let last = *dims.last().ok_or_else(|| ModelError::Shape("group_rms: скаляр".into()))?;
    if group == 0 || last % group != 0 {
        return Err(ModelError::Shape(format!(
            "group_rms: {last} не делится на группу {group}"
        )));
    }
    if last == group {
        return rms(x, weight, eps);
    }
    let groups = last / group;
    let mut grouped = dims.clone();
    grouped.pop();
    grouped.push(groups);
    grouped.push(group);

    let out_dtype = x.dtype();
    let xf = coerr(x.to_dtype(DType::F32))?;
    let xg = coerr(coerr(xf.reshape(grouped))?.contiguous())?;
    let axis = xg.rank() - 1;
    let var = coerr(coerr(xg.sqr())?.mean_keepdim(axis))?;
    let inv = coerr(coerr(coerr(var.add_scalar(eps))?.sqrt())?.recip())?;
    let normed = coerr(xg.broadcast_mul(&inv))?;
    let flat = coerr(normed.reshape(dims))?;
    let w = coerr(weight.to_dtype(DType::F32))?;
    coerr(coerr(flat.broadcast_mul(&w))?.to_dtype(out_dtype))
}

pub fn load_one_plus(
    weights: &dyn synaptix_llm_common::WeightSource,
    key: &str,
    device: synaptix_core::device::Device,
    dtype: DType,
) -> Result<Tensor, ModelError> {
    let w = weights.tensor(key, device, DType::F32)?;
    let w = w.add_scalar(1.0).map_err(|e| ModelError::Load(e.to_string()))?;
    w.to_dtype(dtype).map_err(|e| ModelError::Load(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;

    #[test]
    fn group_rms_matches_manual() {
        synaptix_kernels_cpu::ensure_registered();
        let x = Tensor::from_vec::<_, f32>(
            vec![1.0, -2.0, 3.0, 0.5, 4.0, -1.0, 2.0, -3.0],
            vec![1, 8],
            Device::Cpu,
        )
        .unwrap();
        let w = Tensor::from_vec::<_, f32>(vec![1.0; 8], vec![8], Device::Cpu).unwrap();
        let out = group_rms(&x, &w, 4, 1e-6).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let src = [1.0f32, -2.0, 3.0, 0.5, 4.0, -1.0, 2.0, -3.0];
        for g in 0..2 {
            let s: f32 = src[g * 4..g * 4 + 4].iter().map(|v| v * v).sum::<f32>() / 4.0;
            let inv = 1.0 / (s + 1e-6).sqrt();
            for i in 0..4 {
                let want = src[g * 4 + i] * inv;
                assert!((got[g * 4 + i] - want).abs() < 1e-5, "{} vs {}", got[g * 4 + i], want);
            }
        }
    }
}
