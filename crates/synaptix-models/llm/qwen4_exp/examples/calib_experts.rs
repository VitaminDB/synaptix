//! Калибровка квантования экспертов: вторые моменты входов каждой матрицы
//! каждого эксперта на своём тексте → файл для `synaptix quantize --calib`.
//!
//!   calib_experts <bundle.syn> <текст.txt> <calib.safetensors> [токенов, 32768]
//!
//! Текст режется на окна (`CALIB_WINDOW`, 4096 токенов) с чистым кэшем и
//! прогоняется префиллом кусками (`CALIB_CHUNK`, 1024). Ключи файла:
//! `<стопка>#<эксперт>` и `<стопка>` (среднее по слою — для экспертов,
//! которых роутер на этом тексте не выбрал).
//!
//! Редкий эксперт видит мало строк, и его оценка шумна, поэтому она
//! стягивается к априорной с весом `CALIB_PRIOR` строк (64): у `gate_up` —
//! к среднему по слою (вход общий для всех экспертов), у `down` — к ровной
//! (координаты промежуточного вектора у экспертов свои).

use std::collections::BTreeMap;
use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::quant::calib::{self, InputStats};
use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn shrink(stats: &mut BTreeMap<String, InputStats>, n0: f64) {
    // Сумма по слою: имя стопки → (Σ sum_sq, Σ rows).
    let mut pooled: BTreeMap<String, InputStats> = BTreeMap::new();
    for (name, s) in stats.iter() {
        let Some((base, _)) = name.split_once('#') else { continue };
        let p = pooled.entry(base.to_string()).or_insert_with(|| InputStats { sum_sq: vec![0.0; s.sum_sq.len()], rows: 0 });
        for (a, b) in p.sum_sq.iter_mut().zip(&s.sum_sq) {
            *a += *b;
        }
        p.rows += s.rows;
    }
    for (name, s) in stats.iter_mut() {
        let Some((base, _)) = name.split_once('#') else { continue };
        let p = &pooled[base];
        let pn = p.rows.max(1) as f64;
        if base.ends_with("gate_up_proj") {
            for (a, b) in s.sum_sq.iter_mut().zip(&p.sum_sq) {
                *a += n0 * b / pn;
            }
        } else {
            let own = s.sum_sq.iter().sum::<f64>() / s.sum_sq.len() as f64 / s.rows.max(1) as f64;
            for a in s.sum_sq.iter_mut() {
                *a += n0 * own;
            }
        }
        s.rows += n0 as u64;
    }
    for (base, p) in pooled {
        if base.ends_with("gate_up_proj") {
            stats.insert(base, p);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: calib_experts <bundle.syn> <текст> <calib.safetensors> [токенов, 32768]".into());
    }
    let bundle = PathBuf::from(&a[1]);
    let text = String::from_utf8_lossy(&std::fs::read(&a[2])?).into_owned();
    let out = PathBuf::from(&a[3]);
    let limit: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(32768);
    let window = env("CALIB_WINDOW", 4096);
    let chunk = env("CALIB_CHUNK", 1024);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pipeline = Qwen4ExpPipeline::load_with_precision(&bundle, Device::Cuda(0), PrecisionConfig::nvfp4(), Some(window + 64))?;
    let mut ids = pipeline.encode(&text)?;
    ids.truncate(limit);
    let n = ids.len();
    eprintln!("токенов {n}, окно {window}, кусок {chunk}");

    calib::start();
    let t = std::time::Instant::now();
    let mut done = 0usize;
    for win in ids.chunks(window) {
        let mut cache = pipeline.make_cache(win.len() + 8)?;
        for part in win.chunks(chunk) {
            pipeline.model.forward(part, &mut cache)?;
            done += part.len();
            eprint!("\r  {done}/{n}  {:.0} с", t.elapsed().as_secs_f64());
        }
    }
    eprintln!();
    let mut stats = calib::finish();
    let seen = stats.len();
    let rows: Vec<u64> = stats.values().map(|s| s.rows).collect();
    let rare = rows.iter().filter(|r| **r < 64).count();
    shrink(&mut stats, env("CALIB_PRIOR", 64) as f64);
    // Матрицы связей входов слоёв (`<стопка>@gram`) — для компенсации ошибки.
    let grams = calib::take_grams();
    let n_grams = grams.len();
    stats.extend(grams);
    eprintln!("матриц связей входов: {n_grams}");
    calib::save(&out, &stats)?;
    println!(
        "калибровка: {} — {seen} матриц экспертов, из них {rare} видели меньше 64 строк; {n} токенов за {:.0} с",
        out.display(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
