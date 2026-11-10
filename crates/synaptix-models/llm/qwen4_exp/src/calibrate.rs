//! Калибровка квантования экспертов MoE: вторые моменты входов каждой
//! матрицы каждого эксперта и матрицы связей входов слоёв на своём тексте →
//! файл для `synaptix quantize --calib` (важность столбцов и компенсация
//! ошибки, см. `synaptix_core::quant::{calib, gptq}`).
//!
//! Текст режется на окна с чистым кэшем и прогоняется префиллом кусками.
//! Ключи файла: `<стопка>#<эксперт>`, `<стопка>` (среднее по слою — для
//! экспертов, которых роутер на этом тексте не выбрал) и `<стопка>@gram`.
//!
//! Редкий эксперт видит мало строк, и его оценка шумна, поэтому она
//! стягивается к априорной с весом `prior` строк: у `gate_up` — к среднему
//! по слою (вход общий для всех экспертов), у `down` — к ровной (координаты
//! промежуточного вектора у экспертов свои).

use std::collections::BTreeMap;
use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::quant::calib::{self, InputStats};

use crate::Qwen4ExpPipeline;

pub struct CalibOpts {
    /// Сколько токенов текста взять.
    pub tokens: usize,
    /// Окно с чистым кэшем.
    pub window: usize,
    /// Кусок префилла.
    pub chunk: usize,
    /// Вес априорной оценки, строк.
    pub prior: usize,
}

impl Default for CalibOpts {
    fn default() -> Self {
        Self { tokens: 32768, window: 4096, chunk: 1024, prior: 64 }
    }
}

pub struct CalibReport {
    pub tokens: usize,
    /// Матриц экспертов со статистикой.
    pub matrices: usize,
    /// Из них видели меньше 64 строк.
    pub rare: usize,
    /// Матриц связей входов слоёв (под компенсацию ошибки).
    pub grams: usize,
    pub seconds: f64,
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


/// Прогнать `text` через модель `bundle` и записать калибровку в `out`.
/// Ядра устройства должны быть зарегистрированы вызывающим.
pub fn calibrate_experts(bundle: &Path, text: &str, out: &Path, device: Device, opts: &CalibOpts, progress: &dyn Fn(usize, usize)) -> Result<CalibReport, Box<dyn std::error::Error>> {
    let pipeline = Qwen4ExpPipeline::load_with_precision(bundle, device, PrecisionConfig::nvfp4(), Some(opts.window + 64))?;
    let mut ids = pipeline.encode(text)?;
    ids.truncate(opts.tokens);
    let n = ids.len();
    calib::start();
    let t = std::time::Instant::now();
    let mut done = 0usize;
    for win in ids.chunks(opts.window.max(1)) {
        let mut cache = pipeline.make_cache(win.len() + 8)?;
        for part in win.chunks(opts.chunk.max(1)) {
            pipeline.model.forward(part, &mut cache)?;
            done += part.len();
            progress(done, n);
        }
    }
    let mut stats = calib::finish();
    let matrices = stats.len();
    let rare = stats.values().filter(|s| s.rows < 64).count();
    shrink(&mut stats, opts.prior as f64);
    let grams = calib::take_grams();
    let n_grams = grams.len();
    stats.extend(grams);
    calib::save(out, &stats)?;
    Ok(CalibReport { tokens: n, matrices, rare, grams: n_grams, seconds: t.elapsed().as_secs_f64() })
}
