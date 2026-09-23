use crate::error::{AudioError, Result};

/// Нулей sinc-ядра по каждую сторону (на частоте среза).
const SINC_ZERO_CROSSINGS: f64 = 16.0;
/// Срез чуть ниже Найквиста целевой частоты — запас под спад окна.
const SINC_ROLLOFF: f64 = 0.95;

/// Ограниченная по полосе передискретизация (windowed-sinc, окно Блэкмана).
///
/// Линейная интерполяция ([`resample_linear`]) не фильтрует: при понижении
/// 48/44.1 кГц → 16 кГц всё выше 8 кГц заворачивается в слышимую полосу
/// (алиасинг), и в ASR/диаризацию/TTS-референс уходит искажённый сигнал.
/// Здесь срез — `0.95 × min(src, dst) / 2`, как у ffmpeg/torchaudio по
/// смыслу. Длина выхода — `round(len × dst/src)`, как у линейной.
pub fn resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(AudioError::invalid_arg("sample rates must be > 0"));
    }
    if src_rate == dst_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    // Ширина полосы в долях частоты дискретизации входа (1.0 = Найквист×2).
    let band = ratio.min(1.0) * SINC_ROLLOFF;
    let half = SINC_ZERO_CROSSINGS / band;
    let kernel = |x: f64| -> f64 {
        let arg = std::f64::consts::PI * band * x;
        let sinc = if arg.abs() < 1e-12 { 1.0 } else { arg.sin() / arg };
        let u = std::f64::consts::PI * x / half;
        sinc * (0.42 + 0.5 * u.cos() + 0.08 * (2.0 * u).cos())
    };
    // Полифазная таблица: позиция выхода i во входе — i·M/L, её дробная
    // часть пробегает L значений. Веса на каждую фазу считаются один раз —
    // иначе sin/cos на каждый отвод: час аудио ресемплился минутами.
    let g = gcd(src_rate, dst_rate);
    let (l, m) = ((dst_rate / g) as usize, (src_rate / g) as usize);
    let phases: Vec<(isize, Vec<f64>)> = (0..l)
        .map(|p| {
            let frac = p as f64 / l as f64;
            let j0 = (frac - half).ceil() as isize;
            let j1 = (frac + half).floor() as isize;
            (j0, (j0..=j1).map(|j| kernel(frac - j as f64)).collect())
        })
        .collect();
    let n = input.len() as isize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i * m;
        let (base, (j0, weights)) = ((pos / l) as isize, &phases[pos % l]);
        let start = base + j0;
        let (mut acc, mut norm) = (0.0f64, 0.0f64);
        let skip = (-start).max(0) as usize;
        let take = weights.len().min((n - start).max(0) as usize);
        for (w, &x) in weights[skip.min(take)..take]
            .iter()
            .zip(&input[(start + skip as isize).min(n) as usize..])
        {
            acc += x as f64 * w;
            norm += w;
        }
        // Нормировка на сумму весов: единичное усиление и на краях, где часть
        // ядра выходит за сигнал.
        out.push(if norm.abs() > 1e-12 { (acc / norm) as f32 } else { 0.0 });
    }
    Ok(out)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(AudioError::invalid_arg("sample rates must be > 0"));
    }
    if src_rate == dst_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let frac = (pos - lo as f64) as f32;
        let v = input[lo] * (1.0 - frac) + input[hi] * frac;
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod sinc_tests {
    use super::*;

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        let n = (rate as f64 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        let body = &x[x.len() / 10..x.len() * 9 / 10];
        (body.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / body.len() as f64).sqrt()
    }

    #[test]
    fn passband_tone_keeps_amplitude() {
        let out = resample(&tone(1000.0, 48_000, 0.5), 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 8000);
        let r = rms(&out);
        assert!((r - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02, "rms {r}");
    }

    /// 10 кГц выше Найквиста 16 кГц: линейная заворачивает его в 6 кГц почти
    /// без потерь, sinc — гасит.
    #[test]
    fn above_nyquist_tone_is_filtered() {
        let x = tone(10_000.0, 48_000, 0.5);
        let sinc = rms(&resample(&x, 48_000, 16_000).unwrap());
        let linear = rms(&resample_linear(&x, 48_000, 16_000).unwrap());
        assert!(sinc < 0.02, "sinc rms {sinc}");
        assert!(linear > 0.3, "linear rms {linear} — тест не показывает алиасинг");
    }

    /// Час аудио 48 → 16 кГц — не минуты (полифазная таблица).
    #[test]
    fn long_input_is_fast() {
        let x = vec![0.25f32; 48_000 * 60];
        let t = std::time::Instant::now();
        let out = resample(&x, 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 16_000 * 60);
        assert!((out[500_000] - 0.25).abs() < 1e-4);
        assert!(t.elapsed().as_secs_f64() < 5.0, "{:?} на минуту аудио", t.elapsed());
    }

    #[test]
    fn odd_ratio_44100_to_16000() {
        let out = resample(&tone(1000.0, 44_100, 0.5), 44_100, 16_000).unwrap();
        assert_eq!(out.len(), 8000);
        assert!((rms(&out) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02);
    }

    #[test]
    fn upsampling_keeps_tone() {
        let out = resample(&tone(440.0, 16_000, 0.5), 16_000, 44_100).unwrap();
        assert_eq!(out.len(), 22_050);
        assert!((rms(&out) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02);
    }
}
