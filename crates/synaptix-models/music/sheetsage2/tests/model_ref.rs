//! Сверка нейросетевой части с питоновским релизом (`#[ignore]`: нужен бандл
//! с компонентом `sheetsage2` и GPU или терпение на CPU).
//!
//! Эталон снят релизом в FP32 на CPU, с весами матриц, округлёнными до BF16
//! ровно так, как они лежат в бандле, — чтобы сравнивать вычисления, а не
//! округление при упаковке. Вход — 20 с синтетики формулой (мелодия по гамме,
//! бас, щелчки на долях), без RNG.
//!
//! ```sh
//! cargo test --release -p synaptix-music-sheetsage2 --test model_ref -- --ignored --nocapture
//! ```
//! Бандл: `SHEETSAGE2_BUNDLE` (по умолчанию `~/Storage/syn_models/yue2-3b.syn`),
//! устройство: `SHEETSAGE2_DEVICE` = `cuda` | `cpu`.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_sheetsage2::SheetSage2;

const SR: usize = 24000;

/// Та же формула, что в эталонном скрипте (numpy, f64 → f32).
fn synth(seconds: f64) -> Vec<f32> {
    let n = (seconds * SR as f64) as usize;
    let scale = [0i64, 2, 4, 5, 7, 9, 11, 12];
    (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let idx = (t / 0.5) as i64;
            let midi = 60 + scale[((idx * 3) % 8) as usize];
            let f = 440.0 * 2f64.powf((midi - 69) as f64 / 12.0);
            let phase_in = t - idx as f64 * 0.5;
            let env = (-3.0 * phase_in).exp();
            let x = 0.3 * env * (2.0 * std::f64::consts::PI * f * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 110.0 * t).sin();
            let click = if phase_in < 0.005 { 0.5 * (2.0 * std::f64::consts::PI * 2000.0 * t).sin() } else { 0.0 };
            (x + click) as f32
        })
        .collect()
}

fn bundle() -> std::path::PathBuf {
    std::env::var("SHEETSAGE2_BUNDLE").map(Into::into).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::PathBuf::from(home).join("Storage/syn_models/yue2-3b.syn")
    })
}

fn device() -> Device {
    match std::env::var("SHEETSAGE2_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::Cuda(0),
    }
}

fn values(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

fn stats(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

const REF_TOKENS: &[u32] = &[1, 4, 5, 6, 7, 9, 11, 3, 260, 517, 30537, 30709, 30966, 30988, 31350, 31586, 31657, 264, 567, 30711, 31591, 31657, 264, 617, 30713, 31597, 31657, 264, 667, 30715, 31588, 31657, 264, 717, 30709, 31374, 31593, 31657, 264, 767, 30711, 31598, 31657, 264, 817, 30713, 31590, 31657, 264, 867, 30715, 31595, 31657, 264, 917, 30709, 31350, 31586, 31657, 264, 967, 30711, 31591, 31657, 264, 1017, 30713, 31597, 31657, 264, 1067, 30715, 31588, 31657, 264, 1117, 30709, 31374, 31593, 31657, 264, 1167, 30711, 31598, 31657, 264, 1217, 30713, 31590, 31657, 264, 1267, 30715, 31595, 31657, 264, 1317, 30709, 31350, 31586, 31657, 264, 1367, 30711, 31591, 31657, 264, 1417, 30713, 31597, 31657, 264, 1467, 30715, 31588, 31657, 264, 1517, 30709, 31374, 31593, 31657, 264, 1567, 30711, 31598, 31657, 264, 1621, 30713, 31590, 31657, 264, 1667, 30715, 31595, 31657, 264, 1721, 30709, 31350, 31586, 31657, 264, 1771, 30711, 31591, 31657, 264, 1817, 30713, 31597, 31657, 264, 1867, 30715, 31588, 31657, 264, 1917, 30709, 31374, 31593, 31657, 264, 1967, 30711, 31598, 31657, 264, 2017, 30713, 31590, 31657, 264, 2063, 30715, 31595, 31657, 264, 2117, 30709, 31350, 31586, 31657, 264, 2171, 30711, 31591, 31657, 264, 2217, 30713, 31597, 31657, 264, 2267, 30715, 31588, 31657, 264, 2317, 30709, 31374, 31593, 31657, 264, 2367, 30711, 31598, 31657, 264, 2417, 30713, 31590, 31657, 264, 2463, 30715, 31595, 31657, 264, 2513, 30709, 31037, 31593, 31657, 264, 2575, 2];

#[test]
#[ignore]
fn matches_release_fp32() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let audio = synth(20.0);
    assert_eq!(&audio[..4], &[0.0, 0.27340880036354065, 0.47972649335861206, 0.5698168873786926]);
    let sum: f64 = audio.iter().map(|&x| x as f64).sum();
    assert!((sum - 21.67341427244162).abs() < 1e-3, "сумма синтетики {sum}");

    let model = SheetSage2::open(bundle(), device(), DType::F32).expect("SheetSage2 из бандла");
    let mut window = audio.clone();
    window.resize(model.config.window_samples(), 0.0);

    // Лог-мел: среднее/СКО и точки.
    let mel = model.mel_window(&window).unwrap();
    assert_eq!(mel.dims(), &[1, 30000, 128]);
    let mv = values(&mel);
    let (mean, std) = stats(&mv);
    eprintln!("mel mean {mean} std {std}");
    assert!((mean - -7.3488426363267125).abs() < 1e-4 && (std - 1.6175012872112442).abs() < 1e-4);
    for (i, j, want) in [(0, 0, 1.633929967880249), (100, 17, 0.9864645600318909), (1999, 127, -1.5521447658538818),
        (500, 64, 1.2688463926315308), (29999, 5, -7.962731838226318)]
    {
        let got = mv[i * 128 + j] as f64;
        assert!((got - want).abs() < 2e-3, "mel[{i},{j}] = {got}, релиз {want}");
    }

    // Память энкодера.
    let memory = model.encode_window(&window).unwrap();
    assert_eq!(memory.dims(), &[1, 7500, 512]);
    let m = values(&memory);
    let (mean, std) = stats(&m);
    eprintln!("memory mean {mean} std {std}");
    assert!((mean - -0.0012530290658778049).abs() < 2e-4 && (std - 0.18802990377706505).abs() < 1e-3);
    for (i, j, want) in [(0, 0, -0.18956035375595093), (10, 100, -0.24920664727687836), (499, 511, -0.08899055421352386),
        (250, 7, -0.29097169637680054), (7499, 300, -0.02961711958050728)]
    {
        let got = m[i * 512 + j] as f64;
        assert!((got - want).abs() < 5e-3, "memory[{i},{j}] = {got}, релиз {want}");
    }

    // Жадная генерация под грамматикой: токены целиком.
    let tokens = model.generate_window(&audio, 20.0).unwrap();
    let first_diff = tokens.iter().zip(REF_TOKENS).position(|(a, b)| a != b);
    eprintln!("токенов {} (релиз {}), первое расхождение {:?}", tokens.len(), REF_TOKENS.len(), first_diff);
    assert_eq!(tokens, REF_TOKENS);
}
