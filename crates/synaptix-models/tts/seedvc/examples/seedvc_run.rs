//! Преобразование голоса Seed-VC из бандла:
//! `seedvc_run BUNDLE.syn ИСТОЧНИК.wav ОБРАЗЕЦ.wav OUT.wav [timbre|style|anon|style-anon|v1] [seed] [шагов]`
//! (`v1` — с бандлом `seed-vc-v1.syn`, остальные режимы — `seed-vc-v2.syn`).
//! `SEEDVC_DEVICE=cuda` — на видеокарте (по умолчанию процессор).
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_vc_seed::{pipeline::wav_bytes, SeedVc, SeedVcRequest};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("seedvc_run BUNDLE.syn ИСТОЧНИК.wav ОБРАЗЕЦ.wav OUT.wav [timbre|style|anon|style-anon|v1] [seed] [шагов]");
        std::process::exit(2);
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = if std::env::var("SEEDVC_DEVICE").as_deref() == Ok("cuda") {
        synaptix_kernels_cuda::ensure_registered();
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let (source, source_rate) = synaptix_audio::read_wav_mono_f32(&a[1]).expect("источник");
    let (reference, reference_rate) = synaptix_audio::read_wav_mono_f32(&a[2]).expect("образец");
    let mode = a.get(4).map(String::as_str).unwrap_or("timbre");
    let src_s = source.len() as f64 / source_rate as f64;
    let seed = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(30);
    let mut last = String::new();
    let mut show = |stage: &str, i: usize, n: usize| {
        if stage != last || i % 10 == 0 || i == n {
            eprint!("\r{stage}: {i}/{n}      ");
            last = stage.to_string();
        }
        true
    };
    let t = Instant::now();
    if mode == "v1" {
        let m = synaptix_vc_seed::SeedVcV1::load(&a[0], dev).expect("бандл v1");
        let load_s = t.elapsed().as_secs_f64();
        let req = synaptix_vc_seed::SeedVcV1Request { source, source_rate, reference, reference_rate, seed, steps, ..Default::default() };
        let t = Instant::now();
        let out = m.convert(&req, &mut show).expect("преобразование");
        let total = t.elapsed().as_secs_f64();
        eprintln!();
        std::fs::write(&a[3], wav_bytes(&out.pcm, out.sample_rate)).expect("wav");
        let out_s = out.pcm.len() as f64 / out.sample_rate as f64;
        println!("v1: источник {src_s:.1} с → {out_s:.1} с за {total:.1} с (×{:.2} от реального времени); загрузка {load_s:.1} с, содержание {:.1} с, звук {:.1} с, вокодер {:.1} с → {}", total / out_s.max(1e-9), out.content_seconds, out.cfm_seconds, out.vocoder_seconds, a[3]);
        return;
    }
    let m = SeedVc::load(&a[0], dev).expect("бандл");
    let load_s = t.elapsed().as_secs_f64();
    let req = SeedVcRequest {
        source,
        source_rate,
        reference,
        reference_rate,
        convert_style: mode.starts_with("style"),
        anonymization_only: mode.ends_with("anon"),
        seed,
        steps,
        ..Default::default()
    };
    let t = Instant::now();
    let out = m.convert(&req, &mut show).expect("преобразование");
    let total = t.elapsed().as_secs_f64();
    eprintln!();
    std::fs::write(&a[3], wav_bytes(&out.pcm, out.sample_rate)).expect("wav");
    let out_s = out.pcm.len() as f64 / out.sample_rate as f64;
    println!(
        "{mode}: источник {src_s:.1} с → {out_s:.1} с за {total:.1} с (×{:.2} от реального времени); загрузка {load_s:.1} с, содержание {:.1} с, манера {:.1} с ({} токенов), звук {:.1} с, вокодер {:.1} с → {}",
        total / out_s.max(1e-9),
        out.content_seconds,
        out.ar_seconds,
        out.ar_tokens.len(),
        out.cfm_seconds,
        out.vocoder_seconds,
        a[3]
    );
}
