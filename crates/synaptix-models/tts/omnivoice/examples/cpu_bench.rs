//! Замер синтеза: `cpu_bench <omnivoice.syn> [f32|f16|bf16|int8|nvfp4|mxfp8] [шаги]`.
//! Печатает время загрузки, синтеза и RTF; WAV — в `$OUT` (если задан).
//! `DEV=npu` — на NPU (счёт F16; `int8` — веса q8row, int8 по строке).
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::config::OmniVoiceGenerationConfig;
use synaptix_tts_omnivoice::pipeline::OmniVoicePipeline;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: cpu_bench <bundle.syn> [precision] [steps]");
    let prec = a.next().unwrap_or_else(|| "f32".into());
    let steps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let (dtype, quant) = match prec.as_str() {
        "f16" => (DType::F16, None),
        "bf16" => (DType::BF16, None),
        "int8" if std::env::var("DEV").as_deref() == Ok("npu") => (DType::F16, Some(DType::Q8Row)),
        "int8" => (DType::F32, Some(DType::Sq { bits: 8 })),
        "q8row" => (DType::F32, Some(DType::Q8Row)),
        "nvfp4" => (DType::F32, Some(DType::NVFP4)),
        "mxfp8" => (DType::F32, Some(DType::MXFP8)),
        _ => (DType::F32, None),
    };
    let npu = std::env::var("DEV").as_deref() == Ok("npu");
    let (dev, dtype) = if npu {
        synaptix_kernels_npu::register().expect("NPU");
        (Device::Npu(0), DType::F16)
    } else {
        (Device::Cpu, dtype)
    };
    let t0 = std::time::Instant::now();
    let p = OmniVoicePipeline::from_syn_quant(&path, dev, dtype, quant).expect("загрузка");
    println!("{prec}: загрузка {:.1} с", t0.elapsed().as_secs_f32());
    // SEED — зерно шума Гумбеля (без него — случайное), PT — температура позиций.
    let mut gen = OmniVoiceGenerationConfig { num_step: steps, seed: std::env::var("SEED").ok().and_then(|v| v.parse().ok()), ..Default::default() };
    if let Some(pt) = std::env::var("PT").ok().and_then(|v| v.parse().ok()) {
        gen.position_temperature = pt;
    }
    let texts = ["Привет! Так звучит эта модель с заданными настройками.", "Сегодня хорошая погода, пойдём гулять в парк у реки."];
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    // CONT=1 — как служба: каждая фраза продолжает голос уже сказанного.
    let mut voice = None;
    for run in 0..runs {
        let text = if std::env::var_os("VARY").is_some() { texts[run % 2] } else { texts[0] };
        let t = std::time::Instant::now();
        let wav = if std::env::var_os("CONT").is_some() {
            p.synthesize_continuing(text, &mut voice, Some("ru"), Some("female, young adult"), 1.0, &gen, std::env::var("REF_SEC").ok().and_then(|v| v.parse().ok()).unwrap_or(6.0)).expect("синтез")
        } else {
            p.synthesize(text, None, Some("ru"), Some("female, young adult"), 1.0, None, &gen).expect("синтез")
        };
        let sec = wav.len() as f32 / p.output_sample_rate() as f32;
        let dt = t.elapsed().as_secs_f32();
        println!("{prec}: прогон {run}: {sec:.1} с звука за {dt:.1} с (RTF {:.2})", dt / sec);
        if let (0, Ok(out)) = (run, std::env::var("OUT")) {
            let spec = hound::WavSpec { channels: 1, sample_rate: p.output_sample_rate() as u32, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::create(out, spec).unwrap();
            for x in wav {
                w.write_sample((x.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
            }
            w.finalize().unwrap();
        }
    }
}
