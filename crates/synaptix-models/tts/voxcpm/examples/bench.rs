//! Замер синтеза: `bench <voxcpm2.syn> [f32|f16|bf16|int8] ["(описание)текст"]`.
//! `DEV=npu` — на NPU (счёт F16; `int8` — веса q8row). WAV — в `$OUT`.
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_voxcpm::pipeline::{GenerateOptions, VoxCpmPipeline};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: bench <bundle.syn> [precision] [text]");
    let prec = a.next().unwrap_or_else(|| "f32".into());
    let text = a.next().unwrap_or_else(|| "Привет! Так звучит эта модель с заданными настройками.".into());
    let npu = std::env::var("DEV").as_deref() == Ok("npu");
    let (dtype, quant) = match prec.as_str() {
        "f16" => (DType::F16, None),
        "bf16" => (DType::BF16, None),
        "int8" if npu => (DType::F16, Some(DType::Q8Row)),
        "int8" => (DType::F32, Some(DType::Sq { bits: 8 })),
        _ => (DType::F32, None),
    };
    let (dev, dtype) = if npu {
        synaptix_kernels_npu::register().expect("NPU");
        (Device::Npu(0), DType::F16)
    } else {
        (Device::Cpu, dtype)
    };
    if std::env::var_os("CFG").is_some() {
        let ck = synaptix_tts_voxcpm::loader::VoxCheckpoint::open(&path, Device::Cpu, DType::F32).unwrap();
        println!("{:#?}", ck.config);
        return;
    }
    let t0 = std::time::Instant::now();
    let p = VoxCpmPipeline::from_bundle_quant(&path, dev, dtype, quant).expect("загрузка");
    println!("{prec}: загрузка {:.1} с", t0.elapsed().as_secs_f32());
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    for run in 0..runs {
        let t = std::time::Instant::now();
        // REF=<wav> — клон голоса с записи, PROMPT=<текст записи> — продолжение.
        let w = match (std::env::var("REF"), std::env::var("PROMPT")) {
            (Ok(r), Ok(pt)) => p.synthesize_continuation(&text, &pt, &r, &GenerateOptions::default()),
            (Ok(r), Err(_)) => p.synthesize_with_reference(&text, &r, &GenerateOptions::default()),
            _ => p.synthesize(&text, &GenerateOptions::default()),
        }
        .expect("синтез");
        let sec = w.pcm.len() as f32 / w.sample_rate as f32;
        let dt = t.elapsed().as_secs_f32();
        println!("{prec}: прогон {run}: {sec:.1} с звука за {dt:.1} с (RTF {:.2})", dt / sec);
        if let (0, Ok(out)) = (run, std::env::var("OUT")) {
            let spec = hound::WavSpec { channels: 1, sample_rate: w.sample_rate as u32, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut wr = hound::WavWriter::create(out, spec).unwrap();
            for x in w.pcm {
                wr.write_sample((x.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
            }
            wr.finalize().unwrap();
        }
    }
}
