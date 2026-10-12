//! Распознавание на CPU: `cpu_check <gigaam.syn> <речь.wav> [f32|f16|mxfp8|nvfp4|int8]`.
use synaptix_asr_gigaam::GigaAm;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

fn main() {
    let mut a = std::env::args().skip(1);
    let (model, wav) = (a.next().expect("модель"), a.next().expect("wav"));
    let prec = a.next().unwrap_or_default();
    let (dtype, quant) = match prec.as_str() {
        "f16" => (DType::F16, None),
        "mxfp8" => (DType::F32, Some(DType::MXFP8)),
        "nvfp4" => (DType::F32, Some(DType::NVFP4)),
        "int8" => (DType::F32, Some(DType::Sq { bits: 8 })),
        _ => (DType::F32, None),
    };
    let r = hound::WavReader::open(&wav).expect("wav");
    let spec = r.spec();
    let mx = (1i64 << (spec.bits_per_sample - 1)) as f32;
    let pcm: Vec<f32> = r.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / mx).collect();
    synaptix_kernels_cpu::ensure_registered();
    let m = GigaAm::from_syn_quant(&model, &Device::Cpu, dtype, quant).expect("загрузка");
    let t = std::time::Instant::now();
    let text = m.transcribe_pcm(&pcm, spec.sample_rate).expect("распознавание");
    println!("{:.2} с: «{text}»", t.elapsed().as_secs_f32());
}
