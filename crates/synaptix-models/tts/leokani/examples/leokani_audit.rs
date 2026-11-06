//! Текст → речь по бандлу; сверка с эталоном (сырые f32 из reference/leokani_ref.py --text).
//! leokani_audit <bundle.syn> "<text>" [--voice dala|pono] [--speed 1] [--device cpu|cuda] [--ref a.f32] [--out a.wav] [--repeat N]
use std::path::Path;
use synaptix_core::device::Device;
use synaptix_tts_leokani::LeokaniPipeline;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let dev = match get("--device").as_deref() { Some("cuda") => Device::Cuda(0), Some("npu") => Device::Npu(0), _ => Device::Cpu };
    synaptix_kernels_cpu::ensure_registered();
    if dev.is_cuda() { synaptix_kernels_cuda::ensure_registered(); }
    if matches!(dev, Device::Npu(_)) { let i = synaptix_kernels_npu::register().expect("npu"); eprintln!("NPU: {i:?}"); }
    let t = std::time::Instant::now();
    let p = LeokaniPipeline::from_syn(Path::new(&a[1]), dev).expect("load");
    eprintln!("загрузка {:.2} с, голоса {:?}", t.elapsed().as_secs_f64(), p.voice_names());
    let voice = get("--voice").unwrap_or_else(|| "dala".into());
    let speed: f32 = get("--speed").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    for c in p.g2p().phonemize_chunks(&a[2]) { eprintln!("  фонемы: {c}"); }
    let mut pcm = Vec::new();
    for _ in 0..get("--repeat").map(|s| s.parse().unwrap()).unwrap_or(1) {
        let t = std::time::Instant::now();
        pcm = p.synthesize(&a[2], &voice, speed).expect("synth");
        let tm = p.last_timings();
        eprintln!("кусков {}, звук {:.2} с за {:.3} с (G2P {:.3}, модель {:.3}), RTF {:.4}", tm.chunks, tm.audio_s, t.elapsed().as_secs_f64(), tm.g2p_s, tm.model_s, t.elapsed().as_secs_f64() / tm.audio_s);
    }
    if let Some(r) = get("--ref") {
        let b = std::fs::read(r).unwrap();
        let rf: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let n = rf.len().min(pcm.len());
        let (mut d, mut s) = (0f64, 0f64);
        for i in 0..n { d += ((pcm[i] - rf[i]) as f64).powi(2); s += (rf[i] as f64).powi(2); }
        eprintln!("эталон {} семплов, наш {}, отн. ошибка {:.2e}", rf.len(), pcm.len(), (d / s).sqrt());
    }
    if let Some(o) = get("--out") { synaptix_audio::io::write_wav_mono_f32(o, &pcm, 24000).unwrap(); }
}
