//! Слова с таймкодами и уверенностью (Whisper, DTW по cross-attn).
//!   words <модель.syn> <аудио.wav> [--lang en] [--verbatim] [--prompt "…"] [--cpu]
use synaptix_asr_whisper::{WhisperPipeline, WordOptions};
use synaptix_core::{device::Device, dtype::DType};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("words <модель.syn> <аудио.wav> [--lang en] [--verbatim] [--prompt \"…\"] [--cpu]");
        std::process::exit(2);
    }
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let (dev, dt) = if a.iter().any(|x| x == "--cpu") { (Device::Cpu, DType::F32) } else { (Device::Cuda(0), DType::F16) };
    let p = WhisperPipeline::from_syn(&a[1], dev, dt).expect("модель");
    let audio = WhisperPipeline::load_audio(&a[2]).expect("аудио");
    let opts = WordOptions { verbatim: a.iter().any(|x| x == "--verbatim"), initial_prompt: arg("--prompt"), snap_to_pauses: !a.iter().any(|x| x == "--no-snap") };
    let t = std::time::Instant::now();
    let r = p.transcribe_words(&audio, arg("--lang").as_deref(), &opts).expect("распознавание");
    let dt = t.elapsed().as_secs_f64();
    println!("язык: {}  ({:.2} с на {:.1} с аудио)", r.language, dt, audio.len() as f64 / 16000.0);
    println!("{}", r.text);
    for w in &r.words {
        println!("{:8.2} {:8.2}  {:5.3} {:5.3}  {}", w.start, w.end, w.prob, w.prob_min, w.text);
    }
}
