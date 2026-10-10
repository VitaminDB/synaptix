//! Greedy-распознавание фонем/символов: `recognize <model_dir|.syn> <wav> [cpu|cuda] [f32|f16|bf16]`.

use synaptix_asr_wav2vec2::Wav2Vec2Ctc;
use synaptix_core::{device::Device, dtype::DType};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("model");
    let wav = args.get(2).expect("wav");
    let device = match args.get(3).map(String::as_str) {
        Some("cpu") => {
            synaptix_kernels_cpu::ensure_registered();
            Device::Cpu
        }
        _ => {
            synaptix_kernels_cuda::ensure_registered();
            Device::Cuda(0)
        }
    };
    let dtype = match args.get(4).map(String::as_str) {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(wav).expect("wav");
    let t = std::time::Instant::now();
    let m = Wav2Vec2Ctc::load(model, &device, dtype).expect("load");
    eprintln!("load {:?}", t.elapsed());
    for _ in 0..2 {
        let t = std::time::Instant::now();
        let ph = m.recognize(&pcm, sr).expect("recognize");
        eprintln!("{:?} ({:.2} s audio)", t.elapsed(), pcm.len() as f32 / sr as f32);
        println!("{}", ph.join(" "));
    }
}
