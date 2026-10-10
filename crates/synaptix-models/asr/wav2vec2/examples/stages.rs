//! Время стадий прямого прохода на CPU.
use std::time::Instant;
use synaptix_asr_wav2vec2::{normalize_audio, Wav2Vec2Model, Wav2Vec2Weights};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let a: Vec<String> = std::env::args().collect();
    let w = Wav2Vec2Weights::from_unpacked(&a[1], Device::Cpu, DType::F32).unwrap();
    let m = Wav2Vec2Model::load(&w).unwrap();
    let (pcm, _) = synaptix_audio::io::read_wav_mono_f32(&a[2]).unwrap();
    let x = normalize_audio(&pcm);
    let n = x.len();
    let x = Tensor::from_vec(x, (1, n), Device::Cpu).unwrap();
    for _ in 0..2 {
        let t = Instant::now();
        let f = m.extract_features(&x).unwrap();
        let t1 = t.elapsed();
        let p = m.project(&f).unwrap();
        let t2 = t.elapsed();
        let (_pos, h) = m.encode_with_pos(&p).unwrap();
        let t3 = t.elapsed();
        let _ = m.head(&h).unwrap();
        let t4 = t.elapsed();
        println!("features {:?}, proj {:?}, encoder {:?}, head {:?}", t1, t2 - t1, t3 - t2, t4 - t3);
    }
}
