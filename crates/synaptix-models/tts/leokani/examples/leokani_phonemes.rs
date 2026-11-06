//! Акустика + вокодер по готовым фонемам (без G2P): сверка с `reference/leokani_ref.py`.
//! leokani_phonemes <model.safetensors> "<фонемы>" <config.json> [--device cpu|cuda] [--ref ref.f32] [--out out.wav] [--speed 1]
use std::path::Path;
use synaptix_core::device::Device;
use synaptix_tts_leokani::model::{Acoustic, Vocoder, W};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let dev = if get("--device").as_deref() == Some("cuda") { Device::Cuda(0) } else { Device::Cpu };
    synaptix_kernels_cpu::ensure_registered();
    if dev.is_cuda() { synaptix_kernels_cuda::ensure_registered(); }
    let l = synaptix_io::SafetensorsLoader::open(Path::new(&a[1])).expect("weights");
    let w = W { l: &l, prefix: String::new(), device: dev, dtype: synaptix_core::dtype::DType::F32 };
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[3]).unwrap()).unwrap();
    let vocab = cfg["vocab"].as_object().unwrap();
    let mut ids = vec![0i64];
    for c in synaptix_tts_leokani::model_clusters(&a[2]) {
        if let Some(v) = vocab.get(&c) { ids.push(v.as_i64().unwrap()); }
    }
    ids.push(0);
    let ac = Acoustic::load(&w, 100).unwrap();
    let voc = Vocoder::load(&w, 8, 100).unwrap();
    let speed: f32 = get("--speed").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let t = std::time::Instant::now();
    let mel = ac.forward(&ids, speed, false).unwrap();
    let pcm = voc.forward(&mel, mel.len() / 100).unwrap();
    eprintln!("ids {}, кадров {}, звук {:.2} с за {:.3} с", ids.len(), mel.len() / 100, pcm.len() as f32 / 24000.0, t.elapsed().as_secs_f64());
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
