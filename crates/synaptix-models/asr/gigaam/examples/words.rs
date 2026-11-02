//! Слова с таймкодами по greedy-CTC: `words <gigaam.syn> <речь.wav> [cuda]`.
use synaptix_asr_gigaam::GigaAm;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

fn main() {
    let mut a = std::env::args().skip(1);
    let (model, wav) = (a.next().expect("модель"), a.next().expect("wav"));
    let cuda = a.next().as_deref() == Some("cuda");
    let r = hound::WavReader::open(&wav).expect("wav");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let mx = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / mx).collect()
        }
    };
    let pcm: Vec<f32> = pcm.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let (dev, dt) = if cuda { (Device::Cuda(0), DType::F16) } else { (Device::Cpu, DType::F32) };
    let m = GigaAm::from_syn(&model, &dev, dt).expect("загрузка");
    let t = std::time::Instant::now();
    let (text, words) = m.transcribe_pcm_words(&pcm, spec.sample_rate).expect("распознавание");
    println!("{:.2} с: «{text}»", t.elapsed().as_secs_f32());
    for w in &words {
        println!("{:7.2} {:7.2}  {:5.3}  {}", w.start, w.end, w.prob, w.text);
    }
}
