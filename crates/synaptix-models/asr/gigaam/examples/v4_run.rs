//! GigaAM-v4: `v4_run МОДЕЛЬ(.syn | каталог HF) [cuda] ЗАПИСЬ.wav…` — оба текста
//! (raw и e2e), число слов и время.
use synaptix_asr_gigaam::GigaAm4;
use synaptix_core::device::Device;

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).expect("wav");
    let s = r.spec();
    let raw: Vec<f32> = match s.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let k = 1.0 / (1i64 << (s.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.unwrap() as f32 * k).collect()
        }
    };
    let ch = s.channels as usize;
    (raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect(), s.sample_rate)
}

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    let model = a.remove(0);
    let device = if a.first().map(String::as_str) == Some("cuda") {
        a.remove(0);
        synaptix_kernels_cuda::ensure_registered();
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let t0 = std::time::Instant::now();
    let m = GigaAm4::load(&model, &device).expect("модель");
    eprintln!("загрузка {:.2} с", t0.elapsed().as_secs_f64());
    for f in &a {
        let (pcm, sr) = read_wav(f);
        let t = std::time::Instant::now();
        let r = m.transcribe(&pcm, sr).expect("распознавание");
        let (secs, took) = (pcm.len() as f64 / sr as f64, t.elapsed().as_secs_f64());
        println!("== {f}: {secs:.1} с звука за {took:.2} с (×{:.0} быстрее реального времени), слов {}", secs / took.max(1e-9), r.words.len());
        println!("raw: {}", r.raw);
        println!("e2e: {}", r.e2e);
        if let (Some(a), Some(b)) = (r.words.first(), r.words.last()) {
            println!("первое слово «{}» {:.2}–{:.2} с (p {:.2}), последнее «{}» {:.2}–{:.2} с", a.text, a.start, a.end, a.prob, b.text, b.start, b.end);
        }
    }
}
