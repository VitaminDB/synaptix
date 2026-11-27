//! Песня MiniMax Music 3 из бандла: `music3_run BUNDLE.syn OUT.wav СЕКУНД SEED "описание" "текст песни"`
//! (`\n` в тексте — перевод строки). `MM3_QUANT=mxfp8|nvfp4` — квант LM на лету.
use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_minimax3::{pipeline::wav_bytes, Music3, Music3Request};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mut m = Music3::open(&a[0], Device::Cuda(0)).expect("бандл");
    m.quant = match std::env::var("MM3_QUANT").as_deref() {
        Ok("mxfp8") => Some(DType::MXFP8),
        Ok("nvfp4") => Some(DType::NVFP4),
        _ => None,
    };
    let req = Music3Request { description: a[4].clone(), lyrics: a[5].replace("\\n", "\n"), seconds: a[2].parse().unwrap(), seed: a[3].parse().unwrap(), ..Default::default() };
    let (ids, _) = m.prompt_ids(&req.description, &req.lyrics).expect("промпт");
    eprintln!("промпт {} токенов", ids.len());
    let mut last = String::new();
    let out = m
        .generate(&req, &mut |stage, i, n| {
            if stage != last || i % 25 == 0 || i == n {
                eprint!("\r{stage}: {i}/{n}      ");
                last = stage.to_string();
            }
            true
        })
        .expect("генерация");
    eprintln!();
    std::fs::write(&a[1], wav_bytes(&out.pcm, out.sample_rate, out.channels)).expect("wav");
    println!(
        "кадров {} ({:.1} с звука): сочинение {:.1} с ({:.1} кадр/с), звук {:.1} с, декодер {:.1} с → {}",
        out.frames,
        out.pcm.len() as f64 / 2.0 / out.sample_rate as f64,
        out.ar_seconds,
        out.frames as f64 / out.ar_seconds,
        out.flow_seconds,
        out.decode_seconds,
        a[1]
    );
}
