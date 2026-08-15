//! Диагностика источника MiniMax-H3: каталог или `.syn`-бандл.
//!
//! Печатает вариант, найденные компоненты, разобранные конфиги и размер
//! каждого набора весов — без загрузки тензоров на устройство.
//!
//! Запуск:
//!     cargo run --release -p synaptix-video-minimax-h3 --example inspect_source -- <path> [encoder_path]

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_video_minimax_h3 as h3;
use h3::source::{H3Component, H3EncoderSource, H3Source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or("использование: inspect_source <path> [encoder]")?);
    let encoder_arg = args.next().map(PathBuf::from);

    let src = H3Source::open(&path, h3::H3Variant::Fl2va)?;
    println!("источник:  {} ({})", src.path().display(), if src.is_bundle() { "syn-бандл" } else { "каталог" });
    println!("вариант:   {:?}", src.variant());

    let comps = [
        H3Component::Transformer,
        H3Component::VideoVae,
        H3Component::AudioVae,
        H3Component::TextEncoder,
    ];
    for c in comps {
        if !src.has_component(c) {
            println!("компонент {:<13} —", c.name());
            continue;
        }
        let loader = src.loader(c, Device::Cpu)?;
        let (n, bytes) = summarize(&loader);
        println!("компонент {:<13} {n} тензоров, {:.2} ГБ", c.name(), bytes as f64 / 1e9);
    }

    let cfg = h3::H3Config::from_source(&src)?;
    println!(
        "DiT:       hidden={} layers={} heads={} partition={:?}",
        cfg.hidden_size, cfg.num_layers, cfg.num_attention_heads, cfg.partition
    );
    let vae = h3::VaeConfig::from_source(&src)?;
    println!("video VAE: z_channels={} clip_length={}", vae.z_channels, vae.clip_length);
    let avae = h3::AudioVaeConfig::from_source(&src)?;
    println!(
        "audio VAE: latent_channels={} sample_rate={}",
        avae.vae_latent_channels, avae.sampling_rate
    );

    let enc = match &encoder_arg {
        Some(p) => Some(H3EncoderSource::open(p)?),
        None => H3EncoderSource::from_model(&src),
    };
    match enc {
        Some(e) => {
            let loader = e.loader(Device::Cpu)?;
            let (n, bytes) = summarize(&loader);
            let cfg_len = e.read("config.json")?.len();
            let tok_len = e.read("tokenizer.json")?.len();
            println!(
                "энкодер:   {} — {n} тензоров, {:.2} ГБ (config {cfg_len} Б, tokenizer {tok_len} Б)",
                e.path().display(),
                bytes as f64 / 1e9
            );
        }
        None => println!("энкодер:   не найден (укажите отдельный .syn/каталог)"),
    }
    Ok(())
}

fn summarize(loader: &synaptix_io::weights::safetensors::SafetensorsLoader) -> (usize, usize) {
    let mut n = 0;
    let mut bytes = 0;
    for (_, dt, shape) in loader.infos() {
        n += 1;
        bytes += dt.bytes_for_numel(shape.iter().product::<usize>());
    }
    (n, bytes)
}
