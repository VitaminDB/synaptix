//! Прогон башни MiniCPM-V 4.7 по картинке: раскладка, число токенов,
//! заполнитель, нормы эмбеддингов.
//! cargo run --profile fast-release -p synaptix-vlm-minicpm --example encode -- model.syn картинка.png [--device cpu|cuda]
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_vlm_minicpm::{
    load_from_bundle, placeholder, prepare_image, PlaceholderTokens, PreprocessLimits,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: encode MODEL.syn IMAGE [--device cpu|cuda]");
    let image = args.next().expect("usage: encode MODEL.syn IMAGE [--device cpu|cuda]");
    let mut device = "cuda".to_string();
    while let Some(a) = args.next() {
        if a == "--device" {
            device = args.next().unwrap_or_default();
        }
    }
    synaptix_kernels_cpu::ensure_registered();
    let (dev, dtype) = if device == "cpu" {
        (Device::Cpu, DType::F32)
    } else {
        synaptix_kernels_cuda::ensure_registered();
        (Device::Cuda(0), DType::BF16)
    };
    let t0 = std::time::Instant::now();
    let tower = load_from_bundle(&model, dev, dtype).expect("башня");
    eprintln!("башня загружена за {:.2} с ({dev:?}, {dtype:?})", t0.elapsed().as_secs_f32());
    let cfg = &tower.config;
    let mf = cfg.merge_factor();
    let t1 = std::time::Instant::now();
    let prep = prepare_image(&image, cfg, PreprocessLimits::default(), dev).expect("картинка");
    let l = &prep.layout;
    eprintln!(
        "раскладка: миниатюра {}×{} патчей, срезы {:?} × {}×{}, токенов {} (препроцессинг {:.2} с)",
        l.thumb.h,
        l.thumb.w,
        l.slice,
        l.rows,
        l.cols,
        l.tokens(mf),
        t1.elapsed().as_secs_f32()
    );
    let ph = placeholder(l, mf, Some(0), &PlaceholderTokens::default());
    let short = if ph.len() > 200 { format!("{}…{}", &ph[..120], &ph[ph.len() - 60..]) } else { ph.clone() };
    eprintln!("заполнитель ({} символов): {short}", ph.len());
    let t2 = std::time::Instant::now();
    let emb = tower.forward_image(&prep).expect("forward");
    let _ = emb.to_device(Device::Cpu);
    eprintln!("эмбеддинги {:?} за {:.2} с", emb.dims(), t2.elapsed().as_secs_f32());
    let rows = emb.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
    for (i, r) in rows.iter().enumerate().take(3) {
        let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("строка {i}: норма {norm:.4}, первые {:?}", &r[..6.min(r.len())]);
    }
    let last = rows.len() - 1;
    let norm = rows[last].iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("строка {last}: норма {norm:.4}, первые {:?}", &rows[last][..6.min(rows[last].len())]);
}
