use std::time::Instant;
use synaptix_asr_whisper::mel::whisper_log_mel;
use synaptix_asr_whisper::{WhisperEncoder, WhisperPipeline, WhisperWeights};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn audio(arg: Option<&String>) -> R<Vec<f32>> {
    let mut a = match arg {
        Some(p) => WhisperPipeline::load_audio(p)?,
        None => (0..16000 * 10)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.3 * (2.0 * std::f32::consts::PI * (220.0 + 80.0 * t) * t).sin() * (1.0 + (3.0 * t).sin()) * 0.5
            })
            .collect(),
    };
    a.resize(16000 * 30, 0.0);
    Ok(a)
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("encoder_npu <model.syn> <cpu|npu> <out.f32> [эталон.f32] [аудио.wav]");
        std::process::exit(2);
    }
    let (device, dtype) = match args[1].as_str() {
        "npu" => {
            let info = synaptix_kernels_npu::register()?;
            println!("NPU: {} ({})", info.model.name, info.model.id);
            (Device::Npu(0), DType::F16)
        }
        _ => {
            synaptix_kernels_cpu::ensure_registered();
            (Device::Cpu, DType::F32)
        }
    };
    let dtype = match std::env::var("DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("f32") => DType::F32,
        _ => dtype,
    };
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    let t = Instant::now();
    let w = WhisperWeights::open(&args[0], device, dtype)?;
    let enc = WhisperEncoder::load(&w)?;
    let cfg = w.config.clone();
    println!(
        "энкодер: {} слоёв, d_model {}, загрузка {:.1} с",
        cfg.encoder_layers,
        cfg.d_model,
        t.elapsed().as_secs_f64()
    );

    let a = audio(args.get(4))?;
    let (flat, n_mels, n_frames) = whisper_log_mel(&a, cfg.num_mel_bins, cfg.max_source_positions * 2)?;

    let mut out = Vec::new();
    for i in 0..iters {
        let t = Instant::now();
        let mel = Tensor::from_vec(flat.clone(), (1, n_mels, n_frames), device)?.to_dtype(dtype)?;
        let y = match std::env::var("LAYERS").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(n) => enc.forward_partial(&mel, n)?,
            None => enc.forward(&mel)?,
        };
        let v = y.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        println!("прогон {i}: {:.1} мс, выход {:?}", t.elapsed().as_secs_f64() * 1e3, y.dims());
        out = v;
    }
    if device.is_npu() {
        println!("{:?}", synaptix_kernels_npu::stats());
    }
    std::fs::write(&args[2], bytemuck_bytes(&out))?;

    if let Some(r) = args.get(3).filter(|s| s.as_str() != "-") {
        let rb = std::fs::read(r)?;
        let reference: Vec<f32> = rb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if reference.len() != out.len() {
            return Err(format!("эталон {} значений, выход {}", reference.len(), out.len()).into());
        }
        let d = cfg.d_model;
        let mut worst = (1.0f64, 0usize);
        let mut sum = 0f64;
        let mut maxd = 0f32;
        for (row, (a, b)) in out.chunks(d).zip(reference.chunks(d)).enumerate() {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                dot += x as f64 * y as f64;
                na += (x as f64).powi(2);
                nb += (y as f64).powi(2);
                maxd = maxd.max((x - y).abs());
            }
            let c = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
            sum += c;
            if c < worst.0 {
                worst = (c, row);
            }
        }
        let rows = out.len() / d;
        println!(
            "сверка с эталоном по {rows} строкам: худший cos {:.6} (строка {}), средний {:.6}, макс |Δ| {maxd:.4}",
            worst.0,
            worst.1,
            sum / rows as f64
        );
    }
    Ok(())
}

fn bytemuck_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
