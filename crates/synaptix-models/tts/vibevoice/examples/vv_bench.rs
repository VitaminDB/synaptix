use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_vibevoice::generate::SpeechGenerator;
use synaptix_tts_vibevoice::loader::VibeVoiceCheckpoint;
use synaptix_tts_vibevoice::model::VibeVoiceModel;
use synaptix_tts_vibevoice::processor::VibeVoiceProcessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let bundle = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/run/media/storage/syn_models/vibevoice-1.5b.syn".into());
    let dtype = match std::env::args().nth(2).as_deref() {
        Some("f32") => DType::F32,
        Some("f16") => DType::F16,
        _ => DType::BF16,
    };
    let device = Device::Cuda(0);
    let ckpt = VibeVoiceCheckpoint::open(&bundle, device, dtype)?;
    let processor = VibeVoiceProcessor::new(&ckpt.tokenizer_json, &ckpt.preprocessor)?;
    let model = VibeVoiceModel::load(&ckpt, 4096)?;
    let mut gen = SpeechGenerator::new(&model, &processor, 1)?;

    let hidden = model.lm.hidden_size();
    let vae = model.config.acoustic_vae_dim();
    let iters = 40usize;

    let mut lm_cache = model.lm.new_cache(512)?;
    let warm = model.lm.embed_tokens(&[processor.speech_start_id])?;
    let _ = model.lm.forward(&warm, &mut lm_cache)?;

    let embed = model.lm.embed_tokens(&[processor.speech_diffusion_id])?;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = model.lm.forward(&embed, &mut lm_cache)?;
    }
    sync(device);
    println!("lm decode step (1 ветвь): {:.2} ms", ms(t, iters));

    let mut lm_cache_b = model.lm.new_cache(512)?;
    let warm_b = model.lm.embed_tokens(&[processor.speech_start_id])?;
    let _ = model.lm.forward(&warm_b, &mut lm_cache_b)?;
    let single = model.lm.embed_tokens(&[processor.speech_diffusion_id])?;
    let _ = model
        .lm
        .forward_pair(&single, &single, &mut lm_cache, &mut lm_cache_b)?;
    sync(device);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = model
            .lm
            .forward_pair(&single, &single, &mut lm_cache, &mut lm_cache_b)?;
    }
    sync(device);
    println!("lm decode pair (2 ветви): {:.2} ms", ms(t, iters));

    let pos = Tensor::zeros(vec![1usize, hidden], dtype, device)?;
    let neg = Tensor::zeros(vec![1usize, hidden], dtype, device)?;
    let init = Tensor::zeros(vec![1usize, vae], dtype, device)?;
    let _ = gen.sample_latent(&pos, &neg, 1.3, 20, &init)?;
    sync(device);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = gen.sample_latent(&pos, &neg, 1.3, 20, &init)?;
    }
    sync(device);
    println!("diffusion head (20 steps): {:.2} ms", ms(t, iters));

    let mut acache = model.acoustic.new_cache();
    let lat = Tensor::zeros(vec![1usize, 1usize, vae], dtype, device)?;
    let chunk = model.acoustic.decode(&lat, Some(&mut acache))?;
    sync(device);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = model.acoustic.decode(&lat, Some(&mut acache))?;
    }
    sync(device);
    println!("acoustic streaming decode: {:.2} ms", ms(t, iters));

    let mut scache = model.semantic.new_cache();
    let _ = model.semantic.encode(&chunk, Some(&mut scache))?;
    sync(device);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = model.semantic.encode(&chunk, Some(&mut scache))?;
    }
    sync(device);
    println!("semantic streaming encode: {:.2} ms", ms(t, iters));
    Ok(())
}

fn ms(t: Instant, iters: usize) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn sync(device: Device) {
    if let Device::Cuda(_) = device {
        let _ = Tensor::zeros(vec![1usize], DType::F32, device)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>());
    }
}
