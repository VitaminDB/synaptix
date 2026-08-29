use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bundle = PathBuf::from(args.next().ok_or("usage: vision_probe <bundle.syn> <image>")?);
    let image = PathBuf::from(args.next().ok_or("нужен путь к картинке")?);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = if synaptix_core::device::cuda::get(0).is_ok() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let mut pipeline = Qwen4ExpPipeline::load_with_precision(
        &bundle,
        device,
        PrecisionConfig::nvfp4(),
        Some(2048),
    )?;
    println!("модель поднята, устройство {device:?}");

    if !pipeline.load_vision(&bundle, DType::F16)? {
        println!("в бандле нет компонента vision");
        return Ok(());
    }
    let tower = pipeline.vision.as_ref().expect("башня");
    println!(
        "башня: {} блоков, hidden {}, выход {}",
        tower.config.depth, tower.config.hidden_size, tower.config.out_hidden_size
    );

    let limits = synaptix_vlm_qwen3::PreprocessLimits::default();
    let tokens = pipeline.image_token_count(&image, limits)?;
    let (feats, grid) = pipeline.encode_image(&image, limits)?;
    let host = feats
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())?;
    let finite = host.iter().all(|x| x.is_finite());
    let mean = host.iter().sum::<f32>() / host.len() as f32;
    let norm = (host.iter().map(|x| x * x).sum::<f32>() / host.len() as f32).sqrt();
    println!(
        "картинка → {tokens} токенов, сетка {grid:?}, эмбеддинги {:?}; все конечны: {finite}, среднее {mean:.4}, СКО {norm:.4}",
        feats.dims()
    );

    let Some(question) = args.next() else { return Ok(()) };
    let pad_id = pipeline
        .config
        .image_token_id
        .ok_or("в конфиге нет id заполнителя картинки")?;
    // Собираем id напрямую: спецтокены в тексте пришлось бы прогонять через
    // decode, а он их выбрасывает.
    let mut ids = pipeline.encode("<|im_start|>user\n")?;
    if let Some(t) = pipeline.config.vision_start_token_id {
        ids.push(t);
    }
    ids.extend(std::iter::repeat(pad_id).take(tokens));
    if let Some(t) = pipeline.config.vision_end_token_id {
        ids.push(t);
    }
    ids.extend(pipeline.encode(&question)?);
    ids.extend(pipeline.encode("<|im_end|>\n<|im_start|>assistant\n")?);
    let pads = ids.iter().filter(|t| **t == pad_id).count();
    println!("промпт {} токенов, заполнителей {pads} (нужно {tokens})", ids.len());
    if pads != tokens {
        return Err("число заполнителей не совпало с числом строк эмбеддингов".into());
    }

    let cfg = synaptix_llm_common::GenerationConfig {
        max_new_tokens: 64,
        temperature: 0.0,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let (out, stats) = pipeline.generate_media_streaming(
        &ids,
        &[synaptix_llm_qwen4_exp::pipeline::MediaInput {
            pad: pad_id,
            embeds: feats,
            grids: vec![grid],
        }],
        cfg,
        &mut |_: u32| true,
    )?;
    println!(
        "ответ ({} токенов за {:.1} с, префилл {} мс):\n{}",
        out.len(),
        started.elapsed().as_secs_f32(),
        stats.prefill_ms,
        pipeline.decode(&out)?
    );
    Ok(())
}
