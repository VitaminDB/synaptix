//! Проверка головы многотокенного предсказания на живой модели.
//!
//! В transformers этой головы нет, схема восстановлена по раскладке весов,
//! поэтому единственная честная проверка — доля совпадений с основной моделью:
//! на каждом шаге декода голова получает поток последнего слоя и эмбеддинг
//! только что выбранного токена, а её ответ сравнивается со следующим токеном,
//! который выберет сама модель. Случайное совпадение на словаре в четверть
//! миллиона — доли процента, так что даже скромная доля отличает работающую
//! схему от неверной.

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen4_exp::mtp::{present, MtpHead};
use synaptix_llm_qwen4_exp::{Qwen4ExpPipeline, Qwen4ExpWeights};

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn host(t: &synaptix_core::tensor::Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .expect("на хост")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bundle = PathBuf::from(args.next().ok_or("usage: mtp_probe <bundle.syn> [промпт] [шагов]")?);
    let prompt = args.next().unwrap_or_else(|| "Кратко объясни, что такое энтропия.".into());
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = if synaptix_core::device::cuda::get(0).is_ok() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let precision = PrecisionConfig::nvfp4();

    let pipeline =
        Qwen4ExpPipeline::load_with_precision(&bundle, device, precision, Some(4096))?;
    let cfg = pipeline.config.clone();

    let weights = Qwen4ExpWeights::open(&bundle, device, precision.compute)?;
    if !present(&weights) {
        println!("в бандле нет головы многотокенного предсказания");
        return Ok(());
    }
    let head = MtpHead::load(
        &weights,
        &cfg,
        device,
        precision.compute,
        precision.mlp_w,
        None,
        None,
        cfg.num_hidden_layers,
    )?;
    println!("голова поднята");

    let ids = pipeline.encode(&prompt)?;
    let mut cache = pipeline.make_cache(ids.len() + steps + 8)?;
    let mut head_cache = head.make_cache(&cfg, ids.len() + steps + 8, device, precision.compute)?;
    let rope = pipeline.model.rope();

    let (mut hidden, mut stream) = pipeline.model.forward_with_stream(&ids, &mut cache)?;
    let mut hits = 0usize;
    let mut checked = 0usize;
    let mut guess: Option<u32> = None;

    for _ in 0..steps {
        let last = hidden.dims()[0] - 1;
        let h_last = hidden.narrow(0, last, 1)?.contiguous()?;
        let token = argmax(&host(&pipeline.model.lm_head_forward(&h_last)?));

        if let Some(g) = guess.take() {
            checked += 1;
            if g == token {
                hits += 1;
            }
        }
        if cfg.eos_token_ids.contains(&token) {
            break;
        }

        // Голова смотрит на поток последней позиции и на только что выбранный токен.
        let s_last = stream.narrow(0, last, 1)?.contiguous()?;
        let embed = pipeline.model.embed_tokens(&[token])?;
        let head_hidden = head.forward(&s_last, &embed, &mut head_cache, rope)?;
        guess = Some(argmax(&host(&pipeline.model.lm_head_forward(&head_hidden)?)));

        let next = pipeline.model.forward_with_stream(&[token], &mut cache)?;
        hidden = next.0;
        stream = next.1;
    }

    println!(
        "совпадений {hits} из {checked} ({:.0}%)",
        if checked > 0 { 100.0 * hits as f32 / checked as f32 } else { 0.0 }
    );
    Ok(())
}
