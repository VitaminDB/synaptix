use std::path::PathBuf;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::moe::{ExpertCache, MoeFfn};
use synaptix_llm_qwen4_exp::model::LM_PREFIX;
use synaptix_llm_qwen4_exp::Qwen4ExpWeights;

fn model_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        std::env::var("SYN_QWEN4EXP_MODEL")
            .unwrap_or_else(|_| "/home/master/models/Qwen/Qwen3.8-Flash-Next".to_string()),
    );
    p.join("config.json").exists().then_some(p)
}

/// Прогон настоящего MoE-слоя (512 экспертов) с весами в системной памяти и
/// кэшем на карте. Тяжёлый: читает ~5 ГБ стопки и квантует её поматрично,
/// поэтому включается только по `SYN_QWEN4EXP_CUDA_MOE=1`.
#[test]
fn real_moe_layer_offloads_to_host() {
    if std::env::var("SYN_QWEN4EXP_CUDA_MOE").is_err() {
        eprintln!("SYN_QWEN4EXP_CUDA_MOE не задан — пропуск");
        return;
    }
    let Some(dir) = model_dir() else {
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);

    let weights = Qwen4ExpWeights::open(&dir, device, DType::F16).expect("open");
    let cfg = weights.config.clone();
    let mut moe_cfg = cfg.moe.clone();
    moe_cfg.chunk = 64;

    let cache_bytes = std::env::var("SYN_QWEN4EXP_EXPERT_CACHE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0)
        * (1u64 << 30) as f64;
    let cache = ExpertCache::new(device, cache_bytes as usize);

    let t0 = Instant::now();
    let moe = MoeFfn::load_offloaded(
        &weights,
        &format!("{LM_PREFIX}.layers.0.mlp"),
        moe_cfg,
        device,
        DType::F16,
        DType::NVFP4,
        cache.clone(),
        0,
    )
    .expect("load offloaded");
    eprintln!("загрузка слоя: {:.1} с", t0.elapsed().as_secs_f32());

    let tokens = 64usize;
    let x = Tensor::randn(vec![tokens, cfg.hidden_size], Device::Cpu)
        .and_then(|t| t.mul_scalar(0.05))
        .and_then(|t| t.to_dtype(DType::F16))
        .and_then(|t| t.to_device(device))
        .expect("активации");

    let t1 = Instant::now();
    let out = moe.forward(&x).expect("forward");
    let first = t1.elapsed();
    let t2 = Instant::now();
    let again = moe.forward(&x).expect("forward");
    let second = t2.elapsed();

    assert_eq!(out.dims(), &[tokens, cfg.hidden_size]);
    let v = again
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "в выходе MoE не число");
    assert!(v.iter().any(|x| *x != 0.0));

    let stats = cache.stats();
    eprintln!(
        "MoE слой 0: {tokens} токенов — первый проход {:.0} мс, повтор {:.0} мс; \
         кэш {} экспертов ({:.2} ГБ), попаданий {}, промахов {}",
        first.as_secs_f32() * 1000.0,
        second.as_secs_f32() * 1000.0,
        stats.resident,
        stats.bytes as f64 / (1 << 30) as f64,
        stats.hits,
        stats.misses
    );
    assert!(stats.misses > 0);
    assert!(stats.bytes <= cache_bytes as usize);
}
