//! Сравнение бандлов гибрида Qwen3.5/3.6 по логитам (см.
//! `synaptix_llm_common::logit_eval`): KL к эталону, top-1, перплексия.
//!
//!   logit_kl <bundle.syn> <текст.txt> --save <ref.bin> [токенов]   — эталон
//!   logit_kl <bundle.syn> <текст.txt> --ref  <ref.bin> [токенов]   — сравнение
//!
//! `LOGIT_KL_PRECISION`: `bf16` (эталон из плотного бандла), `mxfp8`, по
//! умолчанию `nvfp4` (у квантованного бандла веса берутся как записаны).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::logit_eval;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let precision = match std::env::var("LOGIT_KL_PRECISION").as_deref() {
        Ok("bf16") => PrecisionConfig::dense(DType::BF16),
        Ok("mxfp8") => PrecisionConfig::mxfp8(),
        _ => PrecisionConfig::nvfp4(),
    };
    logit_eval::cli(&args, &|bundle, limit| {
        let dev = Device::Cuda(0);
        let p = std::rc::Rc::new(HybridPipeline::load_with_precision(bundle, dev, precision, Some(limit + 64))?);
        let vocab = p.config.vocab_size;
        let mut kv = p.model.make_kv_cache(1, limit + 8)?;
        let (pf, pe) = (p.clone(), p);
        Ok((
            Box::new(move |ids: &[u32]| {
                let t = Tensor::from_vec(ids.to_vec(), vec![1usize, ids.len()], dev)?;
                Ok(synaptix_core::grad::no_grad(|| -> Result<Tensor, Box<dyn std::error::Error>> {
                    let h = pf.model.forward_trunk(&t, &mut kv)?;
                    Ok(pf.model.heads_all(&h)?)
                })?)
            }),
            Box::new(move |t: &str| Ok(pe.encode(t)?)),
            vocab,
        ))
    })
}
