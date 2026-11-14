//! Сравнение бандлов Qwen4Exp по логитам (см. `synaptix_llm_common::logit_eval`).
//!
//!   logit_kl <bundle.syn> <текст.txt> --save <ref.bin> [токенов]   — эталон
//!   logit_kl <bundle.syn> <текст.txt> --ref  <ref.bin> [токенов]   — сравнение

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_common::logit_eval;
use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    logit_eval::cli(&args, &|bundle, limit| {
        let p = std::rc::Rc::new(Qwen4ExpPipeline::load_with_precision(bundle, Device::Cuda(0), PrecisionConfig::nvfp4(), Some(limit + 64))?);
        let vocab = p.config.vocab_size;
        let mut cache = p.make_cache(limit + 8)?;
        let (pf, pe) = (p.clone(), p);
        Ok((
            Box::new(move |ids: &[u32]| Ok(pf.model.forward(ids, &mut cache)?)),
            Box::new(move |t: &str| Ok(pe.encode(t)?)),
            vocab,
        ))
    })
}
