//! KL по логитам (см. `synaptix_llm_common::logit_eval`):
//!   ink2_kl <модель> <текст.txt> --save <ref.bin> [токенов]   — эталон
//!   ink2_kl <модель> <текст.txt> --ref  <ref.bin> [токенов]   — сравнение

use synaptix_core::device::Device;
use synaptix_llm_common::logit_eval;
use synaptix_llm_inkling2::Inkling2Pipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    logit_eval::cli(&args, &|path, limit| {
        let p = std::rc::Rc::new(Inkling2Pipeline::load(path, Device::Cuda(0))?);
        let vocab = p.config.text.unpadded_vocab_size;
        let mut cache = p.model.make_cache(limit + 8)?;
        let (pf, pe) = (p.clone(), p);
        Ok((
            Box::new(move |ids: &[u32]| Ok(pf.model.forward(ids, &mut cache, true)?)),
            Box::new(move |t: &str| Ok(pe.encode(t)?)),
            vocab,
        ))
    })
}
