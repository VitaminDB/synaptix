pub use synaptix_llm_common::{DecodeState, DecoderModel, KvCache, KvCacheLayer, LayerCache, ModelError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::GemmaWeights;
    use std::path::PathBuf;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;
    use synaptix_core::tensor::Tensor;

    fn gemma_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/gemma-3-12b-qat");
        if p.join("config.json").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn forward_returns_finite_logits() {
        if std::env::var("SYN_GEMMA_FWD").is_err() {
            return;
        }
        let Some(dir) = gemma_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = GemmaWeights::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let cfg = w.config.to_decoder_config();
        let model = DecoderModel::build(&cfg, &w, Device::Cpu, DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16, 64)
            .expect("build");
        let mut kv = model.make_kv_cache(1, 64).expect("kv");
        let ids = Tensor::from_vec(vec![2u32, 1841, 563], vec![1usize, 3], Device::Cpu).unwrap();
        let logits = model.forward(&ids, &mut kv).expect("forward");
        assert_eq!(logits.dims(), &[1, model.config.vocab_size]);
        let lf = logits.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(lf.iter().all(|x| x.is_finite()), "logits must be finite");
        let (am, mv) = lf.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(a, m), (i, &x)| if x > m { (i, x) } else { (a, m) },
        );
        eprintln!("[gemma fwd] argmax={am} max_logit={mv:.3}");
    }
}
