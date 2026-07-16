pub use synaptix_llm_common::{DecodeState, DecoderModel, KvCache, KvCacheLayer, LayerCache, ModelError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::Qwen3Weights;
    use std::path::PathBuf;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;
    use synaptix_core::tensor::Tensor;

    fn qwen3_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/Qwen/Qwen3-1.7B");
        if p.join("config.json").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn forward_single_token_returns_logits() {
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = Qwen3Weights::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let cfg = w.config.to_decoder_config();
        let model = DecoderModel::build(&cfg, &w, Device::Cpu, DType::BF16, DType::BF16, DType::BF16, DType::BF16, 64)
            .expect("build model");
        let mut kv = model.make_kv_cache(1, 64).expect("make kv cache");
        let ids = Tensor::from_vec(vec![151643u32, 9707, 1834], vec![1usize, 3], Device::Cpu).unwrap();
        let logits = model.forward(&ids, &mut kv).expect("forward");
        assert_eq!(logits.dims(), &[1, model.config.vocab_size]);
        assert_eq!(kv.seq_len, 3);
        let lf = logits.to_dtype(DType::F32).unwrap();
        let v = lf.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
        let (argmax, max_v) = v.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(am, mv), (i, &x)| if x > mv { (i, x) } else { (am, mv) },
        );
        eprintln!("[qwen3 fwd] argmax={argmax} max_logit={max_v:.3}");
    }
}
