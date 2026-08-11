pub use synaptix_llm_common::{DecodeState, DecoderModel, KvCache, KvCacheLayer, LayerCache, ModelError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::LlamaWeights;
    use std::path::PathBuf;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;
    use synaptix_core::tensor::Tensor;

    fn llama_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/mlx-community/Llama-3.2-1B-Instruct-4bit");
        if p.join("config.json").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn forward_returns_finite_logits() {
        let Some(dir) = llama_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = LlamaWeights::load(&dir, Device::Cpu, DType::F32).expect("load");
        let cfg = w.config.to_decoder_config();
        let model = DecoderModel::build(&cfg, &w, Device::Cpu, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, 64)
            .expect("build model");
        let mut kv = model.make_kv_cache(1, 64).expect("make kv cache");
        let ids = Tensor::from_vec(vec![128000u32, 9906, 1917], vec![1usize, 3], Device::Cpu).unwrap();
        let logits = model.forward(&ids, &mut kv).expect("forward");
        assert_eq!(logits.dims(), &[1, model.config.vocab_size]);
        assert_eq!(kv.seq_len, 3);
        let v = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
        let (argmax, max_v) = v.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(am, mv), (i, &x)| if x > mv { (i, x) } else { (am, mv) },
        );
        eprintln!("[llama fwd] argmax={argmax} max_logit={max_v:.3}");
    }
}
