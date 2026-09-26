//! Отладка: один шаг декода обычным путём (`forward`) и device-путём
//! (`forward_decode_dev`, без графа) из одинакового состояния — разница
//! логитов при плотном и MXFP8-KV. Гейтед: `STEP_MODEL=<.gguf|.syn>`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;

fn logits_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn dev_step_matches_eager() {
    let Ok(path) = std::env::var("STEP_MODEL") else { return };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    for kv in [DType::F16, DType::MXFP8] {
        let mut prec = PrecisionConfig::dense(DType::F16);
        prec.kv = kv;
        let p = Qwen3Pipeline::load_with_precision(std::path::Path::new(&path), Device::Cuda(0), prec, Some(512)).unwrap();
        let ids = p.encode("List ten animals and one fact about each.").unwrap();
        let l = ids.len();
        let m = &p.model;
        let mut kv_e = m.make_kv_cache(1, 256).unwrap();
        let mut kv_d = m.make_kv_cache(1, 256).unwrap();
        let prompt = Tensor::from_vec(ids.clone(), vec![1, l], Device::Cuda(0)).unwrap();
        let _ = m.forward(&prompt, &mut kv_e).unwrap();
        let lg0 = m.forward(&prompt, &mut kv_d).unwrap();
        let v0 = logits_vec(&lg0);
        let tok0 = v0.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        // Несколько шагов подряд: ошибка может копиться.
        let mut tok = tok0;
        for step in 0..12 {
            let pos = (l + step) as u32;
            let t = Tensor::from_vec(vec![tok], vec![1, 1], Device::Cuda(0)).unwrap();
            let le = logits_vec(&m.forward(&t, &mut kv_e).unwrap());
            let mut st = m.make_decode_state().unwrap();
            st.update(tok, pos).unwrap();
            m.forward_decode_dev(&mut st, &mut kv_d).unwrap();
            kv_d.seq_len = pos as usize + 1;
            synaptix_core::device::cuda::synchronize_all(0).unwrap();
            let ld = logits_vec(&st.logits);
            let maxd = le.iter().zip(&ld).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            eprintln!("kv {kv:?} шаг {step}: max|Δlogit| = {maxd:.5}, argmax {} vs {}", am(&le), am(&ld));
            tok = am(&le) as u32;
        }
    }
}
