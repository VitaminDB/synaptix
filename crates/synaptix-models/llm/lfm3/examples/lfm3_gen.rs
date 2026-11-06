//! Жадная генерация LFM3 (сверка с эталоном): lfm3_gen <dir|.syn> "<промпт>" [токенов=32] [f32|bf16|f16] [--ids]
use synaptix_core::{device::Device, dtype::DType, precision::PrecisionConfig};
use synaptix_llm_lfm3::pipeline::{GenerationConfig, Lfm3Pipeline};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    synaptix_kernels_cpu::ensure_registered();
    let dt = match a.get(4).map(String::as_str) {
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        _ => DType::F32,
    };
    let t = std::time::Instant::now();
    let p = Lfm3Pipeline::load_with_precision(&a[1], Device::Cpu, PrecisionConfig::dense(dt), Some(4096)).expect("load");
    eprintln!("загрузка {:.1} с", t.elapsed().as_secs_f64());
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let ids: Vec<u32> = match a.iter().position(|x| x == "--prompt-ids") {
        Some(i) => a[i + 1].split(',').map(|x| x.parse().unwrap()).collect(),
        None => {
            let mut v = vec![p.config.bos_token_id.unwrap_or(1)];
            v.extend(p.encode(&a[2]).expect("enc"));
            v
        }
    };
    if let Some(i) = a.iter().position(|x| x == "--logits") {
        // логиты последней позиции промпта → f32 LE
        let mut kv = p.model.make_kv_cache(1, ids.len() + 1).unwrap();
        let t = synaptix_core::tensor::Tensor::from_vec(ids.clone(), vec![1usize, ids.len()], Device::Cpu).unwrap();
        let lg = p.model.forward(&t, &mut kv).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b: Vec<u8> = lg.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&a[i + 1], b).unwrap();
    }
    let prof = std::env::var_os("LFM_PROF").is_some();
    synaptix_llm_common::model::set_decode_prof(prof);
    let (out, st) = p.generate(&ids, GenerationConfig { max_new_tokens: n, temperature: 0.0, ..Default::default() }).expect("gen");
    if a.iter().any(|x| x == "--ids") {
        println!("{ids:?}\n{out:?}");
    }
    println!("{}", p.decode(&out).unwrap());
    if prof {
        eprintln!("{}", synaptix_llm_common::model::decode_prof_report_and_clear());
    }
    eprintln!("промпт {} ток., префилл {} мс, {} ток. за {} мс ({:.1} ток/с)", st.prompt_tokens, st.prefill_ms, st.new_tokens, st.decode_ms, st.new_tokens as f64 * 1000.0 / st.decode_ms.max(1) as f64);
}
