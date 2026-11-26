//! Проба текстового кодера LTX (Gemma 3 / Gemma 4): число и формы скрытых
//! состояний, нормы первых/последних. `ltx_text_probe ENC.syn "промпт"`
use synaptix_core::{device::Device, dtype::DType, precision::PrecisionConfig};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let prec = PrecisionConfig { compute: DType::BF16, attn_w: DType::MXFP8, mlp_w: DType::MXFP8, lm_head: DType::BF16, embed: DType::BF16, kv: DType::BF16 };
    let enc = synaptix_gen::ltx_text::LtxTextEncoder::load(std::path::Path::new(&a[0]), dev, prec, Some(1024)).expect("load");
    let (st, mask) = enc.encode_for_ltx(&a[1], 1024, dev).expect("encode");
    let real = mask.iter().filter(|&&m| m == 1).count();
    println!("состояний {}, форма {:?}, реальных токенов {real}", st.len(), st[0].dims());
    for (i, t) in st.iter().enumerate().filter(|(i, _)| *i < 2 || *i + 2 >= st.len() || i % 12 == 0) {
        let n = t.to_dtype(DType::F32).unwrap().sqr().unwrap().mean().unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0].sqrt();
        println!("  [{i}] rms {n:.4}");
    }
}
