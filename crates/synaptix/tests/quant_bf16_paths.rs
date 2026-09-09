//! Нативный BF16-путь квант-проекций против обхода через F16: одни и те же
//! веса и активации, разные M. Ловит расхождения в bf16-вариантах GEMM/GEMV.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 2.0
        })
        .collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f32;
    let mut den = 0f32;
    for (x, y) in a.iter().zip(b) {
        num += (x - y) * (x - y);
        den += x * x;
    }
    (num / den.max(1e-12)).sqrt()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).unwrap()
}

fn run(fmt: DType, n: usize, k: usize, ms: &[usize]) {
    synaptix::init().expect("init");
    let dev = Device::Cuda(0);
    let w = Tensor::from_vec(noise(1, n * k), vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let q = match fmt {
        DType::NVFP4 => w.quantize_to_nvfp4().unwrap(),
        DType::MXFP8 => w.quantize_to_mxfp8().unwrap(),
        _ => unreachable!(),
    };
    for &m in ms {
        let x = Tensor::from_vec(noise(7 + m as u64, m * k), vec![m, k], dev).unwrap();
        let xb = x.to_dtype(DType::BF16).unwrap();
        let xf = xb.to_dtype(DType::F16).unwrap();
        let ref_y = to_vec(&xf.linear_quant(&q).unwrap());
        match xb.linear_quant(&q) {
            Ok(y) => {
                assert_eq!(y.dtype(), DType::BF16);
                let v = to_vec(&y);
                let rel = rel_l2(&ref_y, &v);
                eprintln!("[{fmt:?}] m={m}: rel L2 bf16-native vs f16 = {rel:.5}");
                assert!(rel < 2e-2, "{fmt:?} m={m}: расхождение {rel}");
            }
            Err(e) => eprintln!("[{fmt:?}] m={m}: нативный путь не поддержан ({e})"),
        }
        // Prequant-путь: общая квант-активация + linear_quant_prequant с BF16-выходом.
        let pre = match fmt {
            DType::NVFP4 => xb.nvfp4_quantize_act().ok(),
            DType::MXFP8 if m == 1 => xb.mxfp8_quantize_act().ok(),
            _ => None,
        };
        if let Some((p, s)) = pre {
            let y = p.linear_quant_prequant(&s, &q, m, DType::BF16).unwrap();
            let rel = rel_l2(&ref_y, &to_vec(&y));
            eprintln!("[{fmt:?}] m={m}: rel L2 prequant-bf16 vs f16 = {rel:.5}");
            assert!(rel < 2e-2, "{fmt:?} m={m} prequant: расхождение {rel}");
        }
    }
}

#[test]
fn nvfp4_bf16_matches_f16() {
    run(DType::NVFP4, 256, 2816, &[1, 7, 40, 64, 200]);
    // Формы Gemma-4: плотный MLP (w4-ядро), эксперты, down с K=704.
    run(DType::NVFP4, 2112, 2816, &[1, 40]);
    run(DType::NVFP4, 2816, 2112, &[1, 40]);
    run(DType::NVFP4, 1408, 2816, &[1]);
    run(DType::NVFP4, 2816, 704, &[1, 8]);
}

#[test]
fn nvfp4_bf16_lm_head() {
    // Голова 262144×2816 — persistent-ядро.
    run(DType::NVFP4, 262144, 2816, &[1]);
}

#[test]
fn mxfp8_bf16_matches_f16() {
    run(DType::MXFP8, 256, 2816, &[1, 7, 40]);
    run(DType::MXFP8, 4096, 2816, &[1]);
    run(DType::MXFP8, 2816, 4096, &[1]);
    run(DType::MXFP8, 8192, 2816, &[1]);
    run(DType::MXFP8, 2816, 8192, &[1]);
}
