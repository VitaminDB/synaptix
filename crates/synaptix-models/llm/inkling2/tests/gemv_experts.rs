//! Индексный GEMV экспертов движка на формах Inkling-2 (NVFP4 и MXFP8, gate_up 2304×2304,
//! down 2304×1152, 8 экспертов из `E` на шаг): скорость (ГБ/с по байтам весов выбранных
//! экспертов) и отпечаток выхода для сверки бит в бит между версиями ядра.
//!
//! `cargo test --profile fast-release -p synaptix-llm-inkling2 --test gemv_experts -- --ignored --nocapture`
//! `GEMV_DUMP=f` — записать выходы, `GEMV_CMP=f` — сравнить с записанными.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::{ExpertTable, QuantWeight};
use synaptix_core::tensor::Tensor;

const E: usize = 48;
const H: usize = 2304;
const I: usize = 1152;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.4
        })
        .collect()
}

fn stack(dtype: DType, n: usize, k: usize, seed: u64) -> (Vec<QuantWeight>, usize) {
    use synaptix_kernels_cpu::quant::encode as enc;
    let dev = Device::Cuda(0);
    let ws: Vec<(Vec<u8>, Vec<u8>)> = {
        use rayon::prelude::*;
        (0..E)
            .into_par_iter()
            .map(|e| {
                let w = noise(seed * 1000 + e as u64, n * k);
                match dtype {
                    DType::NVFP4 => enc::quantize_nvfp4_rows(&w, n, k),
                    _ => enc::quantize_mxfp8_rows(&w, n, k),
                }
                .expect("квант")
            })
            .collect()
    };
    let bytes = ws[0].0.len() + ws[0].1.len();
    let q = ws
        .iter()
        .map(|(p, s)| {
            let pt = Tensor::from_raw_slice(p, vec![p.len()], DType::U8, dev).unwrap();
            let st = Tensor::from_raw_slice(s, vec![s.len()], DType::U8, dev).unwrap();
            QuantWeight::new(pt.storage_arc(), st.storage_arc(), dtype, n, k).unwrap()
        })
        .collect();
    (q, bytes)
}

fn raw(t: &Tensor) -> Vec<u8> {
    let t = if t.dtype() == DType::BF16 { t.to_dtype(DType::F32).unwrap() } else { t.clone() };
    t.to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore]
fn gemv_experts_bench() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let dev = Device::Cuda(0);
    let sync = || synaptix_core::device::cuda::synchronize(0).unwrap();
    let mut dump = Vec::new();
    for dtype in [DType::NVFP4, DType::MXFP8] {
        let (gu, gu_b) = stack(dtype, 2 * I, H, 1);
        let (dn, dn_b) = stack(dtype, H, I, 2);
        let gt = ExpertTable::build_dense(&gu.iter().collect::<Vec<_>>()).unwrap();
        let dt = ExpertTable::build_dense(&dn.iter().collect::<Vec<_>>()).unwrap();
        let x = Tensor::from_vec(noise(7, H), vec![1, H], dev).unwrap().to_dtype(DType::BF16).unwrap();
        let hx = Tensor::from_vec(noise(8, 8 * I), vec![8, I], dev).unwrap().to_dtype(DType::BF16).unwrap();
        let w = Tensor::from_vec(noise(9, 8), vec![8], dev).unwrap();
        let steps = 200usize;
        let idxs: Vec<Tensor> = (0..steps)
            .map(|s| Tensor::from_vec((0..8).map(|j| ((s * 11 + j * 5) % E) as u32).collect::<Vec<u32>>(), vec![8], dev).unwrap())
            .collect();
        let mut acc = Tensor::zeros(vec![H], DType::F32, dev).unwrap();
        dump.extend(raw(&gt.gemv_indexed_dense(&idxs[3], &x, false).unwrap()));
        // _acc складывает пары атомарно (порядок пар не задан) — сверяется тело без накопления.
        dump.extend(raw(&dt.gemv_indexed_dense(&idxs[3], &hx, true).unwrap()));
        sync();
        let t = std::time::Instant::now();
        for i in &idxs {
            let _ = gt.gemv_indexed_dense(i, &x, false).unwrap();
        }
        sync();
        let t_gu = t.elapsed().as_secs_f64() / steps as f64;
        let t = std::time::Instant::now();
        for i in &idxs {
            dt.gemv_indexed_dense_acc(i, &hx, true, &w, &mut acc).unwrap();
        }
        sync();
        let t_dn = t.elapsed().as_secs_f64() / steps as f64;
        println!(
            "{dtype:?}: gate_up {:.1} мкс ({:.0} ГБ/с), down+acc {:.1} мкс ({:.0} ГБ/с)",
            t_gu * 1e6,
            8.0 * gu_b as f64 / t_gu / 1e9,
            t_dn * 1e6,
            8.0 * dn_b as f64 / t_dn / 1e9
        );
    }
    if let Ok(f) = std::env::var("GEMV_DUMP") {
        std::fs::write(f, &dump).unwrap();
    }
    if let Ok(f) = std::env::var("GEMV_CMP") {
        let want = std::fs::read(f).unwrap();
        let same = want == dump;
        println!("выход {} с записанным", if same { "совпадает бит в бит" } else { "РАСХОДИТСЯ" });
        assert!(same);
    }
}
