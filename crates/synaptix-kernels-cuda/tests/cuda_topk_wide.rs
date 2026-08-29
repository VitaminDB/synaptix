//! Top-k по широким строкам против отбора на процессоре.
//!
//! Порядок индексов в выдаче не определён (порог ищется radix-select'ом, а
//! равные порогу добираются атомарным счётчиком), поэтому сверяется набор:
//! он обязан совпасть с k наибольшими значениями строки.

use std::collections::HashSet;

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Скоры индексатора идут после relu: половина обнуляется.
            let v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.4;
            v.max(0.0)
        })
        .collect()
}

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn check(rows: usize, cols: usize, k: usize, valid_of: impl Fn(usize) -> usize) {
    let device = Device::Cuda(0);
    let data = noise(11, rows * cols);
    let valid: Vec<u32> = (0..rows).map(|r| valid_of(r) as u32).collect();
    let scores = Tensor::from_vec(data.clone(), vec![rows, cols], device).expect("скоры");
    let valid_t = Tensor::from_vec(valid.clone(), vec![rows], device).expect("valid");
    let out = scores.topk_wide(&valid_t, k).expect("top-k");
    let host = out
        .to_device(Device::Cpu)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<u32>())
        .expect("индексы");

    for r in 0..rows {
        let n = valid[r] as usize;
        let row = &data[r * cols..r * cols + n];
        let got: Vec<u32> = host[r * k..(r + 1) * k].to_vec();
        let filled: Vec<u32> = got.iter().copied().filter(|i| *i != u32::MAX).collect();
        assert!(filled.iter().all(|i| (*i as usize) < n), "строка {r}: индекс вне валидной части");
        let unique: HashSet<u32> = filled.iter().copied().collect();
        assert_eq!(unique.len(), filled.len(), "строка {r}: повтор индекса");

        if n <= k {
            assert_eq!(filled.len(), n, "строка {r}: короткая строка обязана войти целиком");
            continue;
        }
        assert_eq!(filled.len(), k, "строка {r}: слотов заполнено {}", filled.len());
        let mut sorted: Vec<f32> = row.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[k - 1];
        // Порог ищется по старшим двадцати четырём битам, поэтому у самой
        // границы выбор может взять соседа с почти тем же значением.
        let chosen_min = filled.iter().map(|i| row[*i as usize]).fold(f32::INFINITY, f32::min);
        let slack = threshold.abs() * 1e-4 + 1e-6;
        assert!(
            chosen_min >= threshold - slack,
            "строка {r}: взят элемент {chosen_min} ниже порога {threshold}"
        );
    }
}

#[test]
fn wide_topk_matches_cpu_threshold() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    check(64, 4096, 512, |_| 4096);
}

#[test]
fn wide_topk_respects_valid_prefix() {
    if !ready() {
        return;
    }
    // У ранних запросов виден лишь префикс контекста, а у самых первых
    // кандидатов меньше, чем слотов.
    check(32, 2048, 128, |r| 40 + r * 60);
}
