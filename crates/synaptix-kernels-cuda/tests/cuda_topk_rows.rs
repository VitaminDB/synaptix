//! Top-k по строкам на карте против отбора на процессоре.
//!
//! Проверяется и порядок (по убыванию), и разрешение ничьих: при равных
//! значениях побеждает меньший индекс — иначе выбор экспертов разъедется с
//! прежним поведением роутера.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn reference(row: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    order.sort_by(|a, b| {
        row[*b as usize]
            .partial_cmp(&row[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    order.truncate(k);
    order
}

fn check(rows: usize, cols: usize, k: usize, data: Vec<f32>) {
    let device = Device::Cuda(0);
    let t = Tensor::from_vec(data.clone(), vec![rows, cols], device).expect("вход");
    let (vals, idx) = t.topk_rows(k).expect("top-k");
    let host_idx = idx
        .to_device(Device::Cpu)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<u32>())
        .expect("индексы");
    let host_val = vals
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("значения");

    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let want = reference(row, k);
        let got = &host_idx[r * k..(r + 1) * k];
        assert_eq!(got, &want[..], "строка {r}: индексы разошлись");
        for s in 0..k {
            assert_eq!(host_val[r * k + s], row[got[s] as usize], "строка {r}, слот {s}");
        }
    }
}

#[test]
fn topk_matches_cpu_on_router_shape() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    check(64, 512, 10, noise(7, 64 * 512));
}

#[test]
fn topk_breaks_ties_by_index() {
    if !ready() {
        return;
    }
    // Половина строки — одно и то же значение: выбор обязан взять первые по
    // порядку индексы, как это делает отбор на процессоре.
    let cols = 128;
    let mut data = vec![0.0f32; cols * 2];
    for r in 0..2 {
        for c in 0..cols {
            data[r * cols + c] = if c % 2 == 0 { 1.0 } else { 0.5 };
        }
    }
    check(2, cols, 8, data);
}
