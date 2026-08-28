//! Батч NVFP4-GEMV против тех же умножений по одному.
//!
//! Батчевое ядро выбирает эксперта по `blockIdx.z` и читает его указатели из
//! массива; арифметика внутри та же, что у одиночного GEMV, поэтому расхождений
//! быть не должно вовсе — сверка идёт на точное равенство.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;

const N: usize = 256;
const K: usize = 512;
const EXPERTS: usize = 6;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn cuda_ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

#[test]
fn batched_matches_one_by_one() {
    if !cuda_ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let device = Device::Cuda(0);

    let mut weights: Vec<QuantWeight> = Vec::with_capacity(EXPERTS);
    let mut acts: Vec<(Tensor, Tensor)> = Vec::with_capacity(EXPERTS);
    let mut expected: Vec<f32> = Vec::with_capacity(EXPERTS * N);

    for e in 0..EXPERTS {
        let w = Tensor::from_vec::<_, f32>(noise(1 + e as u64, N * K), vec![N, K], Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F16))
            .and_then(|t| t.to_device(device))
            .expect("вес");
        let qw = w.quantize_to_nvfp4().expect("квант веса");
        let x = Tensor::from_vec::<_, f32>(noise(100 + e as u64, K), vec![1, K], Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F16))
            .and_then(|t| t.to_device(device))
            .expect("активация");

        // Одиночный путь заодно строит перемешанную копию веса — без неё
        // батчевый вызов и не должен работать.
        let single = x.linear_quant(&qw).expect("одиночный GEMV");
        expected.extend(
            single
                .to_device(Device::Cpu)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .unwrap(),
        );
        let (packed, scales) = x.nvfp4_quantize_act().expect("квант активации");
        weights.push(qw);
        acts.push((packed, scales));
    }

    let ws: Vec<&QuantWeight> = weights.iter().collect();
    let av: Vec<(&Tensor, &Tensor)> = acts.iter().map(|(p, s)| (p, s)).collect();
    let rows = vec![0usize; EXPERTS];
    let batched = QuantWeight::gemv_batched(&ws, &av, &rows).expect("батч");
    assert_eq!(batched.dims(), &[EXPERTS, N]);
    let got = batched
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap();

    assert_eq!(got.len(), expected.len());
    let max_abs = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("батч против одиночных: max_abs={max_abs:.3e}");
    assert!(max_abs == 0.0, "батч разошёлся с одиночным GEMV: {max_abs:.3e}");
}

/// Батч читает свою строку из общего кванта активаций: так выходит после
/// фьюза swiglu, посчитанного разом на всех экспертах. Проверяется смещение
/// строки в tile-раскладке масштабов.
#[test]
fn batched_reads_its_own_row_of_shared_activation() {
    if !cuda_ready() {
        return;
    }
    let device = Device::Cuda(0);

    let x_all = Tensor::from_vec::<_, f32>(noise(31, EXPERTS * K), vec![EXPERTS, K], Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F16))
        .and_then(|t| t.to_device(device))
        .expect("активации");
    let (packed_all, scales_all) = x_all.nvfp4_quantize_act().expect("общий квант");

    let mut weights: Vec<QuantWeight> = Vec::with_capacity(EXPERTS);
    let mut expected: Vec<f32> = Vec::with_capacity(EXPERTS * N);
    for e in 0..EXPERTS {
        let w = Tensor::from_vec::<_, f32>(noise(41 + e as u64, N * K), vec![N, K], Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F16))
            .and_then(|t| t.to_device(device))
            .expect("вес");
        let qw = w.quantize_to_nvfp4().expect("квант веса");
        let row = x_all
            .narrow(0, e, 1)
            .and_then(|t| t.contiguous())
            .expect("строка");
        let single = row.linear_quant(&qw).expect("одиночный GEMV");
        expected.extend(
            single
                .to_device(Device::Cpu)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .unwrap(),
        );
        weights.push(qw);
    }

    let ws: Vec<&QuantWeight> = weights.iter().collect();
    let av: Vec<(&Tensor, &Tensor)> = (0..EXPERTS).map(|_| (&packed_all, &scales_all)).collect();
    let rows: Vec<usize> = (0..EXPERTS).collect();
    let batched = QuantWeight::gemv_batched(&ws, &av, &rows).expect("батч");
    let got = batched
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap();

    let max_abs = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("батч по строкам общего кванта: max_abs={max_abs:.3e}");
    assert!(max_abs == 0.0, "строки активации разъехались: {max_abs:.3e}");
}

#[test]
fn batch_without_shuffled_copy_is_refused() {
    if !cuda_ready() {
        return;
    }
    let device = Device::Cuda(0);
    let w = Tensor::from_vec::<_, f32>(noise(7, N * K), vec![N, K], Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F16))
        .and_then(|t| t.to_device(device))
        .expect("вес");
    let qw = w.quantize_to_nvfp4().expect("квант");
    let x = Tensor::from_vec::<_, f32>(noise(8, K), vec![1, K], Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F16))
        .and_then(|t| t.to_device(device))
        .expect("активация");
    let (packed, scales) = x.nvfp4_quantize_act().expect("квант активации");
    let err = QuantWeight::gemv_batched(&[&qw], &[(&packed, &scales)], &[0]);
    assert!(err.is_err(), "батч без перемешанной копии обязан отказать");
}
