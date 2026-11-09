//! argmax по строкам на карте против CPU: широкие строки (словарь), ничьи,
//! F32/F16, смещённое представление.
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

#[test]
fn argmax_rows_matches_cpu() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    for (rows, cols) in [(1usize, 248320usize), (3, 1000), (5, 7)] {
        let mut s = 12345u64 + cols as u64;
        let mut v: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
            })
            .collect();
        // ничья: максимум в двух местах — победить обязан меньший индекс
        for r in 0..rows {
            v[r * cols + cols / 3] = 100.0;
            v[r * cols + cols - 1] = 100.0;
        }
        for dt in [DType::F32, DType::F16] {
            let t = Tensor::from_vec(v.clone(), vec![rows, cols], Device::Cpu).unwrap().to_dtype(dt).unwrap();
            let cpu = t.argmax(1).unwrap().to_dtype(DType::U32).unwrap().flatten_all().unwrap().to_vec1::<u32>().unwrap();
            let gpu = t
                .to_device(Device::Cuda(0))
                .unwrap()
                .argmax(1)
                .unwrap()
                .to_device(Device::Cpu)
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(cpu, gpu, "rows={rows} cols={cols} {dt:?}");
            assert!(gpu.iter().all(|i| *i as usize == cols / 3), "ничья: {gpu:?}");
        }
    }
}
