//! index_select на CUDA против CPU: разрозненные индексы идут ядром сбора строк,
//! подряд идущие — одним memcpy; все оси, f32 и bf16.
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn check(dims: Vec<usize>, dim: usize, idx: Vec<i64>, dtype: DType) {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) * 0.25 - 7.0).collect();
    let cpu = Tensor::from_vec(data, dims.clone(), Device::Cpu).unwrap().to_dtype(dtype).unwrap();
    let gpu = cpu.to_device(Device::Cuda(0)).unwrap();
    let ic = Tensor::from_vec(idx.clone(), vec![idx.len()], Device::Cpu).unwrap();
    let ig = ic.to_device(Device::Cuda(0)).unwrap();
    let a = cpu.index_select(dim, &ic).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = gpu.index_select(dim, &ig).unwrap().to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "dims {dims:?} dim {dim} {dtype:?}");
}

#[test]
fn scattered_and_contiguous_all_axes() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let scattered: Vec<i64> = (0..40).map(|k| (k * 7 % 13) as i64).collect();
    for dtype in [DType::F32, DType::BF16] {
        for dim in 0..3 {
            let mut dims = vec![5usize, 6, 7];
            dims[dim] = 13;
            check(dims.clone(), dim, scattered.clone(), dtype);
            check(dims, dim, (3..9).collect(), dtype);
        }
        check(vec![4 * 10 * 10], 0, (0..400).map(|k| (399 - k) as i64).collect(), dtype);
    }
}
