use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn have_gpu() -> bool {
    synaptix_core::device::cuda::get(0).is_ok()
}

fn det(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap()
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: длина {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(d / scale <= tol, "{what}: [{i}] {x} vs {y} (tol {tol})");
    }
}

#[derive(Clone, Copy)]
enum Op {
    Sum,
    Mean,
    Max,
}

fn apply(t: &Tensor, op: Op, dims: &[usize], keepdim: bool) -> Tensor {
    match (op, keepdim) {
        (Op::Sum, false) => t.sum(dims).unwrap(),
        (Op::Sum, true) => t.sum_keepdim(dims[0]).unwrap(),
        (Op::Mean, true) => t.mean_keepdim(dims[0]).unwrap(),
        (Op::Mean, false) => t.mean_keepdim(dims[0]).unwrap(),
        (Op::Max, false) => t.max(dims).unwrap(),
        (Op::Max, true) => t.max_keepdim(dims[0]).unwrap(),
    }
}

fn without_rows_path<R>(f: impl FnOnce() -> R) -> R {
    synaptix_kernels_cuda::kernels::reduce::set_reduce_rows_enabled(false);
    let r = f();
    synaptix_kernels_cuda::kernels::reduce::set_reduce_rows_enabled(true);
    r
}

#[test]
fn rows_path_matches_generic_kernel() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let cases: &[(&[usize], &[usize])] = &[
        (&[2, 1536], &[1]),
        (&[1, 1, 151936], &[2]),
        (&[3, 5, 7], &[1, 2]),
        (&[128, 40, 33], &[2]),
        (&[64, 4096], &[1]),
    ];
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for &(dims, red) in cases {
            let n: usize = dims.iter().product();
            let data = det(0x5EED + n as u64, n);
            let gpu = Tensor::from_vec(data, dims.to_vec(), Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_device(Device::Cuda(0))
                .unwrap();
            let inner: usize = red.iter().map(|d| dims[*d]).product();
            for op in [Op::Sum, Op::Mean, Op::Max] {
                if red.len() > 1 && matches!(op, Op::Mean) {
                    continue;
                }
                if matches!(op, Op::Sum) && dtype == DType::F16 && inner > 4096 {
                    continue;
                }
                let fast = host(&apply(&gpu, op, red, false));
                let generic = without_rows_path(|| host(&apply(&gpu, op, red, false)));
                let tol = if matches!(op, Op::Max) { 0.0 } else { 2e-5 };
                close(
                    &fast,
                    &generic,
                    tol,
                    &format!("rows vs generic {dims:?} red {red:?} {dtype:?}"),
                );
            }
        }
    }
}

#[test]
fn reduce_trailing_axes_matches_cpu() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let cases: &[(&[usize], &[usize])] = &[
        (&[2, 1536], &[1]),
        (&[1, 1, 1536], &[2]),
        (&[7, 13], &[1]),
        (&[3, 5, 7], &[2]),
        (&[3, 5, 7], &[1, 2]),
        (&[1, 1, 151936], &[2]),
        (&[64, 4096], &[1]),
        (&[5, 1], &[1]),
        (&[2, 3, 4, 5], &[3]),
        (&[128, 40, 33], &[2]),
    ];
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let tol = if dtype == DType::F32 { 1e-5 } else { 3e-2 };
        for &(dims, red) in cases {
            let n: usize = dims.iter().product();
            let data = det(0xBEEF + n as u64, n);
            let cpu = Tensor::from_vec(data.clone(), dims.to_vec(), Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let gpu = cpu.to_device(Device::Cuda(0)).unwrap();
            let inner: usize = red.iter().map(|d| dims[*d]).product();
            for op in [Op::Sum, Op::Mean, Op::Max] {
                if red.len() > 1 && matches!(op, Op::Mean) {
                    continue;
                }
                if matches!(op, Op::Sum) && dtype == DType::F16 && inner > 4096 {
                    continue;
                }
                let cpu32 = cpu.to_dtype(DType::F32).unwrap();
                let want = host(&apply(&cpu32, op, red, false));
                let got = host(&apply(&gpu, op, red, false));
                close(&got, &want, tol, &format!("{dims:?} red {red:?} {dtype:?}"));
                if red.len() == 1 {
                    let want_k = host(&apply(&cpu32, op, red, true));
                    let got_k = host(&apply(&gpu, op, red, true));
                    close(
                        &got_k,
                        &want_k,
                        tol,
                        &format!("keepdim {dims:?} red {red:?} {dtype:?}"),
                    );
                }
            }
        }
    }
}

#[test]
fn reduce_leading_and_strided_still_match_cpu() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let dims = vec![6usize, 9usize];
    let n: usize = dims.iter().product();
    let data = det(0xC0FFEE, n);
    let cpu = Tensor::from_vec(data, dims.clone(), Device::Cpu).unwrap();
    let gpu = cpu.to_device(Device::Cuda(0)).unwrap();

    close(
        &host(&gpu.sum([0usize]).unwrap()),
        &host(&cpu.sum([0usize]).unwrap()),
        1e-5,
        "sum по ведущей оси",
    );

    let cpu_n = cpu.narrow(1, 2, 5).unwrap();
    let gpu_n = gpu.narrow(1, 2, 5).unwrap();
    close(
        &host(&gpu_n.sum([1usize]).unwrap()),
        &host(&cpu_n.sum([1usize]).unwrap()),
        1e-5,
        "sum по хвосту narrow-вью",
    );

    let cpu_t = cpu.transpose(0, 1).unwrap();
    let gpu_t = gpu.transpose(0, 1).unwrap();
    close(
        &host(&gpu_t.sum([1usize]).unwrap()),
        &host(&cpu_t.sum([1usize]).unwrap()),
        1e-5,
        "sum по хвосту транспонированного",
    );
}

#[test]
fn reduce_offset_view_matches_cpu() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let dims = vec![8usize, 64usize];
    let n: usize = dims.iter().product();
    let data = det(0xFACE, n);
    let cpu = Tensor::from_vec(data, dims, Device::Cpu).unwrap();
    let gpu = cpu.to_device(Device::Cuda(0)).unwrap();
    let cpu_v = cpu.narrow(0, 3, 4).unwrap();
    let gpu_v = gpu.narrow(0, 3, 4).unwrap();
    close(
        &host(&gpu_v.sum([1usize]).unwrap()),
        &host(&cpu_v.sum([1usize]).unwrap()),
        1e-5,
        "sum со смещением по строкам",
    );
    close(
        &host(&gpu_v.mean_keepdim(1).unwrap()),
        &host(&cpu_v.mean_keepdim(1).unwrap()),
        1e-5,
        "mean_keepdim со смещением",
    );
}

#[test]
fn copy_from_view_over_larger_buffer() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let src = Tensor::from_vec(det(0xA11CE, 4 * 64), vec![4usize, 64], Device::Cpu)
        .unwrap()
        .to_device(Device::Cuda(0))
        .unwrap();
    let view = src.narrow(0, 0, 2).unwrap().contiguous().unwrap();
    assert!(view.is_contiguous(), "narrow по ведущей оси даёт contiguous-вью");

    let mut dst = Tensor::zeros(vec![2usize, 64], DType::F32, Device::Cuda(0)).unwrap();
    dst.copy_from(&view).expect("copy_from вью в меньший буфер");

    let want = host(&view);
    let got = host(&dst);
    close(&got, &want, 0.0, "copy_from вью");

    let mut tail = Tensor::zeros(vec![2usize, 64], DType::F32, Device::Cuda(0)).unwrap();
    let view_tail = src.narrow(0, 2, 2).unwrap();
    tail.copy_from(&view_tail).expect("copy_from вью со смещением");
    let want_tail: Vec<f32> = host(&src)[2 * 64..].to_vec();
    close(&host(&tail), &want_tail, 0.0, "copy_from вью со смещением");
}
