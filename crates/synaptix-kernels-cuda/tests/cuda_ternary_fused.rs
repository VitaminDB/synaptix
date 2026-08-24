use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn have_gpu() -> bool {
    synaptix_core::device::cuda::get(0).is_ok()
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (x >> 33) as u32;
            ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap()
}

fn check(a: &[f32], b: &[f32], dtype: DType, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: длина");
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if dtype != DType::F32 {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: 16-бит путь обязан быть бит-в-бит");
        }
        let d = (x - y).abs() / x.abs().max(y.abs()).max(1.0);
        worst = worst.max(d);
    }
    assert!(worst <= 2.5e-7, "{what}: worst_rel {worst:e} > 2.5e-7");
}

fn make(dims: &[usize], seed: u64, dtype: DType) -> Tensor {
    let n: usize = dims.iter().product();
    Tensor::from_vec(det(seed, n, 0.7), dims.to_vec(), Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .to_device(Device::Cuda(0))
        .unwrap()
}

#[test]
fn mod_flat_matches_decomposed() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let cases: &[&[usize]] = &[&[2, 1536], &[1, 1536], &[8, 64], &[4, 3, 128], &[2, 4608]];
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for dims in cases {
            let x = make(dims, 0x11, dtype);
            let scale = make(dims, 0x22, dtype);
            let shift = make(dims, 0x33, dtype);

            let fused = x.fused_mod_row(&scale, &shift).unwrap();
            let decomposed = x
                .broadcast_mul(&scale.affine(1.0, 1.0).unwrap())
                .unwrap()
                .broadcast_add(&shift)
                .unwrap();
            check(&vals(&fused), &vals(&decomposed), dtype, &format!("mod_flat {dims:?} {dtype:?}"));
        }
    }
}

#[test]
fn mod_row_broadcast_matches_decomposed() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let x = make(&[4, 512], 0x44, dtype);
        let scale = make(&[512], 0x55, dtype);
        let shift = make(&[512], 0x66, dtype);
        let fused = x.fused_mod_row(&scale, &shift).unwrap();
        let decomposed = x
            .broadcast_mul(&scale.affine(1.0, 1.0).unwrap())
            .unwrap()
            .broadcast_add(&shift)
            .unwrap();
        check(&vals(&fused), &vals(&decomposed), dtype, &format!("mod_rowb {dtype:?}"));
    }
}

#[test]
fn gate_residual_matches_decomposed() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let x = make(&[2, 1536], 0x77, dtype);
        let y = make(&[2, 1536], 0x88, dtype);
        let g = make(&[2, 1536], 0x99, dtype);
        let fused = x.fused_gate_residual(&y, &g).unwrap();
        let decomposed = x.add(&g.mul(&y).unwrap()).unwrap();
        check(&vals(&fused), &vals(&decomposed), dtype, &format!("fma_flat {dtype:?}"));
    }
}
