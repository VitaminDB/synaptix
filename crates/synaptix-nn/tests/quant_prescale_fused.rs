//! Слитый предмасштаб NVFP4 (BF16-активация) бит-в-бит с прежней цепочкой
//! abs → max → affine → recip → broadcast_mul → linear_quant → broadcast_mul.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;

fn det(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|i| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (x >> 33) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0;
            // редкие выбросы — как остаточный поток Qwen-Image
            if i % 997 == 0 { v * amp * 1000.0 } else { v * amp }
        })
        .collect()
}

#[test]
fn prescale_fused_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    for (m, k, n, amp) in [(300usize, 1024usize, 512usize, 3.0f32), (1031, 3072, 768, 50.0), (17, 512, 256, 0.01)] {
        let w = Tensor::from_vec(det(1, n * k, 0.05), vec![n, k], dev).unwrap().to_dtype(DType::BF16).unwrap();
        let ql = QuantLinear::build(w, None, DType::NVFP4, DType::BF16).unwrap();
        let QuantLinear::Quant { w: qw, .. } = &ql else { panic!("не квант") };
        let x = Tensor::from_vec(det(2, m * k, amp), vec![m, k], dev).unwrap().to_dtype(DType::BF16).unwrap();
        assert!(x.nvfp4_prescale_quant(64.0).is_ok(), "слитый предмасштаб не работает");
        let y = ql.forward(&x).unwrap();
        // прежняя цепочка
        let scale = x.abs().unwrap().max_keepdim(1).unwrap().max_all().unwrap().to_dtype(DType::F32).unwrap()
            .mul_scalar(1.0 / 64.0).unwrap().add_scalar(f32::MIN_POSITIVE).unwrap().reshape(vec![1]).unwrap();
        let inv = scale.recip().unwrap().to_dtype(DType::BF16).unwrap();
        let sb = scale.to_dtype(DType::BF16).unwrap();
        let yr = x.broadcast_mul(&inv).unwrap().linear_quant(qw).unwrap().broadcast_mul(&sb).unwrap();
        let a: Vec<f32> = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = yr.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        let diff = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        assert_eq!(diff, 0, "m={m} k={k} n={n}: {diff} элементов отличаются");
    }
}

#[test]
fn forward_many_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    for q in [DType::NVFP4, DType::MXFP8] {
        let (m, k) = (300usize, 1024usize);
        let ls: Vec<QuantLinear> = (0..3)
            .map(|i| {
                let w = Tensor::from_vec(det(10 + i, 512 * k, 0.05), vec![512, k], dev).unwrap().to_dtype(DType::BF16).unwrap();
                let b = Tensor::from_vec(det(20 + i, 512, 0.1), vec![512], dev).unwrap().to_dtype(DType::BF16).unwrap();
                QuantLinear::build(w, Some(b), q, DType::BF16).unwrap()
            })
            .collect();
        let x = Tensor::from_vec(det(3, m * k, 5.0), vec![1, m, k], dev).unwrap().to_dtype(DType::BF16).unwrap();
        let refs: Vec<&QuantLinear> = ls.iter().collect();
        let many = QuantLinear::forward_many(&x, &refs).unwrap();
        for (l, y) in ls.iter().zip(&many) {
            let r = l.forward(&x).unwrap();
            assert_eq!(r.dims(), y.dims());
            let a: Vec<f32> = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = r.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "{q:?}: forward_many ≠ forward");
        }
    }
}
