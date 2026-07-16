//! dwconv1d/dwconvt1d (CUDA-ядро) vs канальный decompose (CPU-путь ops).
//! Ядро аккумулирует FMA (без раунда произведения), decompose раундит каждый
//! tap отдельным tensor-add → расходятся на ULP (~1e-7); порог 2e-6.
#![cfg(feature = "cuda")]

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered as ensure_cpu;
use synaptix_kernels_cuda::ensure_registered as ensure_cuda;
use synaptix_ops::conv::conv_transpose1d::conv_transpose1d;
use synaptix_ops::conv::depthwise::depthwise_conv;

fn randn(shape: Vec<usize>, dev: Device) -> Tensor {
    Tensor::randn(shape, Device::Cpu).unwrap().mul_scalar(0.3).unwrap().to_device(dev).unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let d = a.to_device(Device::Cpu).unwrap().sub(&b.to_device(Device::Cpu).unwrap()).unwrap().abs().unwrap();
    d.max_all().unwrap().to_scalar::<f32>().unwrap()
}

#[test]
fn dwconv1d_matches_decompose() {
    ensure_cpu();
    ensure_cuda();
    let dev = Device::Cuda(0);
    for (c, l, k, s, pad) in [(768usize, 1002, 12, 2, 0), (33, 517, 12, 2, 3), (16, 240, 5, 1, 2)] {
        let x = randn(vec![1, c, l], dev);
        let w = randn(vec![c, 1, k], dev);
        let b = randn(vec![c], dev);
        let fast = depthwise_conv(&x, &w, Some(&b), s, pad, c).unwrap();
        let xc = x.to_device(Device::Cpu).unwrap();
        let wc = w.to_device(Device::Cpu).unwrap();
        let bc = b.to_device(Device::Cpu).unwrap();
        let refr = depthwise_conv(&xc, &wc, Some(&bc), s, pad, c).unwrap();
        assert_eq!(fast.dims(), refr.dims(), "shape C={c} L={l} K={k}");
        let d = max_abs_diff(&fast, &refr);
        assert!(d < 2e-6, "dwconv C={c} L={l} K={k} s={s} pad={pad}: max|Δ|={d}");
    }
}

#[test]
fn dwconvt1d_matches_decompose() {
    ensure_cpu();
    ensure_cuda();
    let dev = Device::Cuda(0);
    for (c, l, k, s, pad) in [(768usize, 501, 12, 2, 0), (2, 4007, 43, 3, 0), (24, 333, 12, 2, 5)] {
        let x = randn(vec![1, c, l], dev);
        let w = randn(vec![c, 1, k], dev);
        let fast = conv_transpose1d(&x, &w, None, s, pad, 0, c, 1).unwrap();
        let xc = x.to_device(Device::Cpu).unwrap();
        let wc = w.to_device(Device::Cpu).unwrap();
        let refr = conv_transpose1d(&xc, &wc, None, s, pad, 0, c, 1).unwrap();
        assert_eq!(fast.dims(), refr.dims(), "shape C={c} L={l} K={k}");
        let d = max_abs_diff(&fast, &refr);
        assert!(d < 2e-6, "dwconvT C={c} L={l} K={k} s={s} pad={pad}: max|Δ|={d}");
    }
}
