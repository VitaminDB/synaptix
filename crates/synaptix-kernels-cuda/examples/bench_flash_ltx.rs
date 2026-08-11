//! Бенч flash-attention на LTX stage2-форме (B=1,H=32,Tq=Tkv=14080,HD=128,
//! bf16, non-causal) через прод-путь Tensor::flash_attention.
//! TFLOP/s = 4·B·H·Tq·Tkv·HD / t. Корректность: малая форма vs naive f32.
//! SYN_BENCH_SHAPE="Tq Tkv H HD" — другая форма.

use std::time::Instant;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn sync() {
    synaptix_core::device::cuda::synchronize(0).unwrap();
}

fn mk(shape: Vec<usize>, dev: Device, seed_mul: f32) -> Tensor {
    Tensor::randn(shape, Device::Cpu)
        .unwrap()
        .mul_scalar(seed_mul)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn naive_ref(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
    // f32 naive: softmax(scale·QKᵀ)·V
    let qf = q.to_dtype(DType::F32).unwrap();
    let kf = k.to_dtype(DType::F32).unwrap();
    let vf = v.to_dtype(DType::F32).unwrap();
    let kt = kf.transpose(2, 3).unwrap().contiguous().unwrap();
    let s = qf.broadcast_matmul(&kt).unwrap().mul_scalar(scale).unwrap();
    let m = s.max_keepdim(3).unwrap();
    let e = s.broadcast_sub(&m).unwrap().exp().unwrap();
    let z = e.sum_keepdim(3).unwrap();
    let p = e.broadcast_div(&z).unwrap();
    p.broadcast_matmul(&vf).unwrap()
}

fn main() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let _ng = synaptix_core::grad::NoGradGuard::new();
    let dev = Device::Cuda(0);

    // ── корректность на малой форме (Tq≥1024 → v2-путь) ──
    {
        let (h, tq, hd) = (4usize, 2048usize, 128usize);
        let q = mk(vec![1, h, tq, hd], dev, 0.5);
        let k = mk(vec![1, h, tq, hd], dev, 0.5);
        let v = mk(vec![1, h, tq, hd], dev, 0.5);
        let scale = 1.0 / (hd as f32).sqrt();
        let o = q.flash_attention(&k, &v, scale, false).unwrap().to_dtype(DType::F32).unwrap();
        let r = naive_ref(&q, &k, &v, scale);
        let d = o.sub(&r).unwrap().abs().unwrap();
        let per_row = d.max([3usize]).unwrap();
        let worst = per_row.max_all().unwrap().to_scalar::<f32>().unwrap();
        let scale_o = r.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        println!("CORRECTNESS small (h={h},t={tq}): per-row max|Δ|={worst:.5} (|ref|max={scale_o:.3})");
        assert!(worst < 0.02, "корректность провалена");
    }

    // ── перф на целевой форме ──
    let (mut tq, mut tkv, mut h, mut hd) = (14080usize, 14080usize, 32usize, 128usize);
    if let Ok(s) = std::env::var("SYN_BENCH_SHAPE") {
        let v: Vec<usize> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if v.len() == 4 {
            tq = v[0]; tkv = v[1]; h = v[2]; hd = v[3];
        }
    }
    let q = mk(vec![1, h, tq, hd], dev, 0.5);
    let k = mk(vec![1, h, tkv, hd], dev, 0.5);
    let v = mk(vec![1, h, tkv, hd], dev, 0.5);
    let scale = 1.0 / (hd as f32).sqrt();
    for _ in 0..3 {
        let o = q.flash_attention(&k, &v, scale, false).unwrap();
        std::hint::black_box(&o);
    }
    sync();
    let iters = 10usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let o = q.flash_attention(&k, &v, scale, false).unwrap();
        std::hint::black_box(&o);
    }
    sync();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flops = 4.0 * (h as f64) * (tq as f64) * (tkv as f64) * (hd as f64);
    println!(
        "FLASH B=1 H={h} Tq={tq} Tkv={tkv} HD={hd} bf16: {:.2} ms  {:.1} TFLOP/s",
        dt * 1e3,
        flops / dt / 1e12
    );
}
