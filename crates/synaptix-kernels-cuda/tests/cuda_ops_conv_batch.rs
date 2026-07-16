#![cfg(feature = "cuda")]

//! synaptix-ops conv1d/conv_transpose1d на CUDA при batch>1 (DiT proj_in/out:
//! K=2, stride=2, b=2). Гейтит 2D×3D matmul-broadcast (баг convT ронял cfg=7 в NaN).

use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_ops::conv::{conv1d, conv_transpose1d};

fn check(name: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{name}: len {} vs {}", cpu.len(), gpu.len());
    let mut md = 0f32;
    for i in 0..cpu.len() {
        assert!(gpu[i].is_finite(), "{name}: CUDA NaN/inf at {i}");
        md = md.max((cpu[i] - gpu[i]).abs());
    }
    eprintln!("{name}: CUDA-vs-CPU maxdiff={md:.3e}");
    assert!(md < 2e-3, "{name}: maxdiff {md} (2D×3D broadcast bug?)");
}

fn vec1(t: Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn ops_conv1d_convt_batch2_cuda() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA недоступна — пропуск");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let cpu = Device::Cpu;
    let gpu = Device::Cuda(0);

    // DiT proj_in: conv1d K=2 stride=2 b=2, большой C_in (как hidden).
    let (b, cin, l, cout, k) = (2usize, 320usize, 18usize, 64usize, 2usize);
    let xd: Vec<f32> = (0..b * cin * l).map(|i| ((i % 37) as f32 * 0.11).sin() * 0.5).collect();
    let wd: Vec<f32> = (0..cout * cin * k).map(|i| ((i % 53) as f32 * 0.07).cos() * 0.2).collect();
    let mk = |d: Device| {
        (
            Tensor::from_vec(xd.clone(), vec![b, cin, l], d).unwrap(),
            Tensor::from_vec(wd.clone(), vec![cout, cin, k], d).unwrap(),
        )
    };
    let (xc, wc) = mk(cpu);
    let (xg, wg) = mk(gpu);
    check(
        "conv1d_b2_s2",
        &vec1(conv1d(&xc, &wc, None, 2, 0).unwrap()),
        &vec1(conv1d(&xg, &wg, None, 2, 0).unwrap()),
    );

    // DiT proj_out: conv_transpose1d K=2 stride=2 b=2, groups=1.
    // weight [C_in, C_out, K].
    let (cint, coutt) = (64usize, 32usize);
    let lt = 9usize;
    let xtd: Vec<f32> = (0..b * cint * lt).map(|i| ((i % 41) as f32 * 0.09).sin() * 0.5).collect();
    let wtd: Vec<f32> = (0..cint * coutt * k).map(|i| ((i % 47) as f32 * 0.05).cos() * 0.2).collect();
    let mkt = |d: Device| {
        (
            Tensor::from_vec(xtd.clone(), vec![b, cint, lt], d).unwrap(),
            Tensor::from_vec(wtd.clone(), vec![cint, coutt, k], d).unwrap(),
        )
    };
    let (xtc, wtc) = mkt(cpu);
    let (xtg, wtg) = mkt(gpu);
    check(
        "convT_b2_s2",
        &vec1(conv_transpose1d(&xtc, &wtc, None, 2, 0, 0, 1, 1).unwrap()),
        &vec1(conv_transpose1d(&xtg, &wtg, None, 2, 0, 0, 1, 1).unwrap()),
    );
}
