#[test]
fn resize_bilinear_small() {
    synaptix_kernels_cpu::ensure_registered();
    use synaptix_core::{device::Device, tensor::Tensor};
    let v = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 1, 1, 2, 2],
        Device::Cpu,
    )
    .unwrap();
    let up = synaptix_video_minimax_h3::pipeline::resize_latent_bilinear(&v, 4, 4).unwrap();
    let out: Vec<f32> = up.flatten_all().unwrap().to_vec1().unwrap();
    eprintln!("{:?}", out);
    assert_eq!(out.len(), 16);
    assert!((out[0] - 1.0).abs() < 1e-4);
    assert!((out[3] - 2.0).abs() < 1e-4);
    assert!((out[12] - 3.0).abs() < 1e-4);
    assert!((out[15] - 4.0).abs() < 1e-4);
    assert!((out[5] - (1.0 * 0.75 * 0.75 + 2.0 * 0.25 * 0.75 + 3.0 * 0.75 * 0.25 + 4.0 * 0.25 * 0.25)).abs() < 1e-4);
}
