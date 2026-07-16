use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_diffusion::guidance::cfg::{apply_cfg, Cfg};
use synaptix_diffusion::guidance::Guidance;
use synaptix_ops::rng::Philox4x32;

fn make_tensor(val: f32, shape: &[usize]) -> Tensor {
    synaptix_kernels_cpu::ensure_registered();
    let n: usize = shape.iter().product();
    Tensor::from_vec(vec![val; n], shape.to_vec(), Device::Cpu).unwrap()
}

fn to_flat_f32(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn apply_cfg_zero_scale_returns_uncond() {
    let uncond = make_tensor(1.0, &[1, 4, 8, 8]);
    let cond = make_tensor(3.0, &[1, 4, 8, 8]);
    let result = apply_cfg(&uncond, &cond, 0.0).unwrap();
    for v in to_flat_f32(&result) {
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0 got {v}");
    }
}

#[test]
fn apply_cfg_scale_one_returns_cond() {
    let uncond = make_tensor(1.0, &[1, 4, 8, 8]);
    let cond = make_tensor(3.0, &[1, 4, 8, 8]);
    let result = apply_cfg(&uncond, &cond, 1.0).unwrap();
    for v in to_flat_f32(&result) {
        assert!((v - 3.0).abs() < 1e-5, "expected 3.0 got {v}");
    }
}

#[test]
fn apply_cfg_scale_7_5() {
    let uncond = make_tensor(0.0, &[1, 4]);
    let cond = make_tensor(1.0, &[1, 4]);
    let result = apply_cfg(&uncond, &cond, 7.5).unwrap();
    for v in to_flat_f32(&result) {
        assert!((v - 7.5).abs() < 1e-5, "expected 7.5 got {v}");
    }
}

#[test]
fn cfg_guidance_prepare_doubles_batch() {
    synaptix_kernels_cpu::ensure_registered();
    let g = Cfg::new(7.5);
    let mut rng = Philox4x32::new(1);
    let latent = synaptix_diffusion::schedulers::randn_seeded(&[1, 4, 8, 8], Device::Cpu, &mut rng).unwrap();
    let doubled = g.prepare_latents(&latent).unwrap();
    assert_eq!(doubled.dims()[0], 2);
}
