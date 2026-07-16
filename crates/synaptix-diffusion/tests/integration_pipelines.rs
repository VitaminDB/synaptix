use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::euler::{EulerConfig, EulerScheduler};
use synaptix_diffusion::pipelines::txt2img::{Txt2ImgConfig, Txt2ImgPipeline};
use synaptix_diffusion::DiffusionError;

fn make_euler() -> EulerScheduler {
    synaptix_kernels_cpu::ensure_registered();
    EulerScheduler::new(EulerConfig::default())
}

#[test]
fn txt2img_run_cfg_produces_correct_shape() {
    let mut scheduler = make_euler();
    let config = Txt2ImgConfig {
        height: 64,
        width: 64,
        latent_channels: 4,
        vae_scale_factor: 8,
        n_steps: 5,
        guidance_scale: 7.5,
        seed: 42,
    };
    let mut pipeline = Txt2ImgPipeline::new(config, &mut scheduler);
    let result = pipeline.run_cfg(
        Device::Cpu,
        DType::F32,
        7.5,
        |input: &Tensor, _sigma: f32| {
            let uncond = input.affine(0.1, 0.0).map_err(DiffusionError::from)?;
            let cond = input.affine(0.2, 0.0).map_err(DiffusionError::from)?;
            Tensor::cat(&[&uncond, &cond], 0).map_err(DiffusionError::from)
        },
        None,
    ).unwrap();

    let dims = result.latents.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 4);
    assert_eq!(dims[2], 8);
    assert_eq!(dims[3], 8);
}

#[test]
fn txt2img_run_with_zero_denoiser_is_finite() {
    let mut scheduler = make_euler();
    let config = Txt2ImgConfig {
        height: 64,
        width: 64,
        latent_channels: 4,
        vae_scale_factor: 8,
        n_steps: 3,
        guidance_scale: 1.0,
        seed: 7,
    };
    let mut pipeline = Txt2ImgPipeline::new(config, &mut scheduler);
    let mut denoiser: synaptix_diffusion::pipelines::DenoiserFn =
        Box::new(|x: &Tensor, _: f32| {
            x.affine(0.0, 0.0).map_err(DiffusionError::from)
        });

    let result = pipeline.run_with_denoiser(Device::Cpu, DType::F32, &mut denoiser, None).unwrap();
    let data = result.latents.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for v in &data {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}
