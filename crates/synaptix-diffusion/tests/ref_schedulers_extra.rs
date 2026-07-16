use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_diffusion::schedulers::{
    dpm_pp_3m::{DpmPp3MConfig, DpmPp3MScheduler},
    dpm_pp_sde::{DpmPpSdeConfig, DpmPpSdeScheduler},
    edm::{EdmConfig, EdmScheduler},
    euler::SigmaSchedule,
    heun::{HeunConfig, HeunScheduler},
    lcm::{LcmConfig, LcmScheduler},
    pndm::{PndmConfig, PndmScheduler},
    tcd::{TcdConfig, TcdScheduler},
    unipc::{UniPcConfig, UniPcScheduler},
    BetaConfig, BetaSchedule, Scheduler, TimestepSpacing,
};
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::load_case;

fn setup() { ensure_registered(); }

fn linear_beta() -> BetaConfig {
    BetaConfig {
        num_train_timesteps: 1000,
        beta_start: 0.0001,
        beta_end: 0.02,
        schedule: BetaSchedule::Linear,
        rescale_zero_snr: false,
    }
}

fn scaled_linear_beta() -> BetaConfig {
    BetaConfig {
        num_train_timesteps: 1000,
        beta_start: 0.00085,
        beta_end: 0.012,
        schedule: BetaSchedule::ScaledLinear,
        rescale_zero_snr: false,
    }
}

#[test]
fn t30_1_heun_init_noise_sigma() {
    setup();
    let t = load_case("diffusion_extra", "heun");
    let mut sch = HeunScheduler::new(HeunConfig {
        beta: linear_beta(),
        spacing: TimestepSpacing::Linspace,
        sigma_schedule: SigmaSchedule::BetaSchedule,
        use_scale_model_input: true,
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();
    let ref_sigmas: Vec<f32> = t["sigmas"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let actual_init = sch.init_noise_sigma();
    let ref_init = ref_sigmas[0];
    assert!((actual_init - ref_init).abs() < 1e-2,
        "init_noise_sigma mismatch: actual={:.4} expected={:.4}", actual_init, ref_init);
}

#[test]
fn t30_2_unipc_sigmas() {
    setup();
    let t = load_case("diffusion_extra", "unipc");
    let mut sch = UniPcScheduler::new(UniPcConfig {
        beta: linear_beta(),
        spacing: TimestepSpacing::Leading,
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();
    let _ = &t;
    assert_eq!(sch.sigmas().len(), 21);
}

#[test]
fn t30_3_pndm_sigmas() {
    setup();
    let _t = load_case("diffusion_extra", "pndm");
    let mut sch = PndmScheduler::new(PndmConfig {
        beta: linear_beta(),
        spacing: TimestepSpacing::Leading,
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();
    assert!(!sch.sigmas().is_empty());
}

#[test]
fn t30_4_edm_set_timesteps() {
    setup();
    let _t = load_case("diffusion_extra", "edm");
    let mut sch = EdmScheduler::new(EdmConfig::default());
    sch.set_timesteps(20).unwrap();
    let sigmas = sch.sigmas();
    assert_eq!(sigmas.len(), 21);
    assert!(sigmas[0] > sigmas[sigmas.len() - 2]);
}

#[test]
fn t30_5_lcm_sigmas() {
    setup();
    let _t = load_case("diffusion_extra", "lcm");
    let mut sch = LcmScheduler::new(LcmConfig {
        beta: scaled_linear_beta(),
        ..Default::default()
    });
    sch.set_timesteps(4).unwrap();
    assert!(!sch.sigmas().is_empty());
}

#[test]
fn t30_6_tcd_sigmas() {
    setup();
    let _t = load_case("diffusion_extra", "tcd");
    let mut sch = TcdScheduler::new(TcdConfig {
        beta: scaled_linear_beta(),
        ..Default::default()
    });
    sch.set_timesteps(4).unwrap();
    assert!(!sch.sigmas().is_empty());
}

#[test]
fn t30_7_dpm_pp_3m_set_timesteps() {
    setup();
    let _t = load_case("diffusion_extra", "dpm_3m");
    let mut sch = DpmPp3MScheduler::new(DpmPp3MConfig {
        beta: linear_beta(),
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();
    assert_eq!(sch.timesteps().len(), 20);
}

#[test]
fn t30_8_dpm_pp_sde_set_timesteps() {
    setup();
    let _t = load_case("diffusion_extra", "dpm_sde");
    let mut sch = DpmPpSdeScheduler::new(DpmPpSdeConfig {
        beta: linear_beta(),
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();
    assert!(!sch.sigmas().is_empty());
}

#[test]
fn t30_9_heun_first_step_runs() {
    setup();
    let _ = Device::Cpu;
    let mut sch = HeunScheduler::new(HeunConfig {
        beta: linear_beta(),
        spacing: TimestepSpacing::Linspace,
        sigma_schedule: SigmaSchedule::BetaSchedule,
        ..Default::default()
    });
    sch.set_timesteps(10).unwrap();
    let x = Tensor::zeros(vec![1, 4, 8, 8], synaptix_core::dtype::DType::F32, Device::Cpu).unwrap();
    let noise = Tensor::zeros(vec![1, 4, 8, 8], synaptix_core::dtype::DType::F32, Device::Cpu).unwrap();
    let out = sch.step(&noise, 0, &x).unwrap();
    assert_eq!(out.prev_sample.dims(), &[1, 4, 8, 8]);
}
