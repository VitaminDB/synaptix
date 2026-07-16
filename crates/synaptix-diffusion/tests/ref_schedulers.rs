use synaptix_diffusion::apply_cfg;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() {
    ensure_registered();
}

#[test]
fn t13_6_cfg_guidance_scales() {
    setup();
    for scale_str in ["1_0", "3_5", "7_5", "15_0"] {
        let t = load_case("diffusion", &format!("cfg_scale_{}", scale_str));
        let scale = t["scale"].to_vec1::<f32>().unwrap()[0];
        let result = apply_cfg(&t["uncond"], &t["cond"], scale).unwrap();
        assert_allclose(&result, &t["output"], 1e-6, 1e-6);
    }
}

#[test]
fn t13_1_ddpm_steps() {
    setup();
    use synaptix_diffusion::schedulers::ddpm::{DdpmConfig, DdpmScheduler, VarianceType};
    use synaptix_diffusion::schedulers::{BetaConfig, BetaSchedule, Scheduler, TimestepSpacing};

    let t = load_case("diffusion", "ddpm_steps");
    let mut x = t["noisy_input"].clone();

    let mut sch = DdpmScheduler::new(DdpmConfig {
        beta: BetaConfig {
            num_train_timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule: BetaSchedule::Linear,
            rescale_zero_snr: false,
        },
        spacing: TimestepSpacing::Leading,
        variance_type: VarianceType::FixedSmall,
        clip_sample: Some(1.0),
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();

    for step_idx in 0..20 {
        let noise_pred = &t[&format!("step_{:02}_noise_pred", step_idx)];
        let variance_noise = &t[&format!("step_{:02}_variance_noise", step_idx)];
        let out = sch.step_with_noise(noise_pred, step_idx, &x, variance_noise).unwrap();
        x = out.prev_sample;
        let expected = &t[&format!("step_{:02}_output", step_idx)];
        assert_allclose(&x, expected, 1e-4, 1e-4);
    }
}

#[test]
fn t13_2_ddim_steps() {
    setup();
    use synaptix_diffusion::schedulers::ddim::{DdimConfig, DdimScheduler};
    use synaptix_diffusion::schedulers::{BetaConfig, BetaSchedule, Scheduler, TimestepSpacing};

    let t = load_case("diffusion", "ddim_steps");
    let mut x = t["noisy_input"].clone();

    let mut sch = DdimScheduler::new(DdimConfig {
        beta: BetaConfig {
            num_train_timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule: BetaSchedule::Linear,
            rescale_zero_snr: false,
        },
        spacing: TimestepSpacing::Leading,
        eta: 0.0,
        clip_sample: Some(1.0),
        set_alpha_to_one: true,
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();

    for step_idx in 0..20 {
        let noise_pred = &t[&format!("step_{:02}_noise_pred", step_idx)];
        let out = sch.step(noise_pred, step_idx, &x).unwrap();
        x = out.prev_sample;
        let expected = &t[&format!("step_{:02}_output", step_idx)];
        assert_allclose(&x, expected, 1e-4, 1e-4);
    }
}

#[test]
fn t13_3_euler_steps() {
    setup();
    use synaptix_diffusion::schedulers::euler::{EulerConfig, EulerScheduler, SigmaSchedule};
    use synaptix_diffusion::schedulers::{BetaConfig, BetaSchedule, Scheduler, TimestepSpacing};

    let t = load_case("diffusion", "euler_steps");
    let mut x = t["noisy_input"].clone();

    let mut sch = EulerScheduler::new(EulerConfig {
        beta: BetaConfig {
            num_train_timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule: BetaSchedule::Linear,
            rescale_zero_snr: false,
        },
        spacing: TimestepSpacing::Linspace,
        sigma_schedule: SigmaSchedule::BetaSchedule,
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();

    for step_idx in 0..20 {
        let key = format!("step_{:02}_noise_pred", step_idx);
        let noise_pred = &t[&key];
        let out = sch.step(noise_pred, step_idx, &x).unwrap();
        x = out.prev_sample;
        let out_key = format!("step_{:02}_output", step_idx);
        let expected = &t[&out_key];
        assert_allclose(&x, expected, 1e-4, 1e-4);
    }
}

#[test]
fn t13_4_dpm_steps() {
    setup();
    use synaptix_diffusion::schedulers::dpm_pp_2m::{DpmPp2MConfig, DpmPp2MScheduler};
    use synaptix_diffusion::schedulers::Scheduler;

    let t = load_case("diffusion", "dpm_steps");
    let mut x = t["noisy_input"].clone();
    let mut sch = DpmPp2MScheduler::new(DpmPp2MConfig::default());
    sch.set_timesteps(20).unwrap();
    for step_idx in 0..20 {
        let noise_pred = &t[&format!("step_{:02}_noise_pred", step_idx)];
        let out = sch.step(noise_pred, step_idx, &x).unwrap();
        x = out.prev_sample;
        let expected = &t[&format!("step_{:02}_output", step_idx)];
        assert_allclose(&x, expected, 1e-4, 1e-4);
    }
}

#[test]
fn t13_5_flowmatch_steps() {
    setup();
    use synaptix_diffusion::schedulers::flow_match::{FlowMatchConfig, FlowMatchScheduler};
    use synaptix_diffusion::schedulers::Scheduler;

    let t = load_case("diffusion", "flowmatch_steps");
    let mut x = t["noisy_input"].clone();
    let mut sch = FlowMatchScheduler::new(FlowMatchConfig::default());
    sch.set_timesteps(20).unwrap();
    for step_idx in 0..20 {
        let noise_pred = &t[&format!("step_{:02}_noise_pred", step_idx)];
        let out = sch.step(noise_pred, step_idx, &x).unwrap();
        x = out.prev_sample;
        let expected = &t[&format!("step_{:02}_output", step_idx)];
        assert_allclose(&x, expected, 1e-4, 1e-4);
    }
}

#[test]
fn t13_7_scheduler_sigmas() {
    setup();
    use synaptix_diffusion::schedulers::euler::{EulerConfig, EulerScheduler, SigmaSchedule};
    use synaptix_diffusion::schedulers::{BetaConfig, BetaSchedule, Scheduler, TimestepSpacing};

    let t = load_case("diffusion", "scheduler_sigmas");
    let expected_sigmas: Vec<f32> = t["sigmas"].to_vec1().unwrap();
    let expected_timesteps: Vec<f32> = t["timesteps"].to_vec1().unwrap();

    let mut sch = EulerScheduler::new(EulerConfig {
        beta: BetaConfig {
            num_train_timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule: BetaSchedule::Linear,
            rescale_zero_snr: false,
        },
        spacing: TimestepSpacing::Linspace,
        sigma_schedule: SigmaSchedule::BetaSchedule,
        ..Default::default()
    });
    sch.set_timesteps(20).unwrap();

    let our_sigmas = sch.sigmas();
    let our_timesteps = sch.timesteps();

    assert_eq!(our_sigmas.len(), expected_sigmas.len());
    for (i, (a, b)) in our_sigmas.iter().zip(expected_sigmas.iter()).enumerate() {
        let diff = (a - b).abs();
        let tol = b.abs() * 1e-3 + 1e-3;
        assert!(diff <= tol, "sigma[{}]: ours={}, expected={}, diff={}", i, a, b, diff);
    }
    assert_eq!(our_timesteps.len(), expected_timesteps.len());
    for (i, (a, b)) in our_timesteps.iter().zip(expected_timesteps.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff < 1.0, "timestep[{}]: ours={}, expected={}", i, a, b);
    }
}
