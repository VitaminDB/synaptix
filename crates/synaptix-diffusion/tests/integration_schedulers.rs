use synaptix_core::device::Device;
use synaptix_diffusion::schedulers::{randn_seeded, Scheduler};
use synaptix_diffusion::schedulers::{
    ddpm::{DdpmConfig, DdpmScheduler},
    ddim::{DdimConfig, DdimScheduler},
    euler::{EulerConfig, EulerScheduler, SigmaSchedule},
    dpm_pp_2m::{DpmPp2MConfig, DpmPp2MScheduler},
    flow_match::{FlowMatchConfig, FlowMatchScheduler},
    rectified_flow::{RectifiedFlowConfig, RectifiedFlowScheduler},
};
use synaptix_ops::rng::Philox4x32;

fn run_scheduler_converges(scheduler: &mut dyn Scheduler, n_steps: usize, tol: f32) {
    synaptix_kernels_cpu::ensure_registered();
    scheduler.set_timesteps(n_steps).unwrap();
    let mut rng = Philox4x32::new(42);
    let shape = [1, 4, 8, 8];
    let noise = randn_seeded(&shape, Device::Cpu, &mut rng).unwrap();
    let init_sigma = scheduler.init_noise_sigma();
    let mut sample = noise.affine(init_sigma, 0.0).unwrap();

    let n = scheduler.n_steps();
    for i in 0..n {
        let scaled = scheduler.scale_model_input(&sample, i).unwrap();
        let out = scheduler.step(&scaled, i, &sample).unwrap();
        sample = out.prev_sample;
    }

    let data = sample.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let rms = (data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32).sqrt();
    assert!(rms < tol, "scheduler did not converge: rms={rms:.4} tol={tol}");
}

#[test]
fn ddpm_converges() {
    let cfg = DdpmConfig::default();
    let mut s = DdpmScheduler::new(cfg);
    run_scheduler_converges(&mut s, 20, 5.0);
}

#[test]
fn ddim_converges() {
    let cfg = DdimConfig::default();
    let mut s = DdimScheduler::new(cfg);
    run_scheduler_converges(&mut s, 20, 5.0);
}

#[test]
fn euler_default_converges() {
    let cfg = EulerConfig::default();
    let mut s = EulerScheduler::new(cfg);
    run_scheduler_converges(&mut s, 20, 5.0);
}

#[test]
fn euler_karras_converges() {
    let cfg = EulerConfig { sigma_schedule: SigmaSchedule::Karras, ..EulerConfig::default() };
    let mut s = EulerScheduler::new(cfg);
    run_scheduler_converges(&mut s, 20, 5.0);
}

#[test]
#[ignore = "mock model_output = scaled даёт огромный x0_pred через DPM++ formula (нерепрезентативный test); ref_schedulers/t13_4 даёт bit-exact validation"]
fn dpm_pp_2m_converges() {
    let cfg = DpmPp2MConfig::default();
    let mut s = DpmPp2MScheduler::new(cfg);
    run_scheduler_converges(&mut s, 20, 5.0);
}

#[test]
fn flow_match_sd3_converges() {
    let mut s = FlowMatchScheduler::new(FlowMatchConfig::sd3());
    run_scheduler_converges(&mut s, 20, 1.5);
}

#[test]
fn flow_match_flux_converges() {
    let mut s = FlowMatchScheduler::new(FlowMatchConfig::flux());
    run_scheduler_converges(&mut s, 20, 1.5);
}

#[test]
fn rectified_flow_ltx_converges() {
    let mut s = RectifiedFlowScheduler::new(RectifiedFlowConfig::ltx());
    run_scheduler_converges(&mut s, 20, 1.5);
}

#[test]
fn sigma_chain_decreases_for_euler() {
    synaptix_kernels_cpu::ensure_registered();
    let mut s = EulerScheduler::new(EulerConfig::default());
    s.set_timesteps(10).unwrap();
    let sigmas = s.sigmas().to_vec();
    assert!(sigmas.len() >= 10);
    for w in sigmas[..sigmas.len() - 1].windows(2) {
        assert!(w[0] >= w[1] - 1e-6, "sigmas not monotone: {:.4} > {:.4}", w[0], w[1]);
    }
    assert_eq!(*sigmas.last().unwrap(), 0.0);
}
