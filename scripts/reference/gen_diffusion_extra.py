"""Generate reference data for extra schedulers (Heun/UniPC/PNDM/EDM/LCM/TCD/DPM++3M/DPM++SDE).

Run:
    python scripts/reference/gen_diffusion_extra.py

Saves sigmas + step_0 output for each. Step-only-zero чтобы избежать state mgmt в multistep solvers.
"""

import pathlib

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EDMDPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LCMScheduler,
    PNDMScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
)
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/diffusion_extra")
N_STEPS = 20
LATENT_SHAPE = (1, 4, 8, 8)


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def _common_setup(scheduler, seed=42):
    scheduler.set_timesteps(N_STEPS)
    torch.manual_seed(seed)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    init_sigma = float(scheduler.init_noise_sigma) if hasattr(scheduler, "init_noise_sigma") else 1.0
    x = latent.clone() * init_sigma
    return latent, x


def _save_sigmas_and_step_zero(name, scheduler, seed):
    latent, x = _common_setup(scheduler, seed)
    sigmas = scheduler.sigmas.float() if hasattr(scheduler, "sigmas") else torch.zeros(N_STEPS + 1)
    timesteps = scheduler.timesteps.float()

    t = scheduler.timesteps[0]
    model_input = scheduler.scale_model_input(x, t) if hasattr(scheduler, "scale_model_input") else x
    torch.manual_seed(seed + 1000)
    noise_pred = torch.randn_like(model_input) * 0.1
    out = scheduler.step(noise_pred, t, x)
    step0_out = out.prev_sample

    save_case(name, {
        "sigmas": sigmas,
        "timesteps": timesteps,
        "initial_latent": latent,
        "noisy_input": x,
        "noise_pred_0": noise_pred,
        "step_0_output": step0_out,
    })


LINEAR = dict(num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0001, beta_end=0.02)
SCALED = dict(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)


def case_heun():
    sch = HeunDiscreteScheduler(**LINEAR)
    _save_sigmas_and_step_zero("heun", sch, seed=10)


def case_unipc():
    sch = UniPCMultistepScheduler(**LINEAR, solver_order=2)
    _save_sigmas_and_step_zero("unipc", sch, seed=11)


def case_pndm():
    sch = PNDMScheduler(**LINEAR)
    _save_sigmas_and_step_zero("pndm", sch, seed=12)


def case_edm():
    sch = EDMDPMSolverMultistepScheduler(
        sigma_min=0.002, sigma_max=80.0, sigma_data=0.5, num_train_timesteps=1000, solver_order=2,
    )
    _save_sigmas_and_step_zero("edm", sch, seed=13)


def case_lcm():
    sch = LCMScheduler(**SCALED)
    _save_sigmas_and_step_zero("lcm", sch, seed=14)


def case_tcd():
    sch = TCDScheduler(**SCALED)
    _save_sigmas_and_step_zero("tcd", sch, seed=15)


def case_dpm_3m():
    sch = DPMSolverMultistepScheduler(**LINEAR, solver_order=3)
    _save_sigmas_and_step_zero("dpm_3m", sch, seed=16)


def case_dpm_sde():
    sch = DPMSolverSDEScheduler(**LINEAR)
    _save_sigmas_and_step_zero("dpm_sde", sch, seed=17)


def main():
    print("Generating extra diffusion scheduler reference data...")
    case_heun()
    case_unipc()
    case_pndm()
    case_edm()
    case_lcm()
    case_tcd()
    case_dpm_3m()
    case_dpm_sde()
    print("Done.")


if __name__ == "__main__":
    main()
