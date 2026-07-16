"""Generate reference SafeTensors for Session 9 — Diffusion schedulers.

Run:
    python scripts/reference/gen_diffusion.py

Uses HuggingFace diffusers schedulers as ground truth.
Saves step-by-step noisy latents + denoised outputs for 20 steps.
Outputs data/ref/diffusion/<case>.safetensors.
"""

import pathlib

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/diffusion")
N_STEPS = 20
LATENT_SHAPE = (1, 4, 8, 8)


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def _fake_noise_pred(
    noisy: torch.Tensor,
    t: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn_like(noisy) * 0.1


def case_ddpm_steps() -> None:
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    scheduler.set_timesteps(N_STEPS)
    torch.manual_seed(0)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    noisy_latent = scheduler.add_noise(
        latent,
        torch.randn_like(latent),
        scheduler.timesteps[:1],
    )
    tensors = {"initial_latent": latent, "noisy_input": noisy_latent.clone()}
    x = noisy_latent.clone()
    for step_idx, t in enumerate(scheduler.timesteps):
        noise_pred = _fake_noise_pred(x, t, seed=step_idx)
        variance_noise = torch.randn(
            x.shape,
            generator=torch.Generator().manual_seed(500 + step_idx),
            dtype=torch.float32,
        )
        gen = torch.Generator().manual_seed(500 + step_idx)
        result = scheduler.step(noise_pred, t, x, generator=gen)
        x = result.prev_sample
        tensors[f"step_{step_idx:02d}_noise_pred"] = noise_pred
        tensors[f"step_{step_idx:02d}_variance_noise"] = variance_noise
        tensors[f"step_{step_idx:02d}_output"] = x.clone()
    save_case("ddpm_steps", tensors)


def case_ddim_steps() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    scheduler.set_timesteps(N_STEPS)
    torch.manual_seed(1)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    noisy_latent = scheduler.add_noise(
        latent,
        torch.randn_like(latent),
        scheduler.timesteps[:1],
    )
    tensors = {"initial_latent": latent, "noisy_input": noisy_latent.clone()}
    x = noisy_latent.clone()
    for step_idx, t in enumerate(scheduler.timesteps):
        noise_pred = _fake_noise_pred(x, t, seed=100 + step_idx)
        result = scheduler.step(noise_pred, t, x, eta=0.0)
        x = result.prev_sample
        tensors[f"step_{step_idx:02d}_noise_pred"] = noise_pred
        tensors[f"step_{step_idx:02d}_output"] = x.clone()
    save_case("ddim_steps", tensors)


def case_euler_steps() -> None:
    scheduler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_schedule="linear")
    scheduler.set_timesteps(N_STEPS)
    torch.manual_seed(2)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    x = latent.clone() * scheduler.init_noise_sigma
    tensors = {"initial_latent": latent, "noisy_input": x.clone()}
    for step_idx, t in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(x, t)
        noise_pred = _fake_noise_pred(model_input, t, seed=200 + step_idx)
        result = scheduler.step(noise_pred, t, x)
        x = result.prev_sample
        tensors[f"step_{step_idx:02d}_noise_pred"] = noise_pred
        tensors[f"step_{step_idx:02d}_output"] = x.clone()
    save_case("euler_steps", tensors)


def case_dpm_steps() -> None:
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        solver_order=2,
    )
    scheduler.set_timesteps(N_STEPS)
    torch.manual_seed(3)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    x = latent.clone() * scheduler.init_noise_sigma
    tensors = {"initial_latent": latent, "noisy_input": x.clone()}
    for step_idx, t in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(x, t)
        noise_pred = _fake_noise_pred(model_input, t, seed=300 + step_idx)
        result = scheduler.step(noise_pred, t, x)
        x = result.prev_sample
        tensors[f"step_{step_idx:02d}_noise_pred"] = noise_pred
        tensors[f"step_{step_idx:02d}_output"] = x.clone()
    save_case("dpm_steps", tensors)


def case_flowmatch_steps() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(N_STEPS)
    torch.manual_seed(4)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    x = latent.clone()
    tensors = {"initial_latent": latent, "noisy_input": x.clone()}
    for step_idx, t in enumerate(scheduler.timesteps):
        noise_pred = _fake_noise_pred(x, t, seed=400 + step_idx)
        result = scheduler.step(noise_pred, t, x)
        x = result.prev_sample
        tensors[f"step_{step_idx:02d}_noise_pred"] = noise_pred
        tensors[f"step_{step_idx:02d}_output"] = x.clone()
    save_case("flowmatch_steps", tensors)


def case_cfg_guidance() -> None:
    torch.manual_seed(5)
    noise_pred_uncond = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    noise_pred_cond = torch.randn(LATENT_SHAPE, dtype=torch.float32)
    for scale in [1.0, 3.5, 7.5, 15.0]:
        guided = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
        scale_str = str(scale).replace(".", "_")
        save_case(
            f"cfg_scale_{scale_str}",
            {
                "uncond": noise_pred_uncond,
                "cond": noise_pred_cond,
                "output": guided,
                "scale": torch.tensor([scale]),
            },
        )


def case_scheduler_sigmas() -> None:
    euler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_schedule="linear")
    euler.set_timesteps(N_STEPS)
    sigmas = euler.sigmas
    timesteps = euler.timesteps
    save_case(
        "scheduler_sigmas",
        {
            "sigmas": sigmas.float(),
            "timesteps": timesteps.float(),
        },
    )


def main() -> None:
    print("Generating diffusion scheduler reference data...")
    case_ddpm_steps()
    case_ddim_steps()
    case_euler_steps()
    case_dpm_steps()
    case_flowmatch_steps()
    case_cfg_guidance()
    case_scheduler_sigmas()
    print("Done.")


if __name__ == "__main__":
    main()
