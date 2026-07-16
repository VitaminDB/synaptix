"""Дамп вход/выход полного FLUX-денойза для изоляции косяка качества.
Реплицируем FluxPipeline вручную при 512², дампим:
  init_latent [1,seq,64] (packed noise), pooled [1,768], t5_seq [1,512,4096],
  final_latent [1,seq,64] (после денойза, до unpack/VAE).
Всё bf16. Затем синапс-пайплайн прогоняет ЭТОТ init_latent+embeds и сравнивает."""

import pathlib
import torch
from diffusers import FluxPipeline

FLUX = "models/black-forest-labs/FLUX.1-dev"
OUT = pathlib.Path("tests/reference_data/flux_io")
H = W = 512
STEPS = 28
PROMPT = "a photo of a red apple on a wooden table, cinematic lighting"


def save(name, d):
    from safetensors.torch import save_file
    OUT.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in d.items()}, str(OUT / f"{name}.safetensors"))
    print("saved", name, {k: tuple(v.shape) for k, v in d.items()})


@torch.no_grad()
def main():
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    dev = pipe._execution_device

    (prompt_embeds, pooled, text_ids) = pipe.encode_prompt(
        prompt=PROMPT, prompt_2=PROMPT, device=dev, num_images_per_prompt=1, max_sequence_length=512)

    num_ch = pipe.transformer.config.in_channels // 4
    gen = torch.Generator("cpu").manual_seed(0)
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_ch, H, W, prompt_embeds.dtype, dev, gen, None)
    init_latent = latents.clone()

    import numpy as np
    image_seq_len = latents.shape[1]
    mu = pipe.scheduler.config.get("base_shift", 0.5)
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    sigmas = np.linspace(1.0, 1 / STEPS, STEPS)
    mu = calculate_shift(image_seq_len, pipe.scheduler.config.base_image_seq_len,
                         pipe.scheduler.config.max_image_seq_len,
                         pipe.scheduler.config.base_shift, pipe.scheduler.config.max_shift)
    pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=dev)
    timesteps = pipe.scheduler.timesteps
    guidance = torch.full([1], 3.5, device=dev, dtype=torch.float32)

    vel0 = None
    sigma0 = float(timesteps[0] / 1000)
    inter = {}
    def hook(name):
        def f(_m, _i, out):
            if isinstance(out, tuple):
                for j, o in enumerate(out):
                    if torch.is_tensor(o): inter[f"{name}_{j}"] = o.detach().float().cpu()
            elif torch.is_tensor(out): inter[name] = out.detach().float().cpu()
        return f
    hs = []
    hs.append(pipe.transformer.x_embedder.register_forward_hook(hook("x_emb")))
    hs.append(pipe.transformer.context_embedder.register_forward_hook(hook("ctx_emb")))
    hs.append(pipe.transformer.time_text_embed.register_forward_hook(hook("temb")))
    hs.append(pipe.transformer.transformer_blocks[0].register_forward_hook(hook("db0")))
    hs.append(pipe.transformer.single_transformer_blocks[0].register_forward_hook(hook("sb0")))
    hs.append(pipe.transformer.pos_embed.register_forward_hook(hook("rope")))
    b0 = pipe.transformer.transformer_blocks[0]
    hs.append(b0.norm1.register_forward_hook(hook("db0_norm1")))
    hs.append(b0.norm1_context.register_forward_hook(hook("db0_norm1ctx")))
    hs.append(b0.attn.register_forward_hook(hook("db0_attn")))
    hs.append(b0.ff.register_forward_hook(hook("db0_ff")))
    hs.append(b0.ff_context.register_forward_hook(hook("db0_ffc")))
    for di in (9, 18):
        hs.append(pipe.transformer.transformer_blocks[di].register_forward_hook(hook(f"depthD{di}")))
    b14 = pipe.transformer.transformer_blocks[14]
    def pre_hook(_m, args, kwargs):
        hs_in = kwargs.get("hidden_states", args[0] if args else None)
        enc = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        if hs_in is not None: inter["b14in_img"] = hs_in.detach().float().cpu()
        if enc is not None: inter["b14in_txt"] = enc.detach().float().cpu()
    hs.append(b14.register_forward_pre_hook(pre_hook, with_kwargs=True))
    hs.append(b14.norm1.register_forward_hook(hook("b14_norm1")))
    hs.append(b14.norm1_context.register_forward_hook(hook("b14_norm1ctx")))
    hs.append(b14.attn.register_forward_hook(hook("b14_attn")))
    hs.append(b14.ff.register_forward_hook(hook("b14_ff")))
    hs.append(b14.register_forward_hook(hook("b14_out")))
    for si in (9, 18, 37):
        hs.append(pipe.transformer.single_transformer_blocks[si].register_forward_hook(hook(f"depthS{si}")))
    for si, t in enumerate(timesteps):
        ts = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents, timestep=ts / 1000, guidance=guidance,
            pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=latent_image_ids, return_dict=False)[0]
        if si == 0:
            vel0 = noise_pred.clone()
            for h in hs: h.remove()
            save("inter_real", inter)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    print("sigma0", sigma0)
    save("io", {"init_latent": init_latent, "pooled": pooled,
                "t5_seq": prompt_embeds, "final_latent": latents,
                "vel0": vel0, "sigma0": torch.tensor([sigma0])})


if __name__ == "__main__":
    main()
