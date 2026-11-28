# InstantX/MiniMax-H3-Turbo-Lora-Diffusers

Источник: https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers · скачано 2026-11-28
Загрузок: 3365 · лайков: 13 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-08-06T09:16:21.000Z

---

---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
  - text-to-video
  - text-to-audio
  - audio-video
  - lora
  - minimax-h3
  - diffusers
  - peft
pipeline_tag: text-to-video
library_name: diffusers
---

# MiniMax-H3 Turbo LoRA — Diffusers

Diffusers / PEFT conversion of [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora): a LoRA that lets [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) render joint **video + synchronized stereo audio** in about **4 sampling steps** instead of the usual ~20.

This repo ships:

- converted LoRA weights in Diffusers PEFT layout (`transformer.*.lora_A/B.weight`)
- `convert.py` to turn the original ComfyUI / `generate.py` safetensors into that layout

> ⚠️ **Early prototype.** Same caveat as the upstream release: under-trained preview weights, not production quality. They already beat the base model at 4 steps (sharper detail, cleaner / better-synced audio), but treat this as a work-in-progress taste, not a finished product. Prefer the non-EMA `ckpt500` weights by default.

## Weights

Converted from the upstream Turbo LoRA (bf16, `W_eff = W + lora_B @ lora_A`, **alpha = rank** so scale is 1). QKV is split into `to_q` / `to_k` / `to_v`, and SwiGLU `fc1` halves are swapped to match Diffusers' `[value; gate]` layout.

| file | source (ComfyUI layout) | notes |
|---|---|---|
| `minimax_h3_turbo_4step_ckpt500_diffusers.safetensors` | `minimax_h3_turbo_4step_ckpt500.safetensors` | **recommended default** — newest non-EMA @ ~500 steps, usually sharpest |
| `minimax_h3_turbo_4step_ema_ckpt500_diffusers.safetensors` | `minimax_h3_turbo_4step_ema_ckpt500.safetensors` | EMA @ ~500 steps — smoother, but early EMA can show **ghosting / motion smear** |

Ranks: attention / MLP = 64, AdaLN = 16. Keys are prefixed with `transformer.` for `MiniMaxH3Transformer3DModel.load_lora_adapter`.

## Requirements

MiniMax-H3 is not in a released Diffusers build yet. Install Diffusers from `main`, plus PEFT:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers.git
```

Base weights: Diffusers-format [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (or your local conversion). Use a **non-pruned** DiT; pruned time-conditioning layouts are **not** compatible with this LoRA (same restriction as upstream).

## Quick start (Diffusers)

```python
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def network_alphas_alpha_eq_rank(state_dict):
    # Turbo LoRA: alpha == rank. Required when ranks differ (attn/mlp=64, adaln=16).
    alphas = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_B.weight") and tensor.ndim > 1:
            base = key[: -len(".lora_B.weight")]
            alphas[f"{base}.alpha"] = float(tensor.shape[1])
    return alphas

lora_path = hf_hub_download(
    "InstantX/MiniMax-H3-Turbo-Lora-Diffusers",
    "minimax_h3_turbo_4step_ckpt500_diffusers.safetensors",
)

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

lora_sd = load_file(lora_path, device="cpu")
pipe.transformer.load_lora_adapter(
    lora_sd,
    prefix="transformer",
    adapter_name="turbo_4step",
    network_alphas=network_alphas_alpha_eq_rank(lora_sd),
)

# Load LoRA *before* enabling offload so PEFT injects into resident modules.
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

# Optional: FlashAttention-3 on Hopper (kernels from the Hub).
try:
    pipe.transformer.set_attention_backend("_flash_3_hub")
except Exception:
    pipe.transformer.set_attention_backend("native")

# MiniMaxH3Scheduler: num_inference_steps is the sigma grid length *including* terminal 0,
# so it drives (num_inference_steps - 1) model evals.
#   5 -> 4 evals  (matches upstream generate.py --steps 4)
#   7–9 -> 6–8 evals  (upstream comfort zone for sharpness at this early checkpoint)
results = pipe(
    prompt="A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark.",
    num_frames=124,   # 17*k+5, ~5.17s @ 24fps
    height=768,
    width=1344,
    num_inference_steps=5,
    generator=torch.Generator().manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)

encode_video(
    results["videos"][0],
    fps=24,
    output_path="out.mp4",
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
```

Diffusers already runs **dual video / audio schedules** (`scheduler` shift 12, `audio_scheduler` shift 3). You do **not** need the ComfyUI Turbo custom sampler node; a wrong single-schedule sampler is what blows up audio at 4 steps in ComfyUI.

## Convert from the original Turbo LoRA

Original weights live in [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (ComfyUI module names, fused `qkv_proj` / `mlp.fc1`).

```bash
pip install safetensors torch

python convert.py \
  --input minimax_h3_turbo_4step_ckpt500.safetensors \
  --output minimax_h3_turbo_4step_ckpt500_diffusers.safetensors
```

What `convert.py` does:

1. Renames ComfyUI paths onto `MiniMaxH3Transformer3DModel` (`blocks.*` → `transformer_blocks.*`, `mlp.fc*` → `ff.net.*`, `final_layer.adaln_proj` → `norm_out.linear`, …).
2. Splits fused `attn.qkv_proj` LoRA into `to_q` / `to_k` / `to_v` (shared `A`, row-split `B` in `[q_all; k_all; v_all]` layout).
3. Swaps `mlp.fc1` LoRA halves from `[gate; value]` to Diffusers SwiGLU `[value; gate]`.
4. Writes keys with a `transformer.` prefix for `load_lora_adapter`.

## Notes

- **Steps**: 4 model evals (`num_inference_steps=5`) works; at this early checkpoint **6–8 evals** (`num_inference_steps=7…9`) are usually sharper. Any count ≥ 4 evals is valid; more steps look better.
- **Resolution / duration**: `height` / `width` multiples of 32 (short edge typically 768). `num_frames` at 24 fps snaps up to the video VAE’s `17·k+5` grid (124 ≈ 5 s). Validated roughly 5–15 s.
- **VRAM**: the base DiT is ~33B. An 80–96 GB GPU is comfortable with `ComponentsManager.enable_auto_cpu_offload`; smaller cards need quantization / group offload as in the [MiniMax-H3 Diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3).
- **Audio**: 32 kHz stereo aligned to the video; video and audio ride different flow schedules inside one transformer call.
- **ComfyUI**: for the original graph / custom Turbo sampler, use the [upstream repo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) and [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo).

## Credit

- Turbo LoRA training & original release: [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- Base model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Diffusers MiniMax-H3 integration: Hugging Face Diffusers (modular pipeline)
