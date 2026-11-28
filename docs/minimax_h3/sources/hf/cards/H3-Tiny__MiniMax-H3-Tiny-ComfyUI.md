# H3-Tiny/MiniMax-H3-Tiny-ComfyUI

Источник: https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI · скачано 2026-11-28
Загрузок: 47824 · лайков: 37 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui · обновлено: 2026-11-24T11:53:08.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI/blob/main/LICENSE-MiniMax-H3
library_name: diffusion-single-file
pipeline_tag: text-to-video
tags:
- minimax-h3
- video-generation
- audio-generation
- comfyui
- pruning
- h3-tiny
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: finetune
---

# MiniMax-H3-Tiny for ComfyUI

[![Project Page](https://img.shields.io/badge/Project-Page-2f6feb)](https://fuyi25.github.io/MiniMax-H3-Tiny/) [![GitHub](https://img.shields.io/badge/GitHub-MiniMax--H3--Tiny-181717?logo=github)](https://github.com/fuyi25/MiniMax-H3-Tiny) [![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-H3--Tiny-ffcc4d)](https://huggingface.co/collections/H3-Tiny/h3-tiny-v01-6afeea67147c27cf3b80820a)

H3-Tiny transformers repackaged as single-file checkpoints for ComfyUI's native MiniMax H3 support.
Part of **[H3-Tiny](https://github.com/fuyi25/MiniMax-H3-Tiny)**: structured pruning of MiniMax-H3 for joint video + audio
generation.

| Model | Workflow | Params | bf16 | INT8 |
|---|---|---|---|---|
| [MiniMax-H3-Tiny-17B-Pruned-Ref2VA](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-Ref2VA) | Ref2VA | 17.05B | `minimax_h3_tiny_17b_ref2va_bf16.safetensors` · 34.2 GB | `minimax_h3_tiny_17b_ref2va_int8_convrot.safetensors` · 18.1 GB |
| [MiniMax-H3-Tiny-17B-Pruned-FL2VA](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-FL2VA) | FL2VA / T2VA | 17.05B | `minimax_h3_tiny_17b_fl2va_bf16.safetensors` · 34.2 GB | `minimax_h3_tiny_17b_fl2va_int8_convrot.safetensors` · 18.1 GB |
| [MiniMax-H3-Tiny-14B-OKD-Ref2VA-v0.1](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-14B-OKD-Ref2VA-v0.1) | Ref2VA | 13.98B | `minimax_h3_tiny_14b_okd_ref2va_bf16.safetensors` · 28.1 GB | `minimax_h3_tiny_14b_okd_ref2va_int8_convrot.safetensors` · 15.1 GB |
| [MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1) | Ref2VA | 17.08B | `minimax_h3_tiny_17b_fewstep_ref2va_bf16.safetensors` · 34.2 GB | `minimax_h3_tiny_17b_fewstep_ref2va_int8_convrot.safetensors` · 18.1 GB |

FewStep samples in 6 steps (4 or more work) and needs our nodes, see below; the others use 50 steps with stock ComfyUI.

These are the same weights as the diffusers checkpoints, in ComfyUI's key layout. ComfyUI reads the pruned depth and
FFN width from the tensors, so no custom node is needed.

The INT8 files load the same way and run on ComfyUI's INT8 kernels (W8A8 with Hadamard rotation, made with
[`scripts/quantize_comfyui_int8.py`](https://github.com/fuyi25/MiniMax-H3-Tiny/blob/main/scripts/quantize_comfyui_int8.py)).
At 1344×768 and 243 frames on one H200, the DiT takes about 17 GB of VRAM instead of 33 GB, sampling runs as fast as
bf16, and the outputs keep the composition and motion of bf16. If you start ComfyUI with `--enable-triton-backend`, also
install our nodes (see [FewStep](#fewstep)): they split INT8 calls that are too large for the Triton kernels.

## Usage

1. Put a file in `ComfyUI/models/diffusion_models/`.
2. Get the text encoder and VAEs from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
   `text_encoders/qwen3vl_32b_minimax_h3_*.safetensors` → `models/text_encoders/`,
   `vae/minimax_h3_video_vae_fp16.safetensors` and `vae/minimax_h3_audio_vae_fp32.safetensors` → `models/vae/`.
3. Open ComfyUI's MiniMax H3 workflow (`MiniMax H3 Image to Video` for FL2VA / T2VA, `MiniMax H3 Reference to Video`
   for Ref2VA) and select the H3-Tiny file in **Load Diffusion Model**.
4. Sample without the turbo LoRA: 50 steps, `simple` scheduler, guider without CFG (`BasicGuider`). We tested
   `euler`.

Use the `ref2va` file with the Reference to Video node and the `fl2va` file with the Image to Video node (with first
and last frames for FL2VA, without for T2VA).

We tested these files on ComfyUI commit `afcb984` (2026-11-21) at 1344×768 and 243 frames on one H200: the outputs
match our diffusers pipeline. The Comfy-Org turbo LoRAs were trained on the
33B model and are untested with H3-Tiny.

The files were made with [`scripts/convert_comfyui.py`](https://github.com/fuyi25/MiniMax-H3-Tiny/blob/main/scripts/convert_comfyui.py).
It writes AdaLN in ComfyUI's curve form (`adaln_t_table`), which reproduces the diffusers AdaLN to about 1e-10
relative error.

### FewStep

The few-step checkpoint also needs the step endpoint, which stock ComfyUI does not pass to the model. Install the
H3-Tiny nodes:

```bash
git clone https://github.com/fuyi25/MiniMax-H3-Tiny
ln -s "$PWD/MiniMax-H3-Tiny/comfyui" ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Tiny
```

In the Reference to Video workflow, load the file with **H3-Tiny Few-Step Loader** (bf16 or INT8) instead of Load Diffusion Model,
and feed **H3-Tiny Few-Step Sigmas** (6 steps by default) to `SamplerCustomAdvanced`, with the `euler` sampler and
`BasicGuider`.

## Contact

Questions and bug reports: [GitHub issues](https://github.com/fuyi25/MiniMax-H3-Tiny/issues) or the Community tab here. For anything else, email Haoshuai Fu at [fohoo@outlook.com](mailto:fohoo@outlook.com).

## License

These weights are Model Derivatives of MiniMax H3 and are distributed under the
[MiniMax H3 Community License Agreement](LICENSE-MiniMax-H3); see [NOTICE](NOTICE). Use of the weights must follow that
agreement, including its territorial scope and acceptable-use policy.

*H3-Tiny is an independent project and is not affiliated with or endorsed by MiniMax or Comfy Org.*
