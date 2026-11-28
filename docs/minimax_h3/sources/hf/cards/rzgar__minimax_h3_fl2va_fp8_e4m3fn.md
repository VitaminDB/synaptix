# rzgar/minimax_h3_fl2va_fp8_e4m3fn

Источник: https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn · скачано 2026-11-28
Загрузок: 3665 · лайков: 25 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-07T06:18:57.000Z

---

---
license: apache-2.0
base_model:
- MiniMaxAI/MiniMax-H3
base_model_relation: quantized
pipeline_tag: image-to-video
library_name: diffusers
tags:
  - text-to-video
  - image-to-video
  - image-text-to-video
  - video-to-video
  - text-to-audio-video
  - image-to-audio-video
  - image-text-to-audio-video
  - video-to-audio-video
  - audio-to-audio-video
  - audio-video-generation
  - multimodal
  - synchronized-audio-video
  - reference-to-audio-video
---



# MiniMax H3 MXFP8
MXFP8 quantization of the MiniMax H3 `fl2va` (first-last-frame to video+audio) diffusion checkpoint.

_better audio at 5 (talking heads, less motions) & 8-steps (shift 8)_

<video controls width="1280" height="720">
  <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/mxfp8_vs_fp8/mxfp8_vs_fp8.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# MiniMax H3 FP16attn
Balanced quality and performance

_fp16attn 8-steps | shift: 8.0_
<video controls width="1280" height="720">
  <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/fp16attn/fp16-fp8_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# MiniMax H3 FP8 (E4M3FN) Quantized

Based on testing (this model and pruned_fp8), the minimal steps to get results similar to Lightx2v using 4-step LoRAs is 5step & 5steps. 
The sweet spot for quality and speed is 8step and 8steps. This quantization requires a minimum of 8step.

- **Sampler:** dpmpp_2m
- **Scheduler:** sgm_uniform
  <img src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/minimax-5step-ksampler.png" width="1196" height="709"  />

| 5-Step pruned fp8 | 5-Step mxfp8 | 8-Step fp8 |
|---------|-------|-------|
| <video controls width="50%" height="50%"> <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/MiniMax_H3_5step_pruned_fp8.mp4" type="video/mp4"></video> | <video controls width="50%" height="50%"> <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/mxfp8_vs_fp8/MiniMax_H3_mxfp8_5step_5steps_.mp4" type="video/mp4"></video> | <video controls width="50%" height="50%"> <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/MiniMax_H3_8step_fp8_.mp4" type="video/mp4"></video> |

| 8-steps fp16attn |
|---------|
| <video controls width="50%" height="50%"> <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/fp16attn/MiniMax_H3_fp16-fp8_8steps_.mp4" type="video/mp4"></video> |

FP8 quantization of the MiniMax H3 `fl2va` (first-last-frame to video+audio)
diffusion checkpoint.

<video controls width="1280" height="720">
  <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/demo/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


**Original model:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

## File sizes

| Variant | Download |
|---------|------|
| minimax_h3_fl2va_mxfp8.safetensors | [Download 47.6GB](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_mxfp8.safetensors) |
| minimax_h3_fl2va_fp16attn_fp8.safetensors | [Download 28.7GB](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_fp16attn_fp8.safetensors) |
| minimax_h3_fl2va_fp8_e4m3fn.safetensors | [Download 47GB](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_fp8_e4m3fn.safetensors) |
| minimax_h3_fl2va_pruned_fp8_scaled.safetensors | [Download 21GB](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors)


### Simple Upscalers
Old but gold upscale models. Lower image resolution for faster inference and more steps.

| Variant | Download |
|---------|------|
| 2xNomosUni_esrgan_multijpg.pth | [Download](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/upscale_models/2xNomosUni_esrgan_multijpg.pth) |
| RealESRGAN_x2plus.pth | [Download](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/upscale_models/RealESRGAN_x2plus.pth) |

_736x416 | ~56 sec |  1472x832 60fps_
<video controls width="1280" height="720">
  <source src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/upscale_models/demo/2xNomosUni_60_fps.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


### Left at native precision (quality-critical)

| Tensors | Dtype | Note |
|---------|-------|--------|
| `adaln_proj.linear.*` (100 tensors) | BF16 | Per-block modulation, largest single weight |
| `token_refiner.*` (17 tensors) | BF16 | Text embedding refinement, 2 layers only |
| `condition_proj.*` (2 tensors) | BF16 | Cross-modal conditioning bridge |
| All norm weights (`norm1`, `norm2`, `q_norm`, `k_norm`, `final_norm`) | BF16 | 104 tensors, tiny and precision-sensitive |
| All biases | BF16 | 56 tensors |
| `time_embedder.*` (4 tensors) | FP32 | Timestep signal, original precision preserved |
| `video_patch_proj.*`, `audio_patch_proj.*` (4 tensors) | FP32 | Input/output projections |
| `final_layer.*_out.*` (4 tensors) | FP32 | Decoder output heads |
| `rope.inv_freq` (1 tensor) | FP32 | Position encoding |


# ComfyUI-MiniMaxH3-Text-Enhancer node
[What this is:](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/tree/main/ComfyUI-MiniMaxH3-Text-Enhancer) the result of looking into why male anatomy comes out misshaped in text-to-video scenes  
or when a nude male appears in the I2V scene.
Along the way we learned some things worth knowing about how this model really works, read [README.md](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/blob/main/ComfyUI-MiniMaxH3-Text-Enhancer/README.md) 
and [notes.md](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/blob/main/ComfyUI-MiniMaxH3-Text-Enhancer/notes.md) in case of you are interested.

**Q: Is it a text-encoder problem? Would an abliterated Qwen3-VL solve it?**
- A: No. The stock encoder doesn't refuse NSFW prompts, the tokenizer and embedding table handle them normally. 
Its only strong divergence from stock is on harmful-instruction content (~11% embedding shift); sexual content shifts only ~1–3%. 
Abliteration targets the wrong thing here.

**Q: Is the base model censored? Would a domain-specific LoRA be ineffective?**
- A: No, the base is uncensored. The misshape comes from weak text conditioning: only ~3–5% of the embedding actually carries the prompt, 
so the DiT's generic body prior wins over the anatomy request. Modifying internal values (condition_proj ×1.2 + 12,658 FFN columns ×1.2) improves the result slightly
and a properly trained LoRA would refine anatomy further on top of it.

_Experimental: 1.2 strength in both nodes improves NSFW rhythmic animations in some scenarios_
<img src="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/ComfyUI-MiniMaxH3-Text-Enhancer/txt_enhancer_exprmntl.png" width="1280" height="720" />
