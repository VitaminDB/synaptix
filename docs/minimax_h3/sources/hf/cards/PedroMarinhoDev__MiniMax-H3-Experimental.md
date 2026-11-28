# PedroMarinhoDev/MiniMax-H3-Experimental

Источник: https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental · скачано 2026-11-28
Загрузок: 73 · лайков: 4 · базовая: ["Comfy-Org/MiniMax-H3","TaoLiveAIGC/TaoMate-H3","Asirus/TaoMate_H3_3_Step_LoRA"] · теги: comfyui,base_model:Asirus/TaoMate_H3_3_Step_LoRA,base_model:finetune:Asirus/TaoMate_H3_3_Step_LoRA · обновлено: 2026-10-28T10:37:46.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
- Comfy-Org/MiniMax-H3
- TaoLiveAIGC/TaoMate-H3
- Asirus/TaoMate_H3_3_Step_LoRA
tags:
- comfyui
- video-generation
- text-to-video
- image-to-video
- minimax-h3
- int8
---

# MiniMax H3 Experimental

Experimental MiniMax H3 files for ComfyUI.

## minimax_h3_fl2va_pruned_int8_convrot_taomax3.safetensors

MiniMax H3 FL2VA (pruned, int8 ConvRot) with the **TaoMate H3 3-step MAXQUALITY** step-distillation LoRA merged into the
weights. It renders a video with sound in **3 sampling steps** instead of 20, with no LoRA node, and keeps ComfyUI's
fast int8 path at every resolution. When the model is streamed from RAM (large sizes on 24 GB cards), a LoRA loaded at run
time makes every patched int8 layer run in bf16; a merged file avoids that.

**This is a modified file.** Made from Comfy-Org's
[`minimax_h3_fl2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3) and
[`taomate_h3_3step_MAXQUALITY_bf16.safetensors`](https://huggingface.co/Asirus/TaoMate_H3_3_Step_LoRA) (the official
TaoMate-H3 3-step LoRA from Alibaba's TaoLive AIGC team, converted for ComfyUI by Asirus) at strength 1.0 by ComfyUI's own LoRA merge: each patched int8 ConvRot linear (attention and MLP of the 50
blocks, and the token refiner) was dequantized, had the LoRA added, and was requantized. Every tensor the LoRA does not
touch is byte-identical to the official file, in its original dtype. Same 932 tensors, same layout and size (21.0 GB) as the
official file.

### Running it

Stock ComfyUI with MiniMax H3 support loads it like the official int8 file: put it in `ComfyUI/models/diffusion_models/`
and pick it in **Load Diffusion Model** (weight dtype `default`). Use the rest of the official MiniMax H3 setup
(text encoder `qwen3vl_32b_minimax_h3_nvfp4_awq` or `qwen3vl_32b_minimax_h3_int8_convrot`, VAE
`minimax_h3_video_vae_int8_convrot`, all from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)). As
with the official int8 files, PyTorch built for CUDA 13.0 or newer is recommended.

Sampling (do not add the TaoMate LoRA again; it is already inside):

| Setting | Value |
|---|---|
| Sampler | `res_multistep` (KSamplerSelect) |
| Sigmas | `1.0, 0.9601, 0.8575, 0` (ManualSigmas), i.e. 3 steps |
| Guider | BasicGuider (no CFG) |
| Sampler node | SamplerCustomAdvanced |

The LoRA's own card also suggests `euler` or `er_sde`, the
`simple` scheduler, 3 steps and a video shift of 8-12 (audio 3); those settings apply to this file the same way.

### Samples

The first 3 seconds of each test render (5.2 s were generated), with sound. Five hard text-to-video prompts (`samples/prompts.py`), seed 42, 3 steps with the settings above, RTX 4090 at locked clocks (2235 MHz core), ComfyUI with SageAttention. Times are for the whole 124-frame clip: the sampler, then the total including decoding and saving (the text encoding was cached). For scale, stock MiniMax H3 int8 at 20 steps takes about 51 s (480p) and 184 s (768p) for the sampler on the same card. Full-length videos and frame sheets are in `samples/`.

#### Market dialogue

Two faces, alternating lip sync, two voices. Good at both sizes: both faces stable, the tomato and gestures follow the text.

| 864×480 | 1344×768 |
|---|---|
| <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/dialogue_480p_3s.mp4" controls width="100%"></video><br>sampler 8.7 s · total 16.1 s | <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/dialogue_768p_3s.mp4" controls width="100%"></video><br>sampler 26.3 s · total 35.2 s |

#### Pianist

Fast hands on the keys, music in sync. Hands hold together; a blurred extra hand at the left edge at 480p; the tilt to his face works.

| 864×480 | 1344×768 |
|---|---|
| <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/pianist_480p_3s.mp4" controls width="100%"></video><br>sampler 8.7 s · total 16.1 s | <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/pianist_768p_3s.mp4" controls width="100%"></video><br>sampler 27.5 s · total 38.9 s |

#### Dog and lake

Splash, wet fur, fast animal motion. The weakest: the dog never visibly leaps; it sits in a splash at the end of the dock.

| 864×480 | 1344×768 |
|---|---|
| <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/dog_splash_480p_3s.mp4" controls width="100%"></video><br>sampler 8.7 s · total 16.3 s | <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/dog_splash_768p_3s.mp4" controls width="100%"></video><br>sampler 27.6 s · total 39.0 s |

#### Neon sign in rain

Readable text, rain, reflections, a camera move. "OPEN ALL NIGHT" reads cleanly at 480p; at 768p the sign moves to the window and NIGHT is garbled. Quiet soundtrack.

| 864×480 | 1344×768 |
|---|---|
| <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/neon_rain_480p_3s.mp4" controls width="100%"></video><br>sampler 8.7 s · total 16.4 s | <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/neon_rain_768p_3s.mp4" controls width="100%"></video><br>sampler 27.5 s · total 39.0 s |

#### Skateboard trick

Full-body motion, handheld camera, a crowd. Coherent rider and cheering friends; the kickflip is only clear at 768p.

| 864×480 | 1344×768 |
|---|---|
| <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/skate_480p_3s.mp4" controls width="100%"></video><br>sampler 8.7 s · total 16.4 s | <video src="https://huggingface.co/PedroMarinhoDev/MiniMax-H3-Experimental/resolve/main/samples/clips/skate_768p_3s.mp4" controls width="100%"></video><br>sampler 27.5 s · total 39.2 s |

### License and use restrictions

MiniMax H3 and this file are licensed under the **MiniMax H3 Community License Agreement** (see `LICENSE` and `NOTICE`).
By downloading or using this file you agree to that agreement, including its Acceptable Use Policy (Exhibit A), its use
restrictions (Section V) and its territorial terms: the license does not cover the European Union, the United Kingdom,
the Republic of Korea or the United States of America. The TaoMate LoRA is published under the same license.

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

Credits: [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax H3), [TaoLive AIGC](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)
(TaoMate), [Asirus](https://huggingface.co/Asirus/TaoMate_H3_3_Step_LoRA) (ComfyUI conversion of the LoRA),
[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) (int8 ConvRot repack).
