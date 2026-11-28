# AtomicChat/MiniMax-H3-Image-GGUF

Источник: https://huggingface.co/AtomicChat/MiniMax-H3-Image-GGUF · скачано 2026-11-28
Загрузок: 59617 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3-Image · теги: gguf,comfyui,quantized,base_model:quantized:MiniMaxAI/MiniMax-H3-Image · обновлено: 2026-11-11T13:59:42.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3-Image
base_model_relation: quantized
quantized_by: AtomicChat
pipeline_tag: text-to-image
library_name: gguf
tags:
- gguf
- comfyui
- stable-diffusion.cpp
- minimax-h3-image
- minimax-h3
- text-to-image
- image-editing
- imatrix
- quantized
- atomic-chat
---

# How to Run MiniMax-H3-Image Locally
<p style="margin-top: 0; margin-bottom: 0;">
  <em>MiniMax's MiniMax-H3-Image, the 6-step image generation and editing model of the H3 family, as GGUF for ComfyUI and stable-diffusion.cpp: per-tensor layouts from a sensitivity scan of this model and an importance matrix from its BF16 weights, measured on one stand against Comfy-Org's own files and every other MiniMax-H3-Image GGUF we found. The <a href="https://huggingface.co/datasets/AtomicChat/MiniMax-H3-Image-GGUF-metrics">renders and measurements</a> are public.</em>
</p>
<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">
  <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_minimax_h3_image&utm_content=btn_atomic"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_atomic.png" width="162" alt="Atomic Chat"></a>
  <a href="https://discord.gg/8wGSsvmg4V"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_discord.png" width="119" alt="Discord"></a>
  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_github.png" width="115" alt="GitHub"></a>
  <a href="https://x.com/atomic_chat_hq"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_x.png" width="108" alt="X"></a>
</div>
<ul style="margin: 0 0 12px 0;">
  <li>Built by the team behind <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_minimax_h3_image&utm_content=bullet_app">Atomic Chat</a>, the app that runs open-weight models locally and offline.</li>
  <li>The image model from 7.6 to 21.4 GB on disk for ComfyUI (<b>Unet Loader (GGUF)</b>) and stable-diffusion.cpp; the files load in stock ComfyUI-GGUF as they are. Of the five other MiniMax-H3-Image GGUF repos we found on 11 November, only one loads that way.</li>
  <li>Closer to the original at the same size: <code>AD-Q5_K</code> (13.8 GB) is 16% closer to the BF16 picture than Abiray's <code>Q5_K_M</code> (14.1 GB), <code>AD-Q3_K</code> (8.7 GB) 10 to 20% closer than the three <code>Q3_K_M</code> files (8.8 to 8.9 GB), and our <code>Q8_0</code> 30% closer than Comfy-Org's <code>int8_convrot</code>. At Q4 the pictures are a tie; see <a href="#compared-with-the-other-files">the comparison</a>.</li>
  <li>No uncensored version: the MiniMax-H3-Image license forbids circumventing safety guardrails.</li>
</ul>
<hr style="margin: 0 0 16px 0;">

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) is MiniMax's text-to-image and editing model of
the [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) family: one checkpoint for both, **6 steps, no CFG**, 4 to 20
megapixels, edits with up to 16 reference images. Its diffusion transformer is H3's 33B design without the audio
layers; with the AdaLN branches folded into a small table (Comfy-Org's pruned repack) it is 20.1B weights, 40.2 GB in
BF16. This repo has that image model as GGUF. The text encoder (MiniMax-H3's Qwen3-VL-32B, unchanged) and the VAE stay
the files you already use: Comfy-Org's for ComfyUI, or any H3 encoder GGUF for stable-diffusion.cpp.

## Pick a file

Every file rendered the same 40 captions at 2048×2048 from the same starting noise as the BF16 weights. Two numbers per
file, both against BF16:

- **Output error** follows the BF16 path: at each of the 6 steps the file gets exactly the input the BF16 model got,
  and we measure how far its prediction lands (relative L2; 0 = identical). It is the stable number.
- **LPIPS** is the picture you actually get, compared with the BF16 picture of the same caption and seed (lower is
  closer). `vs. new seed` puts it on a scale: 100% is how far BF16 itself moves when only the seed changes
  (LPIPS 0.627).

| File | Size on disk | Bits per weight | Output error [95% CI] | LPIPS [95% CI] | vs. new seed |
|---|---:|---:|---|---|---:|
| `BF16` (Comfy-Org's `minimax_h3_image_pruned_bf16.safetensors`, default attention) | 40.23 GB | 16 | 0 | 0 | 0% |
| `Q8_0` | 21.45 GB | 8.53 | 0.020 [0.017, 0.023] | 0.090 [0.072, 0.111] | 14% |
| **`AD-Q6_K`** | **16.67 GB** | 6.63 | 0.033 [0.027, 0.039] | 0.130 [0.105, 0.159] | 21% |
| **`AD-Q5_K`** | **13.84 GB** | 5.51 | 0.049 [0.041, 0.057] | 0.164 [0.138, 0.191] | 26% |
| **`AD-Q4_K`** | **11.57 GB** | 4.60 | 0.076 [0.064, 0.088] | 0.254 [0.224, 0.285] | 41% |
| **`AD-Q3_K`** | **8.73 GB** | 3.47 | 0.132 [0.117, 0.147] | 0.334 [0.305, 0.366] | 53% |
| **`AD-Q2_K`** | **7.57 GB** | 3.01 | 0.170 [0.152, 0.187] | 0.379 [0.349, 0.409] | 60% |

- **`Q8_0`** is the closest to the original. Pick it if it fits. It is closer to the BF16 picture than Comfy-Org's
  own BF16 file as ComfyUI ran it on our H100 (that file switches on int8 attention where ComfyUI offers it: LPIPS
  0.109, ours 0.090, 32 of 40 captions closer), and 30% closer than Comfy-Org's `int8_convrot` (0.129) at 0.5 GB more.
- **`AD-Q6_K`** matches Comfy-Org's `int8_convrot` picture for picture (0.130 vs 0.129) at 4.3 GB less.
- **`AD-Q5_K`** and **`AD-Q4_K`** keep the composition of most pictures; small objects, faces and text strokes move.
- **`AD-Q3_K`** and **`AD-Q2_K`** are for tight memory: more pictures change their layout (53% and 60% of a new
  seed); text and objects stay readable in the samples below. `AD-Q2_K` has a Q2_K base but 3.0 bits per weight: the
  sensitive last 20 blocks are at Q3_K-Q4_K.
- The text encoder comes on top: Comfy-Org's int8 encoder is 27.1 GB on disk (`nvfp4_awq` 15.7 GB, unsloth's
  `Q4_K_M` GGUF 18.2 GB); ComfyUI moves the encoder and the image model in and out of VRAM as needed.

![Quality against size: LPIPS and output error of every file](quality-vs-size.png)

![BF16 and our files, four captions, seed 42](samples-grid.png)

## Compared with the other files

Comfy-Org's own smaller files and the GGUFs of the five other MiniMax-H3-Image GGUF repos we found on 11 November,
on the same stand. Hearthscale's `Q3_K_M`, `Q4_K_M` and `Q5_K_M` are the plain rows of the next table (the same
tensor data); realrebelai's other four files and Abiray's `Q4_K_S` were not measured. "Loads" means stock ComfyUI
with ComfyUI-GGUF `6ea2651` (current main) and **Unet Loader (GGUF)**, nothing patched. Files that do not load were
measured through a copy with the metadata ComfyUI-GGUF needs written in (the architecture, and the `config` key where
it was missing; tensor data unchanged byte for byte), so the quality columns show what the weights can do once a
loader accepts them.

| File | Size on disk | Loads as published | Output error | LPIPS [95% CI] | Closest of ours (LPIPS) |
|---|---:|---|---|---|---|
| Comfy-Org `minimax_h3_image_pruned_int8_convrot.safetensors` | 20.98 GB | yes | - | 0.129 [0.108, 0.151] | `Q8_0` 21.45 GB: 0.090 [0.072, 0.111] |
| Comfy-Org `minimax_h3_image_pruned_w4a8.safetensors` | 12.55 GB | yes | - | 0.274 [0.242, 0.306] | `AD-Q4_K` 11.57 GB: 0.254 [0.224, 0.285] |
| Comfy-Org `minimax_h3_image_pruned_bf16.safetensors` as published (int8 attention) | 40.23 GB | yes | - | 0.109 [0.090, 0.129] | `Q8_0` 21.45 GB: 0.090 [0.072, 0.111] |
| Hearthscale `Q8_0` | 21.45 GB | no: no metadata at all | - | 0.098 [0.080, 0.117] | `Q8_0` 21.45 GB: 0.090 [0.072, 0.111] |
| LSDJesus `Q6_K` | 17.52 GB | no: architecture `minimax_h3`, not yet accepted by ComfyUI-GGUF | - | 0.137 [0.115, 0.160] | `AD-Q6_K` 16.67 GB: 0.130 [0.105, 0.159] |
| realrebelai `Q6_K` | 16.74 GB | yes | - | 0.131 [0.111, 0.151] | `AD-Q6_K` 16.67 GB: 0.130 [0.105, 0.159] |
| lingjiu09 `Q4_0` | 14.22 GB | no: architecture `minimax_h3`, not yet accepted by ComfyUI-GGUF | - | 0.295 [0.268, 0.323] | `AD-Q5_K` 13.84 GB: 0.164 [0.138, 0.191] |
| Abiray `Q5_K_M` | 14.08 GB | no: no `config` key, so ComfyUI stops at a missing `pdd_sigmas` | - | 0.195 [0.164, 0.232] | `AD-Q5_K` 13.84 GB: 0.164 [0.138, 0.191] |
| LSDJesus `Q4_K_imat` | 12.55 GB | no: architecture `minimax_h3`, not yet accepted by ComfyUI-GGUF | 0.090 [0.080, 0.100] | 0.255 [0.227, 0.285] | `AD-Q4_K` 11.57 GB: 0.254 [0.224, 0.285] |
| Abiray `Q4_K_M` | 11.57 GB | no: no `config` key, so ComfyUI stops at a missing `pdd_sigmas` | 0.097 [0.083, 0.110] | 0.247 [0.223, 0.273] | `AD-Q4_K` 11.57 GB: 0.254 [0.224, 0.285] |
| LSDJesus `Q3_K_imat` | 9.99 GB | no: architecture `minimax_h3`, not yet accepted by ComfyUI-GGUF | - | 0.363 [0.334, 0.394] | `AD-Q3_K` 8.73 GB: 0.334 [0.305, 0.366] |
| LSDJesus `Q2_K_mix_imat` | 9.25 GB | no: architecture `minimax_h3`, not yet accepted by ComfyUI-GGUF | - | 0.429 [0.401, 0.457] | `AD-Q3_K` 8.73 GB: 0.334 [0.305, 0.366] |
| Abiray `Q3_K_M` | 8.91 GB | no: no `config` key, so ComfyUI stops at a missing `pdd_sigmas` | - | 0.397 [0.366, 0.430] | `AD-Q3_K` 8.73 GB: 0.334 [0.305, 0.366] |
| lingjiu09 `Q3_K_M` | 8.81 GB | no: architecture `minimax_h3`, not yet accepted by ComfyUI-GGUF | - | 0.418 [0.391, 0.445] | `AD-Q3_K` 8.73 GB: 0.334 [0.305, 0.366] |
| Abiray `Q2_K` | 6.88 GB | no: no `config` key, so ComfyUI stops at a missing `pdd_sigmas` | - | 0.546 [0.527, 0.567] | `AD-Q2_K` 7.57 GB: 0.379 [0.349, 0.409] |

- **Loads.** Our files load in stock ComfyUI-GGUF as they are. Of the other files we tried only realrebelai's `Q6_K`
  does: Abiray's files lack the `config` key that tells ComfyUI the 6 PDD steps; LSDJesus's and lingjiu09's declare
  the architecture `minimax_h3`, which ComfyUI-GGUF does not accept yet (city96/ComfyUI-GGUF#493 is open);
  Hearthscale's carry no metadata. They may work in forks of the node or in stable-diffusion.cpp, which ignores the
  metadata.
- **Same size, closer picture** (paired over the same 40 captions): `AD-Q5_K` vs Abiray's `Q5_K_M` 0.164 vs 0.195
  (31 of 40 closer), `AD-Q3_K` vs Abiray's `Q3_K_M` 0.334 vs 0.397 (34 of 40) and vs lingjiu09's `Q3_K_M` 0.418
  (37 of 40), `AD-Q2_K` vs Abiray's `Q2_K` 0.379 vs 0.546 (40 of 40, ours is 0.7 GB larger), `AD-Q4_K` vs
  Comfy-Org's `w4a8` 0.254 vs 0.274 at 1 GB less.
- **Where we do not win:** at Q4 the free-running pictures are a tie with Abiray's `Q4_K_M` (0.247, same 11.57 GB)
  and LSDJesus's `Q4_K_imat` (0.255, 12.55 GB); our output error is 22% and 16% lower (16 of 16 and 15 of 16
  captions), but a 6-step model redraws enough of a picture at Q4 that LPIPS cannot separate them. At Q6
  realrebelai's and LSDJesus's `Q6_K` tie with `AD-Q6_K`, and Hearthscale's `Q8_0` ties with ours.
- LSDJesus's files also keep Comfy-Org's int8-attention entries, so they run int8 attention in ComfyUI; their numbers
  include that.

## What the layouts change

The plain types are what stable-diffusion.cpp's converter writes for one type everywhere. The AD files keep the
same base type and move bits by a sensitivity scan of this model: each tensor kind (attention qkv and output, the two
MLP matrices) and each band of ten blocks was pushed to Q2_K alone, and a planner spent the bytes where the output
error drops most per byte.

| Size class | Plain sd.cpp type | Output error | LPIPS | AD file | Output error | LPIPS |
|---|---|---|---|---|---|---|
| Q6 | `Q6_K` 16.60 GB | 0.033 [0.029, 0.037] | - | `AD-Q6_K` 16.67 GB | 0.033 [0.027, 0.039] | 0.130 [0.105, 0.159] |
| Q5 | `Q5_K` 13.93 GB | 0.060 [0.050, 0.070] | 0.195 [0.165, 0.226] | `AD-Q5_K` 13.84 GB | 0.049 [0.041, 0.057] | 0.164 [0.138, 0.191] |
| Q4 | `Q4_K` 11.43 GB | 0.093 [0.080, 0.105] | 0.258 [0.230, 0.286] | `AD-Q4_K` 11.57 GB | 0.076 [0.064, 0.088] | 0.254 [0.224, 0.285] |
| Q3 | `Q3_K` 8.77 GB | 0.165 [0.145, 0.183] | 0.372 [0.343, 0.400] | `AD-Q3_K` 8.73 GB | 0.132 [0.117, 0.147] | 0.334 [0.305, 0.366] |
| Q2 | `Q2_K` 6.73 GB | 0.265 [0.242, 0.287] | - | `AD-Q2_K` 7.57 GB | 0.170 [0.152, 0.187] | 0.379 [0.349, 0.409] |

- Hearthscale's `Q3_K_M`, `Q4_K_M` and `Q5_K_M` carry the same tensor data byte for byte as the plain `Q3_K`,
  `Q4_K` and `Q5_K` rows (stable-diffusion.cpp's converter without an importance matrix); they were not rendered
  again.
- The importance matrix alone lowers the output error by 5% at Q3_K and not at all at Q4_K; the layout does the rest, as
  with P-H3-Image.
- The published `AD-Q4_K` is row `AD-Q4_K-c` in the metrics (no tensor below Q3_K; 11.57 GB). The
  first planner row `AD-Q4_K` (11.37 GB, Q2_K floor) measured 0.262 / 0.079 and is not published.

## In ComfyUI

1. **ComfyUI with MiniMax-H3-Image support** (Comfy-Org/ComfyUI#17655, 10 November; we measured with `4d64a8b`) and
   [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).
2. **Image model:** one of these files in `models/unet/`, loaded with **Unet Loader (GGUF)**.
3. **Text encoder and VAE:** from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image):
   `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (or `_nvfp4_awq`) with **Load CLIP**, type
   `minimax`, and `vae/minimax_h3_image_vae_fp16.safetensors` with **Load VAE**.
4. **Graph:** the ComfyUI template *MiniMax H3 Image* (`image_minimax_h3_image_t2i`), with **Unet Loader (GGUF)** in
   place of **Load Diffusion Model**. Keep its prompt expander: the model expects a long, detailed English
   description of the whole image, and its 6-step schedule comes from the file.

The files carry `general.architecture = wan` (the value ComfyUI-GGUF accepts today; ComfyUI itself recognises the
model from its tensors) and the checkpoint's `config` with the 6 PDD steps and sigmas, which ComfyUI needs to build
the model.

## In stable-diffusion.cpp

MiniMax-H3-Image support is in stable-diffusion.cpp `master` since leejet/stable-diffusion.cpp#2324 (10 November).
Built from `32d516c`, our `AD-Q4_K` renders at 2048×2048 with unsloth's H3 encoder GGUF and Comfy-Org's VAE:

```bash
sd-cli --diffusion-model MiniMax-H3-Image-AD-Q4_K.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_image_vae_fp16.safetensors \
  -p "$(cat caption.txt)" -W 2048 -H 2048 --steps 6 --cfg-scale 1 --sigmas minimax-h3-image --diffusion-fa -o out.png
```

The encoder is MiniMax-H3's Qwen3-VL-32B unchanged (the same files as in `MiniMaxAI/MiniMax-H3`, `text_encoder/`), so
an H3 encoder GGUF you already have works, e.g. `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` from
[unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF). sd.cpp renders the prompt as given: write a
full caption, or expand it first.

Atomic Chat support is in progress ([app](https://github.com/AtomicBot-ai/Atomic-Chat/pull/413), [catalog](https://github.com/AtomicBot-ai/atomic-chat-conf/pull/54)): the app
needs its stable-diffusion.cpp engine at build 1088 or newer, which is being built now.

## How it was measured

- **Engine:** ComfyUI `4d64a8b` + ComfyUI-GGUF `6ea2651` on NVIDIA H100 80 GB, PyTorch 2.9.1 + CUDA 12.8.
- **Graph:** the Comfy-Org template without the prompt expander: MiniMax H3 Image conditioning at 2048×2048 (the
  model's default area), KSampler 6 steps, cfg 1, euler, simple, seed 42 (43 for the floor).
- **Reference:** Comfy-Org's `minimax_h3_image_pruned_bf16.safetensors` with ComfyUI's default attention. That file
  also asks ComfyUI for int8 attention (50 small `comfy_attention.config` entries); we measured it as published too
  (row "int8 attention" above) and left those entries out of our GGUFs, which run full-precision attention.
- **Held constant:** Comfy-Org's int8 text encoder and fp16 VAE, so the image model file is the only thing that
  changes.
- **Captions:** 40 requests in five categories (photo, illustration, English text, Chinese text, several objects with
  attributes), each expanded once into a long English description by an LLM prompt expander (the same captions as
  our P-H3-Image measurements). The importance matrix used 12 other captions.
- **Output error:** 16 of the captions × 6 steps, the BF16 inputs recorded once and replayed through each file.
- **Not measured:** edits, other resolutions and samplers, speed, aesthetics or prompt adherence as such.
- **In the [metrics repo](https://huggingface.co/datasets/AtomicChat/MiniMax-H3-Image-GGUF-metrics):** every render,
  the captions, the importance matrix, the scan, the per-tensor layout of every file and every log.

## How these files were made

1. Comfy-Org's `diffusion_models/minimax_h3_image_pruned_bf16.safetensors` from
   [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), its 50 attention-method entries
   left out.
2. Converted with `sd-cli -M convert` from stable-diffusion.cpp `32d516c` with `--tensor-type-rules` and
   `--imat-in`; then `general.architecture`, `general.quantization_version` and `config` are written into the
   metadata (the tensors are byte for byte the converter's).
3. The importance matrix comes from ComfyUI itself: a hook on every linear layer of the BF16 model while it rendered
   the calibration captions, written in stable-diffusion.cpp's format.
4. **Kept out of quantization in every file:** the AdaLN table and projections, the norms and biases (BF16;
   ComfyUI-GGUF cannot read quantized 1-D tensors), the patch layer and the 6 PDD output heads (F32) and
   `condition_proj` (BF16).

## License

These files are a modified MiniMax-H3-Image Work under the
[MiniMax-H3-Image Community License Agreement](LICENSE); see [NOTICE](NOTICE). MiniMax-H3-Image is licensed under the
MiniMax-H3-Image Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

- The license is limited to its Applicable Territory: worldwide **excluding the European Union, the United Kingdom,
  the Republic of Korea and the United States** (section I.5). Read it before you use these files.
- Its Acceptable Use Policy (Exhibit A) applies to every use, including the bans on circumventing safety guardrails
  and on harming minors, and the duty to disclose machine-generated content in public.
- A separate written authorization from MiniMax is needed above US$1M annualized revenue or to offer a hosted
  service, and commercial products must display "MiniMax-H3-Image" (section IV).
- The base model is [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by MiniMax.
  Quantized by AtomicChat; not affiliated with or endorsed by MiniMax or Comfy-Org.
