# unsloth/MiniMax-H3-Image-GGUF

Источник: https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF · скачано 2026-11-28
Загрузок: 324826 · лайков: 63 · базовая: ["MiniMaxAI/MiniMax-H3-Image"] · теги: gguf,quantized,base_model:quantized:MiniMaxAI/MiniMax-H3-Image · обновлено: 2026-11-19T06:40:25.000Z

---

---
base_model:
- MiniMaxAI/MiniMax-H3-Image
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- unsloth
- quantized
- stable-diffusion.cpp
- minimax-h3
- image-generation
- image-editing
- text-to-image
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
language:
- en
pipeline_tag: text-to-image
widget:
- text: "a wide photoreal landscape of a glacial lake in the mountains at sunrise, mist over still turquoise water, pine forest on the shore, snow-capped peaks reflected in the lake, golden light, sharp detail"
  output:
    url: assets/lake.png
- text: "a storefront sign that says \"UNSLOTH\" in bold clean letters, daytime, sharp focus"
  output:
    url: assets/sign.png
- text: "a close-up of two human hands carefully folding an origami crane, anatomically correct fingers, soft studio light"
  output:
    url: assets/origami.png
---
# Run MiniMax-H3-Image FP8 or GGUF with [Unsloth Desktop](https://unsloth.ai/docs/models/minimax-h3-image)

These are GGUF quantized versions of the [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) diffusion transformer (DiT), for text-to-image and image editing on GPUs that cannot hold the 66 GB bf16 DiT, on Apple Silicon and on the CPU. <br>
- **Recommended:** `UD-Q4_K_XL` (13.61 GB): same-seed LPIPS 0.172 vs the bf16 DiT at 1024x1024 (lower is better), 14.8 GiB peak VRAM on an A100. For output closest to bf16 use `Q8_0` (LPIPS 0.083); on tight memory use `UD-Q2_K_XL` (8.68 GB, LPIPS 0.347), which is clearly ahead of the smallest file, `Q2_K` (7.75 GB, LPIPS 0.438).
- Runs on a 24 GB L4 at about 1.2 s per step at 1024x1024 with `sd-cli` (1.7 s through Unsloth Studio), and with `--max-vram 12` it renders the same image byte for byte at 3.9 s per step. On the CPU alone, 512x512 takes about 47 s per image.
- Loads in upstream [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MiniMax-H3-Image support landed in [#2324](https://github.com/leejet/stable-diffusion.cpp/pull/2324)) and in [unslothai/stable-diffusion.cpp](https://github.com/unslothai/stable-diffusion.cpp). `UD-` files are Unsloth Dynamic quants.
<div>
  <div style="display: flex; gap: 5px; align-items: center; ">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
    <a href="https://discord.gg/unsloth">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png" width="173">
    </a>
    <a href="https://unsloth.ai/docs">
      <img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="143">
    </a>
  </div>
</div>

| file | size | LPIPS 1024 vs bf16 (95% interval) | PSNR 1024 | LPIPS 2048 | peak VRAM, 1024x1024 |
|---|---|---|---|---|---|
| `minimax_h3_image_pruned-Q2_K.gguf` | 7.75 GB | 0.438 (0.375 to 0.481) | 13.9 dB | 0.435 | 9.4 GiB |
| `minimax_h3_image_pruned-UD-Q2_K_XL.gguf` | 8.68 GB | 0.347 (0.261 to 0.414) | 16.1 dB | 0.332 | 10.2 GiB |
| `minimax_h3_image_pruned-Q3_K_S.gguf` | 8.77 GB | 0.347 (0.262 to 0.414) | 16.1 dB | 0.385 | 10.3 GiB |
| `minimax_h3_image_pruned-Q3_K_M.gguf` | 10.42 GB | 0.271 (0.199 to 0.339) | 17.5 dB | 0.288 | 11.9 GiB |
| `minimax_h3_image_pruned-UD-Q3_K_XL.gguf` | 11.36 GB | 0.242 (0.164 to 0.324) | 18.1 dB | 0.233 | 12.7 GiB |
| `minimax_h3_image_pruned-Q4_K_S.gguf` | 11.54 GB | 0.225 (0.150 to 0.308) | 18.5 dB | 0.211 | 12.9 GiB |
| `minimax_h3_image_pruned-Q4_K_M.gguf` | 12.72 GB | 0.195 (0.136 to 0.246) | 19.2 dB | 0.196 | 14.0 GiB |
| `minimax_h3_image_pruned-UD-Q4_K_XL.gguf` (recommended) | 13.61 GB | 0.172 (0.110 to 0.235) | 20.4 dB | 0.171 | 14.8 GiB |
| `minimax_h3_image_pruned-Q5_K_S.gguf` | 13.93 GB | 0.173 (0.107 to 0.242) | 21.0 dB | 0.197 | 15.2 GiB |
| `minimax_h3_image_pruned-Q5_K_M.gguf` | 14.60 GB | 0.157 (0.098 to 0.210) | 21.4 dB | 0.133 | 15.8 GiB |
| `minimax_h3_image_pruned-UD-Q5_K_XL.gguf` | 15.47 GB | 0.154 (0.092 to 0.211) | 22.2 dB | 0.113 | 16.8 GiB |
| `minimax_h3_image_pruned-Q6_K.gguf` | 16.60 GB | 0.134 (0.076 to 0.194) | 22.8 dB | 0.158 | 17.6 GiB |
| `minimax_h3_image_pruned-UD-Q6_K_XL.gguf` | 17.58 GB | 0.130 (0.072 to 0.185) | 23.2 dB | 0.128 | 18.8 GiB |
| `minimax_h3_image_pruned-Q8_0.gguf` | 22.03 GB | 0.083 (0.039 to 0.129) | 26.8 dB | 0.048 | 23.0 GiB |

Each file replaces the bf16 `transformer` and its per-step heads. The VAE is the release image VAE, [`vae/diffusion_pytorch_model.safetensors`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/vae), and the text encoder is the MiniMax-H3 Qwen3-VL-32B encoder: a GGUF of it is in [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) (`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, 18.2 GB). GGUF loading runs no arbitrary code. These files are Model Derivatives of MiniMax-H3-Image, distributed under its license: the DiT is quantized to GGUF and its time modulation stored in curve form.

### Usage

Text-to-image:

```bash
sd-cli --diffusion-model minimax_h3_image_pruned-UD-Q4_K_XL.gguf \
  --vae diffusion_pytorch_model.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "A weathered lighthouse keeper in a yellow oilcloth coat stands on a wet stone pier at dusk ..." \
  -W 1024 -H 1024 --cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image --rng cpu
```

Image editing: add one or more reference images with `-r` (or `--ref-image`); in argument order they are `<Picture 1>`, `<Picture 2>`, and so on in the prompt:

```bash
sd-cli --diffusion-model minimax_h3_image_pruned-UD-Q4_K_XL.gguf \
  --vae diffusion_pytorch_model.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -r reference1.png -p "Edit <Picture 1>. ..." \
  -W 1024 -H 1024 --cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image --rng cpu
```

For editing, set `-W` / `-H` to the reference image's aspect ratio; a square size crops a wide reference. MiniMax-H3-Image is distilled to 6 steps with no CFG: keep `--cfg-scale 1.0`, `--steps 6` and the `minimax-h3-image` sigma preset. On small GPUs add `--offload-to-cpu` or cap the card with `--max-vram <GB>`; on Macs and CPU-only machines the whole pipeline runs from RAM.

### Samples

Rendered with `UD-Q4_K_XL` and the `Q4_K_M` text encoder through Unsloth Studio's stable-diffusion.cpp backend on an NVIDIA L4, 1024x1024, 6 steps, seed 1234.

<table>
  <tr>
    <td><img src="assets/lake.png" width="256"></td>
    <td><img src="assets/sign.png" width="256"></td>
    <td><img src="assets/origami.png" width="256"></td>
  </tr>
</table>

Image editing with the release example: the reference (left) and the edit (right), same settings.

<table>
  <tr>
    <td><img src="assets/edit_reference.png" width="256"></td>
    <td><img src="assets/edit_podcast.png" width="256"></td>
  </tr>
</table>

### Accuracy

Same prompts, seeds, settings and text-encoder conditioning for every file; only the DiT changes. LPIPS (AlexNet, lower is better) and PSNR against the bf16 DiT rendered by the same stable-diffusion.cpp CUDA build: 8 prompts x 2 seeds at 1024x1024 (95% interval over prompts), plus 4 prompts at 2048x2048, on the 6-step release schedule. A repeated bf16 render is bit-identical. Peak VRAM is DiT sampling plus VAE decode on an A100 40 GB; the text encoder runs first and needs its own memory (or `--offload-to-cpu`).

MiniMax-H3-Image is sensitive to small changes in its inputs: a text encoder quantized to Q4_K_M instead of bf16 alone moves a 1024x1024 image by LPIPS 0.17. Lower-bit files keep the subject, style and text of the prompt but move details and layout more, which is what the LPIPS column tracks.

The files were also run on an NVIDIA L4 (24 GB), an RTX PRO 6000 Blackwell and the CPU. On the L4, `Q8_0`, `UD-Q4_K_XL` and `UD-Q2_K_XL` rank in the same order as on the A100 on every prompt. Through Unsloth Studio, Q8_0 does not fit next to the compute buffers on a 24 GB card and partly runs from host RAM there (about 14 s per step), so on 24 GB cards `UD-Q4_K_XL` is the faster choice there; plain `sd-cli` on the same L4 ran Q8_0 at 1.24 s per step. Image editing with `UD-Q4_K_XL` on the L4 stays within LPIPS 0.025 to 0.066 of the bf16 edit on three release examples.

| `UD-Q4_K_XL`, 1024x1024, 6 steps | s per step | s per image, DiT + VAE | s per image with the Q4_K_M text encoder |
|---|---|---|---|
| NVIDIA L4 24 GB, `sd-cli` | 1.12 to 1.18 | | about 105 to 111 |
| NVIDIA L4 24 GB, Unsloth Studio | 1.74 | 12.1 | 71.4 |
| NVIDIA L4, `--max-vram 12` | 3.91 | 25.1 | 88.1 |
| NVIDIA A100 40 GB | 1.25 | | |
| RTX PRO 6000 Blackwell 96 GB | 0.40 | 2.8 | 3.9 |
| CPU, 16 threads, 512x512 | 6.7 to 7.3 | 47 to 50 | |

On the L4 most of the end-to-end time is the 32B text encoder, which does not fit next to the DiT and runs largely from host RAM.

---

<div align="center">
  <img width="100%" src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/minimax-h3-image.jpg" alt="MiniMax H3 image, Open Weights">
</div>

<p align="center">
  <a href="https://hailuoai.video" target="_blank"><img src="https://img.shields.io/badge/Hailuo%20AI-FF6C37?logo=minimax&logoColor=white" alt="Hailuo AI"></a>
  <a href="https://platform.minimax.io" target="_blank"><img src="https://img.shields.io/badge/API-FF6C37?logo=minimax&logoColor=white" alt="API"></a>
  <a href="https://www.minimax.io" target="_blank"><img src="https://img.shields.io/badge/MiniMax%20Website-FF6C37?logo=minimax&logoColor=white" alt="MiniMax Website"></a>
  <a href="https://github.com/MiniMax-AI/MiniMax-H3" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image" target="_blank"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <br>
  <a href="https://modelscope.cn/organization/minimax" target="_blank" rel="noopener noreferrer"><img alt="ModelScope MiniMax AI" src="https://img.shields.io/badge/ModelScope-MiniMax%20AI-white?labelColor=%23EF3D5D"></a>
  <a href="https://platform.minimaxi.com/docs/faq/contact-us" target="_blank"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="https://discord.com/invite/dbMxutw7tP" target="_blank"><img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-4CAF50?logo=creativecommons&logoColor=white" alt="LICENSE"></a>
</p>

# MiniMax-H3-Image

## Online API

Use MiniMax-H3-Image directly via API.

- Global: [platform.minimax.io](https://platform.minimax.io/docs/guides/image-generation) | CN: [platform.minimax.cn](https://platform.minimax.cn/docs/guides/image-generation)

API docs:
- Text-to-image: [EN-docs](https://platform.minimax.io/docs/api-reference/image-generation-h3-generations), [CN-docs](https://platform.minimax.cn/docs/api-reference/image-generation-h3-generations)
- Image editing: [EN-docs](https://platform.minimax.io/docs/api-reference/image-generation-h3-edits), [CN-docs](https://platform.minimax.cn/docs/api-reference/image-generation-h3-edits)
- Prompt expansion: [EN-docs](https://platform.minimax.io/docs/api-reference/image-generation-h3-context-ir), [CN-docs](https://platform.minimax.cn/docs/api-reference/image-generation-h3-context-ir)

## Online App

Use MiniMax-H3-Image directly via App.

- WebApp Global: [hailuoai.video](https://hailuoai.video/) | CN: [hailuoai.com](https://hailuoai.com/)
- Desktop Global: [agent.minimax.io](https://agent.minimax.io/) | CN: [agent.minimax.cn](https://agent.minimax.cn)

## Introduction

MiniMax-H3-Image is a state-of-the-art open-source model for image generation and editing from
the [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) family. It combines intelligent
understanding of creative intent with the ability to build visual worlds: with precise
instruction following, it develops a creative concept into concrete characters, scenes and
atmosphere. Generation and editing are unified in one checkpoint, which generates images from
text (text-to-image) and edits one or more reference images (image editing), giving creators
and developers a flexible and controllable way to create images.

It generates images of about 16 megapixels at every aspect ratio, from 4096x4096 (1:1) to
6272x2688 (21:9), directly and without an upscaler (see [Sizes](#sizes)). The checkpoint is
distilled to 6 sampling steps (6 NFEs) with no classifier-free guidance. Its diffusion
transformer (DiT) has the H3 architecture without the audio layers, and it uses the
MiniMax-H3 text encoder unchanged.

![Fifteen MiniMax-H3-Image text-to-image results in landscape and portrait formats: a 3D fantasy battle, a blue-green landscape painting, a tree frog, an indigo-dyed rabbit, a 3D white-haired girl, a portrait among flowers, a sci-fi city, dumplings, a highland cow, a mirror portrait, a 3D otter and sheep, a pixel art cat, a jewel beetle, giant cherries and a fisheye selfie](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase_hero.jpg)

**Highlights**

MiniMax-H3-Image brings together three core strengths: speed, image quality, and instruction
following.

1. **16-megapixel images at a practical computational cost.** Compared with the 1 to 4 MP outputs
   common today, 16 MP (4096 × 4096) images offer far more detail but typically require
   substantially more computation. MiniMax-H3-Image keeps that computational cost manageable,
   making 16 MP generation practical for everyday creation.
2. **Rich detail. Remarkable realism.** MiniMax-H3-Image goes beyond the familiar “AI look” to
   produce images with intricate detail and lifelike realism, opening up more possibilities for
   creative and professional use. It brings generated imagery closer to the richness and nuance
   of the real visual world.
3. **Stronger instruction following.** Greater creative control. Great image generation goes
   beyond beautiful pictures: it helps creators bring their ideas to life. MiniMax-H3-Image
   delivers the strongest instruction following among open-source models and supports up to 16
   reference images, giving creators more control over the final result.

## Model Zoo

| Model | Role | Precision | Size | Hugging Face |
| --- | --- | --- | --- | --- |
| **MiniMax-H3-Image** | Text-to-image and image editing | BF16 (DiT) | 68.65 GB | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) |
| **MiniMax-H3-Image-PE** (optional) | Prompt expander for text-to-image | NVFP4 | 17.9 GB | [MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) |

## Model Architecture

MiniMax-H3-Image follows the H3-Base design with the audio branch removed. The text
prompt and the reference images are encoded by the H3 text encoder; the reference images are
also encoded by the image VAE. All tokens form one packed sequence for a single-stream
transformer, which predicts the latent of the target image. The VAE decodes that latent
into the image.

- **Text encoder**: the full Qwen3-VL text encoder of MiniMax-H3, unchanged. The DiT reads
  the hidden states of its 50th layer.
- **Image VAE**: a CNN encoder and a 2D ViT decoder (24 layers), with 16x spatial
  downsampling and 24 latent channels. The DiT patchifies the latent with a 2x2 patch, so each
  transformer token covers 32x32 pixels.
- **Diffusion transformer**: 33.1B parameters, a dense single-stream transformer with 50
  layers (hidden size 5376, 56 attention heads, FFN size 14336) and 3D multimodal RoPE, the same
  as the H3-Omni-Transformer. The weights are BF16; the embedding, timestep and output layers
  are FP32.

## Quick Start

### Download

| Component | Content | Size |
| --- | --- | --- |
| `transformer/` | DiT: BF16, with the embedding, timestep and output layers in FP32. | 66.28 GB |
| `pdd_fused_heads.safetensors`, `pdd_config.json` | The per-step output heads that replace the transformer's output layer, and the sampling schedule. | 12 MB |
| `vae/` | Image VAE, FP32. | 2.34 GB |
| `tokenizer/`, `processor/` | Qwen3-VL tokenizer and processor, identical to MiniMax-H3. | 23 MB |
| Text encoder | Not in this repository. `model_index.json` loads it from [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3), folder `text_encoder/`. If you already have MiniMax-H3, it is not downloaded again. | 66.7 GB |

```text
MiniMax-H3-Image/
├── model_index.json
├── pdd_config.json
├── pdd_fused_heads.safetensors
├── processor/
├── tokenizer/
├── transformer/
└── vae/
```

Download this repository and the MiniMax-H3 text encoder into the Hugging Face cache:

```bash
hf download MiniMaxAI/MiniMax-H3-Image
hf download MiniMaxAI/MiniMax-H3 --include "text_encoder/*"
# optional prompt expander
hf download MiniMaxAI/MiniMax-H3-Image-PE
```

### Local Deployment

We recommend the following inference frameworks to serve the model:

- [SGLang](https://docs.sglang.io/) - see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-image)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - see [vllm-omni recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image)
- [diffusers](https://github.com/huggingface/diffusers) - see [diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)

#### vLLM-Omni Deployment

See the [vLLM-Omni recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image) for
per-hardware deployment configurations and measured latency. Both NVIDIA and AMD accelerators are
supported, including NVIDIA H200, RTX PRO 6000 Blackwell, AMD Instinct MI350X and more.

Start the server on one GPU:

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni --host 0.0.0.0 --port 8000
```

For lower latency, split each request over several GPUs with sequence parallelism
(Ulysses), for example on 4 GPUs:

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni --host 0.0.0.0 --port 8000 \
  --num-gpus 4 --usp 4 --text-encoder-tp-size 4
```

#### SGLang Deployment

Here we use SGLang as a deployment example. See the [MiniMax-H3-Image deployment guide](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-image) for
additional deployment configurations.

Start the server on one GPU:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3-Image --host 0.0.0.0 --port 30000 --warmup-mode off
```

To use a local copy of the MiniMax-H3 `text_encoder/` folder instead of the Hugging Face
cache, add `--text-encoder-path <path to text_encoder>`.

For lower latency, split each request over several GPUs with sequence parallelism
(Ulysses), for example on 4 GPUs:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3-Image --host 0.0.0.0 --port 30000 --warmup-mode off \
  --num-gpus 4 --ulysses-degree 4
```

#### Request parameters

Text-to-image requests are JSON sent to `/v1/images/generations`. Image editing requests send
the reference images as multipart form data to `/v1/images/edits`; their order defines
`<Picture 1>`, `<Picture 2>`, and so on in the prompt. Both return the PNG in
`data[0].b64_json`. Runnable examples are in [Full Workflow](#full-workflow).

| Field | Text-to-image | Image editing | Notes |
| --- | --- | --- | --- |
| `prompt` | Required | Required | Used as given, truncated to 4096 tokens. See [Full Workflow](#full-workflow) |
| `size` | Required | Required | `WIDTHxHEIGHT`. The returned image has exactly this size; see [Sizes](#sizes) |
| `image` | - | One or more files | In upload order they are `<Picture 1>`, `<Picture 2>`, ... in the prompt |
| `seed` | Recommended | Recommended | Integer |
| `generator_device` | `cpu` recommended | `cpu` recommended | Same seed, same initial noise on every machine |
| `response_format` | `b64_json` | `b64_json` | |
| `min_generation_area` | Optional | Optional | Minimum generated area in pixels, 1024-16777216, default 4194304 (2048x2048). A smaller `size` is generated at the same aspect ratio on a larger canvas and resized back with LANCZOS; pass 1024 to turn this off. All sizes in the table below are above the default |
| `num_inference_steps` | Omit or 6 | Omit or 6 | Other values return HTTP 400 |
| `guidance_scale` | Omit or 1 | Omit or 1 | Other values return HTTP 400 |
| `negative_prompt` | Not supported | Not supported | Rejected |

#### Sizes

The effective working range of MiniMax-H3-Image is about 4 MP to 20 MP, and it generally gives
the best results at around 16 MP. Common 16 MP sizes are listed below:

| Aspect ratio | Size (width x height) |
| --- | --- |
| 1:1 | 4096x4096 |
| 4:3 / 3:4 | 4736x3552 / 3552x4736 |
| 3:2 / 2:3 | 5024x3360 / 3360x5024 |
| 16:9 / 9:16 | 5472x3072 / 3072x5472 |
| 21:9 | 6272x2688 |

### Full Workflow

MiniMax-H3-Image expects a detailed English description of the whole image. The MiniMax API
turns a short prompt in any language into such a description with **H3-Image-Context-IR** before
generation. We also provide
[MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE), which can be
deployed locally.

This section reproduces the quality of images generated directly by the MiniMax API with a
locally deployed SGLang server. The scripts are in
[`examples/`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/examples) and use only
the Python standard library; run them in that folder:

```bash
hf download MiniMaxAI/MiniMax-H3-Image --include "examples/*" --local-dir .
cd examples
```

Before running examples, set your MiniMax API endpoint and key to use the H3-Image-Context-IR API
and the reference scripts. The scripts read them from the environment:

```bash
# MiniMax API endpoint (choose one)
# CN
export MINIMAX_API_BASE="https://api.minimax.cn"
# Global
# export MINIMAX_API_BASE="https://api.minimax.io"

# API key obtained from the MiniMax platform
export MINIMAX_API_KEY="<api-key>"
```

API docs:
- H3-Image-Context-IR: use /v1/h3_image_context_ir [EN-docs](https://platform.minimax.io/docs/api-reference/image-generation-h3-context-ir), [CN-docs](https://platform.minimax.cn/docs/api-reference/image-generation-h3-context-ir)
- Text-to-image: use /v1/images/generations [EN-docs](https://platform.minimax.io/docs/api-reference/image-generation-h3-generations), [CN-docs](https://platform.minimax.cn/docs/api-reference/image-generation-h3-generations)
- Image editing: use /v1/images/edits [EN-docs](https://platform.minimax.io/docs/api-reference/image-generation-h3-edits), [CN-docs](https://platform.minimax.cn/docs/api-reference/image-generation-h3-edits)

#### case-T2I

[`instruction.t2i.txt`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/instruction.t2i.txt).

<table>
<thead>
<tr><th colspan="2">Stage</th><th>Request</th><th>Result</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Prompt Enhancement</td><td>H3-Image-Context-IR (MiniMax API)</td><td><code>python3 context_ir.py t2i</code> (<a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/context_ir.py">script</a>)</td><td><code>revised_prompt.txt</code>, <code>revised_size.txt</code>; our run: <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/revised_prompt.t2i.txt"><code>revised_prompt.t2i.txt</code></a>, <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/revised_size.t2i.txt"><code>revised_size.t2i.txt</code></a> (4736x3552)</td></tr>
<tr><td>local alternative: <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE">MiniMax-H3-Image-PE</a></td><td><code>scripts/expand_prompt.py</code> in the PE repository</td><td>Save <code>expanded_caption</code> as <code>revised_prompt.txt</code> and the size for <code>aspect_ratio</code> as <code>revised_size.txt</code></td></tr>
<tr><td colspan="2">MiniMax-H3-Image generation: local SGLang</td><td><code>python3 t2i.py --prompt revised_prompt.txt --size $(cat revised_size.txt)</code> (<a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/t2i.py">script</a>)</td><td><code>t2i.png</code></td></tr>
<tr><td colspan="2">E2E API Reference: text-to-image by directly calling the MiniMax API</td><td><code>python3 api_generate.py t2i</code> (<a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/api_generate.py">script</a>)</td><td><code>api_t2i.png</code></td></tr>
</tbody>
</table>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/workflow_t2i.jpg" alt="Left: t2i.png from local SGLang with the enhanced prompt. Right: api_t2i.png from the MiniMax API. Both show an overhead farm-to-table feast with bread, vegetables and cheeses on a weathered wooden table" width="100%"/>
</p>
<p align="center"><em>t2i.png from local SGLang (left) and api_t2i.png from the MiniMax API (right), 4736x3552 (4:3 picked by H3-Image-Context-IR)</em></p>

#### case-Edit

[`instruction.edit.txt`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/instruction.edit.txt)
with the reference image [`reference1.png`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/reference1.png)
as `<Picture 1>`.

| Stage | Request | Result |
| --- | --- | --- |
| Prompt Enhancement: H3-Image-Context-IR (MiniMax API) | `python3 context_ir.py edit` ([script](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/context_ir.py)) | `revised_prompt.txt`, `revised_size.txt`; our run: [`revised_prompt.edit.txt`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/revised_prompt.edit.txt), [`revised_size.edit.txt`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/revised_size.edit.txt) (4096x4096) |
| MiniMax-H3-Image generation: local SGLang | `python3 edit.py --prompt revised_prompt.txt --size $(cat revised_size.txt)` ([script](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/edit.py)) | `edit.png` |
| E2E API Reference: image editing by directly calling the MiniMax API | `python3 api_generate.py edit` ([script](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/examples/api_generate.py)) | `api_edit.png` |

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/workflow_edit.jpg" alt="Left: reference1.png, the reference image of two podcast hosts with the man wearing headphones. Middle: edit.png from local SGLang with the enhanced prompt. Right: api_edit.png from the MiniMax API. In both results the headphones are gone and the hosts bump fists above the MONEY TALKS sign" width="100%"/>
</p>
<p align="center"><em>Reference image reference1.png (left), edit.png from local SGLang (middle) and api_edit.png from the MiniMax API (right), 4096x4096 (1:1 picked by H3-Image-Context-IR)</em></p>

## Showcase

Text-to-image and image editing results, shown downscaled.

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/photo.jpg" alt="Three photorealistic photographs: a woman hanging white sheets on a rooftop at dusk under a full moon, a close-up profile of a curly-haired young man holding a cup, and flamingos wading in shallow water" width="100%"/>
</p>
<p align="center"><em>Photorealistic photography</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/surreal.jpg" alt="Three surreal editorial photographs: a barber trimming a cloud seated in a barber chair, a person slicing the moon on a lawn at night, and a model stepping out of a giant pear" width="100%"/>
</p>
<p align="center"><em>Surreal editorial photography</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/scene.jpg" alt="Four environment concepts: ruins of an ancient open-air theater in a forest titled ENVIRONMENT CONCEPT: THE ANCIENT THEATER, a dragon rider on a red rock cliff, an abandoned railway station with an ARRIVALS sign, and shipwrecks frozen in an icy sea" width="100%"/>
</p>
<p align="center"><em>Environment concepts</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/char.jpg" alt="Characters, creatures and props: a soldier in a worn helmet and armor, a red-finned dragon-like creature in a forest, a weathered astronaut helmet on Mars, and a three-view character sheet of a figure in winter clothing" width="100%"/>
</p>
<p align="center"><em>Characters, creatures and props</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/art.jpg" alt="Four paintings and illustrations: mandarin ducks in a lotus pond in an oil painting style, a Jiangnan garden in light watercolor, tarot card XVIII THE MOON with a rabbit, and a Chinese gongbi painting of a pelican looking into a mirror with four calligraphy characters" width="100%"/>
</p>
<p align="center"><em>Painting and illustration styles</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/text.jpg" alt="Two logic puzzle boards: a Chinese candidate-selection puzzle with five conditions and the derived answer, and an English board titled CONFIDENTIAL FILE MISSING with six suspects, true/false stamps and a verification table" width="100%"/>
</p>
<p align="center"><em>Text rendering with reasoning: logic puzzles in Chinese and English, solved correctly on the board</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/edit_render.jpg" alt="Left: untextured grey 3D render of a lunar base with habitat modules, a rover and an astronaut. Right: the same scene as a photorealistic film still on the lunar surface" width="100%"/>
</p>
<p align="center"><em>Image editing, one reference: an untextured 3D render (left) turned into a film still of a lunar research base (right)</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/edit_map.jpg" alt="Left: illustrated game map with a river, a waterfall, bridges, a castle, a village, a mine and a harbor, and a red arrow on the southern lake. Right: first-person 3D game view from the lake towards the waterfall, the suspension bridge, the mine and the distant castle, with the map as a minimap in the top-left corner" width="100%"/>
</p>
<p align="center"><em>Image editing, one reference: a game map (left) turned into a first-person 3D game view from the red arrow's position, with the map as a minimap (right)</em></p>

<p align="center">
<img src="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/assets/showcase/edit_multi.jpg" alt="Top: ten reference images, an empty sunlit room and nine pieces of furniture. Bottom: the empty room furnished with all nine pieces as a living room" width="100%"/>
</p>
<p align="center"><em>Image editing, ten references: an empty room and nine pieces of furniture (top) composed into one living room (bottom)</em></p>

## License

- MiniMax-H3-Image is released under the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
- The T2I PE Model ([MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)) is part of MiniMax-H3-Image and is accordingly subject to [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
- The separate written authorization described in Section IV of the license is requested through [Application form](https://platform.minimax.io/console/license)

## Contact Us

Contact us at [model@minimax.io](mailto:model@minimax.io).
