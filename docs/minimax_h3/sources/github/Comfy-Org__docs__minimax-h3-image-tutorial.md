# Comfy-Org__docs__minimax-h3-image-tutorial

Source: https://github.com/Comfy-Org/docs/blob/main/tutorials/image/minimax/minimax-h3-image.mdx
Fetched: 2026-11-28



---

## FILE: tutorials/image/minimax/minimax-h3-image.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/image/minimax/minimax-h3-image.mdx)

---
title: "MiniMax-H3-Image: 16 MP text to image and image editing in ComfyUI"
description: "Run MiniMax-H3-Image in ComfyUI: one open-weights checkpoint for text to image and prompt-driven image editing at about 16 MP, distilled to 6 sampler steps."
sidebarTitle: "MiniMax H3 Image"
---

import UpdateReminder from '/snippets/tutorials/update-reminder.mdx'

**MiniMax-H3-Image** is MiniMax's open-weights image model, and one checkpoint covers both text to image and image editing. The diffusion transformer is 33.1B parameters in BF16, distilled to a 6-step sampler with no CFG. Output is native at about 16 MP, so a 4096 x 4096 image is a single pass rather than an upscale chain.

**Related links**

- [MiniMaxAI/MiniMax-H3-Image (Hugging Face)](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image): official model card and weights
- [MiniMax-H3-Image (ModelScope)](https://modelscope.cn/models/minimax/MiniMax-H3-Image): official mirror for mainland China
- [MiniMax-AI/MiniMax-H3-Image (GitHub)](https://github.com/MiniMax-AI/MiniMax-H3-Image): official examples and skills
- [Comfy-Org/MiniMax-H3-Image (Hugging Face)](https://huggingface.co/Comfy-Org/MiniMax-H3-Image): diffusion model and image VAE
- [Comfy-Org/MiniMax-H3 (Hugging Face)](https://huggingface.co/Comfy-Org/MiniMax-H3): Qwen3-VL-32B text encoder used by this model
- [MiniMaxAI/MiniMax-H3-Image-PE (Hugging Face)](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE): official prompt expander

## Key features

- **Text to image and editing in one checkpoint**: no separate edit model to load
- **6-step sampling**: guidance is fixed at 1, so there is no CFG pass and no negative prompt
- **About 16 MP natively**: 4096 x 4096 up to 6272 x 2688 without an upscale pass
- **Multiple references**: up to 16 input images, addressed as `<Picture 1>` upwards in the prompt
- **Optional prompt expander**: turns a short request into the detailed caption the model reads best
- **Open weights**: full local control in ComfyUI

<UpdateReminder />

## Templates

Three template Workflows ship with the model. All need ComfyUI **0.45.0+**. The text-to-image and image edit templates need template pack **0.11.104+**; character swap needs **0.11.115+**.

### MiniMax-H3-Image Text to Image {#image_minimax_h3_image_t2i}

Generate an image from a text prompt. The template opens at 16 MP (4096 x 4096 at 1:1), the resolution the model is built for. Lower `megapixels` in the resolution selector to about 4 MP on a small GPU or for faster runs.

<img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_t2i-1.webp" alt="MiniMax-H3-Image text to image example output" />

<CardGroup cols={1}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json">
    Download JSON or search "MiniMax-H3-Image: Text to Image" in Template Library
  </Card>
</CardGroup>

### MiniMax-H3-Image Image Edit {#image_minimax_h3_image_edit}

Edit a picture by describing the change. The template loads the source as `<Picture 1>`, builds a reference-aware caption from your instruction, and finishes with a before/after comparison node.

<img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_edit-1.webp" alt="MiniMax-H3-Image image edit example output" />

<CardGroup cols={1}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json">
    Download JSON or search "MiniMax-H3-Image: Image Edit" in Template Library
  </Card>
</CardGroup>

**Input materials**

The template ships with one source picture, loaded as `<Picture 1>` in its `LoadImage` node.

<CardGroup cols={1}>
  <Card title="vintage_yellow_taxi.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/vintage_yellow_taxi.png">
    Source image the template edits
  </Card>
</CardGroup>

### MiniMax-H3-Image Character Swap {#image_minimax_h3_image_character_swap}

Replace the person in a photo with a character from a reference image. The template reads the photo as `<Picture 1>` and the character as `<Picture 2>`, keeps the pose, framing, and lighting of the shot, and carries over the reference character's face, costume, and materials. It runs on the same checkpoint plus the character-swap adapter at strength 1.0.

<img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_character_swap-1.webp" alt="MiniMax-H3-Image character swap, source photo and swapped result" />

<CardGroup cols={1}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_character_swap.json">
    Download JSON or search "MiniMax-H3-Image: Character Swap" in Template Library
  </Card>
</CardGroup>

Two things differ from the other templates. The source photo is scaled to 1 MP, 864 x 1152 with the example images, and that becomes the output size. There is also no prompt expander: the adapter was trained on short editing instructions, so write the instruction yourself and name the references, for example "Swap the person in `<Picture 1>` with the character in `<Picture 2>`, keep the scene, framing, and the target's pose."

**Input materials**

The template ships with a source photo and a reference character.

<CardGroup cols={2}>
  <Card title="h3_char_swap_source.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_char_swap_source.png">
    Photo to edit, read as the first picture
  </Card>
  <Card title="h3_char_swap_reference.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_char_swap_reference.png">
    Character to carry over, read as the second picture
  </Card>
</CardGroup>

## Model downloads

The text-to-image and image edit templates load these four files, all from Comfy-Org mirror repositories. The character-swap template skips the prompt expander and needs one more file, an adapter published by its author.

<CardGroup cols={2}>
  <Card title="Diffusion model: minimax_h3_image_pruned_int8_convrot.safetensors" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3-Image/blob/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors">
    Pruned INT8 DiT, 21 GB. Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="Text encoder: qwen3vl_32b_minimax_h3_int8_convrot.safetensors" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors">
    Qwen3-VL-32B, INT8, 25.28 GB. Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="Prompt expander: qwen3.5_9b_w4a8_gptq.safetensors" icon="download" href="https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors">
    Qwen3.5 9B W4A8, 7.34 GB. Place in <code>ComfyUI/models/text_encoders/</code>. Text-to-image and image edit only
  </Card>
  <Card title="VAE: minimax_h3_image_vae_fp16.safetensors" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3-Image/blob/main/vae/minimax_h3_image_vae_fp16.safetensors">
    Image VAE, FP16, 1.17 GB. Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="Character swap adapter: h3_image_character_swap_v1.safetensors" icon="download" href="https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA/resolve/main/h3_image_character_swap_v1.safetensors">
    Rank-16 LoRA, 155 MB, from the adapter's original repo. Place in <code>ComfyUI/models/loras/</code>
  </Card>
</CardGroup>

**Model Storage Location**

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_image_pruned_int8_convrot.safetensors
│   ├── 📂 loras/
│   │   └── h3_image_character_swap_v1.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   └── qwen3.5_9b_w4a8_gptq.safetensors
│   └── 📂 vae/
│       └── minimax_h3_image_vae_fp16.safetensors
```

The adapter is [akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA), trained on 140 paired edits and evaluated on the INT8 checkpoint above. Adapter use follows the MiniMax H3 Image Community License Agreement.

## Sampling and canvas size

Every template ships the distilled schedule:

| Setting | Value |
| --- | --- |
| Sampler | euler |
| Scheduler | simple |
| Steps | 6 |
| Guidance (CFG) | 1 |
| Denoise | 1 |

With guidance at 1 there is no classifier-free guidance pass, so the templates carry no negative prompt. Write everything you want, including things to avoid, into the positive prompt.

Native output is about 16 MP. Every edge is generated on a multiple of 32 px, and any width and height are accepted. These sizes cover the common ratios:

| Aspect ratio | Size (width x height) |
| --- | --- |
| 1:1 | 4096 x 4096 |
| 4:3 | 4736 x 3552 |
| 3:4 | 3552 x 4736 |
| 3:2 | 5024 x 3360 |
| 2:3 | 3360 x 5024 |
| 16:9 | 5472 x 3072 |
| 9:16 | 3072 x 5472 |
| 21:9 | 6272 x 2688 |

Smaller canvases work too and cost less time. In the text-to-image template the resolution selector sets the aspect ratio and the target megapixels separately, so any target from about 4 MP up to the selector's 16 MP ceiling is one setting away.

## Reference images in image editing

The image editing template sends the source image to the model as `<Picture 1>`. Additional images are labelled in order, `<Picture 2>`, `<Picture 3>`, up to 16 references, and the prompt addresses each one by its label. The first picture is the one being edited; the rest supply identity, props, materials, or composition. A short instruction is enough, for example replacing the vehicle in `<Picture 1>` with the one in `<Picture 2>`.

To add a reference, place another `Load Image` node in the workflow and connect it to both the prompt generation step and the conditioning node. Reference images need no preparation: the template rounds the source to a multiple of 32 px, and ComfyUI scales the internal copies itself, a 1024 px short edge for the vision side and a short edge clamped between 1024 and 2048 px for the VAE side, both on the 32 px grid.

If an edit runs out of memory, enable the `Resize Image/Mask` node in the template and lower its megapixels before sampling.

## Prompt expander

The text-to-image and image editing templates hold a text generation step in front of the sampler. In the text-to-image template it rewrites the prompt into a detailed English caption. In the image editing template it turns a short instruction into a reference-aware description of the finished image.

The official expander for MiniMax-H3-Image is a Qwen3.8-27B FP4 model, published as [MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE). The templates use a lighter Qwen3.5 9B W4A8 build instead, so the expander fits beside the 21 GB diffusion model on one GPU.

The text-to-image and image editing templates ship with the expander switched on, so your prompt is rewritten before sampling. Switch it off to send your prompt to the model exactly as typed.



---

## FILE: tutorials/partner-nodes/minimax/minimax-h3-image.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/partner-nodes/minimax/minimax-h3-image.mdx)

---
title: "MiniMax H3 Image API: Text to Image and Image Edit in ComfyUI"
description: "Generate 2K or 4K images from a prompt, or edit up to 16 reference images with MiniMax H3 Image through the ComfyUI partner API."
sidebarTitle: "MiniMax H3 Image"
---

import ReqHint from "/snippets/tutorials/partner-nodes/req-hint.mdx";
import UpdateReminder from "/snippets/tutorials/update-reminder.mdx";

MiniMax H3 Image is MiniMax's image model, available in ComfyUI through the **MiniMax H3 Image** partner node. Text to image and image editing use the same node: with no reference images connected it generates from your prompt, and with references connected it edits or combines them.

The workflows run on MiniMax's servers through your Comfy API account, so there are no model downloads and no local GPU to manage.

<ReqHint/>
<UpdateReminder/>

## What MiniMax H3 Image is good at

- **2K and 4K output**: pick a size tier and an aspect ratio, or set an exact width and height
- **Image editing**: edit or combine up to 16 reference images, addressed in the prompt as `@Image1` to `@Image16`
- **Prompt rewriting**: MiniMax expands a short prompt into a detailed scene description by default and returns the rewritten prompt with the image
- **Poster-grade detail**: suited to posters, packaging, and editorial images that need legible text and dense detail

## Text to image {#api_minimax_h3_image_t2i}

Generate an image from a prompt at 2K or 4K.

<img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_image_t2i-1.webp" alt="MiniMax H3 Image text to image workflow preview" />

<CardGroup cols={2}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_t2i.json">
    Download JSON or search &quot;MiniMax-H3-Image: Text to Image (API)&quot; in Template Library
  </Card>
</CardGroup>

## Image edit {#api_minimax_h3_image_image_edit}

Connect a source image, describe the change, and only what the prompt asks for is rewritten. The template adds a woman to a finished render and keeps the rest of the frame.

<img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_image_image_edit-1.webp" alt="MiniMax H3 Image edit workflow preview" />

<CardGroup cols={2}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_image_edit.json">
    Download JSON or search &quot;MiniMax-H3-Image: Image Edit (API)&quot; in Template Library
  </Card>
</CardGroup>

**Input materials**

<CardGroup cols={2}>
  <Card title="surreal_rendering_modern.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/surreal_rendering_modern.png">
    `LoadImage` · `surreal_rendering_modern.png`
  </Card>
</CardGroup>

## Options

| Input | Values | Notes |
| :---- | :----- | :---- |
| `model` | `MiniMax-H3-Image`, `MiniMax-H3-Image-Highspeed` | In the template, change the `model` dropdown to switch |
| `size` | `2K`, `4K`, `custom` | 2K and 4K derive width and height from the aspect ratio. `custom` takes a width and height from 512 to 7680 px each, with 655,360 to 20,000,000 px in total and an aspect ratio of at most 3:1 |
| `aspect_ratio` | `auto` plus 15 fixed ratios | `auto` follows the prompt, or the reference image when one is connected |
| `enhance_prompt` | on by default | MiniMax rewrites the prompt into a detailed scene description first. Edits are always enhanced, so turning it off only affects text to image, which then needs a fixed ratio or a custom size |
| `images` | up to 16 | Reference images for editing and combining. Refer to them in the prompt as `@Image1`, `@Image2`, and so on, in connection order |

Prompts can be up to 8,000 characters when reference images are connected. The node also returns the rewritten `revised_prompt` next to the image.

## Get started

1. Update ComfyUI to 0.45.1 or later
2. In the Template Library, search for `MiniMax-H3-Image`
