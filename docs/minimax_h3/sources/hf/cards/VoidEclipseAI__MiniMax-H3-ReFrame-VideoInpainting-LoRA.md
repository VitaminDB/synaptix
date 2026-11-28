# VoidEclipseAI/MiniMax-H3-ReFrame-VideoInpainting-LoRA

Источник: https://huggingface.co/VoidEclipseAI/MiniMax-H3-ReFrame-VideoInpainting-LoRA · скачано 2026-11-28
Загрузок: 157 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-11-26T13:21:14.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: video-to-video
tags:
  - lora
  - video-editing
  - video-inpainting
  - minimax-h3
  - comfyui
  - reframe
  - reframevideo
---

# ReFrameVideo H3 LoRA

<video controls src="https://huggingface.co/VoidEclipseAI/ReFrameVideo-LoRA/resolve/main/assets/reframevideo_promo.mp4"></video>

A LoRA for **MiniMax H3 (Ref2VA)** that turns a roughly edited video into a finished clip.

**[ReFrameVideo](https://github.com/VoidEclipseAI/ReFrameVideo)** is an open-source personal project made up of this LoRA and a companion video editor. In the editor, the elements of a video become layers that you move, scale or delete as in Photoshop, or you change the aspect ratio, with reference images to control the final shot; this LoRA then lets H3 regenerate the whole clip and fill in the gaps the edit left.

> Powered by MiniMax H3. This LoRA is a modification of MiniMax H3 and is distributed under the MiniMax H3 Community License Agreement (see [License](#license)).

## What it does

ReFrameVideo prepares two inputs from your edit:

- a **rough cut**: every subject is placed in its new position, and the areas left empty are painted flat **grey (RGB 128)**;
- a **per-pixel redraw mask**: the empty areas plus a thin blending band.

Without this LoRA, H3 tends to treat the grey areas as part of the picture and keeps or copies them. **The LoRA teaches H3 that grey 128 means "empty, fill me"**: it fills those areas with background or scene content that fits the surrounding shot, keeps the moved subjects, and blends their edges into their new surroundings.

A first-frame reference image is optional. The LoRA works without one, but the background can sometimes drift during the clip; a clean first frame keeps it much more stable.

<video controls src="https://huggingface.co/VoidEclipseAI/ReFrameVideo-LoRA/resolve/main/assets/widen_compare.mp4"></video>

<video controls src="https://huggingface.co/VoidEclipseAI/ReFrameVideo-LoRA/resolve/main/assets/group4_compare.mp4"></video>

<video controls src="https://huggingface.co/VoidEclipseAI/ReFrameVideo-LoRA/resolve/main/assets/reframe_compare.mp4"></video>

## Files

| File | Description |
|---|---|
| `ReFrameVideo-H3-LoRA-v1.safetensors` | The LoRA (v1). Put it in `ComfyUI/models/loras/`. |
| `LICENSE` | MiniMax H3 Community License Agreement (applies to this LoRA). |
| `NOTICE` | Required notice for MiniMax H3 derivatives. |

A mirror is on [ModelScope](https://modelscope.cn/models/VoidEclipseAI/ReFrameVideo-LoRA) (faster downloads in mainland China).

<video controls src="https://huggingface.co/VoidEclipseAI/ReFrameVideo-LoRA/resolve/main/assets/why_the_lora.mp4"></video>

## Recommended setup

| Setting | Value |
|---|---|
| Base model | `minimax_h3_ref2va_int8_convrot.safetensors` (MiniMax H3 Ref2VA) |
| ReFrameVideo LoRA | `ReFrameVideo-H3-LoRA-v1.safetensors` @ **1.0** |
| Speed-up LoRA | `pika_h3_flex_ref2va_v1_bf16.safetensors` @ **1.0** ([Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex); a ComfyUI copy is in [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)) |
| Steps | **6** (with the speed-up LoRA) |
| Resolution | 768p (short side 768, e.g. 1344×768 for 16:9) |

On an RTX 4090 a ~5 s clip takes roughly 4–5 minutes with these settings.

## How to use

### With ReFrameVideo (recommended)

<video controls src="https://huggingface.co/VoidEclipseAI/ReFrameVideo-LoRA/resolve/main/assets/reframevideo_workflow.mp4"></video>

ReFrameVideo builds every input this LoRA expects: the rough cut, the pixel mask, the reference images and the prompt. It then queues the workflow in ComfyUI for you.

1. Install ReFrameVideo and its ComfyUI node pack ([installation guide](https://github.com/VoidEclipseAI/ReFrameVideo/blob/main/docs/安装.md)).
2. Download this LoRA into `ComfyUI/models/loras/`.
3. In ReFrameVideo's **Generate** panel, add the LoRA (and the speed-up LoRA) and set the steps.
4. Edit your shot and click **Generate video**. The [README](https://github.com/VoidEclipseAI/ReFrameVideo#readme) walks through a complete example with screenshots.

### In your own ComfyUI workflow

The LoRA was trained on inputs prepared the same way ReFrameVideo prepares them, so results are best when you match that setup:

- **Rough cut** as the noised base video: subjects in their new positions, empty areas exactly RGB 128.
- **Redraw mask** converted to H3's latent grid with the `ReframeH3InpaintMask` node from ReFrameVideo's `ComfyUI-Reframe` node pack (store the mask losslessly, e.g. PNG-encoded `.mov`, so 255 stays 255).
- **`<Video 1>`**: the same rough cut downscaled (768 px wide), as a reference video.
- **`<Picture N>`** (optional): a clean first frame, a background plate or character references.
- **Prompt** in H3's Ref2VA six-section format.

The easiest way to get a working graph is to use ReFrameVideo's **Export** feature, which saves the exact `workflow_api.json`, rough cut, mask, references and prompt as a zip you can load into ComfyUI.

## Limitations

- **No real 3D understanding.** With strong camera motion the gaps get filled, but depth, occlusion and contact can be wrong (for example, an object that should pass in front of a moved person stays behind them). Static or gently moving shots work best.
- **Trained on live-action footage only** (no AI-generated content), so it may generalize less well to stylized material such as animation, cartoons or 3D renders.
- Large holes shaped like a person can sometimes be filled with a new person; hands and fine details may degrade; the background can drift without a first-frame reference.
- The final video is the model's full output, so areas outside the mask can differ slightly from the original.
- This is v1 of a personal project and still experimental.

## License

This LoRA is a **Model Derivative of MiniMax H3** and is distributed under the **[MiniMax H3 Community License Agreement](LICENSE)**, the same terms as the base model. By downloading or using it you agree to those terms, including the Acceptable Use Policy (Exhibit A). In particular:

- **Territory:** the license currently applies only outside the **Excluded Territories (the European Union, the United Kingdom, the Republic of Korea and the United States of America)**. Users in those regions are not licensed to use MiniMax H3 or its derivatives under this agreement; MiniMax invites them to apply for a license at https://platform.minimax.io/h3-license.
- **Commercial use:** allowed under the agreement. Products whose yearly revenue exceeds USD 20 million need separate written authorization from MiniMax, and commercial products that use MiniMax H3 Works must prominently display "MiniMax H3" in their user interface.
- **Outputs** may not be used to improve other AI models (Section V.3).

See [`NOTICE`](NOTICE):

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

The ReFrameVideo software (editor, backend and ComfyUI nodes) is a separate work, licensed under AGPL-3.0 with a commercial license available; see the [GitHub repository](https://github.com/VoidEclipseAI/ReFrameVideo).

This summary is for convenience only; the [LICENSE](LICENSE) text is authoritative.

## Acknowledgements

- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax): the base model of this LoRA.
- [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) (Pika): the speed-up LoRA behind 6-step generation.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and Comfy-Org's [MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) packaging: where this LoRA runs.
- The ReFrameVideo editor also uses [SAM 3](https://github.com/facebookresearch/sam3), [Qwen Image](https://github.com/QwenLM/Qwen-Image), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and [MatAnyone 2](https://github.com/pq-yang/MatAnyone2); see the [GitHub README](https://github.com/VoidEclipseAI/ReFrameVideo/blob/main/README_en.md#acknowledgements) for the full list.

## About

ReFrameVideo is a one-person project: I did the concept, the LoRA training, the interaction design and the testing; the editor's code was written with the help of an AI coding assistant (Claude Code). It is still rough around the edges. Feedback, failure cases and contributions are very welcome: please open an issue or a discussion on [GitHub](https://github.com/VoidEclipseAI/ReFrameVideo).
