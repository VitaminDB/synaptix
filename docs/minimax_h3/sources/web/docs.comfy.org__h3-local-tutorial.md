# Source: https://docs.comfy.org/tutorials/video/minimax/minimax-h3.md

Fetched: 2026-11-28 (raw markdown)

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.comfy.org/llms.txt
> Use this file to discover all available pages before exploring further.

# ComfyUI MiniMax H3 Video Generation Guide

> Run open-weights MiniMax H3 in ComfyUI: text-to-video, image-to-video, and reference-to-video Workflows with native stereo audio.

[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's open-weights omni-modal model. It understands text, images, video, and audio in one context and generates video with **native stereo audio** (voice, SFX, and music in a single pass). Open weights target about 1 megapixel (768px short edge) at 24fps for up to \~15 seconds. 2K is a hosted-service output and needs a separate upscale pass in ComfyUI.

> [!NOTE]
>
> H3's open weights let you run the model locally. Commercial use of locally generated outputs requires a [MiniMax commercial license](https://comfy.org/minimax/license), available through Comfy, the only official reseller. Generations on Comfy Cloud already include commercial rights.

## Key features

* **Native stereo audio**: Dialogue, SFX, and music in one MP4
* **Multimodal context**: Combine text, image, video, and audio references
* **Reference-driven generation**: Lock identity, style, motion, camera, or voice from references
* **Instruction following**: Describe reference-to-shot relationships in natural language
* **Accurate text rendering**: Clean spelled-out text and brand elements
* **Open weights**: Full local control in ComfyUI

## Choosing a template Workflow

Pick a Workflow from **Template Library** > **Video** and download models from the pop-up. Weights: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

Five `video_minimax_h3_*` template Workflows are available (examples, not an exhaustive list): `t2v`, `i2v`, `r2v`, `multiframe_reference`, and `fun_controlnet_union`.

> [card] [Text to Video (T2V)](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native.md#minimax-h3-text-to-video-t2v)
>
> Generate videos from text prompts with native stereo audio. ComfyUI **0.30.0+**.

> [card] [Image to Video (I2V)](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native.md#minimax-h3-image-to-video-i2v)
>
> Generate videos from an input image, with optional first/last-frame control. ComfyUI **0.30.0+**.

> [card] [Reference to Video (R2V)](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native.md#minimax-h3-reference-to-video-r2v)
>
> Lock in a character, style, motion, camera move, or voice from reference images, videos, and audio. ComfyUI **0.30.0+**.

> [card] [Multiframe Reference](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe.md)
>
> Anchor reference frames at specific points along the output timeline with chained Add Guide Nodes. ComfyUI **0.34.0+**.

> [card] [Fun ControlNet Union](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet.md)
>
> Drive H3 with a Canny, Depth, HED, MLSD, or Pose control video, or run video inpainting with a mask. ComfyUI **0.35.0+** (Union 2.0: **0.38.0+**). Sparse attention and `Model Attention Backend` also need **0.35.0+**.

> [card] [Prompt guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide.md)
>
> Official prompt guides, tips, and embeddings for H3

> [card] [FastH3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fastvideo.md)
>
> FastVideo sparse-attention checkpoints and speed-focused Workflows. ComfyUI **0.36.0+**.

Two templates cover several modes:

* `MiniMax H3 Image to Video` covers T2V and first/last-frame I2V (`t2va` / `fl2va`).
* `MiniMax H3 Reference to Video` covers R2V (`ref2va`).

## Commercial licensing

Commercial use of locally generated outputs requires a license from Comfy. Comfy Cloud generations already include commercial rights.

> [card] [MiniMax commercial license](https://comfy.org/minimax/license)
>
> When you need a license, community vs Professional and Enterprise tiers, and how to request a quote.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.