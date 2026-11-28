# Source: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe.md

Fetched: 2026-11-28 (raw markdown)

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.comfy.org/llms.txt
> Use this file to discover all available pages before exploring further.

# MiniMax H3 Multiframe: Video Continuations and Multi-Shot Scenes

> Build MiniMax H3 video continuations and multi-shot scenes in ComfyUI by pinning reference frames to set points on the timeline with Add Guide nodes.

Generate video continuations and multi-shot scenes by anchoring reference frames at specific points along the output timeline. The workflow combines the Reference to Video node with chained Add Guide nodes: each Add Guide pins a still image (plus optional audio) to a frame index, so the video passes through that exact composition at that moment.

![MiniMax H3 Multiframe Reference workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_multiframe_reference-1.webp)

> [card] [Run on Comfy Cloud](https://cloud.comfy.org/?template=video_minimax_h3_multiframe_reference\&utm_source=docs\&utm_medium=referral\&utm_campaign=minimax-h3)
>
> Open in Comfy Cloud

> [card] [Download Workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_multiframe_reference.json)
>
> Download JSON or search "MiniMax H3: Multiframe Reference" in Template Library

> [card] [Frame Reference 1: h3\_frame\_ref\_1.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_1.png)
>
> Identity reference and first frame of Shot 1, or use your own images.

> [card] [Frame Reference 2: h3\_frame\_ref\_2.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_2.png)
>
> Keyframe anchored at 1.5s (Shot 2), or use your own images.

> [card] [Frame Reference 3: h3\_frame\_ref\_3.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_3.png)
>
> Keyframe anchored at 3.0s (Shot 3), or use your own images.

> [card] [Frame Reference 4: h3\_frame\_ref\_4.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_4.png)
>
> First frame anchored at 5.0s (Shot 4), or use your own images.

## Model downloads

> [card] [Diffusion Model: minimax\_h3\_ref2va\_pruned\_int8\_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
>
> Place in `ComfyUI/models/diffusion_models/`

> [card] [Text Encoder: qwen3vl\_32b\_minimax\_h3\_nvfp4\_awq](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)
>
> Place in `ComfyUI/models/text_encoders/`

> [card] [VAE: minimax\_h3\_video\_vae\_int8\_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors)
>
> Place in `ComfyUI/models/vae/`

> [card] [VAE: minimax\_h3\_audio\_vae\_fp32](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors)
>
> Place in `ComfyUI/models/vae/`

> [card] [LoRA: minimax\_h3\_ref2v\_turbo\_4step\_v0.1\_comfyui\_bf16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors)
>
> Powers the optional Lightning LoRA turbo mode. Place in `ComfyUI/models/loras/`

## Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   └── 📂 loras/
│       └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

## Prompting tips

1. **Reference by tag**: `<Picture 1>` is the image connected to the first `ref_images` slot, `<Picture 2>` the second, and so on. In this workflow, `<Picture 1>` is both the identity lock and the first frame of the video
2. **Timeline anchors vs references**: Pictures connected only to Add Guide nodes pin compositions at specific frames but are not visible to the text encoder. Plug them into extra `ref_images` slots too if the prompt should see them as `<Picture N>`
3. **Frame index**: Each Add Guide takes `frame_idx` in frames at 24fps; the template computes it from seconds with `round(seconds * 24)`. Negative values count from the end of the video. The index plus the guide length must stay inside the generated duration
4. **Add more keyframes**: Copy one guide block (Load Image + Frame Index + Math Expression + Add Guide for MiniMax H3) and keep the `positive` outputs chained in series into the Basic Guider
5. **Turbo mode (optional)**: Enable the **Lightning LoRA** checkbox to use the 4-step turbo LoRA for much faster generation, with slightly lower audio and motion quality

## Prompt writing guide

The template prompt follows the [full-reference mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md): it defines each subject, maps each `<Picture N>` to a shot's first frame or keyframe, and cuts between shots at explicit timestamps. Use the same structure when writing your own multi-shot prompts. The guide is summarized in the [prompt guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide.md).


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.