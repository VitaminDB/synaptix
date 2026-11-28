# Source: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet.md

Fetched: 2026-11-28 (raw markdown)

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.comfy.org/llms.txt
> Use this file to discover all available pages before exploring further.

# MiniMax H3 Fun ControlNet Union: Pose, Depth, and Canny Control

> Guide MiniMax H3 video motion in ComfyUI with Canny, Depth, HED, MLSD, or Pose control videos, plus mask-based inpainting with the Fun ControlNet Union patch.

Control the motion of a MiniMax H3 video with a control video instead of text alone. The Fun ControlNet Union patch ([alibaba-pai/MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union), packaged in [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)) is a single checkpoint that conditions H3 on Canny, Depth, HED, MLSD, or Pose control videos, and also runs video inpainting with a mask. It works with both the `fl2va` and `ref2va` transformer files.

![MiniMax H3 Fun ControlNet Union workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_fun_controlnet_union-1.webp)

> [card] [Run on Comfy Cloud](https://cloud.comfy.org/?template=video_minimax_h3_fun_controlnet_union\&utm_source=docs\&utm_medium=referral\&utm_campaign=minimax-h3)
>
> Open in Comfy Cloud

> [card] [Download Workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json)
>
> Download JSON or search "MiniMax H3 Fun ControlNet Union" in Template Library

**Input materials**

Upload this file to the matching `Load Video` node:

> [card] [dancer\_field\_pose.mp4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/dancer_field_pose.mp4)
>
> Pose control video used by the example workflow, or use your own.

This template requires ComfyUI 0.35.0 or later. The example uses a pose control video: a built-in SDPose subgraph extracts the pose skeleton from the input video, so the output reproduces the dancer's motion. You can also connect any preprocessed control video directly to the `Apply MiniMax H3 Fun ControlNet` node's `control_video` input.

## Model downloads

> [card] [Diffusion Model: minimax\_h3\_ref2va\_pruned\_int8\_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
>
> Place in `ComfyUI/models/diffusion_models/`

> [card] [ControlNet Patch: minimax\_h3\_fun\_controlnet\_union\_pruned\_int8\_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors)
>
> Place in `ComfyUI/models/model_patches/`

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

> [card] [Checkpoint: sdpose\_wholebody\_fp16](https://huggingface.co/Comfy-Org/SDPose/blob/main/checkpoints/sdpose_wholebody_fp16.safetensors)
>
> Pose extractor used by the built-in SDPose subgraph. Place in `ComfyUI/models/checkpoints/`

> [card] [Detector: rt\_detr\_v4-x-hgnet\_fp16](https://huggingface.co/Comfy-Org/SDPose/blob/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors)
>
> Person detector used by the built-in SDPose subgraph. Place in `ComfyUI/models/diffusion_models/`

## Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   │   └── rt_detr_v4-x-hgnet_fp16.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   ├── 📂 loras/
│   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   ├── 📂 model_patches/
│   │   └── minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors
│   └── 📂 checkpoints/
│       └── sdpose_wholebody_fp16.safetensors
```

## Prompting tips

1. **Video length**: The target video runs 5 to 15 seconds on the 17n+5 frame grid at 24fps (124 frames = 5 seconds). A control video longer than the target is trimmed to its first frames; a shorter one holds its last frame. To match the control video's length exactly, link the example's computed `batch_size` output to the length input of the `MiniMax H3 Reference to Video` node
2. **Guidance and strength**: Keep `guidance_scale` at 1.0. Raise the patch `strength` above 1 only if the output drifts away from the control
3. **Video inpainting**: Connect a mask to the `mask` input: regions marked `1` are regenerated under control of the optional `source_video` input, while the rest of the frame stays fixed
4. **Turbo mode (optional)**: Enable the **Lightning LoRA** checkbox to use the 4-step turbo LoRA for much faster generation, with slightly lower audio and motion quality
5. **Preparing control videos**: ComfyUI ships with built-in preprocessing nodes for each control type. See the [Detect Edges (Canny) node](https://docs.comfy.org/built-in-nodes/Canny.md) for edges and the [Depth Anything 3 tutorial](https://docs.comfy.org/tutorials/utility/depth-anything-3.md) for depth maps


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.