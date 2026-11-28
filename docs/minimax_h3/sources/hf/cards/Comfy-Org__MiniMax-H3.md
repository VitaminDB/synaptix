# Comfy-Org/MiniMax-H3

Источник: https://huggingface.co/Comfy-Org/MiniMax-H3 · скачано 2026-11-28
Загрузок: 33871716 · лайков: 2896 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: comfyui · обновлено: 2026-11-15T09:54:16.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
tags:
- comfyui
- diffusion-single-file
base_model:
- MiniMaxAI/MiniMax-H3
---

# MiniMax H3

Repackaged model files for ComfyUI.

Original model repository:

- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://huggingface.co/lightx2v/Minimax-h3-Turbo
- https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union
- https://huggingface.co/Kijai/MiniMax-H3-experimental

The Qwen3-VL-32B `nvfp4_awq` quant is converted from: https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4

This `nvfp4` text encoder does not require Blackwell GPU to use.

For diffusion models prefer `int8_convrot` if you are able to use pytorch with cu130.

`fp8_scaled` should only be used if you cannot use `int8_convrot`.

---

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_fl2va_bf16.safetensors
│   │   ├── minimax_h3_fl2va_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_bf16.safetensors
│   │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_w6a8.safetensors
│   │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
│   │   ├── minimax_h3_ref2va_bf16.safetensors
│   │   ├── minimax_h3_ref2va_int8_convrot.safetensors
│   │   ├── minimax_h3_ref2va_pruned_bf16.safetensors
│   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   │   ├── minimax_h3_ref2va_pruned_w6a8.safetensors
│   │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 loras/
│   │   ├── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
│   │   ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│   │   ├── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   │   ├── pika_h3_flex_fl2va_v1_bf16.safetensors
│   │   └── pika_h3_flex_ref2va_v1_bf16.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_audio_vae_fp32.safetensors
│   │   ├── minimax_h3_audio_vae_fp32.safetensors
│   │   └── minimax_h3_video_vae_int8_convrot.safetensors
│   ├── 📂 model_patches/
│   │   ├── minimax_h3_fun_controlnet_union_pruned_bf16.safetensors
│   │   └── minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors
│   ├── 📂 embeddings/
│   │   ├── minimaxh3_art_is_explosion.safetensors
│   │   ├── minimaxh3_blooming_flowers.safetensors
│   │   ├── minimaxh3_bullet_time.safetensors
│   │   ├── minimaxh3_dark_magic.safetensors
│   │   ├── minimaxh3_fire_breath.safetensors
│   │   ├── minimaxh3_four_seasons.safetensors
│   │   ├── minimaxh3_kiss_camera.safetensors
│   │   ├── minimaxh3_spiral_ascent.safetensors
│   │   ├── minimaxh3_storm_magic.safetensors
│   │   └── minimaxh3_truman_show.safetensors
```

## Embeddings

To use the embeddings in a text prompt, invoke them in the CLIPTextEncode node by
combining `embedding:` with the filename of the embedding like this for `minimaxh3_art_is_explosion`:
```
embedding:minimaxh3_art_is_explosion
```
Just exchange the `minimaxh3_art_is_explosion` part with the filename of the embedding you want to use.

## Workflows

<table>
<thead>
<tr><th align="center" valign="middle" style="text-align:center;vertical-align:middle">Workflow</th><th colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle">Thumb</th></tr>
</thead>
<tbody>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json">MiniMax H3: Image to Video</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_i2v-1.webp" width="200" height="200" alt="MiniMax H3: Image to Video"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json">MiniMax H3: Reference to Video</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_r2v-1.webp" width="200" height="200" alt="MiniMax H3: Reference to Video"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json">MiniMax H3: Text to Video</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_t2v-1.webp" width="200" height="200" alt="MiniMax H3: Text to Video"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_ref2va.json">Pika H3 Flex: Reference to Video</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_ref2va.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_pika_h3_flex_ref2va-1.webp" width="200" height="200" alt="Pika H3 Flex: Reference to Video"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_p_h3_image_edit.json">P-H3-Image: Image Edit</a></td><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_p_h3_image_edit.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_p_h3_image_edit-2.webp" width="200" height="200" alt="P-H3-Image: Image Edit"></a><div align="center">Before</div></td><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_p_h3_image_edit.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_p_h3_image_edit-1.webp" width="200" height="200" alt="P-H3-Image: Image Edit"></a><div align="center">After</div></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_t2v.json">Pika H3 Flex: Text to Video</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_t2v.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_pika_h3_flex_t2v-1.webp" width="200" height="200" alt="Pika H3 Flex: Text to Video"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_t2i.json">MiniMax H3: Text to Image</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_t2i.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_t2i-1.webp" width="200" height="200" alt="MiniMax H3: Text to Image"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_p_h3_t2i.json">P-H3-Image: Text to Image</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_p_h3_t2i.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_p_h3_t2i-1.webp" width="200" height="200" alt="P-H3-Image: Text to Image"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_fl2va.json">Pika H3 Flex: FL2VA</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_fl2va.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_pika_h3_flex_fl2va-1.webp" width="200" height="200" alt="Pika H3 Flex: FL2VA"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v_continuation.json">Image to Video </a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v_continuation.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_i2v_continuation-1.webp" width="200" height="200" alt="Image to Video "></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_multiframe_reference.json">MiniMax H3: Multiframe Reference</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_multiframe_reference.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_multiframe_reference-1.webp" width="200" height="200" alt="MiniMax H3: Multiframe Reference"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json">MiniMax H3 Fun ControlNet Union</a></td><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_fun_controlnet_union-2.webp" width="200" height="200" alt="MiniMax H3 Fun ControlNet Union"></a><div align="center">Before</div></td><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_fun_controlnet_union-1.webp" width="200" height="200" alt="MiniMax H3 Fun ControlNet Union"></a><div align="center">After</div></td></tr>
</tbody>
</table>
Doc: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
