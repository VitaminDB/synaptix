# Comfy-Org/MiniMax-H3-Image

Источник: https://huggingface.co/Comfy-Org/MiniMax-H3-Image · скачано 2026-11-28
Загрузок: 5052954 · лайков: 808 · базовая: ["MiniMaxAI/MiniMax-H3-Image"] · теги: comfyui · обновлено: 2026-11-23T10:46:27.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
tags:
- comfyui
- diffusion-single-file
base_model:
- MiniMaxAI/MiniMax-H3-Image
pipeline_tag: text-to-image
---

# MiniMax H3 Image

Repackaged model files for ComfyUI.

Original model repository:

- https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
- https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE

## Prompt Enhancer

For the best results we recommend to use the Generate Text node for prompt enhancement, with the model `qwen3.8_27b_minimax_h3_image_pe_w4a8_gptq.safetensors`.

Official system prompts for T2I: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/tree/main/prompt

---

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_image_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3.5_9b_w4a8_gptq.safetensors
│   │   ├── qwen3.8_27b_minimax_h3_image_pe_w4a8_gptq.safetensors
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   └── minimax_h3_image_vae_fp16.safetensors
```

## License

Note [MiniMax H3 Image license agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) is different from the original MiniMax H3 license:

> You shall obtain a separate, prior written authorization from MiniMax by sending a request through https://platform.minimax.io/license, if the Annualized Revenue of you and/or your affiliates exceeds 1 Million USD, or, you provide Hosted Service(s) to any Third Parties.

## Workflows

<table>
<thead>
<tr><th align="center" valign="middle" style="text-align:center;vertical-align:middle">Workflow</th><th colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle">Thumb</th></tr>
</thead>
<tbody>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json">MiniMax-H3-Image: Image Edit</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_edit-1.webp" width="200" height="200" alt="MiniMax-H3-Image: Image Edit"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json">MiniMax-H3-Image: Text to Image</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_t2i-1.webp" width="200" height="200" alt="MiniMax-H3-Image: Text to Image"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_character_swap.json">MiniMax-H3-Image: Character Swap</a></td><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_character_swap.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_character_swap-2.webp" width="200" height="200" alt="MiniMax-H3-Image: Character Swap"></a><div align="center">Before</div></td><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_character_swap.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_minimax_h3_image_character_swap-1.webp" width="200" height="200" alt="MiniMax-H3-Image: Character Swap"></a><div align="center">After</div></td></tr>
</tbody>
</table>
