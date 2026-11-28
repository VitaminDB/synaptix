# masafy/minimax-h3-masafy-lora

Источник: https://huggingface.co/masafy/minimax-h3-masafy-lora · скачано 2026-11-28
Загрузок: 318 · лайков: 1 · базовая: ["Comfy-Org/MiniMax-H3","MiniMaxAI/MiniMax-H3"] · теги: lora,character-lora,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-16T14:17:26.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- Comfy-Org/MiniMax-H3
- MiniMaxAI/MiniMax-H3
tags:
- lora
- video
- image-to-video
- minimax-h3
- character-lora
pipeline_tag: image-to-video
---

# minimax-h3-masafy-lora — "Masafy"

A character LoRA for **MiniMax H3**, a ~33B-parameter omni-modal model that
generates video and audio jointly. Trained on the Ref2VA variant and applied at
inference to its GGUF Q3_K_M build. Trigger token: **`masafy_character`**.

The subject is an original red-panda character created by the author. The full
technical report (JA/EN), the training dataset and all measured data are on GitHub:

> **https://github.com/masafykun/minimax-h3-masafy-lora**

## The finding behind this adapter

Training this adapter failed three times before it succeeded, and the reason was
not the one everybody expects. **Peak VRAM during training measured 3,959 MiB —
2.8% of the card in use.** What actually blocked the run was **system RAM**: the
41.2 GB model set (20.0 GB transformer + 15.7 GB Qwen3-VL 32B text encoder +
5.5 GB VAEs) is staged in host memory before it reaches the GPU, so hosts with
50 GB of RAM were killed by the OOM reaper while a host with 188 GB completed.

Rent by **system RAM**, not by VRAM. Advertised RAM tracks neither price nor VRAM.

## Checkpoints

| File | Steps | Notes |
|---|---:|---|
| `masafy_h3_ref2va_lora_v1_000000300.safetensors` | 300 | style already largely acquired |
| `masafy_h3_ref2va_lora_v1_000000400.safetensors` | 400 | |
| `masafy_h3_ref2va_lora_v1_000000500.safetensors` | 500 | **recommended** |
| `masafy_h3_ref2va_lora_v1_000000600.safetensors` | 600 | |
| `masafy_h3_ref2va_lora_v1_000000700.safetensors` | 700 | mild degradation |
| `masafy_h3_ref2va_lora_v1.safetensors` | 800 | final; no better than 500 |

**Recommended: checkpoint 500 at strength 0.8.** Cross-validation over strength
(0.0–1.2) and training length (300–800) put the optimum there. Strength trades
style fidelity against structural accuracy: raising it moves the style toward the
target while degrading hands, and at 1.2 a heavy dark contour appears around the
character — the die-cut border of the sticker artwork it was trained on.

800 steps at strength 1.0, the setting used initially, was excessive in both.

## Usage (ComfyUI)

Place a checkpoint in `ComfyUI/models/loras/` and insert `LoraLoaderModelOnly`
after `UnetLoaderGGUF`:

```
UnetLoaderGGUF → LoraLoaderModelOnly → BasicGuider
                                     → BasicScheduler
```

**Re-point both `BasicGuider` and `BasicScheduler`.** Patching only one leaves the
guider and the scheduler looking at different models, and it fails quietly.

Prompt with the trigger followed by the appearance:

```
masafy_character, a cute anthropomorphic red panda wearing brown aviator goggles
and a pink patterned scarf, waving at the camera with a big smile, standing in a
sunny park, smooth character animation, clean bright colors
```

**The trigger alone is not sufficient.** With 12 training images all captioned
with the same appearance clause, the caption text carries real weight; omitting
the appearance description does not produce a usable result.

## What the adapter contributes

Under reference-image (R2V) conditioning the reference already pins the character:
without the adapter the goggles, the scarf and the pose all render correctly and
without flicker. What changes is the **style** — flat cel illustration with the
adapter, photoreal fur without it. The return on this adapter is rendering the
character *in the author's own style*, not rendering the character at all.

The adapter also reshapes backgrounds, because captions described whole images.

## Requirements

| | |
|---|---|
| Inference VRAM | **9.3 GB** (fits a 12 GB consumer GPU) |
| Generation time | ~13 min per 5 s clip at 864×480 on an RTX 3060 |
| Base model | MiniMax H3 Ref2VA — **not redistributed here** |
| Quantization | trained on pruned INT8, applies cleanly to GGUF Q3_K_M (208/208 modules, zero key mismatches) |

## License and attribution

The adapter is a derivative work of MiniMax H3 and its use is subject to the
**MiniMax H3 Community License Agreement**
([text](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)).
Commercial use is permitted and **requires attribution to "MiniMax H3"**. The
license carries a territorial clause; consult the official terms for your
jurisdiction.

Weights were obtained from [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3),
the ComfyUI redistribution of [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3).
Neither is redistributed here.

Training additionally required the official training adapter,
[`ostris/minimax_h3_training_adapter`](https://huggingface.co/ostris/minimax_h3_training_adapter),
and the [ai-toolkit](https://github.com/ostris/ai-toolkit) trainer — thanks to
ostris for both.

The character "Masafy" is an original character created by the author. The design
remains the author's work — please credit the author if you use it.

## Citation

```bibtex
@techreport{suzuki2026h3lora,
  author = {Suzuki, Masato},
  title  = {Character LoRA Fine-tuning of a 33B Omni-modal Video Model:
            System RAM, Not VRAM, Was the Binding Constraint},
  year   = {2026},
  url    = {https://github.com/masafykun/minimax-h3-masafy-lora}
}
```
