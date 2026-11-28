# akhaliq/MiniMax-H3-Image-Character-Swap

Источник: https://huggingface.co/akhaliq/MiniMax-H3-Image-Character-Swap · скачано 2026-11-28
Загрузок: 484 · лайков: 214 · базовая: ["MiniMaxAI/MiniMax-H3-Image","akatz-ai/MiniMax-H3-Character-Swap-LoRA"] · теги: lora-merge,base_model:akatz-ai/MiniMax-H3-Character-Swap-LoRA,base_model:merge:akatz-ai/MiniMax-H3-Character-Swap-LoRA · обновлено: 2026-11-14T21:17:04.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
library_name: diffusers
pipeline_tag: image-to-image
base_model:
  - MiniMaxAI/MiniMax-H3-Image
  - akatz-ai/MiniMax-H3-Character-Swap-LoRA
base_model_relation: merge
language:
  - en
tags:
  - minimax-h3
  - diffusers
  - character-swap
  - image-editing
  - image-to-image
  - lora-merge
  - ml-intern
---

<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 2em"><a href="https://hf.co/chat?mode=ml-intern"><img src="https://hf.co/datasets/huggingface/badges/resolve/main/built-with-ml-intern-lg.svg" alt="Built with ML Intern" style="margin:0"></a><a href="https://hf.co/chat?mode=ml-intern&prompt=Build%20a%20Gradio%20demo%20Space%20for%20akhaliq%2FMiniMax-H3-Image-Character-Swap" style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border:1.5px solid #eee;border-radius:16px;background:#f9fbfc;color:#c2410c;font-weight:600;font-size:14px;text-decoration:none">Build a demo Space<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M13.5 6H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8.5m-10.5 6L21 3m0 0h-5m5 0v5"/></svg></a><a href="https://hf.co/chat?mode=ml-intern&prompt=Train%20a%20character-swap%20LoRA%20natively%20on%20MiniMax-H3-Image%20using%20the%20recipe%20in%20akhaliq%2FMiniMax-H3-Image-Character-Swap" style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border:1.5px solid #eee;border-radius:16px;background:#f9fbfc;color:#c2410c;font-weight:600;font-size:14px;text-decoration:none">Train a native H3-Image LoRA<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M13.5 6H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8.5m-10.5 6L21 3m0 0h-5m5 0v5"/></svg></a></div>

# MiniMax-H3-Image · Character Swap

**Character swap on images, merged and ready.** This repo is [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
with [akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
fused directly into the image DiT — the first merged checkpoint that ports the video-trained
character-swap adapter onto the H3 image model. No LoRA files, no key remapping, no ComfyUI graph:
load the pipeline and swap.

Feed it a scene photo as `<Picture 1>` and a character reference as `<Picture 2>`, name who to
replace, and it swaps the person while preserving the background, the lighting, the other people
and the pose — including **cross-style swaps (anime/illustrated character → photoreal scene) that
the base image model fails on**.

## Showcase

*Original scene (`<Picture 1>`) · character reference (`<Picture 2>`) · character swap — all single-seed, straight out of the merged model.*

![Anime character swapped into a side-profile bookstore scene — the case native H3-Image fails on every seed](assets/showcase_profile.jpg)

![Photoreal hiker character swapped into a café conversation](assets/showcase_cafe.jpg)

![Anime character swapped into a full-body jogging scene](assets/showcase_fullbody.jpg)

## The case that motivated this repo

Native MiniMax-H3-Image is a strong multi-reference editor, but it **systematically fails**
cross-style swaps onto non-frontal faces: given an anime character and a strict side-profile
photo, it kept the original face on 3 out of 3 seeds (producing a hybrid, pasting the character
into the newspaper, or ignoring the swap). And even where native appears to succeed, it drifts —
skin tones lighten, poses change, outfit details wander (see the comparisons below). The merged
model nails the hard case on the same seed and holds the details everywhere else:

| Scene (`<Picture 1>`) | Character (`<Picture 2>`) | Native H3-Image | **This model** |
| --- | --- | --- | --- |
| ![scene](assets/profile_scene.jpg) | ![character](assets/character_anime.png) | ❌ kept the source face | ✅ |
| ![native fail](assets/profile_native.jpg) | | | ![merged](assets/profile_merged.jpg) |

## More swaps (native vs merged, same seed)

**Photoreal character → café scene.** Native drifts — the swapped woman's skin tone comes out
visibly lighter than the reference. The merged model keeps her identity.

| Scene | Native | Merged |
| --- | --- | --- |
| ![scene](assets/cafe_scene.png) | ![native](assets/cafe_native.png) | ![merged](assets/cafe_merged.jpg) |

**Anime character → full-body scene.** Native changes the pose — the jog becomes a standing
hand-in-pocket. The merged version keeps the stride *and* the reference outfit detail.

| Scene | Native | Merged |
| --- | --- | --- |
| ![scene](assets/fullbody_scene.jpg) | ![native](assets/fullbody_native.jpg) | ![merged](assets/fullbody_merged.jpg) |

Full test set (8 hard cases: profile, small face, occlusion, full-body, group): [akhaliq/h3-image-swap-tests](https://huggingface.co/datasets/akhaliq/h3-image-swap-tests).

**Three-way labeled comparison (native vs standalone LoRA vs merged).** 6 scenes x 2 characters x
3 seeds, same prompt, 4 MP each — 17 scored comparisons. The **merged checkpoint was preferred in
14/17 (82%)**, the standalone LoRA in 2, native H3-Image in 1. Single labeler, so read it as a
strong preference signal rather than a benchmark; the full grid (scene + reference + all three
outputs, every seed) is public in
[akhaliq/h3-image-swap-tests](https://huggingface.co/datasets/akhaliq/h3-image-swap-tests)
(`blind-v1/`, `blind-v2/`).

## FAQ

**“Do you even need a LoRA for this? Doesn't H3-Image do swaps with the right prompt?”**
It produces *something plausible* on most cases — but look closely. In 17 labeled three-way
comparisons (6 scenes x 3 seeds), native was picked best only **once** (6%). And the pattern of its
losses is consistent — across our 8-case test set,
native H3-Image drifts on exactly the details that make a swap a swap: **skin tone shifts** (café
case: the swapped woman comes out visibly lighter than the reference), **pose changes** (full-body
case: the jogging stride becomes a standing hand-in-pocket), outfit details wander, and on
cross-style swaps onto non-frontal faces it fails outright — anime character → strict-profile
photo failed on 3/3 seeds (a hybrid face, the character printed into a prop, and the swap
ignored). The merged model held identity, outfit and pose on every case we validated. If “close
enough at a glance” is fine, the stock model plus our prompt template may do. If the person has to
actually look like your reference — that's what the merge is for.

**“Where's the LoRA? Why is the ‘LoRA’ 40 GB?”**
It isn't a 40 GB adapter — this repo is the adapter *fused into* the full checkpoint (same size as
Comfy-Org's stock DiT), so there's one file and nothing to configure. If you want the 155 MB
adapter by itself:

| File | Layout | Use |
| --- | --- | --- |
| [`lora/h3_character_swap_v1.5.safetensors`](lora/h3_character_swap_v1.5.safetensors) (155 MB) | ComfyUI (akatz's original, unchanged) | `models/loras/` + **LoraLoaderModelOnly** @ strength 1.0 on the stock H3-Image DiT — all 208 targets verified to match |
| [`lora/h3_image_character_swap_diffusers.safetensors`](lora/h3_image_character_swap_diffusers.safetensors) (173 MB) | diffusers | `transformer.load_lora_adapter(state_dict, prefix=None)` for strength control |

No extraction needed — the merged checkpoint is exactly `W + B·A` of these adapters.

**“In some examples the character looks the wrong way.”**
That's the design, not a bug: a swap follows the *source* person's pose, position and gaze (it's
what makes it a swap instead of a paste). The model re-renders the reference character facing the
source's way; on extreme profile/cross-style combos that re-render can look off. Rephrase the
“same pose, position, scale and gaze” part of the prompt if you'd rather follow the reference's
orientation.

**“Can I use it commercially?”**
The checkpoint is under the [MiniMax-H3-Image Community License](LICENSE). Free, including
commercial use, in the Applicable Territory — which is worldwide **except the EU, UK, South Korea
and the USA**. Two triggers require MiniMax's prior written authorization (apply
[here](https://platform.minimax.io/console/license)): annualized revenue above **$1M**, or
providing it as a **hosted service** to third parties. In the excluded territories the community
license grants no rights — contact MiniMax. MiniMax claims no rights over your outputs. (This is a
summary, not legal advice — read the LICENSE.)

**“What prompt should I use?”**
See [Prompting tips](#comfyui) — scene as `<Picture 1>`, character as `<Picture 2>`, name the
person, and say what to preserve.

## ComfyUI

One file, no LoRA nodes, no adapters — character swap is built into the checkpoint itself.

**➡ [Download the character-swap workflow](comfyui/workflows/minimax-h3-image-character-swap.json)**
(drag the JSON onto the ComfyUI canvas; it's the official *MiniMax-H3-Image: Image Edit* template
pre-wired with a second reference input and the swap prompt — you only load your two images.)

Use the pre-merged ComfyUI checkpoint — a drop-in replacement for Comfy-Org's stock file:

| File | Source | Goes in |
| --- | --- | --- |
| `minimax_h3_image_character_swap_bf16.safetensors` (40 GB) | **this repo**, [`comfyui/`](comfyui/minimax_h3_image_character_swap_bf16.safetensors) | `ComfyUI/models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (25 GB) | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) | `ComfyUI/models/text_encoders/` |
| `minimax_h3_image_vae_fp16.safetensors` (1.2 GB) | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) | `ComfyUI/models/vae/` |
| `qwen3.5_9b_w4a8_gptq.safetensors` (7.3 GB, optional prompt expander) | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) | `ComfyUI/models/text_encoders/` |

1. Requires **ComfyUI 0.45.0+** with template pack **0.11.104+**.
2. Open our [character-swap workflow](comfyui/workflows/minimax-h3-image-character-swap.json)
   (or the stock **"MiniMax-H3-Image: Image Edit"** template) and point the diffusion
   model loader at `minimax_h3_image_character_swap_bf16.safetensors`.
3. Load the scene as the first `Load Image` (`<Picture 1>`) and the character reference as a second
   one (`<Picture 2>`), then prompt, e.g.:
   > Swap the person in `<Picture 1>` with the character in `<Picture 2>`. Keep the reference
   > character's face, hair and outfit. Preserve the background, lighting, camera and all other people.
4. Sampling comes pre-set by the template: **euler / simple, 6 steps, CFG 1, denoise 1**. 4 MP is a
   good starting canvas; the model is native at ~16 MP.

The merged file is numerically verified against the LoRA delta (208/208 targeted tensors, bit-exact
`W + B·A`) and ships the same pruned-bf16 layout as Comfy-Org's official file, so every stock
template that loads `minimax_h3_image_pruned_bf16.safetensors` works unchanged — just with
character-swap built in. Fusion provenance: [`comfyui/minimax_h3_image_character_swap_bf16.provenance.json`](comfyui/minimax_h3_image_character_swap_bf16.provenance.json).

Because the adapter is fused in, there is nothing else to install: no LoRA node, no strength to
tune — every image you edit with this checkpoint has character swap built in.
(Prefer the 155 MB adapter by itself with strength control? See [the FAQ](#faq) — it's in [`lora/`](lora/).)

*Note: end-to-end runs on this page were validated through the diffusers pipeline; the ComfyUI
checkpoint is the identical math in Comfy's own layout. Report any ComfyUI-specific surprises in
the discussions tab.*

## Usage (diffusers)

Requires `diffusers==0.43.0`. The text encoder (~67 GB) is referenced from
`MiniMaxAI/MiniMax-H3-Image` and loaded by the modular pipeline automatically.

```python
import torch
from diffusers import ModularPipeline

REPO = "akhaliq/MiniMax-H3-Image-Character-Swap"
blocks = ModularPipeline.from_pretrained(REPO).blocks
pipe = blocks.init_pipeline(REPO)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")  # full local run needs ~140 GB VRAM (DiT 66 GB + Qwen3-VL encoder 67 GB)

prompt = (
    "Swap the man in the purple shirt in <Picture 1> with the character in <Picture 2>. "
    "Keep the replacement character's face, hair, identity and outfit exactly as in <Picture 2>, "
    "including its art style. The replacement must hold the same pose, position, scale and gaze. "
    "Preserve everything else in <Picture 1> unchanged."
)
image = pipe(
    prompt=prompt,
    image=[scene_image, character_image],  # PIL images: <Picture 1>, <Picture 2>
    width=2720, height=1536,               # ~4 MP, multiple of 32; 16 MP also works
    generator=torch.Generator("cpu").manual_seed(904231),
    output="images",
)[0]
```

**Single-GPU trick:** the heavy part is the text encoder, not the swap. Run the encoder remotely
(the official demo Space splits it out the same way) and only the 66 GB DiT + 2 GB VAE touch your
GPU — see [`merge/merge_and_validate.py`](merge/merge_and_validate.py), which runs exactly that way
on one H200. The merged DiT is a drop-in replacement anywhere
`MiniMaxAI/MiniMax-H3-Image`'s `transformer_diffusers/` is used.

**Prompting tips** (from the LoRA author's recipe, adapted to images):

- Order matters: scene first (`<Picture 1>`), character second (`<Picture 2>`).
- Name the target person explicitly ("the man in the yellow sweater, second from the left").
- Say "keep the reference character's face, hair and outfit… preserve the scene, lighting, camera
  and all other people."
- 6 sampling steps, CFG 1 (the model is distilled) — no tuning needed.

## What this is, exactly

- **Merge:** `h3_character_swap_v1.5.safetensors` (rank 16, 312 modules: attention + FFN of all 50
  DiT blocks and the 2 token-refiner blocks) converted with diffusers 0.43.0's
  `_convert_non_diffusers_minimax_h3_lora_to_diffusers` and fused at strength 1.0
  (`W += B @ A`, alpha == rank) into `transformer_diffusers/`. Verified 624/624 key + shape match
  before fusing.
- **Validation:** fused outputs match the LoRA-attached model within bf16 sampler rounding
  (21-27 dB PSNR). In extended 3-way labeled comparisons (17 cases, 6 scenes x 3 seeds) the merged
  checkpoint was preferred in **82% (14/17)** — native won once, the standalone LoRA twice.
- **Not included:** the classic `MiniMaxH3ImagePipeline` layout (`transformer/`, PDD heads) — the
  merge exists only in the modular `transformer_diffusers/` layout. The audio/video capabilities
  of the H3 video model are unaffected (they live in a different checkpoint).
- **Provenance:** merge + validation script: [`merge/merge_and_validate.py`](merge/merge_and_validate.py).

## Limitations

- The LoRA was trained for **video** character swap (on Ref2VA, 24 fps clips); the port to images
  is validated on 8 hard cases, not a broad benchmark. Report surprises in the discussions tab.
- Swaps follow the source person's pose and gaze by design; on extreme profile/cross-style combos
  the re-rendered viewing direction can look off — tune the pose/gaze phrase in the prompt.
- Like the base model, it can drift on extreme poses, heavy occlusion, or very small faces.
- The character-swap LoRA is an experimental community model; identity fidelity varies with the
  reference quality (a clean, multi-view character sheet works best).
- Use responsibly: swapping real people into scenes carries consent and likeness rights. Do not
  use it to depict real people without their permission.

## License & attribution

- Merged checkpoint: [MiniMax-H3-Image Community License Agreement](LICENSE) (same as the base
  model; the character-swap LoRA carries the same MiniMax H3 Community License family).
- Base image model: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) (MiniMax).
- Character-swap LoRA: [akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA) (Akatz Labs),
  trained on [akatz-ai/H3-Character-Swap-v1](https://huggingface.co/datasets/akatz-ai/H3-Character-Swap-v1).
- Example character sheets: from the LoRA's own demo assets.
