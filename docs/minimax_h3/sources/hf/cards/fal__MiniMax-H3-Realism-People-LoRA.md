# fal/MiniMax-H3-Realism-People-LoRA

Источник: https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA · скачано 2026-11-28
Загрузок: 143314 · лайков: 579 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,template:diffusion-lora · обновлено: 2026-08-12T20:46:37.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
pipeline_tag: image-text-to-video
tags:
  - lora
  - safetensors
  - minimax
  - minimax-h3
  - h3
  - video
  - text-to-video
  - video-generation
  - realism
  - people
  - portrait
  - photorealistic
  - cinematic
  - film
  - fal
  - template:diffusion-lora
---

# MiniMax H3 Realism People LoRA

**Trigger word:** `r34l1sm`

A LoRA adapter for MiniMax H3 specialized in realistic people: faces that hold up in close-up, natural skin texture, believable expressions and gestures, film-style lighting and documentary camera movement.

Same prompt, same seed — base model on the left, this adapter on the right:

<video controls src="https://v3b.fal.media/files/b/0aa5ce26/stD_vdWuX1F90Q2OihK1Q_learn_before_after.mp4"></video>

## Before / after

19 pairs, **same prompt, same seed, adapter on vs off** — the only variable is the
LoRA. The trigger word is present on both sides, so it is not doing the work. Each
pair plays the base model first, then freezes and dims while the adapted version
plays beside it.

<video controls src="https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/before-after-comparison.mp4"></video>

Close-up talking faces, arguments, several people speaking at once, weathered
skin, children, ritual and travel scenes. Nothing cherry-picked from a larger
render batch: these are the pairs that were kept, in order.
([download the comparison](./before-after-comparison.mp4), 197s, 1920x1080)

## What it does

MiniMax H3 is already a strong general video model. This adapter pushes it further on human-centered shots: portraits, faces, hands at work, crowds and everyday characters. Skin keeps its texture instead of smoothing out, eyes and micro-expressions stay coherent, light behaves like it does on a film set, and motion gains a subtle handheld quality. It keeps H3's native synchronized audio.

It is the successor of [MiniMax-H3-Realism-LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-LoRA), retrained on a larger dataset focused on people.

## How to use

These are plain LoRA weights. Nothing here is tied to a hosted service — run them
wherever you run MiniMax H3.

### Locally, in ComfyUI

Download the `.safetensors` and drop it in `models/loras/`, then insert a **Load LoRA**
node between your model loader and the sampler. No conversion step: the keys are the
standard H3 layout (`diffusion_model.blocks.N.attn.qkv_proj`, fused QKV), the same one
other working H3 LoRAs use, so ComfyUI loads it as-is.

```
wget https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/h3-realism-people-t2v-i2v-r2v.safetensors
```

Start the prompt with the trigger word `r34l1sm`, then describe the scene. Scale 1.0 is
the intended strength; 0.6-0.8 for a lighter touch. It works on text to video, image to
video and reference to video, since the adapter only touches the shared attention
projections.

### In your own code

Any inference stack that can apply a LoRA to H3 will take it. The application is the
usual `W_eff = W + lora_B @ lora_A`.

### On a hosted endpoint

If you would rather not run it yourself, fal exposes the H3 LoRA endpoints — but this is
one option among others, not a requirement:

- Text to video: [fal.ai/models/minimax/h3/text-to-video/lora](https://fal.ai/models/minimax/h3/text-to-video/lora)
- Image to video: [fal.ai/models/minimax/h3/image-to-video/lora](https://fal.ai/models/minimax/h3/image-to-video/lora)
- Reference to video: [fal.ai/models/minimax/h3/reference-to-video/lora](https://fal.ai/models/minimax/h3/reference-to-video/lora)

```json
{
  "prompt": "r34l1sm, a young woman faces the camera in a quiet apartment at dusk, soft window light on her skin, shallow depth of field, subtle handheld sway, cinematic, photorealistic",
  "loras": [
    {
      "path": "https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/h3-realism-people-t2v-i2v-r2v.safetensors",
      "scale": 1.0
    }
  ],
  "duration": 5,
  "resolution": "768P"
}
```

Start the prompt with the trigger word `r34l1sm`, then describe the scene. A scale of 1.0 is the intended strength; lower it to 0.6-0.8 for a lighter touch.

## Files

| File | Task | Configuration |
| --- | --- | --- |
| `h3-realism-people-t2v-i2v-r2v.safetensors` | Text to video, image to video, reference to video | rank 32, 1500 steps, trained at high resolution |

Direct link:

```text
https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/h3-realism-people-t2v-i2v-r2v.safetensors
```

## Training

- Base model: MiniMax H3, trained with the fal H3 trainer.
- Dataset: 176 hand-curated live-action clips centered on people - portraits, faces, workers, athletes and everyday characters - building on the strongest shots from the first Realism dataset. Slow-motion footage was detected and retimed to natural speed, everything normalized to strict 24.000 fps with structured scene captions.
- Sixteen configurations (steps, rank, learning rate, training resolution) were trained and compared through side-by-side human review on same-seed prompt pairs — same prompt, same seed, adapter on vs off.
- The version published here is **rank 32, 1500 steps, trained at the high resolution bucket**. It was not the highest rank or the longest run that won: training resolution turned out to matter more than either. Skin texture, pores, fine hair and grain live in high spatial frequencies, and at a lower training bucket the latents barely carry them, so there is little for the adapter to learn.

## Credits

Created by [Lovis Odin](https://github.com/lovisdotio) at [fal](https://fal.ai).

## License

The adapter follows the MiniMax H3 Community License of the base model.
