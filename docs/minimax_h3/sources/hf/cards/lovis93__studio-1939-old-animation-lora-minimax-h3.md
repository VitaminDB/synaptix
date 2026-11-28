# lovis93/studio-1939-old-animation-lora-minimax-h3

Источник: https://huggingface.co/lovis93/studio-1939-old-animation-lora-minimax-h3 · скачано 2026-11-28
Загрузок: 2228 · лайков: 89 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-08-25T14:40:49.000Z

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
- video
- text-to-video
- animation
- style
- minimax-h3
---

# STUDIO 1939

**A hand-painted, golden-age animation style LoRA for video generation.**

STUDIO 1939 is a style LoRA **for MiniMax H3** that turns any text prompt into footage that looks like it was painted and inked by a 1930s animation studio — gouache backgrounds with visible brushwork, warm celluloid palettes, and classical character animation. It was trained on a carefully curated selection from a public-domain 1939 animated feature. The weights are available now in two strengths — **light** and **strong** — and run today on MiniMax H3 on [fal](https://fal.ai).

Two entire short films were written, generated, edited and scored with it **in one night** (see below).

---

## Gallery

| | | |
|---|---|---|
| ![](stills/sby_bigbang.jpg) | ![](stills/r16_ocean.jpg) | ![](stills/sby_nebula.jpg) |
| ![](stills/sd_harbor.jpg) | ![](stills/sd_cloudsea.jpg) | ![](stills/sd_clockwork.jpg) |
| ![](stills/sd_whitecity.jpg) | ![](stills/sd_stormheart.jpg) | ![](stills/sd_frozen.jpg) |
| ![](stills/sby_giants.jpg) | ![](stills/sby_trex.jpg) | ![](stills/sby_cave.jpg) |

## Two flavors

| | |
|---|---|
| ![](stills/r16_horse.jpg) | ![](stills/r64_horse.jpg) |
| **studio1939-light (r16) — painterly.** Softer, more pictorial: detailed painted texture everywhere, closer to an illustrated storybook in motion. | **studio1939-strong (r64) — full cel.** Bold flat-shaded characters with clean outlines over painted backgrounds: the pure golden-age cartoon look. |

Same prompt, same seed — only the LoRA rank changes.

## Character consistency, from text alone

No character sheet, no reference images. Freeze a character description word-for-word and repeat it verbatim in every prompt — the same character comes back, scene after scene, world after world:

| | | |
|---|---|---|
| ![](stills/char_desert.jpg) | ![](stills/char_ice.jpg) | ![](stills/char_dive.jpg) |

## Made with STUDIO 1939

- **SIX BILLION YEARS** — an 8-minute animated film about the entire history of the universe. 87 scenes; all 261 shots generated in **28 minutes**.
- **THE SEVEN DOORS** — a 5-minute adventure: a boy chases a storm-bird through seven worlds to get his kite back. All 108 shots generated in **21 minutes**, the whole film made in under an hour.

Both films: one frozen character description, exhaustively art-directed prompts (palette, camera, and an explicit list of what must *not* appear), voiceover passed through a 1930s optical-track effect, and a fully scripted edit (crossfades inside sequences, hard cuts between worlds).

## Recipe

- Trigger: start the prompt with `gulliv3r, `
- Prompt expansion: **off** — write the full art direction yourself
- Works in 21:9, 16:9 and 4:3
- Scale: `1.0` for the full look; `0.4–0.8` to blend it under a modern art direction
- Best results with exhaustive prompts: name every color, every element, and end with what must not appear

## Get the weights

Both versions are **available now** and can already be used with several video models:

| Version | Rank | Character | Download |
|---|---|---|---|
| **studio1939-light** | 16 | painterly, storybook | [studio1939-light.safetensors](./studio1939-light.safetensors) |
| **studio1939-strong** | 64 | full golden-age cel | [studio1939-strong.safetensors](./studio1939-strong.safetensors) |

Use them today on the standard MiniMax H3 LoRA endpoints on fal — `minimax/h3/text-to-video/lora`:

```json
{
  "prompt": "gulliv3r, a white horse gallops across a windswept meadow at golden hour...",
  "aspect_ratio": "21:9",
  "prompt_expansion_mode": "disabled",
  "loras": [{ "path": "<safetensors url>", "scale": 1.0 }]
}
```

**The new massively accelerated endpoints (the ~20-seconds-per-shot workflow used for the films) land on [fal](https://fal.ai) this Wednesday.**

---

*by [Lovis Odin](https://x.com/OdinLovis) — style source: a public-domain 1939 animated feature.*
