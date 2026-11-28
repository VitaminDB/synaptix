# neonforestmist/Minimax_H3_Monet_Style_LoRA

Источник: https://huggingface.co/neonforestmist/Minimax_H3_Monet_Style_LoRA · скачано 2026-11-28
Загрузок: 186 · лайков: 1 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,style-lora · обновлено: 2026-11-24T16:17:06.000Z

---

---
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
datasets:
  - neonforestmist/GPT_Monet_Style_Images
pipeline_tag: text-to-video
tags:
  - lora
  - text-to-video
  - safetensors
  - minimax-h3
  - monet-style
  - impressionism
  - style-lora
---

# MiniMax H3 Monet Style LoRA

A Monet-inspired style LoRA for MiniMax H3 text-to-video. Use it to explore Impressionist light and color in scenes ranging from quiet landscapes to surreal modern worlds.

**Model file:** [minimax-h3-monet-style-lora.safetensors](./minimax-h3-monet-style-lora.safetensors)

## Sample Videos

Sample videos and their prompts are below.

### Sample 1 — The Forest City

<video src="https://huggingface.co/neonforestmist/Minimax_H3_Monet_Style_LoRA/resolve/main/assets/video-1.mp4" controls width="100%"></video>

[View sample video 1](./assets/video-1.mp4)

> monetstyle, at eye level on a flower-lined walkway in a futuristic forest city, a woman brushes her hand along soft green leaves beside a narrow stream. Vines climb the curved white buildings nearby, and droplets from a small waterfall float upward through warm sunlight as a transit pod glides past in the background. The camera follows gently beside her, lingering on shimmering water, dappled emerald light, and visible oil brushstrokes.

### Sample 2 — The Ballroom Dance

<video src="https://huggingface.co/neonforestmist/Minimax_H3_Monet_Style_LoRA/resolve/main/assets/video-2.mp4" controls width="100%"></video>

[View sample video 2](./assets/video-2.mp4)

> monetstyle, a woman with long blonde hair fading into a soft pink gradient dances in an opulent ballroom, wearing a flowing red dress with a fitted bodice and layered skirt. She turns beneath a crystal chandelier, her hair and dress sweeping around her as floating roses drift across the marble floor. The camera glides gently around her in a full-body shot, capturing flickering candlelight, rich crimson fabric, soft reflections, and visible oil brushstrokes.

### Sample 3 — Midnight Laundromat

<video src="https://huggingface.co/neonforestmist/Minimax_H3_Monet_Style_LoRA/resolve/main/assets/video-3.mp4" controls width="100%"></video>

[View sample video 3](./assets/video-3.mp4)

> monetstyle, inside a quiet laundromat, a young man leans close to a washing machine and watches a tiny moon rise behind its round glass door. The moon floats out toward his outstretched hand, scattering silver light across the machines while pink and blue neon shimmers on the rain-streaked window. The camera moves slowly closer in a medium shot, emphasizing luminous reflections, soft indigo shadows, and visible oil brushstrokes.

### Sample 4 — The Whale Above the City

<video src="https://huggingface.co/neonforestmist/Minimax_H3_Monet_Style_LoRA/resolve/main/assets/video-4.mp4" controls width="100%"></video>

[View sample video 4](./assets/video-4.mp4)

> monetstyle, a woman stands among flowers in a rooftop garden as a translucent whale swims slowly just above her, its immense body filling the sky. She reaches toward glowing droplets rising from a nearby fountain and drifting along the whale’s side. The camera moves gently beside her at eye level, catching golden light through the whale, rippling reflections, soft peach and turquoise colors, and visible oil brushstrokes.

### Sample 5 — The Floating Café

<video src="https://huggingface.co/neonforestmist/Minimax_H3_Monet_Style_LoRA/resolve/main/assets/video-5.mp4" controls width="100%"></video>

[View sample video 5](./assets/video-5.mp4)

> monetstyle, three friends sit together at a sidewalk café on a modern city street, sharing coffee beneath a canopy of flowering vines. As they laugh, their cups lift gently from the table and tiny water lilies bloom in the coffee, while passing headlights shimmer in the rain-soaked pavement. The camera drifts closer at table height, capturing warm café light, soft blue and pink reflections, and visible oil brushstrokes.

## Key Visual Traits

- Expressive oil brushstrokes and soft, atmospheric detail
- Shimmering light, color, and reflections
- Surreal scenes across natural and modern settings

## Trigger Word

`monetstyle`

## Usage

Load `minimax-h3-monet-style-lora.safetensors` in a MiniMax H3 FL2VA-compatible LoRA workflow and include `monetstyle` in the prompt. For the strongest visual expression of the style, a **LoRA weight/scale of 2 or higher** is recommended. Adjust the weight and generation settings to suit your scene.

## Training Details

- **Base model:** MiniMax H3 FL2VA (`minimax-h3-fl2va`).
- **Training dataset:** [GPT Monet Style Images](https://huggingface.co/datasets/neonforestmist/GPT_Monet_Style_Images).
- **Training strategy:** Text-to-video.
- **Training steps:** 2,000.
- **LoRA rank:** 16.
