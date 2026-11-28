# circlestone-labs/MiniMax-H3-Image-Undistill-Lora

Источник: https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Undistill-Lora · скачано 2026-11-28
Загрузок: 0 · лайков: 5 · базовая:  · теги:  · обновлено: 2026-11-19T00:37:18.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
---

Undistills MiniMax-H3-Image, making it act more similar to a base model. Designed mainly for training. Unlike Ostris's V1 adapter, this one is trained using a continuous timestep distribution, rather than a discrete timestep schedule. You should train on top of it like any other base model, without additional tricks or special configuration. Works well for training in diffusion-pipe. AI-Toolkit might be hardcoded for Ostris's adapter and the discrete timestep schedule, so it may not work properly there.

Trained on 10k diverse images with Gemma-4-31b captions for 15 epochs. Equal mix of 512 and 1024 resolution. The model was trained without architecture modifications: it still selects a PDD head based on the timestep, following the ComfyUI implementation, but it is trained to accept any continuous timestep between 0 and 1.

Weights are in ComfyUI format.
