# berryber09/10Eros-Max-h3-turbo-beta6-pruned-w4a8

Источник: https://huggingface.co/berryber09/10Eros-Max-h3-turbo-beta6-pruned-w4a8 · скачано 2026-11-28
Загрузок: 2666 · лайков: 3 · базовая: TenStrip/10Eros-Max · теги: comfyui,quantized,turbo · обновлено: 2026-10-12T13:04:16.000Z

---

---
license: other
tags:
  - minimax-h3
  - comfyui
  - w4a8
  - quantized
  - text-to-video
  - turbo
base_model: TenStrip/10Eros-Max
---

# MiniMax-H3 10Eros-Max beta6 pruned (8-step TURBO) — W4A8

W4A8 quantization of **`10Eros_Max_h3_8stepTURBO_beta6_pruned`** (bf16 source: [TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max), `10Eros_Max_h3_8stepTURBO_beta6_pruned.safetensors`) — the 8-step turbo build (TenStrip/10Eros-Max).

- **Format:** `asym_w4a8_int8`, `group_size=16`, `convrot_groupsize=256`, per-tensor Lloyd-Max codebook + fp8 group scales (Kijai / comfy-kitchen W4A8 layout).
- **Size:** ~11.68 GB — 68.8% smaller than the ~37.5 GB bf16 source; fits a 16 GB card with headroom.
- **Layers:** 200 2D-linear layers quantized (96% of policy-targeted bytes); norms / first / last kept higher-precision (332 kept).
- **Quantizer:** [comfyui-mixed-quantizer](https://github.com/NidAll/comfyui-mixed-quantizer) `--format w4a8 --group-size 16 --codebook-mode fit`; the quantizer's full validation passes (421 passed, 0 failed).
- **Verified:** loads via UNETLoader (convrot) and generates coherent video with speech in ComfyUI.
- **sha256:** `96c5d85b1f658d820e2a66e33f41baf31002e7f838eaa2bf6487bd43fc94bbf8`

## Requirements
- ComfyUI ≥ v0.31.0 (native W4A8 loader) **or** the `comfyui_w4a8_loader.patch`.
- comfy-kitchen with `AsymW4A8Int8Layout` (PR #90).
- CUDA SM ≥ 8.0.

Inherits the source model's license.
