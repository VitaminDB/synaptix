# Alissonerdx/TE-PK-Minimax-H3

Источник: https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3 · скачано 2026-11-28
Загрузок: 6198 · лайков: 88 · базовая:  · теги: comfyui · обновлено: 2026-11-28T08:27:30.000Z

---

---
license: apache-2.0
tags:
  - comfyui
  - minimax-h3
  - text-encoder
  - platos-key
  - ridge-regression
  - qwen3.5
  - gemma4
---

# ComfyUI-H3-Text-Encoders

> [!IMPORTANT]
> **NEW (`v0.4.1` — Experimental `pk_lora_vision` Variant for Qwen 3.5 & Google Gemma 4 + `pk_lora` Unified Adapters):**
> - **Experimental VLM Vision Variant (`*_pk_lora_vision_*.safetensors` — Qwen 3.5 `9B`/`2B`/`0.8B` & Gemma 4 `E4B`/`E2B`):**
>   - **What it does:** Includes the full `pk_lora` text adapter **plus** a dedicated calibrated **`W_vision`** projection matrix. When you pass reference images (`minimax_ref_items`, Ref2VA / RefPacks) or keyframes (`images`, FL2VA / FLF2V / H3-Image), the node automatically routes those images through the donor model's native Vision Tower, projects the visual tokens (`tag = 0`) with `W_vision` and text tokens (`tag = 1`) with `pk_lora`, and emits exact `minimax_token_tags` — **zero workflow changes**.
>   - **Why it is kept separate:** `pk_lora_vision` is **experimental** and has **not yet been 100% exhaustively tested** across all scenes, seeds, and workflows to guarantee it never alters visual/motion quality compared to the standard text-only `pk_lora` (where reference images are already handled by MiniMax-H3's Video VAE latents inside the DiT). Because of this — and because `W_vision` adds extra file size — both `*_pk_lora_*.safetensors` (standard, lighter) and `*_pk_lora_vision_*.safetensors` (experimental VLM vision) are provided side-by-side.
> - **Where we started → Why `pk_lora`:** This project began with **pure zero-training Plato's Key** (closed-form linear Ridge). To achieve crisp **spoken audio & multi-shot quality** (especially on cross-family tokenizers like Gemma 4), we combined **Plato's Key** with **Forced Qwen-Token Segmentation** and a **lightweight trained Projection + Rank-32 Text Encoder LoRA** — all bundled inside **1 single `.safetensors` file** (`*_pk_lora_*.safetensors`, works on `bf16` / `fp8` / `int8` / `nvfp4`, **zero DiT changes**).
> - **Automatic Start-of-Video Speech Fix (`inject_token0_sink = True` in `v0.3.3+`):** Restores `Qwen3-VL-32B`'s universal 20 KB Token-0 Attention Sink vector in `<0.01 ms` to eliminate clipped opening words across all keys.
> - **All Variants Stay Available Side-by-Side:**
>   - **Qwen 3.5 / Qwen 3.8:** **`*_pk_lora_*.safetensors`** *(Recommended standard — up to `cos 98.0%` / `quotes 96.4%`, lighter)* · **`*_pk_lora_vision_*.safetensors`** *(Experimental +VLM Vision on `9B`/`2B`/`0.8B`, `v0.4.0+`)* · **`*_pk_ridge_*.safetensors`** *(pure zero-training Plato's Key, `R² 0.985–0.992`)*.
>   - **Gemma 4 / Inkling-2-Mini:** **`*_pk_lora_*.safetensors`** *(Recommended standard — Speech + Multi-Shot, lighter)* · **`*_pk_lora_vision_*.safetensors`** *(Experimental +VLM Vision on `E4B`/`E2B`, `v0.4.1+`)* · **`*_pk_ridge_charov_*.safetensors`** *(pure zero-training linear, `v0.2.0+`)* · **`*_pk_ridge_4layer_*.safetensors`** *(legacy word-span, `v0.1.8+`)*.

> [!WARNING]
> **Update your custom node (`v0.4.1+` recommended for ALL users!)** Run `cd ComfyUI/custom_nodes/ComfyUI-H3-Text-Encoders && git pull` (or update via ComfyUI Manager) and restart ComfyUI.

Drop-in **Plato's Key** text encoder adapters to run modern LLMs/VLMs (**Qwen 3.5**, **Qwen 3.8**, **Google Gemma 4**, **Inkling-2-Mini**, or any HuggingFace `transformers` model) as a conditioning encoder for **MiniMax-H3** (**T2VA**, **FL2VA**, **Ref2VA**) in ComfyUI — **1 single Text Encoder, 1 `.safetensors` key, zero DiT changes**.

Custom node repository: [**`alisson-anjos/ComfyUI-H3-Text-Encoders`**](https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders).

![Plato's Key + Forced Qwen Segmentation + Bundled TE LoRA Architecture](showcase/H3_PlatosKey_v030_PK_LoRA_Diagram.png)

> **Deep-Dive Docs:** [`PK_LORA.md`](https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders/blob/main/docs/PK_LORA.md) · [`CHAR_OVERLAP_V6.md`](https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders/blob/main/CHAR_OVERLAP_V6.md) · [`ENGINEERING_NOTES.md`](https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders/blob/main/docs/ENGINEERING_NOTES.md)

---

## Pre-Calibrated Weights & Direct Download Links

### Same-Tokenizer Encoders (Qwen 3.5 / Qwen 3.8 — 1:1 Token Ridge & `pk_lora`)

| Encoder | ComfyUI Loader | Text Encoder Download | **`pk_lora` Unified (`v0.3.3+`, Best Audio)** | **`pk_ridge` Zero-Train (`Cos` / `R²`)** |
|---|---|---|---|---|
| **Qwen 3.5 9B** 🏆⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`int8` `9.8 GB` / `bf16` `19.3 GB`) | [`pk_lora_qwen3.5_9b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_qwen3.5_9b.safetensors) (**cos 97.9%**, `H3 R² 0.912`) · [`pk_lora_vision_qwen3.5_9b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_vision_qwen3.5_9b.safetensors) *(exp. +VLM Vision, `vis cos 85.6%`)* | [`pk_ridge_qwen3.5_9b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors) (**93.05%** / `R² 0.9905`) |
| **Qwen 3.5 2B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`bf16` `4.6 GB`) | [`pk_lora_qwen3.5_2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_qwen3.5_2b.safetensors) (`cos 97.1%`, `H3 R² 0.874`) · [`pk_lora_vision_qwen3.5_2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_vision_qwen3.5_2b.safetensors) *(exp. +VLM Vision, `vis cos 84.7%`)* | [`pk_ridge_qwen3.5_2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors) (**92.57%** / `R² 0.9916`) |
| **Qwen 3.5 4B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`bf16` `9.3 GB`) | [`pk_lora_qwen3.5_4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_qwen3.5_4b.safetensors) (`cos 97.6%`, `H3 R² 0.899`, `quotes 95.6%`) | [`pk_ridge_qwen3.5_4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_4b.safetensors) (**91.82%** / `R² 0.9902`) |
| **Qwen 3.5 0.8B** | `PlatosKeyHFTextEncoderLoader` | [**`Qwen/Qwen3.5-0.8B`**](https://huggingface.co/Qwen/Qwen3.5-0.8B) | [`pk_lora_qwen3.5_0.8b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_qwen3.5_0.8b.safetensors) (`cos 96.2%`, `H3 R² 0.835`) · [`pk_lora_vision_qwen3.5_0.8b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_lora_vision_qwen3.5_0.8b.safetensors) *(exp. +VLM Vision, `vis cos 84.1%`)* | [`pk_ridge_qwen3.5_0.8b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_0.8b.safetensors) (**91.11%** / `R² 0.9901`) |
| **Qwen 3.8 27B** 🏆 | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.8-27B`**](https://huggingface.co/Comfy-Org/Qwen3.8-27B/tree/main/text_encoders) (`w4a8` `17.3 GB`) | [`pk_lora_qwen3.8_27b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.8/minimax_h3_32b_pk_lora_qwen3.8_27b.safetensors) (**cos 98.0%**, `H3 R² 0.917`, `quotes 96.4%`) | [`pk_ridge_qwen3.8_27b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.8/minimax_h3_32b_pk_ridge_qwen3.8_27b.safetensors) (**89.94%** / `R² 0.9846`) |

### Cross-Family Encoders (`pk_lora` Unified vs `charov` Zero-Training vs `4layer` Legacy)

> **Why Cross-Family Encoders Need `pk_lora` for Audio & Spoken Dialogue:** Unlike `Qwen 3.5 / 3.8` (which share `Qwen3-VL-32B`'s exact BPE tokenizer 1:1), **Gemma 4** and **Inkling-2-Mini** use completely different tokenizers and internal hidden-state geometry. Purely linear keys (**`4layer`** / **`charov`**) transfer visuals well, but struggle with crisp spoken dialogue and multi-shot choreography. **`pk_lora`** bundles **Forced Qwen-Segmentation + Multi-Layer Plato's Key + a lightweight Rank-32 Text Encoder LoRA** in 1 single `.safetensors` file to fix spoken audio & multi-shot alignment (zero DiT changes).

| Encoder | ComfyUI Loader | **`pk_lora` Unified (`v0.3.2+` · ✅ Speech + Multi-Shot)** | **`charov` Zero-Train (`v0.2.0+` · ⚠️ Visuals OK, Speech Limited)** | **`4layer` Legacy (`v0.1.8+` · ❌ Visuals Only, Garbled Speech)** |
|---|---|---|---|---|
| **Gemma 4 E4B** 🏆 | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`Comfy-Org/gemma-4`](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders)) | [`pk_lora_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_gemma4_e4b.safetensors) (**cos 97.6%**, `H3 R² 0.894`) · [`pk_lora_vision_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_vision_gemma4_e4b.safetensors) *(exp. +VLM Vision, `vis cos 84.5%`)* | [`charov_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_e4b.safetensors) (`95.59%`) | [`4layer_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_e4b.safetensors) (`94.86%`) |
| **Gemma 4 E2B** ⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`Comfy-Org/gemma-4`](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders)) | [`pk_lora_gemma4_e2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_gemma4_e2b.safetensors) (`cos 96.8%`, `H3 R² 0.864`) · [`pk_lora_vision_gemma4_e2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_vision_gemma4_e2b.safetensors) *(exp. +VLM Vision, `vis cos 84.3%`)* | [`charov_gemma4_e2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_e2b.safetensors) (`94.77%`) | [`4layer_gemma4_e2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_e2b.safetensors) (`94.46%`) |
| **Gemma 4 12B** | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`Comfy-Org/gemma-4`](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders)) | [`pk_lora_gemma4_12b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_gemma4_12b.safetensors) (`cos 97.3%`, `H3 R² 0.889`) | [`charov_gemma4_12b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_12b.safetensors) (`95.26%`) | [`4layer_gemma4_12b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_12b.safetensors) (`93.67%`) |
| **Inkling-2-Mini** | `PlatosKeyHFTextEncoderLoader` ([`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)) | [`pk_lora_inkling2_mini`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/inkling2/minimax_h3_32b_pk_lora_inkling2_mini.safetensors) (`cos 96.8%`, `H3 R² 0.863`, `quotes 94.2%`) | [`charov_inkling2_mini`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/inkling2/minimax_h3_32b_pk_ridge_charov_inkling2_mini.safetensors) (`93.52%`) | [`4layer_inkling2_mini`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/inkling2/minimax_h3_32b_pk_ridge_4layer_inkling2_mini.safetensors) (`96.23%`) |
| **Gemma 4 26B-A4B** | `PlatosKeyHFTextEncoderLoader` ([`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)) | [`pk_lora_gemma4_26b_a4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_gemma4_26b_a4b.safetensors) (`cos 97.4%`, `H3 R² 0.892`, `quotes 95.2%`) | [`charov_gemma4_26b_a4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_26b_a4b.safetensors) (`95.32%`) | [`4layer_gemma4_26b_a4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_26b_a4b.safetensors) (`95.43%`) |
| **Gemma 4 31B** 🏆 | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`text_encoders/`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/tree/main/text_encoders)) | [`pk_lora_gemma4_31b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_gemma4_31b.safetensors) (**cos 97.9%**, `H3 R² 0.912`, `quotes 96.0%`) | [`charov_gemma4_31b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_31b.safetensors) (`95.90%`) | [`4layer_gemma4_31b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_31b.safetensors) (`95.97%`) |

---

## Quick Setup

1. **Install or update (`v0.3.2+`):**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders.git
cd ComfyUI-H3-Text-Encoders && pip install -r requirements.txt
```

2. **Place files:**
```text
ComfyUI/models/
├── text_encoders/
│   ├── qwen3.5_9b_int8_convrot.safetensors
│   └── gemma4_e4b_it_fp8_scaled.safetensors
└── clip_projections/
    ├── minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors
    └── minimax_h3_32b_pk_lora_gemma4_e4b.safetensors
```

3. **Wire in ComfyUI (`strength = 1.00`, `apply_to_pooled = False`):**
   - **Native ComfyUI encoders:** `Load CLIP` → **`Plato's Key Patch CLIP (H3 Plug & Play)`** → `MiniMax H3 Reference to Video` / `MiniMax H3 Image to Video`.
   - **Any HuggingFace model:** **`Plato's Key Load Any HF Text Encoder (H3)`** (`offload_to_cpu = True`) → `MiniMax H3 Reference to Video` / `MiniMax H3 Image to Video`.

---

## Visual Showcases

### Explainer & 3-Way Comparison
<video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_X_Demo_Final_thumb.jpg">
  <source src="https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_X_Demo_Final.mp4" type="video/mp4" />
</video>

### 4-Way Ref2VA & Pure T2VA Comparison
<video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_MultiLayer_4Way_Showcase_thumb.jpg">
  <source src="https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_MultiLayer_4Way_Showcase.mp4" type="video/mp4" />
</video>

### 10-Second 6-Way Action Stress Test (Qwen + Gemma 4 + Inkling-2-Mini)
<video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_10s_6Way_Action_Showcase_thumb.jpg">
  <source src="https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_10s_6Way_Action_Showcase.mp4" type="video/mp4" />
</video>

---

## License

Apache 2.0 — see [LICENSE](https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders/blob/main/LICENSE). Inspired by Stanford Hazy Research's [Plato's Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key).
