# OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD

Источник: https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD · скачано 2026-11-28
Загрузок: 32154 · лайков: 25 · базовая: ["Comfy-Org/MiniMax-H3","pdmd2026/pdmd_4NFE_lora","ZhengmingYu/DMAD"] · теги: comfyui,quantized,distilled,base_model:Comfy-Org/MiniMax-H3,base_model:finetune:Comfy-Org/MiniMax-H3 · обновлено: 2026-11-04T08:24:26.000Z

---

---
language:
  - en
license: other
license_name: minimax-license
license_link: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/LICENSE
tags:
  - text-to-video
  - image-to-video
  - video-generation
  - minimax-h3
  - comfyui
  - quantized
  - int8
  - w6a8
  - convrot
  - dmad
  - pdmd
  - distilled
  - accelerated
base_model:
  - Comfy-Org/MiniMax-H3
  - pdmd2026/pdmd_4NFE_lora
  - ZhengmingYu/DMAD
pipeline_tag: text-to-video
---

# 🎞️ Minimax-H3-Hybrid-DMAD-PDMD

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Card-blue?style=for-the-badge)](https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD)
[![Base Model](https://img.shields.io/badge/Base-MiniMax--H3%20(33B)-orange?style=for-the-badge)](https://huggingface.co/Comfy-Org/MiniMax-H3)
[![Distillation](https://img.shields.io/badge/Distillation-DMAD%20%2B%20PDMD-green?style=for-the-badge)](https://huggingface.co/pdmd2026/pdmd_4NFE_lora)
[![ComfyUI Native](https://img.shields.io/badge/Engine-ComfyUI%20Native-7057ff?style=for-the-badge)](https://github.com/comfyanonymous/ComfyUI)

### *The High-Speed, Low-VRAM 33B Foundation Model for MiniMax-H3 Video Generation*

**4–8 Step Accelerated Inference** • **Smooth S-Curve Hybrid Backbone** • **Native INT8 & W6A8 Quantization**

---

</div>

<div align="center">
  <video autoplay loop playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD/resolve/main/Merge_Sample.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<div align="center">
  <video autoplay loop muted playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD/resolve/main/MergeSample.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

**Minimax-H3-Hybrid-DMAD-PDMD** is an accelerated, multimodal reference-capable 33B video-generation foundation model engineered natively for **ComfyUI**.

It resolves the trade-off in the open-weights ecosystem between MiniMax-H3's **FL2VA** (high dynamic range, natural motion, but missing multi-reference conditioning) and **Ref2VA** (reference following, but prone to crushed contrast and dark shadow banding). By executing a **Smooth S-Curve Hybrid AdaLN Merge**, permanently fusing two distillation LoRAs (**0.85× DMAD** + **0.333× PDMD**), and applying low-rank SVD pruning, this model runs generation workflows in **4 to 8 steps** on consumer GPUs.

---

## ?? Model Releases & Formats

Available in two precision tiers tailored for consumer hardware:

| File Name | Format | Precision | File Size | Recommended Hardware | Minimum VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`Minimax-H3-Hybrid-DMAD-PDMD_w6a8_g32.safetensors`](https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD/resolve/main/Minimax-H3-Hybrid-DMAD-PDMD_w6a8_g32.safetensors) | **W6A8_G32** | 6-bit packed weights + FP8 E4M3 scale | **~14.95 GB** | **RTX 3090 / 4080 / 4090**, Apple M-Series (32GB+) | **16 GB** |
| [`Minimax-H3-Hybrid-DMAD-PDMD_int8.safetensors`](https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD/resolve/main/Minimax-H3-Hybrid-DMAD-PDMD_int8.safetensors) | **INT8 ConvRot** | 8-bit symmetric tensorwise | **~18.89 GB** | **RTX 3090 / 4090 / A5000 / A6000**, Apple M-Series (64GB+) | **24 GB** (Fits 100% in VRAM) |
| *Baseline Dense (Unpruned)* | *BF16* | *16-bit Float* | *~61.70 GB* | *A100 (80GB) / H100* | *>64 GB* |

> [!TIP]
> **Which one should you pick?**
> * **Choose W6A8_G32 (14.95 GB)** if you want maximum VRAM savings and run on a 16 GB card (like an RTX 4080 or RTX 4070 Ti Super 16GB). It uses `comfy-kitchen`'s sub-byte bitpacking to reduce weight size with near-lossless reconstruction.
> * **Choose INT8 ConvRot (18.89 GB)** if you have a 24 GB card (RTX 3090 / 4090) and want standard INT8 execution.

---

## ??? Pipeline Architecture

```text
+-----------------------------------+    +-----------------------------------+
¦     FL2VA Backbone (61.7 GB)      ¦    ¦     Ref2VA Guidance (61.7 GB)     ¦
¦   (Spatial Attention, MLPs, Norm) ¦    ¦     (Multimodal AdaLN Routing)    ¦
+-----------------------------------+    +-----------------------------------+
                  ¦                                        ¦
                  +----------------------------------------+
                                      ?
                      +-------------------------------+
                      ¦  Smooth S-Curve Hybrid Merge  ¦
                      ¦  • Blocks 0–19:  100% FL2VA   ¦
                      ¦  • Blocks 20–34: Smoothstep S ¦
                      ¦  • Blocks 35–49: 100% Ref2VA  ¦
                      +-------------------------------+
                                      ?
                      +-------------------------------+
                      ¦  Dual-Distillation LoRA Merge ¦
                      ¦  • + 0.850 × DMAD (8-Step)    ¦
                      ¦  • + 0.333 × PDMD (4-Step v6) ¦
                      +-------------------------------+
                                      ?
                      +-------------------------------+
                      ¦ Low-Rank Affine Time Pruning  ¦
                      ¦ (Dense 2688-D --? Rank-8 +    ¦
                      ¦  folded bias + adaln_t_table) ¦
                      +-------------------------------+
                                      ?
                   +-------------------------------------+
                   ?                                     ?
      +-------------------------+           +-------------------------+
      ¦  INT8 ConvRot Compiler  ¦           ¦   W6A8_G32 Sub-Byte     ¦
      ¦  • Normalized H_256 Rot ¦           ¦   • Normalized H_256    ¦
      ¦  • FP32 Channel Scale   ¦           ¦   • Uniform 6-bit codes ¦
      ¦  • Symmetric INT8       ¦           ¦   • FP8 E4M3 Rel Scale  ¦
      ¦    [18.89 GB]           ¦           ¦    [14.95 GB]           ¦
      +-------------------------+           +-------------------------+
```

---

## ?? Technical Highlights

### 1. Dual-Distillation Superposition (4–8 Steps)
* **DMAD (Distribution Matching Adversarial Distillation)** at **0.85 strength**: Preserves high-frequency spatial textures, eye/skin micro-details, and motion sharpness.
* **PDMD (Projected Distribution Matching Distillation v6)** at **0.333 strength**: Eliminates student–critic endpoint residuals and straightens flow trajectories down to 4 NFE.
* **SwiGLU Vector Re-alignment**: The PDMD adapter was integrated using native ComfyUI layout mapping ($[value; gate] 	o [gate; up]$ row ordering).

### 2. Smooth S-Curve Hybrid AdaLN Injection
* **Blocks 0–19 (100% FL2VA)**: Guarantees balanced exposure, uncrushed shadow dynamics, and stable background lighting.
* **Blocks 20–34 (Cubic Smoothstep Transition)**: Blends between generation and reference pathways without latent shock:
  $$S(t) = t^2(3 - 2t), \quad t \in [0, 1]$$
* **Blocks 35–49 (100% Ref2VA)**: Anchors identity consistency, multimodal character cues, and reference image guidance.
* **Final Layer**: 50/50 balance protecting the output color gamut while locking subject identity.

### 3. Native Quantization Options
* **INT8 ConvRot**: Multiplies weights by normalized Sylvester matrices ($H_{256}$) prior to quantization, diffusing activation spikes and fitting entirely in 24 GB VRAM.
* **W6A8_G32 Layout**: Quantizes rotated weights to uniform 6-bit codes bit-packed 4-into-3 bytes (`torch.int8`), backed by per-group relative scales in `torch.float8_e4m3fn` and per-channel FP32 scales via `comfy-kitchen`'s `AsymW4A8Int8Layout`.

---

## ?? Mathematical Formulations

### 1. Adaptive S-Curve Block Merging

For each transformer block $i \in \{0, \dots, 49\}$ and its AdaLN modulation projection $W^{(i)}$:

$$
lpha(i) = egin{cases} 
0.0 & 	ext{if } i < 20 \
t^2(3 - 2t), \quad 	ext{where } t = rac{i - 20}{35 - 20} & 	ext{if } 20 \le i < 35 \
1.0 & 	ext{if } i \ge 35 
nd{cases}
$$

$$
W^{(i)}_{	ext{hybrid}} = (1 - lpha(i)) 