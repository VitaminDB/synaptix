# OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM

Источник: https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM · скачано 2026-11-28
Загрузок: 579945 · лайков: 434 · базовая: ["WarmBloodAban/Minimax-h3_Singularity","TenStrip/10Eros-Max"] · теги:  · обновлено: 2026-11-03T20:00:23.000Z

---

---
tags:
  - merge
  - minimax-h3
  - ref2va
base_model:
  - WarmBloodAban/Minimax-h3_Singularity
  - TenStrip/10Eros-Max
license: other
---

<p align="center">
  <img src="https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM/resolve/main/Merging_2_AI_models.jpg" alt="Merging 2 AI models" width="100%">
</p>

# Minimax-h3_Singularity-x-10Eros-Max_PRISM

Merge of two Minimax-H3 Ref2VA finetunes with a custom algorithm (**PRISM**, no mergekit):

* A: `WarmBloodAban/Minimax-h3_Singularity` / `Minimax-h3_Singularity_ref2va_v2_bf16.safetensors`
* B: `TenStrip/10Eros-Max` / `10Eros_Max_h3_8stepTURBO_beta6/7.safetensors` (8-step turbo)

Weights: `Minimax-h3_Singularity_x_10Eros-Max_PRISM_b_fp16.safetensors` (fp16). Licence follows the source models - check them before use.

## Method

1. **Spectral denoise** - D=B-A split by randomised SVD into structured part L (rank<=64) and diffuse remainder R; B' = A + L + gamma\*R with adaptive gamma in [0.92, 1].
2. **Polar geodesic** - per output row, direction by SLERP, magnitude by log-domain (geometric) interpolation.
3. **Sign-conflict mediation** - salient opposite-sign elements are pulled toward the weighted winner.
4. **Magnitude re-anchor** - rows rescaled to the geometric target norm.

# ⚠️ **UPD1:** Test of the new version: 10Eros_Max_h3_8stepTURBO_beta7_SingularityV2_int8HQ.safetensors
# ✨ **UPD2:** Test of the classic mix: 10Eros_Max_h3_TURBO-hybrid_beta5_Singularity_v1.3_int8.safetensors
# 💝 **UPD3:** Test of the PIKA 1.00-strength mix: 10Eros_Max_SingularityV2_PikaFlex_1.00_int8HQ.safetensors
# 💘 **UPD3:** Test of the PIKA 0.85-strength mix: 10Eros_Max_SingularityV2_PikaFlex_0.85_int8HQ.safetensors
# 💖 **UPD4:** Test of the PRISM+PIKA 0.85-strength mix: 10Eros_Max_SingularityV2_PRISM2_PikaFlex_0.85_int8HQ.safetensors
# ⚡ **UPD5:** The best out-of-the-box version for ultimate realistic crisp ultra-detailed skin textures is the PRISM-DMAD-PDMD Merge: use 8 steps, no Turbo Lora needed, Euler/Beta57, 12/3 Shift: PRISM_DMAD_PDMD_int8.safetensors
# ⚡ **UPD5.5:** Same merge but for low-VRAM GPUs - PRISM_DMAD_PDMD_w6a8_g32.safetensors

[🔗 My CivitAI Model Mirror](https://civitai.red/models/3019421/singularity-v2-x-10eros-max "Go to my CivitAI Mirror")

<div align="center">
  <video autoplay loop playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM/resolve/main/PRISM_DMAD_PDMD.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<div align="center">
  <video autoplay loop muted playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM/resolve/main/Pika0.85Sample.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<div align="center">
  <video autoplay loop muted playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM/resolve/main/TestMe.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<div align="center">
  <video autoplay loop muted playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM/resolve/main/TestOfBeta7.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
