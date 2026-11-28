# WarmBloodAban/Minimax-h3_Singularity

Источник: https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity · скачано 2026-11-28
Загрузок: 1125221 · лайков: 1028 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: comfyui · обновлено: 2026-10-23T15:20:56.000Z

---

---
license: apache-2.0
language:
- en
- zh
pipeline_tag: image-to-video
tags:
- video-generation
- text-to-video
- image-to-video
- video-to-video
- reference-to-video
- minimax-h3
- comfyui
- fine-tuned
- hdr
- singularity
- int8
base_model:
- MiniMaxAI/MiniMax-H3
---

# 🚀 V2 INT8 — Latest Update

**Minimax-h3_Singularity V2 INT8** builds on V1 with expanded action coverage, stronger prompt adherence, a fix for audio reference conditioning, and enhanced visual style rendering.

## ✨ What's New in V2

* ⚔️ **Expanded Action Coverage**: Further optimized on top of V1 to support a wider range of action scenes and more varied motion sequences, extending the model's versatility across dynamic video generation scenarios.
* 📝 **Stronger Prompt Adherence**: V2 follows prompt instructions more closely, improving control over subjects, actions, environments, and visual styles. We recommend keeping prompts within approximately **5,000 characters**, using clear, focused descriptions and avoiding excessive or repetitive instructions.
* 🎵 **Audio Reference Fix**: Fixed the audio reference conditioning issue. Audio references can now be used to help adjust the character's color tone.
* 🎨 **Enhanced Visual Styles**: Strengthened visual style rendering, with strong results across **photorealistic, East Asian realism, 3D CG, and cinematic CG** scenes. For best results, explicitly include the desired style keywords in your prompt.
* 📦 **INT8 Release**: The updated V2 model is available in **INT8 format**.

### 💡 Recommended Prompting for V2

Clearly describe the **subject, action, environment, camera movement, and intended visual style**. Keep instructions consistent and prioritize the details that matter most.

Include style keywords that match your desired result. Examples:

| Visual Style | Suggested Prompt Keywords |
| --- | --- |
| Photorealistic | `photorealistic, natural lighting, realistic textures` |
| East Asian Realism | `East Asian realism, realistic East Asian aesthetics, natural lighting` |
| 3D CG | `3D CG, detailed materials, stylized 3D rendering` |
| Cinematic CG | `cinematic CG, cinematic lighting, film-quality rendering` |

**Recommended prompt length: within approximately 5,000 characters.** This is a practical recommendation, not a target to fill. Shorter, well-structured prompts are welcome; avoid unnecessary details, repetition, and conflicting style instructions.

---

# Minimax-h3_Singularity

<p align="center">
  <a href="https://www.runninghub.ai/zh-cn/post/2112912294795993090/?inviteCode=rh-v1559">
    <img src="https://img.shields.io/badge/🚀_Online_Demo-RunningHub-blue.svg" alt="Online Demo">
  </a>
  <a href="https://youtube.com/@AIGC-Singularity">
    <img src="https://img.shields.io/badge/YouTube-AIGC--Singularity-red.svg" alt="YouTube">
  </a>
  <a href="https://space.bilibili.com">
    <img src="https://img.shields.io/badge/Bilibili-AIGC--Singularity_Space-00a1d6.svg" alt="Bilibili">
  </a>
</p>

## 📖 Model Overview

**Minimax-h3_Singularity** is a comprehensive fine-tuned fusion model designed to enhance the capabilities of **MiniMax-H3**. As a versatile **multimodal video generation model**, it supports **Text-to-Video (T2V)**, **Image-to-Video (I2V)**, **Reference-to-Video (Ref2V)**, and **Video-to-Video (V2V)** workflows within **ComfyUI**.

Built upon a strategic fusion of key checkpoints, including `ref`, `fl`, and `b25-49`, the model underwent extensive high-step fine-tuning. To preserve the original model's foundational strengths and broad generalization while addressing artifacts introduced during training, we spent **3 full days on precise model pruning and weight optimization**. The result is a clean, sharp, and highly dynamic video generation model.

The latest **V2 INT8** release builds on this foundation with broader action coverage, improved prompt adherence, corrected audio reference conditioning, and stronger visual style rendering.

---

## ✨ Key Improvements & Features

* 🎬 **HDR Image Quality & Blur Reduction**: Fine-tuned on high-dynamic-range (HDR) video datasets to enhance visual clarity and reduce unwanted motion blur during high-speed action.
* 👤 **Distant Face Restoration**: Reduces facial distortion, blurriness, and structural collapse in medium and long shots.
* 🎨 **Clean & Natural Aesthetic**: Reduces excessive skin shine and unnatural glossy textures for more natural lighting and realistic materials.
* ⚔️ **Enhanced Dynamic Motion**: Improves motion fluidity and physical impact, including complex sequences such as **sword fighting, martial arts, and melee combat**. V2 expands support to a wider range of action scenarios.
* 🌌 **VFX & Fantasy Effects**: Optimized for fantasy spellcasting, particle auras, and magical combat visual effects.
* 🎭 **Expressive Facial Dynamics**: Captures subtle facial expressions and emotional nuances more vividly.
* 📹 **Cinematography & Camera Control**: Improves responsiveness to camera instructions, including pans, tilts, zooms, and tracking shots.
* 📝 **Improved Prompt Control in V2**: Stronger adherence to prompt instructions enables more precise control over scene content and presentation.
* 🎵 **Corrected Audio Reference Conditioning in V2**: Resolves the audio reference issue and supports using audio references to help adjust the character's color tone.
* 🖼️ **Stronger Style Rendering in V2**: Delivers improved results in **photorealistic, East Asian realism, 3D CG, and cinematic CG** styles when the intended aesthetic is specified in the prompt.
* 🛡️ **Preserved Base Capabilities**: Designed to retain MiniMax-H3's core multimodal generation capabilities, style adaptability, and broad generalization.

---

## 🎬 Showcase

<video src="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/video/1.mp4" controls autoplay loop muted width="20%"></video>
<video src="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/video/3.mp4" controls autoplay loop muted width="20%"></video>
<video src="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/video/AIGCTYD2_00001_p84-audio_ganvc_1788634594%20(1).mp4" controls autoplay loop muted width="20%"></video>
<video src="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/video/AIGCTYD2_00010_p87-audio_alipa_1788652671.mp4" controls autoplay loop muted width="20%"></video>

---

## 💡 Usage Guide

### Multimodal Pipeline Support

The model supports the following workflows within **ComfyUI**:

* **Text-to-Video (T2V)**
* **Image-to-Video (I2V)**
* **Reference-to-Video (Ref2V)**
* **Video-to-Video (V2V)**

### 📝 Prompt Recommendations for V2

For best results:

* Keep prompts within approximately **5,000 characters**.
* Use clear descriptions of the subject, action, setting, lighting, and camera movement.
* Explicitly include the intended visual style, such as `photorealistic`, `East Asian realism`, `3D CG`, or `cinematic CG`.
* Avoid repetitive descriptions, unnecessary instructions, and conflicting visual styles.
* For action scenes, describe the movement and sequence of events clearly.

### 🎨 Example Style Prompts

**Photorealistic**

> Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking.

**East Asian Realism**

> East Asian realism, a traditional courtyard, realistic fabric textures, soft natural lighting, detailed architecture, fluid martial arts movements, cinematic composition.

**3D CG**

> 3D CG style, detailed character materials, expressive facial animation, fluid action, depth and dimensionality, polished 3D rendering.

**Cinematic CG**

> Cinematic CG style, dramatic cinematic lighting, detailed environments, dynamic action choreography, atmospheric effects, film-quality rendering.

### 🎵 Audio Reference

V2 fixes the audio reference conditioning issue. You can use audio references to help adjust the character's color tone as part of your generation workflow.

### 🚀 Recommended Acceleration LoRA

For high-speed generation, the recommended acceleration LoRA is:

* **`minimax_h3_ref2v_turbo_4step_v0.1`** — enables 4-step fast inference.

### 🌐 Online Interactive Demo

Test the model directly in your browser without a local GPU setup:

👉 [**Try it on RunningHub Workflows**](https://www.runninghub.ai/post/2096339589492432897/?inviteCode=rh-v1559)

---

## 🙏 Acknowledgements

Special thanks to the **MiniMax** open-source team for creating and releasing the powerful **MiniMax-H3** multimodal video model and providing a solid foundation for the open-source community! 🤝

---

## 🤝 Community & Commercial Inquiries

Connect with us for tutorials, community discussions, workflow sharing, or commercial collaborations:

* **YouTube Channel**: [AIGC-Singularity](https://youtube.com/@AIGC-Singularity)
* **Bilibili Channel**: AIGC-Singularity Space
* **QQ Group 1**: `1058747239` (Request to join)
* **QQ Group 2**: `1072010342` (Request to join)
* **Business Inquiries (WeChat)**: `aigctyd`
* **Email**: `a592991299@gmail.com`
