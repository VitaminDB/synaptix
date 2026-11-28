# lightx2v/Minimax-h3-Turbo

Источник: https://huggingface.co/lightx2v/Minimax-h3-Turbo · скачано 2026-11-28
Загрузок: 1835232 · лайков: 1252 · базовая: MiniMaxAI/MiniMax-H3 · теги: distillation,lora · обновлено: 2026-11-24T07:04:24.000Z

---

---
base_model: MiniMaxAI/MiniMax-H3
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
library_name: minimax-h3
tags:
  - video-generation
  - audio-video-generation
  - distillation
  - lora
  - lightx2v
---

# MiniMax-H3 Turbo

> 🌐 **[Create online with LightX2V Studio](https://x2v.light-ai.top/)** — generate videos with audio from text, images, or reference materials. No local installation or GPU setup required.

For local inference and reproducibility, see the [MiniMax-H3 Turbo GitHub repository](https://github.com/ModelTC/Minimax-H3-Turbo), [LightX2V MiniMax-H3 examples](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README.md), and [model specifications](https://github.com/ModelTC/Minimax-H3-Turbo#model-specs).

## Online App: LightX2V Studio

[**LightX2V Studio**](https://x2v.light-ai.top/) is an online AI avatar and multimodal content production platform built on the LightX2V video generation inference framework. With AI avatar video creation at its core, it integrates video, image, and audio creation capabilities and supports workflow orchestration and reuse.

### MiniMax-H3 Online Tasks

- **Text-to-Audio/Video:** Generate video with audio from text prompts.
- **Image-to-Audio/Video:** Generate video with audio from an input image.
- **Omni-Reference Audio/Video:** Generate video with audio guided by reference materials.

### Quick Start

1. Open [LightX2V Studio](https://x2v.light-ai.top/), sign in, and select a MiniMax-H3 task.
2. Enter a prompt, upload any required materials, and configure the generation settings.
3. Review the settings and credit cost, then submit your request.
4. Preview and download the generated video.

### Studio Preview

![LightX2V Studio online creation interface](https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/SjpJ1h2wuwqZchY18D_8Z.png)

*LightX2V Studio — MiniMax-H3 Online Video Creation*

## Online API: LightX2V Studio

Integrate MiniMax-H3 audio/video generation into your applications and workflows through the **LightX2V Studio API**.

- [API & Skills Documentation](https://x2v.light-ai.top/api-docs)
- [Pricing](https://x2v.light-ai.top/pricing)

Online service usage follows Studio's pricing and terms. Model assets are governed by the repository license.

## Video Comparison

We selected cases where MiniMax-H3 Turbo v1.1 struggled and re-evaluated them with **[⭐ MiniMax-H3 Turbo v1.2 (8 steps)](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors)**. Results from MiniMax-H3 (29 steps), Pika LoRA (6 steps), and v1.1 (8 steps) are shown alongside for comparison.

<table>
  <thead>
    <tr>
      <th width="25%">MiniMax-H3<br>29 steps</th>
      <th width="25%">Pika LoRA<br>6 steps</th>
      <th width="25%">MiniMax-H3 Turbo v1.1 (Ours)<br>8 steps</th>
      <th width="25%">
        <strong>⭐ <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors">MiniMax-H3 Turbo v1.2</a> (Ours)<br>8 steps</strong>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="4"><b>Example 1 — Prompt:</b> integrated_multimodal_description: [Shot 1] A wide lens looks down a rose-red railing beside broad concrete steps partly covered by runoff in a leafy urban park. The camera glides toward the steps, then tilts up to a vigorous artificial-looking but natural-rock-lined cascade descending several short ledges. Behind the cascade, pedestrians pause on a footbridge in waterproof coats, some holding vivid umbrellas against a passing spring shower. overall_soundscape: Rushing stepped water, splashes against concrete edges, light spring rain, softened city ambience far beyond the park, bridge footsteps, and faint distant conversation. non_diegetic_music: N/A</td>
    </tr>
    <tr>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/hJnCjTyk446eSqrxnlWxH.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/hMoNaggEJTYJn_JMBkaDp.mp4" controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/lJ15Eg6srZPdk_Bl5-m_q.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/8SjPoDxADxS9WKqqTqcqi.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
    </tr>
    <tr>
      <td colspan="4"><b>Example 2 — Prompt:</b> integrated_multimodal_description: [Shot 1] In a kitchen close-up, a hand picks up a small green cooking pot from the counter and carefully sets it on the stovetop space between two gas burners. The pot lands centered and stable; the burners remain unlit. Warm indoor light, realistic countertop and metal stove details. overall_soundscape: Soft kitchen room tone, a faint clink as the pot is lifted, and a gentle ceramic-on-metal placement sound. non_diegetic_music: N/A</td>
    </tr>
    <tr>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/ZBkGV-djSKWkLktO3XsdO.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/dXReRVzmfaLm7XIUv5wEM.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/pVzcOtGdstA98dVi46LyG.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/sM-yY3R4a-IyKkDW5vC3h.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
    </tr>
    <tr>
      <td colspan="4"><b>Example 3 — Prompt:</b> integrated_multimodal_description: [Shot 1] Close-up of a stovetop burner with a red fork resting on the right side. A hand gently slides the fork across the cooktop and places it on the left side of the burner, then withdraws, leaving the fork still and clearly visible. overall_soundscape: Soft kitchen room tone, a faint scrape as the fork moves across the cooktop, and a subtle clink when it settles. non_diegetic_music: N/A</td>
    </tr>
    <tr>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/LnGiwrcFNNNFWCp20Vonm.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/GFDanv1BWM9O8oRnpzkQm.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/qwh4Ilmd6KKAJuqD6vWpP.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
      <td>
        <video src=https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/fUjOa8TTAqpLcKyTCq-Ja.mp4 controls muted playsinline preload="metadata" width="100%"></video>
      </td>
    </tr>
  </tbody>
</table>

In these selected examples, **MiniMax-H3 Turbo v1.2** delivers better prompt adherence and visual clarity than Pika LoRA and v1.1, producing results closer to the **29-step base model with just 8 steps**.
