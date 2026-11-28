# moe-kill/H3-Turbo-Neat

Источник: https://huggingface.co/moe-kill/H3-Turbo-Neat · скачано 2026-11-28
Загрузок: 3101 · лайков: 32 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-10-24T16:39:12.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/moe-kill/H3-Turbo-Neat/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
- lora
- minimax-h3
- video-generation
- image-generation
- natural-look
language:
- en
- zh
---

# H3 Turbo Neat

A practical family of low-step MiniMax H3 LoRAs for both video and still-image generation, focused on image quality, motion, acting, prompt stability, audio, and real-world usability.

H3 Turbo Neat currently has two branches:

- **Neat Classic** — lightweight, expressive, character-driven, and camera-oriented.
- **Neat Prime** — a reorganized **Neat × DMAD** branch focused on higher overall production quality.

> **Important:** Neat Classic and Neat Prime use different workflows and sampling setups.  
> Do not mix their sampler nodes or recommended settings.

---

# 中文说明

## 版本选择

| 版本 | 定位 | 推荐步数 | 工作流 |
|---|---|---:|---|
| **Neat Classic** | 轻量、短剧表演、人物互动、镜头个性 | 4-step | Classic 专属工作流 |
| **Neat Prime** | 综合画质、动态完整性、综合色调、音频、正式生产 | **7-step 推荐** | **DMAD re-noise 工作流** |
| **Neat Prime Image Mode** | 高质量静帧、参考图、角色卡、Lookdev | **10-step 推荐（3-step 预览）** | **DMAD re-noise Image workflow** |


## 成片效果 / 30 组视频盲测

想直接看实际生成效果，可以观看这期 30 组视频盲测：

**[你能看出来吗？H3加速LoRA，30组视频盲测](https://www.bilibili.com/video/BV1XFpt6xEXy/)**

包含多组真实生成案例，适合直观看 Neat 系列在画质、动态、人物表现和整体成片效果上的实际表现。

---

## Neat Classic

Neat Classic 是最初的轻量版本。

它的价值并不只是“旧版”或“更小的版本”。Classic 本身拥有非常鲜明的镜头语言与画面个性，在人物表演、短剧互动、情绪场景、对白镜头和镜头组织方面尤其有特色。

它经常呈现出一种更自由、更有导演感的表现方式，人物状态更活，镜头变化也更有个性。部分用户在角色展示、Ref2VA、短剧情境和人物特写中也获得了非常有辨识度的效果。

### Classic 的主要特点

- 人物表演思维活跃
- 适合短剧、对白、人物互动和情绪场景
- 镜头感更自由、更有个性
- 画面具有鲜明风格与辨识度
- 体积较小，适合快速 4-step 生成

### Classic 的音频

Classic 的音频表现相对 Prime 较弱，部分场景会带有较明显的电子感，但这并不影响它在人物表演、镜头语言和独特画面风格上的使用价值。

### Classic 文件

`H3Turbo_4Step_Neat_v1.safetensors`

请使用 Classic 对应的原工作流。

详见：[NEAT_CLASSIC.md](./NEAT_CLASSIC.md)

---

## Neat Prime

**Neat Prime = Neat × DMAD 重组版。**

Prime 不是简单叠加，而是对 Neat 与 DMAD 的能力重新组织，并围绕 DMAD re-noise 采样方式重新调校。

它的目标不是只追求更少步数，而是在低步数下同时改善画质、动态、结构完整性、音频、提示词响应和整体成片感。

### 视觉表现

相较 Neat Classic，Prime 拥有：

- 更成熟的综合色调
- 更现代的视觉风格
- 更通透的色彩
- 更完整的高光、暗部与中间调关系
- 更自然的肤色、环境色和材质色分离

相较 DMAD，Prime 在保留画面干净、不油腻特点的同时：

- 色彩更通透
- 灰感更少
- 局部层次更鲜明
- 材质关系更清楚

Prime 的综合色彩并不是简单的 LUT 风格变化，而更像是在生成过程中形成了更成熟的色彩与材质关系。

### 动态与动作

Prime 重点改善了 DMAD 在部分场景中可能出现的：

- 动作缺失
- 动态偏弱
- 快速运动不完整
- 高速动作容易花或失去冲击力

在 7-step 下，Prime 可以保持较强的快速动作清晰度、肢体完整性和打击感。打斗、碰撞等场景更容易保持“拳拳到肉”的感觉。

### 音频

Prime 的音频是这一版本的重要提升之一。

相较 Neat Classic：

- 电子感明显减少
- 声音更自然
- 细节和层次更完整

相较 MiniMax H3 原版 20-step，在我们的测试中 Prime 往往表现出：

- 更饱满的输出
- 更明显的动态起伏
- 更强的瞬态冲击
- 更有力度的动作音效
- 更明显的左右立体声声场
- 更强的空间层次

这些差异在打斗、碰撞、环境声和复杂声场中尤其明显。

### 提示词响应与非预期字幕

低步数 H3 加速 LoRA 普遍更容易出现非预期文字、字幕或提示词响应不稳定的问题。

Prime 建议配合 **H3 Skill 标准六段式提示词** 使用。

在我们的测试中，使用标准六段式提示词可以显著提高提示词响应稳定性，并有效避免非预期字幕生成。

### 推荐设置

- **Sampler:** DMAD re-noise
- **Steps:** **7-step 推荐**
- **Video Shift:** 12
- **Audio Shift:** 2
- **Prompt:** H3 Skill 标准六段式提示词

4-step 仍可用于快速预览或动作测试，但 7-step 是当前推荐的质量 / 速度平衡档。


### Prime Image Mode：高质量生图 / 试镜模式

Neat Prime 不仅可以用于视频生成，也可以直接作为 H3 的高质量静帧生成器。

它最大的实用价值之一，是可以让 **H3 自己为 H3 制作参考图**：先在相同模型审美空间中完成角色、服装、灯光、场景与镜头设计，再将参考图直接回喂给 H3 做视频。相比使用外部生图模型，这种同源参考图与 H3 的色彩、人物塑形和材质逻辑天然一致，能够显著减少风格不匹配和参考图“进视频后变味”的问题。

提供的 Image Mode 工作流：

`H3_Neat_Prime_Image_3Step_5F_Renoise.json`

该工作流可以使用低步数进行极速试镜，也可以提高步数用于最终高质量生图。

### H3 Neat ImageFusion 自定义节点

- GitHub: https://github.com/bt420000-collab/ComfyUI-H3-Neat-ImageFusion

H3 Neat ImageFusion 提供多帧图像融合功能，用于 H3 Neat 图片工作流。

#### 推荐生图参数

**最终高质量生图：**

- **Steps:** **10（推荐）**
- **Sampler:** DMAD re-noise
- **Resolution:** **16:9 / 3.5 MP / 2560 × 1440**
- **Video Shift:** 12
- **Audio Shift:** 2
- **Output:** 5-frame image burst

**极速预览 / Lookdev：**

- **3 steps**
- 其它参数保持一致

3-step 适合快速寻找角色、构图、服装和灯光方向；10-step 推荐用于最终参考图、角色卡和高质量静帧输出。

#### 推荐用途

- 角色定妆与角色卡
- H3 视频参考图制作
- Lookdev / 美术方向探索
- 分镜与机位预演
- 服装、灯光、场景试镜
- 提示词快速预览
- 大批量候选图筛选

#### H3 自参考工作流

1. 使用 Prime Image Mode 生成 5 张候选静帧
2. 选择最佳图片，或使用 **H3 NEAT ImageFusion** 做多图融合精修
3. 将最终参考图直接回喂给 H3 生成视频

这样可以在不依赖其它生图模型的情况下完成从参考图到视频的一体化 H3 工作流。

#### 生图质量参考

下面这张图片可作为 Prime Image Mode + ImageFusion 的实际生成质量参考：

![H3 Neat Prime Image Mode quality reference](./H3_IMAGE_FUSED_00009_.png)

下面这张图可以作为另一张 Prime Image Mode Demo 图参考：

![H3 Neat Prime Image Mode demo](./H3_NEAT_IMAGE_00013_.png)

#### 三视图 / 角色设定参考图

Prime Image Mode 还可以直接生成适合后续视频制作的 **角色三视图参考图**。  
下面这张 Demo 展示了同一角色的正面、侧面和背面设定，可用于角色造型确认、服装结构检查、发型设计以及后续 H3 参考图生视频。

![H3 Neat Prime three-view character reference](./H3_IMAGE_FUSED_00007_.png)

这种三视图由 H3 自己生成，因此人物审美、服装材质、综合色调和后续视频生成天然处于同一套模型分布中，更适合作为 H3 自参考工作流的一部分。



> Image Mode 是独立的高质量静帧 / 视觉预演工作流，不替代 7-step Prime 正式视频工作流。


### 盲测说明

在内部盲测中，Prime 7-step 在部分场景的视觉质量可以达到或超过原版 H3 20-step，但这并不代表所有场景都全面优于原版。对于极端提示词忠实度或特殊高风险镜头，原版 20-step 仍然是重要参考。

### Prime 文件

`H3_Neat_Prime_v0.1.safetensors`

请使用 Prime 专属 DMAD re-noise 工作流。

详见：[NEAT_PRIME.md](./NEAT_PRIME.md)

---

## Classic 与 Prime 的区别

| 项目 | Neat Classic | Neat Prime |
|---|---|---|
| 定位 | 轻量、短剧、人物表演、镜头个性 | 综合画质、动态、声音、正式生产 |
| 推荐步数 | 4-step | **7-step** |
| 工作流 | Classic workflow | **DMAD re-noise workflow** |
| 人物表演 | 更活、更有短剧感 | 更稳、更完整 |
| 镜头语言 | 更自由、更有个性 | 更成熟、更偏成片型 |
| 画面 | 风格鲜明、镜头感强、画质有特色 | 更稳定、更通透、综合色调更成熟 |
| 动态 | 强 | 更完整、更稳定 |
| 音频 | 稍弱，偶有电子感 | 更自然、更饱满、声场更宽 |
| 提示词稳定性 | 一般 | 更高，推荐六段式提示词 |
| 文件体积 | 较小 | 较大 |

### 一句话选择

**Choose Classic for acting and camera personality.**  
**Choose Prime for overall production quality.**

---

# English

## Which version should I use?

| Version | Focus | Recommended steps | Workflow |
|---|---|---:|---|
| **Neat Classic** | Lightweight, acting, short drama, camera personality | 4-step | Classic workflow |
| **Neat Prime** | Overall image quality, motion, color, audio, production use | **7-step recommended** | **DMAD re-noise workflow** |
| **Neat Prime Image Mode** | High-quality stills, references, character sheets, lookdev | **10-step recommended (3-step preview)** | **DMAD re-noise image workflow** |


## Video Demo / 30-Scene Blind Test

For a direct look at real generated results, watch the 30-scene blind test:

**[你能看出来吗？H3加速LoRA，30组视频盲测](https://www.bilibili.com/video/BV1XFpt6xEXy/)**

It includes a wide range of real generation samples and is a useful reference for image quality, motion, character performance, and overall production results.

---

## Neat Classic

Neat Classic is the original lightweight release.

It should not be treated simply as an outdated version. Classic has a very distinctive visual identity and camera language, and it remains especially strong for character acting, short-form drama, dialogue scenes, emotional interaction, and expressive shot composition.

Its images often feel more spontaneous and director-driven, with lively performance and more individual camera behavior. Some users have also reported particularly strong results for character presentation, Ref2VA, short-form drama, and cinematic character shots.

### Classic strengths

- expressive character acting
- strong short-drama and dialogue behavior
- spontaneous interpersonal interaction
- distinctive camera personality
- recognizable visual character
- lightweight 4-step generation

### Classic audio

Compared with Prime, Classic has weaker audio quality and can sometimes sound slightly electronic. Its main value remains its acting behavior, camera personality, and distinctive image character.

### Classic file

`H3Turbo_4Step_Neat_v1.safetensors`

Use the Classic workflow shipped for this branch.

See: [NEAT_CLASSIC.md](./NEAT_CLASSIC.md)

## Neat Prime

**Neat Prime is a reorganized Neat × DMAD variant.**

Prime is not a simple LoRA stack. It reorganizes capabilities from Neat and DMAD and is tuned around the DMAD re-noise sampling workflow.

The goal is not only fewer sampling steps, but stronger overall production quality at low NFE.

### Visual quality

Compared with Neat Classic, Prime provides:

- a more mature overall color balance
- a more modern visual style
- clearer and more transparent color
- stronger highlight / shadow / midtone relationships
- more natural separation between skin, environment, and material colors

Compared with DMAD, Prime keeps the clean, non-greasy rendering character while providing:

- clearer color
- less gray-looking output
- stronger local separation
- more defined material relationships

The difference is closer to a more mature color and material relationship being formed during generation, rather than a simple LUT-like color shift.

### Motion and action

Prime improves several motion issues that can appear in low-step DMAD generation:

- missing actions
- weak dynamics
- incomplete fast motion
- motion degradation during high-speed scenes

At 7 steps, Prime can retain strong fast-motion clarity, better action completeness, and stronger impact. Combat and collision scenes are more likely to preserve a convincing sense of physical force.

### Audio

Audio is one of the major improvements in Prime.

Compared with Neat Classic:

- less electronic coloration
- more natural sound
- better detail and layering

Compared with the original MiniMax H3 20-step output, our tests often show:

- fuller output
- stronger perceived dynamics
- stronger transient impact
- more powerful action sound effects
- a more obvious left-right stereo image
- stronger spatial layering

These differences are especially noticeable in combat, impacts, environmental sound, and complex sound scenes.

### Prompt response and unwanted subtitles

Low-step H3 acceleration LoRAs can sometimes produce unwanted text, subtitles, or unstable prompt responses.

For Prime, we strongly recommend using the **H3 Skill standard six-part prompt format**.

In our testing, properly structured six-part prompts significantly improve prompt-response stability and can effectively prevent unwanted subtitle generation.

### Recommended settings

- **Sampler:** DMAD re-noise
- **Steps:** **7-step recommended**
- **Video Shift:** 12
- **Audio Shift:** 2
- **Prompt:** H3 Skill standard six-part prompt format

4-step generation can still be used for fast previews or action tests, while 7-step is currently recommended as the best quality / speed balance.


### Prime Image Mode: High-Quality Still Image / Lookdev

Neat Prime can also be used directly as a high-quality H3 still-image generator.

One of the most useful advantages of Image Mode is that **H3 can generate its own reference images for H3 video generation**. Characters, costumes, lighting, environments, and shot ideas can all be developed inside the same H3 aesthetic space before the selected reference is fed back into H3 for video generation.

Compared with using an unrelated external image model, this self-referencing workflow keeps color, character design, material response, and overall visual logic naturally aligned with H3 and greatly reduces style mismatch.

Included Image Mode workflow:

`H3_Neat_Prime_Image_3Step_5F_Renoise.json`

The workflow can run at very low steps for rapid visual exploration, or at higher steps for final-quality still images.

#### Recommended Image Settings

**Final high-quality stills:**

- **Steps:** **10 (recommended)**
- **Sampler:** DMAD re-noise
- **Resolution:** **16:9 / 3.5 MP / 2560 × 1440**
- **Video Shift:** 12
- **Audio Shift:** 2
- **Output:** 5-frame image burst

**Fast preview / lookdev:**

- **3 steps**
- Keep the other settings unchanged

3-step is ideal for quickly exploring characters, framing, costume, and lighting. 10-step is recommended for final reference images, character sheets, and higher-quality still output.

#### Recommended Uses

- character look development
- character sheets
- reference images for H3 video generation
- shot ideation and framing
- costume / lighting exploration
- prompt preview
- rapid visual auditioning

#### Self-Referencing H3 Workflow

1. Generate a 5-frame still-image burst with Prime Image Mode
2. Select the best result, or refine multiple candidates with **H3 NEAT ImageFusion**
3. Feed the final reference image directly back into H3 video generation

This creates a complete H3-native reference-to-video workflow without requiring a separate external image model.

#### Image Quality Reference

The following image is a practical quality reference from the Prime Image Mode + ImageFusion pipeline:

![H3 Neat Prime Image Mode quality reference](./H3_IMAGE_FUSED_00009_.png)

The following image can be used as an additional Prime Image Mode demo reference:

![H3 Neat Prime Image Mode demo](./H3_NEAT_IMAGE_00013_.png)

#### Three-View / Character Reference Sheet

Prime Image Mode can also generate **three-view character reference sheets** for later H3 video production.  
The demo below shows front, side, and back views of the same character, useful for character design, costume structure, hairstyle validation, and reference-to-video workflows.

![H3 Neat Prime three-view character reference](./H3_IMAGE_FUSED_00007_.png)

Because the reference sheet is generated by H3 itself, character styling, costume materials, color balance, and the downstream video model remain naturally aligned inside the same model distribution.



> Image Mode is a dedicated high-quality still-image / look-development workflow. It does not replace the 7-step Prime production video workflow.


### Blind A/B note

In internal blind comparisons, Prime 7-step can match or outperform the original H3 20-step in some scenes. This does not mean Prime is universally better in every scenario. Original 20-step remains an important reference for edge cases and maximum prompt fidelity.

### Prime file

`H3_Neat_Prime_v0.1.safetensors`

Use the dedicated Prime DMAD re-noise workflow.

See: [NEAT_PRIME.md](./NEAT_PRIME.md)

## Compatibility

Neat Classic and Neat Prime are **not** drop-in replacements for each other.

They use different sampler setups and should be treated as separate runtime configurations.

- Neat Classic → use the Classic workflow
- Neat Prime → use the Prime DMAD re-noise workflow

Do not mix sampler nodes or recommended settings between the two branches.

## Credits

All credit for training the MiniMax H3 base model and upstream LoRAs / distillation models belongs to their original authors.

See [CREDITS.md](./CREDITS.md).

## License

This project is a modified derivative of upstream MiniMax H3 community work.

Upstream licenses, model licenses, notices, and applicable restrictions remain in effect.

See `LICENSE`, `LICENSE_APACHE_2.0.txt`, and `NOTICE`.
