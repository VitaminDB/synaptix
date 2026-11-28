# Jojocodex/minimax-h3-spatial-physics-lora

Источник: https://huggingface.co/Jojocodex/minimax-h3-spatial-physics-lora · скачано 2026-11-28
Загрузок: 17927 · лайков: 143 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-19T21:42:56.000Z

---

---
license: apache-2.0
base_model: Comfy-Org/MiniMax-H3
tags:
- minimax-h3
- lora
- text-to-video
- physics
- spatial-reasoning
- object-motion
- video-generation
---

# 空间物理 LoRA · Spatial & Physics LoRA (MiniMax-H3)

空间思维与物理逻辑 LoRA，基于 **MiniMax-H3**（Comfy-Org/MiniMax-H3）训练，让模型学会**纯物体**的空间关系与物理运动（碰撞、堆叠、掉落、遮挡等）。

> 本 LoRA 专注**物体物理**。

目前还是训练和测试阶段，一些素材片段以及打标问题导致LORA还不是特别稳定。下一阶段准备修复后重新训练。目前LORA也是可以使用，强度建议0.3 ~ 0.5。纯属是在原模型基础上稍微增强一点物理反馈。
最新是重新训练到了wushu_spatial_physics_clean_3000_pruned.safetensors 版本。

## 视频对比
实测用了这个LORA，视频整体提升真实感物理的逻辑，比如物体碰撞的真实反馈。也可以用于一些打斗场景，人物真实碰撞的效果。

**用了空间物理LORA的武打片段（强度 0.3）**

<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/f6BFLovWOi5kxKX96rrzs.mp4" controls width="40%"></video>

**没有用空间物理LORA的武打片段（同样提示词）**

<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/AaCiljUOQUkzsOAdcMdsn.mp4" controls width="40%"></video>

## 其他测试视频 （wushu_spatial_physics_v2_3000.safetensors）

<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/iuQzNUb4YALFsDGilD4uI.mp4" controls width="40%"></video> <video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/cWdxixHdowNF690Zo-1Bd.mp4" controls width="40%"></video>
<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/II8HI8f7TIPjJmILXL8XD.mp4" controls width="40%"></video> <video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/qLpKJouZtbgoT-Y764PWt.mp4" controls width="40%"></video>
<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/gMSayhbcjLSvAvI2wMl0d.mp4" controls width="40%"></video> <video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/qqyzWY6v0yqqW0x2FspGC.mp4" controls width="40%"></video>


## 使用方法（ComfyUI）

1. 下载 `*.safetensors`，放入 ComfyUI `models/loras/`
2. LoraLoader 加载，strength 建议 0.8~1.0
3. 用空间/物理语言 prompt 描述物体运动

### 关键词库（从训练数据提炼）

**① 主体/场景**
- `several colored objects` 多个彩色物体（CLEVRER 风格）
- `rigid objects` 刚体 / `elastic objects` 弹性物体
- `metal objects` 金属物体
- `a ball` / `balls` 球
- `billiard balls` 台球
- `objects` 通用物体
- `on a table` 桌面上（PhyCo 台球场景）
- `in a synthetic scene` 合成场景（CLEVRER 风格）

**② 物理行为**
- `collision` / `colliding` 碰撞
- `a single collision between objects` 单次碰撞
- `bouncing` / `rebound` 弹跳/回弹
- `with visible impact and bounce` 可见撞击与反弹
- `with minimal bounce` 小幅度回弹
- `rolling into a wall` 滚向墙
- `scattering` 散射 / 四散
- `momentum transfer` 动量传递
- `in motion` 运动中的

**③ 空间一致性短语（关键！）**
- `clear spatial layout` 清晰的空间布局
- `visible depth` 可见的纵深
- `consistent trajectories` 一致的轨迹
- `gravity-consistent` / `gravity-consistent movement` 重力一致的运动
- `gravity-consistent rebound trajectories` 重力一致的回弹轨迹
- `objects maintaining spatial relationships` 物体保持空间关系
- `realistic restitution` 真实的恢复系数（回弹性）
- `rigid collision with momentum transfer` 带动量传递的刚性碰撞

### 负面提示词建议

训练时 neg 为空（模型未学负向），建议只加画质/物理负向，不要加与物理矛盾的内容：
```
blurry, distorted, low quality, jittery motion, objects passing through each other, objects floating without gravity, morphing shapes, inconsistent lighting, flickering
```

### 提示词示例（可直接使用）

**碰撞类**
1. `several colored objects, a single collision between objects, metal objects with rigid, elastic collisions, clear spatial layout, visible depth, objects maintaining spatial relationships in a synthetic scene`
2. `several colored objects, objects colliding with visible impact and bounce, clear spatial layout, gravity-consistent trajectories`
3. `objects colliding with visible impact and bounce, clear spatial layout, gravity-consistent trajectories`

**弹跳/回弹类**
4. `Elastic objects bouncing with realistic restitution, clear spatial layout, gravity-consistent rebound trajectories`
5. `objects bouncing with realistic restitution, clear spatial layout, gravity-consistent rebound`
6. `a ball rolling into a wall and rebounding with minimal bounce, clear spatial layout, visible depth, rigid collision with momentum transfer`

**多物体散射/台球类**
7. `billiard balls colliding and scattering on a table, momentum transfer between balls, clear spatial layout, visible depth`
8. `several balls on a table, a ball striking the group, momentum transfer, balls scattering with realistic collision, clear spatial layout, gravity-consistent`

**轨迹/无碰撞类**
9. `Rigid objects in motion with clear spatial layout, consistent trajectories, gravity-consistent movement and visible depth`
10. `several colored objects, objects moving with consistent trajectories, no collision, clear spatial layout, visible depth, objects maintaining spatial relationships in a synthetic scene`

**自由发挥模板**（替换行为词）
11. `a ball thrown across the scene, [parabolic arc / gravity-consistent trajectory], clear spatial layout, visible depth`
12. `two balls colliding in mid-air, clear spatial layout, consistent trajectories, gravity-consistent rebound, visible depth`
```

## 加速适配

与 MiniMax-H3 Turbo 加速 LoRA 兼容，可叠加使用。

## 训练信息

- 框架：ai-toolkit · Rank 16 · 分辨率 512 · 90 帧 @ 24fps
