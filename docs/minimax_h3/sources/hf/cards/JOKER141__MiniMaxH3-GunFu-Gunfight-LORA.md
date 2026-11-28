# JOKER141/MiniMaxH3-GunFu-Gunfight-LORA

Источник: https://huggingface.co/JOKER141/MiniMaxH3-GunFu-Gunfight-LORA · скачано 2026-11-28
Загрузок: 5016 · лайков: 25 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: lora · обновлено: 2026-10-25T16:48:49.000Z

---

---
language:
- en
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- minimax-h3
- lora
- video-generation
- action
- motion
- gunfu
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
---
<div align="center">

🐰 GUNFU — MiniMax H3

Cinematic Gunfights & Close-Quarters Action

电影感枪战 · 近身枪斗 · 动作衔接

Trigger: BUNNY 🐇

From ranged firefights to John Wick–style close combat—bring firearms, strikes, throws, and physical struggles into one continuous action sequence.

从远程交火到疾速追杀式近身搏杀，让枪械、打击、摔投与身体争夺衔接成完整的动作场面。

</div>

</div>
🎬 DEMO

<video src="https://huggingface.co/JOKER141/MiniMaxH3-GunFu-Gunfight-LORA/resolve/main/asset/demo.mp4" controls width="100%"></video>

<div align="center">

### 🔗 Recommended Companion · 推荐搭配

**For complex action scenes, highly recommended: pair this LoRA with [BUNNY H3 Conditioning Bridge](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge) to help improve multi-character logic, action ownership, and scene stability.**  

**复杂动作场景强烈推荐搭配 [BUNNY H3 Conditioning Bridge（语义桥）](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge)，可进一步改善多人互动逻辑、动作归属和场景稳定性。**

---

**Cinematic Gunfights & Close-Quarters Action**  

**电影感枪战 · 近身枪斗 · 动作衔接**

From ranged firefights to John Wick–style close combat—bring firearms, strikes, throws, and physical struggles into one continuous action sequence.  

从远程交火到疾速追杀式近身搏杀，让枪械、打击、摔投与身体争夺衔接成完整的动作场面。

GUNFU focuses on how the action connects: engagement, contact, impact reactions, and follow-through.  

GUNFU 注重动作之间的联系：交锋、接触、受击反应，以及紧随其后的动作。

🔑 **Trigger word:** `BUNNY`  

🔑 **触发词：** `BUNNY`

🎚️ **Recommended starting weight:** `0.9`  

🎚️ **推荐起始权重：** `0.9`

---

### 🧩 Independent Training · 独立训练

GUNFU was trained independently, without incorporating BASE into its training.  

GUNFU 为独立训练，训练中没有加入 BASE 作为基底。

Choose your setup based on the scene: ranged firefights or close-quarters gun-fu.  

根据场景选择搭配：远程枪战，或融合体术的近身枪斗。

### 🎯 Ranged Gunfights · 远程枪战

**First pass:** `GUNFU 0.9` alone, or `GUNFU 0.9 + Motion Continuity Repair 0.5`.  

**一采：** `GUNFU 0.9` 独立使用，或 `GUNFU 0.9 + 动作修复 0.5`。

**Second pass:** `GUNFU 0.65 + Motion Continuity Repair 0.25`.  

**二采：** `GUNFU 0.65 + 动作修复 0.25`。

### 👊 John Wick–Style Gun-Fu · 疾速追杀式近战枪战

**First pass:** `GUNFU 0.9 + COMBAT 0.5 + Motion Continuity Repair 0.3`.  

**一采：** `GUNFU 0.9 + COMBAT 0.5 + 动作修复 0.3`。

**Second pass:** `GUNFU 0.65 + Motion Continuity Repair 0.25`.  

**二采：** `GUNFU 0.65 + 动作修复 0.25`。

GUNFU provides the gunfight focus, COMBAT supports physical combat, and Motion Continuity Repair supports continuity between actions.  

GUNFU 负责枪战与枪斗表现，COMBAT 加强体术表现，动作修复辅助动作之间的连续衔接。

---

### 📦 Optional Downloads · 可选文件

**Two prompt-writing skills + a one-click workflow with the Conditioning Bridge node.**  

**两个提示词编写技能 + 包含语义桥节点的一键工作流。**

🎬 **Cinematic Gunfight Director** — Helps design the action, camera work, pacing, and sound.  

🎬 **枪战场面导演**——负责动作、镜头、节奏与声音设计。

✍️ **H3 Video Prompt Enhancer** — Turns your scene into a structured H3 prompt.  

✍️ **H3 视频提示词增强**——将场面设计整理成正式的 H3 提示词。

⚙️ **One-click workflow** — Includes the Conditioning Bridge node.  

⚙️ **一键工作流**——包含语义桥节点。

Download both skill ZIP files and upload them together to GPT or Codex, then ask it to read the skills and write your scene.  

下载两个技能 ZIP，一起上传给 GPT 或 Codex，让它读取技能并按你的要求编写场景即可。

---

### 💡 Get More from Your Prompts · 提示词小贴士

**Make the protagonist and targets clear.**  

**明确主角与攻击目标。**

Identify each opponent and describe who is attacking whom, especially when several characters share the frame.  

明确每个敌人以及谁在攻击谁，多人同框时尤其重要。

**Write actions with consequences.**  
**让动作产生可见的后果。**

Connect contact, loss of balance, hit reactions, and changes in position instead of listing unrelated moves.  

将接触、失衡、受击反应和位置变化连起来，不要只罗列互不关联的招式。

**Specify the weapon and firing behavior.**  

**明确枪械与开火方式。**

Describe what the character carries and how it fires instead of only writing “intense gunfight.”  

写清人物拿什么枪、如何开火，不要只写“激烈枪战”。

**Give the scene a voice.**  

**别让人物默默打完整场。**

Include gunshots, casing ejection, impact sounds, pain cries, and breathing where the action calls for them.  

根据动作加入枪声、抛壳、撞击声、痛叫和喘息。

Results depend on your input images, prompts, and generation settings—clear choreography still matters.  

效果会受到输入图片、提示词和生成设置影响，清晰的动作设计仍然重要。

<div align="center">

BUNNY 🐇

</div>
