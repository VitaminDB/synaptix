# Jojocodex/minimax-h3-wushu-action-lora

Источник: https://huggingface.co/Jojocodex/minimax-h3-wushu-action-lora · скачано 2026-11-28
Загрузок: 3067 · лайков: 71 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-30T00:37:44.000Z

---

---
license: apache-2.0
base_model: Comfy-Org/MiniMax-H3
tags:
- minimax-h3
- lora
- text-to-video
- martial-arts
- kung-fu
- action
- video-generation
---

# 1 武打动作 LoRA · Wushu Action LoRA (MiniMax-H3).最新更新 wushu_action_h3_lora_v5 · 3000 步。

武打动作 LoRA，基于 **MiniMax-H3**（Comfy-Org/MiniMax-H3）训练，让模型生成**人体武打/功夫动作**（拳、腿、踢、棍法等），专注人体打斗动作优化。
个人在测试中4步和8步加速LORA偶尔会有画面糊的情况。Wushu-Action-LoRA 0.5的强度搭配0.3强度spatial-physics-lora，打斗的效果会更好。

### 测试视频（对比原本）
### 对比视频
使用了LORA（0.5强度）
<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/3i3_sJ_BULn85mn6UlTIN.mp4" width="35%" controls muted loop></video>
没使用LORA
<video src="https://cdn-uploads.huggingface.co/production/uploads/6a6748afef317ed5adb30050/JRyBD2AZXWIgXE0zDA7Uu.mp4" width="35%" controls muted loop></video>

### 1.1 触发词（必须放开头）

**`wushu_action`** 是本 LoRA 的触发词，**放在提示词最前面**。训练数据中 FIGHT 类 caption 全部以它开头，模型已把它和"武术动作"绑定。

```
wushu_action, [动作主体] + [具体招式/行为] + [环境/镜头] + [可选画质词]
```

> 不带触发词时，LoRA 也有基础的动作先验，但针对性会弱很多。**正式使用请务必带上。**

### 1.2 提示词结构模板（4 段式）

```
wushu_action, [1 动作主体] + [2 具体招式与发力] + [3 场景/镜头] + [4 可选画质词]
```

例：
> `wushu_action,`（触发词）`a martial artist`（主体）`performing punches and kicks in fast combat, with explosive force`（招式与发力）`, indoor gym, medium shot, stable camera`（场景镜头）`, high quality, detailed`（画质）

**第 2 段是灵魂** —— 训练 caption 全是"具体招式 + 身体协调 + 发力"的详细描述，写得越具体，动作越到位。

### 1.3 关键词库（从训练数据提炼）

**① 动作主体**
- `a martial artist` 武术家 / `a fighter` 格斗者 / `a wushu practitioner` 武术练习者
- `two fighters` 两名格斗者（对打）
- `a martial artist with a staff` / `with a sword` / `with a spear` 持械

**② 招式与发力（核心）**
- 拳脚：`punches and kicks` 拳脚 / `fast combat` 快速战斗 / `explosive force` 爆发力
- 攻防：`exchanging strikes` 互换打击 / `in an intense fight` 激烈打斗 / `fast, powerful movements` 快速有力的动作
- 兵器：`fighting with a sword, slashing and thrusting` 剑的劈砍刺 / `precise technique` 精准技术
- 身体：`weight transfer` 重心转移 / `body rotation` 身体旋转 / `coordinated hip and shoulder rotation` 髋肩协同旋转
- 步法：`grounded footwork` 稳固步法 / `forward lunge` 前冲弓步 / `low bow stance (gong bu)` 弓步

**③ 场景/镜头**
- `indoor training studio` 室内训练馆 / `dojo` 道场 / `outdoor arena` 室外场地
- `medium shot` 中景 / `full-body shot` 全身景 / `stable camera` 稳定镜头 / `dynamic angle` 动态角度

**④ 画质词（可选）**
- `high quality, detailed, sharp motion, smooth animation`

### 1.4 提示词示例（可直接使用）

**单人拳脚**
1. `wushu_action, a martial artist performing punches and kicks in fast combat, with explosive force, indoor training studio, medium shot, stable camera`
2. `wushu_action, a wushu practitioner executing a rapid combination of punches and kicks, powerful hip rotation and weight transfer, gym background, full-body shot`
3. `wushu_action, a fighter performing a spinning kick, explosive leg extension, balanced landing, dojo setting, dynamic camera angle`

**双人对打**
4. `wushu_action, two fighters exchanging strikes in an intense fight, with fast, powerful movements, indoor arena, medium shot, stable camera`
5. `wushu_action, two martial artists in close combat, blocking and countering, realistic impact and recoil, training studio, full-body shot`

**兵器**
6. `wushu_action, a martial artist fighting with a sword, slashing and thrusting, with precise technique, dark studio, dramatic lighting`
7. `wushu_action, a martial artist with a staff performing spinning and striking techniques, explosive power, warehouse setting, dynamic angle`

**爆发力/跳跃**
8. `wushu_action, a fighter leaping and striking in mid-air, explosive force, grounded landing with bent knees, outdoor arena, slow motion feel`

**自由发挥模板**（替换招式词）
9. `wushu_action, a martial artist [performing X technique], [发力/协调描述], [场景], [镜头]`
   → 例：`wushu_action, a martial artist performing a flying side kick, explosive leg drive, full body extension, indoor studio, medium shot`

### 1.5 负面提示词建议

训练时 neg 为空（模型未学负向），建议只加画质/动作负向：
```
blurry, distorted, low quality, jittery motion, extra limbs, deformed hands, morphing bodies, inconsistent lighting, flickering, static pose, frozen motion
```


## 2 新增4500步的打斗动作LORA(这是收集的其他素材重新训练的)KungFu_MiniMaxH3_LoRA_step4500.safetensors 

### 2.1 触发词
触发词:kungfu combat
- 必须出现在提示词中,LoRA 才会被激活
- 建议放在提示词开头,权重最高
- 可以加权重语法:(kungfu combat:1.2) 增强激活

### 2.2 提示词结构模板
[触发词], [主体/人物], [动作], [场景环境], [镜头语言], [风格质感], [画质词]

### 2.3 示例提示词(可直接使用)

【示例 1:双人对打,经典武侠】
英文:
kungfu combat, two martial artists in a dynamic kung fu fight, exchanging
punches and kicks, blocking and dodging, cinematic action, outdoor arena,
dust and motion blur, dramatic lighting, 4k, high quality

中文:
功夫对战,两名武术家在激烈的功夫对打中,拳脚相交,格挡闪避,电影级动作,
室外竞技场,尘土飞扬与动态模糊,戏剧性光影,4K,高质量

【示例 2:古装武侠】
英文:
kungfu combat, ancient Chinese wuxia style, two warriors in traditional robes,
flying kicks and sword clashes, bamboo forest, misty atmosphere, slow motion
impact, epic music vibe, cinematic composition

中文:
功夫对战,中国古装武侠风格,两名身着传统长袍的武士,飞踢与刀剑交锋,
竹林,雾气缭绕,慢动作冲击,史诗氛围,电影构图

【示例 3:单人套路(武术表演)】
英文:
kungfu combat, a solo martial artist performing a traditional kung fu form,
fluid movements, punches and spinning kicks, temple courtyard, morning light,
graceful and powerful, smooth camera follow, high detail

中文:
功夫对战,一名武术家单独表演传统功夫套路,动作流畅,拳法与回旋踢,
寺庙庭院,晨光,优雅而有力,流畅跟拍,高细节

【示例 4:现代街头格斗】
英文:
kungfu combat, modern street fight scene, two fighters in casual clothes,
fast close-range combat, urban alley, neon lights at night, gritty realism,
intense expression, handheld camera shake

中文:
功夫对战,现代街头格斗场景,两名穿着休闲服的打斗者,快速近身格斗,
城市小巷,夜晚霓虹灯,粗粝写实风,表情激烈,手持镜头抖动

【示例 5:宏大战争场景】
英文:
kungfu combat, epic battle sequence, dozens of martial artists fighting,
wide shot, sweeping crane camera, golden hour, battlefield with flags,
choreographed chaos, blockbuster movie style

中文:
功夫对战,史诗级战斗片段,数十名武术家混战,广角镜头,扫动摇臂摄影,
黄金时刻,插旗战场,编排有序的混乱,大片风格

### 2.4 常用修饰词库(自由组合)

人物:two martial artists / a lone warrior / a female fighter / a shaolin monk
动作:spinning kick, flying jump kick, palm strike, grappling, sword fight,
      staff spinning, acrobatic flips, counter-attack
场景:outdoor arena, bamboo forest, temple courtyard, mountain peak, night
      market, rainy street, desert, dojo
镜头:close-up, wide shot, slow motion, camera follow, aerial shot, Dutch angle
风格:cinematic, wuxia, realistic, anime, gritty, epic, dramatic lighting,
      golden hour, neon lights, film grain, 4k, high quality
