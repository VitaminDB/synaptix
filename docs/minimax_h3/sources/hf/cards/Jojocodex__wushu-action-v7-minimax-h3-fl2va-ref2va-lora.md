# Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora

Источник: https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora · скачано 2026-11-28
Загрузок: 17894 · лайков: 93 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-10-05T16:02:35.000Z

---

---
license: other
tags:
- text-to-video
- image-to-video
- lora
- minimax-h3
- wushu
- martial-arts
- action
base_model: MiniMaxAI/MiniMax-H3
---

# Wushu Action LoRA — MiniMax H3（E3：全量 int8 DiT + bf16 文本编码器）

武打动作风格 LoRA（ComfyUI 即插即用）。基于 924 段精选武打片段（832×480 @ 24fps，专业招式术语三段式打标），以 **全量 int8-convrot DiT（未剪枝）+ bf16 Qwen3-VL 文本编码器** 训练 1000 步——当前量化精度最高的版本。触发词 `wushu_action`。

A wushu action style LoRA for MiniMax H3, trained 2000 steps on the **unpruned full int8-convrot DiT + bf16 Qwen3-VL text encoder** — the highest-fidelity quantization configuration. Trigger word: `wushu_action`.

之前PRUNED底座的训练的模型我都移除了，目前测得这版算是比较好一点的。但是表现我觉得还是没有达到我的预期，我觉得还是要再优化和增加素材，做一版新的LORA，届时会增加大量的运动动作。目前再整理再整理素材中，预计训练好也还需要2日时间。

## 大家可以尝试搭配JOKER141 的Combat-Base-V2和 Motion Continuity Repair LoRA来使用，效果更加。或者加入少许强度的Spatial & Physics LoRA弥补一点物理上的补偿。
## 本人郑重声明，本人是尊重原创的，不会将其他的作者的作品改成自己的署名上传。本人也喜欢和尊重四只兔子大佬的LORA，也就是JOKER141。都是公益性作者，收集素材整理打标训练，这些都是需要时间成本和资金的，并没有想通过大家去赚这点利益，喜欢就支持，不喜欢就无视就好了。你如果是做推广性作者，可以去推销你打包整理的东西，但麻烦尊重支持一下原创，不要随意改别人的署名，谢谢，同时也感谢大家的喜欢和支持。

## 文件 / Files

| 文件 | 说明 |
|---|---|
| `wushu_action_v7_fl2va_aitoolkit_adaln_full-int8convrot_bf16te_1000step.safetensors` | **E3 主模型**（含 adaln 校准层，配全量 int8-convrot 底模） |
| `Minimax h3多参考双采打斗工作流.json` | 多参考双采打斗 ComfyUI 工作流 |
| `招式TAGS完整清单.md` | 训练集全量招式标签清单（写提示词用） |

> 底模：`minimax_h3_fl2va_int8_convrot.safetensors`（34GB，Comfy-Org 未剪枝 int8 版）
> 已知问题：低步数采样出片偏模糊，请提高步数或配合修复类插件。

---

# ComfyUI 使用教程

## 第 1 步：放置模型文件

| 文件 | 放到哪 |
|---|---|
| 底模 `minimax_h3_fl2va_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` |
| 文本编码器 `qwen3vl_32b_minimax_h3_bf16.safetensors` 或 `nvfp4_awq` 版 | `ComfyUI/models/text_encoders/` |
| 视频 VAE `minimax_h3_video_vae_fp16.safetensors` + 音频 VAE `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` |
| **LoRA**（本仓库 E3 主模型） | `ComfyUI/models/loras/` |

## 第 2 步：搭工作流

用仓库内的**多参考双采打斗工作流 JSON** 直接导入，或用 ComfyUI 模板库搜 **MiniMax-H3**。手动搭线：

```
UNet Loader (minimax_h3_fl2va_int8_convrot)
  → LoraLoader (E3 LoRA, strength 0.8–1.0)
  → KSampler → VAE Decode → Save Video (mp4, 含音轨)

CLIPLoader (qwen3vl_32b) ↘
双 VAE (video fp16 + audio fp32) → VAE Decode ↗
```

## 第 3 步：KSampler 参数（照抄即可）

| 参数 | 值 | 说明 |
|---|---|---|
| sampler | **euler** | 只用这个 |
| scheduler | **simple** | 只用这个 |
| steps | **25** | turbo 底模 4–8 |
| **cfg** | **1.0** | ⚠️ H3 是 CFG 蒸馏模型，>1 必出坏图 |
| denoise | 1.0 | — |
| width × height | **832 × 480** | 32 的倍数；竖屏 480×832 |
| frames | **124** | ≈5.2 秒；必须 17n+5（22/39/56/73/90/107/124/175…） |
| fps | 24 | — |

## 第 4 步：提示词（核心！）

**规则**：以触发词 `wushu_action,` 开头；**角色外貌单独定义在「角色设定」块**（发色+服装+体型+兵器一次定义，全篇逐字固定），镜头里只称"角色A/角色B"不重复外貌（人物一致性关键）；招式写具体术语（"扫堂腿" ✔ "踢腿" ✘）；三段式分镜；结尾带音效描述。

### 模板格式（角色设定与镜头分离）

```
wushu_action, 电影级硬核写实格斗。【场景+光线】。

角色设定：
角色A（攻方）：【性别+发色发型+服装+体型+兵器】
角色B（守方）：【性别+发色发型+服装+体型+兵器】

开局即战，攻防轮换，击中反馈分级，终结重击可击飞。

integrated_multimodal_description:
[Shot 1] 胸口手持跟拍。角色A【招式起势发力】，角色B【防守/受击反馈】。
[Shot 2] 短弧侧绕跟拍。角色B【反击】，角色A【防守】，双方换位。
[Shot 3] 急推近景。角色A【终结重击命中角色B】，角色B【沿作用线击飞+侧滚撞墙】，角色A【追击收势】。

overall_soundscape: 拳腿破空声、衣料剧烈摩擦声、格挡沉闷撞击声、重拳轰鸣命中声、急促喘息声。

non_diegetic_music:
None.
```

### 成品模板 T1 扫堂腿 + 腾空侧踹（入门首选）

```
wushu_action,

integrated_multimodal_description:
[Shot 1] Medium shot, side angle, outdoor courtyard with traditional architecture. A martial artist in a dark training uniform sinks into a low crouch, palms open at his ribs, weight loaded on the rear leg.
[Shot 2] He drives off the ground, sweeping low with a 扫堂腿 that scrapes dust off the floor, then springs airborne into a full-extension 腾空侧踹, heel leading, arms counter-swinging for balance.
[Shot 3] He lands rolling to dissipate force, rises into a front stance with palms pushed forward, holding the final guard.

overall_soundscape:
The whoosh of the sweeping leg, cloth friction, footwork scraping the floor, a dull impact on landing, controlled breathing.

non_diegetic_music:
None.
```

### 模板 T2 长剑套路（缠头裹脑 + 回马枪）

```
wushu_action,

integrated_multimodal_description:
[Shot 1] Wide tracking shot, bamboo grove at dusk. A white-robed swordsman holds a straight jian low, blade tip down, body turned sideways in a narrow stance.
[Shot 2] He opens with 缠头裹脑, the blade circling behind his neck and back, then snaps into a turning reverse thrust like a 回马枪, sword arcing close to the body before a rising 撩剑 flicks upward.
[Shot 3] He settles into a horse stance, sword extended point-forward, bamboo leaves drifting down around him.

overall_soundscape:
Sword blade cutting through air, cloth rustling, soft footsteps on fallen leaves, the ring of the blade settling.

non_diegetic_music:
None.
```

### 模板 T3 双人对打（贴山靠 + 鞭腿）

```
wushu_action,

integrated_multimodal_description:
[Shot 1] Medium shot, front angle, wooden stage. The red fighter initiates with a shoulder charge 贴山靠, the blue fighter angles sideways to deflect and unload the force 格挡卸力.
[Shot 2] Red presses in with an elbow strike, blue retreats half a step and counters with a whipping 鞭腿; red absorbs it on a hardened guard, both locked for a beat.
[Shot 3] They break apart simultaneously, stepping back into guarded stances, fists saluted, the exchange cleanly reset.

overall_soundscape:
Punch impacts, cloth rustle, footwork thuds on wooden planks, strained breathing, crowd murmur in the far background.

non_diegetic_music:
None.
```

### 模板 T4 枪术回马枪

```
wushu_action,

integrated_multimodal_description:
[Shot 1] Full shot, fixed camera, indoor white-wall studio. A spearman holds a vertical spear-holding ready stance, shaft planted, body angled.
[Shot 2] He slides into a charging thrust, then feints a retreat — at the turn he snaps the spear backward over his shoulder in a textbook 回马枪, the point whipping to full extension.
[Shot 3] He whirls the shaft overhead and slams the butt into the ground, settling into a bow stance, spear vertical.

overall_soundscape:
Wooden shaft whooshing through air, the shaft scraping against his palm, sharp foot slides, a heavy butt-end thud.

non_diegetic_music:
None.
```

### 模板 T5 女子双钩剑（纯英文）

```
wushu_action,

integrated_multimodal_description:
[Shot 1] Medium tracking shot, minimalist white room. A female martial artist in a yellow training robe holds twin hook swords inverted, blades hugging her forearms, weight coiled on the rear foot.
[Shot 2] She explodes forward, hooks sweeping in crossing arcs, then leaps into a full-rotation aerial spin, both blades extended in opposite directions, robe flaring with centrifugal force.
[Shot 3] She lands in a low horse stance, hooks crossed before her chest, holding the final pose as her sleeve settles.

overall_soundscape:
Twin blades whistling in counter-rotation, fabric snapping, a sharp landing thud, controlled breathing.

non_diegetic_music:
None.
```

## 招式替换词库

完整清单见仓库内 `招式TAGS完整清单.md`（训练集全量提取，含频次）。速查：

| 类别 | 高频术语 |
|---|---|
| 腿法 | 扫堂腿 sao tang tui ｜ 腾空侧踹 ｜ 鞭腿 ｜ 后旋踢 ｜ 旋风脚 |
| 拳掌肘靠 | 直拳 zhi quan ｜ 贴山靠 tie shan kao ｜ 肘击 ｜ 崩拳 |
| 摔拿 | 擒拿 joint-lock ｜ 过肩摔 ｜ 接腿摔 |
| 剑术 | 缠头裹脑 chan tou guo nao ｜ 撩剑 liao jian ｜ 劈剑 ｜ 回马枪 |
| 枪棍 | 扎枪 zha qiang ｜ 中平枪 ｜ 拦拿 ｜ 横扫 |
| 步法 | 弓步 gongbu ｜ 马步 mabu ｜ 虚步 xubu ｜ 点步 dianbu |
| 身法 | 腾空 ｜ 纵跃 ｜ 翻滚卸力 ｜ 旋身横扫 ｜ 蹬墙借力 |
| 衔接 | 格挡卸力 ge dang xie li ｜ 闪身 shanshen ｜ 借力打力 ｜ 以攻代守 |

## 避坑清单

1. 招式宁专勿泛："扫堂腿" ✔ "踢腿" ✘
2. 一个 Shot 只放 1–2 个招式，别堆砌
3. 不要写"静止站立/呆立"——模型学的是连续动作
4. 保留 `non_diegetic_music: None.`
5. 画质/分辨率交给参数，不要写进提示词
6. 人物/兵器崩坏 → LoRA 强度降到 0.6–0.8；动作不够武 → 升到 1.1

# wushu_h3_v8 LoRA （这个素材主要用大量的3D模型跑的动作，所以真实感会欠缺，但是可以作为优化动画去使用，权重太高画面可能会显黑，因为素材都是黑色背景。）
> 适配 MiniMax-H3 FL2VA 的武术动作 LoRA。本LoRA学习人体运动与发力逻辑，不学习画面质感、角色贴图。
> 触发词：**`wushu_action`**｜底模要求：`minimax_h3_fl2va` 系列，

## 使用前必读
1. **触发词 `wushu_action` 必须放在提示词最前面**。所有训练标注均以此词开头，模型将该词与武术动作强绑定。
2. 该LoRA训练素材全部为人体模型三视图动画，默认容易输出白模+纯黑背景。这是设计特性，不是BUG。
3. LoRA权重推荐范围：
    - 0.9–1.0：生成三视图人体动作参考素材
    - **0.6–0.8**：正常场景出片（推荐起点）
    - 若大量出现白模黑底，降低至0.5–0.6

## 提示词写法
### 模式A：生成训练风格三视图人体动作参考
直接复制模板，仅替换招式描述
```
wushu_action, 白色无贴图男性人体模型, 同一角色三视图同步并排, 纯黑虚空, 固定全身机位, 动作完全同步, 不是三个战士,【填入招式】, 实时爆发力, 全身发力传导, 无慢动作, 无停顿
```

### 模式B：生成真实角色与场景（视频创作推荐）
保留触发词和动作关键词，把模板里的人体模型、三视图、黑底替换为你需要的角色和场景
```
wushu_action,【角色与场景描述】,【填入招式】, 实时爆发力, 全身发力传导, 无慢动作, 无停顿
```
> 提示：优先使用训练集中已存在招式词，成功率更高：直拳、高侧踢、疾步冲刺、举盾格挡、长枪突刺、翻滚、连续连招

## H3 生成参数
- 采样器：euler｜调度器：simple
- 采样步数：25
- CFG：**1.0，请勿大于1**
- 视频帧数：遵守 `17n+5` 规则（可选：22 / 39 / 73 / 90 / 124）
- 分辨率：宽高必须为32的倍数，推荐 832×480

## 负面提示词
```
模糊，扭曲，低画质，画面抖动，多余肢体，手部畸形，形体变形，光影错乱，画面闪烁，静止姿势，动作僵硬
```

## 已知限制
- 训练片段时长仅1.3–3秒。长视频建议分段生成后剪辑拼接。
- 训练素材均为单人，双人对打属于外推效果，稳定性较差。
- 本LoRA只学习动作；角色皮肤、服装、面部质感依靠底模或画风LoRA实现。

## ComfyUI部署方法
1. 将LoRA文件放入 `models/loras/` 文件夹
2. 加载 `minimax_h3_fl2va` 扩散模型，搭配对应的Qwen3VL文本编码器、H3视频VAE
3. UNet加载节点后接入LoRA加载器，设置上面推荐的权重
4. LoRA挂载校验：对比权重0与权重1的出片效果。若无明显差别，需要对safetensors做键名重映射。
