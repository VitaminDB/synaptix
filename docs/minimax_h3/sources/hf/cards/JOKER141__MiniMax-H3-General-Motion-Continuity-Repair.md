# JOKER141/MiniMax-H3-General-Motion-Continuity-Repair

Источник: https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair · скачано 2026-11-28
Загрузок: 0 · лайков: 123 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: lora · обновлено: 2026-10-25T16:47:17.000Z

---

---
language:
- en
- zh
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- MINIMAX
- MINIMAXH3
- MOTION
- lora
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
---
General Motion Continuity Repair LoRA

通用动作连续性修复 LoRA

This is a general-purpose motion support LoRA, not a combat-only LoRA. It can be used for running / sports / dance / acrobatics / character interaction / combat / weapon motion and other dynamic scenes.

这是一个通用动态辅助 LoRA，并不是战斗专用模型，可用于跑步 / 体育 / 舞蹈 / 翻滚跳跃 / 人物互动 / 战斗 / 武器动作等动态场景。

*🎬 BUNNY Motion Repair V2 —— 高难动作不该再靠抽卡！*
*BUNNY Motion Repair V2 — High-Difficulty Motion Shouldn't Be a Lottery.*

*🎯 核心升级 | What's New*

*⚡ 专攻“极危禁区”： 重点修复 MiniMax H3 过去最易崩溃的高速空翻、旋转、翻滚、倒置及恢复。大幅增加 Full-Body 倒置旋转训练，同时延续 V1 的泛用修复能力。*

*Fixes H3's hardest failure zones: flips, spins, rolls, inversions, and orientation recovery, with expanded rotational training while keeping V1's general power.*

*🌪️ 地面支撑与空中翻腾双提升： 回旋、风车、Airflare、转体跳跃与落地缓冲稳定性暴涨！告别人体扭成一团、肢体接错或头躯不同步。*

*Major stability gains for flares, windmills, airflares, spins, and landing recovery. Stops limbs reconnecting wrong or bodies twisting into unreadable masses.*

*🔄 全链路衔接优化： 强化起跳 ➔ 倒置 ➔ 连续旋转 ➔ 重新找正 ➔ 落地全过程，不只是中段好看。*

*Improves the full motion sequence from takeoff, inversion, and spinning to reorientation and landing.*

*💡 底层哲学 | Pure Capability*

*🧠 能力导向打标： 学习的是人体结构、朝向感知与动量恢复而非死记具体招式，拥有极强泛化能力。*

*Trains core mechanics—structure, momentum, and recovery—rather than specific trick names for maximum generalization.*

*🎬 从废片到可用： 依然是修复 LoRA（极端过渡帧仍可能偶发小瑕疵），但足矣让输出从“直接废弃”变成“直接可用”。*

*Still a repair tool, but turns discarded clips into usable final shots.*

---

Recommended Usage / 推荐使用

Standalone use: around 0.9 is currently recommended.

单独使用时，目前推荐权重约为 0.9。

When used alone, Motion Continuity Repair needs a relatively high weight to produce a clear effect on motion continuity / slow-motion tendency / missing transitions / prompt-following.

单独使用时，需要相对较高的权重，才能比较明显地影响动作连续性 / 慢动作倾向 / 动作断链 / 提示词遵循度。

With a Combat LoRA: Stage 1 around 0.5–0.7 is recommended.

搭配 Combat LoRA 时，一采推荐约 0.5–0.7。

Because Combat already provides strong motion speed and impact, Motion Continuity Repair only needs a medium weight to correct continuity / coordination / action logic without unnecessarily taking over the motion.

因为 Combat 已经提供了较强的动作速度和打击感，此时 Motion Continuity Repair 只需要中等权重来修正动作衔接 / 协调性 / 动作逻辑，避免过度接管整个动态。

More complex sequences can gradually increase the Stage 1 weight, but very high weights may begin to reshape the entire motion logic / affect visual style / or alter audio characteristics.

动作越复杂可以逐渐提高一采权重，但过高时可能开始重构整段动作逻辑 / 改变画面表现 / 影响声音特征。

Stage 2: around 0.2–0.3 is recommended.

二采建议约 0.2–0.3。

Higher Stage 2 weights may over-stabilize the sequence and reduce motion intensity / speed / impact.

二采权重过高可能会把动作收得太稳，从而削弱动作幅度 / 速度感 / 冲击力。

最简单地说就是：

单独使用：≈0.9 / 搭配 Combat：0.5–0.7 / 二采：0.2–0.3

这个才是目前测试出来的正确推荐方式。

Although it is a general-purpose LoRA, it works especially well together with a Combat LoRA. Combat provides speed and impact / Motion Continuity Repair focuses on continuity, coordination, prompt-following and interaction logic.

虽然它是一个通用 LoRA，但目前非常推荐与 Combat LoRA 搭配使用。Combat 负责速度与打击感 / Motion Continuity Repair 负责动作衔接、协调性、提示词遵循与交互逻辑。

Trigger Word / 触发词
Trigger word: bunny_crisp_motion — optional. It mainly serves as an explicit identifier for the LoRA and is not required for normal use.

触发词：bunny_crisp_motion —— 可选。它主要用于明确标识这个 LoRA，正常使用时并不是必须填写。

You can test with or without the trigger word depending on your workflow and prompt.

你可以根据自己的工作流和 Prompt，自行测试是否加入触发词。

What does it do? / 它主要做什么？

Ever had a great clip ruined by less than one second of strange motion? A sudden slowdown / a missing action / an awkward limb trajectory / a broken interaction can easily make an otherwise usable clip unusable.

你是否遇到过这种情况：一段视频明明整体很好，却偏偏有不到 1 秒突然慢下来 / 漏掉一个动作 / 肢体轨迹变得诡异 / 交互突然断掉，最后整段素材因此报废？

This LoRA is designed around repairing those moments rather than simply making every motion stronger.

这个 LoRA 的重点就是修复这些“突然掉链子”的瞬间，而不是粗暴地把所有动作统一增强。

Current A/B testing shows improvements in unwanted slow-motion behavior / missing motion transitions / abnormal limb trajectories / action coordination / prompt-following / interaction continuity / and some mismatched motion-feedback events.

目前 A/B 测试中已经观察到对 异常慢动作 / 动作断链 / 诡异肢体轨迹 / 动作协调性 / 提示词遵循度 / 交互连续性 / 部分错误动作反馈 的改善。

When H3 already handles a scene well, the difference may be very small / when the base model begins to struggle, the difference can become much more obvious.

当 H3 本来就很擅长某段动态时，开启前后的区别可能非常小 / 当底模开始掉速、断动作或失去逻辑时，差异反而会明显很多。

It doesn't simply make H3's strengths stronger — it tries to keep its weaker motion states from falling apart.

它不是单纯让 H3 的强项更强，而是尽量不让 H3 在弱项动态里突然掉下去。

Current A/B Results / 当前对比表现
Different weights can progressively change motion intensity and eventually begin reshaping the overall motion rhythm.

随着一采权重提高，可以逐步改变原本偏弱的动态表现，高权重下甚至会开始重构整体动作节奏。

In complex action sequences, the LoRA can reconnect missing actions and reduce strange limb paths, making the motion chain more complete.

在复杂连续动作中，它可以补回部分没有连接上的动作，并减少诡异肢体轨迹，让动作链更加完整。

In scenes where even a Combat LoRA cannot fully suppress H3's slow-motion tendency, Motion Continuity Repair can further pull the sequence back toward a faster real-time motion state.

在一些连 Combat LoRA 都无法完全压住慢动作倾向的场景中，Motion Continuity Repair 还能进一步把动作往正常高速节奏拉回。

It can also improve action-prompt adherence, turning sequences that originally move randomly into results that follow the intended action order more closely.

它也可以提升动作提示词遵循度，让原本只是胡乱运动的结果，更容易按照 Prompt 中指定的动作顺序执行。

Weapon Interaction / 武器交互

A portion of weapon finisher / execution animation data was included in the dataset.

训练数据中还加入了一部分武器处决 / 终结动画。

Current testing shows some improvement in weapon disappearance / clipping / unstable fast swings / incomplete hit reactions / and certain finishing or kill animations.

目前测试中，对 武器突然消失 / 穿模 / 高速挥动不稳定 / 命中反馈不完整 / 部分终结与击杀动作 都有一定改善。

It is not a dedicated Weapon Combat LoRA, so sudden hand switching / unstable grips / occasional weapon disappearance / complex collision errors may still occur.

不过它不是专门的 Weapon Combat LoRA，因此突然换手 / 握持跳变 / 偶尔消失 / 复杂碰撞错误仍然可能发生。

How it was built / 制作方式

Before collecting the final dataset, a detailed screening plan was created specifically around the motion problems targeted by this project.

在正式筛选素材前，先根据这个项目想解决的动态问题，专门制定了一套详细筛选方案。

Every candidate clip was manually reviewed frame by frame, especially around motion / interaction / contact moments, and unsuitable material was removed one by one.

所有候选片段都会由我人工逐帧观看，重点检查动作 / 交互 / 接触过程，并逐条剔除不符合要求的素材。

The remaining clips were then reviewed again with large-model AI under strict project-specific rules, followed by a second filtering pass.

剩余素材还会再由大模型 AI 按照本项目专门制定的严格规则进行第二轮复查与筛选。

Only 100 clips were ultimately retained, followed by a project-specific captioning strategy and final training.

最终只有 100 段片段被保留下来，再经过专门为本项目定制的特化打标策略进行标注与训练。

Important / 重要说明

This LoRA is not a universal “+20% motion quality” switch. If H3 already generates the scene correctly, you may barely notice it. Its value becomes much easier to see when the original result contains a brief failure.

它并不是一个“打开以后所有动态统一 +20%”的开关。如果 H3 原本就生成得很好，你可能几乎感觉不到它；真正容易看出区别的，是原版已经出现短暂异常的场景。

Very high weights are experimental / they may reshape the overall action logic / introduce stronger CG or 3D-game characteristics in extreme dynamic scenes / or affect generated audio.

高权重属于实验区 / 可能重构整体动作逻辑 / 在极端动态场景中增加 CG 或 3D 游戏感 / 也可能影响生成声音。

Don't let one bad second ruin an otherwise great clip.

别让不到 1 秒的掉链子，毁掉整段本来很好的视频。
