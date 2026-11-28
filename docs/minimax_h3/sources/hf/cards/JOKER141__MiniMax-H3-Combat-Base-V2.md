# JOKER141/MiniMax-H3-Combat-Base-V2

Источник: https://huggingface.co/JOKER141/MiniMax-H3-Combat-Base-V2 · скачано 2026-11-28
Загрузок: 39124 · лайков: 55 · базовая: ["Comfy-Org/MiniMax-H3"] · теги: lora,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-10-25T16:48:06.000Z

---

---
language:
- en
- zh
base_model:
- Comfy-Org/MiniMax-H3
tags:
- minimax-h3
- lora
- combat
- action
- motion
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
---
Combat Base V2 is a major upgrade over V1, focusing not only on faster combat, but on better action continuity, hit reactions, cumulative damage, spatial movement, and physical cause-and-effect.
Combat Base V2 is no longer just a combat LoRA — it now works for both action and dialogue scenes, adding richer body movement, finer sound details, and stronger shot-to-shot continuity to character performance.
Combat Base V2 不再只是一个武打 LoRA——它现在可文可武，在文戏中也能增加更丰富的身体动作、更细致的声音表现，以及更强的镜头与状态连续性。
If the LoRA looks muddy or less impactful than expected, check the sampler setup first.
Sampler choice matters a lot for this LoRA.

For best clarity and motion definition, I recommend one of the following setups:

Option A

Sampler: res_multistep

Scheduler:simple

Option B

Sampler: euler

Scheduler: beta

Other sampler / scheduler combinations may produce noticeably blurrier or softer results.

Core Physics & Motion Logic / 核心物理与动作逻辑
Combat Base V2 moves away from "attack → reset" loops into a continuous physical chain where missed strikes carry momentum, blocks change direction smoothly, and limbs separate cleanly without sticking.

Combat Base V2 摆脱了“攻击 → 复位”的回合制循环，形成持续的运动链，招式落空可延续动量，格挡可自然顺势变向，且接触点脱离干净不黏连。

Hit reactions now persist and accumulate—sustained torso bending, lowered guards, and impaired mobility directly build toward convincing heavy finishers.

受击状态具备持续性与累计效应，躯干弯曲、护架变慢及移动受限等伤害会自然保留，让最后的重击终结更加合理成立。

Combat Skillset & Staging / 动作技法与镜头调度
Upgrades include continuous rotational kicks, sustained grappling and takedowns, dynamic 3-stage environmental collisions, and multi-phase fall trajectories.

全面升级连续旋转腿法、持续近身缠抱摔投、三阶段环境碰撞（人体→环境→材质），以及拥有完整轨迹的特技跌落表现。

Supports cinematic 1vMany choreography via staggered attacker entrances, spatial lane control, continuous protagonist-centered tracking, and motivated cross-cut motion continuity.

支持电影级“一对多”围攻调度，利用交替切入、环境通道压迫、主角中心跟随镜头以及跨镜动作连贯性提升画面表现。

Dialogue Performance & Benchmarks / 文戏表现与测试数据
Internal same-seed dialogue tests revealed +19.5% in-shot motion, +34.2% audio RMS, and +22.4% edge sharpness, significantly enhancing physical acting, breathing, and vocal dynamics outside of fight scenes.

同 Seed 文戏测试显示镜内动作量增加 +19.5%、Audio RMS 提升 +34.2%、边缘锐度提升 +22.4%，大幅增强了武戏之外的肢体表演、呼吸与声音动态。

Workflow & Trigger Usage / 工作流与触发词使用
For optimal flow, structure prompt timing as ACTION → DIALOGUE; start with No Trigger for maximum anatomical accuracy, add prfight2 for higher intensity, or prfight2, prfin1 for heavy stunt impact.

建议将提示词结构设为“动作 → 对白”；优先使用 No Trigger 换取最稳动作与人体，需要强攻击性时加入 prfight2，重终结/特技受击则使用 prfight2, prfin1。

Includes two specialized combat-director workflows (General & 1vMany) with Chinese instructions that support prompt output in any target language.

附带“通用”与“一对多”两个专用武戏导演工作流，内置中文说明并支持生成任意语言的最终 Prompt。

Direct the fight first. Write the prompt second. 先把战斗导演好，再去写提示词。

Upgrade Notes：

测试发现：同一镜头包含打斗和台词时，必须先完整完成击打动作再说台词，否则容易退回底模的慢动作表现。

Test finding: When combat and dialogue are in the same shot, complete the attack before the dialogue, or the motion may revert to the base model’s slower behavior.

About This LoRA｜关于这个 LoRA
先说明一下：这不是一个魔法 LoRA，也不会自动帮你编排完整的武打动作。
First of all: this is not a magic LoRA, and it will not automatically choreograph a complete fight for you.

如果你只写 two people fighting、a woman fights a man 之类非常简单的 Prompt，模型很可能只是让两个人高速乱挥拳，最后变成一段很快但没有逻辑的“王八拳”。
You still need to tell the model who attacks first, what they do, where the strike lands, how the opponent dodges or reacts, and why the next attack can naturally follow.

Prompt 负责决定“他们怎么打”，LoRA 负责增强“这些动作怎么表现”。
The prompt decides what the fighters do, while the LoRA enhances how those actions are performed.
