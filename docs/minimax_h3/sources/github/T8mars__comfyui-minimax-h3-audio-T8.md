# T8mars__comfyui-minimax-h3-audio-T8

Source: https://github.com/T8mars/comfyui-minimax-h3-audio-T8
Fetched: 2026-11-28

Description: null
Stars: 1299  Updated: 2026-11-26T19:56:54Z



---

## FILE: README.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/README.md)

# MiniMax H3 Audio T8

用于 ComfyUI 的 MiniMax H3 视频与声音节点：参考图／音频、双采与长视频、人物口型、运镜编辑，以及可选高清后处理。

简体中文 | [English](README_EN.md) · 当前版本：**1.107.0** · [更新日志](CHANGELOG.md)

本版新增 Sulphur-3 Corrector 加载与应用节点。[版本说明](docs/RELEASE_1.107.0.md) · [22套已审创作工作流](examples/workflows/98-nov12-reviewed/README.md)

## 安装

需要较新的 ComfyUI 原生 H3 支持。手动安装：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

安装或更新后，完全退出并重启 ComfyUI，再刷新网页。Manager 可搜索 `MiniMax H3 Audio T8`；Registry 与 GitHub 发布进度独立，未显示最新版时使用 GitHub 安装。

Windows 更新提示 `Filename too long`？见[短路径与恢复说明](docs/WINDOWS_PATHS.md)；不要删除本地修改或已保存画布。

## 选一个工作流开始

先替换自己的模型和素材，不要直接运行示例里的占位文件。

直接写整段提示词：[普通提示词工作流总索引](examples/workflows/PLAIN_PROMPTS.md)，含双采、分块与恢复副本。

|你想做什么|入口|
|---|---|
|Sulphur-3 可选 Corrector 修正网络|[节点与接线](docs/SULPHUR3_CORRECTOR.md) · [候选工作流](examples/workflows/99-sulphur3)|
|2D放大／LoRA区间／DMAD／IC／Flash6+3／声音尾接（EXP）|[22套已审图](examples/workflows/98-nov12-reviewed/README.md) · [接线与范围](docs/NOV12_CREATION_EXP.md)|
|TaoMate-HD／关联噪声／NAG／阶段复用／局部音效（EXP）|[实验接线](examples/workflows/96-nov05-exp/README.md) · [用法与验证范围](docs/NOV05_INTEGRATION.md)|
|Pika H3 Flex 6步文字／首尾帧／参考图（EXP）|[三套已审工作流](examples/workflows/94-pika-flex/README.md) · [用法](docs/PIKA_H3_FLEX.md)|
|Pika双采4+2／3+3、普通模型与可选2K 6+3（EXP）|[八套已审工作流](examples/workflows/95-pika-dual/README.md) · [用法](docs/PIKA_DOUBLE_SAMPLING.md)|
|同条件先音后画／Wuli普通Hybrid（EXP）|[H13两份已审工作流](examples/workflows/88-h13-reviewed/README.md) · [接线与边界](docs/H13_AUDIO_FIRST_HYBRID.md)|
|Spark／Wuli原生4与8步、仅音频参考／耗时卡／桥参数侧车（EXP）|[H11八份已审工作流](examples/workflows/87-h11-reviewed/README.md) · [接线与边界](docs/H11_NATIVE_EXTENSIONS.md)|
|Veda R2VA、FreeVideo INT8、人物移除与原声／参考提案（EXP）|[H09九张已审来源图](examples/workflows/86-h09-reviewed/README.md) · [接线与限制](docs/RELEASE_1.98.0.md)|
|给现有视频生成拟音／H07已审配方（EXP）|[七张工作流](examples/workflows/85-h07-reviewed/README.md) · [拟音用法](docs/H07_FOLEY_USAGE.md)|
|第一次跑 H3／首帧图生视频|[基础生成](examples/workflows/01-basic-generation)|
|音频参考、人物说话／唱歌|[音频控制](examples/workflows/02-audio-control) · [原生音色／情绪与 Avatar](examples/workflows/36-avatar-voice/README.md)|
|双模型 4+4、分段长视频|[长视频](examples/workflows/04-long-video)|
|独立一采／二采、外置 EAV／Relay|[分离采样](docs/MODULAR_SAMPLING_EXP.md) · [Veda 8 步](examples/workflows/63-veda-t2va/README.md)|
|参考图框选人物与目标（EXP）|[视觉标记工作流](examples/workflows/84-visual-marker/README.md) · [接线与限制](docs/VISUAL_MARKER_EXP.md)|
|FreeVideo FP8：8+2／真正4+4（EXP）|[六张分离工作流](examples/workflows/72-freevideo-split/README.md) · [独立环境](docs/FREEVIDEO_EXP.md)|
|Kijai 九原件、Acc8 Full／Pruned 分离双采|[18张工作流](examples/workflows/71-kijai-experimental-split/README.md) · [模型配对](docs/KIJAI_EXPERIMENTAL_LORAS_EXP.md)|
|外片续拍、可见脸 MASK、无脸旁路及曲线双采|[RADAR 新工作流与限制](docs/RELEASE_1.89.0.md)|
|可视化编排镜头、素材和声音|[曜石导演台](examples/workflows/39-director-console/README.md)（也可点 ComfyUI 左侧独立的 `T8 曜石导演台`）|
|OpenVDN、FastH3、低显存|[加速工作流](examples/workflows/10-speed) · [低显存说明](docs/H3_MEMORY_NODES_EXP.md)|
|Meridian 图片／视频运镜、源时间编辑|[四节点工作流](examples/workflows/37-meridian/README.md)|
|一采动态预览、定向取消|[TAEH3 预览](docs/TAEH3_SAMPLING_PREVIEW_EXP.md)|
|成片高清放大／插帧|[Topaz](docs/TOPAZ_EXP.md) · [DLSS-NR](examples/workflows/25-dlss-nr) · [DLSS 插帧](examples/workflows/29-dlss-fi)|
|其他功能与详细操作|[完整工作流目录](examples/workflows) · [详细使用说明](docs/README_DETAILS_ZH.md)|

## 模型下载与位置

下载前阅读对应模型仓库的 README 和许可。不要把不同版本同名权重互相覆盖。

|模型／资源|下载与 ComfyUI 目录|
|---|---|
|H3 主模型、Qwen、视频／音频 VAE|分别放 `models/diffusion_models`、`models/text_encoders`、`models/vae`；[基础安装说明](docs/README_ComfyUI.md)|
|TAEH3 时序／2D 预览模型|[Taeh3-Comfy](https://huggingface.co/t8star/Taeh3-Comfy) → `models/vae_approx/`；2D 文件另名保存|
|Meridian ConvRot INT8 ＋ Omega 1B512|[Meridian-Comfy](https://huggingface.co/t8star/Meridian-Comfy) → `models/meridian/`；Omega 在 `vggt-omega/checkpoints/`，源码／assets 和 H3 VAE 另备|
|Semantic Bridge|[Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy) → `models/semantic_bridge/t8_compat/`|
|OpenVDN 完整包|[Vdn-Minimax-H3-Comfy](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)；保持仓库目录结构|
|H3-World 动作 LoRA|[Minimax-H3-World-Comfy](https://huggingface.co/t8star/Minimax-H3-World-Comfy)；[接线说明](examples/workflows/26-h3-world)|

Meridian 的 DMD 已合并，**不要再加载同一 DMD LoRA**。Omega 是官方原始 PT 几何模型，不是 INT8 或普通 VGGT。TAEH3 只做近似预览，不替换最终 VAE，不生成声音。

## 使用注意

- `Advanced`／`EXP` 使用对应工作流；指定样片通过不代表所有素材、显卡或长窗口都能得到相同效果。
- 旧工作流不强制迁移；未验证的 LoRA／Sage／Sol 组合只提示边界，不一刀切禁止连接。见[补丁共存策略](docs/PATCH_STACK_POLICY.md)。
- 不保证精确声纹、逐字时序、无接缝或通用 16GB 性能。双采仍有轻微接缝变色的已知限制。
- Topaz／DLSS 是独立后处理；各自的程序、资源和运行条件见对应文档，不能只下载 H3 权重。

节点变红、声音异常或模型不兼容时，先看[常见问题与高级配置](docs/README_DETAILS_ZH.md)。反馈请附完整报错、工作流与 Core／节点版本：[提交 Issue](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues)，移除密钥和隐私素材。

## T8 链接

|平台|入口|
|---|---|
|B站／YouTube|[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/)|
|API／在线 AI 应用|[API（推广链接）](https://api.seedance.nz/sign-up?aff=5f4w) · [RunningHub（邀请链接）](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)|
|整合包／模型网盘|[ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [模型网盘](https://pan.quark.cn/s/c9c267081fbf)|
|模型／节点|[Hugging Face](https://huggingface.co/t8star) · [节点 GitHub](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)|

## 许可与文档

节点源码：[GPL-3.0-or-later](LICENSE)。本 GitHub 不含模型权重；模型及外部资源各自遵守原许可。Meridian 衍生权重遵守 MiniMax H3 Community License，Omega 单独遵守 FAIR 非商用研究许可；转换不改变这些边界。

[更新日志](CHANGELOG.md) · [第三方说明](THIRD_PARTY_NOTICES.md) · [功能清单](features.json) · [首页维护规则](docs/README_POLICY.md)


---

## FILE: README_EN.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/README_EN.md)

# MiniMax H3 Audio T8

MiniMax H3 video/audio nodes for ComfyUI: reference control, two-pass and long-video workflows, speech and singing, camera editing, and optional enhancement.

[简体中文](README.md) | English · Current version: **1.107.0** · [Changelog](CHANGELOG.md)

This release adds Sulphur-3 Corrector loader and apply nodes. [Release notes](docs/RELEASE_1.107.0.md) · [22 reviewed creation workflows](examples/workflows/98-nov12-reviewed/README.md)

## Install

Requires recent native ComfyUI H3 support. Manual installation:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

Exit and restart ComfyUI after installing/updating, then refresh the page. Manager can search `MiniMax H3 Audio T8`; Registry activation is separate from GitHub publication. Use GitHub if the latest version is not available there.

Windows update reports `Filename too long`? See [path limits and recovery](docs/WINDOWS_PATHS.md); preserve local changes and saved workflows.

## Choose a workflow

Replace placeholder models and media with your own before running.

Use a single complete prompt: [ordinary-prompt workflow index](examples/workflows/PLAIN_PROMPTS.md), including two-pass, chunked and resume variants.

|Goal|Start here|
|---|---|
|Sulphur-3 optional corrector network|[Nodes and wiring](docs/SULPHUR3_CORRECTOR.md) · [Candidate graph](examples/workflows/99-sulphur3)|
|2D upscaling / LoRA windows / DMAD / IC / Flash6+3 / audio carry (EXP)|[22 reviewed graphs](examples/workflows/98-nov12-reviewed/README.md) · [Scope and wiring](docs/NOV12_CREATION_EXP.md)|
|TaoMate-HD / paired noise / NAG / stage reuse / local SFX (EXP)|[Experimental wiring](examples/workflows/96-nov05-exp/README.md) · [Usage and validation scope](docs/NOV05_INTEGRATION.md)|
|Pika H3 Flex six-step text / endpoint / reference generation (EXP)|[Three picture-reviewed workflows](examples/workflows/94-pika-flex/README.md) · [Guide](docs/PIKA_H3_FLEX.md)|
|Pika two-pass 4+2 / 3+3, regular and optional 2K 6+3 (EXP)|[Eight reviewed workflows](examples/workflows/95-pika-dual/README.md) · [Guide](docs/PIKA_DOUBLE_SAMPLING.md)|
|Same-condition AudioFirst / ordinary Wuli Hybrid (EXP)|[Two reviewed H13 workflows](examples/workflows/88-h13-reviewed/README.md) · [Wiring and limits](docs/H13_AUDIO_FIRST_HYBRID.md)|
|Spark / Wuli native 4/8-step, audio-only references, phase cards and Bridge sidecars (EXP)|[Eight reviewed H11 workflows](examples/workflows/87-h11-reviewed/README.md) · [Wiring and limits](docs/H11_NATIVE_EXTENSIONS.md)|
|Veda R2VA, FreeVideo INT8, Person Removal, source PCM and proposals (EXP)|[Nine H09 source templates](examples/workflows/86-h09-reviewed/README.md) · [Wiring and limits](docs/RELEASE_1.98.0.md)|
|Generate Foley for an existing video / reviewed H07 recipes (EXP)|[Seven workflows](examples/workflows/85-h07-reviewed/README.md) · [Foley usage](docs/H07_FOLEY_USAGE.md)|
|Box-marked actors and targets in a reference image (EXP)|[Visual marker workflows](examples/workflows/84-visual-marker/README.md) · [Wiring and limits](docs/VISUAL_MARKER_EXP.md)|
|First H3 generation / image-to-video|[Basic generation](examples/workflows/01-basic-generation)|
|Audio references, talking or singing|[Audio control](examples/workflows/02-audio-control) · [Native voice/emotion and Avatar](examples/workflows/36-avatar-voice/README.md)|
|Dual-model 4+4 or segmented long video|[Long-video workflows](examples/workflows/04-long-video)|
|Independent stages and external EAV/Relay|[Separated sampling](docs/MODULAR_SAMPLING_EXP.md) · [Veda eight-step](examples/workflows/63-veda-t2va/README.md)|
|FreeVideo FP8: 8+2 / true 4+4 (EXP)|[Six separated workflows](examples/workflows/72-freevideo-split/README.md) · [Independent runtime](docs/FREEVIDEO_EXP.md)|
|Nine Kijai originals, Acc8 Full/Pruned independent stages|[18 workflows](examples/workflows/71-kijai-experimental-split/README.md) · [Model pairing](docs/KIJAI_EXPERIMENTAL_LORAS_EXP.md)|
|External continuation, visible-face masks, no-face bypass and curve stages|[RADAR workflows and limits](docs/RELEASE_1.89.0.md)|
|Visually arrange shots, media and audio|[Obsidian Director](examples/workflows/39-director-console/README.md) (or use the dedicated `T8 Obsidian Director` entry in the ComfyUI left sidebar)|
|OpenVDN, FastH3, memory-saving patches|[Acceleration](examples/workflows/10-speed) · [Memory nodes](docs/H3_MEMORY_NODES_EXP.md)|
|Meridian camera and source-time editing|[Four-node workflows](examples/workflows/37-meridian/README.md)|
|First-pass dynamic preview / scoped cancellation|[TAEH3 preview](docs/TAEH3_SAMPLING_PREVIEW_EXP.md)|
|Enhancement / frame interpolation|[Topaz](docs/TOPAZ_EXP.md) · [DLSS-NR](examples/workflows/25-dlss-nr) · [DLSS interpolation](examples/workflows/29-dlss-fi)|
|More features and advanced setup|[All workflows](examples/workflows) · [Detailed guide](docs/README_DETAILS_EN.md)|

## Download models

Read each repository's setup and licenses. Do not overwrite different architectures with identically named checkpoints.

|Model / resource|Download and ComfyUI location|
|---|---|
|H3 transformer, Qwen, video/audio VAEs|`models/diffusion_models`, `models/text_encoders`, `models/vae`; [base setup](docs/README_ComfyUI.md)|
|TAEH3 temporal / 2D previews|[Taeh3-Comfy](https://huggingface.co/t8star/Taeh3-Comfy) → `models/vae_approx/`; the 2D file has a separate name|
|Meridian ConvRot INT8 + Omega1B512|[Meridian-Comfy](https://huggingface.co/t8star/Meridian-Comfy) → `models/meridian/`; Omega in `vggt-omega/checkpoints/`; code/assets and H3 VAE are separate|
|Semantic Bridge|[Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy) → `models/semantic_bridge/t8_compat/`|
|OpenVDN bundle|[Vdn-Minimax-H3-Comfy](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy); retain the repository layout|
|H3-World action LoRA|[Minimax-H3-World-Comfy](https://huggingface.co/t8star/Minimax-H3-World-Comfy); [setup](examples/workflows/26-h3-world)|

Meridian's DMD is already merged: **do not apply the same DMD LoRA again**. Omega is the original PT geometry checkpoint, not INT8 or ordinary VGGT. TAEH3 is approximate preview only, not a final VAE or audio model.

## Important limits

- Use the matching workflows for `Advanced` / `EXP`. Accepted examples are not universal quality, hardware, or long-window guarantees.
- Existing graphs need no forced migration. Unknown LoRA/Sage/Sol stacks receive warnings rather than blanket connection bans; see [patch-stack policy](docs/PATCH_STACK_POLICY.md).
- Exact voice identity, word timing, seamless joins and universal 16GB performance are not promised. Slight seam color changes remain a known limitation.
- Topaz and DLSS are independent postprocessing with their own runtimes and resources; H3 weights alone are insufficient.

For red nodes, audio errors and model compatibility, see the [detailed guide](docs/README_DETAILS_EN.md). [Report issues](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues) with the full error, workflow and Core/node versions; remove credentials and private media.

## T8 links

|Platform|Link|
|---|---|
|Bilibili / YouTube|[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/)|
|API / online apps|[API (affiliate)](https://api.seedance.nz/sign-up?aff=5f4w) · [RunningHub (invite)](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)|
|ComfyUI bundle / model drive|[ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Models](https://pan.quark.cn/s/c9c267081fbf)|
|Models / nodes|[Hugging Face](https://huggingface.co/t8star) · [Node GitHub](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)|

## Licenses and documentation

Node code: [GPL-3.0-or-later](LICENSE). No weights are included in this GitHub repository. Models/external assets retain their own licenses: Meridian weights use the MiniMax H3 Community License; Omega separately uses the FAIR Noncommercial Research License. Conversion grants no additional rights.

[Changelog](CHANGELOG.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Feature list](features.json) · [Homepage policy](docs/README_POLICY.md)


---

## FILE: docs/AUDIO_REFINE_EXTERNAL_EFFECTS_EXP.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/AUDIO_REFINE_EXTERNAL_EFFECTS_EXP.md)

# Audio Refine：独立尾采效果接口（本地 EXP）

这组新增接口接在已有 Audio Refine Setup 和 Stage Bind 后面。旧十种配方、旧节点、原二十张冻结／冷恢复工作流不替换；原来的 NOISE、SAMPLER、SIGMAS、联合 AV 和视频 0／音频 1 遮罩继续使用。没有新增隐藏采样器，也不重跑冻结的视频生成阶段。

## 接线

```text
原 Setup → 原 Stage Bind ── NOISE / SAMPLER / SIGMAS / stage_latent ───────┐
                 │                                                    │
                 └→ External Effects Context → [外置 Relay] → Stage EAV│
                                                positive ─┐       │    │
                                                          └→ Effects Guider
                                                                 │    │
                                                                 ▼    ▼
                                                       SamplerCustomAdvanced
                                                                 │
                                             Stage EAV Audit → 原 Stage Audit
                                                                 │
                                                原 Delivery Audit → 原 Quality Gate
```

- `MiniMaxH3AudioRefineEffectsBindEXPT8` 接原 Stage Bind 的 `stage_boundary` 及六个采样输入，输出独立 MODEL 和 `T8_STAGE_CONTEXT`。不改采样参数。
- 现有 `Stage EAV Config`／`Stage EAV Apply`／`Stage EAV Audit` 分别保留为画布节点。Apply 的 SIGMAS 和 AV 必须来自同一个尾采边界。
- `MiniMaxH3AudioRefineEffectsGuiderEXPT8` 接效果后的 MODEL、上述 context／SIGMAS／AV。`positive` 不接时沿用原 Setup 条件；接时检查原媒体／时间线及 Relay 的 MODEL–CONDITIONING 配对。输出连接原 Core `SamplerCustomAdvanced.guider`，CFG 保持 1。
- 可在未绑定 Relay 的尾采 MODEL 上使用现有独立 Relay Plan／Conditioning，再将其 MODEL 接 EAV、positive 接 Effects Guider。Relay 新建的空 AV **不能**替换冻结的尾采 AV。
- 原本已有 Relay 的两种配方，继续修改冷恢复图中独立的精修 Relay Plan；不要向同一 MODEL 重复叠加两个 Relay Conditioning。任意后置替换已装 Relay 的便利节点尚未实现。
- 长视频必须连接实际 `segment_index`／`context_frames`，首段为 0／0；续接上下文支持原有 5／22／39。接口核对实际 motion keyframes 与原长视频修复合同，不重算时间坐标或续接音频。

## 边界与兼容

ABSTAIN 仍为空 SIGMAS、零模型调用、原 AV 直通；即使 EAV 选择 `apply_exp`，也不会把 ABSTAIN 升为采样。效果节点不会接受候选音频，原 Quality Gate 默认保留原片，人工听审和明确选择仍必需。视频的逐值回锁仍由原 Quality Gate 完成，不把视频零遮罩当成采样输出逐位不变的保证。

EAV 的原定义是对目标视频 attention 输出施加 FETA 增益。它在联合 AV 尾采中的执行不等于“专门音频增强”或已证实音质更好。进度仍取 `1 - video_sigma`，不会按四步尾采重新归一化；原 `g_hard_limit` 保护保留。很短的测试网格可能触发保护，不会自动钳制增益或关闭报错。

未知用户 LoRA、attention delegate、hook 保留；实际覆盖不足按原 EAV 审计标为未验证。这组接口不授予可移植 StageResult 缓存资格；恢复继续走已有最终视频 AV 的显式 checkpoint 路径。

## 当前验收范围

已取得三类签名计划及十种尾采配方的 tiny 原生 H3 CPU 效果调用证据，包括长视频首段／续接、Relay＋EAV、disabled／report-only 数值不变、ABSTAIN 零调用、错误 AV／遮罩／SIGMAS／配对拒绝。CLIP／VAE 和权重使用标明的测试替身，这不是预训练 GPU 成片或音质验收。

`tools/build_modular_audio_tail_effect_workflows.py --matrix --output <新的artifacts目录>` 生成26张私有效果草稿：十张 EAV、另外八种配方各一张 Relay-only 和 Relay＋EAV。逐一校验来源图，不覆盖公开文件。已有 Relay 的两图保留其独立 Plan。Relay-only 不包含 EAV 节点；所有新增 Relay 都保留原全局提示词，默认空事件、`report_only`、`video_only_paper`，不会自动添加新剧情。需要实际时间路由时，填写至少两个局部事件，再显式选择 `apply_exp`；独立 Query Route 可选 `joint_av_exp`，它是实验性联合音画扩展，不是论文质量保证。Relay 的空 AV 输出不连接尾采。

这26张图已增量保存在 [59目录的effects子目录](../examples/workflows/59-audio-refine-split/effects/README.md)，旧20张不变。与交付图仅有 CRLF／LF 换行差异的候选26/26原生画布可见编辑／另存／刷新重开通过，未排队；八种新增配方的已保存图经 tiny Core 实执行，验证独立 Relay 的真实调用及 EAV 组合，八条还各在全新 Core 进程仅运行尾采。画布连线序列化、CPU尾采、真实权重及人审是不同资格。S26 尚未完成。

### 原尺寸 PDD8 原生画布完整／冷恢复补验（2026-09-30，本地）

新增一条独立链使用原 PDD8 配方尺寸：124 帧、736×416、24fps，约 5.167 秒。真实 Chromium 画布点击 Queue，完整执行 8 步视频生成＋4 步尾采并显式写入原生 AV checkpoint；全新 Core 严格读取相同路径／manifest／文件 SHA，仅执行 4 步尾采。模型、原参考图、采样器、sigma、3D／音频政策与原公开 JSON 保留，没有搬入历史 tiny 完成阶段。

组合尾采保留原 EAV 窗口、tau4 和 report-only 默认，四次认证 forward、200 次配对 Relay 调用已观察。分别保存原版、未经接受的候选、默认交付三路完整 H.264＋AAC；三路解码画音签名、候选／默认交付的 AV 张量内容都在完整／冷恢复间一致，原冻结文件不变。Quality Gate 仍为 `ABSTAIN_HUMAN_REVIEW_REQUIRED`，默认交付的原视频和音频逐值不变，解码画音与原版相同。候选的解码画面和音频与原版不同，不能把零视频遮罩／小数值差当成候选视频逐位不变或质量通过。

十种原配方的四类效果选择另有当前实际 Core／ClipProj 0.1.13 schema 准备检查：40 组完整／冷图的具名边、独立编号和 DAG，加 8 项错误连线／质量自动接受等拒绝门。它不加载权重、不 Queue，不替代剩余配方的原尺寸成片。其余路线、任意 `apply_exp` 强度／窗口、多素材和人工听审继续单列；旧失败证据保留。以下两条小尺寸结果仍只具有原来的历史资格。

### 十种原配方的分项 full／fresh-cold 补验（本地，2026-10-01）

十条原配方目前各自取得真实 Chromium 画布 full/fresh-cold 三路完整 AV 收据：PDD8、PDD 4+4、Phase2 base/same、Turbo4/8、EAV Turbo8、H3 learned upscale、Relay 与 long Relay。它们沿用各自原尺寸、步数、sigma、VAE、音频和交付区间；PDD 4+4 原来只有22帧，long Relay 原120帧，其余124帧，不把短配方改成五秒。冷图只执行原尾采，不重新生成已冻结的视频。各条候选／默认 AV tensor 和原片／候选／默认的完整解码 RGB/PCM 都在 full/cold 间一致，默认选择仍严格保留原片，没有候选被接受。

这些分项发生在不同冻结源码 epoch，仅覆盖独立外置 Relay apply 加 EAV report-only，不是统一当前源码的全效果矩阵、任意 apply 强度／窗口、更多素材或音质通过。Relay 这次明确预留4GiB，原512MiB余量和16GiBcommit保护未降低；Turbo8 保留原2.5GiB和stock保存器。旧资源／codec失败和证据保留，codec根因尚未证实。所有 `apply_exp` 和人工完整看听仍各自验收，不把候选视频零遮罩或成片可解码当作质量保证。

### 历史两条 PDD 小尺寸冷尾采

2026-09-28 补充了 PDD8（128×128）和 PDD 4＋4（192×192）两条来源的验证。两份源 AV 均来自先前已经核验的真实 PDD 视频生成及 checkpoint；本轮不重跑这些前段。每次在新隔离 Core 中，核对公开分离图与当前 Core 序列化结果，严格读取原冻结文件，仅执行四步尾采。模型、Qwen 编码器、VAE 使用本机真实权重，没有 tiny loader 或资源快照替身。

两条都使用独立 Relay Plan、`joint_av_exp`、`apply_exp`，EAV 效果窗口明确为 0–1。初次 `tau=0.25` 的 200 次 FETA 测量全部为增益 1，只证明路由执行。随后以 `tau=4` 补测，并要求观测到非单位增益：两条均记录 4 次模型前向、200 次 Relay attention 和 200 次 FETA 测量，最大增益分别约 1.2002／1.2482，原 `g_hard_limit=1.5` 保持不变。这不是全部默认配置或增强质量认证。

每次分别保存原片、未经接受的候选片、默认选择片，三份 H.264＋AAC 媒体均完整解码，均为 22 帧／24fps（约 0.917 秒）。默认 Quality Gate 保留原片，两者解码后画音 SHA 相同；候选音频变化且有限，两条候选的视频解码 SHA 也与各自原片相同。没有自动接受候选音频。

这次是隔离 API 的真实权重执行证据，**不是新增画布点击执行，也不是 5 秒成片**。公开工作流和原采样控制不变。原尺寸、多素材、十配方完整生成与冷恢复对照及人工听审仍需继续，不能以这两条小尺寸尾采代替。

### 已有 Relay 两配方的完整／冷恢复 CPU 对照

普通 Prompt Relay 与 Long Video Prompt Relay 的已保存效果图，另有 tiny CPU 整图对照：保留上游已经绑定的 Relay，不新增第二个 Relay；将原冻结图的最终 AV 直接接入效果尾采，完整执行 8＋4 步，再在全新 Python／Core 进程严格读取同一 checkpoint，仅执行四步尾采。覆盖 EAV disabled／report_only／apply_exp；普通配方还覆盖真实 rank-1 测试 LoRA 的 0／0.7 强度，共九种组合。

九组合的原始／候选／默认选中 AV 哈希，以及三路完整解码媒体哈希，均在完整图与冷恢复之间相同。默认 Gate 保留原片；长视频续接输出也保留原 AV，磁盘 context 与冻结文件未被尾采改写。长视频媒体比较采用相同的原 Output Trim 区间，不能把裁切前音轨与裁切后音轨混作同一对照。相同权重设置在完整图内由 Core 正常缓存共享，执行步数由真实 Relay 调用计数核验，不能以“只载一次模型”推断漏跑前段。

模型为固定随机小网络，CLIP／VAE／资源信息明确用测试替身；VAE 解码随 latent 内容变化，非恒定黑帧／恒定音轨。这里只证明 CPU 状态与输出恢复合同，长视频仅首段 segment 0。它不替代这两配方的真实权重、多段续接、原尺寸或人工画音验收。


---

## FILE: docs/AVATAR_PROGRESSIVE_EXP.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/AVATAR_PROGRESSIVE_EXP.md)

# 录音驱动 Avatar 渐进采样（v1.85.0 EXP）

## 正式发布状态（2026-09-18）

用户最终确认全部指定样片通过并要求发布。正式示例见
[36-avatar-voice](../examples/workflows/36-avatar-voice/README.md)，源码在正式`h3_t8`目录。
下面是原始开发／失败和修复记录；旧“未发布／待人审”不是当前状态。
首期仍限独立单段；NA接缝、未知组合和Core同进程驻留问题不被这次验收升级。

2026-09-18 补核此前漏读的真实五组反馈：精确绑定的普通8步与新73帧Avatar4+4
样片画面／音频／口型均“正常／可接受”。接缝NA保持，不扩展为长视频或所有组合通过。
见本地`human-feedback-20260918-v1/analysis.json`；这补足下方历史“仍待人审”的固定配方门禁，
不改写既有失败或TAE同进程Core驻留边界。尚未发布。

新增独立入口 `MiniMaxH3AvatarProgressiveEXPT8`。原渐进节点、旧工作流、已验收双采长视频保持不变；本功能还未发布。

人物首帧和录音接原 `MiniMaxH3AudioConditioningT8`：选择 I2VA、`audio_mode=lock_source`，录音接 `drive_audio`。同一个 DualClock Setup 的 MODEL／sampler／sigmas接新入口，默认 LOW4、原 learned3D 放大、HIGH4。可选独立 HIGH MODEL／LoRA沿用已有接口。

音频 latent 由录音编码，零 audio mask 在两阶段中锁定原生音频状态，HIGH重启噪声和clean anchor明确分离。保存时明确接原录音；不要用VAE重建音轨代替它。这不是单纯贴音轨，也不是按该音色生成新台词的声音克隆节点。

首期只支持独立单段，没有空间tile／长视频资格。无需新的 Avatar 专用权重；使用匹配原生H3、现有EMA B加速LoRA、CLIP、video/audio VAE和原 learned3D latent upscaler。用户的注意力／LoRA组合保留，但未实测组合不自动获得质量／提速认证。

## 示例与帧数

开发候选是人物2:3首帧、512×768，LOW256×384，73帧／24fps。AudioWindow设置时长73/24秒、warmup/cooldown=0，`ensure_minimum_context=false`，不把短测扩成124帧。这个选项只改新图，原节点默认不改。换图按自身比例选择32像素对齐尺寸，不直接套用2:3。

录音需覆盖成片；过短部分补零，过长通过窗口裁切。首字不要被过大的静音／淡入削掉。角色身份、口型、表现和声音仍需人审。

## 资格边界

原phase1入口现在另有专门CPU effects检查：I2VA完整录音零audio-mask条件下，EAV独立和EAV+joint Relay两组合实际LOW4/HIGH4，核对两阶段mask证据／完整时钟不重启、Relay各4次路由、report_only与关闭逐值相同、apply产生视频差异而录音锚不变，原输入和用户wrapper保持。已列入最新618范围回归；这是tiny Core+测试放大器的接口／数值锁定资格，不是新预训练模型组合GPU、人审或提速声明。生产Avatar与已验收接缝代码未改。

2026-09-18 v6真实普通8步与Avatar LOW4→原learned3D→HIGH4均完成，
取消LOW后同进程恢复4+4、收到TAEH3真解码帧后取消再恢复也完成。
独立完整媒体审计逐帧解码三片，确认73帧、512×768、H264、视频PTS递增，
音频有限且非空，媒体原SHA不变。主Avatar恢复片110.375秒，
源录音最大误差1.19e-7是数值锁定检查，不代替口型／画质验收。
审计见 `avatar-v6-media-audit-v1/report.json`。

附加预览开关实验最终video latent差异2.1114254、audio差异0，整个v6
终态因此失败，不能把三片的媒体完整性写成整轮质量／只读通过。
后续OFF/OFF对照也有差异，发生在任何预览解码前；同一已加载模型立即
重复三次原生前向则AV逐值相同。具体跨采样权重驻留／恢复原因仍在定位，
不据此更改旧采样、LoRA、注意力后端或已验收接缝。

追加OFF/OFF诊断已确认首步参数相同而10处实际Core权重读取变化，其中9处在BF16/INT8之间切换。变化发生在任何TAE解码前；详见[预览诊断边界](TAEH3_SAMPLING_PREVIEW_EXP.md)。这不是同进程恢复已修复的声明，也不使旧失败通过。没有为新入口修改旧Core驻留或量化数学。

新独立进程关／开TAEH3完整4+4，八次前向与最终AV全部逐值相同；ON三次真解码。
最终ON latent已独立完整解码73帧512×768 H264与明确原录音，完整AV审计通过，
新候选SHA ec1bb42b40038dd940436c8e9e1cedf10369c2cadf9b34709caab8c697ca726d。
见 `taeh3-fresh-pair-audit-v1.json`、`avatar-fresh-decode-gpu-v1/terminal.json` 和
`avatar-fresh-media-audit-v1/report.json`。不覆盖旧同进程恢复边界，不替代口型／画质人审。

范围CPU回归和实际Core保存图验证已通过；六份前端EXP图在正式节点 `examples/workflows/36-avatar-voice`。
真实预训练模型GPU终态及失败读取对应artifact，不是仅凭节点变绿确认。早期v5两片输出124帧，最后严格73帧断言失败，证据保留，不能算73帧验收。

节点继承原渐进采样器的资源保护。独立测试Core使用6GiB权重驻留保留预算，目的是让Core多卸载权重，未修改用户服务／全局设置；这不是所有硬件都必须设置6GiB的承诺。真实OOM或输入错误正常传出。

首次使用本地新Python需要用户自行重启自己的ComfyUI；测试不会替用户重启、清队列或改画布。机器资格不替代最终人审，不表示任意素材／显卡／模型都通过。


---

## FILE: docs/CHUNKED_STANDARD_4PLUS4_EXP.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/CHUNKED_STANDARD_4PLUS4_EXP.md)

# 非 PDD 标准 4＋4 时间分块（Advanced EXP）

这是独立的新执行合同，不修改已验收的双模型长视频循环，也不改变旧 Chunked Plan 的缺省行为。
必须使用包含本次实现的新代码并重启 ComfyUI；旧版本只有 JSON 没有实现，不能运行此模式。
本次实现随 GitHub 源码更新，不另发 Registry 版本；历史实机证据保持原范围。

## 用户自定义组合图（额外发布，EXP）

用户要求将 `2026-09-17_H3_Chunked_4plus4_接线修正版（低显存双分块双采样）.json`
一并公开。它不是下面纯净0.401MP／8秒图的另一份同参数已验收成片。

- 保留 Ref2VA INT8 底模、所选 FL2V Turbo alpha8 LoRA 强度0.7，以及
  `T8 ChunkFFN → KJ Memory Efficient Sage → ModelAttentionBackend → SolAttnMiniMax → LoRA`
  的自定义链。跨任务底模／LoRA搭配和组合执行效果由使用者判断，不硬性禁止，也不认证全部生效。
- 需安装 `ComfyUI-KJNodes`、`ComfyUI_Comfyroll_CustomNodes`、`comfyui-various`、
  `ComfyUI-Easy-Use`，及与本机版本匹配的 Sage/Sol依赖；本项目不安装这些第三方组件。
  新 Core 自带 `ResolutionSelector`、`ComfyMathExpression` 与 `ModelAttentionBackend`。
- LOW 尺寸来自 ResolutionSelector：16:9、0.4MP、32像素倍数，实际864×480。
  Plan `scale_by=2` 算得1728×960，面积约1.659MP，而不是整条输出0.4MP。
  修改LOW面积或倍率时看 Plan 的INT输出，HIGH宽高已连接，不用手改隐藏旧值。
- 时长来自 `JWFloat=5` 与17n+5对齐公式，得到124帧／24fps≈5.17秒；
  `temporal_chunk_frames=136` 下只有一个二采窗，一采4＋二采4＝8次前向。
  输入8秒会得到192帧／两窗／12次前向；增加视频时间分块与模型FFN分块是两件事。
- LOW／HIGH实际都消费 `CR Prompt Text` 的同一个文本；面板中旧缓存文本不代表另一套生效提示词。
  改说话内容只改这个共享提示词。图片 `10A.jpg` 是占位文件名，需用户自行选择有使用权的参考图，
  本次不上传人物图片、模型或任何私人媒体。
- 保留原参数与接线，仅纠正旧备注、语义相同的Ref2VA控件值和当前Core保存控件序列。
  未连线的第二个ResolutionSelector没有作用，可保留作辅助控件。

CPU实际Core API及外部原类schema检查不等于执行第三方patch／GPU内核，也不代表完整音画合格。
外部组合可能覆盖另一节点的优化；看运行警告、实际覆盖与成片，不依据节点串联就认定全部已生效。

## 与旧路线的区别

| Plan 合同 | 一采 | 二采 | 最终音频 |
| --- | --- | --- | --- |
| 原 Plan 默认 `video_only_legacy` | 沿用旧输入 | 沿用旧视频分块执行 | 原输入音频透传 |
| 旧低 sigma v3/v4 示例 | 完整8步至0 | 3步、denoise0.30 | 原输入音频透传 |
| 新 `standard_joint_4plus4_exp` | simple8前4步的denoised_output | 原始后4步，逐窗联合AV | 二采完成音频 |

新合同只开放完整空间画布 `full_frame_safe` 和 T8 双时钟 Euler，不开放独立空间tile。
Plan 原有15个参数位置和默认行为保留，新字段追加在末尾，不增加节点ID或改注册顺序。

## 正确接线

底模 → 正确alpha8 Turbo LoRA → Dual Clock的MODEL → Learned Two-Pass Parity Plan。
保留 `base_steps=8 / coarse_steps=4 / refine_steps=4 / shift_video=12 / shift_audio=3`。

- Parity `coarse_sigmas` → 第一采样器；第一采样器 **denoised_output** → Chunked Upscale `latent`。
- Parity `refine_sigmas` → Chunked Upscale `sigmas`，不另建低denoise调度器。
- Parity `report_json` → Chunked Plan `parity_report_json`，选择 `standard_joint_4plus4_exp`。
- 同一Dual Clock的MODEL和SAMPLER接Chunked Upscale；执行器按每窗放大后的AV形状重新绑定SAMPLER。
- HIGH Conditioning → Chunked Upscale `conditioning`。LOW/HIGH必须同提示词、媒体、首尾图片、时间长度，只改空间尺寸。
- 第一阶段和第二阶段连接独立RandomNoise。二采只生成一次完整目标AV噪声场，再按全片绝对坐标切片。
- Chunked Upscale的LATENT → AV Decode；使用Decode输出的二采音频保存，不旁路一采中间音频。

二采video sigma固定为 `0.9035, 0.8, 0.6316, 0.3158, 0`，恰好4个Euler区间。
Report用于验证配方与时钟，但不能证明外部用户图真的执行了一采4步；必须同时检查实际采样器接线和日志。

## 倍率、尺寸与 INT 输出

Chunked Plan 现与普通学习型放大节点共用 `learned_upscale_geometry`，不维护第二套尺寸算法。
除原手填模式外，可选择 `size_mode=scale_by`（倍率）或 `target_megapixels`（目标面积）。

- 一采 **denoised_output** 同时接 Plan 的 `source_latent` 与 Upscale 的 `latent`，Plan 从真实视频 latent 读取原始宽高，不从 HIGH 条件或手填目标推测。
- 倍率模式设 `scale_by=2.0`，例如源448×224得到896×448；`target_width/height` 在此模式不参与计算。
- Plan 新增 `width`、`height` 两个 **INT** 输出（原输出0/1不变），连接 HIGH Conditioning 的宽、高输入。先将其 `width/height` 控件转换为输入；最终二采条件与 Plan 使用同一计算结果，避免另行手填尺寸。
- 倍率范围与普通学习型节点一致：1～4。按原比例选取32像素对齐尺寸，并检查不缩小、任何单轴不超过4倍、非等比误差限制；不是将宽高分别随意取整。非整数倍率以输出 INT／`report_json.geometry` 为准。
- `target_megapixels` 按原比例与目标面积计算；`target_dimensions + preserve_source` 按手填面积保持源比例，`honor_dimensions_exp` 才按手填宽高分别对齐并检查 `max_anisotropy`（默认1.05）。

旧图仍默认 `target_dimensions`，不接 `source_latent` 时保留原手填尺寸及旧Plan内容；因此旧图无需额外连接源。
倍率／面积模式必须连接源。执行 Upscale 前会复核其输入的源尺寸与 Plan 记录相符，防止错误源或目标尺寸漂移。
输出仅为像素宽高，不携带 LATENT；Plan 不保存源张量，也不改变音频、sigma、步数或时间分块。
`tile_width/height` 不是最终画布，标准4＋4的 `full_frame_safe` 不用它们分割空间。
最新正式示例已连好源与 HIGH 宽高，调整 LOW 尺寸或倍率无需再维护 HIGH 的第二套尺寸。

## 时间分块的目的及计算量

3D学习型放大器先对整个联合输入的视频部分只放大一次，保持音频不变。
随后高分Transformer只看到当前时间窗，而非整段时间轴，旨在降低该阶段激活峰值。
它不是模型内部的MLP/FFN分块；无需因此追加那两个MODEL分块节点。
整段高分latent、噪声场、已完成输出、3D放大及模型权重仍占内存，不保证16GB安全、一定更快或更清晰。

本例448×224 → 896×448（0.401MP）、192帧／24fps＝8秒，窗口136帧、重叠34帧。
H3的5-token／17-frame网格产生两窗：[0,136)、[102,192)。
共享一采4次 + 每窗二采4次 = **12次模型前向**。
“4＋4”是每个区域的两阶段配方，**不是多窗整片总计8次**。
单窗才是整片4＋4共8次；缩窗增加重叠计算和独立预测轨迹。

## 接缝、音频和保护措施

后窗读取前窗已经完成二采的重叠video/audio latent，并将两路该区域noise_mask置0作为只读上下文。
执行后严格检查该前缀逐值相同，拼接时保留前窗结果，只追加后窗的新尾部。
原生Euler的最后一次float32加减本身可能产生几个ULP的误差。新路线在检查前只允许
`8*float32_eps*(1+abs(source))` 的逐元素界限，并精确恢复零mask值；大于该界限直接报错。
部分mask及可编辑值不改写。每窗报告恢复前最大误差和受影响值数，不靠全局大阈值隐藏真实漂移。
没有latent端点平移、RGB/latent叠化或音频增益，也不会把尚未完成的一采音频硬锁为全片最终音频。
全局噪声只生成一次；源latent不原地修改；原输入两路合法noise_mask继承；最后音频padding完整保留。

输入NaN/Inf、错误HIGH参考尺寸、错误sigma、错误双时钟、未知时间元数据、分块时间缺口或只读区域变化会明确拒绝。
执行过程中继续响应取消，不引入持久阶段缓存；本节点没有 `resume_existing` 功能。
已验收长视频循环中的22/39帧上下文参数并未改动；此节点使用自己的17帧网格overlap，不要将两个参数等同。

`anchor_strength`、`overlap_blend`和空间tile参数仍为旧合同保留，在新标准合同中不参与拼接。
上下文由 `temporal_overlap_frames` 控制；多窗要求非零重叠。不能靠增大重叠承诺自动消除内容漂移。

## 验证范围

必须区分CPU机制检查、整模型GPU完成／严格解码、以及完整人工音画质量验收。
本次新合同仍为Advanced EXP，不能仅凭mask相同或可解码宣称接缝、人脸、后段或声音正常。
EAV、Prompt Relay、Sol、额外串联LoRA及未知第三方MODEL patch不在本次组合资格承诺内。
旧接受样例、旧v1/v2/v3/v4路线和双模型长视频接缝策略不迁移。

回归：`tests/test_chunked_two_pass_parity.py`、旧Chunked全局噪声测试、Learned Upscale测试，
以及 `docs/DUAL_MODEL_SEAM_FIX_20260913.md` 指定的5组旧双模型防回归测试。
独立GPU工具：`tools/run_chunked_parity_probe.py`；工具只拥有隔离测试服务，保留失败证据，GPU串行且有资源保护。
GPU工具显式追加KJ全局Sage selector；正式JSON保留普通MODEL路线，两者均用原生H264保存。
GPUv2最初从VHS模板提取采样图并替换为原生保存；最终正式JSON同步为实测原生保存。
节点ID变化不改变执行合同；普通MODEL路线未额外运行GPU对照，不能冒称后端完全相同。

首轮GPU在第二窗触发严格逐值mask检查，未输出成片，失败记录保留。
CPU调用真实双时钟Euler并让微型model返回完全精确的锁定x0，200个保护值中34个发生变化，
最大绝对误差为一个float32 epsilon（1.1920928955078125e-7）；这验证了舍入机制，
不以CPU机制复现替代整模型误差记录或成片审查。

## 本地实机结果（2026-09-17，非人审验收）

最终数值执行路径的0.401MP、8秒两窗GPUv2已完成：896×448、24fps、192帧，H264视频和AAC音频均为8秒，
视频／音频／联合流分别严格完整解码通过。执行耗时约173.55秒，周期性整卡观察最低空闲约4.34GiB，
不是精确峰值或进程独占显存；不是与整段二采的显存／速度A/B。
WebSocket记录为一采4次、窗口一4次、窗口二4次采样回调，sigma与声明一致，3D放大调用一次。
后窗video376320个只读值中23038个发生舍入差，最大2.3841858e-7；audio3648个只读值中524个发生舍入差，
最大1.1920929e-7。全部通过逐元素小界限检查后精确恢复，未改可编辑部分。
CPU指定回归164项通过，注册339个唯一节点，旧位置断言保留；没有跑全仓或所有第三方组合。
最终正式JSON另经真实Core CPU验证：22个执行节点、38条边、75个显式参数，API验证通过，
未创建CUDA上下文或提交推理。保存节点采用当前原生DynamicCombo四控件序列，与已验收V2示例一致；
API嵌套format/codec/encoding结构也已验证。该检查不是原生浏览器导入／导出证明。

证据位于本地 `artifacts/development/chunked-standard4plus4-20260917/gpu-0p4mp-8s-v2`。
整片SHA256：`730c778286eac9a16eb9a22ebf4bb2945c3062799e52d2e378ec7c0f0c2aa51e`。
前一轮失败仍保留为v1，不交付为示例。测试服务已退出，无拥有的残留子进程。
人物、口型、时间窗内容漂移及接缝还需要完整人审，不根据边界抽帧宣称普遍质量通过。

### 倍率／输出补齐的本地检查（2026-09-17）

本轮只复用现有空间尺寸计算并补接线，不修改数值采样器、3D网络、上下文、音频或接缝策略。
195项指定CPU回归通过，覆盖倍率、目标面积／尺寸、32对齐、旧Plan兼容、实际INT输出、尺寸漂移检查、
二采音频／12次前向不变，以及上述5组双模型防回归门禁。
更新后的正式JSON经真实Core校验：340个节点ID、22个执行节点／41条边／78个显式参数，
API及序列化检查通过；无CUDA、未排队，不是浏览器导入／导出或本轮新增GPU画质验收。
2×示例仍计算为896×448；历史GPUv2证据不改写为新版JSON的额外GPU实跑。
证据保存在本地 `artifacts/development/chunked-plan-scale-20260917`。

## 旧测试副本覆盖正式节点的报错

若出现 `build_chunked_two_pass_plan() got an unexpected keyword argument 'parity_report_json'`，
先核对堆栈路径及 `/object_info/MiniMaxH3ChunkedTwoPassPlanT8Advanced` 的 `python_module`。
2026-09-17用户实例8189实际返回 `custom_nodes._minimax_h3_release_test_20260913`，只提供旧15个输入和2个输出；
该副本的builder不含 `sampling_contract/parity_report_json`，新工作流参数被传给旧函数因此报TypeError。
这不是Sage／Sol或倍率导致的错误，不能靠删除Parity参数退回旧执行路线来“修复”标准4＋4。

正式安装仅保留一份本项目节点。关闭ComfyUI后，可将旧测试副本目录
`_minimax_h3_release_test_20260913` 改名为 `_minimax_h3_release_test_20260913.disabled` 或移出custom_nodes；
开头下划线不代表禁用，当前Core扫描仅跳过 `.disabled` 后缀。
然后重启ComfyUI、刷新网页并重新导入更新示例；核对API来源为正式 `minimax-h3-audio-T8`，
有尺寸模式／源LATENT以及 `width/height` INT输出后再运行。
初次诊断只读核验了用户实例。随后用户要求继续收尾，已将这个精确旧副本改名为 `.disabled`，
没有删除内容或移动核心目录。改名前8189已不监听；未停止、启动或重启用户服务，也未提交任务。
合并两个计算分块节点／实际KJ微型组合／Sol注册／倍率／Relay缓存及接缝回归后，322项CPU通过，
新一轮正式Core340注册和22节点／41边／78参数校验通过。下一步须启动／重启真实用户实例，
确认API已指向正式目录；不能把隔离CPU资格检查写成用户实例已重启或本轮整模型GPU通过。


---

## FILE: docs/COMMUNITY_ISSUES_20261117.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/COMMUNITY_ISSUES_20261117.md)

# 社区问题核对与 Core 0.45 兼容

2026-11-17，本轮开始时共有15个开放Issue和2个PR，已逐项核对。10个已交付修复的Issue与2个PR完成收尾；#12/#17/#18/#19/#26继续跟踪。以下区分代码修复、使用说明和外部依赖，不把尚未验证的画质或外部审核写成完成。

## 新发现并修复：FinalLayer显式网格

本机实际Core已更新为v0.45.0（`989d93d0d968e704c589ce0146bb18c77f6b50cc`）。它把原生FinalLayer的`shifts`输入改为`grids`。旧适配器将这个真实接口判为不完整，影响多关键帧、使用旧外部接口的block cache，以及相关HyperFlow路径。

现在根据实际调用接口分派：旧四参数接口保持原调用；旧`sigma/sample_sigmas/shifts`接口原样传入；新`grids`接口使用已安装Core自己的`pdd_grid`及实际输出头数，显式给出的网格直接保留。调用仍进入原layer或用户delegate一次，真实内部错误不捕获重试。最终v1.106.3覆盖旧用户层只有可选grids参数，以及完整旧shift接口加可选grids这两种负例；保留原四参数或shift调用，旧协议优先。均保留修复前真实失败及三Core修复后记录。未改模型、LoRA、步数、采样sigma、声音或缓存完整性要求。

本次当前Core相关用例按测试ID去重后836项通过、6项有明确前提缺失而跳过；这些用例跨数个定向批次核对，初次失败保留，不宣称整个历史测试树全绿。v0.43.0同组236项通过/30项跳过，v0.41.1为233项通过/33项跳过。旧Core环境的跳过包括未装外部BlockCache/KJ/Sol、缺少历史源码、旧Core不存在的网格/PDD接口，以及实际CPython3.13编译器元数据用例；元数据注入和负例在Python3.12执行。当前Core另跳过2.8GB可选原权重及两个用户运行服务的live入口，使用独立CPU/浏览器测试验证相应基础交互。

实际原生FinalLayer普通头与PDD多头均与直接调用逐位对照；分块投影仍与对应原生调用比较并保留原容差。Topaz183项独立回归通过。列宽刷新初次测试有一次时序失败，独立诊断快照确认65%正确保存，原始测试随后连续两次通过，未据此修改UI。未重新生成GPU成片，不把CPU数学/接口验证称为新完整模型质量验收。

## PR结论

- [#31](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/31)：问题已由现有`_sampling_class_members`覆盖，保留其更严格的元数据类型、非执行读取和真实方法检查。Core SigmaShift的整文件误拒也已在`96f0381`修复。该贡献作为已被现有实现覆盖处理，未重复合并较弱实现。
- [#29](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/29)：固定head `71a99d6a681aeb3d81e1f229aaf2ad78af5633db`仍会错误规整真实VFR。实际运行其两段纯计算函数，`[(900,512),(50,256),(50,768)]`会返回512；虽然总时长变化为0，各段真实速度仍被改变。此前要求的完整时序/音轨审计、受控取消worker和注册顺序等也未更新，因此拒绝合并此实现。感谢yangala的提案；不会把拒绝此PR写成已经交付时间轴修复节点。

## Issue处理结论

|Issue|结论与使用边界|
|---|---|
|[#32](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/32)|已修复匹配原生状态导出hook的非持久buffer误判；真实Dynamic加载、原始backup还原、伪造导出拒绝均有回归。仅换run_name不能修复旧代码。|
|[#30](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/30)、[#24](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/24)|已有v1.98.2的显式software/CUVID选择，视频/音频探测、PCM、增强和最终解码均传递；保留Main10帧级位深审计。CPU模拟及原有官方程序验证不冒充报告者机器的新完整增强实测。|
|[#23](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/23)|Topaz已有独立NVML读取器，不再套用DLSS的单GPU门禁；模拟两卡选择及越界清理覆盖，不声明实际双NVIDIA硬件验收。|
|[#28](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/28)|已有`/api/`前缀及`ui-modules`路由修复，更新T8后需重启后端并刷新页面。|
|[#22](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/22)|LoRA行选择、LOW/HIGH有序链、取消/应用及项目保存恢复已有修复和独立浏览器回归。时间轴等建议与选择器bug分开；定制分发遵循LICENSE及第三方NOTICE。|
|[#27](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/27)|已缩短示例名并保留JSON内容；旧名见filename-map.tsv。相对路径140、工作流名96的门禁继续执行。|
|[#25](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/25)|相同MP不是固定尺寸承诺。把放大器width/height连到HIGH Conditioning，再接Reconcile；精确尺寸用target_dimensions。保留旧图默认，不让HIGH独立重选另一套宽高。|
|[#21](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/21)|已有独立HyperFlow Loader/Plan/Sampler及工作流；Flash也是独立消费者，不当普通LoRA加载。组合资格按各示例验收范围。|
|[#20](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/20)|已有native慢/快ByteLevel兼容；仍核对实际token与完整提示词字节，不替换成任意tokenizer。|
|[#19](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/19)|Registry激活由官方处理，上传与Active/installable分开核验；保留公开问题继续跟踪当前目标版本。|
|[#12](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/12)|上游前端仍未正确归一化Registry CUDA分类值，跟踪ComfyUI_frontend#15364及已有PR#15468。保持合法classifier，不修改用户安装的前端资产。|
|[#17](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/17)|MPS/Ref2VA声音前导仍需原工作流、参考和输出FLAC确认；trim_exact_target是有完整ASR匹配条件的生成后裁切，不是模型级预防。|
|[#18](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/18)|原生joint双采和时间分块已有实现；空间tile的细节/接缝质量仍未获得通用资格，不能凭可执行就宣称达到全画幅质量。保留特性跟踪。|
|[#26](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/26)|原云端部署、DeepCleaner合法原包和私人参考素材仍需来源方提供，不能从本地节点代码补造。|

更新应针对T8自定义节点源码并重启对应后端；仅刷新浏览器或只更新ComfyUI本体不会载入Python修复。本次维护不修改或重启用户正在使用的服务。


---

## FILE: docs/CORE_COMPAT_20261031.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/CORE_COMPAT_20261031.md)

# ComfyUI新旧版本兼容与Issues／PR核对

2026-10-31，本机ComfyUI `7c4390a6f70cd8eb41014d6af52969397a752fc4` 与当时上游HEAD一致。另用本地Git中的原始源码检查了v0.43.0（`426f5ee`）和v0.41.1（`45bf90d`），没有修改用户Core、安装包或下载模型。

## 修复的两处兼容问题

**原生SigmaShift被整个Core文件哈希误拒。** 三个版本的`MiniMaxH3SigmaShift.execute`实现相同，但同文件新增其它节点、文件换行或辅助代码更新会改变整文件SHA。之前会提前报`Core SigmaShift source is outside the audited sampling factory`，影响组合采样、保存／恢复等路径。

现在继续核对实际Core来源、类和父类、作用域、执行函数的精确源码，以及按当前文件重新编译后的实际字节码。整个文件的真实SHA仍记录在缓存身份中，只是不再作为“是否允许这个未改变函数”的判断。函数改变、执行方法／hook替换、字节码与文件不符仍不能借用可移植缓存资格。没有更改sigma、采样步数、音频或用户LoRA。

**显式导出非持久buffer的状态字典被误判损坏。** 对应[#32](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/32)的错误类型：`MODEL nonpersistent buffer unexpectedly appears in persistent state`。部分原生／自定义`state_dict`导出hook会明确加入已注册的非持久buffer。动态加载改变dtype后，不能仅凭“它出现在导出字典”就拒绝。

现在只对确实匹配已注册live buffer或原始backup内容的导出项，还原原始字节并保留该导出键；旧版没有导出的非持久项依旧不伪造键。未知路径、形状错误、NaN及伪造导出内容仍报错。回归使用真实Core Dynamic加载器和原生VAE的状态导出hook，核对加载前后身份、live对象和backup均不被修改。

## 语义桥验证边界

原有语义桥154项基础CPU检查在改动前即通过；复现的失败在其组合使用的原生采样身份路径，不是已证实的Bridge权重计算错误。修复前组合检查18项失败，修复后同批193项通过。旧v0.43.0、v0.41.1所选检查各227项通过。各批均另有3项要求真实Python 3.13解释器的编译器元数据检查跳过；当前运行时为Python 3.12.10，已有注入式元数据正反例仍执行，不能宣称做了真实3.13运行验收。

最终本机Core综合回归为462项通过、3项上述跳过；独立源码门禁批次25项通过。两批存在覆盖重叠，不相加成独立功能数。旧Core另各完成2项原生Bridge到AV采样检查。所有本轮运行均未初始化CUDA。

额外检查原生Bridge应用节点到SigmaShift、Guider和完整AV采样的文字／参考图两条tiny Core路径；本机已有Original、BUNNY、Comic Combat和Wushu四份实际Bridge权重也完成了CPU原生AV衔接。H3去噪器是明确标注的随机小模型，因此这些检查证明软件接线，不代表新增完整训练H3的GPU画质。用户原报告未提供完整trace，不能断言它与这两处复现完全相同。

旧节点ID／输入／默认参数、已审工作流、Bridge数学路径均保留。修改的是身份兼容与完整性恢复。新增有限源码后继单独绑定六份明确源码／CI文件和原始preimage；历史H21／H24／H27／Pika账本、采样调用站点和人工成片资格不重录。

### 多桥反馈的补充核查

用户进一步说明“老单桥正常，新多桥报错”后，额外执行了真正的组合路径，不以四份权重分别通过代替多桥通过：

- 在最新Core、v0.43.0、v0.41.1分别检查自动参数节点、链接图预检、Core V3执行分发、组合节点和Apply；每版所选多桥／Transformer／输入验证批次56项通过，其中新增6项完整多桥集成回归。
- 本机已有Original、BUNNY、Comic Combat、Wushu权重组成五种顺序：Original→BUNNY、Comic→Wushu、BUNNY→Comic、Wushu→Comic、四桥串联。每版Core各完成文字／参考图两种原生AV CPU衔接，共各10例。H3仍为随机tiny测试模型，只证明软件兼容。
- 最新Core另在RTX 4060 Ti上执行五种真实训练Bridge组合、FP32／FP16／BF16、CPU条件显式送CUDA及CUDA条件自动设备两种路径，共30例全部通过，输出有限、dtype／设备及原输入保持正确。本次仅加载Bridge适配器，峰值张量分配约46 MiB；没有加载完整H3检查点或生成新的验收视频。

在这些条件下未复现新的多桥专属异常，因此本次补检**没有猜测并修改多桥运行代码**。前面两处共享身份兼容修复已经包含在GitHub源码中；是否修复反馈者那一条错误，仍需其实际报错节点、完整堆栈及所选模型确认。原报告缺失不等于所有其它环境或模型组合已通过。

## 开放PR处理结论

- [#31 Python 3.13](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/31)：现有`_sampling_class_members`已覆盖，并额外检查元数据的真实类型及可执行成员。本次保留更完整实现，不重复合入较弱版本。PR正文提到的Core整文件哈希问题在本次另行修复。
- [#29 Topaz输入时间轴](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/29)：未直接合并。对其固定head `71a99d6`的实际代码，用CPU生成的合法MP4／MKV复现两处模式分发错误：CFR MP4显式`reencode`仍提前返回原片；合法MKV显式`reencode`仍在调用FFmpeg前被拒绝。新增节点还插入旧Topaz注册列表中间，未保持完整节点前缀；FFmpeg子进程缺少超时／持续取消检查和失败产物清理。容器表修复也需要更多时间表、音轨和编辑列表一致性验证。修正这些边界前不能把该PR视为已安全集成。

## 15个开放Issue的核对

| Issue | 当前结论 |
| --- | --- |
| #32 扩画状态字典 | 本次修复匹配的非持久buffer导出误判；原报告的具体完整工作流仍未收到复测。 |
| #30／#24 Topaz解码 | 已有v1.98.2显式software／CUVID修复和既有验证，等待原环境复测；不把所有Topaz异常归为同一根因。 |
| #23 Topaz多GPU | 已有独立Topaz NVML读取器；无真实双NVIDIA硬件的新验收。 |
| #28 导演台打不开 | 已有模块路由404修复；本次没有新原环境复测，不盲关。 |
| #22 LoRA选择无变化 | 已有LOW／HIGH独立列表和保存恢复修复；时间轴、许可与第三方工具建议不算同一bug。 |
| #27 路径过长 | 现有短文件名和Windows路径门禁仍保留；不重命名用户已保存文件。 |
| #25 放大尺寸 | 现有width／height直连及Reconcile契约；相同MP不保证两种算法选出相同尺寸，不能静默改旧默认。 |
| #21 HyperFlow | 已有专用加载、采样与工作流；不因此认证任意组合。 |
| #20 tokenizer byte contract | 已有慢／快ByteLevel兼容和精确token等价检查，不用任意tokenizer替换。 |
| #19 Registry | 本次官方只读API显示1.103.0为公开最新，目标1.104.0仍Pending。保持审核问题独立，不用GitHub推送或绿色Action代替可安装状态。 |
| #18 空间分块二采 | 能执行不等于无接缝；真实质量验收仍未满足，不能宣布完成。 |
| #17 MPS音频前导 | 仍缺少同平台的最终修复／试听确认，本轮不凭空宣称解决。 |
| #12 CUDA能力识别 | 历史桌面／Registry元数据识别反馈，缺原安装环境的新复测。 |
| #26 云端原版材料 | 外部云端组件、素材和准确部署版本无法凭本地代码补造，仍需原材料。 |

这些Issue的旧回复已与现有源码核对，没有批量关闭尚需用户、硬件或外部服务确认的条目。

更新时应更新**T8自定义节点源码**并重启对应ComfyUI后端；仅更新ComfyUI本体或刷新浏览器不会加载本次Python修复。本次维护不自动覆盖或重启用户正在使用的服务。


---

## FILE: docs/CORE_VDN_COMPATIBILITY.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/CORE_VDN_COMPATIBILITY.md)

# Core compatibility and VDN two-pass refinement

Local 2026-09-17 policy override for all owned nodes: LoRA, KJ Sage, SageAttention, Sol
and foreign callable owners are not admission bans merely because composition is unverified.
Warn, preserve/delegate, and report actual coverage. Real input/kernel errors and own
receipt/cache integrity remain enforced; opaque state requires a fresh execution identity.
See [patch-stack policy](PATCH_STACK_POLICY.md). Historical qualification below is not a
universal guarantee for newly selected combinations.

Version1.75.0 verification record, 2026-09-08. Outpaint remains paused and is not part of this delivery.

## Latest delivery checkpoint — 2026-09-08

The isolated Core/VDN candidate passed its complete CPU suite: **2435 passed, no failures/errors/skips**, 21 warnings. The report checks real isolated Core import paths and unchanged Python sources. The candidate starts from published ad43472 (299 nodes), adds two nodes at299/300 and excludes paused outpaint. It is not the main workspace's full-v7 result (2948 passes and15 still-failing paused-outpaint checks).

All four normalized T2VA/I2VA × VDN/native-H3 workflows have now completed a new actual native sidebar open, Save As and API-download roundtrip. Every execution input/edge matches the expected graph, and16 altered API configurations are rejected. The first tab's failed download attempts remain recorded; reopening the actual saved workflows in a fresh tab of the same browser restored downloads without changing security settings. Evidence: `artifacts/core-vdn-isolated-delivery-20260908/final-api-roundtrip-receipt-v1.json`.

The four JSON files and their bilingual `VDN_TWO_PASS.md` are installed in the main project `examples/workflows/10-speed` and actual `ComfyUI/user/default/workflows/MiniMax H3 T8/10-speed`. Source/user bytes match. Deployment preserves the exact tested JSON graph, adding at most a terminal newline. The user's older10-speed README was missing the existing OpenVDN installation section; that missing text was added after proving all pre-existing text matched. Post-install workflow regression:25 passes, no failures/errors/skips. Evidence: `local-workflow-install-receipt-v1.json` under the same artifact directory. Old single-pass workflows were not replaced.

The isolated candidate has now also passed a trained-model DynamicVRAM run: VDN8+4,320x192→640x384x39, full INT8 base, global Sage, official sparse installed after VDN, reserve5 and the registered isolated saver. The run started only after free VRAM reached14214MiB. It exited0; all mechanical/media/source gates passed, actual logs show DynamicVRAM enabled, and minimum free VRAM was2231MiB. Evidence: `artifacts/core-vdn-isolated-delivery-20260908/candidate-trained-gpu-v1/completion.json`, validation-report SHA256 `382023864ab46caa94e6dd66c5f839fad69e434ebbdbd93fd297003ff394c8de`. Candidate identity is bound by the isolated custom-node layout/config and actual Core import log, not an independent per-object Python-origin probe. The earlier main-project run (minimum2117MiB) remains separately recorded; neither run is an additional human quality verdict.

Both human feedback items are closed: the T2VA classical-music/“你在哪里” pair has normal music, speech and both-side lip-sync with a visual tie; the separate I2VA pair has no ambient noise and a visual tie. No preferred route or universal quality guarantee is inferred.

**Version1.75.0 validation complete within the stated scope.** This version has301 nodes and consistent release metadata. Full regression, trained-model integration, four installed workflows and an official Comfy archive with isolated301-node CPU import pass. The available hardware is RTX4060Ti16GB, not a physical4070TiSUPER or Blackwell certification. Outpaint stays paused. Historical pending descriptions below are superseded by these results; they are not an active to-do list.

**Later same-day update:** the native-H3 I2VA comparator passed at1024x512x73 with EMA B259 targets, native-branch ownership checks and2850MiB minimum free. Its actual first-pass decoded RGB and PCM match the historical VDN-refinement I2VA candidate exactly; final PCM also matches. `artifacts/vdn-native-i2va-comparator-20260908/pair_validation.json` binds both media/report hashes. The user reviewed this pair and reported “没有杂音，画面效果基本一样”: visual tie and no audible ambient noise. The separate `human_feedback_v1.json` binds the exact feedback to the page, pair receipt, media and reports without rewriting the original mechanical evidence. This pair contains no dialogue or music, so neither lip-sync nor music quality is accepted by this verdict. Subsequent sparse-hooks fixes changed runtime sources, so the18/18 matrix and full-v6 below are historical pre-fix snapshots, not a final baseline for the new changes.

**CUDA compilation update:** four serial small-model tests passed using the actual official TorchCompileModel node and inductor: native and multikeyframe paths, each with compilation installed before/after the VDN ownership guard. Session17200 exited0. Reports are bound in `artifacts/attention-hooks-core-compat-20260907/cuda-compile-inductor-summary-v3-20260908.json`. Real compiled regions/kernels, numerical agreement, error/interrupt recovery and ownership restoration were checked. Existing VS2022 developer-shell settings and Python UTF-8 were required; no software was installed and no global environment or Core source was changed. Graph breaks and local eager fallback remain (`fullgraph=False`); this is not trained-VDN branch inference, an all-compiled guarantee or a speed benchmark. Cudagraphs/allocator and broader complete-graph combinations remain open.

**Latest result: the current-Core two-pass input matrix is complete, 18/18 mechanical passes.** On 2026-09-08, session16067 ran only the five remaining pruned cases serially and exited0. All18 report/media receipts and current source/CPU-scope bindings were revalidated afterward. The previous13 were not rerun. New-five minimum free VRAM was3371MiB; the combined18 minimum was3294MiB. Evidence: `artifacts/vdn-two-pass-resume-20260908/completion_summary.json` (SHA256 `e1fcb787accf5adb668d5cb350f8228ac289c0ff2ec5eaab8cc0463012995655`). This completes the input matrix, not human quality acceptance or the remaining compatibility/release gates. No production Python changes or workflow installation were made in this continuation.

**Resumed at the user's request on 2026-09-08.** All 13 completed two-pass receipts were independently revalidated against report/media hashes, 17 runtime sources, Core sources/import paths and six input assets. The interrupted pruned/ref2va attempt20260907-222749 remains unchanged and has no final report. A new continuation under `artifacts/vdn-two-pass-resume-20260908` refreshes the full CPU baseline first, then runs only the remaining five pruned-input cases, strictly serially. No old completed case is rerun. The separate DLSS v1.74.1 patch changed six Python files since full-v5; each matches the published release workspace, and none changes the 17 recorded GPU-path sources. GPU launch still requires the existing 12000 MiB free-memory preflight; desktop applications are not closed automatically.

## What is fixed, and what is still being checked?

**Graph lifecycle follow-up:** four additional CUDA Graph cases passed with an explicit `torch.compiler.cudagraph_mark_step_begin()` in the probe. Each Chrome trace contains56 actual `cudaGraphLaunch` calls; numerical results, interruption recovery and ownership checks pass. Without that explicit boundary, both the short and repeated-warmup controls ran correctly but did not demonstrate replay. Production code was not changed to insert the marker, so default-workflow acceleration remains unproven. Graph breaks and skipped regions remain visible.

The real Core `PromptExecutor` also passed14 allocator lifecycle checks: native and multikeyframe small resident CUDA models, each normal → error → retry → interrupt → retry → error → retry. Each block observed a live native AIMDO graph handle. Core's own execution `finally` removed the active graph, module graph references and prefetch state; successful retries were bit-exact. This establishes the tested cleanup boundary, not VBAR eviction, nonempty offload-allocation replay, full-model memory safety or trained VDN execution. The first fixture attempt failed because AIMDO's lazy Linear weights had no checkpoint; the corrected fixture creates random resident weights before enabling AIMDO for all actual execution. Failed attempts remain intact.

All report/source and trace hashes are verified in `artifacts/attention-hooks-core-compat-20260907/cuda-graphs-allocator-summary-v1-20260908.json`. Fresh complete-suite `current-full-v7` finished:2948 passed,15 failed,0 errors/skips,21 warnings in934.82 seconds. Its independent `scope-qualification.json` verifies all664 focused Core/VDN cases pass, unchanged recorded Python sources and correct Core origins. The15 failures match full-v6 exactly:14 paused-outpaint workflow references to the removed DLSS input, plus one old workflow aggregate baseline whose four changes match the published DLSS patch. The full suite remains failed; this is not release approval. Outpaint was not changed.

The VDN compatibility changes let the global Sage launch option remain enabled and distinguish a normal attention backend from a patch that replaces VDN's algorithm. The two-pass route now runs a complete VDN first pass, learned latent upscaling, and a second refinement pass while retaining the first-pass audio.

The historical full CPU suite and five selected Core source trees pass their recorded checks. Fresh full-v6 completed with 2940 passes and 15 failures, all in paused outpaint workflow support: 14 references to the removed DLSS acknowledgement input and one old aggregate workflow baseline. All 656 selected Core/VDN CPU cases passed; source/origin checks passed. The exact failure set and four published DLSS workflow changes are independently qualified in `artifacts/vdn-two-pass-resume-20260908/scope-qualification.json`. The full suite and release gate remain failed; outpaint is not modified. The user subsequently freed GPU memory (14801MiB available), satisfying the unchanged12000MiB preflight; the five-case continuation is now complete.

All18 single-pass GPU cases and all18 two-pass GPU input cases are complete. Both cover full/pruned bases across nine inputs. The final five two-pass cases were pruned Ref2VA, multiple reference images, reference video/audio, reference audio, and hybrid first-frame/audio. Each retained the complete VDN8 first pass, learned2x latent upscale and VDN4 refinement, with identical first/final decoded audio. This is local current-Core input coverage, not all-hardware, all-combination, human-quality or release acceptance.

For the two existing 1024×512 comparison clips, the user reported **“左右差不多” — no clear visual preference**, then explicitly confirmed **“音乐、人声、两边口型都正常”** on2026-09-08. Classical music, Mandarin speech (“你在哪里”) and both candidates' lip-sync are accepted for this pair. The new `artifacts/vdn-two-pass-0p5mp-human-review-20260907/human_audio_lipsync_feedback_v1.json` binds the exact feedback to the page, both media/reports and the unchanged prior visual receipt. No winner or broader input/hardware acceptance is inferred.

## Attention compatibility policy

| Combination | Behavior | Evidence and limit |
|---|---|---|
| Global `--use-sage-attention` + VDN | Leave the launch option enabled; VDN-specific rows keep exact SDPA | Real current-Core DMD8 and two-pass GPU runs; this does not mean VDN internally uses Sage |
| Core plain backend selector + VDN | Recognize the actual backend wrapper, not its name or a copied marker | CPU identity tests and a real PyTorch-selector-before-VDN run |
| Official BlockSparseAttention + VDN | Remove authenticated sparse replacements only from the VDN MODEL branch, including downstream installation and prepare callbacks | CPU ordering/lifecycle tests and real before/after GPU configurations, including 0.52MP refinement |
| Official sparse + T8 SLA Precision V2, FastH3 VSA, H3-World or basic EAV | Keep the T8 algorithm on its own branch and report the local bypass | CPU ownership checks; broader GPU combinations remain open |
| Official sparse + SPEED | Retain official sparse attention; SPEED changes resolution/sigma transitions | CPU composition/lifecycle checks, not sparse-SPEED visual acceptance |
| Official sparse + FFN Activation Chunk | Keep the attention selected by the official callback and chunk only the token-local FFN | CPU composition and numerical checks; eligible GPU producer path still needs validation |
| Unknown algorithm-changing patch | Explain the conflict without erasing arbitrary hooks or silently changing algorithms | Negative identity and ownership tests |

The global Sage flag does not itself install the per-model `optimized_attention_override`. The previous VDN guard rejected every such override. Compatibility must distinguish normal backend selection from genuine algorithm replacement; turning off Sage globally is not the solution.

The external Sol plugin is a separate integration. Recognizing Core's official sparse factory does not automatically authorize external Sol implementations, compiler wrappers or arbitrary object-level forward patches.

### Other affected paths

- **Native H3 and multikeyframe:** capability adapters cover old/new FinalLayer argument lists, attention arguments and prefetch interfaces. Core revisions and hashes are diagnostic evidence, not the runtime capability selector.
- **EAV + STG:** enabled/disabled combinations, main/weak branches, post-CFG re-entry and cancellation/retry have real-Core CPU checks. With both effects disabled, the original MODEL is returned unchanged.
- **Prompt Relay and EAV/LongVideo composers:** authenticated official sparse state is handled on the model clone. Actual wrapper/binding identity is checked; copied markers and unknown hooks do not become trusted. Independent routed-attention/FETA math and lifecycle tests pass.
- **EAV + external BlockCache:** a non-mutating finalization view adapts the old external wrapper to the current Core sigma/head interface. Small real 50-layer CPU models test full/hit paths, dual PDD heads, audio shifts, both node orders and cancellation/retry. The external plugin was not edited; this does not certify its standalone node or trained GPU output.
- **Attention Hooks:** the oldest selected Core has raw tensors rather than AttentionTensorContainer. That mismatch was reproduced (7 failures/1 pass), fixed through capability selection, and the same eight cases then passed. Hooks/RoPE/math/cancellation tests are included in the current matrix.
- **Compilation:** four small real H3 CPU tests execute captured FX graphs with a counting eager backend, both installation orders and cancellation/retry. Four additional real CUDA inductor cases now pass through the official compiler node, as detailed above. All use `fullgraph=False` and retain graph-break warnings. Allocator/cudagraphs and broader compiler combinations remain separate open checks.

The actual CUDA probe reproduced hook bypass in all six Sol/SLA/VSA x before/after configurations: direct control called QKV/output hooks once; the eligible official chunked producer called neither. Source-bound evidence is `artifacts/attention-hooks-core-compat-20260907/cuda-native-producer-pre-fix-20260908/report.json`.

The fix authenticates the actual Core-injected attention closure and adapts it at the MODEL-local block boundary, including the T8 FFN-chunk path. No-hook calls retain Core's original chunked producer. Hook-bearing calls materialize full QKV because a hook may mix arbitrary token rows; they still call the real sparse kernel and retain VSA tile ordering, live-row lengths, prefix sinks and the learned coarse gate. This may increase VRAM use with hooks; it is not a low-memory equivalence claim. Unknown external attention callbacks are not replaced. Layout/dtype/device-changing QKV hooks are rejected explicitly instead of silently using a different algorithm.

Three v3 extended CUDA reports under that same directory each pass8 cases:188-token padding controls,5197-token multi-chunk controls, and5197-token FFN-chunk composition. Each verifies actual sparse operator dispatch, hook effect/call counts, no-hook bit identity, abort/retry and cleanup restoration. VSA cases exercise padding and fine/coarse variants; removing the real coarse gate changes output. These use small randomly initialized native blocks, not trained-video quality or complete graph/compiler coverage. Fresh five-Core expanded-v15 CPU regression completed with zero failures/errors; actual origins and source snapshots were revalidated after the terminal result. Final full-suite, broader composition and delivery gates remain open.

## Two-pass routes and required assets

Both routes use VDN DMD8 for the first pass, the existing learned 3D latent upscaler, rebuilt high-resolution conditioning and locked first-pass audio. Neither is the previously tested LightX2V 4+4 recipe.

1. **VDN → upscale → VDN:** refine using the VDN stage's own trailing sigma grid. Do not add a separate Turbo/EMA LoRA to this VDN branch.
2. **VDN → upscale → native H3:** use an independent base MODEL with the new EMA B LoRA. Do not send the VDN Composer output into the native-refinement loader. The GPU probe checks every native forward for VDN state and audits 259 EMA targets; that is not a tensor-by-tensor weight-equality proof.

Existing assets are reused; no additional converted or trained adapter has been shown necessary:

- `models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`
- `models/diffusion_models/OpenVDN/vdn-minimax-h3/`, keeping its complete internal structure
- `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`
- `models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors` for the independent native refinement route only
- The usual H3 text encoder, video/audio VAEs and FFmpeg.

The two-pass GUI drafts use the registered experimental SafeAVSave node to encode raw IMAGE/AUDIO and return a reusable VIDEO without another SaveVideo re-encode.

## Verification record

### CPU: current source baseline

| Core source | Expanded-v15 passed | Skipped |
|---|---:|---:|
| Current `fbed745` | 664 | 0 |
| `cf10c5c` | 604 | 60 |
| `e7051b0` | 605 | 59 |
| `86aedfd` | 604 | 60 |
| `563b98e` | 583 | 81 |

All five have zero failures/errors, correct actual import origins and unchanged recorded Python source hashes. Missing historical APIs produce explicit skips, not inferred passes. Historical Core sources use this host's current dependencies, not reconstructed historical package installations.

The expanded-v15 reports/XML are under `artifacts/core-contract-matrix-20260908`. Historical `current-full-v5` had2947 passes; later full-v6 had2940 passes and15 paused-outpaint failures. Current full-v7 has2948 passes and the same15 failures, independently qualified as above; an isolated delivery candidate still requires qualification. The18 historical GPU report/media receipts are unchanged; of their17 recorded runtime sources, only `vdn_attention_compat.py` changed (addition of the authenticated attention-closure kind). This difference is recorded explicitly, not hidden by rewriting old reports.

### GPU: available hardware and scope

All real runs here use **RTX 4060 Ti 16GB**, serially. Installed Core is `fbed745c8d7d62573b099cd61fe51cb64b9b807e`. They do not establish 4070 Ti SUPER or Blackwell compatibility.

| Evidence | Completed scope |
|---|---|
| `vdn-current-nine-input-matrix-20260907/completion_summary.json` | All 18 full/pruned × nine-input single-pass cases; 320×192×39, DMD8, global Sage, legacy VRAM reserve5, registered SafeAVSave; minimum free VRAM across the batch 2802 MiB |
| `vdn-current-two-pass-nine-input-matrix-20260907/progress-05.json` | Original first five full-base two-pass cases; later sixth attempt interrupted without a final report |
| `vdn-current-two-pass-nine-input-matrix-resume-20260907-v2/progress-13.json` | 13/18: all nine full-base inputs plus pruned T2VA/I2VA/L2VA/FL2VA; all 13 receipts revalidated on 2026-09-08 |
| `vdn-two-pass-resume-20260908/completion_summary.json` | 18/18: previous13 revalidated plus five new pruned cases; VDN8+4,320x192->640x384x39, Sage and registered saver; first/final PCM identical; current source and scoped CPU bindings checked; full-v6 still has15 paused-outpaint failures |
| `vdn-old-core-cf10c5c-sage-two-pass-20260907/20260907-182527` | Exact historical cf10c5c source, current host packages, Sage, VDN8+4, 640×384×39, registered saver, 3097 MiB minimum free; one old-Core configuration, not the entire historical GPU matrix |
| `vdn-core-compat-sage-sparse-before-legacy-vram-20260907/20260907-143858` | Official sparse before VDN, Sage, VDN8+4, 640×384×39; 1156 MiB minimum free |
| `vdn-two-pass-0p5mp-safe-review-20260907/20260907-155312` | VDN8+4, 1024×512×73, isolated raw-AV encoding, identical first/final decoded PCM; 1275 MiB minimum free |
| `vdn-native-refine-0p5mp-safe-review-20260907/20260907-160134` | VDN8 + independent native4, 1024×512×73, all four native forwards checked, no VDN branch state, 259 EMA targets; 2895 MiB minimum free |
| `vdn-i2va-two-pass-registered-save-v2-20260907/20260907-174246` | I2VA, VDN8+4, 1024×512×73, Sage and official sparse before VDN, registered SafeAVSave, identical first/final PCM; 2830 MiB minimum free |

Each accepted mechanical report checks adapter shapes, absence of LoRA runtime errors, actual output dimensions/frames, strict H.264 decoding, finite nonempty audio, audio preservation, resource margin and recorded project/Core source identity. These checks do not grade naturalness, lip-sync or music quality.

The resumed batch independently revalidated the original five reports/media, 17 runtime sources, five Core sources and six inputs. The six Python changes between full-v4 and full-v5 were confined to the audited Hooks/shared-controls/tests/runner changes and do not occur in the previous GPU graphs. The new batch binds full-v5, uses new case directories and stops on any failure. Python runtime sources stay frozen while it runs.

The old-Core GPU archive is verified against all 1063 tracked Git blobs, with explicit source and runtime roots and actual imported-module checks. The earlier CRLF CPU archive remains intact; accidentally inheriting the parent plugin repository's Git HEAD is not accepted as Core provenance.

### Preserved failures and interruptions

- The 0.52MP attempt `20260907-153645` failed strict H.264 reference-frame checks and had only 280 MiB free. It remains failed. The later run encoded new raw IMAGE/AUDIO through an isolated encoder and changed reserve from 4 to 5 GiB; it is not a repair of that corrupt MP4 or a single-variable benchmark.
- I2VA attempt `20260907-163127` stopped during first-pass sampling with no final report. The separately successful `174246` does not retroactively pass it.
- Original two-pass attempt `full/multi_ref_images/20260907-210409` has no final report. Its batch/processes were confirmed absent before resuming in a new artifact root; the exit cause remains unknown.
- Earlier failing CPU regressions and earlier test matrices remain in their original artifact directories. This document summarizes their resolved or still-open outcomes rather than presenting old “running” notes as current state.

### Workflow and human review

The four v2 drafts in `artifacts/vdn-two-pass-workflow-drafts-20260907-v2` share one prompt and one existing Duration Planner across both conditioning lengths and both output trim durations. At default values, complete resolved API graphs equal their original four-route baselines; the fixtures are included in formal tests.

All four completed actual native sidebar open, Save As and API-download checks in an isolated user directory:

| Route | Execution nodes | Checked edges |
|---|---:|---:|
| T2VA → VDN refinement | 33 | 61 |
| T2VA → native refinement | 37 | 63 |
| I2VA → VDN refinement | 35 | 64 |
| I2VA → native refinement | 39 | 66 |

Each rejects four altered configurations. Receipts are in `artifacts/vdn-workflow-native-roundtrip-20260907-v2`. These checks did not install production workflows or queue inference. The file-chooser route is not verified; actual sidebar serialization and saved API files are.

The 0.52MP public comparison has identical first-pass RGB and decoded audio across the candidates; final images differ. Actual browser playback and single-audio routing were checked, then the user reported no clear visual preference. This feedback does not choose a winner or establish audio/lip-sync acceptance for unreviewed cases.

## Delivery checklist and explicit limits

| Scope | Local outcome |
|---|---|
| P0 old/new interfaces | Capability-based adapters; five real Core source matrices pass their available APIs; missing historical APIs are explicit skips |
| P1 attention ownership/backend | Global Sage stays enabled; known ordinary selectors compose, authenticated algorithm overrides are isolated per MODEL; unknown patches are not erased. Exact-SDPA selection has CPU policy tests and available Ada GPU evidence, not physical Blackwell proof |
| P2 affected nodes | Native/multikeyframe, SLA/VSA, hooks, SPEED, EAV/World, FFN/prefetch, audio/long-video/Face Refine and saver/DLSS are covered by the source-bound regression. Eligible sparse hooks, bounded compiler/graph/allocator paths have separate real CUDA evidence. This is not exhaustive trained inference for every cross-plugin combination |
| P3 two-pass | Existing learned latent2x, high-resolution conditioning, VDN8+4 and independent native4 routes; retained first-pass audio. Two reviewed0.52MP pairs accepted within the limits above; no new model conversion or universal improvement claim |
| P4 validation |18/18 single-pass and18/18 two-pass input cases; an actual old-Core Sage VDN8+4 run; current candidate2435-test full suite plus trained DynamicVRAM integration. Earlier failures and source snapshots remain intact |
| P5 delivery | Four native UI/API-verified JSON files and bilingual usage notes installed to source/user folders;25 post-install tests;1.75.0 metadata, official archive and isolated301-node import. No commit/push |

The remaining unverified areas are physical4070TiSUPER/Blackwell, every full-model offload/compiler/external-plugin combination, exhaustive long-video stress and general quality improvement over single pass. These are not silently certified by the current bounded checks. Optional future extensions need their own tests; repeated inference of the already completed matrix is not required unless its affected runtime code changes.

The Comfy configuration check passes with one retained S102 security warning for a test fixture executing pinned historical Core source; the fixture is excluded from the runtime archive. No warning suppression or test weakening was used. Final local receipts are in `artifacts/core-vdn-isolated-delivery-20260908/package-v2`; the archive contains the compatibility document despite the normal development-doc exclusion.

The user authorized GitHub publication on2026-09-08 after local acceptance. Outpaint remains paused and is excluded from this version.

Historical text removed from this summary is preserved locally in `artifacts/core-vdn-doc-consolidation-20260907/compatibility-before-consolidation.md`. Raw reports and media were not rewritten.


---

## FILE: docs/DANCE_ACCEPTED_PICTURE_20260913.md (https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/HEAD/docs/DANCE_ACCEPTED_PICTURE_20260913.md)

# Dance 4+4：已验收示例与续段背景修复

2026-09-13，用户完整播放指定8秒样片，画面／身份／细节、运动与接缝、音频三项均为“正常／可接受”。视频 SHA256：`58a8a8fb616a741be4c76f75e5857dff098e0b649959470ac0ba9b8ff7877039`。

导入[已验收 Dance 4+4／KJ 工作流](../examples/workflows/04-long-video/2026-09-13_H3_Dance_4plus4_Accepted_Picture_KJ.json)。模型、素材需自行提供；所保留文件名用于复现，不代表仓库附带权重或人物／舞蹈素材。该示例已进入2026-09-13 GitHub主线更新；旧安装需要更新节点代码并重启ComfyUI后再导入。

## 固定配置

- Ref2VA INT8 convrot，两个独立 EMA B LoRA 强度1；一采512×256／4步，原learned 3D latent upscaler，二采1024×512／4步。
- 24fps、192帧、两段共8秒；window=124，context=22，接缝frame124，约5.17秒。不是两段各4秒。
- 每路KJ Memory Efficient Sage → Low VRAM Attention（head_chunks=4）→ ChunkFeedForward（chunks=2，seq_threshold=4096）。需要对应KJ/Sage环境，不叠加SOL或其他attention补丁。
- `low_context_source=accepted_picture_low_context_v1`，`video_context_mode=high_native_mask_exp`，`second_audio_source=auto`；Relay/EAV关闭，颜色处理保留原配置。
- 实测禁用pinned memory，reserve-vram=5 GiB、vram-headroom=2 GiB。运行约13分57秒，周期采样最低可用显存约2.44 GiB、RAM约36.14 GiB；不是最低硬件要求、精确峰值或通用速度保证。

## 问题如何定位、这次改变了什么

旧样片加强衣着提示词后，第一段人物服装改善，但第二段已生成的独立candidate由白棚跳到暗舞蹈室，片尾再回白棚。原动作源对应位置没有相同跳切；调色报告未应用变换，最终拼接之前变化已经存在。不能归因为“视频拼接造成了背景变化”，也不能说没有上下文：真实HIGH已知前缀和22帧条件均存在。

经用户授权，本次只把后段LOW参考从上一段partial low-x0换成**上一段实际成片末39帧，RGB24解码→已有resize→原视频VAE编码**。音频对象、HIGH逻辑、4+4步表、3D放大器、提示词、seed、模型和其他参数不变。39是保存尾部容量；context22仍选7个latent单元，不强制变成39上下文。

第一段low_x0、high_input、high_output三份tensor哈希与上一条完全相同；第二段确实采用直接前段的[85,124)帧，额外扩散NFE为0。实际HIGH前7个latent单元保持前段成片上下文，最大数值差约2.38e-7，mask已知区0、自由区1。边界RGB差值的邻域比由30.91降至1.356；**用户完整动态审片才是本例视觉接受依据，不是这个指标本身。**

这证明该干预解决了此固定Dance样例的续段问题，不证明所有素材或所有low-x0 continuation都有同一根因。不要追加生成后latent端点平移、视频叠化等失败处理。更新相关路径时同时遵守[通用双采接缝防回归说明](DUAL_MODEL_SEAM_FIX_20260913.md)。

## 后续更新必须保留的防回归条件

1. 后段LOW只能读取同一`chain_id`中**已经接受的直接前段**；必须核对段号、候选状态、视频路径和SHA256，不能取重试片、未接受候选或更早段。
2. 从前段实际成片解码末39帧，沿用现有resize和视频VAE；`context_frames=22`仍决定采样使用的上下文量。不要把“保存39帧”和“采样强制39帧”混为一件事。
3. HIGH原生mask、4+4 sigma／NFE、learned 3D latent upscaler、模型／LoRA、seed和音频交付策略必须保持不变。修接缝时不得顺便换采样数学或加入生成后叠化。
4. 缓存身份必须包含LOW上下文来源策略；策略、模型、素材或提示词改变时使用新的`chain_id`。恢复只允许命中同参数、同父段的已接受结果。
5. 代码更新至少运行LOW上下文、直接父段身份、时间坐标、阶段失败清理及工作流序列化回归；涉及采样／上下文数学时，再跑两段8秒短GPU候选并完整动态审片。静态边界指标不能代替人审。

## 提示词与输入规则

`<Picture 1>`表示目标人物；`<Video 1>`表示当前原动作参考。写明目标脸、发型和图中可见衣着，并明确源视频只提供动作、时序、方向和镜头，不继承其人物和服装。本例的深色卷发、浅灰开衫、浅粉V领必须随新图修改；不要把图中没有展示的下装说成已验证参考。RGB参考是软约束，不是严格骨骼控制，也不保证换任意图都能成功。

先用8–15秒恒定帧率短源。LoadVideo会解码整条源到RAM；连接GetVideoComponents的实际fps，不把30fps源硬当24fps。本次源共362帧；两段内部取0..123、102..225，输出仅前192帧。`source_start_seconds`同时移动动作和原音乐起点。

原音乐通过`aligned_original_audio`接`final_audio`，整条音轨只编码一次。251个AAC包及时间戳与上一条完全相同；AAC仍有损。本次音频接受不代表模型重唱或新生成对白口型已验证。

首次使用新的`chain_id`；`resume_existing=true`仅用于同参数恢复。更换图、视频、提示词、模型、LoRA、VAE、策略、尺寸或时长必须换chain_id，不混用旧缓存。`filename_prefix`仅为输出名前缀。

## 保留的边界

示例导出后核对了当前Core/T8/KJ实际schema、API验证及17节点20连线73显式参数的序列化一致性；不把CPU验证称为新一轮浏览器导入测试或GPU采样。直连KJ的逐kernel调用和回退未独立计数，不能声称实测零回退。

此前失败的原生Dance、旧双采Dance及两条Depth实验保留为本地失败证据，不作为推荐成片或成功模板。本次不重跑已接受Tao/LTX，不推广到任意时长、素材或后端；其他暂停项目不恢复。


---

# ISSUES / PRs of T8mars/comfyui-minimax-h3-audio-T8


### #33 [open] ISSUE: [BUG] Patch Sol-Attn (MiniMax) morton=true: "aimdo memory compile error" under DynamicVRAM, and per-token AdaLN rows not reordered
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/33  by cinnom 2026-11-26T16:05:55Z

## Summary

With `morton = true`, **Patch Sol-Attn (MiniMax)** (`SolAttnMiniMax`, `sol_attn_minimax_v2.py`) has two problems. Both are in `install_h3_morton`.

1. **Crash under DynamicVRAM.** With ComfyUI 0.45's default DynamicVRAM, sampling stops after the first step with `RuntimeError: aimdo memory compile error`. It runs with `--disable-dynamic-vram`.
2. **Wrong per-token modulation, with or without DynamicVRAM.** When the video segment has per-token timesteps (a non-uniform video `denoise_mask`, for example a continuation with frozen hand-off frames), the hooks reorder the hidden states and the RoPE table but not the per-token AdaLN row index in `mod_segments`. Some tokens then get another token's timestep modulation.

Reordering once around the block loop, instead of in block hooks, fixes both. A reference implementation and test results are below.

## Environment

- comfyui-minimax-h3-audio-T8 1.106.3. `sol_attn_minimax_v2.py` is unchanged on current main (v1.107.0, d4c430a).
- ComfyUI 0.45.0, comfy-aimdo 0.6.0, comfy-kitchen 0.2.41, torch 2.14.1+cu130, Python 3.13.14 (portable), Windows 11, RTX 4090 24 GB.
- DynamicVRAM on (the default).

## Reproduction

A two-pass H3 graph. Second pass at 1440×1088, 15 s (168,598 packed tokens), int8 checkpoint, `Model Attention Backend` set to comfy kitchen attention, then `Patch Sol-Attn (MiniMax)` with tau 1.3, start_percent 0.2, end_percent 1.0, `morton = true`, `morton_curve = 2d_frame`, other inputs at their defaults.

```
[sol_attn] H3 Morton: ACTIVE: video span [4888, 168598), grid (107, 34, 45), curve 2d_frame
[sol_attn] sparse (1, 168598, 56, 128) tau=1.3 cuda-int8
Comfy model compiler graph breaks: 1, rogues: 10
!!! Exception during processing !!! aimdo memory compile error
  ...
  File "custom_nodes/minimax-h3-audio-T8/sol_attn_minimax_v2.py", line 297, in _forward
  File "comfy/ldm/minimax/model.py", line 803, in _forward
  File "comfy/ldm/minimax/model.py", line 670, in _run_blocks
  File "comfy/model_prefetch.py", line 268, in prefetch_queue_pop
  File "comfy_aimdo/malloc_graph.py", line 55, in iterate
  File "comfy_aimdo/malloc_graph.py", line 25, in pop
RuntimeError: aimdo memory compile error
```

With the same settings and `morton = false` there is no error (`rogues: 0`). With `--disable-dynamic-vram` there is no error either.

## Cause 1: allocations outlive their block scope

In Core 0.45, `MiniMaxH3Model._run_blocks` opens one malloc-graph scope per block (`prefetch_queue_pop(..., malloc_scope="block")`, `model.py` line 658) and pops the last one after the loop (line 670). An allocation is expected to end inside its scope, or to pass to the next block as the residual stream.

The Morton hooks break that:

- Block 0's `pre_hook` creates the reordered RoPE table (`rope.index_select`) and keeps it in `model._sol_morton_state` for all 50 blocks. The `post_hook` on the last block frees it inside the last block's scope.
- `_perm_for` creates `perm` and `inverse` on the GPU inside block 0's scope and keeps them in `_DEVICE_CACHE` for the life of the process.

The compiler counts these as rogue allocations, and popping the last scope fails.

## Cause 2: per-token AdaLN rows are not reordered

`MiniMaxH3Model._forward` builds `mod_segments`. When `video_rows_t` is set (a non-uniform video denoise mask, `model.py` lines 794–795), the video segment's entry holds a per-token `LongTensor`, which `_mod_scale_shift` and `_mod_gate` use (lines 259 and 266). The hooks reorder `h` and the RoPE table only, so after the reorder those rows index the wrong tokens.

With `2d_frame` and frame-wise timesteps, this only affects rows that the 64-row alignment roll moves across a frame boundary (40 rows per boundary when the video starts at row 4888). So it shows where adjacent frames have different timesteps. With `3d`, or with masks that vary within a frame, most rows are affected.

## Proposed fix

Reorder once around the block loop instead of in block hooks:

- wrap `_run_blocks` and reorder `h` in place before it;
- build the RoPE table from reordered `position_ids` (`rope_rotation_table` works per row, so this equals reordering the table);
- reorder the video segment's per-token rows in `mod_segments`;
- restore the order of the output after the loop;
- keep the permutations on the CPU and copy them each forward, so nothing persistent is allocated while Core records.

With no block hooks, every block allocates exactly as it does without Morton.

`h3_t8/res_sol_identity.py` checks the current hook roles and pins `_SOL_SOURCE_SHA256`, so it and its tests would need a matching update.

<details>
<summary>Reference replacement for <code>install_h3_morton</code> (drop-in for <code>sol_attn_minimax_v2.py</code>; <code>_perm_for</code> and <code>_DEVICE_CACHE</code> are no longer used)</summary>

```python
_CPU_PERMS = {}
_FULL_ROWS = {}


def _perm_cpu(grid, curve, start):
    """Morton permutation and inverse for a video span starting at row ``start``, cached on the CPU.

    Same alignment roll as before. Kept on the CPU and copied per forward, so nothing persistent is
    allocated while Core records a block's allocations.
    """
    pad = (-int(start)) % BLOCK_SIZE
    key = (tuple(int(x) for x in grid), curve, pad)
    hit = _CPU_PERMS.get(key)
    if hit is None:
        perm, inverse = morton_perm(grid, torch.device("cpu"), curve)
        if pad:
            perm = torch.roll(perm, pad)
            inverse = torch.argsort(perm)
        hit = (perm.contiguous(), inverse.contiguous())
        _CPU_PERMS[key] = hit
    return hit


def _full_rows(seq_len, start, stop, perm):
    key = (int(seq_len), int(start), int(stop), perm.data_ptr())
    rows = _FULL_ROWS.get(key)
    if rows is None:
        rows = torch.arange(int(seq_len), dtype=torch.int64)
        rows[start:stop] = perm + start
        _FULL_ROWS[key] = rows
    return rows


def install_h3_morton(model):
    """Idempotently install the reorder. Inert without transformer_options['sol_morton'].

    Reorders once around the block loop: no block hooks, so every block allocates as without Morton.
    """
    if id(model) in _INSTALLED:
        return
    for attr in ("rope_freqs", "_forward", "_run_blocks", "blocks"):
        if not hasattr(model, attr):
            raise RuntimeError(f"MiniMax-H3 Morton needs .{attr} on the diffusion model")

    _patch_packed_layout(sys.modules[type(model).__module__])

    original_forward = model._forward
    original_rope_freqs = model.rope_freqs
    original_run_blocks = model._run_blocks

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        previous = getattr(model, "_sol_morton_active", False)
        model._sol_morton_active = bool(transformer_options.get("sol_morton"))
        model._sol_morton_curve = transformer_options.get("sol_morton_curve", "3d")
        model._sol_transformer_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._sol_morton_active = previous
            model._sol_morton_rows = None
            model._sol_transformer_options = None
            transformer_options.pop("sol_h3_video_span", None)
            transformer_options.pop("sol_h3_audio_span", None)

    def rope_freqs(position_ids, device):
        """Publish the segment spans for the conditioning sink. While _run_blocks reorders, build the
        table from reordered position ids (the table is per row, so this equals reordering it)."""
        entry = _SPANS.get(id(position_ids))
        if entry is None:
            _h3_log_once("no layout registered; Morton and the conditioning sink are inactive")
        else:
            _layout, bounds, audio, _span = entry
            options = getattr(model, "_sol_transformer_options", None)
            if options is not None:
                options["sol_h3_video_span"] = bounds
                options["sol_h3_audio_span"] = audio
        rows = getattr(model, "_sol_morton_rows", None)
        if rows is not None and rows[0] is position_ids:
            position_ids = position_ids.index_select(0, rows[1].to(position_ids.device))
        return original_rope_freqs(position_ids, device)

    def _run_blocks(h, t_emb, mod_segments, layout, transformer_options, *args, **kwargs):
        def plain():
            return original_run_blocks(h, t_emb, mod_segments, layout, transformer_options, *args, **kwargs)

        if not getattr(model, "_sol_morton_active", False):
            return plain()
        position_ids = getattr(layout, "position_ids", None)
        entry = _SPANS.get(id(position_ids))
        span = entry[3] if entry is not None else None
        if span is None:
            _h3_log_once("video grid does not match the segment; Morton inactive")
            return plain()
        start, stop, grid = span
        if (not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] < stop
                or position_ids.shape[0] != h.shape[0]):
            _h3_log_once(f"hidden states do not cover the video span {(start, stop)}; inactive")
            return plain()
        # per-token AdaLN rows (non-uniform video timesteps) must move with their tokens
        moved = []
        for i, (a, b, row) in enumerate(mod_segments):
            if torch.is_tensor(row) and row.ndim > 0:
                if a == start and b == stop and row.shape[0] == stop - start:
                    moved.append(i)
                elif b > start and a < stop:
                    _h3_log_once("per-token modulation crosses the video span; Morton inactive")
                    return plain()
        curve = getattr(model, "_sol_morton_curve", "3d")
        perm_cpu, inverse_cpu = _perm_cpu(grid, curve, start)
        _h3_log_once(f"ACTIVE: video span [{start}, {stop}), grid {grid}, curve {curve}")
        perm = perm_cpu.to(h.device)
        segments = list(mod_segments)
        for i in moved:
            a, b, row = segments[i]
            segments[i] = (a, b, row.index_select(0, perm.to(row.device)))
        h[start:stop] = h[start:stop].index_select(0, perm)
        model._sol_morton_rows = (position_ids, _full_rows(h.shape[0], start, stop, perm_cpu))
        try:
            out = original_run_blocks(h, t_emb, segments, layout, transformer_options, *args, **kwargs)
        finally:
            model._sol_morton_rows = None
        out[start:stop] = out[start:stop].index_select(0, inverse_cpu.to(out.device))
        return out

    model._forward = _forward
    model.rope_freqs = rope_freqs
    model._run_blocks = _run_blocks
    _INSTALLED.add(id(model))
```

</details>

## Results with this change

- **Float64 stand-in for the block stack.** It has per-row RoPE from `position_ids`, segment and per-token AdaLN rows, and exact softmax attention: 3 blocks, 48 conditioning rows plus 210 video tokens, with the video start not 64-aligned. The reordered output matched the unreordered one to a relative difference of at most 6.2e-15, for `2d_frame` and `3d`, with and without per-token rows. The current version matches without per-token rows but differs by about 1.0 (rows completely wrong) with them.
- **Real model with DynamicVRAM on.** No error and `rogues: 0` with Morton on the 168,598-token second pass, and on both passes (60,528 and 168,598 tokens). The second pass took 36.0 s per step against 34.7 s without Morton (RTX 4090).
- **Comparison with the current Morton.** Against the current Morton run with `--disable-dynamic-vram`, the output scored 33.8 dB PSNR. For scale, two dense runs across the two memory modes score 35.2 dB against each other.

End-to-end PSNR can't prove exactness on this int8/bf16 stack. An exact reorder changes a one-step first-pass result by about 27 dB, about as much as switching from int8 to PyTorch attention. The float64 test is the exactness evidence.

I can share the stand-in test, or open a PR if you'd prefer one.


> **Comment by cinnom (2026-11-26T18:38:52Z):**

## Follow-up: skip the reorder on steps that run dense

Morton also reorders the steps that `SolAttnMiniMax` runs dense, outside the `start_percent` / `end_percent` window or below `min_tokens`. That reorder is mathematically exact, but it changes rounding. Early in a first pass, where motion is still being decided, that was enough to change the take.

**Test.** First pass only (832×640, 60,528 tokens, 6 steps, beta57). Patch Sol-Attn with start_percent 0.7 (steps 1–4 dense, 5–6 sparse) and tau 1.0. Same seed, compared with the dense first pass:

| | PSNR | Lowest 1-s window | SSIM |
|---|---|---|---|
| morton = true, reordering every step | 24.3 dB | 16.3 dB | 0.80 |
| morton = false | 32.4 dB | 30.4 dB | 0.95 |
| morton = true, reordering only sparse steps | 31.5 dB | 29.3 dB | 0.94 |

With every step reordered, a hand gesture came out differently in seconds 11–14. That's why the lowest 1-second window drops to 16.3 dB, even though only the last two steps were sparse. Changing tau (1.0 against 1.3) didn't affect it. Different seeds score 14–16 dB against each other here.

**Suggested change.** In the reference `_run_blocks` wrapper above, keep raster order when the attention override would run the whole step dense. The node already publishes the gate as `sol_compose`:

```python
def _sparse_step(transformer_options, h):
    gate = (transformer_options or {}).get("sol_compose")
    sigmas = (transformer_options or {}).get("sigmas")
    if not isinstance(gate, dict):
        return True
    if h.shape[0] < gate["min_tokens"]:
        return False
    if sigmas is None:
        return True
    sigma = float(sigmas[0])
    return not (sigma > gate["sigma_start"] or sigma < gate["sigma_end"])

# in _run_blocks, after the sol_morton check:
#     if not _sparse_step(transformer_options, h):
#         return plain()
```

These steps are then bit-identical to `morton = false`: checked in the float64 stand-in, and in the run above, where the reorder became active only at step 5. Blocks kept dense through `dense_blocks` are still reordered, because the reorder happens once per step, around all blocks.




### #32 [closed] ISSUE: Outpaint_01_Generate_Candidate error "MODEL nonpersistent buffer unexpectedly appears in persistent state"
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/32  by djdance 2026-10-27T14:52:00Z

First outpaint WF **2026-09-07_H3_Video_Outpaint_01_Generate_Candidate_EXP.json** fails with error

`- **Node ID:** 9
- **Node Type:** MiniMaxH3VideoOutpaintPrepareT8
- **Exception Type:** ValueError
- **Exception Message:** ValueError: MODEL nonpersistent buffer unexpectedly appears in persistent state`

I changed run_name, it not helped. What else should I do to fix it? Thanks.

ComfyUI v0.43.0

> **Comment by T8mars (2026-11-17T05:40:30Z):**

Rechecked for the 2026-11-17 maintenance release: the nonpersistent-buffer export fix is present in [v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1) (original fix: 96f0381). A native state-dict hook may legitimately export a registered nonpersistent buffer. Matching live/original contents are restored without inventing persistent keys; corrupt exports, paths, shapes and nonfinite buffers still fail. Real Core Dynamic-loader/state-hook regressions cover this path.

Please update the T8 extension and restart the backend; changing run_name alone cannot update the old implementation. Closing as fixed in the shipped code. If it still fails on current source, reopen with the new traceback and a minimal workflow without private media. [Validation scope](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/COMMUNITY_ISSUES_20261117.md).



### #31 [closed] PR: Fix sampling identity checks on Python 3.13
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/31  by Anson2048 2026-10-22T20:43:09Z

## Problem

On Python 3.13 (the interpreter shipped with the current ComfyUI Windows portable build), every `MiniMaxH3NativeDualStageSetupEXPT8` run fails before sampling:

```
UnverifiedModelStack: V2 sampling class contains unknown execution members
  File "h3_t8/long_video_dual_identity.py", line 265, in _v2_sampling_identity
```

Since 3.13, every `class` statement adds `__firstlineno__` and `__static_attributes__` to the class namespace. `_v2_sampling_identity` and `_native_sigma_shift_factory` compare `vars(cls)` against an exact allow-list, so all three carriers are rejected: `MiniMaxH3FlowSampling`, `_make_sampling.<locals>.MiniMaxH3NativeAVSampling`, and Core's `MiniMaxH3SigmaShift` `ModelSamplingAdvanced`.

```python
class C(Base):
    @property
    def audio_scale(self): return 1.0
set(vars(C)) - {"__module__", "__doc__", "audio_scale"}
# 3.13.12 -> {'__firstlineno__', '__static_attributes__'}
```

## Change

- Adds `_PY313_CLASS_META = {"__firstlineno__", "__static_attributes__"}` and excludes it in the three `vars(cls)` checks.
- These names are interpreter metadata, not callables. Every other check (bases, module, qualname, instance state, hooks, Core source/bytecode audit) is unchanged, and older Python versions behave exactly as before.

No version bump, so the Registry publish workflow is not triggered.

## Tests

- `tests/test_native_sigma_shift_identity.py`: new `test_t8_sampling_carriers_ignore_interpreter_class_metadata`, parametrized over `dual_clock_euler` (`t8_raw_audio_flow`) and `er_sde` (`native_av_carrier`). It injects the two metadata names when absent so it is meaningful on any Python version, and it checks that a real added method is still rejected. It fails on current `main` under 3.13.12 and passes with this change.
- Run locally on Python 3.13.12 + ComfyUI `a071826fd`. On `main`, 22 failures in this file and `tests/test_sampling.py` report `lacks a portable native sampling schema`; with this change none do.
- Not fully green locally, and unrelated to this change: the remaining failures in this file stop at `Core SigmaShift source is outside the audited sampling factory`, because my Core `nodes_minimax_h3.py` hash differs from `CORE_SIGMA_SHIFT_SOURCE_SHA256`. `tests/test_h09_sampling_admission_ci.py` and `tools/audit_modular_new_sampling.py guard` fail identically on `main` (`chunked_two_pass_upscale_advanced.py` AST). `check_windows_paths --working-tree`, `check_readme_homepage`, `prepare_release_version --check`, and `check_no_roadmap` pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)


> **Comment by T8mars (2026-11-17T05:42:21Z):**

Thank you for the Python3.13 report and tests. The current implementation already handles all three sampling carriers through _sampling_class_members (7835d92), with additional strict checks for inert metadata types and real executable members. The unrelated Core SigmaShift whole-file mismatch was fixed in 96f0381. These fixes are retained in [v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1); this maintenance also adds the new Core0.45 FinalLayer grids ABI.

Closing as superseded by the shipped implementation, without merging the weaker metadata exemption again. CPU metadata-injection/negative tests were rechecked; the actual Python3.13 compiler-only cases are explicitly skipped on this machine's Python3.12, rather than claimed as a new full3.13 runtime qualification. [Validation details](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/COMMUNITY_ISSUES_20261117.md).



### #30 [closed] ISSUE: Topaz 节点在没有 Intel 核显的机器上必然失败（官方 FFmpeg 无 H.264/HEVC 软件解码器，默认落到 h264_qsv）
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/30  by wonkwq 2026-10-06T11:01:54Z

## 现象

`MiniMaxH3TopazVideoEXPT8` 在读取源视频的第一步就失败，换视频、先启动一次 Topaz 软件都没用：

```
RuntimeError: Official Topaz source_video_probe failed with exit code3221225477; see task stderr. No fallback/download was attempted.
```

`3221225477` 即 `0xC0000005`（ACCESS_VIOLATION），任务目录里的 `source_video_probe.stderr` 是 0 字节。

## 环境

| 项目 | 值 |
|---|---|
| 插件 | T8mars/comfyui-minimax-h3-audio-T8 @ `4c6abda`（v1.93.0，与 origin/main 一致） |
| ComfyUI | 0.38.0，Python 3.12.10，Windows 11 Pro 10.0.26300 |
| CPU | Intel Core i5-12600KF（**KF 型号，没有核显**） |
| GPU | NVIDIA GeForce RTX 4080 SUPER，驱动 610.88 |
| Topaz | Topaz Video（`Topaz Video.exe` 文件版本 1.7.1.0），自带 FFmpeg 8.1 |

## 根因

1. Topaz 自带的 FFmpeg **没有 H.264 / HEVC 软件解码器**，`ffmpeg -decoders` 里只有硬件解码器：
   - H.264：`h264_qsv`、`h264_amf`、`h264_cuvid`
   - HEVC：`hevc_qsv`、`hevc_amf`、`hevc_cuvid`

   其他格式（VP9、AV1、ProRes、FFV1 等）都有软件解码器。
2. 不指定解码器时，FFmpeg 选第一个注册的 `h264_qsv`，它需要 Intel 核显。在没有核显的机器上：
   - `ffprobe` 打开文件即崩溃 `0xC0000005`，没有任何输出；
   - `ffmpeg` 报 `[h264_qsv] Error creating a MFX session: -9.`，退出码 69。
3. 插件里所有读视频的命令都没有指定解码器，所以源视频探测、增强主命令、输出探测、PCM 校验、strict_decode 都会走到 QSV。只探测音频（`-select_streams a -show_packets`）也会崩，因为 ffprobe 打开文件时会为所有流打开解码器。

有核显的机器上默认的 QSV 能用，所以不会遇到这个问题。

## 复现（插件外，直接运行 Topaz 自带程序）

```bat
cd /d "<Topaz 安装目录>"
ffprobe -version                                                        :: 正常
ffprobe -v error -show_format input.mp4                                 :: 崩溃 0xC0000005
ffmpeg -hide_banner -nostdin -i input.mp4 -frames:v 1 -f null -         :: QSV 报错，退出码 69
ffprobe -v error -codec:v h264_cuvid -show_format input.mp4             :: 正常
ffmpeg -hide_banner -nostdin -c:v h264_cuvid -i input.mp4 -frames:v 1 -f null -   :: 正常
```

测了 4 个不同来源的视频（3 个 H.264 8bit、1 个 HEVC 10bit），结果一致；这些文件用 PyAV 都能正常解码，可以排除文件本身的问题。

## 修复思路（补丁附在最后）

NVENC 输出路线本来就要求 NVIDIA 显卡，所以对 H.264 / HEVC 明确使用 CUVID 解码器：

- **`topaz_contract.py`**：新增 `CUVID_DECODERS = ('h264_cuvid', 'hevc_cuvid')` 和 `decoder_options()`；`regular_command` / `interpolation_command` 新增可选参数 `decoder=None`，指定时在 `-i` 前插入 `-codec:v <解码器>`。
- **`topaz_media.py`**：
  - 新增 `video_decoder(runtime, path, environment, candidates)`：逐个用 `ffprobe -codec:v <候选> -select_streams v:0 -show_entries stream=codec_name` 试探。解码器与流不匹配时，ffprobe 在打开阶段报 `Codec type or id mismatches` 并以退出码 1 结束（约 0.1 秒，不会崩溃），第一个成功的候选就是匹配的解码器；都不匹配时返回 `None`，保持 FFmpeg 的默认选择（其他格式走软件解码器，行为不变）。
  - `probe_command` / `decoded_pcm_digests` 新增可选参数 `decoder=None`。
  - `analyze_video`：所有解码帧的 `pix_fmt` 一致时，以帧的格式为准。**原因是 `hevc_cuvid` 在流级别报告的是临时格式 `nv12`（10bit 也一样），解码出来的帧才是 `p010le`**。不处理的话 10bit 源会被判成 8bit，`delivery_hevc_main10` 的输出校验会直接失败。
- **`topaz_worker.py` / `topaz_fi_worker.py`**：对源视频和输出视频各检测一次解码器，传给所有读视频的步骤（视频/音频探测、PCM 校验、增强或插帧主命令、strict_decode）。
- 只改了 worker 侧模块，父进程未改动；`decoder=None` 时所有命令与原来完全相同。
- **测试**：`test_topaz_worker_dimensions.py` 的假 subprocess 增加了对检测命令的响应，并断言所有读视频的调用都带上 `h264_cuvid`、增强命令里的 `-codec:v` 位于 `-i` 之前；`test_topaz_media.py` 新增“流级 `nv12` + 帧级 `p010le`”的用例。

这只是其中一种做法。如果你更倾向于给用户加一个解码器选项，或按机器能力自动选择，也请保留 `hevc_cuvid` 流级像素格式的处理，否则 10bit 会出问题。

## 验证

- **插件测试**：`tests/test_topaz_*.py` 加 `tests/test_nodes_topaz.py`，改前 160 passed，改后 161 passed。
- **真实 Topaz 程序**：

  | 文件 | 检测到的解码器 | `analyze_video` 结果 |
  |---|---|---|
  | H.264 8bit，960×540 | `h264_cuvid` | 200 帧，8bit |
  | HEVC 10bit，1024×576 | `hevc_cuvid` | 158 帧，`p010le`，10bit |
  | FFV1（mkv） | `None` | 24 帧，软件解码，行为不变 |

- **端到端**：`ghq-5`、2x、`delivery_h264` 的工作流运行成功，960×540 → 1920×1080，200 帧，约 11.5 fps，结果为 `media_audit_pass_human_pending`。增强日志：`Stream #0:0 -> #0:0 (h264 (h264_cuvid) -> h264 (h264_nvenc))`。
- **带音轨的文件**（HEVC + 音频）：补丁前只做音频探测就会崩溃；补丁后音频探测和 PCM 校验都正常。
- **补丁本身**：在 `4c6abda` 的干净代码上 `git apply --check` 通过。
- **未覆盖**：带音轨视频的完整增强流程，以及插帧节点的端到端运行。插帧 worker 的改动与常规 worker 对称，目前只经过了代码检查和插件测试。

## 补丁

<details>
<summary>git diff（基于 4c6abda，可直接 git apply，6 个文件，+93 −25）</summary>

```diff
diff --git a/h3_t8/topaz_contract.py b/h3_t8/topaz_contract.py
index 7219c61..18ad135 100644
--- a/h3_t8/topaz_contract.py
+++ b/h3_t8/topaz_contract.py
@@ -17,6 +17,18 @@ REGULAR_PARAMETERS = frozenset({'preblur', 'noise', 'details', 'halo', 'blur',
     'compression', 'prenoise', 'grain', 'gsize', 'blend', 'estimate', 'kcolor'})
 DELIVERY_OUTPUT_PROFILES = frozenset({'delivery_h264', 'delivery_hevc_main10'})
 REGULAR_OUTPUT_PROFILES = DELIVERY_OUTPUT_PROFILES | {'lossless_master'}
+# Official FFmpeg ships no software H.264/HEVC decoder. Left to choose, it opens
+# h264_qsv first, which fails without an Intel iGPU (e.g. KF CPUs). The NVENC
+# route already needs NVIDIA, so its CUVID decoders are named explicitly.
+CUVID_DECODERS = ('h264_cuvid', 'hevc_cuvid')
+
+
+def decoder_options(decoder):
+    if decoder is None:
+        return []
+    if decoder not in CUVID_DECODERS:
+        raise ValueError('Unsupported Topaz input decoder')
+    return ['-codec:v', decoder]
 
 
 @dataclass(frozen=True)
@@ -154,7 +166,7 @@ def native_dimension_evidence(stderr, expected_frames):
 
 
 def regular_command(runtime, source, destination, model_id, width, height, *, size_mode='scale',
-                    output_profile='delivery_h264', audio_mode='copy', **settings):
+                    output_profile='delivery_h264', audio_mode='copy', decoder=None, **settings):
     source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
     suffix = destination.suffix.lower()
     if output_profile not in REGULAR_OUTPUT_PROFILES:
@@ -199,7 +211,8 @@ def regular_command(runtime, source, destination, model_id, width, height, *, si
     # No -r, frame interpolation or silent FPS rounding. MP4-incompatible
     # audio is automatically encoded to AAC; compatible audio remains copied.
     return [str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-nostdin', '-n',
-        '-protocol_whitelist', 'file,pipe', '-copyts', '-start_at_zero', '-i', str(source),
+        '-protocol_whitelist', 'file,pipe', '-copyts', '-start_at_zero',
+        *decoder_options(decoder), '-i', str(source),
         '-map', '0:v:0', '-map', '0:a?', '-vf', filter_text, '-fps_mode', 'passthrough',
         '-enc_time_base:v', 'demux',
         *encoder, '-threads', '2', *audio_encoder,
@@ -229,7 +242,7 @@ def interpolation_filter(runtime, model_id, output_fps, *, device=0, vram=.8, in
 
 
 def interpolation_command(runtime, source, destination, model_id, output_fps, *,
-                          output_profile='delivery_h264', audio_mode='copy', **settings):
+                          output_profile='delivery_h264', audio_mode='copy', decoder=None, **settings):
     source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
     if output_profile not in DELIVERY_OUTPUT_PROFILES:
         raise ValueError('Unknown Topaz interpolation output profile')
@@ -252,7 +265,8 @@ def interpolation_command(runtime, source, destination, model_id, output_fps, *,
     audio_encoder = (['-c:a', 'copy'] if audio_mode == 'copy'
         else ['-c:a', 'aac', '-b:a', '192k'])
     return [str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-nostdin', '-n',
-        '-protocol_whitelist', 'file,pipe', '-copyts', '-start_at_zero', '-i', str(source),
+        '-protocol_whitelist', 'file,pipe', '-copyts', '-start_at_zero',
+        *decoder_options(decoder), '-i', str(source),
         '-map', '0:v:0', '-map', '0:a?', '-vf', filter_text, '-fps_mode', 'passthrough',
         '-enc_time_base:v', 'filter', *encoder,
         *(['-movflags', '+faststart'] if destination.suffix.lower() == '.mp4' else []),
diff --git a/h3_t8/topaz_fi_worker.py b/h3_t8/topaz_fi_worker.py
index f449913..419b662 100644
--- a/h3_t8/topaz_fi_worker.py
+++ b/h3_t8/topaz_fi_worker.py
@@ -50,11 +50,14 @@ def main():
     source_bit_depths = media.qualified_input_bit_depths(output_profile)
     output_bit_depths = (10,) if output_profile == 'delivery_hevc_main10' else (8,)
     source = Path(spec['source']['path'])
+    source_decoder = media.video_decoder(runtime, source, env, contract.CUVID_DECODERS)
     source_video = media.analyze_video(json.loads(run(
-        media.probe_command(runtime, source, frames=True), 'source_video_probe')),
+        media.probe_command(runtime, source, frames=True, decoder=source_decoder), 'source_video_probe')),
         allowed_bit_depths=source_bit_depths)
-    source_audio = json.loads(run(media.probe_command(runtime, source, packets=True), 'source_audio_probe'))
-    source_pcm = media.decoded_pcm_digests(runtime, source, source_audio, env, job / 'source_pcm.stderr')
+    source_audio = json.loads(run(media.probe_command(runtime, source, packets=True,
+        decoder=source_decoder), 'source_audio_probe'))
+    source_pcm = media.decoded_pcm_digests(runtime, source, source_audio, env, job / 'source_pcm.stderr',
+        decoder=source_decoder)
     multiplier = settings['multiplier']
     output_fps = Fraction(source_video['fps']) * multiplier
     if output_fps > 240:
@@ -80,16 +83,19 @@ def main():
         output_fps, device=settings['device'], vram=settings['vram'],
         instances=settings.get('instances', 0),
         duplicate_threshold=settings['duplicate_threshold'], output_profile=output_profile,
-        audio_mode=audio_mode)
+        audio_mode=audio_mode, decoder=source_decoder)
     (job / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf8')
     revalidate()
     run(command, 'interpolation')
+    output_decoder = media.video_decoder(runtime, pending, env, contract.CUVID_DECODERS)
     output_video = media.analyze_video(json.loads(run(
-        media.probe_command(runtime, pending, frames=True), 'output_video_probe')),
+        media.probe_command(runtime, pending, frames=True, decoder=output_decoder), 'output_video_probe')),
         allowed_bit_depths=output_bit_depths)
-    output_audio = json.loads(run(media.probe_command(runtime, pending, packets=True), 'output_audio_probe'))
+    output_audio = json.loads(run(media.probe_command(runtime, pending, packets=True,
+        decoder=output_decoder), 'output_audio_probe'))
     video_audit = media.compare_interpolated_video(source_video, output_video, multiplier)
-    output_pcm = media.decoded_pcm_digests(runtime, pending, output_audio, env, job / 'output_pcm.stderr')
+    output_pcm = media.decoded_pcm_digests(runtime, pending, output_audio, env, job / 'output_pcm.stderr',
+        decoder=output_decoder)
     if audio_mode == 'copy':
         audio_audit = media.compare_audio_packets(source_audio, output_audio, video_audit['common_shift'])
         audio_audit['pcm'] = media.compare_pcm_digests(source_pcm, output_pcm)
@@ -97,8 +103,8 @@ def main():
         audio_audit = media.compare_transcoded_audio(source_audio, output_audio,
             video_audit['common_shift'], source_pcm, output_pcm)
     run([str(runtime.executable('ffmpeg.exe')), '-v', 'error', '-xerror', '-err_detect', 'explode',
-        '-nostdin', '-protocol_whitelist', 'file,pipe', '-i', str(pending), '-map', '0:v:0',
-        '-map', '0:a?', '-f', 'null', '-'], 'strict_decode')
+        '-nostdin', '-protocol_whitelist', 'file,pipe', *contract.decoder_options(output_decoder),
+        '-i', str(pending), '-map', '0:v:0', '-map', '0:a?', '-f', 'null', '-'], 'strict_decode')
     revalidate()
     target = job / ('enhanced' + suffix)
     if target.exists():
diff --git a/h3_t8/topaz_media.py b/h3_t8/topaz_media.py
index 50d4c44..1323e42 100644
--- a/h3_t8/topaz_media.py
+++ b/h3_t8/topaz_media.py
@@ -50,11 +50,12 @@ def file_identity(path):
     return {'path': str(path), 'bytes': after.st_size, 'sha256': digest.hexdigest()}
 
 
-def probe_command(runtime, source, *, frames=False, packets=False):
+def probe_command(runtime, source, *, frames=False, packets=False, decoder=None):
     source = Path(source).resolve(strict=True)
     if not source.is_file() or source.suffix.lower() not in ('.mp4', '.mkv', '.mov', '.webm', '.avi'):
         raise ValueError('Use an existing local video container, not a URL/playlist')
     args = [str(runtime.executable('ffprobe.exe')), '-v', 'error', '-protocol_whitelist', 'file,pipe',
+        *([] if decoder is None else ['-codec:v', decoder]),
         '-of', 'json', '-show_streams', '-show_format']
     if frames:
         args += ['-select_streams', 'v:0', '-show_frames', '-show_entries',
@@ -65,6 +66,25 @@ def probe_command(runtime, source, *, frames=False, packets=False):
     return args + [str(source)]
 
 
+def video_decoder(runtime, path, environment, candidates):
+    """First candidate that opens path's video stream, or None for FFmpeg's choice.
+
+    A mismatched decoder is rejected at open time before any decoding, so this
+    identifies the codec without touching FFmpeg's default hardware decoder.
+    """
+    path = Path(path).resolve(strict=True)
+    for decoder in candidates:
+        result = subprocess.run([str(runtime.executable('ffprobe.exe')), '-v', 'error',
+            '-protocol_whitelist', 'file,pipe', '-codec:v', decoder, '-select_streams', 'v:0',
+            '-show_entries', 'stream=codec_name', '-of', 'json', str(path)],
+            cwd=runtime.install, env=environment, stdin=subprocess.DEVNULL,
+            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
+            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), shell=False)
+        if result.returncode == 0:
+            return decoder
+    return None
+
+
 def analyze_video(probe, *, allowed_bit_depths=(8,)):
     videos = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
     if len(videos) != 1:
@@ -80,6 +100,11 @@ def analyze_video(probe, *, allowed_bit_depths=(8,)):
         raise ValueError('Qualified bit-depth policy is invalid')
     allowed_bit_depths = frozenset(allowed_bit_depths)
     pixel_format = stream.get('pix_fmt')
+    # CUVID reports a provisional stream format (nv12 even for 10-bit HEVC);
+    # the decoded frames carry the real one.
+    frame_formats = {frame.get('pix_fmt') for frame in probe.get('frames', [])} - {None}
+    if len(frame_formats) == 1:
+        pixel_format = frame_formats.pop()
     raw_bits = stream.get('bits_per_raw_sample')
     bit_depth = next((depth for depth, formats in QUALIFIED_SDR_PIXEL_FORMATS.items()
                       if pixel_format in formats), None)
@@ -276,7 +301,7 @@ def compare_transcoded_audio(source, result, common_shift, source_pcm, output_pc
         'source_pcm': source_pcm, 'output_pcm': output_pcm}
 
 
-def decoded_pcm_digests(runtime, path, probe, environment, stderr_path):
+def decoded_pcm_digests(runtime, path, probe, environment, stderr_path, decoder=None):
     """Stream decoded float32 PCM without retaining a full audio track in RAM.
 
     Called only inside the owned, deadline/resource-guarded worker. Any decoder
@@ -292,6 +317,7 @@ def decoded_pcm_digests(runtime, path, probe, environment, stderr_path):
                 raise ValueError('Invalid PCM source stream index')
             command = [str(runtime.executable('ffmpeg.exe')), '-v', 'error', '-xerror',
                 '-err_detect', 'explode', '-nostdin', '-protocol_whitelist', 'file,pipe',
+                *([] if decoder is None else ['-codec:v', decoder]),
                 '-i', str(path), '-map', f'0:{index}', '-c:a', 'pcm_f32le',
                 '-threads', '2', '-f', 'f32le', 'pipe:1']
             process = subprocess.Popen(command, cwd=runtime.install, env=environment,
diff --git a/h3_t8/topaz_worker.py b/h3_t8/topaz_worker.py
index 45721c3..060087d 100644
--- a/h3_t8/topaz_worker.py
+++ b/h3_t8/topaz_worker.py
@@ -44,10 +44,12 @@ def main():
     output_bit_depths = ({'delivery_h264': (8,), 'delivery_hevc_main10': (10,),
                           'lossless_master': (16,)}[output_profile])
     source = Path(spec['source']['path'])
+    source_decoder = media.video_decoder(runtime, source, env, contract.CUVID_DECODERS)
     source_video = media.analyze_video(json.loads(run(
-        media.probe_command(runtime, source, frames=True), 'source_video_probe')),
+        media.probe_command(runtime, source, frames=True, decoder=source_decoder), 'source_video_probe')),
         allowed_bit_depths=source_bit_depths)
-    source_audio = json.loads(run(media.probe_command(runtime, source, packets=True), 'source_audio_probe'))
+    source_audio = json.loads(run(media.probe_command(runtime, source, packets=True,
+        decoder=source_decoder), 'source_audio_probe'))
     if output_profile in contract.DELIVERY_OUTPUT_PROFILES:
         suffix = media.delivery_suffix(source_audio)
         audio_mode = media.delivery_audio_mode(source_audio)
@@ -60,7 +62,8 @@ def main():
     encoder_help = (job / 'encoder_probe.stdout').read_text(encoding='utf8')
     if ('Encoder ' + encoder + ' ') not in encoder_help:
         raise RuntimeError('Official FFmpeg lacks the required Topaz output encoder')
-    source_pcm = media.decoded_pcm_digests(runtime, source, source_audio, env, job / 'source_pcm.stderr')
+    source_pcm = media.decoded_pcm_digests(runtime, source, source_audio, env, job / 'source_pcm.stderr',
+        decoder=source_decoder)
     settings = dict(spec['settings'])
     scale = settings.pop('scale')
     size_mode = settings.pop('size_mode', 'scale')
@@ -91,7 +94,7 @@ def main():
     (job / 'disk_preflight.json').write_text(json.dumps(disk_preflight, indent=2), encoding='utf8')
     pending = job / ('enhanced.pending' + suffix)
     command = contract.regular_command(runtime, source, pending, size_mode=size_mode,
-        output_profile=output_profile, audio_mode=audio_mode, **settings)
+        output_profile=output_profile, audio_mode=audio_mode, decoder=source_decoder, **settings)
     (job / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf8')
     revalidate()
     run(command, 'enhancement')
@@ -101,12 +104,15 @@ def main():
         geometry['post_ai_resampling'] = 'lanczos_exact_target_dimensions'
     else:
         geometry['post_ai_resampling'] = 'none'
+    output_decoder = media.video_decoder(runtime, pending, env, contract.CUVID_DECODERS)
     output_video = media.analyze_video(json.loads(run(
-        media.probe_command(runtime, pending, frames=True), 'output_video_probe')),
+        media.probe_command(runtime, pending, frames=True, decoder=output_decoder), 'output_video_probe')),
         allowed_bit_depths=output_bit_depths)
-    output_audio = json.loads(run(media.probe_command(runtime, pending, packets=True), 'output_audio_probe'))
+    output_audio = json.loads(run(media.probe_command(runtime, pending, packets=True,
+        decoder=output_decoder), 'output_audio_probe'))
     video_audit = media.compare_video(source_video, output_video, width, height)
-    output_pcm = media.decoded_pcm_digests(runtime, pending, output_audio, env, job / 'output_pcm.stderr')
+    output_pcm = media.decoded_pcm_digests(runtime, pending, output_audio, env, job / 'output_pcm.stderr',
+        decoder=output_decoder)
     if audio_mode == 'copy':
         audio_audit = media.compare_audio_packets(source_audio, output_audio, video_audit['common_shift'])
         audio_audit['pcm'] = media.compare_pcm_digests(source_pcm, output_pcm)
@@ -114,8 +120,8 @@ def main():
         audio_audit = media.compare_transcoded_audio(source_audio, output_audio,
             video_audit['common_shift'], source_pcm, output_pcm)
     run([str(runtime.executable('ffmpeg.exe')), '-v', 'error', '-xerror', '-err_detect', 'explode',
-        '-nostdin', '-protocol_whitelist', 'file,pipe', '-i', str(pending), '-map', '0:v:0',
-        '-map', '0:a?', '-f', 'null', '-'], 'strict_decode')
+        '-nostdin', '-protocol_whitelist', 'file,pipe', *contract.decoder_options(output_decoder),
+        '-i', str(pending), '-map', '0:v:0', '-map', '0:a?', '-f', 'null', '-'], 'strict_decode')
     revalidate()
     target = job / ('enhanced' + suffix)
     if target.exists():
diff --git a/tests/test_topaz_media.py b/tests/test_topaz_media.py
index 903a9ce..bedf64e 100644
--- a/tests/test_topaz_media.py
+++ b/tests/test_topaz_media.py
@@ -54,6 +54,16 @@ def test_explicit_sdr_bit_depth_profiles_never_silently_convert():
         analyze_video(video(), allowed_bit_depths=(10,))
 
 
+def test_cuvid_frame_format_overrides_provisional_stream_format():
+    # hevc_cuvid reports nv12 on the stream while its frames are p010le.
+    probe = video(pix_fmt='nv12')
+    for frame in probe['frames']:
+        frame['pix_fmt'] = 'p010le'
+    assert analyze_video(probe, allowed_bit_depths=(10,))['bit_depth'] == 10
+    with pytest.raises(ValueError, match='8-bit'):
+        analyze_video(probe)
+
+
 @pytest.mark.parametrize('kind', ['missing', 'duplicate', 'vfr', 'missing_pts', 'size',
     'interlace', 'hdr', 'ten_bit', 'rotation', 'sar'])
 def test_media_hazards_are_not_silently_normalized(kind):
diff --git a/tests/test_topaz_worker_dimensions.py b/tests/test_topaz_worker_dimensions.py
index f042edc..646a66e 100644
--- a/tests/test_topaz_worker_dimensions.py
+++ b/tests/test_topaz_worker_dimensions.py
@@ -39,7 +39,7 @@ def test_worker_geometry_and_publication_gates(runtime, monkeypatch, tmp_path, c
     monkeypatch.setattr(worker.sys, 'argv', ['worker', str(request)])
     monkeypatch.setattr(worker, 'local_module', lambda name:
         {'topaz_contract': contract, 'topaz_media': media}[name])
-    monkeypatch.setattr(media, 'decoded_pcm_digests', lambda *a: [])
+    monkeypatch.setattr(media, 'decoded_pcm_digests', lambda *a, **k: [])
     monkeypatch.setattr(worker.shutil, 'disk_usage',
         lambda _: SimpleNamespace(free=700 * 1024**2 if case == 'disk_advisory' else 10**12))
     calls = []
@@ -67,6 +67,8 @@ def test_worker_geometry_and_publication_gates(runtime, monkeypatch, tmp_path, c
             code = 1 if case == 'engine_failed' else 0
         elif '-xerror' in command:
             code = 1 if case == 'decode_failed' else 0
+        elif 'stream=codec_name' in command:
+            code = 0 if 'h264_cuvid' in command else 1
         else:
             pytest.fail(f'Unexpected external call: {command}')
         return SimpleNamespace(returncode=code)
@@ -99,3 +101,7 @@ def test_worker_geometry_and_publication_gates(runtime, monkeypatch, tmp_path, c
     enhancement = [command for command in calls if '-vf' in command]
     assert len(enhancement) == (0 if case == 'invalid_target' else 1)
     assert all('download=0' in command[command.index('-vf') + 1] for command in enhancement)
+    # Every call that opens video carries the detected CUVID decoder.
+    assert all('h264_cuvid' in command for command in calls if '-h' not in command)
+    assert all(command[command.index('-i') - 2:command.index('-i')] == ['-codec:v', 'h264_cuvid']
+               for command in enhancement)
```

</details>


> **Comment by T8mars (2026-10-11T16:44:56Z):**

感谢你完整的诊断和补丁。已参考并署名实现，修复随 [v1.98.2正式发行](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.98.2) 推入main，commit `359a3569cbb58bdaf9ba5c459fb4729d4cc84975`。

除了你提到的所有读取点，还保留旧版官方构建的软件路径：先读已安装解码器清单，H.264/HEVC软件解码器存在时显式选它，否则选CUVID，避免默认QSV。只有明确codec-ID不匹配允许其他编码沿用默认；崩溃、硬件失败、损坏及超时不静默回退。探测均在自有worker里，有30秒单次超时和原Job清理/资源门禁。放大与插帧都传递source/output decoder，Main10按完整decoded frame格式判断，混合格式和真实时序/音频异常仍拒绝。

117项定向CPU回归通过，后续资源继承简化只重核受影响10项；本机官方程序对既有72帧H.264的software/CUVID两路、FFV1严格解码通过，H.264时间轴与audio packet/PCM一致。本机官方构建仍有软件解码器，所以你的“缺软件解码器”环境由CPU模拟覆盖，尚不能代替你那台4080 SUPER/Topaz1.7.1的实际复测，也不冒称新增完整增强或插帧GPU质量验收。

请更新GitHub main/v1.98.2后重启后端，在原素材上再跑一次。若仍失败，请发新任务的decoder inventory/decoder probe stderr、request.json（脱敏）和完整错误。Registry目前Pending，Manager旧公开版本并不含修复。本Issue保持开放等你确认。



> **Comment by T8mars (2026-11-17T05:40:43Z):**

2026-11-17复核：此前参考并署名实现的修复已保留在[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)，本轮Topaz183项回归通过。输入decoder显式传到视频/音频探测、增强/插帧、PCM及最终解码；Main10仍按解码帧格式判断，真实硬件/损坏/超时错误继续传出。

原“默认QSV、无Intel核显即失败”的代码问题按已交付修复收尾关闭。缺软件解码器及两卡场景有模拟覆盖，未冒称本轮在你原4080 SUPER/Topaz1.7.1环境完成新GPU增强实测。更新T8后重启后端；当前源码仍有问题可带新decoder日志及脱敏请求重开。



### #29 [closed] PR: Add Topaz source timing repair before enhancement
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/29  by yangala 2026-10-05T15:46:42Z

## Problem

Official Topaz rejects any source whose video clock is not exactly constant. A single miswritten frame duration in an MP4 `stts` table shifts the timestamp of every later frame, and a splice or transcode can write it silently, so a finished cut gets refused right before enhancement starts.

The error text also reports "VFR, missing/duplicate frames" for this case, which sends the reader looking for a frame-count problem that is not there.

## Change

Adds `MiniMaxH3TopazSourcePrepareEXPT8` to sit between `Load Video` and the official Topaz nodes.

- Rewrites only the container duration table first. Encoded frames and the audio stream stay byte for byte identical, and the source file itself is never modified.
- Re-encodes only when no single duration explains the whole track, so a genuinely variable-rate source is never silently turned into a fake constant rate.
- A source that already conforms is passed straight through without writing a file.
- `repair_mode` picks between the two; `lossless` refuses to re-encode and reports the reason instead.

No version bump, so the Registry publish workflow is not triggered.

## Tests

- `tests/test_mp4_timing.py` (new): constant table, one miswritten run, a genuine variable-rate table that must be refused, a rate change past the drift limit, non-ISO-BMFF input, a track without a sample table, and a byte-level check that only the duration table is rewritten.
- `tests/test_nodes_topaz.py`, `tests/test_topaz_interpolation.py`: updated for the new registered node id. No existing case was weakened or removed.

Checked against a real 40s 10-bit source: 977 frames at 24fps, 10100587 bytes in and out, byte-level diff confined to the duration table.


> **Comment by T8mars (2026-11-17T05:42:28Z):**

Thanks for the proposal. I rechecked the unchanged head71a99d6a681aeb3d81e1f229aaf2ad78af5633db and reproduced the blocking counterexample by running its two pure timing functions: entries=[(900,512),(50,256),(50,768)] have1000frames/512000ticks, yet repair_delta returns512. The total duration is unchanged while genuine VFR timing is silently flattened.

The prior review's remaining points (complete timing/audio audit, owned cancelable worker, mode semantics and original registration order) are also not addressed by a new revision. Closing this implementation without merging to avoid changing valid source timing. This is a declined PR, not a claim that the source-timing feature has been delivered. A revised proposal addressing those points can be submitted separately; the Topaz decoder fixes remain independent. [Maintainer review summary](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/COMMUNITY_ISSUES_20261117.md).



### #28 [closed] ISSUE: 曜石导演台打开失败
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/28  by lin-zhenhua 2026-10-03T15:26:18Z

我用绘世启动器打开的comfyui，不知道会不会有影响，一开始每次打开是这样：

<img width="2560" height="1323" alt="Image" src="https://github.com/user-attachments/assets/bc341753-4c75-48c6-a80a-f5c13a5e8d87" />

之后用ai修复了，给大佬看看有没有参考建议：

<img width="1449" height="1132" alt="Image" src="https://github.com/user-attachments/assets/91b2834e-db9e-4f44-a2a7-24d57aeab48a" />

<img width="1439" height="1020" alt="Image" src="https://github.com/user-attachments/assets/a6e55bfb-8e6d-4b39-ba87-f2f58c07522e" />

改动的文件：

<img width="1482" height="1271" alt="Image" src="https://github.com/user-attachments/assets/e619e55b-1c79-4228-9e7c-1927e294f6f7" />

改完之后打开就正常了：

<img width="2560" height="1323" alt="Image" src="https://github.com/user-attachments/assets/60dcabef-a6f2-45eb-8323-dac825427e1e" />

> **Comment by lin-zhenhua (2026-10-03T15:29:44Z):**

改动文件没截全，补一张：

<img width="2305" height="684" alt="Image" src="https://github.com/user-attachments/assets/e440af7a-afce-409d-aef8-71b32d5bc8b1" />


> **Comment by T8mars (2026-10-11T16:35:01Z):**

感谢截图。已看到其中的具体原因：`api.apiURL` 添加 `/api` 前缀后，旧页面只识别 `/minimax_h3_t8/director/`，动态模块走错相对路径，导致 creation_library 等模块404。

当前main（v1.98.1起）已修复这一路径：页面同时识别 `/minimax_h3_t8/director/ui` 和 `/api/minimax_h3_t8/director/ui`，从服务的 `ui-modules/` 加载模块；后端提供对应白名单路由，静态安装文件夹改名也不再用于推断API基址。旧图不需要迁移。

请更新GitHub main，保存画布后重启自己的后端并硬刷新页面，再从侧栏打开。Registry仍停留旧公开版本，Manager提示已最新不代表已拿到此修复。若最新main仍失败，请提供浏览器Network里失败的完整URL/HTTP状态、控制台错误和插件commit，方便继续定向修复。先不直接关闭该Issue。



> **Comment by T8mars (2026-11-17T05:41:10Z):**

2026-11-17复核：/api前缀、服务ui-modules路由和安装目录名无关的加载路径修复已包含在[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)；本轮独立浏览器和路由回归已核对。原模块404问题按已修复关闭。

请更新T8节点源码并重启对应后端，再刷新前端。若当前源码仍打不开，可附完整失败URL/状态及控制台错误重开；仅刷新浏览器不会更新Python后端。



### #27 [closed] ISSUE: 文件名称太长，extract不了
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/27  by wywc2025 2026-10-02T23:25:09Z

这文件名称太长了吧，怎么搞？我之前尝试放C：也遇到太长的问题，extract不了。教教怎么做。

> **Comment by T8mars (2026-10-11T16:35:05Z):**

已处理：main中的37个超长示例已经缩短文件名，JSON内容和接线不变；旧名可通过 [filename-map.tsv](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/examples/workflows/filename-map.tsv) 查找。

现在仓库相对路径预算140个UTF-16单位、工作流JSON文件名最多96，已有CI阻止后续超长命名。当前发行检查最长137。无需开启Windows长路径权限；安装根目录本身过深仍建议缩短。

更新恢复方法见 [Windows路径说明](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/WINDOWS_PATHS.md)。请先保存自己画布/工作流，不要对整个ComfyUI做reset或删除。Manager的Registry旧包可能尚不含此修复，GitHub main与Registry可安装状态是两回事。请用最新main复测解压/更新，确认后再关闭本Issue。



> **Comment by T8mars (2026-11-17T05:41:19Z):**

已复核[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)发行包与工作区：仓库相对路径最长137/140、工作流JSON文件名均不超过96，22套已审图保留原JSON字节。长文件名修复和CI门禁已交付，按已修复关闭。

旧名对应关系仍在[filename-map.tsv](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/examples/workflows/filename-map.tsv)。安装根目录本身过深仍建议缩短；不需要删除自己的画布或本地改动。



### #26 [open] ISSUE: 请求海外30秒文武均衡版的原版部署材料
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/26  by liangzhazha2025-wq 2026-10-02T20:37:03Z

您好，想原样本地部署您发布的“30秒直出-最强文武双修均衡版”：
https://www.runninghub.ai/post/2103216908233744385

希望节点、模型、连线和参数与作者原版完全一致，不使用兼容替代。请问能否提供：
1. 对应原版整合包和准确版本清单，并确认V21是否对应这份工作流；
2. RunningHub Deepcleaner的合法本地原包，或确认它仅支持云端；
3. 节点6的原参考图，以及8个保存为None但仍连接的图片节点的正确处理方式。

已取得您公开的T8 1.85.0源码，但尚无法确认云端实际版本。烦请提供官方材料入口，谢谢！

<!-- support-request: T8-OVS-D30-RAW1 -->

> **Comment by T8mars (2026-10-11T16:35:11Z):**

已核对请求范围，但目前无法把RunningHub作品页的原云端部署认定为本项目已有的本地等价版本，也不能保证同名DeepCleaner就对应你要求的V21。

若要整理准确部署材料，需要原发布者提供：可导出的原工作流JSON、各自定义节点仓库与commit、具体模型文件名/revision/SHA、V21来源与允许本地使用/分发的授权说明，以及可公开的参考素材（可用脱敏替代）。尤其不能凭8NFE或几个模型名字臆造“原版部署”，也不代为分发私人素材/不明授权商业组件。

请补充以上资料中可公开部分；本Issue保持开放作为材料请求，不冒称复刻已完成。与节点运行bug的修复分开跟进。



> **Comment by T8mars (2026-11-17T05:42:17Z):**

2026-11-17复核：现有资料仍不足以确认该RunningHub作品的原版V21部署清单、DeepCleaner合法本地原包及私人参考图。当前节点源码和公开示例不等于同一份云端原版，不能补造节点commit/模型SHA或代发来源不明组件。

此材料请求保留开放，待来源方提供可公开原JSON、准确节点/模型版本及本地使用/分发授权后继续核对。



### #25 [closed] ISSUE: MiniMax H3 Learned Latent Upscale / H3学习型潜空间放大 (Advanced)    这个里面的像素大小 尺寸和h3的尺寸对不上，有些条件下对不上会出错。
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/25  by keng-666 2026-09-28T07:45:33Z

MiniMax H3 Learned Latent Upscale / H3学习型潜空间放大 (Advanced)    这个里面的像素大小 尺寸和h3的尺寸对不上，有些条件下对不上会出错。

如0.9   h3为1280*736     但这个节点里的是1280*704
如2    h3为1920*1088    但这个节点里是1856×1056

> **Comment by T8mars (2026-10-11T16:34:59Z):**

检查了当前实现：H3 Conditioning 的分辨率预设与 Learned Upscale 的 `target_megapixels` 不是同一个尺寸选择算法。后者按实际 LOW latent 的宽高比选择32对齐尺寸，不保证0.9MP/2MP自动得到某一个固定宽高。不能直接改旧默认，否则会改变已有工作流结果。

当前正确接法：Learned Upscale 输出的 `width`／`height` 必须连到 HIGH Conditioning 对应输入，再把 learned latent 与 HIGH av_latent 接入 Two-Pass Latent Reconcile，不能再让 HIGH 独立按同一MP重选另一套尺寸。

若要严格对齐已经指定的HIGH尺寸，可显式选择 `size_mode=target_dimensions`，填写1280×736或1920×1088；要保留明确目标尺寸而不是再次按源比例换算，可选择现有 `aspect_policy=honor_dimensions_exp`，但仍受放大倍率与最大宽高畸变检查，不保证任意源比例都能强拉伸。

请补充发生报错的工作流JSON（去掉私人素材）、LOW实际宽高、完整错误以及节点版本，我会据此检查是否还存在真实几何bug。本Issue先保持开放；不会只根据MP不同就静默改所有旧图。



> **Comment by T8mars (2026-11-17T05:41:27Z):**

已核对当前尺寸计算与回归，正确接线在[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)继续保留：相同MP不承诺两套算法选出相同宽高。将Learned Upscale的width/height接到HIGH Conditioning，再把对应latent送入Reconcile；HIGH不要独立按同一MP再选尺寸。需要固定宽高时显式用target_dimensions；honor_dimensions_exp仍保留真实倍率/畸变检查。

本条按已提供并验证的尺寸契约说明收尾，不静默修改旧图默认。若按此接线仍出现几何错误，可带LOW实际宽高、完整错误及脱敏最小图重开。



### #24 [closed] ISSUE: Official Topaz · 视频高清增强 (T8 EXP)一直报错
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/24  by aije 2026-09-26T02:27:06Z

节点在执行过程中发生错误。

[error.txt](https://github.com/user-attachments/files/32675474/error.txt)

这里是报错日志


<img width="4046" height="2167" alt="Image" src="https://github.com/user-attachments/assets/9e64b23d-2417-4219-b288-d0ab6576a29f" />





> **Comment by T8mars (2026-10-11T16:44:58Z):**

已读取附件日志，实际失败发生在 `source_video_probe`，exit `3221225477`（0xC0000005），与 #30 报告的默认QSV崩溃属于同一错误家族；仅凭日志尚不能确认你的硬件根因完全相同。

已在 [v1.98.2](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.98.2) /main `359a356` 修复H.264/HEVC的显式软件/CUVID解码选择，覆盖视频/音频探测、增强、PCM和最终decode，且保留原音画/取消审计。

请更新GitHub版本后重启后端，复测原素材；Registry仍Pending，Manager旧版可能拿不到修复。如果仍失败，请给新任务decoder日志、Topaz版本、CPU/GPU型号及完整错误。不会把所有Topaz报错都认定为QSV，也不靠强制转码隐藏问题。本Issue保持开放待确认。



> **Comment by T8mars (2026-11-17T05:40:52Z):**

2026-11-17复核：附件所示source_video_probe/0xC0000005错误对应的显式解码选择修复已包含在[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)；本轮Topaz183项回归通过，说明与#30保持一致。视频/音频探测及后续读取均不再依赖H.264/HEVC默认QSV选择。

按该已交付修复收尾；不把所有Topaz错误都归为同一原因。更新T8并重启后端后若仍报错，可重开并附新的完整错误、decoder日志、Topaz版本和CPU/GPU信息。



### #23 [closed] ISSUE: Initial DLSS FI route is single-GPU only
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/23  by cabenn 2026-09-21T09:28:29Z

老大，在使用TOPAZ两个结点的时候报 这个错误是为啥

[INFO] got prompt
[ERROR] !!! Exception during processing !!! Initial DLSS FI route is single-GPU only
[ERROR] Traceback (most recent call last):
  File "D:\ComfyUI2026\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI2026\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI2026\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\ComfyUI2026\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\ComfyUI2026\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\ComfyUI2026\ComfyUI\comfy_api\latest\_io.py", line 2046, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\ComfyUI2026\ComfyUI\custom_nodes\minimax-h3-audio-T8\h3_t8\nodes_topaz.py", line 206, in execute
    path, report = run_regular(runtime, source, root / ('task-' + uuid.uuid4().hex),
                   ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        model_id=model_id, width=target_width if size_mode == 'target_dimensions' else None,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<7 lines>...
        device=gpu_device_index, vram=vram_fraction, parameters=parameters,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        interrupt=throw_exception_if_processing_interrupted)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI2026\ComfyUI\custom_nodes\minimax-h3-audio-T8\h3_t8\topaz_runtime.py", line 448, in run_regular
    with SerialProbeLease(lease_path), NvmlResourceReader() as reader:
                                       ~~~~~~~~~~~~~~~~~~^^
  File "D:\ComfyUI2026\ComfyUI\custom_nodes\minimax-h3-audio-T8\h3_t8\dlss_fi_backend\resources.py", line 85, in __enter__
    raise RuntimeError("Initial DLSS FI route is single-GPU only")
RuntimeError: Initial DLSS FI route is single-GPU only

[INFO] Prompt executed in 2.33 seconds

> **Comment by cabenn (2026-09-21T10:14:52Z):**

COMY只开一块也没用，我只能设备管理暂时禁用就可了。


> **Comment by T8mars (2026-10-11T16:45:00Z):**

已确认并修复：Topaz控制器确实误用了DLSS专用NvmlResourceReader的 `count != 1` 门禁。现在 [v1.98.2](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.98.2) /main `359a356` 的增强和插帧改用Topaz专用reader，按选择的NVIDIA物理index读取UUID/资源，不再要求禁用第二张卡。DLSS自己的旧单卡门禁、取消与资源安全阈值保持。

双卡CPU模拟已覆盖index0/1、UUID及资源选取、越界清理，并确认DLSS门禁未放松；当前没有实际双NVIDIA硬件，不能把模拟说成真实双卡验收。GPU序号监控按nvidia-smi/NVML；混合厂商或设备排序不同的机器还需核对Topaz导出命令的设备映射。

请更新GitHub版本、重启后端，在不禁用第二张卡的情况下复测，并提供gpu_device_index/nvidia-smi顺序及新resources.jsonl（可脱敏）。Registry仍Pending。先保持Issue开放，确认后再关闭。



> **Comment by T8mars (2026-11-17T05:41:01Z):**

2026-11-17复核：Topaz增强/插帧已使用独立TopazResourceReader，修复保留在[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)。两卡模拟覆盖选择index0/1、UUID/资源及越界清理；Topaz不再调用DLSS的单卡门禁。原先需要禁用第二张卡的代码原因按已修复关闭。

这里没有宣称本轮完成真实双NVIDIA硬件验收。若当前版仍有设备映射问题，可提供nvidia-smi顺序、gpu_device_index及脱敏新任务日志重开。



### #22 [closed] ISSUE: 曜石导演台：LoRA 叠加下拉选择后界面无变化（附时间轴建议、定制咨询与工作台推荐）
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/22  by deciia 2026-09-21T00:08:15Z

先感谢 T8 的曜石导演台，非常喜欢，我准备在此基础上做深度定制。下面把使用中发现的问题、几个建议和一点咨询一并反馈。

## 环境

- Windows 11，Chrome 最新版
- ComfyUI（前端 1.52.7），安装后已完全重启
- comfyui-minimax-h3-audio-T8 源码 v1.85.0（commit 5def4aa，当前 GitHub 最新）

## 1. 问题：LoRA 叠加的下拉选择不生效

位置：曜石导演台 → 全局「模型、LoRA 与分辨率」→「LoRA 叠加」

复现步骤：

1. 点「＋ 添加 LoRA」，列表里出现一行 LoRA 条目
2. 在该行的 LoRA 名称下拉里选择一个 LoRA
3. 界面没有任何变化

实际表现：选择后界面没有任何变化，看不出这次选择被记录下来，也没有在别处体现所选 LoRA。

期望表现：选中的 LoRA 应被记录，并在界面上体现出来。

## 2. 建议：增加时间轴编辑

能否增加时间轴编辑功能，在时间轴上统一调整镜头与音频的时间关系（分镜时长、剪辑点、音频起止位置等）？目前镜头与声音的时长关系还要分开处理，如果有统一时间轴会方便很多。

## 3. 关于定制与分发的许可咨询

我计划深度定制曜石导演台，之后把定制版发布到 GitHub。注意到本仓库节点源码是 GPL-3.0-or-later，我的包同样采用 GPL-3.0-or-later。这种情况是否需要调整许可协议？还是有其它需要注意的地方（署名、NOTICE、版本同步等）？想先确认清楚再动手，避免后续麻烦。

## 4. 关于节点注册名的一点建议

之前 Deciia 的分块 PASS2 采样器被整合进了本仓库（节点 ID `DeciiaChunkedPass2Sampler`），我这边为了避免注册名冲突，不得不把自己的壳节点改名为 `DeciiaChunkedPass2SamplerLegacy`。

如果后续还要整合 Deciia 的节点，建议换一个名称（加前缀或后缀区分），或者不直接整合——两个同名节点同时存在会冲突。

## 5. 顺带推荐一个我做的工具：可自定义的工作台入口

受曜石导演台侧栏入口的启发，我把「工具启动器」做成了一个可自定义的 ComfyUI 节点（仓库 ComfyUI_Deciia_All），相关脚本：

- `workstation/store.py` —— 工具清单持久化（保存在 ComfyUI 用户目录，不在仓库内）+ HTTP 路由
- `workstation/nodes.py` —— 节点 `DeciiaWorkstation`（画布节点上的「打开工作台」按钮）
- `web/deciia_workstation.js` —— 侧栏「工作台」面板（卡片式入口，风格参考曜石导演台）

工具条目不写死、可自行增删；每条可选择在 ComfyUI 内浮层打开（与曜石导演台相同的体验）或在新窗口打开，并支持在线状态探测。这样导演台和本机其它服务、外部工具都能集中到一个面板里，不再占用宝贵的侧边栏按钮位。

仓库：https://github.com/deciia/ComfyUI_Deciia_All

仅供参考，也欢迎指正。

---

本 issue 由我的智能体助手小艾（Hermes Agent）代为发布。


> **Comment by T8mars (2026-10-11T16:35:03Z):**

当前main已经是独立LOW/HIGH LoRA列表：采样弹窗的名称下拉会在change时写入对应row并重新渲染，保存到项目时保留有序链；旧v1.85界面不作为当前版本复测依据。相关导演台模块404也已修复，见 #28 的处理说明。

请在当前main硬刷新后复测“添加→选择LoRA→保存项目→重开”，如果仍无变化，请发这个最小过程的录屏/控制台错误、commit和脱敏项目JSON，我会定位具体失效环节。时间轴可视化需求与这个选择器bug分开处理；不会为修选择器重写旧采样工作流。定制/第三方推荐不是已接受的功能或费用承诺。本Issue保持开放待复测。



> **Comment by T8mars (2026-11-17T05:41:35Z):**

已核对[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)的LoRA行选择、LOW/HIGH独立有序链、取消/应用及保存恢复路径，相关独立浏览器回归通过。原v1.85选择后无变化的bug按已交付修复关闭。

时间轴属于独立功能建议；定制分发请参照仓库LICENSE与第三方NOTICE。现有兼容节点ID保留，新增自有节点使用T8唯一命名。这里的关闭仅针对原选择器bug，没有把其它建议宣称为已全部开发。当前源码仍可复现时，请带最小步骤和脱敏项目重开。



### #21 [closed] ISSUE: 大佬會更新，支持hyperflow嗎？
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/21  by john2stai 2026-09-19T19:50:20Z

https://github.com/Saganaki22/ComfyUI-Hyperflow



> **Comment by T8mars (2026-10-11T16:35:07Z):**

已经加入独立HyperFlow路线，不是普通LoRA加载器硬套：专用Loader、Plan和Sampler，支持独立单采8步、连续4+4，以及LOW/HIGH分开接的8+4／部分4+4放大实验。原节点和普通4+4工作流保留。

接线、所需原始权重与限制见 [HyperFlow runtime说明](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/HYPERFLOW_RUNTIME_EXP.md)。目前要求完整H3 backbone，不能把pruned模型或普通加速LoRA当成同一合同；实跑/潜空间一致性并不等于所有场景质量认证。

请使用当前GitHub main。Registry最新公开版本仍是旧版，Manager未显示不代表源码中没有。你若遇到具体加载/接线错误，请发脱敏工作流和完整错误，我继续检查；先保持开放待确认。



> **Comment by T8mars (2026-11-17T05:41:42Z):**

已交付并核对：当前[v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1)包含专用HyperFlow Loader/Plan/Sampler与示例；Flash也走独立消费者，不当普通加速LoRA加载。本轮还修复Core0.45 FinalLayer网格接口，旧Core接口保留。

接线和各路线验证范围见[HyperFlow说明](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/HYPERFLOW_RUNTIME_EXP.md)及[本版社区说明](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/COMMUNITY_ISSUES_20261117.md)。本支持请求按已提供功能收尾；具体模型/接线问题可另附脱敏最小图继续定位。



### #20 [closed] ISSUE: native MiniMax H3 tokenizer lacks the byte-token contract
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/20  by Cloudreadypc 2026-09-15T19:50:47Z

ComfyUI version: 0.35.0

"MiniMax H3 Prompt Relay Conditioning / 分段提示词条件 (Advanced)" node gives an error. Thanks.


`[ERROR] !!! Exception during processing !!! Prompt Relay could not construct ComfyUI's native MiniMax H3 tokenizer for compatibility verification. Update or repair ComfyUI before retrying.
[ERROR] Traceback (most recent call last):
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\h3_t8\prompt_relay_advanced.py", line 484, in _verified_native_tokenizer_fallback
    raise TypeError("native MiniMax H3 tokenizer lacks the byte-token contract")
TypeError: native MiniMax H3 tokenizer lacks the byte-token contract

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\comfy_api\latest\_io.py", line 2046, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\h3_t8\nodes_prompt_relay_advanced.py", line 342, in execute
    return io.NodeOutput(*build_prompt_relay_conditioning(**kwargs))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\h3_t8\prompt_relay_advanced.py", line 1313, in build_prompt_relay_conditioning
    binding = build_prompt_relay_binding(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\h3_t8\prompt_relay_advanced.py", line 589, in build_prompt_relay_binding
    prompt_ids, hf = _prompt_token_ids(clip, conditioned_prompt)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\h3_t8\prompt_relay_advanced.py", line 520, in _prompt_token_ids
    return _verified_native_tokenizer_fallback(clip, prompt)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\h3_t8\prompt_relay_advanced.py", line 494, in _verified_native_tokenizer_fallback
    raise RuntimeError(
RuntimeError: Prompt Relay could not construct ComfyUI's native MiniMax H3 tokenizer for compatibility verification. Update or repair ComfyUI before retrying.`

> **Comment by T8mars (2026-09-17T16:14:01Z):**

Thanks for the report. A compatibility fix aimed at this byte-token contract failure is now in GitHub main, first published in [v1.84.0 / 4f54cdb](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/commit/4f54cdbd14ea5ea5eb21045cb36644b0b9aab6f8) and retained in [current main / 8e1476f](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/commit/8e1476ff786fab454d3d610094687599713dcf53).

Previously, a fast Qwen tokenizer without the legacy `byte_decoder` attribute could fail this check despite using a valid ByteLevel backend. The adapter now supports both the legacy byte decoder and an inspected fast ByteLevel decoder. Token IDs still come from the native connected H3 tokenizer; byte reconstruction must exactly match the compiled prompt. We have not disabled token-equivalence checks or substituted an arbitrary tokenizer.

Related regressions passed with real native slow/fast tokenizer implementations, including Chinese, Korean, media/special tags, emoji and whitespace. This is not yet confirmation that your specific StabilityMatrix / ComfyUI 0.35.0 installation is resolved.

Please update **the T8 custom node itself to current GitHub main**, then fully restart the ComfyUI backend and retry the same workflow. Updating only ComfyUI or refreshing the browser will not load this node-source fix. Registry currently still exposes an older version; see [#19](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/19), so please verify that your update actually fetched the GitHub source rather than reinstalling Registry 1.47.0. Use your normal update flow and preserve any local edits.

If it still fails, please attach the new complete traceback, the T8 and ComfyUI commits, transformers/tokenizers versions, and a minimal workflow showing the CLIP loader and Relay connections, with credentials/private material removed. No model weights are needed. We will keep this issue open until your current-version retest confirms the result.



> **Comment by T8mars (2026-11-17T05:41:49Z):**

Revalidated the native slow/fast ByteLevel tokenizer regressions for [v1.106.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.1). The existing adapter supports the native fast backend without requiring the legacy byte_decoder attribute, while preserving exact token/prompt-byte equivalence checks. No arbitrary tokenizer substitution was added.

Closing the implemented byte-token compatibility bug. Please update the T8 extension itself and restart the backend. If a current-source StabilityMatrix workflow still fails, reopen with its new traceback, Core/T8 commits, tokenizer library versions and minimal CLIP/Relay connections.



### #19 [open] ISSUE: Comfy Registry 新版激活跟踪：1.105.0 Active，v1.106.3 Pending
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/19  by ping13579246810 2026-09-11T17:45:04Z

## 問題描述

ComfyRegistry (api.comfy.org) 顯示 MiniMax H3 Audio T8 的最新版本是 `1.47.0` (2026-08-25)，但 GitHub repository 的實際最新版本是 `1.78.1`。

這導致 ComfyUI-Manager 誤判版本，顯示「衝突」警告。

## 證據

- GitHub pyproject.toml: `version = "1.78.1"`
- ComfyRegistry API: `Latest version: 1.47.0`
- Registry last updated: 2026-08-25T18:14:34Z

## 影響

用戶從 Manager 升級時，會看到 1.47.0 被標記為「最新版本」，產生虛假衝突警告。

## 建議

請更新 ComfyRegistry 的版本數據，或確認是否有其他方式發布新版本到 Registry。

以上由AI代筆整理

> **Comment by T8mars (2026-09-14T12:17:14Z):**

已复核确认，这不是“GitHub 没有发布成功”，而是 Registry 审核状态导致的：

- v1.80.0 已由官方 Action 上传成功；
- Registry API 中 v1.80.0 为 `NodeVersionStatusFlagged`，节点提取仍为 `pending`；
- 因此公开 `latest_version` 继续回退到最后一个 Active 版本 1.47.0；
- 当前扫描结果含 98 条信息级命中，其中既有把普通 `.bind()` 判为网络连接、把模型哈希判为钱包地址的明显误报，也有动态兼容编译、固定参数子进程和 Windows 进程存活检查等需要人工复核的真实敏感操作。

已经向 Comfy Registry 后端提交正式人工复核请求：
https://github.com/Comfy-Org/registry-backend/issues/233

另外，Manager 的语义版本字符串比较问题由上游 https://github.com/Comfy-Org/ComfyUI-Manager/issues/3272 单独跟踪；它不是 Registry 停在 1.47.0 的根因。

此 Issue 保持打开，直到 Registry 将新版本激活并且公开 `latest_version` 实际更新后再关闭。


> **Comment by T8mars (2026-09-14T12:49:08Z):**

紧急加固已合入 main：5ad5c49。现在发布动作上传后会再次查询 Registry 权威接口，只有目标版本为 NodeVersionStatusActive 且已成为公开 latest_version 才算成功；Flagged/Banned 会立即让 Action 失败，避免再次出现“上传绿灯但用户无法安装”。未来 Registry 包还会排除未注册、未被正式节点引用的 VRetouchEr 研究适配器及其上游 SpyNet 下载代码；Topaz、DLSS、FFmpeg、长视频和兼容功能继续保留。已通过 22 项定向测试、Ruff、版本同步检查与真实 comfy node pack 内容审计。注意：已上传的 v1.80.0 压缩包没有被替换，目前仍是 Flagged，公开最新版仍为 1.47.0，仍需等待 Comfy-Org/registry-backend#233 人工审核；本次没有重复发布同版本。


> **Comment by T8mars (2026-09-14T16:59:14Z):**

2026-09-15 处理进度：权威 API 仍是 v1.80.0 `Flagged`、提取 `pending`、公开 latest v1.47.0，尚未恢复 Registry 安装。

已在官方原工单补交扫描分类及需要具体确认的兼容编译位置，并纠正此前留言里的完整提交链接：
https://github.com/Comfy-Org/registry-backend/issues/233#issuecomment-5667621732

98 条 finding 虽然均为 info，但不意味着全部是误报；已明确请官方给出允许激活或需要整改的逐项结论，不以删除正常功能或隐藏代码绕审核。现有发布门禁的 7 项纯 Python 独立检查和发行版本一致性检查通过；在线检查正确返回审核阻塞。

本次没有重复上传 1.80.0，没有改节点运行代码或关闭本 Issue。只有目标版本 Active、公开 latest 更新且对应安装包核实后才会记作解决。GitHub 的后续 Topaz 提交并未进入已上传的旧 1.80.0 归档，二者不混为一谈。



> **Comment by T8mars (2026-09-17T16:13:55Z):**

2026-09-18 进度更新：已重查权威接口，**v1.84.0 仍为 NodeVersionStatusFlagged，公开 latest_version 仍是1.47.0**。此前只提到1.80.0的留言是历史状态，不能当成当前最新版已恢复。

[v1.84.0官方发布流程](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/actions/runs/35228092167)中，“Publish custom node”上传成功，但“Require the published version to be active and installable”检查失败。因此 GitHub 已发布、归档已上传与 Registry 可安装是三种不同状态。

已在官方原审核工单补交当前1.84.0版本ID及上述证据，并请求人工复核或明确整改清单：
https://github.com/Comfy-Org/registry-backend/issues/233#issuecomment-5717600844

补充说明：最新H16完成范围提交 [8e1476f](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/commit/8e1476ff786fab454d3d610094687599713dcf53)是GitHub源码更新，**没有替换已上传的1.84.0归档，也没有另发Registry版本**。当前等待官方给出激活或整改结论；本次没有重复上传、绕过安全检查或改节点运行代码。

需要最新修复的用户可先使用GitHub源码的正常更新方式并重启ComfyUI，保留本地修改；不要把Registry1.47.0误认为已包含这些新修复。本Issue保持开放，直到目标版本Active、公开latest更新并核实安装包后再关闭。



> **Comment by T8mars (2026-10-11T16:45:02Z):**

2026-10-12更新：当前GitHub正式版已到v1.98.2，但只读官方API显示 **1.98.0、1.98.1、1.98.2均为Pending，公开latest_version仍是1.47.0 Active**。这不是“Registry正常/新版本已可安装”，不能把GitHub Release或上传成功当成Manager可用。

官方复核单 [Comfy-Org/registry-backend#233](https://github.com/Comfy-Org/registry-backend/issues/233) 仍Open，最近更新9月27日，尚无新官方处理决定。当前发布流程 [38156589622](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/actions/runs/38156589622) 继续以Active/installable门禁核验，不绕过审核、不重复上传旧版本、不删除正常功能凑通过。

新代码可从GitHub main或正式Release取得；Registry审核独立跟踪，本Issue继续开放。用户已关闭的本地自动监控不恢复。



> **Comment by T8mars (2026-11-17T05:41:58Z):**

2026-11-17更新：Registry已不再停在1.47.0，当前权威API公开最新版为1.105.0 Active。但本次GitHub补丁v1.106.3仍为Pending，不能把上传成功说成Manager已可安装。

[v1.106.3](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/releases/tag/v1.106.3)已提供Core0.45兼容修复及两个下载包；官方发布流程继续保留Active/installable门禁，不重复上传同版本，也不删除正常功能绕扫描。本Issue继续作为当前新版激活跟踪；原后端审核单Comfy-Org/registry-backend#233仍Open。



### #18 [open] ISSUE: [Feature Request] Native tiled VAE/second-pass sampling compatible with the T8 dual-clock shape-bound sampler
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/18  by deciia 2026-09-10T21:42:50Z

## Background / 背景

MiniMax-H3 native resolution is limited by VRAM (12 GB cards like the RTX 5070 Ti can only handle ~0.6-0.7 MP for the first pass). The Goohai integration ships a tiled second-pass sampler (`GoohaiMinimaxH3TiledSampler`) that raises the resolution ceiling substantially, but it only works with the **Goohai conditioning chain** (`MiniMaxH3DualClockT8GH` + GH-built conds).

The T8 chain (`TwoPassDetailMixerT8Advanced` → shape-bound dual-clock sampler → PASS 2) produces noticeably better second-pass quality (DetailMixer tail/bias/STG + dedicated second-pass LoRA ratios), but **cannot be combined with tiled sampling today**.

## What we tried / 我们尝试过的方案

1. **Feeding the DetailMixer sampler into `GoohaiMinimaxH3TiledSampler`** — fails immediately: the T8 dual-clock sampler function (`sample_minimax_h3_dual_clock_euler`) is *shape-bound*: it validates the packed latent layout at setup and the per-tile packed layout doesn't match → `ValueError: packed latent changed after sampler setup`.

2. **A standalone tiled executor that re-implements the "synchronized Euler" idea** (per-step tile forwards + weighted blending, re-packing each tile's layout) and consumes the T8 DetailMixer guider/sampler/sigmas — hits `noise_scaling` mismatch: the BasicGuider/KSampler path builds `noise` from the *unpacked* video shape while `latent_image` stays packed (e.g. `RuntimeError: The size of tensor a (12960) must match the size of tensor b (311552) at dimension 2`), because the DetailMixer model wrapper carries shape-bound registrations that the generic KSampler pipeline doesn't propagate.

3. **Wrapping per-tile model payloads and re-registering `latent_shapes` in conds** — partial progress but the same class of mismatch resurfaces inside `model_sampling.noise_scaling` because the T8 model/conds pair re-derives the packed layout from the registered shapes each step.

## Root cause as we understand it / 根因理解

The T8 dual-clock sampler bakes the **full packed layout** (text tokens + full-frame video + audio) into the sampler at setup, and DetailMixer's model wrapper re-derives/validates that layout on every forward. Tiled sampling needs the layout to be **re-derived per tile** — which is exactly what the Goohai `sample_tiled._sample_synchronized_euler` does for its own (non-shape-bound) model chain, but the T8 chain's contracts (`TwoPassLatentReconcile` output + DetailMixer sampler + packed shape registration) have no hook point for that re-derivation.

## Request / 请求

Would it be possible to add native tiled-sampling support to the T8 second pass, e.g.:

- an optional `tiled` mode in `TwoPassDetailMixerT8Advanced`/PASS 2 that re-packs the per-tile payload and runs synchronized Euler per step (like Goohai's approach), while keeping DetailMixer's tail/bias/STG contracts, or
- exposing the shape-bound registration as an overridable hook so external tiled executors can drive the T8 sampler correctly?

This would let 12 GB users get the DetailMixer-quality second pass at 1.5-3x resolution without OOM. Happy to provide repro workflows, logs and test results.

## Environment / 环境

- ComfyUI 0.34.4 (aki v3.2 package), Python 3.11, RTX 5070 Ti 12 GB
- comfyui-minimax-h3-audio-T8 (09-09 build)
- Goohai-MiniMax-H3_Integration (09-09 build)
- Repro: P8a "T8 dual-pass" workflow, second pass at scale_by=1.5, tiling requested at 4 tiles/128 overlap

> **Comment by T8mars (2026-09-14T12:17:12Z):**

Thanks for the precise shape-bound failure analysis. T8 does now have an internal per-tile sampler rebind path for the low-sigma v3 and masked low-sigma v4 plan schemas: each tile rebuilds the dual-clock sampler against that tile's packed video/audio payload instead of reusing the full-frame registration. This addresses the immediate `packed latent changed after sampler setup` class of failure.

It does not yet satisfy the full request as a stable 12 GB quality route. Real H3 tests of `independent_tiles_exp` showed tile-local coordinate/texture divergence and visible seams. The quality-oriented default therefore remains `full_frame_safe + full_clip_safe`, which preserves global Transformer context but does not provide the requested spatial VRAM reduction.

The relevant diagnostic workflow is:
`examples/workflows/13-latent-upscale/2026-08-30_H3_Mask_Preserving_Low_Sigma_TwoPass_v4_Advanced_EXP.json`

Please treat `independent_tiles_exp` as research-only. If you test it on the reported 5070 Ti case, please share the workflow, peak VRAM, tile/overlap values and seam result. We are keeping this request open with the acceptance requirement that a future route must both fit the target hardware and pass seam/texture review; merely avoiding the shape-bound exception is not enough.


> **Comment by deciia (2026-09-16T07:40:13Z):**

**时间分块二采路线实测反馈（12GB / RTX 5070 Ti）**

基于 v3/v4 per-piece sampler rebind 机制，用薄壳节点（未改 T8 代码）跑通了 DetailMixer 直 feed 的时间分块二采。测试情况如下。

## 0. 测试条件

- 生成规格：12 秒 / 294 帧 / 24fps / `audio_mode=native`（audio_denoise_strength=1.0）
- 提示词：GH 导演台 text_keyframes 模式，多镜头时间轴提示词（走廊奔跑踩地砖点亮字母 + 空翻 + "GOOHAI" 定格，含 overall_soundscape 与 non_diegetic_music），完整覆盖 0–12 秒
- 模型栈：主模型 dasiwaHybridV2_int8；TE qwen3vl_32b nvfp4_awq；一采 turbo 4step 768p LoRA（bypass）；二采 LoRA 串（Combat@0.49 / dasiwa turbo v4@0.5 / 10ErosMax 4step@0.7 / Motion_Repair@0.2 / cinemagrade@0.5 / lms_r64@0.3）；SageAttention（Memory Efficient）+ SolAttnMiniMax(tau 1.3) + pytorch attention
- 采样：一采 dual_clock_euler（shift 12/3，parity plan → 4 NFE）；二采 DetailMixer (Advanced)（shift 12/3，Tail/Bias/STG/Restart 全关）→ 每段 5 NFE
- A/B 两组（同种子、同提示词）参数差异仅 3 处：一采画布 0.4MP vs 0.5MP、放大 1.2× vs 1.5×、PASS 2 路线（原生全画幅 vs 分块）

## 1. 时间分块可跑通

`DeciiaChunkedPass2Sampler` 作为 PASS 2 `SamplerCustomAdvanced` 的 1:1 替换：DetailMixer (Advanced) 的 sampler/sigmas 原样喂入；代码内构建 `masked_low_sigma.v4` 计划（`full_frame_safe + guarded_overlap_exp`、34帧/块、17帧重叠、`anchor_strength=0.999`、`joint_av_preserve_input`、`fp16`、`offload_after`）；目标尺寸=输入 latent×16，内部学习放大走 noop 路径（上游 LearnedLatentUpscale 是唯一放大器，544×960→832×1440 由它完成）。

- 分块：17 段全部完成（832×1440，294帧），16分38秒，节点显存约 3.0GB
- 对照：同规模全画幅二采约 9.6GB 被迫中止
- 正文报告的 `packed latent changed after sampler setup` 失败类未再出现

壳侧适配记录：`guider.original_conds` 是 `convert_cond()` 后的 dict 列表，调执行器前需逆变换为 `[tensor, metadata]` 对；执行器 `model` 参数是 ModelPatcher 不是裸模型。

## 2. 音频缺口

`joint_av_preserve_input` 按设计丢弃二采音频、返回一采原声。A/B 实测：原生全画幅 PASS 2 音频干净（二采联合精修过），分块版把一采杂音（0.5MP 直出）原样带进成片。

壳侧验证：临时代换模块级 `sample_piece`（已核实全局引用可拦截），截获每段被丢弃的二采精修音频，按段绝对音频区间（FRAME_RESCALE 映射）拼回全长、重叠区线性交叉渐变——一采杂音消除。

## 3. 曾出现的问题 1：尾块音频幻觉 → 已定位原因并缓解

首版测试（提示词分镜只覆盖 0–7s，7–12s 无描述）中，第 17 块（音频约 11.3s–12.22s）：输入音频自然衰减至接近静音（逐窗 RMS ~500→~100），精修输出却持续 ~4000–8000 RMS 直到文件末尾被硬切；11.12s（块 16/17 交界附近）另有一个能量尖峰。

定位：尾块输入处于"无提示词描述 + 接近静音"状态，模型在低σ精修时幻觉出输入中不存在的持续声音。

缓解验证：把提示词覆盖扩展到全部 12 秒（末段明确设计"音乐渐弱收尾 + 画面定格"）后重测——尾块不再出现持续幻觉音，末段能量为音乐自然收尾，无 0.47s 周期块接缝条纹，无块边界能量尖峰。

## 4. 曾出现的问题 2：块边界动作断裂 → 已定位原因并缓解

首版测试中第 9 秒（≈272 帧，恰为块 16→17 边界）：人物已站定收势，接最后一块后突然又开始跑——新块对运动方向重新决策，与上一块收势状态不一致（`anchor_strength=0.999` 非全锁，17 帧重叠不足以约束运动连续性；该区间亦无提示词镜头描述）。

缓解验证：同一次全时长提示词重测中，9 秒边界埋在连续奔跑镜头中间，视频侧动作断裂消失，音频侧边界能量尖峰同步消失。

结论（供参考）：`full_frame_safe + guarded_overlap_exp` 路线下，提示词对全部时间段的显式覆盖（尤其让分块边界落在连续动作/持续音乐中间、给尾块明确收尾描述）对块间一致性影响显著；无描述区间是块边界决策不一致的高发区。

——
薄壳节点、测试工作流与完整日志可按需提供。环境：RTX 5070 Ti 12GB / 动态显存加载 / 32GB RAM / T8 v1.81.0 (1464a9f)。



> **Comment by T8mars (2026-09-17T15:48:56Z):**

@deciia 感谢这份详细的实测反馈。我们想进一步复现你描述的**时间分块二采音频精修**，验证能否整理成独立可选功能；现有保留一采原音的默认路径会保持不变。

方便提供你提到的以下材料吗？

1. **完整薄壳节点源码**：`DeciiaChunkedPass2Sampler` 及所需辅助模块，尤其是截获 `sample_piece` 二采音频、`FRAME_RESCALE` 绝对区间映射、重叠交叉渐变、尾块裁剪／补齐的实现。如有修改前后的版本，也请注明。
2. **实际可复现的工作流 JSON**：包含完整时长提示词／时间线、seed、模型与 LoRA 标识及权重、采样器／sigma、时间块大小与重叠、音频策略，以及相关节点和 ComfyUI 的版本或 commit。
3. **完整运行日志**：包括每块的视频／音频区间、样本数、尾块处理及遇到的错误；如果已有波形或边界能量分析，也欢迎一起提供。
4. **对照成片或音频**：一采原声、分块二采保留一采音频、分块二采使用精修音频，最好来自同一输入／seed；如果已有全画幅二采对照，也请一并提供。你提到的尾块幻觉／边界尖峰失败样例及全时长提示词修正后的样例，对定位也很有帮助。

现有测试产物即可，不必重新跑长片；可以直接贴代码／附件，或提供仓库、压缩包和下载链接。素材不便公开时，可提供可公开的最小复现。请移除 API 密钥、账号信息等敏感内容，**无需上传模型权重**。如方便，也请注明薄壳代码许可或是否允许我们参考／集成并保留署名。

取得材料后，我们会重点核验音频绝对时间、重叠与尾段连续性，并分别检查画面、声音和口型，不把单次可跑通或外部显存数据当作普遍质量／性能结论。



> **Comment by deciia (2026-09-19T01:41:08Z):**

**@T8mars 材料整理完毕，逐项对应：**

**先说定位**：`refined_exp` 是 v4 `joint_av_preserve_input` 时期的临时自救方案，只能作为参考实现，不建议正式使用。实测 v1.84 的 `standard_joint_4plus4_exp` 后：节点更少、二采音频原生交付，与薄壳精修路线没有听感/画质的明显区别。所以对"能否整理成独立可选功能"：薄壳代码可作为当时缺口的行为参考，正式使用以官方 4plus4 为准，默认保留一采原音的路径维持不变即可。

**1. 薄壳源码** → https://github.com/deciia/ComfyUI_Deciia_All （GPL-3.0-or-later，与 T8 同许可；欢迎参考/集成，保留署名即可）
- `minimax_h3/h3_sampling/chunked_pass2.py`：完整壳。初版实跑通过后音频逻辑未改过，无修改前后两版。要点：临时代换 T8 模块全局 `sample_piece` 截获二采精修音频、`FRAME_RESCALE` 帧号→绝对音频区间映射、重叠区线性交叉渐变、异常时回退一采原声。
- `minimax_h3/h3_sampling/tiled_second_pass.py`：GH 备援壳，与音频无关，供参照。

**2. 可复现工作流** → 同仓 `examples/workflows/P8a-T8-H3-ChunkedPASS2-testB.json`
先说声抱歉：这份是我日常生产环境的工作流，外部依赖偏多（GH Integration、本包桥接节点、KJNodes/VHS/Easy-Use/Comfyroll 等，完整清单与上游链接见同目录 README），直接跑起来会比较麻烦。
双路线静音切换并存（官方 4plus4 与薄壳 refined_exp），ParityPlan [8,4,4]，窗口 136 / overlap 34 / anchor 0.999 / guarded_overlap_exp，二采独立 RandomNoise(fixed)。环境：ComfyUI 0.36.0 / 前端 1.52.7 / T8 v1.84.0 / RTX 5070 Ti 12GB。

**3. 日志**：只有控制台日志——薄壳没有逐块遥测输出，这是它的缺口之一；逐块区间/样本数日志如需要，后续跑长片时补。尾块幻觉（第 17 块输入 RMS ~100 而输出持续 4000–8000）与全时长提示词修正后现象消失的量化记录见 09-16 反馈，不重复贴。

**4. 同 seed 对照样片**（seed 683072603085674，dasiwaHybridV2_int8）→ 同仓 `examples/samples/`：
- `pass1_direct.mp4`（原 T8a_分块采样测试_一采_00002）：一采直连，第一遍原声，可听杂音
- `chunked_keep_first_pass_audio.mp4`（原 learned_..._exp_00002）：分块二采默认路径，同 seed 同提示词，音频按设计原样返回一采
- `chunked_refined_exp.mp4`（原 learned_..._exp_00005）：薄壳精修终版，同 seed、全时长修正版提示词，**非严格受控对照，特此注明**；主观听感杂音明显降低、无尾块幻觉

全画幅二采对照成片已清理，可用工作流 + fixed seed 复现。


> **Comment by T8mars (2026-10-11T16:35:09Z):**

感谢整理材料，已按你明确的定位分开处理：当前main保留原一采音频默认路径，同时提供原生joint 4+4及分离二采音频路线；时间分块多window对白作用域问题也已独立修复，不通过薄壳全局替换/异常静默回退掩盖错误。

空间切块不能等同时间分块：当前推荐仍为全画幅安全路线，独立空间tile是显式EXP，不声称人物/细节/接缝质量已普适解决。你的 `refined_exp` 仍按历史临时参考而不是正式推荐方案对待。

本Issue保留用于空间tile质量与native VAE需求，暂不因部分功能已覆盖就全项关闭。若有当前main仍缺的具体接线/显存问题，请给出最小复现图，避免重复开发已经提供的joint音频部分。感谢 deciia 的源码与测试反馈，后续实际复用会保留署名与GPL许可。



> **Comment by T8mars (2026-11-17T05:42:14Z):**

2026-11-17复核：已有原生joint双采及时间分块路径保留，本次补齐Core0.45 FinalLayer grids兼容。空间tile的shape重打包与接缝/细节质量仍需要独立完整模型验收，不能把时间分块或CPU可执行当成这项需求全部完成。

此Issue继续作空间tile/native VAE需求跟踪；不会用原先临时薄壳的全局替换或异常静默回退来制造通过。



### #17 [open] ISSUE: Unwanted audio before target speech in Ref2VA; similar BGM-like sound observed in video generation
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/17  by Atsushi888 2026-09-08T08:59:45Z

I’m reporting unwanted audio generated before the requested speech in a Japanese Reference Voice workflow.

**Environment**
- Apple M5, 32 GB unified memory
- ComfyUI on macOS / MPS
- PyTorch 2.12.1
- T8 1.74.0, commit `91c1b4e`
- Model: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`

**Reproduction**
Use `MiniMaxH3SpeechStudioT8` with:
- Japanese reference voice
- Target text: `こんにちは。今日はいい天気ですね。`
- Seed: `2608099002`
- `render_seconds=5.17`, resolution `32`
- 20 steps, `res_multistep`, `simple`
- Video/audio shifts: `12 / 3`
- `trim_mode=none`, `verify_mode=off`
- WavLM speaker verification: `report_cosine`

**Expected**
Only the requested speech, with natural breathing or brief silence.

**Observed**
The output contains a quiet, BGM-like introduction before the target speech. The saved file is 5.875 seconds long. In the stock20 outputs, approximately the first 2.7 seconds contain low-level sound before the signal level rises substantially.

The requested Japanese speech itself is usable in stock20.

**Comparisons**
- The introduction remained with both the default sub-quadratic attention and PyTorch SDPA.
- It also remained in non-EMA and EMA Turbo4 tests.
- Adding instructions to begin speaking immediately, with no music or ambient sound, did not remove it.
- Changing only the reference recording to a different speaker produced unwanted voice-like sound in the introductory section.
- I also recall hearing similar unwanted BGM-like sound during ordinary video generation, before these speech-only tests. This is an observation, not a controlled comparison proving a shared cause.

The T8 speech prompt already includes `No music` and `non_diegetic_music: N/A`. I also noticed that it uses `[audio reuse]` while describing the source as a voice-identity reference. I do not know whether this is related.

Your speech validation report documents an unrelated lead-in before target speech and its removal with `trim_exact_target`. Is the behavior above an expected Ref2VA limitation, a conditioning/workflow issue, or something that can be prevented during generation?

I can provide the workflow and generated FLAC examples. Please let me know which additional diagnostics would help.

> **Comment by T8mars (2026-09-14T12:17:11Z):**

Thanks for the controlled report. The current evidence does not show that changing attention, EMA/Turbo4, or adding “start immediately / no music” prompt text prevents this Ref2VA lead-in reliably.

`trim_exact_target` is a guarded post-generation workaround: it can remove material before the requested line only when local ASR finds the complete target text in order. It does not prevent H3 from generating the introductory sound, and it must not be described as a model-level fix. The `[audio reuse]` instruction tells Ref2VA to use the reference as voice identity while not repeating its words; we currently have no controlled evidence that this tag is the cause.

Please attach the reproducible workflow plus the reference and generated FLAC files you offered (with any private material removed). The most useful pair is the same seed with `verify_mode=off` and `trim_exact_target`, together with the report JSON, so we can distinguish generated lead-in from container/timeline padding and verify the ASR boundary. This Issue remains open pending those artifacts and a current-version reproduction.


> **Comment by T8mars (2026-11-17T05:42:10Z):**

2026-11-17复核：本次Core接口维护没有改变Ref2VA声音配方，也没有获得MPS原工作流/参考及生成FLAC的新受控复现。trim_exact_target仍是完整目标文本ASR匹配后才允许的生成后裁切，不能当作预防生成前导声的模型修复。

此项保留开放。继续定位需要脱敏工作流、参考/生成FLAC及同seed的未裁切/trim_exact_target对照和报告；现有材料不足以承诺已经解决。



### #16 [closed] ISSUE: Hard process abort (no traceback) in OpenVDN window_softmax_sdpa on SM120 Blackwell (torch 2.11) — mem-efficient SDPA dies on >100k-token packed sequences; cuDNN backend fixes it
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/16  by tuanlai1234 2026-09-06T11:11:48Z

## Summary

On Windows + SM120 Blackwell (RTX PRO 6000 Workstation / RTX 5090 class), any OpenVDN
composition (`MiniMaxH3VDNModelComposerT8Advanced`, stage `stage_dmd_8nfe`) crashes the
whole ComfyUI process at the first sampling step once the packed sequence is long
(~122k tokens at 2 MP / 8 s). Small workloads (0.1 MP / 1 s, ~22k tokens) run fine,
which makes it look intermittent.

`Fatal Python error: Aborted` — no Python traceback is printed, because the crash is a
native abort inside the mem-efficient SDPA kernel path, not a raised exception.

## Environment

- Windows 11, python_embeded 3.13.12
- torch 2.11.0+cu130 (allocator: cudaMallocAsync, DynamicVRAM/aimdo enabled)
- ComfyUI 0.34.5, frontend 1.49.6
- Node pack: minimax-h3-audio-t8 **v1.74.0** (commit `91c1b4e`)
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120, 96 GB)
- Model: `minimax_h3_fl2va_bf16.safetensors`, task I2VA, 2.0 MP 16:9, 8 s @ 24 fps
  → layout: 57 latent frames × 2040 tokens/frame, 122334 packed tokens

## Reproduction

1. UNETLoader → `MiniMaxH3VDNModelComposerT8Advanced` (stage_dmd_8nfe) → ExecutionPlan → SamplerCustomAdvanced
2. 2.0 MP + 8 s → crash at step 1, process gone, log ends right after
   `Model MiniMaxH3 prepared for dynamic VRAM loading. 63208MB Staged. 259 patches attached.`
3. Same graph at 0.1 MP + 1 s → completes normally (95 s)

faulthandler stack (captured via stderr tee; `--port`-side debug instance):

```
Fatal Python error: Aborted
Stack (most recent call first):
  File "...\comfyui-minimax-h3-audio-T8\vdn_h3_advanced.py", line 642 in _sdpa_rows
  File "...\comfyui-minimax-h3-audio-T8\vdn_h3_advanced.py", line 680 in window_softmax_sdpa
  File "...\comfyui-minimax-h3-audio-T8\vdn_h3_advanced.py", line 1328 in _vdn_attention
  File "...\comfyui-minimax-h3-audio-T8\vdn_h3_advanced.py", line 1373 in _vdn_block
  File "...\comfyui-minimax-h3-audio-T8\vdn_h3_advanced.py", line 1471 in hook
  File "...\comfy\ldm\minimax\model.py", line 706 in _forward
  ...
```

## Root cause (verified with a per-backend probe patch)

Instrumenting `_sdpa_rows` to try each `SDPBackend` explicitly, first call
`q=(10134, 56, 128) k=v=(122334, 56, 128) bf16, 70 GB VRAM free`:

| backend | result |
| --- | --- |
| `FLASH_ATTENTION` | `RuntimeError: No available kernel. Aborting execution.` — torch 2.11 ships no fused FA kernel for sm120 |
| `EFFICIENT_ATTENTION` | launches "OK", then the process **aborts** at the next ops (cutlass mem-efficient kernel on SM120, K=122334) |
| `CUDNN_ATTENTION` | **passes every call**, full 8-step DMD run completes (594 s) |
| `MATH` | not viable at this size (would materialize ~139 GB scores) |

So on SM120 the stock SDPA dispatcher silently routes to the mem-efficient cutlass
kernel, which hard-aborts on these long packed sequences. This is specific to consumer
Blackwell: the official OpenVDN stack targets B200/SM100 where FA4 exists, and
`window_softmax_sdpa`'s docstring already notes it exists as the "no FlexAttention/FA4/
Triton" Windows-safe fallback — it just needs a backend hint on sm120.

## Fix (validated end-to-end, 8/8 steps, 594 s, 2.0 MP / 8 s)

Prefer cuDNN attention when compute capability >= 12.0, keep the stock dispatcher
elsewhere:

```diff
--- a/vdn_h3_advanced.py
+++ b/vdn_h3_advanced.py
@@ -639,6 +639,28 @@
 def _sdpa_rows(
     q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
 ) -> torch.Tensor:
+    # Blackwell (SM120, e.g. RTX PRO 6000 / 5090) workaround: torch 2.11 ships
+    # no FlashAttention kernel for sm120 ("No available kernel") and the
+    # default dispatcher then silently routes to the mem-efficient cutlass
+    # kernel, which hard-aborts the process on OpenVDN's long packed
+    # sequences (>100k tokens).  cuDNN attention supports sm120 and these
+    # shapes, so prefer it there and keep the stock dispatcher elsewhere.
+    if _prefer_cudnn_sdpa():
+        try:
+            from torch.nn.attention import SDPBackend, sdpa_kernel
+
+            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
+                out = F.scaled_dot_product_attention(
+                    q.permute(1, 0, 2).unsqueeze(0),
+                    k.permute(1, 0, 2).unsqueeze(0),
+                    v.permute(1, 0, 2).unsqueeze(0),
+                    dropout_p=0.0,
+                    is_causal=False,
+                    scale=scale,
+                )
+            return out.squeeze(0).permute(1, 0, 2)
+        except (ImportError, RuntimeError):
+            pass  # cuDNN SDPA unavailable on this build: stock dispatcher
     out = F.scaled_dot_product_attention(
         q.permute(1, 0, 2).unsqueeze(0),
         k.permute(1, 0, 2).unsqueeze(0),
@@ -650,6 +672,20 @@
     return out.squeeze(0).permute(1, 0, 2)
+
+
+_CUDNN_SDPA_PREFERRED: bool | None = None
+
+
+def _prefer_cudnn_sdpa() -> bool:
+    global _CUDNN_SDPA_PREFERRED
+    if _CUDNN_SDPA_PREFERRED is None:
+        try:
+            major, _minor = torch.cuda.get_device_capability()
+            _CUDNN_SDPA_PREFERRED = major >= 12  # Blackwell and newer
+        except Exception:  # noqa: BLE001 - CPU-only or driver probe failed
+            _CUDNN_SDPA_PREFERRED = False
+    return _CUDNN_SDPA_PREFERRED
```

## Notes

- The project already guards the **Sage** path on this exact GPU family:
  `environment_audit.py` emits `sage_sm120_high_token_output_corruption_risk`
  ("SageAttention on compute capability 12.x at high H3 token counts has a reported
  pure-noise output failure") and the Strict Sage composer refuses >= 50,000 packed rows
  (`sage_sm120_guard` in features.json). The OpenVDN SDPA path has no equivalent guard —
  `compose_vdn_model` only records `compute_capability` into the audit receipt
  (vdn_h3_advanced.py:500/522) and then dispatches into the crashing stock SDPA path.
  A sibling guard, or the cuDNN fix above, closes the same class of failure.
- Numerically the run with cuDNN completes and produces correct-looking output; a
  parity A/B against the stock path on an sm90/sm1x0 GPU would still be worthwhile.
- Same crash is expected on RTX 5090 / 5070/5080 (all sm120) with torch 2.11.
- Alternative for the future: upstream Dao-AILab flash-attention is adding sm120 FA4
  support (PRs #2336/#2349/#2389/#2484, still unmerged as of 2026-09); once torch
  integrates it, the capability gate above can be relaxed.


> **Comment by T8mars (2026-09-14T12:17:12Z):**

Thanks for the detailed SM120 diagnosis. This was fixed in v1.75.0 (commit `3769d70`) and the fix remains in current main/v1.80.0.

For CUDA capability major 12, the OpenVDN exact-attention path now enters an explicit `sdpa_kernel([CUDNN_ATTENTION, FLASH_ATTENTION])` context. It never offers memory-efficient SDPA as a fallback, because that backend can abort the process before Python can catch an exception; it also avoids silently falling back to the quadratic math score matrix. Non-SM120 devices retain their previous dispatch behavior.

The routing is covered by `tests/test_vdn_sdpa_backend.py`. Please reopen with the current commit, Torch build and reproduction details if an SM120 hard abort persists.



### #15 [closed] ISSUE: OrcaRouter provider support for MiniMax H3 Audio T8 Prompt Provider Router
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/15  by martinzudergaming-a11y 2026-09-04T03:05:38Z

MiniMax H3 Audio T8 is a ComfyUI node pack for joint H3 video-and-audio generation, and I most appreciate its prompt discipline: the Prompt Provider Router rewrites a raw prompt into a pinned three-field contract (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`), protecting exact `<d>` dialogue and reference-image ordinals, and validates the result before sampling. For users running long serial H3 or MV lip-sync jobs, that guardrail is what makes a 288-node pack usable rather than daunting.

The same contract is exposed through a provider selector: local passthrough by default, plus OpenAI-compatible servers (OpenAI, LM Studio, llama.cpp) or Ollama via an endpoint and an `api_key_env`. The gap I see is a hosted-model option for users who want stronger prompt rewriting than a local 8B model can give without running their own server. A well-tested option here would give those users a new choice inside an abstraction they already trust.

## Proposal

Add [OrcaRouter](https://www.orcarouter.ai) as an optional, additional provider for the Prompt Provider Router node. It would not replace or change any existing provider: local passthrough stays the no-network default, and current LM Studio / llama.cpp / Ollama setups keep working unchanged.

I'm an engineer on the OrcaRouter team. This is a feature request, not a finished patch: I have not written or tested any code against this node.

## Why it fits

OrcaRouter exposes an OpenAI-compatible chat-completions API and uses standard API-key authentication, which maps directly onto the node's existing `openai_compatible` branch. The expected integration point is the current provider surface: `provider_mode`, an HTTPS `endpoint`, `provider_model`, and `api_key_env` under the existing `allow_remote_endpoint` guard. The envelope this node already speaks is the one OrcaRouter serves, so the work would be a documented first-class entry with sensible defaults rather than a new transport.

Relevant capabilities for this project:

- Many chat and reasoning models behind one endpoint, so different rewriters for the three-field contract can be compared without rewiring nodes.
- Automatic model routing and provider failover, so a stalled upstream does not kill a long serial queue.
- Usage tracking and budgets for batching long-video or Prompt Relay segments.

## Ecosystem and disclosure

OrcaRouter is already shipped by open-source projects such as RAGFlow, Dify, and promptfoo. Transparency note: OrcaRouter runs an optional open-source partner program in which approved OSS projects can receive a 5% revenue share from OrcaRouter usage attributed to their integration. Participation is not a prerequisite for adding support, and I am happy to follow whatever disclosure or governance rules this project prefers. The [built-with](https://www.orcarouter.ai/built-with) page lists existing integrations.

I would welcome the maintainers' view on whether this fits the node's direction. If it does, I am glad to submit an implementation PR once approved. Thanks again for the carefully engineered pack.


> **Comment by T8mars (2026-09-04T09:28:47Z):**

Thanks for reaching out. The current Prompt Provider Router already supports an OpenAI-compatible endpoint through its openai_compatible mode, including a custom HTTPS endpoint, custom model ID, and an API key read from an environment variable. Remote calls remain explicitly opt-in. If OrcaRouter follows that contract, it may already work without provider-specific transport code.`n`nWe will not add a branded preset based only on a promotional request. To evaluate one, please provide public technical documentation and a tested minimal configuration: the exact endpoint path, authentication header, model IDs, request/response and error examples, privacy and data-retention terms, routing/failover behavior, and a clear disclosure of affiliation or commercial relationship. A small reproducible PR or test fixture is welcome. We will keep this Issue open as an integration proposal while that evidence is gathered.


> **Comment by martinzudergaming-a11y (2026-09-08T17:03:04Z):**

Thanks — verified below against OrcaRouter's public docs and its unauthenticated API surface (I have no account key, so no keyed end-to-end call). The OpenAI-compatible contract matches the node's existing `openai_compatible` branch, so no provider-specific transport code appears necessary.

**Endpoint & auth**
- Base URL `https://api.orcarouter.ai/v1`; chat completions: `POST https://api.orcarouter.ai/v1/chat/completions`.
- Header: `Authorization: Bearer sk-orca-...` (keys start `sk-orca-`). Docs/examples use env var `ORCAROUTER_API_KEY`. Maps directly onto the node's `endpoint` / `provider_model` / `api_key_env` under `allow_remote_endpoint` + `confirm_provider_request`.

**Model IDs**
Provider-prefixed `<provider>/<model>` form, e.g. `openai/gpt-4o-mini`, `openai/gpt-5`, `anthropic/claude-haiku-4.5`, `deepseek/deepseek-chat`, `qwen/qwen3-max`, `google/gemini-2.5-flash`. Public `GET /v1/models` returns 194 IDs without auth (confirmed HTTP 200). An account-scoped virtual model `orcarouter/auto` (adaptive per-request model selection) is documented for every account.

**Request/response**
- Request body is standard chat.completions JSON: `model`, `messages`, `temperature`, `top_p`, `max_tokens` — the exact fields the node sends in `openai_compatible` mode.
- Response is read via `choices[0].message.content` — the same access path the node already implements, so the existing `_response_text` parser works unchanged.
- Caveat: the node always sends `temperature`. OrcaRouter forwards it, but its docs state some reasoning-model families (OpenAI GPT-5 series, Anthropic Opus 4.x, DeepSeek reasoner) reject `temperature`. A plain chat model (`openai/gpt-4o-mini`, `anthropic/claude-haiku-4.5`, `qwen/qwen3-max`) keeps the node's envelope valid; that is the model class to pin.

**Errors** (live check: invalid bearer key → HTTP 401)
- OpenAI-compatible envelope `{"error":{"message","type","code"}}`. `type` ∈ `orcarouter_api_error` / `upstream_error` / `openai_error` / `claude_error` / `gemini_error`; codes include `model_not_found`, `insufficient_user_quota`, `access_denied`. Status mapping documented: 400/401/403/404, 425 (model announced not live), 429 (workspace rate limit, carries `Retry-After`), 502 (all upstreams failed), 503. The node's existing non-2xx/parse-failure handling covers these envelopes.

**Privacy & data retention**
- Docs state zero retention of prompt/output content on the gateway; requests are forwarded in-memory and discarded. Retained per-request metadata: timestamp, key id, model, token counts, latency, HTTP status, truncated upstream error, source IP (billing/abuse) — kept 13 months; billing records 7 years. Opt-in request-log capture (stores prompt/response bodies) is off by default, 30-day default / 180-day server-enforced maximum. Upstream providers receive prompt content under their own terms.
- Public pages: docs.orcarouter.ai/operations/data-handling, /operations/zero-data-retention, /security/compliance/retention.

**Routing / failover**
- `orcarouter/auto`: per-account adaptive router (strategy configurable; `gated_adaptive` default on new accounts), picks a model per request.
- Explicit fallback chains: pass `models: [id1, id2, ...]` plus `route: "fallback"` (max 5; fails over on upstream 5xx/429; billing at the served model's rate; resolved model in `X-Orca-Resolved-Model` / `X-Orca-Fallback-Model` response headers).
- Since the three-field prompt contract wants reproducible output, a pinned plain-chat model is the safer default for the node; the auto router is better suited to comparing candidate rewriters than to fixed serial batches.

**Affiliation** (repeating the opening disclosure): I am an engineer on the OrcaRouter team. The OSS partner revenue-share program is optional and not a prerequisite for any integration.

**Minimal configuration to test against the existing node (no code change):**
- `provider_mode` = `openai_compatible`
- `endpoint` = `https://api.orcarouter.ai/v1/chat/completions`
- `provider_model` = `openai/gpt-4o-mini` (or another plain chat ID from `GET /v1/models`)
- `api_key_env` = `ORCAROUTER_API_KEY`
- `allow_remote_endpoint` = true, `confirm_provider_request` = true

If useful, I can put this into a small reproducible fixture: docs excerpt plus an example request/response and an error example, without any secret material, ready to drop into the repo's test conventions. Which form do you prefer — a fixture file here, or the implementation PR you described?



> **Comment by T8mars (2026-09-14T12:17:11Z):**

Thanks for supplying the concrete endpoint, envelope, error behavior, retention terms and affiliation disclosure.

The conclusion is that no OrcaRouter-specific transport is required: the existing `openai_compatible` mode already exposes the needed HTTPS endpoint, model ID and environment-variable API-key fields under explicit remote-request confirmation. A pinned plain-chat model is the supported configuration. Models which reject `temperature` are a general OpenAI-compatible parameter-negotiation concern, not a reason to add a branded provider branch.

If you would like to contribute, the preferred form is a small documentation/test-fixture PR showing a secret-free request/response/error example against the existing provider mode. We are not adding a branded preset or commercial routing default, so this feature request is being closed as already supported by the generic interface.



### #14 [closed] ISSUE: Where is Beta57?
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/14  by gamerzedbt-stack 2026-09-03T19:20:34Z

Scheduler Beta57 is missing. Any plans on adding it?

> **Comment by T8mars (2026-09-04T09:29:24Z):**

Implemented and released in v1.72.0. beta57 is now a normal scheduler option in MiniMax H3 Dual-Clock Sampler (T8), not an EXP-only path. It delegates to the installed ComfyUI beta scheduler with alpha=0.5 and beta=0.7, so RES4LYF is not required and no global ComfyUI scheduler registration is changed. native_flow remains the default, so existing workflows keep their previous behavior. Both the custom dual_clock_euler route and current native FLOW_AV samplers are covered by tests; one isolated real euler + beta57 H3 run also completed and passed strict video, audio, and combined decode. This establishes execution compatibility, not a universal quality claim for every model or Turbo LoRA.



### #13 [closed] ISSUE: [BUG] Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/13  by jakehkw 2026-08-25T12:18:54Z

Version: ComfyUI v0.33.4 + 52 commits (a1079ba)

使用"MiniMax H3 Prompt Relay Conditioning / 分段提示词条件 (Advanced)" 出現如下error:

Node threw an error during execution.


# ComfyUI Error Report
## Error Details
- **Node ID:** 73
- **Node Type:** MiniMaxH3PromptRelayConditioningT8Advanced
- **Exception Type:** RuntimeError
- **Exception Message:** RuntimeError: Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer

## Stack Trace
```
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\nodes_prompt_relay_advanced.py", line 342, in execute
    return io.NodeOutput(*build_prompt_relay_conditioning(**kwargs))
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 1127, in build_prompt_relay_conditioning
    binding = build_prompt_relay_binding(
        clip,
    ...<3 lines>...
        details["tokens"],
    )

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 503, in build_prompt_relay_binding
    prompt_ids, hf = _prompt_token_ids(clip, conditioned_prompt)
                     ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 433, in _prompt_token_ids
    inner, hf = _inner_tokenizer(clip)
                ~~~~~~~~~~~~~~~~^^^^^^

  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 428, in _inner_tokenizer
    raise RuntimeError("Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer")

```
## System Information
- **ComfyUI Version:** 0.33.0
- **Arguments:** ComfyUI\main.py --feature-flag show_signin_button=true --feature-flag enable_telemetry=true --enable-manager --extra-model-paths-config C:\Users\jackh\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1787474888383.yaml --input-directory C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input --output-directory C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output
- **OS:** win32
- **Python Version:** 3.13.12 (main, Feb 12 2026, 00:38:53) [MSC v.1944 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.12.1+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 4090 Laptop GPU : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 17170956288
  - **VRAM Free:** 15806234624
  - **Torch VRAM Total:** 0
  - **Torch VRAM Free:** 0

## Logs
```
2026-08-25T19:26:26.359875 - ^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.schemas
2026-08-25T19:26:26.359960 - ^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.tables
2026-08-25T19:26:26.360009 - ^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.types
2026-08-25T19:26:26.360039 - ^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.constraints
2026-08-25T19:26:26.360068 - ^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.defaults
2026-08-25T19:26:26.360097 - ^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.comments
2026-08-25T19:26:26.604929 - ^[[32m[INFO]^[[0m Adding extra search path checkpoints C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\checkpoints
2026-08-25T19:26:26.604979 - ^[[32m[INFO]^[[0m Adding extra search path classifiers C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\classifiers
2026-08-25T19:26:26.605003 - ^[[32m[INFO]^[[0m Adding extra search path clip_vision C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\clip_vision
2026-08-25T19:26:26.605021 - ^[[32m[INFO]^[[0m Adding extra search path configs C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\configs
2026-08-25T19:26:26.605042 - ^[[32m[INFO]^[[0m Adding extra search path controlnet C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\controlnet
2026-08-25T19:26:26.605060 - ^[[32m[INFO]^[[0m Adding extra search path controlnet C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\t2i_adapter
2026-08-25T19:26:26.605076 - ^[[32m[INFO]^[[0m Adding extra search path diffusers C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\diffusers
2026-08-25T19:26:26.605092 - ^[[32m[INFO]^[[0m Adding extra search path diffusion_models C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models
2026-08-25T19:26:26.605107 - ^[[32m[INFO]^[[0m Adding extra search path embeddings C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\embeddings
2026-08-25T19:26:26.605123 - ^[[32m[INFO]^[[0m Adding extra search path gligen C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\gligen
2026-08-25T19:26:26.605139 - ^[[32m[INFO]^[[0m Adding extra search path hypernetworks C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\hypernetworks
2026-08-25T19:26:26.605154 - ^[[32m[INFO]^[[0m Adding extra search path latent_upscale_models C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\latent_upscale_models
2026-08-25T19:26:26.605168 - ^[[32m[INFO]^[[0m Adding extra search path loras C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\loras
2026-08-25T19:26:26.605182 - ^[[32m[INFO]^[[0m Adding extra search path model_patches C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\model_patches
2026-08-25T19:26:26.605196 - ^[[32m[INFO]^[[0m Adding extra search path audio_encoders C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\audio_encoders
2026-08-25T19:26:26.605210 - ^[[32m[INFO]^[[0m Adding extra search path photomaker C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\photomaker
2026-08-25T19:26:26.605223 - ^[[32m[INFO]^[[0m Adding extra search path style_models C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\style_models
2026-08-25T19:26:26.605235 - ^[[32m[INFO]^[[0m Adding extra search path text_encoders C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\text_encoders
2026-08-25T19:26:26.605248 - ^[[32m[INFO]^[[0m Adding extra search path upscale_models C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\upscale_models
2026-08-25T19:26:26.605260 - ^[[32m[INFO]^[[0m Adding extra search path background_removal C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\background_removal
2026-08-25T19:26:26.605275 - ^[[32m[INFO]^[[0m Adding extra search path frame_interpolation C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\frame_interpolation
2026-08-25T19:26:26.605289 - ^[[32m[INFO]^[[0m Adding extra search path geometry_estimation C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\geometry_estimation
2026-08-25T19:26:26.605301 - ^[[32m[INFO]^[[0m Adding extra search path optical_flow C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\optical_flow
2026-08-25T19:26:26.605315 - ^[[32m[INFO]^[[0m Adding extra search path detection C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\detection
2026-08-25T19:26:26.605328 - ^[[32m[INFO]^[[0m Adding extra search path vae C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae
2026-08-25T19:26:26.605342 - ^[[32m[INFO]^[[0m Adding extra search path vae_approx C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae_approx
2026-08-25T19:26:26.605356 - ^[[32m[INFO]^[[0m Adding extra search path LLM C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\LLM
2026-08-25T19:26:26.605370 - ^[[32m[INFO]^[[0m Adding extra search path onnx C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\onnx
2026-08-25T19:26:26.605385 - ^[[32m[INFO]^[[0m Adding extra search path sams C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\sams
2026-08-25T19:26:26.605398 - ^[[32m[INFO]^[[0m Adding extra search path ultralytics C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\ultralytics
2026-08-25T19:26:26.605412 - ^[[32m[INFO]^[[0m Adding extra search path clip C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\clip
2026-08-25T19:26:26.605425 - ^[[32m[INFO]^[[0m Adding extra search path unet C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\unet
2026-08-25T19:26:26.605451 - ^[[32m[INFO]^[[0m Setting output directory to: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output
2026-08-25T19:26:26.605480 - ^[[32m[INFO]^[[0m Setting input directory to: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input
2026-08-25T19:26:26.935851 - [START] Security scan2026-08-25T19:26:26.935867 - 
2026-08-25T19:26:27.884934 - [DONE] Security scan2026-08-25T19:26:27.884944 - 
2026-08-25T19:26:27.886820 - ** ComfyUI startup time:2026-08-25T19:26:27.886855 -  2026-08-25T19:26:27.886870 - 2026-08-25 19:26:27.8862026-08-25T19:26:27.886884 - 
2026-08-25T19:26:27.886899 - ** Platform:2026-08-25T19:26:27.886910 -  2026-08-25T19:26:27.886922 - Windows2026-08-25T19:26:27.886933 - 
2026-08-25T19:26:27.886944 - ** Python version:2026-08-25T19:26:27.886954 -  2026-08-25T19:26:27.886966 - 3.13.12 (main, Feb 12 2026, 00:38:53) [MSC v.1944 64 bit (AMD64)]2026-08-25T19:26:27.886976 - 
2026-08-25T19:26:27.886987 - ** Python executable:2026-08-25T19:26:27.886998 -  2026-08-25T19:26:27.887009 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe2026-08-25T19:26:27.887020 - 
2026-08-25T19:26:27.887031 - ** ComfyUI Path:2026-08-25T19:26:27.887041 -  2026-08-25T19:26:27.887051 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI2026-08-25T19:26:27.887061 - 
2026-08-25T19:26:27.887073 - ** ComfyUI Base Folder Path:2026-08-25T19:26:27.887083 -  2026-08-25T19:26:27.887093 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI2026-08-25T19:26:27.887103 - 
2026-08-25T19:26:27.887114 - ** User directory:2026-08-25T19:26:27.887125 -  2026-08-25T19:26:27.887479 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user2026-08-25T19:26:27.887523 - 
2026-08-25T19:26:27.887536 - ** ComfyUI-Manager config path:2026-08-25T19:26:27.887546 -  2026-08-25T19:26:27.887554 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\__manager\config.ini2026-08-25T19:26:27.887563 - 
2026-08-25T19:26:27.887575 - ** Log path:2026-08-25T19:26:27.887583 -  2026-08-25T19:26:27.887590 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\comfyui.log2026-08-25T19:26:27.887598 - 
2026-08-25T19:26:27.887633 - [ComfyUI-Manager] WARN: Unsafe - SSL verification bypass option is Enabled. (see C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\__manager\config.ini)2026-08-25T19:26:27.887643 - 
2026-08-25T19:26:28.765337 - ^[[32m[INFO]^[[0m [PRE] ComfyUI-Manager
2026-08-25T19:26:28.768225 - ^[[32m[INFO]^[[0m 
Prestartup times for custom nodes:
2026-08-25T19:26:28.768295 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\rgthree-comfy
2026-08-25T19:26:28.768324 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-easy-use
2026-08-25T19:26:28.768347 - ^[[32m[INFO]^[[0m 
2026-08-25T19:26:29.823638 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\cuda\__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
2026-08-25T19:26:31.985669 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'na3d', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_nvfp4', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}
2026-08-25T19:26:31.985757 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend eager: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_embedding', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_mxfp8', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'na3d', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_mxfp8', 'scaled_mm_nvfp4', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}
2026-08-25T19:26:31.985828 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend triton: {'available': True, 'disabled': True, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'int8_linear', 'na3d', 'quantize_and_rotate_rowwise', 'quantize_int8_rowwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'w4a8_int8_linear']}
2026-08-25T19:26:31.985862 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend hip: {'available': False, 'disabled': False, 'unavailable_reason': 'PyTorch ROCm/HIP runtime not available', 'capabilities': []}
2026-08-25T19:26:31.993298 - ^[[32m[INFO]^[[0m Checkpoint files will always be loaded safely.
2026-08-25T19:26:32.054280 - ^[[32m[INFO]^[[0m Total VRAM 16376 MB, total RAM 32505 MB
2026-08-25T19:26:32.054394 - ^[[32m[INFO]^[[0m pytorch version: 2.12.1+cu130
2026-08-25T19:26:32.054662 - ^[[32m[INFO]^[[0m Set vram state to: NORMAL_VRAM
2026-08-25T19:26:32.054765 - ^[[32m[INFO]^[[0m Device: cuda:0 NVIDIA GeForce RTX 4090 Laptop GPU : cudaMallocAsync
2026-08-25T19:26:32.069951 - ^[[32m[INFO]^[[0m Using async weight offloading with 2 streams
2026-08-25T19:26:32.070316 - ^[[32m[INFO]^[[0m Enabled pinned memory 13002.0
2026-08-25T19:26:32.139962 - ^[[32m[INFO]^[[0m Using pytorch attention
2026-08-25T19:26:35.362222 - ^[[32m[INFO]^[[0m aimdo: src-win/cuda-detour.c:38:INFO:aimdo_setup_hooks: installing 6 hooks
2026-08-25T19:26:35.381153 - ^[[32m[INFO]^[[0m aimdo: src/control.c:262:INFO:comfy-aimdo NVML pressure enabled
2026-08-25T19:26:35.418347 - ^[[32m[INFO]^[[0m aimdo: src-win/shmem-detect.c:80:INFO:comfy-aimdo WDDM adapter match: NVIDIA GeForce RTX 4090 Laptop GPU runtime_luid=00000000:0001089b dxgi_luid=00000000:0001089b
2026-08-25T19:26:35.418416 - ^[[32m[INFO]^[[0m aimdo: src/control.c:277:INFO:comfy-aimdo inited for GPU: NVIDIA GeForce RTX 4090 Laptop GPU (VRAM: 16375 MB)
2026-08-25T19:26:35.418495 - ^[[32m[INFO]^[[0m DynamicVRAM support detected and enabled
2026-08-25T19:26:35.418528 - ^[[32m[INFO]^[[0m Python version: 3.13.12 (main, Feb 12 2026, 00:38:53) [MSC v.1944 64 bit (AMD64)]
2026-08-25T19:26:35.418552 - ^[[32m[INFO]^[[0m ComfyUI version: 0.33.0
2026-08-25T19:26:35.419064 - ^[[32m[INFO]^[[0m comfy-aimdo version: 0.4.13
2026-08-25T19:26:35.419653 - ^[[32m[INFO]^[[0m comfy-kitchen version: 0.2.31
2026-08-25T19:26:35.423631 - ^[[32m[INFO]^[[0m comfyui-frontend-package version: 1.49.6
2026-08-25T19:26:35.423690 - ^[[32m[INFO]^[[0m comfyui-workflow-templates version: 0.11.46
2026-08-25T19:26:35.423720 - ^[[32m[INFO]^[[0m comfyui-embedded-docs version: 0.5.10
2026-08-25T19:26:35.423744 - ^[[32m[INFO]^[[0m comfy-kitchen version: 0.2.31
2026-08-25T19:26:35.423767 - ^[[32m[INFO]^[[0m comfy-aimdo version: 0.4.13
2026-08-25T19:26:35.425040 - ^[[32m[INFO]^[[0m [Prompt Server] web root: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfyui_frontend_package\static
2026-08-25T19:26:35.425617 - ^[[32m[INFO]^[[0m Asset seeder disabled
2026-08-25T19:26:35.425693 - ^[[32m[INFO]^[[0m [START] ComfyUI-Manager
2026-08-25T19:26:35.563581 - [ComfyUI-Manager] Using GitPython backend2026-08-25T19:26:35.563613 - 
2026-08-25T19:26:35.600413 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] network_mode: public
2026-08-25T19:26:35.612540 - ^[[1m^[[33m[WARNING]^[[0m [ComfyUI-Manager] The matrix sharing feature has been disabled because the `matrix-nio` dependency is not installed.
	To use this feature, please run the following command:
	C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe -m pip install matrix-nio

2026-08-25T19:26:36.808386 - ^[[32m[INFO]^[[0m No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
2026-08-25T19:26:37.772552 - [Crystools ^[[0;32mINFO^[[0m] Crystools version: 1.27.4
2026-08-25T19:26:37.814704 - [Crystools ^[[0;32mINFO^[[0m] Platform release: 11
2026-08-25T19:26:37.814777 - [Crystools ^[[0;32mINFO^[[0m] JETSON: Not detected.
2026-08-25T19:26:37.817219 - [Crystools ^[[0;32mINFO^[[0m] CPU: Intel(R) Core(TM) i9-14900HX - Arch: AMD64 - OS: Windows 11
2026-08-25T19:26:37.817322 - [Crystools ^[[0;32mINFO^[[0m] pynvml (NVIDIA) initialized.
2026-08-25T19:26:37.817442 - [Crystools ^[[0;32mINFO^[[0m] GPU/s:
2026-08-25T19:26:37.817517 - [Crystools ^[[0;32mINFO^[[0m] 0) NVIDIA GeForce RTX 4090 Laptop GPU
2026-08-25T19:26:37.817573 - [Crystools ^[[0;32mINFO^[[0m] NVIDIA Driver: 610.88
2026-08-25T19:26:40.322883 - [ComfyUI-Easy-Use] server: 2026-08-25T19:26:40.322912 - v1.3.6 2026-08-25T19:26:40.322925 - Loaded2026-08-25T19:26:40.322940 - 
2026-08-25T19:26:40.322958 - [ComfyUI-Easy-Use] web root: 2026-08-25T19:26:40.322969 - C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-easy-use\web_version/v2 2026-08-25T19:26:40.322978 - Loaded2026-08-25T19:26:40.322989 - 
2026-08-25T19:26:40.339333 - ^[[32m[INFO]^[[0m ### Loading: ComfyUI-Impact-Pack (V8.28.3)
2026-08-25T19:26:40.680743 - ^[[32m[INFO]^[[0m [Impact Pack] Wildcard total size (0.00 MB) is within cache limit (50.00 MB). Using full cache mode.
2026-08-25T19:26:40.682028 - ^[[32m[INFO]^[[0m [Impact Pack] Wildcards loading done.
2026-08-25T19:26:40.683281 - ^[[32m[INFO]^[[0m ### Loading: ComfyUI-Impact-Subpack (V1.3.5)
2026-08-25T19:26:40.685944 - ^[[32m[INFO]^[[0m [Impact Pack/Subpack] Using folder_paths to determine whitelist path: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\ComfyUI-Impact-Subpack\model-whitelist.txt
2026-08-25T19:26:40.686143 - ^[[32m[INFO]^[[0m [Impact Pack/Subpack] Ensured whitelist directory exists: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\ComfyUI-Impact-Subpack
2026-08-25T19:26:40.686410 - ^[[32m[INFO]^[[0m [Impact Pack/Subpack] Loaded 0 model(s) from whitelist: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\ComfyUI-Impact-Subpack\model-whitelist.txt
2026-08-25T19:26:40.763430 - ^[[32m[INFO]^[[0m [Impact Subpack] ultralytics_bbox: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\models\ultralytics\bbox
2026-08-25T19:26:40.763498 - ^[[32m[INFO]^[[0m [Impact Subpack] ultralytics_segm: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\models\ultralytics\segm
2026-08-25T19:26:40.847139 - hus' utility nodes: 2026-08-25T19:26:40.847164 - loaded2026-08-25T19:26:40.847179 - 
2026-08-25T19:26:41.093392 - [tinyterraNodes] 2026-08-25T19:26:41.093421 - Loaded2026-08-25T19:26:41.093434 - 
2026-08-25T19:26:41.097098 - ^[[1m^[[33m[WARNING]^[[0m Traceback (most recent call last):
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\nodes.py", line 2263, in load_custom_node
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 1023, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_tinyterranodes\__init__.py", line 110, in <module>
    update_config()
    ~~~~~~~~~~~~~^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_tinyterranodes\__init__.py", line 34, in update_config
    config_write("Versions", node, version)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_tinyterranodes\__init__.py", line 80, in config_write
    config = get_config()
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_tinyterranodes\__init__.py", line 28, in get_config
    config.read(config_path)
    ~~~~~~~~~~~^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\configparser.py", line 737, in read
    self._read(fp, filename)
    ~~~~~~~~~~^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\configparser.py", line 1052, in _read
    ParsingError._raise_all(self._read_inner(fp, fpname))
                            ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\configparser.py", line 1081, in _read_inner
    self._handle_rest(st, line, fpname)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\configparser.py", line 1105, in _handle_rest
    raise MissingSectionHeaderError(fpname, st.lineno, line)
configparser.MissingSectionHeaderError: File contains no section headers.
file: 'C:\\Users\\jackh\\AppData\\Local\\Comfy-Desktop\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\custom_nodes\\comfyui_tinyterranodes\\config.ini', line: 1
'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

2026-08-25T19:26:41.097374 - ^[[1m^[[33m[WARNING]^[[0m Cannot import C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_tinyterranodes module for custom nodes: File contains no section headers.
file: 'C:\\Users\\jackh\\AppData\\Local\\Comfy-Desktop\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\custom_nodes\\comfyui_tinyterranodes\\config.ini', line: 1
'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
2026-08-25T19:26:41.110029 - USDU batch patches applied successfully.
2026-08-25T19:26:41.110055 - ^[[32m[INFO]^[[0m USDU batch patches applied successfully.
2026-08-25T19:26:41.265475 - 
2026-08-25T19:26:41.265515 - [rgthree-comfy] Loaded 48 epic nodes. 🎉2026-08-25T19:26:41.265528 - 
2026-08-25T19:26:41.265539 - 
2026-08-25T19:26:41.265568 - [rgthree-comfy] ComfyUI's new Node 2.0 rendering may be incompatible with some rgthree-comfy nodes and features, breaking some rendering as well as losing the ability to access a node's properties (a vital part of many nodes). It also appears to run MUCH more slowly spiking CPU usage and causing jankiness and unresponsiveness, especially with large workflows. Personally I am not planning to use the new Nodes 2.0 and, unfortunately, am not able to invest the time to investigate and overhaul rgthree-comfy where needed. If you have issues when Nodes 2.0 is enabled, I'd urge you to switch it off as well and join me in hoping ComfyUI is not planning to deprecate the existing, stable canvas rendering all together.
2026-08-25T19:26:41.265581 - 
2026-08-25T19:26:41.618644 - WAS Node Suite: 2026-08-25T19:26:41.618675 - OpenCV Python FFMPEG support is enabled2026-08-25T19:26:41.618690 - 
2026-08-25T19:26:41.618723 - WAS Node Suite 2026-08-25T19:26:41.618733 - Warning: 2026-08-25T19:26:41.618745 - `ffmpeg_bin_path` is not set in `C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\was-ns\was_suite_config.json` config file. Will attempt to use system ffmpeg binaries if available.2026-08-25T19:26:41.618758 - 
2026-08-25T19:26:41.938539 - WAS Node Suite: 2026-08-25T19:26:41.938566 - Finished.2026-08-25T19:26:41.938578 -  2026-08-25T19:26:41.938587 - Loaded2026-08-25T19:26:41.938596 -  2026-08-25T19:26:41.938604 - 2202026-08-25T19:26:41.938612 -  2026-08-25T19:26:41.938620 - nodes successfully.2026-08-25T19:26:41.938632 - 
2026-08-25T19:26:41.938656 - 
	2026-08-25T19:26:41.938665 - "Art is the freedom to create, explore, and inspire."2026-08-25T19:26:41.938674 -  - Unknown2026-08-25T19:26:41.938683 - 
2026-08-25T19:26:41.938692 - 
2026-08-25T19:26:41.942053 - ^[[32m[INFO]^[[0m 
Import times for custom nodes:
2026-08-25T19:26:41.942120 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\video-output-bridge
2026-08-25T19:26:41.942152 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\websocket_image_save.py
2026-08-25T19:26:41.942178 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_hus_utils
2026-08-25T19:26:41.942199 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\cg-use-everywhere
2026-08-25T19:26:41.942219 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-image-studio
2026-08-25T19:26:41.942240 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context-MultiRef
2026-08-25T19:26:41.942261 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-minimax-h3-latent-upscaler
2026-08-25T19:26:41.942282 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-h3-motion-context-auto-chain-addon
2026-08-25T19:26:41.942302 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_essentials
2026-08-25T19:26:41.942323 - ^[[32m[INFO]^[[0m    0.0 seconds (IMPORT FAILED): C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_tinyterranodes
2026-08-25T19:26:41.942347 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-custom-scripts
2026-08-25T19:26:41.942367 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-sol-attn-rocm
2026-08-25T19:26:41.942397 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_ultimatesdupscale
2026-08-25T19:26:41.942415 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\rgthree-comfy
2026-08-25T19:26:41.942435 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-kjnodes
2026-08-25T19:26:41.942455 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-videohelpersuite
2026-08-25T19:26:41.942474 - ^[[32m[INFO]^[[0m    0.1 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-impact-subpack
2026-08-25T19:26:41.942492 - ^[[32m[INFO]^[[0m    0.1 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8
2026-08-25T19:26:41.942512 - ^[[32m[INFO]^[[0m    0.2 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Crystools
2026-08-25T19:26:41.942531 - ^[[32m[INFO]^[[0m    0.2 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui_nvidia_rtx_nodes
2026-08-25T19:26:41.942552 - ^[[32m[INFO]^[[0m    0.3 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-impact-pack
2026-08-25T19:26:41.942571 - ^[[32m[INFO]^[[0m    0.7 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\was-ns
2026-08-25T19:26:41.942597 - ^[[32m[INFO]^[[0m    2.5 seconds: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-easy-use
2026-08-25T19:26:41.942619 - ^[[32m[INFO]^[[0m 
2026-08-25T19:26:41.945200 - ^[[32m[INFO]^[[0m Context impl SQLiteImpl.
2026-08-25T19:26:41.945272 - ^[[32m[INFO]^[[0m Will assume non-transactional DDL.
2026-08-25T19:26:42.016865 - ^[[32m[INFO]^[[0m Using RAM pressure cache.
2026-08-25T19:26:42.017375 - ^[[32m[INFO]^[[0m Starting server

2026-08-25T19:26:42.017733 - ^[[32m[INFO]^[[0m To see the GUI go to: http://127.0.0.1:8188
2026-08-25T19:26:42.055543 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 遠端主機已強制關閉一個現存的連線。
2026-08-25T19:26:43.022227 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/buttonGroup.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-08-25T19:26:43.022859 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-08-25T19:26:43.025322 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/clipspace.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-08-25T19:26:43.026445 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/widgetInputs.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-08-25T19:26:43.327636 - Error. No styles.csv found. Put your styles.csv in C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_hus_utils\styles.csv. Then press "Refresh".
            2026-08-25T19:26:43.327671 - 
2026-08-25T19:26:43.327820 - Error. No styles.csv found. Put your styles.csv in C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_hus_utils\styles.csv. Then press "Refresh".
            2026-08-25T19:26:43.327836 - 
2026-08-25T19:26:43.740258 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/button.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-08-25T19:26:44.120432 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] The ComfyRegistry cache update is still in progress, so an outdated cache is being used.
2026-08-25T19:26:44.223771 - FETCH DATA from: C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfyui_manager\custom-node-list.json2026-08-25T19:26:44.236587 -  [DONE]2026-08-25T19:26:44.236626 - 
2026-08-25T19:26:44.430570 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 遠端主機已強制關閉一個現存的連線。
2026-08-25T19:26:57.040093 - ^[[32m[INFO]^[[0m got prompt
2026-08-25T19:26:57.102480 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
2026-08-25T19:26:57.272337 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
2026-08-25T19:26:57.334510 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-08-25T19:26:57.335231 - ^[[32m[INFO]^[[0m Using MixedPrecisionOps for text encoder
2026-08-25T19:26:58.025278 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-08-25T19:26:58.039818 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-08-25T19:26:58.039941 - ^[[32m[INFO]^[[0m Detected mixed precision quantization
2026-08-25T19:26:58.040234 - ^[[32m[INFO]^[[0m Using mixed precision operations
2026-08-25T19:26:58.040318 - ^[[32m[INFO]^[[0m Native ops: float8_e4m3fn, asym_w4a8_int8, convrot_w4a4, int8_tensorwise, float8_e5m2 , emulated ops: nvfp4, mxfp8
2026-08-25T19:26:58.058536 - ^[[32m[INFO]^[[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-08-25T19:26:58.060934 - ^[[32m[INFO]^[[0m model_type FLOW_AV
2026-08-25T19:26:58.448821 - ^[[32m[INFO]^[[0m Using sage attention mode: auto
2026-08-25T19:26:58.502398 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-08-25T19:26:58.612223 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-25T19:26:59.531508 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-25T19:26:59.796183 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-08-25T19:27:00.010253 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-08-25T19:27:11.741496 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer
2026-08-25T19:27:11.750100 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\nodes_prompt_relay_advanced.py", line 342, in execute
    return io.NodeOutput(*build_prompt_relay_conditioning(**kwargs))
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 1127, in build_prompt_relay_conditioning
    binding = build_prompt_relay_binding(
        clip,
    ...<3 lines>...
        details["tokens"],
    )
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 503, in build_prompt_relay_binding
    prompt_ids, hf = _prompt_token_ids(clip, conditioned_prompt)
                     ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 433, in _prompt_token_ids
    inner, hf = _inner_tokenizer(clip)
                ~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Users\jackh\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-t8\prompt_relay_advanced.py", line 428, in _inner_tokenizer
    raise RuntimeError("Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer")
RuntimeError: Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer

2026-08-25T19:27:11.753994 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 14.71 seconds^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)


> **Comment by T8mars (2026-08-27T16:08:02Z):**

感谢反馈，已在 **v1.52.3** 修复并发布：d2f89b3。

原因不是 Prompt Relay 的时间控制算法本身，而是部分 CLIP 包装/代理对象仍能正常提供 ComfyUI 公共 `tokenize()`，却隐藏了内部的 `tokenizer.qwen3vl_32b.tokenizer` 路径，旧版因此在采样前直接拒绝。

本次没有取消安全校验：当内部路径不可见时，节点会把连接 CLIP 的实际 token ID 与 ComfyUI 原生 MiniMax H3 tokenizer **逐项比较**，完全一致才继续；任何长度或 ID 不一致仍会拒绝，避免动作时间段错位。原生直连路径保持不变，旧工作流结构和参数顺序也没有调整。

请更新到 v1.52.3 后：
1. 完全退出并重启 ComfyUI；
2. 重新载入原工作流直接复测；
3. 建议仍使用 ComfyUI 原生 `Load CLIP`，`type=minimax`。

本地已通过 Prompt Relay 全范围 87 项、注册/前端兼容 48 项及项目全量 **1680 项**测试。若更新后仍失败，请提供新的完整报错，以及 Load CLIP 节点截图和所选模型文件名；工作流太大时无需整份上传，保留 Load CLIP → Prompt Relay 最小链路即可。


> **Comment by T8mars (2026-09-04T09:28:47Z):**

已在 v1.52.3 修复并发布，后续没有收到新的复现记录。现在关闭本 Issue；如果更新后仍遇到问题，请重新打开或新建 Issue，并附完整报错、Load CLIP 节点截图和实际模型文件名。



### #12 [open] ISSUE: GPU/加速器不受支持（可用：CUDA，要求：GPU :: NVIDIA CUDA）
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/12  by thluozw 2026-08-23T10:31:34Z

## System Info
操作系统: win32
Python 版本: 3.12.11 (main, Aug 18 2025, 19:17:54) [MSC v.1944 64 bit (AMD64)]
嵌入式 Python: false
PyTorch 版本: 2.13.0+cu130
启动参数: ComfyUI\main.py --feature-flag show_signin_button=true --feature-flag enable_telemetry=true --base-directory F:\Documents\ComfyUI --user-directory F:\Documents\ComfyUI\user --database-url sqlite:///F:\Documents\ComfyUI\user\comfyui.db --port 8001 --enable-manager --extra-model-paths-config C:\Users\thluo\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1782268083402.yaml --input-directory F:\Documents\ComfyUI\input --output-directory F:\Documents\ComfyUI\output
总内存: 23.26 GB
可用内存: 5.53 GB
模板版本: 0.11.44

## Devices
- cuda:0 NVIDIA GeForce RTX 4060 : cudaMallocAsync (cuda)
  VRAM Total: 8 GB
  VRAM Free: 6.93 GB
  Torch VRAM Total: 0 B
  Torch VRAM Free: 0 B

> **Comment by T8mars (2026-08-24T15:16:24Z):**

感谢反馈。这个提示并不是显卡没有 CUDA，也不是本节点运行时判断失败，而是当前 ComfyUI 前端/Manager 的加速器名称归一化问题：本项目按照 Comfy Registry 官方格式声明 `Environment :: GPU :: NVIDIA CUDA`，Registry 返回 `GPU :: NVIDIA CUDA`，但前端把本机 CUDA 简写成 `CUDA` 后进行了不相等比较。

ComfyUI 上游已有完全相同的问题：
https://github.com/Comfy-Org/ComfyUI_frontend/issues/15364

因此暂时不删除本项目正确的 Registry 元数据。本 Issue 保持开放，用于跟踪上游修复。需要注意，RTX 4060 8GB 是否能够运行具体 H3 工作流是另一项显存条件，与这个安装兼容性误报不是同一问题。


> **Comment by T8mars (2026-09-04T09:28:47Z):**

状态更新（2026-09-04）：ComfyUI 前端上游问题 https://github.com/Comfy-Org/ComfyUI_frontend/issues/15364 目前仍为 Open。本项目的 Registry CUDA 元数据保持官方格式不变，本 Issue 继续保持开放并跟踪上游；这仍是前端名称归一化误报，不代表本机 CUDA 运行时不可用。


> **Comment by T8mars (2026-09-14T12:17:12Z):**

进度更新：截至 2026-09-14，上游 ComfyUI_frontend #15364 仍处于 Open 状态：

https://github.com/Comfy-Org/ComfyUI_frontend/issues/15364

问题仍是 Registry 返回长分类值 `GPU :: NVIDIA CUDA`，而 Manager 前端按短值 `CUDA` 做精确比较；这是前端归一化缺失，并非本节点真的不支持 CUDA。我们不会把本项目的合法 Registry classifier 改成不符合规范的值来规避提示。

本 Issue 继续作为上游跟踪项保留。另请注意，Registry 新版本停在 1.47.0 是独立的安全扫描审核问题，已在本仓库 #19 和 Registry 后端 #233 跟踪。


> **Comment by T8mars (2026-11-17T05:42:05Z):**

2026-11-17重新核对：上游[ComfyUI_frontend#15364](https://github.com/Comfy-Org/ComfyUI_frontend/issues/15364)及已有修复PR[#15468](https://github.com/Comfy-Org/ComfyUI_frontend/pull/15468)均仍Open，尚未合并。近期其它用户也报告分类长值与CUDA短值比较误报。

本项目保留合法Registry classifier；不会为了隐藏提示改成错误元数据或修改用户安装的前端资产。此项继续跟踪上游，不能写成本节点已修复该前端提示。



### #11 [closed] ISSUE: 导入失败
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/11  by gangdeq 2026-08-18T11:12:13Z

[INFO]    0.3 seconds (IMPORT FAILED): /root/ComfyUI/custom_nodes/comfyui-minimax-h3-audio-T8

[WARNING] Error while calling comfy_entrypoint in /root/ComfyUI/custom_nodes/comfyui-minimax-h3-audio-T8: '/.xgcos/.links/c9718bfd-08ce-53c7-8023-dc98a20a626b' is not in the subpath of '/root/ComfyUI/models'

> **Comment by T8mars (2026-08-24T15:16:32Z):**

这个问题与 #10 是同一根因：模型目录中的别名解析到外部公共存储后，旧版本错误地把它判断成越界路径。当前 main 已支持目录软链接和单文件软链接，并保留路径穿越防护。

修复提交：
https://github.com/T8mars/comfyui-minimax-h3-audio-T8/commit/620d87d

请更新本节点到当前 main 并完整重启 ComfyUI。若更新后仍有错误，请新开 Issue 并附上新的完整 traceback。这里作为已修复的重复问题关闭。



### #10 [closed] ISSUE: [BUG] comfyui-minimax-h3-audio-T8 fails to load due to hardcoded path to public model storage not under folder_paths.models_dir
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/10  by losday2060-debug 2026-08-18T09:15:18Z

When ComfyUI starts, the node `comfyui-minimax-h3-audio-T8` fails to import with the following error:

[WARNING] Error while calling comfy_entrypoint in /root/comfyui/ComfyUI/custom_nodes/comfyui-minimax-h3-audio-T8: '/datasets/ComfyUI/models/yolo/person_yolov8s-seg.pt' is not in the subpath of '/home/waas/ComfyUI/models'

The plugin attempts to access a hardcoded absolute path `/datasets/ComfyUI/models/yolo/person_yolov8s-seg.pt`. In my environment:
- `/datasets/` is a shared public storage provided by the cloud service, readable by all users.
- My ComfyUI `folder_paths.models_dir` is configured as `/home/waas/ComfyUI/models` (my private space).
- The actual model file exists in the public storage at `/datasets/ComfyUI/models/yolo/person_yolov8s-seg.pt`.
- I have created a symbolic link at `/root/comfyui/ComfyUI/models/yolo/person_yolov8s-seg.pt` pointing to the public file, so ComfyUI can access it when resolved.
- However, ComfyUI's security check (used by `_resolve_detector_path`) compares the literal path string against `folder_paths.models_dir`. Since `/datasets/...` is not a subpath of `/home/waas/ComfyUI/models`, the check fails, even though the file is reachable through the symlink.
This prevents the node from loading.



.

> **Comment by T8mars (2026-08-24T15:16:29Z):**

该问题已修复。当前实现保留 `ComfyUI/models` 下的词法别名，同时允许该别名通过目录软链接、文件软链接或挂载点解析到外部公共模型存储；仍会拒绝绝对路径和 `..` 路径穿越。

修复提交：
https://github.com/T8mars/comfyui-minimax-h3-audio-T8/commit/620d87d

后续版本还增加了与报告场景一致的 `yolo/person_yolov8s-seg.pt` 文件软链接回归测试。请更新到当前 main 后重启 ComfyUI。如仍失败，请新开 Issue 并附上更新后的完整 traceback。这里按已修复关闭。



### #9 [closed] PR: Add dialogue-safe audio workflows
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/9  by T8mars 2026-08-10T03:59:14Z

## What changed
- Append three EXP nodes for dialogue boundary analysis, dialogue-safe master assembly, and timed background-bed latent locking.
- Preserve all existing 51 node IDs, ordering, defaults, and legacy workflows.
- Add API/UI example workflows, README guidance, feature metadata, and verification notes.
- Document scientific limits: this mitigates unwanted extra speech without truncating the full soundtrack, but does not claim source separation, sample-exact speech stopping, or seamless latent locking.

## Verification
- Full suite before release: 235 passed, 4 warnings.
- Final targeted suite in the ComfyUI embedded runtime: 37 passed, 4 warnings.
- Ruff checks passed.
- Python compile, JSON parsing, diff checks, isolated ComfyUI import/object-info, real H3 four-step latent-lock probe, and real Audio VAE boundary probe passed.

## Scope
- `SKILL.md`, `roadmap.md`, and local artifacts are intentionally excluded.


### #8 [closed] PR: Add experimental H3 source AV repaint nodes
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/8  by T8mars 2026-08-10T01:59:23Z

## What changed

- add three isolated experimental Source AV nodes for 24fps media windowing, strict H3 video/audio latent assembly, per-stream lock/remix/regenerate masks, and VAE-free AV separation
- append the new nodes after the existing 48 registrations so prior node IDs, ordering, defaults, and workflows remain unchanged
- add API and frontend example workflows
- update project metadata and README to version 1.11.0

## Validation

- 222 tests passed
- Ruff, compileall, JSON validation, and `git diff --check` passed
- isolated ComfyUI registration and workflow conversion passed
- one real H3 mechanical end-to-end probe produced a 124-frame 24fps H.264 video with 32kHz stereo AAC

## Limits

The real probe reached only 44.52 MiB minimum free VRAM on a 16 GiB GPU, below the 512 MiB safety gate. The nodes remain Experimental and do not claim `memory_safe`, calibrated denoise strength, arbitrary-video support, or perceptual quality validation.


### #7 [closed] PR: Release 1.10.0 speech reliability and long-form workflows
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/7  by T8mars 2026-08-09T20:24:33Z

## What changed

- adds guarded cleanup for speech failures before Finalize and a non-claiming VRAM preflight
- adds durable long-form accepted state, resume/cancel/compose, chunk previews, exact-sample ADR, explicit voice-library lifecycle, and isolated experimental joint dialogue
- adds multilingual, identity, performance, VRAM, and long-form validation tools
- adds API/frontend example workflows and updates the 1.10.0 documentation and validation report

## Why

The experimental H3 speech path needed reliable failure cleanup, resumable long-form production, creator-facing workflows, and evidence-based release gates without claiming 16GiB memory safety, high-fidelity cloning, calibrated performance control, or stable joint dialogue before those gates pass.

## Compatibility

The original 36 node IDs and order remain unchanged; the 12 new speech reliability/creator nodes are appended. Existing stable sampling, conditioning, and old workflows keep their prior defaults.

## Validation

- 204 pytest tests passed
- Ruff passed
- compileall passed
- all example JSON parsed
- git diff --check passed
- isolated ComfyUI registration confirmed 48 nodes and 12 speech frontend workflows


### #6 [closed] PR: fix: preserve H3 task labels and legacy protocol
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/6  by gluttony-10 2026-08-09T14:57:02Z

## Summary

- preserve H3 task labels in the frontend
- keep the legacy protocol mapping compatible with existing workflows
- add regression coverage for the sampling behavior

## Why

Task labels and legacy protocol values can be lost during frontend normalization, which changes the behavior of existing saved workflows. This keeps the display labels and serialized protocol stable.

## Validation

- 71 tests passed
- frontend JavaScript syntax check passed


> **Comment by T8mars (2026-08-24T15:16:36Z):**

感谢贡献。这个 PR 创建后项目已经经历了多轮兼容更新：任务类型中文显示、旧工作流 canonical 值保留，以及新旧 H3 `audio_scale` 协议识别都已在当前 main 中用更完整的回归测试实现。

当前 PR 已与 main 冲突（DIRTY），其中旧节点列表也只覆盖早期的一小部分节点；直接合并会把已经扩展的任务类型/长视频支持重新带回旧结构。PR 中尚未进入 main 的 RunningHub `ensureNodesRegistered` 逻辑属于独立功能，不应再和旧协议测试混在一起。

因此这里关闭该 PR，不直接合并。如果仍需要 RunningHub 预注册支持，欢迎基于最新 main 提交一个只包含该功能的新 PR，并动态覆盖当前注册节点、附带对应前端兼容测试。



### #5 [closed] PR: Release 1.9.0: long video, speech, and visual reference EXP
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/5  by T8mars 2026-08-09T10:02:03Z

## What changed

- Adds the isolated resumable long-video orchestration, accepted-state, background queue, release-policy, recovery, and workflow stack developed on this branch.
- Adds experimental native H3 speech nodes for described voice, authorized reference voice, ASR/speaker reporting, per-turn dialogue assembly, and explicit release policies.
- Adds the isolated `MiniMaxH3VisualReferenceStrengthEXPT8` post-conditioning node with controlled Ref2VA A/B examples and warnings.
- Publishes the complete API/frontend workflow set and updates the project to version 1.9.0 with 36 registered nodes.

## Compatibility and user impact

- Existing node IDs, the original 35-node order, stable conditioning, stable dual-clock sampling, multi-rate sampling, and still-image paths remain unchanged.
- The visual-reference control is opt-in and only writes `minimax_visual_cond_noise_aug` to copied conditioning metadata.
- Speech and visual-reference features remain explicitly Experimental; no model weights are included.
- `SKILL.md`, `roadmap.md`, local model files, generated media, and validation artifacts are excluded from Git.

## Validation

- 188 project tests passed.
- Final targeted registration/visual-reference suite: 39 passed.
- Ruff passed and staged diff/JSON checks passed.
- Full Ref2VA INT8 matrix completed for no-node, 0.999, 0.995, 0.990, 0.980, and 0.950.
- No-node and explicit 0.999 decoded video/audio were exactly equal; repeated 0.950 was also exactly repeatable under the fixed local controls.
- Stable `sampling.py`, `conditioning.py`, `sampling_multirate_exp.py`, and `still_image.py` remained byte-for-byte unchanged.

## Scientific limits

The Ref2VA control is not presented as an oiliness fix. The single controlled case showed material pose/composition changes at lower strengths and did not establish a monotonic quality improvement. Observed 16GiB headroom remained below the project's 512MiB safety gate, so no memory-safe claim is made.


### #4 [closed] ISSUE: How do I make the person in the reference image move their lips in accordance with the input audio?
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/4  by gastonbarreto2-ai 2026-08-08T05:40:37Z

I've put the audio in drive_audio, but the character's mouth moves while saying something else. I want it to move in time with the audio input.

> **Comment by T8mars (2026-08-24T15:17:03Z):**

Thanks for the report. `drive_audio` is a joint MiniMax H3 generation condition; it is not a deterministic phoneme-to-mouth-motion constraint. H3 may use the audio semantically while generating different speech or imperfect mouth timing.

For the closest supported workflow:

1. Use `audio_mode=lock_source`.
2. Enable `add_source_as_reference`.
3. Refer to the actual `<Audio N>` in the prompt.
4. If the transcript is known, include the exact words inside `<d>...</d>`.
5. Save the node's `mux_audio` output, not the AV decoder's generated audio.

This preserves the supplied final soundtrack and strengthens the speech condition, but it still cannot guarantee phoneme-accurate lip sync. Production-accurate sync requires a dedicated lip-sync post-process. The current README now documents this boundary, so this is being closed as a model capability/usage question rather than a plugin bug.



### #3 [closed] ISSUE: sageattention is not new enough version or could not determine CUDA architecture
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/3  by ahkimkoo 2026-08-07T03:33:45Z

```
[INFO] model_type FLOW
[ERROR] !!! Exception during processing !!! sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.
[ERROR] Traceback (most recent call last):
  File "/home/jovyan/work/python/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/work/python/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/work/python/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/home/jovyan/work/python/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "/home/jovyan/work/python/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/work/python/ComfyUI/comfy_api/latest/_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/work/python/ComfyUI/custom_nodes/comfyui-kjnodes/nodes/ltxv_nodes.py", line 2149, in execute
    raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.")
RuntimeError: sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.
```

MiniMax H3 Mem Eff Sage Attention Patch节点报错。sageattention是最新版。

> **Comment by aotou (2026-08-08T08:51:30Z):**

安装最新的 SageAttention Windows wheel 文件（支持 cu130 + torch 2.10+，也兼容 cu132）：
C 
:\comfyUI\venv\Scripts\python.exe -m pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl


> **Comment by ykxxxxx (2026-08-13T05:39:43Z):**

一样，等T8出来解决



> **Comment by guojuxia-op (2026-08-13T14:09:41Z):**

> 安装最新的 SageAttention Windows wheel 文件（支持 cu130 + torch 2.10+，也兼容 cu132）： C :\comfyUI\venv\Scripts\python.exe -m pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl

我安装了，还是不行哦


> **Comment by T8mars (2026-08-24T15:16:59Z):**

这段 traceback 的抛错位置在 `comfyui-kjnodes/nodes/ltxv_nodes.py`，不是本项目节点代码。根因通常是 SageAttention wheel 与当前 Torch/CUDA/Python ABI 不匹配、`sageattention.core` 缺少 KJNodes 所需符号，或 wheel 没有当前 GPU 的 sm 架构。

当前本项目已经提供只读环境审计，可分别报告 SageAttention 包导入、core 导入、KJ 所需符号、wheel 支持的 sm 架构与当前 GPU 架构，不再把所有原因混成一个错误。请先使用当前版本的 Environment Audit 检查，再根据实际 Torch/CUDA/Python 组合安装对应 wheel；不要仅依据“最新版”判断兼容。

由于故障来自外部 KJNodes/SageAttention 安装组合，本项目无法在节点内部安全修复或替换二进制 wheel，这里作为外部依赖问题关闭。如审计确认是本项目误报，请新开 Issue 并附上完整审计 JSON。



### #2 [closed] ISSUE: AttributeError: 'MiniMaxH3FlowSampling' object has no attribute 'audio_scale'
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/2  by glolionline 2026-08-07T03:13:00Z

`[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
[ERROR] !!! Exception during processing !!! 'MiniMaxH3FlowSampling' object has no attribute 'audio_scale'
[ERROR] Traceback (most recent call last):
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 2172, in extra_conds
    payload["audio_scale"] = self.audio_scale()
                             ^^^^^^^^^^^^^^^^^^
  File "C:\Users\kevin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 2117, in audio_scale
    return self.model_sampling.audio_scale
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1964, in __getattr__
    raise AttributeError(
AttributeError: 'MiniMaxH3FlowSampling' object has no attribute 'audio_scale'
[INFO] Prompt executed in 7.13 seconds

`

Comfyui更新到最近commit(a464ac3)后无法运行了，应该是这个[PR #15243](https://github.com/Comfy-Org/ComfyUI/pull/15243)相关

> **Comment by heisenbergxh-bot (2026-08-13T15:25:07Z):**

同样的问题


> **Comment by T8mars (2026-08-24T15:16:26Z):**

该问题已在 1.3.1 的兼容更新中修复，并持续保留在当前 main：
https://github.com/T8mars/comfyui-minimax-h3-audio-T8/commit/1c8040b

当前实现会区分新版 ComfyUI FLOW_AV 协议与旧版 H3 音频速度协议；T8 自定义双时钟采样使用自己的中性 `audio_scale=1.0`，新版原生采样则使用 ComfyUI 当前协议。相关回归测试也已经覆盖。

请更新到当前 main 后重启 ComfyUI。如仍能复现，请新开 Issue 并附上当前 ComfyUI commit、本节点版本及完整日志。这里先按已修复关闭。



### #1 [closed] PR: Publish MiniMax H3 Audio T8 nodes
URL: https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/1  by T8mars 2026-08-05T15:34:53Z

## What changed

- Add a standalone ComfyUI MiniMax H3 audio node package with nine registered nodes.
- Support native, reference-only, locked-source, and remix audio modes across T2VA/I2VA/FL2VA/L2VA/Ref2VA/Hybrid conditioning.
- Add official media-tag normalization, dynamic audio ordinals, duration/window planning, preflight validation, AV decode, audio mix, and synchronized output trim.
- Add documentation, structured metadata, an API workflow example, and automated tests.

## Why

The previous implementation mixed reference and target-audio responsibilities, could misnumber audio references when video soundtracks were present, and did not provide a complete timing/validation/output pipeline. This package follows the current native ComfyUI MiniMax H3 contracts and remains independent from the legacy node pack.

## Impact

Users get a self-contained `T8/MiniMax H3/Audio` category with guarded H3 conditioning and end-to-end audio-driven workflow helpers. Existing ComfyUI or legacy custom-node files are not changed.

## Validation

- `ruff check .`
- `pytest -q -p no:cacheprovider` — 27 passed
- ComfyUI `load_custom_node` — 9 nodes registered successfully

