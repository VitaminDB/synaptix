# T8mars__comfyui-minimax-h3-prompt-enhancer-T8

Source: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8
Fetched: 2026-11-28

Description: null
Stars: 427  Updated: 2026-11-21T18:54:36Z



---

## FILE: README.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/README.md)

<p align="right">
  <strong>简体中文</strong> | <a href="./README_EN.md">English</a>
</p>

1.42.2 修复部分容器机器的云端代理兼容，见 [RunningHub 与代理配置](docs/cloud-network.md)。
明确配置的后端代理与 `NO_PROXY` 会参与云端线路选择；无代理机器保留直连方式。

## 新增：MiniMax H3 Image 提示词扩写（1.41.0）

1.41.1 修复外接文本中的 URL/JSON/大括号被改写、外接比例断线后的默认值，以及复制整张工作流后的恢复结果串用。

搜索 **MiniMax H3 Image Prompt Enhancer T8**。独立支持文生图和最多16张有序参考图的图像编辑，
使用官方 [H3 Image PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) 两套固定版本提示词。
沿用现有API、本地GGUF和Strata；支持画幅比例、软目标英文词数（0=官方自动，无1000上限）、
逐字锁定画面文案、最多一次格式纠正与成功结果恢复。英文描述保留中文等原文。
`expanded_prompt` 接下游H3正向，编辑原图仍需按相同顺序接下游。
[使用说明](docs/h3-image.md) · [文生图示例](example_workflows/h3_image_t2i_example.json) · [双图编辑示例](example_workflows/h3_image_edit_example.json)。
结构验证不代表生成图质量或4096 Token精确核验；未捆绑或下载模型。

## 更新：H3 普通开启落实色彩与导演拍法（1.40.0）

H3 的普通 **开启 / On** 会要求把色彩与导演拍法写到具体画面中。已有可见表面的颜色未指定或待定时，会要求从所选色卡补全；明确锁定的原色、参考色与材质仍保留。旧 **开启（H3强化试验）** 保留兼容，在本版与普通 On 使用相同提示。两项配置默认关闭；如需停用，关闭对应配置并重新生成。

搜索 `T8 Color Style Config` / `T8 Creator Style Config`，分别连接 H3 或 Seedance 的色彩/导演可选输入。两个入口独立组合，带本地分类搜索与预览；不接 MOSS、不改原图、不限制时长。点击“浏览风格与帮助”可看200种配色的同场景对照，以及200位创作者的400条构图/机位/节奏示意；动态方法可手动播放。导演条目是有来源依据的拍法参考，不是完整编剧器；弱来源有明确标记，示意图不是模型输出。[使用说明](docs/visual-styles.md) · [颜色→H3](example_workflows/visual_style_color_h3.json) · [导演→Seedance](example_workflows/visual_style_creator_seedance20.json) · [组合→H3](example_workflows/visual_style_combined_h3.json)

本版沿用旧试验模式的执行提示，并增加开放颜色补全说明；Off 与 Seedance 的生成文本不变，没有新增字段或独立风格请求。当前验证的是提示词中的配色与具体拍法，预览仍是教学示意。[验证进度、模型参数与历史限制](docs/visual-styles.md#style-validation)集中记录在使用说明中。

## 新增：H3 空间标记增强（实验）

新增 `T8H3SpatialMarkerCanvas` / “手动空间画框”：直接上传图片、拖人物/目标框、填写描述与动作，应用后保存，**无需手写JSON或安装audio插件**。[手动画框说明](docs/h3-visual-marker.md) · [独立Ref2VA](example_workflows/h3_spatial_canvas_ref2va.json) / [独立Hybrid](example_workflows/h3_spatial_canvas_hybrid.json)。原有空间配置与旧示例仍兼容。

搜索 `T8H3VisualMarkerConfig` / `Visual Marker Config`。只给 H3 添加可选入口，默认关闭；读取现有 Marker Prepare 的计划和同一标记参考图，完善角色—目标—动作描述。两个参考项目无需改动。普通 Ref2VA / Hybrid 额外参考图，不新增时长限制或独立 LLM 调用；不保证硬轨迹、去框或成片收益。[接线与边界](docs/h3-visual-marker.md) · [Ref2VA 示例](example_workflows/h3_visual_marker_ref2va.json) · [Hybrid 示例](example_workflows/h3_visual_marker_hybrid.json)

## 新增：独立分镜强化（实验）

`T8 Storyboard Director Config` 独立连接 H3 / Seedance，默认关闭；与既有导演、情感演绎协同，**不限制时长、不截断、不自动拆片**，不增加独立规划请求。未连接/关闭保持旧行为，不给 MOSS 添加此功能。效果尚待 A/B 验证：[使用与验收说明](docs/storyboard-director.md) · [H3 对照](example_workflows/storyboard_director_h3_comparison.json) / [Seedance 对照](example_workflows/storyboard_director_seedance20_comparison.json)。

## 新增：MOSS Image 提示词增强

独立 **MOSS Image Prompt Enhancer T8** 节点，生成 MOSS 原生 Caption 或 Layout；自动格式默认开启。Layout 使用官方固定版本编译器转换像素坐标，不让模型猜最终坐标。支持固定宽高、锁定画面文案、可选参考图（批次展开合计最多 10 张），沿用多渠道 API / 本地 GGUF / 已启动 Strata。参考图仅供增强语言模型理解，**不是 MOSS 原生图像编辑**。

“仅校验编译”无需 API 或模型。最多追加一次纠正；合法首稿不会因可选纠正失败而丢失，硬校验失败则停止下游并可查看失败草稿。4096 目标 Token 未自动测量，不能以字符数保证不截断。[用法、边界与示例](docs/moss-image.md)。原有节点及旧工作流不改。

主节点现在直接显示“最大描述字数”，并可选 `Classic / 原版`（默认）、`Fisher / 通用长版`、`Comic / 漫画卡通`。旧 27 字段位置和六输出不变，新模式只追加在末尾；旧工作流保持 Classic。字数是描述正文的软字符上限，不是 Token，不会截断草稿。Fisher 自动中文长版以约 1500–2500 汉字为软目标，明确短版/上限优先；主 Comic 仍是一张图、一种描述语言。

多页或多参考图请搜索 **MOSS Authoring Director T8**，生成时必接 **T8 LLM Provider Config**。Comic 最多 16 页，先规划再逐页产出共几何的英文/中文稿；Fisher 默认逐参考图独立成稿，只有显式选择 `fusion` 才融合一张。批次展开合计最多 10 张参考图。Director 的 `bundle_json` 接 **MOSS Authoring Page T8**，选页码和语言，再把 Page 的输出 0 正向、1 宽、2 高接下游；整册、报告、源稿绝不直接接正向。[完整接线、修订与调用费用](docs/moss-image.md#整册与多参考图director--page)。这些是独立改编的创作规则，不是 H3/Seedance 分镜配置，也不证明实际成图或翻译质量。

入门示例：[零模型 CPU 选页](example_workflows/moss_authoring_cpu_page.json) · [Comic 两页双语](example_workflows/moss_authoring_comic_two_pages.json) · [两图独立](example_workflows/moss_authoring_fisher_independent.json) / [显式融合](example_workflows/moss_authoring_fisher_fusion.json) · [固定布局修订第二页](example_workflows/moss_authoring_fixed_layout_revision.json)。两图示例需先将 `example_workflows` 中的 `moss_authoring_fisher_independent.jpg`、`moss_authoring_fisher_fusion.jpg` 分别上传至两个 LoadImage；不会自动写入 ComfyUI input。CPU 示例内置手工教学稿，不调用模型，也不是成图效果证明。

画布支持 **8 的倍数**（16–16384），包括 1256×1672；生成、Layout 编译、保存重载和恢复都保持原宽高，不要求改成 16 倍数。下游 latent 可能自行对齐，精确排版请核对最终画布。

## 新增：Strata 本地服务桥接

API 模式选 **Strata 本地服务（桥接）**，使用已经启动的 Strata 服务当前模型；默认连接 ComfyUI 后端本机 `http://127.0.0.1:8080`。无需云端 API Key 或另装 Strata ComfyUI 插件。可选接入 **T8 LLM Provider Config → Strata** 设置地址、模型、思考和采样参数，连接后共享配置优先。

支持文本、图片和视频视觉样本，不读取音频。沿用原提示词格式、质量检查和输出，原 API / GGUF 工作流不变。桥接不会启动、停止、切换或卸载 Strata 模型。[使用说明、限制与排错](docs/strata-bridge.md)

## 入口导航

| 入口 | 适合用户 | 说明 | 打开 |
| --- | --- | --- | --- |
| 贞贞的平价AI小铺（国内版） | 国内用户、国内模型优先 | 主要调用国内模型，适合国内模型工作流。 | <a href="https://api.seedance.nz/sign-up?aff=5f4w"><kbd>进入国内版平价AI小铺</kbd></a> |
| 贞贞的AI工坊（海外版） | 海外用户、海外模型优先 | 主要调用海外模型，也包含部分国内模型。 | <a href="https://ai.t8star.org/register?aff=dP7j"><kbd>进入海外版AI工坊</kbd></a> |
| RunningHub APIKEY（国内版） | 需要适配更多 AI 应用的国内用户 | 适配更多 AI 应用，并可体验最新模型。 | <a href="https://www.runninghub.cn/user-center/1819214514410942465/webapp?inviteCode=rh-v1121"><kbd>获取国内版 APIKEY</kbd></a> |
| RunningHub APIKEY（海外版） | 海外模型、更宽松审核场景 | 审核更宽松，支持海外模型。 | <a href="https://www.runninghub.ai/user-center/1907375370302308353/webapp?inviteCode=rh-v1121"><kbd>获取海外版 APIKEY</kbd></a> |

# ComfyUI MiniMax-H3 / Seedance 2.0 / Music 3 / YuE2 Prompt Enhancer T8

## 新增：可选语音／字幕参考

新增默认关闭的 **T8 语音参考**：已有文字、SRT/VTT/JSON 离线复用，或通过已启动 Strata 的 CPU 语音组件转写 AUDIO／视频音轨。H3、Seedance 增加可选入口；默认只理解内容，明确选“保留对白”并指派角色才锁定原词。原 API、GGUF、视觉 Strata 及旧工作流保留，不改服务设备或模型生命周期。

[接线、角色映射、资源与费用说明](docs/speech-reference.md) · [H3 字幕示例](example_workflows/speech_reference_h3.json) · [Seedance 字幕示例](example_workflows/speech_reference_seedance20.json)。这不是音乐／环境音理解，也不代表已完成成片口型验收。

## 新增：H3 Hybrid 关键帧＋参考混合

在原 H3 节点的「生成类型」选择 **Hybrid**，连接首帧和/或尾帧，再连接至少一个额外参考图或视频。首尾锁定目标边界，额外素材提供外观、身份或指定运动参考；采用 H3 六段参考格式，支持原有四种渠道、本地 GGUF 与 Prompt Relay。旧五种模式、默认 T2VA、38 个保存字段和旧工作流不变。

每槽单图，最多 9 张额外参考图＋2 张关键帧、3 个视频；节点不限制视频时长，不分析音轨。需要音频混合请用 H3 音频插件。结构检查不等于像素/成片验收。[使用说明与边界](docs/h3-hybrid.md) · 示例：[首帧＋图](example_workflows/h3_hybrid_first_image.json) / [尾帧＋图](example_workflows/h3_hybrid_last_image.json) / [首尾＋视频](example_workflows/h3_hybrid_first_last_video.json)。

## 新增：可选质量流程与因果动作优化

**可选战斗运镜配置**：新增 `T8 战斗运镜配置 / Combat Camera Config`，连接 H3 或 Seedance 左侧的可选输入。
支持 AUTO/强化、连续镜头偏好、写实/风格化，可叠加表演导演；只优化已有动作的拍法，不加敌人、招式或胜负。
未连接/关闭保持原行为，不新增规划请求。[使用说明与验收边界](docs/combat-camera.md) ·
[H3 示例](example_workflows/combat_camera_h3.json) / [Seedance 示例](example_workflows/combat_camera_seedance20.json)。

原 H3 / Seedance 节点新增 **质量检查、最多一次质量纠正、因果动作优化**。默认 Off / Original，旧工作流与有效原生输出保持不变；H3 已确认的说话人格式错误会自动免费校正，不需新增表格或连线。检查失败和纠正次数会在节点说明卡与脱敏诊断中显示；追加模型纠正可能计费，失败保留完整稿。复杂语义、物理可达性及成片仍需人工验收，不把结构分当创作质量。

[简明中英文说明与费用边界](docs/h3-quality-creation.md) · [只检查示例](example_workflows/h3_seedance_quality_check_example.json) · [因果优化＋纠正示例](example_workflows/h3_seedance_causal_creation_example.json)

## 新增：H3 / Seedance 定向创作 Skill

**兔子-舞蹈 / Dance**：在两个视频增强节点的「定向创作 Skill」里选择。细化左右手位、支撑脚、重心、卡点和续接姿势；不新增时长限制、不分析音轨。默认关闭，旧案例与工作流不变。[用法与续接示例](docs/directional-skills.md#兔子-舞蹈--dance) · 工作流：[H3](example_workflows/directional_rabbit_dance_h3.json) / [Seedance](example_workflows/directional_rabbit_dance_seedance20.json)。

新增 **Jojocodex-武术打斗 / Wushu combat**：让身体发力、避线/接触、支撑变化与下一动作衔接，跨镜继承未完动作。两个增强节点现有下拉项直接选择，沿用云端/本地渠道与原生格式；不强加三镜、伤害、胜负或 LoRA，不改变旧工作流，默认仍关闭。[用法与边界](docs/directional-skills.md#jojocodex-武术打斗--wushu-combat) · 独立示例：[H3](example_workflows/directional_wushu_combat_h3.json) / [Seedance](example_workflows/directional_wushu_combat_seedance20.json)。已做工程回归，尚未做该技能的真实 LLM/成片效果验收。

新增 **贞贞-POV剧情导演 / Zhenzhen POV**：把第一人称视角、观众参与、人物接收与动作结果组织清楚；不把剧情简化成贴脸、递物和微笑。沿用两节点的云端／本地渠道、语言、时长和原生格式；「贞贞」只作方法署名，不锁人物或渠道，不自动加 LoRA。选择现有定向技能即可，默认仍关闭。[用法与边界](docs/directional-skills.md#贞贞-pov剧情导演--zhenzhen-pov) · 独立示例：[H3](example_workflows/directional_zhenzhen_pov_h3.json) / [Seedance](example_workflows/directional_zhenzhen_pov_seedance20.json)。

新增 **宁版-文武双全 / Ning · Drama & Action**：文戏围绕信息与听者反应安排观看重点，武戏围绕发力、避让与受力变化选择摄影；适合对白转动作的混合场景。支持明确授权的原创创作，保留原句、素材角色、时长、镜数与尾态。选择现有定向技能下拉即可，云端/API兼容接口与本地GGUF共用此方法。[宁版双平台示例](example_workflows/directional_ning_wenwu_comparison.json) · [三组输入及说明](docs/directional-skills.md#宁版-文武双全--ning-drama--action)。默认仍关闭。

原 H3 与 Seedance 2.0 增强节点提供默认关闭的九种定向创作方法。作者署名只改变显示名称，旧名称与稳定技能ID继续兼容。只吸收非官方创作方法，保留两平台原生格式、原有渠道与输出，不需要新增连线。开启时当前请求暂停旧案例/手动模板编排及 H3 可选官方场景，保留原选择，关闭后恢复；只有「连续战斗长镜头」将镜头数 `AUTO` 按一镜处理，固定两镜以上会在付费调用前提示冲突。

两项新戏剧技能同一期加入：一项侧重人物目标、说话意图与关系，一项侧重处境、期待与回应；不强制冲突、转折、笑点或角色反应。对白默认原样保留，只有明确要求才补缺句，LOCK/禁止新增仍优先。角色圣经选填，技能不增加规划/评分调用；已有纠正/重试仍可能计费。[中英文授权、格式及使用说明](docs/directional-skills.md#戏剧场面与情境戏剧--dramatic-and-situational-scenes) · 戏剧场面：[H3](example_workflows/directional_drama_scene_h3.json) / [Seedance](example_workflows/directional_drama_scene_seedance20.json) · 情境戏剧：[H3](example_workflows/directional_situational_drama_h3.json) / [Seedance](example_workflows/directional_situational_drama_seedance20.json)。来源是[固定版本的非官方方法改编](directional_skills/SCREENWRITING-NOTICE.md)，不含书籍/剧本摘录。

[简明中英文使用说明](docs/directional-skills.md) · 云端双节点示例：[Fisher-长镜头](example_workflows/directional_continuous_combat_comparison.json) / [土豆-连续攻防](example_workflows/directional_high_density_combat_comparison.json) / [兔子-电影枪战](example_workflows/directional_cinematic_gunfight_comparison.json) / [宁版-文武双全](example_workflows/directional_ning_wenwu_comparison.json)。示例 Key 留空，完整运行会分别调用 H3 和 Seedance 两个文本增强请求；技能本身不增加规划或评分调用，已有修复/重试仍按原规则执行。

## 新增：H3 Prompt Relay 编排

原 H3 增强节点新增默认关闭的 Relay 模式：输出全局提示词、逐事件提示词、秒数范围、对齐帧数和校验报告。普通增强及第一个输出保持原用途；支持原有云端／本地渠道。无需加载视频模型即可生成文本，接入执行项目的 Relay Plan 后才启用时序路由。

[中英文使用说明与接线](web/js/docs/h3_prompt_relay.md) · [8 秒云端示例工作流](example_workflows/h3_prompt_relay_example.json)

## 新增：YuE2 音乐提示词与歌词创作

新增独立节点 **YuE2 音乐提示词与歌词创作（T8）**：支持一句话写歌、原词逐字保留、指定段落改词、换风格、可选 ABC 乐谱校验与保留旋律去和弦。输出 `style`、`lyrics`、`abc`、官方请求 JSON 和创作报告；不生成音频、不下载音乐模型，不影响原有三个核心节点。

支持现有四类云端／本地 GGUF 渠道与共享配置。默认中文歌词＋英文风格描述；高级设置折叠，并提供参考案例、使用说明与完整结果恢复按钮。

[使用说明与费用边界](web/js/docs/yue2_music.md) · [云端创作](example_workflows/yue2_cloud_creation_example.json) · [保留原词](example_workflows/yue2_preserve_lyrics_example.json) · [本地 GGUF](example_workflows/yue2_local_qwen_example.json) · [ABC 旋律](example_workflows/yue2_abc_melody_example.json)

## 新增：独立作词与编曲规划器

新增两个互相独立的文字规划节点：**T8 作词规划器 / Lyric Writer** 输出歌词与版本化 `lyric_plan_json`；**T8 编曲规划器 / Arrangement Planner** 输出段落、节拍、调性、配器和能量曲线的 `arrangement_plan_json`。它们只生成可复用的文本计划，不生成音频、ABC 或 MIDI。两节点沿用 YuE2 的四类渠道、共享配置、本地 GGUF 和原有默认模型；断开计划 socket 时，Music 3 与 YuE2 的旧工作流完全按原逻辑运行。

作词计划可接入 Music 3 / YuE2，编曲计划只投影受控音乐字段，不把外部 Skill 格式或整段自由文本直接注入 Caption。计划采用 `t8-lyric-plan/v1` 与 `t8-arrangement-plan/v1`，连接前会校验版本、大小和凭据泄漏；质量审校仍是文本审校，不代表听感或音频验收。

## 新增：Qwen Image 2.1 图像提示词增强

独立节点 **Qwen Image 2.1 Prompt Enhancer**：把文字提示词改写为图像生成提示词，也支持连接 1–10 张有序参考图进行图像编辑。默认 **经典兼容 / Classic** 保留冻结 Skill、英文描述与旧工作流。高级设置新增 **编辑规则 → 编辑专用（T8）/ Edit-aware**，仅图像编辑生效：只改指定部分，其他身份/饰品/产品/构图引用原图保留；中文需求默认中文，画内文字不自动翻译。多图使用 `<imageN>` 明确来源，自动比例可跟随目标原图的真实宽高比，固定 UI 比例优先；四个输出端口及原有渠道不变。节点显示上次编辑结果状态，配置改动后会标记尚未重新执行。

默认渠道为贞贞平价小屋，默认模型为 `qwen/qwen3.8-flash-next`。云端默认 8192 个输出 Token（含思考），共享配置明确的 `max_tokens` / `max_completion_tokens` 优先。最多一次格式/长度纠正，失败时保留完整可用原稿并警告，不再额外触发第三次生成。`auto` 不强加预设比例，例如原图 1080×1590 跟随时输出 `36:53`，请确认下游支持，不会近似或裁剪；透明模式只要求 RGBA、Alpha 通道和透明背景的提示词语义，不生成图片。经典模式有历史云端实测记录；新编辑规则的代码/本地浏览器回归不等于真实 API A/B、RunningHub 发布重载或最终出图质量验收。请求 JSON 为元数据，不是可直接提交的渲染 API 请求体。专用 PE 请安装 [独立 Qwen Prompt Rewrite 节点](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8)，这里不重复接入专用模型。

[图像提示词节点说明](web/js/docs/qwen_image21.md) · [文生图工作流](example_workflows/qwen_image21_text_example.json) · [多图编辑工作流](example_workflows/qwen_image21_edit_example.json)

未提供已有 ABC 时，`full / melody` 默认调用当前 LLM 创作谱面（T8 扩展）：完整模式含旋律与和弦，旋律模式不含和弦；已有谱面优先保留。高级设置可切回「交给下游 YuE2 规划」，此时 ABC 留空。自动作谱通常增加 1 次请求，校验失败最多再修正 1 次。

> **ABC 作谱模型建议：本地 9B LLM 可能生成不符合要求的 ABC 格式或小节时值。** 建议优先使用 API，或尝试参数规模更大的模型。本次本地 9B 的 full、melody 实测均未通过谱面校验；已有 API 通过记录，但不代表所有 API 或更大模型都能通过，更大模型本次尚未实测。ABC 失败仍输出风格、歌词和请求 JSON，坏谱不输出，并显示失败原因。详见 [实测记录](web/js/docs/yue2_abc_acceptance.md)。

创作审校评分只代表文本评审，不是音乐听感评分。YuE2 的 `full / melody / off` 是谱面模式，不等于 LLM 思考强度；时长仅为规划参考。

一组面向 MiniMax-H3、Seedance 2.0、Qwen Image 2.1 视频/图像创作和 MiniMax Music 3、YuE2 音乐准备的 ComfyUI 提示词增强节点。H3 与 Seedance 2.0 节点支持文字与真实 `IMAGE` / `VIDEO` 素材；Qwen Image 2.1 支持文字和最多 10 张参考图；Music 3 与 YuE2 按各自官方格式输出。所有核心节点均可选择贞贞平价小屋、贞贞的 AI 工坊、用户自己的 OpenAI 兼容接口，或本地 llama.cpp 兼容 GGUF。

各节点共享已经验证的 API、密钥和错误处理，但提示词协议完全隔离：MiniMax-H3 使用官方字段、任务类型和时间码；Seedance 2.0 使用任务意图、`镜头N` 事件顺序和官方多模态引用语法；Qwen Image 2.1 使用独立的英文图像观察合同；Music 3 和 YuE2 分别执行各自音乐协议。

## 社区与资源链接

| 资源 | 链接 |
| --- | --- |
| B站 | [T8 的哔哩哔哩空间](https://space.bilibili.com/385085361) |
| YouTube | [T8star-Aix](https://www.youtube.com/@T8star-Aix/) |
| API | [获取贞贞 API Key](https://api.seedance.nz/sign-up?aff=5f4w) |
| 在线 AI 应用 | [RunningHub 在线应用与作品](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) |
| ComfyUI 整合包 | [夸克网盘下载](https://pan.quark.cn/s/264edb7e36bd) |
| 模型网盘 | [夸克网盘下载](https://pan.quark.cn/s/c9c267081fbf) |
| Hugging Face | [t8star](https://huggingface.co/t8star) |

## 独立核心节点

| 节点 | 用途 | 主要任务 |
| --- | --- | --- |
| `MiniMax H3 Prompt Enhancer (Cloud / Local GGUF)` | 生成 MiniMax-H3 提示词 | T2VA / I2VA / FL2VA / L2VA / Ref2VA / Hybrid |
| `Seedance 2.0 Prompt Enhancer (Cloud / Local GGUF)` | 生成 Seedance 2.0 提示词 | T2V、首帧、首尾帧、多模态参考、编辑、延长、轨道补齐和组合任务 |
| `MiniMax Music 3 Prompt & Lyrics Enhancer (T8)` | 生成 Music 3 歌词与音乐描述 | AUTO、生成歌词、严格保留、局部润色、纯器乐 |
| `YuE2 音乐提示词与歌词创作（T8）` | 生成 YuE2 风格与歌词 | 原创、保留原词、定向改词、纯器乐、可选 ABC 准备 |
| `Qwen Image 2.1 Prompt Enhancer` | 生成图像提示词 | 文生图、1–10 图像编辑、比例、透明 RGBA |
| `MOSS Image Prompt Enhancer T8` | 生成 MOSS 原生提示词 | Caption / Layout、Fisher、Comic |
| `MiniMax H3 Image Prompt Enhancer T8` | 生成 H3 图像提示词 | 文生图、1–16 图编辑、比例、软词数与锁文案 |

本项目目前不包含 Seedance 2.5 提示词节点，也不调用视频或音乐生成、轮询、试听或下载接口。

## 功能特点

- 平价小屋固定视觉模型 `bytedance/doubao-seed-evolving`；AI 工坊默认 `gemini-3.5-flash` 并支持 Custom；OpenAI 兼容模式支持填写供应商自己的视觉模型 ID。
- 云端视觉渠道分析文字、图片和完整视频；本地 GGUF 明确使用真实时间戳抽样帧与有序联系表，只推断可见时序，不读取音轨，也不冒充完整视频字节理解。
- 支持首帧、尾帧、首尾帧以及多图、多视频参考。
- 集成 MiniMax-H3 官方核心 Skill，规则冻结于经审阅的官方提交 `d21241f0a4b3acbb34c97dae47fa417b7065e438`，并固定全部 4 个文件及归一化内容树哈希。
- 支持现有中英文兼容协议，以及官方所有说明字段强制英文的严格协议。
- 官方共 9 个 Skill：1 个 H3 核心写作 Skill 始终启用，另有 `无 / AUTO` 和全部 8 个可选场景写作 Skill；其中“音乐 MV 动态字幕（官方）”已同步 MiniMax `music-video-subtitle-generator` v0.6.6。选择具体场景 Skill 后，节点内会显示用途、适用范围、推荐输入、结构锚点、可安全填入的示例、MiniMax 官方 GIF 与来源链接。8 个场景 Skill 的完整原生流程依赖 MiniMax Hub agent/canvas/hub 工具；本节点只适配其中可落入单条 H3 提示词的写作约束，不运行或声称移植资产生成、审批、剪辑、画布或外部工作流。
- 两个节点共享独立的 `非官方模板（案例 / 社区 Skill）` 列表：652 条已发布案例事实归并为 305 个稳定案例 selector（含 347 个同机制证据变体），另有 2 个独立用户贡献社区 Skill，共 307 个非官方下拉项；全部提供中文名称、用途、简约推荐输入、2–5 个结构锚点和轻量 GIF 预览。
- 支持中文 / English 输出。
- 支持 `strict / balanced / creative` 三档改写。
- 支持 `AUTO` 或固定 1–20 个镜头的下拉控制。
- 支持用户参考模板融合，主提示词与可观察媒体事实优先。
- 提供随机种子以及 `fixed / randomize / increment / decrement` 状态。
- 提供节点内 API Key 输入、遮罩显示、保存、清空和注册链接。
- 支持贞贞平价小屋、贞贞的 AI 工坊、显式配置的 OpenAI 兼容接口，以及免 API Key 的本地 llama.cpp GGUF；Qwen3.8 两个固定型号继续实测支持，同时开放其他 llama.cpp 兼容文字模型与带匹配 mmproj 的视觉模型。
- H3/Seedance 输出单一提示词 `STRING`；Music 3 分别输出歌词、Structured Caption、payload JSON 和安全增强报告四个 `STRING`。
- 新增独立 Seedance 2.0 节点：简单/复杂双路径、AUTO/固定 1–20 镜头、官方/Seedance.nz 引用语法、字幕与稳定性策略。
- 新增独立 Music 3 节点：完整内置官方 `music-caption-rewriter`、18 个流派索引和 1000 个模板，并严格按 router → 最多 2 个索引 → 最多 3 个模板逐级披露，绝不会把全库塞进一次 LLM 请求。
- Music 3 的官方 Caption 与歌词完全分离；生成/润色歌词属于清楚标注的 T8 非官方扩展，严格保留模式不会改动用户歌词。
- T8 案例支持独立浏览器：按分类筛选、搜索、收藏、最近使用和单 GIF 懒加载预览；原有下拉值与工作流 selector 保持兼容。
- 三个节点提供 ComfyUI 原生进度、仅驻留内存的脱敏执行诊断，以及“恢复上次云端结果（不重新生成）”按钮；恢复只读取当前进程内的完整结果，不会重新提交付费请求。
- 三个节点新增“渠道能力预检”和“查看/复制脱敏诊断”入口；OpenAI 兼容未知渠道不会被标记为已验证视觉模型。
- 可选 `T8 LLM Provider Config` 辅助节点可统一渠道、模型、Base URL、本地 Qwen 设置、`temperature` 策略和白名单参数；断开后立即恢复三个原节点自己的值。
- 可选 `T8 Performance Director Config` 为人物/角色/情绪/对白请求增加可观察的表演结构，支持 `AUTO / 强化 / 关闭 / 极致`；不新增 LLM 请求，不属于 MiniMax 官方 Skill。
- 可选本地凭据别名让工作流只保存别名、真实 Key 留在 ComfyUI 用户目录；原有 API Key `STRING` 接线和工作流保存方式继续可用并具有更高优先级。
- 可选 `T8 Prompt Inspector` 只在本地检查 H3、Seedance 2.0 与 Music 3 的可复算结构，并给出不计分的表演风险 advisory；原文逐字直出、不调用 LLM 或判断创意质量。
- T8 模板浏览器支持确定性的本地 Top-3 推荐与 2–3 项并排对比；推荐不会自动改动当前模板，也不会把用户输入或 GIF 发到外部服务。

## 可选的 P0–P2 辅助节点

`T8 LLM Provider Config`、`T8 Performance Director Config`、`T8 Prompt Inspector`、`T8 Prompt Text` 和 `T8 Show Text` 是仓库自带的辅助节点，不替代三个核心 enhancer，也不改变它们的既有输出。`T8 Prompt Text` 可替代第三方多行文本输入节点，`T8 Show Text` 可在节点内只读显示并原样透传 `STRING`。H3/Seedance 只追加可选 `character_performance_bible`、`performance_director_config` 与 `provider_config` socket，Music 3 只追加 `provider_config`；恢复功能使用隐藏运行时字段和节点属性，不加入原有 `widgets_values` 数组。旧工作流的 31/35/38 个序列化 widget、API Key `STRING` 接线、默认渠道和输出顺序保持不变。

### “共享 LLM 渠道配置”怎么连接

绿色的“共享 LLM 渠道配置（可选）”不是 `STRING`，不能连接普通文本或 API Key 节点。请搜索并添加 `T8 LLM 共享渠道配置`，把它右侧的“渠道配置”输出接到 H3、Seedance 2.0 或 Music 3 节点同名输入：

1. “共享渠道”选择 `Local Qwen`、平价小屋、AI 工坊或 OpenAI Compatible。
2. 本地模式从 `ComfyUI/models/LLM` 递归发现主模型；H3/Seedance 分析图片或视频采样帧时把视觉投影器设为 `AUTO（自动匹配）` 或显式选择对应 mmproj，Music 3 会忽略视觉投影器。
3. 本地模式不填 API Key；云端可继续把普通 `STRING` Key 直接接到原节点，或在共享配置中使用本地凭据别名。
4. 连接共享配置后，以共享节点中的渠道、模型和本地参数为准；断开连接会立即恢复原节点自身设置。
5. 不想使用共享节点时，也可以直接在任一核心节点的 API 模式中选择“本地 GGUF（llama.cpp / Qwen，离线）”。旧工作流保存的“本地 Qwen3.8-27B”值仍会正常执行。

共享配置的凭据优先级为：原节点执行时收到的 `api_key`（包括外部 STRING 接线或已保存值）→ 本地凭据别名 → 原节点既有环境变量后备。凭据管理器可列出、保存、更新、删除和显式测试云端连接；测试连接会先提示并只发送一次最小请求，界面不显示 Key、URL 或上游正文。

OpenAI 兼容默认继续发送 `temperature`，保持 1.2.0 行为。只有已知 Kimi Coding Plan 地址的 `AUTO` 策略会省略该字段，也可在共享配置中显式选择发送或省略。附加参数只接受白名单，不能覆盖模型、消息、鉴权、媒体或流式控制等核心字段。

### “表演导演配置”怎么用

`T8 Performance Director Config（表演导演）` 的紫色 Custom Type 输出可以连接到 H3、Seedance 2.0 或 `T8 Storyboard Pack` 的同名可选输入。它不调用 LLM，只改变下一次增强请求中的表演导演系统规则：

1. `AUTO（按人物 / 表演意图）`：默认模式。只有文本或素材确实包含人物、角色、情绪、对白、反应或表演意图时才应用；纯风景、产品、文字动效和音乐请求跳过。
2. `强化（明确表演结构）`：要求按“触发 → 接收 → 一个主反应 → 收束”显式组织人物表演，适合剧情、微表情和对白镜头。
3. `关闭（保持原编译）`：完全不加入表演导演规则，恢复原 H3/Seedance 编译路径。
4. `极致（深度表演重构）`：新增模式放在原有三个选项之后，保证旧选项顺序稳定。它把已有增强提示词也视为可编辑草稿，在保留人物事实、素材身份、逐字对白、时长/镜头数和原生输出格式的前提下，实质性重写表演相关的节拍、因果和时间顺序。只有形容词、标点或同义词变化不算完成；非人物请求不会被强行加入角色或剧情。
5. `土豆-情绪演绎（声画协同）`：可选新策略，追加在旧四项之后。区分“什么情绪、强度多大、声音多响”，衔接前后余绪，协调逐句演法、口型和景别内动作；不是比极致更激烈。例如“含泪但低声清楚、不哭腔”不会一律变成抽泣，明确爆发也不会被一律压成微表情。可以与独立戏剧/战斗 Skill 和角色圣经组合，不额外要求填写表格。

新策略不改变原生三/六字段、参考图、首尾帧或 Relay，不增加规划/评分调用；选择它不授权新对白、事件或声音。已有完整稿可以保留。来源、边界与单独的视频验收状态见[情绪演绎说明](./docs/EMOTION_PERFORMANCE.md)；API 文本表现不等于成片效果。

[真实 API 验收报告](./docs/EMOTION_API_ACCEPTANCE.md)保留失败及不同代码阶段：初版 48 槽有 47 份完整稿，不能宣称全面胜过极致。H3 现在会免费自动校正全角说话人编号、明确归属的漏号和已知语言标签别名，本地/API、质量 Off 均生效，不改原台词。多人归属不明确时不猜测、不拒绝完整稿，诊断会提示；需要进一步纠正可选择已有“质量纠正”（最多追加一次）。见[格式修复说明](./docs/H3_VOCAL_PROTOCOL_FIX.md)。

规则会把抽象情绪转换为视线、眼睑、呼吸、肩颈、手部或姿态等可见线索，先确定镜头侧与视线目标，再描述面部方向；每个角色、每个触发/接收/反应/收束节拍最多保留三个高信号线索通道，避免短时长画面堆成动作清单。强状态变化可借闭眼、低头、前景/动作遮挡或切镜隐藏，但用户要求连续可见的变化不会被遮住。它不会发明对白，并要求把吞咽、闭嘴等冲突口部动作放到发声跨度之外。

AUTO/强化/极致还会锁定用户明确写出的身体部位、产品零件、颜色、左右关系和引号文字：例如“瞳孔变金”不能转写成“虹膜变金”。用户固定时长/镜头数、逐字对白、素材角色、官方 Skill、硬性要求和 LOCK 锚点始终优先。需要检查已生成结果时，可把原始提示词接入 `T8 Prompt Inspector` 新增的“原始提示词（可选，用于语义漂移检查）”输入；检查完全本地执行，只报警、不改写、不增加请求。

H3 与 Seedance 使用两套独立规则：Seedance 不会收到 H3 的字段、标签、说话人编号、毫秒时间码或帧网格。Storyboard 会额外返回可选逐镜表演 IR：`dramatic_trigger`、`reception_beat`、`primary_performance_beat`、`observable_cues`、`gaze_target`、`speech_span`、`state_transition_strategy` 和 `performance_risks`；非表演镜头保留空值，不为了填表编造情绪。

这套能力受到社区项目 [`phileiny/h3-storyboard-skill`](https://github.com/phileiny/h3-storyboard-skill) 固定提交 `ab65851f599435a1ff94ea4931949bd7bcaf069b` 的启发，但不是 MiniMax 官方 Skill，也不是已证明的跨模型规律。来源文件哈希、许可证、证据范围和禁止过度外推项见 [`research_sources/h3-storyboard-skill.lock.json`](./research_sources/h3-storyboard-skill.lock.json) 与 [`research_sources/NOTICE.md`](./research_sources/NOTICE.md)。

维护者可创建完全离线、初始全部为 `pending` 的 200 项成对 A/B 基准；工具不会生成视频，也不会把空结果算作成功：

```bash
python tools/verify_research_source.py
python tools/performance_benchmark.py init --output performance-benchmark.json
python tools/performance_benchmark.py validate performance-benchmark.json
python tools/performance_benchmark.py summarize performance-benchmark.json
```

只有补齐模型版本、成对视频哈希、注册指标和盲评后，报告才会从 `no_observations` 进入实际观察状态；全画面 PSNR 不会被当作人脸表演质量。

## T8 创作导演套件（独立辅助节点）

创作导演套件扩展的是“策划、探索、修改和交付”链路，不替代三个核心 enhancer。三个核心 Node ID、既有 widget 顺序、默认值和输出保持不变；不使用这些新节点的旧工作流不需要迁移。

| 节点 | 是否调用 LLM | 用途 |
| --- | --- | --- |
| `T8 Film Project Router` | 否 | 维护阶段、权威输入、版本修订、世界合同与上游修改造成的下游失效状态 |
| `T8 Character Performance Bible` | 否 | 把角色目标、阻力、策略、身体惯性、声音与倾听反应编译成每节拍最多三个可观察通道的轻量合同 |
| `T8 Character Performance Bible Stack` | 否 | 汇总 1–8 个独立角色合同，拒绝重复标识，并让多角色分镜逐人保留目标、策略与可观察线索 |
| `T8 Creative Director` | 否 | 建立人物、世界、视觉、动作、声音等统一创作总纲，维度支持 `LOCK / EVOLVE / AUTO` |
| `T8 Creative Context Assembler` | 否 | 把总纲、素材角色、DNA 和个人预设组装成可接现有核心节点的 `STRING` |
| `T8 Directed Revision` | 1 次 | 只修改点名范围，输出修订结果、报告与本地 diff |
| `T8 Long-form Planner` | 1 次 | 任意正整数总时长拆段，分别输出 H3、Seedance 2.0、世界规则/知情差检查和段间交接 JSON |
| `T8 Reference Role Mapper` | 1 次 | 分析用户实际连接的图片/视频，分配身份、服装、场景、动作、镜头、风格和禁止借用角色 |
| `T8 Creative Candidate Lab` | 1 次 | 一次生成 2–4 个创作机制不同的候选及软比较 |
| `T8 Candidate Selector` | 否 | 从候选 JSON 本地选择一个方向 |
| `T8 Storyboard Pack` | 1 次 | 输出全局提示词、分镜、关键帧、转场/声音、素材绑定、逐镜表演 IR 与因果/价值/必要性/伏笔审计 |
| `T8 Creative DNA Mixer` | 否 | 最多融合三个 T8 非官方案例的结构、镜头和高潮/收尾机制，不发送案例媒体 |
| `T8 Personal Creative Preset` | 否 | 在工作流内保存用户自己的文字型创作规则，不修改内置库或写外部文件 |
| `T8 Music Creative Lab` | 1 次 | 歌词/Caption 候选、定向歌词修改、歌曲标题和中日韩文字层软 QA |
| `T8 Creative Version Stack` | 否 | 保存、选择和回退最多 8 个提示词、歌词或 Caption 版本 |
| `T8 Music-to-Video Beat Sheet` | 1 次 | 根据歌词、Caption 和用户已知 BPM/时间点生成 H3/Seedance 视频节拍表 |

需要 LLM 的节点默认使用 Seedance NZ；推荐连接 `T8 LLM Provider Config`，统一选择平价小屋、AI 工坊、OpenAI Compatible 或本地 GGUF。每次执行只有一次逻辑生成请求；Seedance NZ 遇到已确认安全重试的 500/502/503/504/520–530 网关状态时最多进行 6 次有界传输尝试，401/402/429 和可能已完成计费的读超时不会盲目重复。上游返回非空但没有命中 JSON 时，节点保留原文并在报告中标记 `structured_response=false`，不会因为创意评分或格式偏差丢弃内容。

候选的七维软分数由节点对最终文本做确定性的本地启发式计算，不要求模型额外“自评分”，不冒充客观艺术质量，也不阻塞非空候选。分镜包只让模型返回一次逐镜合同，关键帧表与转场/声音表在本地从逐镜字段派生，避免付费响应重复同一内容。2026-08-28 使用真实 `bytedance/doubao-seed-evolving` 完成 7 类脱敏验收：候选、定向修改、长片分段、4 镜头分镜、多模态角色映射、中文歌词候选和 5 镜头音乐节拍表均通过二值合同检查；该验收分数不等于主观艺术评分。

2026-08-30 又完成 6 组真实成对文本 A/B：因果/价值变化、伏笔回收、策略切换、无对白表演、世界规则/永久代价、知情差/下游失效。平价小屋的 `qwen/qwen3.6-flash` 在同题同 seed 下，旧合同平均 44，新合同最终平均 100，6/6 组均未退步；分数只来自可复现的 JSON 结构、字面锚点、顺序、线索预算和状态字段检查，不是主观艺术评分。最终复测复用了 6 条匹配的真实基线并新发 6 条增强请求；另有 2 次 `bytedance/doubao-seed-evolving` 增强试验返回 HTTP 524，未计入分数且响应正文未保存。脱敏报告见 [`tests/fixtures/film_workflow_paid_ab_2026-08-30.json`](./tests/fixtures/film_workflow_paid_ab_2026-08-30.json)，复现实验工具见 [`film_workflow_ab_benchmark.py`](./film_workflow_ab_benchmark.py)。新报告会记录仓库版本/提交、模型 ID、参数、seed 策略和合同哈希；`--model` 可对不同云端模型分别复测，只有合同与模型完全匹配时才复用基线。本地 Qwen 的参数传递由本地回归集覆盖，但尚未把本地模型结果冒充真实跨模型质量结论。

若要验收最终 H3 / Seedance 成片，可使用 [`film_workflow_render_acceptance.py`](./film_workflow_render_acceptance.py) 读取真实视频清单：它记录文件哈希、ffprobe 时长/视频流证据和具名人工五维评分，不生成视频、不保存视频或密钥，也不会把文本 100 分当成成片质量满分。

素材边界：Reference Role Mapper 只会读取用户直接连接到该节点的媒体。官方/T8 案例 GIF、来源视频和人类预览永远不会作为模型参考素材。本地 GGUF 的视频理解仍只基于带真实时间戳的采样画面，不分析音轨；Music-to-Video Beat Sheet 没有 AUDIO 输入，不会声称听歌、检测 BPM 或转写歌词。

推荐连接方式：

1. `T8 Film Project Router` → `T8 Long-form Planner` / `T8 Storyboard Pack`，把权威输入、世界规则、人物知情差和失效状态传给下游；下一版 Router 可直接接上一版紫色状态，空白权威字段自动继承。需要删除某一个旧字段时，在对应输入中填写 `[清空继承]`（或 `[CLEAR_INHERITED]`），其他留空字段仍继续继承。
2. 单角色：`T8 Character Performance Bible` → H3 / Seedance 2.0 / `T8 Storyboard Pack`；多角色：多个 Bible → `T8 Character Performance Bible Stack` → 下游。两者都不增加 LLM 请求。
3. `T8 Creative Director` → `T8 Creative Candidate Lab` / `T8 Long-form Planner` / `T8 Storyboard Pack`。
4. 有素材时先运行 `T8 Reference Role Mapper`，把 `reference_context_for_enhancer` 接到现有 H3 的“参考素材补充”，或把角色表接到 `T8 Creative Context Assembler`。
5. `Candidate Lab` → `Candidate Selector` → 现有 H3/Seedance 的主提示词，或继续进入 `Storyboard Pack`。
6. `Music Creative Lab` 的候选可进入 `Creative Version Stack`；选定歌词与 Music Caption 再进入 `Music-to-Video Beat Sheet`。

`T8 Long-form Planner` 与 `T8 Storyboard Pack` 运行后会在节点底部显示结构协议状态：绿色表示合同通过，红色会列出错误数量与部分稳定错误码。状态卡跟随 ComfyUI 中英文界面并按内容自动增高。新增的“协议失败处理”默认保持旧工作流的“警告并保留输出（兼容）”；改为“严格阻止下游（推荐）”后，合同失败会用 ComfyUI 原生阻断器停止下游，同时保留状态卡和 JSON `validation_errors`。这些本地检查不会增加 LLM 调用或付费次数。

## 1.1.0 更新

- 新增免 API Key 的本地 Qwen3.8-27B GGUF 渠道，并支持官方与第三方 Uncensored Q4_K_M 变体、视觉投影器、显存释放、执行后卸载策略和真实时间戳视频联系表。
- MiniMax H3 核心 Skill 更新为固定官方快照；维护工具可核验内容树，并由每周工作流只读检查 H3 核心、8 个官方创意 Skill 和 Music 3 官方 Skill 的上游路径是否发生变化。
- 662 个官方/T8 GIF 全部可用；主仓库保留 8 个官方 GIF 和既有 377 个 T8 GIF，新加入的 T8 预览不再继续扩充主包。全部 654 个 T8 GIF 都可由独立版本化资源通道按需校验、下载和更新，避免 GitHub/Registry 包随案例增长持续膨胀。
- 提供 13 个 ComfyUI 原生 `example_workflows` 与同名缩略图：原有 9 个基础/本地工作流保持不变，另增 4 个创作导演套件组合案例；全部示例不依赖 Comfyroll、EasyUse 等第三方节点。
- 三种云端渠道统一使用一个经过测试的传输层，同时保留各自既有付费重试、HTTP 分类和错误文案；不记录请求或响应正文。
- 新增版本门禁、密钥/大文件/JSON/GIF 预算检查、Changelog、License、Registry 元数据与发布前 CI。
- 旧工作流迁移规则和稳定接口见 [`COMPATIBILITY.md`](./COMPATIBILITY.md)。

## 安装

> **更新建议：优先从 GitHub 更新。** 当 Registry 的新版本处于审核、`Pending` 或 `Flagged` 状态时，ComfyUI-Manager 可能自动回退安装较旧的 `Active` 版本；界面虽然会显示“安装/重装成功”，但新节点、新功能和紧急修复仍然不可用。更新后必须完整重启 ComfyUI，并对浏览器执行一次 `Ctrl+F5`。不要在 `custom_nodes` 中保留本插件的多个副本。

### GitHub 安装 / 更新（推荐）

首次安装时，在 ComfyUI 的 `custom_nodes` 目录执行：

```bash
git clone https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8.git
```

已经通过 Git clone 安装时，在插件目录执行：

```bash
git pull --ff-only
```

如果原来通过 Manager 或 ZIP 安装，更新前请先关闭 ComfyUI，把旧插件目录移出 `custom_nodes`，再克隆到唯一的标准目录；不要让旧目录与新目录同时存在。目录结构应为：

```text
ComfyUI/
└─ custom_nodes/
   └─ comfyui-minimax-h3-prompt-enhancer-T8/
```

### ComfyUI-Manager / Registry

在 ComfyUI-Manager 的节点管理器中搜索：

```text
MiniMax H3 / Seedance 2.0 / Music 3 Prompt Enhancer (T8)
```

也可以使用 Comfy CLI：

```bash
comfy node install minimax-h3-seedance-music3-prompt-enhancer-t8
```

Registry 发布包会包含节点运行代码、官方 Skill、非官方案例库、8 个官方 GIF 和一份轻量资源通道清单；654 个 T8 GIF 不会塞入 Registry ZIP。用户查看某个 T8 案例时，节点默认只下载该案例所在的小分片并缓存到 `ComfyUI/user/t8_prompt_enhancer/preview_assets`。发布包仍不包含测试、来源批次、API Key、独立 `llama-server` 运行时、安装/下载脚本或 GGUF 模型。Manager 安装的本地 GGUF 模式使用当前 ComfyUI Python 中的 `llama-cpp-python`；如果需要固定 `llama-server`、PATH 自动回退、环境变量默认值或显式连接探测，请按上面的推荐方式从 GitHub 完整安装。

云端模式不增加额外 Python 依赖，使用 ComfyUI 已安装的 `requests`、NumPy、Pillow 和原生媒体类型；本地视频抽样复用 ComfyUI 自带的 PyAV。安装后重启 ComfyUI；如果节点或前端界面没有更新，请对浏览器执行一次 `Ctrl+F5`。

在节点菜单中搜索任一节点：

```text
MiniMax H3 Prompt Enhancer (Cloud / Local GGUF)
Seedance 2.0 Prompt Enhancer (Cloud / Local GGUF)
MiniMax Music 3 Prompt & Lyrics Enhancer (T8)
```

## 快速使用

1. 添加 `MiniMax H3 Prompt Enhancer (Cloud / Local GGUF)`。
2. 在“视频创意 / 提示词”中输入基础意图。
3. 选择生成类型、时长、镜头数量、改写模式、输出语言、官方 Skill 协议和 MiniMax 官方创意预设；需要时再选择一个非官方案例或社区 Skill 模板。
4. I2VA / FL2VA / L2VA / Ref2VA / Hybrid 按任务要求连接图片或视频。
5. 云端渠道填写 API Key；本地 Qwen 渠道不需要 Key。
6. 点击节点底部的“运行提示词优化”。
7. 从 `enhanced_prompt` 获取最终字符串。

可以直接把 [`example_workflows/minimax_h3_prompt_enhancer_example.json`](./example_workflows/minimax_h3_prompt_enhancer_example.json) 拖入 ComfyUI。示例工作流不包含 API Key；同目录 JPG 是 ComfyUI 模板缩略图。

## MiniMax Music 3 快速使用

1. 添加 `MiniMax Music 3 Prompt & Lyrics Enhancer (T8)`。
2. 在“音乐创意”中写明风格、主题、情绪、用途和希望的编曲发展。
3. 选择歌词模式：`AUTO / 生成新歌词 / 严格保留歌词 / 按要求润色 / 纯器乐`。已有歌词可以通过普通 `STRING` 接入。
4. “官方完整（2–4 次请求）”会按官方逐级披露选择参考；“快速核心（1–2 次请求）”只执行官方核心合同。调用次数取决于是否还要生成或润色歌词，不会同时调用三个供应商。
5. 云端渠道填写当前 API Key；本地 Qwen 不需要 Key。点击“运行 Music 3 提示词与歌词优化”。
6. 将 `lyrics` 接到 Music 3 的 `input`，将 `music_caption` 接到 `instructions`；也可以直接解析 `music3_payload_json`。
7. `enhancement_report_json` 只记录阶段、付费请求数、缓存命中、官方快照哈希、Token 预算估算与警告代码，不包含歌词、用户创意、API Key、供应商 URL、模板 ID 或模板正文。

可以直接导入 [`example_workflows/music3_prompt_lyrics_enhancer_example.json`](./example_workflows/music3_prompt_lyrics_enhancer_example.json)。示例不包含 API Key、音频或官方模板正文。

## Seedance 2.0 快速使用

1. 添加 `Seedance 2.0 Prompt Enhancer (Cloud / Local GGUF)`。
2. 填写“视频创意 / 提示词”。任务意图、组织方式、时长和镜头数都可先保持 `AUTO`。
3. 按任务连接首帧、尾帧、参考图片或完整参考视频。
4. 选择官方中文 `@图片N/@视频N` 或 Seedance.nz 英文 `@Image N/@Video N` 引用格式。
5. 可选一个 `非官方模板（案例 / 社区 Skill）`，用于迁移因果结构、节奏和镜头语法。
6. 云端渠道填写“提示词增强 LLM API Key”；本地 Qwen 不需要 Key。然后点击节点底部运行按钮。
7. 把 `enhanced_prompt` 连接到下游 Seedance 2.0 视频节点的提示词输入。

可以直接导入 [`example_workflows/seedance20_prompt_enhancer_example.json`](./example_workflows/seedance20_prompt_enhancer_example.json)。示例不包含 API Key。

## 示例工作流

| 示例 | 用途 |
| --- | --- |
| [`h3_visual_marker_ref2va.json`](./example_workflows/h3_visual_marker_ref2va.json) | H3空间入门：青框人物走向黄框门口；在空间配置节点点「示例与接线」下载配套PNG |
| [`h3_visual_marker_hybrid.json`](./example_workflows/h3_visual_marker_hybrid.json) | 干净首帧＋标记参考图；标记为Picture 2，含完整接线与操作说明 |
| [`minimax_h3_prompt_enhancer_example.json`](./example_workflows/minimax_h3_prompt_enhancer_example.json) | H3 云端基础示例 |
| [`seedance20_prompt_enhancer_example.json`](./example_workflows/seedance20_prompt_enhancer_example.json) | Seedance 2.0 云端基础示例 |
| [`music3_prompt_lyrics_enhancer_example.json`](./example_workflows/music3_prompt_lyrics_enhancer_example.json) | Music 3 云端基础示例 |
| [`basic_workflow_multi_task_connections.json`](./example_workflows/basic_workflow_multi_task_connections.json) | 一个共享配置同时连接 H3、Seedance 2.0 与 Music 3；只使用本仓库节点 |
| [`minimax_h3_local_qwen_example.json`](./example_workflows/minimax_h3_local_qwen_example.json) | H3 + 共享配置 + 本地 Qwen3.8-27B |
| [`seedance20_local_qwen_example.json`](./example_workflows/seedance20_local_qwen_example.json) | Seedance 2.0 + 共享配置 + 本地 Qwen3.8-27B |
| [`music3_local_qwen_example.json`](./example_workflows/music3_local_qwen_example.json) | Music 3 + 共享配置 + 本地 Qwen3.8-27B，不加载 mmproj |
| [`prompt_inspector_local_qwen_example.json`](./example_workflows/prompt_inspector_local_qwen_example.json) | H3 本地增强后接 T8 Prompt Inspector 本地结构检查 |
| [`text_utilities_example.json`](./example_workflows/text_utilities_example.json) | 仓库自带 T8 Prompt Text → T8 Show Text，演示标准 STRING 输入、显示与透传 |
| [`creative_direction_revision_example.json`](./example_workflows/creative_direction_revision_example.json) | 创作总控 → 多方案实验室 → 候选选择 → 定向修改 |
| [`creative_longform_storyboard_example.json`](./example_workflows/creative_longform_storyboard_example.json) | 项目状态、角色表演圣经与共享创作总纲共同驱动长视频分段和分镜审计 |
| [`creative_music_video_suite_example.json`](./example_workflows/creative_music_video_suite_example.json) | 音乐候选、版本选择与 H3/Seedance 2.0 音乐视频节拍导演 |
| [`creative_reference_dna_preset_example.json`](./example_workflows/creative_reference_dna_preset_example.json) | 素材角色、T8 Creative DNA、个人预设组装为旧增强节点可接收的 STRING 上下文 |

所有内置工作流均不包含 API Key。涉及三个原有核心节点的工作流继续保存完整的 31/35/38 项控件数组；旧版 21/22、25/26、31 项工作流会在下拉值校验前自动补齐本地 Qwen 默认值，避免把 `randomize` 错读为 GGUF 模型。

### Seedance 2.0 任务意图

| 选项 | 用途 | 素材规则 |
| --- | --- | --- |
| `AUTO` | 按意图和已连接素材判断 | 无歧义时自动选择 |
| `T2V` | 文生视频 | 不连接素材 |
| `I2V` | 首帧图生视频 | 只连接 `first_frame` |
| `FL-I2V` | 首尾帧过渡 | 连接 `first_frame` 与 `last_frame` |
| `多模态参考生成` | 借用图片或视频中的主体、动作、运镜、风格等 | 至少连接一个参考素材 |
| `视频编辑` | 对已有视频增删改 | 至少一个参考视频，提示词直接写 `@视频N` |
| `视频延长` | 向前或向后延长 | 恰好一个参考视频 |
| `轨道补齐` | 在视频间生成衔接 | 2–3 个参考视频 |
| `组合任务` | 参考一个素材并编辑另一个视频 | 被编辑视频加至少一个其他素材 |

视频编辑与组合任务默认把 `@视频1` / `@Video 1` 作为被编辑目标；其他连接素材是支持参考。需要不同角色时，在高级“素材用途”中明确指定。

### Seedance 2.0 提示词组织

- 官方八要素按需补齐：主体、动作、场景、光影色调、单一主运镜、风格、画质和约束。
- 简单任务使用一个紧凑自然段；复杂任务才使用 `镜头1 / 镜头2 / 镜头3`。
- 镜头数支持 `AUTO` 或固定 1–20。固定镜头是传给 LLM 的软生成约束；时长较短却使用较高镜头数时，上游可能合并或省略镜头，节点不会因此判错。
- 目标时长支持 `AUTO` 或手动输入任意正整数秒数，节点不设上限；它只控制提示词整体内容密度，不生成毫秒级时间码，也不默认写逐镜头绝对秒数。下游视频模型或工作流自身的时长限制仍然有效。
- 编辑、延长和组合任务直接指代 `@视频N`，不写成“参考 @视频N”。
- “素材用途”可补充 `@图片1=人物外观；@视频1=动作和运镜`，但不能替代真实素材分析。
- “参考模板融合”只迁移结构、节奏、运镜、转场、风格和声音设计，不复制模板中的人物、道具和剧情事实。

### Seedance 2.0 音频边界

Seedance 2.0 目标视频模型能够处理音频，不代表提示词增强渠道能分析音频文件。2026-08-05 的真实能力探测中，平价小屋/兼容模式使用的 `bytedance/doubao-seed-evolving` 明确拒绝 OpenAI 兼容 `input_audio`，返回“audio input is not supported by this model”。新增 AI 工坊渠道不会改变本节点的输入合同：两个节点都没有 `AUDIO` 输入，也不会声称听过上传音频。

用户仍可在文字里描述对白、环境声、动作声、音乐和音色，或保留文字形式的 `@音频N/@Audio N` 意图；这些内容只作为文字理解。探测结论的脱敏记录位于 [`tests/fixtures/seedance20_audio_probe_2026-08-05.txt`](./tests/fixtures/seedance20_audio_probe_2026-08-05.txt)。

## MiniMax Music 3 官方 Skill 与歌词边界

Music 3 节点内置的是 MiniMax 官方 `music-caption-rewriter` 完整快照，固定于提交 `91410fb657c007ae57c60df8240f5ece5be089c7`：18 个 family index、1000 个完整模板、共 1022 个文件。来源、数量与归一化内容树哈希记录在 [`official_skills/SOURCE.json`](./official_skills/SOURCE.json)，测试会现场重算并阻止残缺发布。

节点里的“歌词模式”不是官方 Skill 选择器：生成新歌词和按要求润色属于 T8 非官方扩展，严格保留属于本地保护，AUTO/纯器乐属于工作流控制。官方 Skill 的正式能力是生成 `Global Metadata / Vocal Details / Arrangement` 三段 `music_caption`；“官方 Skill 质量模式”选择“官方完整”时，才会在三段合同之外继续执行流派路由、最多两个索引和最多三个官方模板的逐级披露。

官方 Caption 固定按以下顺序输出，默认使用英文并以约 250–450 词作为软目标：

```text
### Global Metadata
### Vocal Details
### Arrangement
```

- 歌词文本只用于 Music 3 的 `input`；Caption 只用于 `instructions`。节点不会在 Caption 中引用、改写、概括或复述歌词正文，只会传递方括号 section tags 和整体音乐意图。
- `生成新歌词` 与 `按要求润色` 是 T8 非官方扩展，不冒充官方 Skill；`严格保留歌词` 在本地逐字直通，不产生歌词 LLM 请求。
- “歌词语言”只约束 `lyrics` 输出，与官方 Caption 的描述语言完全分离。AUTO 会从音乐创意和已有歌词文字推断中/英/日/韩；生成阶段若上游仍返回错误语言，节点只追加一次歌词语言纠正请求，保持标签与结构，并且不修改 `music_caption`。生成新歌词模式会隐藏不参与请求的“已有歌词”输入框，避免把语言名称误填到歌词正文。
- 节点先把用户要求整理为带来源标记的私有 `Music Brief`，再把同一份 BPM、调式、拍号、器乐/人声状态、乐器、排除项和结构约束传给 router、参考选择器与 Caption 编译器。否定句、融合风格或官方歧义词不会被本地关键词强行路由，而是交给官方 router 规则判断。
- 方括号时间线同时保留 section tags 与安全 control tags，例如 `[Verse 2: breathy vocal]`、`[drums drop out]`；纯器乐的自定义结构也会完整进入 Music Brief。URL、工具调用、系统提示、密钥等疑似注入内容不会作为编曲指令进入官方 Caption 阶段。
- “按要求润色”可限定全部歌词、指定段落或第 N 次出现；未命中的目标会在付费前报错，目标以外段落由本地逐字合并保护。`严格保留歌词` 永远逐字直通。
- 歌词语义默认“隐私隔离”：Caption 不读取歌词正文。用户可手填宽泛语义摘要，或明确选择额外一次 LLM 分析；LLM 只返回情绪、叙事强度、能量弧线和人声密度四个枚举，不向 Caption 阶段传递歌词原句。
- 官方资源、固定核心 Skill 哈希和 18/1000 索引模板完整性会在创建供应商会话和任何付费请求之前检查；残缺或被意外修改时直接本地失败。
- “阶段缓存”是当前 Python 进程内、10 分钟、仅成功结果的短缓存，用于后段失败后续跑；缓存键只保存加盐哈希并隔离不同凭据，不落盘保存歌词。画布上的请求估算会提示预计阶段和缓存可能减少的调用数。
- `纯器乐` 输出 `[Instrumental]`，并要求 Caption 不引入歌手。歌词/Caption 目标长度、歌曲时长、BPM、结构和押韵均是软约束，上游返回非空可用内容就不会因偏差报错。
- 该节点没有 `IMAGE`、`VIDEO` 或 `AUDIO` 输入，不分析参考音乐，也不调用 Music 3 音频生成 API。它只准备文字。

## 生成类型

| 选项 | 含义 | 素材要求 |
| --- | --- | --- |
| `T2VA（文生音视频）` | 文本生成音视频提示词 | 不连接媒体 |
| `I2VA（首帧图生音视频）` | 从首帧向后发展 | 必须连接 `first_frame` |
| `FL2VA（首尾帧生音视频）` | 在首尾帧之间设计运动 | 必须连接 `first_frame` 和 `last_frame` |
| `L2VA（尾帧图生音视频）` | 从合理前态收束到尾帧 | 必须连接 `last_frame` |
| `Ref2VA（参考图/视频生音视频）` | 完整参考生成或编辑 | 至少一张参考图或一个参考视频 |
| `Hybrid（关键帧+参考混合生成）` | 首尾边界＋额外视觉参考 | 首帧和/或尾帧，另需至少一张参考图或一个参考视频 |

## 提示词模式

### 官方增强

根据用户意图、真实媒体和 MiniMax-H3 规则直接生成提示词。一般只需要填写主提示词，LLM 会自行补充镜头、动作、声音和节奏。

### 参考模板融合

在官方规则之外，允许提供一段参考模板。模板用于迁移镜头组织、节奏、运镜、转场、视觉风格和声音设计，不会默认照搬模板里的角色、道具、剧情、对白或固定镜头数量。

优先级为：

```text
硬性要求 > 用户主提示词与媒体事实 > MiniMax-H3 核心规则 > MiniMax 官方创意预设 > 更具体的用户手动参考模板 > T8 非官方案例 / 社区 Skill 模板
```

## H3 核心写作 Skill（始终启用）

MiniMax 官方 9 个 Skill 的组成是“1 个核心写作 Skill + 8 个场景 Skill”。核心 `h3-prompt-writing` 负责所有任务共用的 H3 字段、时间线、素材标签与声音合同，因此不是一种可选视频风格；节点界面的协议选项只控制它采用兼容输出还是严格英文输出。

| 选项 | 行为 |
| --- | --- |
| `现有兼容（保留中英文）` | 默认值；保留现有中文 / English 正文体验，同时使用新版结构、说话人、声音和 Ref2VA 素材角色规则 |
| `官方 Skill 严格（全英文协议）` | 所有说明字段和描述正文强制使用英文；仅用户原始对白、歌词、品牌/UI 文案和画面可见文字保留原语言 |

严格档位优先于“输出语言”选择。例如同时选择“中文”和“官方 Skill 严格”，实际说明正文仍为英文。这样不会静默改变旧工作流，新节点也继续默认中文兼容模式。Ref2VA 生成类任务在严格档位下默认以约 350–500 English words 作为软目标；未达到目标字数仍不是节点错误。

新版核心规则还包括：按目标视频首次真实发声顺序分配 `(S1)`，多人同声使用 `(S1,S2)`，跨切镜对白在两侧使用 `<scenetrans>`，只在片尾截断时使用 `<cutoff>`；Ref2VA 严格区分 Subject、Picture、Video 的实际角色，普通视频内声音不会被伪装成独立 Audio 素材。

## MiniMax 官方场景 Skill（8 个可选）

| 预设 | 主要强化内容 |
| --- | --- |
| `无（仅核心规则）` | 不附加场景风格规则 |
| `AUTO（根据意图判断）` | 意图明确时最多选择一个匹配预设，否则只用核心规则 |
| `极简产品广告` | 产品身份、材质、负空间、单节拍主动作、稳定闭幕和品牌事实安全 |
| `3D 动画短片` | 角色/场景连续性、可读轮廓、动作预备与跟随、单镜重要角色控制 |
| `品牌宣传短片` | 可核验品牌事实、精确文案、安全空间和具体功能证明 |
| `音乐 MV 动态字幕（官方）` | MiniMax 官方 `music-video-subtitle-generator` v0.6.6：锁定歌词、条件式口型/表演、空间文字、文本已知节拍和人物/场景/文字参考隔离；明确授权时可创作适配当前时长的短篇原创歌词，但不假装分析音频附件 |
| `双人合作游戏开场` | 双人身份与左右位置、玩家名、UI 文案、按钮层级和颜色控制 |
| `纸拼贴讲解` | 半调纸片、视觉隐喻、停格组装动作和触感音效 |
| `立体纸艺停格讲解` | 分层纸景、折叠/弹起/翻页/纸偶动作、材料与景深连续性 |
| `手绘实拍融合` | 实拍接触、同一实体连续变形、慢半拍追拍、粗糙发光笔触和非恐怖基调 |

这些选项只把官方场景 Skill 中适合“单次 H3 提示词改写”的规则传给 LLM。节点不会安装远程 Skill、生成角色卡或锚点图、研究官网、分析音频文件、调用 H3 视频生成、拼接片段或执行交付流程。

如果同时选择了一个 `非官方模板（案例 / 社区 Skill）`，则 **T8 非官方模板优先**：本次请求不会再应用或 AUTO 推断这 8 个可选官方场景 Skill。界面会把官方场景 Skill 标为“当前停用”并隐藏其详情卡；取消 T8 模板后，工作流中原先保存的官方场景 Skill 会自动恢复。该优先级不影响始终启用的 H3 核心写作 Skill，也不会改变旧工作流的节点 ID、输入或保存值。

## 非官方模板（案例 / 社区 Skill）

该列表与节点已有的 MiniMax 官方 9 个 Skill 完全分开，绝不修改或重复导入官方模板。下拉框显示中文名称，工作流内部保存稳定模板 ID；旧名称会自动迁移。MiniMax H3 会把机制写进 H3 原生字段、`[Shot N]` 时间线和声音合同；Seedance 2.0 只使用它自己的任务意图、自然段或连续 `镜头N` 语法，不会混入 H3 字段或逐镜头绝对时间码。

| 模板 | 可复用机制 |
| --- | --- |
| `产品广告｜功能证据递进` | 先给结果，再用至少三个可见证明状态逐层回答产品问题，最后收束到行动 |
| `固定机位｜从线稿生成成品` | 锁定构图，让同一对象经过至少四个不可逆制作阶段并轻微活化 |
| `形态奇观｜平面升级到现实` | 同一载体从平面、空间、体积升级到环境峰值，再回到初始锚点 |
| `人物旅程｜从困境走向目标` | 明确起点困境与目标，经过连续物理过渡并在抵达后完成目标动作 |
| `二维角色｜真人接触三级反应` | 一次跨媒介接触触发三级反应，由平面媒介自身完成结尾 |
| `一对多反转｜单一证据胜出` | 多数方案先制造冲突，克制的单一信号通过物理结果完成反转 |
| `风险揭晓｜证据交给人物反应` | 干预制造冲突，可视证据证明风险，再以有物理动机的镜头交接落到反应 |
| `未来证据｜异常闯入现实` | 同一证据载体连续更新，异常先出现在证据内，延迟后跨入稳定现实 |
| `第一视角探险｜穿越后巡检` | 清楚穿越阈值后保持空间连续，只检查一个局部线索且不虚构诊断 |
| `复古手持｜日常送别回眸` | 观察者媒介记录日常动作，外部信号改变目标，以一次回应和有动机失误收尾 |
| `反差喜剧｜越从容越失控` | 附着主体的负担触发至少两次因果连锁，主体始终沉着 |
| `人物档案｜能力逐层点亮` | 固定档案框中按身份、诊断、能力、工具和待命状态逐层激活 |
| `画中物成真｜小光点变奇观` | 被创作标记先自主活动，再越过实体边界、扩张并改变环境 |
| `工艺过程｜材料覆盖当进度` | 同一工艺物由一个只增不减的材料变量经过至少四个阶段完成 |
| `角色登场｜细节到全身揭晓` | 从可识别细节逐步扩大到面部、一次代表动作和全身身份定格 |
| `机械同行｜启动后稳定伴行` | 可见启动达到阈值后负载才移动，并由连续镜头证明稳定伴行 |
| `平面重组｜几何节奏品牌片` | 单一几何母题反复拆解重组，高密阶段后留短暂停顿并用既有图形收束 |
| `景别收紧｜从世界到眼神` | 景别从宏大环境持续收紧，证据由环境转到局部、装置和近景反应 |
| `角色卡验真｜四类证据锁定角色` | 固定角色卡反复校验姿态、操作、步态和面部四类不同证据 |
| `双系统碰撞｜同时证明两种规则` | 两套不同规则先独立成立，只汇聚一次并形成同时保留双方证据的第三状态 |
| `单目标逃生｜前后灾害交替逼近` | 单一终点持续可读，前后危险交替，完整恢复后用近失结果证明能力 |
| `对决节奏｜静止密战一击定局` | 安静建立、连续密集交换、一次决定性干预、假稳定和延迟结果 |
| `路线坍塌｜安全空间逐段消失` | 方向与终点固定，后方路线只减不增，主体完全过阈值后最后连接才失效 |
| `程序错位｜扫描到实物双确认` | 超尺度主体经历不同程序失配，扫描先揭晓，随后由同一实物完成第二次确认 |
| `技能展示｜基础动作串联升级` | 每项递增技能都从基础循环出发并返回，每个机位只承担一个证明任务 |
| `微缩介入｜外部物只进入一次` | 静态标准物先校准尺度，微缩日常稳定后由另一个外部执行物单次介入 |
| `第一视角查岗｜遮挡失效到摆烂收口` | 第一人称持续推进，每次遮挡失败都暴露更深证据，最终主动停止遮挡 |
| `雨夜追逐｜街巷近战到机车脱身` | 建立追逐方向，以一次近身阻断改变节奏，再交接到更快载具完成脱身 |
| `单人表演弧｜坐起前倾再释放` | 固定直视镜头中用姿态、手势和释放动作完成完整单人表演弧 |
| `多功能救援线｜跨裂隙送补给再撤离` | 同一绳线承担多种路线职责，接触被救者后由一人带领共同离开 |
| `微缩逃亡｜巨物追赶到资源补给` | 巨物校准微缩尺度，跨材质改变动作动词，最终取得并使用资源 |
| `气闸异境｜物理异常后空间崩坏` | 密闭空间先完成普通操作，再积累异常并开向不可能外部，由遗留物证明连续 |
| `异质航行｜维修出发到巨尺度揭示` | 出发前先证明载具与成员就绪，再交替展示局部操控和广域速度，最终用巨尺度揭示重构旅程大小 |
| `角色登场｜失误后借身体特征破局` | 先建立角色生活与移动方式，技术任务发生可恢复失误后，以身份特征作为工具完成解决 |
| `双角色职责｜独立建立后共同完成` | 两个角色先各自完成可见职责，再在共享接口形成缺一不可的共同成果 |
| `冲击后揭示｜从私密困境到城市尺度真因` | 狭小空间先建立私密困境，巨大外力制造局部后果，最后用城市广角解释真因 |
| `分层接力｜小物体穿越多区域完成传递` | 同一小物体沿相连区域改道传递，见证者持续追踪，最终由全景闭合完整接力 |
| `尺度奇观阶梯｜每镜重置参照再逼近` | 每段先重置熟悉尺度参照，再让不可能形态逐级逼近到近距离遭遇 |
| `背景异常｜先被观众发现再由记录设备证实` | 异常先在背景出现，记录设备把它隔离成证据，人物最后才产生延迟反应 |
| `非对称突破｜密集阻力到空间制度改变` | 单一非对称力量突破密集阻力，局部与广角共同证明推进，并改变空间规则 |
| `表面生长｜沿既有几何扩散并回收到手作载体` | 彩色线条沿真实几何生长并连接多个载体，最终回收到可物理关闭的纪念物 |
| `步速到抵达｜目的地延迟出现并由环境庆祝` | 旅人从步行加速到奔跑，目的地延迟揭示，并由环境在接近时触发庆祝 |
| `动作成字｜方向笔画驱动角色切换与终态标记` | 第一位角色生成方向笔画，第二位角色沿同一图形轴接力并完成终态标记 |
| `连续推进｜重复地标计时并抵达更深通道` | 受限通道持续推进，以重复地标和视差计时，仅在末段跨入更深次级通道 |
| `双人递进｜共享接触从地面升到空中峰值` | 两个主体先建立距离差，再让同一次共享接触由低位递进到唯一空中峰值，最后落地保持关系终态 |
| `数据折叠｜总量下钻到当前值` | 超大总量先建立尺度，展开分类明细作为证据，再折回聚合结果并切换到当前状态值 |
| `手势对齐｜前景动作触发后景状态` | 固定前后景先给基准状态，前景动作完成清楚对齐并只触发一次后景状态变化，最后确认结果持续 |
| `轮廓接力｜闪光换形后完成攻势闭环` | 先建立清楚动作轮廓和落地姿态，用一次闪光完成换形交接，再以效果动作、位移和落地终态证明新形态 |
| `多人攻势接力｜独立入场后同步收束` | 多名角色沿独立路线依次完成职责证明，待角色关系清楚后再同步投入并收束 |
| `微表情问候｜侧身发现到笑容回稳` | 在近乎锁定的人像镜头中，用侧身、直视、垂眼、闭眼笑容和回稳完成微表情弧线 |
| `地貌变形揭示｜局部异常到巨体逼近` | 先让局部异常仍像地貌，再逐步揭示身份、完整几何和尺度影响，最终回到低位完成逼近 |
| `表演让位于环境｜近脸到全景的连续退镜` | 表演不中断，单次连续退镜把视觉主导权从面部交给身体，再交给响应中的环境 |
| `连续跟随｜步态与弯曲地标共同计时` | 在弯曲受限通道中持续后方跟随，以稳定步态、重复地标和消失点变化证明前进深度 |
| `分组对话覆盖｜两边接力后全桌合流` | 第一组与第二组依次完成可见对话轮次，最后通过共享桌面锚点和宽构图重新合流 |
| `群演聚焦接力｜全场到主位再回合奏` | 从完整群体建立空间，隔离主位完成证明段，再重新引入共同主位并以群体同步收束 |
| `载具启程｜微观检查累积到离场证明` | 先建立载具与操作者，以多个局部检查逐项证明准备状态，再通过明确激活和远景离场完成系统兑现 |
| `巨体攻防｜共享空间中的连续闪避` | 巨体用慢重动作占据空间，小体型主体连续穿过负空间，始终保持双方地理关系与未决对峙 |
| `表情校准｜固定身份下的动作单元序列` | 锁定身份、光线和视角，让局部面部动作单元分段变化并复位，最后收紧景别校验稳定性 |
| `踏面异变｜重复触点到基底破裂` | 重复触点先建立局部表面反应规则，一次更强接触使基底破裂并进入新的下层介质状态 |
| `移动兑现｜载具跃迁到生活结果` | 把载具身份、出发、性能跃迁和地理抵达串成因果链，最终用具体生活结果兑现技术奇观 |
| `类型预告｜同一主角跨场景升级到片名` | 同一主角和少数关系锚点贯穿多场景升级，从私人压力走向公共危险与大尺度奇观，片名卡封口 |
| `静态伏击｜从被动锚点升级到异能出口` | 长停留先固定主角与地点，威胁逐次压缩空间，最后只用一次超常出口改变冲突状态 |
| `环境前沿｜局部点燃到全域占据` | 同一地理中的可追踪前沿从局部起点连续扩张、跨越地标，最终由广角证明全场改变 |
| `尺度球局｜共享球体串联反差挑战` | 一个持续可见球体串联尺度反差对手、固定目标与动作阶段，以双方可见反应结束而不臆测得分 |
| `连续变线｜逐层防守到垂直终结` | 固定球场、球与篮筐关系，连续穿过多层可见防守门并逐次改变路线，最后只用一次垂直篮筐动作收束 |
| `技艺递进｜原料变形到成品特写` | 从单一原料开始，以功能不同且难度递增的手部操作改变形态，经一次不可逆热处理后用成品微距证明结果 |
| `材质引路｜空间生长到版式定格` | 同一连续材质从微小锚点引导运镜、生成实体并扩展成环境，最后一次升高压成保留历史关系的平面版式 |
| `世界悬停｜持续靠近到末端复流` | 复杂环境整体冻结而观察者连续穿行，以多个冻结地标持续证明悬停，片尾只恢复一次全局运动 |
| `动态承载｜平衡升级到失败落点` | 先建立稳定穿越能力，再进入动态承载面并持续证明平衡，最后用一次转移失败、落点与本人反应收束 |
| `屏中蒙太奇｜外层连续内屏切换` | 实体显示设备与外部现场保持同一连续镜头，内屏累积至少三种状态，结尾让内屏结果、设备载体和外部语境同时可读 |
| `逐物改写｜局部对象到整片环境` | 从一个局部对象开始逐项传播可见变化，保持改写规则连续，最终把局部机制扩展为整片环境的新状态 |
| `尺度展开｜微小材质扩张后原样回收` | 从微小材质锚点连续扩张到大尺度空间结果，再沿同一材质关系精确回收至原始尺度与状态 |
| `图形复位｜身份卡与动作证明交替` | 先完成紧凑身份激活，让固定图形身份卡与短促动作证明交替出现，后段提高动作难度并把峰值动作轮廓收回完整主体锁定图 |
| `计划对照｜固定分镜条见证执行升级` | 让持续可读的计划板与独立执行区同框，以按计划累积的现场变化证明执行推进，并让终态与计划末项闭合 |
| `日程闭环｜重复锚点串联整日任务` | 用稳定身份与随身物锚点串联准备、通勤、目的地任务和返回休息，在高密度切换中保持整日因果顺序 |
| `镜头交接｜观察者跟随转主体自拍` | 外部观察镜头先完成跟随、地点交互和环境证明，再经有动机的连续桥只交接一次拍摄权给主体 |
| `感官封闭｜人群静止点到异地恢复` | 用稳定主体与独立流动环境建立压力，在一次完整感官封闭后只发生一次全画幅换境，并以同身份恢复动作证明新空间持续 |
| `双支座弹性偶｜控制器驱动形变复位` | 两个可见支座分别绑定图形角色，依次验证共享响应、独立响应和第三种关系测试，最后稳定复位为可辨双角色 |
| `光学搜索｜远景线索递进到细节确认` | 在同一受限光学界面内从远景地点线索逐层收紧到中景主体、近景身份与极近景局部证据 |
| `画内涂鸦｜现实互动逐步绘出隐藏世界` | 让现实互动逐次生成锚定在表面的同族线稿，从局部符号、路线与新图形升级到环境范围的连通绘制世界 |
| `造型轮播｜身份锁定跨风格世界` | 固定主体身份与镜头关系，让背景、材质、配色和配件成套切换并逐步增密，最后回到可读身份终态 |

652 条来源案例中有 305 个案例 selector 和 347 个 evidence variant，当前没有 pending case；证据变体只增加同一机制的 GIF 与证据，不重复增加下拉。`batch-2026-08-17-01` 新增 `身份机体跃迁`、`独演成敌`、`样本倍增` 3 个稳定 selector，并为 `固定机位｜从线稿生成成品` 增加 2 个 evidence variant、为 `造型轮播｜身份锁定跨风格世界` 增加 1 个 evidence variant；`batch-2026-08-17-02` 新增 `规则闯关｜吸收能力后从平面赛道升维`；`batch-2026-08-20-01` 新增 `单次决策兑现｜准备—执行—后果—回归`、`延迟载体释放｜进入—静置—开启—扩散`、`画外定位悬念｜反应递进但不揭示来源`；`batch-2026-08-21-01` 新增 `关系重排｜群体建立—双人收紧—前后站位`、`见证释放｜封闭近景—外部尺度—双视图证明`、`信息接力｜稳定载体穿越多种实体媒介`；`batch-2026-08-22-01` 新增 `价值阶梯｜原料—转化—装配—使用回报`、`能力见证｜装备特写—连贯表演—人群反应—挑战交接`、`公共兑现｜席间期待—揭晓—群体释放—登台锁定`；`errata-2026-08-24-01` 以一份累计快照补齐 8 月 23 日已批准成果：新增 66 个稳定 selector，并为 8 个既有 selector 补充 10 个 evidence variant；`errata-2026-08-24-02` 新增作者明确授权分发的 `指尖控制｜四向同拍全身响应` selector；`batch-2026-08-25-01` 新增 `时尚密度复位｜身份锁定、拼贴回顾与英雄主标`；`batch-2026-08-25-02` 新增 `手势边界换媒｜局部窗口、多形态验真与回归`；`batch-2026-08-25-03` 为该模板补充 1 个同机制 evidence variant，不新增 selector；`batch-2026-08-25-04` 新增 14 个稳定 selector，并把 6 个同机制证据变体归并到既有/同批模板，不创建重复下拉；`batch-2026-08-26-01` 新增 10 个稳定 selector，并把 10 个同机制证据变体归并到既有/同批模板；`batch-2026-08-26-02` 新增 1 个稳定 selector，不新增 evidence variant；`batch-2026-08-27-01` 为 9 个既有模板补充 15 个同机制 evidence variant，不增加重复下拉；`batch-2026-08-28-01` 再为 16 个既有模板补充 20 个 evidence variant；`batch-2026-08-28-02` 又为 19 个既有模板补充 20 个 evidence variant；`batch-2026-08-29-01` 再为 19 个既有模板补充 20 个 evidence variant；`batch-2026-08-30-01` 累计快照再补齐 40 个同机制 evidence variant，其中 20 个为本批明确增量，均不增加重复下拉；`batch-2026-08-31-01` 新增 `路线物品累积｜终点回返` 1 个稳定 selector，并把其余 19 条同机制证据变体归并到既有模板；`batch-2026-09-01-02` 新增 7 个稳定 selector，并把本批其余 13 条以及累计快照补齐的 20 条同机制证据变体归并到既有/同批模板；`batch-2026-09-02-01` 再新增 7 个稳定 selector，并把其余 13 条同机制证据变体归并到既有/同批模板；`batch-2026-09-03-01` 新增 6 个稳定 selector，并把其余 14 条同机制证据变体归并到既有/同批模板；`batch-2026-09-04-01` 新增 `错过阈值转向｜急行路线、离站证据与新方向`、`产品表面交替｜整体基线、微距材质与英雄复归`、`物件仪式增幅｜交付基线、环形锁定与能量释放` 3 个稳定 selector，并把其余 17 条同机制证据变体归并到既有模板。`batch-2026-09-08-01` 新增 `意象载体预告接力｜花果传递、角色轮换与巨像回标`、`情绪物证清空｜低谷表情、旧物处置与价值回报`、`界面恶作剧实体化｜屏内内容、屏外清除与手写回执` 3 个稳定 selector，并把其余 17 条同机制证据变体归并到既有模板。`batch-2026-09-09-01` 为 15 个既有模板补充 20 个同机制 evidence variant，不增加重复下拉。`batch-2026-09-12-01` 为 17 个既有模板补充 19 个同机制 evidence variant，不增加重复下拉。`batch-2026-09-13-01` 为 15 个既有模板补充 19 个同机制 evidence variant，不增加重复下拉。`batch-2026-09-21-01` 再新增 20 个稳定案例 selector，不增加 evidence variant；来源批次见 `case_templates/source_batches/2026-09-21-01.json`。`batch-2026-09-22-01` 新增 13 个稳定案例 selector，并把其余 7 条归并为同机制 evidence variant；来源批次见 `case_templates/source_batches/2026-09-22-01.json`。旧工作流中的 `声画错位递进` / `t8-case-audio-cause-lead-ladder-v1` 会迁移到证据支持的 `景别收紧｜从世界到眼神`。

`batch-2026-10-15-02` 是 `batch-2026-10-15-01` 的完成补充，不能当作第二批 20 例重复相加。相对节点此前库存，新增 15 个案例 selector、5 个同机制 evidence variant；累计为 633 条案例、288 个案例 selector、345 个 evidence variant，加 2 个社区 Skill 共 290 个非官方选项，635 个 T8 预览。累计快照同时刷新 9 月 22 日 20 例的时间字串勘误与适配哈希绑定，既有模板的名称、机制、推荐输入及结构锚点不变。

`batch-2026-10-15-03` 新增 19 条来源案例：17 个新案例 selector、2 个既有模板的同机制 evidence variant；另 1 条完全重复的视频已关闭，不重复导入。累计为 652 条案例、305 个案例 selector、347 个 evidence variant，加 2 个社区 Skill 共 307 个非官方选项、654 个 T8 预览。动态预览通道为 `2026.10.16.1`，既有模板名称、机制和工作流标识保持不变。

另外两个独立的用户贡献社区 Skill 不合并进案例 registry，也不冒充官方 Skill：

| 社区 Skill | 可复用机制 |
| --- | --- |
| `自然街拍互动｜边走边聊到清楚落点` | 连续路线跟拍中用距离、步态、视线、手势、视差与现场声推进到可读互动结果 |
| `突遇惊吓到和解｜贴近反转与求和手势` | 平静基线、一次非致命惊慌、距离反转和被看见的求和手势组成关系反转 |

打开 `MiniMax 官方场景 Skill（8 个可选）` 或 `非官方模板（案例 / 社区 Skill）` 下拉时，鼠标悬停任一具体选项会在菜单右侧即时显示对应 GIF、用途和简约推荐输入；菜单靠近窗口右边时预览会自动移到左侧。键盘上下选择和搜索过滤也会同步预览。选择任一具体官方场景 Skill 或非官方模板后，节点内都会显示用途、适用/推荐输入格式、必须实现的结构锚点、推荐示例、GIF 和来源链接。`填入推荐示例` 只有在主提示词为空时才写入；已有输入会显示“已有输入，未覆盖”。推荐示例只是可编辑的实例意图，不是最终提示词。即使只输入“美丽的女人”，节点也会保留这个主体，并要求 LLM 原创建立模板需要的场景、触发、事件链和可见结果，不能退化成普通人像运镜。

GIF 只供浏览器中的人类界面预览，绝不会连接到首帧、尾帧、参考图片、参考视频，也不会发送给 LLM。8 个官方预设 GIF 继续内置；主仓库保留此前的 377 个 T8 轻量 GIF 作为离线基线，当前全部 654 个 T8 案例/社区 Skill 预览统一通过独立的 [`comfyui-minimax-h3-prompt-enhancer-T8-assets`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8-assets) Release 动态更新。节点会校验安装目录对应的案例全集、来源 SHA-256、分片 SHA-256、每个 GIF 的大小/哈希/文件头和安全解包路径；不兼容或损坏的资源不会显示，也不会影响提示词增强。

H3 与 Seedance 2.0 节点都提供非序列化按钮 `管理 / 更新 T8 动态预览`：可选择“智能按需（推荐）”“自动补齐全部”或“仅手动下载”，并可检查更新、下载缺失、校验修复或清空缓存。这些设置保存在 ComfyUI 用户目录，不写入工作流，所以旧工作流结构、节点输入输出和 API Key 均不会改变。离线时已缓存/内置 GIF 继续显示；未缓存 GIF 只显示提示，提示词生成仍正常运行。

维护者仍可选用本地原始案例清单，以显示来源链接或在开发环境优先检查原始预览；普通用户不需要此配置：

```powershell
Copy-Item .t8-case-library.example.json .t8-case-library.local.json
```

`.t8-case-library.local.json` 已加入 `.gitignore`。也可以分别用环境变量 `T8_UNOFFICIAL_CASE_LIBRARY_V2`、`T8_STANDALONE_COMMUNITY_SKILLS_V1` 指定两个 manifest；若社区 manifest 与案例 manifest 位于同一目录，后端会自动发现它。本地清单缺失、路径失效或校验失败时，节点会自动回退到仓库内置 GIF。

默认 `无（不使用 T8 案例）`，因此旧工作流请求保持不变。案例独立于“官方增强 / 官方优化”和“参考模板融合”，用户提示词、媒体事实、硬性要求、时长与固定镜头数始终优先。节点把所有结构锚点作为一次请求内的硬写作合同传入上游，并要求静默检查；按此前的上游兼容策略，非空响应仍直接放行，不会因本地语义猜测误杀有效输出。

### 维护者：标准化每日导入

日常批次先由 curator 生成并独立复审 `unofficial-case-library-v2.json`，再用 [`tools/import_unofficial_case_library_v2.py`](./tools/import_unofficial_case_library_v2.py) 同步人类名称、推荐输入、结构锚点、证据变体和双模型 Creative DNA：

```powershell
python tools/import_unofficial_case_library_v2.py `
  --library <handoff-dir>\unofficial-case-library-v2.json `
  --community-skills <handoff-dir>\standalone-community-skills-v1.json `
  --existing-catalog case_templates\catalog.json `
  --source-batch-dir case_templates\source_batches `
  --output case_templates/catalog.json
```

每次案例目录发生变化，都必须先生成完整预览候选做覆盖率、编码预算和哈希验收，但不再把新增 T8 GIF 提交到主节点仓库；主仓库中的 377 份旧预览是冻结的离线基线。验收通过后，用资产仓库工具生成新的、不可变的 Release 分片和匹配的 `channel.json`，再只把该通道快照带入节点版本。Registry 同样只分发通道清单，不分发 T8 GIF。原生成器要求输出目录不存在或为空；建议生成到仓库外的待审目录：

```powershell
python tools/bundle_t8_case_previews.py `
  --library <handoff-dir>\unofficial-case-library-v2.json `
  --community-skills <handoff-dir>\standalone-community-skills-v1.json `
  --catalog case_templates\catalog.json `
  --output-dir web\js\assets\t8-case-previews-next `
  --ffmpeg <ffmpeg-dir>\bin\ffmpeg.exe
```

导入器按每次累计库声明的库存做严格闭合校验；本次为 652 条案例记录、305 个 selector、347 个 evidence variant、0 个 pending case，并另验 2 个独立社区 Skill，最终生成 307 个非官方下拉项。已发布案例必须 `released/approved`；H3 与 Seedance 2.0 适配器必须同为可执行的无媒体 recipe，或同为已验证的 disconnected direct-final 交接，两种格式都必须通过哈希、`media_connections=[]` 与 secret 边界检查。可执行 recipe 的 provider/secret 字段必须为空、不得含旧 `openai_upload_url`，双模型 Creative DNA 必须一致；direct-final 的成品提示词只用于验证，节点目录仅从已校验的 `creative-dna.json` 提取可复用机制、结构锚点、可替换槽位和失败修复，不导入成品提示词。若以后再次出现 pending case，必须保持未发布、未复审、有明确 blocker、无 adapter，并且不能进入下拉或预览分发。社区 Skill 必须保持非官方/用户贡献、不得合并进案例 registry，并现场重算 Skill、摘要、双模型 guidance 与 GIF 哈希。提示词目录不会写入本地路径、来源 URL、来源视频或成品提示词；分发包只保存作者授权的轻量 GIF、不可逆哈希与编码参数。Markdown 报告只是说明，不能代替机器 JSON、实时 recipe、内置预览覆盖率与哈希核验。

### 音乐 MV 动态字幕（官方）填写方式

选择 `音乐 MV 动态字幕（官方）` 后，主提示词会显示一份动态填写提示；它只是界面占位文字，不会写入工作流或自动成为提示词。旧工作流中保存的 `MV / 歌词贴字` 会自动迁移到这个正式名称。可以直接按下面的结构填写，未涉及的项目留空：

```text
MV类型/音乐类型/视觉风格：暗色抒情 MV，低饱和蓝黑色，空间歌词排版
歌词原文（逐字锁定，可空）：夜色落在我肩上，别回头。
无歌词时：器乐 / 允许生成原创歌词
演唱者或离屏人声：画内女歌手演唱
已知 BPM、歌词时间点或节拍事件（可空，节点不分析音频）：00:04.200 鼓点进入，00:09.000 drop
目标平台/画幅（可空）：抖音竖屏，9:16
字体包装与禁止项：文字在中景展开，不遮挡眼睛和关键口型
```

MV 规则按以下方式工作：

- 用户提供的歌词逐字锁定，保持原语言、标点、顺序和重复，不翻译、不润色、不补写；只有未提供歌词且明确写出“允许生成原创歌词”时，才会创作一段能放进当前目标时长的原创歌词，并将同一文本用于演唱与可见文字。否则不会自动创建歌词、歌手或口型。
- 只有提示词或可见参考素材明确存在画内演唱者时，才补充嘴唇、下颌、呼吸、表情和手势；器乐、纯 Typography 与离屏人声 MV 不强制出现歌手。
- 歌词文字作为前景、中景或背景的空间图形层，同一时刻保持一个主阅读焦点；可轻微穿插遮挡，但不会遮挡眼睛、主要表情和关键口型。
- 只有文本明确给出 BPM、时间码、drop、snare、808、歌词重音或音乐段落时，才进行精确卡点。节点没有 AUDIO 输入，也不会声称听过歌曲或参考视频声音。
- Trap / Dark-pop / Cyber-grunge 才会条件化使用硬切、glitch、扫描错位、颗粒和 zine 拼贴；抒情、氛围或原声 MV 不会被统一套用硬切风格。
- `strict / balanced / creative` 控制扩写幅度；`官方 Skill 协议`控制正文语言和格式。即使选择全英文严格协议，用户原始歌词仍保留原语言。

多张参考图建议在高级选项的 `reference_context` 中做角色隔离：

```text
<Picture 1>=人物外观；
<Picture 2>=场景与灯光；
<Picture 3>=字体包装，只参考字体、版式和动效，不参考人物与场景。
```

字体参考图中的示例文字、人物和场景不会自动复制。参考视频只提供可见表演、运镜、剪辑节奏和画面结构，不会被当作独立音频、BPM 或歌词时间轴。

使用“参考模板融合”时，模板只贡献镜头组织、节奏、运镜、转场和视觉语法；模板中的人物、歌词、BPM、标题、剧情和镜头数不能覆盖基础提示词、真实媒体、硬性要求或固定镜头数量。


## 改写模式

| 模式 | 温度 | 行为 |
| --- | ---: | --- |
| `strict` | `0.2` | 尽量保持用户原意和媒体事实，只补必要格式与连续性 |
| `balanced` | `0.7` | 在保真基础上补充合理镜头、灯光、动作、环境声和节奏 |
| `creative` | `1.2` | 加强视觉风格、镜头设计、动作连接、声音层次和配乐 |

温度映射是本节点的产品设置，不代表供应商官方推荐值。

## 主要输入

| 输入 | 说明 |
| --- | --- |
| `prompt` | 用户基础视频意图，不能为空 |
| `task_type` | T2VA / I2VA / FL2VA / L2VA / Ref2VA / Hybrid |
| `duration_seconds` | 目标时长，任意正整数秒数，节点不设上限；下游模型限制仍然有效 |
| `shot_count` | `AUTO` 由模型按内容、素材、时长和节奏判断；也可固定为 1–20 个镜头 |
| `rewrite_mode` | `strict / balanced / creative` |
| `description_word_target` | `0` 为自动；非零至少 80，已取消原有 1000 上限，可指定 1500、2000、5000 等目标。中文按约数汉字、英文按单词理解；实际输出仍受模型及 Token 预算影响 |
| `output_language` | `中文 / English`，默认中文 |
| `prompt_mode` | `官方增强 / 参考模板融合` |
| `official_skill_profile` | 界面显示为“H3 核心写作 Skill（始终启用）”；`现有兼容 / 官方 Skill 严格`，默认兼容，严格档位强制英文说明正文 |
| `creative_preset` | 界面显示为“MiniMax 官方场景 Skill（8 个可选）”；`无 / AUTO / 8 个官方场景写作预设`，选择具体项后显示详情卡和官方 GIF；与 T8 非官方模板同时选择时暂不生效 |
| `case_template` | `无 / 307 个非官方模板`：305 个 T8 案例 selector + 2 个独立社区 Skill；显示中文名称，工作流保存稳定 ID，H3 与 Seedance 2.0 使用独立原生适配规则；H3 中与官方场景 Skill 同选时优先生效 |
| `reference_template` | 仅参考模板融合使用 |
| `first_frame` | I2VA / FL2VA / Hybrid 首帧 |
| `last_frame` | FL2VA / L2VA / Hybrid 尾帧 |
| `reference_images` | Ref2VA / Hybrid 额外参考图，Autogrow，最多 9 张；Hybrid 每槽单图 |
| `reference_videos` | Ref2VA / Hybrid 参考视频，Autogrow，最多 3 个 |
| `reference_context` | 高级可选；补充画面无法判断的身份或关系 |
| `constraints` | 高级可选；必须保留或禁止改变的内容 |
| `seed` | 随机种子，配合运行后状态控制缓存和变体 |

`reference_context` 和 `constraints` 默认折叠。正常使用不需要填写，避免把简单任务变成大量表单输入。

## 镜头数量

- `AUTO（系统自动判断）`：结合基础提示词、参考素材、目标时长、动作密度和节奏决定镜头数；没有必要切镜时优先使用单镜头内运镜。
- `1–20`：要求 LLM 在时间线中使用恰好对应数量的 `[Shot N]`，连续编号，并让后续镜头的时间码严格递增且小于目标时长。
- 固定数量优先于基础提示词或参考模板中的模糊镜头数量范围。选择 `AUTO` 时，基础提示词中的镜头要求仍会参与模型判断。

镜头数量是传给上游模型的明确生成约束，不是本地响应验收条件；只要上游返回非空内容，节点仍按“输出与错误行为”中的规则放行。

## 随机种子

种子使用 ComfyUI 原生运行后控制：

- `randomize`：每次运行前随机更换种子。
- `fixed`：保持当前种子，输入相同时允许使用 ComfyUI 缓存。
- `increment`：每次运行递增。
- `decrement`：每次运行递减。

Seedance 没有为本节点调用的 Chat Completions 公开确定性 `seed` 请求参数，因此节点不会发送未声明的 API 字段。种子会作为提示词变体标识并参与 ComfyUI 缓存，但不保证强制重跑后逐字复现。

## API Key 与接口模式

三个节点中的 Key 都用于当前所选云端 LLM 渠道的提示词增强请求，不是下游 Seedance 2.0 视频生成或 Music 3 音频生成 API Key；本地 Qwen 渠道完全不需要 Key。

### 贞贞平价小屋（推荐）

聊天与素材上传地址固定为：

```text
https://api.seedance.nz
```

可以把任意 `STRING` 输出连接到节点的 API Key 接口，也可以在节点底部填写 Key，或点击[获取贞贞 API Key](https://api.seedance.nz/sign-up?aff=5f4w)。接线值优先；没有接线且节点中的 Key 留空时读取环境变量 `SEEDANCE_API_KEY`。

PowerShell：

```powershell
$env:SEEDANCE_API_KEY="你的 API Key"
```

Linux / macOS：

```bash
export SEEDANCE_API_KEY="你的 API Key"
```

### 贞贞的 AI 工坊（图片/视频）

聊天地址固定为：

```text
https://ai.t8star.org/v1/chat/completions
```

默认模型是 `gemini-3.5-flash`。选择 `Custom（自定义）` 后会显示“自定义模型 ID”，可填写 AI 工坊模型列表中的完整 ID。H3/Seedance 自定义模型必须自行确认具备图片与视频理解能力；Music 3 节点是纯文本请求，只要求所选模型支持 Chat Completions 文本输入。

可以点击节点底部的[获取 AI 工坊 API Key](https://ai.t8star.org/register?aff=dP7j)。节点中的 Key 留空时读取环境变量 `T8STAR_API_KEY`：

```powershell
$env:T8STAR_API_KEY="你的 API Key"
```

AI 工坊模式不需要上传 URL。图片和完整视频以 Base64 Data URL 放进同一次 Chat Completions 请求。2026-08-06 的真实协议探测确认：该网关对 `gemini-3.5-flash` 的视频 Data URL 必须使用 OpenAI 多模态 `image_url` 部件（Data URL 自身仍是 `video/mp4`）；`video_url` 虽会返回 200，但会丢失或误读视频事实。因此本节点使用实测通过的表示法，而不是照搬旧参考节点。

### OpenAI 兼容接口（备用）

备用模式只需一个 API Base URL 和模型 ID。Base URL 支持服务根地址、以 `/vN` 版本段结尾的地址（例如 `/v1` 或火山方舟的 `/api/v3`），或完整 `/chat/completions` 地址。节点 Key 留空时读取 `OPENAI_API_KEY`，Base URL 留空时读取 `OPENAI_BASE_URL`；H3/Seedance 需要供应商实际支持的视觉模型，Music 3 只需要文本模型。

三个核心节点会自动把 `OpenAI Base URL` 和 `OpenAI 模型 ID` 保存到当前工作流：输入时即时同步，点击节点运行按钮时还会在提交前再次读取尚未失焦的输入框；重新打开工作流或切换 API 模式后会自动恢复，无需每次重填。该兼容状态不包含 API Key，Key 仍按下方安全规则单独处理。旧工作流无需重连，首次保存后会补充这份兼容状态。

#### 修复后的节点配置

MiniMax H3 与 Seedance 2.0 两个节点使用相同的 OpenAI 兼容配置：

| 节点字段 | 是否必填 | 用法 |
| --- | --- | --- |
| `API Key` | 是 | 可连接任意 `STRING`，也可在节点内填写；节点留空时读取 `OPENAI_API_KEY` |
| `OpenAI 模型 ID` | 是 | 填写兼容服务商提供的完整视觉模型 ID，不再固定为 `bytedance/doubao-seed-evolving` |
| `OpenAI Base URL` | 是 | 只填写一个聊天接口地址；服务根地址补全为 `/v1/chat/completions`，已有 `/vN` 版本段则直接追加 `/chat/completions` |
| `视频素材 URL` | 否 | 仅用于明确支持 `video_url` 的渠道；每行一个，按已连接 `VIDEO` 顺序透传。未填写的已连接视频自动抽帧成图片 |

不再需要、也不要填写单独的“兼容素材上传 URL”。图片没有素材 URL 字段，始终由节点编码成 Base64 并随聊天请求直接发送。

该模式不再使用第二个素材上传端点：

- 图片统一编码为 PNG，通过 `image_url.url` 中的 `data:image/png;base64,...` 内联到同一次 Chat Completions 请求。
- 未填写“视频素材 URL”的已连接视频会按时间顺序抽帧，均匀保留最多 9 帧，以带时间戳的 JPEG `image_url` 发送；这兼容 llama.cpp 等只接受图片内容部件的视觉端点，且不会分析音轨。
- “视频素材 URL”可选，每行一个，按已连接 VIDEO 的顺序以 `video_url` 原样透传；只有供应商明确声明支持该内容部件时才填写。
- 视频 URL 数量不能超过已连接 VIDEO 数量，且必须是 HTTP(S) 地址。

OpenAI 官方 Chat Completions 明确定义了图片输入，但通用 `video_url` 并不是所有兼容供应商都支持的统一能力。默认抽帧路径可避免 llama.cpp/Qwen 视觉端点返回 `HTTP 400 unsupported content[].type`。如果抽帧在请求前失败，节点会直接给出可操作错误且不会发起可能计费的模型请求，也不会静默退回已知不兼容的完整视频格式。

### 本地 GGUF（llama.cpp / Qwen，离线推理）

本地渠道是三个现有节点的第 4 个互斥 provider，不会新增或替换节点，也不会改变 H3、Seedance 2.0、Music 3 各自的提示词合同。运行时不需要 API Key、Base URL 或云端模型 ID。节点不再把可用范围锁死为两个文件名：会递归扫描 `ComfyUI/models/LLM`，读取轻量 GGUF 元数据，区分主模型与 mmproj，并为视觉模型推荐同名/同目录投影器。

云平台可把 `models/LLM` 中的模型做成文件软链接以去重。链接目标可以位于目录外、使用相对地址，也可以只是没有 `.gguf` 后缀的 UUID；链接入口本身也允许是无后缀 UUID。无后缀条目只有在实际内容以 GGUF 魔数开头时才会出现在下拉列表，普通 UUID 文件、目录和失效链接会被忽略。节点保存并显示 `models/LLM` 内的相对条目标识，执行时再解析到真实文件，因此不会要求云平台暴露绝对存储地址。更新后请完整重启 ComfyUI，或点击“检查本地 Qwen 安装 / 扫描 GGUF”刷新目录。

GitHub 完整安装采用自动回退顺序：本节点固定安装器生成的 `llama-server` → 系统 `PATH` 中的 `llama-server` → 当前 ComfyUI Python 环境已经安装的 `llama-cpp-python`。Manager/Registry 安装为通过 Registry 自动安全审查，仅使用进程内 `llama-cpp-python`，不随包分发外部进程启动器。两种安装都会递归扫描同一个 `models/LLM`，状态窗口会显示实际命中的后端、来源和版本。

如果当前 ComfyUI Python 没有可用的 `llama-cpp-python`，三个核心节点都提供“获取 llama-cpp-python 预编译 Wheel”按钮，跳转到 [JamePeng 预编译 Releases](https://github.com/JamePeng/llama-cpp-python/releases)。必须选择与 **ComfyUI 实际 Python 版本、操作系统和 CUDA 版本**匹配的 Wheel，并用 ComfyUI 自己的 Python 安装，而不是系统 Python：

```powershell
& "你的 ComfyUI Python 路径\python.exe" -m pip install "下载到本地的 llama_cpp_python-....whl"
```

这是可选的第三方预编译来源，本节点不会静默下载或自动安装 Wheel。安装后请完整重启 ComfyUI，再点击“检查本地 Qwen 安装 / 扫描 GGUF”确认实际运行时；如果 Wheel 与 Python/CUDA 不匹配，状态窗口会明确显示导入或原生库加载错误。GitHub 完整安装还可以直接运行本仓库的 `install_local_qwen.py --runtime` 安装固定 `llama-server`，无需 Python Wheel；Manager 包不包含该下载/启动脚本。

模型和运行时体积很大，不会随 Git 仓库分发，也不会在执行节点时静默下载。请在节点目录显式运行：

```powershell
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py
```

只看安装计划：

```powershell
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --dry-run
```

只安装或核验其中一部分：

```powershell
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --runtime
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --model --model-variant official
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --model --model-variant uncensored
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --model --model-variant heretic-9b
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --model --model-variant all
& "你的 ComfyUI Python 路径\python.exe" install_local_qwen.py --offline
```

`official` 是默认模型；`uncensored` 是用户可选的第三方 27B FP8 衍生 Q4_K_M 量化；`heretic-9b` 是体积更小的第三方 9B i1-Q6_K 量化；`all` 安装三种文字模型及默认 27B 视觉投影器。9B 上游仓库没有发布匹配的 mmproj，因此固定安装器只安装其文字模型；纯文字 H3/Seedance、Music 3 可直接使用，图片/视频必须由用户另行提供兼容的 9B mmproj，交给 AUTO 匹配或手动选择。安装后重启 ComfyUI，或点击节点的“检查本地 Qwen 安装 / 扫描 GGUF”，下拉会重新列出 `models/LLM` 及任意子目录内的 GGUF。第三方模型不会替换现有 27B 默认值，也不会破坏旧工作流或绕过本项目的提示词合同、输入验证与输出保护。

安装器固定并核验以下资产，支持 `.part` 断点续传、磁盘空间预检和原子改名：

| 资产 | 固定版本 | 体积 |
| --- | --- | --- |
| 默认 Qwen 模型 | `Qwen3.8-27B-Q4_K_M.gguf`，固定 HF revision 与 SHA-256 | 约 15.93 GiB |
| 可选第三方模型 | `qwen3.8-27b-uncensored-fp8-q4_k_m.gguf`，固定 `theresa00l/...` revision 与 SHA-256 | 约 15.66 GiB |
| 可选轻量第三方模型 | `Qwen3.8-9B-heretic-uncensored.i1-Q6_K.gguf`，固定 `mradermacher/...` revision 与 SHA-256 | 约 6.85 GiB |
| 视觉投影器 | `mmproj-F16.gguf`，固定 SHA-256 | 约 0.86 GiB |
| 推理运行时 | llama.cpp `b10436`，固定安装器提交与 SHA-256 | 依平台/后端而定 |

GGUF 与 mmproj 统一放入 ComfyUI 的 `models/LLM/` 或任意子目录；历史目录 `models/LLM/Qwen3.8/` 和只保存文件名的旧工作流继续兼容。固定安装器的 llama.cpp 仍放在本节点已忽略的 `runtime/local_qwen/`，但也会复用当前 ComfyUI 已安装的 `llama-cpp-python`。默认模型来自 `unsloth/Qwen3.8-27B-GGUF`；27B 可选模型来自 `theresa00l/Qwen3.8-27B-Uncensored-FP8-Q4_K_M-GGUF`；轻量 9B 可选模型来自 `mradermacher/Qwen3.8-9B-heretic-uncensored-i1-GGUF`。9B 量化仓库标注其来源为 `rohit267/Qwen3.8-9B-heretic-uncensored`、语言为英语，但没有声明许可证；本项目只提供固定下载入口，不分发或重新授权模型文件。参考安装器、参考插件和 llama.cpp 为 MIT。具体固定提交、哈希及归属见 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。

任意被扫描到的文字 GGUF 都可交给当前 llama.cpp 运行时加载；“被发现”不等于“本项目已验收”。H3/Seedance 有图片或视频输入时必须有匹配 mmproj；AUTO 优先按 `general.name`、架构、同目录和文件名相似度匹配。当前 `llama-cpp-python` 回退路径会自动选择 Qwen2.5-VL、Qwen3-VL、Qwen3.5、Gemma 3/4 或 MTMD 处理器（取决于已安装版本实际提供的类）；不具备对应处理器的视觉架构会明确报兼容性错误，文字模式不受影响。

可选模型已用本项目固定的 `mmproj-F16.gguf` 和 llama.cpp 本地运行时完成真实兼容验收：文字精确 token、图片 OCR/颜色/形状、视频采样帧的双阶段 OCR、早晚运动方向与时间顺序共 12/12 项通过。它支持节点现有的“图片 + 按时间戳采样视频画面”路径；这仍不等于读取完整视频字节或分析音轨。脱敏证据见 [`tests/fixtures/local_qwen_uncensored_compatibility_2026-08-20.json`](./tests/fixtures/local_qwen_uncensored_compatibility_2026-08-20.json)。

轻量 9B Q6_K 模型也完成了真实 llama.cpp 兼容验收：文字精确 token、图片 OCR/颜色/形状、视频早晚代码、运动方向、时间顺序、单图单视频计数和“不分析音轨”共 12/12 项通过，耗时 22.078 秒。视觉测试使用用户另行放置的 `mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf`；该投影器不由本安装器下载或分发。脱敏证据见 [`tests/fixtures/local_qwen_heretic_9b_compatibility_2026-08-25.json`](./tests/fixtures/local_qwen_heretic_9b_compatibility_2026-08-25.json)。

2026-08-21 又对可选第三方模型执行了完整发布质量验收，确定性总分 `100/100`、`passed=true`，5 个用例全部通过：同 seed 三次逐字一致、H3 T2VA、Music 3 中文歌词与官方完整 Skill、H3 图像+视频证据、Seedance 2.0 图像+视频证据。视频用例同时核验早晚阶段代码、运动方向、硬切和首次出现顺序；全套耗时 476.938 秒，峰值显存 15971 MiB，卸载后回到 2007 MiB（基线 1613 MiB）。脱敏报告见 [`tests/fixtures/local_qwen_uncensored_quality_2026-08-21.json`](./tests/fixtures/local_qwen_uncensored_quality_2026-08-21.json)，仅保存分数、布尔检查、哈希、字符数和资源采样，不保存提示词、歌词、媒体、API Key 或临时令牌。

建议 24GB 以上显存获得较好的交互体验；16GB 显存会由独立 `llama-server` 的 `--fit` 自动部分卸载到系统内存，建议至少 32GB RAM，速度会明显下降。CPU-only 功能上可运行，但本项目不承诺交互速度。首次使用前可点击三个节点中的“检查本地 Qwen 安装 / 扫描 GGUF”查看模型数量、mmproj 配对、运行时后端和实际模型目录；旁边的路径按钮可直接复制 `ComfyUI/models/LLM` 绝对路径。

本地选项说明：

- H3/Seedance 的图片编码为 Base64 JPEG；原生 `VIDEO` 按真实时间戳采样，默认 2 fps，每 6–9 帧组成一个有时间标签的联系表，并按上下文预算最多发送 16 个视觉部件。
- 本地视频只分析可见画面顺序，不读取、转写或声称理解原声音轨。云端完整视频行为不变。
- Music 3 只发送文字，整个多阶段任务复用同一模型进程，并且绝不加载 `mmproj`。
- 三个节点共享互斥运行锁，避免同时抢占显存；独立 `llama-server` 等待与请求期间支持 ComfyUI 取消。`llama-cpp-python` 是兼容回退路径，其底层同步推理的中途取消能力取决于已安装绑定版本。
- 默认关闭思考以降低延迟；可在高级选项开启思考、选择推理强度、上下文、最大输出、视频采样率、显存释放及执行后卸载/驻留/10 分钟 TTL。
- 跨工作流使用 GGUF 时，`AUTO` 显存策略会在开始任务时重新检查宿主显存，包括复用驻留模型的情况；不足时按原策略释放 ComfyUI 模型。选择“保留 ComfyUI 模型”仍不自动释放。ComfyUI 的释放内存操作不等于卸载本节点的驻留 GGUF；交替运行大视频模型时，优先使用节点默认的“执行后卸载”。1.42.1 修复了驻留复用时漏掉该检查的问题，但尚未在 #25 所报的相同硬件和模型上复现完整卡死过程。
- `本地单次生成 Token（含思考）` 是模型思考过程和最终正文共享的生成上限，旧版 8192 是项目的保守限制，并非 Qwen/llama.cpp 硬上限。新建节点及仓库示例默认 `16384`，可手动设置 `256–61440`；已有工作流保留其原先保存值，需用户按需改大。输入提示词和用户明确连接的全部图片/视频优先占用上下文；当配置的生成上限会挤掉合法参考媒体时，节点会按本次实际输入自动下调有效生成上限，而不是把 Ref2VA 错报成只能上传一张图。若连最低 256 Token 输出也装不下全部媒体，错误会给出估算所需上下文；此时应将 `local_context_size` 从 `32768` 提高到 `65536`，或缩短模板/提示词、降低视频采样率。更大上下文会增加内存/显存占用。
- 运行时只监听 `127.0.0.1` 随机端口，使用随机临时令牌并关闭 Web UI；成功、异常、取消和 ComfyUI 退出都会执行相应清理。
- 本地与云端模型都会在完整官方 Skill 与案例之后收到最终语言锁，确保“输出语言”对 H3、Seedance 2.0 和 Music 3 Structured Caption 的说明正文生效。只有在完整响应已经收到、且本地检测到正文明显整体跑到另一种语言时，节点才追加一次低温“只纠正语言、不改内容”请求；已符合语言要求时不增加调用，网络状态未知时也绝不会借语言纠正名义重投。协议字段、时间码、参考标签以及用户原始对白、歌词、品牌/UI 文案和画面文字保持原样。H3 的“官方 Skill 严格（全英文协议）”仍按设计优先于中文选项。
- 本地输出继续遵守既有 content-first 规则：H3 仅在完整命中字段时重排，Seedance 非空放行，Music 3 保留歌词保护和官方 Caption 阶段合同。

## API Key 安全

三个节点都提供标准 `STRING` API Key 接口，并保留底部的遮罩、显示、保存和清空按钮。外部接线值优先。遮罩只能避免画布上直接显示明文：点击“保存到工作流”后，Key 会进入工作流 JSON。

- 分享工作流前务必点击“清空”。
- 更安全的方式是把节点 Key 留空并使用环境变量。
- 节点不会把 Key、请求正文、素材 URL 或响应正文写入日志。
- 仓库示例和测试不包含真实 API Key。

执行期间，三个节点会向 ComfyUI 进度条报告本地准备、模型请求和结果整理阶段。最近 50 次执行的脱敏阶段信息可从本机 ComfyUI 的 `/t8-prompt-enhancer/diagnostics` 查看；记录仅含节点类别、渠道名称、结果、阶段耗时、可用时的尝试次数、素材数量、缓存命中与错误类别，进程重启后自动消失。该接口不会返回 API Key、供应商 URL、请求/响应正文、提示词、歌词、模板、媒体或模型推理。

## 图片与视频处理

- 云端图片编码为 PNG；平价小屋模式上传，AI 工坊与 OpenAI 兼容模式内联为 Base64 Data URL。本地 Qwen 图片编码为限边 JPEG Data URL。
- 云端视频使用 ComfyUI 原生 `VIDEO`：平价小屋上传完整流，AI 工坊按其实测协议内联完整字节；OpenAI 兼容模式默认发送带时间戳的有序抽帧，只有用户填写 HTTP(S) 视频 URL 时才以 `video_url` 透传。本地 Qwen 同样读取真实帧率/PTS 与活动裁剪窗口生成有序视觉证据，不读取音轨。
- 支持 MP4、AVI、MOV、MKV，单文件不超过 50 MB。
- H3 Ref2VA 与 Seedance 2.0 增强器不限制参考视频的单段或合计时长，也没有 2 秒下限；视频时长元数据须有效且大于 0。渠道上传容量、模型上下文与显存等资源限制仍适用，不代表下游视频生成服务取消了自身限制。
- Ref2VA 最多 9 张图片、3 个视频，总素材数最多 12。
- 云端渠道仍拒绝带活动裁剪窗口的原生 `VIDEO`，因为上传底层流可能仍指向未裁剪原文件；本地 Qwen 不上传原视频，会按活动裁剪窗口和 PTS 采样，因此允许直接使用裁剪窗口。
- 平价小屋返回的素材上传 URL 是临时链接，应只用于当前模型请求；AI 工坊与 OpenAI 兼容模式没有中间上传步骤。

## 输出与错误行为

H3 与 Seedance 2.0 输出：

```text
enhanced_prompt: STRING
```

只要 Chat Completions 返回非空正文，节点就会输出，不会因为字段缺失、Markdown、镜头编号、时间码、目标长度或 `finish_reason=length` 报错。

MiniMax-H3 节点唯一的本地整理规则：

1. 当前任务的全部预期字段均以精确字段名命中；
2. 每个字段只出现一次；
3. 字段顺序确实错误。

三个条件同时成立时按官方顺序重排；任何字段未命中、缺失、重复，或者本来就是正确顺序时，保持上游原文。

Seedance 2.0 没有 H3 固定字段，因此不做字段重排和格式验收；非空上游正文直接输出。两个节点都不会因目标字数或镜头数量偏差报错，也不会为修复格式自动发起第二次付费请求。

Music 3 输出四个 `STRING`：

```text
lyrics
music_caption
music3_payload_json = {"input": lyrics, "instructions": music_caption}
enhancement_report_json
```

Music 3 只在三个官方标题都被唯一命中但顺序错误时进行本地重排；标题缺失或上游使用其他可用格式时直接放行非空正文，同时在增强报告中给出警告代码。目标字数、歌词长度或结构没有精确命中不属于节点错误。报告还会做歌词原句泄漏、所选官方参考短语重叠、纯器乐误加人声、段落时间线遗漏和官方 5000 Token 预算的软检查；这些检查不把已有非空结果变成节点错误。

以下情况仍会报错：

- API Key 无效或余额不足；
- 网络、超时、限流或供应商 5xx；
- 素材编码、上传或多模态请求失败；
- 响应不是合法 JSON；
- 响应缺少正文或正文全空白。
- 本地渠道缺少/损坏 GGUF、mmproj 或 llama.cpp 运行时，视觉/上下文预算越界，或本地推理进程异常退出。

平价小屋的 `https://api.seedance.nz/v1/chat/completions` 首次使用显式直连，避开失效的系统代理；只有确认发生在建连前的 SSL/ConnectTimeout，或 HTTP 500/502/503/504 与 Cloudflare 520–526/530 网关状态才进入有界重试，路径按“显式直连 → 环境代理 → 显式直连”切换，间隔 0.5 秒和 1 秒。401、余额不足、429、ProxyError、读取超时、流式响应中断和用户自定义 OpenAI 兼容接口不会盲目重投。Seedance Chat 使用流式响应并在内存中持续写入检查点；SSE 始终从原始字节按 UTF-8 解码，不信任网关缺失或错误的字符集声明。收到完整结束标记后即使代理在关闭连接时断开，也会直接返回已经完整接收的结果。2026-09-01 真实验收捕获到 `text/event-stream` 被客户端默认识别为 `ISO-8859-1` 的现场响应；修复后单次请求得到 228 个中文字符、0 个乱码替换字符，且恢复读取没有新增 API 请求。素材上传同样直连优先并在安全故障下切换路径；上传遇到 429 时仍按服务端提示等待后重试。

三个核心节点底部都有“恢复上次云端结果（不重新生成）”按钮。它只读取当前 ComfyUI 进程最近一小时内、同一节点已经完整完成的内存结果，不调用任何 LLM；复制节点时会自动生成独立恢复槽，重启 ComfyUI 后缓存清空。中途断流得到的残缺文本不会冒充最终提示词。如果上游已经生成、但零个响应字节到达本机，按钮会明确提示无法安全取回：2026-09-01 对当前接口的真实探测确认，相同 `Idempotency-Key` 仍会产生不同 completion 并分别计费，且没有可用的按 completion ID 查询结果端点，因此节点绝不会用“重试获取”偷偷再发一次付费生成。

Music 3 的 `official_reference_selection` 是“官方完整”模式必经阶段：必须成功选中至少 1 个官方模板才能继续编译 Caption。在 `api.seedance.nz` 遇到临时网关故障时最多尝试 6 次，按 0.5、1、2、4、8 秒有界退避；不会静默降级为无模板生成。

## 测试

在 ComfyUI 根目录的上一级运行：

```powershell
.\python\python.exe -m unittest discover -s ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tests -v
```

单元测试使用 mock API、本地媒体夹具和官方 Skill 静态资源，不联网、不上传素材、不产生费用。`tests/test_music3.py` 还会核验全部 18 个索引、1000 个模板、固定内容树哈希、逐级披露上限、歌词隔离、局部润色边界、安全 control tags、阶段缓存、诊断脱敏、云端 provider 重试与 Caption 语言纠正合同；`tests/test_local_qwen.py` 核验第四渠道免 Key、模型路径安全、seed 映射、图片 Base64、真实时间戳视频抽样、按首次出现顺序描述视频阶段、无音轨声明、三节点接线以及 Music 不加载视觉能力。`tests/test_platform_11.py` 独立验证共享传输重试、SSE 原始 UTF-8 解码、诊断字段白名单与 URL/Key/错误正文脱敏，以及三节点旧工作流迁移矩阵。

维护者发布前还应运行确定性仓库门禁；它会检查禁入文件、疑似密钥、全部已跟踪 JSON、官方 Skill 快照、Python/JavaScript 语法、`.comfyignore` 实际发布面、Registry YARA 触发模式和 90 MiB GIF 总预算。该上限专门为 Comfy Registry 的 100 MB ZIP 扫描保留余量，避免 Action 发布成功但版本随后被标记为 `Flagged`：

```powershell
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\verify_repository.py
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\release.py --check-prepush
```

本地 Qwen 发布验收不调用云端 API，但会真实加载约 18GB 权重并长时间占用本机 GPU/CPU。只有明确接受本地算力消耗时运行：

```powershell
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\local_qwen_live_smoke.py --confirm-local-large-model
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\local_qwen_live_smoke.py --confirm-local-large-model --model qwen3.8-27b-uncensored-fp8-q4_k_m.gguf --output ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tests\fixtures\local_qwen_uncensored_quality_2026-08-21.json
```

2026-08-19 在 RTX 4060 Ti 16GB / llama.cpp CUDA 12 上的固定验收结果为 `100/100`、`passed=true`：同 seed 三次逐字一致、H3 T2VA、Music 3 中文新歌词与官方完整 Skill、H3 图像+视频、Seedance 2.0 图像+视频五项全部通过。Music 约束逐项核验 92 BPM、D major、4/4、女主唱、钢琴、原声吉他与排除 rap；视觉夹具逐项核验图片代码、视频早晚代码、运动方向、硬切和时序。全套耗时约 964 秒，峰值显存 15091 MiB，卸载后回到 2246 MiB（基线 2223 MiB）。脱敏报告见 `tests/fixtures/local_qwen_quality_2026-08-19.json`；不包含提示词正文、歌词、Base64、API Key 或本地临时令牌。

`live_smoke.py` 用于平价小屋的 MiniMax-H3；`seedance20_live_smoke.py` 用于平价小屋的 Seedance 2.0。`workshop_live_smoke.py` 会让两个节点各走一次 AI 工坊：H3 使用默认模型，Seedance 2.0 使用 Custom 路径但填写同一个 `gemini-3.5-flash`。测试会生成带独有文字、颜色、形状和两阶段运动的本地图片与完整 4 秒 MP4，并核验两个节点是否真的识别图片和视频时间顺序。真实测试会产生 Token 费用，只有明确接受费用时才运行：

```powershell
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\live_smoke.py --confirm-paid
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\seedance20_live_smoke.py --confirm-paid
$env:T8STAR_API_KEY="你的 AI 工坊 API Key"
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\workshop_live_smoke.py --confirm-paid
```

前两个脚本运行前设置 `SEEDANCE_API_KEY`；AI 工坊脚本设置 `T8STAR_API_KEY`。不要把真实 Key 写进命令参数、脚本或工作流后上传到公开仓库。

Music 3 发布质量验收使用 4 个真实工作流：中文新歌词、原歌词逐字保留、纯器乐融合、只改第二段主歌；另用一次批量结构化评审。确定性合同分占 70%，LLM 内容质量评审占 30%，每案总分必须至少 85、内容评审至少 80，且没有标题合同、payload、歌词隔离、器乐或编辑边界硬失败。结果写入脱敏 JSON，不包含 API Key。若个别案例或最终评审失败，脚本会以原子 checkpoint 续跑而不重复已完成案例；通过后自动删除 checkpoint：

```powershell
$env:SEEDANCE_API_KEY="你的平价小屋 API Key"
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3_live_smoke.py --confirm-paid
```

官方 Skill 不会在用户运行节点时联网更新。维护者拿到经过审阅的 MiniMax-Music3 本地 checkout 后，可先只读核验，再显式更新固定快照；工具会检查 18 个索引、1000 个模板、1022 个文件和归一化哈希，并同步 `SOURCE.json` 与运行时常量：

```powershell
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\update_music3_official_skill.py --check-current
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\update_music3_official_skill.py --source-dir D:\MiniMax-Music3 --commit 40位提交SHA --apply
```

H3 核心 Skill 同样使用审阅后显式更新，不会在用户运行时联网拉取：

```powershell
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\update_h3_official_skill.py --check-current
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\update_h3_official_skill.py --source-dir D:\MiniMax-H3 --commit 40位提交SHA --apply
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\check_upstream_skills.py
```

## 参考资料

- [MiniMax-H3 基础提示词指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax-H3 完整参考模式指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [MiniMax-H3 官方创意 Skills 快照](https://github.com/MiniMax-AI/MiniMax-H3/tree/743d51e83329cbae6c7694f1c7b89576e7c25e07/skills)
- [MiniMax-H3 官方核心 Prompt Writing Skill（固定提交）](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/SKILL.md)
- [MiniMax-H3 官方 Music Video Subtitle Generator Skill v0.6.6](https://github.com/MiniMax-AI/MiniMax-H3/blob/743d51e83329cbae6c7694f1c7b89576e7c25e07/skills/music-video-subtitle-generator/SKILL.cn.md)
- [Seedance API 文档](https://api.seedance.nz/docs/llms.txt)
- [Seedance 模型页面](https://api.seedance.nz/pricing/bytedance%2Fdoubao-seed-evolving)
- [Seedance 2.0 官方模型页](https://seed.bytedance.com/en/seedance2_0)
- [Seedance 2.0 官方 Prompt Optimizer Skill](https://arkdocs.tos-cn-beijing.volces.com/files/video-generation/SKILL.md)
- [MiniMax Music 3 官方仓库](https://github.com/MiniMax-AI/MiniMax-Music3)
- [MiniMax Music 3 官方 `music-caption-rewriter` Skill（固定提交）](https://github.com/MiniMax-AI/MiniMax-Music3/tree/91410fb657c007ae57c60df8240f5ece5be089c7/skills/music-caption-rewriter)
- [Qwen3.8-27B GGUF（固定 revision 下载来源）](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [本地 GGUF 参考项目 ComfyUI_Qwen_H3_Prompt](https://github.com/chflame163/ComfyUI_Qwen_H3_Prompt)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

## 说明

本项目是第三方 ComfyUI 自定义节点，不隶属于 MiniMax、ByteDance、Seedance 或 ComfyUI 官方。API 能力、价格和可用性以服务商最新说明为准。使用真人参考素材时，用户仍需自行确认身份授权和下游平台规则。

## 本地 Skill 与整合包

三个核心节点底部均提供“MiniMax & Seedance本地Skill和整合包”入口，点击会在新标签页打开 [T8 本地 Skill 与整合包](https://github.com/T8mars/minimax-h3-prompt-skill-T8)。该入口仅用于访问资源，不会写入工作流，也不会参与或改变任何 LLM 请求。


---

## FILE: README_EN.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/README_EN.md)

<p align="right">
  <a href="./README.md">简体中文</a> | <strong>English</strong>
</p>

Version 1.42.2 fixes cloud proxy routing on selected container hosts; see [RunningHub and proxy routing](docs/cloud-network.md).
Explicit backend proxy settings and `NO_PROXY` are respected; hosts without proxy settings retain direct routing.

# ComfyUI MiniMax H3 / Seedance 2.0 / Music 3 / YuE2 Prompt Enhancer T8

## Updated: concrete H3 color and creator execution with ordinary On (1.40.0)

Ordinary H3 **On** requests concrete palette placement and creator methods in the scene description. Unspecified or undetermined colors on eligible visible surfaces are completed from the selected palette; locked original colors, reference colors and materials remain protected. The legacy **On (H3 experimental)** choice stays compatible and uses the same instructions as ordinary On in this version. Both configs default Off. To stop applying a dimension, switch that config Off and generate again.

Search `T8 Color Style Config` or `T8 Creator Style Config`. Connect either to H3 / Seedance, browse the local catalog, select, then enable On. They combine independently, with 200 palette comparisons and 400 creator-method diagrams; no MOSS integration, image processing or duration cap. Creator methods are not a screenplay generator, and limited-source entries are labeled. Diagrams are not model outputs. [Guide](docs/visual-styles.md) · [Color→H3](example_workflows/visual_style_color_h3.json) · [Creator→Seedance](example_workflows/visual_style_creator_seedance20.json) · [Combined→H3](example_workflows/visual_style_combined_h3.json)

This version keeps the earlier experimental execution instructions and adds the open-color clarification. Off and Seedance generation text are unchanged; no new fields or separate style requests are added. Validation covers palette and method execution in prompt text, while previews remain teaching illustrations. [Validation progress, model settings and historical limitations](docs/visual-styles.md#style-validation) are maintained in the guide.

## New: H3 visual marker authoring (experimental)

`T8H3SpatialMarkerCanvas` now provides upload, actor/target drawing, move/resize, descriptions/actions, undo/redo and local preview. No handwritten JSON or audio plugin is required. [Standalone Ref2VA](example_workflows/h3_spatial_canvas_ref2va.json) / [Hybrid](example_workflows/h3_spatial_canvas_hybrid.json); the previous helper and examples remain compatible.

H3 also has an independent, default-Off **T8H3VisualMarkerConfig / Visual Marker Config** entry.
It reads the existing audio plugin's MarkerPlan and matches the actual marked reference image; neither reference project needs modification.
Normal Ref2VA/Hybrid extra references only, no duration cap or extra planner. This is soft prompt guidance, not a trajectory/removal guarantee.
Click **📖 空间标记：示例与接线 / Examples** on the spatial helper for the wiring diagram,
companion reference PNG and two ready-wired JSON downloads. Save your current graph, import a sample,
upload the PNG, configure your enhancer provider and switch the sample from Off to On.
[Marker wiring and limits](docs/h3-visual-marker.md) · [Ref2VA](example_workflows/h3_visual_marker_ref2va.json) / [Hybrid](example_workflows/h3_visual_marker_hybrid.json).

## New: experimental storyboard direction

The independent `T8 Storyboard Director Config` connects to H3 / Seedance only, default Off. It cooperates with existing directors and acting, with **no duration cap, truncation or automatic splitting**, and no separate planning request. None/Off retains old behavior. No MOSS integration; benefit remains unverified. [Guide and evaluation](docs/storyboard-director.md) · [H3 pair](example_workflows/storyboard_director_h3_comparison.json) / [Seedance pair](example_workflows/storyboard_director_seedance20_comparison.json).

## New: optional speech/subtitle reference

The default-Off **T8 Speech Reference** helper reuses provided text/subtitles offline, or transcribes AUDIO/file-VIDEO through an already running CPU-guarded Strata. H3/Seedance append one optional socket. Semantic reference is distinct from explicitly mapped, verbatim dialogue. Original providers/workflows and service lifecycle are retained. [Guide, privacy and correction costs](docs/speech-reference.md) · [H3](example_workflows/speech_reference_h3.json) / [Seedance](example_workflows/speech_reference_seedance20.json).

## New: MOSS Image prompt authoring

The independent **MOSS Image Prompt Enhancer T8** creates native Caption or Layout drafts. A pinned official compiler converts pixel boxes locally. Fixed canvas dimensions, exact visible-copy locks and up to ten expanded reference images work with the existing cloud/GGUF/running-Strata providers. References inform the enhancement LLM; this is **not native MOSS image editing**.

T8 accepts 8-pixel canvas multiples in 16–16384, including 1256×1672, without rounding generation, Layout or restored dimensions. Upstream sources stay unchanged; the compatibility adapter validates and centers boxes against the real canvas. Downstream latent nodes may align dimensions independently.

Compile-only is offline. At most one additional correction is allowed; a hard-valid first draft survives a failed optional correction. Hard-invalid drafts stop downstream execution and remain available for explicit inspection. The native 4096-token budget is unmeasured, not guaranteed by a character count. [Guide and examples](docs/moss-image.md). Existing nodes/workflows are unchanged.

The main node now exposes the existing description-character limit and appends an opt-in profile: default `Classic`, `Fisher`, or `Comic`. The original 27 saved fields and six outputs retain their positions; old workflows remain Classic. The limit is a soft Unicode description count, not Tokens, and never truncates a draft. Automatic Chinese Fisher expansion prefers roughly 1500–2500 Han characters unless an explicit short/length constraint overrides it. Main-node Comic still produces one image in one descriptive language.

For collections, search **MOSS Authoring Director T8** and connect **T8 LLM Provider Config** for generation. Comic plans up to 16 pages, then produces English/Chinese sources sharing geometry. Fisher defaults to one independent unit per reference; only explicit `fusion` combines references into one image. At most ten expanded references are accepted. Connect Director `bundle_json` to **MOSS Authoring Page T8**, select a 1-based page and language, and wire Page outputs 0/1/2 to positive prompt/width/height. Never wire a bundle, report or source JSON directly to positive. [Wiring, revisions and costs](docs/moss-image.md#整册与多参考图director--page). These independently adapted creative rules are separate from H3/Seedance storyboard configuration; structural checks do not certify translations or rendered images.

Start with [zero-model CPU page selection](example_workflows/moss_authoring_cpu_page.json), [two-page bilingual Comic](example_workflows/moss_authoring_comic_two_pages.json), [two independent references](example_workflows/moss_authoring_fisher_independent.json) / [explicit fusion](example_workflows/moss_authoring_fisher_fusion.json), or [fixed-layout page-two revision](example_workflows/moss_authoring_fixed_layout_revision.json). For either two-reference lesson, upload `example_workflows/moss_authoring_fisher_independent.jpg` and `moss_authoring_fisher_fusion.jpg` into its two LoadImage nodes first; examples never write ComfyUI input automatically. The CPU example uses hand-authored teaching data, not model calls or rendered-quality evidence.

## New: existing Strata service bridge

Select **Strata 本地服务（桥接）** to use an already running Strata service. The default is `http://127.0.0.1:8080` **from the ComfyUI backend**, not the browser. No cloud API key or separate Strata plugin is needed. Connect **T8 LLM Provider Config → Strata** for advanced address/model/thinking/sampling settings; shared configuration takes precedence.

The original chat bridge supports text, images and sampled visual video, without listening to audio. The separate optional speech helper supplies transcribed text, not native chat audio. Original prompt formats, quality checks, outputs and API/GGUF workflows are retained. The bridge never starts, stops, switches or unloads the service model. [Usage and troubleshooting](docs/strata-bridge.md).

## New: H3 Hybrid keyframes + references

Select **Hybrid** in the existing H3 task combo. Connect first and/or last frame,
plus at least one additional reference image/video. Keyframes anchor the target
boundaries; other assets supply requested identity, appearance or motion evidence.
Hybrid uses H3's six reference fields and the existing four cloud/local providers
and Prompt Relay. The five old modes, T2VA default, 38 saved fields and old workflows
are unchanged. Each Hybrid image slot must contain one image; up to nine extra
images plus two keyframes and three videos. There is no enhancer-imposed video
duration cap. No audio track is analyzed; use the H3 audio plugin for audio Hybrid.
Text checks are not pixel or rendered-video acceptance.

[Guide and limits](docs/h3-hybrid.md) · [First + image](example_workflows/h3_hybrid_first_image.json) /
[Last + image](example_workflows/h3_hybrid_last_image.json) /
[First/last + video](example_workflows/h3_hybrid_first_last_video.json).

## New: optional quality workflow and causal creation

**Optional Combat Camera Config:** connect the new utility node to the H3 or Seedance
enhancer's `combat_camera_config` socket. AUTO/Strong, continuity preference and
Natural/Stylized impact support existing action without inventing combat. It can coexist
with Performance Director. Unconnected/Off preserves old behavior; no extra planning
request. [Guide and validation limits](docs/combat-camera.md) ·
[H3 example](example_workflows/combat_camera_h3.json) / [Seedance example](example_workflows/combat_camera_seedance20.json).

The existing H3/Seedance nodes offer **Check, one bounded quality correction, and Causal creation**. Off/Original preserve old workflows and valid native outputs; confirmed H3 vocal-format errors are automatically corrected for free. No extra form or wiring is required. Node guidance and redacted diagnostics show failures and correction counts. Model corrections may cost one additional logical request; failures retain the complete draft. Text contracts do not prove complex semantics, physics or rendered video quality.

[Bilingual guide and cost boundaries](docs/h3-quality-creation.md) · [Check example](example_workflows/h3_seedance_quality_check_example.json) · [Causal + repair example](example_workflows/h3_seedance_causal_creation_example.json)

## New: H3 / Seedance directional creation Skills

**兔子-舞蹈 / Dance** adds grounded footwork, visible beat accents and motion handoffs through the existing selector in both video enhancers. Off by default; it adds no duration limit, audio analysis or workflow fields. [Guide](docs/directional-skills.md#兔子-舞蹈--dance) · Examples: [H3](example_workflows/directional_rabbit_dance_h3.json) / [Seedance](example_workflows/directional_rabbit_dance_seedance20.json).

**Jojocodex-Wushu combat** adds body drive, contact/evasion, support and next-action continuity, including unfinished movement across cuts. Select it in either existing enhancer; native formats and all cloud/local providers remain unchanged. No forced three-shot plan, injury, winner or LoRA. Off remains default and historical workflows are preserved. [Scope](docs/directional-skills.md#jojocodex-武术打斗--wushu-combat) · Standalone examples: [H3](example_workflows/directional_wushu_combat_h3.json) / [Seedance](example_workflows/directional_wushu_combat_seedance20.json). Engineering regressions are not live LLM or rendered-video acceptance for this Skill.

**Zhenzhen POV** is a seventh independent method for readable viewpoint ownership, viewer participation, reception and consequences. It keeps the existing cloud/local providers, language, duration and native formats. The credit does not select an identity or provider; no LoRA, fixed batch or video workflow is added. Off remains the default. [Usage and limits](docs/directional-skills.md#贞贞-pov剧情导演--zhenzhen-pov) · Standalone examples: [H3](example_workflows/directional_zhenzhen_pov_h3.json) / [Seedance](example_workflows/directional_zhenzhen_pov_seedance20.json).

Two independent choices ship together: **Dramatic scene — relationship & subtext** and **Situational drama — expectations & payoff**, alongside the four existing methods below. No new node/wiring, mandatory conflict, reversal, gag or reaction. Supplied dialogue stays exact by default; explicit requests may authorize missing lines, with LOCK/no-extra-speech taking priority. Character Bible is optional. They retain all existing providers and native H3/Seedance output contracts and add no planning/scoring call; existing corrections/retries may still cost extra. [Bilingual scope and help](docs/directional-skills.md#戏剧场面与情境戏剧--dramatic-and-situational-scenes) · Dramatic: [H3](example_workflows/directional_drama_scene_h3.json) / [Seedance](example_workflows/directional_drama_scene_seedance20.json) · Situational: [H3](example_workflows/directional_situational_drama_h3.json) / [Seedance](example_workflows/directional_situational_drama_seedance20.json). [Pinned non-official adaptation and rights notice](directional_skills/SCREENWRITING-NOTICE.md); no book/script excerpts bundled. Off remains default and all legacy methods retain their published behavior.

The existing H3 and Seedance 2.0 enhancers offer four independent, opt-in methods: **Fisher-Continuous combat, 土豆-High-density combat, 兔子-Cinematic gunfight, and Ning · Drama & Action**. Author prefixes change display names only; stable IDs and historical unprefixed labels remain compatible. Ning directs attention around meaningful information, permitted reactions, and changes in force or distance, including dialogue leading into action. It supports authorized original writing while keeping exact supplied words and scene constraints. [Ning workflow](example_workflows/directional_ning_wenwu_comparison.json). Off remains the default. These non-official creative methods retain each platform's native format, providers and outputs without extra wiring. While enabled, they pause old case/manual-template choreography and optional H3 scene presets for the current request; saved choices resume when switched off. Continuous combat resolves AUTO to one shot and rejects an explicit count above one before paid calls.

[Concise bilingual guide](docs/directional-skills.md) · Side-by-side cloud examples: [Fisher-Continuous combat](example_workflows/directional_continuous_combat_comparison.json) / [土豆-High-density combat](example_workflows/directional_high_density_combat_comparison.json) / [兔子-Cinematic gunfight](example_workflows/directional_cinematic_gunfight_comparison.json) / [Ning Drama & Action](example_workflows/directional_ning_wenwu_comparison.json). Keys are blank. Running the full graph invokes both text enhancers; the Skill itself adds no planning or scoring call, while existing repairs/retries may still run.

## New: H3 Prompt Relay authoring

The existing H3 enhancer has an opt-in Relay mode: global prompt, event prompts, seconds ranges, aligned frame count and a validation report. Normal mode and the original first output keep their purpose. Existing cloud/local providers remain available; text authoring does not load a video model or activate downstream attention routing.

[Bilingual guide and wiring](web/js/docs/h3_prompt_relay.md) · [8-second cloud workflow](example_workflows/h3_prompt_relay_example.json)

## New: YuE2 music prompt and lyrics creation

The independent **YuE2 Music Prompt & Lyrics Enhancer (T8)** writes original lyrics,
preserves supplied lyrics exactly, edits a selected section, restyles a song, and
validates optional native ABC or removes chords while preserving both melodies.
Outputs: `style`, `lyrics`, `abc`, official request JSON, and a creation report.
This is text preparation, not audio generation; no music weights are downloaded.

Use the existing cloud/OpenAI-compatible/local GGUF providers and shared provider
config. Defaults are Chinese lyrics with English style descriptions, independently
controlled. Advanced fields are collapsed, with examples, help and completed-result
recovery. Existing core nodes and workflows remain unchanged.

[Guide and call costs](web/js/docs/yue2_music.md) · [Cloud creation](example_workflows/yue2_cloud_creation_example.json) · [Preserve lyrics](example_workflows/yue2_preserve_lyrics_example.json) · [Local GGUF](example_workflows/yue2_local_qwen_example.json) · [ABC melody](example_workflows/yue2_abc_melody_example.json)

## New: independent lyric and arrangement planners

Two independent text-only planning nodes are included: **T8 Lyric Writer** returns lyrics plus a versioned `lyric_plan_json`; **T8 Arrangement Planner** returns section, tempo, meter, key, instrumentation, and energy-curve data as `arrangement_plan_json`. They do not generate audio, ABC, or MIDI. Both reuse the YuE2 provider family, shared config, local GGUF runtime, and existing model defaults. With their plan sockets disconnected, Music 3 and YuE2 follow the original workflow behavior unchanged.

A lyric plan can feed Music 3 or YuE2. An arrangement plan is projected only through a small allow-list of music fields; external Skill markup or free-form text is never injected wholesale into a Caption. Plans use `t8-lyric-plan/v1` and `t8-arrangement-plan/v1`, with schema, size, and credential-leak checks before connection. Text review is not an audible-music or audio-render acceptance test.

## New: Qwen Image 2.1 prompt enhancement

The independent **Qwen Image 2.1 Prompt Enhancer** turns a text brief into an image prompt and accepts one to ten ordered reference images. **Classic** remains the default frozen English contract. Advanced **Edit rules → Edit-aware (T8)** applies only to image editing: clarify the requested operation, preserve untargeted content by reference, separate descriptive language from exact lettering, and identify multiple sources with `<imageN>`. Chinese briefs normally get Chinese prose; English/other briefs get English. The four output ports and existing providers are unchanged. A nonserializable action shows the last result's structural status and marks it stale after configuration changes.

The default provider is ZhenZhen Affordable AI Shop, model `qwen/qwen3.8-flash-next`. Cloud budget remains 8192 tokens including reasoning unless shared config sets `max_tokens`/`max_completion_tokens`. There is at most one logical format/length repair total; a failed repair retains a valid complete first draft with warnings, not a third generation. Edit-aware can follow original image dimensions: 1080×1590 resolves to `36:53`, never a nearest preset or silent crop; verify downstream support. A fixed UI ratio wins. Transparency requests prompt semantics only. Request JSON is metadata, not a render API body. Classic has historical live-cloud evidence; new local code/browser regressions are not live API A/B, RunningHub publishing/reload, or rendered-image quality acceptance. For dedicated PE models use the separate [Qwen Prompt Rewrite nodes](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8); there is no duplicated PE bridge here.

[Qwen Image guide](web/js/docs/qwen_image21.md) · [Text-to-image workflow](example_workflows/qwen_image21_text_example.json) · [Multi-image edit workflow](example_workflows/qwen_image21_edit_example.json)

Without an existing ABC score, full/melody now use the selected LLM to compose one
(a T8 extension, not the YuE2 model planner). Full includes melody and chord symbols;
melody is chord-free. Existing scores take priority. Advanced settings retain the
Downstream option for an empty ABC output and native YuE2 planning. Composition adds
one call, plus at most one structural repair call.

> **ABC model guidance: local 9B LLMs may produce invalid ABC formatting or measure timing.** Prefer an API, or try a larger-parameter model. Both local 9B full/melody tests failed score validation. Recorded API tests passed, but this does not guarantee other APIs or larger models; larger models were not tested in this round. Failed ABC does not block style, lyrics or request JSON: the rejected score is omitted and the reason is shown. See the [test evidence](web/js/docs/yue2_abc_acceptance.md).

Optional review scores evaluate text, not audible music. YuE2 `full / melody / off`
select score planning, not LLM reasoning strength. Target duration is advisory.

A ComfyUI node suite for improving MiniMax H3 and Seedance 2.0 video prompts, preparing Qwen Image 2.1 image prompts, and preparing MiniMax Music 3 and YuE2 music text. The H3 and Seedance nodes accept text plus real ComfyUI `IMAGE` and `VIDEO` inputs. Qwen Image 2.1 accepts text plus up to ten ordered images; Music 3 and YuE2 keep their own text contracts.

All five core nodes can use ZhenZhen Affordable AI Shop, ZhenZhen AI Workshop, a user-supplied OpenAI-compatible endpoint, or a local llama.cpp-compatible GGUF model. Provider transport and error handling are shared, while the prompt contracts remain isolated for each target model.

> The repository homepage intentionally remains Chinese through [`README.md`](./README.md). Use the language switch above to move between the Chinese and English guides.

## Service entry points

| Service | Best for | Open |
| --- | --- | --- |
| ZhenZhen Affordable AI Shop (Mainland China) | Mainland users and domestic models | [Get an API key](https://api.seedance.nz/sign-up?aff=5f4w) |
| ZhenZhen AI Workshop (International) | International users and overseas models | [Create an account](https://ai.t8star.org/register?aff=dP7j) |
| RunningHub China | More hosted AI applications in Mainland China | [Open RunningHub China](https://www.runninghub.cn/user-center/1819214514410942465/webapp?inviteCode=rh-v1121) |
| RunningHub International | International models and applications | [Open RunningHub International](https://www.runninghub.ai/user-center/1907375370302308353/webapp?inviteCode=rh-v1121) |

## Core nodes

| Node | Purpose | Main modes |
| --- | --- | --- |
| `MiniMax H3 Prompt Enhancer (Cloud / Local GGUF)` | Generate MiniMax H3 prompts | T2VA, I2VA, FL2VA, L2VA, Ref2VA, Hybrid |
| `Seedance 2.0 Prompt Enhancer (Cloud / Local GGUF)` | Generate Seedance 2.0 prompts | Text-to-video, first frame, first/last frame, multimodal reference, edit, extend, track completion, combined tasks |
| `MiniMax Music 3 Prompt & Lyrics Enhancer (T8)` | Prepare Music 3 lyrics and captions | AUTO, new lyrics, preserve, scoped rewrite, instrumental |
| `YuE2 音乐提示词与歌词创作（T8）` | Prepare native YuE2 style and lyrics | Original lyrics, preserve/restyle, section edits, instrumental, optional ABC preparation |
| `Qwen Image 2.1 Prompt Enhancer` | Prepare image-generation prompts | Text-to-image, one-to-ten image editing, ratios, transparent RGBA |

This project does not currently provide a Seedance 2.5 prompt node. It does not call video or music generation, polling, preview, or download APIs.

## Highlights

- Cloud providers can analyze text, images, and complete video inputs for H3 and Seedance 2.0. Music 3 is deliberately text-only.
- Local GGUF mode recursively discovers compatible models under `ComfyUI/models/LLM`, supports matched visual projectors, and requires no API key.
- Supports first frame, last frame, first-and-last-frame, multiple reference images, and multiple reference videos.
- Includes the pinned official MiniMax H3 core prompt-writing Skill and all eight optional official scene Skills.
- Includes the complete pinned MiniMax Music 3 `music-caption-rewriter` snapshot: 18 family indexes and 1,000 templates with progressive disclosure.
- Includes 652 released T8 case records consolidated into 305 stable selectors, including 347 evidence variants, plus two independent community Skills. All 307 non-official selectors include Chinese names, concise suggested inputs, structural anchors, and lightweight GIF previews.
- Chinese and English prompt output.
- H3 `description_word_target`: `0` selects automatic length; nonzero targets start at 80 and can exceed the former 1000 limit (for example, 1500, 2000 or 5000). The target is approximate, and actual output still depends on the model and Token budget.
- `strict`, `balanced`, and `creative` rewrite levels.
- `AUTO` or fixed 1–20 shot control.
- Optional user-template fusion with user intent and observable media facts taking priority.
- ComfyUI seed controls: `fixed`, `randomize`, `increment`, and `decrement`.
- In-node API-key entry, masked display, workflow save, clear, and registration links; standard `STRING` API-key connections remain supported.
- OpenAI-compatible Base URL and model ID persist in the workflow and do not need to be re-entered on every run. API keys remain separate.
- Native progress reporting, memory-only redacted diagnostics, provider capability preflight, local prompt inspection, and a no-resubmit recovery button on all three core enhancers.
- Optional `T8 Performance Director Config` adds observable acting structure with `AUTO / Strong / Off / Extreme`, plus opt-in **Tudou emotion direction** (`土豆-情绪演绎（声画协同）`). The fifth option separates emotion from intensity/volume, carries preceding-state residue and coordinates each line, lip motion and visible cues. It is not a higher intensity tier, adds no planning/scoring call, and preserves exact words, sound permissions and native H3/Seedance/Relay contracts. Existing character Bibles and independent Skills remain usable. See [method and evidence boundaries](./docs/EMOTION_PERFORMANCE.md); text quality is not rendered performance evidence.
- A searchable T8 template browser with categories, search, favorites, recent items, lazy GIF previews, deterministic Top-3 recommendations, and comparison.

## T8 Creative Director suite

The [Tudou real-API report](./docs/EMOTION_API_ACCEPTANCE.md) retains failures
and separates the initial 48-slot A/B from revised follow-ups. It does not show
a universal advantage over Extreme. H3 now automatically normalizes fullwidth
speaker IDs, safely bound missing IDs and explicit language-tag aliases, even
with Quality Off, on local/API paths. This free step preserves actual words.
Ambiguous multiple voices retain the complete draft and a diagnostic rather
than guessed identities; optional Quality Repair still allows at most one paid
correction. See [vocal-format fix and evidence](./docs/H3_VOCAL_PROTOCOL_FIX.md).

The suite adds planning, exploration, revision, and delivery helpers without replacing the three core enhancers. Existing core node IDs, serialized widget order, defaults, and outputs remain unchanged.

| Node | LLM calls | Purpose |
| --- | ---: | --- |
| `T8 Film Project Router` | 0 | Maintain stages, authoritative inputs, revisions, world contracts, and downstream invalidation caused by upstream changes |
| `T8 Character Performance Bible` | 0 | Compile objective, obstacle, tactics, physical inertia, voice, and listening behavior into a lightweight contract capped at three observable cue channels per beat |
| `T8 Character Performance Bible Stack` | 0 | Combine 1–8 independent character contracts, reject duplicate IDs, and keep each character's objective, tactic, and cues separate in multi-character shots |
| `T8 Creative Director` | 0 | Shared story, identity, world, visual, motion, and sound brief with `LOCK / EVOLVE / AUTO` policies |
| `T8 Creative Context Assembler` | 0 | Compile the brief, reference map, Creative DNA, and personal preset into a `STRING` for existing nodes |
| `T8 Directed Revision` | 1 | Apply only the requested delta and return a local diff |
| `T8 Long-form Planner` | 1 | Split any positive duration into H3/Seedance segments with continuity, world-rule, knowledge-state, and stale-stage checks |
| `T8 Reference Role Mapper` | 1 | Analyze only directly connected user images/videos and assign identity, costume, scene, motion, camera, style, and exclusion roles |
| `T8 Creative Candidate Lab` / `T8 Candidate Selector` | 1 / 0 | Generate 2–4 structurally distinct directions and select one locally |
| `T8 Storyboard Pack` | 1 | Global prompt, shot JSON, keyframes, transitions/sound, media bindings, performance IR, and causality/value/necessity/setup-payoff audit |
| `T8 Creative DNA Mixer` | 0 | Blend structure/camera/payoff mechanics from up to three non-official T8 cases without using preview media |
| `T8 Personal Creative Preset` | 0 | Store user-owned text rules inside the workflow only |
| `T8 Music Creative Lab` | 1 | Lyrics/caption candidates, scoped lyric revision, titles, and soft CJK singability text checks |
| `T8 Creative Version Stack` | 0 | Select or roll back among up to eight prompt, lyric, or caption versions |
| `T8 Music-to-Video Beat Sheet` | 1 | Convert text lyrics/captions and user-known timing facts into H3/Seedance directing constraints |

LLM-backed helpers default to Seedance NZ. Connect `T8 LLM Provider Config` to use AI Workshop, an OpenAI-compatible endpoint, or local GGUF instead. Each execution represents one logical generation request. Seedance NZ may make up to six bounded transport attempts only for the safe gateway set 500/502/503/504/520–530; authentication, balance, rate-limit, and ambiguous read-timeout failures are not blindly repeated. A non-empty unstructured provider response is preserved and reported with `structured_response=false` instead of being rejected.

The seven candidate scores are deterministic local text heuristics, not model self-scores or objective artistic judgments, and they never reject a non-empty candidate. Storyboard keyframe and transition/sound tables are derived locally from the returned shot contract to avoid duplicated paid output. On 2026-08-28 the real `bytedance/doubao-seed-evolving` endpoint passed redacted binary-contract checks for candidates, directed revision, long-form planning, a four-shot storyboard, image role mapping, Chinese lyric candidates, and a five-event MV beat sheet. That acceptance score is not a subjective art-quality score.

For Router continuations, blank authoritative fields inherit. Enter `[CLEAR_INHERITED]` (or `[清空继承]`) in one field to delete only that inherited value while other blank fields continue to inherit. Router, Long-form, and Storyboard status cards follow the ComfyUI locale and grow with wrapped diagnostics. Long-form and Storyboard default to compatibility warning so existing workflows retain their behavior; select the recommended strict contract policy to stop downstream execution with ComfyUI's native blocker while preserving the card and JSON `validation_errors`. These local checks do not add an LLM call.

On 2026-08-30, six real paired text A/B groups covered causality/value shift, setup/payoff, tactic progression, silent acting, world-rule/permanent-cost continuity, and knowledge-gap/downstream invalidation. With identical cases and seeds on Seedance NZ `qwen/qwen3.6-flash`, the former contract averaged 44 and the final enhanced contract averaged 100; all six enhanced groups were non-regressive. These are deterministic JSON-shape, literal-anchor, ordering, cue-budget, and state-field checks, not subjective artistic scores. The final retest reused six matching paid baselines and issued six new enhanced requests. Two earlier `bytedance/doubao-seed-evolving` enhanced trials returned HTTP 524, were excluded from scoring, and stored no response text. See the redacted [`film_workflow_paid_ab_2026-08-30.json`](./tests/fixtures/film_workflow_paid_ab_2026-08-30.json) report and the reproducible [`film_workflow_ab_benchmark.py`](./film_workflow_ab_benchmark.py) runner. New reports record repository version/commit, model ID, parameter and seed policies, and contract hashes. `--model` supports separate cloud-model reruns and baseline reuse requires an exact model/contract match. Local-Qwen parameter delivery is covered by local regression tests, but is not presented as real cross-model quality evidence until an actual local run is recorded.

For final H3 / Seedance render acceptance, [`film_workflow_render_acceptance.py`](./film_workflow_render_acceptance.py) consumes a real-video manifest and records hashes, ffprobe duration/stream evidence, and a named five-axis human review. It neither generates nor stores video/credentials and never treats a perfect text-contract score as a perfect rendered film.

Only media connected directly to Reference Role Mapper is analyzed. Bundled official/T8 GIFs and case-source videos are never model references. Local video mode uses timestamped visual samples and does not analyze audio. The Beat Sheet node has no AUDIO input and never claims to hear a song or detect BPM.

## Installation

> **Update recommendation: prefer GitHub.** When a new Registry version is under review, `Pending`, or `Flagged`, ComfyUI-Manager may silently fall back to an older `Active` release. Installation can appear successful while new nodes, features, and urgent fixes remain unavailable. Fully restart ComfyUI after updating, refresh the browser with `Ctrl+F5`, and never keep duplicate copies of this node pack under `custom_nodes`.

### GitHub installation / update (recommended)

For a first installation, run this inside the ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8.git
```

For an existing Git clone, run this inside the plug-in directory:

```bash
git pull --ff-only
```

If the previous copy came from Manager or a ZIP, close ComfyUI, move the old plug-in directory out of `custom_nodes`, and then clone into the single canonical directory. Do not keep the old and new directories together.

Expected layout:

```text
ComfyUI/
└─ custom_nodes/
   └─ comfyui-minimax-h3-prompt-enhancer-T8/
```

### ComfyUI-Manager / Registry

Search for:

```text
MiniMax H3 / Seedance 2.0 / Music 3 Prompt Enhancer (T8)
```

Or install with Comfy CLI:

```bash
comfy node install minimax-h3-seedance-music3-prompt-enhancer-t8
```

The Registry package contains the node runtime code, pinned official Skills, the non-official case library, eight official GIFs, and a small preview-channel manifest. The 654 T8 GIFs are not embedded in the Registry ZIP. When a user opens a T8 case, the node downloads only its small release shard and caches it under `ComfyUI/user/t8_prompt_enhancer/preview_assets`. The package still excludes API keys, tests, source delivery batches, the standalone `llama-server` runtime, install/download scripts, and GGUF weights. Local GGUF mode in a Manager install uses `llama-cpp-python` from the active ComfyUI Python environment. Use the recommended full GitHub install when you need the pinned `llama-server`, PATH fallback, environment defaults, or the explicit credential connection probe.

All 662 official/T8 previews remain available. The main repository retains the eight official GIFs and the previous 377 T8 GIFs as an offline baseline; all 654 T8 previews, including new additions, are updated through the versioned [`comfyui-minimax-h3-prompt-enhancer-T8-assets`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8-assets) release channel. Shards, individual GIFs, catalog identities, and source hashes are verified before cache installation. Offline or failed preview downloads never stop prompt enhancement.

Restart ComfyUI after installation. If the frontend still shows an older node layout, refresh the browser with `Ctrl+F5`.

## Quick start: MiniMax H3

1. Add `MiniMax H3 Prompt Enhancer (Cloud / Local GGUF)`.
2. Enter a basic idea in the video concept/prompt field.
3. Select the generation type, duration, shot count, rewrite mode, output language, core Skill protocol, and optional official or T8 template. The H3 duration field accepts any positive whole number of seconds; the node imposes no maximum, although downstream generators may still do so.
4. Connect the images or videos required by I2VA, FL2VA, L2VA, Ref2VA or Hybrid.
5. Supply the selected cloud provider's API key, or choose local GGUF mode without a key.
6. Click the run button at the bottom of the node.
7. Read the final `STRING` from `enhanced_prompt`.

H3 generation types:

| Mode | Meaning | Required media |
| --- | --- | --- |
| `T2VA` | Text-to-audio-video prompt | None |
| `I2VA` | Develop forward from a first frame | `first_frame` |
| `FL2VA` | Design motion between first and last frames | `first_frame` and `last_frame` |
| `L2VA` | Build a plausible lead-in that resolves at the last frame | `last_frame` |
| `Ref2VA` | Reference-based generation or editing | At least one reference image or video |
| `Hybrid` | Keyframe boundaries + visual references | First and/or last frame, plus an additional reference image/video |

Import [`example_workflows/minimax_h3_prompt_enhancer_example.json`](./example_workflows/minimax_h3_prompt_enhancer_example.json) for a ready-to-use cloud example.

## Quick start: Seedance 2.0

1. Add `Seedance 2.0 Prompt Enhancer (Cloud / Local GGUF)`.
2. Enter the video concept. Task intent, organization, duration, and shot count can remain `AUTO` initially. Seedance duration also accepts any positive whole number of seconds, with no node-imposed maximum; downstream generator limits still apply.
3. Connect the required first frame, last frame, reference images, or reference videos.
4. Choose official Chinese references such as `@图片N/@视频N` or Seedance.nz English references such as `@Image N/@Video N`.
5. Optionally select a T8 case or community Skill to transfer causal structure, rhythm, camera grammar, and transitions without copying its subject matter.
6. Supply a cloud API key or select local GGUF mode.
7. Connect `enhanced_prompt` to the prompt input of the downstream Seedance 2.0 video node.

Task intentions:

| Mode | Purpose | Media rule |
| --- | --- | --- |
| `AUTO` | Infer from intent and connected media | Used when the task is unambiguous |
| `T2V` | Text-to-video | No media |
| `I2V` | First-frame image-to-video | `first_frame` only |
| `FL-I2V` | First/last-frame transition | `first_frame` and `last_frame` |
| Multimodal reference | Reuse identity, motion, camera, style, or other observable traits | At least one reference asset |
| Video edit | Add, remove, or change content in an existing video | At least one reference video |
| Video extend | Extend a video forward or backward | Exactly one reference video |
| Track completion | Generate a bridge between clips | Two or three reference videos |
| Combined task | Reference one asset while editing another | An edited video plus at least one other asset |

Simple requests use one compact paragraph. Complex requests use consecutive shot events. Fixed shot counts are generation constraints sent to the LLM, not local response-validation rules.

Import [`example_workflows/seedance20_prompt_enhancer_example.json`](./example_workflows/seedance20_prompt_enhancer_example.json) for a cloud example.

## Quick start: MiniMax Music 3

1. Add `MiniMax Music 3 Prompt & Lyrics Enhancer (T8)`.
2. Describe genre, theme, mood, use case, vocal direction, and arrangement development.
3. Choose `AUTO`, generate new lyrics, strictly preserve lyrics, scoped lyric rewrite, or instrumental.
4. Use the full official mode for official routing and progressive reference selection, or the fast core mode for the three-section caption contract only.
5. Supply a cloud API key or choose local GGUF mode.
6. Send `lyrics` to Music 3 `input` and `music_caption` to Music 3 `instructions`, or parse `music3_payload_json` directly.

Outputs:

```text
lyrics
music_caption
music3_payload_json = {"input": lyrics, "instructions": music_caption}
enhancement_report_json
```

The official Skill generates this ordered caption contract, normally in English:

```text
### Global Metadata
### Vocal Details
### Arrangement
```

Lyrics and captions are intentionally isolated. Lyric generation and scoped rewriting are clearly marked T8 extensions; strict preserve mode passes the original lyrics through byte-for-byte. The lyric-language setting controls only `lyrics` and does not change the language of `music_caption`.

Import [`example_workflows/music3_prompt_lyrics_enhancer_example.json`](./example_workflows/music3_prompt_lyrics_enhancer_example.json).

## Example workflows

| Workflow | Purpose |
| --- | --- |
| [`minimax_h3_prompt_enhancer_example.json`](./example_workflows/minimax_h3_prompt_enhancer_example.json) | H3 cloud example |
| [`seedance20_prompt_enhancer_example.json`](./example_workflows/seedance20_prompt_enhancer_example.json) | Seedance 2.0 cloud example |
| [`music3_prompt_lyrics_enhancer_example.json`](./example_workflows/music3_prompt_lyrics_enhancer_example.json) | Music 3 cloud example |
| [`basic_workflow_multi_task_connections.json`](./example_workflows/basic_workflow_multi_task_connections.json) | One shared provider configuration connected to all three core nodes |
| [`minimax_h3_local_qwen_example.json`](./example_workflows/minimax_h3_local_qwen_example.json) | H3 with local Qwen and shared provider configuration |
| [`seedance20_local_qwen_example.json`](./example_workflows/seedance20_local_qwen_example.json) | Seedance 2.0 with local Qwen and shared provider configuration |
| [`music3_local_qwen_example.json`](./example_workflows/music3_local_qwen_example.json) | Music 3 with local Qwen and no visual projector |
| [`prompt_inspector_local_qwen_example.json`](./example_workflows/prompt_inspector_local_qwen_example.json) | Local H3 enhancement followed by local structural inspection |
| [`text_utilities_example.json`](./example_workflows/text_utilities_example.json) | Built-in `T8 Prompt Text` to `T8 Show Text` STRING workflow |
| [`creative_direction_revision_example.json`](./example_workflows/creative_direction_revision_example.json) | Creative Director → Candidate Lab → selection → directed revision |
| [`creative_longform_storyboard_example.json`](./example_workflows/creative_longform_storyboard_example.json) | Film state, a character-performance bible, and one creative brief drive long-form segmentation and storyboard audit |
| [`creative_music_video_suite_example.json`](./example_workflows/creative_music_video_suite_example.json) | Music candidates, version selection, and H3/Seedance 2.0 beat-sheet direction |
| [`creative_reference_dna_preset_example.json`](./example_workflows/creative_reference_dna_preset_example.json) | Reference roles, T8 Creative DNA, and a personal preset assembled into a legacy-compatible STRING context |

All 13 bundled workflows omit API keys and use only nodes included in this repository.

## Official Skills and T8 templates

MiniMax's nine H3 Skills consist of one always-on core writing Skill and eight optional scene Skills. The core Skill defines shared fields, timelines, speaker labels, media roles, and sound contracts; it is not a visual preset and cannot be disabled.

The eight optional scene Skills cover minimalist product ads, 3D animation shorts, brand promos, music-video typography, co-op game intros, paper-collage explainers, papercraft stop-motion explainers, and live-action/hand-drawn fusion. The node adapts the prompt-writing constraints that fit a single H3 prompt. It does not claim to run MiniMax Hub agents, canvas tools, asset generation, approval, editing, or delivery workflows.

T8 non-official cases and community Skills are maintained separately from all official Skills. When a T8 non-official selector and an optional official scene Skill are both selected, the T8 selector takes precedence and the optional official scene Skill is temporarily inactive. The always-on H3 core Skill remains active. Clearing the T8 selector restores the saved official scene selection without changing workflow wiring.

Preview GIFs are human-interface previews only. They are never attached to or sent as model reference material. Both video nodes expose a non-serialized **Manage / update T8 dynamic previews** button with smart on-demand (default), full auto, and manual modes, plus check, install, repair, and clear actions. Settings live in the ComfyUI user directory and do not alter workflow JSON, node IDs, inputs, outputs, API keys, or prompt execution.

## Rewrite modes

| Mode | Temperature | Behavior |
| --- | ---: | --- |
| `strict` | `0.2` | Preserve intent and media facts; add only necessary structure and continuity |
| `balanced` | `0.7` | Add reasonable shots, lighting, motion, ambience, sound, and rhythm while preserving the brief |
| `creative` | `1.2` | Expand visual direction, camera design, action transitions, sound layers, and music direction |

These temperatures are product defaults, not provider recommendations.

## Providers

### ZhenZhen Affordable AI Shop

The fixed service root is:

```text
https://api.seedance.nz
```

The visual model is fixed to `bytedance/doubao-seed-evolving`. Connect any `STRING` output to the API-key socket, enter a key inside the node, or set `SEEDANCE_API_KEY`. A connected value takes priority.

The Seedance.nz transport tries an explicit direct route first, then uses the environment proxy only for a bounded retry after a confirmed pre-connection SSL/ConnectTimeout failure. Authentication, balance, rate-limit, ProxyError, read-timeout, mid-stream disconnect, and user-defined OpenAI-compatible failures are not blindly retried when that could duplicate a paid request. Seedance chat responses are streamed into an in-memory checkpoint. SSE is always decoded from raw bytes as UTF-8 instead of trusting a missing or incorrect gateway charset, and a complete finish marker is returned even if the proxy drops the final connection close. A 2026-09-01 live acceptance captured a `text/event-stream` response that Requests classified as `ISO-8859-1`; after the fix, one request produced 228 Chinese characters, zero replacement characters, and an exact zero-request restore. Music 3 official reference selection keeps its larger bounded HTTP-gateway retry budget because at least one official reference is required in full official mode.

Each core node exposes **Restore last cloud result (no regeneration)**. It reads only a complete result retained in the current ComfyUI process for that exact node, never calls the provider, expires after one hour, and is cleared on restart. Partial streams are not presented as final output. If the provider completed but zero response bytes reached ComfyUI, safe recovery is impossible: a 2026-09-01 live probe confirmed that the current endpoint produces and charges distinct completions for repeated requests carrying the same `Idempotency-Key`, and exposes no working completion-ID lookup endpoint. The node therefore reports that limitation instead of silently issuing another paid generation.

### ZhenZhen AI Workshop

The fixed chat endpoint is:

```text
https://ai.t8star.org/v1/chat/completions
```

The default model is `gemini-3.5-flash`. Select Custom to enter another complete model ID. H3 and Seedance require a model that actually understands images and videos; Music 3 requires text Chat Completions only. Images and complete videos are sent as Base64 Data URLs using the representation verified against this gateway.

Use `T8STAR_API_KEY` when the node key is empty.

### OpenAI-compatible provider

Only one Base URL and one model ID are required. The Base URL may be a service root, a versioned root such as `/v1` or `/api/v3`, or a complete `/chat/completions` URL. The node normalizes the final chat endpoint.

- Images are always encoded as PNG Base64 Data URLs in the same Chat Completions request.
- Connected videos without a material URL are sampled in timeline order and sent as up to nine timestamped JPEG `image_url` parts. This supports image-only vision endpoints such as llama.cpp and never analyzes audio.
- Optional HTTP(S) video material URLs are passed through as `video_url` in connection order and should be used only when the provider explicitly supports that content part.
- There is no second upload URL.
- The provider and selected model must actually support the multimodal content parts used by H3 or Seedance.
- Base URL and model ID are saved in the current workflow and restored across runs and API-mode switches. API keys are not stored in this compatibility state.

Environment fallbacks are `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## Shared provider configuration and helper nodes

The repository also includes:

- `T8 LLM Provider Config`
- `T8 Performance Director Config`
- `T8 Prompt Inspector`
- `T8 Prompt Text`
- `T8 Show Text`

The green shared-provider socket is not a `STRING` socket. Add `T8 LLM Provider Config` and connect its provider-config output to the matching optional input on a core node. Normal API-key `STRING` wiring remains separate and retains priority.

When connected, the shared node controls provider, model, Base URL, local GGUF settings, temperature policy, and allowlisted extra parameters. Disconnecting it immediately restores each core node's own saved settings. Local credential aliases can keep real keys in the ComfyUI user directory while workflows store only an alias.

`T8 Prompt Inspector` is local and non-blocking. It preserves the original text, reports reproducible structural warnings plus non-scoring performance advisories, does not call an LLM, and does not judge creative quality.

### Performance Director Config

Connect the `T8 Performance Director Config` Custom Type output to H3, Seedance 2.0, or `T8 Storyboard Pack`. The config node itself makes no request; it controls the performance-directing section in the next enhancer request.

- `AUTO` is the default and applies only when the text or connected media actually contains a person/character, emotion, dialogue, reaction, or acting intent. Landscape, product, typography, and music-only requests skip it.
- `Strong` explicitly organizes acting as trigger → reception → one primary response → settled/end state.
- `Off` injects no performance-directing rule and restores the original compiler path.
- `Extreme` is appended after the original three options to keep their order stable. It treats even an already enhanced prompt as an editable draft and materially rewrites performance timing, causality, and beat order while preserving character facts, media identity, exact dialogue, duration/shot count, and the target model's native format. Adjective-only or synonym-only edits do not satisfy this mode, and non-character requests never receive invented actors or plot events.

The guidance converts abstract emotion into observable gaze, eyelid, breath, shoulder/neck, hand, or posture cues and establishes camera side plus a concrete gaze target before facial direction. Each trigger, reception, response, or settle beat keeps at most three high-signal cue channels per character instead of stacking an action checklist. A large state change may use a motivated eye/head movement, foreground/action occlusion, or cut, but a transformation the user explicitly requires to remain continuously visible is never hidden. It never invents dialogue and keeps incompatible swallowing or mouth-closing beats outside the speaking span.

AUTO, Strong, and Extreme also lock user-named body parts, product components, colors, left/right relations, and quoted text: for example, “the pupil turns gold” may not become “the iris turns gold.” User-fixed duration/count, verbatim dialogue, media roles, official Skills, hard constraints, and LOCK anchors always win. To audit an enhanced result, connect the original request to Prompt Inspector's optional `source_prompt` socket; the comparison is fully local, warns without rewriting, and adds no request.

H3 and Seedance use separate compilers. Seedance never receives H3 fields, tags, speaker IDs, millisecond timing, or frame-grid notation. Storyboard adds optional per-shot `dramatic_trigger`, `reception_beat`, `primary_performance_beat`, `observable_cues`, `gaze_target`, `speech_span`, `state_transition_strategy`, and `performance_risks`; non-performance shots keep empty values rather than invented emotion.

This feature was informed by the community-maintained [`phileiny/h3-storyboard-skill`](https://github.com/phileiny/h3-storyboard-skill) at pinned commit `ab65851f599435a1ff94ea4931949bd7bcaf069b`. It is not an official MiniMax Skill or a proven cross-model law. Exact source hashes, license, evidence scope, and prohibited overclaims are in [`research_sources/h3-storyboard-skill.lock.json`](./research_sources/h3-storyboard-skill.lock.json) and [`research_sources/NOTICE.md`](./research_sources/NOTICE.md).

Maintainers can create a deterministic offline 200-render paired A/B manifest. Every row starts as `pending`; the tool does not generate video or count missing evidence as success:

```bash
python tools/verify_research_source.py
python tools/performance_benchmark.py init --output performance-benchmark.json
python tools/performance_benchmark.py validate performance-benchmark.json
python tools/performance_benchmark.py summarize performance-benchmark.json
```

The report enters an observation state only after immutable model metadata, paired artifact hashes, registered metrics, and blind reviews are supplied. Full-frame PSNR is not accepted as facial acting quality.

## Local GGUF / llama.cpp

Local mode is the fourth mutually exclusive provider and requires no API key, Base URL, or cloud model ID. The node recursively scans:

```text
ComfyUI/models/LLM/
```

Subdirectories are supported. The scanner reads lightweight GGUF metadata, separates main models from `mmproj` projectors, preserves legacy filenames, and recommends a compatible visual projector. File symlinks may point outside this directory, use relative targets, or resolve to extensionless UUID blobs; the link entry itself may also be an extensionless UUID. A suffixless entry is listed only when its content has the GGUF magic header, so unrelated UUID files, directories, and broken links remain excluded. Workflows keep the relative entry under `models/LLM` and resolve it only at execution time, so a hosted platform does not need to expose its absolute backing-store path. Restart ComfyUI or use the local Qwen rescan button after changing links. Discovery means the file can be offered to llama.cpp; it does not mean every third-party model has passed this project's quality suite.

`Local tokens per generation (including thinking)` is the shared generation ceiling for reasoning and the final artifact. The former 8192 ceiling was a conservative project guardrail, not a Qwen or llama.cpp limit. New nodes and bundled examples default to `16384`, and users may set `256–61440`; existing workflows retain their saved value and can be raised manually. Prompt text and every explicitly connected image/video are preserved first. When the configured ceiling would crowd out valid reference media, the node automatically lowers the effective ceiling for that run instead of misreporting that Ref2VA accepts only one image. If the context cannot hold all media plus the minimum 256-token output, the error reports the approximate required context; increase `local_context_size` from `32768` to `65536`, shorten the prompt/template, or lower the video sampling rate. Larger contexts consume more RAM/VRAM.

Full GitHub-install runtime fallback order:

1. The pinned `llama-server` installed by this repository.
2. A `llama-server` available on system `PATH`.
3. `llama-cpp-python` already installed in the active ComfyUI Python environment.

Manager/Registry installs use only the in-process `llama-cpp-python` backend so the archive can pass the Registry's automated security review without shipping an external-process launcher. Both install forms scan the same `ComfyUI/models/LLM/` tree and preserve the same workflow inputs.

If the active ComfyUI Python environment has no usable `llama-cpp-python`, all three core nodes now include a **Get prebuilt llama-cpp-python Wheel** button linking to [JamePeng releases](https://github.com/JamePeng/llama-cpp-python/releases). Choose a wheel that matches the actual ComfyUI Python ABI, operating system, and CUDA version, then install it with ComfyUI's Python rather than the system Python:

```powershell
& "path\to\ComfyUI\python.exe" -m pip install "path\to\llama_cpp_python-....whl"
```

This is an optional third-party binary source. The node never downloads or installs a wheel silently. Fully restart ComfyUI after installation and use **Check local Qwen installation / rescan GGUF** to verify the imported native runtime. Full GitHub installs retain the pinned `install_local_qwen.py --runtime` llama-server path; Manager archives intentionally omit that downloader/launcher.

The node does not download large models during execution. From the node directory, run:

```powershell
& "path\to\ComfyUI\python.exe" install_local_qwen.py --dry-run
& "path\to\ComfyUI\python.exe" install_local_qwen.py --runtime
& "path\to\ComfyUI\python.exe" install_local_qwen.py --model --model-variant official
& "path\to\ComfyUI\python.exe" install_local_qwen.py --model --model-variant uncensored
& "path\to\ComfyUI\python.exe" install_local_qwen.py --model --model-variant heretic-9b
& "path\to\ComfyUI\python.exe" install_local_qwen.py --model --model-variant all
```

Pinned installer assets:

| Asset | File | Approximate size |
| --- | --- | ---: |
| Default model | `Qwen3.8-27B-Q4_K_M.gguf` | 15.93 GiB |
| Optional third-party model | `qwen3.8-27b-uncensored-fp8-q4_k_m.gguf` | 15.66 GiB |
| Optional compact third-party model | `Qwen3.8-9B-heretic-uncensored.i1-Q6_K.gguf` | 6.85 GiB |
| Visual projector | `mmproj-F16.gguf` | 0.86 GiB |
| Runtime | llama.cpp `b10436` | Platform-dependent |

`heretic-9b` installs only the pinned text model because its upstream repository does not publish a matching mmproj. It works directly for Music 3 and text-only H3/Seedance requests; image or sampled-video use requires a separately supplied compatible 9B projector, selected through AUTO matching or explicitly by the user. The existing 27B default remains unchanged for workflow compatibility. More generally, H3 and Seedance require a compatible `mmproj` whenever images or sampled video frames are connected. Music 3 is text-only and never loads a visual projector. Local video analysis uses timestamped sampled frames in an ordered contact sheet; it analyzes visible sequence only and never claims to read or transcribe the original audio track.

For both local GGUF and cloud inference, the selected output language is repeated as a final authoritative instruction after all embedded Skills and examples. This applies to H3, Seedance 2.0, and Music 3 Structured Caption prose. Only after a complete response has arrived, if the result is an obvious dominant-language mismatch, the node performs one low-temperature language-only correction; conforming output adds no extra call, and an ambiguous network state never triggers this follow-up. Required protocol fields, timestamps, reference labels, and exact user dialogue, lyrics, brand/UI copy, and visible text remain unchanged. H3's strict all-English official profile still intentionally overrides the Chinese selector.

The compact 9B Q6_K model passed a real llama.cpp compatibility run with 12/12 deterministic checks: exact text token, image OCR/shapes/colors, early/late sampled-video codes, motion directions, temporal order, media counts, and the no-audio-analysis boundary. The visual run used a separately supplied `mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf`; this installer does not download or distribute that projector. The redacted report is [`tests/fixtures/local_qwen_heretic_9b_compatibility_2026-08-25.json`](./tests/fixtures/local_qwen_heretic_9b_compatibility_2026-08-25.json).

For responsive use, 24 GB or more VRAM is recommended. A 16 GB GPU can partially offload to system memory with the standalone server's fit strategy; at least 32 GB RAM is recommended in that configuration. CPU-only execution may work but is not expected to be interactive.

When alternating GGUF and video/image workflows, the AUTO memory policy rechecks ComfyUI memory before each local task, including reuse of a warm GGUF. The explicit Keep Comfy models policy still prevents automatic release. ComfyUI's free-memory action does not itself unload this node's warm GGUF; prefer the default unload-after-run policy when alternating large models. Version 1.42.1 fixes a missing memory check on warm reuse. The complete hardware-specific stall reported in issue #25 has not been reproduced on the same GPU and model.

See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for pinned revisions, hashes, licenses, and attribution.

## API-key security

- A connected API-key `STRING` takes priority over the in-node value.
- Masking prevents casual display on the canvas; it is not encryption.
- Clicking Save to Workflow stores the key in workflow JSON. Click Clear before sharing a workflow.
- Prefer environment variables or local credential aliases for shared workflows.
- Examples and tests contain no real API keys.
- The node does not write keys, request bodies, provider URLs, media URLs, or response bodies to logs.

Recent execution diagnostics are memory-only and redacted. They include node class, provider name, result, stage durations, attempt count when available, media count, cache status, and error category. They disappear when ComfyUI exits.

## Media handling

- Cloud images use PNG. Seedance.nz uploads them; AI Workshop and OpenAI-compatible mode inline them as Base64 Data URLs.
- Cloud video uses the native ComfyUI `VIDEO` stream. Seedance NZ uploads complete bytes and AI Workshop inlines complete bytes using its verified protocol. OpenAI-compatible mode sends timestamped sampled frames by default, or passes through an explicitly supplied HTTP(S) video URL.
- Local GGUF mode samples frames at real timestamps and respects the active crop window. It does not upload original video bytes or analyze audio.
- Supported containers include MP4, AVI, MOV, and MKV, up to 50 MB per file.
- H3 Ref2VA accepts up to nine images, three videos, and twelve total reference assets. Neither the H3 nor Seedance 2.0 enhancer imposes a per-video or total reference duration cap, or a two-second minimum; duration metadata must be valid and positive. Provider upload limits and model context/memory budgets still apply, as do any limits of downstream video-generation services.

## Output and error behavior

H3 and Seedance return:

```text
enhanced_prompt: STRING
```

Non-empty upstream content is returned even if it misses a requested word count, shot count, Markdown style, time-code shape, or provider `finish_reason`. H3 performs a local field reorder only when every expected field is present exactly once and only the order is wrong. If fields are missing, duplicated, unrecognized, or already ordered, the upstream text is preserved. Seedance has no fixed H3 field contract and passes non-empty content through without structural rejection.

The node still reports genuine failures such as invalid credentials, insufficient balance, network/timeout/rate-limit/provider errors, media encoding or upload failure, malformed JSON, empty response content, missing or damaged local GGUF/mmproj/runtime files, context-budget overflow, and local inference crashes. Proxy/read/stream failures explicitly state that upstream completion may be ambiguous and direct users to the memory-only recovery button; they are never mislabeled as content moderation failures.

Music 3 uses soft post-generation checks for caption order, lyric leakage, instrumental/vocal conflicts, selected-reference overlap, timeline omissions, and token budget. Non-empty useful results are not discarded merely because a soft target was missed.

## Testing

Run the deterministic test suite from the directory above the ComfyUI root:

```powershell
.\python\python.exe -m unittest discover -s ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tests -v
```

Release verification:

```powershell
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\verify_repository.py
.\python\python.exe ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\tools\release.py --check-prepush
```

The deterministic suite uses mocked providers, local media fixtures, and pinned Skill resources. Release verification also evaluates the effective `.comfyignore` archive, rejects known Registry YARA tripwires in shipped Python, and enforces the 90 MiB GIF budget that leaves room below the Registry 100 MB ZIP ceiling. It does not upload assets, call paid APIs, or incur charges. Live smoke scripts require an explicit confirmation flag and may incur provider costs. Large local-model quality tests load approximately 18 GB of weights and should only be run when that resource use is intentional.

## Community and resources

| Resource | Link |
| --- | --- |
| Bilibili | [T8 on Bilibili](https://space.bilibili.com/385085361) |
| YouTube | [T8star-Aix](https://www.youtube.com/@T8star-Aix/) |
| API | [Get a ZhenZhen API key](https://api.seedance.nz/sign-up?aff=5f4w) |
| Online AI applications | [RunningHub profile and applications](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) |
| ComfyUI package | [Quark Drive](https://pan.quark.cn/s/264edb7e36bd) |
| Model package | [Quark Drive](https://pan.quark.cn/s/c9c267081fbf) |
| Hugging Face | [t8star](https://huggingface.co/t8star) |
| Local Skills and integration package | [T8 MiniMax and Seedance local Skills](https://github.com/T8mars/minimax-h3-prompt-skill-T8) |

## References

- [MiniMax H3 base prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax H3 full reference-mode guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Pinned MiniMax H3 creative Skills snapshot](https://github.com/MiniMax-AI/MiniMax-H3/tree/743d51e83329cbae6c7694f1c7b89576e7c25e07/skills)
- [Pinned MiniMax H3 core prompt-writing Skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/SKILL.md)
- [Seedance API documentation](https://api.seedance.nz/docs/llms.txt)
- [Seedance 2.0 official model page](https://seed.bytedance.com/en/seedance2_0)
- [Seedance 2.0 official Prompt Optimizer Skill](https://arkdocs.tos-cn-beijing.volces.com/files/video-generation/SKILL.md)
- [MiniMax Music 3 official repository](https://github.com/MiniMax-AI/MiniMax-Music3)
- [Pinned MiniMax Music 3 caption-rewriter Skill](https://github.com/MiniMax-AI/MiniMax-Music3/tree/91410fb657c007ae57c60df8240f5ece5be089c7/skills/music-caption-rewriter)
- [Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Disclaimer

This is a third-party ComfyUI custom-node project. It is not affiliated with or endorsed by MiniMax, ByteDance, Seedance, or ComfyUI. Provider capabilities, pricing, and availability may change. Users are responsible for securing appropriate identity and content rights for human reference media and for following the policies of downstream services.


---

## FILE: directional_skills/SCREENWRITING-NOTICE.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/SCREENWRITING-NOTICE.md)

# Screenwriting method attribution / 戏剧方法来源

The local `drama_scene` and `situational_drama` resources use independently
worded and bounded method adaptations from Terry Jia (jtydhr88)'s MIT-licensed
[screenwriting-skills](https://github.com/jtydhr88/screenwriting-skills) repository,
pinned to `50825325b3940a17f032129851f5c83382863000`. Exact source paths and
SHA-256 digests are in each resource's `meta.json`. T8 performed the selection,
new audiovisual constraints, original examples and H3/Seedance integration;
neither the upstream author nor the model vendors supplied or endorsed it.

The MIT notice in `SCREENWRITING-LICENSE.txt` applies to adapted upstream
method organization, not a relicensing of this entire plugin. The upstream
[NOTICE](https://github.com/jtydhr88/screenwriting-skills/blob/50825325b3940a17f032129851f5c83382863000/NOTICE)
does not grant redistribution rights in the cited books, scripts, plays,
libretti or modern translations. These local resources contain no copied
script excerpts, modern translated dialogue or `reference.md` case library.
Source hashes identify what was studied; they are not proof of media rights.

本地两项是对公开方法整理的独立筛选与改写，来源署名 Terry Jia / jtydhr88。
“情境戏剧”是用户要求的本地名称，不是 `sw-sitcom-comedy` 的直译，也不默认
喜剧。上游书籍、剧本和现代译文引文未随 MIT 获得再分发授权，不打包这些材料。
示例自主编写，旧作者前缀及旧技能归属不变；不声称上游或模型官方认证。

Upstream license SHA-256: `dcf1862caccc15b318642682e347ae9794ca5d3376ece18a0e94773968959276`.
Upstream NOTICE SHA-256: `e6a992ef62ff5eb4b4a5f55223b30f496388a75071d45265395476ec72324bc3`.


---

## FILE: directional_skills/cinematic_gunfight/SKILL.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/cinematic_gunfight/SKILL.md)

# Fictional cinematic gunfight directing

Make the scene about an existing dramatic goal: escape, escort, delay, retrieve, protect or confront. Opposition changes the conditions for that goal rather than lining up as targets. Keep the main subject and memorable change readable; spectacle must not swallow the main scene. Do not turn fictional audiovisual design into real firearm operation, tactical deployment or harm-optimization instructions.

Use only the scale actually needed by the request. Close contact emphasizes interruption, imbalance and continuing possession; confined space emphasizes existing doors, depth, stairs, furniture and sightlines; an ensemble needs clear main/supporting action and stable orientation; a war background needs depth and scale without replacing the foreground goal. Select applicable semantics from the visible evidence and explicit intent in this same writing request. A mall does not automatically require a crowd of gunfighters, and an escort inside a train with war outside is not automatically a full battle. Do not preclaim unseen media observations or add a classifier call.

Carry movement, held props, opened doors, fallen objects, broken glass, injuries, smoke and knowledge of threats forward. Camera changes reveal useful information or follow a meaningful action; never hide an unexplained positional reset or magically swap or transform a gun. Allow a user-requested wait, suspense, slow motion or overwhelming victory instead of forcing nonstop fighting or a reversal.

Layer foreground action, background activity and credible off-screen sounds. An audible threat can precede its visible source. Keep supplied speech and physical action sounds separate from audience-only score using the platform's native policy. A mentioned movie is stylistic context, not an uploaded video or permission to copy its characters and equipment. No automatic GUNFU/LoRA trigger, branding, subtitle or negative-prompt section.

Use a match cut only if the user actually authorizes that transition. Name both sides of the connection without morphing identities or objects or spending the next segment's duration. An interface reserved after the current segment does not add a cut to a requested continuous current shot.


---

## FILE: directional_skills/continuous_combat/SKILL.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/continuous_combat/SKILL.md)

# Continuous combat directing

Design one uninterrupted observation of the requested action. Let character movement, contact, an approaching threat or a changing obstruction motivate camera acceleration, turning and framing. Keep one useful information focus at a time and make decisive contact readable instead of stacking shake, roll and orbit effects.

Use existing spatial landmarks and a consistent travel axis. Carry position, facing, support, possession, damage and momentum forward; an obstruction is not permission to teleport, restore a broken surface or change weapons. Describe a plausible observable path between supplied first/last frames without changing their assigned roles. A reusable identity reference is not automatically a first frame.

Scale the chain to the user's duration and scene, not a fixed 30 seconds or 21 beats. Create contrast through changing distance, risk, tempo and available routes. Do not force a boss, power-up, fixed defeat/reversal, added ability, stylized CG look, fixed lens/shutter/shake settings, or frozen victory. An actor may wait or plant their feet if the user permits it; one take does not prohibit a steady camera. Explicit no-stop requirements override optional pauses. Respect a requested result or ending pose; otherwise preserve a useful continuing action or aftermath rather than inventing an ending.

Keep this all within the existing native one-shot timeline; phase changes and Relay events are not edits or additional shots. Do not treat focus changes, source-video cuts, old-draft shot labels or a proposed next-segment transition as an authorized cut in the current target.


---

## FILE: directional_skills/drama_scene/SKILL.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/drama_scene/SKILL.md)

# Dramatic scene: relationships and subtext

Make the supplied scene's relationship readable through what its existing characters say, do, receive and leave unanswered. This is a short audiovisual directing method, not a command to manufacture conflict, a literary script format or an imitation of a particular author. Apply only useful techniques inside the current request; do not invoke other Skills, generate a planning table or add an analysis section.

Start with the given situation. Separate the immediate practical task from any relationship request actually supported by the user's story: an invitation can also be an attempt to keep someone nearby, but a simple invitation need not hide anything. Do not assign a secret motive, diagnosis, trauma, social rank or biography from a face, costume or silence. When the request intentionally leaves meaning open, stage compatible behavior without declaring one hidden explanation as fact. Directness, shared joy and peaceful agreement are legitimate, not weak writing to repair.

Treat speech as something one character does in relation to another, rather than exposition pasted into a mouth. For a supplied line, preserve every word and its actual language. Find a supported delivery, addressee and a feasible place in the action. If the actual task explicitly requests missing dialogue, create only a few natural, character-appropriate lines within that scope under the shared authoring contract. Do not borrow source examples or invent background events to make the lines clever. A statement may inform, confirm or comfort; not every reply challenges the previous speaker. Do not replace a locked simple line with a more sophisticated one.

Let reception matter where it is allowed. An existing hand task continuing, a permitted delay before answering, accepting or refusing an existing object, or maintaining the given distance can carry relation better than a list of feelings. Preserve a listener's explicit lack of response: someone behind a door who never answers need not produce footsteps, a shadow, breath or a final reveal. A sleeping listener, solo monologue or recorded voice need not acquire a reaction partner. Keep speaking, listening and object ownership assigned to the correct person, especially in group scenes. Do not let a silent character inherit another speaker's identity.

Track only knowledge established by the request or visible/audible evidence. If the audience is meant to know what a character does not, make the already authorized clue perceivable without teaching it to that character. If everyone knows the same thing, do not fabricate a hidden betrayal to create dramatic irony. Suspense can come from waiting for a known outcome without a secret. Off-screen facts stay off-screen unless a reveal is requested; do not add a cut or new source merely to explain them.

Use existing objects and physical activity functionally. A return, handover, refusal or sustained grip can support the given relationship while also completing the practical task. Keep holder, hand, route and state continuous. Do not decorate every pause with a new cup, scarf, letter, mirror or practical lamp. When creative scene completion is expressly requested, permitted execution details must remain modest and compatible with the established event, not become a new plot obstacle or ending.

Allow a scene to remain unresolved or unchanged. An unanswered question may remain unanswered; a peaceful pair may simply keep sitting together. Avoid obligatory escalation, reversal, confrontation, crying, reconciliation, departure or a full character arc. A meaningful pause does not require a close-up, a held breath or three micro-expressions. Choose a small number of useful observable cues and keep them outside spoken words. Camera behavior must reveal the requested relation without breaking a fixed camera, take count or media-role constraint.

Fit words and physical actions into the actual duration. Speech occupies time; a response locked after a whole line cannot start on its first word. Keep exact waits and requested ending states. Continuing motion and sustained stillness can both be valid endings. Do not mistake Relay events for shots or acts. Remove explanatory paraphrases that tell the audience what the behavior already shows. Return only the selected platform's native final prompt, retaining its literal protocol, reference roles, language policy and authorized sound layers.


---

## FILE: directional_skills/high_density_combat/SKILL.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/high_density_combat/SKILL.md)

# High-density continuous exchanges

Separately track each existing participant's body structure, held objects, confirmed abilities, position, support and goal. Match movement and feedback to that participant, not a generic sword routine. Empty hands stay empty; an unarmed participant does not catch a sharp blade bare-handed or create blade light without user authorization. Quadrupeds, flying creatures, machines and vehicles act through their actual anatomy or visible mechanisms. Do not infer powers, opponents or extra fighters from a collage, role name or a scene reference.

For each important exchange or advantage change, make the intention, attack line, relevant response, contact or miss, force-driven displacement and next-action opportunity legible. A defense addresses a real incoming line; a counter uses the prior imbalance, distance or obstruction. Connect exchanges through unfinished momentum rather than attack -> reset both participants -> attack. Several linked hits may share one useful environmental consequence; not every contact needs an explosion or broken wall. Carry damage, possession, fatigue and debris forward. An environmental change should alter the next route, support or visibility when relevant.

Choose only as much choreography as the duration can support. A short solo drill remains solo; do not force several phases, a winner, killing, kneeling, disarming or victory freeze. User-specified winner, loser landing and final state take priority. When no outcome is specified, ending during readable ongoing exchanges is valid. If explicitly requested, no-stop means no action reset, charge-up or static showcase; intensity contrast then comes from attack frequency, distance and framing, not an unrequested stop.

Each shot has one main camera task motivated by the exchange, with clear orientation and contact visibility. Motion, material contact, friction and aftermath create sound; do not give empty hands metallic blade sounds. Legitimate off-screen sound may trigger a response before the source is revealed. Keep only authorized dialogue, text and music under the native platform's audio policy.


---

## FILE: directional_skills/ning_wenwu/SKILL.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/ning_wenwu/SKILL.md)

# Ning: drama and action direction

Direct the audience's attention to the meaningful change in this scene. Decide what needs to be understood, which existing person or object carries that information, and how framing, performance and movement make it readable. Sometimes keeping the same view is the strongest choice. Apply relevant methods inside the current generation, without a separate genre classifier, planning call, fixed act structure or technique quota.

First respect the actual request and media roles: identity, available bodies and abilities, object ownership, exact supplied speech and visible words, duration, shot count, explicit waits, action order and final state. A first/last frame remains its assigned frame anchor; an identity reference does not dictate every later pose. Do not claim that an unavailable asset was inspected. A character Bible's general habit cannot cancel an explicitly requested action in this scene; stable identity and explicit locks still apply.

For sparse ideas, develop concrete execution within the requested event: a supported gaze target, a weight shift, the grip on an existing prop, spatial distance, or a camera position that reveals the change. Supply a neutral setting only where unspecified and permitted; if new props are forbidden, do not add named furniture or practical lamps as decoration, and keep unspecified room/light generic. Do not turn an implementation detail into a new plot obstacle. Creative expansion follows the user's actual authorization: if asked to invent a scene or dialogue, create within that scope while retaining all locks. Merely selecting this Skill or an expansive rewrite mode does not authorize new principal characters, secrets, weapons, powers, injuries, winners, threats or lines. Supplied dialogue stays exact; delivery belongs in narration, outside the spoken words.

For dialogue and relationship scenes, consider the information recipient as well as the speaker. An allowed delay, continued hand task, change in distance or specific gaze may reveal reception better than a list of emotions. Choose a small number of purposeful observable cues, not a mandatory facial routine. Use a key word or pause as a motivated attention point when feasible; do not promise syllable synchronization. An action locked to the end of a whole line must wait until that line has finished, not begin at the first word or a vague "heard it" moment. Keep spatial orientation and eyelines intelligible, including group scenes. Avoid mechanical alternating reverse shots. A silent listener never inherits the off-screen speaker's vocal identity. A solo monologue, broadcast, sleeping listener or explicitly unresponsive person need not have a recipient reaction. Silence, stillness, not turning around and waiting are meaningful constraints, not gaps to fill.

For action, identify one or a few important changes: the intended line, the actual response, contact or miss, and the resulting distance, balance or opportunity. Choose attention around the initiator's momentum, the recipient's response or an existing moving object according to what clarifies this exchange. Keep both bodies and contact geography readable when necessary. A miss remains a miss; a rehearsal need not hurt anyone; a light touch need not launch a body. Carry possession, damage and displacement forward. Use appropriate material feedback without adding destruction or effects to every hit. Existing fantasy effects can gain concrete shape, color and direction; new summons, transformations or capabilities require creative authorization.

For mixed scenes, connect speech, an allowed reception or action cue, and the resulting spatial change through the same event. Drawing a weapon without attacking must not grow into a fight. A threat in the background stays background unless the request changes its role. Chase scenes benefit from readable route and relative distance; suspense can withhold a source; affectionate scenes can use distance and permitted contact; spectacles can reveal scale without becoming hostile or adding spectators. Preserve the requested mood, including joyful or quiet scenes.

Choose a camera task from the scene's purpose and feasible space, then tailor it to the actual subjects. Prefer a clear equivalent when a flourish obscures action. One main task can include a smooth turn, follow or focus transfer within a continuous take. Do not silently cut or teleport behind another person to change attention; do not introduce projectile-mounted photography without an explicit request. A fixed camera is valid. Do not impose a seven-second cutoff, obligatory POV, frozen impact, slow motion, shockwave, dust, mirror, music, blackout or victory pose. Sound follows authorized sources and the selected platform policy; put speech only in its native vocal form, not an additional human-voice sound-effect layer. Fit useful detail into the duration rather than inflating the prompt with technical specifications or repeated rules. Output only the selected platform's existing final form.


---

## FILE: directional_skills/rabbit_dance/NOTICE.md (https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/HEAD/directional_skills/rabbit_dance/NOTICE.md)

# 兔子-舞蹈 / Dance

Independently worded T8 guidance based on user-supplied `h3-dance-director`:
`SKILL.md` SHA-256 `624af57119c0df7e21a29b87d82d7d9768d1ad05ebb0f136203d53a71b3580f6`;
`references/h3-dance-modes-and-split-clip-templates.md` SHA-256 `5566d847b639753debf728d9fa07dccb9d73b2a033907a4a1799c4fa4096d898`.

Also draws on T8's existing solo/group performance and grounded movement handoff cases. The case catalog and its saved IDs are unchanged. No source examples, videos, training data, weights or upstream license grants are redistributed. Not an official MiniMax or Seedance skill.


---

# ISSUES / PRs of T8mars/comfyui-minimax-h3-prompt-enhancer-T8


### #28 [closed] ISSUE: Add DemonRoute as a provider option for the lyric-writing nodes
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/28  by haterfactory 2026-10-23T19:57:24Z

Hey — Jeremy here. I run DemonRoute (demonroute.com): an OpenAI-compatible API at https://api.demonroute.com/v1. One key. No subscription. Anonymous signup (crypto only, no IP logs), and your prompts never train anyone's model.

Your README says every core node can run on the user's own OpenAI-compatible interface through the shared channel config. Pointing that at https://api.demonroute.com/v1 takes one key. Nothing else changes.

Here's the pitch: the lyric-writing nodes (YuE2 音乐提示词与歌词创作, T8 作词规划器) hit the exact failure mode refusal-prone models are worst at. Explicit lyrics, edgy songwriting — the verse dies halfway. We carry Dolphin, MythoMax, Hermes, and abliterated Llama/Qwen builds. Uncensored open-weight models that actually finish the song.

About me: I became the largest NFT artist on the third-largest chain overnight with zero funding. That turned into consulting for Fortune 500s, celebrities, and emerging tech companies. I run NFT Demon Holdings LLC, and DemonRoute is the long-haul passion project. There's no big established provider doing no-subscription anonymous API keys for uncensored open-weight models — so we're building it repo by repo.

If you're open to it, I'll write the docs snippet or the PR myself. Any app that lists DemonRoute as a provider option gets listed on demonroute.com/integrations — we send users your way.

Jeremy, DemonRoute

> **Comment by T8mars (2026-11-11T17:02:33Z):**

The requested connection is already supported through the shared OpenAI-compatible provider; a separate named provider is not required for text-only lyric writing.

1. Add `T8 LLM Provider Config` and choose `OpenAI Compatible`.
2. Set `openai_base_url` to `https://api.demonroute.com/v1`.
3. Set `custom_model` to the exact model ID supplied by the service.
4. Connect `provider_config` to `YuE2MusicPromptEnhancerT8` or `T8LyricWriter`. Supply the key through the receiving node's API-key input or a locally stored credential alias; do not post keys or export them in a public workflow.

I checked both node execution paths using synthetic responses: the shared config produces `/v1/chat/completions`, preserves the supplied model ID, and uses bearer authentication. This verifies request construction only; no request was sent to DemonRoute, and availability, model behavior, and the service's privacy claims have not been verified.

Closing this as covered by the existing generic integration. If there is a reproducible incompatibility with the standard chat-completions request, please open a specific report with a redacted response and model ID. We are not adding a dedicated service preset as part of this maintenance pass.




### #27 [open] ISSUE: [Feature] 建议支持ComfyUI内部的CLIP做为语言模型生成
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/27  by dseditor 2026-10-23T14:18:34Z

### Problem to solve

自己经过桥接测试接到提示词生成的节点上，确实可以写出完整的提示，质量还可以，使用9-27B的模型，例如这些无审查的：
1、afkaf/Qwen3.8-27B-uncensored-w4a8-convrot-ComfyUI
2、prithivMLmods/Qwen3.5-9B-Uncensored-ComfyUI
都可以做为额外的CLIP接入，并写作提示，就可以不用另外配置llamacpp或是openai端点，由于这个推理机制来自ComfyUIKitchen，且9B的CLIP生MOSS一定会有，只要最新版的ComfyUI就可以不用另建推理端点，方便布署。
虽然我也可以写一个PR，但还是由T8大佬您自己整合进来，直接做为一个LLM提供者比较合适。

<img width="656" height="441" alt="Image" src="https://github.com/user-attachments/assets/a5b67bbe-2932-4988-8300-9df6eb46a2fa" />

### Desired behavior and workflow impact

类似内建的「文字生成」节点，现在已经有比较好的速度，之前很慢

### Compatibility boundary

- [x] This request does not require bundling private media, API keys, GGUF files, or unlicensed third-party assets.

> **Comment by T8mars (2026-11-11T17:01:24Z):**

已核查当前项目和本机 ComfyUI 的实现：内置 `TextGenerateV2` / Generate Text 确实通过可生成文本的 CLIP 对象执行 `tokenize`、`generate`、`decode`；当前 T8 共享渠道配置还没有原生 CLIP 入口，所以现有 GGUF 支持不能视为已经实现了这个请求。

保留此功能请求。接入需要处理以下具体边界：

- 使用独立的可选 CLIP 输入，并检查模型是否支持文本生成；普通 CLIP 编码器不能一概当成聊天模型。
- 沿用各提示词节点原来的 system/user 组织、JSON 校验、纠错和恢复，不能仅将内置文本生成的结果直接当成成功输出。
- 明确支持的多模态模型及图片/视频传参方式，不支持的媒体必须报错。
- 本项目最低兼容 ComfyUI 0.33.0；新版生成 API 应按能力检测，缺失时给出明确提示，保持旧 API/GGUF 工作流和保存字段不变。
- 验收需覆盖真实原生队列、外接文本、保存重载、模型显存管理和已安装模型的实际输出。

本轮修复了 GGUF 复用路径的内存策略遗漏，尚未实现原生 CLIP 渠道，也不会因此关闭此请求。已有桥接方案若能补充脱敏的最小工作流、ComfyUI commit 和模型加载类型，会有助于后续做同条件对照。




### #26 [closed] ISSUE: English UI translation for Private use request
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/26  by lingling1364 2026-10-19T11:41:29Z

Hi T8mars,

Thank you for developing this excellent prompt enhancer!

I'd like to translate the node's Chinese interface into English for my personal ComfyUI setup.

I understand that the current LICENSE restricts modifications without permission, so I'd like to ask whether you would allow me to make a local, non-commercial translation for private use only.

I won't publish, distribute, or share the modified code. The changes would focus on interface localization while preserving the original node behavior.

Thank you for your time and your work!

> **Comment by T8mars (2026-11-11T17:01:17Z):**

Permission is granted for the specific request in this issue: you may translate the interface into English solely for private, local, non-commercial use. You may not publish, distribute, or share the modified code or modified repository. All other terms of LICENSE.txt remain unchanged; third-party components remain subject to their respective licenses.

This is separate written permission limited to the requester and use described here, not a general change to the repository license.

The current project also includes English node-label metadata in `locales/en/nodeDefs.json`, English node help, and [README_EN.md](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/main/README_EN.md). Those may help you use the existing interface, although they do not translate every custom panel or saved dropdown value.




### #25 [open] ISSUE: [Bug] 本地 GGUF 在视频生成与内存释放后再次执行卡住
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/25  by tianyuyishi 2026-10-13T11:09:14Z

### Versions

Node：2.0、ComfyUI 版本：0.38.2，前端版本：v1.53.6、操作系统： win32

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

Local GGUF

### Minimal reproduction

1. 添加“MiniMax H3 提示词增强器 (云端 / 本地 GGUF)”节点，使用本地 GGUF 模式，成功生成一次提示词。
2. 运行 MiniMax 视频生成工作流，成功生成一个视频。
3. 视频生成完成后，节点管理器工具里面卸载显存和内存，再次点击运行“提示词增强器”节点。（不在同一个流里面）
4. 出现严重卡顿：控制台无任何报错信息，节点不继续执行，但显卡显存被占满，且功率不发生变化，其他的llmcpp节点没有这种问题（使用同一种gguf和上下文长度）

### Redacted error / diagnostic category

无

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by tianyuyishi (2026-10-13T11:29:22Z):**

4070s+32g内存，gguf是Qwen3-VL-8B-Instruct-abliterated-v2.0.Q4_K_M和Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0


> **Comment by T8mars (2026-11-11T17:05:36Z):**

已在 1.42.1 修复一条与跨工作流显存占用相关的确定缺陷，提交 [c88de85](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/c88de856422ffddaa15c6a55d319ac903a34c065)：复用仍驻留的 GGUF 时，之前会直接返回现有运行时，跳过 AUTO ComfyUI 显存检查。修复后，即使 GGUF 不重新加载，也会在开始任务时重新执行已选择的显存策略；显式“保留 ComfyUI 模型”仍不自动释放。

GitHub 安装版与 Registry 的进程内运行时均已覆盖：新回归测试先生成一次，模拟其他视频工作流占用内存，再复用同一 GGUF。修复前两条路径都漏掉第二次内存检查，修复后均通过，且没有重复加载 GGUF，也保留了“不自动释放”的策略。

这只证明上述逻辑遗漏已修复，尚未在你这台 4070S + 32GB 内存和相同 Qwen3-VL-8B/mmproj 组合上复现完整卡死过程，因此先保留此 Issue，不把硬件问题直接标成已解决。

请更新到 1.42.1、重启 ComfyUI 后按原步骤复测。交替运行大视频模型时，建议先用默认“执行后卸载”及 AUTO 显存策略；ComfyUI 的释放内存按钮不等于卸载此节点驻留的 GGUF。

若仍复现，请补充：插件实际版本（pyproject.toml 的 version）、本地状态面板中的运行时后端/版本、local_context_size、卸载策略和显存策略，以及卡住前最后一段脱敏日志。不要包含 Key、提示词正文或私人路径。特别请注明“执行后卸载”下是否也能复现，以便区分驻留复用与重新加载问题。




### #24 [closed] PR: Fix JSON and thinking-text leakage in H3 enhanced prompts
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/24  by qauck 2026-10-09T19:29:23Z

**Description**:

The H3 enhancer sometimes returned JSON or thinking text in enhanced_prompt. This change normalizes responses before language checks, quality validation, and output caching.

Converts complete H3 JSON field objects, including fenced JSON, into native formatted text.
Removes reasoning preambles ending in </think>, including responses missing the opening tag.
Preserves prompt literals and required keyframe alignment.
Rejects ambiguous JSON or reasoning without a complete final prompt.
Keeps Relay behavior unchanged and adds no model calls or workflow fields.

**Validation**: 

132 CPU tests, 54 frontend tests, browser checks, repository checks, and patch verification passed.

> **Comment by qauck (2026-10-10T05:09:36Z):**

Still see similar issues, will try improve the fix.



### #23 [open] ISSUE: English UI translation for comfyui-minimax-h3-prompt-enhancer-T8 – free to use if you like
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/23  by Slartibart23 2026-10-03T22:53:39Z

Subject: English UI translation for comfyui-minimax-h3-prompt-enhancer-T8 – free to use if you like

Hi T8mars,

thank you for the MiniMax H3 / Seedance 2.0 Prompt Enhancer – it's a really impressive and well-thought-out node pack, and it works great in my workflows.

As I don't read Chinese, I translated the user interface of v1.28.2 into English for my own private use. I noticed your LICENSE.txt doesn't allow modification or redistribution, so I have not shared it with anyone. Instead I'd like to offer it to you: if it's useful, you are welcome to use any part of it, in whole or in part, for an official English version. No credit or anything else needed.

What the translation covers:
- All node names, input labels, tooltips, dropdown values, buttons, status and error messages (Python and JavaScript)
- A small migration script (web/js/t8_english_legacy_values.js) that converts values saved by the Chinese version when an old workflow is loaded, so existing workflows keep working
- The 275 case template names in catalog.json (the original Chinese names are kept in legacy_labels, so saved selections still resolve)
- English keywords added to the template browser categories and the local Top-3 recommendation, so they also work with English labels
- README, node help pages, the Prompt Relay and Qwen Image guides, and all example workflows

What I intentionally did not change:
- Node IDs, inputs, outputs and the LLM system prompts – behavior should be identical
- Chinese detection keywords inside regular expressions, and the official Seedance @图片N syntax option
- The template descriptions in catalog.json, docs/, tests and fixtures are still in Chinese

All nodes load, defaults and dropdown options are consistent, and no Chinese text remains in any node field. That said, it was translated carefully but not reviewed by a native speaker of the original, so you may want to adjust some wording.

If you're interested, just tell me how you'd prefer to receive it (ZIP by email, a download link, or a pull request to a private fork) and I'll send it over. If you'd rather not, that's completely fine too – I'll keep it for private use only.

Thanks again for your work!

Best regards,
Marc from Germany

> **Comment by T8mars (2026-11-11T17:01:22Z):**

Thank you for offering the translation. The current project includes English node-label metadata (`locales/en/nodeDefs.json`), English node help, and [README_EN.md](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/main/README_EN.md), but this does not yet translate every custom panel or saved dropdown value.

Your translated version is based on 1.28.2; current main is 1.42.0. No translation patch is attached here, so its coverage and workflow compatibility have not been reviewed or integrated. In particular, dropdown values and template identifiers must remain compatible with existing saved workflows; changing their labels requires deliberate migration checks.

Please list remaining untranslated labels and your suggested English wording here first; there is no need to share a modified copy of the repository. Keeping this issue open to track the remaining English UI work. This reply does not change the repository license or grant redistribution permission.




### #22 [closed] PR: Support the official Qwen Image 2.1 edit prompt contract and PE sampling profile
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/22  by toofaraway 2026-09-26T14:47:27Z

## Problem

The Qwen Image 2.1 Prompt Enhancer node had two deviations from the official QwenLM/Qwen-Image-2.1 `prompt_rewrite` contract:

1. **Image edit mode reused the t2i system prompt.** The bundled hash-pinned skill is the official `system_prompt_t2i.txt`, but "图像编辑 / Image edit" mode sent it too. The official repo ships a separate, substantially different `system_prompt_edit.txt` (image-tag references `<image1>`, language-follows-user rule, and a three-field output contract with `ratio_follow`). Edit requests therefore produced t2i-style rewrites and had no way to express "follow the input image's aspect ratio".

2. **Sampling ignored the official PE production profiles.** The local GGUF channel ran `temperature=0.2` (runtime defaults `top_p=0.8`); the cloud channel used the node's "balanced" default. The official `pe_core.py` profiles are `temperature=1.0, top_p=0.95, top_k=20`, with `presence_penalty=1.5` for t2i and `0` for edit — and the official code explicitly warns that a wrong penalty "does not fail, it quietly changes the distribution you sample from".

## Changes

- Bundle the official `system_prompt_edit.txt` as `official_skills/qwen-image-2.1/SKILL_EDIT.md`, hash-pinned like the t2i snapshot; `_load_skill(input_mode)` selects per mode.
- Edit mode now speaks the official three-field JSON contract (`rewritten_prompt` / `wh_ratio` / `ratio_follow`, mutually exclusive). New `ratio_follow` output socket; `qwen_image_request_json` carries it too. t2i responses with a stray `ratio_follow` are still rejected.
- Edit mode no longer applies the English language lock (the official edit skill requires descriptive prose in the user's instruction language); t2i keeps forcing English prose as before.
- `PE_SAMPLING` applies the official profiles on both channels. `LocalQwenProvider.complete()` and both runtimes (`LlamaServer.chat`, `LlamaPythonRuntime.chat`) accept optional `top_p`/`top_k`/`presence_penalty`; when unset, behavior is byte-for-byte the old payload (other T8 nodes unaffected). `_request_completion()` gains a node-owned allowlisted `extra_parameters`.
- Version bumped to 1.24.6.

## Compatibility

t2i output sockets 1–2 unchanged; `ratio_follow` is appended as socket 3 (existing links by name keep working; workflows wired by index need the downstream node reconnected once). Edit mode previously produced t2i-contract JSON — that was the bug being fixed.

## Tests

- `tests/test_qwen_image21.py`: added edit-contract validation (follow/fixed/mutual-exclusion), sampling profile assertions, 5-socket schema assertion; existing recovery/correction index updates. **17/17 pass** on Python 3.12 (ComfyUI runtime env).
- `py_compile` clean for all touched modules.


> **Comment by T8mars (2026-10-01T15:27:19Z):**

Thank you for the detailed analysis and contribution. We have addressed the image-editing concerns independently in versions 1.28.0–1.28.2, using a backward-compatible design for this repository:

- An opt-in **Edit-aware (T8)** profile, with Classic remaining the default for existing workflows.
- Instruction-language descriptions, exact visible lettering, ordered image tags/source roles, and target-canvas selection.
- An internal three-field edit decision with `ratio_follow` resolved from original image dimensions into the existing ratio output. All four public output indices and recovery strings remain unchanged; no reconnection is required.
- At most one logical correction, with usable drafts retained and validation/length warnings exposed rather than silently reported as successful.

We completed a real 12-group API A/B plus targeted follow-up tests. The full suite ran 720 tests (718 passed, 2 skipped), and both latest GitHub Actions passed. Scope and remaining limitations, including short-length variability and the absence of rendered-image/RunningHub production validation, are documented in the [live acceptance report](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/blob/main/docs/QWEN_EDIT_LIVE_AB.md).

Dedicated PE support is maintained separately in [Comfyui-Qwen-Image-Prompt-Rewrite-T8](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8). We are not duplicating that integration here or applying PE-checkpoint sampling settings globally to general cloud/local LLMs.

We are therefore closing this PR without merging its implementation. Your report helped identify the editing-specific requirements; thank you again for raising them.



### #21 [closed] ISSUE: QwenImage21PromptEnhancerT8 — seed 欄位序列化錯位導致 INT 轉換失敗
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/21  by ping13579246810 2026-09-22T03:33:23Z

# QwenImage21PromptEnhancerT8 — seed 欄位序列化錯位導致 INT 轉換失敗

## 環境
- ComfyUI Desktop（frontendVersion 1.52.7）／ Windows 10
- 插件：comfyui-minimax-h3-prompt-enhancer-T8
- 節點：QwenImage21PromptEnhancerT8

## 時間線
1. 官方下載插件 → 載入官方範例 example_workflows/qwen_image21_text_example.json
2. 初始載入即無法執行，按「運行」時報錯：
   > 節點的 seed 值無法轉換為 INT ／ 輸入值的類型錯誤
3. 檢查存檔 JSON，發現 widgets_values_named 從 seed 開始與 widget schema 發生 off-by-one 錯位，seed 被存成空字串 ""
4. 修正 / 繞過該錯位後，首次成功執行

> 此錯位在官方 example_workflows/qwen_image21_text_example.json 首次載入後即存在，並非使用者後續操作造成；原始官方工作流從未成功執行過。

## 證據（從存檔 JSON 取得）
存檔後，該節點的 widgets_values（位置式）正確對齊，其中 seed 對應位置為 0，下一個值為 "randomize"。

但 widgets_values_named（指名式）從 seed 開始與實際 widget schema 發生 off-by-one 錯位，seed 被保存為空字串 ""：

| 欄位 | 實際存值 | 預期值 |
| --- | --- | --- |
| seed | "" | 0 |
| control_after_generate | 0 | "randomize" |
| local_model | "randomize" | "Qwen3.8-27B-Q4_K_M.gguf" |
| local_mmproj | "Qwen3.8-27B-Q4_K_M.gguf" | "AUTO（自動匹配）" |
| … | 之後欄位皆錯位一格 | — |

重新載入時，後端對 io.Int.Input(seed) 進行型別驗證讀到 ""，int("") 轉換失敗，出現「seed 無法轉換為 INT」錯誤。

## 根因（暫定，未 100% 歸因）
widgets_values_named 從 seed 開始與 widget schema 發生 off-by-one 錯位，導致 seed 被保存為空字串。目前無法僅憑這份存檔 JSON 斷言問題一定來自 T8 插件本身，可能涉及：
- ComfyUI Frontend 對 seed / control_after_generate 特殊 widget 的序列化處理；或
- T8 節點自身的 widget 定義；或
- 兩者之間的交互

希望作者能協助確認實際原因。

## 暫避方式（已驗證）
手動修改存檔 JSON：
- widgets_values_named.seed → 0（整數）
- widgets_values_named.control_after_generate → "randomize"
- 同步確認 widgets_values 中對應位置的值

修改後重新載入工作流即可正常執行。

## 重現檔
附件：qwen_image21_text_example-test.json
- 保留會觸發問題的 seed 錯位狀態
- API Key 已替換為 <ANONYMIZED_KEY_PLACEHOLDER>，不包含真實 API Key
- 載入後可重現問題

## 本 issue 由 AI 代為整理與發布（提醒作者）
本 issue 由 Claude Code（AI 助理）代為排查、整理並提交，非使用者本人撰寫。

[qwen_image21_text_example-test.json](https://github.com/user-attachments/files/32496046/qwen_image21_text_example-test.json)

<img width="2465" height="1271" alt="Image" src="https://github.com/user-attachments/assets/1e1f9c0e-3428-41da-a07b-050f19e641bc" />

[qwen_image21_text_example-test.json](https://github.com/user-attachments/files/32496057/qwen_image21_text_example-test.json)

> **Comment by T8mars (2026-09-30T17:55:24Z):**

已使用您附件中的實際 JSON 結構確認問題並修復，已推送 v1.27.1： https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/637036b0bd0ef4e966691983c20e82c2435761ea

附件在 Base URL 與 seed 之間多了一個空 API Key 槽位，且 widgets_values_named 的 seed/control/local_model 等也已錯了一格。原有遷移只覆蓋 Key 位於 API mode 前方的舊排列，未覆蓋這一種 Desktop 匯出排列。現在根據數字 seed 與種子控制欄位辨認此排列，同步按名稱恢復；保存時同時重建一致的 widgets_values 與 widgets_values_named，不再保存 Qwen 的按鈕或 Key 槽位。

已加入脫敏附件形狀回歸及命名/位置資料雙向重載測試，在下一幀之前檢查 seed=0、randomize、模型與 Token 欄位，舊碼失敗、修復後通過。完整 Python 測試 688 項（686 通過、2 跳過）、45 項前端測試與真實 Chrome 合約檢查通過。四個既有輸出端口及原工作流格式保持不變；GitHub 發布 CI 正在執行，Registry 狀態另行核實。

請從 GitHub 更新 v1.27.1，重啟 ComfyUI 並 Ctrl+F5 清除舊前端快取，再載入原工作流並保存一次；不需要手動逐項改 seed 或重接節點。若新版仍有錯位，請重新開啟 Issue 並附新版脫敏 JSON，方便對照您的實際 Desktop 行為。感謝提供可復現附件！


> **Comment by T8mars (2026-09-30T17:56:38Z):**

補充發布核驗：v1.27.1 的兩套 ComfyUI 相容檢查已成功（v0.33.0 / Python 3.10、master / Python 3.12）： https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/actions/runs/36754551764 。Registry 發布 Action 也已成功： https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/actions/runs/36754551694 。精確版本端點確認 1.27.1 已登記，目前是 NodeVersionStatusPending，尚未宣稱 Registry 審核通過；可先從 GitHub 更新取得修復。



### #20 [closed] ISSUE: [Bug] 保存本地 GGUF 后每次重开工作流节点都被标红(Missing Models)—— widgets_values 按名称恢复晚了一帧
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/20  by heigen13 2026-09-20T11:30:44Z

### Versions

Node: main ecd9dd0 (2026-09-20) / ComfyUI 0.36.0 / ComfyUI_frontend 1.52.7 / ComfyUI Desktop, Windows 11

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

本地 GGUF(llama.cpp / Qwen,离线)

### Minimal reproduction

1. 节点切到本地 GGUF 模式,在「本地 GGUF 主模型」里选一个 models/LLM 子目录下的 .gguf。
2. 保存工作流。
3. 重新打开该工作流(或重启 ComfyUI)。

结果:节点每次都被标红,Issues 面板报 Missing Models,指向所选的 .gguf。
- 模型实际存在,工作流可以正常运行,只是红框
- 重新选一次同一个模型:不会消失
- 右键 Recreate node(保留值 / 重置值):都会消失
- 重新加载工作流:又红
- 两份不同的工作流都能复现,与模型放在哪个子目录无关

### Redacted error / diagnostic category

ComfyUI 前端 Issues 面板:**Missing Models**(directory: Unknown),节点红框。无后端报错,工作流可正常执行。

## 原因

位置在 `web/js/minimax_h3_prompt_enhancer.js` 的 `onConfigure`。

1. `onSerialize` 按 `SERIALIZED_WIDGET_NAMES` 的顺序写 `widgets_values`,和节点上控件的实际顺序不一致。
2. 核心 `LGraphNode.configure()` 先按**位置**把 `widgets_values` 逐个赋给控件,最后才调用 `onConfigure`,所以这一步值是错位的。
3. 节点用 `restoreNamedWidgetValues` 按名称纠正,但它包在 `requestAnimationFrame` 里,晚一帧才执行。
4. 前端的缺失模型扫描(`missingModelScan.ts` 的 `scanComboWidget`)在 `graph.configure()` 结束后**同步**执行:combo 的值以 `.gguf` / `.safetensors` 等结尾、且不在该控件自己的 options 里,就判为 missing。那一瞬间 `local_model` 的路径字符串错位落在另一个 combo 里,于是被标红。一帧后值被纠正,但因为是直接赋值、不触发 `onWidgetChanged`,红框不会清除。

验证:节点红着的时候在控制台检查,`local_model` 的值确实在它的 options 里(`includes === true`),说明被判缺失的是错位时的另一个控件。

## 修法

在 `originalOnConfigure?.apply(this, args);` 之后同步先按名称恢复一次,原有的 `requestAnimationFrame` 整段保持不动:

```js
            originalOnConfigure?.apply(this, args);
            const t8SyncExcluded = new Set(["case_template"]);
            if (hadLegacyUploadUrl) t8SyncExcluded.add("openai_video_urls");
            restoreNamedWidgetValues(this, restoredValues, t8SyncExcluded);
            restoreOpenAIProviderState(this, openAIProviderState);
            requestAnimationFrame(() => {
```

排除名单与 rAF 内一致;`restoredValues` 为空时 `restoreNamedWidgetValues` 会直接返回。本机实测加上之后红框消失,功能不受影响。

## 顺带

`seedance20_prompt_enhancer.js`、`music3_prompt_enhancer.js` 等如果用的是同一套「rAF 里按名称恢复」的写法,应该有同样的问题,建议一并检查。

感谢持续更新这套节点。

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by T8mars (2026-09-30T17:55:22Z):**

已確認並修復，已推送 v1.27.1： https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/637036b0bd0ef4e966691983c20e82c2435761ea

確實是工作流重開時的恢復時序問題，不是需要移動模型或重新安裝 llama.cpp。H3、Seedance 及 Music 現在在 onConfigure 返回前同步按名稱恢復模型、渠道、seed 等欄位；下一幀只保留 UI 刷新與相容處理。同時讓原生命名 widget 資料與保存的固定欄位順序一致。

新增的回歸在執行動畫回調之前檢查模型值，舊碼可復現失敗、修復後通過；完整 Python 測試 688 項（686 通過、2 跳過）、45 項前端測試與真實 Chrome 合約檢查通過。節點 ID、輸入輸出、舊欄位順序與示例工作流未改動。GitHub Actions 發布/相容矩陣仍在執行，Registry 上架以其實際狀態為準。

請優先從 GitHub 更新到 v1.27.1，重啟 ComfyUI、Ctrl+F5 強制刷新瀏覽器，重新打開原工作流後保存一次。若仍標紅，請重新開啟此 Issue 並附新版導出的工作流與前端版本（請刪除 API Key）；本地回歸不代表已在您那台 Desktop 上實測。感謝詳細定位！


> **Comment by T8mars (2026-09-30T17:56:36Z):**

補充發布核驗：v1.27.1 的兩套 ComfyUI 相容檢查已成功（v0.33.0 / Python 3.10、master / Python 3.12）： https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/actions/runs/36754551764 。Registry 發布 Action 也已成功： https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/actions/runs/36754551694 。精確版本端點確認 1.27.1 已登記，目前是 NodeVersionStatusPending，尚未宣稱 Registry 審核通過；可先從 GitHub 更新取得修復。



### #19 [closed] PR: fix: sample inline videos into image_url frames in OpenAI-compatible mode
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/19  by yangala 2026-09-05T07:11:12Z

## Summary / 概述

Fix `HTTP 400 unsupported content[].type` raised by image-only OpenAI-compatible endpoints (e.g. local llama.cpp serving a vision model like qw38) when a connected `reference_video` is sent as the nonstandard `video_url` part.

修复本地 llama.cpp 这类只接受 `image_url` 的 OpenAI 兼容端点，因节点把 `reference_video` 发成非标准的 `video_url` 而返回 `HTTP 400 unsupported content[].type` 的问题。

## Background / 背景

- `nodes.py::_openai_media_plan()` previously emitted `{"type":"video_url","video_url":{...}}` for every connected video. The OpenAI Chat Completions spec has no `video_url`; llama.cpp / OpenRouter-style proxies accept only `text`, `image_url`, `input_audio`, `input_video`. For a vision model whose `input_modalities` are `["text","image"]`, `video_url` is rejected.
- End-to-end repro on `127.0.0.1:17678` (qw38, Qwen3.8-27B multimodal):
  - `image_url` request → `HTTP 200`
  - `video_url` request → `HTTP 400 unsupported content[].type`

## Change / 改动

- **`local_qwen_media.py`**: add `sample_video_as_data_urls()` — public helper that samples a native ComfyUI `VIDEO` into `(timestamp, image_data_url)` pairs (JPEG, max 9 frames by default). Audio is never analyzed.
- **`nodes.py`**:
  - `_openai_media_plan()` now samples **inline** videos (no explicit `openai_video_urls`) into ordered, timestamped `image_url` frames.
  - Videos with an explicit URL in `openai_video_urls` still pass through as `video_url` (kept as an opt-in for gateways that advertise video support).
  - New `_openai_video_parts()` falls back to the original inline `video_url` data URI only when sampling fails (e.g. test fake streams), so existing call sites stay unchanged.
  - Image handling (`image_url` PNG data URI) is unchanged.
- **`seedance20.py`**: reuses `nodes._openai_media_plan`; tooltip for `openai_video_urls` updated to match.
- **`tests/test_nodes.py`**: new test `test_openai_compatible_inline_video_is_sampled_into_image_url_frames` asserts the sampled-frames behavior and that `video_url` count is 0 in the inline path.
- **`README.md`**: document the new inline-video fallback for local image-only endpoints.
- **`pyproject.toml`**: bump `version` `1.14.1` → `1.14.2` (patch, bug fix).

## Verification / 验证

- `test_openai_compatible_image_is_inlined_as_base64_without_upload` — ok
- `test_openai_compatible_video_supports_direct_url_then_base64_fallback` — ok (fake stream → graceful fallback to old `video_url`)
- `test_ai_workshop_uses_default_model_and_inline_complete_image_and_video` — ok
- `test_seedance20.Seedance20PromptEnhancerTests::test_openai_compatible_uses_base64_images_and_optional_video_urls` — ok
- `test_openai_compatible_inline_video_is_sampled_into_image_url_frames` — ok (new)
- **End-to-end with real 2 s mp4 + qw38 @ 127.0.0.1:17678**: produced `[text,text,image_url,text,image_url,text,image_url,text,image_url]` (4 `image_url`, 0 `video_url`) and returned `HTTP 200`. Pre-fix, the same media emitted as `video_url` returned `HTTP 400`.

## Compatibility / 兼容性

- No API change for users who supply `openai_video_urls`: those videos still pass through as `video_url`.
- Users targeting a `video_url`-capable gateway but *without* filling `openai_video_urls` will now see sampled `image_url` frames instead of `video_url`. This is a strict improvement for image-only endpoints and matches the local GGUF path's behavior.
- AI Workshop and Seedance cloud paths (`_inline_media_plan` / `_upload_media_plan`) are unchanged.

## Notes for maintainers / 维护者备注

- 维护者可自由在此分支上直接编辑 (`Allow edits by maintainers` enabled).
- The repo already has ~900 MB of `web/js/assets/t8-case-previews/*.gif` checked into history, which makes the very first fork-push to a brand-new GitHub fork upload the full pack. To work around that, this PR was pushed after first materialising the common ancestor commit on the fork as a throwaway branch. Consider `git rm --cached` + `.gitignore`ing `web/js/assets/t8-case-previews/*.gif` in a follow-up cleanup commit to shrink clones.

> **Comment by T8mars (2026-09-06T08:21:51Z):**

感谢提交和真实 llama.cpp/qw38 复现证据。这个问题已经基于当前 `main` 移植并加强，发布于 [`1.14.4`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/02d663e91467930ca9a14af285775f878d9f489e)：

- 未提供 URL 的原生 `VIDEO` 会转成有序、带时间戳的 JPEG `image_url` 帧，H3 与 Seedance 2.0 共用该路径；
- 显式 HTTP(S) 视频 URL 仍保留为支持 `video_url` 渠道的 opt-in；
- 抽帧失败时在请求前停止，不再回退为会再次触发 400 的内联 `video_url`，避免无效或可能计费的请求；
- 使用节点中的采样率参数，并补齐真实 MP4 抽帧、H3、Seedance 2.0 和零请求失败路径测试；
- 中英文 README、tooltip、CHANGELOG 与版本已同步。

本地 353 项测试、23 节点/13 工作流校验、22 项前端契约测试均通过；[Verify Action](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/actions/runs/34021511242) 与 [Registry Publish Action](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/actions/runs/34021511225) 也均成功。提交保留了你的 Co-authored-by 署名。

由于原 PR 已落后当前主分支并存在版本/代码冲突，因此没有直接合并旧分支；现按“已在主分支解决”关闭。再次感谢贡献。



### #18 [closed] ISSUE: 新版 llama‑cpp‑python API
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/18  by gangdeq 2026-08-29T16:22:46Z

# ComfyUI Error Report
## Error Details
- **Node ID:** 1
- **Node Type:** MiniMaxH3PromptEnhancerT8
- **Exception Type:** /root/ComfyUI/custom_nodes/comfyui-minimax-h3-prompt-enhancer-T8.nodes.PromptEnhancerError
- **Exception Message:** /root/ComfyUI/custom_nodes/comfyui-minimax-h3-prompt-enhancer-T8.nodes.PromptEnhancerError: llama-cpp-python inference failed. TypeError: Llama.create_chat_completion() got an unexpected keyword argument 'presence_penalty'

## Stack Trace
```
  File "/root/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "/root/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^

  File "/root/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/ComfyUI/comfy_api/latest/_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/ComfyUI/custom_nodes/comfyui-minimax-h3-prompt-enhancer-T8/nodes.py", line 1998, in execute
    result = enhance_prompt(
             ^^^^^^^^^^^^^^^

  File "/root/ComfyUI/custom_nodes/comfyui-minimax-h3-prompt-enhancer-T8/nodes.py", line 1555, in enhance_prompt
    raise PromptEnhancerError(str(error)) from error

```
## System Information
- **ComfyUI Version:** 0.34.2

> **Comment by T8mars (2026-08-29T17:09:14Z):**

已在 **v1.9.10** 修复，提交：[`abbe956`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/abbe9566c727d719b221d2e8ec12d2e3b345604c)。

原因是部分旧版或第三方修改的 `llama-cpp-python` Wheel，其 `Llama.create_chat_completion()` 使用严格签名且没有 `presence_penalty` 参数。现在两个本地运行时都会先检查当前实际方法签名：

- 支持 `presence_penalty`：继续传入原有配置，不改变标准运行时行为；
- 不支持：仅省略这个可选参数，避免本地 H3、Seedance 与 Music 3 直接报错；
- 不使用捕获 `TypeError` 后重复推理的方式，因此不会吞掉模型内部错误，也不会重复生成。

验证结果：

- 新增严格签名回归测试，同时覆盖 Registry 内置运行时和 GitHub 独立回退运行时；
- 本地完整 **267 项测试通过**；
- 仓库结构、Registry 打包、密钥扫描与真实浏览器前端测试通过；
- GitHub Actions 已在 ComfyUI 0.33 / Python 3.10 和 ComfyUI master / Python 3.12 两套环境通过；
- Registry v1.9.10 已提交，目前平台状态为 `Pending`，正式下载地址已经生成。

建议 Registry 变为 Active 后通过 Manager 更新；也可以直接从 GitHub `main` 更新并重启 ComfyUI。感谢反馈。



### #17 [closed] ISSUE: OpenAI-compatible mode: strip inline <think> reasoning from cloud chat responses
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/17  by SherryTOT 2026-08-29T12:09:33Z

When using an OpenAI-compatible endpoint backed by a reasoning model (e.g. MiniMax M3 via `api.minimaxi.com/v1` with `custom_model=MiniMax-M3`), the endpoint returns the model's chain-of-thought inline in `choices[0].message.content`, wrapped in `<think>...</think>` — there is no separate reasoning field. The enhanced prompt fed to the video model therefore starts with the entire reasoning trace.

v1.7.4's `_finalize_local_content()` already strips think tags, but only for the local GGUF runtime; the cloud path (`_request_completion`) returns `result.text` untouched.

Suggestion: apply the same strip (`re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()`, with a fallback to the raw text if the result comes back empty) to cloud responses as well — or at least when `api_mode` is OpenAI-compatible. We run exactly this as a local patch in production with MiniMax-M3 and it works well. Happy to test a build.


> **Comment by T8mars (2026-09-02T07:12:35Z):**

感谢反馈，这个问题已经在 1.9.8（提交 b404595）修复，并在当前 1.12.3 继续保留。

现在 OpenAI-compatible 云端的普通响应与流式响应都会经过共享清理：
- 移除内嵌的 `<think>...</think>` 推理内容；
- 只保留最终提示词；
- 如果响应只有推理、没有最终答案，则按空响应失败，不会把思维链传给视频模型。

仓库已有自动化测试覆盖“云端内嵌 think 清理”和“仅推理响应拒绝”两条路径。请更新到最新版本使用；如最新版仍能复现，请新开 Issue 并附脱敏响应结构与 API 模式。



### #16 [closed] ISSUE: 无法识别LLM目录下模型
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/16  by gangdeq 2026-08-27T07:21:39Z

无法正确识别LLM目录下模型，一直显示默认Qwen3.8-27B-Q4_K_M.gguf

> **Comment by T8mars (2026-08-27T12:58:10Z):**

感谢反馈。当前版本会递归扫描实际 ComfyUI 模型目录下的 `models/LLM` 及任意子目录；点击节点底部的“🧩 检查本地 Qwen 安装 / 扫描 GGUF”后，会刷新当前节点的模型下拉列表，但不会自动把当前选中值从默认模型切换为新模型。

为了确认是目录识别、前端刷新还是模型文件问题，请补充以下信息（API Key 等敏感内容请继续隐藏）：

1. 本插件版本或 Git 提交号，以及 ComfyUI 版本/提交号；
2. GGUF 的完整文件名、文件大小，以及相对路径，例如 `ComfyUI/models/LLM/子目录/模型.gguf`；
3. 点击“扫描 GGUF”后弹窗的完整截图，尤其是“模型目录”“已发现主模型数量/投影器数量”；
4. 扫描后展开“本地 GGUF 主模型”下拉列表的截图；
5. 放入模型后是否已经重启 ComfyUI，并对浏览器执行强制刷新（Ctrl+F5）。

如果弹窗显示主模型数量大于 0，但下拉列表仍只有默认项，请同时附上浏览器开发者工具 Console 中的报错。拿到这些信息后我们可以准确定位，不会靠猜。


> **Comment by gangdeq (2026-08-27T15:20:49Z):**

点击节点底部的“🧩 检查本地 Qwen 安装 / 扫描 GGUF 弹出提示
-----------------------------------------------------------------
文字能力:未就绪(需要运行时和至少一个主模型GGUF)。
视觉能力:未就绪;文字模型仍可用于Music3或纯文字提示词。
已发现:0个主模型,0个视觉投影器。
可用运行时:
- llama-cpp-python 0.3.47 (ComfyUl Python environment)
AUTO会优先使用固定|lama-server,也可自动复用当前Comfyl
的Ilama-cpp-python。
当前模型:Qwen3.8-27B-Q4K_M.gguf(未在本次扫描中找到,
保留旧工作流值)
模型目录:/root/ComfyUl/models/LLM
支持目录下任意层级子文件夹;点击本按钮会重新扫描并刷新节点下
拉列表。
兼容边界:GGUF被发现不等于质量已验证;图像/视频分析必须有
匹配mmproj。
视频边界:按真实时间戳采样可见画面,不读取视频音轨。
Music3边界:仅处理文字,不加载视觉投影器。
------------------------------------------------------------------
节点号----c78015b1ced46544633d447f04a090669f5fb1d4
comfyui版本0.34.0----12d5279438bfefc058a269eae805ceab6047777f
------------------------------------------------------------------

<img width="1004" height="180" alt="Image" src="https://github.com/user-attachments/assets/6429506a-e8e0-4fff-9e27-1b9c71b7751b" />

<img width="944" height="268" alt="Image" src="https://github.com/user-attachments/assets/5ec5aedb-7ad7-4c54-8876-bf72b05bc7e5" />

<img width="1054" height="238" alt="Image" src="https://github.com/user-attachments/assets/6643150e-a030-40b3-92e2-1958ac3ad218" />

<img width="1200" height="581" alt="Image" src="https://github.com/user-attachments/assets/d62760b1-9ef9-4e08-9f39-0e3ff2459615" />

<img width="1200" height="581" alt="Image" src="https://github.com/user-attachments/assets/3972a777-35b1-42b3-9471-5d7278fd420d" />

<img width="1402" height="614" alt="Image" src="https://github.com/user-attachments/assets/ebc0b343-a5a7-40d8-b0c8-5bcb035e05a1" />


> **Comment by gangdeq (2026-08-29T11:36:10Z):**

-------------------------------------------------------------8哥，是这个问题-------------------------------------------------------------------(comfyui) root@269af8a30d96:~/ComfyUI/custom_nodes# # 1) 看 ComfyUI 真正认定的 models 根目录
python3 -c "
import sys
sys.path.insert(0, '/root/ComfyUI/custom_nodes/comfyui-minimax-h3-prompt-enhancer-T8')
import local_gguf_catalog as c
print('models_dir:', c.llm_model_directory())
print('legacy dir:', c.legacy_qwen_model_directory())
print('roots:', c._registered_model_roots())
" 2>&1

# 2) 看 /root/ComfyUI/models/LLM 里到底有什么
ls -la /root/ComfyUI/models/LLM/ 2>&1

# 3) 看真实 mount / 权限
stat /root/ComfyUI/models/LLM
mount | grep -E "ComfyUI|models|root"

# 4) 跨平台找找你的 GGUF 实际在哪里
find / -name "*.gguf" -type f 2>/dev/null | head -20
models_dir: /root/ComfyUI/models/LLM
legacy dir: /root/ComfyUI/models/LLM/Qwen3.8
roots: (PosixPath('/root/ComfyUI/models/LLM'),)
总计 38
drwxr-xr-x  4 root root 11  8月 29 09:36 .
drwxr-xr-x 87 root root 87  8月 12 08:50 ..
lrwxrwxrwx  1 root root 51  8月 29 09:36 9B-mmproj-BF16.gguf -> /.xgcos/.links/41013012-5cf8-5716-905b-8d1665d19822
drwxr-xr-x  3 root root 19  8月 29 09:35 Florence-2-base
drwxr-xr-x  3 root root 21  8月 29 09:36 Florence-2-large
lrwxrwxrwx  1 root root 51  8月 29 09:36 gemma-4-26B-A4B-it-heretic-ara-v2.i1-Q5_K_M.gguf -> /.xgcos/.links/be621c48-0ba6-5777-b799-f8a89d8a3b28
lrwxrwxrwx  1 root root 51  8月 29 09:36 gemma-4-26B-A4B-it-heretic-mmproj.bf16.gguf -> /.xgcos/.links/c5facc6a-4fa3-5b64-b8e7-db32f9baa5ea
lrwxrwxrwx  1 root root 51  8月 29 09:36 mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf -> /.xgcos/.links/8acb440d-5c8e-5f8f-9261-845a65efbd26
lrwxrwxrwx  1 root root 51  8月 29 09:35 Qwen3.5-9B-Q8_0.gguf -> /.xgcos/.links/0de4d3b6-d786-57fe-8c8d-42d9d7a67467
lrwxrwxrwx  1 root root 51  8月 29 09:36 Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf -> /.xgcos/.links/f5dc3d9e-b7c0-59f8-ab53-76788c30b0b3
lrwxrwxrwx  1 root root 51  8月 29 09:35 Qwen3.8-27B-ABLITERATED-Q5_K_M.gguf -> /.xgcos/.links/5a9f9f80-772e-51ec-9aca-0f8bb56a24b3
  文件：/root/ComfyUI/models/LLM
  大小：11              块：17         IO 块大小：1024   目录
设备：3bh/59d   Inode：242958      硬链接：4
权限：(0755/drwxr-xr-x)  Uid: (    0/    root)   Gid: (    0/    root)
访问时间：2026-08-29 09:35:06.078622587 +0000
修改时间：2026-08-29 09:36:16.552080462 +0000
变更时间：2026-08-29 09:36:16.552080462 +0000
创建时间：2026-08-29 09:35:06.078622587 +0000
:webdav{Ks0OR}: on /root/cloud type fuse.rclone (rw,nosuid,nodev,relatime,user_id=0,group_id=0)

----------------还有就是T8浏览库中，点击使用此模板，切换工作流再切回来就会出现节点慢慢拉长---------------

<img width="1651" height="1184" alt="Image" src="https://github.com/user-attachments/assets/0dd8bf60-d426-451a-a889-faa989e5019d" />
<img width="491" height="1145" alt="Image" src="https://github.com/user-attachments/assets/3839bb75-32bf-477d-9ea3-bd52de735467" />


> **Comment by T8mars (2026-08-29T17:02:00Z):**

感谢继续补充。我们正在同时检查“模型扫描为 0”和“切换工作流后节点逐渐拉长”两个现象。

当前 1.9.9 代码会跟随 `models/LLM` 下的 GGUF 符号链接；本地符号链接回归测试以及 300 次模板预览切换测试均通过。尺寸重算每次都会清除 DOM widget 的缓存高度并重新执行 `computeSize()`，目前还没有复现到持续累加高度。因此这个 Issue 先保持开启，请帮忙补充以下最小复现信息（不要包含 API Key、私有提示词或私有素材）：

1. 节点逐渐拉长发生在 MiniMax H3、Seedance 2.0，还是两个节点都会？
2. 请把完整操作写成编号步骤：从刚打开工作流开始，是否需要点击“使用此模板”、关闭模板库、切换顶部工作流标签，再切回来？不运行节点是否也会增长？通常第几次切换开始明显增长？
3. 请分别导出“首次选择模板后”和“已经明显拉长后”的最小工作流 JSON；只保留加载图像/文本、该增强节点和文本输出即可。
4. 请提供插件版本/提交号、ComfyUI 后端提交号、ComfyUI Frontend 版本、浏览器版本，以及是否安装会修改节点尺寸或 DOM widget 的前端扩展。
5. 请录一段从正常高度到拉长的完整屏幕录像，并附浏览器开发者工具 Console 的报错。
6. 对任意一个扫描不到的 GGUF 符号链接，请提供以下只读命令输出，用于确认 FUSE/WebDAV 目标是否真实可读：
   - `readlink -f /root/ComfyUI/models/LLM/模型名.gguf`
   - `stat -L /root/ComfyUI/models/LLM/模型名.gguf`
   - `python3 -c "from pathlib import Path; p=Path('/root/ComfyUI/models/LLM/模型名.gguf'); print(p.is_file(), p.resolve(), p.stat().st_size)"`

拿到最小工作流和精确切换步骤后，我们会按同一路径继续复现，不靠猜测关闭问题。


> **Comment by gangdeq (2026-08-30T04:33:13Z):**

我重新安装了这个插件的最新版，模型可以正常发现，可运行。
1-节点逐渐拉长发生在MiniMax H3、Seedance 2.0两个节点
2-点击“使用此模板”、关闭模板库、切换顶部工作流标签，再切回来会拉长
3-ComfyUI v0.34.2 --169fcf35a2fc163fec31338b816503ddac0d3fcf
comfyui-minimax-h3-prompt-enhancer-T8 最新

https://github.com/user-attachments/assets/0b491345-04cc-4bbb-936f-a70727aa6315
https://github.com/user-attachments/assets/0f42a5aa-9fe1-42ee-ab23-d24af54a42f4
[minimax-seedance示例.json](https://github.com/user-attachments/files/31608028/minimax-seedance.json)
<img width="1293" height="1243" alt="Image" src="https://github.com/user-attachments/assets/3c81e818-d274-4611-afb7-9c58658c9894" />
<img width="2197" height="1069" alt="Image" src="https://github.com/user-attachments/assets/ca755be1-10ea-4871-b450-2a06b423ffc2" />


> **Comment by gangdeq (2026-09-02T07:06:20Z):**

8总是不是把这个Issues抽时间先完善一下~~


> **Comment by T8mars (2026-09-02T07:25:50Z):**

感谢持续补充复现材料。这个 Issue 中的两个现象现已分别处理：

1. **LLM 模型扫描**：你已经确认重新安装最新版后可以正常发现并运行模型，因此原始扫描问题已恢复。
2. **选择 T8 模板后切换工作流，H3 / Seedance 节点逐渐拉长**：已在 **1.12.4**（提交 `88de1ce`）修复。

我们直接检查了你上传的最小工作流，确认异常不是视觉错觉：正常 H3 / Seedance 节点高度约为 784 / 832，而异常节点已被保存为 8267 / 8315。根因是模板详情卡读取 `scrollHeight` 时，把 ComfyUI 上一次分配并写回元素的 inline 高度再次当成真实内容高度，工作流恢复时形成高度反馈。

1.12.4 改为临时移除布局分配高度，只测量真实内容高度；T8 模板与 MiniMax 官方模板详情卡都应用了同一修复。已经异常变高的旧工作流会在下一次详情布局时自动恢复，不改节点 ID、参数值或连接。

验证证据：
- 347 项单元/兼容测试全部通过；
- 新增真实 Chromium 回归，从你工作流中的 8267px 异常高度开始，验证可恢复到真实内容高度；
- 重复恢复工作流后高度保持不变；
- ComfyUI v0.33.0 / Python 3.10 与 ComfyUI master / Python 3.12 两套 CI 均通过；
- GitHub 版 1.12.4 已发布，Comfy Registry 已提交审核。

请优先从 GitHub 更新到 1.12.4，并执行 Ctrl+F5 后重新打开工作流。若更新后仍能复现，请新开 Issue 并附更新后的最小工作流，我们会按新的证据继续处理。



### #15 [closed] ISSUE: [Bug] R2V无法识别参考图片
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/15  by QTHunYuan 2026-08-27T03:13:51Z

### Versions

a8164ea

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

好

### Minimal reproduction

<img width="766" height="546" alt="Image" src="https://github.com/user-attachments/assets/2359f188-6fef-43db-bb9e-0a8020e2582f" />

[INFO] got prompt
[ERROR] Failed to validate prompt for output 7:
[ERROR] * MiniMaxH3PromptEnhancerT8 70:
[ERROR]   - Exception when validating inner node: MiniMaxH3PromptEnhancer.validate_inputs() got an unexpected keyword argument 'reference_images'
[ERROR] Output will be ignored
[WARNING] invalid prompt: {'type': 'prompt_outputs_failed_validation', 'message': 'Prompt outputs failed validation', 'details': '', 'extra_info': {}}

### Redacted error / diagnostic category

comfyui是最新开发版
扩展切换回4959673版后工作正常

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by Copipi (2026-08-27T12:02:23Z):**

 I2V 也會出現一樣的問題
我的comfyui 是 d8e7bbc9 (2026-08-26) 版



> **Comment by T8mars (2026-08-27T12:58:06Z):**

已修复并发布到 `main`，修复版本为 `v1.7.7`，提交：1f725dd。

原因是 `v1.7.4` 新增的自定义 `validate_inputs()` 只接收本地模型字段，而新版 ComfyUI 会同时传入 Autogrow 动态输入组 `reference_images` / `reference_videos`，因此 Ref2VA 在执行前被 Python 参数校验拦截。

本次修复：
- MiniMax H3 验证器显式兼容 `reference_images` 和 `reference_videos`；
- 同步修复 Seedance 2.0 的相同潜在问题；
- 不使用宽泛 `**kwargs`，继续保留 ComfyUI 对其他输入的原生枚举与范围校验；
- 增加动态参考图/视频验证回归测试。

验证结果：214 项 Python 测试和 14 项前端测试全部通过。

更新到 `v1.7.7` 并重启 ComfyUI、刷新前端后，Ref2VA 参考图片连接应可正常通过验证。如仍遇到问题，欢迎携带新的错误日志重新反馈。



### #14 [closed] ISSUE: [Bug] 本地gguf模型思考无法关闭
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/14  by tingyaolab 2026-08-25T07:07:41Z

### Versions

4a99b25

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

本地gguf模型

### Minimal reproduction

已经配置本地思考模式为关闭，但还是会输出思考内容

### Redacted error / diagnostic category

 {
            "identifier": "Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
            "filename": "Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
            "size": 16810714400,
            "architecture": "qwen35",
            "model_type": "model",
            "name": "Huihui Qwen3.8 27B Abliterated",
            "context_length": 262144,
            "projector_type": "",
            "has_vision_encoder": false,
            "has_chat_template": true,
            "metadata_readable": true,
            "metadata_error": "",
            "is_projector": false,
            "text_capable": true,
            "recommended_projector": "mmproj-model-bf16.gguf",
            "vision_capable": true,
            "verification_tier": "runtime_supported_unverified"
        }

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by T8mars (2026-08-26T14:31:29Z):**

已在 [`a8164ea`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/a8164eafd6c89c7437e1a9255b8684fb569b226f) / `v1.7.4` 修复。

修复内容：
- 对支持该能力的 `llama-server`，关闭本地思考时会在模型启动阶段明确传入 `--reasoning off`，而不只依赖单次请求参数。
- 请求仍传入 `enable_thinking=false` 与 `preserve_thinking=false`，兼容不同 llama.cpp / Qwen 模板。
- 对仍然无视开关的第三方模板增加最终兜底：关闭思考时移除泄漏到 `message.content` 的 `<think>...</think>` 内容；独立 `reasoning_content` 不会进入节点字符串输出。
- `llama-cpp-python` 回退运行时也会在支持时应用加载阶段的模板参数。
- 已加入“启动参数为 reasoning off、关闭时不泄漏思考、显式开启时保留”的回归测试。

本地完整测试 211 项通过，GitHub 的 ComfyUI `v0.33.0 / Python 3.10` 与 `master / Python 3.12` 验证也均已通过。Comfy Registry 已收到 `1.7.4`，当前处于自动扫描 Pending 状态。



### #13 [closed] ISSUE: [Bug] 使用兼容接口/本地API时，会检测本地GGUF qwen3.8的配置，配置不对时会报错
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/13  by QTHunYuan 2026-08-25T05:30:25Z

### Versions

4a99b25

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

好

### Minimal reproduction

<img width="879" height="1019" alt="Image" src="https://github.com/user-attachments/assets/de9696dc-e756-4441-962f-1b19c9fd114c" />

[INFO] got prompt
[ERROR] Failed to validate prompt for output 7:
[ERROR] * MiniMaxH3PromptEnhancerT8 41:
[ERROR]   - Value not in list: local_model: 'Qwen3.8-27B-Q4_K_M.gguf' not in ['Qwen3-VL-30B-Instruct-abliterated-Q4_K_M/Huihui-Qwen3-VL-30B-Instruct-abliterated.i1-Q4_K_M.gguf', 'Qwen3.6-35B-HauhauCS-Q6_K/Qwen3.6-35B-HauhauCS-Q6_K.gguf', 'Qwen3.8-27B-Uncensored-HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf']
[ERROR] Output will be ignored
[ERROR] Failed to validate prompt for output 5:
[ERROR] Output will be ignored

### Redacted error / diagnostic category

LM Studio已经连通。
在选择兼容接口模式的情况下，应该忽略节点中本地GGUF qwen3.8模型相关配置，直接走兼容接口流程。
目前应该会造成本地没有qwen3.8模型的同学无法使用节点。
在正确配置本地GGUF模型后，改回兼容接口模式，能够正常工作。

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by T8mars (2026-08-26T14:31:25Z):**

已在 [`a8164ea`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/commit/a8164eafd6c89c7437e1a9255b8684fb569b226f) / `v1.7.4` 修复。

修复内容：
- `local_model` / `local_mmproj` 继续保留下拉列表，但不再由 ComfyUI 在节点执行前对旧工作流值做严格列表拦截。
- 只有实际选择本地 GGUF 模式时才解析并校验本地文件；OpenAI 兼容、Seedance 和 AI 工坊模式会正常忽略未使用的旧本地模型值。
- 同步覆盖 MiniMax H3、Seedance 2.0、Music 3 和共享 LLM 渠道配置节点。
- 已加入旧/缺失 GGUF 值的回归测试。

本地完整测试 211 项通过，GitHub 的 ComfyUI `v0.33.0 / Python 3.10` 与 `master / Python 3.12` 验证也均已通过。Comfy Registry 已收到 `1.7.4`，当前处于自动扫描 Pending 状态。



### #12 [closed] ISSUE: [Bug] GPU和CPU都跑不满，最后生成失败
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/12  by QTHunYuan 2026-08-25T04:21:15Z

### Versions

4a99b25

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

无

### Minimal reproduction

<img width="1048" height="557" alt="Image" src="https://github.com/user-attachments/assets/4baacaf1-c3c4-43e6-a04a-b60f8ba43962" />

[ERROR] !!! Exception during processing !!! OpenAI-compatible provider chat network error: ReadTimeout. The response state is ambiguous, so it was not retried automatically to avoid a duplicate paid generation.
[ERROR] Traceback (most recent call last):
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\connectionpool.py", line 468, in _make_request
    six.raise_from(e, None)
  File "<string>", line 3, in raise_from
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\connectionpool.py", line 463, in _make_request
    httplib_response = conn.getresponse()
                       ^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\http\client.py", line 1430, in getresponse
    response.begin()
  File "D:\ComfyUI-aki-v2.7\python\Lib\http\client.py", line 331, in begin
    version, status, reason = self._read_status()
                              ^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\http\client.py", line 292, in _read_status
    line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\socket.py", line 720, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\requests\adapters.py", line 644, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\connectionpool.py", line 802, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\util\retry.py", line 552, in increment
    raise six.reraise(type(error), error, _stacktrace)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\packages\six.py", line 770, in reraise
    raise value
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\connectionpool.py", line 716, in urlopen
    httplib_response = self._make_request(
                       ^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\connectionpool.py", line 470, in _make_request
    self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\urllib3\connectionpool.py", line 358, in _raise_timeout
    raise ReadTimeoutError(
urllib3.exceptions.ReadTimeoutError: HTTPConnectionPool(host='localhost', port=1234): Read timed out. (read timeout=300)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\ComfyUI-aki-v2.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\provider_transport.py", line 46, in request_chat_completion
    response = session.post(
               ^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\requests\sessions.py", line 637, in post
    return self.request("POST", url, data=data, json=json, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\requests\sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\requests\sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\python\Lib\site-packages\requests\adapters.py", line 690, in send
    raise ReadTimeout(e, request=request)
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=1234): Read timed out. (read timeout=300)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\ComfyUI-aki-v2.7\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\ComfyUI-aki-v2.7\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\nodes.py", line 1898, in execute
    result = enhance_prompt(
             ^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\nodes.py", line 1523, in enhance_prompt
    response_text = _request_completion(
                    ^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\nodes.py", line 1257, in _request_completion
    result = request_chat_completion(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-aki-v2.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\provider_transport.py", line 62, in request_chat_completion
    raise network_error(error, attempt, retry_delays) from error
D:\ComfyUI-aki-v2_x_7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8.nodes.PromptEnhancerError: OpenAI-compatible provider chat network error: ReadTimeout. The response state is ambiguous, so it was not retried automatically to avoid a duplicate paid generation.


### Redacted error / diagnostic category

使用本地LM Studio的API服务
CPU和GPU占用过低，生成速度慢
在进度大约99%的时候突然任务失败

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by QTHunYuan (2026-08-25T04:45:46Z):**

CPU和GPU占用过低，应该是LM Studio的问题，和节点无关。




### #11 [closed] ISSUE: [Bug] 【API模式】选择【OpenAl兼容接口（备用）】无效了，强制去读不存在的qwen3.8模型
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/11  by QTHunYuan 2026-08-24T11:17:08Z

### Versions

7da7832

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

本地

### Minimal reproduction

<img width="665" height="1035" alt="Image" src="https://github.com/user-attachments/assets/7e5d3e05-19c6-4c02-89a4-53d460c24bae" />

### Redacted error / diagnostic category

[INFO] got prompt
[ERROR] Failed to validate prompt for output 7:
[ERROR] * MiniMaxH3PromptEnhancerT8 3:
[ERROR] - Value not in list: local_model: 'randomize' not in ['Qwen3.8-27B-Q4_K_M.gguf']
[ERROR] - Failed to convert an input value to a INT value: seed, , invalid literal for int() with base 10: ''
[ERROR] Output will be ignored
[WARNING] invalid prompt: {'type': 'prompt_outputs_failed_validation', 'message': 'Prompt outputs failed validation', 'details': '', 'extra_info': {}}

猜测是模型或种子相关参数部分出问题了

切换到【laee075】版本后，报错消失，节点可以正常工作

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by T8mars (2026-08-24T13:18:34Z):**

感谢反馈。报告中的版本 `7da7832` 对应 v1.1.2。日志里的 `local_model: randomize` 和空 seed 是旧工作流控件数组在新增本地 Qwen 参数后发生的位置偏移，错误发生在 ComfyUI 执行前校验阶段，并不是 OpenAI 兼容模式真的改去调用了本地 Qwen。v1.4.0 已加入旧版 22 项 H3 工作流到当前字段的补齐与按名称恢复；v1.4.1 又补充了 Seedance 字段顺序迁移。当前 v1.4.2 已针对 `randomize`、本地模型默认值、seed 和 OpenAI 模式迁移完成回归验证。请更新至 v1.4.2，重启 ComfyUI 并刷新前端后重新打开原工作流。若 v1.4.2 仍出现相同校验错误，欢迎提供脱敏后的工作流 JSON，我们会继续复核。现按已修复关闭。



### #10 [closed] ISSUE: 使用openai 兼容接口時，openai 模型id 必填，這個欄位的值會自動消失
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/10  by Copipi 2026-08-23T14:35:29Z

使用openai 兼容接口時，我輸入 url 和 openai 模型id， 

只要在comfyui 內切換tab 視窗，再切回來， openai 模型id 的值會消失
每次都要重key 一次 
(但 url 會保留)

希望 openai 模型id 的值 ，也能夠自動保留

> **Comment by T8mars (2026-08-24T13:18:34Z):**

感謝回報。此問題已在 v1.3.0 起修復：`custom_model`（OpenAI 模型 ID）和 `openai_base_url` 現在會按穩定欄位名稱保存及恢復，不會因欄位暫時隱藏、切換工作流 Tab 或重新配置節點而遺失。當前 v1.4.2 已用 `k3` 與 Kimi Coding Base URL 做專門的 configure/serialize 回歸測試，結果通過。請更新至 v1.4.2，重啟 ComfyUI 並重新整理瀏覽器前端。若最新版仍會清空模型 ID，歡迎附上已脫敏的工作流 JSON 與 ComfyUI 前端版本繼續回報。現按已修復關閉。


> **Comment by T8mars (2026-08-24T14:16:23Z):**

收到后续反馈后重新核验，确认此前的回归测试遗漏了一个前端边界：用户输入模型 ID 后如果未让输入框失焦就直接运行，部分 ComfyUI 前端版本的最新文本只存在于 DOM 输入框，尚未同步到 widget.value，因此保存/运行时可能取到旧值或空值。现已重新打开此 Issue。修复已在本地完成：三个核心节点都会在输入、运行前及工作流序列化时提交 Base URL 与模型 ID，并在重载时从工作流属性恢复；API Key 不进入该兼容状态。相关单元、真实浏览器及全量仓库测试均已通过，待新版本发布后再关闭。


> **Comment by T8mars (2026-08-24T14:56:09Z):**

已在 v1.4.3 完成并发布修复。三个核心节点现会即时同步并随工作流保存 OpenAI Base URL 与模型 ID；即使输入框尚未失焦就直接运行，也会在提交前读取 DOM 中的最新值。切换 API 模式和重新载入工作流会自动恢复，API Key 不会写入这份兼容状态。发布前 201 项测试、真实浏览器合同、仓库完整性与密钥扫描均通过；GitHub Registry 发布工作流也已成功上传 v1.4.3。感谢反馈。



### #9 [closed] ISSUE: 大量的空白区域，实际上应该有内容显示[Bug]
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/9  by AaronZhai331 2026-08-23T07:39:43Z

### Versions

最新版

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8

### Minimal reproduction

<img width="477" height="1148" alt="Image" src="https://github.com/user-attachments/assets/3ca02a1b-d4aa-40a6-ba57-d8347e7e35b2" />

### Redacted error / diagnostic category

？？？？

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.


### #8 [closed] ISSUE: [Bug] 接 kimi coding plan 的时候会报错，因为 kimi 不接受 temperature 参数
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/8  by glacjay 2026-08-21T08:58:59Z

### Versions

4337f5b

### Node

MiniMax H3 Prompt Enhancer

### Provider mode

OpenAI 兼容接口

### Minimal reproduction

base url: https://api.kimi.com/coding/
model: k3

### Redacted error / diagnostic category

Node threw an error during execution.


# ComfyUI Error Report
## Error Details
- **Node ID:** 56
- **Node Type:** MiniMaxMusic3PromptEnhancerT8
- **Exception Type:** E:\ComfyUI-aki-v1_x_7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8.music3.Music3PromptEnhancerError
- **Exception Message:** E:\ComfyUI-aki-v1_x_7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8.music3.Music3PromptEnhancerError: OpenAI-compatible provider Music 3 stage 'lyrics_generation' failed: status=400, category=request_rejected, attempts=1. Upstream response text was hidden for privacy.

## Stack Trace
```
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 545, in execute
output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 344, in get_output_data
return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 318, in _async_map_node_over_list
await process_inputs(input_dict, i)

File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 306, in process_inputs
result = f(**inputs)
^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
return method(locked_class, **inputs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
to_return = cls.execute(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 2347, in execute
result = enhance_music3_prompt(
^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 1934, in enhance_music3_prompt
final_lyrics, generated_profile = _generate_or_edit_lyrics(
^^^^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 1287, in _generate_or_edit_lyrics
response = runner.complete(
^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 424, in complete
result = _request_music_completion(
^^^^^^^^^^^^^^^^^^^^^^^^^^

File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 721, in _request_music_completion
_raise_music_http_error(

File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 659, in _raise_music_http_error
raise Music3PromptEnhancerError(

```
## System Information
- **ComfyUI Version:** 0.33.1
- **Arguments:** E:\ComfyUI-aki-v1.7\ComfyUI\main.py --listen 0.0.0.0 --preview-method auto --use-sage-attention --disable-cuda-malloc
- **OS:** win32
- **Python Version:** 3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.7.0+cu128
## Devices

- **Name:** cuda:0 NVIDIA RTX PRO 6000 Blackwell Workstation Edition : cudaMallocAsync
- **Type:** cuda
- **VRAM Total:** 102641500160
- **VRAM Free:** 99916709888
- **Torch VRAM Total:** 0
- **Torch VRAM Free:** 0

## Logs
```
2026-08-21T13:47:56.682912 - [GLM_Nodes] 信息：从 'image_prompts.txt' 加载提示词成功。2026-08-21T13:47:56.682912 -
2026-08-21T13:47:56.682912 - [GLM_Nodes] 信息：从 'image_prompts.txt' 加载提示词成功。2026-08-21T13:47:56.682912 -
2026-08-21T13:48:49.780442 - ^[[32m[INFO]^[[0m got prompt
2026-08-21T13:48:49.780967 - Exception ignored in: 2026-08-21T13:48:49.781501 - <function _ProactorBasePipeTransport.__del__ at 0x00000287BA6AB920>2026-08-21T13:48:49.781501 -
2026-08-21T13:48:49.781501 - Traceback (most recent call last):
2026-08-21T13:48:49.781501 - File "E:\ComfyUI-aki-v1.7\python\Lib\asyncio\proactor_events.py", line 116, in __del__
2026-08-21T13:48:49.782019 - 2026-08-21T13:48:49.782019 - _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)2026-08-21T13:48:49.782019 -
2026-08-21T13:48:49.782019 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - 2026-08-21T13:48:49.782545 - ^2026-08-21T13:48:49.782545 - ^2026-08-21T13:48:49.782545 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 - ^2026-08-21T13:48:49.783069 -
2026-08-21T13:48:49.783069 - File "E:\ComfyUI-aki-v1.7\python\Lib\asyncio\proactor_events.py", line 80, in __repr__
2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - info.append(f'fd={self._sock.fileno()}')2026-08-21T13:48:49.783602 -
2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.783602 - 2026-08-21T13:48:49.784127 - 2026-08-21T13:48:49.784127 - 2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 - ^2026-08-21T13:48:49.784127 -
2026-08-21T13:48:49.784127 - File "E:\ComfyUI-aki-v1.7\python\Lib\asyncio\windows_utils.py", line 102, in fileno
2026-08-21T13:48:49.796260 - 2026-08-21T13:48:49.796260 - raise ValueError("I/O operation on closed pipe")2026-08-21T13:48:49.796260 -
2026-08-21T13:48:49.796260 - ValueError2026-08-21T13:48:49.796260 - : 2026-08-21T13:48:49.796260 - I/O operation on closed pipe2026-08-21T13:48:49.796764 -
2026-08-21T13:48:50.256717 - ^[[1m^[[33m[WARNING]^[[0m WARNING: PlaySound.IS_CHANGED() missing 1 required positional argument: 'self'
2026-08-21T13:48:58.588374 - prompt token count:2026-08-21T13:48:58.588374 - 2026-08-21T13:48:58.604972 - tensor([1], device='cuda:0')2026-08-21T13:48:58.605972 -
2026-08-21T13:48:58.630321 - prompt token count:2026-08-21T13:48:58.630321 - 2026-08-21T13:48:58.652230 - tensor([126], device='cuda:0')2026-08-21T13:48:58.652750 -
2026-08-21T13:49:13.844486 - SDPose Node: Loading model on device: cuda:0 with UNet precision: bf162026-08-21T13:49:13.844486 -
2026-08-21T13:49:15.938228 - SDPose Node: Received 271 frames. Starting 3-stage process with batch_size=1.2026-08-21T13:49:15.938228 -
2026-08-21T13:49:15.938228 - SDPose Node: Ensuring models are on device: cuda:0...2026-08-21T13:49:15.938228 -
2026-08-21T13:49:15.938228 - SDPose Node: Instantiating pipeline...2026-08-21T13:49:15.938228 -
2026-08-21T13:49:15.938228 - SDPose Node: Stage 1/3 - Detecting/Collecting persons...2026-08-21T13:49:15.938228 -
2026-08-21T13:49:15.942235 - SDPose Node: No detector specified. Processing full image.2026-08-21T13:49:15.942235 -
2026-08-21T13:49:20.433315 - SDPose Node: Stage 1 complete. Found 271 total persons in 271 frames.2026-08-21T13:49:20.433315 -
2026-08-21T13:49:20.433315 - SDPose Node: Stage 2/3 - Running inference in batches of 1...2026-08-21T13:49:20.433315 -
2026-08-21T13:50:00.098174 - SDPose Node: Stage 2 complete. Processed 271 persons.2026-08-21T13:50:00.099174 -
2026-08-21T13:50:00.099174 - SDPose Node: Stage 3/3 - Reconstructing frames...2026-08-21T13:50:00.099174 -
2026-08-21T13:50:02.647400 - SDPose Node: Stage 3 complete. Reconstruction finished.2026-08-21T13:50:02.647400 -
2026-08-21T13:50:04.062968 - SDPose Node: Unloading models from VRAM to CPU.2026-08-21T13:50:04.062968 -
2026-08-21T13:50:05.501324 - ^[[32m[INFO]^[[0m Requested to load CLIPVisionModelProjection
2026-08-21T13:50:05.521443 - ^[[32m[INFO]^[[0m loaded completely; 92392.69 MB usable, 1208.10 MB loaded, full load: True
2026-08-21T13:50:05.544383 - ^[[32m[INFO]^[[0m Clip embeds shape: torch.Size([1, 257, 1280]), dtype: torch.float32
2026-08-21T13:50:05.544383 - ^[[32m[INFO]^[[0m Combined clip embeds shape: torch.Size([1, 257, 1280])
2026-08-21T13:50:09.119577 - ^[[32m[INFO]^[[0m CUDA Compute Capability: 12.0
2026-08-21T13:50:09.119577 - ^[[32m[INFO]^[[0m Detected model in_channels: 36
2026-08-21T13:50:09.119577 - ^[[32m[INFO]^[[0m Model cross attention type: i2v, num_heads: 40, num_layers: 40
2026-08-21T13:50:09.119577 - ^[[32m[INFO]^[[0m Model variant detected: 14B
2026-08-21T13:50:09.792371 - ^[[32m[INFO]^[[0m Loading LoRA: WanAnimate_relight_lora_fp16 with strength: 1.0
2026-08-21T13:50:09.950560 - ^[[32m[INFO]^[[0m Loading LoRA: Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16 with strength: 1.0
2026-08-21T13:50:10.075405 - ^[[32m[INFO]^[[0m Loading LoRA: FastWan_T2V_14B_480p_lora_rank_128_bf16 with strength: 1.0
2026-08-21T13:50:10.210483 - ^[[32m[INFO]^[[0m Loading LoRA: Wan21_PusaV1_LoRA_14B_rank512_bf16 with strength: 1.0
2026-08-21T13:50:10.362318 - ^[[32m[INFO]^[[0m Loading LoRA: Wan2.2-Fun-A14B-InP-LOW-HPS2.1_resized_dynamic_avg_rank_15_bf16 with strength: 0.5
2026-08-21T13:50:10.471885 - ^[[32m[INFO]^[[0m Loading and assigning model weights to device...
2026-08-21T13:50:10.695739 - ^[[32m[INFO]^[[0m -------------------------
2026-08-21T13:50:10.695739 - ^[[32m[INFO]^[[0m Transformer weights loaded:
2026-08-21T13:50:10.695739 - ^[[32m[INFO]^[[0m Device: cuda:0 | Memory: 1,171,497.69 MB
2026-08-21T13:50:10.695739 - ^[[32m[INFO]^[[0m Device: cpu | Memory: 11,160,176.01 MB
2026-08-21T13:50:10.695739 - ^[[32m[INFO]^[[0m Using 1133 LoRA weight patches for WanVideo model
2026-08-21T13:50:13.045917 - ^[[32m[INFO]^[[0m ------- Scheduler info -------
2026-08-21T13:50:13.046419 - ^[[32m[INFO]^[[0m Total timesteps: tensor([999, 937, 833, 624], device='cuda:0')
2026-08-21T13:50:13.046935 - ^[[32m[INFO]^[[0m Using timesteps: tensor([999, 937, 833, 624], device='cuda:0')
2026-08-21T13:50:13.046935 - ^[[32m[INFO]^[[0m Using sigmas: tensor([1.0000, 0.9375, 0.8333, 0.6249, 0.0000])
2026-08-21T13:50:13.046935 - ^[[32m[INFO]^[[0m ------------------------------
2026-08-21T13:50:13.644539 - ^[[32m[INFO]^[[0m Rope function: comfy
2026-08-21T13:50:14.307483 - ^[[32m[INFO]^[[0m WanAnimate: Face input torch.Size([1, 3, 271, 512, 512]) padded to shape torch.Size([1, 3, 321, 512, 512])
2026-08-21T13:50:14.806743 - ^[[32m[INFO]^[[0m WanAnimate: Pose images torch.Size([3, 271, 1280, 720]) padded to shape torch.Size([3, 321, 1280, 720])
2026-08-21T13:50:14.806743 - ^[[32m[INFO]^[[0m Sampling 269 frames in 4 windows, at 720x1280 with 4 steps
2026-08-21T13:50:24.187372 - Generated new RoPE frequencies2026-08-21T13:50:24.187372 -
2026-08-21T13:56:32.745179 - ^[[32m[INFO]^[[0m [Sampling] Max allocated memory: max_memory=11.226 GB
2026-08-21T13:56:32.745179 - ^[[32m[INFO]^[[0m [Sampling] Max reserved memory: max_reserved=15.344 GB
2026-08-21T13:56:39.558219 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 469.71 seconds^[[0m
2026-08-21T13:57:49.627266 - ^[[32m[INFO]^[[0m got prompt
2026-08-21T13:57:49.769940 - ^[[1m^[[33m[WARNING]^[[0m WARNING: PlaySound.IS_CHANGED() missing 1 required positional argument: 'self'
2026-08-21T13:58:07.563934 - SDPose Node: Received 184 frames. Starting 3-stage process with batch_size=1.2026-08-21T13:58:07.564474 -
2026-08-21T13:58:07.564474 - SDPose Node: Ensuring models are on device: cuda:0...2026-08-21T13:58:07.564474 -
2026-08-21T13:58:07.564474 - SDPose Node: Moving 'unet' from cpu to cuda:0.2026-08-21T13:58:07.564474 -
2026-08-21T13:58:08.199075 - SDPose Node: Moving 'vae' from cpu to cuda:0.2026-08-21T13:58:08.199075 -
2026-08-21T13:58:08.323209 - SDPose Node: Moving 'decoder' from cpu to cuda:0.2026-08-21T13:58:08.323209 -
2026-08-21T13:58:08.329210 - SDPose Node: Instantiating pipeline...2026-08-21T13:58:08.329210 -
2026-08-21T13:58:08.329210 - SDPose Node: Stage 1/3 - Detecting/Collecting persons...2026-08-21T13:58:08.329210 -
2026-08-21T13:58:08.331210 - SDPose Node: No detector specified. Processing full image.2026-08-21T13:58:08.331210 -
2026-08-21T13:58:10.992494 - SDPose Node: Stage 1 complete. Found 184 total persons in 184 frames.2026-08-21T13:58:10.992494 -
2026-08-21T13:58:10.992494 - SDPose Node: Stage 2/3 - Running inference in batches of 1...2026-08-21T13:58:10.992494 -
2026-08-21T13:58:36.567256 - SDPose Node: Stage 2 complete. Processed 184 persons.2026-08-21T13:58:36.567256 -
2026-08-21T13:58:36.567256 - SDPose Node: Stage 3/3 - Reconstructing frames...2026-08-21T13:58:36.567256 -
2026-08-21T13:58:37.748503 - SDPose Node: Stage 3 complete. Reconstruction finished.2026-08-21T13:58:37.749034 -
2026-08-21T13:58:38.508623 - SDPose Node: Unloading models from VRAM to CPU.2026-08-21T13:58:38.508623 -
2026-08-21T13:58:41.830652 - ^[[32m[INFO]^[[0m Loading and assigning model weights to device...
2026-08-21T13:58:41.952160 - ^[[32m[INFO]^[[0m -------------------------
2026-08-21T13:58:41.952160 - ^[[32m[INFO]^[[0m Transformer weights loaded:
2026-08-21T13:58:41.952160 - ^[[32m[INFO]^[[0m Device: cuda:0 | Memory: 1,171,497.69 MB
2026-08-21T13:58:41.952160 - ^[[32m[INFO]^[[0m Device: cpu | Memory: 11,160,176.01 MB
2026-08-21T13:58:41.952160 - ^[[32m[INFO]^[[0m Using 1133 LoRA weight patches for WanVideo model
2026-08-21T13:58:42.867491 - ^[[32m[INFO]^[[0m ------- Scheduler info -------
2026-08-21T13:58:42.872490 - ^[[32m[INFO]^[[0m Total timesteps: tensor([999, 937, 833, 624], device='cuda:0')
2026-08-21T13:58:42.872490 - ^[[32m[INFO]^[[0m Using timesteps: tensor([999, 937, 833, 624], device='cuda:0')
2026-08-21T13:58:42.873490 - ^[[32m[INFO]^[[0m Using sigmas: tensor([1.0000, 0.9375, 0.8333, 0.6249, 0.0000])
2026-08-21T13:58:42.873490 - ^[[32m[INFO]^[[0m ------------------------------
2026-08-21T13:58:43.845265 - ^[[32m[INFO]^[[0m Rope function: comfy
2026-08-21T13:58:44.881255 - ^[[32m[INFO]^[[0m WanAnimate: Face input torch.Size([1, 3, 184, 512, 512]) padded to shape torch.Size([1, 3, 241, 512, 512])
2026-08-21T13:58:45.244501 - ^[[32m[INFO]^[[0m WanAnimate: Pose images torch.Size([3, 184, 1280, 720]) padded to shape torch.Size([3, 241, 1280, 720])
2026-08-21T13:58:45.244501 - ^[[32m[INFO]^[[0m Sampling 181 frames in 3 windows, at 720x1280 with 4 steps
2026-08-21T14:03:17.195998 - ^[[32m[INFO]^[[0m [Sampling] Max allocated memory: max_memory=11.226 GB
2026-08-21T14:03:17.195998 - ^[[32m[INFO]^[[0m [Sampling] Max reserved memory: max_reserved=14.031 GB
2026-08-21T14:03:23.785380 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 334.08 seconds^[[0m
2026-08-21T14:49:00.517358 - ^[[32m[INFO]^[[0m got prompt
2026-08-21T14:49:00.668739 - ^[[1m^[[33m[WARNING]^[[0m WARNING: PlaySound.IS_CHANGED() missing 1 required positional argument: 'self'
2026-08-21T14:49:18.997948 - SDPose Node: Received 307 frames. Starting 3-stage process with batch_size=1.2026-08-21T14:49:18.997948 -
2026-08-21T14:49:18.997948 - SDPose Node: Ensuring models are on device: cuda:0...2026-08-21T14:49:18.997948 -
2026-08-21T14:49:18.998949 - SDPose Node: Moving 'unet' from cpu to cuda:0.2026-08-21T14:49:18.998949 -
2026-08-21T14:49:19.708166 - SDPose Node: Moving 'vae' from cpu to cuda:0.2026-08-21T14:49:19.708166 -
2026-08-21T14:49:19.859041 - SDPose Node: Moving 'decoder' from cpu to cuda:0.2026-08-21T14:49:19.859041 -
2026-08-21T14:49:19.865111 - SDPose Node: Instantiating pipeline...2026-08-21T14:49:19.865111 -
2026-08-21T14:49:19.866111 - SDPose Node: Stage 1/3 - Detecting/Collecting persons...2026-08-21T14:49:19.866111 -
2026-08-21T14:49:19.868112 - SDPose Node: No detector specified. Processing full image.2026-08-21T14:49:19.868112 -
2026-08-21T14:49:24.457674 - SDPose Node: Stage 1 complete. Found 307 total persons in 307 frames.2026-08-21T14:49:24.457674 -
2026-08-21T14:49:24.457674 - SDPose Node: Stage 2/3 - Running inference in batches of 1...2026-08-21T14:49:24.457674 -
2026-08-21T14:50:09.084019 - SDPose Node: Stage 2 complete. Processed 307 persons.2026-08-21T14:50:09.084549 -
2026-08-21T14:50:09.084549 - SDPose Node: Stage 3/3 - Reconstructing frames...2026-08-21T14:50:09.084549 -
2026-08-21T14:50:11.974557 - SDPose Node: Stage 3 complete. Reconstruction finished.2026-08-21T14:50:11.974557 -
2026-08-21T14:50:13.718211 - SDPose Node: Unloading models from VRAM to CPU.2026-08-21T14:50:13.718211 -
2026-08-21T14:50:18.169659 - ^[[32m[INFO]^[[0m Loading and assigning model weights to device...
2026-08-21T14:50:18.269760 - ^[[32m[INFO]^[[0m -------------------------
2026-08-21T14:50:18.270762 - ^[[32m[INFO]^[[0m Transformer weights loaded:
2026-08-21T14:50:18.270762 - ^[[32m[INFO]^[[0m Device: cuda:0 | Memory: 1,171,497.69 MB
2026-08-21T14:50:18.270762 - ^[[32m[INFO]^[[0m Device: cpu | Memory: 11,160,176.01 MB
2026-08-21T14:50:18.270762 - ^[[32m[INFO]^[[0m Using 1133 LoRA weight patches for WanVideo model
2026-08-21T14:50:19.160470 - ^[[32m[INFO]^[[0m ------- Scheduler info -------
2026-08-21T14:50:19.167827 - ^[[32m[INFO]^[[0m Total timesteps: tensor([999, 937, 833, 624], device='cuda:0')
2026-08-21T14:50:19.168358 - ^[[32m[INFO]^[[0m Using timesteps: tensor([999, 937, 833, 624], device='cuda:0')
2026-08-21T14:50:19.168881 - ^[[32m[INFO]^[[0m Using sigmas: tensor([1.0000, 0.9375, 0.8333, 0.6249, 0.0000])
2026-08-21T14:50:19.168881 - ^[[32m[INFO]^[[0m ------------------------------
2026-08-21T14:50:19.952146 - ^[[32m[INFO]^[[0m Rope function: comfy
2026-08-21T14:50:20.630227 - ^[[32m[INFO]^[[0m WanAnimate: Face input torch.Size([1, 3, 307, 512, 512]) padded to shape torch.Size([1, 3, 321, 512, 512])
2026-08-21T14:50:21.077383 - ^[[32m[INFO]^[[0m WanAnimate: Pose images torch.Size([3, 307, 1280, 720]) padded to shape torch.Size([3, 321, 1280, 720])
2026-08-21T14:50:21.077899 - ^[[32m[INFO]^[[0m Sampling 305 frames in 4 windows, at 720x1280 with 4 steps
2026-08-21T14:56:27.066952 - ^[[32m[INFO]^[[0m [Sampling] Max allocated memory: max_memory=11.226 GB
2026-08-21T14:56:27.067473 - ^[[32m[INFO]^[[0m [Sampling] Max reserved memory: max_reserved=14.594 GB
2026-08-21T14:56:34.481821 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 453.88 seconds^[[0m
2026-08-21T14:56:35.133471 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
handle: <Handle _ProactorBasePipeTransport._call_connection_lost(None)>
Traceback (most recent call last):
File "E:\ComfyUI-aki-v1.7\python\Lib\asyncio\events.py", line 84, in _run
self._context.run(self._callback, *self._args)
File "E:\ComfyUI-aki-v1.7\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
self._sock.shutdown(socket.SHUT_RDWR)
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-21T15:28:38.686225 - [GLM_Nodes] 信息：从 'text_prompts.txt' 加载提示词成功。2026-08-21T15:28:38.686225 -
2026-08-21T15:28:38.686225 - [GLM_Nodes] 信息：从 'text_prompts.txt' 加载提示词成功。2026-08-21T15:28:38.686225 -
2026-08-21T15:28:38.686746 - [GLM_Nodes] 信息：从 'image_prompts.txt' 加载提示词成功。2026-08-21T15:28:38.686746 -
2026-08-21T15:28:38.687379 - [GLM_Nodes] 信息：从 'image_prompts.txt' 加载提示词成功。2026-08-21T15:28:38.687379 -
2026-08-21T15:30:04.480293 - ^[[32m[INFO]^[[0m got prompt
2026-08-21T15:30:07.177080 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
2026-08-21T15:30:07.303202 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-08-21T15:30:11.724958 - ^[[32m[INFO]^[[0m Requested to load MiniMaxMusic3TEModel
2026-08-21T15:30:11.776748 - ^[[32m[INFO]^[[0m loaded completely; 8759.61 MB loaded, full load: True
2026-08-21T15:30:11.785975 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cuda:0, dtype: torch.float16
2026-08-21T15:33:59.496790 - ^[[32m[INFO]^[[0m model weight dtype torch.float16, manual cast: None
2026-08-21T15:33:59.499790 - ^[[32m[INFO]^[[0m model_type FLOW
2026-08-21T15:34:01.177360 - ^[[32m[INFO]^[[0m Requested to load MiniMaxMusic3
2026-08-21T15:34:01.819919 - ^[[32m[INFO]^[[0m loaded completely; 84774.98 MB usable, 4686.50 MB loaded, full load: True
2026-08-21T15:34:15.840834 - ^[[32m[INFO]^[[0m Requested to load MiniMaxMusic3DAV
2026-08-21T15:34:15.888106 - ^[[32m[INFO]^[[0m loaded completely; 206.60 MB loaded, full load: True
2026-08-21T15:34:17.907180 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 253.43 seconds^[[0m
2026-08-21T15:57:35.100397 - ^[[32m[INFO]^[[0m got prompt
2026-08-21T15:57:35.244476 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! music_idea is required.
2026-08-21T15:57:35.248161 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 545, in execute
output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 344, in get_output_data
return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 318, in _async_map_node_over_list
await process_inputs(input_dict, i)
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 306, in process_inputs
result = f(**inputs)
^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
return method(locked_class, **inputs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
to_return = cls.execute(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 2347, in execute
result = enhance_music3_prompt(
^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 1754, in enhance_music3_prompt
raise Music3PromptEnhancerError("music_idea is required.")
E:\ComfyUI-aki-v1_x_7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8.music3.Music3PromptEnhancerError: music_idea is required.

2026-08-21T15:57:35.249197 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.15 seconds^[[0m
2026-08-21T16:01:03.340234 - ^[[32m[INFO]^[[0m got prompt
2026-08-21T16:01:04.006782 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! OpenAI-compatible provider Music 3 stage 'lyrics_generation' failed: status=400, category=request_rejected, attempts=1. Upstream response text was hidden for privacy.
2026-08-21T16:01:04.007306 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 545, in execute
output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 344, in get_output_data
return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 318, in _async_map_node_over_list
await process_inputs(input_dict, i)
File "E:\ComfyUI-aki-v1.7\ComfyUI\execution.py", line 306, in process_inputs
result = f(**inputs)
^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
return method(locked_class, **inputs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
to_return = cls.execute(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 2347, in execute
result = enhance_music3_prompt(
^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 1934, in enhance_music3_prompt
final_lyrics, generated_profile = _generate_or_edit_lyrics(
^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 1287, in _generate_or_edit_lyrics
response = runner.complete(
^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 424, in complete
result = _request_music_completion(
^^^^^^^^^^^^^^^^^^^^^^^^^^
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 721, in _request_music_completion
_raise_music_http_error(
File "E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8\music3.py", line 659, in _raise_music_http_error
raise Music3PromptEnhancerError(
E:\ComfyUI-aki-v1_x_7\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8.music3.Music3PromptEnhancerError: OpenAI-compatible provider Music 3 stage 'lyrics_generation' failed: status=400, category=request_rejected, attempts=1. Upstream response text was hidden for privacy.

2026-08-21T16:01:04.008876 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.66 seconds^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)

### Safety check

- [x] I removed API keys, private URLs, prompt/lyrics bodies, personal paths, and private media.

> **Comment by T8mars (2026-08-24T13:18:34Z):**

感谢反馈，已复核并确认此问题自 v1.3.0 起已修复。节点现在会自动识别 `https://api.kimi.com/coding/`（包括其 `/v1/chat/completions` 形式），在 AUTO 兼容策略下不向 Kimi Coding Plan 发送 `temperature`；MiniMax H3、Seedance 2.0 与 Music 3 共用该请求策略，Music 3 的所有生成阶段也已覆盖。当前 v1.4.2 的 Kimi 请求回归、Music 3 全阶段测试均通过。请通过 ComfyUI Manager 或 GitHub 更新到 v1.4.2，并重启 ComfyUI。若 v1.4.2 仍返回 400，欢迎继续提供脱敏诊断和实际 Base URL 结构，我们会继续跟进。现按已修复关闭。



### #7 [closed] ISSUE: 已完成：MiniMax H3 目标时长上限放宽至 30 秒
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/7  by qqlory 2026-08-15T03:42:27Z

作者好！MiniMaxH3Director、ComfyUI-H3-Motion-Context 等节点已支持多段拼接生成长视频，但 `MiniMaxH3PromptEnhancerT8` 的“目标时长”被限制在 4–15 秒，在规划多段长视频时不够用。建议解除或放宽该限制。

涉及位置（nodes.py）：

1. UI 控件：`io.Int.Input("duration_seconds", ..., min=4, max=15)` 的 `max=15`
2. 后端校验：`if not 4 <= int(duration_seconds) <= 15: raise PromptEnhancerError("duration_seconds must be between 4 and 15.")`
3. 提示词文本中 3 处写给模型的 “4-15 second” 描述

---

Hi! Nodes like MiniMaxH3Director and ComfyUI-H3-Motion-Context already support multi-segment long-video generation, but `MiniMaxH3PromptEnhancerT8` caps `duration_seconds` at 4–15s, which is limiting when planning multi-segment videos. Suggest relaxing or removing the cap in:

1. The UI input `max=15`
2. The backend validation `duration_seconds must be between 4 and 15`
3. Three “4-15 second” mentions in the LLM prompt text

> **Comment by T8mars (2026-08-16T14:42:01Z):**

这个需求成立，但不会直接把单次 `duration_seconds` 的上限从 15 秒改大。MiniMax H3 的单段生成合同仍是 4–15 秒；直接让一条提示词声明 30/60 秒，会把“长片总时长”和“单次模型时长”混在一起。

已确定后续采用独立的多段长视频规划模式：

- 总片长 `total_duration_seconds`；
- 每段仍为 4–15 秒；
- 自动或指定分段数量；
- 跨段 continuity anchors（人物、场景、动作轴、光线、声音、结尾状态）；
- 每段独立 H3 提示词；
- 尾态→下一段首态的交接表；
- 可选硬切、动作接力、首尾帧接力或轨道补齐策略。

验收要求是每段单独通过现有 H3 合同，总时长与段数闭合，跨段身份和空间连续性可追踪，并且不把长片总时间码错误写进单段提示词。首版只做规划与提示词输出，不自动调用视频生成或拼接。

该项已进入维护路线图；Issue 保持开放，作为实现和验收跟踪入口。



> **Comment by T8mars (2026-08-16T14:44:25Z):**

更新说明：根据最新产品决定，MiniMax H3 节点的单次目标时长上限已在 `fffe31c` 从 15 秒开放到 30 秒。

本次同步修改了 UI 范围、后端校验、H3/MV/3D 提示词协议、README 和边界测试；30 秒输入会明确传入 LLM，31 秒仍在网络请求前拒绝。Seedance 2.0 的独立时长合同保持不变。完整回归为 143 项通过。前一条仅规划多段模式的说明已由本次实际实现取代，感谢你的建议。


> **Comment by T8mars (2026-08-16T15:08:01Z):**

再次确认并完成收尾：

- MiniMax H3 的 duration_seconds 已正式调整为 **4–30 秒**；
- UI 控件最大值为 30；
- 后端校验接受 30，并在任何网络请求前拒绝 3 秒和 31 秒；
- H3、MV 与 3D 提示词协议已统一为 4–30 秒；
- README 与本地维护路线图已同步；
- Seedance 2.0 仍保持它自己的 4–15 秒官方合同，不受此次修改影响；
- 使用整合包自带 Python 环境完成全量回归：**143/143 tests PASS**。

实现提交：fffe31c。因此本 Issue 按已完成关闭；超过 30 秒的长视频仍会作为独立“多段规划”能力继续设计，而不是突破单次生成合同。




### #6 [closed] PR: fix: support /vN-versioned OpenAI-compatible base URLs (e.g. Volcano Ark /api/v3)
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/6  by khusdasz-cmd 2026-08-12T13:05:42Z

## 问题

`_openai_chat_url` 只识别以 `/v1` 结尾的 Base URL，其他版本化地址会被错误拼成 `<base>/v1/chat/completions`。

例如火山方舟（Volcano Ark）官方文档给出的 Base URL 是 `https://ark.cn-beijing.volces.com/api/v3`，节点会请求：

```
https://ark.cn-beijing.volces.com/api/v3/v1/chat/completions
```

多了一层 `/v1`，服务端返回 HTTP 404（空响应体），报错 `OpenAI-compatible provider chat request failed (HTTP 404)`。

## 修复

将判断从 `endswith("/v1")` 放宽为匹配任意 `/vN` 版本段（`re.search(r"/v\d+$", base_url)`），统一追加 `/chat/completions`：

- `https://ark.cn-beijing.volces.com/api/v3` → `…/api/v3/chat/completions` ✅
- `https://gateway.example/v1` → `…/v1/chat/completions`（行为不变）
- `https://gateway.example` → `…/v1/chat/completions`（行为不变）
- 完整 `/chat/completions` 地址原样使用（行为不变）

README 同步更新了 Base URL 填写说明。

## 验证

- 本地 70 个单元测试全部通过（`tests/test_nodes.py`）
- 修复对两个节点（MiniMax H3 / Seedance 2.0）同时生效，因为 `seedance20.py` 复用 `nodes.py` 中的 `_openai_chat_url`

> **Comment by T8mars (2026-08-16T14:41:50Z):**

感谢你准确定位 `/api/v3` 被错误拼成 `/api/v3/v1/chat/completions` 的问题。

我已基于当前 `main` 在 a32ed998cd749a3af3acdc7cb6ce37b4336dc698 完成等价修复，并补充了自动测试矩阵，覆盖：服务根地址、`/v1`、`/api/v3`、`/v12`、尾部斜杠和完整 `/chat/completions` 地址。完整插件回归为 140 项通过（隔离环境下 1 项本地 GIF 配置测试按设计跳过）。

由于本 PR 已与最新主线冲突，且原提交没有包含针对该 URL 归一化的测试，因此采用了在最新主线上重新实现并验证的版本。再次感谢贡献，这个兼容性问题现在已进入 `main`。



> **Comment by T8mars (2026-08-16T14:41:53Z):**

已由 main 上的 a32ed998cd749a3af3acdc7cb6ce37b4336dc698 替代实现并验证，关闭此冲突分支 PR。


> **Comment by T8mars (2026-08-16T14:44:23Z):**

补充更新：`/vN` 修复已进入 `main`，并在 `fffe31c` 上进一步完善为基于 URL 路径的规范化，而不是只对原始字符串做后缀判断。

当前覆盖服务根地址、`/v1`、`/api/v3`、`/v12`、大小写版本段、尾部斜杠和完整 `/chat/completions`；同时分别验证 MiniMax H3、Seedance 2.0 与 Music 3 三个节点使用 `/api/v3`。完整回归为 143 项通过。感谢你准确定位火山方舟兼容问题。



### #5 [closed] ISSUE: 使用openai 接口，圖片連入first_frame 或是 reference_image，執行後就會發生錯誤
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/issues/5  by Copipi 2026-08-11T16:45:40Z

會出現
執行失敗
節點在執行時發生錯誤。
MiniMax H3 Prompt Enhancer (Seedance / AI Workshop / OpenAI)



如果不連入，單純文生圖，使用openai 接口，是可以運作成功
我是連本地的qwen3.6 模型，有執行多模態

> **Comment by Copipi (2026-08-11T16:46:58Z):**

搞錯，要選 生成類型 



### #4 [closed] PR: fix: retry transient API failures across network routes
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/4  by gluttony-10 2026-08-09T14:57:02Z

## Summary

- retry transient upload and chat failures across direct and configured proxy routes
- use the fixed direct → proxy → direct → proxy order with 1s, 5s, and 10s delays
- retry transport failures and HTTP 429/502/503/504 while leaving permanent failures single-attempt
- use fresh production sessions for every route attempt

## Why

Temporary gateway failures, rate limits, connection resets, and TLS EOF errors currently abort requests immediately. The route-aware retry flow makes the node resilient to those short-lived failures.

Chat requests are not inherently idempotent, so a timeout after the server accepted a request can result in duplicate billing. The implementation logs an explicit warning when such a retry occurs.

## Validation

- 85 Python tests passed
- 2 frontend tests passed
- verified four-attempt route order and delay sequence
- verified HTTP 500 and permanent 4xx responses are not retried


> **Comment by T8mars (2026-08-16T14:41:56Z):**

感谢你为临时网关故障和代理环境提出完整重试方案。我们没有原样采用“直连→代理→直连→代理、最多四次”的实现，原因是 Chat 请求不是幂等操作：读取超时、连接在请求提交后断开或 429 场景中，服务端可能已经接受并计费，再自动重放会造成重复付费；自动切换直连/系统代理也会绕过用户明确配置的网络路线。

当前 `main` 已采用更保守且有测试覆盖的方案：

- 仅对平价小屋固定 Chat 端点的 SSL/建连故障和明确网关状态做 0.5 秒、1 秒的有界快速重试；
- ReadTimeout、401、402、429 和自定义 OpenAI 兼容接口不自动重试；
- 不在直连与系统代理之间自行切换；
- 免费素材上传遇到 429 时最多重试一次；
- 错误信息说明是否发生过自动尝试，并保持密钥和上游正文脱敏。

相关回归已覆盖短重试成功、重试耗尽、读取超时不重试、自定义接口不继承策略、429 不重试与密钥脱敏。本轮又在 README 明确写入“不自动切换网络路线”。因此本 PR 的稳定性目标已经以更安全的策略落地，但四次跨路线付费重放不会进入主线。



> **Comment by T8mars (2026-08-16T14:41:58Z):**

主线已采用有界、渠道限定且避免重复计费的替代策略；关闭此已冲突的旧实现。


> **Comment by T8mars (2026-08-16T14:44:21Z):**

更新说明：根据最新维护决定，我们在 `fffe31c` 吸收了本 PR 中“直连 / 系统代理切换”的有价值部分，并按当前主线的付费安全边界重新实现。

现在 `api.seedance.nz` 的安全重试路线为“环境代理 → 显式直连 → 环境代理”；覆盖 H3、Seedance 2.0、Music 3 的聊天阶段，以及 Seedance 素材上传。仍然不会重试读取超时、401/402/429，也不会把该策略扩散到用户自定义 OpenAI 兼容接口，从而避免状态不明的付费请求被重复提交。

新增测试验证了三次路线顺序、上传路线切换、Music 3 六次参考选择阶段的交替路线，以及自定义接口不继承该策略。完整回归为 143 项通过。前一条“不自动切换网络路线”的说明已由本次实现取代。感谢你提出这项可靠性思路。



### #3 [closed] PR: Add AI Workshop multimodal provider
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/3  by T8mars 2026-08-05T18:16:08Z

## What changed

- Added 贞贞的 AI 工坊 as a dedicated API mode for both prompt enhancer nodes.
- Uses `gemini-3.5-flash` by default and exposes an explicit Custom model option.
- Sends images and complete video bytes inline without requiring a separate upload URL.
- Added conditional ComfyUI widgets, API-key signup handling, and legacy workflow migration.
- Updated example workflows, README documentation, unit tests, and a paid multimodal smoke test.

## Why

The existing nodes supported only Seedance.nz and a generic OpenAI-compatible path. The AI 工坊 gateway needs a provider-specific media representation to perform reliable Gemini video understanding.

## Root cause and protocol fix

The reference implementation used an OpenAI `video_url` part. Live requests returned HTTP 200 but produced incorrect video facts. A minimal protocol probe showed that this gateway correctly processes a `data:video/mp4;base64,...` URL only when carried in an OpenAI `image_url` multimodal part. The dedicated provider path now uses that verified representation while preserving the complete video stream.

## User impact

Users can select AI 工坊 directly, use the verified default visual model, or choose Custom and enter another model ID. Existing workflows retain their previous API mode and receive default values for the new widgets during load.

## Validation

- `74` unit tests passed.
- Python compilation, both frontend JavaScript syntax checks, and example JSON parsing passed.
- Live model listing confirmed `gemini-3.5-flash` exists.
- Paid end-to-end smoke tests passed for both nodes with one image plus a complete four-second two-phase video; exact text, colors, shapes, motion direction, and temporal order were recognized.
- The supplied test API key and ignored `roadmap.md` are not included in the commit.



### #2 [closed] PR: Add Seedance 2.0 enhancer and MiniMax H3 skill presets
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/2  by T8mars 2026-08-05T14:34:23Z

## What changed

- add an independent Seedance 2.0 prompt enhancer with reference-media support, shot-count controls, and Seedance/OpenAI-compatible transports
- add MiniMax-H3 compatibility and strict official Skill profiles plus all eight official prompt-writing presets
- deepen the H3 MV preset with exact lyric locking, conditional performance, spatial typography, evidence-based timing, reference-role isolation, and style-gated editing grammar
- gate AUTO MV rules to explicit trusted MV intent so ordinary product text does not trigger them
- add MV-specific dynamic placeholders/tooltips without changing the 19-value workflow serialization contract
- update documentation and regression coverage without including an API key or `roadmap.md`

## Why

The node needed to incorporate the useful parts of MiniMax's official Skills while preserving its single-prompt scope and existing workflows. The MV update turns the prior summary preset into controllable prompt-writing rules without pretending to analyze audio, generate assets, or edit long-form video.

## Behavior and compatibility

- existing workflows retain the localized compatibility profile and current serialized widget order
- strict Skill mode writes descriptive sections in English while preserving exact original dialogue, lyrics, and visible text
- MV lyrics come only from trusted user text; singer/lip-sync, exact beat timing, and aggressive cut/glitch grammar are conditional
- non-empty upstream output still passes through; complete recognized field sets are reordered only when needed
- no AUDIO socket, remote Skill execution, H3 video generation, paid retry, clip stitching, or delivery workflow was added

## Validation

- 70 H3 and Seedance 2.0 unit tests pass
- H3 frontend JavaScript syntax check passes
- `git diff --check` passes
- `roadmap.md` remains ignored and API keys are absent from committed files



### #1 [closed] PR: Add shot count control
URL: https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8/pull/1  by T8mars 2026-08-04T07:46:37Z

## What changed

- add an AUTO or fixed 1–20 shot-count dropdown
- send the selected shot-count rule to both system and user LLM instructions
- migrate existing 16-value workflows to AUTO without shifting later settings
- update the example workflow, README, and regression coverage

## User impact

Users can let the model choose the shot count or request an exact number of `[Shot N]` timeline entries. Non-empty upstream output is still passed through even if the model does not fully follow the requested count.

## Checks

- 46 unit tests
- Python compile check
- JavaScript syntax check
- example workflow JSON parse
- staged secret scan
