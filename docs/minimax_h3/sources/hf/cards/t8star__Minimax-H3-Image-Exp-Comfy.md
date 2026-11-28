# t8star/Minimax-H3-Image-Exp-Comfy

Источник: https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy · скачано 2026-11-28
Загрузок: 34314 · лайков: 15 · базовая: ["MiniMaxAI/MiniMax-H3-Image"] · теги: comfyui,lora · обновлено: 2026-11-21T08:52:35.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
base_model:
- MiniMaxAI/MiniMax-H3-Image
pipeline_tag: text-to-image
language:
- zh
- en
tags:
- comfyui
- minimax-h3
- image-generation
- image-editing
- lora
- experimental
---

# MiniMax-H3-Image-Exp-Comfy

**完整未裁剪 BF16/FP32 发布主干 + 多步材质近似修正实验 / ComfyUI。**

## 快速开始：50 步工作流先装节点

> **必装自定义节点：[H3 Image Single Head · T8（GitHub）](https://github.com/T8mars/Comfyui-h3-image-single-head-T8)。** 只下载模型和 LoRA 不会自动安装节点。工作流里 `H3ImageSingleHead` 显示红叉时，请先安装节点并重启 ComfyUI。
>
> **只跑原生 6 步 CFG1？** 用 [原生六步工作流](workflows/native_pdd6)，不装此节点，也不加载 direct MLP100。

**EN:** The supplied 50-step direct MLP100 workflows require the [H3 Image Single Head node](https://github.com/T8mars/Comfyui-h3-image-single-head-T8). The native 6-step workflows do not.

1. **安装节点：**[查看 GitHub 安装说明](https://github.com/T8mars/Comfyui-h3-image-single-head-T8#安装) 或 [下载节点 ZIP](https://github.com/T8mars/Comfyui-h3-image-single-head-T8/archive/refs/heads/main.zip)，解压到 `ComfyUI/custom_nodes/`，然后重启。
2. **放好模型：**按下方表格放置完整主干、direct MLP100、文本编码器和 VAE。
3. **导入一个工作流：**第一次用就选 [推荐文生图 JSON](workflows/direct_mlp/violin_maker.json)；有参考图再选 [图生图 / 编辑 JSON](workflows/direct_mlp/shop_wall_edit.json)。

```text
UNETLoader → H3ImageSingleHead → LoraLoaderBypassModelOnly (+1) → CFGGuider (3)
```

推荐工作流已连好，使用 Euler 50 步和内置 shift 3 采样表。[查看完整效果对照](EXAMPLES.md)。

Powered by MiniMax-H3-Image。这是 t8star 的第三方研究与打包仓库，不是 MiniMax 官方新版本，不是 FineView 权重，也不是已剥离 PDD 的原版教师。

## 先说结论：什么值得用？

1. **要完整发布模型：**下载 `diffusion_models/minimax_h3_image_full_bf16_comfy_qkvfix-T8.safetensors`。66.29 GB，保留完整 AdaLN 与发布精度，不是 INT8 反量化，也不是 pruned 模型。**它仍然含 PDD，正常六步路线没有被去掉。**
2. **要试多步下更自然的材质：**使用上面的主干 + 本仓库单头节点 + `loras/experimental_direct_mlp100_rank16-T8.safetensors`。这是目前本项目优先尝试的近似修正候选。实测设置为 **Euler 50 步、CFG 3、shift 3、LoRA 正强度 +1**。不是六步 CFG1，也不是简单调整 SHIFT。
3. **不要把最新日期当成最好：**`experimental/rejected/REJECTED_edit_preservation_mlp100_rank16_DIAGNOSTIC_ONLY.safetensors` 未通过训练数值预算检查，实图也没有证明整体优于前一版。仅供失败诊断，**不推荐替换 direct MLP100**。

没有找到匹配的原始图像 PDD 主干增量，也没有未知教师权重可供对照；不能声称“剥离完成”“99% 原版还原”。8/10 是这批固定样本的人工联合检查计数，不是教师相似度，不是大规模盲测成绩。

## 下载什么、放在哪里？

|文件/目录|用途|安装位置|建议|
|---|---|---|---|
|[完整主干](diffusion_models/minimax_h3_image_full_bf16_comfy_qkvfix-T8.safetensors)|完整发布权重的 ComfyUI 格式转换，66.29 GB|`ComfyUI/models/diffusion_models/`|基础必需|
|[direct MLP100](loras/experimental_direct_mlp100_rank16-T8.safetensors)|独立训练的材质近似补偿，172.07 MB|`ComfyUI/models/loras/`|多步实验优先尝试|
|[文本编码器](text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors)|测试使用的 INT8 文本编码器，27.14 GB|`ComfyUI/models/text_encoders/`|已有同文件可复用|
|[VAE](vae/minimax_h3_image_vae_fp16.safetensors)|测试使用的 FP16 图像 VAE，1.17 GB|`ComfyUI/models/vae/`|已有同文件可复用|
|[单头节点 · GitHub](https://github.com/T8mars/Comfyui-h3-image-single-head-T8)|`__init__.py` 和 `mean_head.safetensors` 必须一起保留|整个目录放入 `ComfyUI/custom_nodes/`|50 步必装；原生六步不使用|
|[历史模型清单](MODEL_INDEX.md)|全部已整理的独立推理导出、历史训练适配器和诊断权重|按各条说明|不要一次性叠加|

完整主干是 **BF16 + FP32 混合存储**；文本编码器是 INT8，VAE 是 FP16。这里**不是全链路 BF16**。未重新上传官方原始十二分片、视频全模型、优化器状态或训练缓存；完整转换主干与本研究的独立模型导出集中在本仓库，第三方文件来源分别标注。

文件大小、SHA256、来源、修改状态见 [模型清单](MODEL_INDEX.md) 和 [机器可读清单](MODEL_MANIFEST.json)。体积单位 GB/MB 为十进制。不要直接下载整个历史实验目录当成一个模型。

## 安装与复现

线上主干和 direct MLP100 文件名带 `-T8`，权重内容不变。若你已下载旧名文件，无需重下；在工作流中从加载器下拉菜单重新选择本地文件即可。

使用支持 MiniMax H3 Image 的 ComfyUI。研究环境基于 ComfyUI `989d93d0d`；需要原生 `MiniMaxH3ImageConditioning`、`LoraLoaderBypassModelOnly`、`ManualSigmas` 和 `CFGGuider`。如果节点不存在，请更新到包含这些节点的版本；不要用普通 LoRA 合并加载器替代旁路加载器。

1. 按上表放置主干、文本编码器、VAE、direct MLP100。
2. 安装 [H3 Image Single Head · T8](https://github.com/T8mars/Comfyui-h3-image-single-head-T8)，然后重启 ComfyUI。节点仓库已包含所需头资产，无需另下；安装命令见下方。如果已安装旧的 `h3_image_single_head`，请移出旧目录，避免重复加载。
3. 导入 [workflows/direct_mlp](workflows/direct_mlp) 下普通 `.json`；`.api.json` 是程序提交格式，不是普通界面工作流。
4. 照片编辑示例先将 [inputs/prepared.png](inputs/prepared.png) 保存到 `ComfyUI/input/prepared.png`，然后在 LoadImage 中选中它。可以换自己的图片，但不应把换图结果当成本仓库固定样本复现。
5. 使用工作流保存的显式 51 点 sigma 表，不需要再叠加 SHIFT 节点；保留正条件与单独的空文本负条件。图像编辑时正负分支均携带同一参考图。

在 ComfyUI 根目录执行以下命令安装节点（也可下载 GitHub ZIP 解压至 `custom_nodes`）：

```bash
git clone https://github.com/T8mars/Comfyui-h3-image-single-head-T8.git custom_nodes/Comfyui-h3-image-single-head-T8
```

搜索节点 **H3 Image 单头多步（实验）**，节点 ID 为 `H3ImageSingleHead`。[Hugging Face 节点镜像](custom_nodes/h3_image_single_head) 与 GitHub 二选一安装，不要同时安装。

多步加载顺序：

```text
UNETLoader → H3ImageSingleHead → LoraLoaderBypassModelOnly → CFGGuider
```

单头节点必须紧接未经 LoRA 修改的主干。均值头只是发布六组 PDD 头按区间加权得到的单头，不是蒸馏前教师头；节点只修改内存中的克隆，不覆盖主干文件。修正 LoRA 用 **+1，不是 -1**。不要叠加旧 v2、PAI 或其它实验补偿。

文生图九项为 2048×2048，照片编辑为 2048×1344。实际运行硬件是 RTX 4060 Ti 16 GB / 128 GB RAM；66 GB 权重需要大量系统内存与卸载，16 GB 显存实测能跑不代表低内存机器同样可跑，也不保证速度。不同硬件、算子和 ComfyUI 版本不保证逐像素一致。

### 原生六步路线

[workflows/native_pdd6](workflows/native_pdd6) 保留完整发布模型的六步生成/编辑示例。使用发布 PDD 日程与 BasicGuider（等效 CFG1），**不接单头节点、不加载修正 LoRA**。这一路线用于原发布模型验证，不是去蒸馏。

## 示例图：多步材质改善到底是什么？

下列均为 **AI 生成图**，未经锐化、磨皮或其它后期处理。左为同一完整主干、均值头、50 步 CFG3、不加载修正；右为只加入 direct MLP100。每一对保持提示词、seed、分辨率、采样器与 sigma 网格一致。

**这不是“官方原生六步 vs 修正版”的胜率测试。** 50 步无修正控制本身会出现明显硬刻纹；不能由它推导对原生六步普遍更好。不要把构图和人物细节变化误称为原样恢复。

|题材|50 步无修正控制|direct MLP100 / +1|
|---|---|---|
|老人皮肤|![AI-generated control portrait](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/off50/violin_maker.png)|![AI-generated direct MLP portrait](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/direct_mlp/violin_maker.png)|
|剪刀与亚麻|![AI-generated control linen](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/off50/scissors_linen.png)|![AI-generated direct MLP linen](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/direct_mlp/scissors_linen.png)|
|牧羊犬|![AI-generated control sheepdog](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/off50/sheepdog.png)|![AI-generated direct MLP sheepdog](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/direct_mlp/sheepdog.png)|
|水彩茶舍|![AI-generated control watercolor](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/off50/watercolor_teahouse.png)|![AI-generated direct MLP watercolor](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/direct_mlp/watercolor_teahouse.png)|

主要观察：皮肤的矩形硬刻块减少，仍保留年龄细纹；亚麻从断裂浮雕感变为连续织纹；毛发减少硬刻感；水彩降低过强纸面浮雕，同时保留文字和水彩风格。不是全部细节都变好：局部细节会简化，面孔、物体位置和构图会变化。

**十组完整三列原图、逐项评价、完整提示词/seed 与对应工作流： [EXAMPLES.md](EXAMPLES.md)。** 第三列是未通过数值门槛的新候选，供公开失败对照，不作为推荐。

|固定样本|direct MLP100 观察|本批联合检查|
|---|---|---|
|剪刀与亚麻|连续细织纹、V 形开口和双环保留；构图变化|通过|
|老人皮肤|硬刻块减轻，发须与年龄细纹保留；脸部细节变化|通过|
|雨伞|红伞布面更自然，但仍缺红伞完整长杆和手柄|**失败**|
|雨后书店|四字可读，木石硬浮雕减轻，仍有粗糙纹理|通过|
|陶艺师与碗|皮肤/布料改善，单碗双手可读；部分手指柔化|通过|
|墙字局部编辑|大字基本去除；顶部浅痕、人物与非目标细节漂移|**失败**|
|修车人|本固定样本双 U 形扳手开口露出；不是工具通用修复|通过|
|牧羊犬|本固定样本四爪/耳/尾完整入框；斑纹和头向变化|通过|
|玻璃杯与双把碗|材质区分、三把连接保留；接点不规则、部分高光偏亮|通过|
|水彩茶舍|“山间茶舍”、两花盆和水彩保留；建筑细节简化|通过|

### 失败示例也保留

|缺柄问题没有解决|局部编辑没有做到非目标内容不变|
|---|---|
|![AI-generated umbrella failure](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/direct_mlp/umbrellas.png)|![AI-edited photo with preservation failure](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/examples/direct_mlp/shop_wall_edit.png)|

编辑照片原作：Faisal Akram，Portrait of an Old Man，CC BY-SA 2.0。本仓库准备版及 AI 编辑版本保留署名与同许可；详见 [ATTRIBUTION.md](ATTRIBUTION.md)。不要将 AI 编辑结果称为原作者作品。

## 权重做了什么？

- 完整主干：官方 native 分片转换成 ComfyUI 所需命名和 QKV 排布；保留未裁剪 AdaLN，原混合精度与非转换张量。PDD 六头转换为首头绝对值 + 后续偏移的布局，有 FP32 运算舍入，非逐字节复制所有头。转换依据及逐张量记录见 [provenance](provenance)。
- direct MLP100：冻结主干，独立训练 224 次更新，覆盖 50 层的 100 个 MLP 模块；rank16/alpha16，200 个 FP32 因子，0 注意力适配器。是新训练的近似补偿，不是原加速 LoRA 的负向相减。
- 旧 `full50_mlp_only_rank16`：联合训练后拆出的 MLP 子集，保留较多材质收益，但不等于独立训练 direct MLP100。不能改名混用。
- rejected 编辑候选：额外 176 次更新，五个编辑锚点超过锁定数值预算。工程导出与四项原生全网络一致性通过，不等于数值/画质门槛通过；十图仍 8/10，没有明确整体新增提升。
- PAI FL2VA / Ref2VA 负载诊断：有实际权重变化和局部材质变化，但没有证据说明它们就是官方 Image 六步 PDD 原始增量；不能拿负载成功当作恢复教师。

## 验证范围与限制

direct 与 rejected 各有十张真实 50 步图，每批累计 500 个采样步、1000 次正负条件分支/原生调用；并非仅改界面标签。direct 四项全网络导出一致性通过。原完整主干未覆盖。发布时再次检查文件哈希、工作流连线、依赖名称、采样参数和远程文件大小/哈希；没有为了发布重新抽样挑图。

已见固定样本的人工回归不是盲测，不同题材/seed 的泛化尚未证明。未证明已去除 CFG 蒸馏、恢复未知教师、复现 FineView 的未公开训练方法。历史失败模型只供研究追溯，不建议作为日常默认。

## 来源与许可

- 基座：[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)，转换源固定 revision `5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`。
- 文本编码器、VAE：[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)，按已测试文件镜像，保持原字节。
- 第三方训练适配器与 PAI 诊断源分别列于 [MODEL_INDEX.md](MODEL_INDEX.md)，不冒称为 t8star 原创。
- 图像权重及其衍生物受 [MiniMax-H3-Image Community License](LICENSE) 约束；H3 文本编码器/视频来源适配器另见 [MiniMax H3 许可](licenses/MiniMax-H3-LICENSE)。相关 NOTICE 一并保留。
- 文本编码器的 Qwen3-VL 上游部分另附 [Apache 2.0 许可](licenses/Qwen3-VL-APACHE-2.0)，来源为 [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。这不将整个 MiniMax 衍生模型重新许可为 Apache。

模型文件和示例图开放下载，无需提交访问申请。

**不是 Apache/MIT 无限制模型。** 上游许可排除欧盟、英国、韩国和美国，并包含用途及商用要求；尤其 Image 许可对 Hosted Service 与收入门槛有单独授权规定。请阅读完整条款，README 摘要不能代替许可。开放下载不改变上游许可条款。

All generated example images are AI-generated. This repository contains experimental derivatives and a full-precision format conversion, **not a verified pre-distillation teacher or extracted original PDD LoRA**. The preferred experimental adapter is direct MLP100 at positive strength 1 with the supplied single-head node and 50-step CFG3 workflows. The newer rejected adapter is diagnostic only. There is no measured 99% teacher-recovery claim.

## T8star · 社媒与资源

- [B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [Hugging Face](https://huggingface.co/t8star)
- [API（贞贞的平价 AI 小屋）](https://www.zzai.app/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用（RunningHub）](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)

API 和 RunningHub 为邀请链接，使用本地节点和工作流不需要注册。
