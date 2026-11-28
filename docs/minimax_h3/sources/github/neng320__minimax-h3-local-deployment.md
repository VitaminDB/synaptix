# neng320__minimax-h3-local-deployment

Source: https://github.com/neng320/minimax-h3-local-deployment
Fetched: 2026-11-28

Description: MiniMax H3 300% speed本地部署指南（消费级 16GB 显存版）：全栈加速链 cu130+SageAttention+TE-Speed 实测、模型组合、避坑清单
Stars: 107  Updated: 2026-08-12T10:56:18Z



---

## FILE: README.md (https://github.com/neng320/minimax-h3-local-deployment/blob/HEAD/README.md)

# MiniMax H3 本地部署指南（消费级显卡版）

MiniMax H3 是 MiniMax 开源的全模态视频生成模型（文生/图生/参考生视频 + 原生音轨，最长 15 秒、最高 2K）。本仓库提供一套**在 16GB 显存消费级显卡上完整跑通 H3 并达到可生产速度**的实测指南（RTX 4060 Ti 16GB + 64GB 实测，2026-08-08 更新）。

## 🎯 核心能力

本项目的价值集中在两件事：

1. **🚀 3-5 倍加速工作流**：从官方模板的「每镜头 20-30 分钟」压到「预热后 3-5 分钟/条」（480p 档），核心加速链 = **cu130 硬件反量化（3-5×）+ SageAttention（~1.5-2×）+ T8 双时钟 turbo 采样（步数减半）**，叠加后实测提速 3-5 倍，画质无损。完整加速决策与实测数据见 [qa-benchmarks.md](qa-benchmarks.md)。
2. **📦 完整部署（可复现）**：模型 6 件套 + 33 节点选配 + 双版本工作流 + 一键脚本 + 换机恢复指南，**任何新机器照本仓库配置即可完整复现生产环境**（详见 [deployment-handbook.md](deployment-handbook.md) 与 `h3-restore-guide.md` 恢复指南）。

> 加速不是堆参数，而是**已验证的组合**：哪些该开（SageAttention）、哪些该弃（TE-Speed 失真）、哪些该分层（480p 抽卡 / 768p 精选）——全部有实测依据。

---

## ⚠️ 两个版本是不同的工作流，先选版本再看对应文档

| | **v1 版（产品/动画向）** | **v2 版（真人/写实向）** |
|---|---|---|
| 模型 | **pruned 20.9G**（`minimax_h3_fl2va_pruned_int8_convrot.safetensors`） | **完整版 INT8 34G**（`minimax_h3_fl2va_int8_convrot.safetensors`） |
| 擅长 | **产品视频、卡通/动画、快节奏抽卡** | **真人视频、写实广告、产品文字特写** |
| 速度 | **~109s/条**（480p 抽卡实测；1056×608 ~180s / 1344×768 ~360s） | **~300s/条**（Sage 预热后，640×640） |
| 画质 | 细节弱（剪枝损失，加分辨率补不回） | 语义准、细节完整 |
| 工作流 | `workflows/v1-pruned-product-animation/MiniMax-H3-双时钟采样8步-v1-pruned.json` | `workflows/v2-full-real-person/MiniMax-H3-双时钟采样8步.json` |
| 脚本 | `h3_i2v_run.py --model pruned` | `h3_i2v_run.py --model full --sage` |
| 测试结果 | [qa-benchmarks.md §一](qa-benchmarks.md#一-v1-版pruned-产品动画向-测试) | [qa-benchmarks.md §二](qa-benchmarks.md#二-v2-版full-真人写实向-测试) |

> **质量是模型决定的，不是像素决定的**：v1 的"别扭感"是剪枝丢层固有损失，加分辨率补不回来；真人/写实/文字特写需求**必须用 v2**。两套模型并存，脚本 `--model` 一键切换。

---

## 亮点

- ✅ **3-5 倍加速实测**：cu130 硬件反量化 + SageAttention + T8 双时钟 turbo，480p 从官方 20-30 分钟压到 **3-5 分钟/条**（预热后），画质无损
- ✅ **16GB 显存即可稳定运行**（INT8 量化 + ComfyUI 动态显存卸载，实测显存占用 ~14GB）
- ✅ **加速实测结论（2026-08-08，4060Ti）**：**SageAttention 是唯一可靠加速（v2 实测 300s/条，画质无损失）**；TE-Speed 虽快（165-195s）但**实测失真弃用**——详见 [qa-benchmarks.md](qa-benchmarks.md)
- ✅ **"480p 生产、768p 精选"策略**：抽卡 480p/640×640（v1 ~109s / v2 ~300s）→ 成片 1344×768（~600-900s）
- ✅ 官方推荐模型组合 + ModelScope 国内直连下载（断点续传）
- ✅ 完整避坑清单（分辨率 32 倍数、许可证地域限制、TE-Speed 两坑、常见报错）

---

## 实测性能（RTX 4060 Ti 16GB + 64GB 内存，2026-08-08 实测）

> 测试条件：T8 双时钟采样 8 步 + 5 秒视频。完整测试矩阵（分版本）见 [qa-benchmarks.md](qa-benchmarks.md)。
> ⚠️ **turbo LoRA 归属**：**仅 v2 实测使用**（`h3_i2v_run.py --model full` 脚本链含 turbo LoRA 节点）；**v1 无 turbo**（fc9369f 实测时 turbo LoRA 尚未发布，且 pruned INT8 不能完整应用 turbo LoRA）——两个版本工作流 json 均无 LoRA 节点。

### v1 版（pruned，产品/动画向）— 数据源 commit `fc9369f`（480p 实测修正）

| 分辨率 | 大小 | 耗时（提交→出片） | 说明 |
|---|---|---|---|
| 864×480 | 0.42MP | **~109s** | 抽卡档（5s·12步，实测修正值） |
| 1056×608 | 0.64MP | **~180s** | 质量确认档（5s） |
| 1344×768 | 1.03MP | **~360s** | 成片档（5s·20步） |

> 🎬 示例成片：`examples/v1-pruned-product-animation/MiniMax_H3_00002_.mp4`
> ⚠️ 细节弱是剪枝固有损失，产品/卡通无碍；真人/写实请用 v2。

### v2 版（full，真人/写实向）— 2026-08-08 实测

| 配置 | 分辨率 | 大小 | 耗时（提交→出片） | 质量 | 结论 |
|---|---|---|---|---|---|
| 无加速（标准 attention） | 640×640 | 0.41MP | **391s**（预热后） | ✅ 全清晰、文字正确 | 可用（基线） |
| **SageAttention（`--sage`）** | 640×640 | 0.41MP | **300s** | ✅ 全清晰、文字正确 | ✅ **推荐生产配置** |
| Sage + TE-Speed | 640×640 | 0.41MP | 195s | ❌ 文字乱码/结构失真 | ❌ **TE 不适配当前配置**（8步 turbo+INT8 下无论参数对错均失真） |

- 🎬 示例成片：`examples/v2-full-real-person/SageOnly_full_8step_1x1_sage_00001-audio.mp4`（Sage 300s 配置出片，质检通过）
- **冷启动（首次）**：约 490-818s（Sage/Triton JIT 编译 + 34GB 模型载入），连续批量抽卡从第 2 条起才是上表速度
- **参考（同 16GB 档，非本机）**：5060Ti 8步 154s / 20步 首轮194s·次轮185s（手册§7）；4080S pruned+Sage ≈180s
- 运行时资源：显存 ~14GB / 系统内存 ~44GB（32GB 内存为最低门槛）

---

## 快速开始（TL;DR）

1. **ComfyUI**：官方包 v0.30.0+（`ComfyUI_windows_portable_nvidia.7z`，自带 torch 2.9.1+cu130）
2. **模型**（约 40GB+，ModelScope 国内直连）：**pruned INT8（20.9G）+ 完整版 INT8（34G）双份** + Qwen3-VL NVFP4 编码器 + 双 VAE + **turbo LoRA**（`minimax_h3_turbo_4STEPS_comfyui.safetensors`，**仅 v2 用**；v1 pruned 不适用）
3. **custom_nodes**：`comfyui-minimax-h3-audio-T8`（T8 节点，必需）+ KJNodes + UniBlockSwap + reservedvram + VHS
4. **加速**：**只用 SageAttention**（KJNodes `PatchSageAttentionKJ` 节点或 `--use-sage-attention`，二选一）；**不要开 TE-Speed**（实测失真）
5. **选版本出片**：
   - 真人/写实 → v2：`python scripts/h3_i2v_run.py --model full --steps 8 --size 1x1 --sage`
   - 产品/动画 → v1：`python scripts/h3_i2v_run.py --model pruned --steps 8 --size 16x9 --seed 7`
   - 或手动加载 `workflows/v1-.../`、`workflows/v2-.../` 对应工作流
6. **质检**：ffmpeg 抽帧 + 视觉模型逐帧检查（流程见 qa-benchmarks.md）

详细步骤见 [deployment-handbook.md](deployment-handbook.md)；测试与踩坑见 [qa-benchmarks.md](qa-benchmarks.md)。

---

## 🎬 案例视频（仅供参考）

> 以下为 RTX 4060 Ti 16GB 本机实测出片（640×640 · 5.17s · 带原生音轨）。GIF 为预览，点击可下载原始 MP4。

| v2 版（full 完整版 · SageAttention 300s 配置 · 真人/写实向） | v1 版（pruned 剪枝版 · 产品/动画向） |
|---|---|
| [![v2 真人案例](examples/v2-full-real-person/preview.gif)](examples/v2-full-real-person/SageOnly_full_8step_1x1_sage_00001-audio.mp4)<br>`SageOnly_full_8step_1x1_sage_00001-audio.mp4`（1.31MB） | [![v1 产品案例](examples/v1-pruned-product-animation/preview.gif)](examples/v1-pruned-product-animation/MiniMax_H3_00002_.mp4)<br>`MiniMax_H3_00002_.mp4`（0.88MB） |

- 完整质检记录见 [qa-benchmarks.md](qa-benchmarks.md)（§一 v1 / §二 v2 分版本测试）
- 更多出片：用 `scripts/h3_i2v_run.py` 按版本配置自行生成

---

## 许可证提示

MiniMax H3 开源协议**排除美/欧/英/韩地区使用**；其他地区可用，商用/出海前请自查协议原文。本仓库内容遵循 MIT。

---

## 致谢与参考

- 官方 ComfyUI 教程 / Comfy-Org/MiniMax-H3 模型仓库 / SageAttention / KJNodes
- T8star 的 `comfyui-minimax-h3-audio-T8` 节点（双时钟/MultiRate 采样器）与低配版整合包
- 社区加速指南 [juemin4-source/minimax-h3-guide](https://github.com/juemin4-source/minimax-h3-guide) 与 TE-Speed-MiniMaxH3-OSS

---

---

# English Version（英文版）

> English version of this README, placed as a standalone section below the Chinese content.

## 🎯 Core Capabilities

This project's value is concentrated in two things:

1. **🚀 3–5× accelerated workflow**: from the official template's "20–30 min per shot" down to **3–5 min/shot** (480p tier, warmed up). The core acceleration chain = **cu130 hardware dequantization (3–5×) + SageAttention (~1.5–2×) + T8 dual-clock turbo sampling (half the steps)**, measured at 3–5× combined speedup with no quality loss. Full acceleration decisions & benchmarks in [qa-benchmarks.md](qa-benchmarks.md).
2. **📦 Complete deployment (reproducible)**: 6-piece model set + optional 33 custom nodes + dual-version workflows + one-click script + restore guide — **any new machine can fully reproduce this production environment** from this repo (see [deployment-handbook.md](deployment-handbook.md) & the `h3-restore-guide.md` restore guide).

> Acceleration is not about stacking parameters — it's a **verified combination**: what to enable (SageAttention), what to drop (TE-Speed, distorted), what to tier (480p drafting / 768p final) — all backed by measurements.

## ⚠️ Two versions are distinct workflows; pick one before reading further

| | **v1 (product / animation)** | **v2 (real-person / photorealistic)** |
|---|---|---|
| Model | **pruned 20.9G** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`) | **full INT8 34G** (`minimax_h3_fl2va_int8_convrot.safetensors`) |
| Best for | **Product videos, cartoon/animation, fast drafting** | **Real-person videos, photorealistic ads, product text close-ups** |
| Speed | **~109s/shot** (480p measured; 1056×608 ~180s / 1344×768 ~360s) | **~300s/shot** (Sage, warmed up, 640×640) |
| Quality | Weaker details (pruning loss, cannot be recovered by resolution) | Accurate semantics, complete details |
| Workflow | `workflows/v1-pruned-product-animation/MiniMax-H3-双时钟采样8步-v1-pruned.json` | `workflows/v2-full-real-person/MiniMax-H3-双时钟采样8步.json` |
| Script | `h3_i2v_run.py --model pruned` | `h3_i2v_run.py --model full --sage` |
| Benchmarks | [qa-benchmarks.md §1](qa-benchmarks.md#一-v1-版pruned-产品动画向-测试) | [qa-benchmarks.md §2](qa-benchmarks.md#二-v2-版full-真人写实向-测试) |

> **Quality comes from the model, not pixels**: v1's "awkwardness" is an inherent loss from pruning — adding resolution won't fix it. For real-person / photorealistic / text close-ups, **use v2**. Both models coexist; switch with `--model`.

## Highlights

- ✅ **3–5× measured speedup**: cu130 hardware dequantization + SageAttention + T8 dual-clock turbo; 480p from official 20–30 min down to **3–5 min/shot** (warmed up), no quality loss
- ✅ **Runs stably on 16GB VRAM** (INT8 quantization + ComfyUI dynamic VRAM offloading; ~14GB VRAM measured)
- ✅ **Acceleration verdict (2026-08-08, 4060Ti)**: **SageAttention is the only reliable acceleration** (v2: 300s/shot, no quality loss); TE-Speed is fast (165–195s) but **measured distorted — deprecated**. See [qa-benchmarks.md](qa-benchmarks.md)
- ✅ **"480p for production, 768p for final selection"**: drafting at 480p/640×640 (v1 ~109s / v2 ~300s) → final at 1344×768 (~600–900s)
- ✅ Official model combo + ModelScope direct download in China (resumable)
- ✅ Complete pitfall checklist (32-multiple resolution, license region limits, two TE-Speed traps, common errors)

## Measured Performance (RTX 4060 Ti 16GB + 64GB RAM, 2026-08-08)

> Test conditions: T8 dual-clock 8-step sampler + 5s video. Full per-version matrix in [qa-benchmarks.md](qa-benchmarks.md).
> ⚠️ **turbo LoRA attribution**: **only used in v2 benchmarks** (the `h3_i2v_run.py --model full` chain includes a turbo LoRA node); **v1 has no turbo** (turbo LoRA wasn't released when fc9369f was measured, and pruned INT8 cannot fully apply it) — neither workflow JSON has a LoRA node.

### v1 (pruned, product/animation) — data source: commit `fc9369f` (480p corrected)

| Resolution | Size | Time (submit→video) | Note |
|---|---|---|---|
| 864×480 | 0.42MP | **~109s** | Draft tier (5s·12 steps, measured correction) |
| 1056×608 | 0.64MP | **~180s** | Quality-check tier (5s) |
| 1344×768 | 1.03MP | **~360s** | Final tier (5s·20 steps) |

> 🎬 Example: `examples/v1-pruned-product-animation/MiniMax_H3_00002_.mp4`
> ⚠️ Weaker details are inherent to pruning — fine for products/cartoons; use v2 for real people/photorealism.

### v2 (full, real-person/photorealistic) — measured 2026-08-08

| Config | Resolution | Size | Time (submit→video) | Quality | Verdict |
|---|---|---|---|---|---|
| No acceleration (standard attention) | 640×640 | 0.41MP | **391s** (warmed) | ✅ Sharp, text correct | Usable (baseline) |
| **SageAttention (`--sage`)** | 640×640 | 0.41MP | **300s** | ✅ Sharp, text correct | ✅ **Recommended production config** |
| Sage + TE-Speed | 640×640 | 0.41MP | 195s | ❌ Garbled text / distorted structure | ❌ **TE not applicable here** (distorted under 8-step turbo + INT8 regardless of params) |

- 🎬 Example: `examples/v2-full-real-person/SageOnly_full_8step_1x1_sage_00001-audio.mp4` (Sage 300s config, QC passed)
- **Cold start (first run)**: ~490–818s (Sage/Triton JIT compile + 34GB model load); steady-state speed above applies from the 2nd draft onward
- **Reference (same 16GB tier, not this machine)**: 5060Ti 8-step 154s / 20-step 194s first·185s repeat (handbook §7); 4080S pruned+Sage ≈180s
- Runtime resources: VRAM ~14GB / system RAM ~44GB (32GB RAM is the minimum)

## Quick Start (TL;DR)

1. **ComfyUI**: official package v0.30.0+ (`ComfyUI_windows_portable_nvidia.7z`, ships torch 2.9.1+cu130)
2. **Models** (~40GB+, ModelScope direct link in China): **pruned INT8 (20.9G) + full INT8 (34G)** + Qwen3-VL NVFP4 text encoder + dual VAEs + **turbo LoRA** (`minimax_h3_turbo_4STEPS_comfyui.safetensors`, **v2 only**; not applicable to v1 pruned)
3. **custom_nodes**: `comfyui-minimax-h3-audio-T8` (T8 nodes, required) + KJNodes + UniBlockSwap + reservedvram + VHS
4. **Acceleration**: **SageAttention only** (KJNodes `PatchSageAttentionKJ` node or `--use-sage-attention`, pick one); **do NOT enable TE-Speed** (measured distorted)
5. **Pick version and generate**:
   - Real-person/photorealistic → v2: `python scripts/h3_i2v_run.py --model full --steps 8 --size 1x1 --sage`
   - Product/animation → v1: `python scripts/h3_i2v_run.py --model pruned --steps 8 --size 16x9 --seed 7`
   - Or load the matching workflow manually from `workflows/v1-.../` / `workflows/v2-.../`
6. **QC**: extract frames with ffmpeg + inspect frame-by-frame with a vision model (see qa-benchmarks.md)

Full steps: [deployment-handbook.md](deployment-handbook.md); tests & pitfalls: [qa-benchmarks.md](qa-benchmarks.md).

## Sample Videos (for reference)

> Rendered locally on RTX 4060 Ti 16GB (640×640 · 5.17s · with native audio). GIFs are previews; click to download the original MP4.

| v2 (full · SageAttention 300s config · real-person) | v1 (pruned · product/animation) |
|---|---|
| [![v2 real-person demo](examples/v2-full-real-person/preview.gif)](examples/v2-full-real-person/SageOnly_full_8step_1x1_sage_00001-audio.mp4)<br>`SageOnly_full_8step_1x1_sage_00001-audio.mp4` (1.31MB) | [![v1 product demo](examples/v1-pruned-product-animation/preview.gif)](examples/v1-pruned-product-animation/MiniMax_H3_00002_.mp4)<br>`MiniMax_H3_00002_.mp4` (0.88MB) |

- Full QC records: [qa-benchmarks.md](qa-benchmarks.md) (§1 v1 / §2 v2, split by version)
- More renders: use `scripts/h3_i2v_run.py` with your version's config

## License

MiniMax H3's open-source license **excludes use in the US/EU/UK/Korea**; usable elsewhere — verify the license text before commercial use / overseas distribution. This repo is MIT.

## Credits & References

- Official ComfyUI tutorials / Comfy-Org/MiniMax-H3 model repo / SageAttention / KJNodes
- T8star's `comfyui-minimax-h3-audio-T8` nodes (dual-clock / MultiRate samplers) & the low-spec integration pack
- Community acceleration guide [juemin4-source/minimax-h3-guide](https://github.com/juemin4-source/minimax-h3-guide) & TE-Speed-MiniMaxH3-OSS


---

## FILE: deployment-handbook.md (https://github.com/neng320/minimax-h3-local-deployment/blob/HEAD/deployment-handbook.md)

# MiniMax H3 本地部署操作手册（消费级显卡版）

> 面向 16GB 显存消费级显卡的 MiniMax H3 本地部署完整指南：模型组合、下载、ComfyUI 配置、全栈加速链（cu130 + SageAttention + TE-Speed）、实测性能与资源占用、避坑清单。
>
> **测试环境**：16GB 显存 NVIDIA 显卡 + 64GB 内存 + Windows 11（2026-08 实测）。显存/内存档位见各节标注，同档位硬件可直接复用结论。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 16GB 显存能跑吗？ | **能，且适合**。16GB 显存 + ≥32GB 内存 = 社区验证的"能稳定制作"档位 |
| 用哪套模型？ | **官方推荐组合**：FL2VA `pruned_int8_convrot` (19.5G) + Qwen3-VL `nvfp4_awq` 编码器 (14.6G) + 双 VAE (5.4G)，共约 **40GB** |
| 默认分辨率策略？ | **480p 批量试镜头（832×480）→ 入选镜头重跑 768p（1344×768）**。全程 768p 慢到不经济 |
| 生成速度预期？ | 480p·5s：约 **2 分钟**（全栈加速后）；768p·5s：约 **6 分钟** |
| 要不要换更大显存？ | **不需要**。24GB 只减少卸载，计算速度提升不足以抵消换卡成本 |
| 满血 8×H200 方案？ | 那是服务器路线（vLLM-Omni 分布式需 ~384GB 内存）。ComfyUI 剪枝 INT8 路线才是消费级正解 |
| 许可证 | 开源协议**排除美/欧/英/韩地区**；中国地区可用，商用/出海前自查协议原文 |

---

## 1. 分辨率策略（最重要认知）：480p 生产、768p 精选

- H3 原生高质量画布 = **1344×768**（短边 768，上限 768×1344，**32 倍数对齐**）
- 消费级日常生成 = **832×480 / 864×480**（约 0.4MP）
- 768p 像素量是 480p 的 **约 2.5 倍**（41.5 万 → 103.2 万），加上显存卸载，实际耗时放大到 **3-5 倍**

### 为什么实际都跑 480p
1. **视频要反复抽卡**：动作自然度、产品变形、镜头运动、口型、文字 Logo 漂移……768p 试 5 次 = 1 小时+；480p 试 5 次 ≈ 15 分钟
2. **480p 足以判断 90% 问题**：构图/动作/运镜/节奏/人物一致性/音画同步都能看出来
3. **只有包装文字、小字体、材质高光**才需要 768p 最终确认

### 分阶段工作流
| 阶段 | 分辨率 | 时长 | 步数 | 用途 |
|---|---|---|---|---|
| 构图预览 | 832×480 | 5s | 8-12 | 快速筛选提示词，每镜头 3-5 版 |
| 质量测试 | 1024×576 | 5s | 16-20 | 判断动作/产品一致性 |
| **最终成片** | **1344×768** | 5s | ~20 | 仅对入选镜头重跑 |

> 非产品特写镜头（人物/环境/氛围/快节奏短视频）可直接 480p 交付或后期放大；电商产品特写才需要 768p 源文件。

---

## 2. 模型组合（官方推荐）

**来源**：ComfyUI 官方文档 T2V/I2V 模板的 Model storage 结构 + ModelScope `Comfy-Org/MiniMax-H3` 仓库。

| 文件 | 大小 | 放哪 |
|---|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB | `models/diffusion_models/` |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.61 GB | `models/text_encoders/` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 GB | `models/vae/` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB | `models/vae/` |

**为什么编码器用 NVFP4 而非 INT8**：
- 官方模板默认就是 `qwen3vl_32b_minimax_h3_nvfp4_awq`（14.6G，比 INT8 的 25.3G 省 10GB）
- NVFP4 的"原生加速面向 Blackwell 新架构"指的是 **DiT 主模型**；编码器只做提示词编码、负载轻，模拟路径影响可忽略，且已有 8GB 显存卡实测跑通
- **DiT 主模型必须用 INT8 ConvRot**（INT8 是消费级共识；4bit 版会损失运动细节）

**R2V（全参考模式，可选）**：需要 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（19.5G），是另一套 DiT 权重。先跑通 FL2VA 的文生/图生，再按需补。

**磁盘空间**：只装 FL2VA 组合预留 **80-100GB**；FL2VA+Ref2VA 都装预留 **120-150GB**。

---

## 3. 模型下载（ModelScope 国内直连）

**仓库**：`https://modelscope.cn/models/Comfy-Org/MiniMax-H3`（ComfyUI 官方整理版）

**下载命令（modelscope CLI，支持断点续传）：**
```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local_dir <你的下载目录>/minimax_h3
```
> 下载后目录结构与 ComfyUI 的 `models/` 完全一致（`diffusion_models/`、`text_encoders/`、`vae/`），装好 ComfyUI 后整体拷入即可。

**备选精度**（非必需）：
| 精度 | DiT 大小 | 适用 |
|---|---|---|
| BF16 全精度 | 61.7 GB | 画质上限，需大内存+磁盘卸载，不推荐 |
| INT8 完整版 | 31.7 GB | 内存 ≥64G 时可替换 pruned 版（首像素更快） |
| **pruned INT8（推荐）** | **19.5 GB** | 官方默认，16GB 显存主流 |
| pruned FP8 | 19.5 GB | 效果接近 INT8，备选 |

---

## 4. ComfyUI 安装（v0.30.0+，硬性要求）

> **INT8 硬性依赖 torch cuda 130+**，必须用官方整合包，第三方包必翻车。

**官方包**（约 1.5GB，自带 Python + torch + 依赖，解压即用）：
```
https://github.com/Comfy-Org/ComfyUI/releases/download/v0.30.0/ComfyUI_windows_portable_nvidia.7z
```
国内网络慢/失败可用镜像前缀：`https://ghproxy.com/https://github.com/...` 或 `https://kgithub.com/...`

**步骤**：7-Zip 解压 → 双击 `run_nvidia_gpu.bat` → 浏览器 `http://127.0.0.1:8188` → 模型文件拷入 `ComfyUI/models/` 对应子目录。

> 升级提示：若使用第三方整合包（如秋叶包），需将 ComfyUI 核心升级到 v0.30.0+（git 方式：`git fetch --tags && git checkout v0.30.2`），其自带 Python/torch 环境满足 cu130 即可不动。

---

## 5. 工作流

### 5.1 官方模板（首推）
ComfyUI → 模板库（Template Library）→ Video → 选 MiniMax H3 工作流（T2V 文生 / I2V 图生+首尾帧 / R2V 全参考）。模板含 **Resolution Selector 节点**（宽高比 + Megapixels + **32 倍数对齐**），默认即"快速预览尺寸"，全质量把 Megapixels 拉到 ~1.0 → 1344×768。

### 5.2 官方提示词指南（重要）
- 先描述整个场景（地点/人物/发生什么），再拆成带时间的分镜
- 一个提示词块内描述：分镜 + 镜头运动 + 音频（对白/SFX/音乐）
- 官方指南：`huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`
- R2V 指南：`.../VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`

### 5.3 两个版本工作流（v1 产品/动画 vs v2 真人/写实，**不同工作流，分开使用**）

> ⚠️ **这是两个版本，不是两个选项**：模型、工作流、测试结果都分开。选版本看 [README](README.md) 顶部对比表。

**v1 版（pruned，产品/动画向）** → `workflows/v1-pruned-product-animation/MiniMax-H3-双时钟采样8步-v1-pruned.json`
- 模型：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`（pruned 20.9G）
- 定位：**产品视频、卡通/动画、快节奏抽卡**；~3-4 分钟/条；细节弱是剪枝固有损失
- 测试：qa-benchmarks.md §一

**v2 版（full，真人/写实向）** → `workflows/v2-full-real-person/MiniMax-H3-双时钟采样8步.json`
- 模型：`minimax_h3_fl2va_int8_convrot.safetensors`（完整版 INT8 34G）
- 定位：**真人视频、写实广告、产品文字特写**；Sage 预热后 5:00/条；语义准、细节完整
- 测试：qa-benchmarks.md §二（8-08 全部实测数据基于此）

**实验工作流（两版本通用）** → `workflows/experimental/MiniMax-H3-全能参考-4V10A.json`
- 79 节点，`MultiRateSamplerEXPT8` 多速率（视频 4 步 turbo + 音频 10 步），Fast Groups Bypasser 一键切 5s/10s/15s
- 极速预览/探索用；UNET 引用 full 模型，若要在 v1 跑需手动改 UNETLoader 指向 pruned

---

## 6. 全栈加速链（实测组合，2026-08-08 修正）

**⚠️ 重要修正（8-08 实测推翻早期结论）**：TE-Speed 在**短步数（4/8 步 turbo）+ INT8 量化**场景下实测**失真弃用**（错参数语义崩坏成"牙膏"、官方参数文字乱码/结构失真）——它只对**官方 20-30 步长视频**有效（README 参考 30 步 45%）。**当前推荐加速 = SageAttention 单独使用**，详见 [qa-benchmarks.md](qa-benchmarks.md)。

**速度来源分解**（社区实测，12GB 显存档）：原始 cu129 480s → +Sage+TE 261s (1.8×) → **+cu130 47-74s (6-8×)** —— **最大收益 = PyTorch cu130 升级**（激活 INT8 convrot 硬件反量化，量化开销从软件模拟变硬件直算）。本仓库 8-08 实测（4060Ti 完整版 34G）：无加速 6:31 → Sage-only 5:00 → Sage+TE 2:45-3:15（失真）。

| 手段 | 收益 | 建议 |
|---|---|---|
| **cu130（torch 2.9.1+cu130）** | 3-5×（最大头） | ✅ 必需，官方包自带 |
| **SageAttention** | ~1.5-2× | ✅ **唯一推荐**（KJNodes `PatchSageAttentionKJ` 节点或 `--use-sage-attention`，**二选一勿重复**） |
| **TE-Speed** | 40-50%（仅长视频） | ❌ **弃用**（8 步 turbo + INT8 实测失真；仅官方 20-30 步工作流可再评估） |
| EasyCache / HyperStep / Spectrum | 20-60% | ⚠️ 预览用，别作最终默认（可能肢体错乱/音频失真/遵从度下降） |

> 核心原则：预览抽卡可开激进缓存，**最终成片只保留 SageAttention**。

### 6.1 SageAttention 安装（官方方法）
1. 从 `github.com/woct0rdho/SageAttention/releases` 下载匹配 PyTorch/CUDA 版本的 Windows wheel → `pip install <wheel>`
2. 装 KJNodes：`github.com/kijai/ComfyUI-KJNodes`（ComfyUI Manager 里搜，或 clone 到 `custom_nodes/`）
3. 工作流加 **Patch Sage Attention KJ** 节点（`sage_attention=auto`），或 ComfyUI 启动加 `--use-sage-attention` 全局启用

> 控制台出现 "Input tensors must be in dtype of torch.float16..." 是**正常的**（部分层非 fp16/bf16 自动回退标准注意力），不影响出片。
>
> **版本澄清**：社区有"必须 1.0.6、2.2.0 崩溃"的说法，但 ComfyUI 0.30.2 与最新 KJNodes 均使用 2.x 的 `sageattn` API——**1.0.6 无此函数，装了 Sage 会直接失效**。应安装与 torch/CUDA 匹配的 2.x wheel（如 `2.2.0+cu130torch2.9.1`，2026-07 修复版）。

### 6.2 TE-Speed 安装（闭源免费，实测 -40~50%）
1. clone `HELPMEEADICE/TE-Speed-MiniMaxH3-OSS` 到 `ComfyUI/custom_nodes/`
2. ComfyUI 根目录运行一次 `python patch_model.py`（自动备份原文件为 `model.py.te_speed.bak`；回滚 `--revert`）
3. 重启 ComfyUI；工作流节点链：`UNETLoader → TESpeedMiniMaxH3 → BasicGuider / BasicScheduler → SamplerCustomAdvanced`
4. 推荐参数：`processing_control_value=0.12, processing_percent_1=0.1, processing_percent_2=0.9, mcs=2, device=auto`
5. 生效铁证：控制台输出 `TE-Speed-MiniMaxH3(OSS): acceleration xx.x%`

> 回滚双保险：`git reset --hard <基线commit>`（若 ComfyUI 是 git 仓库）+ `python patch_model.py --revert`。

---

## 7. 实测性能（RTX 4060 Ti 16GB + 64GB 内存，2026-08-08 实测）

> 测试条件：**完整版 INT8 DiT（34GB）** + turbo LoRA + T8 双时钟 8 步 + 5 秒视频（I2VA）。完整矩阵与质检见 [qa-benchmarks.md](qa-benchmarks.md)。

| 配置 | 分辨率 | 耗时（提交→出片） | 质量 |
|---|---|---|---|
| 无加速（标准 attention） | 640×640 (1:1, 0.41MP) | **6 分 31 秒**（预热后） | ✅ 全清晰、文字正确 |
| **SageAttention only（推荐）** | 640×640 | **5 分 00 秒** | ✅ 全清晰、文字正确 |
| Sage + TE-Speed（官方参数） | 640×640 | 3 分 15 秒 | ❌ 文字乱码/结构失真 |
| Sage + TE-Speed（错参数 1.0） | 640×640 | 2 分 45 秒 | ❌ 语义崩坏（TOOTHPASTE） |
| 无加速（冷启动） | 864×480 (16:9) | ~13 分 38 秒 | ✅（含首次编译+模型载入） |
| 同 seed 重跑 | 任意 | 15 秒 | ⚠️ 结果缓存命中（非真跑） |

- **首条慢是编译的锅**：Sage/Triton JIT 编译 + 34GB 模型载入 → 第一条 8-13 分钟；**连续批量抽卡从第 2 条起才是上表真实速度**
- **pruned 档**（v1 卡通/产品，20.9G）：社区 5060Ti 8 步约 154s 为参照，4060Ti 估算 3-4 分钟（快 ~1.5 分钟但画质/细节弱）
- **分辨率规律**：640×640 与 864×480 像素量相近（0.41MP）耗时几乎一致；**768p（1344×768）是 480p 的 2.5 倍像素、约 3-5 倍耗时（20-30 分钟量级）**，只用于最终精选
- **运行时资源占用**：显存 **~14GB** / 系统内存 **~44GB**（16GB 显存 + 64GB 内存安全；32GB 内存为最低门槛）
- **档位建议**：抽卡 640×640/864×480（Sage ~5 分钟）→ 质量确认 → 成片 **1344×768**（20-30 分钟，H3 原生画布）
- **采样参数**：双时钟 8 步 + shift 12/3（T8）；官方模板 `dpmpp_2m` + `beta` + 20 步为画质/音频安全线

---

## 8. 避坑清单

1. **必须官方 ComfyUI 包 v0.30.0+**（INT8 依赖 torch cuda 130+）；版本/目录结构不对直接报错
2. **分辨率必须是 32 倍数**：Resolution Selector 的 Multiple 保持 32；ImageScaleToTotalPixels 的 multiple_of 也设 32（曾因被改成 8/12 触发 `patchify_video` 崩溃：`shape invalid for input of size`）
3. **许可证地域限制**：H3 开源协议**明确排除美/欧/英/韩地区**；中国不受限，但商用/出海前自查协议原文
4. **内存 ≥32GB 是硬要求**（实测占用 ~44GB）：低显存配置下内存往往最先爆
5. **VAE Decode 100% 崩溃**：显存压力下常见；Decode 前加 **VRAM-Cleanup 节点**缓解
6. **量化≠提速**：INT8 只降磁盘/搬运成本，提速靠 cu130/SageAttention/步数/分辨率
7. **Sage 二选一**：`--use-sage-attention` 与 KJ Patch 节点勿同时用（重复 patch）
8. **NVFP4 只用于编码器**，DiT 主模型用 INT8 ConvRot；4bit DiT 损失运动细节
9. **pruned 19G vs 完整 31.7G**：pruned 省内存不省显存、首像素更慢；内存 ≥64G 可换完整 INT8
10. **中文语音不稳定**：提示词反复强调中文仍可能输出非中文/嘴瓢重复；多试几次或用音频参考
11. **先 480p 跑通再 768p**；报错优先丢给 AI agent 解决
12. **显卡规格常见标注错误**：部分第三方标注的"大显存魔改版"（如 48GB 版）为魔改/错误信息，购买与部署前以厂商官方规格为准

---

## 9. 只要视频不要音频

- 官方工作流默认**原生生成音轨**（含在 MP4 里），audio VAE 文件很小（0.56GB）保留即可
- 出片后处理：`ffmpeg -i in.mp4 -an out_silent.mp4` 抽掉音轨，再合入自己的音频

---

## 10. 备选路径

| 路径 | 说明 |
|---|---|
| RunningHub 在线工作流 | 直接搜"H3"就有，注册送额度 |
| MiniMax 官方 API | 唯一完整 2K 链路（含 H3-Regenerate-2K） |
| OpenI 启智社区免费算力（T4/910B/A100） | 先测再买，确认效果再本地投入 |
| 云 GPU（AutoDL 等） | 批量生产时按需 |

---

## 11. 参考链接

- 官方 ComfyUI 教程：`https://docs.comfy.org/tutorials/video/minimax/minimax-h3`
- 模型仓库（ModelScope）：`https://modelscope.cn/models/Comfy-Org/MiniMax-H3`
- 模型仓库（HuggingFace 上游）：`https://huggingface.co/Comfy-Org/MiniMax-H3`
- ComfyUI 官方包：`https://github.com/Comfy-Org/ComfyUI/releases/download/v0.30.0/ComfyUI_windows_portable_nvidia.7z`
- SageAttention：`https://github.com/woct0rdho/SageAttention`
- KJNodes：`https://github.com/kijai/ComfyUI-KJNodes`
- TE-Speed-MiniMaxH3-OSS：`https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS`
- 社区加速指南：`https://github.com/juemin4-source/minimax-h3-guide`
- 官方部署公众号：`https://mp.weixin.qq.com/s/CNXRcUIt9cKOanYMualVlg`
- 提示词指南：`https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`


---

## FILE: qa-benchmarks.md (https://github.com/neng320/minimax-h3-local-deployment/blob/HEAD/qa-benchmarks.md)

# MiniMax H3 本地视频生成 · 测试记录与避坑基准（QA Benchmarks）

> 本文件是 2026-08 在 RTX 4060 Ti 16GB + 64GB 上的**实测记录**，供 AI agent 部署/调参时**避坑**用。
> **⚠️ 两个版本是完全不同的工作流，测试结果分开记录，切勿混用**：
> - **v1 版（pruned 模型，产品/动画向）** → 见 [§一](#一-v1-版pruned-产品动画向-测试)（数据源：commit `fc9369f` 初始实测）
> - **v2 版（full 模型，真人/写实向）** → 见 [§二](#二-v2-版full-真人写实向-测试)
> 所有耗时均为「提交→出片」实测（秒为单位）。

---

## 一、v1 版（pruned，产品/动画向）测试

### 1.1 版本定位

| 项 | 值 |
|---|---|
| 模型 | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（**pruned 20.9G**） |
| 工作流 | `workflows/v1-pruned-product-animation/MiniMax-H3-双时钟采样8步-v1-pruned.json`（UNET 已指向 pruned） |
| 擅长 | **产品视频、卡通/动画、快节奏抽卡**（速度快，细节要求不高） |
| 脚本 | `h3_i2v_run.py --model pruned` |
| **turbo LoRA** | ❌ **无**（v1 不适用：fc9369f 实测时 turbo LoRA 未发布，且 pruned INT8 不能完整应用 turbo LoRA——T8 README 明确） |

### 1.2 测试矩阵（v1，数据源 = commit `fc9369f` 初始实测记录 + 480p 实测修正，秒为单位）

| # | 分辨率 | 大小(MP) | 参数 | 耗时（提交→出片） | 说明 |
|---|--------|----------|------|------------------|------|
| V1-1 | 864×480 | 0.42 | 5s·12 步 | **~109s** | 抽卡档（实测修正值） |
| V1-2 | 1056×608 | 0.64 | 5s | **~180s** | 质量确认档 |
| V1-3 | 1344×768 | 1.03 | 5s·20 步 | **~360s** | 成片档 |

> 📌 **数据来源**：480p（864×480）实测修正为 **~109s**；1056×608 / 1344×768 为仓库 commit `fc9369f`（2026-08-06 初始 README）记录的 V1 实测（当时配置 = pruned INT8 + cu130 + SageAttention + TE-Speed 全栈加速链）。
>
> 🎬 **示例成片**：`examples/v1-pruned-product-animation/MiniMax_H3_00002_.mp4`（640×640·5.17s·0.88MB，pruned 档出片）
>
> ⚠️ **质量提醒**：v1 的"别扭感"（肢体比例/动作连贯性/物体逻辑）是**剪枝丢层的固有损失**，不是参数问题——**产品/卡通这类细节要求不高的场景无碍**，但真人/写实/文字特写请用 v2。
>
> ⚠️ **环境反例**：未正确装 SageAttention/版本不对时，448×672·5s 曾跑 **30min（差 11 倍）**——确认 Sage 生效是提速前提。

---

## 二、v2 版（full，真人/写实向）测试

### 2.1 版本定位

| 项 | 值 |
|---|---|
| 模型 | `minimax_h3_fl2va_int8_convrot.safetensors`（**完整版 INT8 34G**） |
| 工作流 | `workflows/v2-full-real-person/MiniMax-H3-双时钟采样8步.json`（UNET=full） |
| 擅长 | **真人视频、写实广告、产品文字特写**（语义准、细节完整） |
| 脚本 | `h3_i2v_run.py --model full` |
| **turbo LoRA** | ✅ **有（仅实测脚本链）**：`h3_i2v_run.py` 含 `LoraLoaderModelOnly(minimax_h3_turbo_4STEPS)` 节点；⚠️ **json 工作流本身无 LoRA 节点**，若手动加载 json 需自行加 LoRA 才能复现实测结果 |

### 2.2 测试矩阵（v2，2026-08-08 实测，秒为单位）

| # | 加速链 | 分辨率 | 大小(MP) | 参数 | 耗时（提交→出片） | 质量 | 结论 |
|---|--------|--------|----------|------|------------------|------|------|
| V2-1 | 无（标准 attention） | 640×640 | 0.41 | 8步·5s·seed42 | **391s** | ✅ 全清晰、产品文字"去烟味漱口水"正确 | 可用（基线） |
| V2-2 | 无（冷启动） | 864×480 | 0.42 | 8步·5s·seed42 | **818s** | ✅ 可用 | 冷启动（含首次编译） |
| V2-3 | **SageAttention** | 640×640 | 0.41 | 8步·5s·seed43 | **300s** | ✅ 全清晰、中文漱口水正确 | ✅ **推荐生产配置** |
| V2-4 | Sage + TE-Speed | 640×640 | 0.41 | 8步·5s·seed43 | 195s | ❌ 文字乱码伪中文、手部/嘴部结构失真 | ❌ **TE 不适配当前配置** |

> 📌 **参考数据（同 16GB 档，非本机）**：5060Ti 16GB `864×480·5.17s` 8 步 **154s** / 20 步首轮 **194s**·次轮 **185s**（部署手册 §7）；4080S pruned+Sage 约 180s；未正确装 Sage 时同规格 30min（差 11 倍）。
>
> 🎬 **示例成片**：`examples/v2-full-real-person/SageOnly_full_8step_1x1_sage_00001-audio.mp4`（640×640·5.17s·1.31MB·Sage 300s 配置出片，质检通过）
>
> ⚠️ **TE-Speed 明确不适配**：v2 在 **8 步 turbo + INT8 量化**场景下，TE-Speed 无论参数对错都失真（错参数 control=1.0 时语义崩坏成"TOOTHPASTE 牙膏"，官方参数 0.12 时文字乱码/结构失真）——它只对**官方 20-30 步长视频**有效（README 参考 30 步 45%）。**生产只用 `--sage`，勿开 TE-Speed**。

### 2.3 v2 结论

- **生产配置：SageAttention only（`--sage`）= 300s/条，画质与无加速同级**
- 首条慢是 Sage/Triton JIT 编译 + 34GB 模型载入（约 490-818s），**连续抽卡从第 2 条起才是 300s**
- 同 seed 同参数重跑会被 ComfyUI 结果缓存命中（秒级完成，非真跑）——抽多版请换 `--seed`

---

## 三、踩坑实录（两版本通用，务必先读）

### 坑 1：TE-Speed 参数错 = 画面语义崩坏（最严重）
- **表现**：产品文字变成 "TOOTHPASTE"（牙膏）、出现牙刷、第 2 镜头虚化失真、液体偏色"喂药感"
- **根因**：`processing_control_value` 是"相邻 sigma 差小于该值就走缓存步"的阈值。**官方默认 0.12，误设 1.0 意味着几乎每一步都疯狂跳缓存**，编码语义直接被毁
- **正确值**：`0.12 / processing_percent_1: 0.1 / processing_percent_2: 0.9 / mcs: 2`

### 坑 2：TE-Speed 在 8 步 turbo + INT8 场景下即使官方参数仍失真
- **表现**：产品文字变乱码伪中文、手部/嘴部结构不自然，不可用
- **根因**：TE-Speed 是为**官方 20-30 步长视频**设计的缓存加速（README 参考 30 步 8s 得 45%）。8 步太短（前后 10% 强制完整计算后缓存窗口被挤压）+ INT8 量化 + 残差校正 → 数值/语义崩坏
- **结论**：两个版本当前都不要开 TE-Speed；若未来用官方 20-30 步工作流再评估

### 坑 3：SageAttention 需要手动启动的 ComfyUI
- **表现**：报 `OSError: [safe-delete][SAFE_DELETE_FAIL_CLOSED] ... tmp.pid_xxx`，任务失败不出结果
- **根因**：Triton 编译时创建/删除 `tmp.pid_*` 锁文件，部分进程管理环境（沙箱/托管启动）会拦截该模式的 unlink
- **解决**：**手动启动 ComfyUI**（终端/双击启动器，非自动化托管方式）

### 坑 4：同 seed 同参数重跑秒完成
- **表现**：第 2 次同参数 15 秒"完成"
- **根因**：ComfyUI 结果缓存命中
- **解决**：抽多版换 `--seed`；或删 ComfyUI 缓存

### 坑 5：下载文件名带 hash 后缀
- **表现**：工作流加载报"找不到模型"
- **根因**：HF/ModelScope 下载文件名带 `-<hash>` 后缀，ComfyUI 按完整文件名匹配
- **解决**：重命名去掉 `-<hash>`（如 `minimax_h3_fl2va_int8_convrot-8cb663a6125c.safetensors` → `...int8_convrot.safetensors`）

### 坑 6：首次生成特别慢
- **表现**：第一条 8-9 分钟，第二条起 ~5 分钟
- **根因**：Sage/Triton JIT 编译预热
- **解决**：连续批量抽卡从第 2 条起才是真实速度

### 坑 7：分辨率策略
- 640×640(1:1) 与 864×480(16:9) 像素量相近（0.41MP），耗时几乎一致；**768p(1344×768) 是 480p 的 2.5 倍像素、3-5 倍耗时**，只用于最终精选

---

## 四、已验证可用配置（直接照抄）

### v1（产品/动画向，速度优先）
```bash
python scripts/h3_i2v_run.py --model pruned --steps 8 --size 16x9 --seed 7 --prefix draft
# 预期 ~120-180s/条（fc9369f 实测 480p 抽卡 ~120s），适合批量抽卡与产品/卡通出片
```

### v2（真人/写实向，质量优先）
```bash
python scripts/h3_i2v_run.py --model full --steps 8 --size 1x1 --sage --seed 42
# 预期 300s/条（预热后），真人广告/产品文字特写用
```

> ⚠️ `--sage` 需要**手动启动的 ComfyUI**（见坑 3）。


---

# ISSUES / PRs of neng320/minimax-h3-local-deployment
