# slmonker/minimax-h3-fl2v-turbo-comfyui-fp8

Источник: https://huggingface.co/slmonker/minimax-h3-fl2v-turbo-comfyui-fp8 · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая:  · теги:  · обновлено: 2026-09-11T04:22:32.000Z

---

# H3 Diffusers FP8 → ComfyUI FP8 转换器

这是针对本机这份 `minimax_h3_fl2v_turbo_4step_v1.1_768p_fp8.safetensors` 编写的适配脚本，**不是上游官方发布的转换工具，也不是通用 safetensors 转换器**。

## 文件

- `convert_h3_fp8.py`：流式转换，默认只使用 CPU。
- `test_conversion.py`：当前模型的结构、上游反向映射、FP8 CPU/CUDA 小矩阵测试。
- `validate_native.py`：在独立进程中使用原生 ComfyUI 加载器加载完整模型，可运行一个极小随机 latent 前向。
- `official_convert_minimax_h3_to_diffusers.py`：固定 Git blob 的上游代码，转换器只提取其静态配置和权重计划，不运行上游 CLI。
- `official_transformer_minimax_h3.py`、`lightx2v_mm_weight.py`：审查依据，不参与运行。

## 转换规则与边界

1. 基于 Diffusers 官方 H3 转换计划反向映射键名。
2. Q、K、V 沿输出行拼接为 `[q_all; k_all; v_all]`，不是原始训练分片的逐头交错格式。
3. FFN 第一层从 `[value; gate]` 交换成 `[gate; value]`；行缩放先用于对应行，再交换。
4. 补回标准 H3 的 `rope.inv_freq`，theta=10000、freq_dim=16。本机 BF16 H3 的该张量与生成值逐位相同。
5. 源 FP8 权重先按行缩放在 float32 中反量化，再按目标完整矩阵的 `max(abs(W))/448` 重新量化成单标量缩放的 E4M3FN。
6. FP32/BF16 非量化张量在输出文件里保留原 dtype 和数值（必要的布局变换除外）。
7. 写入原生 `_quantization_metadata`，让 ComfyUI 识别混合精度线性层。默认 `full_precision_matrix_mult=true`：**FP8 权重存储，BF16 运算**，不代表原生 FP8 GEMM 加速。
8. 不修改 ComfyUI、不修改 Python 环境、不启动或向正在运行的服务器提交任务、不覆盖源文件。

**不是无损转换**：逐行缩放改为逐张量缩放会增加量化误差。每个 FP8 目标矩阵扫描所有元素并统计新增相对 L2/RMSE，默认超过 4% 立即失败。该阈值是工程检查阈值，不是视频质量保证。误差相对于源 FP8 反量化值，不是相对于原始 BF16 训练权重。

源文件没有配置元数据，所以非权重配置（RoPE、norm eps）按固定上游标准 H3 配置处理。脚本严格核对完整 50 层 H3、2 层 token refiner 的键和尺寸；有未知键、缺失键、非正/非有限缩放、非有限权重会拒绝继续。不能据此断言所有未来 H3 Turbo 变体均适用。

## 本机运行方式（PowerShell）

```powershell
$python = 'O:\ComfyUI_windows_portable\ComfyUI_windows_portable\python_embeded\python.exe'
$script = 'conversion-tools