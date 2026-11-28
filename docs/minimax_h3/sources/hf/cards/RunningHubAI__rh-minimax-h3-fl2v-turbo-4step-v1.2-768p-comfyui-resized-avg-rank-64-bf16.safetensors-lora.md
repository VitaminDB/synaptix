# RunningHubAI/rh-minimax-h3-fl2v-turbo-4step-v1.2-768p-comfyui-resized-avg-rank-64-bf16.safetensors-lora

Источник: https://huggingface.co/RunningHubAI/rh-minimax-h3-fl2v-turbo-4step-v1.2-768p-comfyui-resized-avg-rank-64-bf16.safetensors-lora · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая:  · теги: comfyui,lora · обновлено: 2026-09-24T00:52:02.000Z

---

---
tags:
  - comfyui
  - lora
---

# rh-minimax-h3-fl2v-turbo-4step-v1.2-768p-comfyui-resized-avg-rank-64-bf16.safetensors-lora

[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-555)](README_cn.md)
[![RunningHub API](https://img.shields.io/badge/RunningHub-API-10B981)](https://www.runninghub.ai/call-api?utm_source=huggingface&utm_medium=badge&utm_campaign=api_promotion&utm_content=rh-2102780347502125058)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai?utm_source=huggingface&utm_medium=badge&utm_campaign=model_upload&utm_content=rh-2102780347502125058)
[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn?utm_source=huggingface&utm_medium=badge&utm_campaign=model_upload&utm_content=rh-2102780347502125058)

## Description

| Field | Content |
|---|---|
| Model Name | rh-minimax-h3-fl2v-turbo-4step-v1.2-768p-comfyui-resized-avg-rank-64-bf16.safetensors-lora |
| Model Type | LoRA |
| Platforms | ComfyUI / RunningHub / Hugging Face |
| Author | RunningHub-[@SVIP专属](https://www.runninghub.cn/user-center/1929731488613310465) |
| Original | https://www.runninghub.cn/model/public/2102780347502125058 |
| Try online | [https://www.runninghub.ai](https://www.runninghub.ai?utm_source=huggingface&utm_medium=basic_info&utm_campaign=model_upload&utm_content=rh-2102780347502125058) |
|  | [https://www.runninghub.cn/runninghub-api-doc-en/](https://www.runninghub.cn/runninghub-api-doc-en/?utm_source=huggingface&utm_medium=basic_info&utm_campaign=api_doc&utm_content=rh-2102780347502125058) |

## About this model

<p><strong>LightX2V/ModelTC团队推出的MiniMax H3 FL2VA专属4步极速蒸馏LoRA v1.2版本，专为ComfyUI原生适配的768p BF16精度模型</strong>‌，是目前社区FL2V方向速度最快的商用级衍生版本之一。</p><ul><li>‌<strong>核心定位</strong>‌：基于MiniMax官方开源的H3-Base-FL2VA基础模型做CFG步数蒸馏得到的轻量LoRA适配器，专门面向首尾帧生视频、文生音视频、首帧/尾帧生视频三类场景，无需依赖官方未开源的模块，完全支持本地ComfyUI部署。</li><li>‌<strong>硬件与性能表现</strong>‌：原生训练分辨率为1344×768 16:9标准画幅，BF16全精度格式，叠加rank_64平均权重裁剪优化后，在RTX 5090上搭配SLA稀疏注意力可实现约2.5倍推理加速，生成15秒768p音视频仅需数分钟，远快于原版30-50步的基础模型。</li><li>‌<strong>输出规格</strong>‌：严格对齐H3官方原生能力，输出24fps视频+32kHz同步立体声音频，支持11种主流语言指令，兼容21:9、9:16等多种常用画幅，最长可生成15秒完整音视频片段。</li><li>‌<strong>适配特性</strong>‌：文件名中的<code>resized_avg_rank_64</code>代表该版本完成了参考图尺寸对齐与低秩权重平均优化，直接放入ComfyUI的<code>models/loras</code>目录即可通过标准LoRA加载器运行，无需额外格式转换。</li></ul>

This repository provides **rh-minimax-h3-fl2v-turbo-4step-v1.2-768p-comfyui-resized-avg-rank-64-bf16.safetensors-lora** weight files. You can load them on RunningHub.

Finetuned from: minimax-h3

## Files

| File | Size | Purpose |
|---|---:|---|
| `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_resized_avg_rank_64_bf16.safetensors` | 929 MiB | LoRA weights |

## License

Published by RunningHub on behalf of the author. Copyright remains with the author. Follow the original project or upstream license.

## Training at RunningHub

Train models on RunningHub: https://www.runninghub.ai/page-model

Run Seedance 2.5 with RunningHub API: [https://www.runninghub.ai/call-api/api-detail/2133100000000700025](https://www.runninghub.ai/call-api/api-detail/2133100000000700025?utm_source=huggingface&utm_medium=related_links&utm_campaign=api_promotion&utm_content=rh-2102780347502125058)

- RunningHub：https://www.runninghub.ai
- RunningHub 中国站：https://www.runninghub.cn
- RunningHub API 文档：https://www.runninghub.cn/runninghub-api-doc-cn/
- RunningHub API Documentation：https://www.runninghub.cn/runninghub-api-doc-en/
- Original project: https://www.runninghub.cn/model/public/2102780347502125058
- Author page: https://www.runninghub.cn/user-center/1929731488613310465
