# RunningHubAI/rh-minimax-h3-turboref2v-turbo-8step-v1.0-768p-lora

Источник: https://huggingface.co/RunningHubAI/rh-minimax-h3-turboref2v-turbo-8step-v1.0-768p-lora · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая:  · теги: comfyui,lora · обновлено: 2026-10-03T15:42:02.000Z

---

---
tags:
  - comfyui
  - lora
---

# rh-minimax-h3-turboref2v-turbo-8step-v1.0-768p-lora

[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-555)](README_cn.md)
[![RunningHub API](https://img.shields.io/badge/RunningHub-API-10B981)](https://www.runninghub.ai/call-api?utm_source=huggingface&utm_medium=badge&utm_campaign=api_promotion&utm_content=rh-2102778612423421953)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai?utm_source=huggingface&utm_medium=badge&utm_campaign=model_upload&utm_content=rh-2102778612423421953)
[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn?utm_source=huggingface&utm_medium=badge&utm_campaign=model_upload&utm_content=rh-2102778612423421953)

## Description

| Field | Content |
|---|---|
| Model Name | rh-minimax-h3-turboref2v-turbo-8step-v1.0-768p-lora |
| Model Type | LoRA |
| Platforms | ComfyUI / RunningHub / Hugging Face |
| Author | RunningHub-[@SVIP专属](https://www.runninghub.cn/user-center/1929731488613310465) |
| Original | https://www.runninghub.cn/model/public/2102778612423421953 |
| Try online | [https://www.runninghub.ai](https://www.runninghub.ai?utm_source=huggingface&utm_medium=basic_info&utm_campaign=model_upload&utm_content=rh-2102778612423421953) |
|  | [https://www.runninghub.cn/runninghub-api-doc-en/](https://www.runninghub.cn/runninghub-api-doc-en/?utm_source=huggingface&utm_medium=basic_info&utm_campaign=api_doc&utm_content=rh-2102778612423421953) |

## About this model

<p><strong>MiniMax H3系列专为ComfyUI打造的8步加速版参考生视频专用模型</strong>‌，核心作用是在本地快速生成768P分辨率、带原生立体声的参考驱动视频。</p><h3>核心功能定位</h3><ul><li>‌<strong>专属生成模式</strong>‌：属于Ref2VA（Reference-to-Video）分支，专门用于参考图/参考视频驱动的视频生成任务，能严格锁定角色形象、画面风格和动作逻辑，保证多镜头内容的一致性。</li><li>‌<strong>极速采样优化</strong>‌：内置Turbo 8步加速方案，仅需8步采样即可完成原本几十步的视频生成，大幅缩短推理耗时，实测在支持BF16的显卡上可实现快速出片。</li><li>‌<strong>原生规格适配</strong>‌：原生输出768P短边分辨率的视频，帧率24fps，同时自动生成32kHz立体声音频，无需额外搭配音频生成模型，直接输出完整音视频MP4文件。</li></ul><h3>格式与适配特性</h3><ul><li>‌<strong>BF16精度</strong>‌：采用Brain Float 16数据格式，在几乎不损失画面细节的前提下，显存占用相比全精度FP32减少约50%，同时在支持BF16硬件加速的显卡上运算效率更高。</li><li>‌<strong>ComfyUI原生适配</strong>‌：权重已完成重打包适配，直接放入ComfyUI的<code>diffusion_models</code>文件夹，配合官方H3工作流即可加载使用，无需额外格式转换。</li><li>‌<strong>低秩优化</strong>‌：内置rank_64的平均权重裁剪，在保留核心生成能力的同时进一步压缩模型体积，降低普通消费级显卡的运行门槛。</li></ul>

This repository provides **rh-minimax-h3-turboref2v-turbo-8step-v1.0-768p-lora** weight files. You can load them on RunningHub.

Finetuned from: minimax-h3

## Files

| File | Size | Purpose |
|---|---:|---|
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_resized_avg_rank_64_bf16.safetensors` | 933 MiB | LoRA weights |

## License

Published by RunningHub on behalf of the author. Copyright remains with the author. Follow the original project or upstream license.

## Training at RunningHub

Train models on RunningHub: https://www.runninghub.ai/page-model

Run Seedance 2.5 with RunningHub API: [https://www.runninghub.ai/call-api/api-detail/2133100000000700025](https://www.runninghub.ai/call-api/api-detail/2133100000000700025?utm_source=huggingface&utm_medium=related_links&utm_campaign=api_promotion&utm_content=rh-2102778612423421953)

- RunningHub：https://www.runninghub.ai
- RunningHub 中国站：https://www.runninghub.cn
- RunningHub API 文档：https://www.runninghub.cn/runninghub-api-doc-cn/
- RunningHub API Documentation：https://www.runninghub.cn/runninghub-api-doc-en/
- Original project: https://www.runninghub.cn/model/public/2102778612423421953
- Author page: https://www.runninghub.cn/user-center/1929731488613310465
