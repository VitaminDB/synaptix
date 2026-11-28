# RunningHubAI/rh-minimax-h3-turbo-4steps-lora

Источник: https://huggingface.co/RunningHubAI/rh-minimax-h3-turbo-4steps-lora · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая:  · теги: comfyui,lora · обновлено: 2026-10-15T06:42:05.000Z

---

---
tags:
  - comfyui
  - lora
---

# rh-minimax-h3-turbo-4steps-lora

[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-555)](README_cn.md)
[![RunningHub API](https://img.shields.io/badge/RunningHub-API-10B981)](https://www.runninghub.ai/call-api?utm_source=huggingface&utm_medium=badge&utm_campaign=api_promotion&utm_content=rh-2092791011117453313)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai?utm_source=huggingface&utm_medium=badge&utm_campaign=model_upload&utm_content=rh-2092791011117453313)
[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn?utm_source=huggingface&utm_medium=badge&utm_campaign=model_upload&utm_content=rh-2092791011117453313)

## Description

| Field | Content |
|---|---|
| Model Name | rh-minimax-h3-turbo-4steps-lora |
| Model Type | LoRA |
| Platforms | ComfyUI / RunningHub / Hugging Face |
| Author | RunningHub-[@SVIP专属](https://www.runninghub.cn/user-center/1929731488613310465) |
| Original | https://www.runninghub.cn/model/public/2092791011117453313 |
| Try online | [https://www.runninghub.ai](https://www.runninghub.ai?utm_source=huggingface&utm_medium=basic_info&utm_campaign=model_upload&utm_content=rh-2092791011117453313) |
|  | [https://www.runninghub.cn/runninghub-api-doc-en/](https://www.runninghub.cn/runninghub-api-doc-en/?utm_source=huggingface&utm_medium=basic_info&utm_campaign=api_doc&utm_content=rh-2092791011117453313) |

## About this model

<p>核心优势<br/>‌显存占用更低‌：相比未剪枝的原版Turbo LoRA，内存占用降低约40%，8GB显存的消费级显卡即可流畅运行。<br/>‌生成效率更高‌：在保持基础画质的前提下，将采样步数压缩至4步，生成速度相比原版20步模型提升3倍以上。<br/>‌音画同步支持‌：支持直接生成32kHz高保真立体声音频，无需后期额外合成音频，避免音画错位问题。<br/>部署使用要点<br/>‌放置目录‌：需将该文件放入ComfyUI的 models/loras/ 目录下，重启ComfyUI后即可在LoRA加载节点中识别选中。<br/>‌配套依赖‌：需搭配MiniMax H3 pruned剪枝版基础扩散模型、对应视频VAE、音频VAE共同使用。<br/>‌参数推荐‌：LoRA强度建议设置在0.8-1.2区间，采样器选择res_multistep，采样步数固定为4即可发挥最佳加速效果。</p>

This repository provides **rh-minimax-h3-turbo-4steps-lora** weight files. You can load them on RunningHub.

Finetuned from: minimax-h3

## Files

| File | Size | Purpose |
|---|---:|---|
| `minimax_h3_turbo_4step_pruned_comfyui.safetensors` | 592 MiB | LoRA weights |

## License

Published by RunningHub on behalf of the author. Copyright remains with the author. Follow the original project or upstream license.

## Training at RunningHub

Train models on RunningHub: https://www.runninghub.ai/page-model

Run Seedance 2.5 with RunningHub API: [https://www.runninghub.ai/call-api/api-detail/2133100000000700025](https://www.runninghub.ai/call-api/api-detail/2133100000000700025?utm_source=huggingface&utm_medium=related_links&utm_campaign=api_promotion&utm_content=rh-2092791011117453313)

- RunningHub：https://www.runninghub.ai
- RunningHub 中国站：https://www.runninghub.cn
- RunningHub API 文档：https://www.runninghub.cn/runninghub-api-doc-cn/
- RunningHub API Documentation：https://www.runninghub.cn/runninghub-api-doc-en/
- Original project: https://www.runninghub.cn/model/public/2092791011117453313
- Author page: https://www.runninghub.cn/user-center/1929731488613310465
