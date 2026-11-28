# alibaba-pai/MiniMax-H3-Acc-LoRAs

Источник: https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs · скачано 2026-11-28
Загрузок: 46784 · лайков: 280 · базовая: ["alibaba-pai/MiniMax-H3-Acc-LoRAs","MiniMaxAI/MiniMax-H3"] · теги:  · обновлено: 2026-08-27T06:13:51.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- alibaba-pai/MiniMax-H3-Acc-LoRAs
- MiniMaxAI/MiniMax-H3
library_name: videox_fun
pipeline_tag: text-to-video
---
# MiniMax-H3-Acc-LoRAs
## Introduction
We apply Parallel Decoding Distillation (PDD) <sup>[1](#ref1)</sup> to [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), enabling efficient video generation in only a few inference steps.

For more details, please refer to our [GitHub repo](https://github.com/aigc-apps/VideoX-Fun).

| Name | Base Model | Hugging Face | Description |
|--|--|--|--|
| MiniMax-H3-FL2VA-Acc-8Step.safetensors | [MiniMax-H3 (FL2VA)](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA) | [🤗Link](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/MiniMax-H3-FL2VA-Acc-8Step.safetensors) | Official 8 Step Acc LoRA (`rank=64` and `network_alpha=64` in BF16) for MiniMax-H3 (FL2VA).|
| MiniMax-H3-Ref2VA-Acc-8Step.safetensors | [MiniMax-H3 (Ref2VA)](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/Ref2VA) | [🤗Link](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/MiniMax-H3-Ref2VA-Acc-8Step.safetensors) | Official 8 Step Acc LoRA (`rank=64` and `network_alpha=64` in BF16) for MiniMax-H3 (Ref2VA).|

## Demo

### FL2VA (768p)

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="33%">MiniMax-H3-FL2VA</th>
            <th style="text-align: center;" width="33%"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors">Minimax-h3-Turbo <br> (fl2v_turbo_4step_v1.1_768p)</a></th>
            <th style="text-align: center;" width="33%">MiniMax-H3-FL2VA-Acc-8Step </th>
        </tr>
    </thead>
    <tr>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_baseline_1.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_turbo_4step_v1.1_768p_1.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_acc_8step_v1_1.mp4" width="100%" controls loop></video>
        </td>
    </tr>
    <tr>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_baseline_2.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_turbo_4step_v1.1_768p_2.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_acc_8step_v1_2.mp4" width="100%" controls loop></video>
        </td>
    </tr>
    <tr>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_baseline_3.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_turbo_4step_v1.1_768p_3.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_fl2va_acc_8step_v1_3.mp4" width="100%" controls loop></video>
        </td>
    </tr>
</table>

### Ref2VA

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="33%">MiniMax-H3-Ref2VA</th>
            <th style="text-align: center;" width="33%"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors">Minimax-h3-Turbo <br> (ref2v_turbo_4step_v0.1)</a></th>
            <th style="text-align: center;" width="33%">MiniMax-H3-Ref2VA-Acc-8Step</th>
        </tr>
    </thead>
    <tr>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_baseline_1.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_turbo_4step_v0.1_768p_1.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_acc_8step_v1_1.mp4" width="100%" controls loop></video>
        </td>
    </tr>
    <tr>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_baseline_2.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_turbo_4step_v0.1_768p_2.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_acc_8step_v1_2.mp4" width="100%" controls loop></video>
        </td>
    </tr>
    <tr>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_baseline_3.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_turbo_4step_v0.1_768p_3.mp4" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/results/minimax_h3_ref2va_acc_8step_v1_3.mp4" width="100%" controls loop></video>
        </td>
    </tr>
</table>


> [!NOTE]
> The above test cases are from <a href="https://github.com/ModelTC/Minimax-H3-Turbo/tree/main/examples">Minimax-H3-Turbo</a>.
> Videos are generated with a LoRA weight of 1.0 at both 4 and 8 NFE.

## Quick Start
Set `model_path` and `pdd_lora_path` to the MiniMax-H3 model and the matching acceleration LoRA checkpoint in [predict_t2v.py](https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/MiniMax-H3-Acc-LoRAs/predict_t2v.py) for FL2VA or [predict_ref2v.py](https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/MiniMax-H3-Acc-LoRAs/predict_ref2v.py) for Ref2VA, then run the corresponding script. Each example uses `apply_pdd_lora` to load the checkpoint and derive the required number of inference steps from its configuration.

> [!NOTE]
> These scripts use Diffusers' MiniMax-H3 `ModularPipeline` and require `diffusers >= 0.40.0`.

## Reference
<ol>
  <li id="ref1">Neta Shaul, et al. "Parallel Decoding Distillation for Fast Image and Video Generation.". arXiv preprint arXiv:2607.26004 (2026).</li>
</ol>
