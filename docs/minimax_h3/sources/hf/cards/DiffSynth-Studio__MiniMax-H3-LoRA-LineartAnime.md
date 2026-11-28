# DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime

Источник: https://huggingface.co/DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime · скачано 2026-11-28
Загрузок: 0 · лайков: 20 · базовая:  · теги:  · обновлено: 2026-08-14T09:54:05.000Z

---

---
license: apache-2.0
---
# Anime Video Line Art Colorization (MiniMax-H3 LoRA)

This model is a LoRA fine-tuned on the video generation model [MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3), capable of generating exquisite anime visuals from input line art videos.

## Examples

<table>
<tr>
<td><video src="assets/video_input.mp4" autoplay muted loop controls></video></td>
<td><video src="assets/example_1.mp4" autoplay muted loop controls></video></td>
</tr>
<tr>
<td><video src="assets/example_2.mp4" autoplay muted loop controls></td>
<td><video src="assets/example_3.mp4" autoplay muted loop controls></video></td>
</tr>
</table>

<details>
<summary>Prompts</summary>

> Prompts can directly describe the visual content. The prompts used in the three examples above are as follows:
> 
> * Prompt 1: A shy, blushing girl with long blue hair in white clothes and a frosty headband, sitting against a glacial ice wall with a bashful expression, surrounded by floating ice shards and soft magical glow.
> * Prompt 2: A shy, blushing girl with long black hair in a yellow top, white headband, and pink bow, sitting against a plain white wall with a gentle, bashful expression, softly and evenly lit from the front.
> * Prompt 3: A shy, blushing girl with long red hair in a white top, black headband, and pink bow, sitting on a bed in a bedroom bathed in bright, golden sunset light. Warm orange and amber rays stream through the window, illuminating the scene with a radiant glow as her gentle, bashful expression is softly highlighted by the luminous evening atmosphere.
> 
> Alternatively, editing instructions can be used as prompts, allowing the model to determine the visual content autonomously:
> 
> * Editing prompt: Generate anime videos from line art outlines.

</details>

## Inference

First, install [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio):

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e ".[all]"
```

```python
import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-ref2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
lora = ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime", origin_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, lora)

dataset_snapshot_download(
    "DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime",
    allow_file_pattern="assets/video_input.mp4",
    local_dir="data"
)

# The height and width must be consistent!
control_video, _, _ = read_video_audio("data/assets/video_input.mp4", height=768, width=1344, num_frames=90, fps=24, audio_sample_rate=32000)
prompt = "A shy, blushing girl with long blue hair in white clothes and a frosty headband, sitting against a glacial ice wall with a bashful expression, surrounded by floating ice shards and soft magical glow."
video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=90, num_inference_steps=20, seed=42,
    references=[
        {"type": "video", "video": control_video},
    ],
    ref_video_short_edge=768, ref_video_max_pixels=768*1344,
)
write_video_audio(
    video=video, audio=audio,
    output_path="output.mp4", fps=24, audio_sample_rate=32000,
)
```

## Training

Please refer to the [example code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh) and [documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/MiniMax-H3.html) provided by DiffSynth-Studio.
