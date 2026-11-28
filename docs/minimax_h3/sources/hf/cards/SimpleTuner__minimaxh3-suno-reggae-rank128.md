# SimpleTuner/minimaxh3-suno-reggae-rank128

Источник: https://huggingface.co/SimpleTuner/minimaxh3-suno-reggae-rank128 · скачано 2026-11-28
Загрузок: 45 · лайков: 0 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,template:sd-lora · обновлено: 2026-08-16T19:57:17.000Z

---

---
license: other
license_name: "minimax-h3-community-license-agreement"
license_link: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
base_model: "MiniMaxAI/MiniMax-H3"
tags:
  - minimaxh3
  - minimaxh3-diffusers
  - text-to-video
  - image-to-video
  - diffusers
  - simpletuner
  - not-for-all-audiences
  - lora


  - template:sd-lora
  - standard
pipeline_tag: text-to-video
inference: true
widget:
- text: '<STYLE>
roots reggae, warm analog dub production, laid-back one-drop drum groove, deep round bassline, skanking guitar upstrokes, organ bubble, spacious mix with tape echo and spring reverb, soulful male lead vocal, mid-tempo

<LYRICS>
[verse]
Morning sun a rise pon di mountain top
River run easy and di worries stop
We carry good vibes from di country road
Every likkle burden turn a lighter load

[chorus]
Lift up yuh heart now, sing it loud and clear
One love a di message and di roots right here
Drum and di bass dem a guide di way
Sweet reggae music till di break of day'
  parameters:
    negative_prompt: ''''
  output:
    url: ./assets/image_0_0.mp4
---

# bghira/minimaxh3-suno-reggae-rank128

This is a PEFT LoRA derived from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

The main validation prompt used during training was:
```
<STYLE>
roots reggae, warm analog dub production, laid-back one-drop drum groove, deep round bassline, skanking guitar upstrokes, organ bubble, spacious mix with tape echo and spring reverb, soulful male lead vocal, mid-tempo

<LYRICS>
[verse]
Morning sun a rise pon di mountain top
River run easy and di worries stop
We carry good vibes from di country road
Every likkle burden turn a lighter load

[chorus]
Lift up yuh heart now, sing it loud and clear
One love a di message and di roots right here
Drum and di bass dem a guide di way
Sweet reggae music till di break of day
```


## Validation settings
- CFG: `1.0`
- CFG Rescale: `0.0`
- Steps: `40`
- Sampler: `MiniMaxH3Scheduler`
- Seed: `42`
- Resolution: `256`


Note: The validation settings are not necessarily the same as the [training settings](#training-settings).

You can find some example images and videos in the following gallery:


<Gallery />

The text encoder **was not** trained.
You may reuse the base model text encoder for inference.


## Training settings

- Training epochs: 12
- Training steps: 1000
- Learning rate: 5e-05
  - Learning rate schedule: constant_with_warmup
  - Warmup steps: 500
- Max grad norm: 0.5
- Effective batch size: 8
  - Micro-batch size: 1
  - Gradient accumulation steps: 1
  - Number of GPUs: 8
- Gradient checkpointing: True
- Prediction type: flow_matching[]
- Optimizer: adamw_bf16 (config=weight_decay=0.0,eps=1e-8)
- Trainable parameter precision: Pure BF16
- Base model precision: `no_change`
- Caption dropout probability: 0.0%



- LoRA Rank: 128
- LoRA Alpha: 128.0
- LoRA Dropout: 0.0
- LoRA initialisation style: default
- LoRA mode: Standard
    

## Datasets

### suno-reggae-audio
- Repeats: 0
- Total number of images: 123
- Total number of aspect buckets: 15
- Resolution: 480 px
- Cropped: False
- Crop style: None
- Crop aspect: None
- Used for regularisation data: No


## Inference


```python
import torch
from diffusers import DiffusionPipeline

model_id = 'MiniMaxAI/MiniMax-H3'
adapter_id = 'bghira/minimaxh3-suno-reggae-rank128'
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) # loading directly in bf16
pipeline.load_lora_weights(adapter_id)

prompt = "<STYLE>
roots reggae, warm analog dub production, laid-back one-drop drum groove, deep round bassline, skanking guitar upstrokes, organ bubble, spacious mix with tape echo and spring reverb, soulful male lead vocal, mid-tempo

<LYRICS>
[verse]
Morning sun a rise pon di mountain top
River run easy and di worries stop
We carry good vibes from di country road
Every likkle burden turn a lighter load

[chorus]
Lift up yuh heart now, sing it loud and clear
One love a di message and di roots right here
Drum and di bass dem a guide di way
Sweet reggae music till di break of day"
negative_prompt = ''

## Optional: quantise the model to save on vram.
## Note: The model was not quantised during training, so it is not necessary to quantise it during inference time.
#from optimum.quanto import quantize, freeze, qint8
#quantize(pipeline.transformer, weights=qint8)
#freeze(pipeline.transformer)
    
pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
model_output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=256,
    height=256,
    guidance_scale=1.0,
).images[0]

model_output.save("output.png", format="PNG")

```
