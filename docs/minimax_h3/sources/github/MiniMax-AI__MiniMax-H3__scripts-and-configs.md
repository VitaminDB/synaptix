# MiniMax-AI__MiniMax-H3__scripts-and-configs

Source: https://github.com/MiniMax-AI/MiniMax-H3/tree/main/scripts/readme
Fetched: 2026-11-28



---

## FILE: FL2VA/model_index.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/FL2VA/model_index.json)

```
{
  "_class_name": "MiniMaxH3Pipeline",
  "_diffusers_version": "0.32.2",
  "text_encoder": [
    "transformers",
    "MiniMaxH3Qwen3VLHFEncoder"
  ],
  "tokenizer": [
    "transformers",
    "Qwen2TokenizerFast"
  ],
  "video_vae": [
    "diffusers",
    "MiniMaxH3VideoVAE"
  ],
  "audio_vae": [
    "diffusers",
    "MiniMaxH3AudioVAE"
  ],
  "scheduler": null,
  "transformer": [
    "diffusers",
    "MiniMaxH3DiTModel"
  ],
  "processor": [
    "transformers",
    "Qwen3VLProcessor"
  ],
  "_minimax_h3": {
    "schema_version": 1,
    "partition": "fl2va",
    "tasks": [
      "t2va",
      "fl2va"
    ],
    "task_aliases": {},
    "sigma_shift_scales": {
      "video": 12.0,
      "audio": 3.0
    }
  }
}

```


---

## FILE: FL2VA/transformer/config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/FL2VA/transformer/config.json)

```
{
  "_class_name": "MiniMaxH3DiTModel",
  "_diffusers_version": "0.32.2",
  "hidden_size": 5376,
  "num_layers": 50,
  "token_refiner_num_layers": 2,
  "num_attention_heads": 56,
  "attention_head_dim": 128,
  "ffn_hidden_size": 14336,
  "latents_dim": 24,
  "audio_latents_dim": 32,
  "patch_size": [
    1,
    2,
    2
  ],
  "text_dim": 5120,
  "timestep_input_dim": 256,
  "time_embed_hidden_size": 5376,
  "time_embed_dim": 2688,
  "adaln_out_features": 96768,
  "final_adaln_out_features": 10752,
  "rope_inv_freq_len": 16,
  "norm_eps": 1e-05,
  "qk_norm_eps": 1e-05,
  "final_norm_eps": 1e-05
}

```


---

## FILE: Ref2VA/model_index.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/Ref2VA/model_index.json)

```
{
  "_class_name": "MiniMaxH3Pipeline",
  "_diffusers_version": "0.32.2",
  "text_encoder": [
    "transformers",
    "MiniMaxH3Qwen3VLHFEncoder"
  ],
  "tokenizer": [
    "transformers",
    "Qwen2TokenizerFast"
  ],
  "video_vae": [
    "diffusers",
    "MiniMaxH3VideoVAE"
  ],
  "audio_vae": [
    "diffusers",
    "MiniMaxH3AudioVAE"
  ],
  "scheduler": null,
  "transformer": [
    "diffusers",
    "MiniMaxH3DiTModel"
  ],
  "processor": [
    "transformers",
    "Qwen3VLProcessor"
  ],
  "_minimax_h3": {
    "schema_version": 1,
    "partition": "ref2va",
    "tasks": [
      "ref2va"
    ],
    "task_aliases": {},
    "sigma_shift_scales": {
      "video": 12.0,
      "audio": 3.0
    }
  }
}

```


---

## FILE: Ref2VA/transformer/config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/Ref2VA/transformer/config.json)

```
{
  "_class_name": "MiniMaxH3DiTModel",
  "_diffusers_version": "0.32.2",
  "hidden_size": 5376,
  "num_layers": 50,
  "token_refiner_num_layers": 2,
  "num_attention_heads": 56,
  "attention_head_dim": 128,
  "ffn_hidden_size": 14336,
  "latents_dim": 24,
  "audio_latents_dim": 32,
  "patch_size": [
    1,
    2,
    2
  ],
  "text_dim": 5120,
  "timestep_input_dim": 256,
  "time_embed_hidden_size": 5376,
  "time_embed_dim": 2688,
  "adaln_out_features": 96768,
  "final_adaln_out_features": 10752,
  "rope_inv_freq_len": 16,
  "norm_eps": 1e-05,
  "qk_norm_eps": 1e-05,
  "final_norm_eps": 1e-05
}

```


---

## FILE: audio_scheduler/scheduler_config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/audio_scheduler/scheduler_config.json)

```
{
  "_class_name": "MiniMaxH3Scheduler",
  "_diffusers_version": "0.36.0.dev0",
  "shift": 3.0
}
```


---

## FILE: audio_vae/config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/audio_vae/config.json)

```
{
  "_class_name": "AutoencoderKLMiniMaxH3Audio",
  "_diffusers_version": "0.36.0.dev0",
  "encoder_dim": 64,
  "encoder_rates": [
    2,
    4,
    4,
    5,
    5
  ],
  "latent_dim": 2048,
  "latent_channels": 32,
  "decoder_dim": 1024,
  "decoder_rates": [
    5,
    5,
    2,
    2,
    2,
    2,
    2
  ],
  "decoder_kernel_sizes": [
    9,
    9,
    4,
    4,
    4,
    4,
    4
  ],
  "num_attention_heads": 8,
  "resblock_kernel_sizes": [
    3,
    7,
    11
  ],
  "resblock_dilation_sizes": [
    [
      1,
      3,
      5
    ],
    [
      1,
      3,
      5
    ],
    [
      1,
      3,
      5
    ]
  ],
  "sampling_rate": 32000,
  "latents_mean": [
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797
  ],
  "latents_std": [
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363
  ]
}
```


---

## FILE: model_index.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/model_index.json)

```
{
  "_class_name": "MiniMaxH3ModularPipeline",
  "_diffusers_version": "0.36.0.dev0",
  "_blocks_class_name": "MiniMaxH3Blocks",
  "text_encoder": [
    "transformers",
    "Qwen3VLForConditionalGeneration",
    {
      "type_hint": [
        "transformers",
        "Qwen3VLForConditionalGeneration"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "text_encoder",
      "variant": null,
      "revision": null
    }
  ],
  "tokenizer": [
    "transformers",
    "Qwen2TokenizerFast",
    {
      "type_hint": [
        "transformers",
        "Qwen2TokenizerFast"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "tokenizer",
      "variant": null,
      "revision": null
    }
  ],
  "processor": [
    "transformers",
    "Qwen3VLProcessor",
    {
      "type_hint": [
        "transformers",
        "Qwen3VLProcessor"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "processor",
      "variant": null,
      "revision": null
    }
  ],
  "vae": [
    "diffusers",
    "AutoencoderKLMiniMaxH3",
    {
      "type_hint": [
        "diffusers",
        "AutoencoderKLMiniMaxH3"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "vae",
      "variant": null,
      "revision": null
    }
  ],
  "audio_vae": [
    "diffusers",
    "AutoencoderKLMiniMaxH3Audio",
    {
      "type_hint": [
        "diffusers",
        "AutoencoderKLMiniMaxH3Audio"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "audio_vae",
      "variant": null,
      "revision": null
    }
  ],
  "transformer": [
    "diffusers",
    "MiniMaxH3Transformer3DModel",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Transformer3DModel"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "transformer",
      "variant": null,
      "revision": null
    }
  ],
  "transformer_ref": [
    "diffusers",
    "MiniMaxH3Transformer3DModel",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Transformer3DModel"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "transformer_ref",
      "variant": null,
      "revision": null
    }
  ],
  "scheduler": [
    "diffusers",
    "MiniMaxH3Scheduler",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Scheduler"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "scheduler",
      "variant": null,
      "revision": null
    }
  ],
  "audio_scheduler": [
    "diffusers",
    "MiniMaxH3Scheduler",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Scheduler"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "audio_scheduler",
      "variant": null,
      "revision": null
    }
  ]
}

```


---

## FILE: modular_model_index.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/modular_model_index.json)

```
{
  "_class_name": "MiniMaxH3ModularPipeline",
  "_diffusers_version": "0.36.0.dev0",
  "_blocks_class_name": "MiniMaxH3Blocks",
  "text_encoder": [
    "transformers",
    "Qwen3VLForConditionalGeneration",
    {
      "type_hint": [
        "transformers",
        "Qwen3VLForConditionalGeneration"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "text_encoder",
      "variant": null,
      "revision": null
    }
  ],
  "tokenizer": [
    "transformers",
    "Qwen2TokenizerFast",
    {
      "type_hint": [
        "transformers",
        "Qwen2TokenizerFast"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "tokenizer",
      "variant": null,
      "revision": null
    }
  ],
  "processor": [
    "transformers",
    "Qwen3VLProcessor",
    {
      "type_hint": [
        "transformers",
        "Qwen3VLProcessor"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "processor",
      "variant": null,
      "revision": null
    }
  ],
  "vae": [
    "diffusers",
    "AutoencoderKLMiniMaxH3",
    {
      "type_hint": [
        "diffusers",
        "AutoencoderKLMiniMaxH3"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "vae",
      "variant": null,
      "revision": null
    }
  ],
  "audio_vae": [
    "diffusers",
    "AutoencoderKLMiniMaxH3Audio",
    {
      "type_hint": [
        "diffusers",
        "AutoencoderKLMiniMaxH3Audio"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "audio_vae",
      "variant": null,
      "revision": null
    }
  ],
  "transformer": [
    "diffusers",
    "MiniMaxH3Transformer3DModel",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Transformer3DModel"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "transformer",
      "variant": null,
      "revision": null
    }
  ],
  "transformer_ref": [
    "diffusers",
    "MiniMaxH3Transformer3DModel",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Transformer3DModel"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "transformer_ref",
      "variant": null,
      "revision": null
    }
  ],
  "scheduler": [
    "diffusers",
    "MiniMaxH3Scheduler",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Scheduler"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "scheduler",
      "variant": null,
      "revision": null
    }
  ],
  "audio_scheduler": [
    "diffusers",
    "MiniMaxH3Scheduler",
    {
      "type_hint": [
        "diffusers",
        "MiniMaxH3Scheduler"
      ],
      "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
      "subfolder": "audio_scheduler",
      "variant": null,
      "revision": null
    }
  ]
}
```


---

## FILE: processor/chat_template.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/processor/chat_template.json)

```
{
    "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' }}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content_item in message.content %}\n                {%- if 'text' in content_item %}\n                    {{- content_item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and message.content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
}
```


---

## FILE: processor/preprocessor_config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/processor/preprocessor_config.json)

```
{
    "size": {
        "longest_edge": 16777216,
        "shortest_edge": 65536
    },
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast"
}
```


---

## FILE: processor/video_preprocessor_config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/processor/video_preprocessor_config.json)

```
{
    "size": {
        "longest_edge": 25165824,
        "shortest_edge": 4096
    },
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "processor_class": "Qwen3VLProcessor",
    "video_processor_type": "Qwen3VLVideoProcessor"
}
```


---

## FILE: requirements.txt (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/requirements.txt)

```
# Python dependencies for the MiniMax H3 inference bundle.
#
# The model is released as a Hugging Face diffusers pipeline and is consumed via
# `trust_remote_code=True`, so it relies on a diffusers build that ships the
# MiniMax-H3 model classes. Pin a known-good diffusers release and a torch
# version that ships the CUDA / MPS wheels needed to run it.
#
# Issue: https://github.com/MiniMax-AI/MiniMax-H3/issues/3

# Core deep-learning stack. torch is left at a recent stable major; pin
# tighter if you need an exact CUDA build (e.g. torch==2.4.1+cu121).
torch>=2.4.0

# MiniMax-H3 was merged after diffusers 0.39.0. Pin the merge
# commit so installs contain the H3 modular pipeline and remain reproducible.
diffusers @ git+https://github.com/huggingface/diffusers.git@f53d552036a0d1bd5570782a39cd40cfabf112bc

# transformers provides the Qwen3-VL text encoder / processor / tokenizer used
# by the H3 pipeline. Qwen3-VL support starts in transformers 4.57.
transformers>=4.57.0

# safetensors is used to load the VAE / Transformer weights directly.
safetensors>=0.4.3

# accelerate is needed for device_map / multi-GPU loading of the 33B
# Omni-Transformer and the visual VAE.
accelerate>=0.34.0

# huggingface_hub provides the `hf` CLI used to download the checkpoint
# (`hf download MiniMaxAI/MiniMax-H3 ...` in the README).
huggingface_hub>=0.25.0

# numpy is used throughout the audio / video VAE code.
numpy>=1.24.0

# Direct dependencies used by the released Visual and Audio VAE code.
torchvision>=0.19.0
einops>=0.8.0
PyYAML>=6.0.1

# Pillow is required by the Qwen3-VL processor for image decoding.
Pillow>=10.0.0

# Audio I/O: the audio VAE returns 32 kHz stereo tensors. soundfile is the
# recommended writer; librosa is optional but useful for resampling.
soundfile>=0.12.0

# Video I/O helpers used by the diffusers / ComfyUI integration.
imageio>=2.34.0
imageio-ffmpeg>=0.5.0
av>=11.0.0

```


---

## FILE: scheduler/scheduler_config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scheduler/scheduler_config.json)

```
{
  "_class_name": "MiniMaxH3Scheduler",
  "_diffusers_version": "0.36.0.dev0",
  "shift": 12.0
}
```


---

## FILE: scripts/readme/full-2k-i2va-h3-base.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-i2va-h3-base.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create the H3-Base request with the expanded prompt and capture the video ID.
video_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    '{
  "task": "fl2va",
  "prompt": $prompt,
  "conditions": [
    {
      "type": "image",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png",
      "role": "keyframe",
      "frame_index": 0
    }
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "auto",
    "duration_seconds": 8
  },
  "seed": 0
}' |
    curl --silent --show-error \
      --request POST \
      --url "$SGLANG_DEPLOYMENT_URL/v1/videos" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.id'
)
# Query the generation status.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id" |
  jq '{status}'
# Download the local H3-Base MP4 after its status becomes completed.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id/content" \
  --output i2va.mp4

```


---

## FILE: scripts/readme/full-2k-i2va-h3-context-ir.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-i2va-h3-context-ir.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create the prompt-expansion task and capture its runtime ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/h3_context_ir" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Pull focus to the people in the background and add more steam to the ramen bowl."
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
      },
      "role": "first_frame"
    }
  ],
  "duration": 8,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
context_ir_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$context_ir_result" | jq .
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')

```


---

## FILE: scripts/readme/full-2k-i2va-h3-regenerate-2k.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-i2va-h3-regenerate-2k.sh)

```
#!/usr/bin/env bash
set -euo pipefail

H3_BASE_VIDEO='./i2va.mp4'
# Encode the local H3-Base video as a Data URL and create the regeneration task.
task_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    --rawfile base_video <(
      printf 'data:video/mp4;base64,'
      base64 -w0 "$H3_BASE_VIDEO"
    ) \
    '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": $prompt
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
      },
      "role": "first_frame"
    },
    {
      "type": "video_url",
      "video_url": {
        "url": $base_video
      },
      "role": "base_video"
    }
  ],
  "resolution": "2K"
}' |
    curl --silent --show-error \
      --request POST \
      --url "$MINIMAX_API_BASE/v2/video_regeneration" \
      --header "Authorization: Bearer $TOKEN" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
regeneration_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$regeneration_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$regeneration_result" | jq -er '.task.content.url')
curl --location "$video_url" --output i2va_2k.mp4

```


---

## FILE: scripts/readme/full-2k-i2va-reference-2k-result-by-directly-calling-open-platform-api.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-i2va-reference-2k-result-by-directly-calling-open-platform-api.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create an 8-second 2K FL2VA video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Pull focus to the people in the background and add more steam to the ramen bowl."
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
      },
      "role": "first_frame"
    }
  ],
  "resolution": "2K",
  "duration": 8,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
generation_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$generation_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curl --location "$video_url" --output i2va_direct_2k.mp4

```


---

## FILE: scripts/readme/full-2k-i2va-reference-768p-result-by-directly-calling-open-platform-api.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-i2va-reference-768p-result-by-directly-calling-open-platform-api.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create an 8-second 768P FL2VA video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Pull focus to the people in the background and add more steam to the ramen bowl."
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
      },
      "role": "first_frame"
    }
  ],
  "resolution": "768P",
  "duration": 8,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
generation_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$generation_result" | jq '{status: .task.status}'
# Download the 768P MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curl --location "$video_url" --output i2va_direct_768p.mp4

```


---

## FILE: scripts/readme/full-2k-ref2va-h3-api-2k-in-open-platform-for-reference.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-ref2va-h3-api-2k-in-open-platform-for-reference.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create a 5-second 2K Ref2VA video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Character speaks: Follow the wind, live free. Leave worries behind, enjoy the moment. Voice timbre follows reference audio 1."
    },
    {
      "type": "video_url",
      "video_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
      },
      "role": "reference_video"
    },
    {
      "type": "audio_url",
      "audio_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
      },
      "role": "reference_audio"
    }
  ],
  "resolution": "2K",
  "duration": 5,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
generation_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$generation_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curl --location "$video_url" --output r2va_direct_2k.mp4

```


---

## FILE: scripts/readme/full-2k-ref2va-h3-base.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-ref2va-h3-base.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create the H3-Base request with the expanded prompt and capture the video ID.
video_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    '{
  "task": "ref2va",
  "prompt": $prompt,
  "conditions": [
    {
      "type": "video",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4",
      "role": "reference"
    },
    {
      "type": "audio",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3",
      "role": "reference"
    }
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "auto",
    "duration_seconds": 5
  },
  "seed": 0
}' |
    curl --silent --show-error \
      --request POST \
      --url "$SGLANG_DEPLOYMENT_URL/v1/videos" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.id'
)
# Query the generation status.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id" |
  jq '{status}'
# Download the local H3-Base MP4 after its status becomes completed.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id/content" \
  --output r2va.mp4

```


---

## FILE: scripts/readme/full-2k-ref2va-h3-context-ir.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-ref2va-h3-context-ir.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create the prompt-expansion task and capture its runtime ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/h3_context_ir" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Character speaks: Follow the wind, live free. Leave worries behind, enjoy the moment. Voice timbre follows reference audio 1."
    },
    {
      "type": "video_url",
      "video_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
      },
      "role": "reference_video"
    },
    {
      "type": "audio_url",
      "audio_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
      },
      "role": "reference_audio"
    }
  ],
  "duration": 5,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
context_ir_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$context_ir_result" | jq .
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')

```


---

## FILE: scripts/readme/full-2k-ref2va-reference-2k-result-by-directly-calling-open-platform-api.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-ref2va-reference-2k-result-by-directly-calling-open-platform-api.sh)

```
#!/usr/bin/env bash
set -euo pipefail

H3_BASE_VIDEO='./r2va.mp4'
# Encode the local H3-Base video as a Data URL and create the regeneration task.
task_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    --rawfile base_video <(
      printf 'data:video/mp4;base64,'
      base64 -w0 "$H3_BASE_VIDEO"
    ) \
    '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": $prompt
    },
    {
      "type": "video_url",
      "video_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
      },
      "role": "reference_video"
    },
    {
      "type": "audio_url",
      "audio_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
      },
      "role": "reference_audio"
    },
    {
      "type": "video_url",
      "video_url": {
        "url": $base_video
      },
      "role": "base_video"
    }
  ],
  "resolution": "2K"
}' |
    curl --silent --show-error \
      --request POST \
      --url "$MINIMAX_API_BASE/v2/video_regeneration" \
      --header "Authorization: Bearer $TOKEN" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
regeneration_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$regeneration_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$regeneration_result" | jq -er '.task.content.url')
curl --location "$video_url" --output r2va_2k.mp4

```


---

## FILE: scripts/readme/full-2k-ref2va-reference-768p-result-by-directly-calling-open-platform-api.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-ref2va-reference-768p-result-by-directly-calling-open-platform-api.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create a 5-second 768P Ref2VA video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Character speaks: Follow the wind, live free. Leave worries behind, enjoy the moment. Voice timbre follows reference audio 1."
    },
    {
      "type": "video_url",
      "video_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
      },
      "role": "reference_video"
    },
    {
      "type": "audio_url",
      "audio_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
      },
      "role": "reference_audio"
    }
  ],
  "resolution": "768P",
  "duration": 5,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
generation_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$generation_result" | jq '{status: .task.status}'
# Download the 768P MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curl --location "$video_url" --output r2va_direct_768p.mp4

```


---

## FILE: scripts/readme/full-2k-t2va-h3-base.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-t2va-h3-base.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create the H3-Base request with the expanded prompt and capture the video ID.
video_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    '{
  "task": "t2va",
  "prompt": $prompt,
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 10
  },
  "seed": 0
}' |
    curl --silent --show-error \
      --request POST \
      --url "$SGLANG_DEPLOYMENT_URL/v1/videos" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.id'
)
# Query the generation status.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id" |
  jq '{status}'
# Download the local H3-Base MP4 after its status becomes completed.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id/content" \
  --output t2va.mp4

```


---

## FILE: scripts/readme/full-2k-t2va-h3-context-ir.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-t2va-h3-context-ir.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create the prompt-expansion task and capture its runtime ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/h3_context_ir" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "duration": 10,
  "ratio": "16:9"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
context_ir_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$context_ir_result" | jq .
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')

```


---

## FILE: scripts/readme/full-2k-t2va-h3-regenerate-2k.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-t2va-h3-regenerate-2k.sh)

```
#!/usr/bin/env bash
set -euo pipefail

H3_BASE_VIDEO='./t2va.mp4'
# Encode the local H3-Base video as a Data URL and create the regeneration task.
task_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    --rawfile base_video <(
      printf 'data:video/mp4;base64,'
      base64 -w0 "$H3_BASE_VIDEO"
    ) \
    '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": $prompt
    },
    {
      "type": "video_url",
      "video_url": {
        "url": $base_video
      },
      "role": "base_video"
    }
  ],
  "resolution": "2K"
}' |
    curl --silent --show-error \
      --request POST \
      --url "$MINIMAX_API_BASE/v2/video_regeneration" \
      --header "Authorization: Bearer $TOKEN" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
regeneration_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$regeneration_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$regeneration_result" | jq -er '.task.content.url')
curl --location "$video_url" --output t2va_2k.mp4

```


---

## FILE: scripts/readme/full-2k-t2va-reference-2k-result-by-directly-calling-open-platform-api.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-t2va-reference-2k-result-by-directly-calling-open-platform-api.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create a 10-second 2K video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "resolution": "2K",
  "duration": 10,
  "ratio": "16:9"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
generation_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$generation_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curl --location "$video_url" --output h3_direct_2k.mp4

```


---

## FILE: scripts/readme/full-2k-t2va-reference-768p-result-by-directly-calling-open-platform-api.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/full-2k-t2va-reference-768p-result-by-directly-calling-open-platform-api.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Create a 10-second 768P video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "resolution": "768P",
  "duration": 10,
  "ratio": "16:9"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
generation_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$generation_result" | jq '{status: .task.status}'
# Download the 768P MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curl --location "$video_url" --output h3_direct_768p.mp4

```


---

## FILE: scripts/readme/reproducible-768p-fl2va-request.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/reproducible-768p-fl2va-request.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Submit the FL2VA request with the complete H3-Context-IR prompt.
response=$(
  curl --fail-with-body --silent --show-error \
    --request POST \
    --url http://localhost:30010/v1/videos \
    --header 'Content-Type: application/json' \
    --data-binary @- <<'JSON'
{
  "task": "fl2va",
  "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] This is a live-action, cinematic shot with a shallow depth of field. The camera holds a perfectly static shot throughout the entire eight-second duration, capturing a cozy family gathering in a traditional Japanese dining room. The scene opens with a large, intricately patterned blue and white ceramic bowl of ramen in the immediate foreground, rendered in crisp, sharp focus. The bowl sits on a smooth, polished long wooden table. Inside the bowl, a rich, oily golden-brown broth surrounds yellow wavy noodles, topped with two thick, round slices of chashu pork featuring visible fat marbling and a distinct spiral meat pattern. A generous mound of freshly chopped, bright green scallions rests in the center, and a crisp, dark green rectangular sheet of nori seaweed is tucked into the right edge. To the left of the bowl, a pair of light brown wooden chopsticks rests horizontally on a small, dark rectangular chopstick rest, near a small cylindrical ceramic teacup with blue painted patterns. On the right side of the table, a spherical paper lantern with a ribbed bamboo frame sits on a black wooden base. In the background, a large family of seven is gathered around the table, initially appearing as a soft, blurred presence. Behind them, traditional Japanese sliding shoji screens with wooden lattice frames are open, revealing a bright outdoor scene with lush green trees. Early in the clip, the thick, white steam rising from the hot ramen broth immediately intensifies, billowing upwards in thick, swirling clouds that dance continuously above the bowl. As the clip progresses into the middle seconds, the camera maintains its static position while the focus begins a deliberate, smooth shift deeper into the room. The foreground ramen bowl, its vibrant ingredients, and the rising steam gradually soften into a hazy, out-of-focus blur. Simultaneously, the family members in the background come into sharp, detailed clarity. The heavy steam continues to rise from the foreground, creating a dynamic, translucent veil between the camera and the family. With the focus now firmly locked on the background, the vibrant family dinner comes alive. The man in the dark navy blue long-sleeved shirt on the left leans forward, his mouth moving animatedly in a silent exchange. The young girl in the crisp white short-sleeved t-shirt beside him smiles brightly, looking toward the center of the table. The woman on the far left, wearing a soft light blue long-sleeved blouse, turns her head slightly, smiling gently. Across the table, the woman in the light grey button-down shirt smiles broadly, her eyes crinkling, as she rests her hands near her plate. The woman in the dark grey top further back uses her wooden chopsticks to pick up a small piece of food from a central ceramic dish filled with bright red pickled vegetables. The woman in the center back in the light grey sweater smiles gently, her hands clasped softly in front of her, observing the interaction. Throughout the remainder of the clip, the family continues their lively physical interaction, their mouths moving in continuous, silent cadences of conversation, while the thick, white steam from the blurred ramen bowl in the foreground never stops rising, adding a comforting atmosphere to the warm gathering.\n\noverall_soundscape: The soundscape begins with a quiet room tone mixed with the faint, airy rustle of the thick steam billowing from the hot ramen bowl in the foreground, accompanied by the subtle, continuous hissing and bubbling of the rich broth. As the visual focus shifts deeper into the room, the physical sounds of the bustling family dinner become dominant in the foreground. The clear, sharp clinking of ceramic bowls and wooden chopsticks touching plates is clearly heard as the family members reach for food. This is followed by the faint, muffled thud of a cup being set down on the smooth wooden table, and the subtle, rhythmic rustle of cotton and wool clothing as the family members lean forward and gesture, perfectly capturing the lively, physical atmosphere of the shared meal.\n\nnon_diegetic_music: A gentle, heartwarming acoustic guitar melody plays softly in the background, accompanied by the subtle, resonant notes of a traditional Japanese koto. The music maintains a slow, comforting tempo that enhances the cozy, nostalgic, and joyful atmosphere of the family gathering.",
  "conditions": [
    {
      "type": "image",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png",
      "role": "keyframe",
      "frame_index": 0
    }
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "auto",
    "duration_seconds": 8
  },
  "seed": 0
}
JSON
)
video_id=$(printf '%s\n' "$response" | jq -er '.id')
# Query the generation status.
curl --fail-with-body --silent --show-error \
  --request GET \
  --url "http://localhost:30010/v1/videos/$video_id" |
  jq '{status}'
# Download the generated video after its status becomes completed.
curl --fail-with-body --silent --show-error \
  --request GET \
  --url "http://localhost:30010/v1/videos/$video_id/content" \
  --output fl2va.mp4

```


---

## FILE: scripts/readme/reproducible-768p-ref2va-request.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/reproducible-768p-ref2va-request.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Submit the Ref2VA request with the complete H3-Context-IR prompt.
response=$(
  curl --fail-with-body --silent --show-error \
    --request POST \
    --url http://localhost:30011/v1/videos \
    --header 'Content-Type: application/json' \
    --data-binary @- <<'JSON'
{
  "task": "ref2va",
  "prompt": "subject_definitions:\n<Subject 1> is the young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, an unbuttoned white shirt, and silver rings, holding a small black lamb in his arms in <Video 1>.\n<Video 1> is the source video for the editing task.\n<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.\n<Audio 2> is the voice timbre reference for <Subject 1>'s voice, containing a spoken male voiceover.\n\nsummary:\n[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. <Subject 1>, wearing a bright pink suit and holding a black lamb, stands in a grassy field with other white lambs in the background. The edit animates <Subject 1>'s face to speak the user-provided dialogue. <Audio 1> is partially reused as the continuous background music, while the target references the calm male voice timbre of <Audio 2> for <Subject 1>'s spoken lines.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - the man retains his identity, wavy blonde hair, pink suit, white shirt, accessories, and the black lamb he holds, with his mouth newly animated to speak.\n<Video 1> (source video editing): fully_preserved - the original camera framing, warm golden hour lighting, grassy hill setting, and background white lambs are maintained while the central character is edited.\n<Audio 1>: partially_copy - the atmospheric background music from <Audio 1> is reused in the target video, mixed beneath the newly added spoken dialogue.\n<Audio 2>: reference - the target audio references the male voice timbre from <Audio 2> to generate <Subject 1>'s spoken dialogue.\n\ndetailed_description:\nThe target video is in realistic photographic style.\n[Shot 1] The shot begins from the source <Video 1>, showing <Subject 1>, a young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, and a casually unbuttoned white shirt. He stands confidently in a sunlit green pasture, gently holding a small black lamb securely in his arms. The warm, golden hour lighting casts soft shadows across his face and the bright pink fabric of his suit. Behind him, several white lambs stand and graze on the rolling grassy hill against a clear, pale blue sky. The atmospheric background music from <Audio 1> plays continuously throughout the scene. <Subject 1> physically speaks, his mouth movements naturally syncing to the new dialogue, with his voice timbre referencing the calm male delivery from <Audio 2>. Looking thoughtfully forward, <Subject 1> (S1) speaks softly, <d>[English] Follow the wind, live free.</d> As he delivers the line, he subtly shifts his weight, cradling the resting black lamb while the camera slowly pushes in. <Subject 1> (S1) continues his thought, <d>[English] Leave worries behind, enjoy the moment.</d> Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion. He then turns his gaze slightly away toward the horizon, gently stroking the black lamb's fleece with his fingers as the camera holds on this tranquil, sunlit state through the end of the video.\n\noverall_soundscape:\nThe soundscape consists of the continuous, atmospheric background music from <Audio 1>, overlaid with the clear, calm male dialogue spoken by the main character, referencing the voice timbre of <Audio 2>.\n\nnon_diegetic_music:\nThe atmospheric, sustained background music from <Audio 1> is reused as the continuous score, playing quietly beneath the spoken dialogue.",
  "conditions": [
    {
      "type": "video",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4",
      "role": "reference"
    },
    {
      "type": "audio",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3",
      "role": "reference"
    }
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "auto",
    "duration_seconds": 5
  },
  "seed": 0
}
JSON
)
video_id=$(printf '%s\n' "$response" | jq -er '.id')
# Query the generation status.
curl --fail-with-body --silent --show-error \
  --request GET \
  --url "http://localhost:30011/v1/videos/$video_id" |
  jq '{status}'
# Download the generated video after its status becomes completed.
curl --fail-with-body --silent --show-error \
  --request GET \
  --url "http://localhost:30011/v1/videos/$video_id/content" \
  --output ref2va.mp4

```


---

## FILE: scripts/readme/reproducible-768p-t2va-request.sh (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/scripts/readme/reproducible-768p-t2va-request.sh)

```
#!/usr/bin/env bash
set -euo pipefail

# Submit the T2VA request with the complete H3-Context-IR prompt.
response=$(
  curl --fail-with-body --silent --show-error \
    --request POST \
    --url http://localhost:30010/v1/videos \
    --header 'Content-Type: application/json' \
    --data-binary @- <<'JSON'
{
  "task": "t2va",
  "prompt": "integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a massive, curved observation window. A female captain, in her late 40s with an athletic build and short silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation against a deep purple space nebula. The fleet's massive rear thrusters begin to glow with an intense, escalating bright blue light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and shakes strongly. The brilliant blue-white light from the fleet's gathering energy reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the window, completely washing out the background as the fleet jumps to hyperspace. The sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to stagger slightly forward, her shoulders tensing as she visibly braces herself against the physical tremors. As the intense white light fades abruptly, leaving only the dim, empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches, and she slowly closes her eyes in the newly emptied space.\noverall_soundscape: A low, resonant hum of the ship's ambient life support systems serves as the baseline, soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its hyperdrives. A massive, deafening, bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating under immense physical stress. The intense roaring impact then cuts abruptly back to a hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.\nnon_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French horn melody over deep, sustained string dissonances that build rapidly in volume and intensity, swelling to a massive orchestral peak before snapping immediately into silence right after the jump.",
  "conditions": [
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 10
  },
  "seed": 0
}
JSON
)
video_id=$(printf '%s\n' "$response" | jq -er '.id')
# Query the generation status.
curl --fail-with-body --silent --show-error \
  --request GET \
  --url "http://localhost:30010/v1/videos/$video_id" |
  jq '{status}'
# Download the generated video after its status becomes completed.
curl --fail-with-body --silent --show-error \
  --request GET \
  --url "http://localhost:30010/v1/videos/$video_id/content" \
  --output t2va.mp4

```


---

## FILE: skills-lock.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills-lock.json)

```
{
  "version": 1,
  "skills": {
    "h3-prompt-writing": {
      "source": "MiniMax-AI/MiniMax-H3",
      "sourceType": "github",
      "skillPath": "skills/h3-prompt-writing/SKILL.md",
      "computedHash": "3d01859464bc9438585c8fdbf7fcd4b4c54404fadd3f1a64ab7970ae8877d086"
    }
  }
}

```


---

## FILE: transformer/config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/transformer/config.json)

```
{
  "_class_name": "MiniMaxH3Transformer3DModel",
  "_diffusers_version": "0.36.0.dev0",
  "num_attention_heads": 56,
  "attention_head_dim": 128,
  "hidden_size": 5376,
  "num_layers": 50,
  "num_refiner_layers": 2,
  "ffn_dim": 14336,
  "in_channels": 24,
  "audio_in_channels": 32,
  "patch_size": [
    1,
    2,
    2
  ],
  "text_dim": 5120,
  "freq_dim": 256,
  "time_embed_hidden_dim": 5376,
  "time_embed_dim": 2688,
  "rope_freq_dim": 16,
  "rope_theta": 10000.0,
  "norm_eps": 1e-05,
  "qk_norm_eps": 1e-05,
  "final_norm_eps": 1e-05
}
```


---

## FILE: transformer_ref/config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/transformer_ref/config.json)

```
{
  "_class_name": "MiniMaxH3Transformer3DModel",
  "_diffusers_version": "0.36.0.dev0",
  "num_attention_heads": 56,
  "attention_head_dim": 128,
  "hidden_size": 5376,
  "num_layers": 50,
  "num_refiner_layers": 2,
  "ffn_dim": 14336,
  "in_channels": 24,
  "audio_in_channels": 32,
  "patch_size": [
    1,
    2,
    2
  ],
  "text_dim": 5120,
  "freq_dim": 256,
  "time_embed_hidden_dim": 5376,
  "time_embed_dim": 2688,
  "rope_freq_dim": 16,
  "rope_theta": 10000.0,
  "norm_eps": 1e-05,
  "qk_norm_eps": 1e-05,
  "final_norm_eps": 1e-05
}
```


---

## FILE: vae/config.json (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/vae/config.json)

```
{
  "_class_name": "AutoencoderKLMiniMaxH3",
  "_diffusers_version": "0.36.0.dev0",
  "in_channels": 3,
  "out_channels": 3,
  "latent_channels": 24,
  "block_out_channels": [
    128,
    256,
    256,
    512,
    512,
    1024
  ],
  "layers_per_block": 2,
  "spatial_downsample_factors": [
    2,
    2,
    2,
    2,
    1,
    1
  ],
  "temporal_downsample_factors": [
    1,
    2,
    2,
    1,
    1,
    1
  ],
  "norm_num_groups": 32,
  "norm_eps": 1e-06,
  "spatial_padding_mode": "reflect",
  "decoder_num_layers": 36,
  "decoder_num_attention_heads": 32,
  "decoder_attention_head_dim": 64,
  "decoder_num_register_tokens": 4,
  "decoder_ffn_mult": 4,
  "decoder_rope_theta": 100.0,
  "decoder_rope_dim_ratio": 0.75,
  "decoder_norm_eps": 1e-05,
  "clip_length": 17,
  "token_drop": 3,
  "latents_mean": [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264
  ],
  "latents_std": [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523
  ]
}
```
