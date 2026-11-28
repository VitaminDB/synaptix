# SceneWorks/minimax-h3-image-pe

Источник: https://huggingface.co/SceneWorks/minimax-h3-image-pe · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая: MiniMaxAI/MiniMax-H3-Image-PE · теги: quantized,base_model:quantized:MiniMaxAI/MiniMax-H3-Image-PE · обновлено: 2026-11-19T02:58:59.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3-Image-PE
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - minimax-h3-image
  - prompt-expansion
  - qwen3_5
  - mlx
  - candle
  - quantized
  - sceneworks
---

# MiniMax-H3-Image-PE: SceneWorks converted tiers (bf16 / q8 / q4)

> **Modified files.** Every file under `bf16/`, `q8/`, `q4/`, `candle-q8/` and `candle-q4/` was
> **modified by SceneWorks**. Each one was converted from, or copied out of, MiniMax's
> [`MiniMaxAI/MiniMax-H3-Image-PE`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) at
> revision `35603d4b63949afa12cb098227fffefedf081afa`. This notice is the one Section III.2 of the
> MiniMax-H3-Image Community License Agreement requires. This is **not** an official MiniMax
> distribution.

MiniMax-H3-Image-PE is the prompt expander for
[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), a fine-tune of
Qwen3.8-27B (`qwen3_5`, with its vision tower). It turns a short prompt, or an edit instruction
with its `<Picture N>` references, into the long caption MiniMax-H3-Image was trained on, and
proposes an aspect ratio.

MiniMax publishes it only as a ModelOpt **NVFP4** checkpoint, which runs natively only on
Blackwell NVIDIA GPUs. This repository holds converted tiers for Apple Silicon (MLX) and for
other CUDA GPUs and CPUs (Candle), as used by [SceneWorks](https://github.com/SceneWorks).

## License

Governed by the **MiniMax-H3-Image Community License Agreement** (November 10, 2026), carried
verbatim as [`LICENSE`](LICENSE). The notice Section III.4 requires is in [`NOTICE`](NOTICE).
Downloading or using these files means accepting that Agreement. Its key terms:

- **Applicable Territory.** Use and distribution are authorized only within the Applicable
  Territory: worldwide **except** the European Union, the United Kingdom, the Republic of Korea
  and the United States of America (Sections I.3, II and III).
- **Acceptable Use Policy** (Section V.1 and Exhibit A). For example, content you publish to a
  public environment must be clearly disclosed as machine-generated.
- **Commercial terms** (Section IV). You need separate written authorization from MiniMax
  (https://platform.minimax.io/license) if you and your affiliates have more than USD 1 million
  of annualized revenue, or if you provide a Hosted Service. A commercial product must
  prominently display "MiniMax-H3-Image" on its user interface.
- **No improving other models** (Section V.3). Neither the model nor its outputs, expanded
  prompts included, may be used to improve any AI model other than MiniMax-H3-Image or its
  Model Derivatives.

Read the full Agreement before downloading.

## Tiers

Each tier directory is a complete, standalone snapshot (weights, `config.json`, tokenizer, chat
template, processor and generation config). Download only the tier you need.

| directory | backend | format | size |
|---|---|---|---|
| `bf16/` | MLX and Candle | dense bf16, sharded safetensors | 54.7 GB |
| `q8/` | MLX (macOS) | MLX affine 8-bit, group size 64 | 31.9 GB |
| `q4/` | MLX (macOS) | MLX affine 4-bit, group size 64 | 19.8 GB |
| `candle-q8/` | Candle (Windows / Linux) | GGML Q8_0 | 31.9 GB |
| `candle-q4/` | Candle (Windows / Linux) | GGML Q4_K | 19.8 GB |

On a Blackwell GPU, use MiniMax's original NVFP4 checkpoint instead.

### What was changed

- **`bf16/`**: each ModelOpt NVFP4 projection was dequantized to a dense bf16 `.weight`, and its
  `weight_scale`, `weight_scale_2` and `input_scale` were dropped. Every other tensor was copied
  byte for byte. `config.json` lost its `quantization_config` and declares `dtype: bfloat16`.
  `hf_quant_config.json` and the source index were removed, and a new
  `model.safetensors.index.json` was written. The tokenizer, chat template, processor and
  generation config are unchanged copies.
- **`q8/`, `q4/`, `candle-q8/`, `candle-q4/`**: quantized from the `bf16/` snapshot above. The
  `config.json` `quantization` block records the bit width and storage format.

### What is not here

MiniMax's `prompt/` directory, which holds the system and task prompts the expander needs, is
**not** redistributed here. Download it from the upstream repository at the same revision.
SceneWorks reads it from there at run time and checks every file against the upstream
`prompt/manifest.json`. Without those prompts, the expander does not produce MiniMax-H3-Image
captions.

## Quality vs. NVFP4

Each tier was run on a fixed set of 8 prompts (text-to-image, and edit with reference images)
and compared with the native NVFP4 output from a Blackwell GPU (SceneWorks Inference run
[43028135206](https://github.com/SceneWorks/inference/actions/runs/43028135206)). Every tier
returned valid JSON for all 8 prompts.

| tier | backend | mean chrF vs NVFP4 | aspect ratio matches NVFP4 |
|---|---|---|---|
| bf16 | Candle | 55.3 | 6 / 8 |
| q8 | Candle | 54.7 | 6 / 8 |
| q4 | Candle | 53.6 | 5 / 8 |
| bf16 | MLX | 53.7 | 4 / 8 |
| q8 | MLX | 56.1 | 5 / 8 |
| q4 | MLX | 52.3 | 7 / 8 |

The NVFP4 reference is itself a single sampled expansion, so a difference of a few chrF points
between tiers is within sampling variation.

## How it was produced

Converted on the CPU with
[SceneWorks Inference](https://github.com/SceneWorks/inference):

```sh
# NVFP4 -> dense bf16 (bf16/)
cargo run --locked --release -p candle-llm --example modelopt_nvfp4_to_bf16 -- <MiniMax-H3-Image-PE @ 35603d4b> bf16
# bf16 -> Candle GGML tiers (candle-q8/, candle-q4/)
cargo run --locked --release -p candle-llm --example prepare_snapshot -- bf16 candle-q8 q8
# bf16 -> MLX affine tiers (q8/, q4/), on macOS
cargo run --locked --release -p mlx-llm --example prepare_snapshot -- bf16 q8 q8
```

The publication workflow is `real-weights-minimax-h3-image.yml` in that repository. It verifies
every uploaded file's size and sha256 against the hub.

## Citation and attribution

MiniMax-H3-Image and MiniMax-H3-Image-PE are by MiniMax. See the
[original model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) for usage, citation
and the upstream serving setup.
