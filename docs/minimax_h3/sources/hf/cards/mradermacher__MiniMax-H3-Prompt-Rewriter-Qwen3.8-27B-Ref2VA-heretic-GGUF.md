# mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF

Источник: https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF · скачано 2026-11-28
Загрузок: 5850 · лайков: 2 · базовая: funnygeeker/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic · теги: gguf,base_model:quantized:funnygeeker/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic · обновлено: 2026-11-18T09:51:33.000Z

---

---
base_model: funnygeeker/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic
language:
- en
library_name: transformers
license: apache-2.0
mradermacher:
  readme_rev: 1
quantized_by: mradermacher
tags:
- safetensors
- qwen3.5
- qwen3.8
- minimax
- heretic
- decensored
- uncensored
- abliterated
- prompt-rewriting
- ref2va
- multimodal
- vision-language
- mtp
- conversational
- en
- zh
- base_model:lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA
- base_model:finetune:lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA
---
## About

<!-- ### quantize_version: 2 -->
<!-- ### output_tensor_quantised: 1 -->
<!-- ### convert_type: hf -->
<!-- ### vocab_type:  -->
<!-- ### tags:  -->
<!-- ### quants:  x-f16 Q4_K_S Q2_K Q8_0 Q6_K Q3_K_M Q3_K_S Q3_K_L Q4_K_M Q5_K_S Q5_K_M IQ4_XS -->
<!-- ### quants_skip:  -->
<!-- ### skip_mmproj:  -->
static quants of https://huggingface.co/funnygeeker/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic

<!-- provided-files -->

***For a convenient overview and download list, visit our [model page for this model](https://hf.tst.eu/model#MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF).***

weighted/imatrix quants are available at https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-i1-GGUF
## Usage

If you are unsure how to use GGUF files, refer to one of [TheBloke's
READMEs](https://huggingface.co/TheBloke/KafkaLM-70B-German-V0.1-GGUF) for
more details, including on how to concatenate multi-part files.

## Provided Quants

(sorted by size, not necessarily quality. IQ-quants are often preferable over similar sized non-IQ quants)

| Link | Type | Size/GB | Notes |
|:-----|:-----|--------:|:------|
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.mmproj-Q8_0.gguf) | mmproj-Q8_0 | 0.7 | multi-modal supplement |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.mmproj-f16.gguf) | mmproj-f16 | 1.0 | multi-modal supplement |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q2_K.gguf) | Q2_K | 11.0 |  |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q3_K_S.gguf) | Q3_K_S | 12.4 |  |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q3_K_M.gguf) | Q3_K_M | 13.6 | lower quality |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q3_K_L.gguf) | Q3_K_L | 14.7 |  |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.IQ4_XS.gguf) | IQ4_XS | 15.5 |  |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q4_K_S.gguf) | Q4_K_S | 15.9 | fast, recommended |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q4_K_M.gguf) | Q4_K_M | 16.9 | fast, recommended |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q5_K_S.gguf) | Q5_K_S | 19.1 |  |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q5_K_M.gguf) | Q5_K_M | 19.6 |  |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q6_K.gguf) | Q6_K | 22.5 | very good quality |
| [GGUF](https://huggingface.co/mradermacher/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic-GGUF/resolve/main/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.Q8_0.gguf) | Q8_0 | 29.1 | fast, best quality |

Here is a handy graph by ikawrakow comparing some lower-quality quant
types (lower is better):

![image.png](https://www.nethype.de/huggingface_embed/quantpplgraph.png)

And here are Artefact2's thoughts on the matter:
https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9

## FAQ / Model Request

See https://huggingface.co/mradermacher/model_requests for some answers to
questions you might have and/or if you want some other model quantized.

## Thanks

I thank my company, [nethype GmbH](https://www.nethype.de/), for letting
me use its servers and providing upgrades to my workstation to enable
this work in my free time.

<!-- end -->
