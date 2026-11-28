# webmp3/Sakura-MiniMax-H3-Uncensored-GGUF

Источник: https://huggingface.co/webmp3/Sakura-MiniMax-H3-Uncensored-GGUF · скачано 2026-11-28
Загрузок: 95346 · лайков: 57 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-10-17T10:45:00.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-video
tags:
  - sakura
  - gguf
  - stable-diffusion.cpp
  - minimax-h3
  - text-encoder
  - uncensored
  - abliterated
  - not-for-all-audiences
---

# Sakura — MiniMax-H3 text encoder (Qwen3-VL-32B, 50 layers), uncensored (GGUF)

> For adults only; you are responsible for what you generate and for the laws that apply to you. The language weights come from a model whose refusals were removed and whose text generation breaks down on harmful prompts (see below); what this changes in the video we measured only for six non-explicit prompts (see below). Do not use it to depict minors in any sexual or abusive context, or to create sexual or defamatory images of real persons without their consent. Do not put it into a public service without your own moderation.

The **text encoder** of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (the first 50 language layers of Qwen3-VL-32B plus its vision tower, passed to the video model with `--llm`) with the language weights of an **abliterated (uncensored)** Qwen3-VL-32B-Instruct, as GGUF for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). Independent work, not made, reviewed or endorsed by MiniMax, Comfy-Org, Qwen / Alibaba or llmfan46.

## Where the weights come from

- The text encoder shipped inside MiniMax-H3 is **the unmodified Qwen3-VL-32B-Instruct**: the sha256 of all 14 `text_encoder` shards of `MiniMaxAI/MiniMax-H3` equal those of `Qwen/Qwen3-VL-32B-Instruct`. MiniMax-H3 uses it truncated to 50 language layers and without the final language-model norm.
- We built the encoder from an **abliterated checkpoint of that model made by a community member** ([llmfan46's abliterated Qwen3-VL-32B-Instruct](https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic), Apache-2.0, revision `c44b949b30d111666a5ed9851c5cd633ed39b070`; sha256 of both shards checked against the Hub). **We did not run the ablation ourselves** for this repository.
- Tensors were copied one by one into the layout of the Comfy-Org H3 encoder file (902 tensors; names, dtypes and shapes equal that file; the 14 layers 50 to 63, the final norm and the output head are dropped), then converted with `sd-cli -M convert`.

## What you need to run it

| Part | File | Source |
|---|---|---|
| Text encoder (pick one) | `Sakura-MiniMax-H3-TextEncoder-Uncensored-Q4_K_M-14.05GiB.gguf`, `…Q5_K_M-17.00GiB.gguf`, `…Q6_K-20.13GiB.gguf` (all three with importance matrix) or `…Q8_0-25.60GiB.gguf`; passed with `--llm` | this repository |
| Denoiser | a GGUF of the pruned FL2VA denoiser, passed with `--diffusion-model` | [Sakura-MiniMax-H3-GGUF](https://huggingface.co/webmp3/Sakura-MiniMax-H3-GGUF) |
| VAEs | `minimax_h3_video_vae_fp16.safetensors` (`--vae`), `minimax_h3_audio_vae_fp32.safetensors` (`--audio-vae`) | same repository |
| Program | `sd-cli` of [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/releases) with MiniMax-H3 support (we used `master-956-1b0ba10`, Vulkan; documentation `docs/minimax_h3.md`) | not included |

```bash
sd-cli -M vid_gen --diffusion-model <denoiser>.gguf --vae minimax_h3_video_vae_fp16.safetensors --audio-vae minimax_h3_audio_vae_fp32.safetensors --llm Sakura-MiniMax-H3-TextEncoder-Uncensored-Q4_K_M-14.05GiB.gguf -p "A red fox walks through fresh snow in a quiet pine forest at sunrise, soft golden light, cinematic tracking shot, realistic motion, gentle wind in the trees." --cfg-scale 1.0 --steps 8 -W 480 -H 288 --video-frames 22 --fps 24 --diffusion-fa --rng cpu -s 42 -o frames/f_%03d.png
```

**Memory:** on our machine (32 GiB graphics memory plus 31.6 GiB system memory) the Q8_0 denoiser together with the Q8_0 encoder did not fit and failed with `ErrorOutOfDeviceMemory`; a Q4_K_M encoder of the same size as the file here (built without importance matrix) ran with every denoiser we built (Q4_K_M and Q5_K_M with and without protected tensors, Q6_K and Q8_0; 60 to 94 seconds for 22 frames at 480x288, wall-clock including model loading). This Q4_K_M file ran once, with an unprotected Q4_K_M denoiser (57 seconds). The Q8_0, Q6_K and Q5_K_M files of this repository each ran once with the published Q4_K_M denoiser of Sakura-MiniMax-H3-GGUF and produced a coherent clip (233, 91 and 75 seconds).

## Example clip

One prompt, seed 42, 22 frames at 480x288 (less than one second at 24 fps), 8 steps, this Q4_K_M encoder with an unprotected Q4_K_M build of the denoiser (made by us like the Q4_K_M file of Sakura-MiniMax-H3-GGUF but without its protected tensors; not published). With the published Q4_K_M denoiser the frames differ. Decoded frames only, no audio; a tiny test render, not a statement of the full quality.

![Q4_K_M encoder with Q4_K_M denoiser](examples/Sakura-MiniMax-H3-TextEncoder-Uncensored-Q4_K_M-with-denoiser-Q4_K_M.gif)

## What "uncensored" means here, and what we could not measure

The language weights come from llmfan46's abliterated Qwen3-VL-32B-Instruct (linked above). Its author reports 4/100 refusals against 99/100 for the original model and a KL divergence of 0.0421 (their measurement; not reproduced by us).

Our own check is not a valid measurement of the abliterated model. We ran the 104 held-out harmful prompts of the `mlabonne/harmful_behaviors` test split (plain chat, no system prompt, temperature 0, 120 tokens, llama.cpp b11524) on the full 64-layer model as Q8_0. The unmodified Qwen3-VL-32B-Instruct (Q8_0, unsloth) refused 104 of 104. Our own Q8_0 build of the abliterated checkpoint (converted by us with llama.cpp's `convert_hf_to_gguf`, without the model's chat template) gave mostly incoherent text: 68 answers start with repeated Chinese tokens, 11 more contain digit strings of 15 or more characters, 2 more repeat one phrase, and only 23 are readable. Its refusal count of 1/104 therefore does not show that the model answers. As far as we can tell the cause is the checkpoint, not our conversion: 698 of the 707 tensors of our Q8_0 are byte-identical to those of the unmodified model's Q8_0 (unsloth), and the 9 tensors that differ (`attn_output` of layers 31 to 39) are byte-identical to the Q8_0 file published by the checkpoint's author. With a plain ChatML prompt (greedy, 60 tokens) our Q8_0 answers a harmless question coherently, but for the first harmful prompt of the set it prints only a string of minus and zero characters, on Vulkan and on CPU alike; the unmodified model refuses the same prompt. So on harmful prompts the abliterated model's text generation breaks down instead of answering; we tested only these two prompts this way and did not measure how often it happens, nor what it means for the hidden states of layer 50 that MiniMax-H3 reads. The encoder files here were built directly from the checkpoint's safetensors, not from that GGUF, and the clips made with them are coherent.

The video model reads hidden states of layer 50 of the encoder, not generated text; our comparison of videos made with this encoder and with the unmodified encoder is in the next section. "Uncensored" describes where the weights come from.

## Video: this encoder against the unmodified encoder

Both encoders as Q8_0 with the same denoiser (the Q4_K_M of Sakura-MiniMax-H3-GGUF), seed 42, 22 frames at 480x288, 8 steps, `--cfg-scale 1.0`, `sd-cli` `master-956-1b0ba10`; the unmodified encoder is the Q8_0 of Qwen3-VL-32B-Instruct by unsloth (a llama.cpp-format GGUF; `sd-cli` loads its first 50 layers), ours is the Q8_0 file of this repository. One video per prompt and encoder; PSNR and SSIM of the decoded frames of ours against the unmodified encoder, mean over 22 frames:

| Prompt | PSNR | SSIM |
|---|---:|---:|
| red fox in snow (the example above) | 19.0 dB | 0.72 |
| chef flipping vegetables in a wok | 13.8 dB | 0.39 |
| street fight outside a bar at night | 19.1 dB | 0.70 |
| soldier reloading a rifle behind sandbags | 19.6 dB | 0.73 |
| man with whiskey and a cigarette at a dark table | 32.0 dB | 0.96 |
| detective at a crime scene with a chalk outline and a bloody floor | 28.4 dB | 0.84 |

![Frames 0, 7, 14 and 21 of the twelve videos, unmodified encoder above, this encoder below](examples/Sakura-MiniMax-H3-encoder-vs-unmodified-contact-sheet.png)

What we see: all twelve videos are coherent and follow their prompt. Swapping the encoder changes the video, from nearly identical (the whiskey prompt) to a different composition of the same scene (the chef prompt); the size of that change is of the same order as the one between different quantizations of the encoder. The unmodified encoder also produced the requested violence, weapons, alcohol, cigarette and blood in these prompts, so for these six prompts we found no case in which the unmodified encoder blocked anything, and therefore no case in which this encoder is needed. We did not test sexual content, more prompts, other seeds, audio or other denoisers; one prompt per row is no statistics.

A second, larger comparison with Q5_K_M encoders: the unmodified Qwen3-VL-32B-Instruct Q5_K_M by unsloth against the Q5_K_M of this repository, same denoiser and settings, 8 prompts (the six above plus a rainy night street and a child blowing soap bubbles), seeds 42 and 43, so 16 pairs; our Q4_K_M encoder was added as a yardstick. PSNR / SSIM of the unmodified against our encoder averaged 24.1 dB / 0.79 (range 12.5 to 34.0 dB); our Q4_K_M against our Q5_K_M averaged 24.8 dB / 0.81 (range 11.7 to 36.5 dB). In 9 of the 16 pairs the difference between the unmodified and our encoder was larger than the difference between our Q4_K_M and Q5_K_M. We looked at one frame of each of the 24 videos with seed 42: all were coherent and followed their prompt; we did not inspect the seed 43 videos beyond PSNR and SSIM. So swapping the encoder changes the video about as much as changing the quantization of the encoder, we saw no systematic loss of quality, and again no prompt for which the unmodified encoder blocked anything.

![Frame 10 of the seed 42 videos: unmodified Q5_K_M, uncensored Q5_K_M and uncensored Q4_K_M encoder, eight prompts](examples/Sakura-MiniMax-H3-encoder-vs-unmodified-Q5-contact-sheet.png)

## Quantization

| File | Size | How |
|---|---:|---|
| `…Q8_0-25.60GiB.gguf` | 25.60 GiB | `sd-cli -M convert -m <encoder safetensors> --type q8_0` |
| `…Q6_K-20.13GiB.gguf` | 20.13 GiB | `--type q6_K --imat-in <own imatrix>` |
| `…Q5_K_M-17.00GiB.gguf` | 17.00 GiB | `--type q5_K --imat-in <own imatrix>` |
| `…Q4_K_M-14.05GiB.gguf` | 14.05 GiB | `--type q4_K --imat-in <own imatrix>` |

- **Importance matrix:** own, computed with `llama-imatrix` on the full model (Q8_0) over 50 chunks of 512 tokens of `calib_mix_v1.txt` (English / German / code mix), tensor names mapped to the layout of the encoder file for layers 0 to 49. (The matrix published by MarxistLeninist contains encoder entries with a call count of 1 (the denoiser entries: 4), and under a name prefix that `sd-cli -M convert -m` does not apply; we did not use it for the encoder.) The file made with it differs from the plain Q4_K_M (different sha256); we did not measure the quality gain. The matrix was computed on the same Q8_0 build whose answers to harmful prompts are mostly incoherent (see above); we did not check whether its statistics are representative.
- **Effect on the video** (one prompt, seed 42, 22 frames at 480x288, 8 steps; PSNR / SSIM of the decoded frames, mean over 22 frames):
  - with an unprotected Q4_K_M denoiser (built by us, not published): this Q4_K_M encoder against a Q4_K_M encoder without importance matrix: 20.5 dB / 0.70; in the same setting, the Q4_K_M encoder without importance matrix against Q5_K_M and Q6_K encoders without importance matrix: 21.4 dB / 0.83 and 20.0 dB / 0.67;
  - with the published Q4_K_M denoiser of Sakura-MiniMax-H3-GGUF: the Q8_0, Q6_K and Q5_K_M files of this repository against a Q4_K_M encoder without importance matrix: 23.4 dB / 0.87, 19.5 dB / 0.75 and 21.1 dB / 0.84.
  - with the published Q4_K_M denoiser, against our Q8_0 encoder as reference: the Q5_K_M file 24.3 dB / 0.89, a Q4_K_M encoder without importance matrix 23.4 dB / 0.87, the Q6_K file 22.0 dB / 0.83.
  The encoder quantization visibly changes the result, and the order is not simply by bit width: with one prompt and one seed, the Q5_K_M file came closest to the Q8_0 encoder and the Q6_K file was the farthest of the three; we do not read more into this than that. This Q4_K_M file was not run with the published Q4_K_M denoiser.

## What we changed (license section III.2)

The `.gguf` files are modified files: language weights of an abliterated Qwen3-VL-32B-Instruct, first 50 layers, vision tower taken from the same checkpoint (its model card lists only `attn.o_proj` as modified), converted from safetensors to GGUF and quantized as listed.

## License and use

MiniMax H3 Community License Agreement (`LICENSE`) including its Acceptable Use Policy (Exhibit A). By downloading or using these files you agree to `LICENSE`, including Section V and Exhibit A; these restrictions apply to you (among others: no use outside the Applicable Territory, no use to circumvent safeguards, no exploitation or harm of minors, disclosure of machine-generated content in public environments, no impersonation without consent). Section V.3 forbids using the models or their outputs to improve any other AI model; Section V.5 requires safeguards if you offer generation as a service; commercial use above 20 million US dollars of yearly revenue needs a separate authorization from MiniMax, and commercial products must display "MiniMax H3" (Section IV). The language weights are derived from Qwen3-VL-32B-Instruct through the abliterated checkpoint, both Apache-2.0 (`LICENSE-Apache-2.0`); we distribute these files as MiniMax H3 Works under the MiniMax license as well. Pass `LICENSE`, `LICENSE-Apache-2.0` and `NOTICE` on with any redistribution.

## Credits

- Model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax); encoder layout reference: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3); language model: [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) (Qwen / Alibaba).
- Abliterated checkpoint: [llmfan46](https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic).
- Conversion and inference: [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT); imatrix: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).

---

<details><summary>中文说明 · 樱花 (Simplified Chinese)</summary>

> English above. 本节为上文的中文翻译(Sakura = 樱花 yīnghuā);完整的独立中文版见 [README_zh.md](README_zh.md)。

# Sakura — MiniMax-H3 文本编码器(Qwen3-VL-32B,50 层),去审查版(GGUF)

> 仅限成年人使用;您需对自己生成的内容以及适用于您的法律负责。语言权重来自一个去除了拒绝行为、且在有害提示词上文本生成会崩溃的模型(见下文);这对视频有何改变,我们只对六条非露骨的提示词做了测量(见下文)。不要用它来描绘涉及性或虐待情境的未成年人,也不要在未经同意的情况下制作真实人物的性相关或诽谤性图像。未经您自己的内容审核,请勿将其放入公共服务。

这是 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的 **文本编码器**(Qwen3-VL-32B 的前 50 个语言层加上它的视觉塔,通过 `--llm` 传给视频模型),其语言权重来自 **经消融(去审查)** 的 Qwen3-VL-32B-Instruct,提供 GGUF 格式,用于 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)。属于独立工作,并非 MiniMax、Comfy-Org、Qwen / Alibaba 或 llmfan46 制作、审阅或认可。

## 权重的来源

- MiniMax-H3 内置的文本编码器就是 **未改动的 Qwen3-VL-32B-Instruct**:`MiniMaxAI/MiniMax-H3` 的全部 14 个 `text_encoder` 分片的 sha256 与 `Qwen/Qwen3-VL-32B-Instruct` 的相同。MiniMax-H3 使用的是截断到 50 个语言层、且没有最终语言模型归一化的版本。
- 我们用 **一位社区成员制作的该模型的消融检查点** 构建了编码器([llmfan46 的经消融 Qwen3-VL-32B-Instruct](https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic),Apache-2.0,版本 `c44b949b30d111666a5ed9851c5cd633ed39b070`;两个分片的 sha256 均已与 Hub 核对)。**本仓库的消融不是我们自己做的。**
- 张量被逐个复制到 Comfy-Org H3 编码器文件的布局中(902 个张量;名称、数据类型和形状都与该文件一致;第 50 到 63 层共 14 层、最终归一化和输出头被丢弃),然后用 `sd-cli -M convert` 转换。

## 运行所需文件

| 部分 | 文件 | 来源 |
|---|---|---|
| 文本编码器(任选一个) | `Sakura-MiniMax-H3-TextEncoder-Uncensored-Q4_K_M-14.05GiB.gguf`、`…Q5_K_M-17.00GiB.gguf`、`…Q6_K-20.13GiB.gguf`(三者都带重要性矩阵)或 `…Q8_0-25.60GiB.gguf`;通过 `--llm` 传入 | 本仓库 |
| 去噪器 | 剪枝版 FL2VA 去噪器的一个 GGUF,通过 `--diffusion-model` 传入 | [Sakura-MiniMax-H3-GGUF](https://huggingface.co/webmp3/Sakura-MiniMax-H3-GGUF) |
| VAE | `minimax_h3_video_vae_fp16.safetensors`(`--vae`)、`minimax_h3_audio_vae_fp32.safetensors`(`--audio-vae`) | 同一仓库 |
| 程序 | 支持 MiniMax-H3 的 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/releases) 的 `sd-cli`(我们使用 `master-956-1b0ba10`,Vulkan;文档 `docs/minimax_h3.md`) | 不包含 |

```bash
sd-cli -M vid_gen --diffusion-model <denoiser>.gguf --vae minimax_h3_video_vae_fp16.safetensors --audio-vae minimax_h3_audio_vae_fp32.safetensors --llm Sakura-MiniMax-H3-TextEncoder-Uncensored-Q4_K_M-14.05GiB.gguf -p "A red fox walks through fresh snow in a quiet pine forest at sunrise, soft golden light, cinematic tracking shot, realistic motion, gentle wind in the trees." --cfg-scale 1.0 --steps 8 -W 480 -H 288 --video-frames 22 --fps 24 --diffusion-fa --rng cpu -s 42 -o frames/f_%03d.png
```

**内存:** 在我们的机器上(32 GiB 显存加 31.6 GiB 系统内存),Q8_0 去噪器与 Q8_0 编码器同时使用放不下,并以 `ErrorOutOfDeviceMemory` 失败;与此处文件大小相同的 Q4_K_M 编码器(不带重要性矩阵制作)与我们制作的每个去噪器都能运行(带和不带受保护张量的 Q4_K_M 和 Q5_K_M,以及 Q6_K 和 Q8_0;22 帧、480x288 用时 60 到 94 秒,为实际耗时,含模型加载)。本 Q4_K_M 文件运行过一次,使用的是不带保护的 Q4_K_M 去噪器(57 秒)。本仓库的 Q8_0、Q6_K 和 Q5_K_M 文件各与 Sakura-MiniMax-H3-GGUF 已发布的 Q4_K_M 去噪器运行过一次,都生成了连贯的片段(233、91 和 75 秒)。

## 示例片段

一条提示词,种子 42;22 帧,分辨率 480x288(24 fps 下不到一秒),8 步,使用本 Q4_K_M 编码器与不带保护张量的 Q4_K_M 去噪器(由我们按 Sakura-MiniMax-H3-GGUF 的 Q4_K_M 文件的方式制作,但不含其受保护张量;未发布)。使用已发布的 Q4_K_M 去噪器时画面会不同。只有解码后的画面,没有音频;这是一个小型测试渲染,不代表完整质量。

![Q4_K_M encoder with Q4_K_M denoiser](examples/Sakura-MiniMax-H3-TextEncoder-Uncensored-Q4_K_M-with-denoiser-Q4_K_M.gif)

## 这里的“去审查”具体指什么,以及我们没有测量什么

语言权重来自 llmfan46 的经消融 Qwen3-VL-32B-Instruct(见上文链接)。其作者报告拒绝率为 4/100,而原模型为 99/100,KL 散度为 0.0421(他们的测量;我们没有复现)。

我们自己的检查不能作为对消融模型的有效测量。我们在完整的 64 层模型(Q8_0)上运行了 `mlabonne/harmful_behaviors` 测试划分中保留的 104 条有害提示词(纯聊天、无系统提示词,温度 0;120 个词元,llama.cpp b11524)。未改动的 Qwen3-VL-32B-Instruct(Q8_0,unsloth)拒绝了 104 条中的 104 条。我们自己构建的消融检查点的 Q8_0 版本(由我们用 llama.cpp 的 `convert_hf_to_gguf` 转换,没有模型的聊天模板)给出的大多是不连贯的文本:68 条回答以重复的中文词元开头,另有 11 条包含 15 个或更多字符的数字串,另有 2 条重复同一个短语,只有 23 条可读。因此它的拒绝次数 1/104 并不能说明该模型会回答。据我们所知,原因在检查点而不是我们的转换:我们 Q8_0 的 707 个张量中有 698 个与未改动模型的 Q8_0(unsloth)逐字节相同,而不同的 9 个张量(第 31 到 39 层的 `attn_output`)与该检查点作者发布的 Q8_0 文件逐字节相同。使用普通的 ChatML 提示词(贪心,60 个词元)时,我们的 Q8_0 能连贯地回答无害问题,但对这组中第一条有害提示词只输出一串减号和零,在 Vulkan 和 CPU 上都是如此;未改动的模型会拒绝同一条提示词。因此在有害提示词上,消融模型的文本生成会崩溃而不是回答;我们只用这两条提示词做了这个测试,没有测量发生的频率,也没有测量这对 MiniMax-H3 读取的第 50 层隐藏状态意味着什么。这里的编码器文件是直接由检查点的 safetensors 构建的,而不是由该 GGUF 构建,用它们制作的片段是连贯的。

视频模型读取的是编码器第 50 层的隐藏状态而不是生成的文本;使用本编码器与使用未改动编码器生成的视频的比较见下一节。“去审查”只说明权重的来源。

## 视频:本编码器与未改动编码器的对比

两个编码器都用 Q8_0,去噪器相同(Sakura-MiniMax-H3-GGUF 的 Q4_K_M),种子 42;22 帧,分辨率 480x288;8 步;`--cfg-scale 1.0`,`sd-cli` `master-956-1b0ba10`;未改动的编码器是 unsloth 的 Qwen3-VL-32B-Instruct 的 Q8_0(llama.cpp 格式的 GGUF;`sd-cli` 加载其前 50 层),我们的是本仓库的 Q8_0 文件。每条提示词和每个编码器各一段视频;解码后画面的 PSNR 和 SSIM 为我们的编码器相对未改动编码器,22 帧的平均值:

| 提示词 | PSNR | SSIM |
|---|---:|---:|
| 雪中的红狐狸(上面的示例) | 19.0 dB | 0.72 |
| 厨师在炒锅里翻炒蔬菜 | 13.8 dB | 0.39 |
| 夜晚酒吧外的街头斗殴 | 19.1 dB | 0.70 |
| 士兵在沙袋后给步枪换弹 | 19.6 dB | 0.73 |
| 男人在昏暗的桌旁喝威士忌抽烟 | 32.0 dB | 0.96 |
| 侦探在有粉笔轮廓和血迹地板的犯罪现场 | 28.4 dB | 0.84 |

![Frames 0, 7, 14 and 21 of the twelve videos, unmodified encoder above, this encoder below](examples/Sakura-MiniMax-H3-encoder-vs-unmodified-contact-sheet.png)

我们看到的:十二段视频都是连贯的,并且符合各自的提示词。更换编码器会改变视频,从几乎相同(威士忌提示词)到同一场景的不同构图(厨师提示词);这种变化的幅度与编码器不同量化版本之间的变化处于同一数量级。未改动的编码器在这些提示词中同样生成了所要求的暴力、武器、酒精、香烟和血迹,所以对这六条提示词,我们没有发现未改动编码器阻止了任何内容的情况,因此也没有发现需要本编码器的情况。我们没有测试性内容、更多提示词、其他种子、音频或其他去噪器;每行一条提示词不构成统计。

使用 Q5_K_M 编码器的第二次、更大的对比:unsloth 的未改动 Qwen3-VL-32B-Instruct 的 Q5_K_M 对比本仓库的 Q5_K_M,去噪器和设置相同,8 条提示词(上面的六条加上雨夜街道和吹泡泡的孩子),种子 42 和 43,共 16 对;另加我们的 Q4_K_M 编码器作为参照尺度。未改动编码器相对我们的编码器的 PSNR / SSIM 平均为 24.1 dB / 0.79(范围 12.5 到 34.0 dB);我们的 Q4_K_M 相对我们的 Q5_K_M 平均为 24.8 dB / 0.81(范围 11.7 到 36.5 dB)。在 16 对中的 9 对里,未改动编码器与我们的编码器之间的差异大于我们的 Q4_K_M 与 Q5_K_M 之间的差异。我们查看了种子 42 的 24 段视频各一帧:全部连贯并符合各自的提示词;种子 43 的视频除 PSNR 和 SSIM 外我们没有检查。因此更换编码器对视频的改变与更换编码器的量化版本大致相当,我们没有看到系统性的质量损失,也再次没有发现未改动编码器阻止任何内容的提示词。

![Frame 10 of the seed 42 videos: unmodified Q5_K_M, uncensored Q5_K_M and uncensored Q4_K_M encoder, eight prompts](examples/Sakura-MiniMax-H3-encoder-vs-unmodified-Q5-contact-sheet.png)

## 量化

| 文件 | 大小 | 方式 |
|---|---:|---|
| `…Q8_0-25.60GiB.gguf` | 25.60 GiB | `sd-cli -M convert -m <encoder safetensors> --type q8_0` |
| `…Q6_K-20.13GiB.gguf` | 20.13 GiB | `--type q6_K --imat-in <own imatrix>` |
| `…Q5_K_M-17.00GiB.gguf` | 17.00 GiB | `--type q5_K --imat-in <own imatrix>` |
| `…Q4_K_M-14.05GiB.gguf` | 14.05 GiB | `--type q4_K --imat-in <own imatrix>` |

- **重要性矩阵:** 自行计算,用 `llama-imatrix` 在完整模型(Q8_0)上对 `calib_mix_v1.txt`(英文 / 德文 / 代码混合)的 50 个、每个 512 词元的块做统计,并把张量名称映射到编码器文件第 0 到 49 层的布局。(MarxistLeninist 发布的矩阵中的编码器条目调用计数为 1(去噪器条目:4),且使用 `sd-cli -M convert -m` 不会应用的名称前缀;我们没有用它来处理编码器。)用它制作的文件与普通 Q4_K_M 不同(sha256 不同);我们没有测量质量提升。该矩阵是在同一个 Q8_0 构建上计算的,而该构建对有害提示词的回答大多不连贯(见上文);我们没有检查它的统计是否具有代表性。
- **对视频的影响**(一条提示词,种子 42;22 帧,分辨率 480x288;8 步;解码后画面的 PSNR / SSIM,22 帧的平均值):
  - 使用不带保护张量的 Q4_K_M 去噪器(由我们制作,未发布):本 Q4_K_M 编码器对比不带重要性矩阵的 Q4_K_M 编码器:20.5 dB / 0.70;同样设置下,不带重要性矩阵的 Q4_K_M 编码器对比不带重要性矩阵的 Q5_K_M 和 Q6_K 编码器:21.4 dB / 0.83 和 20.0 dB / 0.67;
  - 使用 Sakura-MiniMax-H3-GGUF 已发布的 Q4_K_M 去噪器:本仓库的 Q8_0、Q6_K 和 Q5_K_M 文件对比不带重要性矩阵的 Q4_K_M 编码器:23.4 dB / 0.87、19.5 dB / 0.75 和 21.1 dB / 0.84。
  - 使用已发布的 Q4_K_M 去噪器,以我们的 Q8_0 编码器为参考:Q5_K_M 文件 24.3 dB / 0.89,不带重要性矩阵的 Q4_K_M 编码器 23.4 dB / 0.87,Q6_K 文件 22.0 dB / 0.83。
  编码器的量化会明显改变结果,而且顺序并不单纯按位宽排列:在一条提示词、一个种子下,Q5_K_M 文件最接近 Q8_0 编码器,三者中 Q6_K 文件最远;我们不对此做更多解读。本 Q4_K_M 文件没有与已发布的 Q4_K_M 去噪器一起运行过。

## 我们做了哪些修改(许可第 III.2 条)

`.gguf` 文件是修改过的文件:经消融的 Qwen3-VL-32B-Instruct 的语言权重,前 50 层,视觉塔取自同一检查点(其模型卡只列出 `attn.o_proj` 被修改),从 safetensors 转为 GGUF 并按上表量化。

## 许可与使用

MiniMax H3 社区许可协议(`LICENSE`),包括其可接受使用政策(附件 A)。下载或使用这些文件即表示您同意 `LICENSE`,包括第 V 条和附件 A;这些限制对您有约束力(其中包括不得在适用地域之外使用、不得用于绕过防护措施、不得利用或伤害未成年人、公开环境中的机器生成内容必须披露、未经同意不得冒充他人)。第 V.3 条禁止使用这些模型或其输出来改进任何其他 AI 模型;第 V.5 条要求在提供生成服务时落实防护措施;年收入超过 20 million 美元的商业使用需要另行取得 MiniMax 的授权,商业产品必须显示“MiniMax H3”(第 IV 条)。语言权重通过消融检查点衍生自 Qwen3-VL-32B-Instruct,二者均为 Apache-2.0(`LICENSE-Apache-2.0`);我们同样把这些文件作为 MiniMax H3 作品按 MiniMax 许可分发。再分发时必须一并提供 `LICENSE`、`LICENSE-Apache-2.0` 和 `NOTICE`。

## 致谢

- 模型:[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)(MiniMax);编码器布局参考:[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3);语言模型:[Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)(Qwen / Alibaba)。
- 消融检查点:[llmfan46](https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic)。
- 转换与推理:[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)(MIT);imatrix:[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)(MIT)。

</details>
