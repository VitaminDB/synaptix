# MiniMax-AI__awesome-minimax-h3-integration

Source: https://github.com/MiniMax-AI/awesome-minimax-h3-integration
Fetched: 2026-11-28

Description: MiniMax H3 Integrations, a community index of models, tools and projects for MiniMax H3 and MiniMax-H3-Image.
Stars: 601  Updated: 2026-11-12T15:47:55Z



---

## FILE: README.md (https://github.com/MiniMax-AI/awesome-minimax-h3-integration/blob/HEAD/README.md)

<p align="center">
  <a href="https://github.com/MiniMax-AI/MiniMax-H3"><img src="https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/assets/minimax-h3-header.gif" alt="MiniMax H3"></a>
</p>

# MiniMax H3 Integrations

A community-maintained index of models, tools and projects for [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) and [MiniMax-H3-Image](https://github.com/MiniMax-AI/MiniMax-H3-Image).

English | [简体中文](README_cn.md)

## Contents

- [MiniMax H3](#minimax-h3)
  - [Official](#h3-official)
  - [Quantizations](#h3-quantizations)
  - [Components](#h3-components)
  - [Acceleration and distillation](#h3-acceleration)
  - [LoRAs and fine-tunes](#h3-loras)
  - [Runtimes](#h3-runtimes)
  - [Apps and studios](#h3-apps)
  - [API integrations](#h3-api)
  - [ComfyUI nodes](#h3-comfyui)
  - [Workflows](#h3-workflows)
  - [Prompting and agent skills](#h3-prompting)
  - [Training](#h3-training)
  - [Serving and hosting](#h3-serving)
  - [Guides](#h3-guides)
- [MiniMax-H3-Image](#minimax-h3-image)
  - [Official](#img-official)
  - [Quantizations and conversions](#img-quantizations)
  - [Distillation and adapters](#img-adapters)
  - [Runtimes](#img-runtimes)
  - [Apps](#img-apps)
  - [API integrations](#img-api)
  - [ComfyUI nodes and workflows](#img-comfyui)
  - [Training](#img-training)
  - [Prompting](#img-prompting)
  - [Benchmarks](#img-benchmarks)
- [Contributing](#contributing)

## MiniMax H3

MiniMax H3 is MiniMax's open-weight model for video with native stereo audio: it reads text, images, video and audio as context and generates clips of 4 to 15 seconds, up to 2K. It comes in two variants: FL2VA works from text and optional first and last frames, and Ref2VA takes up to 9 images, 3 videos and 3 audio clips as references.

<a id="h3-official"></a><a id="guides"></a><a id="checkpoints"></a>

### Official

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-H3">MiniMax-H3</a></td><td>Official repository: reference implementation and prompt guides.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMaxAI/MiniMax-H3</a></td><td>Original weights for FL2VA (first and last frame) and Ref2VA (reference), with text encoder, video VAE and audio VAE.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630433fea362e7e8b51f5e2b/bsDc8lR72KhKvm-wXxYsv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Comfy-Org/MiniMax-H3">Comfy-Org/MiniMax-H3</a></td><td>ComfyUI single-file weights, including AdaLN-pruned BF16, FP8 and INT8 builds.</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/video/minimax/minimax-h3">ComfyUI tutorial</a></td><td>Official ComfyUI guide for H3, from the <a href="https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui">day-0 announcement</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json">ComfyUI workflow templates</a></td><td>Official T2V, <a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json">I2V</a> and <a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json">R2V</a> templates that ship with ComfyUI.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md">Video prompt writing guides</a></td><td>Official guides for base (FL2VA) and <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md">reference (Ref2VA)</a> prompts.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://platform.minimax.io/docs/api-reference/video-generation-v2-create">MiniMax API</a></td><td>Official video generation API; for mainland China, use <a href="https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create">platform.minimaxi.com</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://hailuoai.video/tools/minimax-h3">Hailuo AI</a></td><td>Use H3 in the browser or the <a href="https://hub.minimax.io/">desktop app</a>; for mainland China, use <a href="https://hailuoai.com/">hailuoai.com</a> and <a href="https://hub.minimaxi.com/">hub.minimaxi.com</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/partner-nodes/minimax/minimax-h3">ComfyUI API nodes</a></td><td>Built-in nodes for text, first and last frame and reference to video that run H3 and H3 Max through your Comfy account.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/cli">MiniMax CLI</a></td><td>Official command-line tool for the MiniMax API, including H3 video generation.</td></tr>
</table>

<a id="h3-quantizations"></a><a id="quants"></a><a id="gguf"></a>

### Quantizations

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/unsloth/MiniMax-H3-GGUF">unsloth/MiniMax-H3-GGUF</a></td><td>Pruned FL2VA and Ref2VA GGUF from Q2_K (6.26 GiB) to Q8_0, plus Qwen3-VL text-encoder GGUF.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/Yy9XdzixLpDyVyN4fUvP5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants">DmitryDB/MiniMax-H3-ComfyUI-Quants</a></td><td>INT8 ConvRot and NVFP4 builds of FL2VA and Ref2VA from 10.86 to 21.91 GiB, plus VAE files.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/Yy9XdzixLpDyVyN4fUvP5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV">DmitryDB/MiniMax-H3-DynTime-sQKV</a></td><td>INT8 DT-sQKV builds that need the repository's ComfyUI runtime patch.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63150d9f8be84e1ab412890e/slVOxsRI070drG6kICymD.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot">rockerBOO/minimax-h3-nvfp4-convrot</a></td><td>NVFP4 FL2VA and Ref2VA for Blackwell GPUs, recommended with INT8 ConvRot; experimental INT4 ConvRot FL2VA for other GPUs.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot">Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot</a></td><td>Pruned NVFP4 (11.67 GiB), INT8 ConvRot and mixed INT4/INT8 ConvRot builds.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6357da9b68659b265eb84aab/8PEgQsPKTJVQQD2_87mP9.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot">tsolful/Minimax_H3_INT4MixedConvRot</a></td><td>Pruned mixed INT4/INT8 ConvRot builds of FL2VA and Ref2VA.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/b3c31e18f3b7cc1c5c059058221195dc.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot">Merserk/MiniMax-H3-INT4-ConvRot</a></td><td>INT4 ConvRot, 16.67 GiB full and 10.56 GiB pruned.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/357a2cb738d2a9ad5c01dd6296792c22.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot">AX1Y2JP/MiniMax-H3-W4A8-ConvRot</a></td><td>Pruned W4A8 ConvRot DiT (11.68 GiB) and a W4A8 text encoder.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3-experimental">Kijai/MiniMax-H3-experimental</a></td><td>Experimental W4A8 mixed DiT builds and an INT8 ConvRot video VAE.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62c3f1dd73c3ba0c801033ce/3STHHoSpTLGI-Alcpf7Cg.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI">Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI</a></td><td>Pruned W4A8 ConvRot DiT with a matching quantized text encoder.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/644717ea27a3e9e0b7c7207a/aNn8akdZfegf46fkZ4R9h.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn">rzgar/minimax_h3_fl2va_fp8_e4m3fn</a></td><td>FP8 E4M3FN, MXFP8, and FP8 with FP16 attention builds of FL2VA.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/6701ed70ea8cd899201c524ab3ccac8c.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/abakanai/Minimax_h3_hybrid">abakanai/Minimax_h3_hybrid</a></td><td>Hybrid NVFP4 Ref2VA builds for Blackwell GPUs, 14.03 and 16.38 GiB.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672c8f0070734f59f146b73c/5lGMq-_Patlincxq97jf5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4">DiffSynth-Studio/MiniMax-H3-NF4</a></td><td>NF4 DiT, text encoder and VAEs for DiffSynth-Studio, from 8 GB of VRAM.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4">WaveCut/MiniMax-H3-OrbitQuant-W4A4</a></td><td>W4A4 build with a quantized text encoder; needs the ComfyUI-OrbitQuant node.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-GGUF">Abiray/MiniMax-H3-GGUF</a></td><td>Full FL2VA and Ref2VA GGUF from Q3_K_S to Q8_0, with text-encoder quants and a workflow.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF">Abiray/MiniMax-H3-Pruned-GGUF</a></td><td>Pruned FL2VA GGUF, Q3_K_M and Q8_0.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68761990332d15464ccc8dee/MaE3RLtzT8Q3tQ5I-n38s.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/realrebelai/MiniMax-H3_GGUFs">realrebelai/MiniMax-H3_GGUFs</a></td><td>Q3_K_M and Q4_K_M GGUF for FL2VA and Ref2VA, plus Q2_K and Q4_K_M text encoders; the Q2_K one is 7.91 GiB.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67b808c770117114270d902e/dLdwkNZowSMsqTZ3Znfg2.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF">vantagewithai/MiniMax-H3-comfyUI-GGUF</a></td><td>Full FL2VA and Ref2VA GGUF from Q3_K_M to Q8_0 for ComfyUI.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/2bfb8e5718e39dd319e9320abab557c1.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF">MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF</a></td><td>IQ1_S (3.78 GiB) and IQ1_M builds, the smallest published DiT.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/638a3f5a8d7cce863e70a91b/Z_KI0kdABbpRkXV51JpOG.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/molbal/MiniMax-H3-GGUF">molbal/MiniMax-H3-GGUF</a></td><td>Pruned Ref2VA GGUF, including hand-tuned mixed layouts.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/92b2fbafe275bf1c126d5a687d6a7f2f.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/leejet/MiniMax-H3-GGUF">leejet/MiniMax-H3-GGUF</a></td><td>Pruned FL2VA GGUF, Q2_K and Q4_K_M.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF">joeygambino/MiniMax-H3-curve-GGUF</a></td><td>Pruned FL2VA and Ref2VA GGUF in Q4_0, Q5_1 and Q8_0; needs the H3-Multishot patch for ComfyUI-GGUF.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68a41bb168b64003eab9cfa4/Gq5_H6FGLr1k3hSsshdfa.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4">jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4</a></td><td>NVFP4 builds of the smhfacct FL2VA and Ref2VA hybrid in four block ranges, 11.67 GiB each, for Blackwell GPUs.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/cac9874103e7c159dc5eb6ec977f3ceb.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4">lilcheaty/MiniMax-H3-NVFP4</a></td><td>Pruned NVFP4 FL2VA and Ref2VA at 11.67 GiB each, plus full and mixed Ref2VA builds, for Blackwell GPUs.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/42b9a338c1a323b57ce3b506de76f8ae.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DeepBeepMeep/MiniMax-H3">DeepBeepMeep/MiniMax-H3</a></td><td>Weights WanGP downloads for the H3 family: FL2VA, Ref2VA, FastH3 Omni, HyperFlow Flash, Fun ControlNet and H3-Image, in BF16 and INT8.</td></tr>
</table>

<a id="h3-components"></a><a id="components"></a><a id="components-vae"></a><a id="tae"></a><a id="imagevae"></a><a id="refpatch"></a><a id="funcontrol"></a><a id="text-encoder"></a>

### Components

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1650342890988-623c6253389748c9f72ca287.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union">alibaba-pai/MiniMax-H3-Fun-Controlnet-Union</a></td><td>One control branch for Canny, depth, HED, MLSD and pose, plus video inpainting.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/NCer_WlbHMkH1ZU8Q_I8S.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lihaoyun6/MiniMax-H3-Ref-Patch">lihaoyun6/MiniMax-H3-Ref-Patch</a></td><td>148 MiB patch that lets FL2VA behave more like Ref2VA; needs its ComfyUI node.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3-TAE">Kijai/MiniMax-H3-TAE</a></td><td>9 MiB tiny autoencoder for previews in ComfyUI-KJNodes.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/2a4082bfe6d2d4d206dbd655197388f0.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE">Mamad8/MiniMax-H3-Image-VAE</a></td><td>Experimental single-frame VAE for still images from the video model; keep the original VAE for video.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67c576e958e13035e6e4b0d5/mECPgQ0bhniFavYJPDpV4.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler">LBH-123-AI/Minimax_h3_latent_Upscaler</a></td><td>Learned latent upscaler weights, 1.0 to 4.0x, for its ComfyUI node.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62b4b5beb25cb80fcf278354/SIddx3hYu5rWXA4-O3Oaj.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K">Open-H3-Regenerate-2K</a></td><td>NVIDIA's three-step refiner that takes a low-resolution H3 draft to 2560x1440, with <a href="https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K">code</a>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6767c79846bb6707b92e5fb2/lqjSqaUQW3WEMVUG6KFu_.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ZawShiShawn/Gestura-MiniMax-H3-VAE-LiteRT">ZawShiShawn/Gestura-MiniMax-H3-VAE-LiteRT</a></td><td>H3 VAE in LiteRT format for Gestura, an Android app for on-device video.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/7ddf7fcf26e8da8d32da7a3c4e73af52.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE">speach1sdef178/MiniMax-H3-X2-Detail-VAE</a></td><td>Video VAE that decodes at twice the resolution; needs the fast VAE node from ComfyUI-MiniMaxH3_LatentUpscaler.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62b4b5beb25cb80fcf278354/SIddx3hYu5rWXA4-O3Oaj.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC">Efficient-Large-Model/Open-H3-Regenerate-2K-IC</a></td><td>In-context LoRA version of NVIDIA's refiner that redraws a 768p draft at 1080p, 2K or 4K through Ref2VA.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF">joeygambino/MiniMax-H3-encoder-GGUF</a></td><td>GGUF builds of the Qwen3-VL-32B text encoder for ComfyUI.</td></tr>
</table>

<a id="h3-acceleration"></a><a id="speed"></a><a id="turbo"></a><a id="pdd"></a><a id="fasth3"></a><a id="fasth3-4-step-preview-8-step-v2--trim"></a><a id="fasth3-omni"></a><a id="pika-h3-flex"></a><a id="ptd"></a><a id="dmad"></a><a id="dmad-four-step-audio-video-distillation"></a><a id="vdn"></a><a id="solh3"></a><a id="vcattention"></a>

### Acceleration and distillation

<table>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/Minimax-H3-Turbo">Minimax-H3-Turbo</a></td><td>LightX2V's 4- and 8-step distillation LoRAs, with <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo">weights on Hugging Face</a>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63783918d925e7dad20748a3/SIMpPx2nIKyEF7Mx9E3hC.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora">larryvrh/MiniMax-H3-Turbo-Lora</a></td><td>Community 4-step Turbo re-bakes; <code>v4_step600_ema</code> is the usual default.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63473b59e5c0717e6737b872/e0rwJa886MD1hmdeFr9Wh.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI">drbaph/MiniMax-H3-Turbo-Lora-ComfyUI</a></td><td>Turbo LoRAs converted for the pruned ComfyUI checkpoints, including rank-21 resizes.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI">Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI</a></td><td>Turbo LoRAs for the pruned ComfyUI checkpoints.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3_comfy">Kijai/MiniMax-H3_comfy</a></td><td>LightX2V Turbo LoRA in ComfyUI format, full and rank-21.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/637745113a63a2983ffbde13/wZKoPSWQKpe2JpFsX3nLa.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers">InstantX/MiniMax-H3-Turbo-Lora-Diffusers</a></td><td>4-step Turbo LoRA in diffusers format.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6864d1f250596cb028f66be3/TOr9t2xU3PJFjSXGQvTYK.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joyfox/MiniMax-H3-Turbo">joyfox/MiniMax-H3-Turbo</a></td><td>4-step FL2VA LoRA.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/674ef761b783a2a16788d7fe/cxPCVoNPpnehsDwy3bg14.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA">Tutu 20-to-8 NFE LoRA</a></td><td>Audio-video LoRA that cuts 20 evaluations to 8.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp">t8star/minimax-h3-4step-turbo-loras-comfyui-exp</a></td><td>Experimental INT8 ConvRot Turbo LoRAs for ComfyUI.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/693c433f38397e97c1c9fb78/VNwN7gvr2RGpxeBUIHNqT.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf">infosave/MiniMax-H3-Turbo-cmf</a></td><td>Turbo checkpoints in CMF format, Q4TP and Q2TP.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1650342890988-623c6253389748c9f72ca287.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs">alibaba-pai/MiniMax-H3-Acc-LoRAs</a></td><td>PDD 8-step acceleration LoRAs for FL2VA and Ref2VA, loaded natively by current ComfyUI.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/1af3b10983f16ee56a7ad7b48c1e5df7.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/aptech0081/MiniMax-H3-PDD-ComfyUI">aptech0081/MiniMax-H3-PDD-ComfyUI</a></td><td>PDD LoRAs converted for ComfyUI, plus INT8 Ref2VA checkpoints with PDD baked in.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy">t8star/Minimax-H3-Super-Acceleration-Comfy</a></td><td>PDD LoRAs repacked for the T8 audio nodes.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/ce52788c884be3d7654e823007579f5a.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8">1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8</a></td><td>SVDQuant FP4 build with the PDD 8-step LoRA folded in; an <a href="https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-int4-pdd8">INT4 build</a> is also available.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree">FastH3 4-step Preview</a></td><td>FastVideo's data-free 4-step checkpoint with sparse attention; also as a <a href="https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA">LoRA</a>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2">FastH3 8-Step V2</a></td><td>8-step text-to-audio-video checkpoint with 80% sparse attention.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4">FastH3 Trim NVFP4</a></td><td>Experimental 8-step model with 42 of 50 blocks, 11.1 GiB in NVFP4 for Blackwell and DGX Spark.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1">FastH3-Omni 8-Step</a></td><td>8-step reference-conditioned video and audio from image, video and audio references.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy">FastVideo/FastVideo-FastH3-Comfy</a></td><td>FastH3 8-Step V2 repackaged for ComfyUI.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68761990332d15464ccc8dee/MaE3RLtzT8Q3tQ5I-n38s.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/realrebelai/FastH3_GGUFs">realrebelai/FastH3_GGUFs</a></td><td>GGUF conversions of FastH3 4-step.</td></tr>
<tr><td width="56"><img src="https://github.com/Pika-Labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Pika-Labs/Pika-H3-Flex">Pika H3 Flex</a></td><td>Pika's six-step FL2VA and Ref2VA adapters for native 480p and 768p video, with CLI, diffusers and ComfyUI examples.</td></tr>
<tr><td width="56"><img src="https://github.com/vita-epfl.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vita-epfl/PTD-inference/tree/release/comfyui-inference">PTD</a></td><td>EPFL's four- and six-step trajectory distillation, with <a href="https://huggingface.co/epfl-vita/PTD-weights">weights</a> and ComfyUI integration.</td></tr>
<tr><td width="56"><img src="https://github.com/Yzmblog.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Yzmblog/DMAD">DMAD</a></td><td>Four- and eight-step text-to-audio-video LoRAs with <a href="https://huggingface.co/ZhengmingYu/DMAD">weights</a>, workflows and training code.</td></tr>
<tr><td width="56"><img src="https://github.com/OpenVDN.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenVDN/vdn-minimax-h3">VDN</a></td><td>Video Delta Net: hybrid linear and softmax attention for H3, 2.6 to 2.9x faster per step, with <a href="https://huggingface.co/OpenVDN/vdn-minimax-h3">weights</a> and training code.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63473b59e5c0717e6737b872/e0rwJa886MD1hmdeFr9Wh.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui">drbaph/vdn-minimax-h3-int8-convrot-comfyui</a></td><td>INT8 ConvRot VDN in ComfyUI layout.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy">t8star/Vdn-Minimax-H3-Comfy</a></td><td>Ready-to-use VDN model bundle for the T8 nodes.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/66213d7f118a0a5cef880b5e/hr3ia85xax2nyW0OO0F7D.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA">RAVEN Streaming LoRA</a></td><td>4-step preview that turns H3 into a causal streaming generator, run through the <a href="https://github.com/mvp-ai-lab/RAVEN">RAVEN</a> code.</td></tr>
<tr><td width="56"><img src="https://www.gravatar.com/avatar/7b0abe8d2e4aff190133f25f13729f6b?d=retro&size=100" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot">MATLOWAI/minimax-h3-fused-turbo-int8-convrot</a></td><td>One 21 GB INT8 ConvRot file with the Turbo and motion LoRAs folded in, for 4-step text-to-video, image-to-video and Ref2VA in ComfyUI.</td></tr>
<tr><td width="56"><img src="https://github.com/Video-Rebirth.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Video-Rebirth/hyperflow-h3-flash">HyperFlow H3 Flash</a></td><td>Video Rebirth's 13.7B compressed H3 for video with stereo audio in 4 to 8 steps, INT8 and GGUF <a href="https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash">weights</a>; also an <a href="https://huggingface.co/VideoRebirthAI/hyperflow">8-step LoRA</a> for full H3.</td></tr>
<tr><td width="56"><img src="https://github.com/TaoLiveAIGC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TaoLiveAIGC/TaoMate-H3">TaoMate-H3</a></td><td>Three-step streaming generation for H3, with <a href="https://huggingface.co/TaoLiveAIGC/TaoMate-H3">weights</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/thu-ml.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thu-ml/TurboDiffusion">TurboDiffusion</a></td><td>W8A8 kernels, sampler and multi-GPU runtime for H3 FL2VA, with TurboH3-FL2VA-I2VA weights.</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/LongLive">LongLive</a></td><td>NVIDIA's LongLive-Plug distillation, with <a href="https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step">few-step</a> and <a href="https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-cfg">CFG</a> adapters for H3.</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/FastGen">FastGen</a></td><td>NVIDIA's distillation framework, with H3 among its video-with-audio models.</td></tr>
<tr><td width="56"><img src="https://github.com/vipshop.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vipshop/cache-dit">cache-dit</a></td><td>Caching and parallel inference for diffusers, covering H3 T2VA, FL2VA and Ref2VA.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6568921511b2bbd6c2e9841c/v-pfaTZmvAiLeBjS2-eDb.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview">Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview</a></td><td>Learned sparse attention keeping about 10% of tiles in every mode, plus a tuned <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview">R2VA predictor</a>; runs in ComfyUI or Miowtion.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680de13385293771bc57400b/qnjz3avU_cKzRZA2aGjuZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA">lightx2v/Minimax-h3-Turbo-SLA</a></td><td>4-step FL2VA LoRA with 85% sparse-linear attention, about 2.5x faster on an RTX 5090 in LightX2V, with a ComfyUI build.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6ad5e8ef64cf8fd69bf5f843/28JY9Lma4ZOqJ8GgMZduD.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs">taobao-content-ai/MiniMax-H3-PTD-LoRAs</a></td><td>Per-step PTD LoRAs that run Ref2VA in 4 or 8 steps, with diffusers code and ComfyUI nodes that need ComfyUI-PTD and the unpruned Ref2VA model.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/f72b4505394ad4d2258dbba49ade6e85.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/pdmd2026/pdmd_4NFE_lora">pdmd2026/pdmd_4NFE_lora</a></td><td>Rank-128 LoRA for 4-step generation, distilled with projected distribution matching, in diffusers key format.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a550d1dc84594e3ada10828/BLGmI3tqOxQmO1ti1rxUH.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF">ChrisColeTech/minimax-h3-turbo-GGUF</a></td><td>Pruned FL2VA and Ref2VA GGUF with the LightX2V Turbo LoRA baked in, plus 4B and 8B text encoders for ComfyUI-ClipProj.</td></tr>
</table>

<a id="h3-loras"></a><a id="lora"></a>

### LoRAs and fine-tunes

<table>
<tr><td width="56"><img src="https://huggingface.co/avatars/a0fe0c3e0ee676535c5cae05e05aacb7.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/matlod/minimax-h3-turnaround">minimax-h3-turnaround</a></td><td>Five rotated views of one subject from a single reference image.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6380ebb8471a4550ff255c62/JLWd_lQkzQN81FVIva1w1.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA">Realism - People</a></td><td>fal's LoRA for natural-looking people in everyday scenes.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/a7bf1af19e638162579666509c490220.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime">Looping Sketch Anime</a></td><td>Hand-drawn 2D anime look, made to loop.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/642b468deb31218a5f1e69f7/FWS9Yc-nckXsKBr5l-pij.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/TenStrip/Krea2-H3-Style-Lora">Krea2 Style</a></td><td>Experimental style LoRA toward Krea 2's look, for text-to-video only.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/39c999386889182c4ac74c6c612f8bcf.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models">smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models</a></td><td>FL2VA base with Ref2VA AdaLN blocks merged in, in four block ranges.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/minimax_h3_ref2va_patchin_hf102">t8star/minimax_h3_ref2va_patchin_hf102</a></td><td>Experimental Ref2VA weight modification that boosts one high-frequency patch by 2%.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6333948fea2dcff925248fca/Bef-LRCRhI3zZzt-lY8mv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Viggle/Viggle-Remake">Viggle Remake</a></td><td>Two Viggle LoRAs that recast people or change objects, backgrounds and weather; <a href="https://huggingface.co/Viggle/Viggle-Animate">Viggle Animate</a> recasts from one repainted frame.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/653772fed7616b72795db894/1GPjjeZP_RGRl1pYgFdZc.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/PrunaAI/P-H3-Image">P-H3-Image</a></td><td>PrunaAI's fine-tune of the video model for text-to-image and image editing.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/zYUGNDVFyLncD1OHov1pB.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI">Alissonerdx/Minimax-H3-ComfyUI</a></td><td>Experimental LoRAs for motion transfer, character and head swap, style transfer, VFX edits and live wallpapers.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/232ea307b6e3f75cffe2568a99541dad.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA">360 Orbit</a></td><td>FL2VA LoRA that turns one photo into a frozen-time 360° orbit.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/n2bexa-LzvGqmxyOGaonu.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity">Minimax-h3 Singularity</a></td><td>Fine-tuned merge of the FL2VA and Ref2VA checkpoints for text-, image-, reference- and video-to-video in ComfyUI; also a <a href="https://huggingface.co/WAS/Minimax-h3_Singularity_v2_pruned">pruned build</a>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65bb837dbfb878f46c77de4c/23gZ_lBEwyoqjexFy9QLD.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp">prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp</a></td><td>Experimental rank-16 LoRA for close-up faces with subtle expressions and micro-movements; prompts start with <code>Facial Realism</code>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6928a3ca735961e729669bb3/RrdhFVLZ35EUaBSoYb4IE.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LiconStudio/Minimax-H3-VBVR">LiconStudio/Minimax-H3-VBVR</a></td><td>Image-to-video LoRA trained on rebuilt VBVR data that reduces duplicated objects and characters passing through objects.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/635dd6cd4fabde0df74aeae6/zDmJWNx95QODS2mbAAxNZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alvdansen/h3-keyframe-animation">alvdansen/h3-keyframe-animation</a></td><td>Three LoRAs that hold a hand-drawn animation cadence from one or two drawings; free for individuals and small businesses, gated.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/nJsOmzl5Jf72CxaaHjoiF.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/XGENlabs/XGEN-JING">XGENlabs/XGEN-JING</a></td><td>Four-step first-person model on Ref2VA that makes video and audio from keyboard camera moves, reference images and text.</td></tr>
<tr><td width="56"><img src="https://github.com/Junchao-cs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Junchao-cs/SolarWM">SolarWM</a></td><td>World model whose H3 release streams camera-controlled video from a first frame and a prompt, with training code and the 4-step SolarRenderer.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a6748afef317ed5adb30050/Ca36mpBOmdmrgVPxtMafQ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora">Wushu Action LoRA</a></td><td>Martial-arts action LoRA trained on 924 wushu clips; trigger word <code>wushu_action</code>.</td></tr>
</table>

<a id="h3-runtimes"></a><a id="engines"></a>

### Runtimes

<table>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/ComfyUI">ComfyUI</a></td><td>Native support since day 0, with INT8 in mainline.</td></tr>
<tr><td width="56"><img src="https://github.com/modelscope.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a></td><td><code>MiniMaxH3Pipeline</code> with NF4 inference from 8 GB of VRAM.</td></tr>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/LightX2V">LightX2V</a></td><td>Inference with parallelism, quantized DiT and feature caching; home of the Turbo LoRAs.</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3">Sana Sol-Engine</a></td><td>NVIDIA's per-GPU optimized runtimes, 3.55 to 4.52x faster than baseline, and the 4-step Sol-H3 runtime for up to 8 GPUs.</td></tr>
<tr><td width="56"><img src="https://github.com/hao-ai-lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hao-ai-lab/FastVideo">FastVideo</a></td><td>Sparse-attention and step-distillation framework behind FastH3.</td></tr>
<tr><td width="56"><img src="https://github.com/aigc-apps.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/aigc-apps/VideoX-Fun">VideoX-Fun</a></td><td>Training and inference code behind the Fun ControlNet-Union and the PDD LoRAs.</td></tr>
<tr><td width="56"><img src="https://github.com/antirez.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/antirez/h3.c">h3.c</a></td><td>Native Metal engine for Apple silicon: text-to-video, first and last frame, and Ref2VA.</td></tr>
<tr><td width="56"><img src="https://github.com/drawthingsai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/drawthingsai/draw-things-community">Draw Things</a></td><td>Swift and Metal engine behind the Draw Things app: FL2VA, Ref2VA and FastH3 Omni on Apple silicon.</td></tr>
<tr><td width="56"><img src="https://github.com/ddalcu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ddalcu/mlx-serve">mlx-serve</a></td><td>Apple silicon app and server with FastH3 Omni reference video in 8 steps.</td></tr>
<tr><td width="56"><img src="https://github.com/komikndr.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/komikndr/raylight">Raylight</a></td><td>Multi-GPU ComfyUI runtime that splits sequences across GPUs.</td></tr>
<tr><td width="56"><img src="https://github.com/unslothai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unslothai/unsloth">Unsloth</a></td><td>Lists MiniMax-H3 among the models it can run and train.</td></tr>
<tr><td width="56"><img src="https://github.com/fluxmatrix.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/fluxmatrix/h3-flash">h3-flash</a></td><td>Inference scripts and a web demo for 8x B200 with sequence parallelism and fused kernels.</td></tr>
<tr><td width="56"><img src="https://github.com/huggingface.png?size=96" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/blog/hf-torchtpu">TorchTPU + Diffusers</a></td><td>H3 on Cloud TPU v6e-4 with native PyTorch, <code>torch.compile</code> and tensor parallelism.</td></tr>
<tr><td width="56"><img src="https://github.com/leejet.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/leejet/stable-diffusion.cpp">stable-diffusion.cpp</a></td><td>C/C++ inference engine that runs H3 and its GGUF builds.</td></tr>
<tr><td width="56"><img src="https://github.com/mudler.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mudler/LocalAI">LocalAI</a></td><td>Local OpenAI-compatible server; version 4.9.0 added H3 video with audio.</td></tr>
<tr><td width="56"><img src="https://github.com/ztxz16.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ztxz16/fastllm">fastllm</a></td><td>Native H3 text-to-audio-video with an HTTP demo.</td></tr>
<tr><td width="56"><img src="https://github.com/0xShug0.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/0xShug0/audio.cpp">audio.cpp</a></td><td>C++ engine that runs H3 text-to-audio-video from Q4_K GGUF, optionally with an INT8 ConvRot DiT.</td></tr>
<tr><td width="56"><img src="https://github.com/1CatAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1CatAI/1Cat-vLLM">1Cat-vLLM</a></td><td>vLLM fork for V100 with native H3 audio-video generation.</td></tr>
<tr><td width="56"><img src="https://github.com/gufo-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gufo-org/gufo">gufo</a></td><td>Inference engine for AMD Strix Halo that runs H3 video with sound.</td></tr>
<tr><td width="56"><img src="https://github.com/localai-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/localai-org/vllm.cpp">vllm.cpp</a></td><td>Python-free C++ engine that serves H3 from Q4_K_M GGUF through <code>/v1/videos</code>, measured at 34.6 s per step on one Jetson Thor.</td></tr>
</table>

<a id="h3-apps"></a>

### Apps and studios

<table>
<tr><td width="56"><img src="https://github.com/agent-next.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agent-next/video-agent">open-video</a></td><td>Runs H3 on your own GPU with <code>install</code>, <code>pull</code> and <code>run</code>, plus a skill that lets coding agents make video.</td></tr>
<tr><td width="56"><img src="https://github.com/FlashML-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlashML-org/FreeVideo">FreeVideo</a></td><td>Local ComfyUI workspace built on VDN-H3, with an 8 GB VRAM path.</td></tr>
<tr><td width="56"><img src="https://github.com/D3voz.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/D3voz/openjutsu">Openjutsu</a></td><td>Character and object swaps in video with SAM 3.1 tracking.</td></tr>
<tr><td width="56"><img src="https://github.com/PxTicks.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/PxTicks/h3-motion">H3 Motion</a></td><td>Editor for keyframed subject motion and masked region regeneration.</td></tr>
<tr><td width="56"><img src="https://github.com/zxcHolmes.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zxcHolmes/minimax-h3-studio">MiniMax-H3 Studio</a></td><td>Self-hosted web and API studio with a persistent queue, including FastH3 and Pika H3 Flex.</td></tr>
<tr><td width="56"><img src="https://github.com/Klimentios.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Klimentios/talemondo-studio">TaleMondo Studio</a></td><td>Noncommercial Mac workspace for film production with local H3.</td></tr>
<tr><td width="56"><img src="https://github.com/thezolon.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thezolon/tiiny-studio">Tiiny Studio</a></td><td>Desktop app for Tiiny Pocket hardware with H3-Turbo text- and image-to-video.</td></tr>
<tr><td width="56"><img src="https://github.com/slogonomo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/slogonomo/lf-studio">LF Studio</a></td><td>Web studio for NVIDIA's Sol-H3 pipeline on one DGX Spark or one RTX 5090; formerly sol-h3-studio.</td></tr>
<tr><td width="56"><img src="https://github.com/omnichar.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/omnichar/OmniChar">Omnichar Studio</a></td><td>Character studio that keeps one face, body, outfit and voice across shots; generates with H3 and trains H3 LoRAs locally. Formerly Inline Studio.</td></tr>
<tr><td width="56"><img src="https://github.com/deepbeepmeep.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/deepbeepmeep/WanGP">WanGP</a></td><td>Video generator for low-VRAM GPUs, with H3 among its models.</td></tr>
<tr><td width="56"><img src="https://github.com/mcmonkeyprojects.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</a></td><td>Web UI with H3 among its video models.</td></tr>
<tr><td width="56"><img src="https://github.com/tin2tin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tin2tin/Pallaidium">Pallaidium</a></td><td>Generative movie studio inside Blender's video editor, with H3 and the Open-H3 Regenerate upscaler.</td></tr>
<tr><td width="56"><img src="https://github.com/mrbizarro.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mrbizarro/Phosphene">Phosphene</a></td><td>Native Mac app that runs H3 and LTX-2.5 offline on Apple silicon; paid Pro version optional.</td></tr>
<tr><td width="56"><img src="https://github.com/neuregex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/neuregex/freefield">freefield</a></td><td>Ready-made H3 apps on Modal: video from text or a photo, extending a clip, and new camera angles.</td></tr>
<tr><td width="56"><img src="https://github.com/AbrahamPaulJ.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AbrahamPaulJ/nightmare-mobile">nightmare-mobile</a></td><td>Android node-graph app for Snapdragon phones that makes H3 video with sound; non-commercial license.</td></tr>
<tr><td width="56"><img src="https://github.com/SwotAtmk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SwotAtmk/infinite-creation">infinite-creation</a></td><td>Agent app that turns a novel or script into a video of any length with local ComfyUI H3; needs 12 GB or more of VRAM.</td></tr>
<tr><td width="56"><img src="https://github.com/saddam213.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/saddam213/AmuseAI">Amuse</a></td><td>Desktop app built on .NET for local image, video and audio generation that lists H3 among its video pipelines, on CUDA or Vulkan.</td></tr>
<tr><td width="56"><img src="https://github.com/james-see.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/james-see/ltx-video-mac">ltx-video-mac</a></td><td>Native macOS app that runs H3 on Apple silicon through h3.c, in BF16, INT8 or a 6-step Turbo build, with a local REST API.</td></tr>
<tr><td width="56"><img src="https://github.com/code-video-model.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/code-video-model/CodeVideoModel">CodeVideoModel</a></td><td>Has a language model build a Three.js layout video, then turns it into realistic video with local H3 Ref2VA.</td></tr>
<tr><td width="56"><img src="https://github.com/OpenDCAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenDCAI/GameFactory-3A">GameFactory-3A</a></td><td>Game-asset framework whose CG-video stage renders cutscenes and trailers with local H3 at 720p.</td></tr>
<tr><td width="56"><img src="https://github.com/siyuan-liu31.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/siyuan-liu31/minimax-h3-video-studio">MiniMax H3 Video Studio</a></td><td>Self-hosted studio and CLI on ComfyUI for H3 text, image, reference and video-to-video, and segment-by-segment long videos.</td></tr>
<tr><td width="56"><img src="https://github.com/RachelForster.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/RachelForster/Shinsekai">Shinsekai</a></td><td>AI RPG maker that can animate character sprites with H3 image-to-video on local ComfyUI.</td></tr>
<tr><td width="56"><img src="https://github.com/jnMetaCode.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jnMetaCode/openshorts">OpenShorts</a></td><td>Local-first short-video pipeline whose AI drama mode generates clips with H3 on your machine through stable-diffusion.cpp, or in the cloud.</td></tr>
</table>

<a id="h3-api"></a>

### API integrations

<table>
<tr><td width="56"><img src="https://github.com/NousResearch.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NousResearch/hermes-agent">Hermes Agent</a></td><td>Self-improving agent whose fal video plugin runs H3, H3 Max and H3 Max Turbo, mapping usual resolutions to their tiers.</td></tr>
<tr><td width="56"><img src="https://github.com/n8n-io.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/n8n-io/n8n/tree/master/packages/@n8n/nodes-langchain/nodes/vendors/MiniMax">n8n</a></td><td>Workflow automation tool whose MiniMax node can generate H3 video through the MiniMax API; fair-code license.</td></tr>
<tr><td width="56"><img src="https://github.com/harry0703.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/harry0703/MoneyPrinterTurbo">MoneyPrinterTurbo</a></td><td>Turns a topic or keyword into a short video and can generate the footage with H3 through Metaso's API.</td></tr>
<tr><td width="56"><img src="https://github.com/OpenMontageInc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenMontageInc/OpenMontage">OpenMontage</a></td><td>Agentic video production system with an H3 skill and tools that call the MiniMax API, fal or local ComfyUI.</td></tr>
<tr><td width="56"><img src="https://github.com/BerriAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BerriAI/litellm">LiteLLM</a></td><td>Python SDK and AI gateway that can create H3 video tasks on the MiniMax API through its OpenAI-style video calls.</td></tr>
<tr><td width="56"><img src="https://github.com/QuantumNous.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/QuantumNous/new-api">New API</a></td><td>Self-hosted AI gateway whose Hailuo task plugin sends H3 multimodal jobs to MiniMax's v2 video API.</td></tr>
<tr><td width="56"><img src="https://github.com/HeyPuter.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HeyPuter/puter">Puter</a></td><td>Self-hostable web desktop whose MiniMax video provider uses H3 by default.</td></tr>
<tr><td width="56"><img src="https://github.com/THU-MAIC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/THU-MAIC/OpenMAIC">OpenMAIC</a></td><td>Multi-agent classroom whose video adapter runs H3 and H3 Max on MiniMax's v2 API, directly or via TokenDance.</td></tr>
<tr><td width="56"><img src="https://github.com/agno-agi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agno-agi/agno">Agno</a></td><td>Agent framework whose MiniMax tool generates H3 video from a text prompt through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/storytold.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/storytold/artcraft">ArtCraft</a></td><td>Desktop app that stages scenes in 2D and 3D, then generates video with H3 through its own cloud or fal.</td></tr>
<tr><td width="56"><img src="https://github.com/agentscope-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agentscope-ai/QwenPaw">QwenPaw</a></td><td>Personal assistant whose Creator plugin makes video with H3 or H3 Max on the MiniMax API, or with self-hosted H3 on SGLang.</td></tr>
<tr><td width="56"><img src="https://github.com/yetone.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yetone/magpie">Magpie</a></td><td>Menu-bar app that sets up models for coding agents; with a MiniMax API key it also makes videos with H3.</td></tr>
<tr><td width="56"><img src="https://github.com/simstudioai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/simstudioai/sim">Sim</a></td><td>Agent workflow workspace whose video tool runs H3 and its Highspeed, Lite and Max versions on the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/vercel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vercel/ai">AI SDK</a></td><td>Vercel's TypeScript toolkit whose MiniMax provider makes H3 and H3 Max video and checks each mode's input limits.</td></tr>
<tr><td width="56"><img src="https://github.com/1Panel-dev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1Panel-dev/MaxKB">MaxKB</a></td><td>Enterprise agent platform whose MiniMax provider adds H3 and H3 Max text-to-video and image-to-video models.</td></tr>
<tr><td width="56"><img src="https://github.com/HBAI-Ltd.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HBAI-Ltd/Toonflow-app">Toonflow</a></td><td>Short-drama studio from script and storyboard to video, calling H3 through the Metaso, CompShare or APIMart API.</td></tr>
<tr><td width="56"><img src="https://github.com/chatfire-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chatfire-AI/huobao-drama">Huobao Drama</a></td><td>Generates a short drama from one sentence to finished video, with an H3 adapter for the MiniMax API; non-commercial license.</td></tr>
<tr><td width="56"><img src="https://github.com/corsairdev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/corsairdev/corsair">Corsair</a></td><td>App integration platform whose MiniMax plugin lets agents generate H3 and H3 Max video through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/waooAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/waooAI/waoowaoo">waoowaoo</a></td><td>Film and video production platform that calls H3 and H3 Max through OpenRouter; Elastic License 2.0.</td></tr>
<tr><td width="56"><img src="https://github.com/maximhq.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/maximhq/bifrost">Bifrost</a></td><td>AI gateway that runs H3 and H3 Max as async tasks on MiniMax's v2 API, fitting size and seconds to each model's range.</td></tr>
<tr><td width="56"><img src="https://github.com/rikkahub.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rikkahub/rikkahub">RikkaHub</a></td><td>Android chat app for many LLM providers that can also generate H3 video through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/TeamWiseFlow.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TeamWiseFlow/xiaobei">xiaobei</a></td><td>Marketing agent for small businesses whose video skill makes clips with H3 through the MiniMax API; MIT with extra terms.</td></tr>
<tr><td width="56"><img src="https://github.com/openJiuwen-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openJiuwen-ai/jiuwenswarm">JiuwenSwarm</a></td><td>AI agent whose video tools send H3 and H3 Max jobs to the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/Devin-AXIS.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Devin-AXIS/iPolloWork">iPolloWork</a></td><td>Agent workbench with H3 video and H3 digital-human tools run as RunningHub workflows; personal non-commercial license.</td></tr>
<tr><td width="56"><img src="https://github.com/ArcReel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ArcReel/ArcReel">ArcReel</a></td><td>Agent workspace that turns novels and scripts into shots and video, with a built-in H3 endpoint for the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/monid-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/monid-ai/monid">Monid</a></td><td>Marketplace of tool APIs for agents, with connectors for H3, H3 Max and H3 Max Turbo on MiniMax's v2 API.</td></tr>
<tr><td width="56"><img src="https://github.com/EverMind-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/EverMind-AI/Raven">Raven</a></td><td>Multi-agent system whose MiniMax media tools create H3 and H3 Max videos through the v2 API.</td></tr>
<tr><td width="56"><img src="https://github.com/pollinations.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pollinations/pollinations">Pollinations</a></td><td>Open-source generative AI platform whose API serves H3, H3 Max and H3 Max Turbo video relayed from fal.</td></tr>
<tr><td width="56"><img src="https://github.com/chaitin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chaitin/MonkeyCode">MonkeyCode</a></td><td>Enterprise AI office workbench whose video service runs H3 text, frame and reference modes on the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/darkzOGx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/darkzOGx/youtube-automation-agent">YouTube Automation Agent</a></td><td>Runs a YouTube channel end to end and can render clips with H3 on the MiniMax API under a paid-seconds cap.</td></tr>
<tr><td width="56"><img src="https://github.com/crmne.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/crmne/ruby_llm">RubyLLM</a></td><td>Ruby AI framework whose MiniMax video protocol lets its <code>animate</code> call create H3 videos.</td></tr>
<tr><td width="56"><img src="https://github.com/Agent-Field.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Agent-Field/agentfield">AgentField</a></td><td>Framework for running agents as services whose MiniMax media provider uses H3 as its default video model.</td></tr>
<tr><td width="56"><img src="https://github.com/jnMetaCode.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jnMetaCode/agency-orchestrator">Agency Orchestrator</a></td><td>Multi-agent workflow tool whose video steps can render H3 clips through the Metaso or APIMart API.</td></tr>
<tr><td width="56"><img src="https://github.com/xuanyustudio.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xuanyustudio/LocalMiniDrama">LocalMiniDrama</a></td><td>Short-drama tool that keeps data on your machine and renders shots with H3 on the MiniMax API or through relays.</td></tr>
<tr><td width="56"><img src="https://github.com/glanderness.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/glanderness/BeefTV">BeefTV</a></td><td>Canvas video workspace with a live H3 Max Director view on fal and an H3 plugin for MiniMax's v2 API; source-available.</td></tr>
<tr><td width="56"><img src="https://github.com/WhiteTowerAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WhiteTowerAI/ohmygame">OhMyGame</a></td><td>AI game studio app whose video tools offer H3 and H3 Max through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/ddcat-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ddcat-ai/open-ai-canvas">open-ai-canvas</a></td><td>Workspace for AI film with H3 prompt cards and H3 video through MiniMax, Alibaba Bailian or Metaso.</td></tr>
<tr><td width="56"><img src="https://github.com/Stonewuu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Stonewuu/ai-fusion-video">ai-fusion-video</a></td><td>Java agent platform for short dramas and comics, with H3 and H3 Max presets for the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/siddsachar.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/siddsachar/row-bot">Row-Bot</a></td><td>Local-first assistant whose video tool runs H3 and its Highspeed, Lite and Max versions on the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/869413421.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/869413421/ai-moive-studio">AICON</a></td><td>Canvas workflow agent for video creation whose default video model is H3, called through the project's own API gateway.</td></tr>
<tr><td width="56"><img src="https://github.com/tigerowo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tigerowo/infinite-canvas">infinite-canvas</a></td><td>AI creation canvas that makes H3 video over a MiniMax-protocol channel, with built-in H3 prompt-writing skills.</td></tr>
<tr><td width="56"><img src="https://github.com/xixihhhh.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xixihhhh/clipforge">ClipForge</a></td><td>Turns one product photo into a short sales video, with H3 and H3 Max available through Atlas Cloud.</td></tr>
<tr><td width="56"><img src="https://github.com/0xsline.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/0xsline/StoryGen-Atelier">StoryGen-Atelier</a></td><td>Storyboard tool where Gemini writes shots and frames, with H3 on the MiniMax API as an alternative to Veo for clips.</td></tr>
<tr><td width="56"><img src="https://github.com/looplj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/looplj/reveric">Reveric</a></td><td>Desktop studio for novels and short dramas that generates video with H3 through the MiniMax or Metaso API.</td></tr>
<tr><td width="56"><img src="https://github.com/anymouschina.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/anymouschina/TapCanvas">TapCanvas</a></td><td>Agent-native board for AI film and comics whose bundled gateway speaks MiniMax's H3 v2 video protocol.</td></tr>
<tr><td width="56"><img src="https://github.com/ChrisChen667788.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ChrisChen667788/wind-comic">wind-comic</a></td><td>Multi-agent pipeline that turns one line of text into a short drama, rendering shots with H3 on the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/agentjido.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agentjido/req_llm">ReqLLM</a></td><td>Elixir LLM library whose <code>generate_video</code> calls H3 on the MiniMax API and can upload local reference files.</td></tr>
<tr><td width="56"><img src="https://github.com/ZMGID.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ZMGID/kivio">Kivio</a></td><td>Screen-level AI assistant for macOS and Windows that can make H3 videos through MiniMax's v2 API.</td></tr>
<tr><td width="56"><img src="https://github.com/tryonlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tryonlabs/opentryon">OpenTryOn</a></td><td>Virtual try-on and fashion toolkit with H3 and H3 Max through the MiniMax API, or H3 on a local GPU; non-commercial.</td></tr>
<tr><td width="56"><img src="https://github.com/ganbo-gab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ganbo-gab/open-storyboard-canvas">Open Storyboard Canvas</a></td><td>AI shot-planning board with camera controls whose video nodes can use a preset MiniMax H3 provider.</td></tr>
<tr><td width="56"><img src="https://github.com/fancyboi999.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/fancyboi999/Loomic">Loomic</a></td><td>Chat-driven design agent on an infinite canvas that makes H3 video from text or frames through Metaso's API.</td></tr>
<tr><td width="56"><img src="https://github.com/chillzhuang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chillzhuang/Kinema">Kinema</a></td><td>Agent for AI comics, films and commentary videos, with a <code>minimax-h3</code> provider on MiniMax's v2 API.</td></tr>
<tr><td width="56"><img src="https://github.com/Finderchangchang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Finderchangchang/brewreel">BrewReel</a></td><td>Turns a product brief into a vertical promo video; its talking-head mode adds AI B-roll made with H3 on the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/LudwigKienle.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LudwigKienle/ai-video-production-editor">AI Video Production Editor</a></td><td>Electron app for AI filmmaking whose node pipelines can run H3 text- and image-to-video on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/sumarilkkxx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sumarilkkxx/Mora">Mora</a></td><td>Local-first promo video workbench that can render clips with H3 through Atlas Cloud.</td></tr>
<tr><td width="56"><img src="https://github.com/saihhold-zhao.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/saihhold-zhao/polox_ai">Polox AI</a></td><td>Generation platform built on DeepSeek Harness, with H3 text, image and reference to video through WaveSpeed.</td></tr>
<tr><td width="56"><img src="https://github.com/s-nagaev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/s-nagaev/chibi">Chibi</a></td><td>Self-hosted Telegram bot whose video generation can use H3 and H3 Max through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/WaveSpeedAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WaveSpeedAI/mcp-server">WaveSpeed MCP Server</a></td><td>MCP server whose H3 family generates, edits and extends video on WaveSpeed's API.</td></tr>
<tr><td width="56"><img src="https://github.com/Sociaro-Research.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Sociaro-Research/open-media-gateway">Open Media Gateway</a></td><td>Self-hosted service whose MiniMax provider covers H3 and its Highspeed, Lite and Max versions on the official API.</td></tr>
<tr><td width="56"><img src="https://github.com/KunLabAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KunLabAI/KunFlix">KunFlix</a></td><td>Platform for film and TV ad creation whose video service calls H3 through MiniMax's v2 API; Apache-2.0 with extra terms.</td></tr>
<tr><td width="56"><img src="https://github.com/Jjat00.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Jjat00/hf-studio">hf-studio</a></td><td>Self-hosted media studio that runs H3 video through Higgsfield and H3-Image through KIE.</td></tr>
<tr><td width="56"><img src="https://github.com/ItusiAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ItusiAI/Open-Magiviz">Magiviz</a></td><td>AI video creation engine whose video models include H3 through KIE's API.</td></tr>
<tr><td width="56"><img src="https://github.com/openfilm.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openfilm/openfilm">OpenFilm</a></td><td>Video agent and editor for coding agents that can replace the people in a clip with H3 Max Recast on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/HaD0Yun.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HaD0Yun/CozyClay">CozyClay</a></td><td>Browser previs tool: block a scene and camera, then render shots with fal's H3 Max Turbo or your own ComfyUI H3 workflow.</td></tr>
<tr><td width="56"><img src="https://github.com/openstory-so.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openstory-so/openstory">OpenStory</a></td><td>Video sequence platform whose motion step offers H3 Max text, image and reference to video on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/lovisdotio.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lovisdotio/video-to-game">video-to-game</a></td><td>31 games, each rebuilt from a 10 to 15 second H3 Max clip made on fal, with the asset pipeline.</td></tr>
<tr><td width="56"><img src="https://github.com/patch-gg.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/patch-gg/anymove">anymove</a></td><td>Motion capture for any creature from two-camera H3 Max clips on fal, with a skill that trains an H3 LoRA there.</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/prompt2rig">prompt2rig</a></td><td>Type a motion and a rigged character performs it, with H3 Max Turbo on fal as the fast video option.</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/falcraft">falcraft</a></td><td>Minecraft mod for fal-generated builds, including a wall TV whose live stream is rendered by H3 Max Director on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/vargHQ.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vargHQ/sdk">varg SDK</a></td><td>TypeScript video SDK with H3 Max text, image and reference to video models on fal; its JSX API is now deprecated.</td></tr>
<tr><td width="56"><img src="https://github.com/orkunisitmak.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/orkunisitmak/banana-studio">Banana Studio</a></td><td>Self-hosted media studio with H3 Max text, image and reference to video, extend, lip sync and recast on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/henjicc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/henjicc/Henji-AI">Henji AI</a></td><td>Desktop app that brings many AI providers together, with H3 Max through fal or APIMart.</td></tr>
<tr><td width="56"><img src="https://github.com/xflare-bot.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xflare-bot/pokemonlive">pokemonlive</a></td><td>Local Pokémon battle prototype whose cinematics come from H3 Max Turbo and H3 Max reference-to-video on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/internetphysics.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/internetphysics/live-classroom">live-classroom</a></td><td>Live TV channel that teaches any topic, with each lesson beat rendered just in time by H3 Max Turbo on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/gokayfem.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gokayfem/h3-max-education">H3 Max Realtime Education</a></td><td>Science tutor you can interrupt by voice, with teaching video generated continuously by H3 Max on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/marketcalls.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/marketcalls/sketchmaker">SketchMaker</a></td><td>Self-hostable creative app whose animate, story and lip-sync features use H3 Max on fal by default.</td></tr>
<tr><td width="56"><img src="https://github.com/gokayfem.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gokayfem/H3-Max-Blender">H3 Max Blender</a></td><td>Blender neural rendering demo: H3 Max on fal restyles a gray ship in four looks as its geometry changes.</td></tr>
<tr><td width="56"><img src="https://github.com/Valiera00.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Valiera00/SPITE">SPITE</a></td><td>Pre-production canvas for filmmakers whose video nodes can use H3 Max through fal with your own key.</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/interdimensional-game">Interdimensional Game</a></td><td>Choose-your-own-adventure game with real-time video from H3 Max on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/unreel">UNREEL</a></td><td>AI streaming app where a language model writes each episode and H3 Max Turbo on fal renders it shot by shot.</td></tr>
<tr><td width="56"><img src="https://github.com/ilkerzg.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ilkerzg/storycast">StoryCast</a></td><td>Turns a topic into a story video, with shots and lip-synced lines from H3 Max on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/eren23.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/eren23/openflipbook">OpenFlipbook</a></td><td>Clickable picture book where every page is AI-drawn, with optional page animation from H3 Max or H3 Max Turbo on fal.</td></tr>
</table>

<a id="h3-comfyui"></a><a id="nodes"></a><a id="recipes"></a><a id="open-h3-regenerate"></a>

### ComfyUI nodes

<table>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-audio-T8">comfyui-minimax-h3-audio-T8</a></td><td>62 nodes for audio, stills, conditioning, long video and speech.</td></tr>
<tr><td width="56"><img src="https://github.com/huangserva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/huangserva/ComfyUI_MiniMaxH3_Director">ComfyUI_MiniMaxH3_Director (huangserva)</a></td><td>Storyboard nodes with five importable templates: t2v, fl2v, r2v, v2v and rv2v.</td></tr>
<tr><td width="56"><img src="https://github.com/AIMixer.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director">ComfyUI_MiniMaxH3_Director (AIMixer)</a></td><td>The original Director nodes.</td></tr>
<tr><td width="56"><img src="https://github.com/seesee75-commits.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director">ComfyUI-MiniMaxH3-Director</a></td><td>Timeline editor: media on tracks, one prompt per shot, live preview and retakes.</td></tr>
<tr><td width="56"><img src="https://github.com/imbutus.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/imbutus/ComfyUI-MiniMaxDirector">ComfyUI-MiniMaxDirector</a></td><td>Timeline of shots, camera moves and audio cues compiled into the structured prompt from the official guides.</td></tr>
<tr><td width="56"><img src="https://github.com/nkxx188.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy">ComfyUI-MiniMaxH3-Easy</a></td><td>One compact workflow for T2V, I2V, first and last frame, and reference video.</td></tr>
<tr><td width="56"><img src="https://github.com/FX-FeiHou.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3">ComfyUI-FeiHou-Easy-H3</a></td><td>Fork of MiniMaxH3-Easy with built-in slots for 9 images, 3 videos and 3 audio references.</td></tr>
<tr><td width="56"><img src="https://github.com/NikoDemon80.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context">ComfyUI-H3-Motion-Context</a></td><td>Chains clips while keeping motion and sound continuous across them.</td></tr>
<tr><td width="56"><img src="https://github.com/princepainter.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/princepainter/ComfyUI-PainterNodes">ComfyUI-PainterNodes</a></td><td><code>MiniMaxRefToVideo2</code> node with the official reference and dialogue format.</td></tr>
<tr><td width="56"><img src="https://github.com/wyzborrero.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/wyzborrero/ComfyUI-H3-FunControl">ComfyUI-H3-FunControl</a></td><td>Loads the Fun ControlNet-Union with its preprocessors and inpaint path.</td></tr>
<tr><td width="56"><img src="https://github.com/lihaoyun6.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lihaoyun6/ComfyUI-MiniMaxH3_Ref-Patch">ComfyUI-MiniMaxH3_Ref-Patch</a></td><td>Applies the Ref Patch to FL2VA.</td></tr>
<tr><td width="56"><img src="https://github.com/zechengtang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zechengtang/VPack">VPack</a></td><td>Compresses Ref2VA reference-video conditioning to cut reference tokens.</td></tr>
<tr><td width="56"><img src="https://github.com/alisson-anjos.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3">TARA-H3</a></td><td>Experimental temporal alignment of Ref2VA video-reference attention, with a technical report.</td></tr>
<tr><td width="56"><img src="https://github.com/matlowai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matlowai/ComfyUI-Dials">ComfyUI-Dials</a></td><td>Training-free steering vectors built from prompt pairs, for motion and style.</td></tr>
<tr><td width="56"><img src="https://github.com/scraed.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scraed/LanPaint">LanPaint</a></td><td>Training-free video and audio inpainting.</td></tr>
<tr><td width="56"><img src="https://github.com/Tr1dae.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler">ComfyUI-MiniMaxH3_LatentUpscaler</a></td><td>Latent upscaler for H3's audio-video latents, for two-pass sampling.</td></tr>
<tr><td width="56"><img src="https://github.com/LBH-123-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler">Comfyui_Minimax_h3_latent_Upscaler</a></td><td>Learned 1.0 to 4.0x latent upscaler that skips the VAE round-trip.</td></tr>
<tr><td width="56"><img src="https://github.com/tazmannner.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tazmannner/ComfyUI-H3-ResCascade">ComfyUI-H3-ResCascade</a></td><td>Hires-fix nodes that refine about 1 MP renders to 1080p or 1440p, keeping the audio.</td></tr>
<tr><td width="56"><img src="https://github.com/rdh9616.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate">ComfyUI Open-H3 Regenerate</a></td><td>Nodes and workflows for NVIDIA's Open-H3-Regenerate-2K refiner.</td></tr>
<tr><td width="56"><img src="https://github.com/hashu786.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hashu786/ComfyUI-Tessera">Tessera</a></td><td>Tiled, chunked video refinement with streamed VAE encode and decode.</td></tr>
<tr><td width="56"><img src="https://github.com/lihaoyun6.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT">ComfyUI-H3VAE_TRT</a></td><td>TensorRT video VAE, up to 1.7x faster decode.</td></tr>
<tr><td width="56"><img src="https://github.com/xmarre.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3">ComfyUI-Spectrum-MiniMax-H3</a></td><td>Skips selected transformer steps by forecasting features; output is approximate.</td></tr>
<tr><td width="56"><img src="https://github.com/kijai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kijai/ComfyUI-SolAttn_triton">ComfyUI-SolAttn_triton</a></td><td>Training-free Sol-Attn sparse attention, tested with H3; deprecated now that ComfyUI has it built in.</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-VDN-H3">ComfyUI-VDN-H3</a></td><td>Runs the VDN adapters in ComfyUI; a <a href="https://github.com/Speach1sdef178/ComfyUI-VDN-H3-24GB">24 GB fork</a> offloads more.</td></tr>
<tr><td width="56"><img src="https://github.com/Kablex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Kablex/ComfyUI-Ref2VA-VSA">ComfyUI-Ref2VA-VSA</a></td><td>Video sparse attention for Ref2VA, about 72 s per 5 s clip on an RTX 4090.</td></tr>
<tr><td width="56"><img src="https://github.com/Jalen-Brunson.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc">ComfyUI-MiniMax-H3-PDD-Acc</a></td><td>Loads the PDD LoRA with its 32 output heads on older ComfyUI.</td></tr>
<tr><td width="56"><img src="https://github.com/xmarre.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xmarre/ComfyUI-Sol-H3">ComfyUI-Sol-H3</a></td><td>ComfyUI front end for the Sol-H3 runtime (GPL-3.0).</td></tr>
<tr><td width="56"><img src="https://github.com/zechengtang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zechengtang/Spark-H3">Spark-H3</a></td><td>Adaptive block-sparse attention with diffusers and ComfyUI packages.</td></tr>
<tr><td width="56"><img src="https://github.com/iamwavecut.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/iamwavecut/ComfyUI-OrbitQuant/tree/feature/minimax-h3-comfyui">ComfyUI-OrbitQuant</a></td><td>Loads the OrbitQuant W4A4 build.</td></tr>
<tr><td width="56"><img src="https://github.com/NikoDemon80.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NikoDemon80/ComfyUI-FastH3-Lora-Converter">ComfyUI-FastH3-Lora-Converter</a></td><td>Converts the FastH3 LoRA to ComfyUI key order.</td></tr>
<tr><td width="56"><img src="https://github.com/BobJohnson24.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BobJohnson24/ComfyUI-INT8-Fast">ComfyUI-INT8-Fast</a></td><td>Largely superseded by native INT8; its <code>convert_comfy_quant.py</code> converts older INT8 files.</td></tr>
<tr><td width="56"><img src="https://github.com/Songssx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector">ComfyUI-MiniMaxH3-TimelineDirector</a></td><td>Editable timeline of reference media for Ref2VA, with an all-in-one director workflow.</td></tr>
<tr><td width="56"><img src="https://github.com/Luisacaotica.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod">ComfyUI-MiniMaxH3Mod</a></td><td>RefMod adapters that compress reference images and videos into small latents without training; still under construction.</td></tr>
<tr><td width="56"><img src="https://github.com/Carasibana.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Carasibana/ComfyUI-H3-FaceRefine">ComfyUI-H3-FaceRefine</a></td><td>Tracks small faces frame by frame, refines them with H3 and stitches them back.</td></tr>
<tr><td width="56"><img src="https://github.com/JGRFW.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/JGRFW/comfyui-AICG3D">comfyui-AICG3D</a></td><td>Short-drama workbench that splits a script into segments for videos of any length.</td></tr>
<tr><td width="56"><img src="https://github.com/ukr8b3g-cmyk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ukr8b3g-cmyk/H3-Character-Sheet-Designer">H3-Character-Sheet-Designer</a></td><td>Character reference sheets with up to 13 views from one character image.</td></tr>
<tr><td width="56"><img src="https://github.com/tritant.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tritant/ComfyUI_MiniMax_H3_Extender">ComfyUI_MiniMax_H3_Extender</a></td><td>Chains clips into long video while keeping motion, picture and audio continuous.</td></tr>
<tr><td width="56"><img src="https://github.com/MajoorWaldi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MajoorWaldi/ComfyUI-Majoor-H3-GuideMaster">ComfyUI-Majoor-H3-GuideMaster</a></td><td>Places images and audio on the H3 timeline as guides.</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3">ComfyUI-Viggle-Animate-H3</a></td><td>Nodes for Viggle Animate, which replaces a character in a driving video with a reference.</td></tr>
<tr><td width="56"><img src="https://github.com/astropuzzo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio">ComfyUI-MiniMax-H3-Image-Studio</a></td><td>Still images and reference editing with the video model, plus Qwen detail refinement.</td></tr>
<tr><td width="56"><img src="https://github.com/thaakeno.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thaakeno/ComfyUI-MiniMax-H3-Studio">ComfyUI-MiniMax-H3-Studio</a></td><td>Reference-aware still-image workflow built on the video model; alpha.</td></tr>
<tr><td width="56"><img src="https://github.com/HELPMEEADICE.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS">TE-Speed-MiniMaxH3-OSS</a></td><td>Step cache that reuses late-block residuals when sigma barely changes, about 45% faster at 30 steps; needs a one-time ComfyUI patch.</td></tr>
<tr><td width="56"><img src="https://github.com/vizart-vj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia">ComfyUI-MiniMax-H3-LongMedia</a></td><td>Long-form video and audio nodes with a Director timeline, single-clip rerenders and lip-sync redubbing.</td></tr>
<tr><td width="56"><img src="https://github.com/ukr8b3g-cmyk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum">ComfyUI-H3-Continuum</a></td><td>Generates long video and audio in chunks with a prompt per chunk, lets you retry one chunk and resumes from saved takes.</td></tr>
<tr><td width="56"><img src="https://github.com/seitanism.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef">ComfyUI-H3-Motion-Context-MultiRef</a></td><td>Fork of H3-Motion-Context that adds music videos built around one song, V2V motion transfer, AV bridges between two clips and masked keyframes.</td></tr>
<tr><td width="56"><img src="https://github.com/chanon.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chanon/comfyui-obvpm-timeline">comfyui-obvpm-timeline</a></td><td>Timeline node that generates, extends, prepends, bridges and loops clips like a small video editor; needs comfyui-obvpm.</td></tr>
<tr><td width="56"><img src="https://github.com/yichengup.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3">ComfyUI-YCNodes-MiniMax-H3</a></td><td>Six nodes, including time-split prompts, dynamic CFG, low-noise refinement and tiled sampling.</td></tr>
<tr><td width="56"><img src="https://github.com/goohai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/goohai/Goohai-MiniMax-H3_Integration">Goohai-MiniMax-H3_Integration</a></td><td>All-in-one conditioning node that picks T2VA, FL2VA, Ref2VA or hybrid mode from the uploaded media and writes the official tags.</td></tr>
<tr><td width="56"><img src="https://github.com/bbaudio-2025.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale">Comfyui-MMH3-UltimateUpscale</a></td><td>Upscales and re-samples a finished latent in time chunks and tiles, so long high-resolution clips fit limited VRAM.</td></tr>
<tr><td width="56"><img src="https://github.com/matlowai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matlowai/ComfyUI-MAINodes">ComfyUI-MAINodes</a></td><td>Motion Lab renders a slowed copy and drops held frames to fix smeared fast motion; also turnaround and low-VRAM nodes.</td></tr>
<tr><td width="56"><img src="https://github.com/knoic.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream">ComfyUI-MiniMaxH3-PrefixStream</a></td><td>Long-video continuation that writes the previous clip's clean latents into the start of the next, with a clip bin for picking which take to continue.</td></tr>
<tr><td width="56"><img src="https://github.com/nicolab28.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nicolab28/ComfyUI-ClipProj">ComfyUI-ClipProj</a></td><td>Swaps the 32B text encoder for Qwen3-VL 4B or 8B through a learned <a href="https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3">projection</a>, down to 4.5 GB.</td></tr>
<tr><td width="56"><img src="https://github.com/scottmudge.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader">ComfyUI_MinimaxH3HybridLoader</a></td><td>Loader that merges chosen Ref2VA tensor groups into FL2VA at load time by block range, without writing a new file.</td></tr>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still">ComfyUI-Fizgig-H3-Still</a></td><td>Two nodes for true single-frame stills, with a one-frame latent and a clean decode at any size; no extra model.</td></tr>
<tr><td width="56"><img src="https://github.com/Adudeguyman.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Adudeguyman/ComfyUI-H3-AudioRefine">ComfyUI-H3-AudioRefine</a></td><td>Freezes the video of a sampled latent and runs extra steps on the audio only, for 4-step Turbo runs whose sound is too rough.</td></tr>
<tr><td width="56"><img src="https://github.com/KadoBOT.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio">ComfyUI-H3-Continuity-Studio</a></td><td>Multi-shot timeline that carries context between clips and across scene or camera changes, with an MCP adapter for coding agents.</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-sol-attn">ComfyUI-sol-attn</a></td><td>Sol-Attn Triton kernel with H3 nodes for zero-copy attention, fused AdaLN modulation and chunked MLP, on RTX 30 series through Blackwell.</td></tr>
<tr><td width="56"><img src="https://github.com/rookiestar28.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rookiestar28/MiniMaxH3-Studio">MiniMaxH3-Studio</a></td><td>ComfyUI sidebar with 28 nodes for prompts, the official H3 workflows, long-video planning and a built-in editor.</td></tr>
<tr><td width="56"><img src="https://github.com/veda-sparse.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/veda-sparse/Veda-on-ComfyUI">Veda-on-ComfyUI</a></td><td>One node that adds Veda sparse attention to native H3, 2.9x faster end to end on an RTX 5070, on NVIDIA, AMD ROCm and Apple silicon.</td></tr>
<tr><td width="56"><img src="https://github.com/WASasquatch.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WASasquatch/was-node-suite-comfyui">WAS Node Suite</a></td><td>Node suite whose H3 nodes condition a multi-segment video on one node, extend it by window and steer segments with Fun ControlNet.</td></tr>
<tr><td width="56"><img src="https://github.com/Astral-xm.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Astral-xm/ComfyUI-MiniMax-ContextIR">ComfyUI-MiniMax-ContextIR</a></td><td>Nodes from prompt to sampling for local H3, with optional prompt rewriting through MiniMax's H3 Context-IR API.</td></tr>
</table>

<a id="h3-workflows"></a><a id="wf"></a><a id="wf-comfyui"></a>

### Workflows

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow">MiniMax-H3-Multishot-Workflow</a></td><td>Strings FL2VA and Ref2VA clips into one sequence with matched audio handoffs.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65dc6589ec3dc9ccf761d936/SxJaAV-JarA1HZj4D1iEV.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows">comfy-MiniMax-H3-workflows</a></td><td>The low-VRAM acceleration stack in one importable workflow.</td></tr>
<tr><td width="56"><img src="https://github.com/FlowForgeLabAi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlowForgeLabAi/MiniMax-H3-8GB-Workflows">MiniMax-H3-8GB-Workflows</a></td><td>Three-tier workflow pack for 8 GB laptops.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/minimax_fl2v_gguf_workflow.json">FL2V GGUF workflow</a></td><td>Runs the GGUF-quantized FL2VA model in ComfyUI.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json">OrbitQuant workflows</a></td><td>T2VA workflow for the W4A4 build, with <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA-api.json">T2VA</a> and <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-Ref2VA-api.json">Ref2VA</a> API forms.</td></tr>
<tr><td width="56"><img src="https://github.com/pixelaigc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pixelaigc/ComfyUI-MiniMax-H3-Workflow">ComfyUI-MiniMax-H3-Workflow</a></td><td>First/last-frame, image-to-video and text-to-video in one workflow, with the Turbo LoRA and a control panel for resolution.</td></tr>
</table>

<a id="h3-prompting"></a><a id="recipes-prompt"></a><a id="agent-skills"></a>

### Prompting and agent skills

<table>
<tr><td width="56"><img src="https://github.com/1038lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor">ComfyUI-MiniMax-H3-Promptor</a></td><td>Builds H3 prompts with <code>&lt;Picture X&gt;</code> tags placed inline in the action.</td></tr>
<tr><td width="56"><img src="https://github.com/ethanfel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide">ComfyUI-MiniMax-H3-Guide</a></td><td>Splits identity, keyframes, motion, voice and score into roles and compiles valid H3 prose.</td></tr>
<tr><td width="56"><img src="https://github.com/ruashots.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ruashots/open-h3-ir">OpenH3-IR</a></td><td>Writes prompts in the Context-IR format MiniMax published, then checks and fixes them; CLI, HTTP or ComfyUI.</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8">comfyui-minimax-h3-prompt-enhancer-T8</a></td><td>Server-side prompt enhancement in ComfyUI.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680de13385293771bc57400b/qnjz3avU_cKzRZA2aGjuZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA">MiniMax-H3-Prompt-Rewriter-LoRA</a></td><td>Qwen3.6-27B LoRA that rewrites short prompts into H3's three-part structure; an <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B">8B vision version</a> also reads keyframes.</td></tr>
<tr><td width="56"><img src="https://github.com/BeatAPI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BeatAPI/awesome-minimax-h3-prompts">awesome-minimax-h3-prompts</a></td><td>Prompt collection with example videos, by category.</td></tr>
<tr><td width="56"><img src="https://github.com/yangzhou-chaofan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yangzhou-chaofan/minimax-h3-1000-prompts">MiniMax H3 1K prompts</a></td><td>1,000 curated text-to-video prompts with attribution.</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/minimax-h3-prompt-skill-T8">minimax-h3-prompt-skill-T8</a></td><td>Case library installable as an agent skill, with a desktop viewer.</td></tr>
<tr><td width="56"><img src="https://github.com/joeVenner.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/joeVenner/awesome-minimax-h3">awesome-minimax-h3 (skills)</a></td><td>Agent skills that take a brief to a finished file through the hosted API.</td></tr>
<tr><td width="56"><img src="https://github.com/okaypic.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/okaypic/okaypic-video-skill">okaypic-video-skill</a></td><td>Agent skill that produces a captioned multi-shot episode through the okaypic API.</td></tr>
<tr><td width="56"><img src="https://github.com/SlavaSexton.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SlavaSexton/ComfyUI-Agent-Kit">ComfyUI-Agent-Kit</a></td><td>Drives a local ComfyUI from coding agents, with an H3 skill.</td></tr>
<tr><td width="56"><img src="https://github.com/unknowlei.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unknowlei/minimax-h3-opencode-skills">minimax-h3-opencode-skills</a></td><td>OpenCode skills for directing, multi-shot planning, prompting and review.</td></tr>
<tr><td width="56"><img src="https://github.com/pytraveler.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI">MiniMax-H3-Prompt-Rewriter-ComfyUI</a></td><td>ComfyUI nodes for LightX2V's Prompt Rewriter LoRA, plus reference captioning.</td></tr>
<tr><td width="56"><img src="https://github.com/benjiyaya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/benjiyaya/ComfyUI-H3-VisionPromptor">ComfyUI-H3-VisionPromptor</a></td><td>Turns an idea and optional reference images into an H3 prompt with a local vision model.</td></tr>
<tr><td width="56"><img src="https://github.com/Rimagination.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Rimagination/h3lite">h3lite</a></td><td>Codex skill that picks a ComfyUI setup for your hardware and generates H3 video locally.</td></tr>
<tr><td width="56"><img src="https://github.com/benjiyaya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill">Minimax-H3-Prompt-AgentSkill</a></td><td>Builds a formatted H3 prompt from your media and idea, then hands it back for you to run.</td></tr>
<tr><td width="56"><img src="https://github.com/Adudeguyman.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder">ComfyUI-Fantastic-MiniMaxH3-PromptBuilder</a></td><td>Fillable prompt templates for every mode, checked against the official guide as you type.</td></tr>
<tr><td width="56"><img src="https://github.com/duckyshell.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer">ComfyUI-MiniMaxH3-Prompt-Writer</a></td><td>Prompt-writing panel in ComfyUI that turns a short brief and reference media into the official H3 format, using Ollama, a local GGUF or an API model.</td></tr>
<tr><td width="56"><img src="https://github.com/gnipbao.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gnipbao/minimax-h3-video-reverse-skill">minimax-h3-video-reverse-skill</a></td><td>Agent skill that reads a reference video and writes an H3 text-to-video prompt or per-shot image-to-video prompts.</td></tr>
<tr><td width="56"><img src="https://github.com/phileiny.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/phileiny/h3-storyboard-skill">h3-storyboard-skill</a></td><td>Claude Code skill that breaks a script into H3 shots and directs facial performance, marking each rule as tested or inferred.</td></tr>
<tr><td width="56"><img src="https://github.com/TFboy1.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TFboy1/oh-my-minimaxh3-director">oh-my-minimaxh3-director</a></td><td>Agent skill that splits a script into shots, batch-runs them through local ComfyUI workflows and assembles a JianYing draft.</td></tr>
<tr><td width="56"><img src="https://github.com/Sac-Y.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Sac-Y/MiniMax-H3-Cloud">MiniMax-H3-Cloud</a></td><td>Codex plugin that rents a GPU on the CompShare cloud, sets up ComfyUI, runs text, keyframe or reference-to-video jobs and shuts the instance down.</td></tr>
<tr><td width="56"><img src="https://github.com/jtydhr88.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jtydhr88/ComfyTV">ComfyTV</a></td><td>Canvas app for ComfyUI whose built-in agents ship with an H3 cinematic-director skill.</td></tr>
<tr><td width="56"><img src="https://github.com/zenstory-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zenstory-ai/drama-skills">drama-skills</a></td><td>Short-drama skills from novel to storyboard to prompts, with an H3 mode that binds reference images and speakers.</td></tr>
<tr><td width="56"><img src="https://github.com/eternityspring.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/eternityspring/shuohao-skills">shuohao-skills</a></td><td>Short-drama skills for character, art and screenplay bibles and storyboards, with the storyboard step aligning cuts and keyframes to H3 prompts.</td></tr>
<tr><td width="56"><img src="https://github.com/LoveRain1997.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LoveRain1997/h3-prompt-journal">h3-prompt-journal</a></td><td>Case studies of music-driven dance and multi-shot prompts, each with the full prompt and what made it work.</td></tr>
<tr><td width="56"><img src="https://github.com/SkyNotSilent.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SkyNotSilent/awesome-MiniMax-H3-cases">awesome-MiniMax-H3-cases</a></td><td>Over 3,000 playable H3 videos with their sources, over 700 complete prompts and guides for running H3 locally.</td></tr>
<tr><td width="56"><img src="https://github.com/BMB12d3.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BMB12d3/minimax-h3-prompt-composer">H3 Prompt Composer</a></td><td>Offline prompt builder in a single HTML file that runs in the browser.</td></tr>
<tr><td width="56"><img src="https://github.com/scenario-labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scenario-labs/skills">Scenario skills</a></td><td>Toolkit for agents on Scenario's API whose video skill covers H3 keyframe and reference modes and native audio prompts.</td></tr>
<tr><td width="56"><img src="https://github.com/ChatCut-Inc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ChatCut-Inc/agent-plugin">ChatCut Agent Plugin</a></td><td>AI video editor for coding agents whose video skill submits H3 and H3 Max jobs to its hosted service.</td></tr>
<tr><td width="56"><img src="https://github.com/zenstory-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zenstory-ai/oh-story-dsh">oh-story-dsh</a></td><td>DeepSeek Harness plugin for web novels and short dramas that renders video with H3 through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/LIUFelix2004.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LIUFelix2004/God-minmax-H3">God MiniMax H3</a></td><td>Prompt library and agent skill for 30-second H3 films built from chained segments, with API parameters for each prompt.</td></tr>
<tr><td width="56"><img src="https://github.com/10291124jcjc-dotcom.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/10291124jcjc-dotcom/ancient-rap-battle">ancient-rap-battle</a></td><td>Claude Code skill for rap battles between historical rivals, with an optional H3 lip-sync route on the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/suihe1.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/suihe1/short-drama-production">short-drama-production</a></td><td>Codex skill that runs a short drama from outline to rough cut and submits H3 jobs to the MiniMax API or CompShare.</td></tr>
<tr><td width="56"><img src="https://github.com/gary149.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gary149/h3-game-sprites">h3-game-sprites</a></td><td>Agent skill that turns OpenRouter H3 clips into 2D sprite sheets, one clip per move, keyed and checked frame by frame.</td></tr>
<tr><td width="56"><img src="https://github.com/guangjun5952.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/guangjun5952/minimax-h3-broll-generator">MiniMax H3 B-roll Generator</a></td><td>Codex skill that finds where a talk needs cutaways, writes H3 prompts and renders approved shots via MiniMax or Metaso.</td></tr>
<tr><td width="56"><img src="https://github.com/gooseworks-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gooseworks-ai/goose-skills">Goose Skills</a></td><td>Growth and marketing skills for coding agents, with a creator part that makes talking-creator clips with H3 Max on fal.</td></tr>
<tr><td width="56"><img src="https://github.com/makevoid.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/makevoid/motion-graphics-music-video-skill">Motion Graphics Music Video Skill</a></td><td>Claude plugin that turns a music track and a prompt into an animated video, with H3 Max image-to-video on fal.</td></tr>
</table>

<a id="h3-training"></a><a id="training"></a>

### Training

<table>
<tr><td width="56"><img src="https://github.com/IAmIronMan42.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/IAmIronMan42/MiniMax-H3-FineTuning">MiniMax-H3-FineTuning</a></td><td>Supervised LoRA training on the diffusers implementation, with audio in the loss.</td></tr>
<tr><td width="56"><img src="https://github.com/verl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/verl-project/verl-omni">verl-omni</a></td><td>RL post-training of H3 LoRAs with FlowGRPO and DiffusionNFT.</td></tr>
<tr><td width="56"><img src="https://github.com/radixark.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/radixark/miles_diffusion">miles_diffusion</a></td><td>Flow-GRPO post-training with an H3 text-to-audio-video recipe.</td></tr>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/Fizgig">Fizgig</a></td><td>LoRA and LoKr training studio with an H3 preset.</td></tr>
<tr><td width="56"><img src="https://github.com/kohya-ss.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kohya-ss/musubi-tuner">musubi-tuner</a></td><td>LoRA training scripts with experimental H3 support.</td></tr>
<tr><td width="56"><img src="https://github.com/tdrussell.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tdrussell/diffusion-pipe">diffusion-pipe</a></td><td>Pipeline-parallel trainer with H3 text-to-image and text-to-audio-video training.</td></tr>
<tr><td width="56"><img src="https://github.com/bghira.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/bghira/SimpleTuner">SimpleTuner</a></td><td>Fine-tuning kit for image, video and audio models, including H3.</td></tr>
<tr><td width="56"><img src="https://github.com/ByteDance-Seed.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ByteDance-Seed/VeOmni">VeOmni</a></td><td>ByteDance's training framework with FL2VA and Ref2VA configs for H3.</td></tr>
<tr><td width="56"><img src="https://github.com/Tencent-Hunyuan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Tencent-Hunyuan/UniRL">UniRL</a></td><td>Tencent Hunyuan's RL framework; H3 text-to-audio-video is supported on the training side only for now.</td></tr>
<tr><td width="56"><img src="https://github.com/X-GenGroup.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/X-GenGroup/Flow-Factory">Flow-Factory</a></td><td>Flow-matching fine-tuning framework with GRPO LoRA recipes for H3 text-, first/last-frame- and reference-to-audio-video.</td></tr>
</table>

<a id="h3-serving"></a><a id="partners"></a><a id="uniserve"></a><a id="hosted"></a><a id="dedicated"></a>

### Serving and hosting

<table>
<tr><td width="56"><img src="https://github.com/sgl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sgl-project/sglang">SGLang</a></td><td>Native H3 serving with a <a href="https://docs.sglang.ai/cookbook/diffusion/MiniMax/MiniMax-H3">cookbook</a> of recipes verified from one RTX 4090 to B300 and AMD MI355X.</td></tr>
<tr><td width="56"><img src="https://github.com/vllm-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vllm-project/vllm-omni">vLLM-Omni</a></td><td>OpenAI-compatible <code>/v1/videos</code> serving with an <a href="https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md">official H3 recipe</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/hao-ai-lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hao-ai-lab/uniserve">UniServe</a></td><td>HTTP serving for FastH3 and FastH3-Omni on 1 to 8 GPUs.</td></tr>
<tr><td width="56"><img src="https://github.com/EzgiTastan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/EzgiTastan/minimax-h3-gb300-ebpf">minimax-h3-gb300-ebpf</a></td><td>Reproducible GB300 benchmarking and tracing of SGLang H3 serving.</td></tr>
<tr><td width="56"><img src="https://github.com/fal-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://fal.ai">H3 Max on fal</a></td><td>Hosted H3 Max, Director and Turbo models, post-trained by fal; closed weights, also reachable from ComfyUI Partner Nodes.</td></tr>
<tr><td width="56"><img src="https://github.com/jd-nuva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://nuvalab.ai/minimax-h3-dedicated-deployment/">Nuva Lab</a></td><td>Dedicated H3 and FastH3 hosts for one company, including customer fine-tuned weights.</td></tr>
<tr><td width="56"><img src="https://github.com/fal-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://fal.ai/models/minimax/h3/text-to-video">MiniMax H3 on fal</a></td><td>H3 itself on fal's API, with text, <a href="https://fal.ai/models/minimax/h3/image-to-video">image</a> and <a href="https://fal.ai/models/minimax/h3/reference-to-video">reference</a> to video at up to 2K.</td></tr>
<tr><td width="56"><img src="https://github.com/replicate.png?size=96" alt="" width="48"></td><td width="240"><a href="https://replicate.com/minimax/h3">MiniMax H3 on Replicate</a></td><td>H3 on Replicate's prediction API, with text, first and last frame and reference to video at 768P or 2K.</td></tr>
<tr><td width="56"><img src="https://github.com/WaveSpeedAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://wavespeed.ai/models/wavespeed-ai/minimax-h3/text-to-video">MiniMax H3 on WaveSpeed</a></td><td>The open H3 weights served as an API, with text, image and reference to video.</td></tr>
<tr><td width="56"><img src="https://github.com/OpenRouterTeam.png?size=96" alt="" width="48"></td><td width="240"><a href="https://openrouter.ai/minimax/hailuo-3">H3 on OpenRouter</a></td><td>H3 and H3 Max video through OpenRouter's API.</td></tr>
<tr><td width="56"><img src="https://github.com/cloudflare.png?size=96" alt="" width="48"></td><td width="240"><a href="https://developers.cloudflare.com/ai/models/minimax/h3/">MiniMax H3 on Cloudflare</a></td><td>H3 in Cloudflare's AI model catalog, with text, first and last frame and reference to video.</td></tr>
</table>

<a id="h3-guides"></a><a id="models"></a><a id="recipes-vram"></a><a id="compat"></a><a id="compat-env"></a>

### Guides

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="docs/run-h3-locally.md">Run H3 locally</a></td><td>This repository's guide: VRAM recipes, every quantized file, components, compatibility and measured environments.</td></tr>
<tr><td width="56"><img src="https://github.com/jd-nuva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://nuvalab.ai/blog/run-minimax-h3-locally-gpu-vram-requirements">GPU and VRAM requirements</a></td><td>Nuva Lab's notes on GPUs and VRAM for local H3, with a <a href="https://nuvalab.ai/blog/minimax-h3-api-price-vs-self-hosting">cost comparison</a> against the API.</td></tr>
<tr><td width="56"><img src="https://github.com/matsuo-koya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matsuo-koya/minimax-h3-notes">minimax-h3-notes</a></td><td>Single-machine run notes.</td></tr>
<tr><td width="56"><img src="https://github.com/IvenKooLab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/IvenKooLab/minimax-h3-turing">minimax-h3-turing</a></td><td>Field handbook for 22 GB Turing cards with W4A8 and Turbo.</td></tr>
<tr><td width="56"><img src="https://github.com/FlowForgeLabAi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlowForgeLabAi/-h3-8gb-traps">-h3-8gb-traps</a></td><td>Measured 8 GB laptop path, with a reproduction for each trap.</td></tr>
</table>

## MiniMax-H3-Image

MiniMax-H3-Image is the open-weight image model of the H3 family, with text-to-image and multi-image editing in one checkpoint. It generates images of about 16 megapixels directly in 6 steps, at aspect ratios from 1:3 to 3:1, and takes up to 16 reference images.

<a id="img-official"></a>

### Official

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-H3-Image">MiniMax-H3-Image</a></td><td>Official repository: usage guide, showcase, local deployment and a prompt writing skill.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image">MiniMaxAI/MiniMax-H3-Image</a></td><td>Original weights for 16-megapixel image generation and editing with up to 16 reference images; also on <a href="https://modelscope.cn/models/MiniMax/MiniMax-H3-Image">ModelScope</a>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE">MiniMaxAI/MiniMax-H3-Image-PE</a></td><td>Optional prompt expander that turns short prompts into the detailed English the model expects.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630433fea362e7e8b51f5e2b/bsDc8lR72KhKvm-wXxYsv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Comfy-Org/MiniMax-H3-Image">Comfy-Org/MiniMax-H3-Image</a></td><td>ComfyUI repack for native support.</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image">ComfyUI tutorial</a></td><td>Official ComfyUI guide with <a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json">text-to-image</a> and <a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json">editing</a> templates; see also the <a href="https://blog.comfy.org/p/minimax-h3-image-is-now-available">launch post</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://platform.minimax.io/docs/guides/image-generation">MiniMax API</a></td><td>Official API for text-to-image, image editing and prompt expansion; for mainland China, use <a href="https://platform.minimax.cn/docs/guides/image-generation">platform.minimax.cn</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://hailuoai.video/">Hailuo AI</a></td><td>Use H3-Image in the browser or the <a href="https://agent.minimax.io/">desktop app</a>; for mainland China, use <a href="https://hailuoai.com/">hailuoai.com</a> and <a href="https://agent.minimax.cn">agent.minimax.cn</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_t2i.json">ComfyUI API node</a></td><td>Built-in node that generates and edits images with H3-Image through your Comfy account, with <a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_image_edit.json">an edit template</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/cli">MiniMax CLI</a></td><td>Official command line that generates images with H3-Image and edits them from up to 16 references.</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-MCP">MiniMax MCP</a></td><td>Official server that lets agents generate images with H3-Image by default and edit from up to 16 references.</td></tr>
</table>

<a id="img-quantizations"></a>

### Quantizations and conversions

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/unsloth/MiniMax-H3-Image-FP8">unsloth/MiniMax-H3-Image-FP8</a></td><td>INT8 ConvRot and FP8 DiT, 20.3 GB each, picked automatically by Unsloth Desktop.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/23f3af29b8e3f4317298458cafc54454.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LSDJesus/MiniMax-H3-Image-fp8-nvfp4">LSDJesus/MiniMax-H3-Image-fp8-nvfp4</a></td><td>FP8 scaled (21 GB) and NVFP4 (12.5 GB) for ComfyUI; <a href="https://huggingface.co/LSDJesus/MiniMax-H3-Image-GGUF">GGUF</a> from Q6_K to Q2_K.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF">Abiray/MiniMax-H3-Image-GGUF</a></td><td>Pruned-DiT GGUF from 6.88 GB with sensitive layers kept in FP32, for stock ComfyUI-GGUF.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a7d89f6560102d6b0e60442/wLhDigI2l39VFPC_OBAFK.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Hearthscale/MiniMax-H3-Image-GGUF">Hearthscale/MiniMax-H3-Image-GGUF</a></td><td>Q8_0, Q5_K_M, Q4_K_M and Q3_K_M for stable-diffusion.cpp.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69cf8fa5b7bee3980535206f/_wFUtjc_L4WrMgAdpDwKn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/AtomicChat/MiniMax-H3-Image-GGUF">AtomicChat/MiniMax-H3-Image-GGUF</a></td><td>GGUF tuned with a layer sensitivity scan, 7.6 to 21.4 GB, with published measurements.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/c88172b8c192ca50aa900233ae6d3c13.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lingjiu09/minimax-h3-image-gguf">lingjiu09/minimax-h3-image-gguf</a></td><td>Q3_K_M and Q4_0, tested on an 8 GB laptop.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6357da9b68659b265eb84aab/8PEgQsPKTJVQQD2_87mP9.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tsolful/Minimax_H3_Image_INT6_INT4GPTQ">tsolful/Minimax_H3_Image_INT6_INT4GPTQ</a></td><td>GPTQ W6A8 (15.6 GiB) and W4A8 (12.8 GiB) for ComfyUI.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/4d09b9d42626184db42666bbf1642ab1.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/speedyrulz/minimax-h3-image-int8-w4a8-hybrid">speedyrulz/minimax-h3-image-int8-w4a8-hybrid</a></td><td>14.8 GiB INT8 and W4A8 mix for 16 GB GPUs.</td></tr>
<tr><td width="56"><img src="https://www.gravatar.com/avatar/d651eb3d698c95ec23696040b7c7c6c3?d=retro&size=100" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/PotatoForge/Minimax-H3-Image-W4A4">PotatoForge/Minimax-H3-Image-W4A4</a></td><td>Experimental W4A4 at 11.35 GB for older GPUs.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8">WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8</a></td><td>W4A8 DiT and text encoder, 27.5 GB in total.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-Image-Pruned">WaveCut/MiniMax-H3-Image-Pruned</a></td><td>Unused weights removed, 135 GB down to 95 GB with bit-identical text-to-image output; <a href="https://github.com/iamwavecut/MiniMax-H3-Image-Pruned">build scripts</a>.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630cc06015433862cfc66a16/Wam_ZPEs8SLh5wtA7Xoms.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/SD-Next/MiniMax-H3-Image-SDNQ-uint4-svd-hadamard-dynamic">SD-Next SDNQ uint4</a></td><td>SDNQ 4-bit for diffusers and SD.Next; also <a href="https://huggingface.co/OzzyGT/minimax_h3_image_sdnq_dynamic_4bit">4-bit</a> and <a href="https://huggingface.co/OzzyGT/minimax_h3_image_sdnq_dynamic_8bit">8-bit</a> builds by OzzyGT.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65273adf77bceabaab23ee5c/6h4l-LlfXHKiqR6-15P8n.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/nativ-community/MiniMax-H3-Image-4bit">nativ-community/MiniMax-H3-Image-4bit</a></td><td>MLX 4-bit (54.5 GB) for mlx-vlm; an <a href="https://huggingface.co/nativ-community/MiniMax-H3-Image-8bit">8-bit build</a> is also available.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69bf5ffd2a6ad8b6410a5f3f/T6MxkTP8x6ZDrrQ5SMVG1.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/VisualInference/MiniMax-H3-Image-MLX-bf16">VisualInference/MiniMax-H3-Image-MLX-bf16</a></td><td>MLX BF16 for the minimax-h3-image-mlx runner, with <a href="https://huggingface.co/cguldogan/MiniMax-H3-Image-MLX-4bit">4-bit</a> and <a href="https://huggingface.co/cguldogan/MiniMax-H3-Image-MLX-8bit">8-bit</a> builds by cguldogan.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/635c610bfa3c172ddf40d116/pgN34290L5I0dhy0W-E8V.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Sawfwair/MiniMax-H3-Image-MLX-8bit">Sawfwair/MiniMax-H3-Image-MLX-8bit</a></td><td>MLX 8-bit export (52 GB) for mere.run.</td></tr>
</table>

<a id="img-adapters"></a>

### Distillation and adapters

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1674929746905-60f1abe7544c2adfd699860c.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step">minimax-h3-image-pdmd-2step</a></td><td>Two-step pDMD distillation LoRA, with a <a href="https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step">demo Space</a>.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/e2e8b6608dad261a033ecf26a75cbb05.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter">Fizgig H3-Image Training Adapter</a></td><td>Rank-16 adapter kept on while training LoRAs.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/64b90c0ce3d41dbd696b0bd0/O6FfUDzAildors6vyrvQK.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Felldude/MiniMaxImage_hdr_diffusers_vae">Felldude/MiniMaxImage_hdr_diffusers_vae</a></td><td>VAE with a fine-tuned decoder that trades pixel accuracy for more high-frequency detail, with metrics against the base VAE.</td></tr>
</table>

<a id="img-runtimes"></a>

### Runtimes

<table>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/ComfyUI">ComfyUI</a></td><td>Native support with official templates.</td></tr>
<tr><td width="56"><img src="https://github.com/sgl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sgl-project/sglang">SGLang</a></td><td>Serving from the main branch, with a <a href="https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-Image">cookbook</a> of recipes from DGX Spark to B300 and offload for 12 to 48 GB cards.</td></tr>
<tr><td width="56"><img src="https://github.com/vllm-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vllm-project/vllm-omni">vLLM-Omni</a></td><td>Serving with an official <a href="https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image">recipe</a>, including AMD Instinct.</td></tr>
<tr><td width="56"><img src="https://github.com/huggingface.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/huggingface/diffusers">diffusers</a></td><td>Modular Diffusers blocks for text-to-image and reference editing, released in v0.43.0; see the <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image">docs</a>.</td></tr>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/LightX2V">LightX2V</a></td><td>Text-to-image and editing with one or more references, on one GPU or several with TP/SP parallelism.</td></tr>
<tr><td width="56"><img src="https://github.com/modelscope.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a></td><td>Inference, low-VRAM inference, and full or LoRA training.</td></tr>
<tr><td width="56"><img src="https://github.com/leejet.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/leejet/stable-diffusion.cpp">stable-diffusion.cpp</a></td><td>C/C++ inference with day-0 H3-Image support and GGUF models.</td></tr>
<tr><td width="56"><img src="https://github.com/unslothai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unslothai/unsloth">Unsloth</a></td><td>Runs H3-Image in its desktop app and web UI with the INT8 or FP8 DiT.</td></tr>
<tr><td width="56"><img src="https://github.com/niw.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/niw/mmh3">mmh3</a></td><td>Rust engine with CUDA and Metal kernels.</td></tr>
<tr><td width="56"><img src="https://github.com/Anionex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Anionex/h3-image-mlx">h3-image-mlx</a></td><td>MLX generation and editing on M1 to M4 Macs with W8A8 weights, about 7 GB peak memory at 4K.</td></tr>
<tr><td width="56"><img src="https://github.com/felix-ab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/felix-ab/minimax-h3-image-mlx">minimax-h3-image-mlx</a></td><td>BF16 MLX port that reads each block from disk while the previous one computes; 5 GB of memory at 2048×2048.</td></tr>
<tr><td width="56"><img src="https://github.com/zhongkaifu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zhongkaifu/TensorSharp">TensorSharp</a></td><td>.NET engine that runs H3-Image text-to-image on ggml; editing is not implemented yet.</td></tr>
<tr><td width="56"><img src="https://github.com/animede.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/animede/minimax-h3-image-server">minimax-h3-image-server</a></td><td>Server with a browser UI for generation and reference edits; about 21 GB of VRAM by default, down to 8 GB with W4A8 and weight streaming.</td></tr>
<tr><td width="56"><img src="https://github.com/devin-lai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/devin-lai/MiniMax-H3-Image-Accel">MiniMax-H3-Image-Accel</a></td><td>SGLang tuned for L20 and RTX 5090, BF16 by default with optional FP8 for lower latency.</td></tr>
<tr><td width="56"><img src="https://github.com/Neywa-Labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Neywa-Labs/mlx-vlm">mlx-vlm</a></td><td>MLX text-to-image and image editing for H3-Image on Apple silicon, with BF16, 8-bit and 4-bit checkpoints.</td></tr>
</table>

<a id="img-apps"></a>

### Apps

<table>
<tr><td width="56"><img src="https://github.com/mcmonkeyprojects.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</a></td><td>Web UI that runs H3-Image with up to 16 prompt images for references or edits.</td></tr>
<tr><td width="56"><img src="https://github.com/tin2tin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tin2tin/Pallaidium">Pallaidium</a></td><td>Blender video editor add-on with H3-Image text-to-image and edits from up to 10 references, in SDNQ 4-bit for 24 GB GPUs.</td></tr>
<tr><td width="56"><img src="https://github.com/SHoogstad.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SHoogstad/diffusion-for-krita">diffusion-for-krita</a></td><td>Krita plugin on a ComfyUI backend that uses H3-Image to generate, inpaint and edit with up to 16 images.</td></tr>
<tr><td width="56"><img src="https://github.com/stepupgaming.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/stepupgaming/styk-paint">styk-paint</a></td><td>Godot add-on that has H3-Image repaint a 3D blockout view by view through ComfyUI, then projects the paint back onto the set.</td></tr>
<tr><td width="56"><img src="https://github.com/ryoppippi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ryoppippi/mmh3-studio">mmh3-studio</a></td><td>macOS and browser client for a resident mmh3 server, so generations skip the cold start.</td></tr>
<tr><td width="56"><img src="https://github.com/BesianSherifaj-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BesianSherifaj-AI/h3-image-editor">h3-image-editor</a></td><td>Windows workspace on ComfyUI for H3-Image generation, reference edits and masked repair, with Qwen, SAM and SeedVR2 tools.</td></tr>
<tr><td width="56"><img src="https://github.com/MrJackSpade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MrJackSpade/ImageGenerationServer">ImageGenerationServer</a></td><td>Multi-user web front end for a local ComfyUI, with H3-Image generation and edits from up to 16 pictures; non-commercial license.</td></tr>
<tr><td width="56"><img src="https://github.com/tong-io.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tong-io/tongflow-modal-minimax-h3-image">tongflow-modal-minimax-h3-image</a></td><td>TongFlow plugin that runs H3-Image generation, editing and fusion of up to 16 references on Modal.</td></tr>
</table>

<a id="img-api"></a>

### API integrations

<table>
<tr><td width="56"><img src="https://github.com/heygen-com.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/heygen-com/hyperframes">HyperFrames</a></td><td>HTML-to-video renderer for agents whose media command can make 2K or 4K images with H3-Image through HeyGen.</td></tr>
<tr><td width="56"><img src="https://github.com/vercel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vercel/ai">AI SDK</a></td><td>Vercel's TypeScript toolkit whose MiniMax provider generates and edits images with H3-Image and its Highspeed version.</td></tr>
<tr><td width="56"><img src="https://github.com/looplj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/looplj/reveric">Reveric</a></td><td>Desktop studio for novels and short dramas that creates and edits images with H3-Image through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/langgenius.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/langgenius/dify-official-plugins/tree/main/tools/minimax_image">Dify MiniMax Image plugin</a></td><td>Tool for Dify apps that generates images with H3-Image and edits from 1 to 16 references through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/ThinkFlowLab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ThinkFlowLab/OpenDirector">OpenDirector</a></td><td>Agent video workflow tool with an H3-Image adapter that edits from up to 16 references through the MiniMax API.</td></tr>
<tr><td width="56"><img src="https://github.com/OctaFuse.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OctaFuse/octafuse-gateway">Octafuse Gateway</a></td><td>Self-hosted API hub with presets for H3-Image generation and edits on MiniMax's OpenAI-compatible Images API.</td></tr>
</table>

<a id="img-comfyui"></a>

### ComfyUI nodes and workflows

<table>
<tr><td width="56"><img src="https://github.com/Rinne414.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Rinne414/ComfyUI-MiniMaxH3-Tools">ComfyUI-MiniMaxH3-Tools</a></td><td>Nodes that free the text encoder before the DiT loads, so H3-Image runs on a 24 GB GPU with 32 GB of RAM.</td></tr>
<tr><td width="56"><img src="https://github.com/nkxx188.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy">ComfyUI-MiniMaxH3-Easy</a></td><td>Its Easy Image node does H3-Image text-to-image and reference edits; type @ in the prompt to insert a reference.</td></tr>
<tr><td width="56"><img src="https://github.com/pxyllvr.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pxyllvr/ComfyUI-H3Image-MediaLoader">ComfyUI-H3Image-MediaLoader</a></td><td>Loads up to 16 reference images with per-slot crop, rotate and reorder.</td></tr>
<tr><td width="56"><img src="https://github.com/DeepNeuralNerd.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/DeepNeuralNerd/H3-Video-LoRA-Loader-for-Image">H3-Video-LoRA-Loader-for-Image</a></td><td>Runs H3 video LoRAs on H3-Image.</td></tr>
<tr><td width="56"><img src="https://github.com/KadoBOT.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio">ComfyUI-H3-Continuity-Studio</a></td><td>Node pack for multi-clip H3 video, with an H3-Image node that makes the still frame guiding each shot.</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8">comfyui-minimax-h3-prompt-enhancer-T8</a></td><td>Prompt enhancement nodes, with H3-Image support since v1.41.0.</td></tr>
<tr><td width="56"><img src="https://github.com/iamwavecut.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/iamwavecut/ComfyUI-OrbitQuant">ComfyUI-OrbitQuant</a></td><td>Runs the OrbitQuant W4A8 build for generation and edits with up to 16 references.</td></tr>
<tr><td width="56"><img src="https://github.com/Hejunlong27.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Hejunlong27/H3GIF">H3GIF</a></td><td>Has H3-Image draw a storyboard grid in one image, then splits it into frames for a looping GIF.</td></tr>
<tr><td width="56"><img src="https://github.com/BuffMcBigHuge.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BuffMcBigHuge/Comfiest">Comfiest</a></td><td>Agent-driven ComfyUI workflow suite with H3-Image text-to-image and edit graphs for 1, 2 or 4 references.</td></tr>
<tr><td width="56"><img src="https://github.com/kuhung.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kuhung/minimax-h3-colab-workflow">minimax-h3-colab-workflow</a></td><td>Colab notebooks that run H3 and H3-Image from open weights on an L4, through ComfyUI quants and stable-diffusion.cpp.</td></tr>
<tr><td width="56"><img src="https://github.com/Timster666.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Timster666/comfyui-minimax-image">comfyui-minimax-image</a></td><td>RunPod template that boots into ComfyUI with the H3-Image models.</td></tr>
<tr><td width="56"><img src="https://github.com/yilives.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yilives/minimax-h3-v100-guide">minimax-h3-v100-guide</a></td><td>Tested setup for H3 and H3-Image on one 16 GB V100 with 32 GB of RAM: configs, launchers and workflows.</td></tr>
</table>

<a id="img-training"></a>

### Training

<table>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/Fizgig">Fizgig</a></td><td>LoRA and fine-tune studio with H3-Image support since v9.6.</td></tr>
<tr><td width="56"><img src="https://github.com/ostris.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ostris/ai-toolkit">ai-toolkit</a></td><td>GUI and CLI trainer; trains H3-Image editing when the dataset has control images.</td></tr>
</table>

<a id="img-prompting"></a>

### Prompting

<table>
<tr><td width="56"><img src="https://github.com/AI-KSK.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AI-KSK/minimax-h3-image-prompts">minimax-h3-image-prompts</a></td><td>Reusable skill for writing generation and editing prompts with up to 16 references.</td></tr>
<tr><td width="56"><img src="https://github.com/BeatAPI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BeatAPI/awesome-minimax-h3-image-prompts">awesome-minimax-h3-image-prompts</a></td><td>63 source-linked H3-Image examples with result images and editing references; only 5 include the prompt itself.</td></tr>
</table>

<a id="img-benchmarks"></a>

### Benchmarks

<table>
<tr><td width="56"><img src="https://github.com/jo-nike.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jo-nike/h3-image-bench">h3-image-bench</a></td><td>600 graded prompts in two rounds at about 16 MP on a 16 GB RTX 5080, with a live results page.</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69c9b513833ed97d9b2f808d/xDH70MJgqXXRZn1ekMirh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench">M5 Pro Mac mini benchmark</a></td><td>ComfyUI INT8 timings on a 64 GB M5 Pro Mac mini: about 9 minutes per 4096×4096 image.</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/9947803a52252be0e6db004495bb2a75.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/datasets/LIOSAM007/h3-image-12gb-tests">12 GB tests</a></td><td>INT8 on a 12 GB RTX 4070 SUPER with 96 GB of RAM: 11 to 12 s at 4 MP, about 41 s at 16 MP.</td></tr>
</table>

## Contributing

Add one row to the matching table: the owner's avatar, the name linked to the project, and one sentence on what it is. Add the same row to `README_cn.md`, then run `python3 scripts/check_readme.py`.

An entry needs its own README, docs or model card that says what it is and that it uses MiniMax H3 or MiniMax-H3-Image, locally or through an API. Measurements, VRAM recipes and per-file details belong in [docs/run-h3-locally.md](docs/run-h3-locally.md).

<a id="credits"></a>

## Acknowledgements

This index is only possible because other people made the models, tools, tests, and documentation it points to. Thank you to the MiniMax, ComfyUI, SGLang, vLLM, NVIDIA, and Unsloth teams, and to the independent maintainers who keep testing H3 on real hardware.

The structure and much of the resource discovery come from the community-maintained [`wildminder/awesome-minimax-H3`](https://github.com/wildminder/awesome-minimax-H3). This page follows that work and adds MiniMax's own GitHub and Hugging Face scan.

More specific thanks go to:

* [Comfy-Org](https://huggingface.co/Comfy-Org) for the official ComfyUI conversions, workflow templates, and day-one support.
* [ModelTC / LightX2V](https://github.com/ModelTC/LightX2V) for the Turbo distillation work and public training configuration.
* [`Larryvrh`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) for the checkpoint comparisons behind the Turbo guidance.
* [`Kijai`](https://github.com/kijai/ComfyUI-SolAttn_triton) for the NVIDIA Sol-Attn implementation and benchmark notes.
* [Alibaba PAI](https://huggingface.co/alibaba-pai) and [`aigc-apps/VideoX-Fun`](https://github.com/aigc-apps/VideoX-Fun) for the PDD 8-step Acc LoRAs and the Fun ControlNet-Union branch.
* [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3) for publishing Video Delta Net's training code alongside the weights.
* [FastVideo](https://huggingface.co/FastVideo) for the FastH3 line (4-step preview, V2, Trim and Omni), its data-free distillation recipe, and UniServe.
* [Nuva Lab](https://nuvalab.ai/fasth3/) for the production grounding behind FastH3, the FastH3-Omni co-development and reference dataset, and the production hardware behind the UniServe benchmarks.
* [Pika Labs](https://github.com/Pika-Labs/Pika-H3-Flex) for the six-step Flex adapters and inference examples; [NVIDIA](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K) for the released draft refiner and fast decoder, building on LBH, LightX2V and LMS components.
* [`IAmIronMan42`](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning) for the training work and the documented fixes.
* [Salvatore Sanfilippo (antirez)](https://github.com/antirez) for the standalone H3 C/Metal inference engine.
* Every quantizer and workflow maintainer represented above. Their files, testing time, and write-ups make local H3 use much easier.

If you spot a wrong number, broken link, or missing compatibility note, please open an issue or send a correction.

## Contact

For MiniMax H3 questions, contact [model@minimax.io](mailto:model@minimax.io).


---

## FILE: README_cn.md (https://github.com/MiniMax-AI/awesome-minimax-h3-integration/blob/HEAD/README_cn.md)

<p align="center">
  <a href="https://github.com/MiniMax-AI/MiniMax-H3"><img src="https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/assets/minimax-h3-header.gif" alt="MiniMax H3"></a>
</p>

# MiniMax H3 Integrations

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) 和 [MiniMax-H3-Image](https://github.com/MiniMax-AI/MiniMax-H3-Image) 的模型、工具和项目索引，由社区共同维护。

[English](README.md) | 简体中文

## 目录

- [MiniMax H3](#minimax-h3)
  - [官方资源](#h3-official)
  - [量化](#h3-quantizations)
  - [组件](#h3-components)
  - [加速与蒸馏](#h3-acceleration)
  - [LoRA 与微调](#h3-loras)
  - [推理引擎](#h3-runtimes)
  - [应用与工作室](#h3-apps)
  - [API 集成](#h3-api)
  - [ComfyUI 节点](#h3-comfyui)
  - [工作流](#h3-workflows)
  - [提示词与 Agent 技能](#h3-prompting)
  - [训练](#h3-training)
  - [部署与托管](#h3-serving)
  - [指南](#h3-guides)
- [MiniMax-H3-Image](#minimax-h3-image)
  - [官方资源](#img-official)
  - [量化与格式转换](#img-quantizations)
  - [蒸馏与适配器](#img-adapters)
  - [推理引擎](#img-runtimes)
  - [应用](#img-apps)
  - [API 集成](#img-api)
  - [ComfyUI 节点与工作流](#img-comfyui)
  - [训练](#img-training)
  - [提示词](#img-prompting)
  - [评测](#img-benchmarks)
- [贡献](#贡献)

## MiniMax H3

MiniMax H3 是 MiniMax 开放权重的音视频生成模型：以文字、图片、视频和音频为上下文，生成 4 到 15 秒、最高 2K、带原生立体声的视频。它有两个版本：FL2VA 根据文字和可选的首尾帧生成，Ref2VA 最多可参考 9 张图片、3 段视频和 3 段音频。

<a id="h3-official"></a>

### 官方资源

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-H3">MiniMax-H3</a></td><td>官方仓库：参考实现和提示词指南。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMaxAI/MiniMax-H3</a></td><td>原始权重，包括 FL2VA（首尾帧）和 Ref2VA（多参考）两个版本，以及文本编码器、视频 VAE 和音频 VAE。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630433fea362e7e8b51f5e2b/bsDc8lR72KhKvm-wXxYsv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Comfy-Org/MiniMax-H3">Comfy-Org/MiniMax-H3</a></td><td>ComfyUI 单文件权重，包括剪枝后的 BF16、FP8 和 INT8 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/video/minimax/minimax-h3">ComfyUI tutorial</a></td><td>ComfyUI 官方 H3 教程，见 <a href="https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui">首发公告</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json">ComfyUI workflow templates</a></td><td>ComfyUI 自带的官方模板：文生视频、<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json">图生视频</a>和<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json">参考生视频</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md">Video prompt writing guides</a></td><td>官方提示词指南：基础模式（FL2VA）和<a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md">参考模式（Ref2VA）</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://platform.minimax.io/docs/api-reference/video-generation-v2-create">MiniMax API</a></td><td>官方视频生成 API；中国大陆请用 <a href="https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create">platform.minimaxi.com</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://hailuoai.video/tools/minimax-h3">Hailuo AI</a></td><td>在网页上或<a href="https://hub.minimax.io/">桌面应用</a>里直接使用 H3；中国大陆请用 <a href="https://hailuoai.com/">hailuoai.com</a> 和 <a href="https://hub.minimaxi.com/">hub.minimaxi.com</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/partner-nodes/minimax/minimax-h3">ComfyUI API nodes</a></td><td>ComfyUI 自带的 API 节点，支持文生视频、首尾帧和参考生视频，通过 Comfy 账号调用 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/cli">MiniMax CLI</a></td><td>MiniMax API 的官方命令行工具，可直接生成 H3 视频。</td></tr>
</table>

<a id="h3-quantizations"></a>

### 量化

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/unsloth/MiniMax-H3-GGUF">unsloth/MiniMax-H3-GGUF</a></td><td>剪枝版 FL2VA 和 Ref2VA 的 GGUF，从 Q2_K（6.26 GiB）到 Q8_0，另有 Qwen3-VL 文本编码器的 GGUF。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/Yy9XdzixLpDyVyN4fUvP5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants">DmitryDB/MiniMax-H3-ComfyUI-Quants</a></td><td>FL2VA 和 Ref2VA 的 INT8 ConvRot 和 NVFP4 版本，10.86 到 21.91 GiB，附 VAE 文件。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/Yy9XdzixLpDyVyN4fUvP5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV">DmitryDB/MiniMax-H3-DynTime-sQKV</a></td><td>INT8 DT-sQKV 版本，需要配合该仓库的 ComfyUI 运行时补丁。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63150d9f8be84e1ab412890e/slVOxsRI070drG6kICymD.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot">rockerBOO/minimax-h3-nvfp4-convrot</a></td><td>面向 Blackwell 显卡的 NVFP4 FL2VA 和 Ref2VA，推荐 INT8 ConvRot 版；另有给其他显卡用的实验性 INT4 ConvRot FL2VA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot">Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot</a></td><td>剪枝版 NVFP4（11.67 GiB）、INT8 ConvRot 和 INT4/INT8 混合 ConvRot 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6357da9b68659b265eb84aab/8PEgQsPKTJVQQD2_87mP9.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot">tsolful/Minimax_H3_INT4MixedConvRot</a></td><td>FL2VA 和 Ref2VA 的剪枝版 INT4/INT8 混合 ConvRot。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/b3c31e18f3b7cc1c5c059058221195dc.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot">Merserk/MiniMax-H3-INT4-ConvRot</a></td><td>INT4 ConvRot，完整版 16.67 GiB，剪枝版 10.56 GiB。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/357a2cb738d2a9ad5c01dd6296792c22.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot">AX1Y2JP/MiniMax-H3-W4A8-ConvRot</a></td><td>剪枝版 W4A8 ConvRot DiT（11.68 GiB）和 W4A8 文本编码器。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3-experimental">Kijai/MiniMax-H3-experimental</a></td><td>实验性的 W4A8 混合精度 DiT 和 INT8 ConvRot 视频 VAE。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62c3f1dd73c3ba0c801033ce/3STHHoSpTLGI-Alcpf7Cg.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI">Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI</a></td><td>剪枝版 W4A8 ConvRot DiT，附配套的量化文本编码器。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/644717ea27a3e9e0b7c7207a/aNn8akdZfegf46fkZ4R9h.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn">rzgar/minimax_h3_fl2va_fp8_e4m3fn</a></td><td>FL2VA 的 FP8 E4M3FN、MXFP8，以及 FP8 加 FP16 注意力版本。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/6701ed70ea8cd899201c524ab3ccac8c.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/abakanai/Minimax_h3_hybrid">abakanai/Minimax_h3_hybrid</a></td><td>面向 Blackwell 显卡的混合 NVFP4 Ref2VA 版本，14.03 和 16.38 GiB。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672c8f0070734f59f146b73c/5lGMq-_Patlincxq97jf5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4">DiffSynth-Studio/MiniMax-H3-NF4</a></td><td>给 DiffSynth-Studio 用的 NF4 DiT、文本编码器和 VAE，8 GB 显存起步。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4">WaveCut/MiniMax-H3-OrbitQuant-W4A4</a></td><td>W4A4 版本，附量化文本编码器；需要 ComfyUI-OrbitQuant 节点。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-GGUF">Abiray/MiniMax-H3-GGUF</a></td><td>完整版 FL2VA 和 Ref2VA 的 GGUF，Q3_K_S 到 Q8_0，附文本编码器量化和工作流。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF">Abiray/MiniMax-H3-Pruned-GGUF</a></td><td>剪枝版 FL2VA 的 GGUF，Q3_K_M 和 Q8_0。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68761990332d15464ccc8dee/MaE3RLtzT8Q3tQ5I-n38s.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/realrebelai/MiniMax-H3_GGUFs">realrebelai/MiniMax-H3_GGUFs</a></td><td>FL2VA 和 Ref2VA 的 Q3_K_M、Q4_K_M GGUF，另有 Q2_K 和 Q4_K_M 文本编码器，Q2_K 只有 7.91 GiB。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67b808c770117114270d902e/dLdwkNZowSMsqTZ3Znfg2.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF">vantagewithai/MiniMax-H3-comfyUI-GGUF</a></td><td>给 ComfyUI 用的完整版 FL2VA 和 Ref2VA GGUF，Q3_K_M 到 Q8_0。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/2bfb8e5718e39dd319e9320abab557c1.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF">MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF</a></td><td>IQ1_S（3.78 GiB）和 IQ1_M 版本，是目前最小的 DiT。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/638a3f5a8d7cce863e70a91b/Z_KI0kdABbpRkXV51JpOG.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/molbal/MiniMax-H3-GGUF">molbal/MiniMax-H3-GGUF</a></td><td>剪枝版 Ref2VA 的 GGUF，包括手工调过的混合精度布局。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/92b2fbafe275bf1c126d5a687d6a7f2f.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/leejet/MiniMax-H3-GGUF">leejet/MiniMax-H3-GGUF</a></td><td>剪枝版 FL2VA 的 GGUF，Q2_K 和 Q4_K_M。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF">joeygambino/MiniMax-H3-curve-GGUF</a></td><td>剪枝版 FL2VA 和 Ref2VA 的 GGUF，Q4_0、Q5_1 和 Q8_0；需要给 ComfyUI-GGUF 打补丁的 H3-Multishot 节点包。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68a41bb168b64003eab9cfa4/Gq5_H6FGLr1k3hSsshdfa.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4">jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4</a></td><td>smhfacct 的 FL2VA 与 Ref2VA 混合方案的 NVFP4 版本，四种层段组合，每个 11.67 GiB，面向 Blackwell 显卡。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/cac9874103e7c159dc5eb6ec977f3ceb.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4">lilcheaty/MiniMax-H3-NVFP4</a></td><td>剪枝版 FL2VA 和 Ref2VA 的 NVFP4 版本，各 11.67 GiB，另有完整和混合精度的 Ref2VA 版本，面向 Blackwell 显卡。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/42b9a338c1a323b57ce3b506de76f8ae.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DeepBeepMeep/MiniMax-H3">DeepBeepMeep/MiniMax-H3</a></td><td>WanGP 使用的 H3 系列权重：FL2VA、Ref2VA、FastH3 Omni、HyperFlow Flash、Fun ControlNet 和 H3-Image，有 BF16 和 INT8 版本。</td></tr>
</table>

<a id="h3-components"></a>

### 组件

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1650342890988-623c6253389748c9f72ca287.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union">alibaba-pai/MiniMax-H3-Fun-Controlnet-Union</a></td><td>一个控制分支同时支持 Canny、深度、HED、MLSD、姿态控制和视频局部重绘。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/NCer_WlbHMkH1ZU8Q_I8S.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lihaoyun6/MiniMax-H3-Ref-Patch">lihaoyun6/MiniMax-H3-Ref-Patch</a></td><td>148 MiB 的补丁，让 FL2VA 更接近 Ref2VA 的效果；需要配套的 ComfyUI 节点。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3-TAE">Kijai/MiniMax-H3-TAE</a></td><td>9 MiB 的小型自编码器，用于 ComfyUI-KJNodes 里的预览。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/2a4082bfe6d2d4d206dbd655197388f0.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE">Mamad8/MiniMax-H3-Image-VAE</a></td><td>实验性的单帧 VAE，用视频模型出静态图；生成视频时仍用原版 VAE。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67c576e958e13035e6e4b0d5/mECPgQ0bhniFavYJPDpV4.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler">LBH-123-AI/Minimax_h3_latent_Upscaler</a></td><td>学习型潜空间放大模型，1.0 到 4.0 倍，配合它的 ComfyUI 节点使用。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62b4b5beb25cb80fcf278354/SIddx3hYu5rWXA4-O3Oaj.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K">Open-H3-Regenerate-2K</a></td><td>NVIDIA 的三步精修模型，把低分辨率的 H3 草稿提升到 2560x1440，附<a href="https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K">代码</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6767c79846bb6707b92e5fb2/lqjSqaUQW3WEMVUG6KFu_.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ZawShiShawn/Gestura-MiniMax-H3-VAE-LiteRT">ZawShiShawn/Gestura-MiniMax-H3-VAE-LiteRT</a></td><td>LiteRT 格式的 H3 VAE，给在安卓手机上本地生成视频的 Gestura 应用使用。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/7ddf7fcf26e8da8d32da7a3c4e73af52.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE">speach1sdef178/MiniMax-H3-X2-Detail-VAE</a></td><td>解码出两倍分辨率的视频 VAE；需要 ComfyUI-MiniMaxH3_LatentUpscaler 里的快速 VAE 节点。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62b4b5beb25cb80fcf278354/SIddx3hYu5rWXA4-O3Oaj.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC">Efficient-Large-Model/Open-H3-Regenerate-2K-IC</a></td><td>NVIDIA 精修模型的上下文 LoRA 版本，借 Ref2VA 把 768p 草稿重新生成为 1080p、2K 或 4K。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF">joeygambino/MiniMax-H3-encoder-GGUF</a></td><td>Qwen3-VL-32B 文本编码器的 GGUF 版本，给 ComfyUI 用。</td></tr>
</table>

<a id="h3-acceleration"></a>

### 加速与蒸馏

<table>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/Minimax-H3-Turbo">Minimax-H3-Turbo</a></td><td>LightX2V 的 4 步和 8 步蒸馏 LoRA，<a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo">权重在 Hugging Face</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63783918d925e7dad20748a3/SIMpPx2nIKyEF7Mx9E3hC.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora">larryvrh/MiniMax-H3-Turbo-Lora</a></td><td>社区重训的 4 步 Turbo LoRA，常用默认版本是 <code>v4_step600_ema</code>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63473b59e5c0717e6737b872/e0rwJa886MD1hmdeFr9Wh.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI">drbaph/MiniMax-H3-Turbo-Lora-ComfyUI</a></td><td>适配 ComfyUI 剪枝版权重的 Turbo LoRA，包括压到 rank 21 的版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI">Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI</a></td><td>适配剪枝版 ComfyUI 权重的 Turbo LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3_comfy">Kijai/MiniMax-H3_comfy</a></td><td>ComfyUI 格式的 LightX2V Turbo LoRA，完整版和 rank 21 版。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/637745113a63a2983ffbde13/wZKoPSWQKpe2JpFsX3nLa.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers">InstantX/MiniMax-H3-Turbo-Lora-Diffusers</a></td><td>diffusers 格式的 4 步 Turbo LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6864d1f250596cb028f66be3/TOr9t2xU3PJFjSXGQvTYK.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joyfox/MiniMax-H3-Turbo">joyfox/MiniMax-H3-Turbo</a></td><td>4 步 FL2VA LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/674ef761b783a2a16788d7fe/cxPCVoNPpnehsDwy3bg14.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA">Tutu 20-to-8 NFE LoRA</a></td><td>把 20 次推理压到 8 次的音视频 LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp">t8star/minimax-h3-4step-turbo-loras-comfyui-exp</a></td><td>实验性的 INT8 ConvRot Turbo LoRA，给 ComfyUI 用。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/693c433f38397e97c1c9fb78/VNwN7gvr2RGpxeBUIHNqT.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf">infosave/MiniMax-H3-Turbo-cmf</a></td><td>CMF 格式的 Turbo 权重，Q4TP 和 Q2TP。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1650342890988-623c6253389748c9f72ca287.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs">alibaba-pai/MiniMax-H3-Acc-LoRAs</a></td><td>PDD 8 步加速 LoRA，FL2VA 和 Ref2VA 各一个，新版 ComfyUI 可以直接加载。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/1af3b10983f16ee56a7ad7b48c1e5df7.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/aptech0081/MiniMax-H3-PDD-ComfyUI">aptech0081/MiniMax-H3-PDD-ComfyUI</a></td><td>转成 ComfyUI 格式的 PDD LoRA，另有合入 PDD 的 INT8 Ref2VA 权重。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy">t8star/Minimax-H3-Super-Acceleration-Comfy</a></td><td>为 T8 音频节点重新打包的 PDD LoRA。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/ce52788c884be3d7654e823007579f5a.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8">1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8</a></td><td>已经合入 PDD 8 步 LoRA 的 SVDQuant FP4 版本，另有 <a href="https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-int4-pdd8">INT4 版本</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree">FastH3 4-step Preview</a></td><td>FastVideo 的 4 步免数据蒸馏权重，带稀疏注意力；也有 <a href="https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA">LoRA 版</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2">FastH3 8-Step V2</a></td><td>8 步文生音视频权重，稀疏注意力的稀疏度为 80%。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4">FastH3 Trim NVFP4</a></td><td>实验性的 8 步模型，保留 50 层中的 42 层，NVFP4 共 11.1 GiB，面向 Blackwell 和 DGX Spark。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1">FastH3-Omni 8-Step</a></td><td>8 步的多参考音视频生成，可用图片、视频和音频作参考。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy">FastVideo/FastVideo-FastH3-Comfy</a></td><td>FastH3 8 步 V2 的 ComfyUI 打包版。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68761990332d15464ccc8dee/MaE3RLtzT8Q3tQ5I-n38s.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/realrebelai/FastH3_GGUFs">realrebelai/FastH3_GGUFs</a></td><td>FastH3 4 步的 GGUF 转换版。</td></tr>
<tr><td width="56"><img src="https://github.com/Pika-Labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Pika-Labs/Pika-H3-Flex">Pika H3 Flex</a></td><td>Pika 的 6 步 FL2VA 和 Ref2VA 适配器，原生输出 480p 和 768p，附命令行、diffusers 和 ComfyUI 示例。</td></tr>
<tr><td width="56"><img src="https://github.com/vita-epfl.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vita-epfl/PTD-inference/tree/release/comfyui-inference">PTD</a></td><td>EPFL 的 4 步和 6 步轨迹蒸馏，附<a href="https://huggingface.co/epfl-vita/PTD-weights">权重</a>和 ComfyUI 集成。</td></tr>
<tr><td width="56"><img src="https://github.com/Yzmblog.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Yzmblog/DMAD">DMAD</a></td><td>4 步和 8 步文生音视频 LoRA，附<a href="https://huggingface.co/ZhengmingYu/DMAD">权重</a>、工作流和训练代码。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenVDN.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenVDN/vdn-minimax-h3">VDN</a></td><td>Video Delta Net：线性和 softmax 混合注意力，每步快 2.6 到 2.9 倍，附<a href="https://huggingface.co/OpenVDN/vdn-minimax-h3">权重</a>和训练代码。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63473b59e5c0717e6737b872/e0rwJa886MD1hmdeFr9Wh.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui">drbaph/vdn-minimax-h3-int8-convrot-comfyui</a></td><td>ComfyUI 格式的 INT8 ConvRot VDN。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy">t8star/Vdn-Minimax-H3-Comfy</a></td><td>配合 T8 节点的 VDN 模型包，放好即可用。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/66213d7f118a0a5cef880b5e/hr3ia85xax2nyW0OO0F7D.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA">RAVEN Streaming LoRA</a></td><td>4 步预览版，把 H3 变成逐段生成的流式模型，配合 <a href="https://github.com/mvp-ai-lab/RAVEN">RAVEN</a> 代码使用。</td></tr>
<tr><td width="56"><img src="https://www.gravatar.com/avatar/7b0abe8d2e4aff190133f25f13729f6b?d=retro&size=100" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot">MATLOWAI/minimax-h3-fused-turbo-int8-convrot</a></td><td>单个 21 GB 的 INT8 ConvRot 文件，已合入 Turbo 和运动 LoRA，在 ComfyUI 里 4 步完成文生视频、图生视频和 Ref2VA。</td></tr>
<tr><td width="56"><img src="https://github.com/Video-Rebirth.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Video-Rebirth/hyperflow-h3-flash">HyperFlow H3 Flash</a></td><td>Video Rebirth 压缩到 13.7B 的 H3，4 到 8 步生成带立体声的视频，有 INT8 和 GGUF <a href="https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash">权重</a>；另有给完整版 H3 用的 <a href="https://huggingface.co/VideoRebirthAI/hyperflow">8 步 LoRA</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/TaoLiveAIGC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TaoLiveAIGC/TaoMate-H3">TaoMate-H3</a></td><td>让 H3 三步流式生成，附<a href="https://huggingface.co/TaoLiveAIGC/TaoMate-H3">权重</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/thu-ml.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thu-ml/TurboDiffusion">TurboDiffusion</a></td><td>面向 H3 FL2VA 的 W8A8 算子、采样器和多卡运行时，附 TurboH3-FL2VA-I2VA 权重。</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/LongLive">LongLive</a></td><td>NVIDIA 的 LongLive-Plug 蒸馏，附 H3 的<a href="https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step">少步数</a>和 <a href="https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-cfg">CFG</a> 适配器。</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/FastGen">FastGen</a></td><td>NVIDIA 的蒸馏框架，支持的音视频模型里有 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/vipshop.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vipshop/cache-dit">cache-dit</a></td><td>diffusers 的缓存加速和并行推理，支持 H3 的 T2VA、FL2VA 和 Ref2VA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6568921511b2bbd6c2e9841c/v-pfaTZmvAiLeBjS2-eDb.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview">Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview</a></td><td>学习型稀疏注意力，各模式下只保留约 10% 的注意力块，另有专门微调的 <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview">R2VA 预测器</a>；可在 ComfyUI 或 Miowtion 里运行。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680de13385293771bc57400b/qnjz3avU_cKzRZA2aGjuZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA">lightx2v/Minimax-h3-Turbo-SLA</a></td><td>4 步 FL2VA LoRA，加入稀疏度 85% 的稀疏线性注意力，在 LightX2V 里用 RTX 5090 约快 2.5 倍，另有 ComfyUI 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6ad5e8ef64cf8fd69bf5f843/28JY9Lma4ZOqJ8GgMZduD.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs">taobao-content-ai/MiniMax-H3-PTD-LoRAs</a></td><td>每一步各用一个 PTD LoRA，让 Ref2VA 4 步或 8 步出片，附 diffusers 代码和 ComfyUI 节点，节点需要 ComfyUI-PTD 和未剪枝的 Ref2VA 模型。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/f72b4505394ad4d2258dbba49ade6e85.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/pdmd2026/pdmd_4NFE_lora">pdmd2026/pdmd_4NFE_lora</a></td><td>用投影分布匹配蒸馏出的 4 步 LoRA，rank 128，diffusers 键名格式。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a550d1dc84594e3ada10828/BLGmI3tqOxQmO1ti1rxUH.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF">ChrisColeTech/minimax-h3-turbo-GGUF</a></td><td>合入 LightX2V Turbo LoRA 的剪枝版 FL2VA 和 Ref2VA GGUF，另附配合 ComfyUI-ClipProj 的 4B 和 8B 文本编码器。</td></tr>
</table>

<a id="h3-loras"></a>

### LoRA 与微调

<table>
<tr><td width="56"><img src="https://huggingface.co/avatars/a0fe0c3e0ee676535c5cae05e05aacb7.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/matlod/minimax-h3-turnaround">minimax-h3-turnaround</a></td><td>从一张参考图生成同一主体的五个旋转视角。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6380ebb8471a4550ff255c62/JLWd_lQkzQN81FVIva1w1.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA">Realism - People</a></td><td>fal 训练的 LoRA，让日常场景里的人物更自然。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/a7bf1af19e638162579666509c490220.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime">Looping Sketch Anime</a></td><td>手绘 2D 动画风格，可以无缝循环。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/642b468deb31218a5f1e69f7/FWS9Yc-nckXsKBr5l-pij.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/TenStrip/Krea2-H3-Style-Lora">Krea2 Style</a></td><td>实验性风格 LoRA，向 Krea 2 的画风靠拢，只对文生视频有效。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/39c999386889182c4ac74c6c612f8bcf.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models">smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models</a></td><td>以 FL2VA 为基础、合入 Ref2VA 的 AdaLN 层，有四种层段组合。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/minimax_h3_ref2va_patchin_hf102">t8star/minimax_h3_ref2va_patchin_hf102</a></td><td>实验性的 Ref2VA 权重修改，把一处高频分量调高 2%。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6333948fea2dcff925248fca/Bef-LRCRhI3zZzt-lY8mv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Viggle/Viggle-Remake">Viggle Remake</a></td><td>Viggle 的两个 LoRA，替换视频里的人物，或修改物体、背景和天气；<a href="https://huggingface.co/Viggle/Viggle-Animate">Viggle Animate</a> 靠一帧重绘画面替换角色。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/653772fed7616b72795db894/1GPjjeZP_RGRl1pYgFdZc.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/PrunaAI/P-H3-Image">P-H3-Image</a></td><td>PrunaAI 基于 H3 视频模型微调的文生图和图像编辑模型。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/zYUGNDVFyLncD1OHov1pB.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI">Alissonerdx/Minimax-H3-ComfyUI</a></td><td>实验性 LoRA，用于动作迁移、角色和头部替换、风格迁移、特效编辑和动态壁纸。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/232ea307b6e3f75cffe2568a99541dad.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA">360 Orbit</a></td><td>FL2VA LoRA，把一张照片变成时间静止的 360 度环绕镜头。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/n2bexa-LzvGqmxyOGaonu.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity">Minimax-h3 Singularity</a></td><td>融合 FL2VA 和 Ref2VA 权重后再微调的模型，在 ComfyUI 里支持文生、图生、参考生和视频生视频；另有<a href="https://huggingface.co/WAS/Minimax-h3_Singularity_v2_pruned">剪枝版</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65bb837dbfb878f46c77de4c/23gZ_lBEwyoqjexFy9QLD.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp">prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp</a></td><td>实验性的 rank 16 LoRA，用于面部特写，表情和细微动作更真实；提示词以 <code>Facial Realism</code> 开头。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6928a3ca735961e729669bb3/RrdhFVLZ35EUaBSoYb4IE.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LiconStudio/Minimax-H3-VBVR">LiconStudio/Minimax-H3-VBVR</a></td><td>用重新整理的 VBVR 数据训练的图生视频 LoRA，减少物体重复和人物与物体互相穿插。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/635dd6cd4fabde0df74aeae6/zDmJWNx95QODS2mbAAxNZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alvdansen/h3-keyframe-animation">alvdansen/h3-keyframe-animation</a></td><td>三个 LoRA，按一到两张参考画保持手绘动画的节奏；个人和小企业免费，下载需同意条款。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/nJsOmzl5Jf72CxaaHjoiF.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/XGENlabs/XGEN-JING">XGENlabs/XGEN-JING</a></td><td>基于 Ref2VA 的 4 步第一人称模型，根据键盘控制的镜头、参考图和文字生成音视频。</td></tr>
<tr><td width="56"><img src="https://github.com/Junchao-cs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Junchao-cs/SolarWM">SolarWM</a></td><td>世界模型，H3 版本从首帧和提示词流式生成可控镜头的视频，附训练代码和 4 步的 SolarRenderer。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a6748afef317ed5adb30050/Ca36mpBOmdmrgVPxtMafQ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora">Wushu Action LoRA</a></td><td>用 924 段武打片段训练的武打动作 LoRA，触发词 <code>wushu_action</code>。</td></tr>
</table>

<a id="h3-runtimes"></a>

### 推理引擎

<table>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/ComfyUI">ComfyUI</a></td><td>首发即原生支持，主线已内置 INT8。</td></tr>
<tr><td width="56"><img src="https://github.com/modelscope.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a></td><td>提供 <code>MiniMaxH3Pipeline</code>，NF4 推理 8 GB 显存起步。</td></tr>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/LightX2V">LightX2V</a></td><td>支持并行、量化 DiT 和特征缓存的推理框架，Turbo LoRA 也出自这里。</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3">Sana Sol-Engine</a></td><td>NVIDIA 针对各型号 GPU 优化的推理方案，比基线快 3.55 到 4.52 倍；另有最多 8 卡的 4 步 Sol-H3。</td></tr>
<tr><td width="56"><img src="https://github.com/hao-ai-lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hao-ai-lab/FastVideo">FastVideo</a></td><td>FastH3 背后的稀疏注意力和步数蒸馏框架。</td></tr>
<tr><td width="56"><img src="https://github.com/aigc-apps.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/aigc-apps/VideoX-Fun">VideoX-Fun</a></td><td>Fun ControlNet-Union 和 PDD LoRA 的训练与推理代码。</td></tr>
<tr><td width="56"><img src="https://github.com/antirez.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/antirez/h3.c">h3.c</a></td><td>苹果芯片上的原生 Metal 引擎，支持文生视频、首尾帧和多参考生成。</td></tr>
<tr><td width="56"><img src="https://github.com/drawthingsai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/drawthingsai/draw-things-community">Draw Things</a></td><td>Draw Things 应用背后的 Swift/Metal 引擎，在苹果芯片上跑 FL2VA、Ref2VA 和 FastH3 Omni。</td></tr>
<tr><td width="56"><img src="https://github.com/ddalcu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ddalcu/mlx-serve">mlx-serve</a></td><td>苹果芯片上的应用和服务端，支持 8 步的 FastH3 Omni 多参考视频。</td></tr>
<tr><td width="56"><img src="https://github.com/komikndr.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/komikndr/raylight">Raylight</a></td><td>多卡 ComfyUI 运行时，把序列拆到多张显卡上并行计算。</td></tr>
<tr><td width="56"><img src="https://github.com/unslothai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unslothai/unsloth">Unsloth</a></td><td>支持运行和训练 MiniMax-H3。</td></tr>
<tr><td width="56"><img src="https://github.com/fluxmatrix.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/fluxmatrix/h3-flash">h3-flash</a></td><td>面向 8 卡 B200 的推理脚本和网页演示，用序列并行和融合算子加速。</td></tr>
<tr><td width="56"><img src="https://github.com/huggingface.png?size=96" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/blog/hf-torchtpu">TorchTPU + Diffusers</a></td><td>在 Cloud TPU v6e-4 上用原生 PyTorch、<code>torch.compile</code> 和张量并行运行 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/leejet.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/leejet/stable-diffusion.cpp">stable-diffusion.cpp</a></td><td>C/C++ 推理引擎，可运行 H3 及其 GGUF 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/mudler.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mudler/LocalAI">LocalAI</a></td><td>本地的 OpenAI 兼容服务，4.9.0 版加入了 H3 音视频生成。</td></tr>
<tr><td width="56"><img src="https://github.com/ztxz16.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ztxz16/fastllm">fastllm</a></td><td>原生支持 H3 文生音视频，附 HTTP 演示。</td></tr>
<tr><td width="56"><img src="https://github.com/0xShug0.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/0xShug0/audio.cpp">audio.cpp</a></td><td>C++ 引擎，用 Q4_K GGUF 运行 H3 文生音视频，DiT 可选 INT8 ConvRot。</td></tr>
<tr><td width="56"><img src="https://github.com/1CatAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1CatAI/1Cat-vLLM">1Cat-vLLM</a></td><td>面向 V100 的 vLLM 分支，原生支持 H3 音视频生成。</td></tr>
<tr><td width="56"><img src="https://github.com/gufo-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gufo-org/gufo">gufo</a></td><td>面向 AMD Strix Halo 的推理引擎，可运行 H3 有声视频。</td></tr>
<tr><td width="56"><img src="https://github.com/localai-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/localai-org/vllm.cpp">vllm.cpp</a></td><td>不依赖 Python 的 C++ 推理引擎，用 Q4_K_M GGUF 通过 <code>/v1/videos</code> 提供 H3 服务，在一台 Jetson Thor 上实测每步 34.6 秒。</td></tr>
</table>

<a id="h3-apps"></a>

### 应用与工作室

<table>
<tr><td width="56"><img src="https://github.com/agent-next.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agent-next/video-agent">open-video</a></td><td>用 <code>install</code>、<code>pull</code>、<code>run</code> 三条命令在自己的显卡上跑 H3，另附让编程 agent 生成视频的技能。</td></tr>
<tr><td width="56"><img src="https://github.com/FlashML-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlashML-org/FreeVideo">FreeVideo</a></td><td>基于 VDN-H3 的本地 ComfyUI 工作台，支持 8 GB 显存。</td></tr>
<tr><td width="56"><img src="https://github.com/D3voz.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/D3voz/openjutsu">Openjutsu</a></td><td>用 SAM 3.1 跟踪替换视频里的角色和物体。</td></tr>
<tr><td width="56"><img src="https://github.com/PxTicks.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/PxTicks/h3-motion">H3 Motion</a></td><td>编辑主体的关键帧运动，并对遮罩区域重新生成。</td></tr>
<tr><td width="56"><img src="https://github.com/zxcHolmes.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zxcHolmes/minimax-h3-studio">MiniMax-H3 Studio</a></td><td>可自托管的网页和 API 工作室，带持久队列，支持 FastH3 和 Pika H3 Flex。</td></tr>
<tr><td width="56"><img src="https://github.com/Klimentios.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Klimentios/talemondo-studio">TaleMondo Studio</a></td><td>非商用的 Mac 影视制作工作台，用本地 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/thezolon.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thezolon/tiiny-studio">Tiiny Studio</a></td><td>Tiiny Pocket 硬件的桌面应用，支持 H3-Turbo 文生视频和图生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/slogonomo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/slogonomo/lf-studio">LF Studio</a></td><td>NVIDIA Sol-H3 的网页工作室，跑在一台 DGX Spark 或一张 RTX 5090 上，原名 sol-h3-studio。</td></tr>
<tr><td width="56"><img src="https://github.com/omnichar.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/omnichar/OmniChar">Omnichar Studio</a></td><td>角色工作室，让同一角色的脸、身材、服装和声音在各个镜头里保持一致；可用 H3 生成，也能在本地训练 H3 LoRA。原名 Inline Studio。</td></tr>
<tr><td width="56"><img src="https://github.com/deepbeepmeep.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/deepbeepmeep/WanGP">WanGP</a></td><td>面向低显存显卡的视频生成工具，支持的模型里有 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/mcmonkeyprojects.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</a></td><td>网页界面，支持的视频模型里有 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/tin2tin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tin2tin/Pallaidium">Pallaidium</a></td><td>嵌在 Blender 视频编辑器里的生成式影视工具，支持 H3 和 Open-H3 Regenerate 放大。</td></tr>
<tr><td width="56"><img src="https://github.com/mrbizarro.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mrbizarro/Phosphene">Phosphene</a></td><td>原生 Mac 应用，在苹果芯片上离线运行 H3 和 LTX-2.5；Pro 版收费，可选。</td></tr>
<tr><td width="56"><img src="https://github.com/neuregex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/neuregex/freefield">freefield</a></td><td>跑在 Modal 上的一组 H3 应用：文字或照片生成视频、续写片段、换机位。</td></tr>
<tr><td width="56"><img src="https://github.com/AbrahamPaulJ.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AbrahamPaulJ/nightmare-mobile">nightmare-mobile</a></td><td>安卓上的节点图应用，在骁龙手机上生成 H3 有声视频；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/SwotAtmk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SwotAtmk/infinite-creation">infinite-creation</a></td><td>Agent 应用，用本地 ComfyUI 的 H3 把小说或剧本做成任意时长的视频；需要 12 GB 以上显存。</td></tr>
<tr><td width="56"><img src="https://github.com/saddam213.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/saddam213/AmuseAI">Amuse</a></td><td>基于 .NET 的桌面应用，在本地生成图片、视频和音频，H3 是它支持的视频模型之一，可用 CUDA 或 Vulkan。</td></tr>
<tr><td width="56"><img src="https://github.com/james-see.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/james-see/ltx-video-mac">ltx-video-mac</a></td><td>macOS 原生应用，通过 h3.c 在苹果芯片上运行 H3，可选 BF16、INT8 或 6 步 Turbo 版本，附本地 REST API。</td></tr>
<tr><td width="56"><img src="https://github.com/code-video-model.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/code-video-model/CodeVideoModel">CodeVideoModel</a></td><td>先让大模型搭 Three.js 场景渲染成布局视频，再用本地 H3 Ref2VA 生成写实视频。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenDCAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenDCAI/GameFactory-3A">GameFactory-3A</a></td><td>游戏资产生成框架，其中 CG 视频环节用本地 H3 以 720p 生成过场动画和宣传片。</td></tr>
<tr><td width="56"><img src="https://github.com/siyuan-liu31.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/siyuan-liu31/minimax-h3-video-studio">MiniMax H3 Video Studio</a></td><td>基于 ComfyUI 的自托管工作室和命令行工具，支持 H3 文生、图生、参考生和视频生视频，以及分段拼接的长视频。</td></tr>
<tr><td width="56"><img src="https://github.com/RachelForster.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/RachelForster/Shinsekai">Shinsekai</a></td><td>AI 角色扮演游戏制作工具，可通过本地 ComfyUI 用 H3 图生视频把角色立绘做成动态立绘。</td></tr>
<tr><td width="56"><img src="https://github.com/jnMetaCode.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jnMetaCode/openshorts">OpenShorts</a></td><td>本地优先的短视频流水线，AI 短剧模式可通过 stable-diffusion.cpp 在本机用 H3 生成片段，也可走云端。</td></tr>
</table>

<a id="h3-api"></a>

### API 集成

<table>
<tr><td width="56"><img src="https://github.com/NousResearch.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NousResearch/hermes-agent">Hermes Agent</a></td><td>会自我改进的 AI Agent，自带的 fal 视频插件可调用 H3、H3 Max 和 H3 Max Turbo，并把常用分辨率换算成各自的档位。</td></tr>
<tr><td width="56"><img src="https://github.com/n8n-io.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/n8n-io/n8n/tree/master/packages/@n8n/nodes-langchain/nodes/vendors/MiniMax">n8n</a></td><td>工作流自动化工具，MiniMax 节点可直接调用官方 API 生成 H3 视频；采用 fair-code 许可。</td></tr>
<tr><td width="56"><img src="https://github.com/harry0703.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/harry0703/MoneyPrinterTurbo">MoneyPrinterTurbo</a></td><td>根据主题或关键词生成短视频，可通过秘塔的 API 用 H3 生成视频素材。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenMontageInc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenMontageInc/OpenMontage">OpenMontage</a></td><td>Agent 驱动的视频制作系统，内置 H3 技能和工具，可调用 MiniMax 官方 API、fal 或本地 ComfyUI。</td></tr>
<tr><td width="56"><img src="https://github.com/BerriAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BerriAI/litellm">LiteLLM</a></td><td>Python SDK 和 AI 网关，可用 OpenAI 风格的视频接口在 MiniMax 官方 API 上创建 H3 任务。</td></tr>
<tr><td width="56"><img src="https://github.com/QuantumNous.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/QuantumNous/new-api">New API</a></td><td>可自托管的 AI 网关，海螺任务插件会把 H3 多模态任务转发到 MiniMax 的 v2 视频接口。</td></tr>
<tr><td width="56"><img src="https://github.com/HeyPuter.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HeyPuter/puter">Puter</a></td><td>可自托管的网页桌面系统，视频生成的 MiniMax 供应商默认用 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/THU-MAIC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/THU-MAIC/OpenMAIC">OpenMAIC</a></td><td>多 Agent 互动课堂，视频适配器走 MiniMax 的 v2 接口调用 H3 和 H3 Max，可直连，也可经 TokenDance 网关。</td></tr>
<tr><td width="56"><img src="https://github.com/agno-agi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agno-agi/agno">Agno</a></td><td>Agent 框架，自带的 MiniMax 工具走官方 API，用 H3 根据文字生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/storytold.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/storytold/artcraft">ArtCraft</a></td><td>桌面应用，先在 2D 和 3D 里布置场景，再通过自家云服务或 fal 用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/agentscope-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agentscope-ai/QwenPaw">QwenPaw</a></td><td>个人 AI 助手，Creator 插件可走 MiniMax 官方 API 调用 H3 或 H3 Max，也能接自己在 SGLang 上部署的 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/yetone.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yetone/magpie">Magpie</a></td><td>菜单栏应用，为各种编程 Agent 统一配置模型；填入 MiniMax API Key 后还能用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/simstudioai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/simstudioai/sim">Sim</a></td><td>搭建 Agent 工作流的平台，视频工具可走 MiniMax 官方 API 调用 H3 及其 Highspeed、Lite 和 Max 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/vercel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vercel/ai">AI SDK</a></td><td>Vercel 的 TypeScript 工具包，MiniMax provider 可生成 H3 和 H3 Max 视频，并按模式检查输入是否合规。</td></tr>
<tr><td width="56"><img src="https://github.com/1Panel-dev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1Panel-dev/MaxKB">MaxKB</a></td><td>企业级智能体平台，在 MiniMax 模型供应商下提供 H3 和 H3 Max 的文生视频与图生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/HBAI-Ltd.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HBAI-Ltd/Toonflow-app">Toonflow</a></td><td>从剧本、分镜到成片的短剧制作工具，通过秘塔、优云智算或 APIMart 的 API 调用 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/chatfire-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chatfire-AI/huobao-drama">Huobao Drama</a></td><td>一句话生成完整短剧的平台，内置调用 MiniMax 官方 API 的 H3 适配器；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/corsairdev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/corsairdev/corsair">Corsair</a></td><td>应用集成平台，MiniMax 插件让 Agent 能通过官方 API 生成 H3 和 H3 Max 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/waooAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/waooAI/waoowaoo">waoowaoo</a></td><td>影视全流程生产平台，通过 OpenRouter 调用 H3 和 H3 Max；采用 Elastic License 2.0。</td></tr>
<tr><td width="56"><img src="https://github.com/maximhq.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/maximhq/bifrost">Bifrost</a></td><td>AI 网关，把 H3 和 H3 Max 作为异步任务发到 MiniMax 的 v2 接口，并把尺寸和时长调整到各模型的范围内。</td></tr>
<tr><td width="56"><img src="https://github.com/rikkahub.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rikkahub/rikkahub">RikkaHub</a></td><td>支持多家大模型的安卓聊天应用，也能通过 MiniMax 官方 API 用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/TeamWiseFlow.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TeamWiseFlow/xiaobei">xiaobei</a></td><td>面向中小企业的自媒体获客 Agent，视频技能走 MiniMax 官方 API 用 H3 生成片段；在 MIT 基础上附加了条款。</td></tr>
<tr><td width="56"><img src="https://github.com/openJiuwen-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openJiuwen-ai/jiuwenswarm">JiuwenSwarm</a></td><td>AI Agent，视频工具可把 H3 和 H3 Max 任务发到 MiniMax 官方 API。</td></tr>
<tr><td width="56"><img src="https://github.com/Devin-AXIS.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Devin-AXIS/iPolloWork">iPolloWork</a></td><td>Agent 工作台，带 H3 视频和 H3 数字人两种生成工具，都在 RunningHub 工作流上运行；仅限个人非商用。</td></tr>
<tr><td width="56"><img src="https://github.com/ArcReel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ArcReel/ArcReel">ArcReel</a></td><td>把小说和剧本做成分镜和视频的 Agent 工作台，内置调用 MiniMax 官方 API 的 H3 接口。</td></tr>
<tr><td width="56"><img src="https://github.com/monid-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/monid-ai/monid">Monid</a></td><td>面向 Agent 的工具 API 市场，提供对接 MiniMax v2 接口的 H3、H3 Max 和 H3 Max Turbo。</td></tr>
<tr><td width="56"><img src="https://github.com/EverMind-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/EverMind-AI/Raven">Raven</a></td><td>多 Agent 系统，自带的 MiniMax 媒体工具通过 v2 接口生成 H3 和 H3 Max 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/pollinations.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pollinations/pollinations">Pollinations</a></td><td>开源生成式 AI 平台，可通过它的 API 生成 H3、H3 Max 和 H3 Max Turbo 视频，上游是 fal。</td></tr>
<tr><td width="56"><img src="https://github.com/chaitin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chaitin/MonkeyCode">MonkeyCode</a></td><td>企业级 AI 办公工作台，视频服务走 MiniMax 官方 API，支持 H3 的文生、首尾帧和参考图模式。</td></tr>
<tr><td width="56"><img src="https://github.com/darkzOGx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/darkzOGx/youtube-automation-agent">YouTube Automation Agent</a></td><td>全流程运营 YouTube 频道的 Agent，可通过 MiniMax 官方 API 用 H3 生成片段，并限制付费秒数。</td></tr>
<tr><td width="56"><img src="https://github.com/crmne.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/crmne/ruby_llm">RubyLLM</a></td><td>Ruby AI 框架，接入 MiniMax 视频协议后，<code>animate</code> 调用可以生成 H3 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/Agent-Field.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Agent-Field/agentfield">AgentField</a></td><td>把 Agent 当作服务来构建和运行的框架，MiniMax 媒体供应商默认用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/jnMetaCode.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jnMetaCode/agency-orchestrator">Agency Orchestrator</a></td><td>多 Agent 工作流工具，视频步骤可通过秘塔或 APIMart 的 API 用 H3 生成片段。</td></tr>
<tr><td width="56"><img src="https://github.com/xuanyustudio.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xuanyustudio/LocalMiniDrama">LocalMiniDrama</a></td><td>数据不出本机的 AI 短剧工具，可通过 MiniMax 官方 API 或中转服务用 H3 生成镜头。</td></tr>
<tr><td width="56"><img src="https://github.com/glanderness.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/glanderness/BeefTV">BeefTV</a></td><td>画布式视频工作台，可在 fal 上实时使用 H3 Max Director，另有对接 MiniMax v2 接口的 H3 插件；源码可见许可。</td></tr>
<tr><td width="56"><img src="https://github.com/WhiteTowerAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WhiteTowerAI/ohmygame">OhMyGame</a></td><td>用 AI 做游戏的工作室应用，视频工具走 MiniMax 官方 API，可选 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/ddcat-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ddcat-ai/open-ai-canvas">open-ai-canvas</a></td><td>AI 影视创作画布，内置 H3 提示词卡片，可通过 MiniMax、阿里云百炼或秘塔调用 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/Stonewuu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Stonewuu/ai-fusion-video">ai-fusion-video</a></td><td>基于 Java 的短剧漫剧 Agent 平台，内置调用 MiniMax 官方 API 的 H3 和 H3 Max 预设。</td></tr>
<tr><td width="56"><img src="https://github.com/siddsachar.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/siddsachar/row-bot">Row-Bot</a></td><td>本地优先的个人助手，视频工具可走 MiniMax 官方 API 调用 H3 及其 Highspeed、Lite 和 Max 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/869413421.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/869413421/ai-moive-studio">AICON</a></td><td>以无限画布驱动的视频创作 Agent，默认视频模型是 H3，通过项目自家的 API 网关调用。</td></tr>
<tr><td width="56"><img src="https://github.com/tigerowo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tigerowo/infinite-canvas">infinite-canvas</a></td><td>AI 创作画布，可按 MiniMax 接口协议生成 H3 视频，内置 H3 提示词技能。</td></tr>
<tr><td width="56"><img src="https://github.com/xixihhhh.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xixihhhh/clipforge">ClipForge</a></td><td>上传一张商品图生成带货短视频，可通过 Atlas Cloud 调用 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/0xsline.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/0xsline/StoryGen-Atelier">StoryGen-Atelier</a></td><td>分镜工具，由 Gemini 写分镜和画面，视频片段除 Veo 外也可通过 MiniMax 官方 API 用 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/looplj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/looplj/reveric">Reveric</a></td><td>小说和短剧桌面创作工作台，可通过 MiniMax 或秘塔的 API 用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/anymouschina.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/anymouschina/TapCanvas">TapCanvas</a></td><td>面向 AI 影视和漫剧的 Agent 画布，自带的网关支持 MiniMax 的 H3 v2 视频协议。</td></tr>
<tr><td width="56"><img src="https://github.com/ChrisChen667788.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ChrisChen667788/wind-comic">wind-comic</a></td><td>多 Agent 流水线，把一句话做成短剧，镜头通过 MiniMax 官方 API 用 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/agentjido.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agentjido/req_llm">ReqLLM</a></td><td>Elixir 大模型库，<code>generate_video</code> 可调用 MiniMax 官方 API 的 H3，也能直接上传本地参考文件。</td></tr>
<tr><td width="56"><img src="https://github.com/ZMGID.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ZMGID/kivio">Kivio</a></td><td>macOS 和 Windows 上的屏幕级 AI 助手，可通过 MiniMax v2 接口生成 H3 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/tryonlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tryonlabs/opentryon">OpenTryOn</a></td><td>虚拟试衣和时尚 AI 工具包，可通过 MiniMax 官方 API 调用 H3 和 H3 Max，也能在本地显卡上跑 H3；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/ganbo-gab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ganbo-gab/open-storyboard-canvas">Open Storyboard Canvas</a></td><td>AI 分镜和导演台画布，视频节点可使用预设的 MiniMax H3 供应商。</td></tr>
<tr><td width="56"><img src="https://github.com/fancyboi999.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/fancyboi999/Loomic">Loomic</a></td><td>在无限画布上用对话做设计的 Agent，可通过秘塔的 API 用 H3 从文字或首尾帧生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/chillzhuang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chillzhuang/Kinema">Kinema</a></td><td>做 AI 漫剧、电影和解说视频的 Agent，带对接 MiniMax v2 接口的 <code>minimax-h3</code> 供应商。</td></tr>
<tr><td width="56"><img src="https://github.com/Finderchangchang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Finderchangchang/brewreel">BrewReel</a></td><td>根据产品简报生成竖版宣传片；口播配画面模式可通过 MiniMax 官方 API 用 H3 生成 AI 画面。</td></tr>
<tr><td width="56"><img src="https://github.com/LudwigKienle.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LudwigKienle/ai-video-production-editor">AI Video Production Editor</a></td><td>面向 AI 电影制作的 Electron 编辑器，节点流程里可通过 fal 运行 H3 文生视频和图生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/sumarilkkxx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sumarilkkxx/Mora">Mora</a></td><td>本地优先的商业推广视频工作台，可通过 Atlas Cloud 用 H3 生成视频片段。</td></tr>
<tr><td width="56"><img src="https://github.com/saihhold-zhao.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/saihhold-zhao/polox_ai">Polox AI</a></td><td>基于 DeepSeek Harness 的多模态生成平台，可通过 WaveSpeed 调用 H3 文生视频、图生视频和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/s-nagaev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/s-nagaev/chibi">Chibi</a></td><td>可自托管的 Telegram 机器人，视频生成可通过 MiniMax 官方 API 调用 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/WaveSpeedAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WaveSpeedAI/mcp-server">WaveSpeed MCP Server</a></td><td>MCP 服务，可在 WaveSpeed 的 API 上用 H3 生成、编辑和延长视频。</td></tr>
<tr><td width="56"><img src="https://github.com/Sociaro-Research.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Sociaro-Research/open-media-gateway">Open Media Gateway</a></td><td>可自托管的媒体服务，MiniMax 供应商覆盖官方 API 上的 H3 及其 Highspeed、Lite 和 Max 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/KunLabAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KunLabAI/KunFlix">KunFlix</a></td><td>影视广告创作平台，视频服务通过 MiniMax v2 接口调用 H3；Apache-2.0 附加条款。</td></tr>
<tr><td width="56"><img src="https://github.com/Jjat00.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Jjat00/hf-studio">hf-studio</a></td><td>可自托管的媒体工作室，通过 Higgsfield 调用 H3 视频，通过 KIE 调用 H3-Image。</td></tr>
<tr><td width="56"><img src="https://github.com/ItusiAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ItusiAI/Open-Magiviz">Magiviz</a></td><td>AI 视频创作引擎，视频模型里有通过 KIE API 调用的 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/openfilm.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openfilm/openfilm">OpenFilm</a></td><td>让编程 Agent 做视频的工具和编辑器，可通过 fal 的 H3 Max Recast 替换片段里的人物。</td></tr>
<tr><td width="56"><img src="https://github.com/HaD0Yun.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HaD0Yun/CozyClay">CozyClay</a></td><td>浏览器里的预演工具，先搭场景和机位，再用 fal 上的 H3 Max Turbo 或自己的 ComfyUI H3 工作流渲染镜头。</td></tr>
<tr><td width="56"><img src="https://github.com/openstory-so.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openstory-so/openstory">OpenStory</a></td><td>视频序列创作平台，运动生成环节可通过 fal 使用 H3 Max 文生视频、图生视频和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/lovisdotio.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lovisdotio/video-to-game">video-to-game</a></td><td>31 款游戏，每款都由 fal 上生成的 10 到 15 秒 H3 Max 片段复刻而来，附素材制作流程。</td></tr>
<tr><td width="56"><img src="https://github.com/patch-gg.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/patch-gg/anymove">anymove</a></td><td>用 fal 上生成的双机位 H3 Max 片段为任意生物做动作捕捉，附在 fal 上训练 H3 LoRA 的技能。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/prompt2rig">prompt2rig</a></td><td>输入一个动作，绑好骨骼的角色就会做出来，快速模式用 fal 上的 H3 Max Turbo 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/falcraft">falcraft</a></td><td>Minecraft 模组，用 fal 生成建筑，其中墙上的电视播放由 fal 上的 H3 Max Director 实时生成的画面。</td></tr>
<tr><td width="56"><img src="https://github.com/vargHQ.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vargHQ/sdk">varg SDK</a></td><td>TypeScript 视频生成 SDK，内置 fal 上的 H3 Max 文生视频、图生视频和参考生视频模型；JSX 写法已弃用。</td></tr>
<tr><td width="56"><img src="https://github.com/orkunisitmak.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/orkunisitmak/banana-studio">Banana Studio</a></td><td>可自托管的媒体工作室，可通过 fal 使用 H3 Max 的文生视频、图生视频、参考生视频、延长、对口型和换人。</td></tr>
<tr><td width="56"><img src="https://github.com/henjicc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/henjicc/Henji-AI">Henji AI</a></td><td>聚合多家供应商的桌面应用，可通过 fal 或 APIMart 调用 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/xflare-bot.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xflare-bot/pokemonlive">pokemonlive</a></td><td>本地宝可梦对战原型，过场动画由 fal 上的 H3 Max Turbo 和 H3 Max 参考生视频生成。</td></tr>
<tr><td width="56"><img src="https://github.com/internetphysics.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/internetphysics/live-classroom">live-classroom</a></td><td>教任何主题的直播电视频道，每段课程内容都由 fal 上的 H3 Max Turbo 即时生成。</td></tr>
<tr><td width="56"><img src="https://github.com/gokayfem.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gokayfem/h3-max-education">H3 Max Realtime Education</a></td><td>可随时用语音打断的实时科学辅导，教学视频由 fal 上的 H3 Max 持续生成。</td></tr>
<tr><td width="56"><img src="https://github.com/marketcalls.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/marketcalls/sketchmaker">SketchMaker</a></td><td>可自托管的创作工具，动画、故事和对口型功能默认用 fal 上的 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/gokayfem.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gokayfem/H3-Max-Blender">H3 Max Blender</a></td><td>Blender 神经渲染演示，用 fal 上的 H3 Max 把一艘灰模飞船渲染成四种风格，随几何体变化更新。</td></tr>
<tr><td width="56"><img src="https://github.com/Valiera00.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Valiera00/SPITE">SPITE</a></td><td>面向电影人的前期制作画布，视频节点可用你自己的 fal 密钥调用 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/interdimensional-game">Interdimensional Game</a></td><td>实时生成视频的互动冒险游戏，视频由 fal 上的 H3 Max 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/unreel">UNREEL</a></td><td>AI 流媒体应用，大模型写剧集，再由 fal 上的 H3 Max Turbo 逐个镜头生成画面。</td></tr>
<tr><td width="56"><img src="https://github.com/ilkerzg.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ilkerzg/storycast">StoryCast</a></td><td>输入主题生成故事视频，镜头和对口型的台词由 fal 上的 H3 Max 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/eren23.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/eren23/openflipbook">OpenFlipbook</a></td><td>每页都是 AI 插画的翻页书，可选的页面动画由 fal 上的 H3 Max 或 H3 Max Turbo 生成。</td></tr>
</table>

<a id="h3-comfyui"></a>

### ComfyUI 节点

<table>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-audio-T8">comfyui-minimax-h3-audio-T8</a></td><td>62 个节点，覆盖音频、静态图、条件控制、长视频和语音。</td></tr>
<tr><td width="56"><img src="https://github.com/huangserva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/huangserva/ComfyUI_MiniMaxH3_Director">ComfyUI_MiniMaxH3_Director (huangserva)</a></td><td>分镜节点，附五个可导入模板：t2v、fl2v、r2v、v2v 和 rv2v。</td></tr>
<tr><td width="56"><img src="https://github.com/AIMixer.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director">ComfyUI_MiniMaxH3_Director (AIMixer)</a></td><td>最早的 Director 节点。</td></tr>
<tr><td width="56"><img src="https://github.com/seesee75-commits.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director">ComfyUI-MiniMaxH3-Director</a></td><td>时间线编辑器：素材拖上轨道、每个镜头一条提示词，支持实时预览和重拍。</td></tr>
<tr><td width="56"><img src="https://github.com/imbutus.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/imbutus/ComfyUI-MiniMaxDirector">ComfyUI-MiniMaxDirector</a></td><td>把镜头、运镜和音频提示排在时间线上，编译成官方指南里的结构化提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/nkxx188.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy">ComfyUI-MiniMaxH3-Easy</a></td><td>一个紧凑的工作流覆盖文生视频、图生视频、首尾帧和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/FX-FeiHou.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3">ComfyUI-FeiHou-Easy-H3</a></td><td>MiniMaxH3-Easy 的分支版，内置 9 图、3 视频、3 音频的参考槽位。</td></tr>
<tr><td width="56"><img src="https://github.com/NikoDemon80.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context">ComfyUI-H3-Motion-Context</a></td><td>把多段视频接起来，保持动作和声音连贯。</td></tr>
<tr><td width="56"><img src="https://github.com/princepainter.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/princepainter/ComfyUI-PainterNodes">ComfyUI-PainterNodes</a></td><td><code>MiniMaxRefToVideo2</code> 节点，支持官方的参考和对白格式。</td></tr>
<tr><td width="56"><img src="https://github.com/wyzborrero.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/wyzborrero/ComfyUI-H3-FunControl">ComfyUI-H3-FunControl</a></td><td>加载 Fun ControlNet-Union，附预处理器和局部重绘流程。</td></tr>
<tr><td width="56"><img src="https://github.com/lihaoyun6.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lihaoyun6/ComfyUI-MiniMaxH3_Ref-Patch">ComfyUI-MiniMaxH3_Ref-Patch</a></td><td>把 Ref Patch 应用到 FL2VA 上。</td></tr>
<tr><td width="56"><img src="https://github.com/zechengtang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zechengtang/VPack">VPack</a></td><td>压缩 Ref2VA 的参考视频条件，减少参考 token 数量。</td></tr>
<tr><td width="56"><img src="https://github.com/alisson-anjos.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3">TARA-H3</a></td><td>实验性的 Ref2VA 视频参考注意力时间对齐，附技术报告。</td></tr>
<tr><td width="56"><img src="https://github.com/matlowai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matlowai/ComfyUI-Dials">ComfyUI-Dials</a></td><td>用提示词对构造免训练的引导向量，调节动作和风格。</td></tr>
<tr><td width="56"><img src="https://github.com/scraed.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scraed/LanPaint">LanPaint</a></td><td>免训练的视频和音频局部重绘。</td></tr>
<tr><td width="56"><img src="https://github.com/Tr1dae.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler">ComfyUI-MiniMaxH3_LatentUpscaler</a></td><td>针对 H3 音视频潜变量的放大节点，用于两段式采样。</td></tr>
<tr><td width="56"><img src="https://github.com/LBH-123-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler">Comfyui_Minimax_h3_latent_Upscaler</a></td><td>学习型潜空间放大，1.0 到 4.0 倍，不必先解码再编码。</td></tr>
<tr><td width="56"><img src="https://github.com/tazmannner.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tazmannner/ComfyUI-H3-ResCascade">ComfyUI-H3-ResCascade</a></td><td>高清修复节点，把约 100 万像素的结果精修到 1080p 或 1440p，保留原音频。</td></tr>
<tr><td width="56"><img src="https://github.com/rdh9616.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate">ComfyUI Open-H3 Regenerate</a></td><td>NVIDIA Open-H3-Regenerate-2K 精修模型的节点和工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/hashu786.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hashu786/ComfyUI-Tessera">Tessera</a></td><td>分块、分段的视频精修，VAE 编解码以流式进行。</td></tr>
<tr><td width="56"><img src="https://github.com/lihaoyun6.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT">ComfyUI-H3VAE_TRT</a></td><td>TensorRT 版视频 VAE，解码最多快 1.7 倍。</td></tr>
<tr><td width="56"><img src="https://github.com/xmarre.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3">ComfyUI-Spectrum-MiniMax-H3</a></td><td>通过预测特征跳过部分 transformer 计算，结果是近似的。</td></tr>
<tr><td width="56"><img src="https://github.com/kijai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kijai/ComfyUI-SolAttn_triton">ComfyUI-SolAttn_triton</a></td><td>免训练的 Sol-Attn 稀疏注意力，在 H3 上测试过；ComfyUI 已内置这项功能，此节点已停止维护。</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-VDN-H3">ComfyUI-VDN-H3</a></td><td>在 ComfyUI 里运行 VDN 适配器；<a href="https://github.com/Speach1sdef178/ComfyUI-VDN-H3-24GB">24 GB 分支版</a>卸载得更多。</td></tr>
<tr><td width="56"><img src="https://github.com/Kablex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Kablex/ComfyUI-Ref2VA-VSA">ComfyUI-Ref2VA-VSA</a></td><td>给 Ref2VA 用的视频稀疏注意力，RTX 4090 上 5 秒视频约 72 秒。</td></tr>
<tr><td width="56"><img src="https://github.com/Jalen-Brunson.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc">ComfyUI-MiniMax-H3-PDD-Acc</a></td><td>在旧版 ComfyUI 上加载带 32 个输出头的 PDD LoRA。</td></tr>
<tr><td width="56"><img src="https://github.com/xmarre.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xmarre/ComfyUI-Sol-H3">ComfyUI-Sol-H3</a></td><td>Sol-H3 运行时的 ComfyUI 前端（GPL-3.0）。</td></tr>
<tr><td width="56"><img src="https://github.com/zechengtang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zechengtang/Spark-H3">Spark-H3</a></td><td>自适应块稀疏注意力，提供 diffusers 和 ComfyUI 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/iamwavecut.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/iamwavecut/ComfyUI-OrbitQuant/tree/feature/minimax-h3-comfyui">ComfyUI-OrbitQuant</a></td><td>加载 OrbitQuant W4A4 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/NikoDemon80.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NikoDemon80/ComfyUI-FastH3-Lora-Converter">ComfyUI-FastH3-Lora-Converter</a></td><td>把 FastH3 LoRA 转成 ComfyUI 的键名顺序。</td></tr>
<tr><td width="56"><img src="https://github.com/BobJohnson24.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BobJohnson24/ComfyUI-INT8-Fast">ComfyUI-INT8-Fast</a></td><td>已基本被原生 INT8 取代，<code>convert_comfy_quant.py</code> 仍可转换旧版 INT8 文件。</td></tr>
<tr><td width="56"><img src="https://github.com/Songssx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector">ComfyUI-MiniMaxH3-TimelineDirector</a></td><td>可编辑的 Ref2VA 参考素材时间线，附一个全功能导演台工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/Luisacaotica.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod">ComfyUI-MiniMaxH3Mod</a></td><td>RefMod 适配器，不用训练就把参考图和视频压成很小的潜变量；仍在开发中。</td></tr>
<tr><td width="56"><img src="https://github.com/Carasibana.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Carasibana/ComfyUI-H3-FaceRefine">ComfyUI-H3-FaceRefine</a></td><td>逐帧跟踪画面里的小脸，用 H3 精修后再贴回去。</td></tr>
<tr><td width="56"><img src="https://github.com/JGRFW.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/JGRFW/comfyui-AICG3D">comfyui-AICG3D</a></td><td>短剧工作台，把剧本拆成多段，生成任意时长的视频。</td></tr>
<tr><td width="56"><img src="https://github.com/ukr8b3g-cmyk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ukr8b3g-cmyk/H3-Character-Sheet-Designer">H3-Character-Sheet-Designer</a></td><td>用一张角色图生成角色设定图，最多 13 个视角。</td></tr>
<tr><td width="56"><img src="https://github.com/tritant.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tritant/ComfyUI_MiniMax_H3_Extender">ComfyUI_MiniMax_H3_Extender</a></td><td>把多段片段接成长视频，动作、画面和声音保持连贯。</td></tr>
<tr><td width="56"><img src="https://github.com/MajoorWaldi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MajoorWaldi/ComfyUI-Majoor-H3-GuideMaster">ComfyUI-Majoor-H3-GuideMaster</a></td><td>把图片和音频放到 H3 时间线上作为引导。</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3">ComfyUI-Viggle-Animate-H3</a></td><td>Viggle Animate 的节点，用参考图替换驱动视频里的角色。</td></tr>
<tr><td width="56"><img src="https://github.com/astropuzzo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio">ComfyUI-MiniMax-H3-Image-Studio</a></td><td>用视频模型生成静态图和做参考图编辑，附 Qwen 细节精修。</td></tr>
<tr><td width="56"><img src="https://github.com/thaakeno.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thaakeno/ComfyUI-MiniMax-H3-Studio">ComfyUI-MiniMax-H3-Studio</a></td><td>基于视频模型、支持参考图的静态图工作流，目前是 alpha 版。</td></tr>
<tr><td width="56"><img src="https://github.com/HELPMEEADICE.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS">TE-Speed-MiniMaxH3-OSS</a></td><td>步间缓存，sigma 变化很小时复用后段层的残差，30 步时约快 45%；需要给 ComfyUI 打一次补丁。</td></tr>
<tr><td width="56"><img src="https://github.com/vizart-vj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia">ComfyUI-MiniMax-H3-LongMedia</a></td><td>长篇音视频节点，带 Director 时间线，可单独重渲某一段，也能重配口型。</td></tr>
<tr><td width="56"><img src="https://github.com/ukr8b3g-cmyk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum">ComfyUI-H3-Continuum</a></td><td>分段生成长视频和音频，每段一条提示词，可单独重试某一段，也能从保存的结果接着生成。</td></tr>
<tr><td width="56"><img src="https://github.com/seitanism.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef">ComfyUI-H3-Motion-Context-MultiRef</a></td><td>H3-Motion-Context 的分支版，新增围绕一首歌生成的音乐视频、V2V 动作迁移、两段视频之间的音视频过渡和带遮罩的关键帧。</td></tr>
<tr><td width="56"><img src="https://github.com/chanon.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chanon/comfyui-obvpm-timeline">comfyui-obvpm-timeline</a></td><td>时间线节点，像小型剪辑软件一样生成、延长、往前补、衔接片段，也能做无缝循环；需要 comfyui-obvpm 节点包。</td></tr>
<tr><td width="56"><img src="https://github.com/yichengup.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3">ComfyUI-YCNodes-MiniMax-H3</a></td><td>6 个节点，包括按时间分段的提示词、动态 CFG、低噪阶段精修和分块采样。</td></tr>
<tr><td width="56"><img src="https://github.com/goohai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/goohai/Goohai-MiniMax-H3_Integration">Goohai-MiniMax-H3_Integration</a></td><td>一体化条件节点，按上传的素材自动选择 T2VA、FL2VA、Ref2VA 或混合模式，并写好官方标签。</td></tr>
<tr><td width="56"><img src="https://github.com/bbaudio-2025.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale">Comfyui-MMH3-UltimateUpscale</a></td><td>把生成好的潜变量放大后按时间分段、按空间分块重新采样，让高分辨率长视频在有限显存里跑完。</td></tr>
<tr><td width="56"><img src="https://github.com/matlowai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matlowai/ComfyUI-MAINodes">ComfyUI-MAINodes</a></td><td>Motion Lab 先渲染放慢版再去掉重复帧，修复快速动作的拖影；另有多视角和低显存节点。</td></tr>
<tr><td width="56"><img src="https://github.com/knoic.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream">ComfyUI-MiniMaxH3-PrefixStream</a></td><td>长视频续写节点，把上一段干净的潜变量写进下一段开头，并用素材箱挑选从哪一段接着生成。</td></tr>
<tr><td width="56"><img src="https://github.com/nicolab28.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nicolab28/ComfyUI-ClipProj">ComfyUI-ClipProj</a></td><td>用学习到的<a href="https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3">投影</a>把 32B 文本编码器换成 Qwen3-VL 4B 或 8B，最低只占 4.5 GB。</td></tr>
<tr><td width="56"><img src="https://github.com/scottmudge.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader">ComfyUI_MinimaxH3HybridLoader</a></td><td>加载时按层段把 Ref2VA 的部分权重合入 FL2VA，不用另存新文件。</td></tr>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still">ComfyUI-Fizgig-H3-Still</a></td><td>两个节点，用单帧潜变量和干净的解码让 H3 出真正的单帧静态图，尺寸不限，也不需要额外模型。</td></tr>
<tr><td width="56"><img src="https://github.com/Adudeguyman.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Adudeguyman/ComfyUI-H3-AudioRefine">ComfyUI-H3-AudioRefine</a></td><td>冻结已采样潜变量里的视频，只给音频多跑几步，适合画面够用但声音粗糙的 4 步 Turbo 结果。</td></tr>
<tr><td width="56"><img src="https://github.com/KadoBOT.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio">ComfyUI-H3-Continuity-Studio</a></td><td>多镜头时间线，在片段之间和换场景、换机位后延续上下文，附给编程 agent 用的 MCP 接口。</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-sol-attn">ComfyUI-sol-attn</a></td><td>Sol-Attn 的 Triton 内核，附 H3 专用节点：零拷贝注意力、融合 AdaLN 调制和分块 MLP，支持 RTX 30 系到 Blackwell。</td></tr>
<tr><td width="56"><img src="https://github.com/rookiestar28.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rookiestar28/MiniMaxH3-Studio">MiniMaxH3-Studio</a></td><td>ComfyUI 侧边栏加 28 个节点，覆盖写提示词、调用官方 H3 工作流、规划长视频和内置剪辑。</td></tr>
<tr><td width="56"><img src="https://github.com/veda-sparse.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/veda-sparse/Veda-on-ComfyUI">Veda-on-ComfyUI</a></td><td>一个节点给原生 H3 加上 Veda 稀疏注意力，在 RTX 5070 上端到端快 2.9 倍，支持 NVIDIA、AMD ROCm 和苹果芯片。</td></tr>
<tr><td width="56"><img src="https://github.com/WASasquatch.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WASasquatch/was-node-suite-comfyui">WAS Node Suite</a></td><td>通用节点套件，H3 节点能在一个节点里写好多段视频的条件、逐窗口延长，并用 Fun ControlNet 控制每段。</td></tr>
<tr><td width="56"><img src="https://github.com/Astral-xm.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Astral-xm/ComfyUI-MiniMax-ContextIR">ComfyUI-MiniMax-ContextIR</a></td><td>覆盖本地 H3 从提示词到采样的节点包，可选用 MiniMax 的 H3 Context-IR API 改写提示词。</td></tr>
</table>

<a id="h3-workflows"></a>

### 工作流

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow">MiniMax-H3-Multishot-Workflow</a></td><td>把多段 FL2VA 和 Ref2VA 视频串成一个序列，衔接处音频对齐。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65dc6589ec3dc9ccf761d936/SxJaAV-JarA1HZj4D1iEV.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows">comfy-MiniMax-H3-workflows</a></td><td>把低显存加速方案打包成一个可导入的工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/FlowForgeLabAi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlowForgeLabAi/MiniMax-H3-8GB-Workflows">MiniMax-H3-8GB-Workflows</a></td><td>面向 8 GB 笔记本的三档工作流合集。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/minimax_fl2v_gguf_workflow.json">FL2V GGUF workflow</a></td><td>在 ComfyUI 里运行 GGUF 量化的 FL2VA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json">OrbitQuant workflows</a></td><td>W4A4 版本的文生音视频工作流，另有 <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA-api.json">T2VA</a> 和 <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-Ref2VA-api.json">Ref2VA</a> 的 API 格式。</td></tr>
<tr><td width="56"><img src="https://github.com/pixelaigc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pixelaigc/ComfyUI-MiniMax-H3-Workflow">ComfyUI-MiniMax-H3-Workflow</a></td><td>首尾帧、图生视频和文生视频合在一个工作流里，带 Turbo LoRA 和分辨率总控面板。</td></tr>
</table>

<a id="h3-prompting"></a>

### 提示词与 Agent 技能

<table>
<tr><td width="56"><img src="https://github.com/1038lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor">ComfyUI-MiniMax-H3-Promptor</a></td><td>生成 H3 提示词，把 <code>&lt;Picture X&gt;</code> 标签直接写进动作描述里。</td></tr>
<tr><td width="56"><img src="https://github.com/ethanfel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide">ComfyUI-MiniMax-H3-Guide</a></td><td>把身份、关键帧、动作、配音和配乐拆成不同角色，再编译成合规的 H3 提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/ruashots.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ruashots/open-h3-ir">OpenH3-IR</a></td><td>按 MiniMax 公开的 Context-IR 格式写提示词，写完自检修正；可用命令行、HTTP 或 ComfyUI。</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8">comfyui-minimax-h3-prompt-enhancer-T8</a></td><td>在 ComfyUI 里调用服务端扩写提示词。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680de13385293771bc57400b/qnjz3avU_cKzRZA2aGjuZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA">MiniMax-H3-Prompt-Rewriter-LoRA</a></td><td>Qwen3.6-27B 的 LoRA，把短提示词改写成 H3 的三段式结构；<a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B">8B 视觉版</a>还能看关键帧。</td></tr>
<tr><td width="56"><img src="https://github.com/BeatAPI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BeatAPI/awesome-minimax-h3-prompts">awesome-minimax-h3-prompts</a></td><td>按类别整理的提示词合集，附示例视频。</td></tr>
<tr><td width="56"><img src="https://github.com/yangzhou-chaofan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yangzhou-chaofan/minimax-h3-1000-prompts">MiniMax H3 1K prompts</a></td><td>精选的 1000 条文生视频提示词，注明作者。</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/minimax-h3-prompt-skill-T8">minimax-h3-prompt-skill-T8</a></td><td>案例库，可作为 Agent 技能安装，附桌面查看器。</td></tr>
<tr><td width="56"><img src="https://github.com/joeVenner.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/joeVenner/awesome-minimax-h3">awesome-minimax-h3 (skills)</a></td><td>通过托管 API 把需求一路做成成片的 Agent 技能。</td></tr>
<tr><td width="56"><img src="https://github.com/okaypic.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/okaypic/okaypic-video-skill">okaypic-video-skill</a></td><td>通过 okaypic API 生成带字幕多镜头成片的 Agent 技能。</td></tr>
<tr><td width="56"><img src="https://github.com/SlavaSexton.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SlavaSexton/ComfyUI-Agent-Kit">ComfyUI-Agent-Kit</a></td><td>让编程 Agent 驱动本地 ComfyUI，内含 H3 技能。</td></tr>
<tr><td width="56"><img src="https://github.com/unknowlei.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unknowlei/minimax-h3-opencode-skills">minimax-h3-opencode-skills</a></td><td>OpenCode 技能集：导演、多镜头规划、写提示词和审片。</td></tr>
<tr><td width="56"><img src="https://github.com/pytraveler.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI">MiniMax-H3-Prompt-Rewriter-ComfyUI</a></td><td>LightX2V 提示词改写 LoRA 的 ComfyUI 节点，另能给参考素材写描述。</td></tr>
<tr><td width="56"><img src="https://github.com/benjiyaya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/benjiyaya/ComfyUI-H3-VisionPromptor">ComfyUI-H3-VisionPromptor</a></td><td>用本地视觉模型把想法和参考图扩写成 H3 提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/Rimagination.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Rimagination/h3lite">h3lite</a></td><td>Codex 技能，按电脑配置选择 ComfyUI 方案，在本地生成 H3 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/benjiyaya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill">Minimax-H3-Prompt-AgentSkill</a></td><td>根据你的素材和想法生成格式化的 H3 提示词，交回给你自己运行。</td></tr>
<tr><td width="56"><img src="https://github.com/Adudeguyman.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder">ComfyUI-Fantastic-MiniMaxH3-PromptBuilder</a></td><td>每种模式的可填写提示词模板，边写边按官方指南检查。</td></tr>
<tr><td width="56"><img src="https://github.com/duckyshell.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer">ComfyUI-MiniMaxH3-Prompt-Writer</a></td><td>ComfyUI 里的提示词写作面板，把一段简单描述和参考素材写成官方 H3 格式，可用 Ollama、本地 GGUF 或 API 模型。</td></tr>
<tr><td width="56"><img src="https://github.com/gnipbao.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gnipbao/minimax-h3-video-reverse-skill">minimax-h3-video-reverse-skill</a></td><td>Agent 技能，读一段参考视频，写成 H3 文生视频提示词或逐镜头的图生视频提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/phileiny.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/phileiny/h3-storyboard-skill">h3-storyboard-skill</a></td><td>Claude Code 技能，把剧本拆成 H3 镜头并指导面部表演，每条规则都标明是实测还是推断。</td></tr>
<tr><td width="56"><img src="https://github.com/TFboy1.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TFboy1/oh-my-minimaxh3-director">oh-my-minimaxh3-director</a></td><td>Agent 技能，把剧本拆成分镜，用本地 ComfyUI 工作流批量生成，再拼成剪映草稿。</td></tr>
<tr><td width="56"><img src="https://github.com/Sac-Y.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Sac-Y/MiniMax-H3-Cloud">MiniMax-H3-Cloud</a></td><td>Codex 插件，在优云智算上租用 GPU 并装好 ComfyUI，跑文生视频、关键帧或参考图生视频，结束后关机。</td></tr>
<tr><td width="56"><img src="https://github.com/jtydhr88.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jtydhr88/ComfyTV">ComfyTV</a></td><td>ComfyUI 上的画布应用，内置 Agent 自带一个 H3 电影导演技能。</td></tr>
<tr><td width="56"><img src="https://github.com/zenstory-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zenstory-ai/drama-skills">drama-skills</a></td><td>从小说到分镜再到提示词的短剧技能集，H3 模式会绑定参考图和说话人。</td></tr>
<tr><td width="56"><img src="https://github.com/eternityspring.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/eternityspring/shuohao-skills">shuohao-skills</a></td><td>短剧技能集，产出角色、美术、剧本设定和分镜，分镜环节把剪辑点和关键帧对齐到 H3 提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/LoveRain1997.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LoveRain1997/h3-prompt-journal">h3-prompt-journal</a></td><td>以音乐节拍驱动的舞蹈和多镜头提示词案例，每篇附完整提示词和经验总结。</td></tr>
<tr><td width="56"><img src="https://github.com/SkyNotSilent.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SkyNotSilent/awesome-MiniMax-H3-cases">awesome-MiniMax-H3-cases</a></td><td>3000 多个可播放的 H3 视频，均注明出处，另有 700 多条完整提示词和本地运行教程。</td></tr>
<tr><td width="56"><img src="https://github.com/BMB12d3.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BMB12d3/minimax-h3-prompt-composer">H3 Prompt Composer</a></td><td>单个 HTML 文件的离线提示词构建工具，直接在浏览器里运行。</td></tr>
<tr><td width="56"><img src="https://github.com/scenario-labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scenario-labs/skills">Scenario skills</a></td><td>调用 Scenario API 的 Agent 技能集，其中的视频技能讲解 H3 的关键帧和参考模式以及原生音频提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/ChatCut-Inc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ChatCut-Inc/agent-plugin">ChatCut Agent Plugin</a></td><td>供编程 Agent 使用的 AI 视频编辑工具，视频技能把 H3 和 H3 Max 任务提交到它的托管服务。</td></tr>
<tr><td width="56"><img src="https://github.com/zenstory-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zenstory-ai/oh-story-dsh">oh-story-dsh</a></td><td>面向网文和短剧的 DeepSeek Harness 插件，视频通过 MiniMax 官方 API 用 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/LIUFelix2004.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LIUFelix2004/God-minmax-H3">God MiniMax H3</a></td><td>提示词库和 Agent 技能，用分段接力做 30 秒的 H3 成片，每条提示词附 API 参数。</td></tr>
<tr><td width="56"><img src="https://github.com/10291124jcjc-dotcom.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/10291124jcjc-dotcom/ancient-rap-battle">ancient-rap-battle</a></td><td>Claude Code 技能，做历史人物说唱对决视频，可选用 MiniMax 官方 API 的 H3 对口型。</td></tr>
<tr><td width="56"><img src="https://github.com/suihe1.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/suihe1/short-drama-production">short-drama-production</a></td><td>Codex 技能，管理短剧从大纲到粗剪的全流程，H3 任务提交到 MiniMax 官方 API 或优云智算。</td></tr>
<tr><td width="56"><img src="https://github.com/gary149.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gary149/h3-game-sprites">h3-game-sprites</a></td><td>Agent 技能，把 OpenRouter 上生成的 H3 动作片段做成 2D 精灵图，每个动作一段，逐帧抠像和检查。</td></tr>
<tr><td width="56"><img src="https://github.com/guangjun5952.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/guangjun5952/minimax-h3-broll-generator">MiniMax H3 B-roll Generator</a></td><td>Codex 技能，根据逐字稿规划 B-roll、写 H3 提示词，审批后通过 MiniMax 官方 API 或秘塔生成镜头。</td></tr>
<tr><td width="56"><img src="https://github.com/gooseworks-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gooseworks-ai/goose-skills">Goose Skills</a></td><td>面向编程 Agent 的增长营销技能库，其中的创作者组件用 fal 上的 H3 Max 生成达人口播视频。</td></tr>
<tr><td width="56"><img src="https://github.com/makevoid.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/makevoid/motion-graphics-music-video-skill">Motion Graphics Music Video Skill</a></td><td>Claude 插件，用一首音乐和一句提示词生成动画视频，可通过 fal 用 H3 Max 图生视频。</td></tr>
</table>

<a id="h3-training"></a>

### 训练

<table>
<tr><td width="56"><img src="https://github.com/IAmIronMan42.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/IAmIronMan42/MiniMax-H3-FineTuning">MiniMax-H3-FineTuning</a></td><td>基于 diffusers 实现的监督式 LoRA 训练，损失函数里包含音频。</td></tr>
<tr><td width="56"><img src="https://github.com/verl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/verl-project/verl-omni">verl-omni</a></td><td>用 FlowGRPO 和 DiffusionNFT 对 H3 LoRA 做强化学习后训练。</td></tr>
<tr><td width="56"><img src="https://github.com/radixark.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/radixark/miles_diffusion">miles_diffusion</a></td><td>Flow-GRPO 后训练，附 H3 文生音视频方案。</td></tr>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/Fizgig">Fizgig</a></td><td>LoRA 和 LoKr 训练工作室，带 H3 预设。</td></tr>
<tr><td width="56"><img src="https://github.com/kohya-ss.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kohya-ss/musubi-tuner">musubi-tuner</a></td><td>LoRA 训练脚本，实验性支持 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/tdrussell.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tdrussell/diffusion-pipe">diffusion-pipe</a></td><td>流水线并行的训练脚本，支持 H3 的文生图和文生音视频训练。</td></tr>
<tr><td width="56"><img src="https://github.com/bghira.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/bghira/SimpleTuner">SimpleTuner</a></td><td>图像、视频和音频模型的微调工具包，支持 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/ByteDance-Seed.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ByteDance-Seed/VeOmni">VeOmni</a></td><td>字节跳动的训练框架，附 H3 FL2VA 和 Ref2VA 的配置。</td></tr>
<tr><td width="56"><img src="https://github.com/Tencent-Hunyuan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Tencent-Hunyuan/UniRL">UniRL</a></td><td>腾讯混元的强化学习框架，H3 文生音视频目前只支持训练端。</td></tr>
<tr><td width="56"><img src="https://github.com/X-GenGroup.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/X-GenGroup/Flow-Factory">Flow-Factory</a></td><td>流匹配模型微调框架，附 H3 文生音视频、首尾帧和参考生音视频的 GRPO LoRA 配置。</td></tr>
</table>

<a id="h3-serving"></a>

### 部署与托管

<table>
<tr><td width="56"><img src="https://github.com/sgl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sgl-project/sglang">SGLang</a></td><td>原生 H3 服务，<a href="https://docs.sglang.ai/cookbook/diffusion/MiniMax/MiniMax-H3">cookbook</a> 里的方案从单卡 RTX 4090 到 B300、AMD MI355X 都实测过。</td></tr>
<tr><td width="56"><img src="https://github.com/vllm-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vllm-project/vllm-omni">vLLM-Omni</a></td><td>兼容 OpenAI 的 <code>/v1/videos</code> 服务，附<a href="https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md">官方 H3 方案</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/hao-ai-lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hao-ai-lab/uniserve">UniServe</a></td><td>FastH3 和 FastH3-Omni 的 HTTP 服务，支持 1 到 8 卡。</td></tr>
<tr><td width="56"><img src="https://github.com/EzgiTastan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/EzgiTastan/minimax-h3-gb300-ebpf">minimax-h3-gb300-ebpf</a></td><td>SGLang 跑 H3 在 GB300 上的可复现性能测试和追踪。</td></tr>
<tr><td width="56"><img src="https://github.com/fal-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://fal.ai">H3 Max on fal</a></td><td>fal 后训练的托管模型 H3 Max、Director 和 Turbo，权重不公开，也可从 ComfyUI Partner Nodes 调用。</td></tr>
<tr><td width="56"><img src="https://github.com/jd-nuva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://nuvalab.ai/minimax-h3-dedicated-deployment/">Nuva Lab</a></td><td>为单家公司提供的专属 H3 和 FastH3 部署，可运行客户自己的微调权重。</td></tr>
<tr><td width="56"><img src="https://github.com/fal-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://fal.ai/models/minimax/h3/text-to-video">MiniMax H3 on fal</a></td><td>fal API 上的原版 H3，支持文生视频、<a href="https://fal.ai/models/minimax/h3/image-to-video">图生视频</a>和<a href="https://fal.ai/models/minimax/h3/reference-to-video">参考生视频</a>，最高 2K。</td></tr>
<tr><td width="56"><img src="https://github.com/replicate.png?size=96" alt="" width="48"></td><td width="240"><a href="https://replicate.com/minimax/h3">MiniMax H3 on Replicate</a></td><td>Replicate 预测 API 上的 H3，支持文生视频、首尾帧和参考生视频，可选 768P 或 2K。</td></tr>
<tr><td width="56"><img src="https://github.com/WaveSpeedAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://wavespeed.ai/models/wavespeed-ai/minimax-h3/text-to-video">MiniMax H3 on WaveSpeed</a></td><td>以 API 形式提供的 H3 开源权重，支持文生视频、图生视频和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenRouterTeam.png?size=96" alt="" width="48"></td><td width="240"><a href="https://openrouter.ai/minimax/hailuo-3">H3 on OpenRouter</a></td><td>通过 OpenRouter 的 API 调用 H3 和 H3 Max 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/cloudflare.png?size=96" alt="" width="48"></td><td width="240"><a href="https://developers.cloudflare.com/ai/models/minimax/h3/">MiniMax H3 on Cloudflare</a></td><td>Cloudflare AI 模型目录里的 H3，支持文生视频、首尾帧和参考生视频。</td></tr>
</table>

<a id="h3-guides"></a>

### 指南

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="docs/run-h3-locally.md">Run H3 locally</a></td><td>本仓库的本地运行指南：显存配置、所有量化文件、组件、兼容性和实测环境。</td></tr>
<tr><td width="56"><img src="https://github.com/jd-nuva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://nuvalab.ai/blog/run-minimax-h3-locally-gpu-vram-requirements">GPU and VRAM requirements</a></td><td>Nuva Lab 关于本地运行 H3 所需显卡和显存的说明，另有和 API 的<a href="https://nuvalab.ai/blog/minimax-h3-api-price-vs-self-hosting">成本对比</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/matsuo-koya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matsuo-koya/minimax-h3-notes">minimax-h3-notes</a></td><td>单机运行笔记。</td></tr>
<tr><td width="56"><img src="https://github.com/IvenKooLab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/IvenKooLab/minimax-h3-turing">minimax-h3-turing</a></td><td>22 GB Turing 显卡的实战手册，用 W4A8 加 Turbo。</td></tr>
<tr><td width="56"><img src="https://github.com/FlowForgeLabAi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlowForgeLabAi/-h3-8gb-traps">-h3-8gb-traps</a></td><td>8 GB 笔记本的实测方案，每个坑都附复现。</td></tr>
</table>

## MiniMax-H3-Image

MiniMax-H3-Image 是 H3 系列中开放权重的图像模型，一个权重同时支持文生图和多图编辑。它 6 步就能直接生成约 1600 万像素的图片，宽高比从 1:3 到 3:1，最多可用 16 张参考图。

<a id="img-official"></a>

### 官方资源

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-H3-Image">MiniMax-H3-Image</a></td><td>官方仓库：使用指南、效果展示、本地部署和提示词编写技能。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image">MiniMaxAI/MiniMax-H3-Image</a></td><td>原始权重，支持 1600 万像素的生成和编辑，最多 16 张参考图；也可从 <a href="https://modelscope.cn/models/MiniMax/MiniMax-H3-Image">ModelScope</a> 下载。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE">MiniMaxAI/MiniMax-H3-Image-PE</a></td><td>可选的提示词扩写模型，把简短提示词扩写成模型需要的详细英文描述。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630433fea362e7e8b51f5e2b/bsDc8lR72KhKvm-wXxYsv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Comfy-Org/MiniMax-H3-Image">Comfy-Org/MiniMax-H3-Image</a></td><td>原生支持所用的 ComfyUI 打包权重。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image">ComfyUI tutorial</a></td><td>ComfyUI 官方教程，附<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json">文生图</a>和<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json">编辑</a>模板；另见<a href="https://blog.comfy.org/p/minimax-h3-image-is-now-available">发布公告</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://platform.minimax.io/docs/guides/image-generation">MiniMax API</a></td><td>官方 API，支持文生图、图像编辑和提示词扩写；中国大陆请用 <a href="https://platform.minimax.cn/docs/guides/image-generation">platform.minimax.cn</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://hailuoai.video/">Hailuo AI</a></td><td>在网页上或<a href="https://agent.minimax.io/">桌面应用</a>里直接使用 H3-Image；中国大陆请用 <a href="https://hailuoai.com/">hailuoai.com</a> 和 <a href="https://agent.minimax.cn">agent.minimax.cn</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_t2i.json">ComfyUI API node</a></td><td>ComfyUI 自带的 API 节点，通过 Comfy 账号用 H3-Image 生成和编辑图片，另有<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_image_edit.json">编辑模板</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/cli">MiniMax CLI</a></td><td>官方命令行工具，可用 H3-Image 生成图片，并用最多 16 张参考图编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-MCP">MiniMax MCP</a></td><td>官方服务，让 Agent 默认用 H3-Image 生成图片，并可用最多 16 张参考图编辑。</td></tr>
</table>

<a id="img-quantizations"></a>

### 量化与格式转换

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/unsloth/MiniMax-H3-Image-FP8">unsloth/MiniMax-H3-Image-FP8</a></td><td>INT8 ConvRot 和 FP8 版 DiT，各 20.3 GB，Unsloth Desktop 会自动选用。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/23f3af29b8e3f4317298458cafc54454.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LSDJesus/MiniMax-H3-Image-fp8-nvfp4">LSDJesus/MiniMax-H3-Image-fp8-nvfp4</a></td><td>给 ComfyUI 用的 FP8 scaled（21 GB）和 NVFP4（12.5 GB）；另有 Q6_K 到 Q2_K 的 <a href="https://huggingface.co/LSDJesus/MiniMax-H3-Image-GGUF">GGUF</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF">Abiray/MiniMax-H3-Image-GGUF</a></td><td>剪枝版 DiT 的 GGUF，6.88 GB 起，敏感层保留 FP32，可直接用 ComfyUI-GGUF 加载。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a7d89f6560102d6b0e60442/wLhDigI2l39VFPC_OBAFK.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Hearthscale/MiniMax-H3-Image-GGUF">Hearthscale/MiniMax-H3-Image-GGUF</a></td><td>给 stable-diffusion.cpp 用的 Q8_0、Q5_K_M、Q4_K_M 和 Q3_K_M。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69cf8fa5b7bee3980535206f/_wFUtjc_L4WrMgAdpDwKn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/AtomicChat/MiniMax-H3-Image-GGUF">AtomicChat/MiniMax-H3-Image-GGUF</a></td><td>按逐层敏感度扫描调过的 GGUF，7.6 到 21.4 GB，附测量数据。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/c88172b8c192ca50aa900233ae6d3c13.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lingjiu09/minimax-h3-image-gguf">lingjiu09/minimax-h3-image-gguf</a></td><td>Q3_K_M 和 Q4_0，在 8 GB 笔记本上测过。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6357da9b68659b265eb84aab/8PEgQsPKTJVQQD2_87mP9.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tsolful/Minimax_H3_Image_INT6_INT4GPTQ">tsolful/Minimax_H3_Image_INT6_INT4GPTQ</a></td><td>给 ComfyUI 用的 GPTQ W6A8（15.6 GiB）和 W4A8（12.8 GiB）。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/4d09b9d42626184db42666bbf1642ab1.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/speedyrulz/minimax-h3-image-int8-w4a8-hybrid">speedyrulz/minimax-h3-image-int8-w4a8-hybrid</a></td><td>INT8 和 W4A8 混合，14.8 GiB，面向 16 GB 显卡。</td></tr>
<tr><td width="56"><img src="https://www.gravatar.com/avatar/d651eb3d698c95ec23696040b7c7c6c3?d=retro&size=100" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/PotatoForge/Minimax-H3-Image-W4A4">PotatoForge/Minimax-H3-Image-W4A4</a></td><td>实验性 W4A4，11.35 GB，面向较老的显卡。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8">WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8</a></td><td>W4A8 的 DiT 和文本编码器，总共 27.5 GB。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-Image-Pruned">WaveCut/MiniMax-H3-Image-Pruned</a></td><td>去掉用不到的权重，从 135 GB 减到 95 GB，文生图结果逐位一致；附<a href="https://github.com/iamwavecut/MiniMax-H3-Image-Pruned">构建脚本</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630cc06015433862cfc66a16/Wam_ZPEs8SLh5wtA7Xoms.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/SD-Next/MiniMax-H3-Image-SDNQ-uint4-svd-hadamard-dynamic">SD-Next SDNQ uint4</a></td><td>给 diffusers 和 SD.Next 用的 SDNQ 4 bit；OzzyGT 另有 <a href="https://huggingface.co/OzzyGT/minimax_h3_image_sdnq_dynamic_4bit">4 bit</a> 和 <a href="https://huggingface.co/OzzyGT/minimax_h3_image_sdnq_dynamic_8bit">8 bit</a> 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65273adf77bceabaab23ee5c/6h4l-LlfXHKiqR6-15P8n.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/nativ-community/MiniMax-H3-Image-4bit">nativ-community/MiniMax-H3-Image-4bit</a></td><td>给 mlx-vlm 用的 MLX 4 bit（54.5 GB），另有 <a href="https://huggingface.co/nativ-community/MiniMax-H3-Image-8bit">8 bit 版</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69bf5ffd2a6ad8b6410a5f3f/T6MxkTP8x6ZDrrQ5SMVG1.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/VisualInference/MiniMax-H3-Image-MLX-bf16">VisualInference/MiniMax-H3-Image-MLX-bf16</a></td><td>给 minimax-h3-image-mlx 用的 MLX BF16，cguldogan 另有 <a href="https://huggingface.co/cguldogan/MiniMax-H3-Image-MLX-4bit">4 bit</a> 和 <a href="https://huggingface.co/cguldogan/MiniMax-H3-Image-MLX-8bit">8 bit</a> 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/635c610bfa3c172ddf40d116/pgN34290L5I0dhy0W-E8V.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Sawfwair/MiniMax-H3-Image-MLX-8bit">Sawfwair/MiniMax-H3-Image-MLX-8bit</a></td><td>给 mere.run 用的 MLX 8 bit 导出（52 GB）。</td></tr>
</table>

<a id="img-adapters"></a>

### 蒸馏与适配器

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1674929746905-60f1abe7544c2adfd699860c.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step">minimax-h3-image-pdmd-2step</a></td><td>两步 pDMD 蒸馏 LoRA，附<a href="https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step">在线演示</a>。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/e2e8b6608dad261a033ecf26a75cbb05.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter">Fizgig H3-Image Training Adapter</a></td><td>训练 LoRA 时一直挂着的 rank 16 适配器。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/64b90c0ce3d41dbd696b0bd0/O6FfUDzAildors6vyrvQK.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Felldude/MiniMaxImage_hdr_diffusers_vae">Felldude/MiniMaxImage_hdr_diffusers_vae</a></td><td>解码器微调过的 VAE，以像素精度换取更多高频细节，附与原版 VAE 的对比数据。</td></tr>
</table>

<a id="img-runtimes"></a>

### 推理引擎

<table>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/ComfyUI">ComfyUI</a></td><td>原生支持，附官方模板。</td></tr>
<tr><td width="56"><img src="https://github.com/sgl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sgl-project/sglang">SGLang</a></td><td>用 main 分支部署，<a href="https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-Image">cookbook</a> 有从 DGX Spark 到 B300 的方案，以及给 12 到 48 GB 显卡的卸载方案。</td></tr>
<tr><td width="56"><img src="https://github.com/vllm-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vllm-project/vllm-omni">vLLM-Omni</a></td><td>提供服务，附官方<a href="https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image">方案</a>，包括 AMD Instinct。</td></tr>
<tr><td width="56"><img src="https://github.com/huggingface.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/huggingface/diffusers">diffusers</a></td><td>以 Modular Diffusers 模块接入，支持文生图和参考图编辑，v0.43.0 起发布，见<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image">文档</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/LightX2V">LightX2V</a></td><td>支持文生图和单图或多图参考编辑，可单卡运行，也可多卡 TP/SP 并行。</td></tr>
<tr><td width="56"><img src="https://github.com/modelscope.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a></td><td>支持推理、低显存推理，以及全量或 LoRA 训练。</td></tr>
<tr><td width="56"><img src="https://github.com/leejet.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/leejet/stable-diffusion.cpp">stable-diffusion.cpp</a></td><td>纯 C/C++ 推理，首发即支持 H3-Image 和 GGUF 模型。</td></tr>
<tr><td width="56"><img src="https://github.com/unslothai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unslothai/unsloth">Unsloth</a></td><td>在桌面应用和网页界面里运行 H3-Image，使用 INT8 或 FP8 版 DiT。</td></tr>
<tr><td width="56"><img src="https://github.com/niw.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/niw/mmh3">mmh3</a></td><td>Rust 写的推理引擎，带 CUDA 和 Metal 算子。</td></tr>
<tr><td width="56"><img src="https://github.com/Anionex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Anionex/h3-image-mlx">h3-image-mlx</a></td><td>在 M1 到 M4 的 Mac 上用 MLX 生成和编辑，支持 W8A8，4K 出图峰值内存约 7 GB。</td></tr>
<tr><td width="56"><img src="https://github.com/felix-ab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/felix-ab/minimax-h3-image-mlx">minimax-h3-image-mlx</a></td><td>BF16 的 MLX 移植，边计算边从磁盘读下一块权重，2048×2048 只占 5 GB 内存。</td></tr>
<tr><td width="56"><img src="https://github.com/zhongkaifu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zhongkaifu/TensorSharp">TensorSharp</a></td><td>.NET 推理引擎，在 ggml 上跑 H3-Image 文生图，编辑还没实现。</td></tr>
<tr><td width="56"><img src="https://github.com/animede.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/animede/minimax-h3-image-server">minimax-h3-image-server</a></td><td>带网页界面的服务端，可生成和参考图编辑；默认约需 21 GB 显存，用 W4A8 加权重流式加载可降到 8 GB。</td></tr>
<tr><td width="56"><img src="https://github.com/devin-lai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/devin-lai/MiniMax-H3-Image-Accel">MiniMax-H3-Image-Accel</a></td><td>针对 L20 和 RTX 5090 调优的 SGLang，默认 BF16，可选 FP8 降低延迟。</td></tr>
<tr><td width="56"><img src="https://github.com/Neywa-Labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Neywa-Labs/mlx-vlm">mlx-vlm</a></td><td>在苹果芯片上用 MLX 跑 H3-Image 文生图和图像编辑，支持 BF16、8 bit 和 4 bit 权重。</td></tr>
</table>

<a id="img-apps"></a>

### 应用

<table>
<tr><td width="56"><img src="https://github.com/mcmonkeyprojects.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</a></td><td>网页界面，跑 H3-Image 时最多可放 16 张图做参考或编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/tin2tin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tin2tin/Pallaidium">Pallaidium</a></td><td>Blender 视频编辑器插件，支持 H3-Image 文生图和最多 10 张参考图的编辑，用 SDNQ 4 bit，24 GB 显卡放得下。</td></tr>
<tr><td width="56"><img src="https://github.com/SHoogstad.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SHoogstad/diffusion-for-krita">diffusion-for-krita</a></td><td>以 ComfyUI 为后端的 Krita 插件，用 H3-Image 生成、局部重绘和编辑，最多 16 张图。</td></tr>
<tr><td width="56"><img src="https://github.com/stepupgaming.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/stepupgaming/styk-paint">styk-paint</a></td><td>Godot 插件，通过 ComfyUI 让 H3-Image 把 3D 简模场景逐个视角重绘，再把画面投影回场景。</td></tr>
<tr><td width="56"><img src="https://github.com/ryoppippi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ryoppippi/mmh3-studio">mmh3-studio</a></td><td>常驻 mmh3 服务的 macOS 和浏览器客户端，出图不用每次重新加载模型。</td></tr>
<tr><td width="56"><img src="https://github.com/BesianSherifaj-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BesianSherifaj-AI/h3-image-editor">h3-image-editor</a></td><td>基于 ComfyUI 的 Windows 编辑工作台，用 H3-Image 生成、参考图编辑和蒙版修补，另有 Qwen、SAM、SeedVR2 等工具。</td></tr>
<tr><td width="56"><img src="https://github.com/MrJackSpade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MrJackSpade/ImageGenerationServer">ImageGenerationServer</a></td><td>多用户网页前端，后端接本地 ComfyUI，可用 H3-Image 生成和最多 16 张图编辑；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/tong-io.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tong-io/tongflow-modal-minimax-h3-image">tongflow-modal-minimax-h3-image</a></td><td>TongFlow 插件，在 Modal 上跑 H3-Image 的生成、编辑和最多 16 张图的融合。</td></tr>
</table>

<a id="img-api"></a>

### API 集成

<table>
<tr><td width="56"><img src="https://github.com/heygen-com.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/heygen-com/hyperframes">HyperFrames</a></td><td>为 Agent 设计的 HTML 转视频工具，素材命令可通过 HeyGen 用 H3-Image 生成 2K 或 4K 图片。</td></tr>
<tr><td width="56"><img src="https://github.com/vercel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vercel/ai">AI SDK</a></td><td>Vercel 的 TypeScript 工具包，MiniMax provider 可用 H3-Image 及高速版生成和编辑图片。</td></tr>
<tr><td width="56"><img src="https://github.com/looplj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/looplj/reveric">Reveric</a></td><td>小说和短剧桌面创作工作台，可通过 MiniMax 官方 API 用 H3-Image 生成和编辑图片。</td></tr>
<tr><td width="56"><img src="https://github.com/langgenius.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/langgenius/dify-official-plugins/tree/main/tools/minimax_image">Dify MiniMax Image plugin</a></td><td>Dify 工具插件，通过 MiniMax 官方 API 用 H3-Image 生成图片，并可用 1 到 16 张参考图编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/ThinkFlowLab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ThinkFlowLab/OpenDirector">OpenDirector</a></td><td>面向 Agent 的视频工作流工具，带 H3-Image 适配器，可通过 MiniMax 官方 API 用最多 16 张参考图编辑图片。</td></tr>
<tr><td width="56"><img src="https://github.com/OctaFuse.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OctaFuse/octafuse-gateway">Octafuse Gateway</a></td><td>可自托管的 AI 网关，提供 H3-Image 生成和编辑的预设，上游是 MiniMax 兼容 OpenAI 的图片接口。</td></tr>
</table>

<a id="img-comfyui"></a>

### ComfyUI 节点与工作流

<table>
<tr><td width="56"><img src="https://github.com/Rinne414.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Rinne414/ComfyUI-MiniMaxH3-Tools">ComfyUI-MiniMaxH3-Tools</a></td><td>编码后先释放文本编码器再加载 DiT，让 H3-Image 能在 24 GB 显存加 32 GB 内存上运行。</td></tr>
<tr><td width="56"><img src="https://github.com/nkxx188.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy">ComfyUI-MiniMaxH3-Easy</a></td><td>其中的 Easy Image 节点做 H3-Image 文生图和参考图编辑，在提示词里输入 @ 插入参考图。</td></tr>
<tr><td width="56"><img src="https://github.com/pxyllvr.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pxyllvr/ComfyUI-H3Image-MediaLoader">ComfyUI-H3Image-MediaLoader</a></td><td>加载最多 16 张参考图，每格可裁剪、旋转和调整顺序。</td></tr>
<tr><td width="56"><img src="https://github.com/DeepNeuralNerd.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/DeepNeuralNerd/H3-Video-LoRA-Loader-for-Image">H3-Video-LoRA-Loader-for-Image</a></td><td>在 H3-Image 上使用 H3 视频 LoRA。</td></tr>
<tr><td width="56"><img src="https://github.com/KadoBOT.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio">ComfyUI-H3-Continuity-Studio</a></td><td>用 H3 视频拼接多段镜头的节点包，其中的 H3-Image 节点生成引导每个镜头的静帧。</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8">comfyui-minimax-h3-prompt-enhancer-T8</a></td><td>提示词扩写节点，v1.41.0 起支持 H3-Image。</td></tr>
<tr><td width="56"><img src="https://github.com/iamwavecut.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/iamwavecut/ComfyUI-OrbitQuant">ComfyUI-OrbitQuant</a></td><td>运行 OrbitQuant W4A8 版本，支持生成和最多 16 张参考图的编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/Hejunlong27.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Hejunlong27/H3GIF">H3GIF</a></td><td>让 H3-Image 在一张图里画出分镜网格，再切成单帧，导出循环 GIF。</td></tr>
<tr><td width="56"><img src="https://github.com/BuffMcBigHuge.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BuffMcBigHuge/Comfiest">Comfiest</a></td><td>由 agent 驱动的 ComfyUI 工作流套件，含 H3-Image 文生图和 1、2、4 张参考图的编辑工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/kuhung.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kuhung/minimax-h3-colab-workflow">minimax-h3-colab-workflow</a></td><td>在 Colab 的 L4 上用开放权重跑 H3 和 H3-Image 的笔记本，基于 ComfyUI 量化权重和 stable-diffusion.cpp。</td></tr>
<tr><td width="56"><img src="https://github.com/Timster666.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Timster666/comfyui-minimax-image">comfyui-minimax-image</a></td><td>启动即进入 ComfyUI、自带 H3-Image 模型的 RunPod 模板。</td></tr>
<tr><td width="56"><img src="https://github.com/yilives.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yilives/minimax-h3-v100-guide">minimax-h3-v100-guide</a></td><td>在一张 16 GB V100 加 32 GB 内存上实测过的 H3 和 H3-Image 部署：配置、启动器和工作流。</td></tr>
</table>

<a id="img-training"></a>

### 训练

<table>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/Fizgig">Fizgig</a></td><td>LoRA 和微调工作室，v9.6 起支持 H3-Image。</td></tr>
<tr><td width="56"><img src="https://github.com/ostris.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ostris/ai-toolkit">ai-toolkit</a></td><td>有图形界面和命令行的训练工具，数据集带控制图时可训练 H3-Image 的编辑能力。</td></tr>
</table>

<a id="img-prompting"></a>

### 提示词

<table>
<tr><td width="56"><img src="https://github.com/AI-KSK.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AI-KSK/minimax-h3-image-prompts">minimax-h3-image-prompts</a></td><td>可复用的技能，用来写生成和编辑提示词，最多 16 张参考图。</td></tr>
<tr><td width="56"><img src="https://github.com/BeatAPI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BeatAPI/awesome-minimax-h3-image-prompts">awesome-minimax-h3-image-prompts</a></td><td>63 个注明出处的 H3-Image 案例，附结果图和编辑参考图；只有 5 个附了完整提示词。</td></tr>
</table>

<a id="img-benchmarks"></a>

### 评测

<table>
<tr><td width="56"><img src="https://github.com/jo-nike.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jo-nike/h3-image-bench">h3-image-bench</a></td><td>在 16 GB 的 RTX 5080 上以约 1600 万像素跑两轮共 600 条打分提示词，附在线结果页。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69c9b513833ed97d9b2f808d/xDH70MJgqXXRZn1ekMirh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench">M5 Pro Mac mini benchmark</a></td><td>在 64 GB 的 M5 Pro Mac mini 上用 ComfyUI 跑 INT8 版：4096×4096 一张约 9 分钟。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/9947803a52252be0e6db004495bb2a75.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/datasets/LIOSAM007/h3-image-12gb-tests">12 GB tests</a></td><td>12 GB 的 RTX 4070 SUPER 加 96 GB 内存跑 INT8 版：4 MP 约 11 到 12 秒，16 MP 约 41 秒。</td></tr>
</table>

## 贡献

在对应的表格里加一行：作者头像、链接到项目的名称、一句话介绍。`README.md` 里也加上同样的一行英文，然后运行 `python3 scripts/check_readme.py` 检查。

收录条件：项目自己的 README、文档或模型卡写明它是什么，并且用到了 MiniMax H3 或 MiniMax-H3-Image，本地运行或通过 API 调用都可以。测速数据、显存配置和逐个文件的说明请写进 [docs/run-h3-locally.md](docs/run-h3-locally.md)。

<a id="credits"></a>

## 致谢

这份索引离不开所有发布模型、工具、测试和文档的人。感谢 MiniMax、ComfyUI、SGLang、vLLM、NVIDIA 和 Unsloth 团队，也感谢在真实硬件上持续测试 H3 的各位独立维护者。

最初的结构和大部分资源来自社区维护的 [`wildminder/awesome-minimax-H3`](https://github.com/wildminder/awesome-minimax-H3)。本页在它的基础上，加入了 MiniMax 自己对 GitHub 和 Hugging Face 的检索结果。

特别感谢：

* [Comfy-Org](https://huggingface.co/Comfy-Org)：官方 ComfyUI 转换、工作流模板和首发支持。
* [ModelTC / LightX2V](https://github.com/ModelTC/LightX2V)：Turbo 蒸馏和公开的训练配置。
* [`Larryvrh`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)：Turbo 推荐背后的权重对比。
* [`Kijai`](https://github.com/kijai/ComfyUI-SolAttn_triton)：NVIDIA Sol-Attn 的实现和测速记录。
* [Alibaba PAI](https://huggingface.co/alibaba-pai) 和 [`aigc-apps/VideoX-Fun`](https://github.com/aigc-apps/VideoX-Fun)：PDD 8 步加速 LoRA 和 Fun ControlNet-Union 控制分支。
* [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)：随权重一起公开了 Video Delta Net 的训练代码。
* [FastVideo](https://huggingface.co/FastVideo)：FastH3 系列（4 步预览、V2、Trim 和 Omni）、免数据蒸馏方案和 UniServe。
* [Nuva Lab](https://nuvalab.ai/fasth3/)：FastH3 的生产验证、FastH3-Omni 的联合开发和参考数据集，以及 UniServe 测速所用的生产硬件。
* [Pika Labs](https://github.com/Pika-Labs/Pika-H3-Flex)：6 步 Flex 适配器和推理示例；[NVIDIA](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K)：公开的草稿精修模型和快速解码器，基于 LBH、LightX2V 和 LMS 的组件。
* [`IAmIronMan42`](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning)：训练工作和记录下来的修复。
* [Salvatore Sanfilippo (antirez)](https://github.com/antirez)：独立的 H3 C/Metal 推理引擎。
* 上面提到的每一位量化和工作流维护者。你们的文件、测试时间和文档让本地运行 H3 方便了很多。

如果发现数字有误、链接失效或缺少兼容性说明，欢迎提 issue 或直接修正。

## 联系我们

MiniMax H3 相关问题请联系 [model@minimax.io](mailto:model@minimax.io)。


---

## FILE: docs/run-h3-locally.md (https://github.com/MiniMax-AI/awesome-minimax-h3-integration/blob/HEAD/docs/run-h3-locally.md)

# Run MiniMax H3 locally

Detailed notes for running MiniMax H3 on your own hardware: which files to pick for your GPU, every community quantization, components, acceleration families, compatibility and measured environments. The project index is in the [README](../README.md).

## Where to start

This navigation guide is not a complete compatibility list.

| Goal | Start with |
| :--- | :--- |
| Run on your GPU | [Run locally](#models) — pick a stack from the [VRAM table](#recipes-vram) |
| Work with audio | [`comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) and [audio VAE](#components-vae) |
| Build in ComfyUI | [Official tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) · [Workflows & nodes](../README.md#nodes) |
| Write better prompts | [Prompting](../README.md#recipes-prompt) |
| Make it faster | [Speed](#speed) — [Turbo](#turbo) · [PDD](#pdd) · [FastH3](#fasth3) · [VDN](#vdn) |
| Control pose / depth / edges | [Fun Control](#funcontrol) |
| Use the hosted model | [Hosted H3 Max](../README.md#hosted) |
| Fine-tune it | [Training & LoRAs](../README.md#training) |
| Serve it as an API | [Serving H3](../README.md#partners) |
| Run it on a dedicated host | [Dedicated deployment](../README.md#dedicated) — a private H3 / FastH3 host for one company |
| Run on Apple Silicon | [`antirez/h3.c`](https://github.com/antirez/h3.c) |

<a id="models"></a>

## Run it locally

MiniMax-H3 generates video with **native stereo audio** from text, images, video, and audio inputs. It supports clips up to **2K** and **15 seconds**. Two base variants exist:

* **H3-Base-FL2VA** (first-and-last-frame mode) — accepts zero, one, or two input images. Zero images for text-to-video; one image for first- *or* last-frame-to-video; two images for first-and-last-frame-to-video.
* **H3-Base-Ref2VA** (omni-reference mode) — accepts up to **9 images, 3 video clips (2–15 s each), and 3 audio clips**, for a maximum of **12 files** total.

Checkpoints are the same size. FL2VA, trained only with keyframes, typically yields better raw output. Ref2VA accepts more reference material but has lower base quality; the [Ref Patch](#refpatch) can partially bridge this gap.

<a id="recipes-vram"></a>

### By VRAM and hardware

Find your GPU in the table, then use the notes to inform your configuration.

| Situation | Stack | Why this combination |
| :--- | :--- | :--- |
| **24 GB, first run** | `pruned_int8_convrot` DiT (19.53 GiB) + TE `nvfp4_awq` (14.61 GiB) + [`ComfyUI-MiniMaxH3-Easy`](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy) | Easy routes T2V, I2V, first/last-frame, and reference input through a single `Media` port. Sampling, LoRAs, and decoding remain outside the node for later modification. |
| **24 GB, want speed** | The above + [TE-Speed-MiniMaxH3](https://github.com/tl2012tl/TE-Speed-MiniMaxH3) + Turbo `v4_step600_ema` at **6–8 steps** | Block-cache acceleration; v3.2 adapts to current ComfyUI's block prefetch and no longer patches core files. 6–8 steps reduce Turbo motion smear. |
| **22 GB, Turing (sm_75)** | [`minimax-h3-turing`](https://github.com/IvenKooLab/minimax-h3-turing) — W4A8 mixed + Turbo 4-step field handbook | 5.7 min/clip baseline; T8 BlockCache drafts −43%; PDD 8-step + T8 combo 210 s/clip. sm_75 has no FP8/BF16 tensor cores — quantization, launch flags and kernel support all differ from Ampere+ |
| **12–16 GB** | Pruned `Q4_K_M` GGUF (10.64 GiB) or pruned `nvfp4` (11.67 GiB) + TE `Q2_K` (7.91 GiB) + fp8mix VAE pair | GGUF offers the most size options, beneficial for tight memory. `IQ1_S` is smaller at 3.78 GiB, but quality noticeably drops. |
| **8 GB** | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) NF4 path | The project states 8 GB as its minimum for this path. Offloading performs most work here; expect slow performance, not just small memory footprint. |
| **8 GB laptop (sm_120) + 16 GB RAM** | [`-h3-8gb-traps`](https://github.com/FlowForgeLabAi/-h3-8gb-traps) — pruned INT8 ConvRot + nvfp4 text encoder, with a reproduction for each trap | The contributor reports 10 s / 768×1024 / **8 steps in 14.3 min**, 20 steps in 33.8–43.5 min. `--disable-pinned-memory` is required — `MAX_PINNED_MEMORY = ram * 0.40` otherwise kills the process. The binding limit is the **16 GB of system RAM, not the 8 GB of VRAM**: a ~34.5 GiB working set pages, and the symptom is a stalled sampler while `utilization.gpu` still reads 99% (`nvidia-smi` power is the tell — flat 34–36 W against 64–98 W healthy; 0.9–6 s/block fresh vs 212 s/block after ~1.7 h), so restart ComfyUI before each run. At H3's real geometry comfy-kitchen INT8 attention measures **22.8 ms** against 143.1 ms for SDPA. Three-tier companion pack: [`MiniMax-H3-8GB-Workflows`](https://github.com/FlowForgeLabAi/MiniMax-H3-8GB-Workflows) |
| **RTX 50-series / Blackwell** | [NVIDIA Sol-Attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | **1.14–1.44×** faster than SageAttention with **−37 %** MLP peak VRAM, measured on a 5090. SM89–SM121, Triton 3.6.0. Also unlocks Blackwell-only hybrid-NVFP4 checkpoints. |
| **H200 / B200, faster than real time** | [`OpenVDN/vdn-minimax-h3`](https://github.com/OpenVDN/vdn-minimax-h3), hybrid-attention H3 + 8-step adapter | Datacenter path, not a VRAM-saving one: the bf16 H3 base plus a linear-attention branch and two LoRAs. A 768p, 14.4 s clip denoises in **90.5 s on one H200**, **51 s on one B200**, and **11.23 s on 8×B200**. See [VDN](#vdn). |
| **Multi-shot / long video** | [`ComfyUI-H3-Motion-Context`](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | H3 generates in blocks up to 15 s. Motion-Context feeds the previous block's final frame **and** audio forward, preserving motion direction and speed. |
| **Pose / depth / edge control** | [`alibaba-pai/MiniMax-H3-Fun-Controlnet-Union`](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) + [`ComfyUI-H3-FunControl`](https://github.com/wyzborrero/ComfyUI-H3-FunControl) | One ~6.8 GiB control branch covers Canny, Depth, HED, MLSD, Pose and video inpainting. See [Fun Control](#funcontrol). |
| **Fewest steps** | [PDD 8-step Acc LoRA](#pdd) or [FastH3](#fasth3) | Alternatives to Turbo: PDD at 8 steps; FastH3 has a 4-step preview and 8-step V2 / Trim releases with different runtime requirements. |
| **Storyboard / timeline** | [`ComfyUI_MiniMaxH3_Director`](https://github.com/huangserva/ComfyUI_MiniMaxH3_Director) | Five importable templates: t2v, fl2v, r2v, v2v, and rv2v. |
| **Inpaint / local edit** | [`scraed/LanPaint`](https://github.com/scraed/LanPaint) | v2.1.0 fixed H3 support. Training-free video **and** audio inpainting. |
| **Apple Silicon** | [`antirez/h3.c`](https://github.com/antirez/h3.c) (MIT, Metal-native) | h3.c supports T2V/A, first-last-frame, and ordered Ref2VA references end-to-end, with M3 Max / M5 Max performance optimization ongoing. |
| **One-command local** | [`agent-next/video-agent`](https://github.com/agent-next/video-agent) | "Ollama for video models" — `install` · `pull` · `run`. |

<a id="checkpoints"></a>

### Checkpoints

| Source | What it is | Files | Total |
| :--- | :--- | :---: | ---: |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Original diffusers weights — `transformer/` (FL2VA) and `transformer_ref/` (Ref2VA) at 14 shards / 61.73 GiB each, plus text encoder, video VAE, audio VAE, and self-contained `FL2VA/` and `Ref2VA/` pipeline folders | 280 | 464.2 GiB |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | ComfyUI-repackaged single-file weights — 10 diffusion models, 3 text encoders, video + audio VAE | 17 | 433.2 GiB |

| Variant | Name | Precision | Size | Download |
| :--- | :--- | :---: | :---: | :---: |
| FL2VA | `minimax_h3_fl2va` | ![bf16][badge-bf16] | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors) |
| FL2VA | `minimax_h3_fl2va` | ![int8][badge-int8] | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) |
| FL2VA | `minimax_h3_fl2va_pruned` | ![bf16][badge-bf16] | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors) |
| FL2VA | `minimax_h3_fl2va_pruned` | ![fp8][badge-fp8] | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors) |
| FL2VA | `minimax_h3_fl2va_pruned` | ![int8][badge-int8] | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| Ref2VA | `minimax_h3_ref2va` | ![bf16][badge-bf16] | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors) |
| Ref2VA | `minimax_h3_ref2va` | ![int8][badge-int8] | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) |
| Ref2VA | `minimax_h3_ref2va_pruned` | ![bf16][badge-bf16] | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors) |
| Ref2VA | `minimax_h3_ref2va_pruned` | ![fp8][badge-fp8] | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors) |
| Ref2VA | `minimax_h3_ref2va_pruned` | ![int8][badge-int8] | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) |

**"Pruned"** means AdaLN-pruned. These models are roughly 40% smaller, work only in ComfyUI, and underpin most of the consumer-GPU quants below. For a 24 GB card, `pruned_int8_convrot` (**19.53 GiB**) is the usual starting point.

<a id="quants"></a>

### Quantized Models

MiniMax provides original BF16 checkpoints. The files below are community conversions and repackaged variants, not official MiniMax releases. Confirm compatibility with your runtime's documentation before downloading.

<details>
<summary><b>Community FL2VA conversions</b></summary>

| Pruned | Precision | Method | Size | Download |
| :---: | :---: | :--- | :---: | :--- |
| | ![bf16][badge-bf16] | BF16 | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors) |
| | ![int8][badge-int8] | ConvRot | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) |
| | ![int8][badge-int8] | ConvRot Lean (HQ) | 21.91 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-INT8-ConvRot-HQ.safetensors) |
| | ![int8][badge-int8] | ConvRot | 20.94 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ConvRot Lite | 20.33 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-INT8-ConvRot-Lite.safetensors) |
| | ![fp8][badge-fp8] | FP8 E4M3FN | 43.78 GiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_fp8_e4m3fn.safetensors) |
| | ![mxfp8][badge-mxfp8] | MXFP8 | 44.34 GiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_mxfp8.safetensors) |
| | ![fp8][badge-fp8] | FP8 + FP16 attention | 26.70 GiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_fp16attn_fp8.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 (HQ) | 13.60 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-NVFP4-HQ.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 10.86 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-NVFP4.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 32.05 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_fl2va_nvfp4.safetensors) |
| | ![int4][badge-int4] | NF4 (DiffSynth) | 15.98 GiB | [![][gh-DiffSynth-Studio]](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4/resolve/main/minimax-h3-fl2va-nf4.safetensors) |
| | | OrbitQuant W4A4 | 17.03 GiB | [![][gh-WaveCut]](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/transformer/diffusion_pytorch_model-00001-of-00005.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot | 21.00 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/FL2VA/MiniMax-H3_FL2VA-DT-sQKV-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot Lean | 27.99 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/FL2VA/MiniMax-H3_FL2VA-DT-sQKV-INT8-ConvRot-HQ.safetensors) |
| ✓ | ![bf16][badge-bf16] | BF16 | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors) |
| ✓ | ![fp8][badge-fp8] | FP8 scaled | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors) |
| ✓ | ![int8][badge-int8] | ConvRot | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) ┊ [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors) |
| ✓ | ![int8][badge-int8] | Hybrid ConvRot (fl2va base + ref2va adaln b15-49) | 19.53 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b15-49-int8.safetensors) |
| ✓ | ![int8][badge-int8] | Hybrid ConvRot (fl2va base + ref2va adaln b20-49) | 19.53 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b20-49-int8.safetensors) |
| ✓ | ![int8][badge-int8] | Hybrid ConvRot (fl2va base + ref2va adaln b25-49) | 19.53 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors) |
| ✓ | ![int8][badge-int8] | Hybrid ConvRot (fl2va base + ref2va adaln b30-49) | 19.53 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_fl2va_pruned_nvfp4.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 + ConvRot INT8 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 11.67 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot | 14.81 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors) ┊ [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_fl2va_pruned_INT4BQ.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot Lean | 17.27 GiB | [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_fl2va_pruned_INT4Q.safetensors) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot | 15.67 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_fl2va_pruned_int4_convrot_simple.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot | 18.92 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_fl2va_pruned_mixed_int4_int8_convrot_simple.safetensors) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot | 16.67 GiB | [![][gh-Merserk]](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot (pruned) | 10.56 GiB | [![][gh-Merserk]](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| ✓ | ![int4][badge-int4] | W4A8 ConvRot | 11.68 GiB | [![][gh-AX1Y2JP]](https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors) ┊ [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors) ┊ [![][gh-Winnougan]](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors) |

*GGUF quants — see the [GGUF section](#gguf).*

</details>

<details>
<summary><b>Community Ref2VA conversions</b></summary>

| Pruned | Precision | Method | Size | Download |
| :---: | :---: | :--- | :---: | :--- |
| | ![bf16][badge-bf16] | BF16 | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors) |
| | ![int8][badge-int8] | ConvRot ┊ *(patchin HF 1.02)* | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) ┊ [![][gh-t8star]](https://huggingface.co/t8star/minimax_h3_ref2va_patchin_hf102/resolve/main/minimax_h3_ref2va_patchin_hf102_T8.safetensors) |
| | ![int8][badge-int8] | ConvRot Lean (HQ) | 21.91 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-INT8-ConvRot-HQ.safetensors) |
| | ![int8][badge-int8] | ConvRot | 20.94 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ConvRot Lite | 20.33 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-INT8-ConvRot-Lite.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 (HQ) | 13.60 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-NVFP4-HQ.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 10.86 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-NVFP4.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 32.05 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_ref2va_nvfp4.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 mixed | 22.76 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors) |
| | ![int4][badge-int4] | NF4 (DiffSynth) | 15.98 GiB | [![][gh-DiffSynth-Studio]](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4/resolve/main/minimax-h3-ref2va-nf4.safetensors) |
| | | OrbitQuant W4A4 | 17.03 GiB | [![][gh-WaveCut]](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/transformer_ref/diffusion_pytorch_model-00001-of-00005.safetensors) |
| | ![nvfp4][badge-nvfp4] | Hybrid NVFP4, FFN-only (Blackwell) | 16.38 GiB | [![][gh-abakanai]](https://huggingface.co/abakanai/Minimax_h3_hybrid/resolve/main/minimax_h3_ref2va_pruned_hybrid_ffn_nvfp4_blackwell.safetensors) |
| | ![nvfp4][badge-nvfp4] | Hybrid NVFP4, QKV+FFN (Blackwell) | 14.03 GiB | [![][gh-abakanai]](https://huggingface.co/abakanai/Minimax_h3_hybrid/resolve/main/minimax_h3_ref2va_pruned_hybrid_nvfp4_blackwell.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot | 21.00 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot Lean | 27.99 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot-HQ.safetensors) |
| ✓ | ![bf16][badge-bf16] | BF16 | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors) |
| ✓ | ![fp8][badge-fp8] | FP8 scaled | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors) |
| ✓ | ![int8][badge-int8] | ConvRot | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) ┊ [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_ref2va_pruned_nvfp4.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 + ConvRot INT8 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot/resolve/main/minimax_h3_ref2va_pruned_nvfp4_convrot_int8.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 11.67 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot | 14.06 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors) ┊ [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_ref2va_pruned_INT4BQ.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot Lean | 17.18 GiB | [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_ref2va_pruned_INT4Q.safetensors) |
| ✓ | ![int4][badge-int4] | W4A8 ConvRot | 11.68 GiB | [![][gh-AX1Y2JP]](https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot/resolve/main/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors) ┊ [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors) ┊ [![][gh-Winnougan]](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors) |

*GGUF quants — see the [GGUF section](#gguf).*

</details>

<details>
<summary><b>Community multi-tier repack</b></summary>

[`DeepBeepMeep/MiniMax-H3`](https://huggingface.co/DeepBeepMeep/MiniMax-H3) collects community files at several precisions. Check the model card for compatibility and licensing.

</details>

<details>
<summary><b>Quants of the accelerated variants (VDN · PDD · FastH3)</b></summary>

These are quantized copies of the [VDN](#vdn), [PDD](#pdd) and [FastH3](#fasth3) models rather than of the base checkpoints. Match the quant to the runtime the parent model expects.

| Parent | Precision | Method | Download |
| :--- | :---: | :--- | :--- |
| VDN | ![int8][badge-int8] | ConvRot, ComfyUI layout | [![][gh-drbaph]](https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui) |
| VDN | ![int8][badge-int8] | ConvRot, T8 repack | [![][gh-t8star]](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy) |
| PDD 8-step | ![nvfp4][badge-nvfp4] | SVDQuant FP4 | [![][gh-1ronman1993]](https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8) |
| PDD 8-step | ![int4][badge-int4] | SVDQuant INT4 | [![][gh-1ronman1993]](https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-int4-pdd8) |
| FastH3 4-step | GGUF | Community conversion | [![][gh-realrebelai]](https://huggingface.co/realrebelai/FastH3_GGUFs) |

</details>

<a id="gguf"></a>

#### GGUF Quantized Models

GGUF files are community conversions, not part of the official MiniMax H3 release. Use a compatible loader and follow its documentation.

<details>
<summary><b>Community FL2VA GGUF conversions</b></summary>

| Pruned | Quant | Size | Download |
| :---: | :---: | :---: | :--- |
| | ![Q3_K_M][badge-Q3_K_M] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q3_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q3_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q3_K_M.gguf) |
| | ![Q3_K_S][badge-Q3_K_S] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q3_K_S.gguf) |
| | ![Q4_0][badge-Q4_0] | 17.36 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q4_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_0.gguf) |
| | ![Q4_1][badge-Q4_1] | 20.41 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_1.gguf) |
| | ![Q4_K_M][badge-Q4_K_M] | 18.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q4_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q4_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_K_M.gguf) |
| | ![Q4_K_S][badge-Q4_K_S] | 18.49 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q4_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_K_S.gguf) |
| | ![Q5_0][badge-Q5_0] | 21.21 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q5_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_0.gguf) |
| | ![Q5_1][badge-Q5_1] | 24.17 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_1.gguf) |
| | ![Q5_K_M][badge-Q5_K_M] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q5_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_K_M.gguf) |
| | ![Q5_K_S][badge-Q5_K_S] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q5_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_K_S.gguf) |
| | ![Q6_K][badge-Q6_K] | 26.28 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q6_K.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q6_K.gguf) |
| | ![Q8_0][badge-Q8_0] | 33.56 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q8_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q8_0.gguf) |
| ✓ | ![IQ1_S][badge-IQ1_S] | **3.78 GiB** — smallest DiT published | [![][gh-MarxistLeninist]](https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF/resolve/main/minimax_h3_fl2va_pruned-IQ1_S.gguf) |
| ✓ | ![IQ1_M][badge-IQ1_M] | 4.22 GiB | [![][gh-MarxistLeninist]](https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF/resolve/main/minimax_h3_fl2va_pruned-IQ1_M.gguf) |
| ✓ | ![Q2_K][badge-Q2_K] | 6.26 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q2_K.gguf) ┊ [![][gh-leejet]](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| ✓ | ![UD-Q2_K_XL][badge-UD-Q2_K_XL] | 7.51 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf) |
| ✓ | ![Q3_K_M][badge-Q3_K_M] | 8.16 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q3_K.gguf) |
| ✓ | ![Q3_K_M][badge-Q3_K_M] | 8.29 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| ✓ | ![UD-Q3_K_XL][badge-UD-Q3_K_XL] | 8.90 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf) |
| ✓ | ![Q4_K_M][badge-Q4_K_M] | 10.64 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q4_K.gguf) ┊ [![][gh-leejet]](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| ✓ | ![Q5_0][badge-Q5_0] | 12.97 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q5_0.gguf) |
| ✓ | ![Q6_K][badge-Q6_K] | 15.45 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q6_K.gguf) |
| ✓ | ![Q8_0][badge-Q8_0] | 19.97 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q8_0.gguf) |
| ✓ | ![Q8_0][badge-Q8_0] | 20.10 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |


</details>

<details>
<summary><b>Community Ref2VA GGUF conversions</b></summary>

| Pruned | Quant | Size | Download |
| :---: | :---: | :---: | :--- |
| | ![Q3_K_M][badge-Q3_K_M] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q3_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-REF2VA-Q3_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q3_K_M.gguf) |
| | ![Q3_K_S][badge-Q3_K_S] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q3_K_S.gguf) |
| | ![Q4_0][badge-Q4_0] | 17.36 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q4_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_0.gguf) |
| | ![Q4_1][badge-Q4_1] | 20.41 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_1.gguf) |
| | ![Q4_K_M][badge-Q4_K_M] | 18.49 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q4_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-REF2VA-Q4_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_K_M.gguf) |
| | ![Q4_K_S][badge-Q4_K_S] | 18.49 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q4_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_K_S.gguf) |
| | ![Q5_0][badge-Q5_0] | 21.21 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q5_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_0.gguf) |
| | ![Q5_1][badge-Q5_1] | 24.17 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_1.gguf) |
| | ![Q5_K_M][badge-Q5_K_M] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q5_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_K_M.gguf) |
| | ![Q5_K_S][badge-Q5_K_S] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q5_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_K_S.gguf) |
| | ![Q6_K][badge-Q6_K] | 26.28 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q6_K.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q6_K.gguf) |
| | ![Q8_0][badge-Q8_0] | 33.56 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q8_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q8_0.gguf) |
| ✓ | ![Q2_K][badge-Q2_K] | 6.22 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q2_K.gguf) |
| ✓ | ![Q3_K_M][badge-Q3_K_M] | 8.12 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q3_K.gguf) |
| ✓ | ![Q4_0][badge-Q4_0] | 10.60 GiB | [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) |
| ✓ | ![Q4_K_M][badge-Q4_K_M] | 10.60 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q4_K.gguf) |
| ✓ | ![Q5_0][badge-Q5_0] | 12.94 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q5_0.gguf) |
| ✓ | ![Q6_K][badge-Q6_K] | 14.00 GiB | [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) ‡ |
| ✓ | ![Q6_K][badge-Q6_K] | 15.42 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q6_K.gguf) |
| ✓ | ![Q8_0][badge-Q8_0] | 18.77 GiB | [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) § |
| ✓ | ![Q8_0][badge-Q8_0] | 19.94 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q8_0.gguf) ┊ [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) |

‡ molbal calls this one **`U16G`** — a hand-tuned mixed layout rather than a stock `Q6_K`. § molbal's **`Q8_CR`** is `Q8_0` with ConvRot applied. Both are in the same repo as the plain quants; read the filename.

</details>

<details>
<summary><b>Fine-tuned checkpoint quants</b></summary>

`DmitryDB` also publishes stock-compatible quants of community **fine-tunes** of H3 — fine-tuned QKV weights in blocks 0–31 preserved alongside a tested quantization layout, no custom node or core patch required (the ConvRot / NVFP4 tiers match the base-model tiers exactly: 21.91 / 20.94 / 13.60 / 10.86 GiB, plus a ⚠️ DT-sQKV build at 21.00 GiB).

The fine-tunes themselves are third-party and are not itemised here; browse [`DmitryDB`'s model list](https://huggingface.co/DmitryDB) directly for the current set.

</details>

#### Notes

* **`DmitryDB/MiniMax-H3-INT8-Lean-ConvRot`** and **`DmitryDB/MiniMax-H3-ComfyUI-Quants`** are the *same repo* — the author merged and rebranded. Likewise **`…-INT8-Lean-ConvRot-Dynamic-Time-Separate-QKV`** and **`…-DynTime-sQKV`**. Both names in each pair resolve to the same files, so don't download twice.
* **`t8star/minimax_h3_ref2va_patchin_hf102`** is a weight *modification*, not a quant: +2 % on the 2×2 spatial high-frequency patch in the video-input projection. The author's own tests showed a weak HF-agent gain and did **not** confirm the "oily/waxy" look was removed. Treat as experimental.
* **`Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI`** ships a matching quantized text encoder: [`qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors`](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors).
* **`unsloth/MiniMax-H3-GGUF`** also carries Qwen3-VL text-encoder GGUFs: `Q2_K_M` 12.2 GiB and `Q4_K_M` 17.0 GiB.
* **`DmitryDB/MiniMax-H3-ComfyUI-Quants`** also carries VAE files: video VAE FP16 4.85 GiB, audio VAE FP32 577 MiB.
* **VDN and PDD quants are not interchangeable with base quants.** A VDN INT8 file expects the VDN adapters loaded; a SVDQuant PDD build already has the 8-step LoRA folded in, so stacking another acceleration LoRA on top will over-distill the result.
* **`DiffSynth-Studio/MiniMax-H3-NF4`** bundles NF4 TE + video VAE + audio VAE. Requires [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio); the project states a **minimum of 8 GB VRAM** on this path.
* **`WaveCut/MiniMax-H3-OrbitQuant-W4A4`** bundles a quantized TE and FP32 VAE copies, and requires the [`ComfyUI-OrbitQuant`](https://github.com/iamwavecut/ComfyUI-OrbitQuant) node — the W4A4 path is not loadable without it. [Workflow JSON](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json).

<a id="text-encoder"></a>

### Text encoders

MiniMax-H3 uses **Qwen3-VL-32B** for text and vision. On a 24 GB card, you will usually need to reduce the text encoder after reducing the DiT. The options are listed below.

#### Comfy-Org (official repackage)

| Model | Precision | Size | Download |
| :--- | :---: | :---: | :--- |
| `qwen3vl_32b_minimax_h3` | ![bf16][badge-bf16] | 47.97 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![int8][badge-int8] | 25.28 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![nvfp4][badge-nvfp4] | **14.61 GiB** | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |

The `nvfp4_awq` build at **14.61 GiB** is the smallest official TE and the one to pair with a pruned INT8 DiT on a 24 GB card.

#### Community quantizations

| Model | Precision | Size | Source |
| :--- | :---: | :---: | :--- |
| `qwen3vl_32b_minimax_h3` | ![Q4_K_M][badge-Q4_K_M] | 13.58 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf) |
| `qwen3vl_32b_minimax_h3` | ![int4][badge-int4] | 13.93 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![nvfp4][badge-nvfp4] | 25.28 GiB † | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![int4][badge-int4] | 15.35 GiB | [![][gh-AX1Y2JP]](https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot) — W4A8 ConvRot |
| `qwen3vl_32b_minimax_h3` | ![int4][badge-int4] | — | [![][gh-Winnougan]](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors) — W4A8 ConvRot |
| `qwen3vl_32b_minimax_h3` | ![Q2_K][badge-Q2_K] | 12.2 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — `Q2_K_M` GGUF |
| `qwen3vl_32b_minimax_h3` | ![Q4_K_M][badge-Q4_K_M] | 17.0 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — `Q4_K_M` GGUF |
| `qwen3vl_32b_minimax_h3` | ![Q2_K][badge-Q2_K] | **7.91 GiB** | [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) — smallest TE published |
| `qwen3vl_32b_minimax_h3` | ![int8][badge-int8] | 24.89 GiB | [![][gh-DeepBeepMeep]](https://huggingface.co/DeepBeepMeep/MiniMax-H3) — quanto-INT8 ⚠️ no license |

† `Abiray`'s `nvfp4_awq` file is byte-for-byte the size of Comfy-Org's **INT8** build, not of an NVFP4 one. Check the file before assuming it is a smaller download.

<a id="components"></a>

### VAE and components

<a id="components-vae"></a>

#### VAE (video & audio)

Both VAEs are **required** for every generation workflow — H3 decodes video and audio through separate autoencoders.

| Component | Source | Precision | Size | Download |
| :--- | :--- | :---: | :---: | :--- |
| Video VAE | Comfy-Org | ![fp16][badge-fp16] | 4.85 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | Comfy-Org | ![fp32][badge-fp32] | 577 MiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| Video VAE | Kijai | ![int8][badge-int8] | 2.95 GiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors) |


<a id="tae"></a>

#### Tiny Autoencoder (TAE) — previews only

A quickly-trained 2D tiny VAE by [Kijai](https://huggingface.co/Kijai/MiniMax-H3-TAE). The author's own assessment: not a great outcome, but it still beats `latent2rgb` for previews. **9 MiB.** Currently only usable through the `ModelPreviewOverride` node in [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).

| Component | Size | Download |
| :--- | :---: | :--- |
| TAE (preview VAE) | 9 MiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors) |

<a id="imagevae"></a>

#### Image VAE (single-frame)

An experimental image-specialised H3 VAE that decodes a single temporal latent (`T=1`) into one still. Merged checkpoint — no custom node needed.

⚠️ **For image workflows only.** The image-tuned decoder materially regresses multi-frame video reconstruction, so keep the original VAE loaded for video.

| Component | Size | Download |
| :--- | :---: | :--- |
| Single-image VAE (step 1597) | 4.85 GiB | [![][gh-Mamad8]](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/resolve/main/minimax_h3_t1_image_vae_step1597.safetensors) |

<a id="refpatch"></a>

#### Ref Patch — FL2VA that behaves more like Ref2VA

Diffs the **112 keys shared** between the `ref2va` and `fl2va` weights and stores the differences as a single 148 MiB patch, letting the lighter FL2VA checkpoint partially mimic Ref2VA behaviour. Apache-2.0. Requires the [`ComfyUI-MiniMaxH3_Ref-Patch`](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3_Ref-Patch) node.

| Component | Size | Download |
| :--- | :---: | :--- |
| Ref Patch | 148 MiB | [![][gh-lihaoyun6]](https://huggingface.co/lihaoyun6/MiniMax-H3-Ref-Patch) |

<a id="funcontrol"></a>

#### Fun Control — Canny / Depth / HED / MLSD / Pose + inpainting

[`alibaba-pai/MiniMax-H3-Fun-Controlnet-Union`](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) adds structural control to H3 without retraining the base model. It is a **single ~6.8 GiB control branch** — one `control_proj_in` plus five `control_blocks` grafted onto layers 0, 10, 20, 30 and 40 of the DiT's 50 — and one checkpoint covers every control type rather than one file per modality.

| | |
| :--- | :--- |
| Control types | Canny · Depth · HED · MLSD · Pose, plus video inpainting and outpainting |
| Control input | `control_in_dim = 49` — the control video and the inpaint mask enter through the same projection |
| Strength | `control_context_scale` from **0** (ignore) to **1** (follow closely) |
| Guidance | **Guidance-distilled**: run at `guidance_scale = 1.0`. Raising CFG does not help and costs a second forward pass. |
| Frame count | Snaps to `17*n + 5`, up to 15 s at 24 fps |
| Upstream | [`aigc-apps/VideoX-Fun`](https://github.com/aigc-apps/VideoX-Fun) (★2244, Apache-2.0) — the training and inference code |
| ComfyUI | [`wyzborrero/ComfyUI-H3-FunControl`](https://github.com/wyzborrero/ComfyUI-H3-FunControl) (★34, Apache-2.0). Core ComfyUI support is open as draft PR [#15860](https://github.com/comfyanonymous/ComfyUI/pull/15860). |

<a id="speed"></a>

## Speed

Three levers stack: **fewer steps**, **cheaper steps**, and — newest — **a cheaper model**, where the architecture itself changes rather than the schedule. Check your PyTorch build first; an outdated CUDA build commonly causes slow generations.

Choose the checkpoint or adapter, sampler and schedule as one recipe. Do not stack distillation adapters unless the project explicitly supports the combination.

| Family | Steps | Comes from | Pick it when |
| :--- | :---: | :--- | :--- |
| [Turbo](#turbo) | 4–8 | LightX2V DMD distillation, plus community re-bakes | You want the most tested option with the widest checkpoint choice. |
| [PDD](#pdd) | 8 | Alibaba PAI, on NVIDIA's Parallel Decoding Distillation | You want a single official-ish 8-step LoRA per variant and native ComfyUI loading. |
| [FastH3](#fasth3) | 4 / 8 | FastVideo, distilled H3 checkpoints and optimized runtimes | Preview, V2 and Trim for T2VA; [Omni](#fasth3-omni) for image/video/audio-reference generation. Choose the checkpoint and runtime together. |
| [Pika H3 Flex](#pika-h3-flex) | 6 | Pika Labs, FL2VA / Ref2VA adapters | Native 480p/768p video with stereo audio and multimodal references, using the released six-step schedule. |
| [VDN](#vdn) | 8 or 50 | OpenVDN, hybrid linear/softmax attention | You want an architectural speedup rather than a step-count trick, and have the disk for it. |
| [PTD](#ptd) | 4 / 6 | EPFL VITA, Parametric Trajectory Distillation | Re-noised SDE adapters for four-/six-step sampling; full-model ComfyUI validation covers the paper's four-step checkpoint only. |
| [DMAD](#dmad) | 4 / 8 | Distribution Matching as Adversarial Distillation | Four- and eight-step T2VA student LoRAs with Diffusers, ComfyUI workflows and public training code. |

Attention-level work sits outside that table: [VC-Attention](#vcattention) makes each step's attention cheaper instead of removing steps, and is published but unreleased.

Field note from the 2080Ti and 4090 threads: PDD at 4 NFE visibly flattens painted-illustration texture where Turbo `v4-600-ema` still holds it. At the LoRA's intended 8 steps the gap mostly closes.

<a id="turbo"></a>

### Turbo (Acceleration LoRA)

Turbo LoRAs are community acceleration models. Start with [`ModelTC/Minimax-H3-Turbo`](https://github.com/ModelTC/Minimax-H3-Turbo) and [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo), then use the workflow instructions provided by the project you choose. At 4 steps the audio track can degrade along with fast motion; 6–8 steps helps both.

<details>
<summary><b>Community Turbo checkpoint reference</b></summary>

| Variant | Steps | Base | Precision | Size | Download |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `fl2v v0.1` | 4 | Full | ![bf16][badge-bf16] | 1.29 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors) |
| `fl2v v1.0 768p` | 4 | Full | ![bf16][badge-bf16] | 1.29 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors) |
| `fl2v v1.0 768p · comfyui` | 4 | Full | ![bf16][badge-bf16] | 1.82 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) |
| `fl2v v1.0` | 8 | Full | ![bf16][badge-bf16] | 1.29 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors) |
| `fl2v v1.0 · comfyui` | 8 | Full | ![bf16][badge-bf16] | 1.82 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) |
| `lightx2v v0.1` | 4 | Full | ![bf16][badge-bf16] | 1.82 GiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors) |
| `lightx2v v0.1 · rank-21 resize` | 4 | Full | ![bf16][badge-bf16] | 300 MiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors) |
| `fl2v` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step.safetensors) |
| `fl2v ema` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema.safetensors) |
| `fl2v ckpt500` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ckpt500.safetensors) |
| `fl2v ema ckpt500` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema_ckpt500.safetensors) |
| `fl2v ckpt850` ← best 4-step under motion | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ckpt850.safetensors) |
| `fl2v ema ckpt850` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema_ckpt850.safetensors) |
| `fl2v v4 step600` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600.safetensors) |
| **`fl2v v4 step600 ema`** ← recommended default | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors) |
| `fl2v pruned` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_pruned_comfyui.safetensors) |
| `fl2v pruned ema` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors) |
| `fl2v pruned ckpt500` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors) |
| `fl2v pruned ema ckpt500` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors) |
| `fl2v pruned ckpt850` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors) |
| `fl2v pruned ema ckpt850` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors) |
| `fl2v pruned v4 step600` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors) |
| `fl2v pruned v4 step600 ema` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors) |
| `fl2v v1.0 768p · rank-21 resize` | 4 | Pruned | ![bf16][badge-bf16] | 298 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors) |
| `fl2v v1.0 · rank-21 resize` | 8 | Pruned | ![bf16][badge-bf16] | 327 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors) |
| `fl2v pruned ckpt500 V1` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_V1.safetensors) |
| `fl2v pruned ckpt600 V4` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt600_V4.safetensors) |
| `fl2v pruned ckpt600 ema V4` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors) |
| `fl2v pruned ckpt850 V1` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt850_V1.safetensors) |
| `fl2v diffusers` | 4 | Full | ![bf16][badge-bf16] | 0.79 GiB | [![][gh-InstantX]](https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers) |
| `fl2v` | 4 | Full | ![bf16][badge-bf16] | 717 MiB | [![][gh-joyfox]](https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/minimax_h3_fl2va_4step_lora.safetensors) |
| `fl2v step 100` | 8 NFE | Full | ![bf16][badge-bf16] | 738 MiB | [![][gh-tutututututu]](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors) |
| `fl2v step 200` | 8 NFE | Full | ![bf16][badge-bf16] | 738 MiB | [![][gh-tutututututu]](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors) |
| `fl2v step 300` | 8 NFE | Full | ![bf16][badge-bf16] | 738 MiB | [![][gh-tutututututu]](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors) |
| `fl2v 4-step` · ConvRot · ⚠️ dual-clock sampler or 8–10 steps | 4 | Full | ![int8][badge-int8] | 779.9 MiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp) |
| `fl2v 4-step ema` · ConvRot | 4 | Full | ![int8][badge-int8] | 779.9 MiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp) |
| `fl2v v4 step600 (T8-convert)` · ConvRot | 4 | Full | ![int8][badge-int8] | 779.9 MiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp/resolve/main/minimax_h3_turbo_v4_step600_comfyui_T8-convert.safetensors) |
| `lightx2v v0.1 · alpha8 T8-convert` · ConvRot · ⚠️ dual-clock sampler or 8–10 steps | 4 | Full | ![int8][badge-int8] | 1.82 GiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert) |
| `fl2v CMF` | 4 | Full | Q4TP (CMF) | 25.20 GiB | [![][gh-infosave]](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/mmh3-turbo-q4tp.cmf) |
| `fl2v CMF · FL2VA` | 4 | Full | Q4TP (CMF) | 25.70 GiB | [![][gh-infosave]](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/mmh3-turbo-fl2va-q4tp.cmf) |
| `fl2v CMF · FL2VA (smaller)` | 4 | Full | Q2TP (CMF) | 20.12 GiB | [![][gh-infosave]](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/mmh3-turbo-fl2va-q2tp.cmf) |

</details>

<a id="pdd"></a>

### PDD (Parallel Decoding Distillation)

PDD comes from NVIDIA — [arXiv 2607.26004](https://arxiv.org/abs/2607.26004), published 2026-07-28 — and was applied to H3 by Alibaba PAI as a pair of 8-step acceleration LoRAs.

[`alibaba-pai/MiniMax-H3-Acc-LoRAs`](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) ships one file per base variant:

| File | For | Rank / alpha | Precision |
| :--- | :--- | :---: | :---: |
| `MiniMax-H3-FL2VA-Acc-8Step.safetensors` | H3-Base-FL2VA | 64 / 64 | ![bf16][badge-bf16] |
| `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` | H3-Base-Ref2VA | 64 / 64 | ![bf16][badge-bf16] |

Released under the MiniMax-H3 Community License, trained and served through [`aigc-apps/VideoX-Fun`](https://github.com/aigc-apps/VideoX-Fun) — the same project behind [Fun Control](#funcontrol).

**Loading it.** The LoRA is not a single flat adapter: it is a backbone LoRA plus **32 output heads**, so a loader that expects one weight set will silently drop most of it. Current ComfyUI handles this natively through its LoRA-bank support. Otherwise use a repacked copy:

| Conversion | Downloads | Note |
| :--- | ---: | :--- |
| [![][gh-aptech0081]](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI) `MiniMax-H3-Acc-LoRAs-ComfyUI` | 31.6k | Straight ComfyUI key remap. |
| [![][gh-t8star]](https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy) `Minimax-H3-Super-Acceleration-Comfy` | 23.1k | T8's repack, aligned with the T8 audio node set. |

ComfyUI node: [`Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc`](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc) (★94, Apache-2.0) — use it if your ComfyUI predates LoRA-bank support.

<a id="fasth3"></a>
<a id="fasth3-4-step-preview-8-step-v2--trim"></a>

### FastH3 (preview, V2, Trim & Omni)

[FastVideo](https://github.com/hao-ai-lab/FastVideo) combines **video sparse attention (VSA)** with step distillation. The original 4-step preview remains available; V2 and experimental Trim use eight steps and target **text-to-audio-video**, without distilled FL2VA or Ref2VA support. The separate [Omni release](#fasth3-omni) adds eight-evaluation reference-conditioned video and audio generation.

| Model | Downloads | Note |
| :--- | ---: | :--- |
| [![][gh-FastVideo]](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) `FastH3-4-step-Preview-v1-VSA-DataFree` | 109.5k | Full weights. Data-free distillation, so no dataset licence question follows it. |
| [![][gh-FastVideo]](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA) `FastH3-4-step-Preview-v1-LoRA` | 20.9k | LoRA form, for stacking on a quant you already hold. |
| [![][gh-realrebelai]](https://huggingface.co/realrebelai/FastH3_GGUFs) `FastH3_GGUFs` | — | GGUF conversions for the low-VRAM path. |

FastVideo also publishes MLX INT8 and INT4 builds for Apple Silicon; check the [FastVideo org page](https://huggingface.co/FastVideo) for the current set. [`NikoDemon80/ComfyUI-FastH3-Lora-Converter`](https://github.com/NikoDemon80/ComfyUI-FastH3-Lora-Converter) (★7) converts the LoRA into ComfyUI key order.

#### V2 and consumer-hardware releases

* [FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) - Eight-forward, data-free DMD2 checkpoint with 80% VSA sparsity. Requires FastVideo's VSA-H3 backend and the checkpoint's trained schedule (video shift 10, not base H3's 12). T2VA only; FL2VA and Ref2VA were not distilled. Weights use the MiniMax H3 Community License.
* [FastH3 Trim NVFP4](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4) - Experimental eight-step T2VA model retaining 42 of 50 transformer blocks, with rank-16 AdaLN compression. The NVFP4 transformer is 11.1 GiB and targets Blackwell GPUs / DGX Spark; pruning trades quality for size and speed. Check the model and component terms separately from FastVideo's Apache-2.0 code license.
* [Consumer-hardware recipes and measurements](https://haoailab.com/blogs/fasth3-rtx/) - October 6 release covering NVFP4 on RTX 5090 / RTX PRO 6000 / DGX Spark, FP8 on RTX 4090, and INT6 through MLX on Apple Silicon. The reported 8 GB Trim result uses **an RTX 4090 capped to 8 GB**, not a physical 8 GB card: 82.0 s for a 5 s, 832×480 clip with audio, warm end-to-end through MP4 export; the authors expect real lower-memory cards to be slower.

<a id="fasth3-omni"></a>

#### FastH3 Omni — reference-conditioned video and audio

* [FastH3-Omni 8-Step V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) - BF16 release checkpoint using **eight transformer evaluations** with ordered image, video and audio references. Requires **at least one image or video**; accepts up to 9 images, 3 videos and 3 audio clips, with at most 12 references total. Outputs 24 FPS video with synchronized 32-kHz stereo audio. MiniMax-H3 Community License.
* [Release variants and runtime recipes](https://huggingface.co/collections/FastVideo/fasth3-omni) - BF16 server deployment through [UniServe](../README.md#uniserve), NVFP4 QAD profiles for Blackwell / DGX Spark / RTX 5090, an FP8 route for RTX 4090, and ComfyUI variants. Keep each recipe's attention, precision and scheduler settings; non-FP4 hardware needs compatible text-encoder components.
* [Release and matched examples](https://haoailab.com/blogs/fastvideo-omniref-update/) - Published October 20, 2026. The authors compare complete serving recipes on **8×GB200** across 27 matched workloads; timing and quality comparisons apply to those configurations, not every deployment.
* Built by FastVideo in collaboration with [Nuva Lab](https://nuvalab.ai/fasth3/). Per the [model card](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1), Nuva Lab co-developed the model and defined and curated the omni reference dataset. For a managed host, see [Dedicated deployment](../README.md#dedicated).

<a id="pika-h3-flex"></a>

### Pika H3 Flex

[Pika H3 Flex](https://github.com/Pika-Labs/Pika-H3-Flex) releases **six-step FL2VA and Ref2VA adapters** for native **480p/768p** video with synchronized stereo audio, with [weights](https://huggingface.co/Pika-Labs/Pika-H3-Flex), a CLI, Diffusers examples and ComfyUI workflows. FL2VA covers text and first/last-frame conditioning; Ref2VA accepts image, video and audio references and requires at least one image or video. Use the released adapter, sampler and schedule together.

The official CLI's [hardware guidance](https://github.com/Pika-Labs/Pika-H3-Flex/blob/main/docs/hardware.md) uses **H100 80 GB** as a validated environment, with CPU offload for larger requests; it runs one request on one GPU. Hosted 1080p/2K/4K refinement, audio stems and prompt enhancement are API extensions, not features of the released local weights. H3 Community License adapters and base weights; Apache-2.0 example code.

<a id="ptd"></a>

### PTD (Parametric Trajectory Distillation)

[`vita-epfl/PTD-inference`](https://github.com/vita-epfl/PTD-inference/tree/release/comfyui-inference) provides standalone **four-/six-step** inference and a native ComfyUI integration, with [PTD weights](https://huggingface.co/epfl-vita/PTD-weights), converters and example workflows. Requires the **original unpruned H3 base** and PTD-specific adapters, sampler and schedule; pruned, quantized, PDD and Turbo variants are unsupported.

The [v0.2.0 release candidate](https://github.com/vita-epfl/PTD-inference/blob/release/comfyui-inference/docs/RELEASE_NOTES.md) adds re-noised `h3_4step_sde` / `h3_6step_sde` recipes for the current Hugging Face checkpoints, which replace the paper's adapters. The authors validate standalone sampling against their research implementation on H100. Full-model ComfyUI validation covers **the paper's four-step T2V checkpoint with stereo audio**, launched with `--disable-dynamic-vram`; the new SDE adapters, six-step and image-conditioned ComfyUI generation remain unvalidated on the full model. Apache-2.0 code; MiniMax-H3 Community License weights.

<a id="dmad"></a>
<a id="dmad-four-step-audio-video-distillation"></a>

### DMAD (four-/eight-step audio-video distillation)

[`Yzmblog/DMAD`](https://github.com/Yzmblog/DMAD) publishes **four- and eight-step MiniMax-H3 text-to-audio-video LoRAs**, [Diffusers and native ComfyUI-layout weights](https://huggingface.co/ZhengmingYu/DMAD), example workflows and training code. The October 29 `8step_full_critic` release is a student trained from scratch for eight-step sampling. Use the matching student weights and documented sampler/schedule; this entry covers T2VA, not an Omni-reference checkpoint. The H3 code is Apache-2.0 and the weights use the MiniMax-H3 Community License; the separate image-training subfolder has different terms.

<a id="vdn"></a>

### VDN (Video Delta Net)

[`OpenVDN/vdn-minimax-h3`](https://github.com/OpenVDN/vdn-minimax-h3) (★397) is the one entry here that changes the architecture instead of the schedule. Softmax attention is quadratic in sequence length and dominates step time at long clips, so VDN splits it in two: a **sliding-window softmax branch** for nearby frames, and a **frame-wise linear-attention branch** for long-range context. The change is added to the frozen H3 weights as a linear branch plus two LoRA adapters, merged at inference. Released 2026-09-06; write-up at [openvdn.github.io](https://openvdn.github.io). Weights [![][gh-OpenVDN]](https://huggingface.co/OpenVDN/vdn-minimax-h3).

Both the training and the inference code are open, which is unusual in this list — most acceleration work publishes weights only. The licence is split: **Apache-2.0 for the code, MiniMax-H3 Community License for the weights.**

**Size.** The repo is roughly **82 GiB** in total. You do not need all of it:

| Directory | Size | What it is |
| :--- | :---: | :--- |
| `h3-base/` | 72 GiB | The rebuilt base. Only needed if you are training or evaluating from scratch. |
| `stage-b-step-2000/` | 4.3 GiB | 50-step adapter — the quality tier. |
| `stage-dmd-step-250/` | 5.1 GiB | 8-step DMD adapter — the speed tier. |

**Published figures.** **2.6–2.9×** faster per step than dense H3 on one H200/B200. A 768p, 14.4 s clip at 8 steps denoises in **90.5 s on one H200**, **51 s on one B200**, and **11.23 s on 8×B200** — faster than the clip plays, on that last configuration. Denoising only: VAE decode and muxing are excluded. All figures are the authors' own; see the [environment report](#compat-env) for the full stack they measured on.

| ComfyUI node | ⭐ | Note |
| :--- | ---: | :--- |
| [`Saganaki22/ComfyUI-VDN-H3`](https://github.com/Saganaki22/ComfyUI-VDN-H3) | 202 | The main node. Apache-2.0. Author reports **17 s/it** at 1280×736 / 145 frames on an RTX 5090. |
| [`Speach1sdef178/ComfyUI-VDN-H3-24GB`](https://github.com/Speach1sdef178/ComfyUI-VDN-H3-24GB) | 4 | 24 GB-targeted fork with heavier offload. |

Ready-made workflows for T2VA, I2VA, FL2VA, Ref2VA and reference audio ship with [`comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8), alongside a place-and-go model bundle [![][gh-t8star]](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy).

**The competing claim.** [`Kablex/ComfyUI-Ref2VA-VSA`](https://github.com/Kablex/ComfyUI-Ref2VA-VSA) (★111) applies video sparse attention to Ref2VA instead, and its author measures it at **2.24× faster than VDN-H3** — about **72 s per 5 s clip on an RTX 4090** in roughly **13.5 GiB**. Both sets of numbers come from the projects themselves, on different hardware; neither has been reproduced independently here.

INT8 conversions of VDN for ComfyUI are listed under [Quantized models](#quants).

<a id="solh3"></a>

### Sol-H3 (datacentre)

The `sol-engine` branch of [`NVlabs/Sana`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3) now also ships **Sol-H3**, a standalone runtime rather than a per-GPU config. It is 4-step distilled, scales with Ulysses sequence parallelism to 8 GPUs, and uses SOL/BSA sparse attention.

Published medians on **8×B300** at 1344×768 / 24 fps with stereo audio, warm, excluding load, warmup and encode:

| Frames | ≈ Duration | T2VA | Ref2VA |
| ---: | ---: | ---: | ---: |
| 124 | 5 s | 1.653 s | 2.192 s |
| 243 | 10 s | 3.732 s | 4.348 s |
| 362 | 15 s | 6.612 s | 5.947 s |

ComfyUI wrapper: [`xmarre/ComfyUI-Sol-H3`](https://github.com/xmarre/ComfyUI-Sol-H3) (★21, **GPL-3.0** — note the licence differs from the rest of this list).

<a id="vcattention"></a>

### VC-Attention (low-bit attention — paper only)

> ⚠️ **No code released.** VC-Attention is a published method, not something you can install. There is no repository, no kernel download and no license. It is indexed here because the measurements are on H3 and the technique is orthogonal to the four families above — not because you can run it today.

Rather than cutting steps, this line of work quantizes attention itself. **V-Smooth** reorders value tokens by lightweight online clustering to cut value-quantization error; **ExpCast-FP8** folds the softmax exponential into a single fused multiply-add. QK runs INT8, PV moves to FP8 E4M3. No retraining is required, and the authors state it composes with existing sparse-attention methods.

Published figures, the authors' own, measuring **the attention kernel** against BF16 FlashAttention-4:

| | B200 | B300 |
| :--- | ---: | ---: |
| VC-Attention | 1.59× | 1.51× |
| Nunchux Attention (proprietary) | 1.91× | 1.83× |

End-to-end generation moves considerably less than the kernel does: the paper reports **1.13–1.19×** on datacentre cards and **1.36–1.70×** on workstation cards. Fidelity across 100 prompts is **20.2 dB** mean PSNR against the BF16 reference, compared with 19.9 dB for SageAttention2.

⚠️ **Two things share one name.** *VC-Attention* is the paper. *Nunchux Attention* — the faster row above — is described by its authors as a **proprietary extension** available through [Nunchux](https://nunchux.ai)'s hosted API, not as code.

- Paper: [arXiv 2609.15810](https://arxiv.org/abs/2609.15810), 14 Sep 2026 — Xingyang Li, Dongyun Zou, Shining Zhang, Jiacheng Chen, Haocheng Xi, Lvmin Zhang, Jun-Yan Zhu, Song Han, Zhekai Zhang, Yujun Lin, Muyang Li (MIT, CMU, UC Berkeley, Stanford, NVIDIA)
- Write-up: [Attention is the video bottleneck](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck)

<a id="compat"></a>

## Compatibility & Licensing

### ComfyUI INT8

ComfyUI includes native INT8 support. Older I8Fast files use different tensor names; use [`convert_comfy_quant.py`](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) or download a native-format quant.

### ComfyUI patches

Some community tools modify or patch ComfyUI. Check the project's documentation and keep your ComfyUI version compatible.

| Type | Project | Notes |
| :--- | :--- | :--- |
| Runtime patch | [`DmitryDB/MiniMax-H3-DynTime-sQKV`](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV) | Required for its DT-sQKV files. |
| Runtime patch | [`ComfyUI-H3-Motion-Context`](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | Checks its ComfyUI assumptions at startup. |

<a id="compat-env"></a>

### Reported environments

* [`comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8): ComfyUI `0.31.0`, commit `cbbc9dab1`, Python 3.10+.
* [`minimax-h3-turing`](https://github.com/IvenKooLab/minimax-h3-turing): RTX 2080Ti 22G (Turing sm_75), ComfyUI 0.33.1 → master 345c919, PyTorch 2.9.1+cu130; W4A8 mixed + fl2v Turbo 4-step; T8 BlockCache −43% (drafts, non-reproducible), PDD 8-step + T8 210 s/clip
* [`ComfyUI_MiniMaxH3_Director`](https://github.com/huangserva/ComfyUI_MiniMaxH3_Director): RTX 4090 48 GB, ComfyUI 0.30.0, PyTorch 2.11.0, CUDA 12.8, Ref2VA INT8.
* [`ComfyUI-VDN-H3`](https://github.com/Saganaki22/ComfyUI-VDN-H3): RTX 5090, 1280×736, 145 frames, 17 s/it.
* [`ComfyUI-Ref2VA-VSA`](https://github.com/Kablex/ComfyUI-Ref2VA-VSA): RTX 4090, ~72 s per 5 s clip, ~13.5 GiB.
* [Sol-H3](#solh3): 8×B300, 1344×768 @ 24 fps with stereo audio, warm medians excluding load / warmup / encode.
* [`OpenVDN/vdn-minimax-h3`](https://github.com/OpenVDN/vdn-minimax-h3) ([VDN](#vdn)): H200 and B200, 1 GPU or 8 GPUs (Ulysses), Python 3.12, PyTorch 2.13.0+cu129, `flash-attn-4` 4.0.0b26, Triton 3.7.1, patched Diffusers, FP8 e4m3 linears; 768p × 14.4 s at 8 NFE: 90.5 s (1×H200), 51 s (1×B200), 11.23 s (8×B200), denoising only, VAE decode and MP4 muxing excluded.
* [`matsuo-koya/minimax-h3-notes`](https://github.com/matsuo-koya/minimax-h3-notes) (★24, MIT) collects further single-machine run notes.
* [`-h3-8gb-traps`](https://github.com/FlowForgeLabAi/-h3-8gb-traps) (MIT): RTX 5060 Laptop 8 GB (sm_120) + 15.26 GiB RAM, Windows 11 25H2, ComfyUI 0.35.0 (`a7b1d39`), PyTorch 2.14.0+cu130, Python 3.13.12; pruned INT8 ConvRot DiT + `nvfp4_awq` TE, launched with `--disable-pinned-memory`; 10 s / 768×1024 / 8 steps in 857.7 s; SDPA 143.1 ms vs comfy-kitchen INT8 22.8 ms vs SageAttention 29.0 ms at 56 heads × 128 / S=8192, bf16
* [Draw Things](https://releases.drawthings.ai/p/harvest-season-new-hardware-new-inference) (first-party benchmark): Draw Things 26.1013.0 vs mlx-serve 26.10.2 and vpipe 0.1.80, each in its best configuration found; 1024×512 × 124 frames, 20 steps; sparse attention and step caching off for every engine. Draw Things uses 8-bit S weights (4-bit S on M6), with the Neural Engine + GPU on M3 Ultra, M4 Pro and M6, and GPU neural accelerators only on M5 Ultra. End-to-end: 201.87 s on M5 Ultra 80c / 256 GiB (fastest other: vpipe 246.79 s), 1239.95 s on M3 Ultra 60c / 96 GiB (vpipe 1377.44 s), 1589.02 s on M6 Mac mini 32 GiB (vpipe 1528.22 s — faster end-to-end, though Draw Things is faster per step, 66.32 s vs 67.73 s), 2761.3 s on M4 Pro 20c / 24 GiB (vpipe 3151.21 s). Draw Things model-loading time is included on low-RAM Macs.

### Licenses

| License | Where |
| :--- | :--- |
| Apache-2.0 | `ModelTC/Minimax-H3-Turbo` and the Turbo LoRA line · Ref Patch · `aigc-apps/VideoX-Fun` · `Saganaki22/ComfyUI-VDN-H3` · `Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc` · `wyzborrero/ComfyUI-H3-FunControl` · `lihaoyun6/ComfyUI-H3VAE_TRT` · `NVlabs/Sana` sol-engine |
| MIT | `antirez/h3.c` · `matsuo-koya/minimax-h3-notes` |
| GPL-3.0 | `xmarre/ComfyUI-Sol-H3` · `FX-FeiHou/ComfyUI-FeiHou-Easy-H3` · `drawthingsai/draw-things-community` — copyleft |
| MiniMax-H3 Community | `alibaba-pai/MiniMax-H3-Acc-LoRAs` · `alibaba-pai/MiniMax-H3-Fun-Controlnet-Union` · RAVEN Streaming · Krea2 Style |
| Apache-2.0 (code) · MiniMax-H3 Community (weights) | `OpenVDN/vdn-minimax-h3` |
| Closed / hosted only | The [H3 Max family](../README.md#hosted) — no weights are distributed |
| No code released | [VC-Attention](#vcattention) — paper only; *Nunchux Attention* is proprietary and hosted |
| No license stated | `DeepBeepMeep/MiniMax-H3` |

For other projects, check the repository or model card.

<!-- MARKDOWN LINKS & IMAGES -->
[gh-MiniMaxAI]: https://img.shields.io/badge/%F0%9F%A4%97-MiniMaxAI-FFD21E?style=flat-square
[gh-Comfy--Org]: https://img.shields.io/badge/%F0%9F%A4%97-Comfy--Org-FFD21E?style=flat-square
[gh-Abiray]: https://img.shields.io/badge/%F0%9F%A4%97-Abiray-FFD21E?style=flat-square
[gh-DmitryDB]: https://img.shields.io/badge/%F0%9F%A4%97-DmitryDB-FFD21E?style=flat-square
[gh-DiffSynth-Studio]: https://img.shields.io/badge/%F0%9F%A4%97-DiffSynth--Studio-FFD21E?style=flat-square
[gh-DeepBeepMeep]: https://img.shields.io/badge/%F0%9F%A4%97-DeepBeepMeep-FFD21E?style=flat-square
[gh-WaveCut]: https://img.shields.io/badge/%F0%9F%A4%97-WaveCut-FFD21E?style=flat-square
[gh-rockerBOO]: https://img.shields.io/badge/%F0%9F%A4%97-rockerBOO-FFD21E?style=flat-square
[gh-Kijai]: https://img.shields.io/badge/%F0%9F%A4%97-Kijai-FFD21E?style=flat-square
[gh-AX1Y2JP]: https://img.shields.io/badge/%F0%9F%A4%97-AX1Y2JP-FFD21E?style=flat-square
[gh-tsolful]: https://img.shields.io/badge/%F0%9F%A4%97-tsolful-FFD21E?style=flat-square
[gh-realrebelai]: https://img.shields.io/badge/%F0%9F%A4%97-realrebelai-FFD21E?style=flat-square
[gh-rzgar]: https://img.shields.io/badge/%F0%9F%A4%97-rzgar-FFD21E?style=flat-square
[gh-larryvrh]: https://img.shields.io/badge/%F0%9F%A4%97-larryvrh-FFD21E?style=flat-square
[gh-drbaph]: https://img.shields.io/badge/%F0%9F%A4%97-drbaph-FFD21E?style=flat-square
[gh-vantagewithai]: https://img.shields.io/badge/%F0%9F%A4%97-vantagewithai-FFD21E?style=flat-square
[gh-Mamad8]: https://img.shields.io/badge/%F0%9F%A4%97-Mamad8-FFD21E?style=flat-square
[gh-NicoLab28]: https://img.shields.io/badge/%F0%9F%A4%97-NicoLab28-FFD21E?style=flat-square
[gh-lightx2v]: https://img.shields.io/badge/%F0%9F%A4%97-lightx2v-FFD21E?style=flat-square
[gh-lihaoyun6]: https://img.shields.io/badge/%F0%9F%A4%97-lihaoyun6-FFD21E?style=flat-square
[gh-tutututututu]: https://img.shields.io/badge/%F0%9F%A4%97-tutututututu-FFD21E?style=flat-square
[gh-t8star]: https://img.shields.io/badge/%F0%9F%A4%97-t8star-FFD21E?style=flat-square
[gh-abakanai]: https://img.shields.io/badge/%F0%9F%A4%97-abakanai-FFD21E?style=flat-square
[gh-Winnougan]: https://img.shields.io/badge/%F0%9F%A4%97-Winnougan-FFD21E?style=flat-square
[gh-unsloth]: https://img.shields.io/badge/%F0%9F%A4%97-unsloth-FFD21E?style=flat-square
[gh-MarxistLeninist]: https://img.shields.io/badge/%F0%9F%A4%97-MarxistLeninist-FFD21E?style=flat-square
[gh-joyfox]: https://img.shields.io/badge/%F0%9F%A4%97-joyfox-FFD21E?style=flat-square
[gh-smhfacct]: https://img.shields.io/badge/%F0%9F%A4%97-smhfacct-FFD21E?style=flat-square
[gh-infosave]: https://img.shields.io/badge/%F0%9F%A4%97-infosave-FFD21E?style=flat-square
[gh-InstantX]: https://img.shields.io/badge/%F0%9F%A4%97-InstantX-FFD21E?style=flat-square
[gh-Merserk]: https://img.shields.io/badge/%F0%9F%A4%97-Merserk-FFD21E?style=flat-square
[gh-molbal]: https://img.shields.io/badge/%F0%9F%A4%97-molbal-FFD21E?style=flat-square
[gh-mvp--lab]: https://img.shields.io/badge/%F0%9F%A4%97-mvp--lab-FFD21E?style=flat-square
[gh-TenStrip]: https://img.shields.io/badge/%F0%9F%A4%97-TenStrip-FFD21E?style=flat-square
[gh-leejet]: https://img.shields.io/badge/%F0%9F%A4%97-leejet-FFD21E?style=flat-square
[gh-fal]: https://img.shields.io/badge/%F0%9F%A4%97-fal-FFD21E?style=flat-square
[gh-matlod]: https://img.shields.io/badge/%F0%9F%A4%97-matlod-FFD21E?style=flat-square
[gh-Inner--Reflections]: https://img.shields.io/badge/%F0%9F%A4%97-Inner--Reflections-FFD21E?style=flat-square
[gh-bghira]: https://img.shields.io/badge/%F0%9F%A4%97-bghira-FFD21E?style=flat-square
[gh-FastVideo]: https://img.shields.io/badge/%F0%9F%A4%97-FastVideo-FFD21E?style=flat-square
[gh-aptech0081]: https://img.shields.io/badge/%F0%9F%A4%97-aptech0081-FFD21E?style=flat-square
[gh-1ronman1993]: https://img.shields.io/badge/%F0%9F%A4%97-1ronman1993-FFD21E?style=flat-square
[gh-OpenVDN]: https://img.shields.io/badge/%F0%9F%A4%97-OpenVDN-FFD21E?style=flat-square

[badge-bf16]: https://img.shields.io/badge/bf16-0077cc?style=flat-square
[badge-fp16]: https://img.shields.io/badge/fp16-0077cc?style=flat-square
[badge-fp32]: https://img.shields.io/badge/fp32-6c757d?style=flat-square
[badge-fp8]: https://img.shields.io/badge/fp8-28a745?style=flat-square
[badge-mxfp8]: https://img.shields.io/badge/mxfp8-20c997?style=flat-square
[badge-int8]: https://img.shields.io/badge/int8-17a2b8?style=flat-square
[badge-int4]: https://img.shields.io/badge/int4-ffc107?style=flat-square
[badge-nvfp4]: https://img.shields.io/badge/nvfp4-6f42c1?style=flat-square
[badge-w4a8]: https://img.shields.io/badge/W4A8-ffc107?style=flat-square
[badge-w4a4]: https://img.shields.io/badge/W4A4-e05d44?style=flat-square
[badge-nf4]: https://img.shields.io/badge/NF4-ffc107?style=flat-square
[badge-Q2_K]: https://img.shields.io/badge/Q2__K-e05d44?style=flat-square
[badge-Q3_K_M]: https://img.shields.io/badge/Q3__K__M-fe7d37?style=flat-square
[badge-Q3_K_S]: https://img.shields.io/badge/Q3__K__S-fe7d37?style=flat-square
[badge-Q4_0]: https://img.shields.io/badge/Q4__0-dfb317?style=flat-square
[badge-Q4_1]: https://img.shields.io/badge/Q4__1-dfb317?style=flat-square
[badge-Q4_K_M]: https://img.shields.io/badge/Q4__K__M-dfb317?style=flat-square
[badge-Q4_K_S]: https://img.shields.io/badge/Q4__K__S-dfb317?style=flat-square
[badge-Q5_0]: https://img.shields.io/badge/Q5__0-97c00f?style=flat-square
[badge-Q5_1]: https://img.shields.io/badge/Q5__1-97c00f?style=flat-square
[badge-Q5_K_M]: https://img.shields.io/badge/Q5__K__M-97c00f?style=flat-square
[badge-Q5_K_S]: https://img.shields.io/badge/Q5__K__S-97c00f?style=flat-square
[badge-Q6_K]: https://img.shields.io/badge/Q6__K-0077cc?style=flat-square
[badge-Q8_0]: https://img.shields.io/badge/Q8__0-28a745?style=flat-square
[badge-UD-Q2_K_XL]: https://img.shields.io/badge/UD-Q2__K__XL-e05d44?style=flat-square
[badge-UD-Q3_K_XL]: https://img.shields.io/badge/UD-Q3__K__XL-fe7d37?style=flat-square
[badge-IQ1_S]: https://img.shields.io/badge/IQ1__S-b02a37?style=flat-square
[badge-IQ1_M]: https://img.shields.io/badge/IQ1__M-d64545?style=flat-square
[badge-noinfo]: https://img.shields.io/badge/no%20description-6c757d?style=flat-square&logoColor=white

[cat-cond]: https://img.shields.io/badge/Conditioning-0077cc?style=flat-square
[cat-prompt]: https://img.shields.io/badge/Prompt-28a745?style=flat-square
[cat-upscale]: https://img.shields.io/badge/Upscaling-fe7d37?style=flat-square
[cat-accel]: https://img.shields.io/badge/Acceleration-6f42c1?style=flat-square
[cat-port]: https://img.shields.io/badge/Port-17a2b8?style=flat-square
[cat-face]: https://img.shields.io/badge/Face%20Refine-e83e8c?style=flat-square


---

# ISSUES / PRs of MiniMax-AI/awesome-minimax-h3-integration


### #35 [open] PR: Document TensorSharp support for H3-Image editing
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/35  by zhongkaifu 2026-11-27T07:13:46Z

The TensorSharp entry in both READMEs still says editing is not implemented. This updates the existing rows to reflect the merged [H3-Image integration](https://github.com/zhongkaifu/TensorSharp/pull/390) and links to the [editing guide](https://github.com/zhongkaifu/TensorSharp/blob/main/docs/models/minimax-h3-image.md#editing).

TensorSharp supports reference-image editing on ggml CUDA and Metal. Masked editing is implemented by TensorSharp, not a native feature of the upstream MiniMax pipeline.

Validation: `python3 scripts/check_readme.py` returned `OK`.

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/run-proof?expires=1798355629&installation_model_id=427615&pr_number=35&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F35&variant=proof_this_pr&signature=b476cac8c40f34f98beaba03f6f4b6512e2cabf17ba47837bce9add347108ff1"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-proof-this-pr-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-proof-this-pr-v2-light.svg"><img alt="Proof this PR" src="https://pr-comments-assets.blacksmith.sh/codesmith/proof-proof-this-pr-v2-dark-3x.png" width="121" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/35?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #34 [open] PR: Add a Postman collection for MiniMax H3 text-to-video
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/34  by xiaodilj-dev 2026-11-25T14:39:45Z

This adds a two-request Postman collection to the MiniMax H3 video API integrations table in both languages. The [setup guide](https://qianliu-api-tools.deed.page/#h3-postman) links to the [importable JSON](https://qianliu-api-tools.deed.page/qianliu-h3-postman.json). It shows how to submit a five-second text prompt and look up the same job by its task ID. Sending the create request requires a funded Qianliu account and incurs the provider's usage charge.

I maintain the collection for Qianliu. I validated the published file against the Postman v2.1 schema and imported it with the Postman Collection SDK to check the bearer configuration and task-ID substitution. I haven't run a funded generation end to end. The repository's bilingual README check passes.

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/run-proof?expires=1798209590&installation_model_id=427615&pr_number=34&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F34&variant=start_proof&signature=1a41964a925fbceed6cdb0b042b57f440aa995364d715b85e3afac60790d05dd"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-start-proof-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-start-proof-v2-light.svg"><img alt="Start Proof" src="https://pr-comments-assets.blacksmith.sh/codesmith/proof-start-proof-v2-dark-3x.png" width="108" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/34?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #33 [open] PR: Add Kaeso to API integrations
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/33  by dvnold 2026-11-22T16:21:51Z

Adds [Kaeso](https://github.com/logicplanes/kaeso) to **API integrations**, after LiteLLM, in both `README.md` and `README_cn.md`.

Kaeso is a self-hosted AI gateway written in Rust, under the MIT license. Apps create MiniMax-H3 video jobs through its OpenAI-style Videos API. Kaeso sends `MiniMax-H3` and its variants to MiniMax's v2 video API with the organization's own MiniMax key, under per-project budgets, and logs each request.

**Sources:**
- [Videos API, provider table](https://docs.kaeso.app/api/videos), which lists `MiniMax-H3` and its variants.
- [How video jobs map to MiniMax's Hailuo and H3 APIs](https://docs.kaeso.app/development/kpp/minimax-videos). `MiniMax-H3` goes to `/v2`, which needs `seconds` and `size`.

H3-Image isn't claimed: Kaeso's MiniMax image support is `image-01` only.

`python3 scripts/check_readme.py` prints OK. Disclosure: I maintain Kaeso.

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/run-proof?expires=1797956515&installation_model_id=427615&pr_number=33&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F33&variant=run_proof_on_this_pr&signature=3d00d739399ccd43ca5ed6bf95dc9cfe7203867dda86469bac37c8089304f58f"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-run-proof-on-this-pr-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-run-proof-on-this-pr-v2-light.svg"><img alt="Run Proof on this PR" src="https://pr-comments-assets.blacksmith.sh/codesmith/proof-run-proof-on-this-pr-v2-dark-3x.png" width="168" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/33?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #32 [open] PR: Add H3Img to MiniMax-H3-Image apps
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/32  by yi8882025 2026-11-22T09:41:16Z

This adds H3Img to the MiniMax-H3-Image Apps table in the English and Chinese READMEs. H3Img is an independently operated third-party browser tool that uses MiniMax-H3-Image through APIMart for text-to-image generation and reference editing. It offers paid monthly subscriptions and one-time credit packs. [Pricing](https://h3img.com/pricing/)

The serum bottle example shows the source packshot, original edit instruction, edited image and provider used. It also documents where the generated surface and bottle placement differed from the request. [Example and test details](https://h3img.com/examples/serum-scene/)

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/run-proof?expires=1797932479&installation_model_id=427615&pr_number=32&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F32&variant=run_proof&signature=66b4b8123a71335db4644e0ef9a5181470cddc8490563f8cd2f6d51ba72c32f4"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-run-proof-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/proof-run-proof-v2-light.svg"><img alt="Run Proof" src="https://pr-comments-assets.blacksmith.sh/codesmith/proof-run-proof-v2-dark-3x.png" width="102" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/32?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #31 [closed] PR: Update the integrations index
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/31  by SaladDay 2026-11-12T15:41:03Z

- `README.md` is now a table index of models, tools and projects for MiniMax H3 and MiniMax-H3-Image. Projects that run the open weights and projects that use the models through an API are both listed. The old explicit anchors are kept.
- `README_cn.md` is the Chinese version. Both files list the same links.
- The per-GPU guide from the old README (model files, VRAM recipes, compatibility) moves to `docs/run-h3-locally.md`. Download links to files that were deleted or renamed upstream are fixed.
- `scripts/check_readme.py` checks that both READMEs list the same links and that every in-page link resolves.

Every entry was checked against the project's own README, docs or model card on 2026-11-12, and every link returns 200.

<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1797090067&installation_model_id=427615&pr_number=31&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F31&signature=baed4341ca65a17c12fe9b5ae388de7eb67734d0721f427776d9e0cf3c939a27"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/31?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #30 [open] ISSUE: Suggestion: distinguish the two H3 still-image branches in the navigation guide
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/issues/30  by diaozxin007 2026-11-12T12:11:39Z

The index already includes the community single-image VAE, but I could not find entries for the two image-generation/editing branches. A short “Still images” row in “Start here” could help readers avoid mixing their checkpoints and workflows.

Primary references:
- [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image): the official image checkpoint follows the H3-Base design without the audio branch and retains the H3 text encoder; its card documents six sampling steps and approximately 16 MP output.
- [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image): an H3-based image generation/editing pipeline; its card documents eight steps and 8 MP output, with a separate Comfy package linked from the card.

These are separate H3-derived image paths, not interchangeable downloads. Their model cards also link different license files, so a shared “H3 license” label would not be sufficient.

Suggested navigation wording: **Generate or edit still images → MiniMax-H3-Image (official image branch) / P-H3-Image (Pruna image branch); follow the workflow and license for the exact release.** The model-card links above are enough for this row.

Disclosure: I maintain Studio Verdict. I also made a [documentation-based comparison and interactive family diagram](https://studioverdict.app/guides/h3-image-vs-p-h3-image/) separating model derivation, prompt components and deployment packaging. If community explanatory guides fit this index, it is an optional supplementary resource; it is not a generation benchmark and makes no measured quality/speed claim. No need to include it if the index prefers primary documentation only.


### #29 [closed] PR: Add TaleMondo Studio to H3 runtimes
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/29  by Klimentios 2026-11-09T09:07:41Z

## Summary
Add TaleMondo Studio to the H3 runtimes table. It is an independent, noncommercial Mac film-production workspace built around local MiniMax H3 through ComfyUI.

The public repository documents role-tagged references, franchise and scene boards, shot direction, a GPU queue, take review, editing, and MCP tools for coding agents. It also includes interface screenshots and an edited tour.

The row states the documented 96 GB VRAM / 256 GB RAM H3 setup, the PolyForm Noncommercial license, and that other imported ComfyUI model paths are unverified. The project credits MiniMax AI, Comfy-Org, and Kijai in its README and NOTICE.

Source: https://github.com/Klimentios/talemondo-studio
Walkthrough and video examples: https://github.com/Klimentios/Klimentios/blob/main/writing/talemondo-studio.md

Disclosure: I maintain TaleMondo Studio.

<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1796807265&installation_model_id=427615&pr_number=29&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F29&signature=075afcbcc16db3c4c28eb697214a918e83b212b641e931b50a05f41fe26b05cc"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/29?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #28 [closed] PR: Index H3 Flex, refinement workflows and local editing tools
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/28  by Odysseusailoon 2026-11-02T18:33:41Z

The index was missing released H3 adapters, refinement workflows and local editing applications, while its PTD and DMAD descriptions no longer matched the available checkpoints. This update adds nine previously unindexed projects/integrations and refreshes those two entries.

- Add Pika H3 Flex's six-step FL2VA/Ref2VA adapters, separating native local output from hosted API extensions.
- Add NVIDIA's Open-H3-Regenerate-2K, its community ComfyUI integration, Tessera and ComfyUI-Dials, with upstream component credits, hardware boundaries and separate code/model terms.
- Add Openjutsu, H3 Motion, MiniMax-H3 Studio and Tiiny Studio, including their required ComfyUI or device environments. Some projects predate this discovery window; these are newly indexed entries, not nine new releases.
- Update PTD for the re-noised SDE adapters and their full-model ComfyUI validation gap; add DMAD's eight-step student. Preserve existing anchors, including the former DMAD heading link.

Validation: primary repository documentation, source artifacts, releases and model cards reviewed; independent content review completed; all 14 newly linked URLs return HTTP 200; Markdown table structure and anchor preservation checks pass; `git diff --check` passes. Documentation-only change; no GPU benchmarks or model inference rerun.

<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1796236429&installation_model_id=427615&pr_number=28&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F28&signature=c8904a14f406cbd2c4bb922d454ea9d817b279c73f1461695e4808af3a11c23c"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/28?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #27 [closed] PR: Add Draw Things to Runtimes
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/27  by liuliu 2026-10-26T16:12:33Z

List the Draw Things engine (drawthingsai/draw-things-community) as an Apple silicon runtime, add its first-party benchmark to Reported environments, and add it to the GPL-3.0 license row.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/27?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v3.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3-3x.png" width="162" height="28"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1795623156&installation_model_id=427615&pr_number=27&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F27&signature=4535f05d7e34ee02d1c79d3a7001ae685ab302ab55a84b2c8a7d10b61d73f3cd"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #26 [closed] PR: Index Dedicated deployment and add FastH3 credits (Nuva Lab)
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/26  by jd-nuva 2026-10-26T01:08:32Z

## Summary

PR #23 (merged) added the "Dedicated deployment" subsection under "Serving H3" with a Nuva Lab entry, and PR #25 added FastH3 Omni to the index after the FastH3-Omni open-weights release on October 20, 2026. The "Start here" table and the "Serving H3" intro did not point to "Dedicated deployment", and the Nuva Lab entry still used the preview-era wording: "One of the two FastH3-Omni preview launch partners, with Reactor." This PR updates `README.md` only (6 lines added, 3 removed) to add the missing links and replace the preview-era wording with release-era credits.

## Changes

1. Add a "Run it on a dedicated host" row to the "Start here" table that links to `#dedicated`.
2. Add one sentence to the "Serving H3" intro that points to Hosted H3 Max (`#hosted`) for a hosted API and to Dedicated deployment (`#dedicated`) for a managed private host.
3. Add one bullet to the "FastH3 Omni" subsection. It states that FastVideo built FastH3 Omni in collaboration with Nuva Lab, that Nuva Lab co-developed the model and defined and curated the omni reference dataset per the model card, and that `#dedicated` covers a managed host.
4. Update the Nuva Lab entry in "Dedicated deployment". Remove the preview-era sentence and keep all other text in the entry unchanged. Add that Nuva Lab runs FastH3-Omni on dedicated hosts with fine-tuning on customer data, add a link to `nuvalab.ai/fasth3`, and list its FastH3 credits: FastH3-Omni co-development and reference dataset, production grounding credit on the 4-step preview and 8-Step V2 model cards, and UniServe serving production traffic at Nuva Lab and running its benchmarks on Nuva Lab hardware.
5. Expand the FastVideo line in "Acknowledgements" from the 4-step preview to the whole FastH3 line (4-step preview, V2, Trim, and Omni) and UniServe. Add one line that credits Nuva Lab for production grounding behind FastH3, the FastH3-Omni co-development and reference dataset, and the production hardware behind the UniServe benchmarks. FastVideo stays the owner and maintainer of FastH3, and no anchor names or other sections change.

## Sources

- [FastH3-Omni-8-Step-V1 model card](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) (public, checked 2026-10-25): "FastH3 Omni was built in close collaboration with Nuva Lab, which co-developed the model with FastVideo, defined and curated the omni reference dataset..." and "For dedicated production deployment, fine-tuning on your own data and a creative agent on top, see Nuva Lab's FastH3-Omni page."
- [FastH3 4-step Preview model card](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) and [8-Step V2 model card](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) (public, checked 2026-10-25): "We thank Nuva Lab for bringing production grounding to FastH3..."
- [UniServe blog](https://haoailab.com/blogs/uniserve-fasth3/) (public, checked 2026-10-25): "UniServe is serving production API traffic on Reactor and Nuva Lab" and "Nuva Lab for access to the production hardware used in our benchmarks."
- [FastH3 Omni release post](https://haoailab.com/blogs/fastvideo-omniref-update/) (public, checked 2026-10-25): lists Nuva Lab among partners.

## Checks

- Ran checks locally (this repo has no CI).
- All 9 URLs in the added lines return HTTP 200.
- The anchors `#dedicated`, `#hosted`, `#partners`, `#uniserve`, and `#fasth3-omni` each exist once.
- Rendered the README with the GitHub GFM markdown API: the "Start here" table has 11 rows (was 10), and the three new `#dedicated` links render.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/26?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v3.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3-3x.png" width="162" height="28"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1795568915&installation_model_id=427615&pr_number=26&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F26&signature=4adff647c074122a8e9373511eb650c4f6cf16fc45a1dcb1f4113e7c9fe67563"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #25 [closed] PR: Add FastH3 Omni, serving integrations and community workflows
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/25  by Odysseusailoon 2026-10-25T23:16:29Z

The index is missing the FastH3-Omni open-weight release and several usable H3 runtimes and workflows. Add Omni, UniServe, PTD, TorchTPU/Diffusers, MLX-Serve, ResCascade, VPack, Spark-H3 and experimental TARA-H3, plus the previously omitted FreeVideo and DMAD projects.

Keep the earlier FastH3 V2/Trim T2VA limitations distinct from Omni's reference-conditioned generation. Entries retain hardware requirements, model/code license distinctions, PTD's validated ComfyUI scope, Spark-H3's current compatibility notice and TARA's experimental status. Update the acceleration overview and preserve existing navigation anchors.

Validation: all 21 newly linked external URLs returned HTTP 200; existing anchors and new internal links were checked; Markdown table structure and `git diff --check` passed. Descriptions were checked against project documentation and model cards. No inference benchmarks were rerun.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/25?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v3.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3-3x.png" width="162" height="28"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1795562192&installation_model_id=427615&pr_number=25&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F25&signature=68d9035dc42821a19152e0e91e002a0a23503e3422741023d0d8af26d74764d9"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #24 [closed] PR: Refresh H3 ecosystem: FastH3 V2 / Trim and community runtimes
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/24  by Odysseusailoon 2026-10-17T00:43:19Z

The index still describes FastH3 only as a four-step preview. Add the released eight-step V2 and experimental Trim checkpoints, including T2VA-only scope, runtime requirements, and the distinction between a memory-capped RTX 4090 measurement and a physical 8 GB GPU.

Also index H3-Flash, Sol-H3 Studio, and FeiHou Easy H3 with hardware, license, and compatibility notes. Clarify contributor attribution on the laptop measurements and fix the Raylight description.

Validation: reviewed upstream READMEs/model cards and repository licenses; all seven newly introduced URLs return HTTP 200; existing FastH3 anchor retained; `git diff --check` passes. Documentation-only update; GPU performance has not been independently reproduced.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/24?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v3.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1794789802&installation_model_id=427615&pr_number=24&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F24&signature=c8c157ced9eb2d7c29009ba29409acffe824fb2e83cb15b8ecb08687321412d7"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #23 [closed] PR: Add Dedicated deployment section (Nuva Lab)
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/23  by jd-nuva 2026-10-15T22:27:48Z

## Summary

This PR adds a short **Dedicated deployment** subsection under "Serving H3", after the hosted H3 Max family. It lists [Nuva Lab](https://nuvalab.ai/minimax-h3-dedicated-deployment/).

The index has open-source serving stacks (SGLang, vLLM-Omni) and the hosted H3 Max API. It has no entry for a private host that runs the open H3 weights for one company. This subsection fills that gap.

## What the entry says

- Dedicated MiniMax H3 and FastH3 hosts. The standard host has 8× RTX PRO 6000. B200, B300 and GB200 hosts are available on request.
- The host runs customer fine-tuned weights.
- Nuva Lab holds a MiniMax authorization for H3 deployments that serve US companies.
- Nuva Lab is one of the two FastH3-Omni preview launch partners, with Reactor.
- Links to two Nuva Lab pages with sourced cost and VRAM data.

The change is in `README.md` only. It adds one anchor (`#dedicated`) and changes no other section.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/23?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v3.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1794695272&installation_model_id=427615&pr_number=23&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F23&signature=3fb589123f4832e4559fca8dcc527956450a95370fc64427e88b604f24edbb47"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #22 [closed] PR: Add okaypic-video-skill to Agent skills
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/22  by shipunyc 2026-10-09T10:26:45Z

Adds [okaypic-video-skill](https://github.com/okaypic/okaypic-video-skill) to the **Agent skills** table, second row (it carries a brief through to a finished file, like `awesome-minimax-h3`), and updates the intro sentence from "one carries" to "two carry".

What it is: a `SKILL.md` skill (Claude Code, Codex) that takes a prompt to a finished, captioned multi-shot H3 episode — shot list, reference sheets, two takes per shot with contact sheets, a local timeline editor, ffmpeg cut with EN/ZH captions. Runs against a hosted API (okaypic), not local weights. MIT.

Example made with it: GOBLIN CITY ep. 1 (28 shots, 3:52) — https://okaypic.b-cdn.net/marketing/goblin-city/goblin-city-ep01-en.mp4

Disclosure: I'm the author; okaypic is my service and the skill needs an okaypic API key (new accounts get a small free credit).

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/22?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v3.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v3.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1794133607&installation_model_id=427615&pr_number=22&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F22&signature=c8c586ecb6d099d05c933ea5a984b47f15e1824b5fdb29dadee211bed899bdc7"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #21 [closed] PR: Add Raylight MiniMax H3 runtime integration
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/21  by komikndr 2026-10-06T15:48:19Z

## Summary

Adds [Raylight](https://github.com/komikndr/raylight) to the MiniMax H3 runtimes section.
Preview [Showcases](https://komikndr.github.io/raylight-showcase/)

Raylight provides distributed multi-GPU execution for ComfyUI diffusion models, including MiniMax H3.

## H3 support

- Unified Sequence Parallelism (USP) via xDiT / xFuser
- Optional FSDP model-weight sharding
- CFG parallelism
- Custom attention support including MiniMax SLA
- Ray-managed multi-GPU execution
- Reduced per-GPU VRAM requirements for large H3 workloads

## Repository

https://github.com/komikndr/raylight

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/21?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1793893702&installation_model_id=427615&pr_number=21&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F21&signature=7541700bba75b131c062ea935d9f28438c5d50070c1a6916f9530c351b6aab8e"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #20 [closed] PR: 929
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/20  by Nuos 2026-09-29T06:56:33Z



<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/20?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1793256997&installation_model_id=427615&pr_number=20&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F20&signature=b26a818a4842efae5af833e5dc463453a691018fe69540466303c957892bdd52"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #19 [closed] PR: Add a measured 8 GB laptop path (sm_120 + 16 GB RAM) to VRAM and environments
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/19  by FlowForgeLabAi 2026-09-27T02:37:21Z

The `8 GB` row currently points at one path (DiffSynth NF4). This adds a second
one that I have measured end to end, in the same shape as the Turing entry.

**What it adds**

- **By VRAM and hardware**: one row for an RTX 5060 Laptop 8 GB (sm_120) with
  16 GB of system RAM — pruned INT8 ConvRot DiT + `nvfp4_awq` text encoder,
  `--disable-pinned-memory`, 10 s / 768×1024 / 8 steps in 14.3 min.
- **Reported environments**: the matching environment line.

**The finding that changes how the 8 GB case is configured.** The binding limit
on this machine is the **16 GB of system RAM, not the 8 GB of VRAM**. The working
set is ~34.5 GiB, so it pages; the symptom is a stalled sampler while
`utilization.gpu` still reads 99%, and `nvidia-smi` power draw is the tell —
flat 34–36 W while paging against 64–98 W healthy. Same machine: 0.9–6 s/block
fresh, 212 s/block after ~1.7 h, which is why restarting ComfyUI before each run
matters more than any sampler setting here.

Also recorded because it is architecture-specific: at H3's real geometry
(56 heads × 128, S=8192, bf16), comfy-kitchen INT8 attention measures 22.8 ms
against 143.1 ms for SDPA and 29.0 ms for SageAttention. ComfyUI core sets
`MAX_PINNED_MEMORY = ram * 0.40`, which kills the process on a 16 GB machine, so
`--disable-pinned-memory` is required rather than optional.

**Scope.** I have deliberately stated only what I measured, and the linked repo
labels the untested items (clips >10 s, `ref_image_size=max`, the RTX 4060's
8/20-step timings) as untested. Both linked repos are MIT.

Happy to trim the row, drop the workflow link, or move it elsewhere in the table
— whatever fits the format best.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/19?autoLogin=true&ref=codesmith_pr_footer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1793068646&installation_model_id=427615&pr_number=19&ref=codesmith_pr_footer&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F19&signature=5e00352d229cf5a30c9ea93577cb29aab0a6fd36154ce4edd617a40bd8b11795"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #18 [closed] PR: Index VC-Attention as a paper-only attention method
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/18  by Odysseusailoon 2026-09-17T08:15:01Z

Adds VC-Attention to the Speed section, in its own subsection rather than the four-family table — it quantizes attention instead of cutting steps, so it is orthogonal to Turbo / PDD / FastH3 / VDN rather than an alternative to them.

**Flagged as unreleased.** There is no repository, no kernel download and no license — only an arXiv report and a company blog post. The rest of this index implies you can go download the thing it lists, so that needed saying up front.

**Two names, separated.** The announcement uses them interchangeably:

| | B200 | B300 |
| :--- | ---: | ---: |
| VC-Attention (the paper) | 1.59× | 1.51× |
| Nunchux Attention (proprietary, hosted) | 1.91× | 1.83× |

Kernel-level, against BF16 FlashAttention-4. End-to-end is much smaller and is recorded alongside it: 1.13–1.19× datacentre, 1.36–1.70× workstation. Fidelity 20.2 dB mean PSNR vs SageAttention2's 19.9 dB across 100 prompts.

Licenses table gains a "No code released" row.

Sources: [arXiv 2609.15810](https://arxiv.org/abs/2609.15810) (14 Sep 2026) · [blog](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck). All links verified live.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/18"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1792224905&installation_model_id=427615&pr_number=18&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F18&signature=f8efc299c9cf7ecc5397084979499e583c5e537397f64ef64a281b64a0f85f28"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #17 [closed] PR: Add ComfyUI-MiniMaxDirector to Conditioning & orchestration
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/17  by imbutus 2026-09-16T22:28:11Z

Adds `imbutus/ComfyUI-MiniMaxDirector` (49★, MIT) to the Conditioning & orchestration table, in star order.

A timeline editor for H3 in ComfyUI: shots, camera moves and audio cues laid out on a track and compiled into the structured prompt the model reads, covering the camera vocabulary, voiceovers, cross-cut dialogue and on-screen text from the FL2VA / Ref2VA prompt guides, with a card per named subject for face, voice and retention, and a linter for the guides' stated rules. Listed in the Comfy Registry; full walkthrough on YouTube (EN, RU, ZH).

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/17"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1792189695&installation_model_id=427615&pr_number=17&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F17&signature=1fa827dfdac0c3d31ff1085f1f8dd8536bce6d253a6ec2c722fb62d84db39383"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #16 [closed] PR: Add verl-omni to Training & LoRAs
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/16  by SamitHuang 2026-09-13T10:23:31Z

## Summary

- Add [`verl-project/verl-omni`](https://github.com/verl-project/verl-omni) to the **Training & LoRAs** table.
- verl-omni is an RL post-training framework with MiniMax-H3 LoRA recipes for **T2VA, FL2VA, and Ref2VA** using [FlowGRPO](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/minimax_h3/README.md) and [DiffusionNFT](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/README.md) (Diffusers FSDP2 Actor + vLLM-Omni rollout).

## Why this is not a duplicate

- `gh pr list --repo MiniMax-AI/awesome-minimax-h3-integration --state all --search "verl-omni"` returned no matching PRs.
- Open PRs/issues on the target repo also did not already list this project.

## Test plan

- [x] Confirm the new row sits in **Training & LoRAs** next to `miles_diffusion`.
- [x] Confirm both recipe links resolve on GitHub.
- [x] Star count taken from GitHub at PR time (1018).

AI assistance (Cursor) was used to draft this change. A human submitter reviewed every changed line.


Made with [Cursor](https://cursor.com)

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/16"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1791887014&installation_model_id=427615&pr_number=16&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F16&signature=fd4bfa53507233476fe9e85042fa717ae7c15704dcea8287c0c25bef27a412ea"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #15 [closed] PR: Add PDD, FastH3, VDN, Sol-H3, Fun Control and hosted H3 Max sections
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/15  by Odysseusailoon 2026-09-11T02:10:13Z

Speed now opens with a comparison of the four step-reduction families — Turbo, PDD, FastH3, VDN — since they are alternatives to each other, not a stack, and readers were loading several at once.

New sections:
* PDD (NVIDIA arXiv 2607.26004, applied to H3 by Alibaba PAI) — 8-step Acc LoRAs, the backbone-plus-32-heads loading trap, and two repacks.
* FastH3 — FastVideo's 4-step VSA preview line.
* VDN — OpenVDN's hybrid linear/softmax rebuild, its 82 GiB directory layout, and the competing Ref2VA-VSA measurement.
* Sol-H3 — the standalone datacentre runtime on the Sana sol-engine branch, with the published 8xB300 medians.
* Fun Control — the single 6.8 GiB ControlNet-Union branch.
* Hosted H3 Max — flagged as closed weights, so nothing else on the page applies to it.

Also: quants of the accelerated variants, ComfyUI-H3VAE_TRT, MiniMax-H3-LongVideos, new environment reports, and a Licenses table that now carries GPL-3.0 (ComfyUI-Sol-H3) and a hosted-only row.

Licenses and star counts verified against the GitHub API; all added links checked live.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/15"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1791684616&installation_model_id=427615&pr_number=15&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F15&signature=94d438bd849969e54bb1587669f8bf973d48162633a19c337d92d9655447c671"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #14 [closed] PR: Add VDN-H3 (Video DeltaNet hybrid attention) to Speed
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/14  by haochengxi 2026-09-04T04:32:27Z

Adds **VDN-H3** ([code](https://github.com/OpenVDN/vdn-minimax-h3), [weights](https://huggingface.co/OpenVDN/vdn-minimax-h3)), a hybrid-attention version of MiniMax H3: a sliding-window softmax branch for nearby frames plus a linear-attention branch for long-range context, shipped as a linear branch and two LoRAs on top of the unmodified H3 weights. Generates a 768p 14.4 s clip in 8 steps in 51 s on one B200 and 11.23 s on 8×B200. Weights, inference, and training code are open-sourced.

Changes, all additive:

- **Speed**: new `Architecture` subsection after Acceleration nodes, with rows for VDN-H3 and its two ComfyUI integrations ([Saganaki22/ComfyUI-VDN-H3](https://github.com/Saganaki22/ComfyUI-VDN-H3), [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)), plus one clause in the Speed intro.
- **By VRAM and hardware**: an `H200 / B200, faster than real time` row.
- **Reported environments** and **Licenses**: one line each.
- `gh-OpenVDN` badge definition.

I am one of the VDN-H3 authors. Happy to answer any questions about the model or the numbers.


> **Comment by Odysseusailoon (2026-09-11T04:08:29Z):**

Merged — thank you, and thanks for stating your affiliation up front. That's the right way to bring your own model to an index like this.

Full disclosure on what changed during the merge, since it wasn't a clean apply. A restructure of `## Speed` landed a few hours before this (#15): PDD, FastH3, VDN and Sol-H3 all appeared within a few weeks of each other, so the section became a comparison of acceleration families, each with its own subsection, instead of a flat node list. That collided with your `### Architecture` section, so I resolved it as follows:

- **Your content won on every overlap.** Your three-lever framing replaced the old "two levers" intro — architecture change genuinely is a distinct lever from step count and step cost, and the page was missing that. Your per-GPU figures (2.6–2.9× per step, 90.5 s on 1×H200, 51 s on 1×B200) and your full environment report replaced our thinner versions, which only had the 8×B200 number.
- **`### Architecture` is folded into the existing `### VDN (Video Delta Net)` subsection** under `#vdn`, so VDN sits alongside Turbo / PDD / FastH3 rather than in a parallel one-entry section. Your mechanism description — sliding-window softmax for nearby frames, linear attention for long-range — is now the opening paragraph there; ours was vaguer.
- **Your H200/B200 row** is in the VRAM table, retargeted to `#vdn`.
- **The `comfyui-minimax-h3-audio-T8` row is demoted to a sentence.** It's an audio node that ships VDN workflows, not an architecture change, and its ~1k stars measure something other than VDN adoption. The bundle link is preserved. The Saganaki22 node kept its table row.
- Star counts were refreshed on the way in (OpenVDN 397, Saganaki22 202).

Two things worth flagging so they're not a surprise:

The VDN subsection carries a competing measurement next to yours — `Kablex/ComfyUI-Ref2VA-VSA` reports 2.24× faster than VDN-H3 on an RTX 4090. The page states plainly that both figures are self-reported, on different hardware, and neither is independently reproduced. This index is run by MiniMax; a vendor page quoting an author's own benchmark without the competing claim would read as an endorsement. No comment on whose number is right.

And your licence line was right where ours was wrong. We had `OpenVDN/vdn-minimax-h3` down as plain Apache-2.0; the model card is `minimax-h3-community-license-agreement`, so the code/weights split you wrote is what the table says now.

If the folded version misstates the mechanism anywhere, open a PR against `#vdn` and we'll take the correction.



> **Comment by haochengxi (2026-09-11T07:17:01Z):**

Thanks so much for documenting the changes so transparently. We really appreciate the effort to maintain this repo. :)

For the competing benchmark, we think the current wording is fair enough. Our initial release was mostly tested and tuned for H200 / B200 (data-center-class) GPUs, and we haven't done the same level of optimization for consumer hardware such as the 4090 / 5090.

For example, the offloading strategy / kernel implementation might be far from optimal on consumer level devices, so we do not view the two reported numbers as very contradictory.

We are working on better consumer-GPU support and tuning. Once that is in a more mature state, we may submit a follow-up PR with updated results for consumer-level GPUs. Thanks again for maintaining this repo and making it a neutral place for documentation :)




### #13 [closed] PR: Add Turing (sm_75) 2080Ti 22G handbook + environment report
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/13  by IvenKooLab 2026-09-03T17:05:04Z

Adds two entries backed by a field-tested handbook (all numbers from a real production pipeline, same-seed A/B measured):

1. **By VRAM and hardware** — a new `22 GB, Turing (sm_75)` row pointing to [minimax-h3-turing](https://github.com/IvenKooLab/minimax-h3-turing). The table currently has no Turing entry, yet modded 2080Ti 22G cards are a popular budget path; the handbook covers the sm_75-specific quantization verdict (W4A8-only, 18× error gap vs W4A4), launch flags, and the cu130 dequantization runtime requirement.

2. **Reported environments** — our environment report: RTX 2080Ti 22G, ComfyUI 0.33.1 → master 345c919, PyTorch 2.9.1+cu130, with measured results: T8 BlockCache −43% on the 4-step Turbo route (drafts), and PDD 8-step + T8 combo at 210 s/clip (−34%, 6/8 cache hits) ported ahead of the official release.

The handbook includes a 13-entry pitfall FAQ, a SageAttention sm_75 crash autopsy, and same-seed A/B scripts.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/13"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1791047108&installation_model_id=427615&pr_number=13&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F13&signature=ed86047f0920fa00821fba6cd012b3b7dbfefdc0516b4d0c6c3e8542f08fb73c"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #12 [closed] PR: Add GB300 benchmarking and observability resource
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/12  by EzgiTastan 2026-08-29T17:15:04Z

Adds a reproducible MiniMax-H3 GB300 benchmarking and observability resource to the Serving H3 table.

It covers SGLang scaling, eBPF CUDA tracing, Grace–GPU traffic, energy, and Nsight comparison.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/12"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1790615708&installation_model_id=427615&pr_number=12&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F12&signature=a2553daf71cf610f71dd298e97079135de885fa41772a4a40a56ec7233003058"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #11 [closed] PR: Add MiniMax H3 1K prompt dataset to the prompt resources
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/11  by tonyclawskill 2026-08-28T07:08:03Z

Alongside the prompt corpora already listed here, this curated 1K text-to-video prompt dataset is a natural companion: 3-field prompt-structure anatomy, 10 hand-picked reusable prompts, and an H3 vs. peer model comparison.\n\nLinks:\n- Interactive atlas of all 1,000 H3 clips: https://neta.art/use-cases/en/h3-1000-prompt-list\n- Curated index repo: https://github.com/yangzhou-chaofan/minimax-h3-1000-prompts\n- Original dataset (ostris/minimax_h3_1k): https://huggingface.co/datasets/ostris/minimax_h3_1k

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/11"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1790492886&installation_model_id=427615&pr_number=11&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F11&signature=d9c9badab666825960a5429c5bc739dcfa0fb5ccf08cff7fda830464a3571959"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #10 [closed] PR: SGLang's verified RTX 5090 lossless recipe
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/10  by Odysseusailoon 2026-08-25T00:31:58Z

- Add a '32 GB (RTX 5090), lossless' row to the VRAM table deep-linking the verified SGLang cookbook config (single 5090, layerwise offload, native precision, 32 GB host-RAM baseline).
- Link the single-5090 native-precision recipe from the Serving H3 row.
- Drop the Blackwell-only hybrid NVFP4 community quants (and the Sol-Attn clause referencing them): superseded by the verified lossless path. Sol-Attn stays for ComfyUI speed.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/10"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1790209921&installation_model_id=427615&pr_number=10&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F10&signature=39875fe07b11be10e7f59482909d60028170e88449f090e87dee883c1a70326a"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #9 [closed] PR: Add NVlabs/Sana Sol-Engine to Runtimes
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/9  by Odysseusailoon 2026-08-25T00:02:00Z

Hardware-specific full-stack H3 runtimes on the sol-engine branch: released full-opt configs for GB200, GB10, H100, A100, RTX 4090/5090, 3.55-4.52x speedup vs matching same-hardware baselines.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/9"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1790208123&installation_model_id=427615&pr_number=9&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F9&signature=ff3f5ec1f354418d6bffac93340c3c46a90cf686351ddcff38fb5415c4f0da74"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #8 [closed] PR: Prompting: collect agent skills into their own section
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/8  by joeVenner 2026-08-24T18:52:49Z

Agent-skill repos are currently spread across two places — one row in the prompt table (`minimax-h3-prompt-skill-T8`) and a trailing sentence listing four more. That made it hard to see what each one actually does for you, since they range from "writes a prompt and hands it back" to "produces a finished file."

This groups them under a new **Agent skills** subsection (anchor `#agent-skills`) with a **Runs against** column, which is the real difference between them.

**On ordering:** the rows run by how much of the pipeline each one covers, most complete first — hosted API → local ComfyUI → prompt-only. That is deliberately not a popularity order, and I want to flag it rather than have you discover it: my own entry is the newest and smallest of the four, and it sits first only because it covers the most of the pipeline. The existing prompt table already orders on something other than stars (`minimax-h3-prompt-skill-T8` is the most-starred entry in this section and sits last), so there's precedent — but if you'd rather this table ran by developer interest like the rest of the index, say so and I'll flip it. My entry goes last under that rule and that's completely fine by me.

**Added:** [`joeVenner/awesome-minimax-h3`](https://github.com/joeVenner/awesome-minimax-h3) — MIT, plain Markdown `SKILL.md` files with no runtime to install. Carries a brief through to a finished file against the hosted MiniMax API: submit, poll with backoff, resolve the returned `file_id`, post-process with `ffmpeg`. It reports that the documented `/v2/…/query` polling form returns 404 and gives `GET /v1/query/video_generation` as the working call.

Kept deliberately narrow:

- The prompt table is untouched, as is the `OpenH3-IR` double-listing (the page already explains that one is intentional).
- `ComfyUI-PainterNodes` and `minimax-h3-prompt-skill-T8` keep their cross-references in a trailing line, so nothing that was linked before is now unreachable.
- The new entry is flagged inline as needing a `MINIMAX_API_KEY` rather than local weights, since the rest of this index is local-first. It also ships Speech 2.8 and Music 3.0 skills; those are named in the cell but not given their own rows, as they're outside this index's scope.

Happy to reword, reorder, or drop my entry — and if you'd rather keep the flat sentence and just take the one addition, I can reduce this to a one-line diff instead.

Context: Ryan pointed me here by email, and Asuka asked me to submit a PR for review.


### #7 [closed] PR: docs: add miles_diffusion to training part
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/7  by Rockdu 2026-08-20T22:09:45Z



<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/7"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1789855788&installation_model_id=427615&pr_number=7&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F7&signature=50bd8018099c0d7cf9a7cbd087519adf5046ab9965b138b1e26ac8f43f70c832"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #6 [closed] ISSUE: 您好,你收录的TE-Speed-MiniMaxH3-OSS此仓库是盗用我代码的仓库,我是插件原作者,希望您修改一下
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/issues/6  by tl2012tl 2026-08-18T04:13:37Z

您好,您文档里收录的https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS
 仓库为盗用我代码者发布的仓库,里面代码来自于AI逆向或猜测,没有任何安全性和可靠性.
此cofmyui加速插件TE-Speed-MiniMaxH3 1.0我于2026-08-03 21:24:51首次发布于B站.链接: https://www.bilibili.com/video/BV1WYMR6mETN/?spm_id_from=333.1387.list.card_archive.click&vd_source=56945eb7482411a19daf38ab98a6e1fa    
当时并未上传github.此人第二天就原名发布在github,冒充我的原名节点,导致很多人认为是我的仓库.此事我已在b站和q群多次说明.您估计未知.
且现在TE-Speed-MiniMaxH3此节点已经更新到3.2版本,也不用再修改cofmyui的文件,而且期间comfyui对于h3的接口已经改过多次.1.0的代码已经无法匹配.

希望您修改为作者(我本人的正确仓库):https://github.com/tl2012tl/TE-Speed-MiniMaxH3


最后,感谢MiniMaxH3团队带来如此振奋人心的作品.



> **Comment by Odysseusailoon (2026-08-18T21:41:29Z):**

没问题


> **Comment by Odysseusailoon (2026-08-19T21:41:23Z):**

您好，感谢您的说明，也抱歉给您带来了困扰。
    
我们已经核实了情况（对比了两个仓库的创建时间和您 2026-08-03 的 B 站首发视频），并已在 commit 1121b20 (https://github.com/MiniMax-AI/awesome-minimax-h3-integration/commit/1121b20) 中将 README 里全部三处引用替换为您的仓库 tl2012tl/TE-Speed-MiniMaxH3 (https://github.com/tl2012tl/TE-Speed-MiniMaxH3)，条目描述也已按 v3.2 的实际情况重写（适配新版 ComfyUI、不再修改核心文件、4/8 步 LoRA 模式、长视频缓存策略）。
                                                                  
感谢您为 H3 社区做的加速工作，也感谢您的支持。<3



### #5 [closed] PR: Add xcc3641/minimax-h3-starter to Prompting
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/5  by xcc3641 2026-08-18T00:19:14Z

Adds [`xcc3641/minimax-h3-starter`](https://github.com/xcc3641/minimax-h3-starter) under **Prompting** only.

It is a beginner lab: T2V/I2V case studies with full prompt revision history, plus a portable `SKILL.md` for T2VA / I2VA / FL2VA / L2VA / Ref2VA. Same shelf as `awesome-minimax-h3-prompts` and `minimax-h3-prompt-skill-T8`.



### #4 [closed] ISSUE: Add OpenH3-IR: an open take on H3's Context-IR stage, with a command line, a service and three ComfyUI nodes
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/issues/4  by ruashots 2026-08-16T05:20:51Z

OpenH3-IR is an open take on H3's Context-IR stage, Apache-2.0, at https://github.com/ruashots/open-h3-ir

You published the Context-IR format and finished examples of it, and kept the part that writes it on your own servers. This follows what you published, so it is not a prompt enhancer: the document comes out with the named sections in a fixed order, numbered picture labels, a length on the frame grid, per-shot cut times and sound described separately. It calls an OpenAI-compatible model you already run, which is where the writing happens. The service itself wants no GPU, and nothing in it calls MiniMax.

It comes with two ways to use it, which is why it fits two of your shelves. One is a command line and a small local service. The other is three ComfyUI nodes that use that same service from the canvas. Both rows are below in your formats. Take either, both, or neither.

Prompting section:

```
| [`OpenH3-IR`](https://github.com/ruashots/open-h3-ir) | Follows the Context-IR format and examples MiniMax published, rather than rewriting your prompt: it writes the document, then checks it and fixes what is wrong. Command line, HTTP, or its own ComfyUI nodes, on a model you already run. |
```

ComfyUI nodes section:

```
| [`OpenH3-IR`](https://github.com/ruashots/open-h3-ir) ![Conditioning][cat-cond] | (new) | The ComfyUI side of OpenH3-IR, using its Context-IR service from the same repo, on a model you already run. Drop your pictures, clips and sounds on one panel, name them, then mention them by name in the sentence. |
```

Two notes on the nodes row. The star count is yours to set and it is low, since the repo is new. The badge is a guess at your taxonomy: `cat-cond` is the closest I can see, because the Main node hands out conditioning and a latent, so rename it if you shelve prompt-layer tools elsewhere.

What it does, if you would rather write your own description. You type one plain sentence. What comes back is the long Context-IR document H3 expects, with each picture you attached tied to the right part of the text, a length H3 can actually render, cuts that land inside the clip, and any line of dialogue spoken exactly as you typed it, because your lines never pass through the writing model. It then checks the result and fixes what is wrong, tells you what it could not fix, and falls back to a plain generated draft rather than shipping something broken.

How it keeps itself honest, since the claim is that it follows your format rather than approximating it. Your published examples are in its test set and have to pass clean, because a check that fires on your own artifact is a wrong check. Fifteen deliberately broken copies of one of them each have to fail on the check that defect earns. 22 checks and 901 tests run with no model, no GPU and no network.

One accuracy note if you list the nodes: they write nothing themselves. They talk to the service in the same repo, so the two are one tool listed twice rather than two projects.



### #3 [closed] PR: Audio section
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/3  by Odysseusailoon 2026-08-14T21:59:16Z



<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/3"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1789336759&installation_model_id=427615&pr_number=3&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F3&signature=17f45dfe981702213b702cc3aa99e6aea0e5e53289e4748fd9011685e35c6f9d"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith-bot</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #2 [closed] PR: Review pass: correct the source table, drop the glyph decoration, bra…
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/2  by Odysseusailoon 2026-08-14T06:54:33Z

…nd the badges

- Checkpoints source table: MiniMaxAI/MiniMax-H3 is 280 files / 464.2 GiB, not 104 / 464.1, and the transformers are 14 shards each, not 13. Filled in the Comfy-Org row that was two em-dashes: 17 files / 433.2 GiB. All figures re-read from the Hugging Face tree API; the ten per-precision sizes and the three text encoders already matched byte for byte.
- Badges use each host's real brand colour instead of grey: Hugging Face FFD21E, GitHub 181717, ComfyUI 211927, plus a MiniMax E73562 badge. Dropped logo=comfyui, which simple-icons does not carry, so it was rendering blank. Applies to all 38 uploader download shields as well.
- Removed the block-glyph heading prefixes and the decorative separator rows. The separators carried the table-of-contents targets, so each one became an invisible <a id> in place; all 24 in-page links still resolve.
- Trimmed the out-of-scope notes and rewrote Acknowledgements as Sources, stating where each figure came from rather than thanking its author.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/2"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1789282476&installation_model_id=427615&pr_number=2&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F2&signature=08569d86b348cc9e125d419dc61aa457c8340117a122db48d6fd69ae61a67838"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #1 [closed] PR: Add the MiniMax-H3 open ecosystem index
URL: https://github.com/MiniMax-AI/awesome-minimax-h3-integration/pull/1  by Odysseusailoon 2026-08-14T04:28:33Z

Merges MiniMax's own ecosystem scan (GitHub + Hugging Face, snapshot 2026-08-13) with the community-curated wildminder/awesome-minimax-H3, as a union of both rather than a rewrite of either.

Covers checkpoints, the Turbo acceleration LoRA, safetensors and GGUF quantizations, text encoders, separated components (VAE / TAE / Image VAE / ClipProj / Ref Patch), style LoRAs, VRAM-tiered workflow recommendations, training, inference engines, ComfyUI nodes, guides, compatibility and patching notes, and licensing.

Sizes are binary units throughout (GiB = bytes / 1024^3), stated explicitly in the scope note, because the two sources label the same figures differently.

<!-- codesmith:footer -->
---
<a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/awesome-minimax-h3-integration/pr/1"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-light-v2.svg"><img alt="View with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/view-with-codesmith-dark-v2.svg"></picture></a> <a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1789273716&installation_model_id=427615&pr_number=1&repository=MiniMax-AI%2Fawesome-minimax-h3-integration&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2Fawesome-minimax-h3-integration%2Fpull%2F1&signature=d44c3f11ac5ba75ed5fc6b832363baaa25235c46d11a29be43e2f5fce095f8aa"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"></picture></a>
<sup>Need help on this PR? Tag <code>@codesmith</code> with what you need. Autofix is disabled.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->
