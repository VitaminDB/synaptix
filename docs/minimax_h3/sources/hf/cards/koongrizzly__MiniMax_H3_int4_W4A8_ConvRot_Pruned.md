# koongrizzly/MiniMax_H3_int4_W4A8_ConvRot_Pruned

Источник: https://huggingface.co/koongrizzly/MiniMax_H3_int4_W4A8_ConvRot_Pruned · скачано 2026-11-28
Загрузок: 29919 · лайков: 11 · базовая:  · теги: quantization · обновлено: 2026-11-21T21:02:10.000Z

---

---
license: apache-2.0
tags:
- minimax
- minimax-h3
- video-generation
- audio-video-generation
- int4
- convrot
- w4a8
- quantization
---

# Files for the MiniMax H3 standalone app : 

https://github.com/Koongrizzly/MiniMax_H3_Standalone_app (also allows loading int8 models)


# MiniMax H3 INT4 ConvRot + Mixed FP16/FP32 Video VAE

This repository contains a collection of MiniMax H3 model components intended for lower-memory local inference.

The original MiniMax H3 model and architecture were released by **MiniMaxAI**:

https://huggingface.co/MiniMaxAI/MiniMax-H3

MiniMax H3 is an omni-modal video generation model capable of generating synchronized video and stereo audio. The official release supports text-to-audio-video, first/last-frame-to-audio-video, and multimodal reference-to-audio-video generation, depending on the selected model variant.

The original MiniMax H3 release, documentation, architecture information, usage instructions and licensing information can be found in the official repository linked above.


# September 19 2026 : updated with another Hybrid model version

This can replace both the rf2va and the ref2va model and allows using start image + reference in one job
Hybrid model one : all credits to berryber09, go here for all info :
https://huggingface.co/berryber09/MiniMax-H3-ref2va-fl2va-hybrid-w4a8
Hybrid model 2 : minimaxH3_hybrid_Sparseref15_prunedPartialINT8V10 by Aki7777777, So all credits to this person for creating this great checkpoint.
For more info go to https://civitai.com/models/2900456/minimax-h3sparseref15hybrid


## INT4 ConvRot model files


The following three quantized model files included in this repository were copied from the MiniMax H3 INT4 ConvRot release by **Winnougan**:

https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI

- `minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors`
- `minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors`
- `minimax_h3_te_w4a8_convrot.safetensors`

These files use the W4A8 / ConvRot quantization format.

The FL2VA and Ref2VA transformer checkpoints are pruned and W4A8 ConvRot quantized versions of the MiniMax H3 diffusion transformers. The text encoder is also provided in W4A8 ConvRot format, while its visual component remains at higher precision as in the source conversion.

All credit for these INT4 ConvRot conversions belongs to **Winnougan** and the respective contributors to the ConvRot/W4A8 tooling.

For additional technical information about the quantization format, requirements and original converted files, see the Winnougan repository linked above.

## Mixed-precision Video VAE

This repository also contains a modified version of the MiniMax H3 Video / Visual VAE.

The original Video VAE weights originate from the official MiniMax H3 release.

For this repository, I converted the majority of the Video VAE tensors from FP32 to **FP16** in order to substantially reduce its memory and storage requirements.

Instead of blindly converting the entire VAE to FP16, selected small and precision-sensitive blocks were intentionally retained in **FP32**.

The resulting file is therefore a **mixed FP16/FP32 Video VAE**:

- Large weight tensors are stored in FP16.
- Selected smaller / precision-sensitive tensors remain FP32.
- The goal is to reduce memory usage while preserving as much reconstruction quality and numerical stability as possible.
- The model architecture itself was not retrained or fine-tuned.


## MiniMax H3 Tubo 4 step loras

Newest Turbo lora step600 was copied from  https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora


## Model components


- FL2VA INT4/W4A8 ConvRot transformer
- Ref2VA INT4/W4A8 ConvRot transformer
- INT4/W4A8 ConvRot MiniMax H3 text encoder
- Mixed FP16/FP32 Video VAE
- MiniMax H3 FP32 Audio VAE
- Supporting configuration files where required


## Intended use

The purpose of this repository is to provide a more practical MiniMax H3 model set for local inference, particularly on consumer GPUs where loading the original full-precision model components can require a very large amount of VRAM and system memory.

The quantized components trade some numerical precision for substantially reduced memory requirements.

The mixed-precision Video VAE is intended to further reduce memory consumption compared with the original FP32 VAE while retaining FP32 precision where it may matter most.

Officially created for use with Comfyui. With some work and the correct (comfyui) loader these also work standalone in other setups.

## Credits

**Original model**

MiniMaxAI — MiniMax H3  
https://huggingface.co/MiniMaxAI/MiniMax-H3

**INT4 / W4A8 ConvRot conversions**

Winnougan — MiniMax-H3-INT4_Convrot_ComfyUI  
https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI

**Mixed FP16/FP32 Video VAE conversion**

Converted from the official MiniMax H3 Video VAE for reduced-memory inference. Most tensors were converted to FP16 while selected small / precision-sensitive tensors were preserved in FP32.

## License and usage

This repository contains components derived from or redistributed from other projects.

Users should review and comply with the licensing and usage terms of the original MiniMax H3 model and any applicable terms associated with the converted components.

This repository is not affiliated with or endorsed by MiniMaxAI.
