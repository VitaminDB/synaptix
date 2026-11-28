# ibyteohdear/MiniMax-H3-Pruned-DMD-FL2VA-Turbo-Fused

Источник: https://huggingface.co/ibyteohdear/MiniMax-H3-Pruned-DMD-FL2VA-Turbo-Fused · скачано 2026-11-28
Загрузок: 599 · лайков: 1 · базовая: multimodalart/MiniMax-H3-Pruned · теги:  · обновлено: 2026-11-28T02:41:08.000Z

---

---
library_name: diffusers
base_model: multimodalart/MiniMax-H3-Pruned
tags: [minimax-h3, text-to-video, image-to-video, audio-video, diffusers, 10eros, not-for-all-audiences]
---

```python
import ctypes         # Ask Linux/glibc to return unused heap memory
import os
import sys 
import tempfile
import random
BASE = tempfile.gettempdir()
IS_HF_SPACES = bool(os.getenv("SPACES_ZERO_GPU"))
HF_TOKEN = os.getenv("HF_TOKEN")
if IS_HF_SPACES:
    import spaces
    clip_dir=BASE
else:
    os.environ["HF_HOME"] = "/mnt/data/huggingface"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    clip_dir="/mnt/data/huggingface"

import gradio as gr
import torch
import gc
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download

BASE_REPO = "ibyteohdear/MiniMax-H3-Pruned-DMD-FL2VA-Turbo-Fused"
CLIP = False
if CLIP:
    b4 = False
    if b4:
        TEXT_REPO = "prithivMLmods/Qwen3-VL-4B-Instruct-c_abliterated-v2"
    else:    
        TEXT_REPO = "prithivMLmods/Qwen3-VL-8B-Instruct-c_abliterated-v3"
else:
    TEXT_REPO = "ibyteohdear/qwen3_vl_32b_bnb_nf4"

diffusers_path = snapshot_download(
    repo_id=BASE_REPO,
    token=HF_TOKEN,
    allow_patterns=["diffusers/*"],
    local_dir="diffusers"
)
sys.path.insert(0, diffusers_path)

from diffusers import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
)
from diffusers.utils.export_utils import encode_video
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, Qwen3VLProcessor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import (
    MiniMaxH3SetupStep,
)
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3DenoiseStep,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3TextEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoKeyframeVaeEncoderStep,
    MiniMaxH3DecodeStep,
    _generation_outputs,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam
import torch.nn.functional as F
from transformers import TorchAoConfig as TransformersTorchAoConfig
from diffusers import AutoModel
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.modular_pipelines import ModularPipelineBlocks
from peft.tuners.tuners_utils import BaseTunerLayer
from contextlib import contextmanager
from diffusers.hooks import apply_group_offloading
import threading

# -----------------------------------------------------------------------------
# 1. Zero-GPU decorator
# -----------------------------------------------------------------------------
def zero_gpu_decorator(duration=1500):
    def decorator(func):
        if IS_HF_SPACES: 
            return spaces.GPU(duration=duration)(func)
        return func
    return decorator

#--------------------------------------------------------------------------
# 2. ClipProj
# -----------------------------------------------------------------------------
if CLIP:
    if b4:
        PROJ_FILENAME = "mmh3-4b-ClipProj-v3.1.safetensors"
        MLP_FILENAME  = "mmh3-4b-ClipProj-v3.1-mlp.safetensors"
    else:
        PROJ_FILENAME = "mmh3-8b-ClipProj-v3.1.safetensors"
        MLP_FILENAME  = "mmh3-8b-ClipProj-v3.1-mlp.safetensors"

    CLIPPROJ_REPO = "NicoLab28/ClipProj-MiniMax-H3"

    clipproj_dir = os.path.join(clip_dir, "clipproj")
    os.makedirs(clipproj_dir, exist_ok=True)

    proj_path = hf_hub_download(
        repo_id=CLIPPROJ_REPO,
        filename=PROJ_FILENAME,
        local_dir=clipproj_dir,
    )

    mlp_path = hf_hub_download(
        repo_id=CLIPPROJ_REPO,
        filename=MLP_FILENAME,
        local_dir=clipproj_dir,
    )

    proj_weights = load_file(proj_path)
    mlp_weights = load_file(mlp_path)

    W = proj_weights["W"].to(torch.bfloat16)
    mean_in = proj_weights["mean_in"].to(torch.bfloat16)
    std_in = proj_weights["std_in"].to(torch.bfloat16)
    mean_out = proj_weights["mean_out"].to(torch.bfloat16)
    std_out = proj_weights["std_out"].to(torch.bfloat16)

    sink_out = proj_weights.get("sink_out")

    if sink_out is not None:
        sink_out = sink_out.to(torch.bfloat16)

# -----------------------------------------------------------------------------
# 3. Classes
# -----------------------------------------------------------------------------
if CLIP:
    class ClipProjModelWrapper(torch.nn.Module):
        def __init__(
            self,
            original_model,
            W,
            mean_in,
            std_in,
            mean_out,
            std_out,
            mlp_weights,
            sink_out=None,
        ):
            super().__init__()

            self.original_model = original_model

            self.register_buffer("W", W)
            self.register_buffer("mean_in", mean_in)
            self.register_buffer("std_in", std_in)
            self.register_buffer("mean_out", mean_out)
            self.register_buffer("std_out", std_out)

            if sink_out is not None:
                self.register_buffer("sink_out", sink_out)
            else:
                self.sink_out = None

            d_in = W.shape[0]
            out_dim = W.shape[1]

            mlp_in = mlp_weights["mlp.0.weight"].shape[1]
            hidden_dim = mlp_weights["mlp.0.weight"].shape[0]
            mlp_out = mlp_weights["mlp.2.weight"].shape[0]

            self.fc1 = torch.nn.Linear(
                d_in,
                hidden_dim,
                bias=True,
                dtype=torch.float16,
            )

            self.fc2 = torch.nn.Linear(
                hidden_dim,
                out_dim,
                bias=True,
                dtype=torch.float16,
            )

            with torch.no_grad():

                self.fc1.weight.copy_(
                    mlp_weights["mlp.0.weight"]
                )

                self.fc1.bias.copy_(
                    mlp_weights["mlp.0.bias"]
                )

                self.fc2.weight.copy_(
                    mlp_weights["mlp.2.weight"]
                )

                self.fc2.bias.copy_(
                    mlp_weights["mlp.2.bias"]
                )

            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)

            self.fc1.eval()
            self.fc2.eval()

        def forward(self, *args, **kwargs):
            kwargs["output_hidden_states"] = True
        
            outputs = self.original_model(
                *args,
                **kwargs,
            )
        
            h = outputs.hidden_states[24]
        
            h_norm = (
                h - self.mean_in
            ) / self.std_in
        
            projected = h_norm @ self.W
        
            with torch.no_grad():
                mlp_input = h_norm.to(self.fc1.weight.dtype)
        
                residual = self.fc2(
                    F.gelu(
                        self.fc1(mlp_input)
                    )
                ).to(projected.dtype)
        
            projected = projected + residual
        
            projected = (
                projected * self.std_out
                + self.mean_out
            )
        
            if self.sink_out is not None:
                projected = projected.clone()
        
                projected[:, 0, :] = self.sink_out.to(
                    device=projected.device,
                    dtype=projected.dtype,
                )
                
            hidden_states = list(outputs.hidden_states)
        
            while len(hidden_states) <= 50:
                hidden_states.append(projected)
        
            hidden_states[50] = projected
        
            outputs.hidden_states = tuple(hidden_states)
            outputs.last_hidden_state = projected
        
            return outputs


_encoder_lock = threading.RLock()
_encoder_offload_done = False


@contextmanager
def use_base_model(encoder, base_model, active_wrapper):

    previous_model = encoder.model

    try:
        if CLIP:
            encoder.model = base_model

        else:
            encoder.model = previous_model

        yield encoder

    finally:
        encoder.model = previous_model
        

@contextmanager
def offload_text_encoder():
    global _encoder_offload_done

    with _encoder_lock:
        text_encoder_b.to("cpu")
        if CLIP and clip_wrapper:
            clip_wrapper.to("cpu")

        torch.cuda.empty_cache()
        gc.collect()

        try:
            text_encoder_b.to("cuda")
            if CLIP and clip_wrapper:
                clip_wrapper.to("cuda")
            yield
        finally:
            text_encoder_b.to("cpu")
            if CLIP and clip_wrapper:
                clip_wrapper.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()


class MiniMaxH3PromptEnhanceStep(ModularPipelineBlocks):
    model_name = "minimax-h3-prompt-enhance"

    @property
    def expected_components(self):
        return [
            ComponentSpec("text_encoder", type(None)),
        ]

    @property
    def inputs(self):
        return [
            InputParam("prompt", type_hint=str, default=""),
            InputParam("image", type_hint=object, default=None),
            InputParam("enhance_prompt", type_hint=bool, default=True),
            InputParam("enhance_temperature", type_hint=float, default=0.7),
        ]

    @property
    def outputs(self):
        return [
            OutputParam(
                "prompt",
                type_hint=str,
                description="Enhanced prompt string.",
            ),
        ]

    @torch.inference_mode()
    def __call__(self, components, state):

        raw_prompt = (getattr(state, "prompt", None) or "").strip()
        image = getattr(state, "image", None)

        enhance_flag = getattr(state, "enhance_prompt", True)
        enhance_temp = getattr(state, "enhance_temperature", 0.7)

        if not enhance_flag or (not raw_prompt and image is None):
            state.prompt = raw_prompt or "A cinematic scene."
            return components, state

        processor = (
            getattr(components, "processor", None)
            or h3_processor
        )

        tokenizer = (
            getattr(components, "tokenizer", None)
            or h3_tokenizer
        )

        system_prompt = (
            "You are an expert video prompt engineer for MiniMax-H3. "
            "You have access to a reference image and the user's original prompt. "
            "Analyze the image carefully, including subjects, appearance, "
            "clothing, environment, composition, lighting, colors and visual style. "
            "Use the image as the primary visual reference. "
            "Preserve the identity and appearance of all subjects. "
            "Create a vivid cinematic video description covering realistic "
            "subject motion, interactions, environmental motion, lighting, "
            "and camera movement. "
            "Do not merely describe a static image. "
            "Describe how the scene should animate over time. "
            "Do not introduce unrelated subjects or contradict the reference image. "
            "Output ONLY the enhanced video prompt."
        )

        user_content = []

        if image is not None:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            if isinstance(image, Image.Image):
                image = image.convert("RGB")

                user_content.append({
                    "type": "image",
                    "image": image,
                })

        user_content.append({
            "type": "text",
            "text": (
                f"Original video prompt: {raw_prompt or 'Create a cinematic video based on the reference image.'}

"
                "Use the supplied image as the visual reference."
            ),
        })

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        model_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        with use_base_model(
            text_encoder_b,
            base_model,
            clip_wrapper,
        ) as gen_model:
            try:
                with torch.inference_mode():
                    outputs = gen_model.generate(
                        **model_inputs,
                        max_new_tokens=320,
                        do_sample=True,
                        temperature=max(0.1, float(enhance_temp)),
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
        
                input_length = model_inputs.input_ids.shape[1]
                generated_tokens = outputs[:, input_length:]
        
                enhanced_prompt = tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )[0].strip()
        
                state.prompt = enhanced_prompt or raw_prompt or "A cinematic scene."
            finally:
                for var_name in ("model_inputs", "outputs", "generated_tokens"):
                    if var_name in locals():
                        del locals()[var_name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return components, state
        

class MiniMaxH3ConditionerBlocks(SequentialPipelineBlocks):
    model_name = "minimax-h3"

    block_classes = [
        MiniMaxH3PromptEnhanceStep,
        MiniMaxH3SetupStep,
        MiniMaxH3TextEncoderStep,
    ]

    block_names = [
        "prompt_enhance",
        "setup",
        "text_encoder",
    ]

    @property
    def outputs(self):
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam(
                "text_token_tags",
                description="The per-row modality tag of every row of prompt_embeds.",
            ),
            OutputParam(
                "height",
                type_hint=int,
                description="Resolved height.",
            ),
            OutputParam(
                "width",
                type_hint=int,
                description="Resolved width.",
            ),
            OutputParam(
                "num_frames",
                type_hint=int,
                description="Resolved number of frames.",
            ),
        ]


class MiniMaxH3GeneratorBlocks(SequentialPipelineBlocks):
    model_name = "minimax-h3"

    block_classes = [
        MiniMaxH3SetupStep,
        MiniMaxH3AutoKeyframeVaeEncoderStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3DenoiseStep,
        MiniMaxH3DecodeStep,
    ]

    block_names = [
        "setup",
        "vae_encoder",
        "prepare_layout",
        "prepare_latents",
        "set_timesteps",
        "denoise",
        "decode",
    ]

    @property
    def outputs(self):
        return _generation_outputs()


# ------------------------------------------------------------------
# 4. Load Qwen3-VL-8b (with explicit CPU offload)
# ------------------------------------------------------------------
dtype = torch.bfloat16

clip_wrapper = None

if CLIP:
    quantization_config = TransformersTorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=["model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"],
    )
    
    text_encoder_b = Qwen3VLForConditionalGeneration.from_pretrained(
        TEXT_REPO,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"}, 
        quantization_config=quantization_config, 
    )
    
else:    
    text_encoder_b = Qwen3VLForConditionalGeneration.from_pretrained(
        TEXT_REPO,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},  
    )
         
base_model = text_encoder_b.model
if hasattr(base_model, "to"):
    base_model.to("cpu")

if CLIP and clip_wrapper:
    clip_wrapper.to("cpu")

torch.cuda.empty_cache()
gc.collect()

# ------------------------------------------------------------------
# 5. Wrap the model with ClipProj (if CLIP)
# ------------------------------------------------------------------
if CLIP:
    original_num_layers = text_encoder_b.config.text_config.num_hidden_layers
    text_encoder_b.config.text_config.num_hidden_layers = 64

    clip_wrapper = ClipProjModelWrapper(
        original_model=base_model,
        W=W,
        mean_in=mean_in,
        std_in=std_in,
        mean_out=mean_out,
        std_out=std_out,
        mlp_weights=mlp_weights,
        sink_out=sink_out,
    )
    text_encoder_b.model = clip_wrapper
    clip_wrapper.to("cpu")

text_encoder_b.requires_grad_(False)
text_encoder_b.eval()

if CLIP:
    clip_wrapper.to("cpu")
text_encoder_b.to("cpu")
torch.cuda.empty_cache()
gc.collect()

# -----------------------------------------------------------------------------
# 6. Build pipeline components
# -----------------------------------------------------------------------------
transformer = AutoModel.from_pretrained(
    BASE_REPO,
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

LORA_REPO = "ibyteohdear/h3-turbo-diffusers"

ADAPTERS = [
    # ("VBVR",       "max/H3_VBVR_Pro-V2.safetensors",                                                  1.0),
    # ("HMM",        "max/hmmotion_minimax-h3_epoch12.safetensors",                                     1.0),
    # ("MLP",        "max/Eros_beta6mix_MLP_reference_diffusers.safetensors",                           1.0),
    # ("EROS",       "max/Eros_beta6mix_Lora_diffusers.safetensors",                                    1.0),
    # ("DREAM",      "max/H3_DR34ML4Y_P1.safetensors",                                                  1.0),
    # ("AIO",        "max/HMNSFW-AIO-V2.5.safetensors",                                                 1.0),
    # ("PINK",       "max/PinkCherry_NaughtyTimes-v2-rank128.safetensors",                              1.0),
    # ("VAG",        "max/minimax_vag_000002500.safetensors",                                           1.0),
    # ("3D",         "max/Tentacles-3D_minimaxh3_3344593_epoch_7.safetensors",                          1.0),
    # ("101",        "max/Tentacles_H3_101_.safetensors",                                               1.0),
    # ("BOOSTER",    "max/motion_booster_v2_mlp_e995_r8_diffusers.safetensors",                         1.0),
    # ("ANEL",       "max/silver_anel_warmup_mlp_e995_r8_diffusers.safetensors",                        1.0),
    # ("SLOP",       "max/slop_bounce_e995_r8_diffusers.safetensors",                                   1.0),
    # ("ASTRO",      "max/Astro nsfw.safetensors",                                                      1.0),    
    # ("SPLIT",      "max/split.safetensors",                                                           1.0),
    # ("LOCK",       "max/Minimax_H3_Unlocked_V2.1.safetensors",                                        1.0),
    # ("MYS",        "max/MysticXXX_MMH3-V5.safetensors",                                               1.0),
    # ("FURRY",      "max/furry_lora_epoch31.safetensors",                                              1.0),
    # ("MOT",        "max/Sulphur-3_Better_NSFW_Motion.safetensors",                                    1.0),
    # ("TIT",        "max/titfuck101_e995_r8.safetensors",                                              1.0),
    # ("SLOP",       "max/slop_twerk_e995_r8.safetensors",                                              1.0),
    # ("LACT",       "max/lactation-minimax_h3-v1.e02850_rank6.safetensors",                            1.0),
    # ("MOT_D",      "Sulphur-3_Better_NSFW_Motion_diffusers.safetensors",                              1.0),
    # ("LACT_D",     "lactation-minimax_h3-v1.e02850_rank6_diffusers.safetensors",                      1.0),
    # ("SLOP_D",     "slop_twerk_e995_r8_diffusers.safetensors",                                        1.0),
    # ("TIT_D",     "titfuck101_e995_r8_diffusers.safetensors",                                         1.0),
    # ("ASTRO_D",    "Astro nsfw_diffusers.safetensors",                                                1.0),
    # ("SPLIT_D",    "split_diffusers.safetensors",                                                     1.0),
    # ("LOCK_D",     "Minimax_H3_Unlocked_V2.1_diffusers.safetensors",                                  1.0),
    # ("MYS_D",      "MysticXXX_diffusers.safetensors",                                                 1.0),
    # ("FURRY_D",    "furry_diffusers.safetensors",                                                     1.0),
    # ("VBVR_D",     "H3_VBVR_Pro-V2_diffusers.safetensors",                                            1.0),
    # ("3D_D",      "Tentacles-3D_minimaxh3_3344593_epoch_7_diffusers.safetensors",                     1.0),
    # ("101_D",      "Tentacles_H3_101_diffusers.safetensors",                                          1.0),
    # ("HMM_D",      "hmmotion_minimax-h3_epoch12_diffusers.safetensors",                               1.0),
    # ("MLP_D",      "Eros_beta6mix_MLP_reference_diffusers_auto.safetensors",                          1.0),
    # ("EROS_D",     "Eros_beta6mix_Lora_diffusers_auto.safetensors",                                   1.0),
    # ("DREAM_D",    "H3_DR34ML4Y_P1_diffusers.safetensors",                                            1.0),
    # ("AIO_D",      "HMNSFW-AIO-V2.5_diffusers.safetensors",                                           1.0),
    # ("PINK_D",     "PinkCherry_NaughtyTimes-v2-rank128_diffusers.safetensors",                        1.0),
    # ("VAG_D",      "minimax_vag_000002500_diffusers.safetensors",                                     1.0),
    # ("SEXGOD256_D","SexGod-NaughtyTimes-v2-rank256_diffusers.safetensors",                            1.0),
    # ("SEXGOD512_D","SexyGod-NaughtyTimes-v2-rank512_diffusers.safetensors",                           1.0),
    # ("SEXGOD256",  "SexGod-NaughtyTimes-v2-rank256.safetensors",                                      1.0),
    # ("SEXGOD512",  "SexyGod-NaughtyTimes-v2-rank512.safetensors",                                     1.0),
    # ("TURBO_LX2V_FUSION", "lightx2v_hybrid-4to8step-full-fusion_Turbo_pruned_diffusers.safetensors",  0.65),
    # ("TURBO_LX2V_FUSION_48", "lightx2v_hybrid-4to8step-Turbo_r48_diffusers.safetensors",              0.65),
    # ("TURBO_DMD",         "minimax_h3_dmd_fl2va_8step_turbo_pruned_diffusers.safetensors",            0.35),
    # ("TURBO_FL2V_4S_768", "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16_diffusers.safetensors", 0.35),
    # ("TURBO_FL2V_8S_768", "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16_diffusers.safetensors", 1.0),
    # ("TURBO_FL2V_8S",     "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16_diffusers.safetensors",      1.0),
]

LORA_SCALE = 1.2

def merge_adapter_cpu(transformer, sd_path, scale=1.0, chunk=4096):
    sd = load_file(sd_path, device="cpu")

    pairs = {}

    for k, v in sd.items():
        if ".lora_A." in k:
            prefix = k.split(".lora_A.")[0]
            pairs.setdefault(prefix, {})["A"] = v
        elif ".lora_B." in k:
            prefix = k.split(".lora_B.")[0]
            pairs.setdefault(prefix, {})["B"] = v

    merged = 0

    with torch.no_grad():
        for prefix, ab in pairs.items():
            if "A" not in ab or "B" not in ab:
                continue

            module_name = prefix
            if module_name.startswith("transformer."):
                module_name = module_name[len("transformer."):]

            try:
                mod = transformer.get_submodule(module_name)
            except AttributeError:
                print(f"Skipping unmatched module: {module_name}")
                continue

            A = ab["A"].float()
            B = ab["B"].float()
            w = mod.weight

            if w.device.type != "cpu":
                raise RuntimeError(f"{module_name} is not on CPU: {w.device}")

            for i in range(0, B.shape[0], chunk):
                delta = (B[i:i + chunk] @ A) * scale
                w[i:i + chunk].add_(delta.to(w.dtype))

            merged += 1
            del A, B

    del sd, pairs
    gc.collect()

    print(f"Merged {merged} LoRA modules from {sd_path}")
    
for name, path, w in ADAPTERS:
    sd_path = hf_hub_download(
        LORA_REPO,
        path,
        token=HF_TOKEN
    )

    merge_adapter_cpu(
        transformer,
        sd_path,
        scale=w * LORA_SCALE,
        chunk=4096
    )

    del sd_path
    gc.collect() 

non_cpu = [n for n, p in transformer.named_parameters() if p.device.type != "cpu"]
if non_cpu:
    raise RuntimeError(f"Parameters left off CPU ({len(non_cpu)}), first: {non_cpu[:5]}")

bad_weights = [
    n for n, p in transformer.named_parameters()
    if not torch.isfinite(p).all()
]
if bad_weights:
    raise RuntimeError(f"Non-finite (NaN/Inf) weights detected after merge ({len(bad_weights)}), first: {bad_weights[:5]}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

_SKIP = (
    "proj_in", "audio_proj_in",
    "proj_out", "audio_proj_out",

    "time_embedder", "rope",

    "norm_out",
    "adaln_basis", "adaln_mean", "adaln_proj",
    "norm", "scale_shift_table",

    "context_embedder", "token_refiner",

    "folded_bias",

    "lora_",
)

def _quant_filter(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return False
    return not any(k in fqn for k in _SKIP)

def _iter_units(model):
    for name, child in model.named_children():
        if isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
            for i, block in enumerate(child):
                yield f"{name}.{i}", block
        else:
            yield name, child

config = Int8WeightOnlyConfig(version=2)

for unit_name, unit in _iter_units(transformer):
    quantize_(
        unit,
        config,
        filter_fn=lambda m, fqn, _p=unit_name: _quant_filter(
            m, f"{_p}.{fqn}" if fqn else _p
        ),
    )
    gc.collect()

transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level", 
    use_stream=False,
)

vae = AutoencoderKLMiniMaxH3.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    subfolder="vae",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    subfolder="audio_vae",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

h3_tokenizer = AutoTokenizer.from_pretrained(
    TEXT_REPO, 
    low_cpu_mem_usage=True,
)

h3_processor = Qwen3VLProcessor.from_pretrained(
    TEXT_REPO, 
    low_cpu_mem_usage=True,
)

conditioner_blocks = MiniMaxH3ConditionerBlocks()

conditioner = conditioner_blocks.init_pipeline(
    "MiniMaxAI/MiniMax-H3"
)

conditioner.update_components(
    text_encoder=text_encoder_b,
    tokenizer=h3_tokenizer,
    processor=h3_processor,
)

generator_blocks = MiniMaxH3GeneratorBlocks()

generator = generator_blocks.init_pipeline(
    "MiniMaxAI/MiniMax-H3"
)

generator.update_components(
    transformer=transformer,
    vae=vae,
    audio_vae=audio_vae,
)

generator.load_components(dtype=dtype)

# -----------------------------------------------------------------------------
# 7. Helpers
# -----------------------------------------------------------------------------
FPS = 24
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
RESOLUTION_PROFILES = {
    "super": { "16:9": (1536, 896), "9:16": (896, 1536),"1:1": (1088, 1088) } ,
    "ultra": { "16:9": (1472, 832), "9:16": (832, 1472), "1:1": (1024, 1024) },
    "high":   { "16:9": (1408, 768), "9:16": (768, 1408), "1:1": (960, 960) },
    "medium": { "16:9": (1344, 704), "9:16": (704, 1344), "1:1": (896, 896) },
    "low":    { "16:9": (1280, 640), "9:16": (640, 1280), "1:1": (832, 832) },
}
MAX_SEED = np.iinfo(np.int32).max

def snap_frames(seconds: float) -> int:
    frames = max(1, round(seconds * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames

def auto_crop_image(image_path: str, target_width: int, target_height: int, session_id: str = "default") -> str:
    print(f"[Auto-Crop] Resizing & Cropping to match target resolution: {target_width}x{target_height}")
    
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    target_aspect = target_width / target_height
    orig_aspect = orig_w / orig_h
    
    if abs(orig_aspect - target_aspect) < 0.01:
        print("[Auto-Crop] Image aspect ratio matches settings. Skipping modifications.")
        return image_path

    if orig_aspect > target_aspect:
        new_h = orig_h
        new_w = int(orig_h * target_aspect)
        left = (orig_w - new_w) // 2
        top = 0
        right = left + new_w
        bottom = orig_h
    else:
        new_w = orig_w
        new_h = int(orig_w / target_aspect)
        left = 0
        top = (orig_h - new_h) // 2
        right = orig_w
        bottom = top + new_h
        
    img_cropped = img.crop((left, top, right, bottom))
    print(f"[Auto-Crop] Centered crop applied from {orig_w}x{orig_h} to bounds: left={left}, top={top}, right={right}, bottom={bottom}")
    
    cropped_out_path = os.path.join(BASE, f"cropped_{session_id}.png")

    if os.path.exists(cropped_out_path):
        try:
            os.remove(cropped_out_path)
        except Exception:
            pass

    img_cropped.save(cropped_out_path)
    return cropped_out_path
    
# -----------------------------------------------------------------------------
# 8. Generation
# -----------------------------------------------------------------------------
@zero_gpu_decorator(duration=1500)
def generate_video(prompt, e_prompt_temp, image_pil, end_image_pil, quality, ratio, duration, seed, randomize_seed, use_enhancer, steps, progress=gr.Progress(track_tqdm=True)):
    if image_pil is None:
        raise ValueError("A starting first frame image is required.")

    width, height = RESOLUTION_PROFILES[quality][ratio]
    num_frames = snap_frames(duration)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    session_id = str(current_seed)

    def crop_if_image(img, suffix):
        if img is None or not isinstance(img, Image.Image):
            return None
        temp_path = os.path.join(BASE, f"temp_{suffix}_{session_id}.png")
        img.save(temp_path)
        cropped_path = auto_crop_image(temp_path, width, height, session_id=suffix)
        return Image.open(cropped_path).convert("RGB")

    image_pil = crop_if_image(image_pil, "first")
    end_image = crop_if_image(end_image_pil, "end")

    with offload_text_encoder():
        cond_result = conditioner(
            prompt=prompt,
            image=image_pil,
            enhance_prompt=bool(use_enhancer),
            enhance_temperature=float(e_prompt_temp),
            height=height,
            width=width,
            num_frames=num_frames,
        )
        final_prompt = cond_result.prompt if (cond_result and getattr(cond_result, "prompt", None)) else prompt
        prompt_embeds = cond_result.prompt_embeds
        text_token_tags = cond_result.text_token_tags
        del cond_result

    gen_kwargs = {
        "prompt_embeds": prompt_embeds,
        "text_token_tags": text_token_tags,
        "image": image_pil,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "generator": torch.Generator().manual_seed(int(current_seed)),
        "output": ["videos", "audio", "sampling_rate"],
        "num_inference_steps": steps,
    }

    del prompt_embeds, text_token_tags

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    if end_image is not None:
        gen_kwargs["last_image"] = end_image

    out_path = os.path.join(BASE, f"minimax_h3_gradio_output_{session_id}.mp4")

    vae.to("cuda")
    audio_vae.to("cuda")
    try:
        with torch.no_grad():
            gen_result = generator(**gen_kwargs)
    finally:
        vae.to("cpu")
        audio_vae.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    with torch.inference_mode():
        encode_video(
            gen_result["videos"][0],
            FPS,
            out_path,
            gen_result["audio"][0],
            gen_result["sampling_rate"]
        )

    del gen_result
    del gen_kwargs
    
    gc.collect()
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    return out_path, final_prompt

@zero_gpu_decorator(duration=120)
def enhance_prompt_standalone(raw_prompt, e_prompt_temp, image_pil, quality="super", ratio="16:9"):

    try:
        if not raw_prompt and image_pil is None:
            return raw_prompt

        cropped_image = image_pil

        if isinstance(image_pil, Image.Image):
            width, height = RESOLUTION_PROFILES[quality][ratio]
            temp_path = os.path.join(BASE, "temp_enhance_first.png")
            image_pil.save(temp_path)

            cropped_path = auto_crop_image(
                temp_path,
                width,
                height,
                session_id="enhance_first"
            )

            cropped_image = Image.open(cropped_path).convert("RGB")

        with offload_text_encoder():

            cond_result = conditioner(
                prompt=raw_prompt,
                image=cropped_image,
                enhance_prompt=True,
                enhance_temperature=float(e_prompt_temp),
            )

            enhanced = cond_result.prompt or raw_prompt
            del cond_result

        return enhanced

    except Exception:
        raise

    finally:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# 9. UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="MiniMax-H3 FL2VA") as demo:
    gr.Markdown("# MiniMax-H3 FL2VA Generator")
    gr.Markdown("Using **Qwen3-VL + ClipProj v3.1 + 8 step Lightning**")

    with gr.Row():
        with gr.Column():
            
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="They all drink a beer and cheer!",
                    lines=2,
                )
            
            with gr.Row():  
                e_prompt_temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.1, label="Enhance prompt temperature")
                
            with gr.Row():    
                enhance_btn = gr.Button("✨ Enhance Prompt", variant="secondary")    
            
            with gr.Row():
                image_input = gr.Image(
                    label="First Frame Image (Required)",
                    type="pil",
                )
                end_image_input = gr.Image(
                    label="Last Frame Image (Optional)",
                    type="pil",
                )
            
            with gr.Row():
                quality_tier = gr.Dropdown(
                    choices=["super", "ultra", "high", "medium", "low"],
                    value="super",
                    label="Quality / Canvas Tier"
                )
                aspect_ratio = gr.Dropdown(
                    choices=["16:9", "9:16", "1:1"],
                    value="16:9",
                    label="Aspect Ratio"
                )

            with gr.Row():
                duration_slider = gr.Slider(
                    minimum=6,
                    maximum=8.5,
                    step=0.5,
                    value=8.5,
                    label="Duration (seconds)",
                )
            
            with gr.Row():
                seed_input = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                )
            
            with gr.Row():
                randomize_seed = gr.Checkbox(value=True, label="Randomize Seed")

            with gr.Row():
                enhance_toggle = gr.Checkbox(
                    label="Auto-Enhance on Generate",
                    value=False,
                    info="Automatically enhance prompt when clicking Generate"
                )

            with gr.Row():
                steps = gr.Slider(1, 16, value=8, step=1, label="Steps")

            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
    
        with gr.Column():
            
            with gr.Row():
                video_output = gr.Video(label="Generated Video")

    gr.Examples(
        examples=[
            "They all drink a beer and cheer!",
        ],
        inputs=[prompt_input],
        example_labels=[
            "They all drink a beer and cheer!",
        ],
        label="💡 Try these:"
    )

    enhance_btn.click(
        fn=enhance_prompt_standalone,
        inputs=[prompt_input, e_prompt_temp, image_input, quality_tier, aspect_ratio],
        outputs=[prompt_input],
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt_input,
            e_prompt_temp,
            image_input,
            end_image_input, 
            quality_tier, 
            aspect_ratio, 
            duration_slider,
            seed_input,
            randomize_seed,
            enhance_toggle,
            steps
        ],
        outputs=[video_output, prompt_input],
    )

if __name__ == "__main__":
    if IS_HF_SPACES:
        demo.queue().launch(debug=False, share=False)
    else:
        demo.queue().launch(debug=True, share=True)
```
