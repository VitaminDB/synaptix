#!/usr/bin/env python3
# Детерминированный reference-дамп masked-decode (для гейта генерационного цикла).
# position_temperature=0 + class_temperature=0 → без gumbel, чистый argmax/topk → детерминизм.
#
# PYTHONPATH=~/Temp/OmniVoice synaptix/scripts/reference/.venv/bin/python \
#   synaptix/crates/synaptix-models/tts/omnivoice/reference/dump_generate.py
#
# Дампит: gen_meta.json (текст, target_len, num_step, guidance), gen_codes.npy (8,T).

import os, json
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
MODEL = os.environ.get("OV_MODEL", "tmp/ov_unpack")
OUT = os.environ.get("OV_OUT", "tmp/ov_ref")
os.makedirs(OUT, exist_ok=True)

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig

torch.manual_seed(0)
model = OmniVoice.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()

TEXT = "Hello world."
TARGET = 12
NUM_STEP = 8
GUIDANCE = 2.0

task = GenerationTask(
    batch_size=1, texts=[TEXT], target_lens=[TARGET], langs=[None], instructs=[None],
    ref_texts=[None], ref_audio_tokens=[None], ref_rms=[None], speed=None,
)
gen = OmniVoiceGenerationConfig(
    num_step=NUM_STEP, guidance_scale=GUIDANCE, t_shift=0.1, layer_penalty_factor=5.0,
    position_temperature=0.0, class_temperature=0.0, denoise=True,
)
# Подготовленный вход одного item (для изолированного гейта decode-логики,
# без текст-фронтенда): cond input_ids[1,8,C] + audio_mask[1,C].
prep = model._prepare_inference_inputs(
    text=TEXT, num_target_tokens=TARGET, ref_text=None, ref_audio_tokens=None,
    lang=None, instruct=None, denoise=True,
)
np.save(f"{OUT}/gen_input_ids.npy", prep["input_ids"].cpu().numpy().astype(np.int64))
np.save(f"{OUT}/gen_audio_mask.npy", prep["audio_mask"].cpu().numpy().astype(np.bool_))
print("gen_input_ids", tuple(prep["input_ids"].shape), "gen_audio_mask", tuple(prep["audio_mask"].shape))

with torch.inference_mode():
    codes = model._generate_iterative(task, gen)[0]  # (8, T)
codes_np = codes.cpu().numpy().astype(np.int64)
np.save(f"{OUT}/gen_codes.npy", codes_np)
json.dump(
    {"text": TEXT, "target_len": TARGET, "num_step": NUM_STEP, "guidance_scale": GUIDANCE,
     "t_shift": 0.1, "layer_penalty_factor": 5.0},
    open(f"{OUT}/gen_meta.json", "w"), ensure_ascii=False, indent=2,
)
print("gen_codes", codes_np.shape, "min", int(codes_np.min()), "max", int(codes_np.max()))
print("codes[:, :6] =", codes_np[:, :6].tolist())
print("DUMP OK")
