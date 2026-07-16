#!/usr/bin/env python3
# Reference-дамп декода HiggsAudioV2 (коды → волна 24кГц) для гейта codec-decode.
#
# PYTHONPATH=~/Temp/OmniVoice synaptix/scripts/reference/.venv/bin/python \
#   synaptix/crates/synaptix-models/tts/omnivoice/reference/dump_codec.py
#
# Берёт gen_codes.npy (8,T) → audio_tokenizer.decode → wav_ref.npy (samples,).

import os
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
MODEL = os.environ.get("OV_MODEL", "tmp/ov_unpack")
OUT = os.environ.get("OV_OUT", "tmp/ov_ref")

from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()

codes = np.load(f"{OUT}/gen_codes.npy")  # (8, T) i64
tokens = torch.from_numpy(codes).to(model.audio_tokenizer.device)  # (8,T)
with torch.inference_mode():
    dec = model.audio_tokenizer.decode(tokens.unsqueeze(0))  # (1,8,T) → audio
    wav = dec.audio_values[0].float().cpu().numpy()  # (1, samples) or (samples,)
wav = np.squeeze(wav).astype(np.float32)
np.save(f"{OUT}/wav_ref.npy", wav)
print("codes", codes.shape, "→ wav", wav.shape, "min", float(wav.min()), "max", float(wav.max()),
      "rms", float(np.sqrt((wav**2).mean())))
print("DUMP OK")
