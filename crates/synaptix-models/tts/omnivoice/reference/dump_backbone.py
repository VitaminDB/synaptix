#!/usr/bin/env python3
# Reference-дамп для bit-exact гейта backbone OmniVoice (PyTorch upstream → synaptix).
#
# Запуск (reference-venv + upstream на PYTHONPATH):
#   PYTHONPATH=~/Temp/OmniVoice \
#   synaptix/scripts/reference/.venv/bin/python \
#   synaptix/crates/synaptix-models/tts/omnivoice/reference/dump_backbone.py
#
# Пути через env (дефолты — распакованный omnivoice.syn в Storage/tmp):
#   OV_MODEL=tmp/ov_unpack   OV_OUT=tmp/ov_ref
#
# Дампит: specials.json, input_ids.npy (1,8,S), audio_mask.npy (1,S),
#         audio_logits.npy (1,8,S,1025) — выход одного bidirectional forward.

import os, json
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
MODEL = os.environ.get("OV_MODEL", "tmp/ov_unpack")
OUT = os.environ.get("OV_OUT", "tmp/ov_ref")
os.makedirs(OUT, exist_ok=True)

from omnivoice import OmniVoice

torch.manual_seed(0)
model = OmniVoice.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()
tok = model.text_tokenizer

specials = {
    t: tok.convert_tokens_to_ids(t)
    for t in ["<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
              "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]
}
json.dump(specials, open(f"{OUT}/specials.json", "w"), ensure_ascii=False, indent=2)

inp = model._prepare_inference_inputs(
    text="Hello world.", num_target_tokens=10, ref_text=None,
    ref_audio_tokens=None, lang=None, instruct=None, denoise=True,
)
input_ids, audio_mask = inp["input_ids"], inp["audio_mask"]
S = input_ids.size(2)
np.save(f"{OUT}/input_ids.npy", input_ids.cpu().numpy().astype(np.int64))
np.save(f"{OUT}/audio_mask.npy", audio_mask.cpu().numpy().astype(np.bool_))

attn = torch.ones(1, 1, S, S, dtype=torch.bool)
with torch.inference_mode():
    logits = model.forward(input_ids=input_ids, audio_mask=audio_mask,
                           attention_mask=attn).logits.float().cpu().numpy()
np.save(f"{OUT}/audio_logits.npy", logits)
print(f"specials={specials}")
print(f"input_ids={tuple(input_ids.shape)} audio_mask={tuple(audio_mask.shape)} "
      f"audio_logits={logits.shape} mean={float(logits.mean()):.4f}")
print("DUMP OK")
