#!/usr/bin/env python3
# Reference-дамп ENCODE-пути нейро-кодека HiggsAudioV2 (ref-аудио → коды [8,T])
# для гейта voice-cloning encode.
#
# PYTHONPATH=Temp/OmniVoice \
#   synaptix/scripts/reference/.venv/bin/python \
#   synaptix/crates/synaptix-models/tts/omnivoice/reference/dump_encoder.py
#
# Вход: extr.wav → mono → resample→24000 → клип до кратного hop_length
# (как create_voice_clone_prompt, без silence removal для детерминизма).
# Сохраняет:
#   enc_input.npy  (N,) f32 24kHz mono  — ИДЕНТИЧНЫЙ вход для synaptix (без ресэмпл-расхождения)
#   enc_codes.npy  (8,T) i64            — model.audio_tokenizer.encode(...).audio_codes[0]
# Опц. (debug, при OV_DUMP_STAGES=1): enc_semfeat.npy, enc_acoustic.npy, enc_embeddings.npy.

import os
import numpy as np
import torch
import torchaudio

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = os.environ.get("OV_MODEL", "tmp/ov_unpack")
OUT = os.environ.get("OV_OUT", "tmp/ov_ref")
WAV = os.environ.get("OV_ENC_WAV", "extr.wav")
DUMP_STAGES = os.environ.get("OV_DUMP_STAGES", "0") == "1"

os.makedirs(OUT, exist_ok=True)

from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()
tok = model.audio_tokenizer
sr_target = tok.config.sample_rate  # 24000

# --- загрузка + mono + resample → 24000 (детерминизм: torchaudio kaiser/hann sinc) ---
import soundfile as sf
data, sr = sf.read(WAV, dtype="float32", always_2d=True)  # (N, C)
wav = torch.from_numpy(data.T.copy())  # [C, N]
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
if sr != sr_target:
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sr_target)
wav = wav.to(torch.float32)  # [1, N]

# Клип до кратного hop_length (как create_voice_clone_prompt).
chunk = tok.config.hop_length  # 960
clip = int(wav.shape[-1] % chunk)
if clip > 0:
    wav = wav[:, :-clip]

# Ограничим длину для скорости гейта (~6s достаточно для покрытия всех 8 квантизаторов).
max_s = float(os.environ.get("OV_ENC_MAX_S", "6.0"))
max_n = int(max_s * sr_target)
max_n -= max_n % chunk
if wav.shape[-1] > max_n:
    wav = wav[:, :max_n]

mono = wav[0].contiguous().cpu().numpy().astype(np.float32)  # (N,)
np.save(f"{OUT}/enc_input.npy", mono)

inp = wav.unsqueeze(0)  # [1, 1, N]
with torch.inference_mode():
    out = tok.encode(inp)  # bandwidth=None → target_bandwidths[-1]
    codes = out.audio_codes[0].to(torch.int64).cpu().numpy()  # (n_q, T)

np.save(f"{OUT}/enc_codes.npy", codes)

print("input wav 24k:", mono.shape, "rms", float(np.sqrt((mono**2).mean())))
print("codes:", codes.shape, "dtype", codes.dtype,
      "min", int(codes.min()), "max", int(codes.max()))

if DUMP_STAGES:
    with torch.inference_mode():
        e_sem_in = tok._extract_semantic_features(inp).detach()  # [1, T_s, 768]
        e_sem = tok.encoder_semantic(e_sem_in.transpose(1, 2))   # [1, 1024, T]
        if tok._get_conv1d_output_lengths(inp.shape[2], tok.acoustic_encoder) != e_sem.shape[2]:
            e_ac = tok.acoustic_encoder(
                torch.nn.functional.pad(inp, (tok.pad, tok.pad))
            )
        else:
            e_ac = tok.acoustic_encoder(inp)
        emb = torch.cat([e_ac, e_sem], dim=1)
        emb = tok.fc(emb.transpose(1, 2)).transpose(1, 2)
    # .contiguous() обязателен: транспонированные view'ы numpy сохранит как C-order,
    # но логический layout надо зафиксировать row-major до save.
    np.save(f"{OUT}/enc_semfeat.npy", e_sem_in[0].contiguous().cpu().numpy().astype(np.float32))    # (T_s,768)
    np.save(f"{OUT}/enc_e_semantic.npy", e_sem[0].contiguous().cpu().numpy().astype(np.float32))    # (768,T)
    np.save(f"{OUT}/enc_e_acoustic.npy", e_ac[0].contiguous().cpu().numpy().astype(np.float32))     # (256,T)
    np.save(f"{OUT}/enc_embeddings.npy", emb[0].contiguous().cpu().numpy().astype(np.float32))      # (1024,T)
    print("stages: semfeat", tuple(e_sem_in[0].shape), "e_sem", tuple(e_sem[0].shape),
          "e_ac", tuple(e_ac[0].shape), "emb", tuple(emb[0].shape))

print("DUMP OK")
