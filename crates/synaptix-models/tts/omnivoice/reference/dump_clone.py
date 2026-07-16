#!/usr/bin/env python3
# Reference-дамп VOICE-CLONE пути OmniVoice для гейтов A (ref-промпт) и B (clone e2e).
#
# PYTHONPATH=Temp/OmniVoice \
#   synaptix/scripts/reference/.venv/bin/python \
#   synaptix/crates/synaptix-models/tts/omnivoice/reference/dump_clone.py
#
# Дампит:
#   clone_ref_tokens.npy (C, T_ref) i64  — create_voice_clone_prompt(...).ref_audio_tokens
#   clone_ref_pre.npy    (N,) f32        — preprocessed ref-wav 24k (после rms+remove_silence+clip)
#   clone_wav.npy        (T,) f32        — детерминированная clone-генерация (codes→decode, БЕЗ post-process)
#   clone_codes.npy      (C, T) i64      — сгенерированные коды (для изоляции от decode)
#   clone_meta.json      {ref_text, ref_rms, text, target_len, num_step, ...}
#
# Детерминизм: position_temperature=0 + class_temperature=0 → чистый argmax/topk.

import os, json
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = os.environ.get("OV_MODEL", "tmp/ov_unpack")
OUT = os.environ.get("OV_OUT", "tmp/ov_ref")
WAV = os.environ.get("OV_ENC_WAV", "extr.wav")
os.makedirs(OUT, exist_ok=True)

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoiceGenerationConfig,
    _combine_text,
)

# ref_text распознан synaptix ASR с extr.wav (используется ДОСЛОВНО).
REF_TEXT = (
    "Я тебе скажу, что реальная тема. Сходи в круглосуточный и купи нам печенье "
    "Джафа. Или шоколадки. Ему шоколадку. А Ириша к вам не купить, чтоб пломбы "
    "повышкакивали, клоуны хуевые! Поберегись!"
)
TEXT_CLONE = "Привет, это тест клонирования голоса."
NUM_STEP = 8
GUIDANCE = 2.0

torch.manual_seed(0)
model = OmniVoice.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()

# ── ЧАСТЬ A: create_voice_clone_prompt + дамп preprocessed wav ────────────────
# Воспроизводим внутренности create_voice_clone_prompt, чтобы заодно сдампить
# preprocessed wav (для изоляции препроцесса от encode в гейте).
from omnivoice.utils.audio import load_audio, remove_silence
from omnivoice.utils.text import add_punctuation

sr = model.sampling_rate  # 24000
ref_wav = load_audio(WAV, sr)  # (1, N)
ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
if 0 < ref_rms < 0.1:
    ref_wav = ref_wav * 0.1 / ref_rms
# ref_text задан → trim НЕ применяется; remove_silence применяется.
ref_wav = remove_silence(ref_wav, sr, mid_sil=200, lead_sil=100, trail_sil=200)
chunk = model.audio_tokenizer.config.hop_length
clip = int(ref_wav.shape[-1] % chunk)
if clip > 0:
    ref_wav = ref_wav[:, :-clip]
pre = ref_wav[0].astype(np.float32).copy()
np.save(f"{OUT}/clone_ref_pre.npy", pre)

# Реальный create_voice_clone_prompt (ref_audio_tokens).
prompt = model.create_voice_clone_prompt(ref_audio=WAV, ref_text=REF_TEXT)
ref_tokens = prompt.ref_audio_tokens.to(torch.int64).cpu().numpy()  # (C, T)
np.save(f"{OUT}/clone_ref_tokens.npy", ref_tokens)
print("ref_rms", ref_rms, "ref_text(+punct)", repr(prompt.ref_text))
print("preprocessed wav", pre.shape, "rms", float(np.sqrt((pre**2).mean())))
print("ref_tokens", ref_tokens.shape, "min", int(ref_tokens.min()), "max", int(ref_tokens.max()))

# ── ЧАСТЬ B: детерминированная clone-генерация (codes→decode) ─────────────────
T_ref = ref_tokens.shape[-1]
target_len = model._estimate_target_tokens(TEXT_CLONE, prompt.ref_text, T_ref, speed=1.0)
print("target_len", target_len)

task = GenerationTask(
    batch_size=1,
    texts=[TEXT_CLONE],
    target_lens=[target_len],
    langs=[None],
    instructs=[None],
    ref_texts=[prompt.ref_text],
    ref_audio_tokens=[prompt.ref_audio_tokens],
    ref_rms=[prompt.ref_rms],
    speed=None,
)
gen = OmniVoiceGenerationConfig(
    num_step=NUM_STEP,
    guidance_scale=GUIDANCE,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=0.0,
    class_temperature=0.0,
    denoise=True,
)

with torch.inference_mode():
    codes = model._generate_iterative(task, gen)[0]  # (C, T)
    codes_np = codes.cpu().numpy().astype(np.int64)
    np.save(f"{OUT}/clone_codes.npy", codes_np)
    # decode БЕЗ post-process (как generate_clone_with_target).
    wav = (
        model.audio_tokenizer.decode(codes.to(model.audio_tokenizer.device).unsqueeze(0))
        .audio_values[0]
        .cpu()
        .numpy()
    )  # (1, T) или (T,)
wav = np.asarray(wav, dtype=np.float32).reshape(-1)
np.save(f"{OUT}/clone_wav.npy", wav)
print("clone_codes", codes_np.shape, "min", int(codes_np.min()), "max", int(codes_np.max()))
print("clone_wav", wav.shape, "rms", float(np.sqrt((wav**2).mean())))

json.dump(
    {
        "ref_text": prompt.ref_text,
        "ref_rms": prompt.ref_rms,
        "text": TEXT_CLONE,
        "target_len": int(target_len),
        "num_step": NUM_STEP,
        "guidance_scale": GUIDANCE,
        "t_shift": 0.1,
        "layer_penalty_factor": 5.0,
        "T_ref": int(T_ref),
    },
    open(f"{OUT}/clone_meta.json", "w"),
    ensure_ascii=False,
    indent=2,
)
print("DUMP OK")
