"""Эталон Leokani-tts (upstream leokani.py, torch fp32, 1 поток): по фонемам Kokoro или по тексту (G2P Leokani).

    python -I leokani_ref.py --model <снимок HF>[/pono] --code <снимок HF>/leokani.py (--phonemes "..." | --text "...") --out a.f32 [--speed 1]
Пишет сырые f32 (little-endian) — PCM 24 кГц до перевода в int16.
"""
import argparse, importlib.util, sys

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--phonemes")
ap.add_argument("--text")
ap.add_argument("--out", required=True)
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("leokani", a.code)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import torch
torch.set_num_threads(1)
tts = m.Leokani(a.model, g2p="leokani")
audio = tts.synthesize(a.text, a.phonemes, a.speed)
audio.float().numpy().astype("<f4").tofile(a.out)
print(f"{audio.shape[0]} семплов")
