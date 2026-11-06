"""Эталон Supertonic 3 (HF Supertone/supertonic-3, onnx/*.onnx через onnxruntime, логика
`py/helper.py` из github supertone-inc/supertonic) для сверки с synaptix (пример supertonic_audit).

    python -I supertonic_ref.py --pylib <каталог с onnxruntime> --repo <клон supertonic>/py
        --onnx <снимок>/onnx --voice <снимок>/voice_styles/M1.json --text "..." --lang en
        --out каталог [--steps 8] [--speed 1.05] [--seed 0]

Начальный шум — из нашего ГПСЧ (`synaptix_tts_supertonic::noise`): splitmix64 со счётчиком,
пары u1,u2 = (z>>40 + 0.5)/2^24 (f64), n = sqrt(-2 ln u1)·cos(2π u2), по порядку C [1, 144, L].
Выход: ids.npy, duration.npy, text_emb.npy, x0.npy, x_final.npy, wav.npy (до обрезки), audio.wav.
"""

import argparse
import json
import os
import sys

import numpy as np

GOLD = 0x9E3779B97F4A7C15


def our_normal(seed: int, n: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        idx = np.arange(1, 2 * n + 1, dtype=np.uint64)
        z = np.uint64(seed) + idx * np.uint64(GOLD)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(40)).astype(np.float64) + 0.5) / float(1 << 24)
    u1, u2 = u[0::2], u[1::2]
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylib")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.pylib:
        sys.path.insert(0, a.pylib)
    sys.path.insert(0, a.repo)
    import soundfile as sf
    import helper

    os.makedirs(a.out, exist_ok=True)
    tts = helper.load_text_to_speech(a.onnx, use_gpu=False)
    style = helper.load_voice_style([a.voice])

    dumps = {}
    orig = tts.sample_noisy_latent

    def noisy(duration):
        x, mask = orig(duration)
        x = our_normal(a.seed, x.size).reshape(x.shape) * mask
        dumps["x0"] = x.copy()
        return x, mask

    tts.sample_noisy_latent = noisy
    ids, mask = tts.text_processor([a.text], [a.lang])
    np.save(os.path.join(a.out, "ids.npy"), ids[0])
    dur, *_ = tts.dp_ort.run(None, {"text_ids": ids, "style_dp": style.dp, "text_mask": mask})
    te, *_ = tts.text_enc_ort.run(None, {"text_ids": ids, "style_ttl": style.ttl, "text_mask": mask})
    np.save(os.path.join(a.out, "duration.npy"), dur)
    np.save(os.path.join(a.out, "text_emb.npy"), te[0])
    wav, d = tts._infer([a.text], [a.lang], style, a.steps, a.speed)
    np.save(os.path.join(a.out, "x0.npy"), dumps["x0"][0])
    np.save(os.path.join(a.out, "wav.npy"), wav[0])
    w = wav[0, : int(tts.sample_rate * d[0])]
    sf.write(os.path.join(a.out, "audio.wav"), w, tts.sample_rate, subtype="FLOAT")
    print(f"ids {ids.shape[1]}, duration {d[0]:.3f} s (dp {dur[0]:.3f}), latent {dumps['x0'].shape}, wav {w.size}")


if __name__ == "__main__":
    main()
