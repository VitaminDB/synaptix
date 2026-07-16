"""Reference data for synaptix Whisper encoder + mel front-end.

Loads `breaking_bad.wav` (first 30 s), produces the HF log-mel `input_features`
and runs the Whisper audio encoder in float32. Dumps audio_16k, input_features
and encoder_out so the Rust side can verify (a) mel front-end against
`input_features` and (b) encoder against `encoder_out` independently.

Run from the synaptix repo root with the reference venv.
"""

import pathlib

import numpy as np
import soundfile as sf
import torch

from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

HF_DIR = "models/whisper-large-v3-turbo-hf"
WAV = "breaking_bad.wav"
OUTPUT_DIR = pathlib.Path("tests/reference_data/asr_whisper")
SR = 16000
N_SAMPLES = 30 * SR  # 480000


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def load_audio_16k_mono(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # downmix to mono
    if sr != SR:
        # linear resample (matches synaptix-audio simple resampler closely enough
        # for the front-end; encoder test feeds HF input_features directly anyway)
        import math

        n_out = int(math.floor(len(audio) * SR / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


def main():
    audio = load_audio_16k_mono(WAV)
    seg = audio[:N_SAMPLES]
    if len(seg) < N_SAMPLES:
        seg = np.pad(seg, (0, N_SAMPLES - len(seg)))

    fe = WhisperFeatureExtractor.from_pretrained(HF_DIR)
    feats = fe(seg, sampling_rate=SR, return_tensors="pt").input_features  # [1,128,3000]
    feats = feats.to(torch.float32)
    print("input_features", tuple(feats.shape), feats.dtype)

    model = WhisperForConditionalGeneration.from_pretrained(HF_DIR, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        enc_out = model.model.encoder(feats).last_hidden_state  # [1,1500,1280]
    print("encoder_out", tuple(enc_out.shape), enc_out.dtype)

    save_case(
        "whisper_enc",
        {
            "audio_16k": torch.from_numpy(seg),
            "input_features": feats.squeeze(0),  # [128,3000]
            "encoder_out": enc_out.squeeze(0),    # [1500,1280]
        },
    )


if __name__ == "__main__":
    main()
