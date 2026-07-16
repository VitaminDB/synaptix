"""Reference logits for synaptix Whisper decoder (teacher-forced prefix).

Reuses the encoder output from `gen_asr_whisper_encoder.py` setup, builds a
forced transcription prefix, runs the decoder in float32 and dumps the prefix
token ids (as float32 for safe round-trip), the encoder output and the logits.

Run from the synaptix repo root with the reference venv.
"""

import pathlib

import numpy as np
import soundfile as sf
import torch

from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

HF_DIR = "models/whisper-large-v3-turbo-hf"
WAV = "breaking_bad.wav"
OUTPUT_DIR = pathlib.Path("tests/reference_data/asr_whisper")
SR = 16000
N_SAMPLES = 30 * SR


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def load_audio_16k_mono(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
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
    feats = fe(seg, sampling_rate=SR, return_tensors="pt").input_features.to(torch.float32)

    tok = WhisperTokenizer.from_pretrained(HF_DIR)

    def tid(s):
        i = tok.convert_tokens_to_ids(s)
        assert i is not None and i >= 0, f"missing token {s}"
        return i

    prefix = [tid("<|startoftranscript|>"), tid("<|en|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
    print("prefix ids:", prefix)
    dec_ids = torch.tensor([prefix], dtype=torch.long)

    model = WhisperForConditionalGeneration.from_pretrained(HF_DIR, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        enc_out = model.model.encoder(feats).last_hidden_state  # [1,1500,1280]
        logits = model(decoder_input_ids=dec_ids, encoder_outputs=(enc_out,)).logits  # [1,S,vocab]

    print("logits", tuple(logits.shape))
    print("argmax per pos:", logits.argmax(-1).squeeze(0).tolist())

    save_case(
        "whisper_dec",
        {
            "token_ids": torch.tensor(prefix, dtype=torch.float32),
            "encoder_out": enc_out.squeeze(0),  # [1500,1280]
            "logits": logits.squeeze(0),         # [S, vocab]
        },
    )


if __name__ == "__main__":
    main()
