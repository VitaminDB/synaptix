"""Generate reference SafeTensors for Session 3 — Embedding layers.

Run:
    python scripts/reference/gen_embed.py

Covers: token_embedding, patch_embed_2d, log_mel_spectrogram (librosa), timestep_embedding.
Outputs data/ref/embed/<case>.safetensors.
"""

import math
import pathlib

import librosa
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/embed")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_token_embedding() -> None:
    torch.manual_seed(0)
    vocab_size, embed_dim = 32000, 256
    emb = nn.Embedding(vocab_size, embed_dim)
    input_ids = torch.randint(0, vocab_size, (4, 32), dtype=torch.int64)
    out = emb(input_ids)
    save_case(
        "token_embedding",
        {
            "weight": emb.weight.data,
            "input_ids": input_ids,
            "output": out.detach(),
        },
    )


def case_patch_embed_2d() -> None:
    torch.manual_seed(1)
    img_size, patch_size, in_channels, embed_dim = 64, 8, 3, 128
    num_patches = (img_size // patch_size) ** 2
    x = torch.randn(2, in_channels, img_size, img_size, dtype=torch.float32)
    proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    out = proj(x).flatten(2).transpose(1, 2)
    save_case(
        "patch_embed_2d",
        {
            "input": x,
            "weight": proj.weight.data,
            "bias": proj.bias.data,
            "output": out.detach(),
        },
    )


def case_log_mel_spectrogram() -> None:
    np.random.seed(2)
    sr = 16000
    n_mels = 80
    n_fft = 400
    hop_length = 160
    duration_sec = 1.0
    audio = np.random.randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr // 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)
    audio_t = torch.from_numpy(audio)
    log_mel_t = torch.from_numpy(log_mel)
    save_case("log_mel_spectrogram", {"audio": audio_t, "output": log_mel_t})


def case_timestep_embedding() -> None:
    dim = 256
    max_period = 10000
    timesteps = torch.tensor([0, 100, 500, 999, 250, 750], dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32)
        / half
    )
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    save_case("timestep_embedding", {"timesteps": timesteps, "output": emb})


def main() -> None:
    print("Generating embedding reference data...")
    case_token_embedding()
    case_patch_embed_2d()
    case_log_mel_spectrogram()
    case_timestep_embedding()
    print("Done.")


if __name__ == "__main__":
    main()
