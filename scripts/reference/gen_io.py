"""Generate reference files for Session 8 — IO + Bundle.

Run:
    python scripts/reference/gen_io.py

Outputs data/ref/io/ with:
  - safetensors F32, multi-dtype safetensors with metadata
  - minimal GGUF header stub (F32 tensor, GGUF format v3)
  - WAV sine 440 Hz (PCM-16)
  - PNG RGB test image
"""

import math
import pathlib
import struct
import wave

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/io")


def save_st(name: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=metadata)
    print(f"  saved {path}")


def case_safetensors_f32() -> None:
    torch.manual_seed(0)
    t = torch.randn(8, 64, dtype=torch.float32)
    save_st("safetensors_f32", {"tensor": t})


def case_safetensors_dtypes() -> None:
    torch.manual_seed(1)
    tensors = {
        "f16": torch.randn(4, 32, dtype=torch.float16),
        "bf16": torch.randn(4, 32, dtype=torch.bfloat16),
        "i32": torch.randint(-100, 100, (4, 32), dtype=torch.int32),
        "i64": torch.randint(-1000, 1000, (4, 32), dtype=torch.int64),
        "f32": torch.randn(4, 32, dtype=torch.float32),
    }
    save_st("safetensors_dtypes", tensors)


def case_safetensors_metadata() -> None:
    torch.manual_seed(2)
    t = torch.randn(4, 16, dtype=torch.float32)
    metadata = {
        "model_name": "synaptix-test",
        "version": "1.0",
        "author": "synaptix",
        "custom_key": "hello world",
    }
    save_st("safetensors_metadata", {"data": t}, metadata=metadata)


def _write_gguf_minimal(path: pathlib.Path) -> None:
    """Write a minimal GGUF v3 file with one F32 tensor of shape [4, 4]."""
    tensor_data = np.arange(16, dtype=np.float32)
    tensor_name = b"test.weight"
    tensor_name_len = len(tensor_name)

    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 0))

        f.write(struct.pack("<Q", tensor_name_len))
        f.write(tensor_name)

        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<Q", 4))

        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<Q", 0))

        alignment = 32
        current_pos = f.tell()
        padding = (alignment - current_pos % alignment) % alignment
        f.write(b"\x00" * padding)

        f.write(tensor_data.tobytes())

    print(f"  saved {path}")


def case_gguf_minimal() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "gguf_minimal.gguf"
    _write_gguf_minimal(path)
    ref_tensor = np.arange(16, dtype=np.float32).reshape(4, 4)
    save_st(
        "gguf_minimal_ref",
        {"tensor": torch.from_numpy(ref_tensor)},
    )


def case_wav_sine_440hz() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sr = 16000
    duration_sec = 0.5
    freq = 440.0
    n_samples = int(sr * duration_sec)
    t = np.arange(n_samples, dtype=np.float32) / sr
    audio_f32 = (np.sin(2.0 * math.pi * freq * t) * 0.5).astype(np.float32)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    wav_path = OUTPUT_DIR / "wav_sine_440hz.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    print(f"  saved {wav_path}")
    audio_readback = audio_i16.astype(np.float32) / 32767.0
    save_st(
        "wav_sine_440hz",
        {
            "samples_f32": torch.from_numpy(audio_readback),
            "samples_i16": torch.from_numpy(audio_i16.astype(np.int32)),
        },
    )


def case_wav_round_trip() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    sr = 16000
    n_samples = 8000
    audio_f32 = np.random.randn(n_samples).astype(np.float32) * 0.1
    audio_i16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)
    wav_path = OUTPUT_DIR / "wav_round_trip.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    print(f"  saved {wav_path}")
    audio_expected = audio_i16.astype(np.float32) / 32767.0
    save_st(
        "wav_round_trip_ref",
        {"expected_f32": torch.from_numpy(audio_expected)},
    )


def case_png_rgb_exact() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(1)
    pixels = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    png_path = OUTPUT_DIR / "png_rgb_exact.png"
    img.save(str(png_path))
    print(f"  saved {png_path}")
    save_st(
        "png_rgb_exact_ref",
        {"pixels": torch.from_numpy(pixels.astype(np.int32))},
    )


def case_png_round_trip() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(2)
    pixels = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    save_st(
        "png_round_trip_ref",
        {"pixels": torch.from_numpy(pixels.astype(np.int32))},
    )


def main() -> None:
    print("Generating IO reference data...")
    case_safetensors_f32()
    case_safetensors_dtypes()
    case_safetensors_metadata()
    case_gguf_minimal()
    case_wav_sine_440hz()
    case_wav_round_trip()
    case_png_rgb_exact()
    case_png_round_trip()
    print("Done.")


if __name__ == "__main__":
    main()
