import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("VIBEVOICE_SRC", "/home/master/Storage/VibeVoice"))

from safetensors.torch import save_file

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from dump_components import det_audio


def patch_zero_noise():
    def zeros(*args, **kwargs):
        kwargs.pop("generator", None)
        if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
            shape = tuple(args[0])
        else:
            shape = tuple(a for a in args if isinstance(a, int))
        return torch.zeros(shape, **kwargs)

    def zeros_like(t, *args, **kwargs):
        return torch.zeros_like(t)

    torch.randn = zeros
    torch.randn_like = zeros_like


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--script", default="Speaker 1: The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    patch_zero_noise()
    device = torch.device(args.device)

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation=args.attn
    )
    model.to(device).eval()
    model.set_ddpm_inference_steps(args.steps)
    processor = VibeVoiceProcessor.from_pretrained(args.model)

    if args.voice:
        import wave

        with wave.open(args.voice, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if width == 2:
            wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            wav = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported wav width {width}")
        if ch > 1:
            wav = wav.reshape(-1, ch).mean(axis=1)
        if sr != 24000:
            idx = np.arange(int(len(wav) * 24000 / sr), dtype=np.float64) * sr / 24000.0
            lo = np.floor(idx).astype(np.int64).clip(0, len(wav) - 1)
            hi = np.minimum(lo + 1, len(wav) - 1)
            frac = (idx - lo).astype(np.float32)
            wav = wav[lo] * (1.0 - frac) + wav[hi] * frac
        voice = wav.astype(np.float32)
    else:
        voice = det_audio(3200 * 8)

    inputs = processor(
        text=[args.script],
        voice_samples=[[voice]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()
    }

    with torch.no_grad():
        out = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            cfg_scale=args.cfg_scale,
            max_new_tokens=None,
            generation_config={"do_sample": False},
            verbose=False,
            show_progress_bar=False,
        )

    seq = out.sequences[0].cpu()
    prompt_len = inputs["input_ids"].shape[1]
    generated = seq[prompt_len:]
    audio = out.speech_outputs[0]
    audio = audio.reshape(-1).float().cpu()

    tensors = {
        "voice_raw": torch.from_numpy(np.asarray(voice, dtype=np.float32)),
        "prompt_input_ids": inputs["input_ids"].cpu(),
        "generated_tokens": generated.contiguous(),
        "audio": audio.contiguous(),
    }
    meta = {
        "script": args.script,
        "cfg_scale": args.cfg_scale,
        "ddpm_steps": args.steps,
        "prompt_len": int(prompt_len),
        "generated_len": int(generated.numel()),
        "audio_len": int(audio.numel()),
    }
    save_file(tensors, args.out, metadata={k: str(v) for k, v in meta.items()})
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
