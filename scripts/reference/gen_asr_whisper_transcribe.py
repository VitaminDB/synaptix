"""Reference greedy transcription (single 30 s segment) for synaptix Whisper.

Reuses the exact audio_16k from whisper_enc.safetensors (np.interp-resampled
first 30 s of breaking_bad.wav) so the Rust comparison isolates the decode loop
from resampling. Dumps generated content ids + decoded text to
tests/reference_data/asr_whisper/transcribe.json.

Run from the synaptix repo root with the reference venv.
"""

import json
import pathlib

import torch
from safetensors.torch import load_file

from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

HF_DIR = "models/whisper-large-v3-turbo-hf"
OUTPUT_DIR = pathlib.Path("tests/reference_data/asr_whisper")
SR = 16000


def main():
    case = load_file(str(OUTPUT_DIR / "whisper_enc.safetensors"))
    audio = case["audio_16k"].numpy()  # [480000], exact same samples the Rust test uses

    fe = WhisperFeatureExtractor.from_pretrained(HF_DIR)
    feats = fe(audio, sampling_rate=SR, return_tensors="pt").input_features.to(torch.float32)

    tok = WhisperTokenizer.from_pretrained(HF_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(HF_DIR, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        gen = model.generate(
            feats,
            language="en",
            task="transcribe",
            num_beams=1,
            do_sample=False,
            return_timestamps=False,
        )
    gen_ids = gen.squeeze(0).tolist()
    print("generated ids:", gen_ids)

    # Контентные BPE-токены Whisper — это id < eot (50257); все спец-токены
    # (eot/sot/lang/task/notimestamps/timestamps) идут >= 50257.
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    content = [t for t in gen_ids if t < eot]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    print("text:", text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "transcribe.json"
    out.write_text(json.dumps({"content_ids": content, "text": text}, ensure_ascii=False))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
