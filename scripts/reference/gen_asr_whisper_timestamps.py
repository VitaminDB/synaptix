"""Reference timestamped decode (single 30 s segment) for synaptix Whisper.

HF generate(return_timestamps=True) → dump raw generated ids (incl. timestamp
tokens) + the timestamp_begin id + chunk (start,end,text) list. Used to
replicate the TimestampLogitsProcessor in Rust and verify parity.

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
    audio = case["audio_16k"].numpy()

    fe = WhisperFeatureExtractor.from_pretrained(HF_DIR)
    feats = fe(audio, sampling_rate=SR, return_tensors="pt").input_features.to(torch.float32)
    tok = WhisperTokenizer.from_pretrained(HF_DIR)
    ts_begin = tok.convert_tokens_to_ids("<|0.00|>")

    model = WhisperForConditionalGeneration.from_pretrained(HF_DIR, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            feats,
            language="en",
            task="transcribe",
            num_beams=1,
            do_sample=False,
            return_timestamps=True,
        )
    out = out["sequences"] if isinstance(out, dict) else out
    gen_ids = out.squeeze(0).tolist()
    print("ts gen ids:", gen_ids)
    print("timestamp_begin:", ts_begin)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / "timestamps.json"
    p.write_text(json.dumps({"gen_ids": gen_ids, "timestamp_begin": ts_begin}))
    print(f"  saved {p}")


if __name__ == "__main__":
    main()
