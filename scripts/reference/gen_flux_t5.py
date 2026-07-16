"""Reference data for synaptix FLUX T5-XXL encoder (bit-exact check on CUDA).

FLUX text_encoder_2 = google/t5-v1_1-xxl (T5EncoderModel, только encoder).
last_hidden_state [B,S,4096]. Грузим в float32, токенизируем фиксированный промпт
(max_length=SEQ, padding, БЕЗ attention_mask на forward — FLUX не маскирует) и
дампим input_ids + last_hidden_state.

SEQ=128 (вместо прод-512) — быстрее, но покрывает весь диапазон relative-bias
(max_distance=128). T5-forward не зависит от длины (relative bias на любую S).

Запуск: scripts/reference/.venv/bin/python scripts/reference/gen_flux_t5.py
"""

import pathlib

import torch
from transformers import T5EncoderModel, T5TokenizerFast

FLUX = "models/black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = pathlib.Path("tests/reference_data/flux_t5")
PROMPT = "a photo of a red apple on a wooden table, cinematic lighting"
SEQ = 128


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def main():
    tok = T5TokenizerFast.from_pretrained(f"{FLUX}/tokenizer_2")
    model = T5EncoderModel.from_pretrained(f"{FLUX}/text_encoder_2", torch_dtype=torch.float32).float()
    model.eval()
    print("d_model", model.config.d_model, "layers", model.config.num_layers,
          "heads", model.config.num_heads, "buckets", model.config.relative_attention_num_buckets)

    ids = tok(
        PROMPT,
        padding="max_length",
        max_length=SEQ,
        truncation=True,
        return_tensors="pt",
    ).input_ids  # [1,SEQ] int64
    print("input_ids", tuple(ids.shape), "first ids", ids[0, :12].tolist())

    with torch.no_grad():
        # FLUX НЕ передаёт attention_mask в T5 (padding-токены участвуют в attn).
        out = model(input_ids=ids, output_hidden_states=False)
    last = out.last_hidden_state  # [1,SEQ,4096]
    print("last_hidden_state", tuple(last.shape), "range", float(last.min()), float(last.max()))

    save_case("t5", {"input_ids": ids.to(torch.int32), "last_hidden_state": last})


if __name__ == "__main__":
    main()
