"""Reference data for synaptix FLUX CLIP-L pooled (bit-exact check on CUDA).

FLUX берёт из CLIP-L (text_encoder/) ТОЛЬКО pooler_output [B,768] —
last_hidden_state(после final_layer_norm, eps=1e-5) в позиции argmax(input_ids)
(= первый EOS-токен 49407). Грузим CLIPTextModel в float32, токенизируем
фиксированный промпт (max_length=77, padding) и дампим input_ids + pooled +
last_hidden_state.

Запуск: scripts/reference/.venv/bin/python scripts/reference/gen_flux_clip.py
"""

import pathlib

import torch
from transformers import CLIPTextModel, CLIPTokenizer

FLUX = "models/black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = pathlib.Path("tests/reference_data/flux_clip")
PROMPT = "a photo of a red apple on a wooden table, cinematic lighting"


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def main():
    tok = CLIPTokenizer.from_pretrained(f"{FLUX}/tokenizer")
    model = CLIPTextModel.from_pretrained(f"{FLUX}/text_encoder", torch_dtype=torch.float32).float()
    model.eval()

    ids = tok(
        PROMPT,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids  # [1,77] int64
    print("input_ids", tuple(ids.shape), "argmax(eos pos)", int(ids.argmax(-1)))

    with torch.no_grad():
        out = model(ids, output_hidden_states=False)
    pooled = out.pooler_output  # [1,768]
    last = out.last_hidden_state  # [1,77,768]
    print("pooled", tuple(pooled.shape), "range", float(pooled.min()), float(pooled.max()))

    save_case("clip_l", {
        "input_ids": ids.to(torch.int32),
        "pooled": pooled,
        "last_hidden_state": last,
    })


if __name__ == "__main__":
    main()
