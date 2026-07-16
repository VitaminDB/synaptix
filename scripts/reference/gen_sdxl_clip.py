"""Reference data for synaptix SDXL CLIP text encoders (bit-exact check).

Loads both SDXL text encoders in float32 (from the fp16 weights, upcast to
match synaptix which loads fp16 -> f32) and dumps, for a fixed prompt:

  * input_ids               [1, 77]   (i32, fed verbatim to the Rust side so
                                        tokenisation is not a variable)
  * penultimate             [1, 77, H] hidden_states[-2] (pre-final-LN); this
                                        is what SDXL feeds the UNet
  * last_hidden_state       [1, 77, H] after final LayerNorm
  * pooled                  [1, H]     EOT-pooled (CLIP-L: raw; bigG: projected
                                        text_embeds -> SDXL add_text_embeds)

CLIP-L  = text_encoder    (CLIPTextModel,               quick_gelu, 12L/768)
bigG    = text_encoder_2  (CLIPTextModelWithProjection, gelu,       32L/1280)

Run from the synaptix repo root with the reference venv.
"""

import pathlib

import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

SDXL = "models/stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = pathlib.Path("tests/reference_data/sdxl_clip")
PROMPT = "a photograph of an astronaut riding a horse"
MAX_LEN = 77


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def tokenize(tok_dir):
    tok = CLIPTokenizer.from_pretrained(tok_dir)
    enc = tok(
        PROMPT,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )
    return enc.input_ids  # [1, 77] int64


def run_clip_l():
    ids = tokenize(f"{SDXL}/tokenizer")
    model = CLIPTextModel.from_pretrained(
        f"{SDXL}/text_encoder", torch_dtype=torch.float16, variant="fp16"
    ).float()
    model.eval()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    penultimate = out.hidden_states[-2]  # [1,77,768] pre-final-LN
    save_case(
        "clip_l",
        {
            "input_ids": ids.to(torch.int32),
            "penultimate": penultimate,
            "last_hidden_state": out.last_hidden_state,
            "pooled": out.pooler_output,  # [1,768] eot, no projection
        },
    )
    print("  clip_l penultimate", tuple(penultimate.shape))


def run_clip_bigg():
    ids = tokenize(f"{SDXL}/tokenizer_2")
    model = CLIPTextModelWithProjection.from_pretrained(
        f"{SDXL}/text_encoder_2", torch_dtype=torch.float16, variant="fp16"
    ).float()
    model.eval()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    penultimate = out.hidden_states[-2]  # [1,77,1280] pre-final-LN
    save_case(
        "clip_bigg",
        {
            "input_ids": ids.to(torch.int32),
            "penultimate": penultimate,
            "last_hidden_state": out.last_hidden_state,
            "pooled": out.text_embeds,  # [1,1280] projected -> add_text_embeds
        },
    )
    print("  clip_bigg penultimate", tuple(penultimate.shape))


def main():
    print("CLIP-L (text_encoder):")
    run_clip_l()
    print("bigG (text_encoder_2):")
    run_clip_bigg()


if __name__ == "__main__":
    main()
