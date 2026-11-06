"""Каталог для `synaptix convert` из снимка HF Audio8-TTS: модель (model.safetensors) — главный
компонент, кодек (codec.pth, torch-pickle) → codec/model.safetensors (F32, имена как в upstream,
weight-norm не сворачивается). Остальное — файлы бандла.

    python -I stage_bundle.py <снимок HF> <каталог-стадия>
"""

import os
import shutil
import sys

import torch
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(dst, "codec"), exist_ok=True)
for name in ["model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "processor_config.json",
             "preprocessor_config.json", "README.md", "LICENSE", "NOTICE"]:
    p = os.path.join(src, name)
    if os.path.exists(p):
        q = os.path.join(dst, name)
        if os.path.lexists(q):
            os.remove(q)
        os.symlink(os.path.realpath(p), q)
state = torch.load(os.path.join(src, "codec.pth"), map_location="cpu", weights_only=True)
if "state_dict" in state:
    state = state["state_dict"]
if any("generator." in k for k in state):
    state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
state = {k: v.contiguous() for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}
save_file(state, os.path.join(dst, "codec", "model.safetensors"))
print(f"codec: {len(state)} тензоров")
