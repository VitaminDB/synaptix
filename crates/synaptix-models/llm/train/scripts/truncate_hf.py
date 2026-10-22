"""Первые N слоёв HF-модели отдельным каталогом — для сверки градиентов
большой модели с PyTorch на CPU.

    python truncate_hf.py <hf_dir> <out_dir> <N>
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

src, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(src / "config.json"))
t = cfg.get("text_config", cfg)
t["num_hidden_layers"] = n
if "layer_types" in t:
    t["layer_types"] = t["layer_types"][:n]
t["mtp_num_hidden_layers"] = 0
json.dump(cfg, open(out / "config.json", "w"), indent=2)


def keep(name):
    if ".layers." not in name:
        return not name.startswith("mtp.")
    i = int(name.split(".layers.")[1].split(".")[0])
    return i < n and not name.startswith("mtp.")


idx = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
by_file = {}
for k, f in idx.items():
    if keep(k):
        by_file.setdefault(f, []).append(k)
tensors = {}
for f, keys in by_file.items():
    with safe_open(src / f, "pt") as h:
        for k in keys:
            tensors[k] = h.get_tensor(k).contiguous()
save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
for f in src.iterdir():
    if f.suffix in (".json", ".jinja", ".txt") and f.name not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(f, out / f.name)
print(f"готово: {out} ({len(tensors)} тензоров)")
