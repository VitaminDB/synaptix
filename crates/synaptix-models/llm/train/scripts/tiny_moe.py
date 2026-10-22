"""Крошечная Qwen3.5-MoE со случайными весами — в раскладке настоящей
Qwen3.6-35B-A3B (башня зрения, `model.language_model.*`, стопки экспертов) —
для сверки градиентов MoE-обучения с PyTorch.

    python tiny_moe.py <config.json настоящей модели> <каталог токенизатора> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

src_cfg, tok_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.load(open(src_cfg))
t = cfg["text_config"]
t.update(
    hidden_size=256, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=256,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=256,
    num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64, shared_expert_intermediate_size=64,
    mtp_num_hidden_layers=0,
)
v = cfg["vision_config"]
v.update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256)
config = Qwen3_5MoeConfig.from_dict(cfg)
torch.manual_seed(0)
model = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.05)
        elif "norm" in n and "weight" in n:
            p.normal_(0.0, 0.1)
# Стопки экспертов как в настоящем чекпоинте (`save_pretrained` разложил бы
# их по экспертам).
out.mkdir(parents=True, exist_ok=True)
config.save_pretrained(out)
from safetensors.torch import save_file
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors", metadata={"format": "pt"})
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"]:
    if (tok_dir / f).exists():
        shutil.copy(tok_dir / f, out / f)
print("готово:", out)
