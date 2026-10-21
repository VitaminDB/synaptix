"""Эталон лосса и градиентов LoRA (transformers + PEFT, F32 на CPU) для
сверки с примером `grad_ref` крейта synaptix-llm-train.

    python grad_ref.py <hf_model_dir> <grad_ref.safetensors>

Упаковка повторяется 4D-маской (каузальная внутри примера) и position_ids.
"""
import sys

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

model_dir, dump = sys.argv[1], sys.argv[2]
d = load_file(dump)
tokens = d["tokens"].long()
labels = d["labels"].long()
pos = d["positions"].long()
seg = d["seg"].long()
T = tokens.numel()

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation="eager")
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
rank = d[next(k for k in d if k.endswith(".lora_A"))].shape[0]
cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets, lora_dropout=0.0, bias="none")
model = get_peft_model(model, cfg)
params = dict(model.named_parameters())
loaded = 0
for k, v in d.items():
    if k.endswith(".lora_A") or k.endswith(".lora_B"):
        mod, kind = k.rsplit(".", 1)
        name = f"base_model.model.{mod}.{kind}.default.weight"
        with torch.no_grad():
            params[name].copy_(v)
        loaded += 1
print(f"загружено матриц LoRA: {loaded}")

i = torch.arange(T)
allowed = (i[None, :] <= i[:, None]) & (seg[None, :] == seg[:, None])
mask = torch.zeros(T, T)
mask[~allowed] = torch.finfo(torch.float32).min
mask = mask[None, None]

out = model(input_ids=tokens[None], attention_mask=mask, position_ids=pos[None])
logits = out.logits[0]
valid = labels >= 0
loss = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
loss.backward()
print(f"loss: torch {loss.item():.6f}  synaptix {d['loss'].item():.6f}")

worst = []
tot_ref = tot_err = 0.0
for k, v in d.items():
    if not (k.endswith(".grad_A") or k.endswith(".grad_B")):
        continue
    mod, kind = k.rsplit(".", 1)
    name = f"base_model.model.{mod}.lora_{kind[-1]}.default.weight"
    ref = params[name].grad
    err = (v - ref).norm().item()
    rel = err / (ref.norm().item() + 1e-12)
    cos = torch.nn.functional.cosine_similarity(v.flatten(), ref.flatten(), dim=0).item()
    tot_ref += ref.norm().item() ** 2
    tot_err += err ** 2
    worst.append((rel, cos, k))
worst.sort(reverse=True)
print(f"общая относительная ошибка градиента: {(tot_err / tot_ref) ** 0.5:.4f}")
print("худшие матрицы (отн. ошибка, косинус):")
for rel, cos, k in worst[:8]:
    print(f"  {rel:.4f}  {cos:.5f}  {k}")
