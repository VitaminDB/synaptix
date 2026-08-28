"""Эталон Qwen4Exp (Qwen3.8-Flash-Next) для теста паритета.

Требует transformers с поддержкой qwen4_exp (main-ветка). Собирает маленькую
модель со случайными весами, прогоняет её и кладёт рядом веса, конфиг, токены и
эталонные тензоры: логиты, hidden по слоям, выход QSA-слоя и маску индексатора.

Запуск:
    python scripts/reference/gen_qwen4_exp_ref.py <каталог> [число токенов]

Проверка:
    SYN_QWEN4EXP_REF=<каталог> cargo test -p synaptix-llm-qwen4-exp --test parity_hf
"""

import json
import math
import os
import sys

import torch
from safetensors.torch import save_file
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    apply_rotary_pos_emb,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/ref/qwen4_exp"
TOKENS = [1, 17, 42, 7, 2, 99, 13, 64, 5, 88, 31, 6, 77, 21, 3, 55, 12, 90, 44, 8]
if len(sys.argv) > 2:
    TOKENS = TOKENS[: int(sys.argv[2])]

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

cfg = Qwen4ExpTextConfig(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=64,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    rms_norm_eps=1e-6,
    max_position_embeddings=512,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    hc_count=4,
    hc_lowrank=16,
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=97,
    make_ngram_vocab_size_divisible_by=128,
    seed=1234,
    split_ngram_parts=4,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    output_gate_type="sigmoid",
    hidden_act="silu",
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    attention_bias=False,
)
cfg._attn_implementation = "eager"

model = Qwen4ExpForCausalLM(cfg)
model.eval()
with torch.no_grad():
    for p in model.parameters():
        if p.dtype.is_floating_point:
            p.normal_(0.0, 0.05)

layer_out = {}
idx_out = {}


def mk_layer_hook(i):
    def hook(mod, args, kwargs, output):
        layer_out[i] = output[0] if isinstance(output, tuple) else output

    return hook


def idx_hook(mod, args, kwargs, output):
    idx_out["mask"] = output.clone()
    h = kwargs.get("hidden_states")
    idx_out["h"] = (h if h is not None else args[0]).clone()


for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(mk_layer_hook(i), with_kwargs=True)
qsa_layer = next(i for i, t in enumerate(cfg.layer_types) if t != "linear_attention")
model.model.layers[qsa_layer].self_attn.indexer.register_forward_hook(idx_hook, with_kwargs=True)

attn_out = {}


def attn_hook(mod, args, kwargs, output):
    attn_out["qsa"] = output[0] if isinstance(output, tuple) else output


model.model.layers[qsa_layer].self_attn.register_forward_hook(attn_hook, with_kwargs=True)

ids = torch.tensor([TOKENS], dtype=torch.long)
with torch.no_grad():
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True)

ref = {"logits": out.logits[0].float().clone().contiguous()}
for i, h in enumerate(out.hidden_states):
    ref[f"hidden_{i}"] = h[0].float().clone().contiguous()
for i, h in layer_out.items():
    ref[f"layer_out_{i}"] = h[0].float().clone().contiguous()
for k, v in attn_out.items():
    ref[k] = v[0].float().clone().contiguous()
ref["index_mask"] = idx_out["mask"][0, 0].float().clone().contiguous()

state = {k: v.to(torch.float32).contiguous() for k, v in model.state_dict().items()}
save_file(state, os.path.join(OUT, "model.safetensors"))
save_file(ref, os.path.join(OUT, "reference.safetensors"))

cfg_dict = cfg.to_dict()
cfg_dict["model_type"] = "qwen4_exp_text"
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(cfg_dict, f, indent=1)
with open(os.path.join(OUT, "tokens.json"), "w") as f:
    json.dump({"tokens": TOKENS}, f)

idx = model.model.layers[qsa_layer].self_attn.indexer
h = idx_out["h"]
T = h.shape[1]
with torch.no_grad():
    positions = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = model.model.rotary_emb(h, positions)
    nh, dh, cr = idx.index_n_heads, idx.index_head_dim, idx.compress_ratio
    qk = idx.index_qk_proj(h)
    q, token_k = torch.split(qk, [nh * dh, idx.index_kv_heads * dh], dim=-1)
    q = idx.q_layernorm(q.reshape(1, T, nh, dh))
    q = apply_rotary_pos_emb(q, cos=cos[:, -T:, :], sin=sin[:, -T:, :], unsqueeze_dim=2)
    raw = token_k.reshape(1, T, dh)
    blocks = torch.arange((T // cr) * cr).view(T // cr, cr)
    groups = raw[0].index_select(0, blocks.flatten()).view(T // cr, cr, dh)
    pooled = idx.k_layernorm(groups.float().mean(dim=1).to(raw.dtype))
    starts = blocks[:, 0]
    block_keys = apply_rotary_pos_emb(
        pooled.unsqueeze(1),
        cos=cos[0].index_select(0, starts),
        sin=sin[0].index_select(0, starts),
    ).squeeze(1)
    scores = {}
    for qi in range(T):
        nb = (qi + 1) // cr
        if nb == 0:
            continue
        sc = torch.matmul(q[0, qi].float(), block_keys[:nb].float().transpose(-1, -2)).transpose(-1, -2)
        scores[qi] = torch.relu(sc).sum(dim=-1) / math.sqrt(dh)

print(f"каталог: {OUT}")
print(f"токенов: {len(TOKENS)}, слой QSA: {qsa_layer}, блоков: {T // cr}")
print(f"логиты: {tuple(ref['logits'].shape)}")
for qi, sc in scores.items():
    print(f"  скоры блоков q={qi}: {[round(float(x), 5) for x in sc]}")
