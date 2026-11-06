"""Эталон LFM3 на PyTorch строго по modeling_lfm3.py (eager, без кэша): логиты всех позиций и жадная
генерация (пересчёт с нуля на каждом шаге). Для сверки с synaptix (пример lfm3_gen --ids / --logits).

    python -I lfm3_ref.py --pylib P <dir> "<промпт-ids через запятую>" <токенов> <out.npy>
"""
import json, os, sys
a = sys.argv
if a[1] == "--pylib":
    sys.path.insert(0, a[2]); a = a[:1] + a[3:]
import numpy as np, torch
import torch.nn.functional as F
from safetensors.torch import load_file

d = a[1]; ids = [int(x) for x in a[2].split(",")]; n_new = int(a[3]); out = a[4]
cfg = json.load(open(os.path.join(d, "config.json")))
W = {k: v.float() for k, v in load_file(os.path.join(d, "model.safetensors")).items()}
H, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
eps = cfg["norm_eps"]; lt = cfg["layer_types"]; win = cfg["sliding_window"]
torch.set_num_threads(16)

def rms(x, w):
    v = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(v + eps))

def inv_freq(kind):
    p = cfg["rope_parameters"][kind]; base = p["rope_theta"]; f = p.get("partial_rotary_factor", 1.0)
    ang = int(f * hd) // 2
    i = 1.0 / (base ** (torch.arange(0, 2 * ang, 2, dtype=torch.float) / hd))
    return torch.cat((i, torch.zeros(hd // 2 - ang)))

def rot(x):
    return torch.cat((-x[..., hd // 2:], x[..., :hd // 2]), -1)

def forward(ids):
    T = len(ids)
    x = W["model.embed_tokens.weight"][torch.tensor(ids)][None]
    pos = torch.arange(T).float()
    ropes = {}
    for k in ("full_attention", "sliding_attention"):
        fr = pos[:, None] * inv_freq(k)[None]
        e = torch.cat((fr, fr), -1); ropes[k] = (e.cos(), e.sin())
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        res = x
        h = rms(x, W[p + "operator_norm.weight"])
        if lt[L] == "conv":
            bcx = (h @ W[p + "conv.in_proj.weight"].T).transpose(1, 2)
            B, C, xx = bcx.chunk(3, dim=-2)
            bx = B * xx
            w = W[p + "conv.conv.weight"]  # [H,1,7]
            y = F.conv1d(bx, w, padding=w.shape[-1] - 1, groups=H)[..., :T]
            y = (C * y).transpose(1, 2)
            o = y @ W[p + "conv.out_proj.weight"].T
        else:
            q = (h @ W[p + "self_attn.q_proj.weight"].T).view(1, T, nh, hd)
            k = (h @ W[p + "self_attn.k_proj.weight"].T).view(1, T, nkv, hd)
            v = (h @ W[p + "self_attn.v_proj.weight"].T).view(1, T, nkv, hd).transpose(1, 2)
            q = rms(q, W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = rms(k, W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            cos, sin = ropes[lt[L]]
            q = q * cos + rot(q) * sin; k = k * cos + rot(k) * sin
            g = nh // nkv
            k = k.repeat_interleave(g, 1); v = v.repeat_interleave(g, 1)
            s = (q @ k.transpose(2, 3)) * hd ** -0.5
            i = torch.arange(T)
            m = i[None, :] <= i[:, None]
            if lt[L] == "sliding_attention":
                m = m & ((i[:, None] - i[None, :]) < win)
            s = s.masked_fill(~m, float("-inf"))
            att = (s.softmax(-1) @ v).transpose(1, 2)  # [1,T,nh,hd]
            gate = torch.sigmoid(h @ W[p + "self_attn.gate_proj.weight"].T)[..., None]
            o = (att * gate).reshape(1, T, nh * hd) @ W[p + "self_attn.o_proj.weight"].T
        o = rms(o, W[p + "post_operator_norm.weight"])
        x = res + o
        f = rms(x, W[p + "ffn_norm.weight"])
        f = (F.silu(f @ W[p + "feed_forward.w1.weight"].T) * (f @ W[p + "feed_forward.w3.weight"].T)) @ W[p + "feed_forward.w2.weight"].T
        x = x + rms(f, W[p + "post_ffn_norm.weight"])
    x = rms(x, W["model.embedding_norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

with torch.no_grad():
    lg = forward(ids)[0]
    np.save(out, lg.numpy().astype(np.float32))
    seq = list(ids); gen = []
    for _ in range(n_new):
        t = int(forward(seq)[0, -1].argmax()); gen.append(t); seq.append(t)
    print(gen)
