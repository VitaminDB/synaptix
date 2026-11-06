"""Эталон LFM3 на PyTorch строго по modeling_lfm3.py (eager, без кэша): логиты всех позиций и жадная
генерация (пересчёт с нуля на каждом шаге). Для сверки с synaptix (пример lfm3_gen --ids / --logits).

    python -I lfm3_ref.py --pylib P <dir> "<промпт-ids через запятую>" <токенов> <out.npy>

Калибровка квантования (E[x²] входов каждой матрицы, формат synaptix-calib-v1 для `synaptix quantize --calib`):

    python -I lfm3_ref.py --pylib P <dir> --calib <файл: строки ids через запятую> <out.calib>
"""
import json, os, sys
a = sys.argv
if a[1] == "--pylib":
    sys.path.insert(0, a[2]); a = a[:1] + a[3:]
import numpy as np, torch
import torch.nn.functional as F
from safetensors.torch import load_file

d = a[1]
CALIB = a[2] == "--calib"
if not CALIB:
    ids = [int(x) for x in a[2].split(",")]; n_new = int(a[3]); out = a[4]
cfg = json.load(open(os.path.join(d, "config.json")))
W = {k: v.float() for k, v in load_file(os.path.join(d, "model.safetensors")).items()}
H, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
eps = cfg["norm_eps"]; lt = cfg["layer_types"]; win = cfg["sliding_window"]
torch.set_num_threads(16)

REC = {}  # имя веса → [сумма x², строк]

def lin(x, n):
    """x @ Wᵀ; в режиме калибровки копит E[x²] входа по столбцам."""
    if CALIB:
        f = x.reshape(-1, x.shape[-1]).double()
        e = REC.setdefault(n, [torch.zeros(x.shape[-1], dtype=torch.float64), 0])
        e[0] += (f * f).sum(0); e[1] += f.shape[0]
    return x @ W[n].T

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
            bcx = lin(h, p + "conv.in_proj.weight").transpose(1, 2)
            B, C, xx = bcx.chunk(3, dim=-2)
            bx = B * xx
            w = W[p + "conv.conv.weight"]  # [H,1,7]
            y = F.conv1d(bx, w, padding=w.shape[-1] - 1, groups=H)[..., :T]
            y = (C * y).transpose(1, 2)
            o = lin(y, p + "conv.out_proj.weight")
        else:
            q = lin(h, p + "self_attn.q_proj.weight").view(1, T, nh, hd)
            k = lin(h, p + "self_attn.k_proj.weight").view(1, T, nkv, hd)
            v = lin(h, p + "self_attn.v_proj.weight").view(1, T, nkv, hd).transpose(1, 2)
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
            gate = torch.sigmoid(lin(h, p + "self_attn.gate_proj.weight"))[..., None]
            o = lin((att * gate).reshape(1, T, nh * hd), p + "self_attn.o_proj.weight")
        o = rms(o, W[p + "post_operator_norm.weight"])
        x = res + o
        f = rms(x, W[p + "ffn_norm.weight"])
        f = lin(F.silu(lin(f, p + "feed_forward.w1.weight")) * lin(f, p + "feed_forward.w3.weight"), p + "feed_forward.w2.weight")
        x = x + rms(f, W[p + "post_ffn_norm.weight"])
    x = rms(x, W["model.embedding_norm.weight"])
    return lin(x, "model.embed_tokens.weight")

def save_calib(path):
    hdr = {}; body = b""
    for n, (sq, rows) in sorted(REC.items()):
        v = (sq / max(rows, 1)).float().numpy().astype("<f4").tobytes()
        hdr[n] = {"dtype": "F32", "shape": [len(v) // 4], "data_offsets": [len(body), len(body) + len(v)]}
        body += v
    hdr["__metadata__"] = {"format": "synaptix-calib-v1"}
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as fo:
        fo.write(len(h).to_bytes(8, "little")); fo.write(h); fo.write(body)

if CALIB:
    with torch.no_grad():
        for line in open(a[3]):
            seq = [int(x) for x in line.strip().split(",") if x]
            for s0 in range(0, len(seq), 512):
                forward(seq[s0:s0 + 512])
    save_calib(a[4])
    print("калибровка:", len(REC), "матриц")
    sys.exit(0)

with torch.no_grad():
    lg = forward(ids)[0]
    np.save(out, lg.numpy().astype(np.float32))
    seq = list(ids); gen = []
    for _ in range(n_new):
        t = int(forward(seq)[0, -1].argmax()); gen.append(t); seq.append(t)
    print(gen)
