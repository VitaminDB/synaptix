#!/usr/bin/env python3
"""Эталон Hy4-flash для сверки synaptix: послойный прогон на CPU по исходным BF16-весам.

Формулы — transformers PR #50903 (modeling_hy_v4_flash.py); engram — по vLLM
(eps норм = eps F32, заполнители без памяти, id вне карты — сами в себя), как
в движке. Потоки iHC держатся в F32 (у transformers — BF16 между слоями).
Эксперты читаются срезами только выбранные, слой за слоем: RAM ~ один слой.

  python hy4_ref.py --model _hf/Hy4-flash --ids ids.txt --out dump/ [--layers N]

ids.txt — id токенов через пробел. В out пишется layerNN.f32 (потоки [T,4,H]
после слоя) и logits.f32 (последняя позиция), top-10 печатается.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)


class W:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def f(self, name):
        fn = self.idx[name]
        if fn not in self.h:
            self.h[fn] = safe_open(os.path.join(self.d, fn), framework="pt")
        return self.h[fn]

    def t(self, name):
        return self.f(name).get_tensor(name).float()

    def sl(self, name, i):
        return self.f(name).get_slice(name)[i].float()

    def has(self, name):
        return name in self.idx


def rms(x, w=None, eps=1e-5):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def rope_inter(x, pos, theta, width):
    # x [..., width] — чередующиеся пары; pos [T] (x — [T, ..., width])
    inv = 1.0 / (theta ** (torch.arange(0, width, 2, dtype=torch.float32) / width))
    ang = pos[:, None].float() * inv[None, :]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [width // 2]
    c, s = ang.cos().view(shape), ang.sin().view(shape)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, b * c + a * s), dim=-1).flatten(-2)


STAGE_DIR = None


def stage(l, name, x):
    """Промежуточные стадии слоя 0 (`l0_<name>.f32`) — для поиска расхождений."""
    if STAGE_DIR and l == 0:
        x.detach().numpy().astype(np.float32).tofile(os.path.join(STAGE_DIR, f"l0_{name}.f32"))


def main():
    global STAGE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    STAGE_DIR = a.out
    cfg = json.load(open(os.path.join(a.model, "config.json")))["text_config"]
    eng = json.load(open(os.path.join(a.model, "hy4_engram.json")))
    w = W(a.model)
    ids = [int(x) for x in open(a.ids).read().split()]
    T, H, M = len(ids), cfg["hidden_size"], cfg["hc_mult"]
    eps = cfg["rms_norm_eps"]
    nl = a.layers or cfg["num_hidden_layers"]
    pos = torch.arange(T)
    types = cfg["layer_types"]
    G = cfg["compressed_indexer_compress_ratio"]
    topk = cfg["index_topk"]

    emb = w.f("model.embed_tokens.weight").get_slice("model.embed_tokens.weight")
    x0 = torch.stack([emb[i].float() for i in ids])  # [T, H]
    X = x0[:, None, :].expand(T, M, H).clone()

    lookup = None
    for l in range(nl):
        p = f"model.layers.{l}"
        # ---- engram ----
        if w.has(f"{p}.engram_layer.lookup_table"):
            g = f"{p}.engram_layer"
            lid = l + 1
            if lookup is None:
                lookup = w.f(f"{g}.lookup_table").get_tensor(f"{g}.lookup_table").long()
            mult = [int(v) for v in eng["multipliers"][str(lid)]]
            primes = [v for o in cfg["engram_layer_head_vocab_sizes"][str(lid)] for v in o]
            offs = np.cumsum([0] + primes[:-1]).tolist()
            pad_tok = cfg["engram_pad_id"]
            tmap = lambda t: int(lookup[t]) if t < lookup.shape[0] else t
            pad = tmap(pad_tok)
            nh = len(primes)
            hd = cfg["engram_n_embed_per_ngram"] // cfg["engram_n_head_per_ngram"]
            table = w.f(f"{g}.multi_head_embedding.embedding.weight").get_slice(f"{g}.multi_head_embedding.embedding.weight")
            norms = torch.stack([w.t(f"{g}.lookup_per_head_norms.{h}.weight") for h in range(nh)])
            E = torch.zeros(T, nh, hd)
            live = torch.zeros(T, dtype=torch.bool)
            for t in range(T):
                if ids[t] == pad_tok:
                    continue
                live[t] = True
                roll, blocked, col = 0, False, 0
                for s in range(cfg["engram_max_ngram_size"]):
                    blocked |= s > t
                    src = pad if blocked else tmap(ids[t - s])
                    roll ^= (src * mult[s]) & 0xFFFFFFFFFFFFFFFF
                    if s > 0:
                        for hh in range(cfg["engram_n_head_per_ngram"]):
                            row = roll % primes[col] + offs[col]
                            E[t, col] = table[row].float()
                            col += 1
            E = rms(E, norms, eps=torch.finfo(torch.float32).eps).flatten(1)
            keys = [rms(E @ w.t(f"{g}.key_projs.{c}.weight").T + w.t(f"{g}.key_projs.{c}.bias"), w.t(f"{g}.norm_key.{c}.weight"), torch.finfo(torch.float32).eps) for c in range(M)]
            val = E @ w.t(f"{g}.value_proj.weight").T + w.t(f"{g}.value_proj.bias")
            main_sig = w.t(f"{g}.unified_pre_gamma") * rms(val, eps=torch.finfo(torch.float32).eps)
            for c in range(M):
                q = rms(X[:, c], w.t(f"{g}.norm_query.{c}.weight"), torch.finfo(torch.float32).eps)
                dot = (keys[c] * q).sum(-1) / math.sqrt(H)
                gate = torch.sigmoid(dot.abs().clamp_min(1e-6).sqrt() * dot.sign())
                upd = (gate[:, None] + w.t(f"{g}.unified_per_hc_scales.{c}")) * main_sig
                X[:, c] = X[:, c] + torch.where(live[:, None], upd, torch.zeros_like(upd))

        def hc(prefix):
            fn, base, sc = w.t(f"{prefix}.hc_fn"), w.t(f"{prefix}.hc_base"), w.t(f"{prefix}.hc_scale")
            flat = X.reshape(T, M * H)
            mixes = (flat @ fn.T) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)
            pre = torch.sigmoid(mixes[:, :M] * sc[0] + base[:M]) + cfg["hc_eps"]
            post = cfg["hc_magnitude"] * torch.sigmoid(mixes[:, M:] * sc[1] + base[M:]) + cfg["hc_eps"]
            return (pre[:, :, None] * X).sum(1), post

        # ---- внимание ----
        h, post = hc(f"{p}.hc_attn_layer.hc_pre")
        hn = rms(h, w.t(f"{p}.input_layernorm.weight"), eps)
        stage(l, "x", X)
        stage(l, "attn_hn", hn)
        A = f"{p}.self_attn"
        sliding = types[l] == "sliding_attention"
        nhd = cfg["swa_num_attention_heads"] if sliding else cfg["num_attention_heads"]
        nope, rd, vd, lat = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"], cfg["kv_lora_rank"]
        theta = cfg["rope_parameters"]["sliding_attention" if sliding else "deepseek_sparse_attention"]["rope_theta"]
        gate = hn @ w.t(f"{A}.linear_gate.weight").T
        qr = rms(hn @ w.t(f"{A}.q_a_proj.weight").T, w.t(f"{A}.q_a_layernorm.weight"), eps)
        q = (qr @ w.t(f"{A}.q_b_proj.weight").T).view(T, nhd, nope + rd)
        q = torch.cat((q[..., :nope], rope_inter(q[..., nope:], pos, theta, rd)), -1)
        ckv = hn @ w.t(f"{A}.kv_a_proj_with_mqa.weight").T
        kp = rms(ckv[:, :lat], w.t(f"{A}.kv_a_layernorm.weight"), eps)
        kr = rope_inter(ckv[:, lat:], pos, theta, rd)
        kv = (kp @ w.t(f"{A}.kv_b_proj.weight").T).view(T, nhd, nope + vd)
        k = torch.cat((kv[..., :nope], kr[:, None, :].expand(T, nhd, rd)), -1)
        v = kv[..., nope:]
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(nope + rd)
        i, j = pos[:, None], pos[None, :]
        allow = j <= i
        if sliding:
            allow &= (i - j) < cfg["sliding_window"]
        elif T > topk:
            I = f"{A}.compressed_indexer"
            tok = rms(hn @ w.t(f"{I}.token_key_projection.weight").T, w.t(f"{I}.token_key_norm.weight"), eps)
            ng = T // G
            grp = tok[: ng * G].view(ng, G, -1)
            vals = grp @ w.t(f"{I}.key_value_projection.weight").T
            gts = grp @ w.t(f"{I}.key_gate_projection.weight").T + w.t(f"{I}.intra_group_ape")
            pooled = rms((vals * gts.softmax(1)).sum(1), w.t(f"{I}.key_norm.weight"), eps)
            ch = pooled.shape[-1]
            gk = torch.cat((pooled[:, : ch - rd], rope_inter(pooled[:, ch - rd :], torch.arange(ng) * G, theta, rd)), -1)
            nhi = cfg["index_n_heads"]
            qi = (qr @ w.t(f"{I}.query_projection.weight").T).view(T, nhi, ch)
            qi = torch.cat((qi[..., : ch - rd], rope_inter(qi[..., ch - rd :], pos, theta, rd)), -1)
            hw = (hn @ w.t(f"{I}.head_weight_projection.weight").T) * (nhi ** -0.5 * ch ** -0.5)
            sc = F.relu(torch.einsum("thc,gc->thg", qi, gk))
            lg = torch.einsum("th,thg->tg", hw, sc)
            own = pos // G
            lg = lg.masked_fill(torch.arange(ng)[None, :] >= own[:, None], float("-inf"))
            kk = min(topk // G - 1, ng)
            sel = torch.cat((lg.topk(kk, -1).indices, own[:, None]), -1)
            tokens = (sel[:, :, None] * G + torch.arange(G)).flatten(1).clamp(max=T - 1)
            keep = torch.zeros(T, T, dtype=torch.bool)
            keep.scatter_(1, tokens, True)
            allow &= keep
        scores = scores.masked_fill(~allow[None], float("-inf"))
        sinks = w.t(f"{A}.learnable_sink_param").view(nhd, 1, 1).expand(nhd, T, 1)
        sc = torch.cat((scores, sinks), -1)
        pr = sc.softmax(-1)[..., :-1]
        o = torch.einsum("hts,shd->thd", pr, v).reshape(T, nhd * vd)
        o = o * torch.sigmoid(gate)
        stage(l, "attn_o", o)
        y = rms(o @ w.t(f"{A}.o_proj.weight").T, w.t(f"{A}.post_proj_norm.weight"), eps)
        stage(l, "attn_y", y)
        X = X + post[:, :, None] * y[:, None, :]

        # ---- MLP ----
        h, post = hc(f"{p}.hc_mlp_layer.hc_pre")
        hn = rms(h, w.t(f"{p}.post_attention_layernorm.weight"), eps)
        stage(l, "mlp_hn", hn)
        Mp = f"{p}.mlp"
        if cfg["mlp_layer_types"][l] == "dense":
            z = (F.silu(hn @ w.t(f"{Mp}.gate_proj.weight").T) * (hn @ w.t(f"{Mp}.up_proj.weight").T)) @ w.t(f"{Mp}.down_proj.weight").T
            z = rms(z, w.t(f"{Mp}.post_fc2_norm.weight"), eps)
        else:
            logits = hn @ w.t(f"{Mp}.gate.weight").T
            s = logits.sigmoid()
            choice = s + w.t(f"{Mp}.gate.e_score_correction_bias")
            ti = choice.topk(cfg["num_experts_per_tok"], -1).indices
            tw = s.gather(1, ti)
            tw = tw / (tw.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
            latin = hn @ w.t(f"{Mp}.routed_expert_down_proj.weight").T
            out = torch.zeros_like(latin)
            lim = cfg["swiglu_limit"]
            for e in sorted(set(ti.flatten().tolist())):
                rows, slot = (ti == e).nonzero(as_tuple=True)
                gu = latin[rows] @ w.sl(f"{Mp}.experts.gate_up_proj", e).T
                gg, uu = gu.chunk(2, -1)
                act = F.silu(gg.clamp(max=lim)) * uu.clamp(-lim, lim)
                out.index_add_(0, rows, (act @ w.sl(f"{Mp}.experts.down_proj", e).T) * tw[rows, slot, None])
            routed = out @ w.t(f"{Mp}.routed_expert_up_proj.weight").T
            sh = (F.silu(hn @ w.t(f"{Mp}.shared_experts.gate_proj.weight").T) * (hn @ w.t(f"{Mp}.shared_experts.up_proj.weight").T)) @ w.t(f"{Mp}.shared_experts.down_proj.weight").T
            z = rms(routed + sh, w.t(f"{Mp}.combine_output_norm.weight"), eps)
        stage(l, "mlp_z", z)
        X = X + post[:, :, None] * z[:, None, :]
        X.numpy().astype(np.float32).tofile(os.path.join(a.out, f"layer{l:02d}.f32"))
        print(f"слой {l}: |X| {X.norm().item():.4e}", flush=True)

    if nl == cfg["num_hidden_layers"]:
        fn, base, sc = w.t("model.hc_head.hc_head_fn"), w.t("model.hc_head.hc_head_base"), w.t("model.hc_head.hc_head_scale")
        flat = X[-1:].reshape(1, M * H)
        mixes = (flat @ fn.T) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)
        pre = torch.sigmoid(mixes * sc + base) + cfg["hc_eps"]
        hh = rms((pre[:, :, None] * X[-1:]).sum(1), w.t("model.norm.weight"), eps)
        logits = (hh @ w.t("lm_head.weight").T)[0]
        logits.numpy().astype(np.float32).tofile(os.path.join(a.out, "logits.f32"))
        top = logits.topk(10)
        print("top-10:", list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])))


if __name__ == "__main__":
    main()
