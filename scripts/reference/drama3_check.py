#!/usr/bin/env python3
"""Сверка порта Fish Audio Drama 3 с эталоном по дампу `drama3_run` (D3_DUMP).

    python -I drama3_check.py ДАМП КАТАЛОГ_HF КАТАЛОГ_sglang-omni

Кодек — эталонный код (`fishaudio_s2_pro/fish_speech/models/dac`) с весами
`codec.pth`: коды образца и волна. Вход основы и башня кодбуков — формулы
эталона (`audio_decoder.py`, `decoder_layers.py`; сам модуль требует sglang),
f32, исходные веса; на вход башне — состояния основы из дампа. Основа с
оригиналом здесь не сверяется.
"""
import importlib.util, json, math, os, sys, types
import numpy as np, torch
from safetensors import safe_open

dump, hf, ref = sys.argv[1:4]
torch.set_grad_enabled(False)
SEM0, SEM1, CB, H = 248099, 252194, 4096, 2048

def f32(name, *shape):
    return torch.from_numpy(np.fromfile(f"{dump}/{name}", dtype="<f4").copy()).reshape(*shape)

def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))

# ---------- кодек ----------
def codec_check():
    base = f"{ref}/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac"
    pkg = "d3ref"
    sys.modules[pkg] = types.ModuleType(pkg); sys.modules[pkg].__path__ = [base]
    def load(n):
        src = open(f"{base}/{n}.py").read().replace("sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac", pkg)
        spec = importlib.util.spec_from_loader(f"{pkg}.{n}", loader=None)
        m = importlib.util.module_from_spec(spec); m.__file__ = f"{base}/{n}.py"; sys.modules[f"{pkg}.{n}"] = m
        exec(compile(src, m.__file__, "exec"), m.__dict__); return m
    md = load("modded_dac"); rvq = load("rvq")
    tcfg = lambda: md.ModelArgs(block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072, n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
    wl = lambda: md.WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=tcfg())
    q = rvq.DownsampleResidualVectorQuantize(input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.5, downsample_factor=[2, 2], post_module=wl(), pre_module=wl(), semantic_codebook_size=4096)
    import functools
    gen = functools.partial(md.ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
    m = md.DAC(sample_rate=44100, encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=1536, decoder_rates=[8, 8, 4, 2], encoder_transformer_layers=[0, 0, 0, 4], decoder_transformer_layers=[4, 0, 0, 0], transformer_general_config=gen, quantizer=q)
    sd = torch.load(f"{hf}/codec.pth", map_location="cpu", weights_only=True); sd = sd.get("state_dict", sd)
    miss = m.load_state_dict(sd, strict=False)
    print(f"кодек: ключей {len(sd)}, не найдено {len(miss.missing_keys)}, лишних {len(miss.unexpected_keys)}")
    m.eval()
    pcm = torch.from_numpy(np.fromfile(f"{dump}/ref_pcm.f32", dtype="<f4").copy())
    mine = torch.from_numpy(np.fromfile(f"{dump}/ref_codes.i32", dtype="<i4").copy()).reshape(10, -1).long()
    n = pcm.numel(); pad = (-n) % 2048
    x = torch.nn.functional.pad(pcm, (0, pad))[None]
    codes, _ = m.encode(x, torch.tensor([x.shape[-1]]))
    codes = codes[0][:, : mine.shape[1]]
    eq = (codes == mine).float()
    print(f"кодек, коды образца {tuple(mine.shape)}: совпало {eq.mean()*100:.2f} % (семантический {eq[0].mean()*100:.2f} %, остаточные {eq[1:].mean()*100:.2f} %)")
    wav = m.from_indices(mine[None].clone()).flatten()
    my = torch.from_numpy(np.fromfile(f"{dump}/ref_decoded.f32", dtype="<f4").copy())
    k = min(wav.numel(), my.numel())
    d = wav[:k] - my[:k]
    print(f"кодек, декод тех же кодов: {k} семплов, макс. разность {d.abs().max():.2e}, SNR {10*math.log10(float(wav[:k].pow(2).sum()/(d.pow(2).sum()+1e-30))):.1f} дБ")

# ---------- веса LM/башни ----------
idx = json.load(open(f"{hf}/model.safetensors.index.json"))["weight_map"]
_open = {}
def W(name):
    f = idx[name]
    if f not in _open: _open[f] = safe_open(f"{hf}/{f}", "pt")
    return _open[f].get_tensor(name).float()

def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)

def tower_check():
    rows = torch.from_numpy(np.fromfile(f"{dump}/rows.i32", dtype="<i4").copy()).reshape(-1, 11).long()
    L = rows.shape[0]
    sem = (rows[:, 0] >= SEM0) & (rows[:, 0] <= SEM1)
    cb = W("audio_decoder.codebook_embeddings.weight")
    emb = W("model.language_model.embed_tokens.weight")
    # вход основы
    x = emb[rows[:, 0]]
    mixed = (x + cb[CB + rows[:, 2].clamp(0, CB - 1)]) / math.sqrt(2.0)
    x = torch.where(sem[:, None], mixed, x)
    mine = f32("embeds.f32", L, H)
    print(f"вход основы [{L}, {H}] (аудиострок {int(sem.sum())}): cos {cos(x, mine):.6f}, макс. разность {(x-mine).abs().max():.3e}")
    # башня на префилле
    h = f32("normed.f32", L, H)[: L - 1]
    inp, nxt = rows[: L - 1], rows[1:]
    n = L - 1
    lab_sem = sem[1:]
    extra = torch.zeros(n, H)
    for k in range(2, 10):
        extra += cb[k * CB + inp[:, 1 + k].clamp(0, CB - 1)]
    extra = torch.where(sem[: L - 1, None], extra, torch.zeros(()))
    inv = 1.0 / 1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    ang = torch.arange(n, dtype=torch.float32)[:, None] * inv
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    def rope(t):  # [n, heads, 256]
        r = t[..., :64]; ev, od = r[..., 0::2], r[..., 1::2]
        rot = torch.stack((ev * c - od * s, od * c + ev * s), -1).flatten(-2)
        return torch.cat((rot, t[..., 64:]), -1)
    mask = torch.full((n, n), float("-inf")).triu(1)
    names = [f"audio_decoder_mtp_shared.layers.{i}" for i in range(4)] + [f"audio_decoder_mtp.layers.{i}" for i in range(9)]
    x = h
    worst = 1.0
    for li, p in enumerate(names):
        t = f"{p}.transformer_layer"
        col = 1 if li < 4 else 1 + (li - 4)
        table = W(f"{p}.{'semantic_embedding' if li < 4 else 'codebook_embedding'}.weight")
        e = torch.where(lab_sem[:, None], table[nxt[:, col].clamp(0, CB - 1)], torch.zeros(()))
        if li == 0: e = e + extra
        mixed = torch.cat((rms(e, W(f"{p}.enorm.weight")), rms(x, W(f"{p}.hnorm.weight"))), -1) @ W(f"{p}.eh_proj.weight").T
        qkv = rms(mixed, W(f"{t}.attention_norm.weight")) @ W(f"{t}.attention.wqkv.weight").T
        q, g, k, v = qkv.split([4096, 4096, 512, 512], -1)
        q = rope(rms(q.view(n, 16, 256), W(f"{t}.attention.q_norm.weight"))).transpose(0, 1)
        k = rope(rms(k.view(n, 2, 256), W(f"{t}.attention.k_norm.weight"))).transpose(0, 1).repeat_interleave(8, 0)
        v = v.view(n, 2, 256).transpose(0, 1).repeat_interleave(8, 0)
        att = torch.softmax(q @ k.transpose(1, 2) / 16.0 + mask, -1) @ v
        a = (att.transpose(0, 1).reshape(n, 4096) * torch.sigmoid(g)) @ W(f"{t}.attention.wo.weight").T
        r1 = mixed + a
        f = rms(r1, W(f"{t}.ffn_norm.weight"))
        r2 = r1 + (torch.nn.functional.silu(f @ W(f"{t}.feed_forward.w1.weight").T) * (f @ W(f"{t}.feed_forward.w3.weight").T)) @ W(f"{t}.feed_forward.w2.weight").T
        x = rms(r2, W(f"{p}.final_layernorm.weight"))
        mine = f32(f"tower_{li:02d}.f32", n, H)
        cc = cos(x, mine); worst = min(worst, cc)
        line = f"башня, слой {li:2d}: cos {cc:.6f}, макс. разность {(x-mine).abs().max():.3e}"
        if li >= 4:
            r = li - 4
            lg = x @ W(f"audio_decoder_mtp.heads.{r}.weight").T
            ml = f32(f"head_{r}.f32", n, CB)
            line += f"; голова {r}: cos {cos(lg, ml):.6f}, argmax совпал {float((lg.argmax(-1)==ml.argmax(-1)).float().mean())*100:.1f} %"
            # насколько голова угадывает следующий код образца (teacher forcing)
            rowsel = lab_sem
            if r + 2 <= 10 and rowsel.any():
                tgt = nxt[:, 2 + r][rowsel]
                line += f", top-1 по кодам образца {float((lg[rowsel].argmax(-1)==tgt).float().mean())*100:.1f} %"
        print(line)
        x = mine  # дальше — от состояний порта, чтобы ошибка не копилась
    print(f"башня: худший cos {worst:.6f}")

if os.path.exists(f"{dump}/ref_codes.i32"):
    codec_check()
tower_check()
