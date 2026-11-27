"""Дампы частей Seed-VC v2 для сверки порта (CPU, f32, заданные входы).

SEEDVC_REF=…/seed-vc python -I seedvc_parts.py OUT_DIR SRC.wav REF.wav [этапы через запятую]
Этапы: audio, mel, style, lr, dit, cfm, ar (по умолчанию все) и, когда докачаны внешние веса,
hubert, astral, bigvgan. Заданные входы — `wave`/`toks` из seedvc_common (то же считает seedvc_check).
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seedvc_common as C

C.setup()
out, src_path, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
stages = set((sys.argv[4] if len(sys.argv) > 4 else "audio,mel,style,lr,dit,cfm,ar").split(","))
D = lambda n, t: C.dump(out, n, t)


def load_f32(name):
    import glob
    f = glob.glob(os.path.join(out, name + ".*.f32"))[0]
    dims = [int(x) for x in os.path.basename(f).split(".")[-2].split("x")]
    return torch.from_numpy(np.fromfile(f, dtype="<f4").reshape(dims))


if "audio" in stages:
    import librosa
    src = librosa.load(src_path, sr=C.SR)[0]
    ref = librosa.load(ref_path, sr=C.SR)[0][: C.SR * 25]
    D("src_22k", src)
    D("ref_22k", ref)
    D("src_16k", librosa.resample(src, orig_sr=C.SR, target_sr=16000))
    D("ref_16k", librosa.resample(ref, orig_sr=C.SR, target_sr=16000))

if "mel" in stages:
    D("mel_src", C.mel_fn(load_f32("src_22k")[None])[0])
    D("mel_ref", C.mel_fn(load_f32("ref_22k")[None])[0])

if "style" in stages:
    w = load_f32("ref_16k")[None]
    feat = C.fbank(w)
    D("fbank_ref", feat)
    cam = C.build_campplus()
    lens = torch.tensor([feat.size(0)], dtype=torch.int32) // 2
    D("style_ref", cam(feat[None], lens)[0])
    D("style_ref_nolen", cam(feat[None])[0])

if stages & {"lr", "dit", "cfm"}:
    cfm, lr = C.build_cfm()
    style = C.wave([1, 192], 0.5, 0.31, 0.2)

if "lr" in stages:
    D("lr_cond", lr(C.toks(100, 2048, 7919, 13)[None], ylens=torch.LongTensor([172]))[0][0])

if "dit" in stages:
    T = 120
    x = C.wave([1, 80, T], 0.8, 0.11, 0.0)
    prompt = C.wave([1, 80, T], 0.5, 0.07, 1.0)
    prompt[..., 50:] = 0
    cond = C.wave([1, T, 512], 0.3, 0.013, 0.5)
    D("dit_v", cfm.estimator(x, prompt, torch.LongTensor([T]), torch.tensor([0.4]), style, cond)[0])

if "cfm" in stages:
    pc = lr(C.toks(60, 2048, 6151, 5)[None], ylens=torch.LongTensor([103]))[0]
    sc = lr(C.toks(100, 2048, 7919, 13)[None], ylens=torch.LongTensor([172]))[0]
    mu = torch.cat([pc, sc], dim=1)
    T = mu.size(1)
    prompt = C.wave([1, 80, 103], 1.5, 0.05, 0.3) - 4.0
    z = torch.randn([1, 80, T], generator=torch.Generator().manual_seed(1234))
    D("cfm_z", z[0])
    for name, rates, rv in (("cfm_mel_77", [0.7, 0.7], False), ("cfm_mel_70", [0.7, 0.0], False), ("cfm_mel_07", [0.0, 0.7], False),
                            ("cfm_mel_00", [0.0, 0.0], False), ("cfm_mel_anon", [0.7, 0.7], True)):
        n = 6 if name == "cfm_mel_77" else 2
        y = cfm.solve_euler(z.clone(), torch.LongTensor([T]), prompt, mu, style, C.t_span(n), rates, rv)
        D(name, y[0])

if "ar" in stages:
    ar, arlr = C.build_ar()
    V = 2049
    narrow = torch.cat([C.toks(40, 32, 7, 3), C.toks(60, 32, 11, 1)])
    cond = arlr(narrow[None])[0]
    target = C.toks(80, 2048, 6151, 5)[None]
    D("ar_cond", cond[0])
    sep = ar.sep_token_emb.expand(1, 1, -1)
    emb = torch.cat([sep, cond, sep], dim=1)
    pos = torch.cat([torch.arange(cond.size(1) + 1), torch.LongTensor([0])])
    temb = ar.model.embeddings(target)
    emb = torch.cat([emb, temb], dim=1)
    pos = torch.cat([pos, torch.arange(temb.size(1)) + 1])
    kv = torch.arange(emb.size(1))
    logits = ar.model.forward_generate(emb, pos, kv).logits[0, -1]
    steps = [logits.clone()]
    seq = []
    N = int(os.environ.get("SEEDVC_AR_STEPS", "120"))
    for i in range(N):
        l = logits.clone()
        if len(seq) < 11:  # эталон: EOS подавлен на первом токене и пока сгенерировано < 10
            l[V - 1] = -float("inf")
        tok = int(torch.argmax(l))
        if tok == V - 1:
            break
        seq.append(tok)
        pos = pos[-1:] + 1
        kv = kv[-1:] + 1
        logits = ar.model.forward_generate(ar.model.embeddings(torch.tensor([[tok]])), pos, kv).logits[0, -1]
        if i < 4:
            steps.append(logits.clone())
    D("ar_logits", torch.stack(steps))
    D("ar_greedy", torch.tensor(seq))

if stages & {"hubert", "astral"}:
    hub = C.build_hubert()
    f_src = C.hubert_features(hub, load_f32("src_16k")[None])
    f_ref = C.hubert_features(hub, load_f32("ref_16k")[None])
    D("hubert_src", f_src[0])
    D("hubert_ref", f_ref[0])
    # Сверка самой обёртки: тот же путь через Wav2Vec2FeatureExtractor, как в AstralQuantizer.
    from transformers import Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(os.path.join(C.EXT, "hubert-large-ll60k"))
    w = load_f32("src_16k")
    inp = fe([w.numpy()], return_tensors="pt", return_attention_mask=True, padding=True, sampling_rate=16000)
    h2 = hub(inp.input_values, attention_mask=inp.attention_mask).last_hidden_state[:, : f_src.size(1)]
    print("экстрактор HF против ручной нормировки: макс. разность", float((h2 - f_src).abs().max()))

if "astral" in stages:
    for kind, tag in (("bsq2048", "wide"), ("bsq32", "narrow")):
        enc, q = C.build_astral(kind)
        for nm, f in (("src", f_src), ("ref", f_ref)):
            xh, idx = C.astral_tokens(enc, q, f)
            D(f"astral_{tag}_hidden_{nm}", xh[0])
            D(f"tok_{tag}_{nm}", idx[0])

if "bigvgan" in stages:
    voc = C.build_bigvgan()
    mel = load_f32("mel_src")[None]
    D("bigvgan_src", voc(mel).squeeze())
