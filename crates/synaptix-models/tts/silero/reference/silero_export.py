"""Перепаковка torch.package Silero TTS v5 в каталог для `synaptix convert`.

    python -I silero_export.py --pylib P package.pt out_dir

Каталог: tts/model.safetensors (+ tts_N для пакетов из нескольких моделей), config.json (символы,
голоса, атрибуты TorchScript-модулей: паддинги свёрток, головы внимания, ISTFT, PQMF, флаги
эмбеддингов типа/фокуса), для русских — accentor/ (n-граммы, исключения, классификаторы) и
homosolver/ (BERT, словарь WordPiece, омографы, фразы). Квантованные эмбеддинги восстанавливаются
как в unpack_q_model: scale · (w − zero_point).
"""
import argparse, gzip, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--pylib")
ap.add_argument("package")
ap.add_argument("out")
a = ap.parse_args()
if a.pylib:
    sys.path.insert(0, a.pylib)
import torch
from safetensors.torch import save_file

imp = torch.package.PackageImporter(a.package)
root = imp.load_pickle("tts_models", "model")
os.makedirs(a.out, exist_ok=True)


def g(o, n, d=None):
    try:
        return getattr(o, n)
    except Exception:
        return d


def sub(o, path):
    for p in path.split('.'):
        o = getattr(o, p)
    return o


def module_attrs(mm):
    """Структурные атрибуты TorchScript-модулей по пути."""
    out = {}
    for name, mod in mm.named_modules():
        on = mod.original_name
        at = {"class": on}
        if on == "Conv1d":
            for k in ("padding", "stride", "dilation", "groups", "kernel_size"):
                v = g(mod, k)
                if v is not None:
                    at[k] = list(v) if isinstance(v, tuple) else v
        elif on == "MultiheadAttention":
            for k in ("num_heads", "head_dim", "embed_dim", "batch_first"):
                at[k] = g(mod, k)
        elif on == "FFTBlock":
            for k in ("use_linear", "pitch_pred"):
                at[k] = g(mod, k)
        elif on in ("HourGlassTransformer",):
            at["shorten_factor"] = g(mod, "shorten_factor")
        elif on == "LinearUpsample":
            at["dim"] = g(mod, "dim")
        elif on == "ISTFT":
            for k in ("n_fft", "hop_length", "win_length", "padding"):
                at[k] = g(mod, k)
        elif on == "PQMF":
            for k in ("N", "taps"):
                at[k] = g(mod, k)
        elif on == "TransformerSeriesPredictor":
            for k in ("utt_type_emb", "focus_emb", "emphasis_emb", "speaker_emb"):
                v = g(mod, k)
                if v is not None:
                    at[k] = v
        elif on in ("JitOldPitchPredictor", "JitPitchPredictorIC", "JitDurPredictor"):
            for k in ("pitch_sil_pred", "pitch_emb_type", "speaker_emb", "log_dur"):
                v = g(mod, k)
                if isinstance(v, (bool, int, float, str)):
                    at[k] = v
        elif on == "LayerNorm":
            at["eps"] = g(mod, "eps")
        if len(at) > 1 or on in ("ISTFT", "PQMF"):
            out[name] = at
    return out


cfg = {"source": os.path.basename(a.package), "packages": []}
for pi, p in enumerate(root.packages):
    if not g(p, "q_model_unpacked", True):
        p.unpack_q_model()  # BERT омографов: scale · (w − zero_point), как при первом apply_tts
        p.q_model_unpacked = True
    pc = {}
    for k in ("symbols", "phons", "sos_token", "eos_token", "phon_sep", "alphabet", "speaker_to_ids", "speaker_to_model",
              "speakers", "symbol_to_id", "ext_alph", "strength2time", "rate2value", "pitch2value", "emb_dim", "emph_pitch", "window"):
        v = g(p, k)
        if v is not None:
            pc[k] = v
    pc["models"] = []
    for mi, mm in enumerate(p.models):
        name = "tts" if (pi, mi) == (0, 0) else f"tts_{pi}_{mi}"
        sd = {k: v.detach().float().contiguous() for k, v in mm.state_dict().items() if v.is_floating_point()}
        wj = g(p, "wrapped_jit_v")
        extra_mods = {}
        if wj is not None:
            # v5/v5_1: ISTFT и PQMF вне TorchScript (WrappedJitV) — та же математика, параметры отсюда
            sd["vocoder.head.istft.window"] = wj.istft.window.detach().float().contiguous()
            extra_mods["vocoder.head.istft"] = {"class": "ISTFT", "n_fft": wj.istft.n_fft, "hop_length": wj.istft.hop_length,
                                                 "win_length": wj.istft.win_length, "padding": wj.istft.padding}
            for n in ("pqmf_6", "pqmf_2"):
                q = getattr(wj, n)
                sd[f"vocoder.{n}.H"] = q.H.detach().float().contiguous()
                extra_mods[f"vocoder.{n}"] = {"class": "PQMF", "N": q.N, "taps": q.taps}
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        save_file(sd, os.path.join(a.out, name, "model.safetensors"))
        mc = {"component": name, "params": sum(v.numel() for v in sd.values())}
        for k in ("punkt_symbols", "point_id", "comma_ids", "mean_std_coef", "sil_value"):
            mc[k] = g(mm, k)
        t = mm.tacotron
        for k in ("pitch_strength", "energy_strength", "pitch_emb_type", "log_dur", "padding_value", "n_mel_channels"):
            mc[k] = g(t, k)
        mc["root_class"] = mm.original_name
        mc["modules"] = module_attrs(mm)
        mc["modules"].update(extra_mods)
        mc["vocoder_outside"] = wj is not None
        # исходники TorchScript корня и предсказателей — для сверки логики
        mc["code_hash"] = hash(mm.code) & 0xffffffff
        with open(os.path.join(a.out, name, "forward.ts.py"), "w") as f:
            f.write(mm.code)
            for n in ("pitch_predictor", "dur_predictor"):
                f.write(f"\n# ---- {n}\n" + sub(mm, n).code)
        pc["models"].append(mc)
    cfg["packages"].append(pc)

    acc = g(p, "accentor")
    if acc is not None and pi == 0:
        ad = os.path.join(a.out, "accentor")
        os.makedirs(ad, exist_ok=True)
        an = acc.accentor
        model = an.model
        w = model.embedding.weight.data
        if w.dtype != torch.float32 or g(model, "scale") is not None:
            # _loadmodel восстанавливает квантованные веса при загрузке; pickled-объект уже восстановлен?
            pass
        sd = {k: v.detach().float().contiguous() for k, v in model.state_dict().items() if v.is_floating_point() and k not in ("scale", "zero_point")}
        save_file(sd, os.path.join(ad, "model.safetensors"))
        ngr = model.embedding.ngram_dict
        inv = sorted(ngr.items(), key=lambda kv: kv[1])
        assert all(i == j for j, (_, i) in enumerate(inv))
        with open(os.path.join(ad, "ngrams.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(k for k, _ in inv) + "\n")
        with open(os.path.join(ad, "exceptions.tsv"), "w", encoding="utf-8") as f:
            for k, (s, y) in sorted(an.exceptions.items()):
                f.write(f"{k}\t{s}\t{y}\n")
        acfg = {"stress_token": an.stress_token, "vowels": an.vowels, "re_cond": an.re_cond,
                "scale": float(model.scale), "zero_point": float(model.zero_point), "emb_dtype": str(w.dtype),
                "stress_clf": str(model.stress_clf.code), "yo_clf": str(model.yo_clf.code)}
        json.dump(acfg, open(os.path.join(ad, "config.json"), "w"), ensure_ascii=False, indent=1)

        hs = acc.homosolver
        hd = os.path.join(a.out, "homosolver")
        os.makedirs(hd, exist_ok=True)
        hm = hs.model
        sd = {k: v.detach().float().contiguous() for k, v in hm.state_dict().items() if v.is_floating_point()}
        save_file(sd, os.path.join(hd, "model.safetensors"))
        tok = hs.tokenizer
        inv = sorted(tok.vocab.items(), key=lambda kv: kv[1])
        with open(os.path.join(hd, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(k for k, _ in inv) + "\n")
        json.dump(hs.homodict, open(os.path.join(hd, "homodict.json"), "w"), ensure_ascii=False)
        # фразы омографов: {слово: [[вариант (з+амок), [фразы с [HOMO] … [/HOMO] по порядку]], …]} —
        # разбор паттерна _load_phrases_dict (варианты по порядку, фразы — длинные первыми)
        grp = re.compile(r"\(\?P<([^>]+)>\(\?<!\[а-яА-ЯёЁ\\-\]\)\(\?:(.*?)\)\(\?!\[а-яА-ЯёЁ\\-\]\)\)(?:\||$)", re.S)
        phrases = {}
        for w, pat in (g(hs, "compiled_phrases", None) or {}).items():
            items = []
            for m in grp.finditer(pat.pattern):
                name, alts = m.group(1), m.group(2)
                parts, cur, i = [], "", 0
                while i < len(alts):
                    ch = alts[i]
                    if ch == "\\" and i + 1 < len(alts):
                        cur += alts[i + 1]
                        i += 2
                        continue
                    if ch == "|":
                        parts.append(cur)
                        cur = ""
                    else:
                        cur += ch
                    i += 1
                parts.append(cur)
                k = next(i for i, c in enumerate(name) if c.isupper())
                items.append([name[:k] + "+" + name[k:].lower(), parts])
            # самопроверка: собранный обратно паттерн совпадает с исходным
            rebuilt = "|".join(rf'(?P<{v[:v.index("+")] + v[v.index("+") + 1].upper() + v[v.index("+") + 2:]}>(?<![а-яА-ЯёЁ\-])(?:{"|".join(map(re.escape, ps))})(?![а-яА-ЯёЁ\-]))' for v, ps in items)
            assert rebuilt == pat.pattern, (w, pat.pattern[:200], rebuilt[:200])
            phrases[w] = items
        json.dump(phrases, open(os.path.join(hd, "phrases.json"), "w"), ensure_ascii=False)
        bert = hm.bert
        hc = {"never_split": sorted(tok.never_split), "tokenize_chinese_chars": tok.tokenize_chinese_chars,
              "unk": tok.unk_token, "sep": tok.sep_token, "pad": tok.pad_token, "cls": tok.cls_token,
              "homo_start_id": tok.homo_start_id, "homo_end_id": tok.homo_end_id, "window_size": g(hs, "window_size"), "homosolver_attrs": sorted(vars(hs)),
              "bert_scale": float(g(bert, "scale", 1.0)), "bert_zero_point": float(g(bert, "zero_point", 0.0)),
              "bert_code": bert.code, "homo_clf": hm.homo_clf.code}
        for name, mod in bert.named_modules():
            if mod.original_name in ("BertSelfAttention",):
                hc["num_attention_heads"] = g(mod, "num_attention_heads")
            if mod.original_name == "LayerNorm":
                hc["layer_norm_eps"] = g(mod, "eps")
        json.dump(hc, open(os.path.join(hd, "config.json"), "w"), ensure_ascii=False, indent=1)

json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), ensure_ascii=False, indent=1)
print("ok", a.out, [m["params"] for pc in cfg["packages"] for m in pc["models"]])
