"""Оценка синтеза речи по набору заданий (формат vv_audit / vibevoice_ref.py):
WER по whisper через службу synaptixd (пример asr_check), «бормотание» — озвученное
время вне распознанных слов (участки ≥ 0,2 с из 20-мс кадров громче −35 дБ от пика вне слов ±0,15 с,
с пиком громче −16 дБ — на уровне речи, а не вдох),
лишние слова, F0 (медиана, yin) против образца голоса в полутонах.

    python -I tts_eval.py задания.tsv каталог [каталог2 …] [--asr путь/asr_check] [--model whisper-large-v3-turbo]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def norm(s):
    return [w for w in (re.sub(r"[^\w']", "", x) for x in s.lower().split()) if w]


def f0_median(wav, sr):
    import librosa

    if len(wav) < sr // 4:
        return float("nan")
    f0 = librosa.yin(wav.astype(np.float32), fmin=60, fmax=500, sr=sr, frame_length=2048)
    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0][: len(f0)]
    keep = rms > rms.max() * 10 ** (-30 / 20)
    return float(np.median(f0[: len(keep)][keep])) if keep.any() else float("nan")


def evaluate(asr, model, wav_path, ref):
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / sr
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    sf.write(tmp, wav, sr, subtype="PCM_16")
    out = subprocess.run([asr, model, tmp, "en", ref], capture_output=True, text=True, timeout=300).stdout
    os.unlink(tmp)
    lines = out.splitlines()
    text = lines[1] if len(lines) > 1 else ""
    words = []
    for l in lines[2:]:
        m = re.match(r"\s+([\d.]+)–\s*([\d.]+)\s+([\d.]+)\s+(.*)", l)
        if m:
            words.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)))
    m = re.search(r"WER ([\d.]+) %", out)
    wer = float(m.group(1)) if m else float("nan")
    hop = int(sr * 0.02)
    fr = np.array([np.sqrt(np.mean(wav[i : i + hop] ** 2) + 1e-12) for i in range(0, len(wav) - hop + 1, hop)])
    loud = fr > fr.max() * 10 ** (-35 / 20)
    cov = np.zeros_like(loud)
    for s, e, _, _ in words:
        cov[max(0, int((s - 0.15) / 0.02)) : int((e + 0.15) / 0.02) + 1] = True
    # Бормотание — громкие (пик > −16 дБ от максимума, на уровне речи) участки
    # ≥ 0,2 с вне слов; тихие вдохи и шорохи между репликами не в счёт.
    bad = loud & ~cov
    mumble, i = 0.0, 0
    while i < len(bad):
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < len(bad) and bad[j]:
            j += 1
        if j - i >= 10 and fr[i:j].max() > fr.max() * 10 ** (-16 / 20):
            mumble += (j - i) * 0.02
        i = j
    extra = max(0, len(norm(text)) - len(norm(ref)))
    tail = dur - (words[-1][1] if words else 0.0)
    return dict(dur=dur, wer=wer, mumble=mumble, extra=extra, tail=tail, text=text, f0=f0_median(wav, sr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--asr", default=os.path.expanduser("~/Storage/Проекты/2027/synshell/target/fast-release/examples/asr_check"))
    ap.add_argument("--model", default="whisper-large-v3-turbo")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    jobs = []
    for l in open(a.jobs, encoding="utf-8").read().splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        name, vs, script = l.split("\t", 2)
        ref = " ".join(re.sub(r"^\s*Speaker\s+\d+\s*:", "", x) for x in script.split("|"))
        jobs.append((name, vs, ref))
    vf0 = {}
    for name, vs, _ in jobs:
        if vs != "-":
            v = vs.split(",")[0]
            if v not in vf0:
                w, sr = sf.read(v, dtype="float32")
                vf0[v] = f0_median(w if w.ndim == 1 else w.mean(axis=1), sr)
    allrows = []
    for d in a.dirs:
        rows = []
        print(f"== {d}")
        for name, vs, ref in jobs:
            p = os.path.join(d, name + ".wav")
            if not os.path.exists(p):
                continue
            r = evaluate(a.asr, a.model, p, ref)
            st = float("nan")
            if vs != "-" and not name.startswith("d"):
                st = 12 * np.log2(r["f0"] / vf0[vs.split(",")[0]])
            r["st"] = st
            rows.append(r)
            print(f"{name:20s} {r['dur']:5.2f}с WER {r['wer']:5.1f}% бормот {r['mumble']:4.2f}с лишних {r['extra']} хвост {r['tail']:4.2f}с F0 {r['f0']:5.0f} Гц ({st:+.1f} пт)")
            if a.v:
                print("    ", r["text"])
        allrows += rows
        if rows:
            n = len(rows)
            bad = sum(1 for r in rows if r["wer"] > 20 or r["mumble"] > 0.5)
            sts = [abs(r["st"]) for r in rows if not np.isnan(r["st"])]
            print(
                f"ИТОГО {n}: WER ср {np.mean([r['wer'] for r in rows]):.1f}%, бормотание ср {np.mean([r['mumble'] for r in rows]):.2f}с, "
                f"брак (WER>20% или бормот>0,5с) {bad}/{n}, |ΔF0| ср {np.mean(sts) if sts else float('nan'):.1f} пт"
            )

    if len(a.dirs) > 1 and allrows:
        n = len(allrows)
        bad = sum(1 for r in allrows if r["wer"] > 20 or r["mumble"] > 0.5)
        sts = [abs(r["st"]) for r in allrows if not np.isnan(r["st"])]
        print(
            f"ВСЕГО {n}: WER ср {np.mean([r['wer'] for r in allrows]):.2f}%, с ошибками {sum(1 for r in allrows if r['wer'] > 0)}/{n}, "
            f"бормотание ср {np.mean([r['mumble'] for r in allrows]):.2f}с, брак {bad}/{n} ({100 * bad / n:.0f}%), "
            f"|ΔF0| ср {np.mean(sts):.1f} пт, длит. ср {np.mean([r['dur'] for r in allrows]):.2f}с"
        )


if __name__ == "__main__":
    main()
