"""Эталон Silero TTS v5 (оригинальный torch.package) для сверки с synaptix (пример silero_audit).

    python -I silero_ref.py --pylib P package.pt --speaker xenia --text "..." --out DIR [--sr 48000]
        [--no-accent] [--texts-file F]  (F — строки текста: только фронтенд, accented.txt по строкам)

Выход: accented.txt (текст после ударений/омографов), seq.npy, type_ids.npy, focus.npy (если есть),
dur.npy (длительности после всех правок корня), pitch.npy (высота тона, вход такотрона), mel.npy, audio.npy.
"""
import argparse, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pylib")
ap.add_argument("package")
ap.add_argument("--speaker", default="xenia")
ap.add_argument("--text")
ap.add_argument("--texts-file")
ap.add_argument("--out")
ap.add_argument("--sr", type=int, default=48000)
ap.add_argument("--no-accent", action="store_true")
a = ap.parse_args()
if a.pylib:
    sys.path.insert(0, a.pylib)
import torch
torch.set_num_threads(4)
torch.set_grad_enabled(False)
imp = torch.package.PackageImporter(a.package)
root = imp.load_pickle("tts_models", "model")
p = root.packages[0]
if not p.q_model_unpacked:
    p.unpack_q_model()
    p.q_model_unpacked = True

if a.texts_file:
    os.makedirs(a.out, exist_ok=True)
    lines = [l.rstrip("\n") for l in open(a.texts_file, encoding="utf-8")]
    out = []
    for l in lines:
        out.append(p.accentor(l) if p.accentor is not None else l)
    open(os.path.join(a.out, "accented_lines.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    sys.exit(0)

os.makedirs(a.out, exist_ok=True)
acc_log = []
if p.accentor is not None:
    orig_acc = p.accentor
    class AccProxy:
        def __call__(self, *args, **kw):
            r = orig_acc(*args, **kw)
            acc_log.append(r)
            return r
        def __getattr__(self, n):
            return getattr(orig_acc, n)
    p.accentor = AccProxy()

calls = []
mm = p.models[0]
class Proxy:
    def __call__(self, **kw):
        calls.append(kw)
        return mm(**kw)
    def __getattr__(self, n):
        return getattr(mm, n)
p.models[0] = Proxy()
kw = dict(text=a.text, speaker=a.speaker, sample_rate=a.sr)
if a.no_accent:
    kw.update(put_accent=False, put_yo=False, put_stress_homo=False, put_yo_homo=False)
audio = root.apply_tts(**kw)
np.save(os.path.join(a.out, "audio.npy"), audio.numpy().astype(np.float32))
open(os.path.join(a.out, "accented.txt"), "w", encoding="utf-8").write((acc_log[0] if acc_log else "") + "\n")
k = calls[-1]
seq = k["sequence"]
np.save(os.path.join(a.out, "seq.npy"), seq.numpy().astype(np.int64))
if k.get("type_ids") is not None:
    np.save(os.path.join(a.out, "type_ids.npy"), k["type_ids"].numpy().astype(np.int64))
if k.get("focus_mask") is not None:
    np.save(os.path.join(a.out, "focus.npy"), k["focus_mask"].numpy().astype(np.int64))
# повтор корня по частям: длительности из корня, высота тона — тем же путём, что в forward
args = {n: k[n] for n in ("sequence", "speaker_ids", "sr", "symb_durs", "durs_rate", "pitch_coefs", "device") if n in k}
for n in ("type_ids", "focus_mask", "gt_durs", "gt_pitch"):
    if k.get(n) is not None:
        args[n] = k[n]
_, dur = mm(**args)
np.save(os.path.join(a.out, "dur.npy"), dur.numpy().astype(np.float32))
from_tacotron = {}
def hook_tac(*inp):
    pass
orig_mask = (seq == 0)
pp = mm.pitch_predictor
if "type_ids" in args:
    if mm.pitch_predictor.original_name == "JitPitchPredictorIC" and "emphasis" in str(pp.code):
        pitch = pp(seq, k["speaker_ids"], orig_mask, args.get("type_ids"), args.get("focus_mask"), None)
    else:
        pitch = pp(seq, k["speaker_ids"], orig_mask, args.get("type_ids"), args.get("focus_mask"))
else:
    pitch = pp(seq, k["speaker_ids"], orig_mask, 1.0)
np.save(os.path.join(a.out, "pitch_raw.npy"), pitch.numpy().astype(np.float32))
if k.get("pitch_coefs") is not None:
    pitch = mm.update_pitch_coef(pitch, k["pitch_coefs"][0].unsqueeze(0), k["speaker_ids"])
# обнуление высоты на пунктуации в корне не срабатывает: cumsum идёт по оси батча [1, T+1],
# срез durs_cum[:-1] пуст, цикл — ноль итераций. Повторяем как есть (ничего не обнуляем).
d = dur[0]
np.save(os.path.join(a.out, "pitch.npy"), pitch.numpy().astype(np.float32))
mel = mm.tacotron(seq, k["speaker_ids"], orig_mask, dur, pitch, None)
np.save(os.path.join(a.out, "mel.npy"), np.ascontiguousarray(mel.numpy().astype(np.float32)))
if not getattr(p, "wrapped_jit_v", None) and not k.get("symb_durs"):
    v = mm.vocoder(mel, a.sr, 0., True)[0]
    n = min(len(v), len(audio))
    print("сверка частей с цельным forward: max |Δ| звука", float((v[:n] - audio[:n]).abs().max()), len(v), len(audio))
print("seq", seq.shape[1], "кадров", int(d.sum()), "звук", audio.shape[0] / a.sr, "с;", (acc_log[0] if acc_log else ""))
