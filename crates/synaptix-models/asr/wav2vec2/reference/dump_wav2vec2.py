#!/usr/bin/env python3
# Reference-дамп Wav2Vec2ForCTC (HF transformers, CPU F32) для гейта synaptix.
# Также конвертирует pytorch_model.bin → model.safetensors рядом с моделью.
#   scripts/reference/.venv/bin/python <this> [model_dir] [wav]
# Выход ($W2V_OUT, по умолчанию tmp/w2v2_ref): wav16.npy, features.npy [1,512,T],
# proj.npy [1,T,D], pos.npy [1,T,D], hidden.npy [1,T,D], logits.npy [1,T,V], text.json.
import os, sys, json
import numpy as np, torch, soundfile as sf
from safetensors.torch import save_file
from transformers import Wav2Vec2ForCTC

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/w2v2-xlsr53-espeak"
WAV = sys.argv[2] if len(sys.argv) > 2 else "tmp/w2v2_ref/input.wav"
OUT = os.environ.get("W2V_OUT", "tmp/w2v2_ref"); os.makedirs(OUT, exist_ok=True)

st_path = os.path.join(MODEL, "model.safetensors")
if not os.path.exists(st_path):
    sd = torch.load(os.path.join(MODEL, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, st_path)
    print("wrote", st_path, len(sd), "tensors")

model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()

audio, sr = sf.read(WAV, dtype="float32")
if audio.ndim > 1: audio = audio.mean(1)
assert sr == 16000, sr
np.save(f"{OUT}/wav16.npy", audio)
x = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
inp = torch.from_numpy(x)[None]

caps = {}
def hk(name):
    def f(_m, _i, o):
        caps[name] = np.ascontiguousarray((o[0] if isinstance(o, tuple) else o).detach().numpy())
    return f
w = model.wav2vec2
w.feature_extractor.register_forward_hook(hk("features"))
w.feature_projection.register_forward_hook(hk("proj"))
w.encoder.pos_conv_embed.register_forward_hook(hk("pos"))
with torch.no_grad():
    out = model(inp)
np.save(f"{OUT}/features.npy", caps["features"])
np.save(f"{OUT}/proj.npy", caps["proj"])
np.save(f"{OUT}/pos.npy", caps["pos"])
with torch.no_grad():
    last = w(inp).last_hidden_state
np.save(f"{OUT}/hidden.npy", np.ascontiguousarray(last.numpy()))
np.save(f"{OUT}/logits.npy", out.logits.numpy())

vocab = json.load(open(os.path.join(MODEL, "vocab.json")))
inv = {v: k for k, v in vocab.items()}
ids = out.logits.argmax(-1)[0].tolist()
seq, prev = [], None
for i in ids:
    if i != prev and i != 0: seq.append(inv[i])
    prev = i
json.dump({"phonemes": seq}, open(f"{OUT}/text.json", "w"), ensure_ascii=False)
print("features", caps["features"].shape, "logits", out.logits.shape)
print(" ".join(seq))
