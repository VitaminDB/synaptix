#!/usr/bin/env python3
# Эталон WavLMModel (CPU F32): features/proj/pos/hidden на wav16.npy из tmp/w2v2_ref.
#   scripts/reference/.venv/bin/python <this> models/wavlm-large
# Выход ($W2V_OUT, по умолчанию tmp/wavlm_ref). Заодно пишет model.safetensors.
import os, sys, shutil
import numpy as np, torch
from safetensors.torch import save_file
from transformers import WavLMModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/wavlm-large"
OUT = os.environ.get("W2V_OUT", "tmp/wavlm_ref"); os.makedirs(OUT, exist_ok=True)
st = os.path.join(MODEL, "model.safetensors")
if not os.path.exists(st):
    sd = torch.load(os.path.join(MODEL, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, st)
model = WavLMModel.from_pretrained(MODEL).eval()
wav = np.load("tmp/w2v2_ref/wav16.npy")
np.save(f"{OUT}/wav16.npy", wav)
x = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
caps = {}
def hk(n):
    def f(_m, _i, o):
        caps[n] = np.ascontiguousarray((o[0] if isinstance(o, tuple) else o).detach().numpy())
    return f
model.feature_extractor.register_forward_hook(hk("features"))
model.feature_projection.register_forward_hook(hk("proj"))
model.encoder.pos_conv_embed.register_forward_hook(hk("pos"))
with torch.no_grad():
    out = model(torch.from_numpy(x)[None])
for k in ["features", "proj", "pos"]:
    np.save(f"{OUT}/{k}.npy", caps[k])
np.save(f"{OUT}/hidden.npy", np.ascontiguousarray(out.last_hidden_state.numpy()))
print({k: v.shape for k, v in caps.items()}, out.last_hidden_state.shape)
