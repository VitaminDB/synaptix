#!/usr/bin/env python3
# Reference-дамп GigaAM-v3-e2e-CTC: mel/encoder/logits/text на extr.wav.
# PYTHONPATH=~/Temp/GigaAM synaptix/scripts/reference/.venv/bin/python <this>
import os, json, struct
import numpy as np, torch
OUT=os.environ.get("GG_OUT","tmp/gigaam_ref"); os.makedirs(OUT,exist_ok=True)
WAV=os.environ.get("GG_WAV","extr.wav")
ST=os.environ.get("GG_ST","tmp/gigaam_unpack/model.safetensors")
import gigaam
from gigaam.preprocess import load_audio

model = gigaam.load_model("e2e_ctc", fp16_encoder=False, device="cpu")
model.eval()

def load_st(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n)); data=f.read()
    out={}
    for k,v in hdr.items():
        if k=='__metadata__': continue
        s,e=v['data_offsets']
        out[k]=torch.from_numpy(np.frombuffer(data[s:e],dtype=np.float32).reshape(v['shape']).copy())
    return out
sd=load_st(ST)
res=model.load_state_dict(sd, strict=False)
print("load_state_dict: missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys))
print("  sample missing:", res.missing_keys[:5])

caps={}
def hk(name):
    def f(m,i,o): caps[name]=o
    return f
model.preprocessor.register_forward_hook(hk('mel'))
model.encoder.register_forward_hook(hk('enc'))
model.head.register_forward_hook(hk('head'))

wav=load_audio(WAV)  # [T] 16k mono f32
np.save(f"{OUT}/wav16.npy", wav.cpu().numpy().astype(np.float32))
with torch.inference_mode():
    text=model.transcribe(WAV)
print("TEXT:", text)
json.dump({"text": text if isinstance(text,str) else str(text)}, open(f"{OUT}/text.json","w"), ensure_ascii=False)

def dump(name,obj):
    t = obj[0] if isinstance(obj,(tuple,list)) else obj
    a=t.detach().float().cpu().numpy()
    np.save(f"{OUT}/{name}.npy", a); print(f"  {name}", a.shape)
if 'mel' in caps: dump('mel', caps['mel'])
if 'enc' in caps: dump('encoder', caps['enc'])
if 'head' in caps: dump('logits', caps['head'])
print("DUMP OK")
