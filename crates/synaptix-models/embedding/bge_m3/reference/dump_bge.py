#!/usr/bin/env python3
# Reference-дамп BGE-M3 (XLM-RoBERTa dense-эмбеддинг = L2-norm(last_hidden[:,0])).
# synaptix/scripts/reference/.venv/bin/python <this>
import os, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
MODEL=os.environ.get("BGE_MODEL","tmp/bge_unpack")
OUT=os.environ.get("BGE_OUT","tmp/bge_ref"); os.makedirs(OUT,exist_ok=True)
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
text=os.environ.get("BGE_TEXT","Привет, это тест эмбеддинга BGE-M3.")
enc=tok(text, return_tensors="pt", truncation=True, max_length=512)
np.save(f"{OUT}/input_ids.npy", enc["input_ids"].numpy().astype(np.int64))
np.save(f"{OUT}/attention_mask.npy", enc["attention_mask"].numpy().astype(np.int64))
with torch.no_grad():
    h=model(**enc).last_hidden_state  # [1,S,1024]
cls=h[:,0]
dense=torch.nn.functional.normalize(cls, p=2, dim=-1)
np.save(f"{OUT}/last_hidden.npy", h.float().numpy())
np.save(f"{OUT}/dense_ref.npy", dense.float().numpy())
print("text=",text)
print("input_ids", tuple(enc["input_ids"].shape), "ids[:12]=", enc["input_ids"][0,:12].tolist())
print("last_hidden", tuple(h.shape), "dense", tuple(dense.shape), "dense[:5]=", dense[0,:5].tolist())
print("DUMP OK")
