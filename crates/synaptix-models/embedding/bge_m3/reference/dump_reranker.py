#!/usr/bin/env python3
# Reference-дамп BGE-reranker-v2-m3 (HF XLMRobertaForSequenceClassification).
# Логиты релевантности для пар (query, passage) + CLS + input_ids первой пары.
#   .venv/bin/python dump_reranker.py
import os, json
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DIR = os.environ.get("RR_DIR", "storage/hf/bge-reranker-v2-m3")
OUT = os.environ.get("RR_OUT", "tmp/reranker_ref"); os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(DIR)
model = AutoModelForSequenceClassification.from_pretrained(DIR, torch_dtype=torch.float32).eval()

QUERY = "What is the capital of France?"
DOCS = [
    "Paris is the capital and most populous city of France.",
    "The Great Wall of China is over 13,000 miles long.",
    "France is a country in Western Europe; its capital city is Paris.",
    "Bananas are a good source of potassium.",
]
pairs = [(QUERY, d) for d in DOCS]

scores = []
with torch.no_grad():
    for i, (q, p) in enumerate(pairs):
        enc = tok(q, p, return_tensors="pt", truncation=True, max_length=512)
        out = model(**enc, output_hidden_states=True)
        logit = out.logits[0, 0].item()
        scores.append(logit)
        if i == 0:
            np.save(f"{OUT}/ids_0.npy", enc["input_ids"][0].cpu().numpy().astype(np.int64))
            np.save(f"{OUT}/cls_0.npy", out.hidden_states[-1][0, 0].float().cpu().numpy())

np.save(f"{OUT}/scores.npy", np.array(scores, dtype=np.float32))
json.dump({"query": QUERY, "docs": DOCS, "scores": scores}, open(f"{OUT}/rerank.json", "w"),
          ensure_ascii=False, indent=2)
order = sorted(range(len(DOCS)), key=lambda i: -scores[i])
print("scores:", [round(s, 3) for s in scores])
print("ranking (desc):", order)
print("DUMP OK")
