#!/usr/bin/env python3
# Эталон градиентов CTC-дообучения Wav2Vec2ForCTC (CPU F32) для гейта synaptix.
# Обучаются два верхних слоя, финальная норма и lm_head; разметка — greedy-фонемы
# из text.json (выравнивание заведомо возможно).
#   scripts/reference/.venv/bin/python <this> [model_dir]
# Выход ($W2V_OUT, по умолчанию tmp/w2v2_ref): grad_meta.json, grad_<имя>.npy.
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
from transformers import Wav2Vec2ForCTC

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/w2v2-lv60-espeak"
OUT = os.environ.get("W2V_OUT", "tmp/w2v2_ref")
model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
n = model.config.num_hidden_layers
train = [f"wav2vec2.encoder.layers.{n-2}.", f"wav2vec2.encoder.layers.{n-1}.", "wav2vec2.encoder.layer_norm.", "lm_head."]
for name, p in model.named_parameters():
    p.requires_grad_(any(name.startswith(t) for t in train))

wav = np.load(f"{OUT}/wav16.npy")
x = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
vocab = json.load(open(os.path.join(MODEL, "vocab.json")))
labels = [vocab[p] for p in json.load(open(f"{OUT}/text.json"))["phonemes"]]

logits = model(torch.from_numpy(x)[None]).logits.float()
lp = F.log_softmax(logits, -1)
loss = F.ctc_loss(lp.transpose(0, 1), torch.tensor([labels]), torch.tensor([lp.shape[1]]),
                  torch.tensor([len(labels)]), blank=0, reduction="sum", zero_infinity=False)
loss.backward()
check = [
    "lm_head.weight", "lm_head.bias", "wav2vec2.encoder.layer_norm.weight",
    f"wav2vec2.encoder.layers.{n-1}.final_layer_norm.weight",
    f"wav2vec2.encoder.layers.{n-1}.attention.q_proj.weight",
    f"wav2vec2.encoder.layers.{n-1}.feed_forward.intermediate_dense.weight",
    f"wav2vec2.encoder.layers.{n-2}.attention.k_proj.weight",
    f"wav2vec2.encoder.layers.{n-2}.layer_norm.bias",
]
params = dict(model.named_parameters())
for c in check:
    np.save(f"{OUT}/grad_{c}.npy", np.ascontiguousarray(params[c].grad.numpy()))
json.dump({"loss": loss.item(), "labels": labels, "check": check, "train": train}, open(f"{OUT}/grad_meta.json", "w"))
print("loss", loss.item(), "labels", len(labels))
