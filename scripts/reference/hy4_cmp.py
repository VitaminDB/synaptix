#!/usr/bin/env python3
"""Сравнить дампы потоков Hy4 по слоям: hy4_cmp.py эталон/ движок/ [--hc 4 --hidden 3072]."""
import os
import sys

import numpy as np

ref, eng = sys.argv[1], sys.argv[2]
for f in sorted(os.listdir(ref)):
    if not f.endswith(".f32") or not os.path.exists(os.path.join(eng, f)):
        continue
    a = np.fromfile(os.path.join(ref, f), dtype=np.float32)
    b = np.fromfile(os.path.join(eng, f), dtype=np.float32)
    if a.shape != b.shape:
        print(f"{f}: формы {a.shape} / {b.shape}")
        continue
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)
    if f.startswith("layer"):
        t = a.size // (4 * 3072)
        ar, br = a.reshape(t, -1), b.reshape(t, -1)
        cos = (ar * br).sum(1) / (np.linalg.norm(ar, axis=1) * np.linalg.norm(br, axis=1) + 1e-30)
        print(f"{f}: rel {rel:.3e}  cos min {cos.min():.6f}")
    else:
        print(f"{f}: rel {rel:.3e}  top1 {a.argmax()} / {b.argmax()}")
