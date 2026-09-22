# Запуск: python -m venv v && v/bin/pip install gguf numpy && v/bin/python scripts/reference/gen_ggml_dequant_ref.py
# Пишет crates/synaptix-core/tests/reference_data/ggml_dequant_ref.bin.
# Эталонные векторы деквантования ggml: случайные валидные блоки → f32 через gguf-py.
import struct, numpy as np
from gguf import quants, GGMLQuantizationType as T, GGML_QUANT_SIZES
rng = np.random.default_rng(20260922)

def sane_f16(n):
    sign = rng.integers(0, 2, n).astype(np.uint16) << 15
    exp = rng.integers(7, 24, n).astype(np.uint16) << 10
    man = rng.integers(0, 1024, n).astype(np.uint16)
    return (sign | exp | man)

# смещения f16-полей шкал (байты) по типам
F16_AT = {
    T.Q4_0:[0], T.Q4_1:[0,2], T.Q5_0:[0], T.Q5_1:[0,2], T.Q8_0:[0], T.Q8_1:[0,2],
    T.Q2_K:[80,82], T.Q3_K:[108], T.Q4_K:[0,2], T.Q5_K:[0,2], T.Q6_K:[208],
    T.IQ2_XXS:[0], T.IQ2_XS:[0], T.IQ2_S:[0], T.IQ3_XXS:[0], T.IQ3_S:[0], T.IQ1_S:[0],
    T.IQ4_NL:[0], T.IQ4_XS:[0], T.TQ1_0:[52], T.TQ2_0:[64], T.Q1_0:[0],
}
types = [T.Q4_0,T.Q4_1,T.Q5_0,T.Q5_1,T.Q8_0,T.Q2_K,T.Q3_K,T.Q4_K,T.Q5_K,T.Q6_K,
         T.IQ2_XXS,T.IQ2_XS,T.IQ2_S,T.IQ3_XXS,T.IQ3_S,T.IQ1_S,T.IQ1_M,T.IQ4_NL,T.IQ4_XS,
         T.TQ1_0,T.TQ2_0,T.MXFP4,T.NVFP4]
out = bytearray()
count = 0
for t in types:
    be, bb = GGML_QUANT_SIZES[t]
    nblk = 3
    raw = rng.integers(0, 256, nblk*bb, dtype=np.uint8).reshape(nblk, bb)
    for b in range(nblk):
        for off in F16_AT.get(t, []):
            raw[b, off:off+2] = np.frombuffer(sane_f16(1).tobytes(), dtype=np.uint8)
        if t == T.IQ1_M:
            raw[b, 55] = (raw[b, 55] & 0x0F) | 0x30   # старший ниббл sc[3] → экспонента f16 в норме
        if t == T.MXFP4:
            raw[b, 0] = rng.integers(100, 140)
        if t == T.Q8_K:
            raw[b, 0:4] = np.frombuffer(np.float32(rng.uniform(-2,2)).tobytes(), dtype=np.uint8)
    n = nblk*be
    ref = quants.dequantize(raw.reshape(-1), t).astype(np.float32).reshape(-1)
    assert ref.shape[0] == n, (t, ref.shape)
    assert np.all(np.isfinite(ref)), t.name
    out += struct.pack('<III', int(t.value), n, raw.size)
    out += raw.tobytes()
    out += ref.astype('<f4').tobytes()
    count += 1
open('/home/master/Projects/2027/synaptix/crates/synaptix-core/tests/reference_data/ggml_dequant_ref.bin','wb').write(out)
print('types', count, 'bytes', len(out))
