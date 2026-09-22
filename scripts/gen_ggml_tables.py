#!/usr/bin/env python3
# Таблицы-решётки ggml → Rust (synaptix-core/src/quant/ggml_tables.rs) и CUDA
# (synaptix-kernels-cuda/src/cu/elementwise/ggml_tables.cuh).
# Вход: ggml-common.h из llama.cpp (ggml/src/ggml-common.h, MIT).
#   python3 scripts/gen_ggml_tables.py path/to/ggml-common.h
import re, sys, pathlib
src = open(sys.argv[1]).read()
root = pathlib.Path(__file__).resolve().parent.parent
want = ['kmask_iq2xs','ksigns_iq2xs','iq2xxs_grid','iq2xs_grid','iq2s_grid','iq3xxs_grid','iq3s_grid','kvalues_iq4nl','kvalues_fp4','iq1s_grid']
tables = {}
for m in re.finditer(r'GGML_TABLE_BEGIN\((\w+), (\w+), (\w+)\)\n(.*?)GGML_TABLE_END\(\)', src, re.S):
    ty, name, size, body = m.groups()
    if name in want:
        tables[name] = (ty, size, re.findall(r'-?0x[0-9a-fA-F]+|-?\d+', body))
sizes = {'NGRID_IQ1S': 2048}
rmap = {'uint8_t': 'u8', 'uint64_t': 'u64', 'uint32_t': 'u32', 'int8_t': 'i8'}
rs = ['//! Таблицы-решётки ggml для IQ-форматов и LUT nl/fp4 — байт в байт из',
      '//! `ggml/src/ggml-common.h` (llama.cpp, MIT). Сгенерировано скриптом,',
      '//! руками не править.', '']
cu = ['// Таблицы-решётки ggml (ggml-common.h, MIT). Сгенерировано, руками не править.',
      '// NVRTC без stdint.h — типы объявляем сами.',
      'typedef unsigned char uint8_t;', 'typedef unsigned short uint16_t;',
      'typedef unsigned int uint32_t;', 'typedef unsigned long long uint64_t;',
      'typedef signed char int8_t;', '']
for name in want:
    ty, size, vals = tables[name]
    n = int(sizes.get(size, size)); assert len(vals) == n, (name, len(vals), n)
    rs.append(f'pub static {name.upper()}: [{rmap[ty]}; {n}] = [')
    rs += ['    ' + ', '.join(vals[i:i+8]) + ',' for i in range(0, n, 8)]
    rs.append('];\n')
    cu.append(f'static const __device__ {ty} {name}[{n}] = {{')
    cu += ['    ' + ', '.join(vals[i:i+8]) + ',' for i in range(0, n, 8)]
    cu.append('};\n')
(root / 'crates/synaptix-core/src/quant/ggml_tables.rs').write_text('\n'.join(rs))
(root / 'crates/synaptix-kernels-cuda/src/cu/elementwise/ggml_tables.cuh').write_text('\n'.join(cu))
print({k: len(v[2]) for k, v in tables.items()})
