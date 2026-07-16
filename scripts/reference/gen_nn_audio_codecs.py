"""Reference SafeTensors для synaptix-nn/audio quantizers (FSQ/LFQ/RVQ).

Run:
    python scripts/reference/gen_nn_audio_codecs.py

Outputs в tests/reference_data/nn_audio_codecs/<case>.safetensors.

Reference math воспроизводит lucidrains `vector-quantize-pytorch`:
- FSQ: round(tanh(z) * half - offset), индексы — mixed-base encoding.
- LFQ: sign(z) → ±1, индексы — бит-маска.
- RVQ: greedy nearest-neighbor по residual.

Codec-wrappers (DAC/EnCodec/SNAC/Mimi/HiggsAudio) — shape-only тесты в Rust,
ref-кейсы откладываются до Phase O (для них нужны pretrained веса HF).
"""

import pathlib

import torch
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_audio_codecs")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def fsq_quantize(z: torch.Tensor, levels: list[int]):
    halves = torch.tensor([(l - 1) * 0.5 for l in levels])
    offsets = torch.tensor([0.5 if l % 2 == 0 else 0.0 for l in levels])
    basis = torch.tensor([1] + [1] * (len(levels) - 1), dtype=torch.long)
    for i in range(1, len(levels)):
        basis[i] = basis[i - 1] * levels[i - 1]

    tanh_z = torch.tanh(z)
    scaled = tanh_z * halves - offsets
    los = -halves - offsets
    his = halves - offsets
    rounded = torch.clamp(scaled.round(), los, his)
    shifted = (rounded + halves + offsets).round().long()
    indices = (shifted * basis).sum(dim=-1)
    return rounded, indices


def fsq_dequantize(indices: torch.Tensor, levels: list[int]):
    halves = torch.tensor([(l - 1) * 0.5 for l in levels])
    offsets = torch.tensor([0.5 if l % 2 == 0 else 0.0 for l in levels])
    out = torch.zeros(*indices.shape, len(levels), dtype=torch.float32)
    for i, l in enumerate(levels):
        shifted = (indices % l).float()
        indices = indices // l
        out[..., i] = shifted - halves[i] - offsets[i]
    return out


def case_fsq_3_3_3():
    torch.manual_seed(600)
    levels = [3, 3, 3]
    z = torch.randn(2, 4, 3)
    codes, indices = fsq_quantize(z, levels)
    dq = fsq_dequantize(indices, levels)
    save_case("fsq_3_3_3", {
        "z": z, "codes": codes, "indices": indices, "dequantized": dq,
    })


def case_fsq_4_4_4_4():
    torch.manual_seed(601)
    levels = [4, 4, 4, 4]
    z = torch.randn(3, 5, 4)
    codes, indices = fsq_quantize(z, levels)
    dq = fsq_dequantize(indices, levels)
    save_case("fsq_4_4_4_4", {
        "z": z, "codes": codes, "indices": indices, "dequantized": dq,
    })


def lfq_quantize(z: torch.Tensor):
    codes = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    bits = (codes > 0).long()
    powers = (1 << torch.arange(z.shape[-1])).long()
    indices = (bits * powers).sum(dim=-1)
    return codes, indices


def lfq_dequantize(indices: torch.Tensor, dim: int):
    out = torch.zeros(*indices.shape, dim, dtype=torch.float32)
    for i in range(dim):
        bit = ((indices >> i) & 1).float()
        out[..., i] = bit * 2.0 - 1.0
    return out


def case_lfq_dim4():
    torch.manual_seed(602)
    dim = 4
    z = torch.randn(2, 5, dim)
    codes, indices = lfq_quantize(z)
    dq = lfq_dequantize(indices, dim)
    save_case("lfq_dim4", {"z": z, "codes": codes, "indices": indices, "dequantized": dq})


def rvq_encode_decode(x: torch.Tensor, codebooks: list[torch.Tensor]):
    *batch_shape, dim = x.shape
    flat = x.reshape(-1, dim)
    num_cb = len(codebooks)
    indices = torch.zeros(flat.shape[0], num_cb, dtype=torch.long)
    residual = flat.clone()
    for c, cb in enumerate(codebooks):
        # L2 distance to each entry: [N, K]
        dists = ((residual.unsqueeze(1) - cb.unsqueeze(0)) ** 2).sum(dim=-1)
        idx = dists.argmin(dim=1)
        indices[:, c] = idx
        residual = residual - cb[idx]
    # decode
    recon = torch.zeros_like(flat)
    for c, cb in enumerate(codebooks):
        recon = recon + cb[indices[:, c]]
    indices = indices.reshape(*batch_shape, num_cb)
    recon = recon.reshape(*batch_shape, dim)
    return indices, recon


def case_rvq_3cb_8sz_4dim():
    torch.manual_seed(603)
    dim, codebook_size, num_cb = 4, 8, 3
    codebooks = [torch.randn(codebook_size, dim) for _ in range(num_cb)]
    x = torch.randn(2, 6, dim)
    indices, recon = rvq_encode_decode(x, codebooks)
    save_case("rvq_3cb_8sz_4dim", {
        "x": x,
        "cb0": codebooks[0], "cb1": codebooks[1], "cb2": codebooks[2],
        "indices": indices, "recon": recon,
    })


def main() -> None:
    print("Generating nn-audio-codecs reference data...")
    case_fsq_3_3_3()
    case_fsq_4_4_4_4()
    case_lfq_dim4()
    case_rvq_3cb_8sz_4dim()
    print("Done.")


if __name__ == "__main__":
    main()
