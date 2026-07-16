"""Generate reference SafeTensors for conv1d/2d/3d.

Run:
    python scripts/reference/gen_conv.py

Outputs tests/reference_data/conv/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/conv")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_conv1d_basic() -> None:
    torch.manual_seed(300)
    b, c_in, l = 2, 3, 16
    c_out, k = 5, 3
    x = torch.randn(b, c_in, l)
    w = torch.randn(c_out, c_in, k)
    bias = torch.randn(c_out)
    out = F.conv1d(x, w, bias=bias, stride=1, padding=1)
    save_case("conv1d_basic", {"x": x, "w": w, "bias": bias, "output": out})


def case_conv1d_stride2() -> None:
    torch.manual_seed(301)
    b, c_in, l = 1, 4, 20
    c_out, k = 8, 3
    x = torch.randn(b, c_in, l)
    w = torch.randn(c_out, c_in, k)
    out = F.conv1d(x, w, bias=None, stride=2, padding=1)
    save_case("conv1d_stride2", {"x": x, "w": w, "output": out})


def case_conv2d_basic() -> None:
    torch.manual_seed(302)
    b, c_in, h, w = 2, 3, 8, 8
    c_out, kh, kw = 4, 3, 3
    x = torch.randn(b, c_in, h, w)
    weight = torch.randn(c_out, c_in, kh, kw)
    bias = torch.randn(c_out)
    out = F.conv2d(x, weight, bias=bias, stride=1, padding=1)
    save_case("conv2d_basic", {"x": x, "w": weight, "bias": bias, "output": out})


def case_conv2d_stride2() -> None:
    torch.manual_seed(303)
    b, c_in, h, w = 1, 3, 16, 16
    c_out, kh, kw = 6, 3, 3
    x = torch.randn(b, c_in, h, w)
    weight = torch.randn(c_out, c_in, kh, kw)
    out = F.conv2d(x, weight, bias=None, stride=2, padding=1)
    save_case("conv2d_stride2", {"x": x, "w": weight, "output": out})


def case_conv2d_dilated() -> None:
    torch.manual_seed(304)
    b, c_in, h, w = 1, 2, 10, 10
    c_out, kh, kw = 3, 3, 3
    x = torch.randn(b, c_in, h, w)
    weight = torch.randn(c_out, c_in, kh, kw)
    out = F.conv2d(x, weight, bias=None, stride=1, padding=2, dilation=2)
    save_case("conv2d_dilated", {"x": x, "w": weight, "output": out})


def case_conv3d_basic() -> None:
    torch.manual_seed(305)
    b, c_in, d, h, w = 1, 2, 4, 6, 6
    c_out, kd, kh, kw = 3, 3, 3, 3
    x = torch.randn(b, c_in, d, h, w)
    weight = torch.randn(c_out, c_in, kd, kh, kw)
    bias = torch.randn(c_out)
    out = F.conv3d(x, weight, bias=bias, stride=1, padding=1)
    save_case("conv3d_basic", {"x": x, "w": weight, "bias": bias, "output": out})


def case_conv3d_stride() -> None:
    torch.manual_seed(306)
    b, c_in, d, h, w = 1, 2, 6, 8, 8
    c_out, kd, kh, kw = 4, 3, 3, 3
    x = torch.randn(b, c_in, d, h, w)
    weight = torch.randn(c_out, c_in, kd, kh, kw)
    out = F.conv3d(x, weight, bias=None, stride=(2, 2, 2), padding=(1, 1, 1))
    save_case("conv3d_stride", {"x": x, "w": weight, "output": out})


def case_transposed_conv_basic() -> None:
    torch.manual_seed(307)
    b, c_in, l = 2, 3, 8
    c_out, k = 4, 3
    x = torch.randn(b, c_in, l)
    w = torch.randn(c_in, c_out, k)  # ConvTranspose1d: [C_in, C_out, K]
    bias = torch.randn(c_out)
    out = F.conv_transpose1d(x, w, bias=bias, stride=1, padding=0)
    save_case("transposed_conv_basic", {"x": x, "w": w, "bias": bias, "output": out})


def case_transposed_conv_stride2() -> None:
    torch.manual_seed(308)
    b, c_in, l = 2, 3, 8
    c_out, k = 4, 3
    x = torch.randn(b, c_in, l)
    w = torch.randn(c_in, c_out, k)
    out = F.conv_transpose1d(x, w, bias=None, stride=2, padding=1)
    save_case("transposed_conv_stride2", {"x": x, "w": w, "output": out})


def case_causal_conv3d_basic() -> None:
    torch.manual_seed(309)
    b, c_in, t, h, w = 2, 2, 5, 6, 6
    c_out, kt, kh, kw = 3, 3, 3, 3
    x = torch.randn(b, c_in, t, h, w)
    weight = torch.randn(c_out, c_in, kt, kh, kw)
    bias = torch.randn(c_out)
    # causal pad по времени (слева kt-1), VALID по пространству, uniform stride=1
    xp = F.pad(x, (0, 0, 0, 0, kt - 1, 0))
    out = F.conv3d(xp, weight, bias=bias, stride=1)
    save_case(
        "causal_conv3d_basic",
        {"x": x, "weight": weight, "bias": bias, "output": out},
    )


def case_causal_conv3d_stride2() -> None:
    torch.manual_seed(310)
    b, c_in, t, h, w = 2, 2, 5, 6, 6
    c_out, kt, kh, kw = 3, 3, 3, 3
    x = torch.randn(b, c_in, t, h, w)
    weight = torch.randn(c_out, c_in, kt, kh, kw)
    xp = F.pad(x, (0, 0, 0, 0, kt - 1, 0))
    out = F.conv3d(xp, weight, bias=None, stride=2)
    save_case("causal_conv3d_stride2", {"x": x, "weight": weight, "output": out})


def main() -> None:
    print("Generating conv reference data...")
    case_conv1d_basic()
    case_conv1d_stride2()
    case_conv2d_basic()
    case_conv2d_stride2()
    case_conv2d_dilated()
    case_conv3d_basic()
    case_conv3d_stride()
    case_transposed_conv_basic()
    case_transposed_conv_stride2()
    case_causal_conv3d_basic()
    case_causal_conv3d_stride2()
    print("Done.")


if __name__ == "__main__":
    main()
