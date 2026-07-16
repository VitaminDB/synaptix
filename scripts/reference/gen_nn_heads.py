"""Generate reference SafeTensors for synaptix-nn heads + adapters.

Run:
    python scripts/reference/gen_nn_heads.py

Outputs tests/reference_data/nn_heads/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_heads")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_lm_head() -> None:
    torch.manual_seed(400)
    hidden_size, vocab_size = 8, 16
    weight = torch.randn(vocab_size, hidden_size)
    x = torch.randn(2, 4, hidden_size)
    out = F.linear(x, weight)
    save_case("lm_head", {"weight": weight, "x": x, "output": out})


def case_cls_head() -> None:
    torch.manual_seed(401)
    hidden_size, num_classes = 8, 4
    dense_w = torch.randn(hidden_size, hidden_size)
    dense_b = torch.randn(hidden_size)
    out_w = torch.randn(num_classes, hidden_size)
    out_b = torch.randn(num_classes)
    x = torch.randn(2, hidden_size)
    h = F.linear(x, dense_w, dense_b).tanh()
    logits = F.linear(h, out_w, out_b)
    save_case(
        "cls_head",
        {
            "dense_w": dense_w, "dense_b": dense_b,
            "out_w": out_w, "out_b": out_b,
            "x": x, "output": logits,
        },
    )


def case_token_cls_head() -> None:
    torch.manual_seed(402)
    hidden_size, num_labels = 8, 5
    weight = torch.randn(num_labels, hidden_size)
    bias = torch.randn(num_labels)
    x = torch.randn(2, 6, hidden_size)
    out = F.linear(x, weight, bias)
    save_case("token_cls_head", {"weight": weight, "bias": bias, "x": x, "output": out})


def case_reward_head() -> None:
    torch.manual_seed(403)
    hidden_size = 8
    weight = torch.randn(1, hidden_size)
    bias = torch.randn(1)
    x = torch.randn(3, hidden_size)
    out = F.linear(x, weight, bias)
    save_case("reward_head", {"weight": weight, "bias": bias, "x": x, "output": out})


def case_lora() -> None:
    torch.manual_seed(404)
    in_f, out_f, r = 8, 8, 4
    alpha = 8.0
    scaling = alpha / r
    base_w = torch.randn(out_f, in_f)
    a = torch.randn(r, in_f)
    b = torch.randn(out_f, r)
    x = torch.randn(2, in_f)
    out = F.linear(x, base_w) + F.linear(F.linear(x, a), b) * scaling
    save_case(
        "lora",
        {"base_w": base_w, "lora_a": a, "lora_b": b, "x": x, "output": out},
    )


def case_dora() -> None:
    torch.manual_seed(405)
    in_f, out_f, r = 8, 8, 4
    alpha = 8.0
    scaling = alpha / r
    base_w = torch.randn(out_f, in_f)
    a = torch.randn(r, in_f)
    b = torch.randn(out_f, r)
    magnitude = torch.rand(out_f) + 0.5
    x = torch.randn(2, in_f)
    eps = 1e-8

    v = base_w + (b @ a) * scaling  # [out, in]
    v_norm = v.norm(p=2, dim=1, keepdim=True) + eps  # [out, 1]
    v_normalized = v / v_norm
    scaled = v_normalized * magnitude.unsqueeze(1)
    out = x @ scaled.t()
    save_case(
        "dora",
        {
            "base_w": base_w, "lora_a": a, "lora_b": b,
            "magnitude": magnitude, "x": x, "output": out,
        },
    )


def case_ia3() -> None:
    torch.manual_seed(406)
    in_f, out_f = 8, 8
    base_w = torch.randn(out_f, in_f)
    scale = torch.rand(out_f) + 0.5
    x = torch.randn(2, in_f)
    out = F.linear(x, base_w) * scale
    save_case(
        "ia3",
        {"base_w": base_w, "scale": scale, "x": x, "output": out},
    )


def case_prefix_tuning() -> None:
    torch.manual_seed(407)
    num_layers, prefix_len, hidden = 3, 4, 8
    pk = torch.randn(num_layers, prefix_len, hidden)
    pv = torch.randn(num_layers, prefix_len, hidden)
    layer = 1
    save_case(
        "prefix_tuning",
        {
            "prefix_keys": pk, "prefix_values": pv,
            "layer_k": pk[layer:layer+1].clone(),
            "layer_v": pv[layer:layer+1].clone(),
        },
    )


def case_ctc_head() -> None:
    torch.manual_seed(420)
    hidden_size, vocab_size = 8, 12
    weight = torch.randn(vocab_size, hidden_size)
    bias = torch.randn(vocab_size)
    x = torch.randn(2, 6, hidden_size)
    logits = F.linear(x, weight, bias)
    out = F.log_softmax(logits, dim=-1)
    save_case("ctc_head", {"weight": weight, "bias": bias, "x": x, "output": out})


def case_mlm_head() -> None:
    torch.manual_seed(421)
    hidden_size, vocab_size = 8, 16
    dense_w = torch.randn(hidden_size, hidden_size)
    dense_b = torch.randn(hidden_size)
    ln_w = torch.randn(hidden_size).abs() + 0.5
    ln_b = torch.randn(hidden_size) * 0.1
    out_w = torch.randn(vocab_size, hidden_size)
    out_b = torch.randn(vocab_size)
    eps = 1e-12
    x = torch.randn(2, 6, hidden_size)
    h = F.linear(x, dense_w, dense_b)
    h = F.gelu(h, approximate="none")
    h = F.layer_norm(h, (hidden_size,), ln_w, ln_b, eps)
    out = F.linear(h, out_w, out_b)
    save_case(
        "mlm_head",
        {
            "dense_w": dense_w, "dense_b": dense_b,
            "ln_w": ln_w, "ln_b": ln_b,
            "out_w": out_w, "out_b": out_b,
            "x": x, "output": out,
        },
    )


def case_qa_head() -> None:
    torch.manual_seed(422)
    hidden_size = 8
    weight = torch.randn(2, hidden_size)
    bias = torch.randn(2)
    x = torch.randn(2, 6, hidden_size)
    out = F.linear(x, weight, bias)
    start = out[..., 0]
    end = out[..., 1]
    save_case(
        "qa_head",
        {
            "weight": weight, "bias": bias, "x": x,
            "output": out, "start": start, "end": end,
        },
    )


def case_segmentation_head() -> None:
    torch.manual_seed(423)
    hidden_size, num_classes = 8, 5
    weight = torch.randn(num_classes, hidden_size)
    bias = torch.randn(num_classes)
    b, h, w = 2, 4, 4
    x = torch.randn(b, hidden_size, h, w)
    x_perm = x.permute(0, 2, 3, 1).contiguous()
    x_flat = x_perm.view(b * h * w, hidden_size)
    logits_flat = F.linear(x_flat, weight, bias)
    logits = logits_flat.view(b, h, w, num_classes).permute(0, 3, 1, 2).contiguous()
    save_case("segmentation_head", {"weight": weight, "bias": bias, "x": x, "output": logits})


def case_bbox_head() -> None:
    torch.manual_seed(424)
    hidden_size, num_classes = 8, 3
    weight = torch.randn(num_classes * 4, hidden_size)
    bias = torch.randn(num_classes * 4)
    x = torch.randn(2, 6, hidden_size)
    logits = F.linear(x, weight, bias)
    sig = torch.sigmoid(logits).view(2, 6, num_classes, 4)
    raw = logits.view(2, 6, num_classes, 4)
    save_case(
        "bbox_head",
        {"weight": weight, "bias": bias, "x": x, "output_sigmoid": sig, "output_raw": raw},
    )


def case_regression_head() -> None:
    torch.manual_seed(425)
    hidden_size, output_dim = 8, 3
    dense_w = torch.randn(hidden_size, hidden_size)
    dense_b = torch.randn(hidden_size)
    out_w = torch.randn(output_dim, hidden_size)
    out_b = torch.randn(output_dim)
    x = torch.randn(2, hidden_size)
    h = F.linear(x, dense_w, dense_b).tanh()
    out = F.linear(h, out_w, out_b)
    save_case(
        "regression_head",
        {
            "dense_w": dense_w, "dense_b": dense_b,
            "out_w": out_w, "out_b": out_b,
            "x": x, "output": out,
        },
    )


def case_keypoint_head() -> None:
    torch.manual_seed(426)
    hidden_size, num_keypoints = 8, 4
    weight = torch.randn(num_keypoints * 3, hidden_size)
    bias = torch.randn(num_keypoints * 3)
    x = torch.randn(2, 6, hidden_size)
    logits = F.linear(x, weight, bias).view(2, 6, num_keypoints, 3)
    xy = logits[..., :2]
    vis = torch.sigmoid(logits[..., 2:3])
    out_sigmoid = torch.cat([xy, vis], dim=-1)
    save_case(
        "keypoint_head",
        {"weight": weight, "bias": bias, "x": x, "output_sigmoid": out_sigmoid, "output_raw": logits},
    )


def case_rnn_t_head() -> None:
    torch.manual_seed(427)
    enc_dim, pred_dim, joint_dim, vocab_size = 6, 5, 8, 12
    enc_w = torch.randn(joint_dim, enc_dim)
    enc_b = torch.randn(joint_dim)
    pred_w = torch.randn(joint_dim, pred_dim)
    pred_b = torch.randn(joint_dim)
    out_w = torch.randn(vocab_size, joint_dim)
    out_b = torch.randn(vocab_size)
    b, t_len, u_len = 2, 3, 4
    enc = torch.randn(b, t_len, enc_dim)
    pred = torch.randn(b, u_len, pred_dim)
    f = F.linear(enc, enc_w, enc_b)
    g = F.linear(pred, pred_w, pred_b)
    joint = (f.unsqueeze(2) + g.unsqueeze(1)).tanh()
    out = F.linear(joint, out_w, out_b)
    save_case(
        "rnn_t_head",
        {
            "enc_w": enc_w, "enc_b": enc_b,
            "pred_w": pred_w, "pred_b": pred_b,
            "out_w": out_w, "out_b": out_b,
            "enc": enc, "pred": pred, "output": out,
        },
    )


def main() -> None:
    print("Generating nn-heads + adapters reference data...")
    case_lm_head()
    case_cls_head()
    case_token_cls_head()
    case_reward_head()
    case_lora()
    case_dora()
    case_ia3()
    case_prefix_tuning()
    case_ctc_head()
    case_mlm_head()
    case_qa_head()
    case_segmentation_head()
    case_bbox_head()
    case_regression_head()
    case_keypoint_head()
    case_rnn_t_head()
    print("Done.")


if __name__ == "__main__":
    main()
