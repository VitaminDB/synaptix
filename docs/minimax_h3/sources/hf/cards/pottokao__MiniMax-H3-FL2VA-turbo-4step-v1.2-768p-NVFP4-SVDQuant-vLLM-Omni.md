# pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-SVDQuant-vLLM-Omni

Источник: https://huggingface.co/pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-SVDQuant-vLLM-Omni · скачано 2026-11-28
Загрузок: 21 · лайков: 0 · базовая:  · теги: turbo,distilled,svdquant · обновлено: 2026-09-05T13:57:29.000Z

---

---
license: apache-2.0
pipeline_tag: image-to-video
library_name: diffusers
tags:
- minimax-h3
- fl2va
- turbo
- 4-step
- distilled
- nvfp4
- svdquant
- nunchaku
- vllm-omni
---

# MiniMax-H3 FL2VA — Turbo 4-step v1.2 (768p) · NVFP4 SVDQuant (vLLM-Omni)

vLLM-Omni serialized DiT for **MiniMax-H3 FL2VA** (First-&-Last-frame → Video + Audio), with the **Turbo 4-step v1.2 768p** LoRA fused in, quantized with **SVDQuant NVFP4**.

- **Format**: vLLM-Omni serialized SVDQuant — `FL2VA/transformer/` (`MiniMaxH3DiTModel`, 50 layers, 3 shards).
- **Quantization**: SVDQuant, precision **NVFP4**, **rank-32 low-rank correction**, activation **W4A4**, curve-basis AdaLN rank-16. `modules_to_not_convert`: `token_refiner`, `adaln_proj`, `condition_proj`.
- **Source**: bf16 FL2VA transformer + turbo-4step v1.2 LoRA fused (rank-128, α=8) — **no fp4 was ever dequantized**; only bf16 base + bf16 LoRA as inputs.
- **Size**: ~12.3 GiB (3 shards, 1785 tensors).

## Requirements
- **vLLM-Omni** with SVDQuant enabled and SM120 (consumer Blackwell) in the capability gate — see the patch below, or use the pre-built image.
- **Text encoder**: [pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-NVFP4-AWQ](https://huggingface.co/pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-NVFP4-AWQ) (W4A16_NVFP4, the vLLM-Omni path) — **not** the GGUF (that one is for ComfyUI).

## Running on consumer Blackwell — SM120 (RTX 50xx) & SM121 (GB10 / DGX Spark)

Upstream vLLM-Omni (v0.28.0) gates SVDQuant-NVFP4 to **SM103 (B300) only** and treats a
present `quant_config` as *online* quantization, which forces the whole DiT onto the GPU at
build time (OOM on 16 GB). Two tiny changes fix both; the same NVFP4 tensor-core path runs
unchanged on **SM120** (RTX 5070 Ti / 5060 Ti) and **SM121** (GB10 / DGX Spark).

**Pre-built image (no patching needed)** — shipped in this repo under `docker/`:
```bash
hf download pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-SVDQuant-vLLM-Omni \
    docker/h3-vllm-omni-nvfp4-0.28.0-sm120.tar.gz --local-dir .
docker load < docker/h3-vllm-omni-nvfp4-0.28.0-sm120.tar.gz   # → h3-vllm-omni-nvfp4:0.28.0-sm120
```
Built on RTX 5060 Ti (consumer Blackwell / SM120), x86_64. Base: `vllm/vllm-openai:v0.28.0`.

### What is patched — `vllm_omni_0.28.0_sm120_svdquant.patch` (93 lines, 3 files)

1. **`vllm_omni/quantization/svdquant_config.py`**
   - `_SUPPORTED_CAPABILITIES = {(10, 3)}` → `{(10, 3), (12, 0), (12, 1)}` — admit SM120 (RTX 50xx) and SM121 (GB10).
   - `DiffusionSVDQuantConfig.__init__`: add `self.is_checkpoint_quantized = True` — mark the
     serialized checkpoint as **offline** so the diffusion loader allows CPU-offload instead of
     forcing the DiT onto the accelerator (this is what lets it fit 16 GB).
   - error-string update (SM103 → SM103/SM120).
2. **`vllm_omni/diffusion/model_loader/host_weight_plan.py`**
   - add `_needs_post_load_quant_processing()` and, in `build_checkpoint_mmap_plan`, fall back
     to the ordinary loader for SVDQuant checkpoints. SVDQuant renames/reshapes params in
     `process_weights_after_loading` (qweight→weight, scale swizzle+pad, `input_global_scale_inv`/
     `alpha` creation); the direct-mmap plan binds the *pre-processing* names and would stream a
     layer missing its kernel runtime params.
3. **`tests/diffusion/quantization/test_svdquant_linear.py`** — capability-gate test update.

```diff
--- a/vllm_omni/quantization/svdquant_config.py
+++ b/vllm_omni/quantization/svdquant_config.py
@@
-_SUPPORTED_CAPABILITIES = {(10, 3)}
+# SM103 (B300) upstream-validated; SM120 (RTX 50xx) and SM121 (GB10 / DGX Spark)
+# run the same NVFP4 tensor-core path via vLLM CUTLASS/FlashInfer kernels;
+# validated on RTX 5070 Ti and GB10 with MiniMax-H3 SVDQuant-NVFP4.
+_SUPPORTED_CAPABILITIES = {(10, 3), (12, 0), (12, 1)}
@@ class DiffusionSVDQuantConfig
         self.modules_to_not_convert = modules_to_not_convert or []
+        # Offline serialized checkpoint: weights are already quantized on disk.
+        # Without this flag the loader assumes *online* quant and forces the whole
+        # DiT onto the accelerator at build time (OOM on 16 GB).
+        self.is_checkpoint_quantized = True
```

### Build it yourself
```bash
# layer 1: official vLLM-Omni CUDA image from the patched v0.28.0 checkout
git clone --branch v0.28.0 <vllm-omni> vllm-omni-src
cd vllm-omni-src && git apply vllm_omni_0.28.0_sm120_svdquant.patch
docker build -f docker/Dockerfile.cuda -t vllm-omni:0.28.0-sm120 .
# layer 2: H3 exporter + example
docker build -f Dockerfile.h3 -t h3-vllm-omni-nvfp4:0.28.0-sm120 .
```

### Run
```bash
docker run --rm --gpus all --ipc=host \
  -v $PWD/FL2VA:/models/FL2VA \
  h3-vllm-omni-nvfp4:0.28.0-sm120 \
  python3 /opt/h3/smoke_gen.py --model /models/FL2VA --quant @/opt/h3/quant_svdquant.json --out /out
```

## Related
- **ComfyUI path** (rotated-NVFP4, nunchaku loader): [pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-rotated-T1](https://huggingface.co/pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-rotated-T1)

_Abliterated / uncensored pipeline component for the MiniMax-H3 text-to-video / image-to-video stack._
