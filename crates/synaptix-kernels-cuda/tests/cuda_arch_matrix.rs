//! Матрица архитектур: каждый `.cu` собирается NVRTC под все цели, на которые
//! движок претендует. Портируемые ядра обязаны собираться под `sm_80` — это
//! гарантия, что в них нет инструкций новее Ampere (NVRTC их не пропустит).
//! Blackwell-ядра (block-scale MMA) обязаны собираться под `sm_120a` и
//! обязаны НЕ собираться под `sm_80`: если вдруг собрались — значит,
//! из них выпал MMA-путь. TMA-семейство `gemm_bf16.cu` под `sm_80` вырезается
//! препроцессором, и в PTX не должно остаться `setmaxnreg`.
//!
//! Только компиляция, без загрузки в контекст (PTX под `sm_90a` на карте
//! sm_120 не загрузить). Нужен NVRTC, карта не нужна.

use synaptix_kernels_cuda::kernels::compile::compile_ptx_only;

const ARCHES: &[&str] = &["sm_80", "sm_86", "sm_89", "sm_90a", "sm_120a"];

macro_rules! cu {
    ($($path:literal),* $(,)?) => {
        &[$(($path, include_str!(concat!("../src/", $path)))),*]
    };
}

/// Собираются под любую цель.
const PORTABLE: &[(&str, &str)] = cu![
    "best_cu/gemm/gemm_bf16.cu",
    "best_cu/gemm/gemm_f16.cu",
    "best_cu/gemm/gemm_f32.cu",
    "best_cu/gemv/gemv_mxfp8.cu",
    "best_cu/gemv/mma_gemv.cu",
    "cu/conv/causal_conv1d_chunk.cu",
    "cu/conv/causal_conv1d.cu",
    "cu/conv/conv1d.cu",
    "cu/conv/conv2d.cu",
    "cu/conv/conv3d_causal.cu",
    "cu/conv/conv3d.cu",
    "cu/conv/depthwise.cu",
    "cu/conv/im2col.cu",
    "cu/conv/implicit_conv.cu",
    "cu/conv/nchw_nhwc.cu",
    "cu/conv/upsample_nearest2x.cu",
    "cu/elementwise/activations.cu",
    "cu/elementwise/kv_append.cu",
    "cu/elementwise/logit_cap.cu",
    "cu/elementwise/mxfp8_kv.cu",
    "cu/elementwise/mxfp8_quant.cu",
    "cu/elementwise/nvfp4_quant.cu",
    "cu/elementwise/rope.cu",
    "cu/embed/embed.cu",
    "cu/fused/attention/chunk_fla.cu",
    "cu/fused/attention/flash_attn_v2.cu",
    "cu/fused/attention/flash_blocks.cu",
    "cu/fused/attention/flash_decode.cu",
    "cu/fused/attention/flash_decode_gqa.cu",
    "cu/fused/attention/flash_decode_mxfp8_v2.cu",
    "cu/fused/attention/flash_mxfp8_prefill.cu",
    "cu/fused/attention/flash_mxfp8_splitq.cu",
    "cu/fused/attention/flash_splitq.cu",
    "cu/fused/attention/linear_attn_raw.cu",
    "cu/fused/attention/sdpa_f32_acc.cu",
    "cu/fused/conv/conv_epilogue.cu",
    "cu/fused/llm_decode.cu",
    "cu/fused/loss/cross_entropy.cu",
    "cu/fused/mlp/geglu_split.cu",
    "cu/fused/mlp/silu_and_mul.cu",
    "cu/fused/moe/moe_dispatch.cu",
    "cu/fused/norm/group_norm.cu",
    "cu/fused/norm/layernorm_residual.cu",
    "cu/fused/norm/pixel_norm.cu",
    "cu/fused/norm/rms_mod_quant.cu",
    "cu/fused/norm/rmsnorm_residual.cu",
    "cu/fused/qwen4_decode.cu",
    "cu/fused/ssm/delta_rule.cu",
    "cu/fused/ssm/gated_delta_rule.cu",
    "cu/fused/ssm/linear_attn_prep_scatter.cu",
    "cu/fused/ssm/mamba2_ssd.cu",
    "cu/fused/ssm/mamba_scan.cu",
    "cu/fused/topk_rows.cu",
    "cu/fused/topk_wide.cu",
    "cu/kernels/dwconv1d.cu",
    "cu/kernels/elementwise.cu",
    "cu/kernels/reduce.cu",
    "cu/reduction/layernorm.cu",
    "cu/reduction/rms_norm.cu",
    "cu/reduction/softmax.cu",
    "cu/reduction/topk.cu",
    "cu/scan/chunk_scan.cu",
    "cu/scan/parallel_scan.cu",
    "cu/ssm/mamba2_bmm.cu",
    "cu/ssm/mamba2_chunked_helpers.cu",
];

/// Только block-scale MMA: `sm_120a` (и `sm_100a`), под остальное — ошибка.
const BLACKWELL_ONLY: &[(&str, &str)] = cu![
    "best_cu/gemm/gemm_nvfp4.cu",
    "best_cu/gemv/gemv_nvfp4.cu",
    "best_cu/gemm/gemm_mxfp8.cu",
    "cu/fused/mlp/nvfp4_geglu_shuf.cu",
    "cu/fused/mlp/nvfp4_swiglu_shuf.cu",
    "cu/fused/projection/nvfp4_qkv_proj_shuf.cu",
];

/// `mbarrier.try_wait` — sm_90+.
const SM90_ONLY: &[(&str, &str)] = cu!["cu/fused/attention/flash_splitq6.cu"];

fn compile(src: &str, tag: &'static str, arch: &'static str) -> Result<String, String> {
    compile_ptx_only(src, tag, &[], arch).map_err(|e| e.to_string())
}

/// `flash_attn_v2_bf16.cu` — надстройка над `flash_attn_v2.cu`, Rust-сторона
/// склеивает их в один исходник (см. `attention/flash_bf16.rs`).
fn flash_bf16_combined() -> String {
    format!(
        "{}\n\n{}",
        include_str!("../src/cu/fused/attention/flash_attn_v2.cu"),
        include_str!("../src/cu/fused/attention/flash_attn_v2_bf16.cu")
    )
}

#[test]
fn portable_kernels_build_for_every_target() {
    let mut failures = Vec::new();
    let combined = flash_bf16_combined();
    let extra: [(&str, &str); 1] = [("cu/fused/attention/flash_attn_v2_bf16.cu", combined.as_str())];
    for (tag, src) in PORTABLE.iter().chain(extra.iter()) {
        for arch in ARCHES {
            if let Err(e) = compile(src, tag, arch) {
                let head: String = e.lines().take(6).collect::<Vec<_>>().join("\n");
                failures.push(format!("{tag} @ {arch}:\n{head}"));
            }
        }
    }
    assert!(failures.is_empty(), "не собрались:\n{}", failures.join("\n\n"));
}

#[test]
fn blockq_dequant_builds_for_every_target() {
    // Таблицы + ядра склеиваются Rust-стороной (см. elementwise/blockq.rs).
    let src = synaptix_kernels_cuda::elementwise::blockq::module_source();
    for arch in ARCHES {
        compile_ptx_only(&src, "blockq_dequant.cu", synaptix_kernels_cuda::elementwise::blockq::MODULE_OPTS, arch)
            .unwrap_or_else(|e| panic!("blockq_dequant.cu @ {arch}: {e}"));
        let gsrc = synaptix_kernels_cuda::elementwise::blockq::gemv_module_source();
        compile_ptx_only(&gsrc, "blockq_gemv.cu", &[], arch)
            .unwrap_or_else(|e| panic!("blockq_gemv.cu @ {arch}: {e}"));
        compile_ptx_only(&gsrc, "blockq_gemv_bf16.cu", &["-DSYN_ACT_BF16"], arch)
            .unwrap_or_else(|e| panic!("blockq_gemv_bf16.cu @ {arch}: {e}"));
    }
}

#[test]
fn blackwell_kernels_need_sm120a() {
    for (tag, src) in BLACKWELL_ONLY {
        compile(src, tag, "sm_120a").unwrap_or_else(|e| panic!("{tag} @ sm_120a: {e}"));
        for arch in ["sm_80", "sm_89", "sm_90a"] {
            assert!(
                compile(src, tag, arch).is_err(),
                "{tag} собрался под {arch} — из него выпал block-scale MMA?"
            );
        }
    }
}

#[test]
fn sm90_kernels_need_sm90() {
    for (tag, src) in SM90_ONLY {
        compile(src, tag, "sm_90a").unwrap_or_else(|e| panic!("{tag} @ sm_90a: {e}"));
        compile(src, tag, "sm_120a").unwrap_or_else(|e| panic!("{tag} @ sm_120a: {e}"));
        assert!(compile(src, tag, "sm_80").is_err(), "{tag} собрался под sm_80");
    }
}

#[test]
fn gemm_bf16_tma_family_follows_target() {
    let (tag, src) = PORTABLE[0];
    assert_eq!(tag, "best_cu/gemm/gemm_bf16.cu");
    let ptx80 = compile(src, tag, "sm_80").unwrap();
    assert!(!ptx80.contains("setmaxnreg"), "sm_80: TMA-семейство должно быть вырезано");
    assert!(!ptx80.contains("gn_bf16_tma_"), "sm_80: точек входа TMA быть не должно");
    assert!(ptx80.contains("gemm_bf16_"), "sm_80: cp.async-семейство на месте");
    let ptx120 = compile(src, tag, "sm_120a").unwrap();
    assert!(ptx120.contains("setmaxnreg"), "sm_120a: TMA-семейство должно остаться");
    assert!(ptx120.contains("gn_bf16_tma_64x64_s3"));
    // «Плоская» sm_120 (SYN_FORCE_ARCH=sm_120): без `a` — без TMA-ядер.
    let ptx120_plain = compile(src, tag, "sm_120").unwrap();
    assert!(!ptx120_plain.contains("setmaxnreg"));
}
