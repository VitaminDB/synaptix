//! Фаза 1: проверка раскладки весов LTX-2.3 22B distilled против разобранной
//! карты чекпойнта. Открывает реальный файл zero-copy (mmap), читает конфиг из
//! `__metadata__` и сверяет per-prefix счётчики, dtype-гистограмму и
//! формы/dtype ключевых тензоров. Численного эталона не требует.
//!
//! Тест пропускается, если веса не лежат локально.

use std::collections::BTreeMap;
use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_video_ltx23::loader::{
    LtxCheckpoint, AUDIO_VAE_PREFIX, DIT_PREFIX, TEXT_PROJ_PREFIX, VAE_PREFIX, VOCODER_PREFIX,
};

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";

#[test]
fn layout_map_matches_checkpoint() {
    if !Path::new(CKPT).exists() {
        eprintln!("skip layout_map_matches_checkpoint: weights absent at {CKPT}");
        return;
    }
    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).expect("open checkpoint");

    // --- config из __metadata__ ---
    assert_eq!(ckpt.config.model_version, "2.3.0");
    let t = &ckpt.config.transformer;
    assert_eq!(t.num_layers, 48);
    assert_eq!(t.num_attention_heads, 32);
    assert_eq!(t.attention_head_dim, 128);
    assert_eq!(t.inner_dim(), 4096);
    assert_eq!(t.ff_inner_dim(), 16384);
    assert_eq!(t.audio_num_attention_heads, 32);
    assert_eq!(t.audio_attention_head_dim, 64);
    assert_eq!(t.audio_inner_dim(), 2048);
    assert_eq!(t.audio_ff_inner_dim(), 8192);
    assert_eq!(t.in_channels, 128);
    assert_eq!(t.out_channels, 128);
    assert_eq!(t.cross_attention_dim, 4096);
    assert_eq!(t.audio_cross_attention_dim, 2048);
    assert_eq!(t.caption_channels, 3840);
    assert_eq!(t.activation_fn, "gelu-approximate");
    assert_eq!(t.qk_norm, "rms_norm");
    assert_eq!(t.rope_type, "split");
    assert_eq!(t.frequencies_precision, "float64");
    assert!(t.apply_gated_attention);
    assert!(t.cross_attention_adaln);
    assert!(t.use_audio_video_cross_attention);
    assert!(t.use_embeddings_connector);
    assert_eq!(t.connector_num_layers, 8);
    assert_eq!(t.connector_num_learnable_registers, 128);
    assert_eq!(t.positional_embedding_max_pos, vec![20, 2048, 2048]);
    assert_eq!(t.audio_positional_embedding_max_pos, vec![20]);
    assert!((t.norm_eps - 1e-6).abs() < 1e-12, "norm_eps = {} (want 1e-6)", t.norm_eps);
    assert_eq!(t.positional_embedding_theta, 10000.0);
    // 188160 = caption_channels(3840) * 49 hidden-states (Gemma-3-12B: 48 слоёв + embed)
    assert_eq!(t.text_aggregate_in(49), 188_160);

    let v = &ckpt.config.vae;
    assert_eq!(v.dims, 3);
    assert_eq!(v.latent_channels, 128);
    assert_eq!(v.in_channels, 3);
    assert_eq!(v.out_channels, 3);
    assert_eq!(v.patch_size, 4);
    assert_eq!(v.norm_layer, "pixel_norm");
    assert_eq!(v.spatial_padding_mode, "zeros");
    assert_eq!(v.encoder_blocks.len(), 9);
    assert_eq!(v.decoder_blocks.len(), 9);
    assert_eq!(v.encoder_blocks[0].0, "res_x");
    assert_eq!(v.encoder_blocks[0].1.num_layers, Some(4));

    assert_eq!(ckpt.config.scheduler.num_train_timesteps, 1000);
    assert_eq!(ckpt.config.scheduler.sampler, "LinearQuadratic");

    // --- per-prefix счётчики + dtype-гистограмма ---
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0usize;
    for (name, dtype, _shape) in ckpt.infos() {
        let pfx = name.split('.').next().unwrap_or("").to_string();
        *counts.entry(pfx).or_default() += 1;
        *dtypes.entry(format!("{dtype:?}")).or_default() += 1;
        total += 1;
    }
    assert_eq!(total, 5947, "total tensor count");
    assert_eq!(counts.get("model"), Some(&4444), "model.diffusion_model count");
    assert_eq!(counts.get("vocoder"), Some(&1227), "vocoder count");
    assert_eq!(counts.get("vae"), Some(&170), "vae count");
    assert_eq!(counts.get("audio_vae"), Some(&102), "audio_vae count");
    assert_eq!(counts.get("text_embedding_projection"), Some(&4), "text_proj count");
    assert_eq!(dtypes.get("BF16"), Some(&5657), "BF16 count");
    assert_eq!(dtypes.get("F32"), Some(&290), "F32 count");

    // --- per-block adaLN scale_shift таблицы: F32 с точными формами ---
    let blk = format!("{DIT_PREFIX}.transformer_blocks.0");
    for (suffix, rows, dim) in [
        ("scale_shift_table", 9usize, 4096usize),
        ("audio_scale_shift_table", 9, 2048),
        ("prompt_scale_shift_table", 2, 4096),
        ("audio_prompt_scale_shift_table", 2, 2048),
        ("scale_shift_table_a2v_ca_video", 5, 4096),
        ("scale_shift_table_a2v_ca_audio", 5, 2048),
    ] {
        let info = ckpt
            .tensor_info(&format!("{blk}.{suffix}"))
            .unwrap_or_else(|| panic!("missing {suffix}"));
        assert_eq!(info.dtype, DType::F32, "{suffix} dtype");
        assert_eq!(info.shape, vec![rows, dim], "{suffix} shape");
    }
    // глобальные финальные таблицы
    let g = ckpt.tensor_info(&format!("{DIT_PREFIX}.scale_shift_table")).unwrap();
    assert_eq!((g.dtype, g.shape), (DType::F32, vec![2, 4096]));
    let ga = ckpt.tensor_info(&format!("{DIT_PREFIX}.audio_scale_shift_table")).unwrap();
    assert_eq!((ga.dtype, ga.shape), (DType::F32, vec![2, 2048]));

    // --- репрезентативные BF16-тензоры DiT (формы из карты раскладки) ---
    let dit = ckpt.component(DIT_PREFIX);
    let shape = |n: &str| dit.tensor_info(n).unwrap_or_else(|| panic!("missing {n}")).shape;
    assert_eq!(shape("patchify_proj.weight"), vec![4096, 128]);
    assert_eq!(shape("audio_patchify_proj.weight"), vec![2048, 128]);
    assert_eq!(shape("proj_out.weight"), vec![128, 4096]);
    assert_eq!(shape("audio_proj_out.weight"), vec![128, 2048]);
    assert_eq!(shape("adaln_single.linear.weight"), vec![36864, 4096]); // 9*4096
    assert_eq!(shape("adaln_single.emb.timestep_embedder.linear_1.weight"), vec![4096, 256]);
    assert_eq!(shape("transformer_blocks.0.attn1.to_q.weight"), vec![4096, 4096]);
    assert_eq!(shape("transformer_blocks.0.attn1.q_norm.weight"), vec![4096]);
    assert_eq!(shape("transformer_blocks.0.attn1.to_gate_logits.weight"), vec![32, 4096]);
    assert_eq!(shape("transformer_blocks.0.ff.net.0.proj.weight"), vec![16384, 4096]);
    assert_eq!(shape("transformer_blocks.0.ff.net.2.weight"), vec![4096, 16384]);
    assert_eq!(shape("transformer_blocks.0.audio_to_video_attn.to_q.weight"), vec![2048, 4096]);
    assert_eq!(shape("transformer_blocks.0.video_to_audio_attn.to_q.weight"), vec![2048, 2048]);

    // последний блок присутствует (48 блоков: 0..47)
    assert!(dit.contains("transformer_blocks.47.attn1.to_q.weight"));
    assert!(!dit.contains("transformer_blocks.48.attn1.to_q.weight"));

    // --- text_embedding_projection: 188160 → 4096/2048 ---
    let tp = ckpt.component(TEXT_PROJ_PREFIX);
    assert_eq!(tp.tensor_info("video_aggregate_embed.weight").unwrap().shape, vec![4096, 188_160]);
    assert_eq!(tp.tensor_info("audio_aggregate_embed.weight").unwrap().shape, vec![2048, 188_160]);

    // --- VAE / audio_vae / vocoder присутствуют (формы — на своих фазах) ---
    assert!(ckpt.component(VAE_PREFIX).contains("encoder.conv_in.conv.bias"));
    assert!(ckpt.component(VAE_PREFIX).contains("per_channel_statistics.mean-of-means"));
    assert!(ckpt.component(AUDIO_VAE_PREFIX).contains("encoder.conv_in.conv.bias"));
    assert_eq!(
        ckpt.component(VOCODER_PREFIX).tensor_info("mel_stft.mel_basis").unwrap().shape,
        vec![64, 257]
    );

    eprintln!(
        "LTX-2.3 layout OK: v{} | {total} tensors | DiT {} / vocoder {} / vae {} / audio_vae {} / text_proj {}",
        ckpt.config.model_version,
        counts["model"], counts["vocoder"], counts["vae"], counts["audio_vae"],
        counts["text_embedding_projection"],
    );
}
