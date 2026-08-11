use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_video_minimax_h3::config::{AudioVaeConfig, VaeConfig};
use synaptix_video_minimax_h3::loader::{ComponentLoader, H3Checkpoint, H3Paths};

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("H3_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join(".local/share/synthos/hf/MiniMax-H3")
        });
    if p.join("FL2VA").is_dir() || p.join("transformer").is_dir() {
        Some(p)
    } else {
        None
    }
}

fn open() -> Option<H3Checkpoint> {
    synaptix_kernels_cpu::ensure_registered();
    let dir = model_dir()?;
    let paths = H3Paths::open(&dir).expect("H3Paths::open");
    Some(H3Checkpoint::open(paths, Device::Cpu, DType::BF16).expect("H3Checkpoint::open"))
}

fn shape_of<'a>(ck: &'a H3Checkpoint, name: &str) -> Vec<usize> {
    ck.tensor_info(name)
        .unwrap_or_else(|| panic!("нет тензора {name}"))
        .shape
        .to_vec()
}

#[test]
fn config_matches_checkpoint() {
    let Some(ck) = open() else { return };
    let c = &ck.config;
    assert_eq!(c.hidden_size, 5376);
    assert_eq!(c.num_layers, 50);
    assert_eq!(c.token_refiner_num_layers, 2);
    assert_eq!(c.num_attention_heads, 56);
    assert_eq!(c.attention_head_dim, 128);
    assert_eq!(c.inner_dim(), 7168);
    assert_eq!(c.ffn_hidden_size, 14336);
    assert_eq!(c.latents_dim, 24);
    assert_eq!(c.audio_latents_dim, 32);
    assert_eq!(c.patch_size, [1, 2, 2]);
    assert_eq!(c.video_patch_dim(), 96);
    assert_eq!(c.text_dim, 5120);
    assert_eq!(c.time_embed_dim, 2688);
    assert_eq!(c.rope_inv_freq_len, 16);
    assert_eq!(c.rope_rot_dim(), 96);
    assert_eq!(c.sigma_shift_video, 12.0);
    assert_eq!(c.sigma_shift_audio, 3.0);
    assert!(c.tasks.iter().any(|t| t == "t2va"));
}

#[test]
fn dit_tensor_shapes_match_config() {
    let Some(ck) = open() else { return };
    let c = ck.config.clone();
    let hidden = c.hidden_size;
    let inner = c.inner_dim();

    assert_eq!(shape_of(&ck, "video_patch_proj.weight"), vec![hidden, c.video_patch_dim()]);
    assert_eq!(shape_of(&ck, "audio_patch_proj.weight"), vec![hidden, c.audio_latents_dim]);
    assert_eq!(shape_of(&ck, "condition_proj.weight"), vec![hidden, c.text_dim]);
    assert_eq!(shape_of(&ck, "rope.inv_freq"), vec![c.rope_inv_freq_len]);
    assert_eq!(
        shape_of(&ck, "time_embedder.proj_in.weight"),
        vec![c.time_embed_hidden_size, c.timestep_input_dim]
    );
    assert_eq!(
        shape_of(&ck, "time_embedder.proj_out.weight"),
        vec![c.time_embed_dim, c.time_embed_hidden_size]
    );

    for i in [0usize, 1] {
        let p = format!("token_refiner.blocks.{i}");
        assert_eq!(shape_of(&ck, &format!("{p}.attn.qkv_proj.weight")), vec![inner * 3, hidden]);
        assert_eq!(shape_of(&ck, &format!("{p}.attn.out_proj.weight")), vec![hidden, inner]);
        assert_eq!(shape_of(&ck, &format!("{p}.attn.q_norm.weight")), vec![c.attention_head_dim]);
        assert_eq!(shape_of(&ck, &format!("{p}.mlp.fc1.weight")), vec![c.ffn_hidden_size * 2, hidden]);
        assert_eq!(shape_of(&ck, &format!("{p}.mlp.fc2.weight")), vec![hidden, c.ffn_hidden_size]);
        assert_eq!(shape_of(&ck, &format!("{p}.norm1.weight")), vec![hidden]);
    }
    assert_eq!(shape_of(&ck, "token_refiner.final_norm.weight"), vec![hidden]);

    for i in [0usize, c.num_layers / 2, c.num_layers - 1] {
        let p = format!("blocks.{i}");
        assert_eq!(shape_of(&ck, &format!("{p}.attn.qkv_proj.weight")), vec![inner * 3, hidden]);
        assert_eq!(shape_of(&ck, &format!("{p}.attn.out_proj.weight")), vec![hidden, inner]);
        assert_eq!(shape_of(&ck, &format!("{p}.attn.q_norm.weight")), vec![c.attention_head_dim]);
        assert_eq!(shape_of(&ck, &format!("{p}.attn.k_norm.weight")), vec![c.attention_head_dim]);
        assert_eq!(shape_of(&ck, &format!("{p}.mlp.fc1.weight")), vec![c.ffn_hidden_size * 2, hidden]);
        assert_eq!(shape_of(&ck, &format!("{p}.mlp.fc2.weight")), vec![hidden, c.ffn_hidden_size]);
        assert_eq!(shape_of(&ck, &format!("{p}.norm1.weight")), vec![hidden]);
        assert_eq!(shape_of(&ck, &format!("{p}.norm2.weight")), vec![hidden]);
        assert_eq!(
            shape_of(&ck, &format!("{p}.adaln_proj.linear.weight")),
            vec![c.adaln_out_features, c.time_embed_dim]
        );
        assert_eq!(shape_of(&ck, &format!("{p}.adaln_proj.linear.bias")), vec![c.adaln_out_features]);
    }

    assert_eq!(
        shape_of(&ck, "final_layer.adaln_proj.linear.weight"),
        vec![c.final_adaln_out_features, c.time_embed_dim]
    );
    assert_eq!(shape_of(&ck, "final_layer.norm.weight"), vec![hidden]);
    assert_eq!(shape_of(&ck, "final_layer.video_out.weight"), vec![c.video_patch_dim(), hidden]);
    assert_eq!(shape_of(&ck, "final_layer.audio_out.weight"), vec![c.audio_latents_dim, hidden]);
}

#[test]
fn f32_islands_are_actually_f32() {
    let Some(ck) = open() else { return };
    for name in [
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_out.weight",
        "rope.inv_freq",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
    ] {
        let info = ck.tensor_info(name).unwrap_or_else(|| panic!("нет {name}"));
        assert_eq!(info.dtype, DType::F32, "{name} должен быть F32");
    }
    let blk = ck.tensor_info("blocks.0.attn.qkv_proj.weight").expect("blocks.0 qkv");
    assert_eq!(blk.dtype, DType::BF16);
}

#[test]
fn adaln_is_thirteen_billion_parameters() {
    let Some(ck) = open() else { return };
    let c = &ck.config;
    let per_block = c.adaln_out_features * c.time_embed_dim + c.adaln_out_features;
    let total = per_block * c.num_layers;
    assert!(
        (12.9e9..13.2e9).contains(&(total as f64)),
        "adaLN = {total} параметров, ожидалось ~13.0B"
    );

    let mut adaln_bytes = 0usize;
    let mut other_bytes = 0usize;
    for (name, dt, shape) in ck.infos() {
        let n: usize = shape.iter().product();
        let b = dt.bytes_for_numel(n);
        if name.starts_with("blocks.") && name.contains("adaln_proj") {
            adaln_bytes += b;
        } else {
            other_bytes += b;
        }
    }
    let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
    assert!(
        gb(adaln_bytes) > 23.0 && gb(adaln_bytes) < 25.5,
        "adaLN весит {:.1} ГБ, ожидалось ~24 ГБ (13B в bf16)",
        gb(adaln_bytes)
    );
    assert!(
        gb(other_bytes) > 34.0 && gb(other_bytes) < 40.0,
        "рабочие веса {:.1} ГБ, ожидалось ~36 ГБ (19.3B в bf16)",
        gb(other_bytes)
    );
    eprintln!(
        "[parity] adaLN {:.1} ГБ / рабочие {:.1} ГБ — предвычисление снимает {:.0}% весов",
        gb(adaln_bytes),
        gb(other_bytes),
        100.0 * adaln_bytes as f64 / (adaln_bytes + other_bytes) as f64
    );
}

#[test]
fn video_vae_tensors_match_config() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = model_dir() else { return };
    let paths = H3Paths::open(&dir).expect("paths");
    let cfg = VaeConfig::from_dir(&paths.root).expect("vae config");
    assert_eq!(cfg.z_channels, 24);
    assert_eq!(cfg.vae_ratio, 16);
    assert_eq!(cfg.vae_ratio_t, 4);
    assert_eq!(cfg.clip_length, 17);
    assert_eq!(cfg.token_drop, 3);
    assert_eq!(cfg.latents_mean.len(), 24);
    assert_eq!(cfg.vit_decoder_kwargs.num_layers, 36);
    assert_eq!(cfg.vit_decoder_kwargs.dim(), 2048);
    assert_eq!(cfg.vit_decoder_kwargs.rope_dim(), 48);

    let w = ComponentLoader::open_file(paths.video_vae_file(), Device::Cpu).expect("vae weights");
    let need = |k: &str| assert!(w.contains(k), "нет тензора VAE {k}");
    need("encoder.conv_in.weight");
    need("encoder.norm_out.weight");
    need("encoder.conv_out.weight");
    need("quant_conv.weight");
    need("post_quant_conv.weight");
    need("decoder.x_embedder.weight");
    need("decoder.register_tokens");
    need("decoder.norm_out.weight");
    need("decoder.proj_out.weight");
    for i in [0usize, 17, 35] {
        for suf in ["attn.to_qkv.weight", "attn.to_out.weight", "ff.w1.weight", "ff.w2.weight", "norm1.weight", "scale1"] {
            need(&format!("decoder.transformer_blocks.{i}.{suf}"));
        }
    }
    for lvl in 0..cfg.num_stages() {
        for b in 0..cfg.num_res_blocks {
            need(&format!("encoder.down.{lvl}.block.{b}.conv1.weight"));
            need(&format!("encoder.down.{lvl}.block.{b}.norm1.weight"));
        }
        if cfg.space_down[lvl] * cfg.time_down[lvl] > 1 {
            need(&format!("encoder.down.{lvl}.downsample.conv.weight"));
        }
    }
}

#[test]
fn audio_vae_tensors_match_config() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = model_dir() else { return };
    let paths = H3Paths::open(&dir).expect("paths");
    let cfg = AudioVaeConfig::from_dir(&paths.root).expect("audio vae config");
    assert_eq!(cfg.hop_length(), 800);
    assert_eq!(cfg.latents_per_second(), 40);
    assert_eq!(cfg.vae_latent_channels, 32);
    assert_eq!(cfg.latent_dim, 2048);
    assert_eq!(cfg.decoder_rates, vec![5, 5, 2, 2, 2, 2, 2]);

    assert_eq!(cfg.latents_mean.len(), 32, "latents_mean живут в config.json, не в весах");
    assert_eq!(cfg.latents_std.len(), 32);

    let w = ComponentLoader::open_file(paths.audio_vae_file(), Device::Cpu).expect("audio weights");
    let need = |k: &str| assert!(w.contains(k), "нет тензора audio VAE {k}");
    need("dec_in_proj.weight");
    need("decoder.conv_pre.weight_v");
    need("decoder.conv_post.weight_v");
    need("decoder.activation_post.act.alpha");
    need("mean_proj.weight");
    need("pre_block.attn.qkv.weight");
    need("pre_block.attn.q_bias");
    need("pre_block.attn.zero_k_bias");
    need("pre_block.mlp.w0.weight");
    need("encoder.block.0.weight_v");
    for i in 0..cfg.decoder_rates.len() {
        need(&format!("decoder.ups.{i}.0.weight_v"));
    }
    for i in 0..cfg.decoder_rates.len() * cfg.resblock_kernel_sizes.len() {
        need(&format!("decoder.resblocks.{i}.convs1.0.weight_v"));
        need(&format!("decoder.resblocks.{i}.activations.0.act.alpha"));
    }
    for lvl in 1..=cfg.encoder_rates.len() {
        need(&format!("encoder.block.{lvl}.block.4.weight_v"));
        for u in 0..3 {
            need(&format!("encoder.block.{lvl}.block.{u}.block.1.weight_v"));
        }
    }
}

#[test]
fn turbo_lora_targets_expected_modules() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = model_dir() else { return };
    let lora = dir.join("lora/turbo/minimax_h3_turbo_v4_step600_ema.safetensors");
    if !lora.exists() {
        return;
    }
    let Some(ck) = open() else { return };
    let lw = synaptix_video_minimax_h3::LoraWeights::open(&lora, Device::Cpu, 1.0)
        .expect("lora open");
    let ck = ck.with_lora(std::sync::Arc::new(lw));

    let mut hit = 0usize;
    let mut adaln_hit = 0usize;
    for i in 0..ck.config.num_layers {
        for key in [
            format!("blocks.{i}.attn.qkv_proj"),
            format!("blocks.{i}.attn.out_proj"),
            format!("blocks.{i}.mlp.fc1"),
            format!("blocks.{i}.mlp.fc2"),
        ] {
            if ck.lora_delta(&key, DType::F32).expect("delta").is_some() {
                hit += 1;
            }
        }
        if ck
            .lora_delta(&format!("blocks.{i}.adaln_proj.linear"), DType::F32)
            .expect("delta")
            .is_some()
        {
            adaln_hit += 1;
        }
    }
    eprintln!("[parity] Turbo LoRA: {hit} линеек блоков, {adaln_hit} adaLN-веток");
    assert!(hit > 0, "LoRA не попала ни в одну линейку — проверь префиксы ключей");
}
