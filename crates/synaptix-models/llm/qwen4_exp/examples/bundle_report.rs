use std::collections::BTreeMap;
use std::path::PathBuf;

use synaptix_bundle::quant_layout::QuantManifest;
use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_llm_common::WeightSource;
use synaptix_llm_qwen4_exp::model::LM_PREFIX;
use synaptix_llm_qwen4_exp::Qwen4ExpWeights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("укажите путь к .syn или каталогу модели")?,
    );
    synaptix_kernels_cpu::ensure_registered();

    if path.is_file() {
        let bundle = Bundle::open(&path)?;
        match QuantManifest::read_from(&bundle) {
            Some(manifest) => report_quant(&manifest),
            None => println!("квант-манифеста нет — бандл плотный"),
        }
    }

    let weights = Qwen4ExpWeights::open(&path, Device::Cpu, DType::BF16)?;
    let cfg = weights.config.clone();
    println!(
        "\nконфиг: hidden {}, слоёв {}, экспертов {} (активно {}), окно {}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.moe.num_experts,
        cfg.moe.num_experts_per_tok,
        cfg.max_position_embeddings
    );

    let mut missing = Vec::new();
    let mut check = |key: String| {
        if !weights.contains(&key) {
            missing.push(key);
        }
    };
    check(format!("{LM_PREFIX}.embed_tokens.weight"));
    check("lm_head.weight".to_string());
    check(format!("{LM_PREFIX}.hyper_connection_mixer.hc_norm.weight"));
    for l in 0..cfg.num_hidden_layers {
        let p = format!("{LM_PREFIX}.layers.{l}");
        check(format!("{p}.mlp.gate.weight"));
        check(format!("{p}.mlp.experts.gate_up_proj"));
        check(format!("{p}.mlp.experts.down_proj"));
        check(format!("{p}.attn_hyper_connection.hc_norm.weight"));
        match cfg.layer_type(l) {
            synaptix_llm_qwen4_exp::LayerType::Linear => {
                check(format!("{p}.linear_attn.in_proj_qkv.weight"));
                check(format!("{p}.linear_attn.conv1d.weight"));
            }
            synaptix_llm_qwen4_exp::LayerType::Qsa => {
                check(format!("{p}.self_attn.q_proj.weight"));
                check(format!("{p}.self_attn.indexer.index_qk_proj.weight"));
            }
        }
    }
    if let Some(ple) = &cfg.ple {
        for layer in &ple.layer_ids {
            let p = format!("{LM_PREFIX}.layers.{layer}.ple");
            check(format!("{p}.key_proj.weight"));
            check(format!("{p}.conv1d.weight"));
            let table = format!("{p}.ple_embedding.ngram_embedding");
            let single = weights.contains(&format!("{table}.weight"));
            let sharded = weights.contains(&format!("{table}.shard_0.weight"));
            println!(
                "n-gram слоя {layer}: {}",
                if single {
                    "единой таблицей".to_string()
                } else if sharded {
                    format!("{} шардов", ple.split_parts)
                } else {
                    "НЕ НАЙДЕНА".to_string()
                }
            );
        }
    }

    if let Some(ple) = &cfg.ple {
        use std::time::Instant;
        let layer = ple.layer_ids[0];
        let table = weights.ngram_rows(layer)?;
        println!(
            "\nтаблица n-грамм: {} строк по {} значений",
            table.rows(),
            table.dim()
        );
        let ids: Vec<i64> = (0..512)
            .map(|i| (i as i64 * 611_953) % table.rows() as i64)
            .collect();
        let mut buf = vec![0f32; ids.len() * table.dim()];
        let t0 = Instant::now();
        table.gather_into(&ids, &mut buf)?;
        let cold = t0.elapsed();
        let t1 = Instant::now();
        table.gather_into(&ids, &mut buf)?;
        let warm = t1.elapsed();
        let finite = buf.iter().all(|x| x.is_finite());
        let nonzero = buf.iter().filter(|x| **x != 0.0).count();
        println!(
            "  {} строк: холодно {:.1} мс ({:.1} мкс/строка), повтор {:.1} мс; ненулевых значений {:.1}%, все конечны: {finite}",
            ids.len(),
            cold.as_secs_f32() * 1000.0,
            cold.as_secs_f32() * 1e6 / ids.len() as f32,
            warm.as_secs_f32() * 1000.0,
            100.0 * nonzero as f32 / buf.len() as f32,
        );
    }

    if missing.is_empty() {
        println!("все ожидаемые тензоры на месте");
    } else {
        println!("нет {} тензоров, первые:", missing.len());
        for key in missing.iter().take(10) {
            println!("  {key}");
        }
    }
    Ok(())
}

fn report_entry(manifest: &QuantManifest, name: &str) {
    match manifest.entry(name) {
        Some(e) => println!(
            "  {name}: {:?}, форма {:?}, packed {:.1} МБ, scales {:.1} МБ",
            e.kind(),
            e.shape,
            e.packed_bytes().unwrap_or(0) as f64 / (1 << 20) as f64,
            e.scales_bytes().unwrap_or(0) as f64 / (1 << 20) as f64,
        ),
        None => println!("  {name}: не квантован"),
    }
}

fn report_quant(manifest: &QuantManifest) {
    let mut by_kind: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for (name, entry) in manifest.tensors.iter() {
        let kind = entry
            .kind()
            .map(|k| format!("{k:?}"))
            .unwrap_or_else(|| "?".to_string());
        let bytes = entry.packed_bytes().unwrap_or(0) + entry.scales_bytes().unwrap_or(0);
        let slot = by_kind.entry(kind).or_insert((0, 0));
        slot.0 += 1;
        slot.1 += bytes;
        let _ = name;
    }
    println!("квант-манифест: {} тензоров", manifest.tensors.len());
    for (kind, (count, bytes)) in &by_kind {
        println!("  {kind}: {count} тензоров, {:.1} ГБ", *bytes as f64 / (1u64 << 30) as f64);
    }
    for name in [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    ] {
        report_entry(manifest, name);
    }
}
