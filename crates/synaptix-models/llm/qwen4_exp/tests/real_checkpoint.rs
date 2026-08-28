use std::path::PathBuf;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_llm_common::WeightSource;
use synaptix_llm_qwen4_exp::config::LayerType;
use synaptix_llm_qwen4_exp::model::LM_PREFIX;
use synaptix_llm_qwen4_exp::ngram::{head_vocabs, layer_multipliers, NGramEmbedding};
use synaptix_llm_qwen4_exp::Qwen4ExpWeights;

fn model_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        std::env::var("SYN_QWEN4EXP_MODEL")
            .unwrap_or_else(|_| "/home/master/models/Qwen/Qwen3.8-Flash-Next".to_string()),
    );
    p.join("config.json").exists().then_some(p)
}

#[test]
fn reads_real_checkpoint_layout() {
    let Some(dir) = model_dir() else {
        eprintln!("чекпойнт Qwen3.8-Flash-Next не найден — пропуск");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let weights = Qwen4ExpWeights::open(&dir, Device::Cpu, DType::BF16).expect("open");
    let cfg = weights.config.clone();

    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.hc_hidden(), 10240);
    assert_eq!(cfg.moe.num_experts, 512);
    assert_eq!(cfg.moe.num_experts_per_tok, 10);
    assert_eq!(cfg.indexer.budget, 2048);
    assert_eq!(cfg.indexer.block_topk(), 512);
    assert_eq!(cfg.rope.rotary_dim, 64);
    assert!(cfg.output_gate_sigmoid);
    assert_eq!(
        cfg.layer_types.iter().filter(|t| **t == LayerType::Qsa).count(),
        12
    );
    let ple = cfg.ple.as_ref().expect("ple");
    assert_eq!(ple.layer_ids, vec![1]);
    assert_eq!(ple.split_parts, 128);
    assert_eq!(ple.head_dim(), 160);

    let shape = |key: &str| -> Vec<usize> {
        weights
            .tensor(key, Device::Cpu, DType::BF16)
            .unwrap_or_else(|e| panic!("{key}: {e}"))
            .dims()
            .to_vec()
    };
    assert_eq!(shape(&format!("{LM_PREFIX}.layers.0.linear_attn.in_proj_qkv.weight")), vec![10240, 2560]);
    assert_eq!(shape(&format!("{LM_PREFIX}.layers.3.self_attn.q_proj.weight")), vec![12288, 2560]);
    assert_eq!(shape(&format!("{LM_PREFIX}.layers.3.self_attn.indexer.index_qk_proj.weight")), vec![640, 2560]);
    assert_eq!(shape(&format!("{LM_PREFIX}.layers.0.attn_hyper_connection.input_mix_weight_down.weight")), vec![320, 10240]);
    assert_eq!(shape(&format!("{LM_PREFIX}.layers.0.attn_hyper_connection.block_inject_weight.weight")), vec![4, 10240]);
    assert_eq!(shape(&format!("{LM_PREFIX}.hyper_connection_mixer.hc_norm.weight")), vec![10240]);
    assert_eq!(shape(&format!("{LM_PREFIX}.layers.1.ple.key_proj.weight")), vec![10240, 2560]);

    let expert_stack = format!("{LM_PREFIX}.layers.0.mlp.experts.gate_up_proj");
    assert!(weights.contains(&expert_stack));
}

#[test]
fn ngram_buffers_match_checkpoint() {
    let Some(dir) = model_dir() else {
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let weights = Qwen4ExpWeights::open(&dir, Device::Cpu, DType::BF16).expect("open");
    let cfg = weights.config.clone();
    let ple = cfg.ple.as_ref().expect("ple");
    let layer = ple.layer_ids[0];
    let prefix = format!("{LM_PREFIX}.layers.{layer}.ple.ple_embedding");

    let read = |name: &str| -> Vec<i64> {
        weights
            .tensor(&format!("{prefix}.{name}"), Device::Cpu, DType::I64)
            .expect(name)
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap()
    };

    let (sizes, offsets, total) = head_vocabs(ple, 0);
    assert_eq!(read("ngram_heads_vocab_sizes"), sizes);
    assert_eq!(read("ngram_heads_offsets"), offsets);
    assert_eq!(
        read("layer_multipliers"),
        layer_multipliers(cfg.vocab_size, ple.ngram_size, 0, ple.seed)
    );

    let padded = total.div_ceil(ple.make_vocab_divisible_by as u64) * ple.make_vocab_divisible_by as u64;
    assert_eq!(padded % ple.split_parts as u64, 0);
    let rows_per_shard = padded / ple.split_parts as u64;
    assert_eq!(rows_per_shard, 2_500_012);
}

#[test]
fn ngram_gather_streams_from_disk() {
    let Some(dir) = model_dir() else {
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let weights = Qwen4ExpWeights::open(&dir, Device::Cpu, DType::F32).expect("open");
    let cfg = weights.config.clone();
    let ple = cfg.ple.as_ref().expect("ple").clone();
    let layer = ple.layer_ids[0];

    let table = weights.ngram_rows(layer).expect("таблица n-грамм");
    assert_eq!(table.dim(), 160);
    assert_eq!(table.rows(), 320_001_536);

    let embedding = NGramEmbedding::new(
        &ple,
        0,
        cfg.vocab_size,
        cfg.eos_token_ids.first().copied().unwrap_or(0),
        table,
        None,
        Device::Cpu,
        DType::F32,
    )
    .expect("n-gram");

    let tokens: Vec<u32> = (0..32).map(|i| (i * 7919 + 13) as u32 % 248_320).collect();
    let mut history = vec![embedding.eos(); embedding.context_len()];
    history.extend_from_slice(&tokens);

    let ids = embedding.ids_for(&history, tokens.len());
    assert_eq!(ids.len(), tokens.len() * 16);
    assert!(ids.iter().all(|id| *id >= 0 && (*id as usize) < 320_001_536));

    let t0 = Instant::now();
    let out = embedding.forward(&history, tokens.len()).expect("gather");
    let elapsed = t0.elapsed();
    assert_eq!(out.dims(), &[tokens.len(), 2560]);
    let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(v.iter().any(|x| *x != 0.0), "все строки нулевые");
    assert!(v.iter().all(|x| x.is_finite()));
    eprintln!(
        "n-gram: {} токенов × 16 голов за {:.1} мс",
        tokens.len(),
        elapsed.as_secs_f32() * 1000.0
    );
}
