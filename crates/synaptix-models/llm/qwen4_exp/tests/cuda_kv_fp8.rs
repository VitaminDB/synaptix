use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::RopePositions;
use synaptix_llm_common::mrope;
use synaptix_llm_common::weights::WeightSource;
use synaptix_llm_common::ModelError;
use synaptix_llm_qwen4_exp::attention::{KvLayer, QsaAttention};
use synaptix_llm_qwen4_exp::config::Qwen4ExpConfig;

const CFG: &str = r#"{
    "model_type": "qwen4_exp",
    "text_config": {
        "full_attention_interval": 2, "hc_count": 2, "hc_lowrank": 32,
        "head_dim": 64, "heads_per_ngram": 2, "hidden_act": "silu",
        "hidden_size": 256, "indexer_budget": 64, "indexer_compress_ratio": 4,
        "indexer_head_dim": 64, "indexer_kv_heads": 1, "indexer_n_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_value_head_dim": 32, "make_ngram_vocab_size_divisible_by": 128,
        "max_position_embeddings": 4096, "moe_intermediate_size": 64,
        "ngram_size": 3, "ngram_vocab_size_base": 20000,
        "num_attention_heads": 8, "num_experts": 4, "num_experts_per_tok": 2,
        "num_hidden_layers": 2, "num_key_value_heads": 2, "output_gate_type": "sigmoid",
        "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 256,
        "ple_layer_ids": [], "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [4, 4, 8],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000,
                            "rope_type": "default"},
        "shared_expert_intermediate_size": 64, "split_ngram_parts": 1,
        "tie_word_embeddings": false, "vocab_size": 512, "eos_token_id": 1
    }
}"#;

struct Fake {
    rows: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Fake {
    fn new(cfg: &Qwen4ExpConfig) -> Self {
        let (h, nh, nkv, hd) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let idx = cfg.indexer;
        let mut rows = HashMap::new();
        let mut put = |key: &str, dims: Vec<usize>, seed: u64| {
            let n: usize = dims.iter().product();
            rows.insert(key.to_string(), (dims, noise(seed, n)));
        };
        put("qsa.q_proj.weight", vec![nh * 2 * hd, h], 1);
        put("qsa.k_proj.weight", vec![nkv * hd, h], 2);
        put("qsa.v_proj.weight", vec![nkv * hd, h], 3);
        put("qsa.o_proj.weight", vec![h, nh * hd], 4);
        put("qsa.q_norm.weight", vec![hd], 5);
        put("qsa.k_norm.weight", vec![hd], 6);
        put(
            "qsa.indexer.index_qk_proj.weight",
            vec![(idx.n_heads + idx.kv_heads) * idx.head_dim, h],
            7,
        );
        put("qsa.indexer.q_layernorm.weight", vec![idx.head_dim], 8);
        put("qsa.indexer.k_layernorm.weight", vec![idx.head_dim], 9);
        Self { rows }
    }
}

impl WeightSource for Fake {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        let (dims, data) = self
            .rows
            .get(key)
            .ok_or_else(|| ModelError::Load(format!("нет веса {key}")))?;
        Tensor::from_vec(data.clone(), dims.clone(), Device::Cpu)
            .and_then(|t| t.to_dtype(dtype))
            .and_then(|t| t.to_device(device))
            .map_err(|e| ModelError::Load(e.to_string()))
    }

    fn contains(&self, key: &str) -> bool {
        self.rows.contains_key(key)
    }
}

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.4
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .expect("на хост")
}

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|x| (*x as f64).powi(2)).sum();
    (num / den.max(1e-12)).sqrt() as f32
}

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

const S: usize = 192;

/// Прогон слоя QSA: `quant` — держать KV в fp8, `tables` — задать позиции
/// таблицами M-RoPE вместо последовательных.
fn run(cfg: &Qwen4ExpConfig, attn: &QsaAttention, device: Device, quant: bool, tables: bool) -> Vec<f32> {
    run_seed(cfg, attn, device, quant, tables, 42)
}

fn run_seed(
    cfg: &Qwen4ExpConfig,
    attn: &QsaAttention,
    device: Device,
    quant: bool,
    tables: bool,
    seed: u64,
) -> Vec<f32> {
    let h = Tensor::from_vec(noise(seed, S * cfg.hidden_size), vec![S, cfg.hidden_size], Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_device(device))
        .expect("вход");
    let mut kv = if quant {
        KvLayer::new_mxfp8(cfg.num_key_value_heads, cfg.head_dim, S + 8, device).expect("kv fp8")
    } else {
        KvLayer::new(cfg.num_key_value_heads, cfg.head_dim, S + 8, device, DType::F32)
            .expect("kv")
    };
    let mut idx = attn.indexer.make_cache(S + 8).expect("кэш индексатора");
    let rope = synaptix_ops::pos::rope_cache::RopeCache::new(
        cfg.rope.rotary_dim.max(2),
        S + 8,
        cfg.rope.theta,
        device,
    )
    .expect("rope");

    let built = tables.then(|| {
        let pos: Vec<[u32; 3]> = (0..S as u32).map(|p| [p, p, p]).collect();
        let inv = cfg.rope.inv_freqs();
        let section = cfg.rope.mrope_section.expect("mrope_section");
        let (cos, sin) = mrope::rope_tables(&pos, &inv, &section, cfg.rope.mrope_interleaved);
        let half = inv.len();
        let make = |v: Vec<f32>| {
            Tensor::from_vec(v, vec![S, half], device).expect("таблица")
        };
        (make(cos), make(sin))
    });
    let pos = match &built {
        Some((cos, sin)) => RopePositions::Tables { cos, sin },
        None => RopePositions::Sequential,
    };
    let (out, selected) = attn.forward(&h, &mut kv, &mut idx, 0, S, &rope, pos).expect("forward");
    assert!(selected.is_some(), "селекция не включилась — сравнивать нечего");
    host(&out)
}

fn build(device: Device) -> (Qwen4ExpConfig, QsaAttention) {
    let cfg = Qwen4ExpConfig::from_hf_bytes(CFG.as_bytes()).expect("конфиг");
    let src = Fake::new(&cfg);
    let attn = QsaAttention::load(&src, "qsa", &cfg, device, DType::F32, DType::F32)
        .expect("слой QSA");
    (cfg, attn)
}

/// KV в fp8 против плотного: тот же слой, тот же вход, разница — только в
/// том, чем хранится кэш.
#[test]
fn kv_fp8_matches_dense() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let device = Device::Cuda(0);
    let (cfg, attn) = build(device);
    let dense = run(&cfg, &attn, device, false, false);
    let quant = run(&cfg, &attn, device, true, false);
    let rel = rel_l2(&quant, &dense);
    // Контроль: на другом входе тот же слой отвечает совсем иначе, так что
    // близость fp8 к плотному — не свойство метрики.
    let other = run_seed(&cfg, &attn, device, false, false, 43);
    let apart = rel_l2(&other, &dense);
    eprintln!("KV в fp8 против плотного: rel_l2={rel:.3e}, чужой вход даёт {apart:.3e}");
    assert!(apart > 10.0 * rel, "метрика не различает входы: {apart:.3e} против {rel:.3e}");
    // Цена самого кванта: E4M3 держит три бита мантиссы, масштаб — на
    // тридцать два элемента; на случайных весах скоры почти равны и
    // возмущение видно сильнее, чем на обученных.
    assert!(rel < 6e-2, "fp8-кэш разошёлся с плотным: {rel:.3e}");
}

/// Позиции таблицами M-RoPE на чистом тексте обязаны дать то же, что и
/// последовательные: у текстового токена все три оси равны его индексу.
#[test]
fn mrope_tables_match_sequential_on_text() {
    if !ready() {
        return;
    }
    let device = Device::Cuda(0);
    let (cfg, attn) = build(device);
    let plain = run(&cfg, &attn, device, false, false);
    let tabled = run(&cfg, &attn, device, false, true);
    let rel = rel_l2(&tabled, &plain);
    eprintln!("таблицы M-RoPE против последовательных позиций: rel_l2={rel:.3e}");
    assert!(rel < 1e-6, "M-RoPE сдвинул текстовый путь: {rel:.3e}");
}
