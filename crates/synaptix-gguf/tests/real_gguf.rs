//! Реальные GGUF-файлы llama.cpp (пропускаются, если файлов нет):
//! `~/Storage/syn_models/gguf-tests/{Qwen3-0.6B-Q8_0, gemma-3-1b-it-Q8_0,
//! Llama-3.2-1B-Instruct-Q4_K_M}.gguf` и Qwen2 Q4_0 из Downloads.
//!
//! Проверяется: план строится, config.json синтезируется с нужным
//! `model_type`, квантованные веса отдаются блоками с верной формой, плотный
//! деквант совпадает с CPU-эталоном, а синтезированный tokenizer.json
//! токенизирует так же, как `llama-tokenize` (если бинарь есть в PATH).

use std::path::PathBuf;
use std::process::Command;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_gguf::GgufSource;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

fn models() -> Vec<(PathBuf, &'static str)> {
    let d = home().join("Storage/syn_models/gguf-tests");
    vec![
        (d.join("Qwen3-0.6B-Q8_0.gguf"), "qwen3"),
        (d.join("gemma-3-1b-it-Q8_0.gguf"), "gemma3_text"),
        (d.join("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), "llama"),
        (home().join("Storage/vitamindb/vitamindb/Downloads/ggml-model-Q4_0.gguf"), "qwen2"),
    ]
    .into_iter()
    .filter(|(p, _)| p.is_file())
    .collect()
}

#[test]
fn plans_configs_and_quant_weights() {
    let ms = models();
    if ms.is_empty() {
        eprintln!("нет эталонных GGUF — пропуск");
        return;
    }
    for (path, want_type) in ms {
        let src = GgufSource::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let cfg: serde_json::Value = serde_json::from_slice(src.file("config.json").unwrap()).unwrap();
        assert_eq!(cfg["model_type"], want_type, "{}", path.display());
        assert!(src.file("tokenizer.json").is_some() && src.file("tokenizer_config.json").is_some());
        let layers = cfg["num_hidden_layers"].as_u64().unwrap() as usize;
        assert!(layers > 0);
        // Первая проекция q — квантованный вес правильной формы.
        let q = "model.layers.0.self_attn.q_proj.weight";
        let (slices, n, k) = src.quant_dims(q).unwrap_or_else(|| panic!("{}: q_proj не квант", path.display()));
        assert_eq!(slices, 1);
        assert_eq!(k, cfg["hidden_size"].as_u64().unwrap() as usize);
        let heads = cfg["num_attention_heads"].as_u64().unwrap() as usize;
        let hd = cfg["head_dim"].as_u64().unwrap() as usize;
        assert_eq!(n, heads * hd, "{}: q_proj N", path.display());
        let qw = src.load_quant(q, Device::Cpu).unwrap().unwrap();
        assert!(matches!(qw.dtype(), DType::Ggml(_)));
        // Плотная норма — плавающая, без кванта.
        assert!(src.quant_dims("model.layers.0.input_layernorm.weight").is_none());
        let norm = src.load_to("model.layers.0.input_layernorm.weight", Device::Cpu, DType::F32).unwrap();
        assert_eq!(norm.dims(), &[k]);
        // Плотный деквант квантованного веса на CPU совпадает с эталоном core.
        let dense = src.load_to(q, Device::Cpu, DType::F32).unwrap();
        assert_eq!(dense.dims(), &[n, k]);
        let v = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let blob = qw.packed_arc().unwrap();
        let blob = blob.as_cpu().unwrap().as_bytes();
        let rb = qw.block_row_bytes().unwrap();
        let mut want = vec![0f32; k];
        synaptix_core::quant::dequant_row_f32(qw.dtype(), &blob[..rb], k, &mut want).unwrap();
        assert_eq!(&v[..k], &want[..], "{}: строка 0 деквант", path.display());
        eprintln!("[{}] {} слоёв, q_proj {:?} [{n}, {k}]", path.file_name().unwrap().to_string_lossy(), layers, qw.dtype());
    }
}

#[test]
fn tokenizer_matches_llama_cpp() {
    let ms = models();
    if ms.is_empty() {
        return;
    }
    let bin = ["/usr/bin/llama-tokenize", "/usr/local/bin/llama-tokenize"].iter().find(|p| std::path::Path::new(p).is_file());
    let Some(bin) = bin else {
        eprintln!("llama-tokenize не найден — пропуск");
        return;
    };
    let texts = [
        "The capital of France is Paris.",
        "Столица Франции — Париж, 2026 год!",
        "  leading spaces\nnew line\ttab 12345 x=1;",
    ];
    for (path, _) in ms {
        let src = GgufSource::open(&path).unwrap();
        let tk = tokenizers::Tokenizer::from_bytes(src.file("tokenizer.json").unwrap()).unwrap();
        for text in texts {
            let out = Command::new(bin)
                .args(["-m", path.to_str().unwrap(), "-p", text, "--ids", "--no-bos", "--no-parse-special", "--log-disable"])
                .output()
                .expect("llama-tokenize");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('['));
            let Some(line) = line else {
                panic!("{}: llama-tokenize не дал ids: {stdout}", path.display());
            };
            let want: Vec<u32> = line.trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let got = tk.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(got, want, "{}: {text:?}", path.display());
        }
        eprintln!("[{}] токенизация совпала", path.file_name().unwrap().to_string_lossy());
    }
}
