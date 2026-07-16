use std::path::PathBuf;

use synaptix_core::dtype::DType;
use synaptix_core::precision::parse_dtype;
use synaptix_llm_qwen3::pipeline::{GenerationConfig, Qwen3Pipeline};

use crate::commands::device::resolve as resolve_device;

pub struct BenchArgs {
    pub model: PathBuf,
    pub n_tokens: usize,
    pub prompt_tokens: usize,
    pub batch_size: usize,
    pub warmup: usize,
    pub device: String,
    pub attn: Option<String>,
    pub dtype: Option<String>,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            n_tokens: 128,
            prompt_tokens: 0,
            batch_size: 1,
            warmup: 3,
            device: "cuda".to_string(),
            attn: None,
            dtype: None,
        }
    }
}

pub fn run(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device = resolve_device(&args.device);
    crate::commands::device::resolve_attn(args.attn.as_deref());
    if !args.model.exists() {
        return Err(format!("model path not found: {}", args.model.display()).into());
    }
    if args.batch_size != 1 {
        eprintln!("warning: batch_size > 1 не поддерживается в MVP — фиксируем batch=1");
    }
    let dtype = match args.dtype.as_deref() {
        Some(s) => parse_dtype(s).ok_or_else(|| format!("bad --dtype '{s}' (f32|bf16|f16)"))?,
        None => DType::BF16,
    };

    eprintln!("synaptix bench: loading {} ({:?}, {:?})", args.model.display(), dtype, device);
    let pipeline = Qwen3Pipeline::load(&args.model, device, dtype)
        .map_err(|e| format!("load: {e}"))?;
    let prompt = "Привет, ";
    let mut prompt_ids = pipeline.encode(prompt).map_err(|e| format!("tokenize: {e}"))?;
    if args.prompt_tokens > prompt_ids.len() {
        let pad_id = *prompt_ids.last().unwrap_or(&0u32);
        prompt_ids.resize(args.prompt_tokens, pad_id);
    }

    eprintln!("synaptix bench: warmup ({} iterations)", args.warmup);
    for i in 0..args.warmup {
        let cfg = GenerationConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            seed: i as u64,
            ..Default::default()
        };
        let _ = pipeline.generate(&prompt_ids, cfg).map_err(|e| format!("warmup: {e}"))?;
    }

    eprintln!("synaptix bench: measuring {} new tokens", args.n_tokens);
    let cfg = GenerationConfig {
        max_new_tokens: args.n_tokens,
        temperature: 0.0,
        ..Default::default()
    };
    let (new_ids, stats) = pipeline
        .generate(&prompt_ids, cfg)
        .map_err(|e| format!("generate: {e}"))?;

    let prefill_tps = if stats.prefill_ms > 0 {
        (stats.prompt_tokens as f32) / (stats.prefill_ms as f32 / 1000.0)
    } else {
        0.0
    };
    let decode_tps = if stats.decode_ms > 0 && new_ids.len() > 1 {
        ((new_ids.len().saturating_sub(1)) as f32) / (stats.decode_ms as f32 / 1000.0)
    } else {
        0.0
    };

    println!("synaptix bench {} ({:?}):", args.model.display(), dtype);
    println!("  prompt:  {} tokens, {} ms ({:.2} tok/s)", stats.prompt_tokens, stats.prefill_ms, prefill_tps);
    println!("  decode:  {} tokens, {} ms ({:.2} tok/s)", new_ids.len(), stats.decode_ms, decode_tps);
    println!("  total:   {} ms", stats.prefill_ms + stats.decode_ms);
    Ok(())
}
