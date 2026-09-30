use std::path::PathBuf;
use std::time::Instant;

use synaptix::facade::arch::detect_llm_arch;
use synaptix::facade::llm::{load_llm, GenerationOptions, LlmGeneration};

use crate::commands::device::resolve as resolve_device;
use crate::commands::llm_facade::{resolve_precision, PrecisionFlags, RuntimeFlags};

pub struct BenchArgs {
    pub model: PathBuf,
    pub n_tokens: usize,
    pub prompt_tokens: usize,
    pub batch_size: usize,
    pub warmup: usize,
    pub repeat: usize,
    pub device: String,
    pub attn: Option<String>,
    pub quant: Option<String>,
    pub compute_dtype: Option<String>,
    pub storage_dtype: Option<String>,
    pub lm_head_dtype: Option<String>,
    pub embed_dtype: Option<String>,
    pub kv_dtype: Option<String>,
    pub no_graph: bool,
    pub no_spec: bool,
    pub prefill_batch: usize,
}

pub fn run(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device = resolve_device(&args.device);
    crate::commands::device::resolve_attn(args.attn.as_deref());
    if !args.model.exists() {
        return Err(format!("model path not found: {}", args.model.display()).into());
    }
    if args.batch_size != 1 {
        return Err("--batch-size > 1: батчевого декода в движке нет".into());
    }
    let arch = detect_llm_arch(&args.model)?;
    let (precision, profile) = resolve_precision(
        &args.model,
        &PrecisionFlags {
            quant: args.quant.as_deref(),
            compute_dtype: args.compute_dtype.as_deref(),
            storage_dtype: args.storage_dtype.as_deref(),
            lm_head_dtype: args.lm_head_dtype.as_deref(),
            embed_dtype: args.embed_dtype.as_deref(),
            kv_dtype: args.kv_dtype.as_deref(),
        },
        &RuntimeFlags { no_graph: args.no_graph, no_spec: args.no_spec, layer_sync: None, prefill_batch: args.prefill_batch },
    )?;
    eprintln!(
        "synaptix bench: loading {} (arch={arch:?}, профиль {profile}: compute={:?}, attn_w={:?}, mlp_w={:?}, kv={:?}, {:?})",
        args.model.display(),
        precision.compute,
        precision.attn_w,
        precision.mlp_w,
        precision.kv,
        device
    );
    let t0 = Instant::now();
    let max_seq = args.prompt_tokens.max(16) + args.n_tokens + 16;
    let (llm, tok) = load_llm(&args.model, device, precision, Some(max_seq))?;
    eprintln!("synaptix bench: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let mut prompt_ids = tok.encode("Привет, ")?;
    if args.prompt_tokens > prompt_ids.len() {
        let pad = *prompt_ids.last().unwrap_or(&0);
        prompt_ids.resize(args.prompt_tokens, pad);
    }
    let measure = |max_new: usize| -> Result<(usize, f64, f64), Box<dyn std::error::Error>> {
        let opts = GenerationOptions {
            max_new_tokens: max_new,
            max_seq_len: max_seq,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 1,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };
        let mut gen = LlmGeneration::new(&llm, opts);
        let started = Instant::now();
        let mut first = None;
        let mut n = 0usize;
        gen.generate_streaming(&prompt_ids, &tok, |_, _| {
            first.get_or_insert_with(Instant::now);
            n += 1;
            true
        })?;
        let end = Instant::now();
        let first = first.unwrap_or(end);
        Ok((n, (first - started).as_secs_f64() * 1e3, (end - first).as_secs_f64() * 1e3))
    };
    eprintln!("synaptix bench: warmup ({} iterations)", args.warmup);
    for _ in 0..args.warmup {
        measure(2)?;
    }
    let mut rows = Vec::with_capacity(args.repeat.max(1));
    for i in 0..args.repeat.max(1) {
        let (n, prefill, decode) = measure(args.n_tokens)?;
        let p_tps = prompt_ids.len() as f64 / (prefill / 1e3).max(1e-9);
        let d_tps = n.saturating_sub(1) as f64 / (decode / 1e3).max(1e-9);
        eprintln!("  прогон {}: prefill {prefill:.1} ms ({p_tps:.1} tok/s) | decode {} ток. {decode:.1} ms ({d_tps:.2} tok/s)", i + 1, n.saturating_sub(1));
        rows.push((p_tps, d_tps));
    }
    let mean = |f: fn(&(f64, f64)) -> f64| rows.iter().map(f).sum::<f64>() / rows.len() as f64;
    println!("synaptix bench {} ({arch:?}, {profile}):", args.model.display());
    println!("  prompt:  {} tokens, {:.1} tok/s", prompt_ids.len(), mean(|r| r.0));
    println!("  decode:  {} tokens, {:.2} tok/s (среднее по {} прогонам)", args.n_tokens, mean(|r| r.1), rows.len());
    Ok(())
}
