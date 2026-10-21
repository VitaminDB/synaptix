//! `synaptix train` — дообучение LLM адаптером LoRA (на квантованной базе —
//! QLoRA), см. [`synaptix::facade::train`].

use std::path::PathBuf;

use synaptix::facade::train::{train_lora, BaseMatmul, LoraConfig, LoraTrainOptions, TrainConfig};
use synaptix_core::dtype::DType;

pub struct TrainArgs {
    pub model: PathBuf,
    pub data: PathBuf,
    pub output: PathBuf,
    pub quant: String,
    pub native_quant: bool,
    pub lora_r: usize,
    pub lora_alpha: Option<f32>,
    pub targets: String,
    pub lr: f32,
    pub epochs: usize,
    pub max_len: usize,
    pub grad_accum: usize,
    pub warmup: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub eval_fraction: f32,
    pub eval_every: usize,
    pub save_every: usize,
    pub max_samples: Option<usize>,
    pub thinking: bool,
    pub resume: Option<PathBuf>,
    pub seed: u64,
}

fn parse_quant(s: &str) -> Result<DType, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "nvfp4" | "fp4" => DType::NVFP4,
        "mxfp8" | "fp8" => DType::MXFP8,
        "bf16" | "none" => DType::BF16,
        q if q.starts_with("sq") => DType::Sq { bits: q[2..].parse().map_err(|_| format!("формат {s}: ожидалось sq1…sq8"))? },
        _ => return Err(format!("формат базы {s}: nvfp4 | mxfp8 | sq1…sq8 | bf16")),
    })
}

fn hms(secs: f32) -> String {
    let s = secs.max(0.0) as u64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

pub fn run(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("модель не найдена: {}", args.model.display()).into());
    }
    if !args.data.exists() {
        return Err(format!("датасет не найден: {}", args.data.display()).into());
    }
    let base_quant = parse_quant(&args.quant)?;
    let targets: Vec<String> = match args.targets.as_str() {
        "all" => LoraConfig::default().targets,
        "attn" => ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"].iter().map(|s| s.to_string()).collect(),
        t => t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
    };
    let opts = LoraTrainOptions {
        base_quant,
        base_matmul: if args.native_quant { BaseMatmul::Native } else { BaseMatmul::Dequant },
        lora: LoraConfig { r: args.lora_r, alpha: args.lora_alpha.unwrap_or(2.0 * args.lora_r as f32), targets, seed: args.seed, ..LoraConfig::default() },
        train: TrainConfig {
            lr: args.lr,
            epochs: args.epochs,
            max_len: args.max_len,
            grad_accum: args.grad_accum,
            warmup_ratio: args.warmup,
            weight_decay: args.weight_decay,
            max_grad_norm: args.max_grad_norm,
            seed: args.seed,
            out_dir: args.output.clone(),
            save_every: args.save_every,
            eval_every: args.eval_every,
            ..TrainConfig::default()
        },
        thinking: args.thinking,
        eval_fraction: args.eval_fraction,
        max_samples: args.max_samples,
        init_adapter: args.resume.clone(),
        ..LoraTrainOptions::default()
    };
    eprintln!("synaptix train: {} ← {}", args.model.display(), args.data.display());
    eprintln!(
        "  база {:?}{}, LoRA r={} α={} на {}, lr {:e}, эпох {}, длина {}, накопление {}",
        base_quant,
        if args.native_quant { " (ядра инференса)" } else { "" },
        opts.lora.r,
        opts.lora.alpha,
        opts.lora.targets.join(","),
        opts.train.lr,
        opts.train.epochs,
        opts.train.max_len,
        opts.train.grad_accum,
    );
    let report = train_lora(
        &args.model,
        &args.data,
        opts,
        |d, n_train, n_eval| {
            eprintln!(
                "  датасет: {} примеров ({} обучение, {} оценка), {} токенов, из них учится {}; пропущено {}, обрезано {}{}",
                d.samples,
                n_train,
                n_eval,
                d.tokens,
                d.target_tokens,
                d.skipped,
                d.truncated,
                if d.last_turn_only > 0 { format!(", только последний ответ у {}", d.last_turn_only) } else { String::new() },
            );
        },
        |p| {
            let eval = p.eval_loss.map(|l| format!("  eval {l:.4}")).unwrap_or_default();
            eprintln!(
                "шаг {:>5}/{}  эпоха {}  loss {:.4}{eval}  lr {:.2e}  |g| {:.3}  {:.0} ток/с  {} / осталось {}",
                p.step,
                p.total_steps,
                p.epoch + 1,
                p.loss,
                p.lr,
                p.grad_norm,
                p.tokens_per_sec,
                hms(p.elapsed_secs),
                hms(p.eta_secs),
            );
            true
        },
    )?;
    eprintln!(
        "готово: адаптер ({} обучаемых параметров) — {}",
        report.trainable_params,
        report.adapter_dir.display()
    );
    Ok(())
}
