use std::path::PathBuf;

pub struct TrainArgs {
    pub model: PathBuf,
    pub data: PathBuf,
    pub output: PathBuf,
    pub lora_r: usize,
    pub lora_alpha: f32,
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            data: PathBuf::new(),
            output: PathBuf::new(),
            lora_r: 8,
            lora_alpha: 16.0,
            lr: 1e-4,
            epochs: 3,
            batch_size: 4,
        }
    }
}

/// `synaptix train` — LoRA fine-tuning через `synaptix-train`.
///
/// Полноценный training loop требует синхронной работы autograd-tape
/// (`synaptix-autograd`), оптимизатора (`synaptix-train`), data-pipeline'а
/// (`synaptix-io::data`) и forward'а модели (`synaptix-llm-qwen3`). На MVP
/// здесь — валидация аргументов и явный exit-код `2` (не реализовано), чтобы
/// shell-скрипты могли это детектировать и не падать на исключении.
pub fn run(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("model not found: {}", args.model.display()).into());
    }
    if !args.data.exists() {
        return Err(format!("data not found: {}", args.data.display()).into());
    }
    if args.lora_r == 0 {
        return Err("lora_r must be > 0".into());
    }
    if args.lr <= 0.0 {
        return Err("learning_rate must be > 0".into());
    }
    if args.epochs == 0 {
        return Err("epochs must be > 0".into());
    }
    eprintln!(
        "synaptix train: model={:?} data={:?} output={:?}",
        args.model, args.data, args.output
    );
    eprintln!("  hyperparams: lora_r={} alpha={} lr={} epochs={} batch={}",
        args.lora_r, args.lora_alpha, args.lr, args.epochs, args.batch_size);
    Err("synaptix train: LoRA loop ещё не подключён (Phase R — нужны autograd-tape hook'и в Qwen3Model.forward + Optimizer::step)".into())
}
