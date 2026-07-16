pub enum LrSchedule {
    Constant(f64),
    Cosine { max_lr: f64, min_lr: f64, total_steps: usize },
    Linear { start_lr: f64, end_lr: f64, total_steps: usize },
    WarmupCosine { warmup_steps: usize, max_lr: f64, min_lr: f64, total_steps: usize },
}

impl LrSchedule {
    pub fn lr_at(&self, step: usize) -> f64 {
        match self {
            Self::Constant(lr) => *lr,
            Self::Cosine { max_lr, min_lr, total_steps } => {
                let t = (step as f64 / *total_steps as f64).min(1.0);
                min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
            }
            Self::Linear { start_lr, end_lr, total_steps } => {
                let t = (step as f64 / *total_steps as f64).min(1.0);
                start_lr + (end_lr - start_lr) * t
            }
            Self::WarmupCosine { warmup_steps, max_lr, min_lr, total_steps } => {
                if step < *warmup_steps {
                    max_lr * step as f64 / (*warmup_steps).max(1) as f64
                } else {
                    let t = ((step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64).min(1.0);
                    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
                }
            }
        }
    }
}
