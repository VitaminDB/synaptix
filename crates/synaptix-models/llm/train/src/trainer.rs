//! Цикл обучения: упаковка примеров в последовательности, накопление
//! градиента, AdamW с расписанием, оценка на отложенных примерах,
//! сохранение адаптера.

use std::path::PathBuf;
use std::time::Instant;

use crate::adapter;
use crate::model::{Batch, TrainModel};
use crate::optim::{grad_norm, lr_at, AdamW};
use crate::Result;

/// Пример: токены целиком и флаг «этот токен модель должна предсказать»
/// (обычно — токены ответов ассистента).
#[derive(Debug, Clone)]
pub struct Sample {
    pub tokens: Vec<u32>,
    pub target: Vec<bool>,
}

impl Sample {
    pub fn n_targets(&self) -> usize {
        // Первый токен предсказывать не из чего.
        self.target.iter().skip(1).filter(|&&t| t).count()
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub lr: f32,
    pub epochs: usize,
    /// Длина упакованной последовательности (длиннее — обрезается).
    pub max_len: usize,
    /// Последовательностей на шаг оптимизатора.
    pub grad_accum: usize,
    pub warmup_ratio: f32,
    pub min_lr_ratio: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub seed: u64,
    /// Каталог адаптера; чекпойнты — `checkpoint-<шаг>` внутри.
    pub out_dir: PathBuf,
    /// Чекпойнт каждые столько шагов (0 — только в конце).
    pub save_every: usize,
    /// Оценка на отложенных примерах каждые столько шагов (0 — только в конце).
    pub eval_every: usize,
    pub base_name: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            epochs: 1,
            max_len: 2048,
            grad_accum: 4,
            warmup_ratio: 0.03,
            min_lr_ratio: 0.1,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            seed: 0,
            out_dir: PathBuf::from("lora-out"),
            save_every: 0,
            eval_every: 0,
            base_name: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainProgress {
    pub step: usize,
    pub total_steps: usize,
    pub epoch: usize,
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub tokens_per_sec: f32,
    pub elapsed_secs: f32,
    pub eta_secs: f32,
    /// Лосс на отложенных примерах, если на этом шаге была оценка.
    pub eval_loss: Option<f32>,
}

/// Перемешивание (xorshift64*, детерминировано зерном).
fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for i in (1..v.len()).rev() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let j = (s.wrapping_mul(0x2545F4914F6CDD1D) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Жадная упаковка примеров по порядку в последовательности до `max_len`.
pub fn pack(samples: &[&Sample], max_len: usize) -> Vec<Batch> {
    let mut out = Vec::new();
    let mut cur = Batch::default();
    let mut seg = 0i32;
    for s in samples {
        let n = s.tokens.len().min(max_len);
        if n < 2 {
            continue;
        }
        if !cur.is_empty() && cur.len() + n > max_len {
            out.push(std::mem::take(&mut cur));
            seg = 0;
        }
        for t in 0..n {
            cur.tokens.push(s.tokens[t]);
            let label = if t + 1 < n && s.target[t + 1] { s.tokens[t + 1] as i32 } else { -100 };
            cur.labels.push(label);
            cur.positions.push(t as u32);
            cur.seg.push(seg);
        }
        seg += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|b| b.n_targets() > 0);
    out
}

pub struct Trainer {
    pub cfg: TrainConfig,
    pub opt: AdamW,
}

impl Trainer {
    pub fn new(cfg: TrainConfig) -> Self {
        let opt = AdamW { weight_decay: cfg.weight_decay, ..AdamW::default() };
        Self { cfg, opt }
    }

    /// Средний CE на отложенных примерах.
    pub fn evaluate(&self, model: &TrainModel<'_>, eval: &[Sample]) -> Result<Option<f32>> {
        if eval.is_empty() {
            return Ok(None);
        }
        let refs: Vec<&Sample> = eval.iter().collect();
        let (mut sum, mut n) = (0f64, 0usize);
        for b in pack(&refs, self.cfg.max_len) {
            let s = model.eval_loss(&b)?;
            sum += s.loss_sum;
            n += s.n_targets;
        }
        Ok((n > 0).then(|| (sum / n as f64) as f32))
    }

    /// Обучить и сохранить адаптер в `cfg.out_dir`. `progress` вызывается
    /// после каждого шага оптимизатора; `false` из него — остановиться
    /// (адаптер всё равно сохраняется).
    pub fn run(
        &mut self,
        model: &mut TrainModel<'_>,
        train: &[Sample],
        eval: &[Sample],
        mut progress: impl FnMut(&TrainProgress) -> bool,
    ) -> Result<()> {
        let cfg = self.cfg.clone();
        let accum = cfg.grad_accum.max(1);
        // План всех шагов наперёд: число шагов нужно расписанию.
        let mut epochs: Vec<Vec<Batch>> = Vec::with_capacity(cfg.epochs);
        for e in 0..cfg.epochs {
            let mut refs: Vec<&Sample> = train.iter().collect();
            shuffle(&mut refs, cfg.seed.wrapping_add(e as u64));
            epochs.push(pack(&refs, cfg.max_len));
        }
        let total_steps: usize = epochs.iter().map(|b| b.len().div_ceil(accum)).sum();
        let warmup = (total_steps as f32 * cfg.warmup_ratio).round() as usize;
        let started = Instant::now();
        let mut step = 0usize;
        let mut tokens_done = 0usize;
        let mut stop = false;

        'outer: for (epoch, batches) in epochs.iter().enumerate() {
            for group in batches.chunks(accum) {
                let n_targets: usize = group.iter().map(|b| b.n_targets()).sum();
                let scale = 1.0 / n_targets.max(1) as f32;
                let (mut loss_sum, mut n) = (0f64, 0usize);
                for b in group {
                    let s = model.forward_backward(b, scale)?;
                    loss_sum += s.loss_sum;
                    n += s.n_targets;
                    tokens_done += b.len();
                }
                let gn = grad_norm(model)?;
                let mul = if cfg.max_grad_norm > 0.0 && gn > cfg.max_grad_norm { cfg.max_grad_norm / gn } else { 1.0 };
                let lr = lr_at(cfg.lr, step, total_steps, warmup, cfg.min_lr_ratio);
                if gn.is_finite() {
                    self.opt.step(model, lr, mul)?;
                } else {
                    // NaN/inf в градиенте — шаг пропускается целиком.
                    model.zero_grad();
                }
                step += 1;
                if crate::prof::enabled() {
                    eprintln!("профиль шага {step}:\n{}", crate::prof::report_and_clear());
                }
                let elapsed = started.elapsed().as_secs_f32();
                let eval_loss = if (cfg.eval_every > 0 && step % cfg.eval_every == 0) || step == total_steps {
                    self.evaluate(model, eval)?
                } else {
                    None
                };
                let p = TrainProgress {
                    step,
                    total_steps,
                    epoch,
                    loss: (loss_sum / n.max(1) as f64) as f32,
                    lr,
                    grad_norm: gn,
                    tokens_per_sec: tokens_done as f32 / elapsed.max(1e-3),
                    elapsed_secs: elapsed,
                    eta_secs: elapsed / step as f32 * (total_steps - step) as f32,
                    eval_loss,
                };
                if cfg.save_every > 0 && step % cfg.save_every == 0 && step < total_steps {
                    adapter::save(model, &cfg.out_dir.join(format!("checkpoint-{step}")), &cfg.base_name)?;
                }
                if !progress(&p) {
                    stop = true;
                    break 'outer;
                }
            }
        }
        let _ = stop;
        adapter::save(model, &cfg.out_dir, &cfg.base_name)
    }
}
