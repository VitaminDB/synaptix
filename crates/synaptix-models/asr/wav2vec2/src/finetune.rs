//! CTC-дообучение Wav2Vec2 (фонемы или символы) по манифесту JSONL.
//!
//! Строка манифеста: `{"audio": "путь.wav", "labels": "ɑ q s ..."}` — токены
//! словаря через пробел. Аудио — WAV любой частоты (ресэмпл до 16 кГц).
//! Свёрточный энкодер и позиционная свёртка заморожены, слои ниже
//! `freeze_below` — тоже; голова и остальные слои учатся. Батч — один пример,
//! градиенты копятся `accumulate` шагов (loss нормирован на длину разметки).

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_audio::resample;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::log_softmax_dim;
use synaptix_train::losses::{ctc_loss, CtcReduction};
use synaptix_train::optimizer::adamw::{AdamW, AdamWConfig};

use crate::loader::Wav2Vec2Weights;
use crate::model::Wav2Vec2Model;
use crate::pipeline::{collapse_ctc, normalize_audio};
use crate::vocab::CtcVocab;
use crate::{Result, Wav2Vec2Error};

#[derive(Debug, Clone)]
pub struct FinetuneConfig {
    /// Базовая модель (каталог HF-снапшота или `.syn`).
    pub base: PathBuf,
    pub train: PathBuf,
    pub dev: Option<PathBuf>,
    /// Новый словарь (`vocab.json`); `None` — словарь базовой модели.
    pub vocab: Option<PathBuf>,
    /// Тёплый старт головы: новый токен → токен базового словаря.
    pub head_init: HashMap<String, String>,
    pub out: PathBuf,
    pub device: Device,
    pub lr: f64,
    pub head_lr: f64,
    pub warmup: usize,
    pub steps: usize,
    pub accumulate: usize,
    /// Слои ниже этого номера заморожены.
    pub freeze_below: usize,
    pub max_seconds: f32,
    pub eval_every: usize,
    pub eval_max: usize,
    pub max_grad_norm: f64,
    pub weight_decay: f64,
    pub seed: u64,
}

#[derive(Debug, Clone)]
struct Example {
    audio: PathBuf,
    labels: Vec<u32>,
}

fn read_manifest(path: &Path, vocab: &CtcVocab) -> Result<(Vec<Example>, usize)> {
    let f = std::fs::File::open(path).map_err(|e| Wav2Vec2Error::Load(format!("{}: {e}", path.display())))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    let mut skipped = 0;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).map_err(|e| Wav2Vec2Error::Load(format!("манифест: {e}")))?;
        let audio = PathBuf::from(v["audio"].as_str().unwrap_or_default());
        let audio = if audio.is_absolute() { audio } else { base.join(audio) };
        let ids: Option<Vec<u32>> = v["labels"].as_str().unwrap_or_default().split_whitespace().map(|t| vocab.id(t)).collect();
        match ids {
            Some(labels) if !labels.is_empty() => out.push(Example { audio, labels }),
            _ => skipped += 1,
        }
    }
    Ok((out, skipped))
}

fn load_audio(path: &Path, max_seconds: f32) -> Result<Vec<f32>> {
    let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(path).map_err(|e| Wav2Vec2Error::Audio(format!("{}: {e}", path.display())))?;
    let pcm = if sr == 16000 { pcm } else { resample(&pcm, sr, 16000).map_err(|e| Wav2Vec2Error::Audio(e.to_string()))? };
    // Обрезать нельзя: разметка осталась бы полной — пример пропускается.
    if pcm.len() as f32 > max_seconds * 16000.0 {
        return Err(Wav2Vec2Error::Audio(format!("длиннее {max_seconds} с")));
    }
    Ok(normalize_audio(&pcm))
}

/// Детерминированная перестановка (xorshift).
fn shuffle<T>(v: &mut [T], seed: &mut u64) {
    for i in (1..v.len()).rev() {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        v.swap(i, (*seed % (i as u64 + 1)) as usize);
    }
}

fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + usize::from(x != y));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn trainable(name: &str, freeze_below: usize) -> bool {
    // Имя без префикса семейства (wav2vec2./wavlm./hubert.).
    let name = ["wav2vec2.", "wavlm.", "hubert."].iter().find_map(|p| name.strip_prefix(p)).unwrap_or(name);
    if name.starts_with("lm_head.") || name.starts_with("encoder.layer_norm.") {
        return true;
    }
    name.strip_prefix("encoder.layers.")
        .and_then(|r| r.split('.').next())
        .and_then(|n| n.parse::<usize>().ok())
        .is_some_and(|n| n >= freeze_below)
}

/// Phone/char error rate greedy-декода на `examples` (не больше `max`).
fn evaluate(model: &Wav2Vec2Model, examples: &[Example], max: usize, blank: u32, cfg: &FinetuneConfig) -> Result<f64> {
    no_grad(|| {
        let (mut errs, mut total) = (0usize, 0usize);
        for ex in examples.iter().take(max) {
            let Ok(x) = load_audio(&ex.audio, cfg.max_seconds) else { continue };
            let n = x.len();
            let t = Tensor::from_vec(x, (1, n), cfg.device)?;
            let logits = model.logits(&t)?;
            let ids: Vec<u32> = logits.argmax(2)?.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
            let hyp = collapse_ctc(&ids, blank);
            errs += edit_distance(&hyp, &ex.labels);
            total += ex.labels.len();
        }
        Ok(errs as f64 / total.max(1) as f64)
    })
}

/// Квадрат нормы тензора (считается на устройстве, на хост — скаляр).
fn sq_norm(t: &Tensor) -> Result<f64> {
    let s = t.to_dtype(DType::F32)?.sqr()?.sum_all()?;
    Ok(s.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?[0] as f64)
}

pub fn run(cfg: &FinetuneConfig) -> Result<()> {
    synaptix_autograd::init().map_err(Wav2Vec2Error::Tensor)?;
    let w = if cfg.base.is_dir() {
        Wav2Vec2Weights::from_unpacked(&cfg.base, cfg.device, DType::F32)?
    } else {
        Wav2Vec2Weights::from_syn(&cfg.base, cfg.device, DType::F32)?
    };
    let base_vocab = w.vocab.clone();
    let mut model = Wav2Vec2Model::load(&w)?;
    let (vocab, vocab_json) = match &cfg.vocab {
        Some(p) => {
            let bytes = std::fs::read(p).map_err(|e| Wav2Vec2Error::Load(format!("{}: {e}", p.display())))?;
            let v = CtcVocab::from_json_bytes(&bytes)?;
            let init: Vec<Option<usize>> = v
                .tokens()
                .iter()
                .map(|t| {
                    let src = cfg.head_init.get(t).map(String::as_str).unwrap_or(t.as_str());
                    base_vocab.id(src).map(|i| i as usize)
                })
                .collect();
            let warm = init.iter().filter(|x| x.is_some()).count();
            eprintln!("finetune: новый словарь {} токенов, тёплый старт {warm}", v.len());
            model.replace_head(v.len(), &init)?;
            (v, bytes)
        }
        None => (base_vocab.clone(), serde_json::to_vec(&base_vocab_map(&base_vocab)).unwrap()),
    };
    let blank = model.config().blank_id() as u32;

    let (mut train, skipped) = read_manifest(&cfg.train, &vocab)?;
    eprintln!("finetune: обучающих примеров {} (пропущено {skipped})", train.len());
    let dev = match &cfg.dev {
        Some(p) => read_manifest(p, &vocab)?.0,
        None => Vec::new(),
    };
    if train.is_empty() {
        return Err(Wav2Vec2Error::Config("пустой обучающий манифест".into()));
    }

    // Обучаемые параметры: свежие листья с requires_grad.
    let mut n_params = 0usize;
    for (name, t) in model.named_parameters_mut() {
        if trainable(&name, cfg.freeze_below) {
            n_params += t.numel();
            *t = t.clone().requires_grad_(true);
        }
    }
    eprintln!("finetune: обучаемых параметров {:.1} M (слои ≥ {})", n_params as f64 / 1e6, cfg.freeze_below);
    let mut opt_body = AdamW::new(AdamWConfig { lr: cfg.lr, weight_decay: cfg.weight_decay, ..AdamWConfig::default() });
    let mut opt_head = AdamW::new(AdamWConfig { lr: cfg.head_lr, weight_decay: 0.0, ..AdamWConfig::default() });

    std::fs::create_dir_all(&cfg.out).map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out.join("train.log"))
        .map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
    if !dev.is_empty() {
        let per = evaluate(&model, &dev, cfg.eval_max, blank, cfg)?;
        eprintln!("finetune: PER до обучения {:.2}%", per * 100.0);
        let _ = writeln!(log, "step 0 dev_per {per:.4}");
    }

    let mut seed = cfg.seed.max(1);
    shuffle(&mut train, &mut seed);
    let mut cursor = 0usize;
    let t0 = Instant::now();
    let (mut loss_acc, mut loss_n) = (0f64, 0usize);
    let mut best_per = f64::INFINITY;
    for step in 1..=cfg.steps {
        // Накопление градиентов.
        for _ in 0..cfg.accumulate {
            if cursor >= train.len() {
                cursor = 0;
                shuffle(&mut train, &mut seed);
            }
            let ex = &train[cursor];
            cursor += 1;
            let x = match load_audio(&ex.audio, cfg.max_seconds) {
                Ok(x) => x,
                Err(_) => continue,
            };
            let n = x.len();
            let xt = Tensor::from_vec(x, (1, n), cfg.device)?;
            let logits = model.logits(&xt)?;
            let lp = log_softmax_dim(&logits.to_dtype(DType::F32)?, 2)?;
            let frames = lp.dims()[1];
            let loss = ctc_loss(&lp, &[frames], &[ex.labels.clone()], blank, CtcReduction::MeanPerLabel, true)
                .map_err(Wav2Vec2Error::Tensor)?;
            let lv = loss.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?[0] as f64;
            if lv > 0.0 && lv.is_finite() {
                let scale = Tensor::from_vec(vec![1.0f32 / cfg.accumulate as f32], vec![1usize], cfg.device)?;
                synaptix_core::grad::backward_with(&loss, scale)?;
                loss_acc += lv;
                loss_n += 1;
            }
        }

        // Шаг оптимизатора (без графа).
        let lr_scale = lr_factor(step, cfg.warmup, cfg.steps);
        let (mut body_p, mut body_g, mut head_p, mut head_g) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut total_sq = 0f64;
        for (name, t) in model.named_parameters_mut() {
            if !trainable(&name, cfg.freeze_below) {
                continue;
            }
            let g = match t.grad() {
                Some(g) => g,
                None => t.zeros_like()?,
            };
            total_sq += sq_norm(&g)?;
            if name.starts_with("lm_head.") {
                head_p.push(t.clone());
                head_g.push(g);
            } else {
                body_p.push(t.clone());
                body_g.push(g);
            }
        }
        let gnorm = total_sq.sqrt();
        let clip = if gnorm > cfg.max_grad_norm { (cfg.max_grad_norm / (gnorm + 1e-6)) as f32 } else { 1.0 };
        no_grad(|| -> Result<()> {
            if clip < 1.0 {
                for g in body_g.iter_mut().chain(head_g.iter_mut()) {
                    *g = g.mul_scalar(clip)?;
                }
            }
            opt_body.config.lr = cfg.lr * lr_scale;
            opt_head.config.lr = cfg.head_lr * lr_scale;
            opt_body.step_params(&mut body_p, &body_g).map_err(|e| Wav2Vec2Error::Config(e.to_string()))?;
            opt_head.step_params(&mut head_p, &head_g).map_err(|e| Wav2Vec2Error::Config(e.to_string()))?;
            Ok(())
        })?;
        // Новые значения — обратно в модель свежими листьями (старые градиенты отброшены).
        let (mut bi, mut hi) = (0usize, 0usize);
        for (name, t) in model.named_parameters_mut() {
            if !trainable(&name, cfg.freeze_below) {
                continue;
            }
            let v = if name.starts_with("lm_head.") {
                hi += 1;
                &head_p[hi - 1]
            } else {
                bi += 1;
                &body_p[bi - 1]
            };
            *t = v.clone().requires_grad_(true);
        }

        if step % 10 == 0 || step == 1 {
            let el = t0.elapsed().as_secs_f64();
            let msg = format!(
                "step {step}/{} loss {:.4} gnorm {gnorm:.2} lr {:.2e} {:.2} с/шаг, осталось ~{:.0} мин",
                cfg.steps,
                loss_acc / loss_n.max(1) as f64,
                cfg.lr * lr_scale,
                el / step as f64,
                el / step as f64 * (cfg.steps - step) as f64 / 60.0
            );
            eprintln!("{msg}");
            let _ = writeln!(log, "{msg}");
            loss_acc = 0.0;
            loss_n = 0;
        }
        let last = step == cfg.steps;
        if (cfg.eval_every > 0 && step % cfg.eval_every == 0) || last {
            let per = if dev.is_empty() { f64::NAN } else { evaluate(&model, &dev, cfg.eval_max, blank, cfg)? };
            eprintln!("step {step}: dev PER {:.2}%", per * 100.0);
            let _ = writeln!(log, "step {step} dev_per {per:.4}");
            // С dev-выборкой хранится лучшая по PER модель (последняя её не
            // затирает), без неё — последняя.
            if dev.is_empty() || per < best_per {
                if !dev.is_empty() {
                    best_per = per;
                }
                save(&mut model, &cfg.out, &w, &vocab_json)?;
                eprintln!("  сохранено в {}", cfg.out.display());
            } else if last {
                eprintln!("  лучшая модель (dev PER {:.2}%) уже в {}", best_per * 100.0, cfg.out.display());
            }
        }
    }
    Ok(())
}

fn base_vocab_map(v: &CtcVocab) -> HashMap<String, u32> {
    v.tokens().iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect()
}

/// Линейный прогрев, затем косинусный спад до 5 %.
fn lr_factor(step: usize, warmup: usize, total: usize) -> f64 {
    if step < warmup {
        return step as f64 / warmup.max(1) as f64;
    }
    let t = (step - warmup) as f64 / (total - warmup).max(1) as f64;
    0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * t.min(1.0)).cos())
}

fn save(model: &mut Wav2Vec2Model, out: &Path, w: &Wav2Vec2Weights, vocab_json: &[u8]) -> Result<()> {
    model.save_safetensors(&out.join("model.safetensors"), DType::F32)?;
    let mut cfg_json: serde_json::Value = serde_json::to_value(&w.config_json).unwrap_or_default();
    if let Some(o) = cfg_json.as_object_mut() {
        o.insert("vocab_size".into(), serde_json::json!(model.config().vocab_size));
    }
    let wr = |name: &str, bytes: &[u8]| {
        std::fs::write(out.join(name), bytes).map_err(|e| Wav2Vec2Error::Load(format!("{name}: {e}")))
    };
    wr("config.json", &serde_json::to_vec_pretty(&cfg_json).unwrap())?;
    wr("vocab.json", vocab_json)?;
    wr(
        "preprocessor_config.json",
        br#"{"do_normalize": true, "feature_extractor_type": "Wav2Vec2FeatureExtractor", "feature_size": 1, "padding_side": "right", "padding_value": 0, "return_attention_mask": true, "sampling_rate": 16000}"#,
    )
}
