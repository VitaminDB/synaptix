//! CTC-дообучение Wav2Vec2 (фонемы или символы) по манифесту JSONL.
//!
//! Строка манифеста: `{"audio": "путь.wav", "labels": "ɑ q s ..."}` — токены
//! словаря через пробел. Аудио — WAV любой частоты (ресэмпл до 16 кГц).
//! Свёрточный энкодер и позиционная свёртка заморожены, слои ниже
//! `freeze_below` — тоже; голова и остальные слои учатся. Батч — один пример,
//! градиенты копятся `accumulate` шагов (loss нормирован на длину разметки).
//!
//! Результат в `out`: `model.safetensors` (F32), конфиги и сразу готовый
//! бандл `model.syn` (F16, purpose=asr). Каждые `save_every` шагов и в конце —
//! чекпойнт `out/checkpoint/` (веса F32, моменты AdamW, шаг, порядок примеров
//! и ГПСЧ перемешивания), с которого `resume` продолжает ровно с того же места.

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
    /// Чекпойнт каждые N шагов (0 — только в конце).
    pub save_every: usize,
    /// Продолжить с `out/checkpoint/`.
    pub resume: bool,
    /// Собирать `out/model.syn` (F16) при каждом сохранении лучшей модели.
    pub bundle: bool,
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
    let ckpt_dir = cfg.out.join("checkpoint");
    let ckpt = if cfg.resume {
        let bytes = std::fs::read(ckpt_dir.join("state.json"))
            .map_err(|e| Wav2Vec2Error::Load(format!("--resume: нет чекпойнта {}: {e}", ckpt_dir.display())))?;
        let st: CheckpointState = serde_json::from_slice(&bytes).map_err(|e| Wav2Vec2Error::Load(format!("state.json: {e}")))?;
        if st.freeze_below != cfg.freeze_below {
            return Err(Wav2Vec2Error::Config(format!(
                "--resume: чекпойнт обучался с --freeze-below {}, задано {}",
                st.freeze_below, cfg.freeze_below
            )));
        }
        eprintln!("finetune: продолжение с шага {} ({})", st.step, ckpt_dir.display());
        Some(st)
    } else {
        None
    };
    let w = if ckpt.is_some() {
        // Чекпойнт — полный каталог модели с уже заменённой головой.
        Wav2Vec2Weights::from_unpacked(&ckpt_dir, cfg.device, DType::F32)?
    } else if cfg.base.is_dir() {
        Wav2Vec2Weights::from_unpacked(&cfg.base, cfg.device, DType::F32)?
    } else {
        Wav2Vec2Weights::from_syn(&cfg.base, cfg.device, DType::F32)?
    };
    let base_vocab = w.vocab.clone();
    let mut model = Wav2Vec2Model::load(&w)?;
    let (vocab, vocab_json) = match &cfg.vocab {
        _ if ckpt.is_some() => {
            let bytes = std::fs::read(ckpt_dir.join("vocab.json")).map_err(|e| Wav2Vec2Error::Load(format!("vocab.json: {e}")))?;
            (base_vocab.clone(), bytes)
        }
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
    let names: Vec<String> = model.named_parameters_mut().into_iter().map(|(n, _)| n).filter(|n| trainable(n, cfg.freeze_below)).collect();

    std::fs::create_dir_all(&cfg.out).map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out.join("train.log"))
        .map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
    if !dev.is_empty() && ckpt.is_none() {
        let per = evaluate(&model, &dev, cfg.eval_max, blank, cfg)?;
        eprintln!("finetune: PER до обучения {:.2}%", per * 100.0);
        let _ = writeln!(log, "step 0 dev_per {per:.4}");
    }

    // Порядок примеров — перестановка индексов манифеста (её и ГПСЧ хранит
    // чекпойнт, чтобы продолжение шло по тем же примерам).
    let mut seed = cfg.seed.max(1);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut cursor = 0usize;
    let mut best_per = f64::INFINITY;
    let mut start = 1usize;
    match &ckpt {
        Some(st) => {
            if st.order.len() != train.len() || st.order.iter().any(|&i| i >= train.len()) {
                return Err(Wav2Vec2Error::Config(format!(
                    "--resume: в чекпойнте {} примеров, в манифесте {} — манифест изменился",
                    st.order.len(),
                    train.len()
                )));
            }
            order = st.order.clone();
            cursor = st.cursor;
            seed = st.seed;
            best_per = st.best_per.unwrap_or(f64::INFINITY);
            start = st.step + 1;
            load_optimizer(&ckpt_dir.join("optimizer.safetensors"), &names, st, &mut opt_body, &mut opt_head, cfg.device)?;
        }
        None => shuffle(&mut order, &mut seed),
    }
    let t0 = Instant::now();
    let (mut loss_acc, mut loss_n) = (0f64, 0usize);
    for step in start..=cfg.steps {
        // Накопление градиентов.
        for _ in 0..cfg.accumulate {
            if cursor >= order.len() {
                cursor = 0;
                shuffle(&mut order, &mut seed);
            }
            let ex = &train[order[cursor]];
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

        if step % 10 == 0 || step == start {
            let el = t0.elapsed().as_secs_f64();
            let msg = format!(
                "step {step}/{} loss {:.4} gnorm {gnorm:.2} lr {:.2e} {:.2} с/шаг, осталось ~{:.0} мин",
                cfg.steps,
                loss_acc / loss_n.max(1) as f64,
                cfg.lr * lr_scale,
                el / (step + 1 - start) as f64,
                el / (step + 1 - start) as f64 * (cfg.steps - step) as f64 / 60.0
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
                if cfg.bundle {
                    write_bundle(&mut model, &cfg.out, &w, &vocab_json)?;
                }
                eprintln!("  сохранено в {}", cfg.out.display());
            } else if last {
                eprintln!("  лучшая модель (dev PER {:.2}%) уже в {}", best_per * 100.0, cfg.out.display());
            }
        }
        if (cfg.save_every > 0 && step % cfg.save_every == 0) || last {
            let st = CheckpointState {
                step,
                cursor,
                seed,
                order: order.clone(),
                best_per: best_per.is_finite().then_some(best_per),
                body_step: opt_body.step,
                head_step: opt_head.step,
                freeze_below: cfg.freeze_below,
                names: names.clone(),
            };
            save_checkpoint(&mut model, &cfg.out, &w, &vocab_json, &opt_body, &opt_head, &st)?;
            eprintln!("  чекпойнт шага {step} → {}", ckpt_dir.display());
        }
    }
    Ok(())
}

/// Состояние обучения в `checkpoint/state.json`.
#[derive(serde::Serialize, serde::Deserialize)]
struct CheckpointState {
    step: usize,
    cursor: usize,
    seed: u64,
    order: Vec<usize>,
    best_per: Option<f64>,
    body_step: usize,
    head_step: usize,
    freeze_below: usize,
    /// Обучаемые параметры в порядке моментов оптимизаторов.
    names: Vec<String>,
}

fn io_err(what: &Path) -> impl Fn(std::io::Error) -> Wav2Vec2Error + '_ {
    move |e| Wav2Vec2Error::Load(format!("{}: {e}", what.display()))
}

/// Чекпойнт пишется во временный каталог и подменяет старый целиком —
/// прерывание посреди записи не портит прошлый.
fn save_checkpoint(
    model: &mut Wav2Vec2Model,
    out: &Path,
    w: &Wav2Vec2Weights,
    vocab_json: &[u8],
    opt_body: &AdamW,
    opt_head: &AdamW,
    st: &CheckpointState,
) -> Result<()> {
    let tmp = out.join("checkpoint.tmp");
    let dst = out.join("checkpoint");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(io_err(&tmp))?;
    save(model, &tmp, w, vocab_json)?;
    let mut tensors: Vec<(String, &Tensor)> = Vec::new();
    let (bn, hn): (Vec<&String>, Vec<&String>) = st.names.iter().partition(|n| !n.starts_with("lm_head."));
    for (names, opt) in [(&bn, opt_body), (&hn, opt_head)] {
        let (m, v) = opt.moments();
        for (i, n) in names.iter().enumerate() {
            if let (Some(m), Some(v)) = (m.get(i), v.get(i)) {
                tensors.push((format!("m.{n}"), m));
                tensors.push((format!("v.{n}"), v));
            }
        }
    }
    write_f32_safetensors(&tmp.join("optimizer.safetensors"), &tensors)?;
    let p = tmp.join("state.json");
    std::fs::write(&p, serde_json::to_vec(st).unwrap()).map_err(io_err(&p))?;
    let _ = std::fs::remove_dir_all(&dst);
    std::fs::rename(&tmp, &dst).map_err(io_err(&dst))
}

fn write_f32_safetensors(path: &Path, tensors: &[(String, &Tensor)]) -> Result<()> {
    let mut blobs = Vec::with_capacity(tensors.len());
    for (n, t) in tensors {
        let host = t.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        blobs.push((n.clone(), t.dims().to_vec(), host.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()));
    }
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = blobs
        .iter()
        .map(|(n, s, b)| {
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                .map(|v| (n.clone(), v))
                .map_err(|e| Wav2Vec2Error::Load(format!("{n}: {e}")))
        })
        .collect::<Result<_>>()?;
    let bytes = safetensors::serialize(views, None).map_err(|e| Wav2Vec2Error::Load(e.to_string()))?;
    std::fs::write(path, bytes).map_err(io_err(path))
}

fn load_optimizer(
    path: &Path,
    names: &[String],
    st: &CheckpointState,
    opt_body: &mut AdamW,
    opt_head: &mut AdamW,
    device: Device,
) -> Result<()> {
    if st.names != names {
        return Err(Wav2Vec2Error::Config("--resume: набор обучаемых параметров не совпадает с чекпойнтом".into()));
    }
    let bytes = std::fs::read(path).map_err(io_err(path))?;
    let sf = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| Wav2Vec2Error::Load(format!("{}: {e}", path.display())))?;
    let get = |key: &str| -> Result<Option<Tensor>> {
        let Ok(v) = sf.tensor(key) else { return Ok(None) };
        let data: Vec<f32> = v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Some(Tensor::from_vec(data, v.shape().to_vec(), device)?))
    };
    for (head, opt, step) in [(false, opt_body, st.body_step), (true, opt_head, st.head_step)] {
        let (mut m, mut v) = (Vec::new(), Vec::new());
        for n in names.iter().filter(|n| n.starts_with("lm_head.") == head) {
            match (get(&format!("m.{n}"))?, get(&format!("v.{n}"))?) {
                (Some(a), Some(b)) => {
                    m.push(a);
                    v.push(b);
                }
                _ if step == 0 => {}
                _ => return Err(Wav2Vec2Error::Load(format!("optimizer.safetensors: нет моментов {n}"))),
            }
        }
        opt.set_moments(m, v, step);
    }
    Ok(())
}

/// Готовый к службе бандл `out/model.syn`: веса в F16, конфиги рядом —
/// то же, что `syn-pack <каталог F16> --purpose asr`.
fn write_bundle(model: &mut Wav2Vec2Model, out: &Path, w: &Wav2Vec2Weights, vocab_json: &[u8]) -> Result<()> {
    let stage = out.join(".bundle-stage");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(io_err(&stage))?;
    write_configs(model, &stage, w, vocab_json)?;
    model.save_safetensors(&stage.join("model.safetensors"), DType::F16)?;
    let berr = |e: synaptix_bundle::Error| Wav2Vec2Error::Bundle(e.to_string());
    let mut plan = synaptix_bundle::pack_plan::PackPlan::scan(&stage).map_err(berr)?;
    plan.meta.purpose = "asr".into();
    if let Some(name) = out.file_name().and_then(|n| n.to_str()) {
        plan.meta.id = synaptix_bundle::pack_plan::normalize_id(name);
    }
    let dst = out.join("model.syn");
    plan.into_builder().map_err(berr)?.write(&dst).map_err(berr)?;
    let _ = std::fs::remove_dir_all(&stage);
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
    write_configs(model, out, w, vocab_json)
}

fn write_configs(model: &Wav2Vec2Model, out: &Path, w: &Wav2Vec2Weights, vocab_json: &[u8]) -> Result<()> {
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
