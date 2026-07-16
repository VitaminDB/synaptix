//! Итеративное masked-diffusion раскрытие (MaskGIT-стиль).
//!
//! Порт `_generate_iterative` + `_predict_tokens_with_scoring` + `_get_time_steps`
//! из `~/Temp/OmniVoice/omnivoice/models/omnivoice.py` для B=1, greedy
//! (position_temperature=0, class_temperature=0 → без gumbel/top-k-фильтра).
//!
//! Каждый step: ДВА отдельных `Backbone::forward` (cond длины C + uncond длины T)
//! — для B=1 это эквивалентно батч-паддингу 2B с диагональю в attention-маске
//! uncond, но проще (full-attention в обоих). Скоринг (log_softmax / CFG / argmax /
//! max / layer-penalty / topk-by-flatten) — на host f32 (логиты приходят f32,
//! точность важнее скорости). См. SPEC.md «Критичные места» п.3-4.

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::backbone::Backbone;
use crate::config::OmniVoiceGenerationConfig;
use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Inference(e.to_string())
}

/// timesteps = linspace(0,1,num_step+1); ts = t_shift*ts/(1+(t_shift-1)*ts).
fn time_steps(num_step: usize, t_shift: f32) -> Vec<f32> {
    let n = num_step;
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            t_shift * t / (1.0 + (t_shift - 1.0) * t)
        })
        .collect()
}

/// schedule[step] = (последний шаг ? rem : min(ceil(total*Δt), rem)); rem -= num.
fn schedule(total_mask: usize, num_step: usize, ts: &[f32]) -> Vec<usize> {
    let mut rem = total_mask as i64;
    let mut sched = Vec::with_capacity(num_step);
    for step in 0..num_step {
        let num = if step == num_step - 1 {
            rem
        } else {
            let delta = (ts[step + 1] - ts[step]) as f64;
            let ceil = (total_mask as f64 * delta).ceil() as i64;
            ceil.min(rem)
        };
        let num = num.max(0);
        sched.push(num as usize);
        rem -= num;
    }
    sched
}

/// Раскрыть один item, B=1.
///
/// `cond_input_ids` [1,8,C] (I64), `cond_audio_mask` [1,C] (U8) — из text-фронтенда
/// (target-хвост = MASK, audio_mask=true на ref+target хвосте). Возврат `codes`
/// [8,T] (I64) на host (Device::Cpu).
pub fn generate_iterative(
    backbone: &Backbone,
    cond_input_ids: &Tensor,
    cond_audio_mask: &Tensor,
    target_len: usize,
    gen: &OmniVoiceGenerationConfig,
) -> Result<Tensor> {
    let dims = cond_input_ids.dims();
    let (n_cb, c_len) = (dims[1], dims[2]);
    let t = target_len;
    let mask_id = backbone.audio_mask_id();
    let vocab = backbone.audio_vocab_size();
    // Тензоры-входы forward + выходные codes — на устройстве модели (CPU/CUDA).
    // Host-скоринг тянет логиты через to_vec1 (кросс-девайс) независимо.
    let device = backbone.device();

    if c_len < t {
        return Err(err(format!("c_len {c_len} < target_len {t}")));
    }

    // cond input_ids/audio_mask на host I64/U8 (мутируем хвост по шагам).
    let mut cond_ids = cond_input_ids
        .to_dtype(DType::I64)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<i64>())
        .map_err(err)?; // [8*C] row-major по (c, s)
    let cond_mask = cond_audio_mask
        .to_dtype(DType::U8)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<u8>())
        .map_err(err)?; // [C]

    // uncond = только target-хвост: input_ids[...,-T:], audio_mask[...,-T:].
    let mut uncond_ids = vec![0i64; n_cb * t];
    for c in 0..n_cb {
        for j in 0..t {
            uncond_ids[c * t + j] = cond_ids[c * c_len + (c_len - t + j)];
        }
    }
    let uncond_mask: Vec<u8> = (0..t).map(|j| cond_mask[c_len - t + j]).collect();
    let uncond_mask_t = Tensor::from_vec(uncond_mask, vec![1, t], device).map_err(err)?;

    // tokens[8,T] = MASK.
    let mut tokens = vec![mask_id; n_cb * t]; // row-major (c, s)

    let ts = time_steps(gen.num_step, gen.t_shift);
    let sched = schedule(t * n_cb, gen.num_step, &ts);
    let g = gen.guidance_scale;
    let lpf = gen.layer_penalty_factor;

    for step in 0..gen.num_step {
        let k = sched[step];
        if k == 0 {
            continue;
        }

        // forward cond (C) и uncond (T).
        let cond_ids_t = Tensor::from_vec(cond_ids.clone(), vec![1, n_cb, c_len], device)
            .map_err(err)?;
        let cond_mask_t = Tensor::from_vec(cond_mask.clone(), vec![1, c_len], device)
            .map_err(err)?;
        let uncond_ids_t = Tensor::from_vec(uncond_ids.clone(), vec![1, n_cb, t], device)
            .map_err(err)?;

        let cond_logits = backbone.forward(&cond_ids_t, &cond_mask_t)?; // [1,8,C,vocab]
        let uncond_logits = backbone.forward(&uncond_ids_t, &uncond_mask_t)?; // [1,8,T,vocab]

        let cond_flat = cond_logits
            .to_dtype(DType::F32)
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())
            .map_err(err)?; // [8*C*vocab]
        let uncond_flat = uncond_logits
            .to_dtype(DType::F32)
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())
            .map_err(err)?; // [8*T*vocab]

        // pred[8,T], score[8,T]; score -= layer_id*lpf; уже раскрытые -> -inf.
        let mut pred = vec![0i64; n_cb * t];
        let mut score = vec![f32::NEG_INFINITY; n_cb * t];
        for c in 0..n_cb {
            for j in 0..t {
                // cond_logits[:, c, (c_len-t)+j, :]
                let c_off = ((c * c_len) + (c_len - t + j)) * vocab;
                let u_off = ((c * t) + j) * vocab;
                let c_row = &cond_flat[c_off..c_off + vocab];
                let u_row = &uncond_flat[u_off..u_off + vocab];

                let (p, s) = predict_row(c_row, u_row, g, mask_id, vocab);
                let idx = c * t + j;
                pred[idx] = p as i64;
                // layer-penalty (layer_id = c по оси кодбука).
                let penalized = s - (c as f32) * lpf;
                // уже раскрытые (tokens != mask) -> -inf.
                score[idx] = if tokens[idx] != mask_id {
                    f32::NEG_INFINITY
                } else {
                    penalized
                };
            }
        }

        // topk(score.flatten(), k); flatten — row-major [8,T] (как PyTorch .flatten()).
        let top = topk_indices(&score, k);
        for &idx in &top {
            tokens[idx] = pred[idx];
        }

        // обновить cond хвост и uncond.
        for c in 0..n_cb {
            for j in 0..t {
                let tok = tokens[c * t + j];
                cond_ids[c * c_len + (c_len - t + j)] = tok;
                uncond_ids[c * t + j] = tok;
            }
        }
    }

    Tensor::from_vec(tokens, vec![n_cb, t], device).map_err(err)
}

/// Один (c,j): CFG log_softmax(c_lp + g*(c_lp-u_lp)) → mask_id=-inf → (argmax, max).
/// greedy (class_temperature=0). Возврат (pred_token, confidence_score).
fn predict_row(
    c_logits: &[f32],
    u_logits: &[f32],
    g: f32,
    mask_id: i64,
    vocab: usize,
) -> (usize, f32) {
    // c_lp = log_softmax(c_logits); u_lp = log_softmax(u_logits).
    let c_lp = log_softmax(c_logits);
    let combined: Vec<f32> = if g != 0.0 {
        let u_lp = log_softmax(u_logits);
        // lp = log_softmax(c_lp + g*(c_lp - u_lp)).
        let mixed: Vec<f32> = (0..vocab)
            .map(|i| c_lp[i] + g * (c_lp[i] - u_lp[i]))
            .collect();
        log_softmax(&mixed)
    } else {
        c_lp
    };

    // lp[mask_id] = -inf, затем argmax + max.
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in combined.iter().enumerate() {
        let val = if i as i64 == mask_id {
            f32::NEG_INFINITY
        } else {
            v
        };
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

/// Численно-устойчивый log_softmax по 1D-строке (как F.log_softmax).
fn log_softmax(x: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &v in x {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in x {
        sum += ((v - m) as f64).exp();
    }
    let log_sum = sum.ln() as f32 + m;
    x.iter().map(|&v| v - log_sum).collect()
}

/// top-k индексов по значению (убывание), tie-break по меньшему индексу.
/// Совпадает с детерминированным `torch.topk(flatten, k)` для stable-данных.
fn topk_indices(score: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    let k = k.min(idx.len());
    idx.sort_by(|&a, &b| {
        match score[b].partial_cmp(&score[a]) {
            Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(o) => o,
        }
    });
    idx.truncate(k);
    idx
}
