//! Streaming-инференс NeMo Sortformer (`forward_streaming` + `sortformer_modules`).
//!
//! Источник истины: NeMo `sortformer_diar_models.forward_streaming(_step)` +
//! `sortformer_modules.{streaming_update,_compress_spkcache,_get_silence_profile,...}`.
//! Прогон batch=1, sync-режим (fifo_len=0). Эмбеддинги остаются на device; скоринг/topk
//! спик-кэша считается на CPU (раз в чанк, дёшево).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct StreamCfg {
    pub spkcache_len: usize,
    pub fifo_len: usize,
    pub chunk_len: usize,
    pub subsampling_factor: usize,
    pub chunk_left_context: usize,
    pub chunk_right_context: usize,
    pub spkcache_sil_frames_per_spk: usize,
    pub spkcache_update_period: usize,
    pub n_spk: usize,
    pub d_model: usize,
    pub pred_score_threshold: f32,
    pub scores_boost_latest: f32,
    pub sil_threshold: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
    pub max_index: usize,
}

/// Состояние стриминга (sync, fifo_len=0): spkcache как device-тензор, mean_sil + n_sil.
pub struct StreamState {
    pub spkcache: Tensor,               // (1, L, d_model) — пустой в начале
    pub spkcache_preds: Option<Tensor>, // (1, L, n_spk)
    pub mean_sil: Tensor,               // (1, d_model)
    pub n_sil: f32,
    device: Device,
    dtype: DType,
}

impl StreamState {
    pub fn new(device: Device, dtype: DType, d_model: usize) -> Result<Self> {
        let spkcache = Tensor::from_vec(Vec::<f32>::new(), (1, 0, d_model), device)?.to_dtype(dtype)?;
        let mean_sil = Tensor::from_vec(vec![0.0f32; d_model], (1, d_model), device)?.to_dtype(dtype)?;
        Ok(Self { spkcache, spkcache_preds: None, mean_sil, n_sil: 0.0, device, dtype })
    }
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn cat2(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dims()[1] == 0 {
        return Ok(b.clone());
    }
    if b.dims()[1] == 0 {
        return Ok(a.clone());
    }
    Ok(Tensor::cat(&[a, b], 1)?)
}

/// NeMo `streaming_update` (sync, **fifo_len=0** — конфиг v2.1: fifo всегда пуст,
/// поэтому chunk напрямую идёт в spkcache, без поддержки fifo и без 0-длинных тензоров).
/// `chunk_embs` (1, lc+chunk_len+rc, d), `preds` (1, spkcache+lc+chunk_len+rc, n_spk).
pub fn streaming_update(
    state: &mut StreamState,
    chunk_embs: &Tensor,
    preds: &Tensor,
    lc: usize,
    rc: usize,
    cfg: &StreamCfg,
) -> Result<Tensor> {
    assert_eq!(cfg.fifo_len, 0, "streaming_update реализован только для fifo_len=0 (v2.1)");
    let sc_len = state.spkcache.dims()[1];
    let total = chunk_embs.dims()[1];
    let chunk_len = total - lc - rc;

    // fifo_len=0 → pop = весь чанк (без контекста); chunk_preds — срез чанка.
    let pop_embs = chunk_embs.narrow(1, lc, chunk_len)?.contiguous()?;
    let chunk_preds = preds.narrow(1, sc_len + lc, chunk_len)?.contiguous()?;

    get_silence_profile(state, &pop_embs, &chunk_preds, cfg)?;

    state.spkcache = cat2(&state.spkcache, &pop_embs)?;
    if let Some(scp) = &state.spkcache_preds {
        state.spkcache_preds = Some(cat2(scp, &chunk_preds)?);
    }

    if state.spkcache.dims()[1] > cfg.spkcache_len {
        if state.spkcache_preds.is_none() {
            // первая компрессия: preds[:, :sc_len] (старый spkcache) + chunk_preds.
            let spkcache_preds = if sc_len > 0 {
                let old = preds.narrow(1, 0, sc_len)?.contiguous()?;
                cat2(&old, &chunk_preds)?
            } else {
                chunk_preds.clone()
            };
            state.spkcache_preds = Some(spkcache_preds);
        }
        let preds_for = state.spkcache_preds.take().unwrap();
        let (sc, scp) = compress_spkcache(&state.spkcache, &preds_for, &state.mean_sil, cfg)?;
        state.spkcache = sc;
        state.spkcache_preds = Some(scp);
    }

    Ok(chunk_preds)
}

/// NeMo `_get_silence_profile`: обновить mean_sil/n_sil по тихим кадрам pop-out.
fn get_silence_profile(
    state: &mut StreamState,
    emb_seq: &Tensor,
    preds: &Tensor,
    cfg: &StreamCfg,
) -> Result<()> {
    let frames = preds.dims()[1];
    let pv = to_vec(preds)?;
    let mut is_sil = vec![0.0f32; frames];
    let mut sil_count = 0f32;
    for f in 0..frames {
        let s: f32 = (0..cfg.n_spk).map(|k| pv[f * cfg.n_spk + k]).sum();
        if s < cfg.sil_threshold {
            is_sil[f] = 1.0;
            sil_count += 1.0;
        }
    }
    if sil_count == 0.0 {
        return Ok(());
    }
    let mask = Tensor::from_vec(is_sil, (1, frames, 1), state.device)?.to_dtype(state.dtype)?;
    let sil_emb_sum = emb_seq.broadcast_mul(&mask)?.sum_keepdim(1)?.reshape(vec![1, cfg.d_model])?;
    let upd_n = state.n_sil + sil_count;
    let total = state.mean_sil.affine(state.n_sil, 0.0)?.add(&sil_emb_sum)?;
    state.mean_sil = total.affine(1.0 / upd_n.max(1.0), 0.0)?;
    state.n_sil = upd_n;
    Ok(())
}

/// NeMo `_compress_spkcache`: оставить spkcache_len важнейших кадров (по preds).
fn compress_spkcache(
    emb_seq: &Tensor,
    preds: &Tensor,
    mean_sil: &Tensor,
    cfg: &StreamCfg,
) -> Result<(Tensor, Tensor)> {
    let f = preds.dims()[1];
    let n = cfg.n_spk;
    let pv = to_vec(preds)?;

    let per_spk = cfg.spkcache_len / n - cfg.spkcache_sil_frames_per_spk;
    let strong = (per_spk as f32 * cfg.strong_boost_rate).floor() as usize;
    let weak = (per_spk as f32 * cfg.weak_boost_rate).floor() as usize;
    let min_pos = (per_spk as f32 * cfg.min_pos_scores_rate).floor() as usize;
    let thr = cfg.pred_score_threshold;
    let ln05 = 0.5f32.ln();
    let neg_inf = f32::NEG_INFINITY;

    // _get_log_pred_scores.
    let mut sc = vec![0.0f32; f * n];
    for fr in 0..f {
        let mut log1sum = 0.0f32;
        for s in 0..n {
            log1sum += (1.0 - pv[fr * n + s]).clamp(thr, f32::INFINITY).ln();
        }
        for s in 0..n {
            let p = pv[fr * n + s];
            let lp = p.clamp(thr, f32::INFINITY).ln();
            let l1 = (1.0 - p).clamp(thr, f32::INFINITY).ln();
            sc[fr * n + s] = lp - l1 + log1sum - ln05;
        }
    }
    // _disable_low_scores.
    let mut pos_count = vec![0usize; n];
    for fr in 0..f {
        for s in 0..n {
            if pv[fr * n + s] <= 0.5 {
                sc[fr * n + s] = neg_inf;
            }
            if sc[fr * n + s] > 0.0 {
                pos_count[s] += 1;
            }
        }
    }
    for fr in 0..f {
        for s in 0..n {
            let speech = pv[fr * n + s] > 0.5;
            let is_pos = sc[fr * n + s] > 0.0;
            if !is_pos && speech && pos_count[s] >= min_pos {
                sc[fr * n + s] = neg_inf;
            }
        }
    }
    // scores_boost_latest: кадры >= spkcache_len.
    if cfg.scores_boost_latest > 0.0 && f > cfg.spkcache_len {
        for fr in cfg.spkcache_len..f {
            for s in 0..n {
                sc[fr * n + s] += cfg.scores_boost_latest;
            }
        }
    }
    boost_topk(&mut sc, f, n, strong, 2.0);
    boost_topk(&mut sc, f, n, weak, 1.0);

    // sil-pad +inf (spkcache_sil_frames_per_spk кадров на конце) → (F+sil, n).
    let sil = cfg.spkcache_sil_frames_per_spk;
    let fp = f + sil;

    // _get_topk_indices: flat spk-major s*fp+frame; top spkcache_len; -inf→max_index; sort asc.
    let mut flat: Vec<(f32, usize)> = Vec::with_capacity(fp * n);
    for s in 0..n {
        for fr in 0..fp {
            let v = if fr < f { sc[fr * n + s] } else { f32::INFINITY };
            flat.push((v, s * fp + fr));
        }
    }
    flat.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<usize> = flat
        .iter()
        .take(cfg.spkcache_len)
        .map(|&(v, idx)| if v == neg_inf { cfg.max_index } else { idx })
        .collect();
    chosen.sort_unstable();

    let mut idx_u32 = vec![0u32; cfg.spkcache_len];
    let mut disabled = vec![0.0f32; cfg.spkcache_len];
    for (p, &val) in chosen.iter().enumerate() {
        if val == cfg.max_index {
            disabled[p] = 1.0;
            continue;
        }
        let frame = val % fp;
        if frame >= f {
            disabled[p] = 1.0;
        } else {
            idx_u32[p] = frame as u32;
        }
    }

    let dev = emb_seq.device();
    let dt = emb_seq.dtype();
    let idx_t = Tensor::from_vec(idx_u32, (cfg.spkcache_len,), dev)?;
    let gathered = emb_seq.index_select(1, &idx_t)?; // (1, spkcache_len, d)
    let dmask = Tensor::from_vec(disabled.clone(), (1, cfg.spkcache_len, 1), dev)?.to_dtype(dt)?;
    let keep = dmask.affine(-1.0, 1.0)?; // 1 − disabled
    let mean_b = mean_sil.reshape(vec![1, 1, cfg.d_model])?;
    let sc_emb = gathered.broadcast_mul(&keep)?.broadcast_add(&mean_b.broadcast_mul(&dmask)?)?;

    let mut scp_out = vec![0.0f32; cfg.spkcache_len * n];
    for (p, &val) in chosen.iter().enumerate() {
        if disabled[p] != 0.0 {
            continue;
        }
        let frame = (val % fp).min(f - 1);
        for s in 0..n {
            scp_out[p * n + s] = pv[frame * n + s];
        }
    }
    let preds_out = Tensor::from_vec(scp_out, (1, cfg.spkcache_len, n), dev)?.to_dtype(dt)?;

    Ok((sc_emb, preds_out))
}

/// Top-`k` кадров по каждому спикеру получают += scale·ln2 (NeMo `_boost_topk_scores`, offset 0.5).
fn boost_topk(sc: &mut [f32], f: usize, n: usize, k: usize, scale: f32) {
    if k == 0 {
        return;
    }
    let delta = -scale * 0.5f32.ln();
    for s in 0..n {
        let mut order: Vec<usize> = (0..f).collect();
        order.sort_by(|&a, &b| {
            sc[b * n + s].partial_cmp(&sc[a * n + s]).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &fr in order.iter().take(k.min(f)) {
            sc[fr * n + s] += delta;
        }
    }
}
