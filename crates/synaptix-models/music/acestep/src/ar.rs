
use synaptix_core::{dtype::DType, tensor::Tensor};
use synaptix_llm_common::{GenerationConfig, TokenSampler};
use synaptix_ops::rng::Philox4x32;

use crate::lm::AceStepLm;
use crate::tokenizer::{AceTokenizer, Metadata, NUM_AUDIO_CODES};
use crate::AceError;

pub const CODES_PER_SECOND: usize = 5;

#[derive(Debug, Clone)]
pub struct CodesGenOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub cfg_scale: f32,
    /// CFG decode as one batch-2 graph (cond+uncond, ~1.7× — weights read once).
    /// `false` keeps the legacy two-graph path (bit-different codes vs batched
    /// due to M=1 vs M=2 GEMM kernel selection; both are valid generations).
    pub batched_cfg: bool,
}

impl Default for CodesGenOptions {
    fn default() -> Self {
        Self {
            temperature: 0.85,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
            cfg_scale: 2.0,
            batched_cfg: true,
        }
    }
}

pub fn generate_codes(
    lm: &AceStepLm,
    tok: &AceTokenizer,
    caption: &str,
    lyrics: &str,
    meta: &Metadata,
    opts: &CodesGenOptions,
) -> Result<Vec<u32>, AceError> {
    let prompt = tok.build_codes_prompt(caption, lyrics, meta);
    let ids = tok.encode(&prompt)?;
    let target = meta.duration as usize * CODES_PER_SECOND;
    let device = lm.device;
    let use_cfg = opts.cfg_scale > 1.0;

    let uncond_ids = if use_cfg {
        tok.encode(&tok.build_codes_prompt_uncond())?
    } else {
        Vec::new()
    };

    let max_seq = ids.len().max(uncond_ids.len()) + target + 8;
    let mut kv = lm.make_kv(1, max_seq)?;
    let id_t = Tensor::from_vec(ids.clone(), vec![1usize, ids.len()], device)?;
    let mut logits = lm.forward(&id_t, &mut kv)?;

    let mut ukv = if use_cfg { Some(lm.make_kv(1, max_seq)?) } else { None };
    let mut ulogits = if let Some(uk) = ukv.as_mut() {
        let ut = Tensor::from_vec(uncond_ids.clone(), vec![1usize, uncond_ids.len()], device)?;
        Some(lm.forward(&ut, uk)?)
    } else {
        None
    };

    let base = tok.audio_base() as usize;
    let n_audio = NUM_AUDIO_CODES as usize;
    let mut rng = Philox4x32::new(opts.seed);

    let code0 = sample_code(&logits, ulogits.as_ref(), base, n_audio, opts, &mut rng)?;
    let mut codes = vec![code0];
    codes.truncate(target);
    if codes.len() >= target {
        return Ok(codes);
    }

    {
        if let synaptix_core::device::Device::Cuda(ord) = device {
            if let Some(mut uk) = ukv.take() {
                // CFG: cond+uncond as one batch-2 decode (weights read once
                // instead of twice) with per-row positions in the KV/RoPE.
                // opts.batched_cfg=false keeps the legacy two-graph path.
                if opts.batched_cfg {
                    decode_codes_graph_batched(lm, kv, uk, &mut codes, target, base, n_audio, ids.len(), uncond_ids.len(), opts, &mut rng, ord)?;
                } else {
                    decode_codes_graph(lm, &mut kv, Some(&mut uk), &mut codes, target, base, n_audio, ids.len(), uncond_ids.len(), opts, &mut rng, ord)?;
                }
            } else {
                decode_codes_graph(lm, &mut kv, None, &mut codes, target, base, n_audio, ids.len(), uncond_ids.len(), opts, &mut rng, ord)?;
            }
            return Ok(codes);
        }
    }

    while codes.len() < target {
        let tok_id = base as u32 + *codes.last().unwrap();
        let nxt = Tensor::from_vec(vec![tok_id], vec![1usize, 1usize], device)?;
        logits = lm.forward(&nxt, &mut kv)?;
        if let Some(uk) = ukv.as_mut() {
            ulogits = Some(lm.forward(&nxt, uk)?);
        }
        let code = sample_code(&logits, ulogits.as_ref(), base, n_audio, opts, &mut rng)?;
        codes.push(code);
    }
    Ok(codes)
}

fn sample_code(
    logits_c: &Tensor,
    logits_u: Option<&Tensor>,
    base: usize,
    n_audio: usize,
    opts: &CodesGenOptions,
    rng: &mut Philox4x32,
) -> Result<u32, AceError> {
    let cond = logits_c.narrow(1, base, n_audio)?.to_dtype(DType::F32)?;
    let merged = if let Some(uc) = logits_u {
        let unc = uc.narrow(1, base, n_audio)?.to_dtype(DType::F32)?;
        let diff = cond.broadcast_add(&unc.affine(-1.0, 0.0)?)?;
        unc.broadcast_add(&diff.affine(opts.cfg_scale, 0.0)?)?
    } else {
        cond
    };
    let logit_vec: Vec<f32> = merged.flatten_all()?.to_vec1()?;
    Ok(sample_logits(&logit_vec, opts, rng))
}

#[allow(clippy::too_many_arguments)]
fn decode_codes_graph(
    lm: &AceStepLm,
    kv: &mut synaptix_llm_common::KvCache,
    ukv: Option<&mut synaptix_llm_common::KvCache>,
    codes: &mut Vec<u32>,
    target: usize,
    base: usize,
    n_audio: usize,
    lc: usize,
    lu: usize,
    opts: &CodesGenOptions,
    rng: &mut Philox4x32,
    ord: usize,
) -> Result<(), AceError> {
    use synaptix_core::grad::no_grad;
    use synaptix_infer::error::InferError;
    use synaptix_infer::graph_capture::GraphCapturer;

    let model = &lm.model;
    let stream = synaptix_core::device::cuda::default_stream(ord)
        .map_err(|e| AceError::Other(format!("default_stream: {e}")))?;
    let code0 = *codes.last().unwrap();

    let mut state_c = model.make_decode_state().map_err(|e| AceError::Other(e.to_string()))?;
    state_c.update(base as u32 + code0, lc as u32).map_err(|e| AceError::Other(e.to_string()))?;
    let mut cap_c = GraphCapturer::new(3);
    let graph_c = {
        let sr = &mut state_c;
        let kr = &mut *kv;
        no_grad(|| {
            cap_c.capture_with(&stream, |_| {
                model.forward_decode_dev(sr, kr).map_err(|e| InferError::Other(e.to_string()))
            })
        })
        .map_err(|e| AceError::Other(format!("graph capture cond: {e}")))?
    };
    graph_c.upload().map_err(|e| AceError::Other(format!("graph upload cond: {e}")))?;

    let mut state_u: Option<synaptix_llm_common::DecodeState> = None;
    let mut graph_u = None;
    if let Some(uk) = ukv {
        let mut su = model.make_decode_state().map_err(|e| AceError::Other(e.to_string()))?;
        su.update(base as u32 + code0, lu as u32).map_err(|e| AceError::Other(e.to_string()))?;
        let mut cap_u = GraphCapturer::new(3);
        let gu = {
            let sr = &mut su;
            let kr = &mut *uk;
            no_grad(|| {
                cap_u.capture_with(&stream, |_| {
                    model.forward_decode_dev(sr, kr).map_err(|e| InferError::Other(e.to_string()))
                })
            })
            .map_err(|e| AceError::Other(format!("graph capture uncond: {e}")))?
        };
        gu.upload().map_err(|e| AceError::Other(format!("graph upload uncond: {e}")))?;
        state_u = Some(su);
        graph_u = Some(gu);
    }

    let code1 = sample_code(&state_c.logits, state_u.as_ref().map(|s| &s.logits), base, n_audio, opts, rng)?;
    codes.push(code1);

    while codes.len() < target {
        let last = *codes.last().unwrap();
        let pos_c = (lc + codes.len() - 1) as u32;
        state_c.update(base as u32 + last, pos_c).map_err(|e| AceError::Other(e.to_string()))?;
        graph_c.launch().map_err(|e| AceError::Other(format!("graph launch cond: {e:?}")))?;
        if let (Some(su), Some(gu)) = (state_u.as_mut(), graph_u.as_ref()) {
            let pos_u = (lu + codes.len() - 1) as u32;
            su.update(base as u32 + last, pos_u).map_err(|e| AceError::Other(e.to_string()))?;
            gu.launch().map_err(|e| AceError::Other(format!("graph launch uncond: {e:?}")))?;
        }
        stream.synchronize().map_err(|e| AceError::Other(format!("sync: {e:?}")))?;
        let code = sample_code(&state_c.logits, state_u.as_ref().map(|s| &s.logits), base, n_audio, opts, rng)?;
        codes.push(code);
    }
    Ok(())
}

/// Concatenate two batch-1 KV caches into a batch-2 cache (row 0 = a / cond,
/// row 1 = b / uncond). One-time copy before the batched decode loop.
fn merge_kv2(
    a: &synaptix_llm_common::KvCache,
    b: &synaptix_llm_common::KvCache,
) -> Result<synaptix_llm_common::KvCache, AceError> {
    use synaptix_llm_common::{KvCache, KvCacheLayer, LayerCache};
    let mut layers = Vec::with_capacity(a.layers.len());
    for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
        match (la, lb) {
            (LayerCache::Full(ka), LayerCache::Full(kb)) => {
                if ka.k_scale.is_some() || kb.k_scale.is_some() {
                    return Err(AceError::Other("batched AR decode: mxfp8 KV not supported".into()));
                }
                let k = Tensor::cat(&[&ka.k, &kb.k], 0)?;
                let v = Tensor::cat(&[&ka.v, &kb.v], 0)?;
                layers.push(LayerCache::Full(KvCacheLayer { k, v, k_scale: None, v_scale: None }));
            }
            _ => return Err(AceError::Other("batched AR decode: non-full layer cache".into())),
        }
    }
    Ok(KvCache { layers, seq_len: a.seq_len.max(b.seq_len), max_seq: a.max_seq })
}

fn sample_code_batched(
    logits2: &Tensor,
    base: usize,
    n_audio: usize,
    opts: &CodesGenOptions,
    rng: &mut Philox4x32,
) -> Result<u32, AceError> {
    let cond = logits2.narrow(0, 0, 1)?;
    let uncond = logits2.narrow(0, 1, 1)?;
    sample_code(&cond, Some(&uncond), base, n_audio, opts, rng)
}

/// Batched CFG decode: cond (row 0) + uncond (row 1) in ONE batch-2 graph with
/// per-row absolute positions (`lc+t` / `lu+t`). Reads the LM weights once per
/// step instead of twice (the dominant decode cost), ~1.8× over two graphs.
#[allow(clippy::too_many_arguments)]
fn decode_codes_graph_batched(
    lm: &AceStepLm,
    kv_c: synaptix_llm_common::KvCache,
    kv_u: synaptix_llm_common::KvCache,
    codes: &mut Vec<u32>,
    target: usize,
    base: usize,
    n_audio: usize,
    lc: usize,
    lu: usize,
    opts: &CodesGenOptions,
    rng: &mut Philox4x32,
    ord: usize,
) -> Result<(), AceError> {
    use synaptix_core::grad::no_grad;
    use synaptix_infer::error::InferError;
    use synaptix_infer::graph_capture::GraphCapturer;

    let model = &lm.model;
    let stream = synaptix_core::device::cuda::default_stream(ord)
        .map_err(|e| AceError::Other(format!("default_stream: {e}")))?;
    let mut kv2 = merge_kv2(&kv_c, &kv_u)?;
    drop(kv_c);
    drop(kv_u);
    let code0 = *codes.last().unwrap();

    let mut state =
        model.make_decode_state_batched(2).map_err(|e| AceError::Other(e.to_string()))?;
    let tok0 = base as u32 + code0;
    state
        .update_batched(&[tok0, tok0], &[lc as u32, lu as u32])
        .map_err(|e| AceError::Other(e.to_string()))?;
    let mut cap = GraphCapturer::new(3);
    let graph = {
        let sr = &mut state;
        let kr = &mut kv2;
        no_grad(|| {
            cap.capture_with(&stream, |_| {
                model.forward_decode_dev(sr, kr).map_err(|e| InferError::Other(e.to_string()))
            })
        })
        .map_err(|e| AceError::Other(format!("graph capture batched: {e}")))?
    };
    graph.upload().map_err(|e| AceError::Other(format!("graph upload batched: {e}")))?;

    let code1 = sample_code_batched(&state.logits, base, n_audio, opts, rng)?;
    codes.push(code1);

    while codes.len() < target {
        let last = *codes.last().unwrap();
        let tok = base as u32 + last;
        let pos_c = (lc + codes.len() - 1) as u32;
        let pos_u = (lu + codes.len() - 1) as u32;
        state
            .update_batched(&[tok, tok], &[pos_c, pos_u])
            .map_err(|e| AceError::Other(e.to_string()))?;
        graph.launch().map_err(|e| AceError::Other(format!("graph launch batched: {e:?}")))?;
        stream.synchronize().map_err(|e| AceError::Other(format!("sync: {e:?}")))?;
        let code = sample_code_batched(&state.logits, base, n_audio, opts, rng)?;
        codes.push(code);
    }
    Ok(())
}

fn sample_logits(logits: &[f32], opts: &CodesGenOptions, rng: &mut Philox4x32) -> u32 {
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i as u32, v)) } else { None })
        .collect();
    if cand.is_empty() {
        return 0;
    }
    if opts.temperature == 0.0 {
        let mut best = cand[0];
        for &c in cand.iter() {
            if c.1 > best.1 {
                best = c;
            }
        }
        return best.0;
    }
    if opts.top_k > 0 && opts.top_k < cand.len() {
        cand.select_nth_unstable_by(opts.top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        cand.truncate(opts.top_k);
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if opts.min_p > 0.0 {
        let max = cand[0].1;
        let exps: Vec<f32> = cand.iter().map(|&(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-10);
        let thr = opts.min_p * (exps[0] / sum);
        let keep = exps.iter().take_while(|&&e| e / sum >= thr).count().max(1);
        cand.truncate(keep);
    }
    if opts.top_p > 0.0 && opts.top_p < 1.0 {
        let max = cand[0].1;
        let exps: Vec<f32> = cand.iter().map(|&(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-10);
        let mut cumsum = 0.0f32;
        let mut keep = cand.len();
        for (i, e) in exps.iter().enumerate() {
            cumsum += e / sum;
            if i > 0 && cumsum > opts.top_p {
                keep = i;
                break;
            }
        }
        cand.truncate(keep.max(1));
    }
    if (opts.temperature - 1.0).abs() > 1e-6 {
        let inv = 1.0 / opts.temperature;
        for c in cand.iter_mut() {
            c.1 *= inv;
        }
    }
    if cand.len() == 1 {
        return cand[0].0;
    }
    let max = cand.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = cand.iter().map(|&(_, v)| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-10);
    let u = rng.next_u32() as f64 as f32 / u32::MAX as f32;
    let mut cumsum = 0.0f32;
    for (i, e) in exps.iter().enumerate() {
        cumsum += e / sum;
        if u < cumsum {
            return cand[i].0;
        }
    }
    cand[cand.len() - 1].0
}

pub fn generate_phase1(
    lm: &AceStepLm,
    tok: &AceTokenizer,
    caption: &str,
    lyrics: &str,
    base: &Metadata,
    opts: &CodesGenOptions,
    max_tokens: usize,
) -> Result<Metadata, AceError> {
    let prompt = tok.build_cot_prompt(caption, lyrics);
    let ids = tok.encode(&prompt)?;
    let device = lm.device;
    let think_end = tok.think_end_id();
    let max_seq = ids.len() + max_tokens + 8;
    let mut kv = lm.make_kv(1, max_seq)?;
    let id_t = Tensor::from_vec(ids.clone(), vec![1usize, ids.len()], device)?;
    let mut logits = lm.forward(&id_t, &mut kv)?;
    let gcfg = GenerationConfig {
        temperature: opts.temperature,
        top_k: opts.top_k,
        top_p: opts.top_p,
        repetition_penalty: opts.repetition_penalty,
        seed: opts.seed,
        ..GenerationConfig::default()
    };
    let mut sampler = TokenSampler::new(&gcfg, &ids);
    let mut gen: Vec<u32> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let tok_id = sampler.sample(&logits).map_err(|e| AceError::Other(e.to_string()))?;
        gen.push(tok_id);
        if Some(tok_id) == think_end || tok_id == tok.eos() {
            break;
        }
        let nxt = Tensor::from_vec(vec![tok_id], vec![1usize, 1usize], device)?;
        logits = lm.forward(&nxt, &mut kv)?;
    }
    let text = tok.decode(&gen)?;
    Ok(tok.parse_metadata(&text, base))
}

#[allow(clippy::too_many_arguments)]
pub fn ar_generate(
    lm: &AceStepLm,
    tok: &AceTokenizer,
    caption: &str,
    lyrics: &str,
    base: &Metadata,
    opts: &CodesGenOptions,
    use_cot: bool,
) -> Result<(Vec<u32>, Metadata), AceError> {
    let meta = if use_cot {
        generate_phase1(lm, tok, caption, lyrics, base, opts, 512)?
    } else {
        base.clone()
    };
    let cap = if meta.caption.is_empty() { caption } else { &meta.caption };
    let codes = generate_codes(lm, tok, cap, lyrics, &meta, opts)?;
    Ok((codes, meta))
}
