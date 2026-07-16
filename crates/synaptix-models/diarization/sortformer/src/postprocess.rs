//! Постпроцессинг per-frame probs (12.5 Hz, 4 спикера) → сегменты диаризации.
//!
//! Алгоритм (порт официального NVIDIA NeMo postprocess):
//! binarize(thr) → median_smooth → per-speaker contiguous intervals (confidence=avg prob)
//! → merge gaps<merge_gap_s → drop<min_segment_s → arrival-time re-id. TODO: реализовать.

use serde::{Deserialize, Serialize};

/// Один сегмент речи одного спикера.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiarizeSegment {
    pub start_s: f32,
    pub end_s: f32,
    /// 0-indexed speaker id (по порядку первого появления — arrival-time re-id).
    pub speaker: u8,
    /// Средняя вероятность спикера по сегменту.
    pub confidence: f32,
}

/// Параметры постпроцессинга (бинаризация + сглаживание + merge/filter).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PostprocessParams {
    pub threshold: f32,
    pub min_segment_s: f32,
    pub merge_gap_s: f32,
    /// Частота кадров post-subsampling (12.5 Hz для v2.1).
    pub frame_rate_hz: f32,
    pub max_speakers: usize,
    /// Окно медианного сглаживания (нечётное; 3 для v2.1).
    pub smoothing_frames: usize,
}

impl Default for PostprocessParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_segment_s: 0.25,
            merge_gap_s: 0.15,
            frame_rate_hz: 12.5,
            max_speakers: 4,
            smoothing_frames: 3,
        }
    }
}

/// Результат диаризации куска аудио.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationResult {
    pub version: u32,
    pub sample_rate: u32,
    pub duration_s: f32,
    pub num_speakers: usize,
    pub segments: Vec<DiarizeSegment>,
}

pub const RESULT_VERSION: u32 = 1;

/// Бинаризация (per-spk threshold, overlap разрешён) → median-smooth(window) → per-speaker
/// непрерывные интервалы (confidence = средняя prob) → merge gaps<merge_gap_s →
/// drop<min_segment_s → arrival-time re-id. `probs` = row-major `[n_frames, n_spk]`.
pub fn frames_to_segments(
    probs: &[f32],
    n_frames: usize,
    n_spk: usize,
    p: &PostprocessParams,
) -> Vec<DiarizeSegment> {
    if n_frames == 0 {
        return Vec::new();
    }
    let dt = 1.0f32 / p.frame_rate_hz;

    // active[s][t] (бинарно) + median smooth по времени (majority в окне).
    let half = p.smoothing_frames / 2;
    let raw = |s: usize, t: usize| probs[t * n_spk + s] > p.threshold;
    let mut active = vec![vec![false; n_frames]; n_spk];
    for s in 0..n_spk {
        for t in 0..n_frames {
            if p.smoothing_frames <= 1 {
                active[s][t] = raw(s, t);
                continue;
            }
            let lo = t.saturating_sub(half);
            let hi = (t + half).min(n_frames - 1);
            let (mut on, mut total) = (0usize, 0usize);
            for u in lo..=hi {
                total += 1;
                if raw(s, u) {
                    on += 1;
                }
            }
            active[s][t] = on * 2 > total;
        }
    }

    // per-speaker непрерывные интервалы (в кадрах) с confidence.
    let mut segs: Vec<DiarizeSegment> = Vec::new();
    for s in 0..n_spk {
        let mut t = 0usize;
        while t < n_frames {
            if !active[s][t] {
                t += 1;
                continue;
            }
            let start = t;
            let mut sum = 0.0f32;
            while t < n_frames && active[s][t] {
                sum += probs[t * n_spk + s];
                t += 1;
            }
            let len = t - start;
            segs.push(DiarizeSegment {
                start_s: start as f32 * dt,
                end_s: t as f32 * dt,
                speaker: s as u8,
                confidence: sum / len as f32,
            });
        }
    }

    // merge gaps < merge_gap_s внутри одного спикера.
    segs.sort_by(|a, b| {
        a.speaker.cmp(&b.speaker).then(a.start_s.partial_cmp(&b.start_s).unwrap())
    });
    let mut merged: Vec<DiarizeSegment> = Vec::new();
    for seg in segs {
        if let Some(last) = merged.last_mut() {
            if last.speaker == seg.speaker && seg.start_s - last.end_s < p.merge_gap_s {
                let w1 = last.end_s - last.start_s;
                let w2 = seg.end_s - seg.start_s;
                last.confidence = (last.confidence * w1 + seg.confidence * w2) / (w1 + w2).max(1e-6);
                last.end_s = seg.end_s;
                continue;
            }
        }
        merged.push(seg);
    }

    // drop коротких.
    merged.retain(|s| s.end_s - s.start_s >= p.min_segment_s);

    // arrival-time re-id: новые id по порядку первого появления спикера.
    let mut order: Vec<(u8, f32)> = Vec::new();
    for s in &merged {
        if !order.iter().any(|(sp, _)| *sp == s.speaker) {
            order.push((s.speaker, s.start_s));
        }
    }
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let remap = |old: u8| order.iter().position(|(sp, _)| *sp == old).unwrap_or(0) as u8;
    for s in &mut merged {
        s.speaker = remap(s.speaker);
    }

    merged.sort_by(|a, b| {
        a.start_s.partial_cmp(&b.start_s).unwrap().then(a.speaker.cmp(&b.speaker))
    });
    merged
}
