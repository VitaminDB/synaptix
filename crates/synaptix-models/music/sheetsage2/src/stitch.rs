//! Песня длиннее окна модели: окна по 300 с, перевод позиций сетки во время,
//! склейка принятых событий и префикс перекрытия для следующего окна.
//!
//! Всё здесь повторяет релиз до бита, включая его численные примитивы:
//! `np.median`, `np.interp`, `np.clip` и питоновский `round` (к чётному).
//! Отличие в любой из этих мелочей сдвигает время события на тик — и ABC,
//! собранный по сетке долей, получается другим.

use std::collections::BTreeMap;

use crate::vocab::{Decoded, Event, Field, TokenType, Vocab};
use crate::SheetError;

const EPS: f64 = 1e-4;

/// Одно окно плана. `generation_stop` — где остановить генерацию (секунды от
/// начала окна); `None` — окно последнее, идти до конца записи.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub start: f64,
    pub end: f64,
    pub accept_start: f64,
    pub accept_end: f64,
    pub prefix_end: f64,
    pub generation_stop: Option<f64>,
}

/// План скользящих окон. Дефолт релиза: окно 300 с, перекрытие 200 с, из них
/// 100 с — правый контекст, который видит энкодер, но не принимает склейка.
pub fn sliding_window_plan(
    duration: f64,
    window_seconds: f64,
    overlap_seconds: f64,
    lookahead_seconds: f64,
) -> Result<Vec<Window>, SheetError> {
    if !(duration > 0.0) || !(window_seconds > 0.0) {
        return Err(SheetError::Config("длительность и окно должны быть положительными".into()));
    }
    if !(0.0 <= lookahead_seconds && lookahead_seconds <= overlap_seconds && overlap_seconds < window_seconds) {
        return Err(SheetError::Config("нужно 0 <= lookahead <= overlap < окна".into()));
    }
    let hop = window_seconds - overlap_seconds;
    let (mut start, mut accepted) = (0.0f64, 0.0f64);
    let mut out = Vec::new();
    loop {
        let last = start + window_seconds >= duration - 1e-6;
        let accept_end = if last { duration } else { start + window_seconds - lookahead_seconds };
        out.push(Window {
            start,
            end: duration.min(start + window_seconds),
            accept_start: accepted,
            accept_end,
            prefix_end: accepted,
            generation_stop: if last { None } else { Some(window_seconds - lookahead_seconds) },
        });
        if last {
            return Ok(out);
        }
        accepted = accept_end;
        start = (start + hop).min(duration - window_seconds);
    }
}

/// Питоновский `round()` — половины к чётному.
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// `np.median`: среднее двух средних при чётной длине.
pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `np.interp` строго внутри `[xp[0], xp[-1]]` — края обрабатывает вызывающий.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    // j: xp[j] <= x < xp[j+1]
    let j = match xp.partition_point(|&v| v <= x) {
        0 => 0,
        p => p - 1,
    };
    if j + 1 >= xp.len() {
        return fp[fp.len() - 1];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let res = slope * (x - xp[j]) + fp[j];
    if res.is_nan() {
        slope * (x - xp[j + 1]) + fp[j + 1]
    } else {
        res
    }
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Позиция сетки → секунды от начала окна, по меткам времени событий.
#[derive(Debug, Clone)]
pub struct TimeMap {
    steps: Vec<f64>,
    times: Vec<f64>,
    step_seconds: f64,
    target: f64,
}

impl TimeMap {
    pub fn new(decoded: &Decoded, target_seconds: f64) -> Self {
        // dict(anchors): на одну позицию — последнее значение; дальше по возрастанию.
        let mut anchors: BTreeMap<i64, f64> = BTreeMap::new();
        for event in &decoded.events {
            if let Some(t) = event.values.timestamp {
                anchors.insert(event.subbeat, t);
            }
        }
        let steps: Vec<f64> = anchors.keys().map(|&s| s as f64).collect();
        let times: Vec<f64> = anchors.values().copied().collect();
        let mut step_seconds = 0.125;
        if steps.len() >= 2 {
            let ratios: Vec<f64> = (1..steps.len())
                .map(|i| (times[i] - times[i - 1]) / (steps[i] - steps[i - 1]).max(1.0))
                .collect();
            let m = median(&ratios);
            if m.is_finite() && m > 0.0 {
                step_seconds = m;
            }
        }
        Self { steps, times, step_seconds, target: target_seconds }
    }

    pub fn lookup(&self, step: i64) -> f64 {
        let step = step as f64;
        if self.steps.is_empty() {
            return self.target.min((step * 0.125).max(0.0));
        }
        let (s0, t0) = (self.steps[0], self.times[0]);
        let (s1, t1) = (*self.steps.last().unwrap(), *self.times.last().unwrap());
        if step <= s0 {
            return clip(t0 + (step - s0) * self.step_seconds, 0.0, self.target);
        }
        if step >= s1 {
            return clip(t1 + (step - s1) * self.step_seconds, 0.0, self.target);
        }
        interp(step, &self.steps, &self.times)
    }
}

/// Разбор выданных токенов: строго, а при «мягких» ошибках (пустое событие,
/// поле неактивной задачи) — нестрого, с предупреждением.
pub fn decode_generated(vocab: &Vocab, tokens: &[u32]) -> Result<(Decoded, Option<String>), SheetError> {
    match vocab.decode_sequence(tokens, true) {
        Ok(d) => Ok((d, None)),
        Err(SheetError::Sequence(msg))
            if msg.contains("empty event at subbeat") || msg.contains("belongs to inactive output field") =>
        {
            let decoded = vocab.decode_sequence(tokens, false)?;
            Ok((decoded, Some(msg)))
        }
        Err(e) => Err(e),
    }
}

/// События окна, попавшие в его полосу приёма, — с абсолютным временем и
/// концами нот.
pub fn stitched_window_events(
    decoded: &Decoded,
    map: &TimeMap,
    window: &Window,
    song_duration: f64,
    window_index: usize,
    global_subbeat_base: i64,
) -> Vec<Event> {
    let mut accepted = Vec::new();
    for event in &decoded.events {
        let local = map.lookup(event.subbeat);
        let abs_time = window.start + local;
        if abs_time < window.accept_start - EPS {
            continue;
        }
        if abs_time >= window.accept_end - EPS || abs_time >= song_duration - EPS {
            continue;
        }
        let mut out = Event {
            subbeat: event.subbeat,
            tokens: event.tokens.clone(),
            values: event.values.clone(),
            ..Default::default()
        };
        let time = clip(abs_time, 0.0, song_duration);
        out.time = Some(time);
        out.window_index = window_index;
        out.window_start = window.start;
        out.source_subbeat = event.subbeat;
        out.global_subbeat = global_subbeat_base + event.subbeat;
        if out.values.timestamp.is_some() {
            out.values.timestamp = Some(time);
        }
        if let Some(notes) = out.values.melody.as_mut() {
            for note in notes.iter_mut() {
                let local_end = map.lookup(event.subbeat + note.duration_steps as i64);
                let end = window.start + local_end;
                note.end_time = Some(song_duration.min((time + 0.04).max(end)));
            }
        }
        accepted.push(out);
    }
    accepted
}

/// Контекст (секция, тональность, аккорд, размер), действующий к моменту `time_abs`.
fn active_context_before(events: &[Event], vocab: &Vocab, time_abs: f64) -> Result<[Option<Vec<u32>>; 4], SheetError> {
    // [structure, key, chord, meter]
    let mut state: [Option<Vec<u32>>; 4] = Default::default();
    for event in events {
        let Some(t) = event.time else { continue };
        if t > time_abs + 1e-6 {
            continue;
        }
        for (slot, field) in [(0, Field::Structure), (1, Field::Key), (2, Field::Chord)] {
            let tokens = event.field_tokens(field);
            if !tokens.is_empty() {
                state[slot] = Some(tokens.to_vec());
            }
        }
        let mut meters = Vec::new();
        for &t in event.field_tokens(Field::Rhythm) {
            if vocab.token_type(t)? == TokenType::Meter {
                meters.push(t);
            }
        }
        if !meters.is_empty() {
            state[3] = Some(meters[..1].to_vec());
        }
    }
    Ok(state)
}

/// Префикс перекрытия: уже принятые события из `[window_start, prefix_end)`,
/// перенумерованные от первой доли и с локальными метками времени. Модель
/// продолжает с них, а не начинает окно с чистого листа. `None` — в перекрытии
/// нет ни одной доли. Возвращает токены префикса и базовую позицию сетки.
pub fn build_overlap_prefix(
    stitched: &[Event],
    vocab: &Vocab,
    prompts: &[usize],
    window_start: f64,
    prefix_end: f64,
) -> Result<Option<(Vec<u32>, i64)>, SheetError> {
    let mut source: Vec<&Event> = stitched
        .iter()
        .filter(|e| {
            let t = e.time.unwrap_or(-1.0);
            window_start - EPS <= t && t < prefix_end - EPS
        })
        .collect();
    source.sort_by(|a, b| {
        a.global_subbeat
            .cmp(&b.global_subbeat)
            .then(a.time.unwrap_or(0.0).total_cmp(&b.time.unwrap_or(0.0)))
    });
    let Some(first) = source
        .iter()
        .position(|e| e.values.timestamp.is_some() || e.values.rhythm.is_some())
    else {
        return Ok(None);
    };
    let source = &source[first..];
    let base = source[0].global_subbeat;
    let context = active_context_before(stitched, vocab, source[0].time.unwrap_or(0.0))?;
    let mut events = Vec::with_capacity(source.len());
    for src in source {
        let mut event = Event { tokens: src.tokens.clone(), ..Default::default() };
        event.subbeat = (src.global_subbeat - base).max(0);
        if !event.tokens[Field::Timestamp as usize].is_empty() {
            let local = src.time.unwrap_or(0.0) - window_start;
            let id = py_round(local * vocab.time_hz as f64) as i64;
            let id = id.clamp(0, vocab.n_time_tokens as i64 - 1) as u32;
            event.tokens[Field::Timestamp as usize] = vec![vocab.time_id_to_token(id)?];
        }
        vocab.refresh_values(&mut event)?;
        events.push(event);
    }
    // Первое событие получает действующие секцию, тональность, аккорд и размер.
    let head = &mut events[0];
    for (slot, field) in [(0, Field::Structure), (1, Field::Key), (2, Field::Chord)] {
        if head.tokens[field as usize].is_empty() {
            if let Some(tokens) = &context[slot] {
                head.tokens[field as usize] = tokens.clone();
            }
        }
    }
    let rhythm = head.tokens[Field::Rhythm as usize].clone();
    let mut has_meter = false;
    let mut has_eighth = false;
    for &t in &rhythm {
        match vocab.token_type(t)? {
            TokenType::Meter => has_meter = true,
            TokenType::EighthPosition => has_eighth = true,
            _ => {}
        }
    }
    if has_eighth && !has_meter {
        if let Some(meter) = &context[3] {
            let mut tokens = meter.clone();
            tokens.extend(rhythm);
            head.tokens[Field::Rhythm as usize] = tokens;
        }
    }
    vocab.refresh_values(head)?;
    let decoded = Decoded { prompts: prompts.to_vec(), events, has_eos: false };
    Ok(Some((vocab.encode_decoded(&decoded)?, base)))
}

/// Склейка окон по мере генерации: префикс перекрытия для очередного окна,
/// приём его событий и итоговая сортировка.
pub struct Stitcher<'a> {
    vocab: &'a Vocab,
    prompts: Vec<usize>,
    duration: f64,
    window_length: f64,
    stitched: Vec<Event>,
    pub warnings: Vec<String>,
}

impl<'a> Stitcher<'a> {
    pub fn new(vocab: &'a Vocab, prompts: &[usize], duration: f64, window_length: f64) -> Self {
        Self {
            vocab,
            prompts: prompts.to_vec(),
            duration,
            window_length,
            stitched: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Префикс для окна `index` (у первого окна его нет) и базовая позиция сетки.
    pub fn prefix_for(&self, index: usize, window: &Window) -> Result<Option<(Vec<u32>, i64)>, SheetError> {
        if index == 0 {
            return Ok(None);
        }
        build_overlap_prefix(&self.stitched, self.vocab, &self.prompts, window.start, window.prefix_end)
    }

    /// Принять выданные окном токены (префикс в их начале). Возвращает число
    /// событий окна и число принятых.
    pub fn accept(
        &mut self,
        index: usize,
        window: &Window,
        tokens: &[u32],
        base: i64,
    ) -> Result<(usize, usize), SheetError> {
        let (decoded, warning) = decode_generated(self.vocab, tokens)?;
        if let Some(w) = warning {
            self.warnings.push(w);
        }
        let map = TimeMap::new(&decoded, self.window_length);
        let accepted = stitched_window_events(&decoded, &map, window, self.duration, index, base);
        let counts = (decoded.events.len(), accepted.len());
        self.stitched.extend(accepted);
        Ok(counts)
    }

    /// Все принятые события по времени (при равенстве — по позиции сетки).
    pub fn finish(mut self) -> (Vec<Event>, Vec<String>) {
        self.stitched.sort_by(|a, b| {
            a.time
                .unwrap_or(0.0)
                .total_cmp(&b.time.unwrap_or(0.0))
                .then(a.global_subbeat.cmp(&b.global_subbeat))
        });
        (self.stitched, self.warnings)
    }
}

/// Окно записи: `seconds` секунд с `start`, хвост дополнен нулями.
pub fn slice_audio(audio: &[f32], sample_rate: u32, start: f64, seconds: f64) -> Vec<f32> {
    let offset = py_round(start * sample_rate as f64) as usize;
    let count = py_round(seconds * sample_rate as f64) as usize;
    let mut out = vec![0f32; count];
    if offset < audio.len() {
        let n = (audio.len() - offset).min(count);
        out[..n].copy_from_slice(&audio[offset..offset + n]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_matches_release_for_long_song() {
        let plan = sliding_window_plan(419.19733333333335, 300.0, 200.0, 100.0).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!((plan[0].start, plan[0].accept_end, plan[0].generation_stop), (0.0, 200.0, Some(200.0)));
        assert_eq!((plan[1].start, plan[1].accept_start, plan[1].accept_end), (100.0, 200.0, 300.0));
        assert_eq!(plan[2].start, 119.19733333333335);
        assert_eq!((plan[2].accept_start, plan[2].generation_stop), (300.0, None));
    }

    #[test]
    fn short_song_is_one_window() {
        let plan = sliding_window_plan(59.998666666666665, 300.0, 200.0, 100.0).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].accept_end, 59.998666666666665);
        assert_eq!(plan[0].generation_stop, None);
    }

    #[test]
    fn numpy_median_and_python_round() {
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        assert_eq!(py_round(-0.5), -0.0);
    }
}
