//! Темп речи без смены высоты (WSOLA) и деление текста на куски для
//! потокового синтеза.

/// Изменить темп: `rate` > 1 — быстрее, < 1 — медленнее; высота голоса та же.
/// WSOLA: окна Ханна по 30 мс с перекрытием 50 %, сдвиг входа подбирается
/// по максимуму корреляции в пределах ±10 мс, чтобы не рвать период голоса.
pub fn time_stretch(pcm: &[f32], sample_rate: u32, rate: f32) -> Vec<f32> {
    let rate = rate.clamp(0.25, 4.0);
    if (rate - 1.0).abs() < 1e-3 || pcm.len() < 4 {
        return pcm.to_vec();
    }
    let win = ((sample_rate as f32 * 0.030) as usize).max(16) & !1;
    let hop_out = win / 2;
    let hop_in = hop_out as f32 * rate;
    let tol = (sample_rate as f32 * 0.010) as usize;
    let window: Vec<f32> = (0..win).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win as f32).cos()).collect();
    let out_len = (pcm.len() as f32 / rate) as usize + win;
    let mut out = vec![0f32; out_len];
    let mut norm = vec![0f32; out_len];
    let at = |i: isize| -> f32 { if i >= 0 && (i as usize) < pcm.len() { pcm[i as usize] } else { 0.0 } };
    // Позиция «естественного продолжения» прошлого куска во входе.
    let mut prev: isize = 0;
    let mut k = 0usize;
    loop {
        let out_pos = k * hop_out;
        let nominal = (k as f32 * hop_in) as isize;
        if nominal as usize >= pcm.len() || out_pos + win > out_len {
            break;
        }
        let pos = if k == 0 {
            0
        } else {
            // Ищем сдвиг около номинала, лучше всего совпадающий с
            // продолжением прошлого окна (prev + hop_out).
            let target = prev + hop_out as isize;
            let (mut best, mut best_c) = (nominal, f32::NEG_INFINITY);
            let step = if tol > 64 { 2 } else { 1 };
            let mut d = -(tol as isize);
            while d <= tol as isize {
                let cand = nominal + d;
                let mut c = 0f32;
                for i in (0..hop_out).step_by(2) {
                    c += at(cand + i as isize) * at(target + i as isize);
                }
                if c > best_c {
                    best_c = c;
                    best = cand;
                }
                d += step;
            }
            best
        };
        for i in 0..win {
            out[out_pos + i] += at(pos + i as isize) * window[i];
            norm[out_pos + i] += window[i];
        }
        prev = pos;
        k += 1;
    }
    let len = ((pcm.len() as f32 / rate) as usize).min(out_len);
    out.truncate(len);
    for (o, n) in out.iter_mut().zip(norm) {
        if n > 1e-3 {
            *o /= n;
        }
    }
    out
}

/// Текст для потокового синтеза: по предложениям (`.!?…` и переводы
/// строк), короткие склеиваются со следующими до `min_chars`; первый кусок
/// может быть короче — раньше начнётся звук.
pub fn speech_chunks(text: &str, min_chars: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '\n' {
            if !cur.trim().is_empty() {
                sentences.push(cur.trim().to_string());
            }
            cur.clear();
            continue;
        }
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？');
        let next = chars.get(i + 1).copied();
        // Конец предложения: знак и пробел/конец (не «3.5», не «e.g.x»).
        if end && next.is_none_or(|n| n.is_whitespace() || n == '"' || n == '»' || n == ')') {
            if let Some(q) = next.filter(|n| matches!(n, '"' | '»' | ')')) {
                cur.push(q);
            }
            sentences.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }
    sentences.retain(|s| !s.is_empty() && s != "\"" && s != "»" && s != ")");
    let mut out: Vec<String> = Vec::new();
    for s in sentences {
        // Первый кусок — отдельно (раньше звук), дальше короткие склеиваются.
        let n = out.len();
        match out.last_mut() {
            Some(last) if n >= 2 && last.chars().count() < min_chars => {
                last.push(' ');
                last.push_str(&s);
            }
            _ => out.push(s),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_length_and_pitch() {
        let sr = 16000;
        let f = 220.0;
        let pcm: Vec<f32> = (0..sr * 2).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin() * 0.5).collect();
        for rate in [0.75f32, 1.25, 1.5] {
            let out = time_stretch(&pcm, sr as u32, rate);
            let want = (pcm.len() as f32 / rate) as usize;
            assert!((out.len() as isize - want as isize).abs() < 10, "{rate}: {} против {want}", out.len());
            // Частота — по пересечениям нуля в середине.
            let mid = &out[out.len() / 4..out.len() * 3 / 4];
            let zc = mid.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
            let est = zc as f32 * sr as f32 / mid.len() as f32;
            assert!((est - f).abs() < 6.0, "{rate}: частота {est}");
        }
    }

    #[test]
    fn chunks() {
        let c = speech_chunks("Hello there. How are you doing today? I am fine, thanks! It costs 3.5 dollars. Ok.", 30);
        assert_eq!(c[0], "Hello there.");
        assert!(c.iter().any(|s| s.contains("3.5 dollars")));
        assert!(c.last().unwrap().ends_with("Ok."));
        let c = speech_chunks("Сәлем! Қалың қалай?\nЖақсы.", 4);
        assert_eq!(c, vec!["Сәлем!", "Қалың қалай?", "Жақсы."]);
    }
}
