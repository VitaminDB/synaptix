//! Слова с таймкодами по методу openai/whisper (`timing.py`): cross-attn
//! выбранных голов → softmax по кадрам → нормализация по токенам → медианный
//! фильтр → DTW токены×кадры (кадр 20 мс) → границы слов по склейке токенов.

/// Слово с таймкодами (секунды от начала аудио) и уверенностью.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// Слово без ведущего пробела, с приклеенной пунктуацией.
    pub text: String,
    pub start: f32,
    pub end: f32,
    /// Среднее вероятностей токенов слова.
    pub prob: f32,
    /// Минимум вероятностей токенов слова.
    pub prob_min: f32,
}

/// Кадров энкодера в секунду (20 мс).
pub const TOKENS_PER_SECOND: f32 = 50.0;

const PREPEND_PUNCT: &str = "\"'“¿([{-";
const APPEND_PUNCT: &str = "\"'.。,，!！?？:：”)]}、";
/// `string.punctuation` Python (проверка — подстрока, как в openai).
const PY_PUNCT: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const SENTENCE_END: &str = ".。!！?？";

/// Слово выравнивания внутри окна (время — от начала окна).
#[derive(Debug, Clone)]
pub(crate) struct RawWord {
    pub word: String,
    pub tokens: Vec<u32>,
    pub start: f32,
    pub end: f32,
    pub probs: Vec<f32>,
}

/// Softmax по кадрам каждой строки `[rows × cols]` на месте.
fn softmax_rows(x: &mut [f32], cols: usize) {
    for row in x.chunks_mut(cols) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in row.iter_mut() {
            *v /= s;
        }
    }
}

/// Медианный фильтр по строкам с отражением краёв (как `median_filter` openai).
pub(crate) fn median_filter_rows(x: &mut [f32], cols: usize, width: usize) {
    let pad = width / 2;
    if cols <= pad {
        return;
    }
    let mut buf = vec![0f32; cols + 2 * pad];
    let mut win = vec![0f32; width];
    for row in x.chunks_mut(cols) {
        // reflect: [x2 x1 | x0 x1 ... | xn-2 xn-3]
        for i in 0..pad {
            buf[i] = row[pad - i];
            buf[pad + cols + i] = row[cols - 2 - i];
        }
        buf[pad..pad + cols].copy_from_slice(row);
        for j in 0..cols {
            if width == 7 {
                let mut w7 = [0f32; 7];
                w7.copy_from_slice(&buf[j..j + 7]);
                row[j] = median7(&mut w7);
            } else {
                win.copy_from_slice(&buf[j..j + width]);
                win.sort_unstable_by(|a, b| a.total_cmp(b));
                row[j] = win[pad];
            }
        }
    }
}

/// DTW по стоимости `x[n × m]` (openai `dtw_cpu`): индексы (токен, кадр) пути.
pub(crate) fn dtw(x: &[f32], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let w = m + 1;
    let mut cost = vec![f32::INFINITY; (n + 1) * w];
    let mut trace = vec![-1i8; (n + 1) * w];
    cost[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[(i - 1) * w + j - 1];
            let c1 = cost[(i - 1) * w + j];
            let c2 = cost[i * w + j - 1];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = x[(i - 1) * m + j - 1] + c;
            trace[i * w + j] = t;
        }
    }
    for j in 0..w {
        trace[j] = 2;
    }
    for i in 0..=n {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::with_capacity(n + m);
    while i > 0 || j > 0 {
        path.push((i.wrapping_sub(1), j.wrapping_sub(1)));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path.into_iter().unzip()
}

/// Медиана 7 значений (сеть сравнений вместо сортировки).
#[inline]
fn median7(v: &mut [f32; 7]) -> f32 {
    #[inline]
    fn cs(v: &mut [f32; 7], a: usize, b: usize) {
        if v[a] > v[b] {
            v.swap(a, b);
        }
    }
    // оптимальная сеть сортировки 7 элементов (16 сравнений)
    for &(a, b) in &[
        (0, 6), (2, 3), (4, 5), (0, 2), (1, 4), (3, 6), (0, 1), (2, 5), (3, 4), (1, 2), (4, 6), (2, 3), (4, 5),
        (1, 2), (3, 4), (5, 6),
    ] {
        cs(v, a, b);
    }
    v[3]
}

/// Одна голова: softmax по кадрам, z-нормировка по токенам, медианный фильтр 7.
fn head_matrix(head: &[f32], rows: usize, t_enc: usize, frames: usize) -> Vec<f32> {
    let mut w: Vec<f32> = (0..rows).flat_map(|r| head[r * t_enc..r * t_enc + frames].iter().copied()).collect();
    softmax_rows(&mut w, frames);
    // std/mean по токенам (на каждый кадр), без смещения.
    for f in 0..frames {
        let mean = (0..rows).map(|r| w[r * frames + f]).sum::<f32>() / rows as f32;
        let var = (0..rows).map(|r| (w[r * frames + f] - mean).powi(2)).sum::<f32>() / rows as f32;
        let std = var.sqrt().max(1e-12);
        for r in 0..rows {
            w[r * frames + f] = (w[r * frames + f] - mean) / std;
        }
    }
    median_filter_rows(&mut w, frames, 7);
    w
}

/// Матрица выравнивания: `qk` по головам `[rows × t_enc]` → нормализованная,
/// отфильтрованная и усреднённая по головам `[rows × frames]`. Головы — параллельно.
pub(crate) fn alignment_matrix(qk: &[Vec<f32>], rows: usize, t_enc: usize, frames: usize) -> Vec<f32> {
    let frames = frames.clamp(1, t_enc);
    let heads: Vec<Vec<f32>> = std::thread::scope(|s| {
        let hs: Vec<_> = qk.iter().map(|h| s.spawn(move || head_matrix(h, rows, t_enc, frames))).collect();
        hs.into_iter().map(|h| h.join().expect("alignment head")).collect()
    });
    let mut acc = vec![0f32; rows * frames];
    for w in &heads {
        for (a, v) in acc.iter_mut().zip(w) {
            *a += v;
        }
    }
    let k = qk.len().max(1) as f32;
    acc.iter_mut().for_each(|a| *a /= k);
    acc
}

/// Языки без пробелов между словами — слова по символам Unicode.
pub(crate) fn no_space_language(lang: &str) -> bool {
    matches!(lang, "zh" | "ja" | "th" | "lo" | "my" | "yue")
}

/// Группы токенов, декодирующиеся в целые символы Unicode.
pub(crate) fn split_on_unicode(tokens: &[u32], decode: &dyn Fn(&[u32]) -> String) -> (Vec<String>, Vec<Vec<u32>>) {
    const REPL: char = '\u{FFFD}';
    let full: Vec<char> = decode(tokens).chars().collect();
    let (mut words, mut groups) = (Vec::new(), Vec::new());
    let mut cur: Vec<u32> = Vec::new();
    let mut offset = 0usize;
    for &t in tokens {
        cur.push(t);
        let d = decode(&cur);
        let ok = match d.chars().position(|c| c == REPL) {
            None => true,
            Some(p) => full.get(offset + p) == Some(&REPL),
        };
        if ok {
            offset += d.chars().count();
            words.push(d);
            groups.push(std::mem::take(&mut cur));
        }
    }
    (words, groups)
}

/// Слова: новое слово с пробела, спецтокена или пунктуации (openai
/// `split_tokens_on_spaces`); для языков без пробелов — по символам.
pub(crate) fn split_to_words(
    tokens: &[u32],
    eot: u32,
    lang: &str,
    decode: &dyn Fn(&[u32]) -> String,
) -> (Vec<String>, Vec<Vec<u32>>) {
    let (sub, sub_tok) = split_on_unicode(tokens, decode);
    if no_space_language(lang) {
        return (sub, sub_tok);
    }
    let (mut words, mut groups): (Vec<String>, Vec<Vec<u32>>) = (Vec::new(), Vec::new());
    for (s, t) in sub.into_iter().zip(sub_tok) {
        let special = t[0] >= eot;
        let with_space = s.starts_with(' ');
        let punct = PY_PUNCT.contains(s.trim());
        if special || with_space || punct || words.is_empty() {
            words.push(s);
            groups.push(t);
        } else {
            words.last_mut().unwrap().push_str(&s);
            groups.last_mut().unwrap().extend(t);
        }
    }
    (words, groups)
}

/// Склейка пунктуации с соседними словами (openai `merge_punctuations`).
pub(crate) fn merge_punctuations(a: &mut [RawWord]) {
    if a.len() < 2 {
        return;
    }
    // ведущая пунктуация → к следующему слову
    let mut i = a.len() as isize - 2;
    let mut j = a.len() - 1;
    while i >= 0 {
        let iu = i as usize;
        let prev = &a[iu];
        if prev.word.starts_with(' ') && PREPEND_PUNCT.contains(prev.word.trim()) {
            let (w, t, p) = (std::mem::take(&mut a[iu].word), std::mem::take(&mut a[iu].tokens), std::mem::take(&mut a[iu].probs));
            a[j].word = w + &a[j].word;
            a[j].tokens.splice(0..0, t);
            a[j].probs.splice(0..0, p);
        } else {
            j = iu;
        }
        i -= 1;
    }
    // замыкающая пунктуация → к предыдущему слову
    let (mut i, mut j) = (0usize, 1usize);
    while j < a.len() {
        if !a[i].word.ends_with(' ') && !a[j].word.is_empty() && APPEND_PUNCT.contains(a[j].word.as_str()) {
            let (w, t, p) = (std::mem::take(&mut a[j].word), std::mem::take(&mut a[j].tokens), std::mem::take(&mut a[j].probs));
            a[i].word.push_str(&w);
            a[i].tokens.extend(t);
            a[i].probs.extend(p);
        } else {
            i = j;
        }
        j += 1;
    }
}

/// Медиана и предел длительности слова (openai: `min(0.7, медиана)`, ×2).
pub(crate) fn duration_limits(words: &[RawWord]) -> (f32, f32) {
    let mut d: Vec<f32> = words.iter().map(|w| w.end - w.start).filter(|&x| x > 0.0).collect();
    if d.is_empty() {
        return (0.0, 0.0);
    }
    d.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = d.len();
    let med = if n % 2 == 1 { d[n / 2] } else { 0.5 * (d[n / 2 - 1] + d[n / 2]) };
    let med = med.min(0.7);
    (med, med * 2.0)
}

/// Обрезка слишком длинных слов у границ предложений (openai).
pub(crate) fn truncate_at_sentence_ends(words: &mut [RawWord], max_dur: f32) {
    if max_dur <= 0.0 {
        return;
    }
    for i in 1..words.len() {
        if words[i].end - words[i].start > max_dur {
            if SENTENCE_END.contains(words[i].word.as_str()) && !words[i].word.is_empty() {
                words[i].end = words[i].start + max_dur;
            } else if SENTENCE_END.contains(words[i - 1].word.as_str()) && !words[i - 1].word.is_empty() {
                words[i].start = words[i].end - max_dur;
            }
        }
    }
}

/// Первое (и второе) слово после паузы не длиннее предела (openai: «hack»).
pub(crate) fn truncate_after_pause(words: &mut [WordTiming], last_speech: f32, med: f32, max_dur: f32) {
    if words.is_empty() || max_dur <= 0.0 {
        return;
    }
    let long_first = words[0].end - words[0].start > max_dur
        || (words.len() > 1 && words[1].end - words[0].start > max_dur * 2.0);
    if words[0].end - last_speech > med * 4.0 && long_first {
        if words.len() > 1 && words[1].end - words[1].start > max_dur {
            let b = (words[1].end / 2.0).max(words[1].end - max_dur);
            words[0].end = b;
            words[1].start = b;
        }
        words[0].start = (words[0].end - max_dur).max(0.0);
    }
}

/// Паузы (тишина ≥ `min_sil` с) по энергии: кадр 10 мс, порог — 25 дБ ниже
/// уровня речи (95-й перцентиль), но не ниже шума (10-й) + 6 дБ. Интервалы в секундах от начала `pcm`.
pub fn silences(pcm: &[f32], sr: u32, min_sil: f32) -> Vec<(f32, f32)> {
    let hop = (sr / 100) as usize;
    let win = hop * 5 / 2;
    if pcm.len() < win {
        return Vec::new();
    }
    let n = (pcm.len() - win) / hop + 1;
    let db: Vec<f32> = (0..n)
        .map(|i| {
            let f = &pcm[i * hop..i * hop + win];
            let e = f.iter().map(|x| x * x).sum::<f32>() / win as f32;
            10.0 * e.max(1e-10).log10()
        })
        .collect();
    let mut sorted = db.clone();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let floor = sorted[n / 10];
    let speech = sorted[(n * 95 / 100).min(n - 1)];
    if speech - floor < 10.0 {
        return Vec::new(); // нет контраста речь/тишина
    }
    // порог от уровня речи (подобрано по MFA на LibriSpeech и KSC)
    let thr = (floor + 6.0).max(speech - 25.0);
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if db[i] < thr {
            let j = (i..n).find(|&k| db[k] >= thr).unwrap_or(n);
            // кадр i покрывает [i·hop, i·hop+win): тишина — центры кадров
            let a = (i * hop + win / 2) as f32 / sr as f32;
            let b = if j == n { pcm.len() as f32 / sr as f32 } else { (j * hop + win / 2) as f32 / sr as f32 };
            let a = if i == 0 { 0.0 } else { a };
            if b - a >= min_sil {
                out.push((a, b));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Подтяжка границ слов к паузам: для каждой паузы — ближайшая граница между
/// словами (или край фразы); слова до неё кончаются не позже начала паузы,
/// слова после — начинаются не раньше её конца; соседние с паузой края,
/// не дотянувшиеся до неё (до `snap` с), прижимаются к ней.
pub fn snap_to_silences(words: &mut [WordTiming], sil: &[(f32, f32)], snap: f32) {
    let n = words.len();
    if n == 0 {
        return;
    }
    for &(a, b) in sil {
        let c = 0.5 * (a + b);
        // граница k: между словами k-1 и k (0 — перед первым, n — после последнего)
        let bt = |k: usize| -> f32 {
            if k == 0 {
                words[0].start
            } else if k == n {
                words[n - 1].end
            } else {
                0.5 * (words[k - 1].end + words[k].start)
            }
        };
        let k = (0..=n).min_by(|&x, &y| (bt(x) - c).abs().total_cmp(&(bt(y) - c).abs())).unwrap();
        let t = bt(k);
        if t < a - snap || t > b + snap {
            continue;
        }
        // слово не должно целиком уйти в паузу: проверка до правки
        if k > 0 && words[k - 1].start >= a {
            continue;
        }
        if k < n && words[k].end <= b {
            continue;
        }
        if k > 0 {
            let w = &mut words[k - 1];
            if w.end > a || a - w.end <= snap {
                w.end = a;
            }
        }
        if k < n {
            let w = &mut words[k];
            if w.start < b || w.start - b <= snap {
                w.start = b;
            }
        }
        for w in words[..k.saturating_sub(1)].iter_mut() {
            w.end = w.end.min(a);
            w.start = w.start.min(w.end);
        }
        for w in words.iter_mut().skip(k + 1) {
            w.start = w.start.max(b);
            w.end = w.end.max(w.start);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_diagonal() {
        // 3 токена × 6 кадров: минимум стоимости — блоки по 2 кадра.
        let mut x = vec![1f32; 18];
        for (r, c) in [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)] {
            x[r * 6 + c] = -1.0;
        }
        let (ti, fi) = dtw(&x, 3, 6);
        assert_eq!(ti, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(fi, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn median7_matches_sort() {
        let mut x: Vec<f32> = (0..200).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
        let mut y = x.clone();
        median_filter_rows(&mut x, 200, 7);
        // эталон — сортировка
        let pad = 3;
        let mut buf = vec![0f32; 206];
        for i in 0..pad {
            buf[i] = y[pad - i];
            buf[pad + 200 + i] = y[200 - 2 - i];
        }
        buf[pad..pad + 200].copy_from_slice(&y);
        for j in 0..200 {
            let mut w = buf[j..j + 7].to_vec();
            w.sort_by(|a, b| a.total_cmp(b));
            y[j] = w[3];
        }
        assert_eq!(x, y);
    }

    #[test]
    fn median_reflect() {
        let mut x = vec![5.0, 1.0, 9.0, 2.0, 7.0];
        median_filter_rows(&mut x, 5, 3);
        // отражение: [1,5,1,9,2,7,2]
        assert_eq!(x, vec![1.0, 5.0, 2.0, 7.0, 2.0]);
    }

    fn rw(w: &str) -> RawWord {
        RawWord { word: w.into(), tokens: vec![0], start: 0.0, end: 0.0, probs: vec![1.0] }
    }

    #[test]
    fn punct_merge() {
        let mut a = vec![rw(" \""), rw(" Hello"), rw(","), rw(" world"), rw(".")];
        merge_punctuations(&mut a);
        let w: Vec<&str> = a.iter().map(|r| r.word.as_str()).filter(|s| !s.is_empty()).collect();
        assert_eq!(w, vec![" \" Hello,", " world."]);
    }
}
