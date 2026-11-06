//! Расстановщик ударений и «ё» Silero (`models/accentor.py`, `AccentorNgram`): fastText-подобный мешок
//! n-грамм слова (`<слово>`, n = 1…len+3, среднее эмбеддингов 16) → два MLP (ударная гласная, «ё»),
//! словарь исключений; правила — как в оригинале (пользовательские «+» и «ё», односложные слова).

use std::collections::HashMap;

use crate::{Result, SileroError};

const VOWELS: &str = "аоуыэиеяёю";
const STRESS: char = '+';

/// Вариант токенизации: v5…v5_5 — по пробелам (стоп-слово «то»), v5_6 — и по пунктуации.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccentorKind {
    Old,
    New,
}

struct Mlp {
    layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>, // (W [out, in], b, in, out)
}

impl Mlp {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        let n = self.layers.len();
        for (li, (w, b, i, o)) in self.layers.iter().enumerate() {
            let mut y = b.clone();
            for r in 0..*o {
                let row = &w[r * i..(r + 1) * i];
                y[r] += row.iter().zip(&h).map(|(a, c)| a * c).sum::<f32>();
            }
            if li + 1 < n {
                y.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            h = y;
        }
        h
    }
}

pub struct Accentor {
    ngrams: HashMap<String, usize>,
    emb: Vec<f32>,
    dim: usize,
    stress: Mlp,
    yo: Mlp,
    exceptions: HashMap<String, (i64, i64)>,
    pub kind: AccentorKind,
}

fn lower(s: &str) -> String {
    s.chars().flat_map(|c| c.to_lowercase()).collect()
}

fn is_cyr_letter(c: char) -> bool {
    matches!(c, 'А'..='Я' | 'а'..='я' | 'ё' | 'Ё')
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

impl Accentor {
    /// `tensors`: имя → (данные, форма) из компонента `accentor`.
    pub fn new(ngrams_txt: &str, exceptions_tsv: &str, tensors: &dyn Fn(&str) -> Result<(Vec<f32>, Vec<usize>)>, kind: AccentorKind) -> Result<Self> {
        let ngrams: HashMap<String, usize> = ngrams_txt.lines().enumerate().map(|(i, l)| (l.to_string(), i)).collect();
        let (emb, shape) = tensors("embedding.weight")?;
        let dim = *shape.last().unwrap_or(&16);
        let mlp = |p: &str| -> Result<Mlp> {
            let mut layers = Vec::new();
            for k in [0, 2, 4, 6] {
                let (w, s) = tensors(&format!("{p}.{k}.weight"))?;
                let (b, _) = tensors(&format!("{p}.{k}.bias"))?;
                layers.push((w, b, s[1], s[0]));
            }
            Ok(Mlp { layers })
        };
        let mut exceptions = HashMap::new();
        for l in exceptions_tsv.lines() {
            let mut it = l.split('\t');
            if let (Some(w), Some(s), Some(y)) = (it.next(), it.next(), it.next()) {
                exceptions.insert(w.to_string(), (s.parse().unwrap_or(0), y.parse().unwrap_or(-1)));
            }
        }
        Ok(Self { ngrams, emb, dim, stress: mlp("stress_clf")?, yo: mlp("yo_clf")?, exceptions, kind })
    }

    /// Эмбеддинг слова: среднее по известным n-граммам (`UNK`, если ни одной).
    fn embed(&self, word: &str) -> Result<Vec<f32>> {
        let s: Vec<char> = format!("<{word}>").chars().collect();
        let wl = word.chars().count();
        let mut ids = Vec::new();
        for n in 1..=wl + 3 {
            if n > s.len() {
                continue;
            }
            for i in 0..=s.len() - n {
                let g: String = s[i..i + n].iter().collect();
                if let Some(&id) = self.ngrams.get(&g) {
                    ids.push(id);
                }
            }
        }
        if wl < 1 {
            if let Some(&id) = self.ngrams.get(word) {
                ids.push(id);
            }
        }
        if ids.is_empty() {
            ids.push(*self.ngrams.get("UNK").ok_or_else(|| SileroError::Other("в n-граммах нет UNK".into()))?);
        }
        let mut e = vec![0f32; self.dim];
        for id in &ids {
            let r = &self.emb[id * self.dim..(id + 1) * self.dim];
            e.iter_mut().zip(r).for_each(|(a, b)| *a += b);
        }
        e.iter_mut().for_each(|v| *v /= ids.len() as f32);
        Ok(e)
    }

    /// (вероятности ударения, ударная гласная, вероятности «ё», номер «е»).
    fn predict(&self, word: &str) -> Result<(Vec<f32>, usize, Vec<f32>, usize)> {
        let e = self.embed(word)?;
        let sp = softmax(&self.stress.forward(&e));
        let yp = softmax(&self.yo.forward(&e));
        let (s, y) = (argmax(&sp), argmax(&yp));
        Ok((sp, s, yp, y))
    }

    /// Токены (сырые), вход модели (только кириллица, нижний регистр), нужна ли обработка.
    fn tokenize(&self, sentence: &str) -> Vec<(String, String, bool)> {
        let chars: Vec<char> = sentence.chars().collect();
        let sep = |c: char| match self.kind {
            AccentorKind::Old => c.is_whitespace(),
            AccentorKind::New => c.is_whitespace() || ".,!?;:<>=()/\\".contains(c),
        };
        // re.split с группой: чередование слов и разделителей (первое и последнее — слова, возможно пустые)
        let mut pieces: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut i = 0;
        while i < chars.len() {
            if sep(chars[i]) {
                pieces.push(std::mem::take(&mut cur));
                let mut s = String::new();
                while i < chars.len() && sep(chars[i]) {
                    s.push(chars[i]);
                    i += 1;
                }
                pieces.push(s);
            } else {
                cur.push(chars[i]);
                i += 1;
            }
        }
        pieces.push(cur);
        let mut out = Vec::new();
        for word in pieces {
            let parts: Vec<&str> = word.split('-').collect();
            let (toks, masks): (Vec<String>, Vec<bool>) = if parts.len() == 1 {
                (vec![word.clone()], vec![true])
            } else {
                let mut t: Vec<String> = parts[..parts.len() - 1].iter().map(|p| format!("{p}-")).collect();
                t.push(parts[parts.len() - 1].to_string());
                let mut m = vec![true; parts.len() - 1];
                m.push(parts[parts.len() - 1] != "то");
                (t, m)
            };
            for (t, m) in toks.into_iter().zip(masks) {
                let clean: String = lower(&t).chars().filter(|&c| is_cyr_letter(c)).collect();
                let need = !clean.is_empty() && m;
                out.push((t, clean, need));
            }
        }
        out
    }

    fn positions(word: &[char], stressed: &[usize], yo_ids: &[i64]) -> (Vec<usize>, Vec<usize>, usize, i64) {
        let vowel_ids: Vec<usize> = word.iter().enumerate().filter(|(_, c)| VOWELS.contains(**c)).map(|(i, _)| i).collect();
        let ye_ids: Vec<usize> = word.iter().enumerate().filter(|(_, &c)| c == 'е').map(|(i, _)| i).collect();
        let stress_pos = stressed.iter().filter(|&&i| i < vowel_ids.len()).map(|&i| vowel_ids[i]).collect();
        let yo_pos = yo_ids.iter().filter(|&&i| i > 0 && ((i - 1) as usize) < ye_ids.len()).map(|&i| ye_ids[(i - 1) as usize]).collect();
        let first = vowel_ids.first().map(|&v| v as i64).unwrap_or(-1);
        (stress_pos, yo_pos, vowel_ids.len(), first)
    }

    fn exception(&self, clean: &str, raw: &str, have_stress: bool) -> String {
        let (exc_stress, exc_yo) = self.exceptions[clean];
        let mut w: Vec<char> = raw.chars().collect();
        let set_yo = |w: &mut Vec<char>, p: usize| {
            if p < w.len() {
                w[p] = if w[p].is_lowercase() { 'ё' } else { 'Ё' };
            }
        };
        if have_stress {
            let user: Vec<usize> = w.iter().enumerate().filter(|(_, &c)| c == STRESS).map(|(i, _)| i).collect();
            let mut a: Vec<char> = w.iter().copied().filter(|&c| c != STRESS).collect();
            if exc_yo != -1 && user.contains(&((exc_yo + 1) as usize)) {
                set_yo(&mut a, exc_yo as usize);
            }
            for p in user {
                a.insert(p.min(a.len()), STRESS);
            }
            a.into_iter().collect()
        } else {
            if exc_yo != -1 {
                set_yo(&mut w, exc_yo as usize);
            }
            w.insert((exc_stress as usize).min(w.len()), STRESS);
            w.into_iter().collect()
        }
    }

    /// Ударения и «ё» в предложении (как `AccentorNgram.__call__` с put_stress, put_yo,
    /// stress_single_vowel = True, без пропусков).
    pub fn apply(&self, sentence: &str) -> Result<String> {
        let mut out = String::new();
        for (raw, clean, need) in self.tokenize(sentence) {
            if !need {
                out.push_str(&raw);
                continue;
            }
            let rl = lower(&raw);
            let have_stress = rl.contains(STRESS);
            let have_yo = rl.contains('ё');
            if have_stress && have_yo {
                out.push_str(&raw);
                continue;
            }
            if !have_stress && have_yo {
                // ударение на каждую «ё» пользователя
                let mut w: Vec<char> = raw.chars().collect();
                let pos: Vec<usize> = rl.chars().enumerate().filter(|(_, c)| *c == 'ё').map(|(i, _)| i).collect();
                for (i, p) in pos.into_iter().enumerate() {
                    w.insert(p + i, STRESS);
                }
                out.extend(w);
                continue;
            }
            if self.exceptions.contains_key(&clean) {
                out.push_str(&self.exception(&clean, &raw, have_stress));
                continue;
            }
            let (sp, s, yp, y) = self.predict(&clean)?;
            let mut stressed = vec![s];
            let set_stress = sp[s] > 0.5 && !have_stress;
            let yo_ids = [y as i64];
            let set_yo = yp[y] > 0.5;
            let rlc: Vec<char> = rl.chars().collect();
            if have_stress {
                stressed = rl.split(STRESS).map(|part| part.chars().filter(|c| VOWELS.contains(*c)).count()).collect();
            }
            let (mut stress_pos, yo_pos, nv, first) = Self::positions(&rlc, &stressed, &yo_ids);
            if nv == 0 {
                out.push_str(&raw);
                continue;
            }
            let mut w: Vec<char> = raw.chars().collect();
            for p in yo_pos {
                if stress_pos.contains(&p) && set_yo && rlc[p] == 'е' {
                    w[p] = if w[p].is_lowercase() { 'ё' } else { 'Ё' };
                }
            }
            let mut set_stress = set_stress;
            if nv == 1 {
                stress_pos = vec![first as usize];
                set_stress = true;
            }
            if !have_stress && set_stress {
                for (i, p) in stress_pos.into_iter().enumerate() {
                    w.insert((p + i).min(w.len()), STRESS);
                }
            }
            out.extend(w);
        }
        Ok(out)
    }
}
