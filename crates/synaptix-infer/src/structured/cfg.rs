//! Контекстно-свободная грамматика над байтами и её разбор «набором стеков»
//! (как GBNF в llama.cpp): состояние — все стеки позиций, которые ещё
//! согласуются с прочитанным. Неоднозначность (anyOf, необязательные поля)
//! разбирается без отката; хвостовая рекурсия стек не растит.

use std::sync::Arc;

/// Множество байтов (битовая маска 256).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ByteSet([u64; 4]);

impl ByteSet {
    pub fn one(b: u8) -> Self {
        let mut s = Self::default();
        s.insert(b);
        s
    }

    pub fn range(lo: u8, hi: u8) -> Self {
        let mut s = Self::default();
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }

    pub fn of(bytes: &[u8]) -> Self {
        let mut s = Self::default();
        for &b in bytes {
            s.insert(b);
        }
        s
    }

    pub fn insert(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1 << (b & 63);
    }

    pub fn union(mut self, o: Self) -> Self {
        for i in 0..4 {
            self.0[i] |= o.0[i];
        }
        self
    }

    pub fn minus(mut self, o: Self) -> Self {
        for i in 0..4 {
            self.0[i] &= !o.0[i];
        }
        self
    }

    #[inline]
    pub fn contains(&self, b: u8) -> bool {
        self.0[(b >> 6) as usize] & (1 << (b & 63)) != 0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Elem {
    Bytes(ByteSet),
    Rule(u32),
}

/// Правило — альтернативы, альтернатива — последовательность элементов.
#[derive(Clone, Debug, Default)]
pub struct Grammar {
    pub rules: Vec<Vec<Vec<Elem>>>,
    pub names: Vec<String>,
    pub start: u32,
}

impl Grammar {
    /// Новое пустое правило (заполнить — [`Self::set`]).
    pub fn rule(&mut self, name: &str) -> u32 {
        self.rules.push(Vec::new());
        self.names.push(name.to_string());
        (self.rules.len() - 1) as u32
    }

    pub fn set(&mut self, id: u32, alts: Vec<Vec<Elem>>) {
        self.rules[id as usize] = alts;
    }

    /// Правило-литерал.
    pub fn literal(&mut self, name: &str, s: &[u8]) -> u32 {
        let id = self.rule(name);
        self.set(id, vec![lit(s)]);
        id
    }

    /// Правила без альтернатив и ссылки на несуществующие правила — ошибка
    /// построения, а не тихая пустая маска на генерации.
    pub fn validate(&self) -> Result<(), String> {
        for (i, alts) in self.rules.iter().enumerate() {
            if alts.is_empty() {
                return Err(format!("правило «{}» пустое", self.names[i]));
            }
            for a in alts {
                for e in a {
                    if let Elem::Rule(r) = e {
                        if *r as usize >= self.rules.len() {
                            return Err(format!("правило «{}» ссылается на несуществующее {r}", self.names[i]));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Последовательность из байтов литерала.
pub fn lit(s: &[u8]) -> Vec<Elem> {
    s.iter().map(|&b| Elem::Bytes(ByteSet::one(b))).collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
struct Pos {
    rule: u32,
    alt: u16,
    idx: u16,
}

type Stack = Vec<Pos>;

/// Состояние разбора: стеки, у каждого наверху — элемент-байты; пустой
/// стек — грамматика может закончиться здесь.
#[derive(Clone, Debug)]
pub struct ParseState {
    stacks: Vec<Stack>,
}

/// Предел глубины раскрытия — защита от левой рекурсии в чужой грамматике.
const MAX_DEPTH: usize = 512;

impl ParseState {
    pub fn new(g: &Grammar) -> Self {
        let mut out = Vec::new();
        for alt in 0..g.rules[g.start as usize].len() {
            expand(g, vec![Pos { rule: g.start, alt: alt as u16, idx: 0 }], &mut out, 0);
        }
        dedup(&mut out);
        Self { stacks: out }
    }

    pub fn is_dead(&self) -> bool {
        self.stacks.is_empty()
    }

    /// Можно закончить здесь.
    pub fn accepting(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// Только закончить — продолжения нет.
    pub fn finished(&self) -> bool {
        !self.stacks.is_empty() && self.stacks.iter().all(|s| s.is_empty())
    }

    /// Байты, которые может принять следующий шаг.
    pub fn next_bytes(&self, g: &Grammar) -> ByteSet {
        let mut set = ByteSet::default();
        for s in &self.stacks {
            if let Some(top) = s.last() {
                if let Elem::Bytes(b) = g.rules[top.rule as usize][top.alt as usize][top.idx as usize] {
                    set = set.union(b);
                }
            }
        }
        set
    }

    pub fn advance(&self, g: &Grammar, byte: u8) -> Self {
        let mut out = Vec::with_capacity(self.stacks.len());
        for s in &self.stacks {
            let Some(top) = s.last() else { continue };
            let Elem::Bytes(set) = g.rules[top.rule as usize][top.alt as usize][top.idx as usize] else { continue };
            if !set.contains(byte) {
                continue;
            }
            let mut ns = s.clone();
            ns.last_mut().unwrap().idx += 1;
            expand(g, ns, &mut out, 0);
        }
        if out.len() > 1 {
            dedup(&mut out);
        }
        Self { stacks: out }
    }

    pub fn advance_bytes(&self, g: &Grammar, bytes: &[u8]) -> Self {
        let mut st = self.clone();
        for &b in bytes {
            st = st.advance(g, b);
            if st.is_dead() {
                break;
            }
        }
        st
    }
}

fn dedup(v: &mut Vec<Stack>) {
    v.sort_unstable();
    v.dedup();
}

/// Довести стек до элемента-байтов наверху (или до пустого).
fn expand(g: &Grammar, mut s: Stack, out: &mut Vec<Stack>, depth: usize) {
    if depth > MAX_DEPTH {
        return;
    }
    loop {
        let Some(top) = s.last().copied() else {
            out.push(s);
            return;
        };
        let seq = &g.rules[top.rule as usize][top.alt as usize];
        if top.idx as usize >= seq.len() {
            s.pop();
            continue;
        }
        match seq[top.idx as usize] {
            Elem::Bytes(_) => {
                out.push(s);
                return;
            }
            Elem::Rule(r) => {
                // Хвостовой вызов: продолжения после ссылки нет — кадр снимаем.
                if top.idx as usize + 1 >= seq.len() {
                    s.pop();
                } else {
                    s.last_mut().unwrap().idx += 1;
                }
                let n = g.rules[r as usize].len();
                for alt in 0..n {
                    let mut ns = if alt + 1 == n { std::mem::take(&mut s) } else { s.clone() };
                    ns.push(Pos { rule: r, alt: alt as u16, idx: 0 });
                    expand(g, ns, out, depth + 1);
                }
                return;
            }
        }
    }
}

/// Префиксное дерево байтов токенов словаря.
#[derive(Debug, Default)]
struct Trie {
    /// Дети узла: (байт, узел), по возрастанию байта.
    kids: Vec<Vec<(u8, u32)>>,
    /// Токены, которые кончаются в узле.
    ends: Vec<Vec<u32>>,
}

impl Trie {
    fn build(bytes: &[Option<Vec<u8>>]) -> Self {
        let mut t = Trie { kids: vec![Vec::new()], ends: vec![Vec::new()] };
        for (id, b) in bytes.iter().enumerate() {
            let Some(b) = b.as_ref().filter(|b| !b.is_empty()) else { continue };
            let mut node = 0usize;
            for &c in b {
                node = match t.kids[node].binary_search_by_key(&c, |k| k.0) {
                    Ok(i) => t.kids[node][i].1 as usize,
                    Err(i) => {
                        let n = t.kids.len() as u32;
                        t.kids.push(Vec::new());
                        t.ends.push(Vec::new());
                        t.kids[node].insert(i, (c, n));
                        n as usize
                    }
                };
            }
            t.ends[node].push(id as u32);
        }
        t
    }
}

/// Словарь модели для маски: байты токенов, их дерево и стоп-токены.
/// Строится один раз на токенизатор (на 150–250 тыс. токенов — десятки мс).
#[derive(Debug)]
pub struct TokenVocab {
    bytes: Vec<Option<Vec<u8>>>,
    trie: Trie,
    eos: Vec<u32>,
}

impl TokenVocab {
    pub fn new(bytes: Vec<Option<Vec<u8>>>, eos: Vec<u32>) -> Self {
        let trie = Trie::build(&bytes);
        Self { bytes, trie, eos }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn token(&self, id: u32) -> Option<&[u8]> {
        self.bytes.get(id as usize).and_then(|b| b.as_deref())
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }
}

/// Ограничитель генерации грамматикой: какие токены допустимы сейчас и
/// сдвиг по выбранному. Пока не встретился `trigger` (например `</think>`
/// у думающих моделей), всё разрешено — грамматика включается после него.
#[derive(Clone, Debug)]
pub struct GrammarMatcher {
    g: Arc<Grammar>,
    vocab: Arc<TokenVocab>,
    state: ParseState,
    trigger: Option<Vec<u8>>,
    /// Триггер — спецтокен (байтов у него нет): `</think>`, `<channel|>`.
    trigger_id: Option<u32>,
    tail: Vec<u8>,
}

impl GrammarMatcher {
    pub fn new(g: Arc<Grammar>, vocab: Arc<TokenVocab>) -> Self {
        let state = ParseState::new(&g);
        Self { g, vocab, state, trigger: None, trigger_id: None, tail: Vec::new() }
    }

    /// Грамматика — только после этой строки в выводе.
    pub fn with_trigger(mut self, trigger: &str) -> Self {
        if !trigger.is_empty() {
            self.trigger = Some(trigger.as_bytes().to_vec());
        }
        self
    }

    /// Грамматика — после этого токена.
    pub fn with_trigger_token(mut self, id: u32) -> Self {
        self.trigger_id = Some(id);
        self
    }

    pub fn active(&self) -> bool {
        self.trigger.is_none() && self.trigger_id.is_none()
    }

    /// Разбор пришёл в конец: дальше — только стоп-токен.
    pub fn finished(&self) -> bool {
        self.active() && self.state.finished()
    }

    pub fn accepting(&self) -> bool {
        self.active() && self.state.accepting()
    }

    /// Допустим ли токен.
    pub fn allows(&self, id: u32) -> bool {
        if !self.active() {
            return true;
        }
        if self.vocab.is_eos(id) {
            return self.state.accepting();
        }
        match self.vocab.token(id) {
            Some(b) if !b.is_empty() => !self.state.advance_bytes(&self.g, b).is_dead(),
            _ => false,
        }
    }

    /// Все допустимые токены (проход по дереву байтов с отсечением).
    pub fn allowed(&self) -> Vec<u32> {
        let mut out = Vec::new();
        if !self.active() {
            return (0..self.vocab.len() as u32).collect();
        }
        if self.state.accepting() {
            out.extend_from_slice(&self.vocab.eos);
        }
        self.walk(0, &self.state, &mut out);
        out
    }

    fn walk(&self, node: usize, st: &ParseState, out: &mut Vec<u32>) {
        let t = &self.vocab.trie;
        let next = st.next_bytes(&self.g);
        for &(b, child) in &t.kids[node] {
            if !next.contains(b) {
                continue;
            }
            let ns = st.advance(&self.g, b);
            if ns.is_dead() {
                continue;
            }
            out.extend_from_slice(&t.ends[child as usize]);
            if !t.kids[child as usize].is_empty() {
                self.walk(child as usize, &ns, out);
            }
        }
    }

    /// Недопустимым токенам — `-inf`.
    pub fn mask_logits(&self, logits: &mut [f32]) {
        if !self.active() {
            return;
        }
        let mut keep = vec![false; logits.len()];
        for id in self.allowed() {
            if let Some(k) = keep.get_mut(id as usize) {
                *k = true;
            }
        }
        for (l, k) in logits.iter_mut().zip(keep) {
            if !k {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Принять выбранный токен. `false` — токен грамматике не подходит
    /// (состояние не меняется).
    pub fn advance(&mut self, id: u32) -> bool {
        if let Some(t) = self.trigger_id {
            if t == id {
                self.trigger_id = None;
                self.trigger = None;
            }
            return true;
        }
        let bytes = self.vocab.token(id).unwrap_or(&[]);
        if let Some(trig) = &self.trigger {
            self.tail.extend_from_slice(bytes);
            if let Some(at) = find(&self.tail, trig) {
                // Хвост после триггера в том же токене — уже под грамматикой.
                let rest = self.tail[at + trig.len()..].to_vec();
                self.trigger = None;
                self.tail.clear();
                let st = self.state.advance_bytes(&self.g, &rest);
                if st.is_dead() {
                    return false;
                }
                self.state = st;
            } else if self.tail.len() > trig.len() * 4 {
                let cut = self.tail.len() - trig.len();
                self.tail.drain(..cut);
            }
            return true;
        }
        if self.vocab.is_eos(id) {
            return self.state.accepting();
        }
        let st = self.state.advance_bytes(&self.g, bytes);
        if st.is_dead() {
            return false;
        }
        self.state = st;
        true
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
