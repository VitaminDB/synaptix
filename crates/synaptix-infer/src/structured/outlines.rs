//! Regex-guided decoding (стиль outlines): компиляция подмножества regex в
//! байтовый Thompson-NFA и вычисление допустимых следующих байтов по
//! последовательности уже принятых байтов.
//!
//! Поддержано: литералы (с экранированием `\`), `.` (любой байт), классы
//! `[...]` с диапазонами `a-z` и отрицанием `[^...]`, квантификаторы `* + ?`,
//! группировка `(...)`, альтернация `|`, конкатенация. Паттерны считаются ASCII.

/// Матчер одного байта.
#[derive(Clone)]
enum Matcher {
    Byte(u8),
    Any,
    Class { negate: bool, ranges: Vec<(u8, u8)> },
}

impl Matcher {
    fn matches(&self, b: u8) -> bool {
        match self {
            Matcher::Byte(x) => b == *x,
            Matcher::Any => true,
            Matcher::Class { negate, ranges } => {
                let hit = ranges.iter().any(|(lo, hi)| b >= *lo && b <= *hi);
                hit != *negate
            }
        }
    }
}

struct State {
    /// `None` — эпсилон-переход; иначе условие на байт.
    trans: Vec<(Option<Matcher>, usize)>,
}

/// Скомпилированный байтовый NFA.
struct Nfa {
    states: Vec<State>,
    start: usize,
    accept: usize,
}

struct Builder {
    states: Vec<State>,
}

impl Builder {
    fn new() -> Self { Self { states: Vec::new() } }
    fn add_state(&mut self) -> usize {
        self.states.push(State { trans: Vec::new() });
        self.states.len() - 1
    }
    fn add_eps(&mut self, from: usize, to: usize) {
        self.states[from].trans.push((None, to));
    }
    fn add_match(&mut self, from: usize, m: Matcher, to: usize) {
        self.states[from].trans.push((Some(m), to));
    }
}

struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
    b: Builder,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> { self.chars.get(self.pos).copied() }
    fn next(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() { self.pos += 1; }
        c
    }

    /// alt := concat ('|' concat)*
    fn parse_alt(&mut self) -> Result<(usize, usize), ()> {
        let (mut s, mut e) = self.parse_concat()?;
        while self.peek() == Some('|') {
            self.next();
            let (s2, e2) = self.parse_concat()?;
            let ns = self.b.add_state();
            let ne = self.b.add_state();
            self.b.add_eps(ns, s);
            self.b.add_eps(ns, s2);
            self.b.add_eps(e, ne);
            self.b.add_eps(e2, ne);
            s = ns;
            e = ne;
        }
        Ok((s, e))
    }

    /// concat := quantified*  (пустая конкатенация = эпсилон-фрагмент)
    fn parse_concat(&mut self) -> Result<(usize, usize), ()> {
        let start = self.b.add_state();
        let mut cur = start;
        loop {
            match self.peek() {
                None | Some('|') | Some(')') => break,
                _ => {}
            }
            let (s, e) = self.parse_quantified()?;
            self.b.add_eps(cur, s);
            cur = e;
        }
        Ok((start, cur))
    }

    /// quantified := atom ('*' | '+' | '?')?
    fn parse_quantified(&mut self) -> Result<(usize, usize), ()> {
        let (s, e) = self.parse_atom()?;
        match self.peek() {
            Some('*') => {
                self.next();
                let ns = self.b.add_state();
                let ne = self.b.add_state();
                self.b.add_eps(ns, s);
                self.b.add_eps(ns, ne);
                self.b.add_eps(e, s);
                self.b.add_eps(e, ne);
                Ok((ns, ne))
            }
            Some('+') => {
                self.next();
                let ne = self.b.add_state();
                self.b.add_eps(e, s);
                self.b.add_eps(e, ne);
                Ok((s, ne))
            }
            Some('?') => {
                self.next();
                let ns = self.b.add_state();
                let ne = self.b.add_state();
                self.b.add_eps(ns, s);
                self.b.add_eps(ns, ne);
                self.b.add_eps(e, ne);
                Ok((ns, ne))
            }
            _ => Ok((s, e)),
        }
    }

    /// atom := '(' alt ')' | '[' class ']' | '.' | '\' c | literal
    fn parse_atom(&mut self) -> Result<(usize, usize), ()> {
        match self.peek() {
            Some('(') => {
                self.next();
                let (s, e) = self.parse_alt()?;
                if self.next() != Some(')') {
                    return Err(());
                }
                Ok((s, e))
            }
            Some('[') => {
                self.next();
                self.parse_class()
            }
            Some('.') => {
                self.next();
                self.single(Matcher::Any)
            }
            Some('\\') => {
                self.next();
                let c = self.next().ok_or(())?;
                self.single(Matcher::Byte(c as u32 as u8))
            }
            Some(c) => {
                self.next();
                self.single(Matcher::Byte(c as u32 as u8))
            }
            None => Err(()),
        }
    }

    fn single(&mut self, m: Matcher) -> Result<(usize, usize), ()> {
        let s = self.b.add_state();
        let e = self.b.add_state();
        self.b.add_match(s, m, e);
        Ok((s, e))
    }

    /// Внутренность класса `[...]` (открывающая `[` уже съедена).
    fn parse_class(&mut self) -> Result<(usize, usize), ()> {
        let mut negate = false;
        if self.peek() == Some('^') {
            self.next();
            negate = true;
        }
        let mut ranges: Vec<(u8, u8)> = Vec::new();
        let mut first = true;
        loop {
            match self.peek() {
                None => return Err(()),
                Some(']') if !first => {
                    self.next();
                    break;
                }
                _ => {}
            }
            first = false;
            let c = self.next().ok_or(())?;
            let lo = if c == '\\' { self.next().ok_or(())? as u32 as u8 } else { c as u32 as u8 };
            // диапазон lo-hi
            if self.peek() == Some('-') && self.chars.get(self.pos + 1).map_or(false, |&n| n != ']') {
                self.next(); // '-'
                let h = self.next().ok_or(())?;
                let hi = if h == '\\' { self.next().ok_or(())? as u32 as u8 } else { h as u32 as u8 };
                ranges.push((lo.min(hi), lo.max(hi)));
            } else {
                ranges.push((lo, lo));
            }
        }
        self.single(Matcher::Class { negate, ranges })
    }
}

fn compile_nfa(pattern: &str) -> Option<Nfa> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut p = Parser { chars: &chars, pos: 0, b: Builder::new() };
    let (s, e) = p.parse_alt().ok()?;
    if p.pos != p.chars.len() {
        return None; // не весь паттерн разобран
    }
    Some(Nfa { states: p.b.states, start: s, accept: e })
}

impl Nfa {
    fn closure(&self, set: &[usize]) -> Vec<usize> {
        let mut stack: Vec<usize> = set.to_vec();
        let mut seen: Vec<bool> = vec![false; self.states.len()];
        let mut out = Vec::new();
        while let Some(s) = stack.pop() {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            out.push(s);
            for (m, t) in &self.states[s].trans {
                if m.is_none() {
                    stack.push(*t);
                }
            }
        }
        out
    }

    fn step(&self, set: &[usize], b: u8) -> Vec<usize> {
        let mut next = Vec::new();
        for &s in set {
            for (m, t) in &self.states[s].trans {
                if let Some(m) = m {
                    if m.matches(b) {
                        next.push(*t);
                    }
                }
            }
        }
        self.closure(&next)
    }

    fn run(&self, consumed: &[u8]) -> Vec<usize> {
        let mut cur = self.closure(&[self.start]);
        for &b in consumed {
            cur = self.step(&cur, b);
            if cur.is_empty() {
                break;
            }
        }
        cur
    }
}

/// Ограничение из regex: маска допустимых следующих байтов по уже принятым.
pub struct OutlinesConstraint {
    pub pattern: String,
    nfa: Option<Nfa>,
}

impl OutlinesConstraint {
    /// Скомпилировать паттерн. При ошибке разбора NFA пуст (ничего не допускает) —
    /// используйте [`OutlinesConstraint::compile`] если нужна явная ошибка.
    pub fn new(pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        let nfa = compile_nfa(&pattern);
        Self { pattern, nfa }
    }

    pub fn regex(r: impl Into<String>) -> Self { Self::new(r) }

    /// Скомпилировать с явной проверкой корректности паттерна.
    pub fn compile(pattern: impl Into<String>) -> Result<Self, String> {
        let pattern = pattern.into();
        match compile_nfa(&pattern) {
            Some(nfa) => Ok(Self { pattern, nfa: Some(nfa) }),
            None => Err(format!("invalid regex: {pattern}")),
        }
    }

    /// Множество байтов, которыми можно продолжить после `consumed`.
    pub fn allowed_bytes(&self, consumed: &[u8]) -> Vec<u8> {
        let Some(nfa) = &self.nfa else { return Vec::new() };
        let cur = nfa.run(consumed);
        if cur.is_empty() {
            return Vec::new();
        }
        let mut allowed = Vec::new();
        for b in 0u16..=255 {
            let b = b as u8;
            if !nfa.step(&cur, b).is_empty() {
                allowed.push(b);
            }
        }
        allowed
    }

    /// Является ли `consumed` полным совпадением (достигнуто accept-состояние).
    pub fn is_match(&self, consumed: &[u8]) -> bool {
        let Some(nfa) = &self.nfa else { return false };
        nfa.run(consumed).contains(&nfa.accept)
    }

    /// Удобный матч строки целиком.
    pub fn accepts(&self, s: &str) -> bool {
        self.is_match(s.as_bytes())
    }
}
