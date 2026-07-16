use crate::error::{InferError, Result};
use crate::structured::outlines::OutlinesConstraint;

/// Минимальный LMQL-style constraint. Полноценный LMQL — query-language с
/// boolean-операторами и переменными; здесь поддерживается AND-комбинация
/// атомарных ограничений вида:
///
/// - `LEN < N` / `LEN <= N` / `LEN == N` / `LEN > N` / `LEN >= N` — токеновый бюджет.
/// - `REGEX("pattern")` — байтовый regex (через [`OutlinesConstraint`]).
/// - `IN [a, b, c, ...]` — фиксированный набор разрешённых токенов.
/// - `STARTS_WITH [a, b, c, ...]` — начальный префикс (linear-grammar style).
///
/// Несколько атомов разделяются ` AND ` (case-insensitive). Все условия должны
/// выполняться одновременно (`allowed_tokens` пересекает множества).
///
/// Пример: `"LEN < 32 AND REGEX(\"[0-9]+\")"` — до 32 токенов, только цифры (байты).
pub struct LmqlConstraint {
    pub query: String,
    atoms: Vec<Atom>,
}

enum Atom {
    LenLt(usize),
    LenLe(usize),
    LenEq(usize),
    LenGt(usize),
    LenGe(usize),
    Regex(OutlinesConstraint),
    In(Vec<u32>),
    StartsWith(Vec<u32>),
}

impl LmqlConstraint {
    pub fn new(query: impl Into<String>) -> Self {
        let query = query.into();
        let atoms = parse_query(&query).unwrap_or_default();
        Self { query, atoms }
    }

    pub fn parse(query: impl Into<String>) -> Result<Self> {
        let query = query.into();
        let atoms = parse_query(&query)?;
        Ok(Self { query, atoms })
    }

    pub fn is_satisfied(&self, state: &[u32]) -> bool {
        for a in &self.atoms {
            match a {
                Atom::LenLt(n) => {
                    if !(state.len() < *n) {
                        return false;
                    }
                }
                Atom::LenLe(n) => {
                    if !(state.len() <= *n) {
                        return false;
                    }
                }
                Atom::LenEq(n) => {
                    if state.len() != *n {
                        return false;
                    }
                }
                Atom::LenGt(n) => {
                    if !(state.len() > *n) {
                        return false;
                    }
                }
                Atom::LenGe(n) => {
                    if !(state.len() >= *n) {
                        return false;
                    }
                }
                Atom::Regex(_) | Atom::In(_) | Atom::StartsWith(_) => {}
            }
        }
        true
    }

    pub fn is_finished(&self, state: &[u32]) -> bool {
        for a in &self.atoms {
            if let Atom::LenEq(n) = a {
                return state.len() >= *n;
            }
            if let Atom::LenLt(n) = a {
                if state.len() + 1 >= *n {
                    return true;
                }
            }
            if let Atom::LenLe(n) = a {
                if state.len() >= *n {
                    return true;
                }
            }
        }
        false
    }

    pub fn allowed_tokens(&self, state: &[u32], vocab_size: usize) -> Vec<u32> {
        if self.is_finished(state) {
            return Vec::new();
        }
        let mut acc: Option<Vec<u32>> = None;
        for a in &self.atoms {
            let mask: Option<Vec<u32>> = match a {
                Atom::LenLt(n) => {
                    if state.len() + 1 > *n {
                        Some(Vec::new())
                    } else {
                        None
                    }
                }
                Atom::LenLe(n) => {
                    if state.len() + 1 > *n {
                        Some(Vec::new())
                    } else {
                        None
                    }
                }
                Atom::LenEq(n) => {
                    if state.len() >= *n {
                        Some(Vec::new())
                    } else {
                        None
                    }
                }
                Atom::LenGt(_) | Atom::LenGe(_) => None,
                Atom::Regex(con) => {
                    let consumed: Vec<u8> = state.iter().filter_map(|&t| u8::try_from(t).ok()).collect();
                    let bytes = con.allowed_bytes(&consumed);
                    Some(
                        bytes
                            .into_iter()
                            .map(|b| b as u32)
                            .filter(|&t| (t as usize) < vocab_size)
                            .collect(),
                    )
                }
                Atom::In(set) => Some(
                    set.iter()
                        .copied()
                        .filter(|&t| (t as usize) < vocab_size)
                        .collect(),
                ),
                Atom::StartsWith(prefix) => {
                    if state.len() < prefix.len() {
                        Some(vec![prefix[state.len()]].into_iter().filter(|&t| (t as usize) < vocab_size).collect())
                    } else {
                        None
                    }
                }
            };
            if let Some(m) = mask {
                acc = Some(match acc {
                    Some(prev) => intersect_sorted_or_set(&prev, &m),
                    None => m,
                });
                if let Some(a) = &acc {
                    if a.is_empty() {
                        return Vec::new();
                    }
                }
            }
        }
        match acc {
            Some(set) => set,
            None => (0u32..vocab_size as u32).collect(),
        }
    }

    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }
}

fn intersect_sorted_or_set(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut set: std::collections::BTreeSet<u32> = b.iter().copied().collect();
    a.iter().copied().filter(|t| set.remove(t)).collect()
}

fn parse_query(q: &str) -> Result<Vec<Atom>> {
    let trimmed = q.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let parts = split_and(trimmed);
    let mut out = Vec::with_capacity(parts.len());
    for raw in parts {
        let p = raw.trim();
        if p.is_empty() {
            continue;
        }
        out.push(parse_atom(p)?);
    }
    Ok(out)
}

fn split_and(q: &str) -> Vec<String> {
    let lower = q.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut start = 0;
    let mut depth_paren = 0;
    let mut depth_brack = 0;
    let mut in_string = false;
    let bytes = q.as_bytes();
    let lower_b = lower.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '"' {
            in_string = !in_string;
        }
        if !in_string {
            match c {
                '(' => depth_paren += 1,
                ')' => depth_paren -= 1,
                '[' => depth_brack += 1,
                ']' => depth_brack -= 1,
                _ => {}
            }
            if depth_paren == 0
                && depth_brack == 0
                && i + 5 <= lower_b.len()
                && &lower_b[i..i + 5] == b" and "
            {
                out.push(q[start..i].to_string());
                start = i + 5;
                i += 5;
                continue;
            }
        }
        i += 1;
    }
    out.push(q[start..].to_string());
    out
}

fn parse_atom(s: &str) -> Result<Atom> {
    let upper = s.trim().to_ascii_uppercase();
    if upper.starts_with("LEN") {
        return parse_len(s.trim());
    }
    if upper.starts_with("REGEX(") {
        let pattern = extract_in_quotes(s)?;
        return Ok(Atom::Regex(OutlinesConstraint::new(pattern)));
    }
    if upper.starts_with("IN ") || upper.starts_with("IN[") {
        let toks = parse_token_list(s)?;
        return Ok(Atom::In(toks));
    }
    if upper.starts_with("STARTS_WITH ") || upper.starts_with("STARTS_WITH[") {
        let toks = parse_token_list(s)?;
        return Ok(Atom::StartsWith(toks));
    }
    Err(InferError::Other(format!("LMQL: unknown atom '{s}'")))
}

fn parse_len(s: &str) -> Result<Atom> {
    let rest = s[3..].trim_start();
    let (op_len, op): (usize, &str) = if rest.starts_with("<=") {
        (2, "<=")
    } else if rest.starts_with(">=") {
        (2, ">=")
    } else if rest.starts_with("==") {
        (2, "==")
    } else if rest.starts_with('<') {
        (1, "<")
    } else if rest.starts_with('>') {
        (1, ">")
    } else if rest.starts_with('=') {
        (1, "=")
    } else {
        return Err(InferError::Other(format!("LMQL LEN: missing operator in '{s}'")));
    };
    let arg = rest[op_len..].trim();
    let n: usize = arg
        .parse()
        .map_err(|e| InferError::Other(format!("LMQL LEN: bad number '{arg}': {e}")))?;
    Ok(match op {
        "<" => Atom::LenLt(n),
        "<=" => Atom::LenLe(n),
        "==" | "=" => Atom::LenEq(n),
        ">" => Atom::LenGt(n),
        ">=" => Atom::LenGe(n),
        _ => unreachable!(),
    })
}

fn extract_in_quotes(s: &str) -> Result<String> {
    let l = s.find('"').ok_or_else(|| InferError::Other(format!("LMQL: expected '\"' in '{s}'")))?;
    let r = s[l + 1..]
        .find('"')
        .ok_or_else(|| InferError::Other(format!("LMQL: expected closing '\"' in '{s}'")))?;
    Ok(s[l + 1..l + 1 + r].to_string())
}

fn parse_token_list(s: &str) -> Result<Vec<u32>> {
    let l = s.find('[').ok_or_else(|| InferError::Other(format!("LMQL: expected '[' in '{s}'")))?;
    let r = s
        .rfind(']')
        .ok_or_else(|| InferError::Other(format!("LMQL: expected ']' in '{s}'")))?;
    if r <= l + 1 {
        return Ok(Vec::new());
    }
    let inner = &s[l + 1..r];
    let mut out = Vec::new();
    for p in inner.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        let n: u32 = p
            .parse()
            .map_err(|e| InferError::Other(format!("LMQL: bad token '{p}': {e}")))?;
        out.push(n);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_len_lt() {
        let c = LmqlConstraint::parse("LEN < 10").unwrap();
        assert_eq!(c.atom_count(), 1);
        assert!(c.is_satisfied(&[1, 2, 3]));
        assert!(c.is_satisfied(&vec![0; 9]));
        assert!(!c.is_satisfied(&vec![0; 10]));
        assert!(c.is_finished(&vec![0; 9]));
    }

    #[test]
    fn parse_len_eq_blocks_at_target() {
        let c = LmqlConstraint::parse("LEN == 3").unwrap();
        assert!(!c.is_finished(&[1, 2]));
        let allowed = c.allowed_tokens(&[1, 2], 16);
        assert!(!allowed.is_empty(), "at len=2, len<3 still room");
        let allowed3 = c.allowed_tokens(&[1, 2, 3], 16);
        assert!(allowed3.is_empty(), "at len=3 must stop");
        assert!(c.is_finished(&[1, 2, 3]));
    }

    #[test]
    fn parse_in_filters_vocab() {
        let c = LmqlConstraint::parse("IN [2, 5, 7]").unwrap();
        let mut allowed = c.allowed_tokens(&[], 16);
        allowed.sort();
        assert_eq!(allowed, vec![2, 5, 7]);
    }

    #[test]
    fn parse_regex_byte_level() {
        let c = LmqlConstraint::parse("REGEX(\"[0-9]+\")").unwrap();
        let allowed = c.allowed_tokens(&[], 256);
        for d in b'0'..=b'9' {
            assert!(allowed.contains(&(d as u32)), "expected digit {} in regex allowed", d as char);
        }
        assert!(!allowed.contains(&(b'a' as u32)));
    }

    #[test]
    fn parse_starts_with_emits_prefix_only_at_position() {
        let c = LmqlConstraint::parse("STARTS_WITH [42, 99]").unwrap();
        let a0 = c.allowed_tokens(&[], 256);
        assert_eq!(a0, vec![42]);
        let a1 = c.allowed_tokens(&[42], 256);
        assert_eq!(a1, vec![99]);
        let a2 = c.allowed_tokens(&[42, 99], 256);
        assert!(!a2.is_empty(), "after prefix any token allowed unless другие atoms ограничат");
    }

    #[test]
    fn intersect_len_and_in() {
        let c = LmqlConstraint::parse("LEN < 4 AND IN [1, 2, 3]").unwrap();
        let mut allowed = c.allowed_tokens(&[1], 16);
        allowed.sort();
        assert_eq!(allowed, vec![1, 2, 3]);
        let allowed_full = c.allowed_tokens(&[1, 2, 3], 16);
        assert!(allowed_full.is_empty());
    }

    #[test]
    fn intersect_regex_and_len() {
        let c = LmqlConstraint::parse("LEN <= 5 AND REGEX(\"[a-z]+\")").unwrap();
        let allowed = c.allowed_tokens(&[], 256);
        for d in b'a'..=b'z' {
            assert!(allowed.contains(&(d as u32)));
        }
        assert!(!allowed.contains(&(b'A' as u32)));
        let allowed_at_len5 = c.allowed_tokens(&vec![b'a' as u32; 5], 256);
        assert!(allowed_at_len5.is_empty(), "at len=5 must stop");
    }

    #[test]
    fn unknown_atom_errors() {
        let err = LmqlConstraint::parse("FOOBAR == 1");
        assert!(err.is_err());
    }

    #[test]
    fn empty_query_is_trivial_satisfied() {
        let c = LmqlConstraint::parse("").unwrap();
        assert_eq!(c.atom_count(), 0);
        assert!(c.is_satisfied(&[1, 2, 3]));
        let allowed = c.allowed_tokens(&[], 4);
        assert_eq!(allowed, vec![0, 1, 2, 3]);
    }

    #[test]
    fn fallible_old_constructor_silent_on_bad_query() {
        let c = LmqlConstraint::new("FOOBAR == 1");
        assert_eq!(c.atom_count(), 0);
        assert!(c.is_satisfied(&[]));
    }

    #[test]
    fn intersect_starts_with_and_regex() {
        let c = LmqlConstraint::parse("STARTS_WITH [97] AND REGEX(\"[a-z]+\")").unwrap();
        let a0 = c.allowed_tokens(&[], 256);
        assert_eq!(a0, vec![97]);
        let a1 = c.allowed_tokens(&[97], 256);
        assert!(a1.iter().all(|&t| t >= b'a' as u32 && t <= b'z' as u32));
    }
}
