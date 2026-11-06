//! Латиница в русском тексте → кириллица по правилам английского чтения (упрощённо): в алфавите
//! русских моделей латиницы нет, и слова вроде «MacBook» молча выпадали бы из речи.

fn word(w: &str) -> String {
    let l: Vec<char> = w.to_ascii_lowercase().chars().collect();
    let n = l.len();
    let mut o = String::new();
    let mut i = 0;
    let vowel = |c: char| "aeiouy".contains(c);
    while i < n {
        let rest: String = l[i..].iter().collect();
        let tri = [("tion", "шн", 4), ("sch", "ш", 3), ("tch", "ч", 3)];
        if let Some((_, r, k)) = tri.iter().find(|(p, _, _)| rest.starts_with(p)) {
            o.push_str(r);
            i += k;
            continue;
        }
        let di = [
            ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ph", "ф"), ("th", "т"), ("ck", "к"), ("qu", "кв"), ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"), ("ow", "оу"), ("ai", "ей"), ("ay", "ей"), ("ey", "ей"), ("oy", "ой"), ("ya", "я"), ("yu", "ю"), ("yo", "ё"),
        ];
        if let Some((_, r)) = di.iter().find(|(p, _)| rest.starts_with(p)) {
            o.push_str(r);
            i += 2;
            continue;
        }
        let c = l[i];
        let next = l.get(i + 1).copied();
        // немое конечное e после согласной в словах длиннее трёх букв
        if c == 'e' && i + 1 == n && n > 3 && i > 0 && !vowel(l[i - 1]) {
            break;
        }
        // удвоенные согласные — одна
        if i > 0 && l[i - 1] == c && !vowel(c) {
            i += 1;
            continue;
        }
        let r = match c {
            'a' => "а",
            'b' => "б",
            'c' => {
                if next.is_some_and(|x| "eiy".contains(x)) {
                    "с"
                } else {
                    "к"
                }
            }
            'd' => "д",
            'e' => "е",
            'f' => "ф",
            'g' => "г",
            'h' => "х",
            'i' => "и",
            'j' => "дж",
            'k' => "к",
            'l' => "л",
            'm' => "м",
            'n' => "н",
            'o' => "о",
            'p' => "п",
            'q' => "к",
            'r' => "р",
            's' => "с",
            't' => "т",
            'u' => "у",
            'v' => "в",
            'w' => "в",
            'x' => "кс",
            'y' => {
                if i == 0 && next.is_some_and(vowel) {
                    "й"
                } else {
                    "и"
                }
            }
            'z' => "з",
            _ => "",
        };
        o.push_str(r);
        i += 1;
    }
    o
}

/// Заменяет латинские слова кириллицей (регистр — по первой букве).
pub fn to_cyrillic(text: &str) -> String {
    let mut out = String::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut String| {
        if !cur.is_empty() {
            let w = word(cur);
            if cur.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                let mut cs = w.chars();
                if let Some(f) = cs.next() {
                    out.extend(f.to_uppercase());
                    out.push_str(cs.as_str());
                }
            } else {
                out.push_str(&w);
            }
            cur.clear();
        }
    };
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
            out.push(c);
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words() {
        assert_eq!(to_cyrillic("MacBook"), "Макбук");
        assert_eq!(to_cyrillic("Apple"), "Апл");
        assert_eq!(to_cyrillic("Silero"), "Силеро");
        assert_eq!(to_cyrillic("Google Chrome"), "Гугл Чром");
        assert_eq!(to_cyrillic("я купил iPhone"), "я купил ифон");
    }
}
