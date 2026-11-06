//! Английские слова в русском тексте: произношение — G2P Leokani (IPA с ударениями), затем ближайшие
//! русские звуки; основное ударение `ˈ` — знак Silero `+` перед ударной гласной. «iPhone» → «+айфоун».

/// Гласные IPA (и дифтонги в однобуквенной записи misaki) → русские буквы.
fn vowel(c: char) -> Option<&'static str> {
    Some(match c {
        'ɑ' | 'a' | 'ʌ' | 'ɐ' => "а",
        'æ' => "э",
        'ə' => "а",
        'ɚ' => "ер",
        'ɜ' => "ё",
        'ɛ' | 'e' => "е",
        'ɪ' | 'i' | 'ᵻ' => "и",
        'u' | 'ʊ' => "у",
        'ɔ' | 'o' | 'ɒ' => "о",
        'A' => "ей",
        'I' => "ай",
        'O' => "оу",
        'W' => "ау",
        'Y' => "ой",
        _ => return None,
    })
}

fn consonant(c: char) -> Option<&'static str> {
    Some(match c {
        'b' => "б",
        'd' => "д",
        'f' => "ф",
        'ɡ' | 'g' => "г",
        'h' => "х",
        'k' => "к",
        'l' => "л",
        'm' => "м",
        'n' => "н",
        'p' => "п",
        's' => "с",
        't' | 'ɾ' => "т",
        'v' => "в",
        'w' => "у",
        'z' => "з",
        'ʃ' => "ш",
        'ʒ' => "ж",
        'θ' => "с",
        'ð' => "з",
        'ŋ' => "нг",
        'ɹ' | 'r' => "р",
        'ʧ' => "ч",
        'ʤ' => "дж",
        'x' => "х",
        _ => return None,
    })
}

/// IPA (Leokani/misaki) → кириллица с `+` перед ударной гласной.
pub fn ipa_to_cyrillic(ipa: &str) -> String {
    // одно слово: пробелы между кусками G2P не нужны
    let c: Vec<char> = ipa.chars().filter(|&x| x != 'ː' && x != 'ʔ' && x != '\u{0303}' && x != ' ').collect();
    let mut out = String::new();
    let mut stress = false;
    let mut i = 0;
    let push_vowel = |out: &mut String, v: &str, stress: &mut bool| {
        if *stress {
            out.push('+');
            *stress = false;
        }
        out.push_str(v);
    };
    while i < c.len() {
        let ch = c[i];
        match ch {
            'ˈ' => {
                stress = true;
                i += 1;
                continue;
            }
            'ˌ' => {
                i += 1;
                continue;
            }
            ' ' => {
                out.push(' ');
                stress = false;
                i += 1;
                continue;
            }
            _ => {}
        }
        // дифтонги двумя символами IPA
        let next = c.get(i + 1).copied();
        let pair = match (ch, next) {
            ('e', Some('ɪ')) => Some("ей"),
            ('a', Some('ɪ')) => Some("ай"),
            ('o', Some('ʊ')) | ('ə', Some('ʊ')) => Some("оу"),
            ('a', Some('ʊ')) => Some("ау"),
            ('ɔ', Some('ɪ')) => Some("ой"),
            _ => None,
        };
        if let Some(v) = pair {
            push_vowel(&mut out, v, &mut stress);
            i += 2;
            continue;
        }
        if ch == 'j' {
            // йотированные: ю, я, е, ё; после согласной — с мягким знаком (ударение может стоять между)
            let after_cons = out.chars().last().is_some_and(|p| "бвгджзклмнпрстфхцчшщ".contains(p));
            let mut k = i + 1;
            while k < c.len() && (c[k] == 'ˈ' || c[k] == 'ˌ') {
                if c[k] == 'ˈ' {
                    stress = true;
                }
                k += 1;
            }
            let next = c.get(k).copied();
            let v = next.and_then(|n| match n {
                'u' | 'ʊ' => Some("ю"),
                'ɑ' | 'a' | 'ʌ' | 'ə' | 'æ' => Some("я"),
                'ɛ' | 'e' => Some("е"),
                'ɔ' | 'o' => Some("ё"),
                _ => None,
            });
            match v {
                Some(v) => {
                    if after_cons {
                        out.push('ь');
                    }
                    push_vowel(&mut out, v, &mut stress);
                    i = k + 1;
                }
                None => {
                    out.push('й');
                    i += 1;
                }
            }
            continue;
        }
        if (ch == 'ə' || ch == 'ɜ') && next == Some('ɹ') {
            push_vowel(&mut out, "ер", &mut stress);
            i += 2;
            continue;
        }
        if ch == 'ŋ' && matches!(next, Some('k') | Some('ɡ') | Some('g')) {
            out.push('н');
            i += 1;
            continue;
        }
        if let Some(v) = vowel(ch) {
            push_vowel(&mut out, v, &mut stress);
        } else if let Some(k) = consonant(ch) {
            // w перед гласной — «у» как в «уи»; удвоение одинаковых согласных — одна
            if !out.ends_with(k) || k.chars().count() > 1 {
                out.push_str(k);
            }
        }
        i += 1;
    }
    // несколько основных ударений (составные слова) — оставляем последнее
    if out.matches('+').count() > 1 {
        let last = out.rfind('+').unwrap_or(0);
        out = out.char_indices().filter(|&(i, c)| c != '+' || i == last).map(|(_, c)| c).collect();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words() {
        assert_eq!(ipa_to_cyrillic("ˈI fˈOn"), "айф+оун");
        assert_eq!(ipa_to_cyrillic("kəmpjˈuɾəɹ"), "кампь+ютер");
        assert_eq!(ipa_to_cyrillic("θˈæŋks"), "с+энкс");
        assert_eq!(ipa_to_cyrillic("mˈIkɹəsˌɑft"), "м+айкрасафт");
    }
}
