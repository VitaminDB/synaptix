//! Числа словами по-русски (количественные, именительный падеж; «1 000 000» и «1000000» — одно
//! число). В оригинале Silero цифр нет в алфавите — они молча выпадали бы из речи.

const UNITS: [&str; 20] = [
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
];
const TENS: [&str; 10] = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"];
const HUNDREDS: [&str; 10] = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"];

/// 1…999; `fem` — женский род единиц (тысячи: «одна», «две»).
fn triple(n: u64, fem: bool) -> Vec<&'static str> {
    let mut w = Vec::new();
    let (h, r) = (n / 100, n % 100);
    if h > 0 {
        w.push(HUNDREDS[h as usize]);
    }
    if r >= 20 {
        w.push(TENS[(r / 10) as usize]);
        let u = r % 10;
        if u > 0 {
            w.push(match (u, fem) {
                (1, true) => "одна",
                (2, true) => "две",
                _ => UNITS[u as usize],
            });
        }
    } else if r > 0 {
        w.push(match (r, fem) {
            (1, true) => "одна",
            (2, true) => "две",
            _ => UNITS[r as usize],
        });
    }
    w
}

fn plural(n: u64, forms: [&'static str; 3]) -> &'static str {
    let (d, h) = (n % 10, n % 100);
    if (11..=14).contains(&h) {
        forms[2]
    } else if d == 1 {
        forms[0]
    } else if (2..=4).contains(&d) {
        forms[1]
    } else {
        forms[2]
    }
}

pub fn spell(n: u64) -> String {
    if n == 0 {
        return UNITS[0].to_string();
    }
    let scales: [(u64, [&str; 3], bool); 4] = [
        (1_000_000_000_000, ["триллион", "триллиона", "триллионов"], false),
        (1_000_000_000, ["миллиард", "миллиарда", "миллиардов"], false),
        (1_000_000, ["миллион", "миллиона", "миллионов"], false),
        (1_000, ["тысяча", "тысячи", "тысяч"], true),
    ];
    let mut rest = n;
    let mut w: Vec<&str> = Vec::new();
    for (s, forms, fem) in scales {
        let k = rest / s;
        if k > 0 {
            w.extend(triple(k, fem));
            w.push(plural(k, forms));
            rest %= s;
        }
    }
    if rest > 0 {
        w.extend(triple(rest, false));
    }
    w.join(" ")
}

/// Заменяет числа в тексте словами: «2026», «1 000» (пробел между группами по 3), «3,5» → «три
/// целых пять десятых»-подобное чтение не делается — дробная часть читается отдельным числом через «и».
pub fn expand(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_ascii_digit() {
            let mut digits = String::new();
            let mut j = i;
            loop {
                while j < c.len() && c[j].is_ascii_digit() {
                    digits.push(c[j]);
                    j += 1;
                }
                // группа тысяч через пробел: «1 000 000»
                if j + 3 < c.len() + 1 && j < c.len() && (c[j] == ' ' || c[j] == '\u{a0}') && c[j + 1..].iter().take(3).filter(|x| x.is_ascii_digit()).count() == 3 && c.get(j + 4).is_none_or(|x| !x.is_ascii_digit()) {
                    j += 1;
                    continue;
                }
                break;
            }
            let word = digits.parse::<u64>().map(spell).unwrap_or_else(|_| digits.chars().map(|d| UNITS[d.to_digit(10).unwrap_or(0) as usize]).collect::<Vec<_>>().join(" "));
            if !out.is_empty() && out.chars().last().is_some_and(|p| p.is_alphabetic()) {
                out.push(' ');
            }
            out.push_str(&word);
            if c.get(j).is_some_and(|n| n.is_alphabetic()) {
                out.push(' ');
            }
            i = j;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spells() {
        assert_eq!(spell(3), "три");
        assert_eq!(spell(21), "двадцать один");
        assert_eq!(spell(2026), "две тысячи двадцать шесть");
        assert_eq!(spell(1_000_000), "один миллион");
        assert_eq!(spell(11_512), "одиннадцать тысяч пятьсот двенадцать");
        assert_eq!(expand("У меня 3 яблока и 1 000 рублей, 5кг"), "У меня три яблока и одна тысяча рублей, пять кг");
    }
}
