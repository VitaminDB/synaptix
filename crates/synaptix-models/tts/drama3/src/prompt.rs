//! Промпт `reference-bracket-v1` (эталон — `prompt.py`): строки-кадры
//! `[токен, c0..c9]`; у текстовых строк коды нулевые, у аудиострок токен =
//! `<|semantic:0|> + c0`. Каждый отрезок токенизируется отдельно — склейка
//! строк меняет id на швах.

use crate::{Drama3Error, Row, CODEBOOKS, R, SEMANTIC_BEGIN};

const REFERENCE_INSTRUCTION: &str = "convert the provided text to speech reference to the following:";
const TTS_INSTRUCTION: &str = "convert the provided text to speech";
const MESSAGE_END: &str = "<|im_end|>\n";
const GENERATION_START: &str = "<|im_start|>assistant\n<|voice|>";
const STRUCTURAL: [&str; 6] = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"];
pub const MAX_SPEAKER: u32 = 15;

/// Образец голоса в промпте: расшифровка и коды `[10][T]`.
pub struct RefCodes<'a> {
    pub transcript: &'a str,
    pub codes: &'a [Vec<u32>],
}

/// Unicode White_Space (как в эталоне; `char::is_whitespace` — то же свойство).
fn lstrip(s: &str) -> &str {
    s.trim_start_matches(char::is_whitespace)
}

/// Длина тега `<|speaker:N|>` в начале строки (N = 0..15 без ведущих нулей).
pub(crate) fn speaker_tag(s: &str) -> Option<usize> {
    let rest = s.strip_prefix("<|speaker:")?;
    let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || !rest[digits..].starts_with("|>") {
        return None;
    }
    Some("<|speaker:".len() + digits + 2)
}

fn valid_speaker(tag: &str) -> bool {
    let n = &tag["<|speaker:".len()..tag.len() - 2];
    (n == "0" || !n.starts_with('0')) && n.parse::<u32>().is_ok_and(|v| v <= MAX_SPEAKER)
}

pub fn validate(text: &str, place: &str) -> R<()> {
    if text.trim_matches(char::is_whitespace).is_empty() {
        return Err(Drama3Error::Input(format!("{place}: пустой текст")));
    }
    for t in STRUCTURAL {
        if text.contains(t) {
            return Err(Drama3Error::Input(format!("{place}: служебный токен {t} в тексте недопустим")));
        }
    }
    for (i, _) in text.match_indices("<|speaker") {
        match speaker_tag(&text[i..]) {
            Some(n) if valid_speaker(&text[i..i + n]) => {}
            _ => return Err(Drama3Error::Input(format!("{place}: тег говорящего пишется <|speaker:N|>, N = 0..{MAX_SPEAKER}"))),
        }
    }
    Ok(())
}

/// Тег говорящего 0, если текст (после пробелов) не начинается со своего.
pub fn with_speaker_tag(text: &str) -> String {
    if speaker_tag(lstrip(text)).is_some() {
        text.to_string()
    } else {
        format!("<|speaker:0|> {text}")
    }
}

/// Строки промпта, заканчивающиеся началом хода голоса.
pub fn build_rows(text: &str, refs: &[RefCodes], encode: &dyn Fn(&str) -> R<Vec<u32>>) -> R<Vec<Row>> {
    build_rows_with_history(text, refs, &[], encode)
}

/// Как [`build_rows`], но с уже сказанными кусками (`history`: текст и кадры
/// `[токен, c0..c9]`) — ходы user/assistant перед целевым текстом, чтобы
/// длинная реплика звучала одним голосом и интонацией (как в MLX-эталоне).
pub fn build_rows_with_history(text: &str, refs: &[RefCodes], history: &[(&str, &[Row])], encode: &dyn Fn(&str) -> R<Vec<u32>>) -> R<Vec<Row>> {
    let mut rows: Vec<Row> = Vec::new();
    let push_text = |rows: &mut Vec<Row>, s: &str| -> R<()> {
        for id in encode(s)? {
            let mut r = [0u32; CODEBOOKS + 1];
            r[0] = id;
            rows.push(r);
        }
        Ok(())
    };
    push_text(&mut rows, "<|im_start|>system\n")?;
    push_text(&mut rows, if refs.is_empty() { TTS_INSTRUCTION } else { REFERENCE_INSTRUCTION })?;
    for (i, r) in refs.iter().enumerate() {
        let place = format!("образец {}", i + 1);
        validate(r.transcript, &place)?;
        if r.codes.len() != CODEBOOKS || r.codes[0].is_empty() || r.codes.iter().any(|c| c.len() != r.codes[0].len()) {
            return Err(Drama3Error::Input(format!("{place}: коды должны быть [{CODEBOOKS}][T], T > 0")));
        }
        // Тег стоит вплотную к «Text: »: ведущие пробелы уходят за него.
        let tagged = with_speaker_tag(r.transcript);
        let body = lstrip(&tagged);
        let lead = &tagged[..tagged.len() - body.len()];
        let n = speaker_tag(body).expect("тег добавлен");
        let transcript = format!("{}{}{}", &body[..n], lead, &body[n..]);
        push_text(&mut rows, &format!("\n\n[Reference {}]\nText: {transcript}\nSpeech: ", i + 1))?;
        for t in 0..r.codes[0].len() {
            let mut row = [0u32; CODEBOOKS + 1];
            row[0] = SEMANTIC_BEGIN + r.codes[0][t];
            for k in 0..CODEBOOKS {
                row[1 + k] = r.codes[k][t];
            }
            rows.push(row);
        }
    }
    validate(text, "текст")?;
    push_text(&mut rows, MESSAGE_END)?;
    for (t, frames) in history {
        push_text(&mut rows, "<|im_start|>user\n")?;
        push_text(&mut rows, &with_speaker_tag(t))?;
        push_text(&mut rows, MESSAGE_END)?;
        push_text(&mut rows, GENERATION_START)?;
        rows.extend_from_slice(frames);
        push_text(&mut rows, MESSAGE_END)?;
    }
    push_text(&mut rows, "<|im_start|>user\n")?;
    push_text(&mut rows, &with_speaker_tag(text))?;
    push_text(&mut rows, MESSAGE_END)?;
    push_text(&mut rows, GENERATION_START)?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags() {
        assert_eq!(with_speaker_tag("привет"), "<|speaker:0|> привет");
        assert_eq!(with_speaker_tag("  <|speaker:3|> привет"), "  <|speaker:3|> привет");
        assert!(validate("<|speaker:16|> а", "т").is_err());
        assert!(validate("<|speaker:01|> а", "т").is_err());
        assert!(validate("<|speaker:15|> а", "т").is_ok());
        assert!(validate("а <|im_end|>", "т").is_err());
        assert!(validate("  ", "т").is_err());
    }

    #[test]
    fn rows_layout() {
        let enc = |s: &str| -> R<Vec<u32>> { Ok(vec![s.len() as u32]) };
        let codes: Vec<Vec<u32>> = (0..CODEBOOKS).map(|k| vec![k as u32 + 1, 7]).collect();
        let rows = build_rows("привет", &[RefCodes { transcript: " раз", codes: &codes }], &enc).unwrap();
        // system, инструкция, заголовок образца, 2 кадра, конец, user, текст, конец, старт
        assert_eq!(rows.len(), 10);
        assert_eq!(rows[3][0], SEMANTIC_BEGIN + 1);
        assert_eq!(rows[3][1], 1);
        assert_eq!(rows[3][10], 10);
        let header = "\n\n[Reference 1]\nText: <|speaker:0|>  раз\nSpeech: ";
        assert_eq!(rows[2][0], header.len() as u32);
    }
}
