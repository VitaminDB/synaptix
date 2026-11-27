//! Разбиение длинного текста на куски (как `SplitText` MLX-эталона): около
//! 300 байт UTF-8 по концам предложений, затем по запятым и пробелам; в
//! куске не больше трёх говорящих; реплики говорящих — строками.

use crate::prompt::speaker_tag;

pub const CHUNK_BYTES: usize = 300;
const CHUNK_SPEAKERS: usize = 3;

/// Куски длиной до `limit` байт: по концам предложений, иначе по запятым, иначе по пробелам.
fn split_by_length(text: &str, limit: usize) -> Vec<String> {
    let text = text.trim();
    let mut out = Vec::new();
    let mut rest = text;
    while rest.len() > limit {
        // Последняя граница в пределах лимита: конец предложения → запятая → пробел.
        let mut cut = [None::<usize>; 3];
        let mut depth = 0i32;
        let mut prev = '\0';
        for (i, c) in rest.char_indices() {
            if i > limit {
                break;
            }
            // Граница — перед пробелом после знака; внутри [событий] не режем.
            if c.is_whitespace() && depth == 0 && i > 0 {
                let k = match prev {
                    '.' | '!' | '?' | '…' | '。' | '！' | '？' => 0,
                    ',' | ';' | ':' | '—' | '，' => 1,
                    _ => 2,
                };
                for slot in cut.iter_mut().skip(k) {
                    *slot = Some(i);
                }
            }
            match c {
                '[' => depth += 1,
                ']' => depth = (depth - 1).max(0),
                _ => {}
            }
            prev = c;
        }
        let at = cut[0].or(cut[1]).or(cut[2]).unwrap_or_else(|| {
            // Ни одного пробела: режем по границе символа.
            let mut i = limit.min(rest.len());
            while !rest.is_char_boundary(i) {
                i -= 1;
            }
            i.max(rest.chars().next().map_or(1, char::len_utf8))
        });
        let head = rest[..at].trim();
        if !head.is_empty() {
            out.push(head.to_string());
        }
        rest = rest[at..].trim_start();
    }
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// Куски текста для синтеза. Короткий текст — один кусок как есть.
pub fn split(text: &str) -> Vec<String> {
    if text.len() <= CHUNK_BYTES {
        return vec![text.to_string()];
    }
    let flat = text.split('\n').map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
    if !flat.contains("<|speaker:") {
        return split_by_length(&flat, CHUNK_BYTES);
    }
    // Ходы говорящих: (номер, текст); до первого тега — говорящий 0.
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut speaker = "0".to_string();
    let mut rest = flat.as_str();
    loop {
        let next = rest.match_indices("<|speaker:").find_map(|(i, _)| speaker_tag(&rest[i..]).map(|n| (i, n)));
        let end = next.map_or(rest.len(), |(i, _)| i);
        if !rest[..end].trim().is_empty() {
            turns.push((speaker.clone(), rest[..end].trim().to_string()));
        }
        match next {
            Some((i, n)) => {
                speaker = rest[i + "<|speaker:".len()..i + n - 2].to_string();
                rest = &rest[i + n..];
            }
            None => break,
        }
    }
    let mut chunks: Vec<Vec<String>> = Vec::new();
    let mut speakers: Vec<String> = Vec::new();
    let mut bytes = 0usize;
    for (sp, body) in &turns {
        for piece in split_by_length(body, CHUNK_BYTES) {
            let tagged = format!("<|speaker:{sp}|> {piece}");
            let new_speaker = !speakers.contains(sp);
            if chunks.is_empty() || (new_speaker && speakers.len() >= CHUNK_SPEAKERS) || bytes + tagged.len() > CHUNK_BYTES {
                chunks.push(Vec::new());
                speakers.clear();
                bytes = 0;
            }
            bytes += tagged.len();
            if !speakers.contains(sp) {
                speakers.push(sp.clone());
            }
            chunks.last_mut().expect("кусок есть").push(tagged);
        }
    }
    chunks.into_iter().map(|c| c.join("\n")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(split("Привет!\nКак дела?"), vec!["Привет!\nКак дела?".to_string()]);
    }

    #[test]
    fn long_text_splits_on_sentences() {
        let s = "Жил-был король, который любил суп. ".repeat(12);
        let c = split(&s);
        assert!(c.len() > 1);
        for x in &c {
            assert!(x.len() <= CHUNK_BYTES, "{}", x.len());
            assert!(x.ends_with('.'));
        }
        assert_eq!(c.join(" "), s.trim());
    }

    #[test]
    fn speakers_keep_tags() {
        let a = "Ваше величество, суп готов, и он сегодня особенно удался, честное слово! ".repeat(3);
        let s = format!("<|speaker:0|> {a}\n<|speaker:1|> {a}");
        let c = split(&s);
        assert!(c.len() >= 2);
        assert!(c.iter().all(|x| x.starts_with("<|speaker:") && x.len() <= CHUNK_BYTES + 16));
        assert!(c.last().unwrap().contains("<|speaker:1|>"));
    }
}
