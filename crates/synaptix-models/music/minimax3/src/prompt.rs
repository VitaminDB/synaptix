//! Сборка промпта MiniMax Music 3 (эталон diffusers `encoders.py`): описание
//! музыки и текст песни между служебными токенами чекпойнта.

use fancy_regex::Regex;

pub const AUDIO_END: u32 = 151_670;
pub const AUDIO_CFG: u32 = 151_654;
pub const CODE_OFFSET: u32 = 151_675;
pub const SEMANTIC_VOCAB: usize = 16_384;
pub const MAX_PROMPT_TOKENS: usize = 5_000;
pub const MAX_FRAMES: usize = 9_000;

fn re(p: &str) -> Regex {
    Regex::new(p).expect("regex")
}

fn sub(r: &Regex, s: &str, rep: &str) -> String {
    r.replace_all(s, rep).into_owned()
}

/// Описание без разметки: `<|a b|>` → `a is b`, заголовки, списки, `**`, `*`,
/// линейки, «• » и отступы убираются, пустые строки схлопываются.
pub fn clean_caption(caption: &str) -> String {
    let tag = re(r"<\|([^|]*)\|>");
    let mut text = String::new();
    let mut last = 0;
    for m in tag.captures_iter(caption).flatten() {
        let all = m.get(0).unwrap();
        text.push_str(&caption[last..all.start()]);
        let inner = m.get(1).map(|x| x.as_str().trim()).unwrap_or("");
        let mut it = inner.splitn(2, char::is_whitespace);
        match (it.next(), it.next()) {
            (Some(a), Some(b)) if !b.trim().is_empty() => text.push_str(&format!("{a} is {}", b.trim_start())),
            _ => text.push_str(inner),
        }
        last = all.end();
    }
    text.push_str(&caption[last..]);
    let (h, l1, l2, bold, ital) = (re(r"^\s{0,3}#{1,6}\s+"), re(r"^\s*[*+-]\s+"), re(r"^\s*\*\s+"), re(r"\*\*([^*]+)\*\*"), re(r"(?<!\*)\*([^*\n]+)\*(?!\*)"));
    let lines: Vec<String> = text
        .lines()
        .map(|line| {
            let mut s = sub(&l2, &sub(&l1, &sub(&h, line, ""), ""), "");
            while s.contains("**") {
                let u = sub(&bold, &s, "$1");
                if u == s {
                    break;
                }
                s = u;
            }
            sub(&ital, &s, "$1").trim_end().to_string()
        })
        .collect();
    let text = sub(&re(r"(?m)^\s*[-*_]{3,}\s*$"), &lines.join("\n"), "");
    let text = text.replace("• ", "").replace("    ", "");
    sub(&re(r"\n{2,}"), &text, "\n")
}

/// Текст песни: строка с ведущими тегами `[..]` — только теги (остаток строки
/// по контракту чекпойнта отбрасывается), теги в нижний регистр, `[start]` в начале.
pub fn normalize_lyrics(lyrics: &str) -> String {
    let lead = re(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)");
    let out: Vec<String> = lyrics
        .split('\n')
        .map(|line| match lead.captures(line).ok().flatten().and_then(|c| c.get(1)) {
            Some(m) => m.as_str().trim().to_string(),
            None => line.to_string(),
        })
        .collect();
    let text = out.join("\n").replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n");
    let tag = re(r"\[([^\]]+)\]");
    let mut res = String::new();
    let mut last = 0;
    for m in tag.captures_iter(&text).flatten() {
        let all = m.get(0).unwrap();
        res.push_str(&text[last..all.start()]);
        res.push_str(&format!("[{}]", m.get(1).unwrap().as_str().to_lowercase()));
        last = all.end();
    }
    res.push_str(&text[last..]);
    format!("[start]\n{res}")
}

/// Полный текст промпта для токенизатора.
pub fn assemble(description: &str, lyrics: &str) -> String {
    format!(
        "<|im_start|><|caption_start|>{}<|caption_end|><|lyrics_start|>{}<|lyrics_end|><|im_end|><|audio_start|>",
        clean_caption(description),
        normalize_lyrics(lyrics)
    )
}

/// Безусловная ветка CFG: все токены, кроме первого и двух последних, — `AUDIO_CFG`.
pub fn unconditional(ids: &[u32]) -> Vec<u32> {
    let n = ids.len();
    ids.iter().enumerate().map(|(i, &t)| if i >= 1 && i + 2 < n { AUDIO_CFG } else { t }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyrics_tags() {
        let l = normalize_lyrics("[Verse] лишнее\nMorning light\n[Chorus]\nSoftly [Bridge] the world ^ breathes");
        assert_eq!(l, "[start]\n[verse]\nMorning light\n[chorus]\nSoftly\n[bridge]\nthe world\nbreathes");
    }

    #[test]
    fn caption_markup() {
        let c = clean_caption("# Global\n**Genre**: pop <|bpm 96|>\n\n\n- soft *female* lead\n---\n• piano");
        assert_eq!(c, "Global\nGenre: pop bpm is 96\nsoft female lead\npiano");
        assert_eq!(unconditional(&[1, 2, 3, 4, 5]), vec![1, AUDIO_CFG, AUDIO_CFG, 4, 5]);
        assert!(assemble("a", "b").ends_with("<|lyrics_end|><|im_end|><|audio_start|>"));
    }
}
