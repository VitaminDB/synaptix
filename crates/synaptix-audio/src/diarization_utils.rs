use std::path::Path;

use crate::error::{AudioError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct RttmTurn {
    pub file_id: String,
    pub channel: u32,
    pub start: f64,
    pub duration: f64,
    pub speaker: String,
}

impl RttmTurn {
    pub fn end(&self) -> f64 {
        self.start + self.duration
    }
}

pub fn parse_rttm(text: &str) -> Result<Vec<RttmTurn>> {
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 9 {
            return Err(AudioError::invalid_arg(format!(
                "RTTM line {} has {} fields (expected >=9)",
                lineno + 1,
                parts.len()
            )));
        }
        if !parts[0].eq_ignore_ascii_case("SPEAKER") {
            continue;
        }
        let file_id = parts[1].to_string();
        let channel: u32 = parts[2]
            .parse()
            .map_err(|e| AudioError::invalid_arg(format!("line {}: channel: {e}", lineno + 1)))?;
        let start: f64 = parts[3]
            .parse()
            .map_err(|e| AudioError::invalid_arg(format!("line {}: start: {e}", lineno + 1)))?;
        let duration: f64 = parts[4]
            .parse()
            .map_err(|e| AudioError::invalid_arg(format!("line {}: duration: {e}", lineno + 1)))?;
        let speaker = parts[7].to_string();
        out.push(RttmTurn { file_id, channel, start, duration, speaker });
    }
    Ok(out)
}

pub fn read_rttm(path: impl AsRef<Path>) -> Result<Vec<RttmTurn>> {
    let p = path.as_ref();
    let text = std::fs::read_to_string(p)
        .map_err(|e| AudioError::Io { path: p.to_path_buf(), source: e })?;
    parse_rttm(&text)
}

pub fn format_rttm(turns: &[RttmTurn]) -> String {
    let mut out = String::new();
    for t in turns {
        out.push_str(&format!(
            "SPEAKER {} {} {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
            t.file_id, t.channel, t.start, t.duration, t.speaker
        ));
    }
    out
}

pub fn write_rttm(path: impl AsRef<Path>, turns: &[RttmTurn]) -> Result<()> {
    let p = path.as_ref();
    std::fs::write(p, format_rttm(turns))
        .map_err(|e| AudioError::Io { path: p.to_path_buf(), source: e })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_rttm() {
        let text = "SPEAKER audio1 1 0.500 1.200 <NA> <NA> spk1 <NA> <NA>\nSPEAKER audio1 1 2.000 0.500 <NA> <NA> spk2 <NA> <NA>\n";
        let turns = parse_rttm(text).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, "spk1");
    }

    #[test]
    fn round_trip_rttm() {
        let turns = vec![
            RttmTurn { file_id: "f".into(), channel: 1, start: 0.0, duration: 1.5, speaker: "a".into() },
            RttmTurn { file_id: "f".into(), channel: 1, start: 1.5, duration: 0.5, speaker: "b".into() },
        ];
        let text = format_rttm(&turns);
        let parsed = parse_rttm(&text).unwrap();
        assert_eq!(parsed, turns);
    }
}
