//! Референсы видеомоделей (MiniMax-H3): тип файла по расширению, метки, под
//! которыми модель видит референсы в промпте, и есть ли у видео звук. Без
//! весов и GPU — это нужно и тому, кто пишет промпт (агент Synthos), и самой
//! модели.

use std::path::Path;
use std::process::Command;

const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "heic", "heif", "tif", "tiff"];
const VIDEO_EXT: &[&str] = &["mp4", "mov", "mkv", "webm", "m4v", "avi"];
const AUDIO_EXT: &[&str] = &["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    Image,
    Video,
    Audio,
}

impl RefKind {
    /// Тип референса по расширению файла.
    pub fn of_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        if IMAGE_EXT.contains(&ext.as_str()) {
            Some(Self::Image)
        } else if VIDEO_EXT.contains(&ext.as_str()) {
            Some(Self::Video)
        } else if AUDIO_EXT.contains(&ext.as_str()) {
            Some(Self::Audio)
        } else {
            None
        }
    }
}

/// Метки, под которыми презентация покажет референсы модели, — на них и
/// ссылается промпт. Вход — тип и «есть дорожка» (для видео: просили звук и
/// он в файле есть), в порядке списка. Нумерация своя у каждого типа; дорожка
/// видео занимает очередной `<Audio j>` раньше отдельных аудио после неё.
pub fn labels(refs: &[(RefKind, bool)]) -> Vec<String> {
    let (mut img, mut vid, mut aud) = (0usize, 0usize, 0usize);
    refs.iter()
        .map(|(kind, with_audio)| match kind {
            RefKind::Image => {
                img += 1;
                format!("<Picture {img}>")
            }
            RefKind::Audio => {
                aud += 1;
                format!("<Audio {aud}>")
            }
            RefKind::Video => {
                vid += 1;
                if *with_audio {
                    aud += 1;
                    format!("<Video {vid}> + <Audio {aud}>")
                } else {
                    format!("<Video {vid}>")
                }
            }
        })
        .collect()
}

/// Есть ли в файле аудиопоток (ffprobe). От этого зависит нумерация
/// `<Audio j>`, так что узнать это надо до прогона — когда пишется промпт.
pub fn has_audio_stream(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("audio"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_labels() {
        assert_eq!(RefKind::of_path(Path::new("a.JPG")), Some(RefKind::Image));
        assert_eq!(RefKind::of_path(Path::new("a.txt")), None);
        let l = labels(&[(RefKind::Image, false), (RefKind::Video, true), (RefKind::Audio, false), (RefKind::Video, false)]);
        assert_eq!(l, ["<Picture 1>", "<Video 1> + <Audio 1>", "<Audio 2>", "<Video 2>"]);
    }
}
