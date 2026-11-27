//! Этап генерации (видео, картинки, музыка) для индикатора службы: снаружи
//! видны только шаги денойза, а долгие этапы — кодирование текста, загрузка
//! DiT, декодер VAE — проходили молча. Архитектура пишет этап и раскладку
//! блоков (сколько на карте); служба добавляет их к идущей работе. Одна
//! генерация на устройство — статус общий на процесс.

use std::sync::Mutex;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenStatus {
    /// Этап словами: «кодирование текста», «загрузка DiT», «денойз», «декодер VAE».
    pub phase: String,
    /// Блоки (слои) модели этапа: на карте и всего; `None` — неизвестно.
    pub blocks: Option<(u32, u32)>,
}

static STATUS: Mutex<Option<GenStatus>> = Mutex::new(None);

/// Превью генерации: маленькая картинка текущей оценки результата (RGB8,
/// строки сверху вниз) и шаг, на котором она снята.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preview {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub step: usize,
    pub total: usize,
}

static PREVIEW: Mutex<Option<Preview>> = Mutex::new(None);

/// Новое превью (прежнее, если его не забрали, заменяется).
pub fn set_preview(p: Preview) {
    *PREVIEW.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
}

/// Коэффициенты линейной проекции латента в RGB из байтов f32 LE: `3·(c+1)`
/// чисел (веса и смещение на канал цвета); другое число байт — `None`.
pub fn parse_factors(bytes: &[u8], c: usize) -> Option<Vec<f32>> {
    (bytes.len() == 3 * (c + 1) * 4).then(|| bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Каждые сколько шагов снимать превью (`SYN_GEN_PREVIEW_EVERY`, 0 — выключить).
pub fn preview_every(default: usize) -> usize {
    std::env::var("SYN_GEN_PREVIEW_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Превью линейной проекцией: `x0` — `h·w` строк по `c` каналов (оценка
/// результата в латенте), `rgb = W·x + b` с обрезкой в [0, 1].
pub fn linear_preview(x0: &[f32], c: usize, width: usize, height: usize, factors: &[f32], step: usize, total: usize) {
    if factors.len() != 3 * (c + 1) || x0.len() != width * height * c {
        return;
    }
    let mut rgb = vec![0u8; width * height * 3];
    for (i, px) in x0.chunks_exact(c).enumerate() {
        for ch in 0..3 {
            let w = &factors[ch * (c + 1)..(ch + 1) * (c + 1)];
            let y = w[c] + px.iter().zip(w).map(|(a, b)| a * b).sum::<f32>();
            rgb[i * 3 + ch] = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    set_preview(Preview { width, height, rgb, step, total });
}

/// Забрать превью, если появилось новое.
pub fn take_preview() -> Option<Preview> {
    PREVIEW.lock().unwrap_or_else(|e| e.into_inner()).take()
}

fn with(f: impl FnOnce(&mut Option<GenStatus>)) {
    f(&mut STATUS.lock().unwrap_or_else(|e| e.into_inner()));
}

/// Новый этап; раскладка блоков прошлого этапа сбрасывается.
pub fn phase(p: impl Into<String>) {
    let p = p.into();
    with(|s| *s = Some(GenStatus { phase: p, blocks: None }));
}

/// Раскладка блоков текущего этапа: `vram` на карте из `total`.
pub fn blocks(vram: usize, total: usize) {
    with(|s| s.get_or_insert_with(Default::default).blocks = Some((vram as u32, total as u32)));
}

pub fn clear() {
    with(|s| *s = None);
    let _ = take_preview();
}

pub fn get() -> Option<GenStatus> {
    STATUS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Снимает статус по выходу из генерации (в том числе по ошибке).
pub struct Scope;

impl Scope {
    pub fn new(first: impl Into<String>) -> Self {
        phase(first);
        Scope
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_and_scope() {
        {
            let _s = Scope::new("кодирование текста");
            blocks(36, 50);
            assert_eq!(get(), Some(GenStatus { phase: "кодирование текста".into(), blocks: Some((36, 50)) }));
            phase("денойз");
            assert_eq!(get().unwrap().blocks, None);
            // Проекция: один пиксель, два канала; R = x0 + x1, G = смещение 0.5, B = 0.
            linear_preview(&[0.25, 0.5], 2, 1, 1, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 3, 8);
            let p = take_preview().unwrap();
            assert_eq!((p.rgb, p.step, p.total), (vec![191, 128, 0], 3, 8));
        }
        assert_eq!(get(), None);
    }
}
