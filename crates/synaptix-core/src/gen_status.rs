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
        }
        assert_eq!(get(), None);
    }
}
