//! Раскладка квантованных весов внутри `.syn` (`syn-quant-v1`).
//!
//! Обычный tensors-чанк хранит веса как есть. Когда бандл собран с
//! квантованием, выбранные тензоры заменяются парой блобов, а описание
//! лежит рядом файловым чанком:
//!
//! * `<имя>.qpacked` — упакованные веса, `U8`. Форма повторяет исходную с
//!   уполовиненной последней осью для NVFP4 (`[N, K/2]`) и без изменений для
//!   MXFP8 (`[N, K]`). Стопка матриц `[E, N, K]` квантуется послойно и
//!   сохраняет ведущую ось.
//! * `<имя>.qscales` — блочные масштабы, `U8`, одномерный блоб. Их раскладку
//!   задаёт ядро, разбирать её снаружи не нужно и нельзя.
//! * [`MANIFEST_NAME`] — JSON: какой тензор каким форматом упакован и какой
//!   формы он был. Без манифеста читатель не отличит квант от обычного `U8`.
//!
//! Бандл с такой раскладкой обязан объявлять [`crate::CAP_QUANT_WEIGHTS`] в
//! `required_caps`: читатель без поддержки тогда честно откажется его
//! открыть, вместо того чтобы не найти половину тензоров и списать это на
//! битый файл.
//!
//! Модуль намеренно не зависит ни от ядер, ни от тензорной библиотеки —
//! здесь только имена, формы и арифметика размеров. Считать сам квант умеет
//! писатель (`synthos`), читать — `synaptix-io`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bundle::Bundle;
use crate::inspect::QuantKind;

/// Имя файла-манифеста внутри бандла.
pub const MANIFEST_NAME: &str = "quant_manifest.json";

pub const PACKED_SUFFIX: &str = ".qpacked";
pub const SCALES_SUFFIX: &str = ".qscales";

/// Текущая версия раскладки.
pub const VERSION: u32 = 1;

/// Машинный ключ формата, как он лежит в манифесте.
pub fn format_key(kind: QuantKind) -> &'static str {
    match kind {
        QuantKind::Nvfp4 => "nvfp4",
        QuantKind::Mxfp8 => "mxfp8",
    }
}

pub fn format_from_key(s: &str) -> Option<QuantKind> {
    match s {
        "nvfp4" => Some(QuantKind::Nvfp4),
        "mxfp8" => Some(QuantKind::Mxfp8),
        _ => None,
    }
}

/// Описание одного квантованного тензора.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantEntry {
    /// `nvfp4` | `mxfp8`.
    pub format: String,
    /// Исходная форма. По ней восстанавливаются `N`/`K` и число срезов —
    /// гадать по размеру блоба не нужно.
    pub shape: Vec<usize>,
}

impl QuantEntry {
    pub fn kind(&self) -> Option<QuantKind> {
        format_from_key(&self.format)
    }

    /// `(число срезов, N, K)`. `None` — форма не матрица и не стопка матриц,
    /// то есть манифест противоречит сам себе.
    pub fn dims(&self) -> Option<(usize, usize, usize)> {
        match self.shape.as_slice() {
            [n, k] => Some((1, *n, *k)),
            [e, n, k] => Some((*e, *n, *k)),
            _ => None,
        }
    }

    /// Сколько байт занимают упакованные веса (без масштабов).
    pub fn packed_bytes(&self) -> Option<u64> {
        let (slices, n, k) = self.dims()?;
        let last = match self.kind()? {
            QuantKind::Nvfp4 => k / 2,
            QuantKind::Mxfp8 => k,
        };
        Some(slices as u64 * n as u64 * last as u64)
    }

    /// Сколько байт занимают масштабы.
    pub fn scales_bytes(&self) -> Option<u64> {
        let total = crate::inspect::quantized_bytes(&self.shape, self.kind()?)?;
        Some(total - self.packed_bytes()?)
    }

    /// Число матриц в стопке: 1 для обычного двумерного веса, `E` для
    /// экспертов MoE.
    pub fn slices(&self) -> Option<usize> {
        Some(self.dims()?.0)
    }

    /// Байты одной матрицы стопки в блобе `.qpacked`.
    ///
    /// Писатель кладёт срезы подряд — сперва все упакованные веса
    /// (`эксперт 0`, `эксперт 1`, …), затем тем же порядком все масштабы.
    /// Поэтому эксперт `i` читается срезом `[i * шаг, (i + 1) * шаг)` в
    /// каждом из двух блобов.
    pub fn packed_bytes_per_slice(&self) -> Option<u64> {
        let slices = self.slices()? as u64;
        Some(self.packed_bytes()? / slices)
    }

    /// Байты масштабов одной матрицы стопки в блобе `.qscales`.
    pub fn scales_bytes_per_slice(&self) -> Option<u64> {
        let slices = self.slices()? as u64;
        Some(self.scales_bytes()? / slices)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantManifest {
    pub version: u32,
    pub packed_suffix: String,
    pub scales_suffix: String,
    pub tensors: BTreeMap<String, QuantEntry>,
}

impl Default for QuantManifest {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantManifest {
    pub fn new() -> Self {
        Self {
            version: VERSION,
            packed_suffix: PACKED_SUFFIX.to_string(),
            scales_suffix: SCALES_SUFFIX.to_string(),
            tensors: BTreeMap::new(),
        }
    }

    /// Прочитать манифест из бандла. `None` — бандл собран без квантования
    /// (обычный случай) либо манифест не разобрался.
    pub fn read_from(bundle: &Bundle) -> Option<Self> {
        let raw = bundle.read_file(MANIFEST_NAME).ok()?;
        let m: Self = serde_json::from_slice(&raw).ok()?;
        // Версию проверяем строго: чужая раскладка под теми же именами
        // хуже отсутствующей.
        (m.version == VERSION).then_some(m)
    }

    pub fn entry(&self, name: &str) -> Option<&QuantEntry> {
        self.tensors.get(name)
    }

    pub fn packed_name(&self, name: &str) -> String {
        format!("{name}{}", self.packed_suffix)
    }

    pub fn scales_name(&self, name: &str) -> String {
        format!("{name}{}", self.scales_suffix)
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(shape: Vec<usize>, format: &str) -> QuantEntry {
        QuantEntry { format: format.into(), shape }
    }

    #[test]
    fn packed_and_scales_add_up_to_the_estimate() {
        for (shape, fmt) in [
            (vec![17408usize, 5120], "nvfp4"),
            (vec![17408, 5120], "mxfp8"),
            (vec![512, 1280, 2560], "nvfp4"),
        ] {
            let e = entry(shape.clone(), fmt);
            let total = crate::inspect::quantized_bytes(&shape, e.kind().unwrap()).unwrap();
            assert_eq!(e.packed_bytes().unwrap() + e.scales_bytes().unwrap(), total);
        }
    }

    #[test]
    fn dims_read_matrices_and_stacks() {
        assert_eq!(entry(vec![128, 256], "nvfp4").dims(), Some((1, 128, 256)));
        assert_eq!(entry(vec![512, 128, 256], "nvfp4").dims(), Some((512, 128, 256)));
        assert_eq!(entry(vec![128], "nvfp4").dims(), None);
    }

    #[test]
    fn unknown_format_is_not_guessed() {
        let e = entry(vec![128, 256], "int4");
        assert_eq!(e.kind(), None);
        assert_eq!(e.packed_bytes(), None);
    }

    #[test]
    fn manifest_round_trips_and_rejects_other_versions() {
        let mut m = QuantManifest::new();
        m.tensors.insert("w".into(), entry(vec![128, 256], "nvfp4"));
        let raw = serde_json::to_vec(&m).unwrap();
        let back: QuantManifest = serde_json::from_slice(&raw).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.packed_name("w"), "w.qpacked");
        assert_eq!(back.scales_name("w"), "w.qscales");

        let mut wrong = m.clone();
        wrong.version = 99;
        let raw = serde_json::to_vec(&wrong).unwrap();
        let parsed: QuantManifest = serde_json::from_slice(&raw).unwrap();
        assert_ne!(parsed.version, VERSION, "чужую версию читатель обязан отвергнуть");
    }
}
