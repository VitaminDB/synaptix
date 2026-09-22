//! Блочные квант-форматы весов, живущие одним блобом (шкалы внутри блока):
//! собственный [`sq`] и типы ggml для файлов llama.cpp ([`ggml`]).
//!
//! Оба представлены в [`crate::dtype::DType`] (`Sq { bits }`, `Ggml(..)`),
//! хранятся в [`crate::tensor::quant::QuantWeight`] без отдельного тензора
//! масштабов и деквантуются на карте одним модулем. Здесь — геометрия, ключи
//! форматов и CPU-эталоны деквантования (и эталонный энкодер SQ).

pub mod ggml;
pub mod ggml_dequant;
pub mod ggml_tables;
pub mod sq;

pub use ggml::GgmlType;

use crate::dtype::DType;

/// Машинный ключ квант-формата (манифест бандла, CLI, настройки):
/// `nvfp4`, `mxfp8`, `sq1`…`sq8`, `ggml:q4_k`, `ggml:iq2_xxs`, …
pub fn format_key(dtype: DType) -> Option<String> {
    match dtype {
        DType::NVFP4 => Some("nvfp4".into()),
        DType::MXFP8 => Some("mxfp8".into()),
        DType::Sq { bits } => Some(format!("sq{bits}")),
        DType::Ggml(t) => Some(format!("ggml:{}", t.key())),
        _ => None,
    }
}

/// Обратное к [`format_key`]. Неизвестный ключ — `None`, а не догадка.
pub fn format_from_key(s: &str) -> Option<DType> {
    let s = s.trim();
    match s {
        "nvfp4" => return Some(DType::NVFP4),
        "mxfp8" | "fp8" => return Some(DType::MXFP8),
        _ => {}
    }
    if let Some(rest) = s.strip_prefix("sq") {
        let bits: u8 = rest.parse().ok()?;
        return sq::check_bits(bits).ok().map(|_| DType::Sq { bits });
    }
    if let Some(rest) = s.strip_prefix("ggml:") {
        return GgmlType::from_key(rest).filter(|t| t.is_weight_format()).map(DType::Ggml);
    }
    None
}

/// Байт на строку `[k]` веса в одноблобном формате; `None` — формат не
/// одноблобный (NVFP4/MXFP8 держат шкалы отдельно) или `k` не кратен блоку.
pub fn block_row_bytes(dtype: DType, k: usize) -> Option<usize> {
    match dtype {
        DType::Sq { bits } => (k % sq::SUB_BLOCK == 0).then(|| sq::row_bytes(bits, k)),
        DType::Ggml(t) => (k % t.block_elems() == 0).then(|| t.bytes_for(k)),
        _ => None,
    }
}

/// Деквант строки `[k]` одноблобного формата в f32 (CPU-эталон).
pub fn dequant_row_f32(dtype: DType, src: &[u8], k: usize, dst: &mut [f32]) -> crate::error::Result<()> {
    match dtype {
        DType::Sq { bits } => sq::dequant_row(bits, src, k, dst),
        DType::Ggml(t) => ggml_dequant::dequantize(t, src, k, dst),
        _ => Err(crate::error::SynaptixError::Unsupported(
            "dequant_row_f32: формат не одноблобный",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip() {
        let cases = [
            DType::NVFP4,
            DType::MXFP8,
            DType::Sq { bits: 1 },
            DType::Sq { bits: 4 },
            DType::Sq { bits: 8 },
            DType::Ggml(GgmlType::Q4K),
            DType::Ggml(GgmlType::Iq2Xxs),
            DType::Ggml(GgmlType::Tq1_0),
        ];
        for d in cases {
            let k = format_key(d).unwrap();
            assert_eq!(format_from_key(&k), Some(d), "{k}");
        }
        assert_eq!(format_key(DType::Sq { bits: 4 }).unwrap(), "sq4");
        assert_eq!(format_key(DType::Ggml(GgmlType::Q4K)).unwrap(), "ggml:q4_k");
        assert_eq!(format_from_key("sq0"), None);
        assert_eq!(format_from_key("sq9"), None);
        assert_eq!(format_from_key("int4"), None);
        assert_eq!(format_from_key("ggml:q8_1"), None, "формат активаций — не вес");
        assert_eq!(format_from_key("ggml:f16"), None);
        assert_eq!(format_key(DType::BF16), None);
    }

    #[test]
    fn row_bytes_by_format() {
        assert_eq!(block_row_bytes(DType::Sq { bits: 4 }, 4096), Some(16 * 148));
        assert_eq!(block_row_bytes(DType::Ggml(GgmlType::Q4_0), 4096), Some(128 * 18));
        assert_eq!(block_row_bytes(DType::Ggml(GgmlType::Q4K), 4096), Some(16 * 144));
        assert_eq!(block_row_bytes(DType::Ggml(GgmlType::Q4K), 4000), None);
        assert_eq!(block_row_bytes(DType::NVFP4, 4096), None);
    }
}
