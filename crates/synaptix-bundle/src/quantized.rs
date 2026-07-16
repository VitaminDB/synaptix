//! Quantized tensors chunk (`CHUNK_TYPE_QUANTIZED_TENSORS = 6`).
//!
//! Payload prefix — 16-байтный [`QuantizedChunkHeader`]; за ним идёт
//! формат-зависимый блок данных, описанный полем `format`:
//!
//! | Format                  | ID | Layout                       |
//! |-------------------------|----|------------------------------|
//! | `QUANT_FORMAT_FP8E4M3`  | 2  | per-block FP8 E4M3 (legacy)  |
//!
//! ID=1 (бывший embedded-GGUF) и ID=2 — deprecated/reserved для forward-compat
//! reader'ов; новые bundle'ы их не записывают. Нативные quant-форматы synaptix —
//! NVFP4 / MXFP8 (block-scale, см. core `Tensor::quantize_to_nvfp4/mxfp8`).
//!
//! Forward-compat: reader без поддержки конкретного формата встречает chunk
//! и должен скипнуть — fallback на стандартный `CHUNK_TYPE_TENSORS = 1`,
//! который остаётся в bundle'е рядом.

use crate::error::{Error, Result};

/// Header перед quantized payload. Сериализуется в `u32 LE` для каждого поля.
/// Размер — 16 байт.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedChunkHeader {
    /// Формат-дискриминатор. См. `QUANT_FORMAT_*`.
    pub format: u32,
    /// Reserved флаги (zero для будущего использования).
    pub flags: u32,
    /// Размер сырого payload'а после header'а (без header'а самого).
    pub payload_len: u64,
}

impl QuantizedChunkHeader {
    pub const SIZE: usize = 16;

    pub fn encode(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&self.format.to_le_bytes());
        out[4..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..16].copy_from_slice(&self.payload_len.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::CborDecode(format!(
                "QuantizedChunkHeader: need {} bytes, got {}",
                Self::SIZE,
                bytes.len()
            )));
        }
        let format = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let payload_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        Ok(Self {
            format,
            flags,
            payload_len,
        })
    }
}

/// FP8 E4M3 — собственный layout (см. `crates/quant-fp8`).
pub const QUANT_FORMAT_FP8E4M3: u32 = 2;

/// Имя стандартного quantized-чанка — соответствует `tensors:main` у
/// обычного [`crate::cdir::CHUNK_TYPE_TENSORS`].
pub const QUANTIZED_CHUNK_NAME: &str = "quantized:main";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = QuantizedChunkHeader {
            format: QUANT_FORMAT_FP8E4M3,
            flags: 0,
            payload_len: 12345,
        };
        let bytes = h.encode();
        let decoded = QuantizedChunkHeader::decode(&bytes).unwrap();
        assert_eq!(decoded, h);
    }

    #[test]
    fn header_too_short_errors() {
        let bytes = [0u8; 8];
        assert!(QuantizedChunkHeader::decode(&bytes).is_err());
    }
}
