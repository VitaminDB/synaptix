//! Деквантование ggml-блоков в f32 — обёртка над CPU-эталоном
//! `synaptix_core::quant::ggml_dequant` с ошибками этого крейта.

use crate::error::{GgufError, Result};
use crate::ggml::GgmlType;

pub fn dequantize(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
    let need = ty.bytes_for(n);
    if src.len() < need {
        return Err(GgufError::Truncated { at: 0, need, have: src.len() });
    }
    if dst.len() < n {
        return Err(GgufError::Truncated { at: 0, need: n, have: dst.len() });
    }
    synaptix_core::quant::ggml_dequant::dequantize(ty, src, n, dst)
        .map_err(|_| GgufError::UnsupportedQuant(ty.name()))
}
