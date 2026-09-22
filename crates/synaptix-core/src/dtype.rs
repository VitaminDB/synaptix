use serde::{Deserialize, Serialize};

use crate::quant::GgmlType;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,

    /// E2M1 + E4M3-шкала на 16, шкалы отдельным тензором (тайл-мажорно).
    NVFP4,
    /// E4M3 + E8M0-шкала на 32, шкалы отдельным тензором.
    MXFP8,

    U8,
    U32,
    I32,
    I64,

    /// Собственный блочный формат `bits ∈ 1..=8`, один блоб — см.
    /// [`crate::quant::sq`].
    Sq { bits: u8 },
    /// Блочный тип ggml (файлы llama.cpp), один блоб байт в байт как в GGUF —
    /// см. [`crate::quant::ggml`]. Только для исполнения, энкодеров нет.
    Ggml(GgmlType),
}

impl DType {
    pub const fn size_in_bits(self) -> usize {
        match self {
            DType::F64 => 64,
            DType::F32 => 32,
            DType::F16 => 16,
            DType::BF16 => 16,
            DType::NVFP4 => 4,
            DType::MXFP8 => 8,
            DType::U8 => 8,
            DType::U32 => 32,
            DType::I32 => 32,
            DType::I64 => 64,
            DType::Sq { bits } => bits as usize,
            // Средняя оценка: шкалы размазаны по блоку.
            DType::Ggml(t) => t.bits_per_weight_x8() / 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::F64 | DType::F32 | DType::F16 | DType::BF16)
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self, DType::NVFP4 | DType::MXFP8 | DType::Sq { .. } | DType::Ggml(_))
    }

    /// Формат «один блоб»: шкалы лежат внутри блока, отдельного тензора
    /// масштабов нет (SQ и типы ggml). NVFP4/MXFP8 — нет.
    pub const fn is_block_quant(self) -> bool {
        matches!(self, DType::Sq { .. } | DType::Ggml(_))
    }

    pub const fn is_integer(self) -> bool {
        matches!(self, DType::U8 | DType::U32 | DType::I32 | DType::I64)
    }

    pub const fn is_sub_byte(self) -> bool {
        match self {
            DType::NVFP4 => true,
            DType::Sq { bits } => bits < 8,
            DType::Ggml(t) => t.block_bytes() * 8 < t.block_elems() * 8,
            _ => false,
        }
    }

    pub fn bytes_for_numel(self, numel: usize) -> usize {
        let bits = self.size_in_bits();
        match self {
            DType::NVFP4 => {
                let block_elems = 16;
                let block_bytes = 8 + 1;
                numel.div_ceil(block_elems) * block_bytes
            }
            // Одна строка из `numel` значений: хвостовой блок — целиком.
            DType::Sq { bits } => crate::quant::sq::row_bytes(bits, numel),
            DType::Ggml(t) => t.bytes_for(numel),
            _ => (numel * bits).div_ceil(8),
        }
    }
}

pub trait SynaptixScalar: bytemuck::Pod + Copy + Send + Sync + 'static {
    const DTYPE: DType;
}

impl SynaptixScalar for f32 { const DTYPE: DType = DType::F32; }
impl SynaptixScalar for f64 { const DTYPE: DType = DType::F64; }
impl SynaptixScalar for half::f16 { const DTYPE: DType = DType::F16; }
impl SynaptixScalar for half::bf16 { const DTYPE: DType = DType::BF16; }
impl SynaptixScalar for u8 { const DTYPE: DType = DType::U8; }
impl SynaptixScalar for u32 { const DTYPE: DType = DType::U32; }
impl SynaptixScalar for i32 { const DTYPE: DType = DType::I32; }
impl SynaptixScalar for i64 { const DTYPE: DType = DType::I64; }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_classification() {
        assert!(DType::F32.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::U32.is_float());
        assert!(!DType::NVFP4.is_float());
    }

    #[test]
    fn quantized_classification() {
        assert!(DType::NVFP4.is_quantized());
        assert!(DType::MXFP8.is_quantized());
        assert!(!DType::F32.is_quantized());
    }

    #[test]
    fn integer_classification() {
        assert!(DType::U32.is_integer());
        assert!(DType::I64.is_integer());
        assert!(!DType::F32.is_integer());
    }

    #[test]
    fn bytes_for_numel_basic() {
        assert_eq!(DType::F32.bytes_for_numel(10), 40);
        assert_eq!(DType::F16.bytes_for_numel(10), 20);
        assert_eq!(DType::BF16.bytes_for_numel(10), 20);
        assert_eq!(DType::U8.bytes_for_numel(7), 7);
        assert_eq!(DType::I64.bytes_for_numel(3), 24);
    }

    #[test]
    fn bytes_for_numel_quantized() {
        assert_eq!(DType::NVFP4.bytes_for_numel(16), 9);
        assert_eq!(DType::MXFP8.bytes_for_numel(32), 32);
        assert_eq!(DType::Sq { bits: 4 }.bytes_for_numel(256), 148);
        assert_eq!(DType::Ggml(GgmlType::Q4K).bytes_for_numel(512), 288);
    }

    #[test]
    fn block_quant_classification() {
        assert!(DType::Sq { bits: 2 }.is_quantized() && DType::Sq { bits: 2 }.is_block_quant());
        assert!(DType::Ggml(GgmlType::Q8_0).is_block_quant());
        assert!(!DType::NVFP4.is_block_quant() && !DType::MXFP8.is_block_quant());
        assert!(DType::Sq { bits: 3 }.is_sub_byte() && !DType::Sq { bits: 8 }.is_sub_byte());
        assert!(DType::Ggml(GgmlType::Q4_0).is_sub_byte() && !DType::Ggml(GgmlType::Q8_0).is_sub_byte());
        assert_eq!(DType::Sq { bits: 5 }.size_in_bits(), 5);
        assert_eq!(DType::Ggml(GgmlType::Q4K).size_in_bits(), 4);
    }

    #[test]
    fn scalar_dtype_links() {
        assert_eq!(<f32 as SynaptixScalar>::DTYPE, DType::F32);
        assert_eq!(<half::f16 as SynaptixScalar>::DTYPE, DType::F16);
        assert_eq!(<half::bf16 as SynaptixScalar>::DTYPE, DType::BF16);
        assert_eq!(<u32 as SynaptixScalar>::DTYPE, DType::U32);
    }
}
