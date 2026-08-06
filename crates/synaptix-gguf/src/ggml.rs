use crate::error::{GgufError, Result};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    BF16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        use GgmlType::*;
        Ok(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => BF16,
            34 => Tq1_0,
            35 => Tq2_0,
            39 => Mxfp4,
            other => return Err(GgufError::BadTensorType(other)),
        })
    }

    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            Q8K => "Q8_K",
            Iq2Xxs => "IQ2_XXS",
            Iq2Xs => "IQ2_XS",
            Iq3Xxs => "IQ3_XXS",
            Iq1S => "IQ1_S",
            Iq4Nl => "IQ4_NL",
            Iq3S => "IQ3_S",
            Iq2S => "IQ2_S",
            Iq4Xs => "IQ4_XS",
            I8 => "I8",
            I16 => "I16",
            I32 => "I32",
            I64 => "I64",
            F64 => "F64",
            Iq1M => "IQ1_M",
            BF16 => "BF16",
            Tq1_0 => "TQ1_0",
            Tq2_0 => "TQ2_0",
            Mxfp4 => "MXFP4",
        }
    }

    pub fn block_elems(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Iq4Nl | Mxfp4 => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K | Iq2Xxs | Iq2Xs | Iq3Xxs | Iq1S | Iq3S | Iq2S
            | Iq4Xs | Iq1M | Tq1_0 | Tq2_0 => QK_K,
        }
    }

    pub fn block_bytes(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            I8 => 1,
            F64 | I64 => 8,
            Q4_0 => 2 + 16,
            Q4_1 => 2 + 2 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 2 + 2 + 4 + 16,
            Q8_0 => 2 + 32,
            Q8_1 => 4 + 4 + 32,
            Mxfp4 => 1 + 16,
            Q2K => QK_K / 16 + QK_K / 4 + 2 + 2,
            Q3K => QK_K / 8 + QK_K / 4 + 12 + 2,
            Q4K => 2 + 2 + K_SCALE_SIZE + QK_K / 2,
            Q5K => 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
            Q6K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
            Q8K => 4 + QK_K + QK_K / 16 * 2,
            Iq2Xxs => 2 + QK_K / 4,
            Iq2Xs => 2 + QK_K / 4 + QK_K / 32,
            Iq2S => 2 + QK_K / 4 + QK_K / 16,
            Iq3Xxs => 2 + QK_K / 4 + QK_K / 8,
            Iq3S => 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + QK_K / 64,
            Iq1S => 2 + QK_K / 8 + QK_K / 16,
            Iq1M => QK_K / 8 + QK_K / 16 + QK_K / 32,
            Iq4Nl => 2 + 16,
            Iq4Xs => 2 + 2 + QK_K / 64 + QK_K / 2,
            Tq1_0 => 2 + 4 * (QK_K / 64) + QK_K / 16,
            Tq2_0 => 2 + QK_K / 4,
        }
    }

    pub fn bytes_for(self, elems: usize) -> usize {
        let be = self.block_elems();
        elems.div_ceil(be) * self.block_bytes()
    }

    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::F64
                | GgmlType::I8
                | GgmlType::I16
                | GgmlType::I32
                | GgmlType::I64
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml() {

        assert_eq!(GgmlType::Q4_0.block_bytes(), 18);
        assert_eq!(GgmlType::Q4_1.block_bytes(), 20);
        assert_eq!(GgmlType::Q5_0.block_bytes(), 22);
        assert_eq!(GgmlType::Q5_1.block_bytes(), 24);
        assert_eq!(GgmlType::Q8_0.block_bytes(), 34);
        assert_eq!(GgmlType::Q2K.block_bytes(), 84);
        assert_eq!(GgmlType::Q3K.block_bytes(), 110);
        assert_eq!(GgmlType::Q4K.block_bytes(), 144);
        assert_eq!(GgmlType::Q5K.block_bytes(), 176);
        assert_eq!(GgmlType::Q6K.block_bytes(), 210);
        assert_eq!(GgmlType::Iq4Nl.block_bytes(), 18);
        assert_eq!(GgmlType::Iq4Xs.block_bytes(), 136);
    }

    #[test]
    fn bytes_for_rounds_up_to_block() {
        assert_eq!(GgmlType::Q8_0.bytes_for(64), 68);
        assert_eq!(GgmlType::F32.bytes_for(7), 28);
        assert_eq!(GgmlType::Q4K.bytes_for(512), 288);
    }
}
