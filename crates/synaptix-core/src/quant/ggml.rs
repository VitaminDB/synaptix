//! Типы тензоров ggml (GGUF) — идентификаторы, геометрия блоков, имена.
//!
//! Это **формат исполнения моделей llama.cpp**: движок читает блоки байт в
//! байт как в файле (один непрерывный блоб на тензор, шкалы внутри блока),
//! деквантует их на карте и умножает. Энкодеров у ggml-форматов в synaptix
//! нет — файлы берутся готовыми (решение Р1 плана
//! `quant_lowbit_gguf_plan_2026.md`); собственный квант движка — [`super::sq`].
//!
//! Номера — `enum ggml_type` из `ggml.h`; удалённые типы (4, 5, 31–33, 36–38)
//! не представлены. Размеры блоков — `GGML_QUANT_SIZES` из gguf-py и
//! `static_assert`'ы `ggml-common.h`.

use serde::{Deserialize, Serialize};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// NVFP4 в раскладке ggml: блок 64 = 4 под-блока по 16 с UE4M3-шкалами
    /// внутри блока. Это не `DType::NVFP4` движка (там шкалы отдельно и
    /// тайл-мажорно) — только формат файла.
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GgmlType {
    pub const ALL: [GgmlType; 35] = {
        use GgmlType::*;
        [
            F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, Iq2Xxs,
            Iq2Xs, Iq3Xxs, Iq1S, Iq4Nl, Iq3S, Iq2S, Iq4Xs, I8, I16, I32, I64, F64, Iq1M, BF16,
            Tq1_0, Tq2_0, Mxfp4, Nvfp4, Q1_0, Q2_0,
        ]
    };

    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
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
            40 => Nvfp4,
            41 => Q1_0,
            42 => Q2_0,
            _ => return None,
        })
    }

    /// Имя как в ggml (`GGML_TYPE_*` без префикса).
    pub const fn name(self) -> &'static str {
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
            Nvfp4 => "NVFP4",
            Q1_0 => "Q1_0",
            Q2_0 => "Q2_0",
        }
    }

    /// Машинный ключ формата (манифест бандла, CLI): имя в нижнем регистре.
    pub fn key(self) -> String {
        self.name().to_ascii_lowercase()
    }

    pub fn from_key(s: &str) -> Option<Self> {
        let up = s.trim().to_ascii_uppercase();
        Self::ALL.iter().copied().find(|t| t.name() == up)
    }

    /// Элементов в блоке.
    pub const fn block_elems(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Iq4Nl | Mxfp4 => 32,
            Nvfp4 => 64,
            Q2_0 => 64,
            Q1_0 => 128,
            _ => QK_K,
        }
    }

    /// Байт в блоке.
    pub const fn block_bytes(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            F64 | I64 => 8,
            I8 => 1,
            Q4_0 => 2 + 16,
            Q4_1 => 4 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 4 + 4 + 16,
            Q8_0 => 2 + 32,
            Q8_1 => 4 + 32,
            Q2K => QK_K / 16 + QK_K / 4 + 4,
            Q3K => QK_K / 8 + QK_K / 4 + 12 + 2,
            Q4K => 4 + K_SCALE_SIZE + QK_K / 2,
            Q5K => 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
            Q6K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
            Q8K => 4 + QK_K + QK_K / 16 * 2,
            Iq2Xxs => 2 + QK_K / 8 * 2,
            Iq2Xs => 2 + QK_K / 8 * 2 + QK_K / 32,
            Iq2S => 2 + QK_K / 4 + QK_K / 16,
            Iq3Xxs => 2 + 3 * QK_K / 8,
            Iq3S => 2 + 13 * QK_K / 32 + QK_K / 64,
            Iq1S => 2 + QK_K / 8 + QK_K / 16,
            Iq1M => QK_K / 8 + QK_K / 16 + QK_K / 32,
            Iq4Nl => 2 + 16,
            Iq4Xs => 2 + 2 + QK_K / 64 + QK_K / 2,
            Tq1_0 => 2 + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
            Tq2_0 => 2 + QK_K / 4,
            Mxfp4 => 1 + 16,
            Nvfp4 => 4 + 32,
            Q1_0 => 2 + 16,
            Q2_0 => 2 + 16,
        }
    }

    /// Байт на `elems` элементов одной строки (хвостовой блок — целиком).
    pub const fn bytes_for(self, elems: usize) -> usize {
        elems.div_ceil(self.block_elems()) * self.block_bytes()
    }

    /// Средних бит на вес (для сводок и `DType::size_in_bits`).
    pub const fn bits_per_weight_x8(self) -> usize {
        self.block_bytes() * 8 * 8 / self.block_elems()
    }

    pub const fn is_quantized(self) -> bool {
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

    /// Формат весов, который умеют ядра движка (деквант на карте). Q8_1 и
    /// Q8_K — форматы активаций для dp4a, как весов они не встречаются.
    pub const fn is_weight_format(self) -> bool {
        self.is_quantized() && !matches!(self, GgmlType::Q8_1 | GgmlType::Q8K)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml() {
        use GgmlType::*;
        let want = [
            (Q4_0, 32, 18),
            (Q4_1, 32, 20),
            (Q5_0, 32, 22),
            (Q5_1, 32, 24),
            (Q8_0, 32, 34),
            (Q8_1, 32, 36),
            (Q2K, 256, 84),
            (Q3K, 256, 110),
            (Q4K, 256, 144),
            (Q5K, 256, 176),
            (Q6K, 256, 210),
            (Q8K, 256, 292),
            (Iq2Xxs, 256, 66),
            (Iq2Xs, 256, 74),
            (Iq3Xxs, 256, 98),
            (Iq1S, 256, 50),
            (Iq4Nl, 32, 18),
            (Iq3S, 256, 110),
            (Iq2S, 256, 82),
            (Iq4Xs, 256, 136),
            (Iq1M, 256, 56),
            (Tq1_0, 256, 54),
            (Tq2_0, 256, 66),
            (Mxfp4, 32, 17),
            (Nvfp4, 64, 36),
            (Q1_0, 128, 18),
            (Q2_0, 64, 18),
        ];
        for (t, be, bb) in want {
            assert_eq!(t.block_elems(), be, "{}", t.name());
            assert_eq!(t.block_bytes(), bb, "{}", t.name());
        }
    }

    #[test]
    fn ids_and_keys_round_trip() {
        for t in GgmlType::ALL {
            assert_eq!(GgmlType::from_u32(t as u32), Some(t));
            assert_eq!(GgmlType::from_key(&t.key()), Some(t), "{}", t.name());
        }
        assert_eq!(GgmlType::from_u32(4), None);
        assert_eq!(GgmlType::from_key("q4_k"), Some(GgmlType::Q4K));
        assert_eq!(GgmlType::from_key("iq2_xxs"), Some(GgmlType::Iq2Xxs));
    }

    #[test]
    fn bytes_for_rounds_up() {
        assert_eq!(GgmlType::Q4_0.bytes_for(33), 36);
        assert_eq!(GgmlType::Q4K.bytes_for(256), 144);
        assert_eq!(GgmlType::Q4K.bytes_for(257), 288);
    }
}
