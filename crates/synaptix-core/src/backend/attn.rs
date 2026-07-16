//! Глобальный селектор attention-backend (long-context, Session 11 Phase 2).
//!
//! `Backend::flash_attention` читает этот режим и роутит запрос на конкретное
//! ядро. `Auto` (default) — эвристика по (Tq, длина KV). Остальные варианты
//! форсят конкретный backend (для A/B-бенчмарка кроссовера и явного выбора).

use std::sync::atomic::{AtomicU8, Ordering};

/// Вариант attention-backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttnMode {
    /// Эвристика: prefill (Tq>1, BF16) → FA-2 tiled; decode (Tq=1) → split-K
    /// flash-decode с адаптивным split_k по длине KV.
    #[default]
    Auto,
    /// Всегда split-K flash-decode (скалярные CUDA-cores; выигрывает на коротком
    /// контексте за счёт occupancy).
    FlashDecode,
    /// Всегда FlashAttention-2 tiled/single-row (reuse K/V в SRAM; выигрывает на
    /// длинном контексте). BF16; не-BF16 → fallback на flash-decode.
    Fa2,
    /// FlashAttention-4 (Blackwell WMMA). Phase 4 — head_dim пока захардкожен 256,
    /// для hd≠256 → fallback на FA-2.
    Fa4,
}

impl AttnMode {
    fn to_u8(self) -> u8 {
        match self {
            AttnMode::Auto => 0,
            AttnMode::FlashDecode => 1,
            AttnMode::Fa2 => 2,
            AttnMode::Fa4 => 3,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => AttnMode::FlashDecode,
            2 => AttnMode::Fa2,
            3 => AttnMode::Fa4,
            _ => AttnMode::Auto,
        }
    }

    /// Разбор из строки (CLI `--attn`, env `SYN_ATTN`). Возвращает `None` для
    /// неизвестных значений.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(AttnMode::Auto),
            "flash-decode" | "flash_decode" | "flashdecode" | "decode" => Some(AttnMode::FlashDecode),
            "fa2" | "flash2" | "flash-attn-2" => Some(AttnMode::Fa2),
            "fa4" | "flash4" | "flash-attn-4" => Some(AttnMode::Fa4),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AttnMode::Auto => "auto",
            AttnMode::FlashDecode => "flash-decode",
            AttnMode::Fa2 => "fa2",
            AttnMode::Fa4 => "fa4",
        }
    }
}

static ATTN_MODE: AtomicU8 = AtomicU8::new(0);

/// Установить глобальный attention-backend (один раз при старте CLI / per-variant
/// в бенчмарке).
pub fn set_mode(m: AttnMode) {
    ATTN_MODE.store(m.to_u8(), Ordering::Relaxed);
}

/// Текущий attention-backend. Если глобально `Auto` и задан `SYN_ATTN` — он
/// разбирается лениво (env как fallback, чтобы не требовать CLI-флаг).
pub fn mode() -> AttnMode {
    AttnMode::from_u8(ATTN_MODE.load(Ordering::Relaxed))
}
