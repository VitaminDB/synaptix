use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttnMode {
    Off,
    Fa2,
    Fa4,
}

impl FlashAttnMode {
    fn to_u8(self) -> u8 {
        match self {
            FlashAttnMode::Off => 0,
            FlashAttnMode::Fa2 => 1,
            FlashAttnMode::Fa4 => 2,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => FlashAttnMode::Off,
            1 => FlashAttnMode::Fa2,
            _ => FlashAttnMode::Fa4,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FlashAttnMode::Off => "off",
            FlashAttnMode::Fa2 => "fa2",
            FlashAttnMode::Fa4 => "fa4",
        }
    }

    pub fn wmma_force(self) -> Option<bool> {
        match self {
            FlashAttnMode::Off | FlashAttnMode::Fa4 => None,
            FlashAttnMode::Fa2 => Some(false),
        }
    }
}

impl std::str::FromStr for FlashAttnMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "off" | "Off" | "OFF" => Ok(FlashAttnMode::Off),
            "fa2" | "Fa2" | "FA2" | "fa-2" | "FA-2" => Ok(FlashAttnMode::Fa2),
            "fa4" | "Fa4" | "FA4" | "fa-4" | "FA-4" => Ok(FlashAttnMode::Fa4),
            _ => Err(()),
        }
    }
}

static FLASH_ATTN_MODE: AtomicU8 = AtomicU8::new(2);

pub fn set_flash_attn_mode(mode: FlashAttnMode) {
    FLASH_ATTN_MODE.store(mode.to_u8(), Ordering::Relaxed);
}

pub fn flash_attn_mode() -> FlashAttnMode {
    FlashAttnMode::from_u8(FLASH_ATTN_MODE.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for m in [FlashAttnMode::Off, FlashAttnMode::Fa2, FlashAttnMode::Fa4] {
            set_flash_attn_mode(m);
            assert_eq!(flash_attn_mode(), m);
        }
    }

    #[test]
    fn str_parse() {
        use std::str::FromStr;
        assert_eq!(FlashAttnMode::from_str("fa4"), Ok(FlashAttnMode::Fa4));
        assert_eq!(FlashAttnMode::from_str("FA-2"), Ok(FlashAttnMode::Fa2));
        assert_eq!(FlashAttnMode::from_str("off"), Ok(FlashAttnMode::Off));
        assert_eq!(FlashAttnMode::from_str("garbage"), Err(()));
    }
}
