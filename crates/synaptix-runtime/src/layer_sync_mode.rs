use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSyncMode {
    Auto,
    On,
    Off,
}

impl LayerSyncMode {
    fn to_u8(self) -> u8 {
        match self {
            LayerSyncMode::Auto => 0,
            LayerSyncMode::On => 1,
            LayerSyncMode::Off => 2,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => LayerSyncMode::On,
            2 => LayerSyncMode::Off,
            _ => LayerSyncMode::Auto,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            LayerSyncMode::Auto => "auto",
            LayerSyncMode::On => "on",
            LayerSyncMode::Off => "off",
        }
    }
}

impl std::str::FromStr for LayerSyncMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "auto" | "Auto" | "AUTO" => Ok(LayerSyncMode::Auto),
            "on" | "On" | "ON" | "1" | "true" => Ok(LayerSyncMode::On),
            "off" | "Off" | "OFF" | "0" | "false" => Ok(LayerSyncMode::Off),
            _ => Err(()),
        }
    }
}

static LAYER_SYNC_MODE: AtomicU8 = AtomicU8::new(0);

pub fn set_layer_sync_mode(m: LayerSyncMode) {
    LAYER_SYNC_MODE.store(m.to_u8(), Ordering::Relaxed);
}

pub fn layer_sync_mode() -> LayerSyncMode {
    LayerSyncMode::from_u8(LAYER_SYNC_MODE.load(Ordering::Relaxed))
}

pub fn layer_sync_should_apply(input_tokens: usize) -> bool {
    match layer_sync_mode() {
        LayerSyncMode::On => true,
        LayerSyncMode::Off => false,
        LayerSyncMode::Auto => input_tokens > 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for m in [LayerSyncMode::Auto, LayerSyncMode::On, LayerSyncMode::Off] {
            set_layer_sync_mode(m);
            assert_eq!(layer_sync_mode(), m);
        }
    }

    #[test]
    fn auto_branches_on_input_tokens() {
        set_layer_sync_mode(LayerSyncMode::Auto);
        assert!(!layer_sync_should_apply(1));
        assert!(layer_sync_should_apply(2));
    }
}
