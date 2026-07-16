pub struct DynamicLossScaler {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
}

impl Default for DynamicLossScaler {
    fn default() -> Self { Self { scale: 65536.0, growth_factor: 2.0, backoff_factor: 0.5 } }
}

impl DynamicLossScaler {
    pub fn update(&mut self, has_overflow: bool) {
        if has_overflow { self.scale *= self.backoff_factor; } else { self.scale *= self.growth_factor; }
    }
}
