pub struct LoraTrainerConfig {
    pub r: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
}

impl Default for LoraTrainerConfig {
    fn default() -> Self { Self { r: 8, alpha: 16.0, target_modules: Vec::new() } }
}
