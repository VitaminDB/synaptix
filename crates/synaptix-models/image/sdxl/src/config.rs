//! Параметры txt2img-генерации.

/// Высокоуровневые параметры одного txt2img-запуска. Архитектурные размеры
/// модели берутся из пресетов `synaptix_nn` (`*::sdxl()` / `ClipTextConfig`),
/// здесь — только то, что меняется от запроса к запросу.
#[derive(Debug, Clone)]
pub struct Txt2ImgParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for Txt2ImgParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            height: 1024,
            width: 1024,
            steps: 30,
            guidance_scale: 5.0,
            seed: 0,
        }
    }
}

impl Txt2ImgParams {
    /// Размер VAE-латента: SDXL ужимает в 8 раз по каждой оси.
    pub const VAE_SCALE: usize = 8;

    pub fn latent_height(&self) -> usize {
        self.height / Self::VAE_SCALE
    }

    pub fn latent_width(&self) -> usize {
        self.width / Self::VAE_SCALE
    }
}
