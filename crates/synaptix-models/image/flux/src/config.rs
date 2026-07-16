//! Параметры txt2img-генерации FLUX.

/// Высокоуровневые параметры одного FLUX txt2img-запуска. Архитектурные
/// размеры берутся из конфигов модели (`transformer/config.json` и т.д.),
/// здесь — только то, что меняется от запроса к запросу.
///
/// FLUX.1-dev guidance-distilled: `guidance_scale` подаётся в трансформер как
/// эмбеддинг (НЕ CFG), negative-промпта нет. Типичные значения: guidance 3.5,
/// steps 28-50.
#[derive(Debug, Clone)]
pub struct Txt2ImgParams {
    pub prompt: String,
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
            height: 1024,
            width: 1024,
            steps: 50,
            guidance_scale: 3.5,
            seed: 0,
        }
    }
}

impl Txt2ImgParams {
    /// VAE ужимает в 8 раз по каждой оси.
    pub const VAE_SCALE: usize = 8;
    /// MMDiT работает на packed-латенте (2×2 patchify) → ещё /2 по каждой оси.
    pub const PATCH: usize = 2;

    /// Высота латента VAE (до packing): `height / 8`.
    pub fn latent_height(&self) -> usize {
        self.height / Self::VAE_SCALE
    }

    pub fn latent_width(&self) -> usize {
        self.width / Self::VAE_SCALE
    }

    /// Длина packed-последовательности для трансформера: `(H/8/2) * (W/8/2)`.
    pub fn packed_seq_len(&self) -> usize {
        (self.latent_height() / Self::PATCH) * (self.latent_width() / Self::PATCH)
    }
}
