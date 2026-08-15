use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::source::{H3EncoderSource, H3Source};
use crate::H3Error;

pub use synaptix_vlm_qwen3::pipeline::{DirWeights, H3Conditioning, H3Encoder, H3_ENCODER_LAYERS};
pub use synaptix_vlm_qwen3::presentation::{H3Presentation, RefItem, TokenTag, VideoBlock};
pub use synaptix_vlm_qwen3::preprocess::ImageGrid;

pub struct EncoderHandle {
    inner: H3Encoder,
}

fn encoder_layers() -> usize {
    std::env::var("H3_ENCODER_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(H3_ENCODER_LAYERS)
}

impl EncoderHandle {
    pub fn load(
        encoder_dir: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, H3Error> {
        let inner = H3Encoder::load(encoder_dir, None, device, compute, quant, encoder_layers())
            .map_err(|e| H3Error::Load(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Энкодер из источника: каталог `text_encoder/`, отдельный `.syn`-бандл
    /// энкодера или компонент `text_encoder` внутри бандла модели.
    pub fn load_source(
        src: &H3EncoderSource,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, H3Error> {
        if let H3EncoderSource::Dir(dir) = src {
            return Self::load(dir, device, compute, quant);
        }
        let cfg = src.read("config.json")?;
        let tok = src.read("tokenizer.json")?;
        let weights = DirWeights::from_loader(src.loader(device)?);
        let inner = H3Encoder::from_parts(
            &cfg,
            &tok,
            &weights,
            device,
            compute,
            quant,
            encoder_layers(),
        )
        .map_err(|e| H3Error::Load(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Энкодер, идущий в комплекте с моделью (подкаталог/компонент
    /// `text_encoder`). Ошибка, если источник его не содержит.
    pub fn load_bundled(
        model: &H3Source,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, H3Error> {
        let src = H3EncoderSource::from_model(model).ok_or_else(|| {
            H3Error::Load(format!(
                "в источнике {} нет text_encoder — укажите отдельный каталог или .syn энкодера",
                model.path().display()
            ))
        })?;
        Self::load_source(&src, device, compute, quant)
    }

    pub fn encoder(&self) -> &H3Encoder {
        &self.inner
    }

    pub fn prepare_image(&self, rgb: &Tensor) -> Result<(Tensor, ImageGrid), H3Error> {
        self.inner
            .prepare_image(rgb)
            .map_err(|e| H3Error::Load(e.to_string()))
    }

    pub fn encode(
        &self,
        presentation: &H3Presentation,
        images: &[(Tensor, ImageGrid)],
    ) -> Result<H3Conditioning, H3Error> {
        self.inner
            .encode(presentation, images)
            .map_err(|e| H3Error::Load(e.to_string()))
    }

    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }
}

pub fn presentation_t2va(prompt: &str) -> H3Presentation {
    H3Presentation::t2va(prompt)
}

pub fn presentation_fl2va(
    prompt: &str,
    grids: &[ImageGrid],
    merge: usize,
) -> H3Presentation {
    H3Presentation::fl2va(prompt, grids, merge)
}

pub fn presentation_ref2va(prompt: &str, refs: &[RefItem], merge: usize) -> H3Presentation {
    H3Presentation::ref2va(prompt, refs, merge)
}
