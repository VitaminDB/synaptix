use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::H3Error;

pub use synaptix_vlm_qwen3::pipeline::{H3Conditioning, H3Encoder, H3_ENCODER_LAYERS};
pub use synaptix_vlm_qwen3::presentation::{H3Presentation, RefItem, TokenTag, VideoBlock};
pub use synaptix_vlm_qwen3::preprocess::ImageGrid;

pub struct EncoderHandle {
    inner: H3Encoder,
}

impl EncoderHandle {
    pub fn load(
        encoder_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let inner = H3Encoder::load(encoder_dir, None, device, dtype, H3_ENCODER_LAYERS)
            .map_err(|e| H3Error::Load(e.to_string()))?;
        Ok(Self { inner })
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
