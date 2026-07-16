use std::path::Path;
use synaptix_core::{device::Device, tensor::Tensor};
use crate::error::Result;
use super::png::{load_image, save_image};

pub fn load_webp(path: impl AsRef<Path>, device: Device) -> Result<Tensor> {
    load_image(path, device)
}

pub fn save_webp(tensor: &Tensor, path: impl AsRef<Path>) -> Result<()> {
    save_image(tensor, path)
}
